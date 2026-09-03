"""Full model: embedding → N transformer layers → LM head.

Streaming attention architecture: ALL attention on GPU0, experts
split across all GPUs via EP. GPU1+ reserved for EP prefill and
HCS decode experts (zero attention allocation).

Loading sequence:
  Phase 0: System RAM budget check (refuse to run if insufficient)
  Phase 1: GPU weights — all on GPU0 (streaming BF16→INT8, one tensor at a time)
  Phase 2: CPU expert weights (Krasis Rust engine, INT4)
"""

import gc
import hashlib
import importlib
import logging
import math
import os
import json
import shutil
import sys
import threading
import time
import base64
import types
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from math import ceil, pi
from typing import List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open

from krasis.attention_backend import (
    attention_quant_cache_nbits,
    attention_quant_nbits,
    HQQ_ATTENTION_CACHE_VERSION,
    HQQ_CACHE_PROFILE_BASELINE,
    hqq_attention_cache_dir,
    hqq_attention_cache_total_bytes,
    hqq_backend_name,
    hqq_cache_algorithm_for_nbits,
    hqq_cache_storage_nbits,
    hqq_attention_manifest_path,
    hqq_attention_pending_manifest_path,
    init_hqq_attention_manifest,
    is_hqq_attention,
    is_quantized_attention,
    delete_hqq_attention_pending_manifest,
    hqq_auto_direct_edge_nbits,
    load_hqq_attention_artifact,
    load_hqq_attention_manifest,
    load_hqq_attention_pending_manifest,
    require_complete_hqq_sidecar_manifest,
    require_complete_hqq_attention_manifest,
    save_hqq_attention_manifest,
    save_hqq_attention_pending_manifest,
    hqq_auto_budget_bytes_from_pct,
    hqq_auto_candidate_from_records,
    hqq_auto_promotion_policy,
    hqq46_auto_budget_bytes_from_mib,
    hqq46_tensor_nbits,
    is_hqq_auto_attention,
    is_hqq_mixed_attention,
    select_hqq_auto_promotions,
    synthesize_hqq_fused_qkv_artifact,
    validate_hqq_cache_nbits,
    validate_hqq_nbits,
    write_hqq_attention_artifact,
)
from krasis.timing import TIMING

from krasis.config import (
    ModelConfig,
    PPRankConfig,
    QuantConfig,
    build_pp_ranks,
    cache_dir_for_model,
    compute_pp_partition,
    marlin_cache_basename,
)
from krasis.weight_loader import WeightLoader, int8_linear
from krasis.layer import TransformerLayer
from krasis.kv_cache import PagedKVCache, SequenceKVState
from krasis.sampler import sample
from krasis.gemma4_vision import (
    Gemma4ImagePreprocessor,
    Gemma4MultimodalEmbedder,
    Gemma4VisionConfig,
    Gemma4VisionModel,
)
from krasis.glm53_vision import Glm53ImagePreprocessor, Glm53VisionConfig, Glm53VisionModel
from krasis.deepseek_v4_vision import (
    DeepseekV4ImagePreprocessor,
    DeepseekV4VisionAligner,
    DeepseekV4VisionModel,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
)
from krasis.step_vision_int4 import quantize_step_vision_modules_int4, quantize_vision_modules_int4

from krasis.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

MAMBA2_PROJECTION_INT4_CACHE_VERSION = 2
MAMBA2_PROJECTION_INT4_CACHE_FORMAT = "krasis_mamba2_projection_marlin_int4"


class KrasisVisionVramError(RuntimeError):
    """Raised when an image request cannot fit transient vision VRAM."""


_TQ4_SEED = 42
_TQ4_LAYER_SEED_STRIDE = 1337
_RAM_LEDGER_LAST_RSS_KB: Optional[int] = None


def _tq4_wht_signs(layer_idx: int, head_dim: int, device: torch.device) -> torch.Tensor:
    """vLLM-style deterministic WHT rotation signs for one attention layer."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_TQ4_SEED + layer_idx * _TQ4_LAYER_SEED_STRIDE)
    bits = torch.randint(0, 2, (head_dim,), generator=gen, device="cpu")
    signs = bits.to(torch.float32).mul_(2.0).sub_(1.0)
    return signs.to(device=device, non_blocking=True)


def _python_trace_enabled(component: str) -> bool:
    if os.environ.get("KRASIS_TRACE") != "1":
        return False
    raw = os.environ.get("KRASIS_TRACE_COMPONENTS", "").strip()
    if not raw:
        return True
    filters = {part.strip().lower() for part in raw.split(",") if part.strip()}
    component = component.lower()
    if "all" in filters or component in filters:
        return True
    prefix = component.split("_", 1)[0]
    return prefix in filters


def _python_trace(component: str, message: str) -> None:
    if _python_trace_enabled(component):
        logger.info("[KRASIS-TRACE] event=mark scope=python component=%s %s", component, message)


def _weight_dtype_code(t: torch.Tensor) -> int:
    """Map torch dtype to Rust GpuWeight dtype code: 0=BF16, 1=FP32, 2=FP16."""
    if t.dtype == torch.float32:
        return 1
    elif t.dtype == torch.float16:
        return 2
    return 0  # BF16 (default)


def _dsa_owner_layers_for_segment(
    cfg: ModelConfig,
    layer_start: int,
    layer_end: int,
) -> list[int]:
    """Return the unique IndexShare owners needed by one decode segment."""
    if not cfg.is_dsa:
        return []
    if layer_start < 0 or layer_end > cfg.num_hidden_layers or layer_start >= layer_end:
        raise ValueError(
            f"Invalid DSA decode segment [{layer_start}, {layer_end}) for "
            f"{cfg.num_hidden_layers} layers"
        )
    dsa_layers = [
        layer_idx
        for layer_idx in range(layer_start, layer_end)
        if cfg.is_dsa_layer(layer_idx)
    ]
    owners = {
        cfg.dsa_indexer_owner_layer(layer_idx)
        for layer_idx in dsa_layers
    }
    if None in owners:
        raise RuntimeError(
            f"DSA decode segment [{layer_start}, {layer_end}) contains an "
            "unowned IndexShare layer"
        )
    return sorted(int(owner) for owner in owners)


def _dsa_resource_layers_for_segment(
    cfg: ModelConfig,
    layer_start: int,
    layer_end: int,
) -> tuple[list[int], list[int]]:
    """Split required IndexShare owners into local compute and replica sets."""
    owners = _dsa_owner_layers_for_segment(cfg, layer_start, layer_end)
    local = [owner for owner in owners if layer_start <= owner < layer_end]
    replicas = [owner for owner in owners if owner not in local]
    return local, replicas


def _dsa_topk_candidate_capacity(context: int, configured_topk: int) -> int:
    """Exact shared ping-pong capacity used by the native DSA selector."""
    if context <= 0 or configured_topk <= 0:
        raise ValueError(
            "DSA top-k planning requires positive context and configured top-k"
        )
    selected = min(context, configured_topk)
    if context <= selected:
        return 0
    padded_selected = 1 << (selected - 1).bit_length()
    sort_width = padded_selected * 2
    initial_runs = (context + sort_width - 1) // sort_width
    return initial_runs * selected if initial_runs > 1 else 0


# GPU-to-GPU P2P transfer may silently fail on some systems (returns zeros).
# Detect this once at import time and use CPU bounce if needed.
_p2p_works: Optional[bool] = None

def _check_p2p() -> bool:
    """Test if direct GPU-to-GPU transfer works."""
    global _p2p_works
    if _p2p_works is not None:
        return _p2p_works
    if torch.cuda.device_count() < 2:
        _p2p_works = True
        return True
    test = torch.tensor([42, 137], dtype=torch.float32, device='cuda:0')
    transferred = test.to('cuda:1')
    torch.cuda.synchronize()
    _p2p_works = bool((transferred.cpu() == test.cpu()).all())
    if not _p2p_works:
        logger.warning("GPU P2P transfer broken — using CPU bounce for cross-device transfers")
    return _p2p_works


def _to_device(tensor: torch.Tensor, target: torch.device) -> torch.Tensor:
    """Transfer tensor to target device, using CPU bounce if P2P is broken."""
    if tensor.device == target:
        return tensor
    if _check_p2p():
        return tensor.to(target)
    return tensor.cpu().to(target)


def _linear(x: torch.Tensor, weight_data) -> torch.Tensor:
    """Dispatch to INT8 or BF16 linear based on weight type."""
    if isinstance(weight_data, tuple):
        return int8_linear(x, *weight_data)
    return torch.nn.functional.linear(x, weight_data)


def _read_meminfo():
    """Read /proc/meminfo and return dict of key -> value in KB."""
    result = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    result[key] = int(parts[1])
    except (OSError, ValueError):
        pass
    return result


def _vram_checkpoint(label: str, devices=None):
    """Print VRAM usage at a checkpoint for diagnostics."""
    import torch
    if devices is None:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    for dev in devices:
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        free, total = torch.cuda.mem_get_info(dev)
        used = total - free
        print(
            f"  \033[33m[VRAM {label}]\033[0m {dev}: "
            f"alloc={alloc // (1024*1024)} MB, reserved={reserved // (1024*1024)} MB, "
            f"used={used // (1024*1024)} MB, free={free // (1024*1024)} MB, total={total // (1024*1024)} MB",
            flush=True,
        )
        logger.info(
            "VRAM_CHECKPOINT [%s] %s: alloc=%d MB, reserved=%d MB, driver_used=%d MB, free=%d MB, total=%d MB",
            label, dev, alloc >> 20, reserved >> 20, used >> 20, free >> 20, total >> 20,
        )


def _vram_ledger_enabled() -> bool:
    return os.environ.get("KRASIS_VRAM_LEDGER", "").strip().lower() in ("1", "true", "yes", "on")


def _ram_ledger_enabled() -> bool:
    return os.environ.get("KRASIS_RAM_LEDGER", "").strip().lower() in ("1", "true", "yes", "on")


def _read_vmrss_kb() -> int:
    """Read VmRSS from /proc/self/status in KB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _read_proc_status_kb() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    key = parts[0].rstrip(":")
                    if key.startswith("Vm") or key in ("RssAnon", "RssFile", "RssShmem"):
                        try:
                            result[key] = int(parts[1])
                        except ValueError:
                            pass
    except OSError:
        pass
    return result


def _read_smaps_rollup_kb() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    try:
                        result[parts[0].rstrip(":")] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return result


def log_ram_ledger(label: str, component_bytes: Optional[dict[str, int]] = None) -> None:
    """Log opt-in process/system RAM checkpoints for startup residency diagnosis."""
    if not _ram_ledger_enabled():
        return

    global _RAM_LEDGER_LAST_RSS_KB

    status = _read_proc_status_kb()
    smaps = _read_smaps_rollup_kb()
    meminfo = _read_meminfo()

    rss_kb = smaps.get("Rss", status.get("VmRSS", 0))
    pss_kb = smaps.get("Pss", 0)
    anon_kb = smaps.get("Anonymous", status.get("RssAnon", 0))
    shmem_kb = smaps.get("Shmem", status.get("RssShmem", 0))
    rss_file_kb = status.get("RssFile", 0)
    file_backed_kb = rss_file_kb if rss_file_kb else max(0, rss_kb - anon_kb - shmem_kb)
    private_kb = smaps.get("Private_Clean", 0) + smaps.get("Private_Dirty", 0)
    shared_kb = smaps.get("Shared_Clean", 0) + smaps.get("Shared_Dirty", 0)
    delta_kb = 0 if _RAM_LEDGER_LAST_RSS_KB is None else rss_kb - _RAM_LEDGER_LAST_RSS_KB
    _RAM_LEDGER_LAST_RSS_KB = rss_kb

    components = ""
    if component_bytes:
        components = " " + " ".join(
            f"{name}_mb={bytes_ / (1024 * 1024):.1f}"
            for name, bytes_ in sorted(component_bytes.items())
            if bytes_ > 0
        )

    logger.info(
        "RAM LEDGER label=%s rss_mb=%.1f delta_mb=%+.1f pss_mb=%.1f anon_mb=%.1f "
        "file_mb=%.1f shmem_mb=%.1f private_mb=%.1f shared_mb=%.1f vm_size_mb=%.1f "
        "vm_data_mb=%.1f vm_swap_mb=%.1f system_available_mb=%.1f system_cached_mb=%.1f%s",
        label,
        rss_kb / 1024.0,
        delta_kb / 1024.0,
        pss_kb / 1024.0,
        anon_kb / 1024.0,
        file_backed_kb / 1024.0,
        shmem_kb / 1024.0,
        private_kb / 1024.0,
        shared_kb / 1024.0,
        status.get("VmSize", 0) / 1024.0,
        status.get("VmData", 0) / 1024.0,
        status.get("VmSwap", 0) / 1024.0,
        meminfo.get("MemAvailable", 0) / 1024.0,
        meminfo.get("Cached", 0) / 1024.0,
        components,
    )


def _estimate_expert_ram_gb(cfg, cpu_expert_bits: int = 4, group_size: int = 128) -> float:
    """Estimate system RAM needed for CPU expert weights in GB.

    Calculates per-expert byte sizes for the unified weight format:
      w13 (gate+up concat, transposed): packed [K//8, 2*N] + scales [K//gs, 2*N]
      w2 (down, transposed):            packed [N//8, K] + scales [N//gs, K]

    Plus shared experts if any.
    """
    h = cfg.hidden_size           # K for w13, N_down for w2
    m = cfg.moe_intermediate_size  # N for w13, K_down for w2
    gs = group_size
    n_experts = cfg.n_routed_experts
    num_moe_layers = cfg.num_moe_layers

    if cpu_expert_bits == 4:
        # w13 packed: (h//8) * 2*m * 4 bytes, w13 scales: (h//gs) * 2*m * 2 bytes
        w13_bytes = (h // 8) * (2 * m) * 4 + (h // gs) * (2 * m) * 2
        # w2 packed: (m//8) * h * 4 bytes, w2 scales: (m//gs) * h * 2 bytes
        w2_bytes = (m // 8) * h * 4 + (m // gs) * h * 2
    else:
        # INT8: gate/up each m*h + scales, down h*m + scales
        w13_bytes = 2 * m * h + (h // gs) * (2 * m) * 2
        w2_bytes = h * m + (m // gs) * h * 2

    per_expert = w13_bytes + w2_bytes
    routed_total = num_moe_layers * n_experts * per_expert

    # Shared experts
    shared_total = 0
    if cfg.n_shared_experts > 0:
        shared_m = cfg.n_shared_experts * m
        if cpu_expert_bits == 4:
            s_w13 = (h // 8) * (2 * shared_m) * 4 + (h // gs) * (2 * shared_m) * 2
            s_w2 = (shared_m // 8) * h * 4 + (shared_m // gs) * h * 2
        else:
            s_w13 = 2 * shared_m * h + (h // gs) * (2 * shared_m) * 2
            s_w2 = h * shared_m + (shared_m // gs) * h * 2
        shared_total = num_moe_layers * (s_w13 + s_w2)

    return (routed_total + shared_total) / 1e9


def _check_system_ram(
    cfg,
    cpu_expert_bits: int = 4,
    group_size: int = 128,
    force_load: bool = False,
):
    """Check system RAM budget before loading expert weights.

    Reads /proc/meminfo, estimates RAM needed, and refuses to run if
    the model would exceed 95% of MemTotal. Warns at 90%.

    Args:
        cfg: ModelConfig with model dimensions
        cpu_expert_bits: 4 or 8
        group_size: expert weight quantization group size
        force_load: If True, log error but don't raise

    Raises:
        RuntimeError: If insufficient RAM and force_load=False
    """
    meminfo = _read_meminfo()
    if not meminfo:
        logger.warning("Could not read /proc/meminfo — skipping RAM budget check")
        return

    mem_total_gb = meminfo.get("MemTotal", 0) / 1024 / 1024
    mem_avail_gb = meminfo.get("MemAvailable", 0) / 1024 / 1024

    expert_ram_gb = _estimate_expert_ram_gb(cfg, cpu_expert_bits, group_size)
    # Compute non-expert RAM overhead from model dimensions:
    # - GPU weights (attention, embedding, lm_head, norms, gates, shared experts) are loaded
    #   to GPU but PyTorch/CUDA uses staging buffers in system RAM during loading.
    # - Estimate ~2x the BF16 weight size for staging overhead (load + transfer).
    # - Add 10% of total expert RAM for PyTorch/CUDA runtime, page tables, etc.
    h = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    # Attention projection params per layer
    if cfg.kv_lora_rank is not None:
        # MLA: Q (or q_a + q_b), KV_A, KV_B, O
        kv_lora = cfg.kv_lora_rank
        qk_nope = cfg.qk_nope_head_dim or 128
        qk_rope = cfg.qk_rope_head_dim or 64
        v_hd = cfg.v_head_dim or 128
        q_params = h * n_heads * (qk_nope + qk_rope)
        if cfg.q_lora_rank:
            q_params = h * cfg.q_lora_rank + cfg.q_lora_rank * n_heads * (qk_nope + qk_rope)
        kv_a_params = h * (kv_lora + qk_rope)
        kv_b_params = kv_lora * n_heads * (qk_nope + v_hd)
        o_params = n_heads * v_hd * h
        attn_params = q_params + kv_a_params + kv_b_params + o_params
    else:
        # GQA: Q + K + V + O
        head_dim = cfg.gqa_head_dim or (h // n_heads)
        attn_params = h * n_heads * head_dim + 2 * h * n_kv * head_dim + n_heads * head_dim * h
    attn_bytes = attn_params * 2 * n_layers  # BF16
    # Embedding + lm_head
    embed_bytes = cfg.vocab_size * h * 2
    lm_head_bytes = 0 if cfg.tie_word_embeddings else cfg.vocab_size * h * 2
    # Gates, norms, shared experts
    n_moe = n_layers - cfg.first_k_dense_replace
    gate_bytes = h * cfg.n_routed_experts * 2 * n_moe if cfg.n_routed_experts > 0 else 0
    norm_bytes = 2 * h * 2 * n_layers
    shared_inter = cfg.shared_expert_intermediate_size or (cfg.n_shared_experts * cfg.moe_intermediate_size)
    shared_bytes = 3 * h * shared_inter * 2 * n_moe if shared_inter > 0 else 0
    gpu_weight_bytes = attn_bytes + embed_bytes + lm_head_bytes + gate_bytes + norm_bytes + shared_bytes
    # Staging overhead (2x GPU weights for load/transfer) + runtime overhead (10% of expert RAM)
    overhead_gb = (gpu_weight_bytes * 2 / 1e9) + (expert_ram_gb * 0.10)
    overhead_gb = max(overhead_gb, 2.0)  # floor at 2 GB for minimal PyTorch/CUDA runtime
    total_needed_gb = expert_ram_gb + overhead_gb

    logger.info(
        "RAM budget: experts=%.1f GB, overhead=%.1f GB, total=%.1f GB needed | "
        "system: %.1f GB total, %.1f GB available",
        expert_ram_gb, overhead_gb, total_needed_gb, mem_total_gb, mem_avail_gb,
    )

    hard_limit = 0.95 * mem_total_gb
    warn_limit = 0.90 * mem_total_gb

    if total_needed_gb > hard_limit:
        msg = (
            f"INSUFFICIENT RAM: model needs ~{total_needed_gb:.1f} GB but system has "
            f"{mem_total_gb:.1f} GB total (95% limit = {hard_limit:.1f} GB). "
            f"Breakdown: {expert_ram_gb:.1f} GB experts + {overhead_gb:.1f} GB overhead. "
            f"Available: {mem_avail_gb:.1f} GB. "
            f"This WILL cause OOM and crash system processes."
        )
        if force_load:
            logger.error("FORCE LOAD: %s", msg)
        else:
            raise RuntimeError(msg)
    elif total_needed_gb > warn_limit:
        logger.warning(
            "LOW RAM HEADROOM: model needs ~%.1f GB, system has %.1f GB total "
            "(%.1f GB available). Monitor memory usage closely.",
            total_needed_gb, mem_total_gb, mem_avail_gb,
        )
    else:
        logger.info(
            "RAM budget OK: %.1f GB needed, %.1f GB headroom",
            total_needed_gb, mem_total_gb - total_needed_gb,
        )


def _check_actual_rss(estimated_gb: float):
    """After loading, compare actual RSS to estimate and warn if >10% deviation."""
    actual_kb = _read_vmrss_kb()
    if actual_kb == 0:
        return
    actual_gb = actual_kb / 1024 / 1024
    if estimated_gb > 0:
        deviation = abs(actual_gb - estimated_gb) / estimated_gb
        if deviation > 0.10:
            logger.warning(
                "RAM estimate deviation: estimated %.1f GB, actual VmRSS %.1f GB "
                "(%.0f%% off) — estimates may need recalibration",
                estimated_gb, actual_gb, deviation * 100,
            )
        else:
            logger.info(
                "RAM estimate accurate: estimated %.1f GB, actual VmRSS %.1f GB (%.0f%% deviation)",
                estimated_gb, actual_gb, deviation * 100,
            )


def _compute_layer_groups(
    rank,
    cfg,
    divisor: int,
) -> List[Tuple[List[int], List[int]]]:
    """Compute layer groups for layer-grouped prefill.

    Splits a rank's layers into groups where each group has at most
    ceil(num_moe_layers / divisor) MoE layers. Dense layers (below
    first_k_dense_replace) are included in the first group.

    Args:
        rank: PPRankConfig with layer_start, layer_end
        cfg: ModelConfig with first_k_dense_replace
        divisor: Number of groups to split MoE layers into

    Returns:
        List of (abs_layer_indices, moe_layer_indices) tuples.
        abs_layer_indices: absolute layer indices for self.layers indexing
        moe_layer_indices: 0-based MoE indices for GpuPrefillManager
    """
    first_k = cfg.first_k_dense_replace
    all_layers = list(range(rank.layer_start, rank.layer_end))

    # Build abs_layer → moe_index mapping for hybrid models
    # For standard models: MoE layers are first_k..N, so moe_idx = l - first_k
    # For hybrid (Nemotron): MoE layers are scattered, need sequential mapping
    if cfg.layer_types is not None:
        _abs_to_moe = {}
        _moe_seq = 0
        for i in range(cfg.num_hidden_layers):
            if cfg.is_moe_layer(i):
                _abs_to_moe[i] = _moe_seq
                _moe_seq += 1
        dense_layers = [l for l in all_layers if l not in _abs_to_moe]
        moe_layers = [l for l in all_layers if l in _abs_to_moe]
    else:
        _abs_to_moe = None
        dense_layers = [l for l in all_layers if l < first_k]
        moe_layers = [l for l in all_layers if l >= first_k]

    if not moe_layers:
        # All dense layers — single group, no experts
        return [(all_layers, [])]

    def _to_moe_idx(abs_layer):
        if _abs_to_moe is not None:
            return _abs_to_moe[abs_layer]
        return abs_layer - first_k

    # Active-only or single group: all layers in one group
    if divisor <= 1:
        moe_indices = [_to_moe_idx(l) for l in moe_layers]
        return [(all_layers, moe_indices)]

    # Split MoE layers into groups
    num_moe = len(moe_layers)
    moe_per_group = ceil(num_moe / divisor)

    groups = []
    for g in range(divisor):
        start = g * moe_per_group
        end = min(start + moe_per_group, num_moe)
        if start >= num_moe:
            break
        group_moe = moe_layers[start:end]
        group_moe_indices = [_to_moe_idx(l) for l in group_moe]

        if _abs_to_moe is not None:
            # Hybrid model: include interleaved non-MoE layers in correct position.
            # Each group spans from first_moe to last_moe (inclusive), plus any
            # non-MoE layers before the first group or between groups.
            range_start = 0 if g == 0 else group_moe[0]
            range_end = group_moe[-1] + 1
            group_all = [l for l in all_layers if range_start <= l < range_end]
        elif g == 0:
            # Standard model: dense layers go in the first group
            group_all = dense_layers + group_moe
        else:
            group_all = list(group_moe)

        groups.append((group_all, group_moe_indices))

    return groups


class CPUHubManager:
    """Manages multi-GPU token coordination using CPU as a star-hub aggregator.

    Owns pinned CPU buffers for N GPUs. 
    Workflow: 
      1. gather(gpu_tensors) -> copies from VRAM to pinned RAM
      2. reduce()            -> calls Rust engine to sum BF16 buffers
      3. broadcast(gpu_tensors) -> copies from pinned RAM to all VRAM
    """

    def __init__(self, num_gpus: int, hidden_size: int, engine):
        self.num_gpus = num_gpus
        self.hidden_size = hidden_size
        self.engine = engine
        
        # Max chunk size for prefill buffers (e.g. 5000 tokens)
        self.max_tokens = 5000
        self.buffer_elements = self.max_tokens * hidden_size
        
        # Pinned input buffers (one per GPU)
        self.input_bufs = [
            torch.empty(self.buffer_elements, dtype=torch.bfloat16, pin_memory=True)
            for _ in range(num_gpus)
        ]
        # Pinned output buffer (result)
        self.output_buf = torch.empty(
            self.buffer_elements, dtype=torch.bfloat16, pin_memory=True
        )
        
        self.input_ptrs = [buf.data_ptr() for buf in self.input_bufs]
        self.output_ptr = self.output_buf.data_ptr()

    def reduce_sum(self, num_tokens: int):
        """Call Rust engine to sum partial results."""
        num_elements = num_tokens * self.hidden_size
        if num_elements > self.buffer_elements:
            raise ValueError(f"Too many tokens for CPU hub: {num_tokens} > {self.max_tokens}")
            
        self.engine.reduce_sum_bf16(
            self.input_ptrs,
            self.output_ptr,
            num_elements
        )

    def gather(self, gpu_tensors: List[torch.Tensor]):
        """Copy from GPUs to pinned CPU memory (asynchronous)."""
        assert len(gpu_tensors) == self.num_gpus
        for i, t in enumerate(gpu_tensors):
            # Non-blocking copy to pinned memory
            M = t.shape[0]
            count = M * self.hidden_size
            self.input_bufs[i][:count].copy_(t.view(-1), non_blocking=True)

    def broadcast(self, gpu_tensors: List[torch.Tensor]):
        """Copy from pinned CPU memory to all GPUs (asynchronous)."""
        # num_tokens is derived from the target tensor shapes
        for i, t in enumerate(gpu_tensors):
            M = t.shape[0]
            count = M * self.hidden_size
            t.view(-1).copy_(self.output_buf[:count], non_blocking=True)

    def all_gather(self, gpu_slices: List[Optional[torch.Tensor]],
                   targets: List[torch.Tensor]):
        """Gather partial hidden states from all GPUs and broadcast full set to all.

        Each GPU owns a slice of tokens. This concatenates all slices via pinned
        CPU memory and copies the full result to every target tensor.

        Args:
            gpu_slices: Per-GPU partial hidden states (may contain None for inactive GPUs).
            targets: Pre-allocated [total_tokens, hidden_size] tensors on each GPU.
        """
        # Copy each GPU's slice into contiguous pinned memory
        offset = 0
        for t in gpu_slices:
            if t is None:
                continue
            count = t.shape[0] * self.hidden_size
            self.output_buf[offset:offset + count].copy_(t.view(-1), non_blocking=True)
            offset += count
        torch.cuda.synchronize()
        # Broadcast concatenated result to all target GPUs
        for target in targets:
            count = target.shape[0] * self.hidden_size
            target.view(-1).copy_(self.output_buf[:count], non_blocking=True)


def _apply_max_context_limit(
    cfg: ModelConfig,
    max_context_tokens: Optional[int],
) -> None:
    """Apply an explicit runtime context cap without extending model support."""
    if max_context_tokens is None:
        return
    requested = int(max_context_tokens)
    if requested <= 0:
        raise ValueError(
            f"max_context_tokens must be positive when set, got {requested}"
        )
    model_limit = int(cfg.max_position_embeddings)
    if requested > model_limit:
        raise ValueError(
            f"max_context_tokens {requested} exceeds model limit {model_limit}"
        )
    cfg.max_position_embeddings = requested


class KrasisModel:
    """Full model with streaming attention (GPU0) + EP MoE (all GPUs) + CPU experts."""

    # Mapping from _extract_layer_weights dict keys to attention module attribute names.
    # Only entries where the names differ need to be listed.
    _WEIGHT_KEY_TO_ATTN_ATTR = {
        "kv_a_proj_with_mqa": "kv_a_proj",
        "kv_a_layernorm": "kv_a_norm_weight",
        "q_a_layernorm": "q_a_norm_weight",
    }

    def __init__(
        self,
        model_path: str,
        pp_partition: Optional[List[int]] = None,
        num_gpus: Optional[int] = None,
        devices: Optional[List[str]] = None,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        krasis_threads: int = 40,
        gpu_prefill: bool = False,  # Rust prefill replaces Python GpuPrefillManager
        gpu_prefill_threshold: int = 300,
        quant_cfg: QuantConfig = None,
        force_load: bool = False,
        layer_group_size: int = 1,
        gguf_path: Optional[str] = None,
        gguf_native: bool = False,
        expert_hqq_diagnostic_cache_spec: Optional[str] = None,
        kv_cache_mb: int = 1000,  # MB for KV cache
        stream_attention: bool = False,
        max_context_tokens: Optional[int] = None,
    ):
        self.cfg = ModelConfig.from_model_path(model_path)
        _apply_max_context_limit(self.cfg, max_context_tokens)
        self.quant_cfg = quant_cfg or QuantConfig()
        if self.cfg.is_deepseek_v4:
            if self.quant_cfg.kv_cache_format not in ("native", "bf16"):
                raise ValueError(
                    "DeepSeek-V4 sequence state supports only architecture-native packed "
                    f"or expanded BF16 storage, got {self.quant_cfg.kv_cache_format!r}. "
                    "Conventional k4v4/k6v6 formats do not represent its shared latent cache."
                )
            if self.quant_cfg.attention not in (
                "hqq4",
                "hqq46_auto",
                "hqq6",
                "hqq68_auto",
                "hqq8",
                "bf16",
            ):
                raise ValueError(
                    "DeepSeek-V4 supports HQQ4/HQQ6/HQQ8, measured 4/6 and 6/8 "
                    "mixed-auto HQQ, or BF16 attention; got "
                    f"{self.quant_cfg.attention!r}."
                )
        elif self.quant_cfg.kv_cache_format == "native":
            raise ValueError(
                "The Native sequence-state format is architecture-owned and currently "
                "implemented only for DeepSeek-V4. Select a cache format supported by "
                f"{self.cfg.model_type}."
            )
        if self.cfg.gemma4_text:
            if self.quant_cfg.gpu_expert_bits != 4 or self.quant_cfg.cpu_expert_bits != 4:
                raise ValueError(
                    "Gemma4 text support currently validates only INT4 routed experts "
                    f"(gpu_expert_bits=4, cpu_expert_bits=4). Got "
                    f"gpu_expert_bits={self.quant_cfg.gpu_expert_bits}, "
                    f"cpu_expert_bits={self.quant_cfg.cpu_expert_bits}."
                )
            if self.quant_cfg.attention not in (
                "bf16",
                "hqq4",
                "hqq46_auto",
                "hqq6",
                "hqq68_auto",
                "hqq8",
            ):
                raise ValueError(
                    "Gemma4 text support validates BF16, fixed HQQ attention, or measured "
                    "4/6 and 6/8 mixed-auto HQQ attention. "
                    f"Got attention_quant={self.quant_cfg.attention!r}."
                )
            if self.quant_cfg.kv_cache_format not in ("bf16", "k6v6", "k4v4"):
                raise ValueError(
                    "Gemma4 text support currently validates only BF16, k6v6, and k4v4 KV cache. "
                    f"Got kv_cache_format={self.quant_cfg.kv_cache_format!r}."
                )
            if self.quant_cfg.ring_window_kv and self.quant_cfg.kv_cache_format == "k4v4":
                raise ValueError(
                    "Gemma4 ring-window KV currently validates only k6v6. "
                    "k4v4 ring-window produced invalid 25K long-prompt output during validation; "
                    "use k6v6 ring-window or k4v4 without ring-window."
                )
        self.gguf_path = gguf_path
        self.gguf_native = gguf_native
        self.expert_hqq_diagnostic_cache_spec = expert_hqq_diagnostic_cache_spec

        # Determine PP partition
        if pp_partition is None:
            if num_gpus is None:
                num_gpus = torch.cuda.device_count()
            pp_partition = compute_pp_partition(self.cfg.num_hidden_layers, num_gpus)

        self.pp_partition = pp_partition
        self.ranks = build_pp_ranks(self.cfg, pp_partition, devices)
        # all_devices = all GPUs for EP replication, not just PP-ranked GPUs
        effective_num_gpus = num_gpus or len(self.ranks)
        self._num_gpus = effective_num_gpus
        self.all_devices = [torch.device(f"cuda:{i}") for i in range(effective_num_gpus)]
        self.kv_dtype = kv_dtype
        self.kv_cache_mb = kv_cache_mb
        self.krasis_threads = krasis_threads
        self.gpu_prefill_enabled = gpu_prefill
        self.gpu_prefill_threshold = gpu_prefill_threshold
        self.force_load = force_load
        # When attention is quantized, it's permanently VRAM-resident (much smaller),
        # so streaming from CPU is unnecessary and would overwrite MarlinWeight attrs.
        if stream_attention and is_quantized_attention(quant_cfg.attention):
            logger.info("Disabling stream_attention: quantized attention (%s) is permanently VRAM-resident",
                        quant_cfg.attention)
            stream_attention = False
        self.stream_attention = stream_attention

        # Double-buffered streaming attention needs layer_group_size >= 2
        if stream_attention and layer_group_size >= 1:
            if layer_group_size < 2:
                logger.warning("stream_attention: auto-adjusting layer_group_size %d → 2 (minimum for double-buffering)", layer_group_size)
                layer_group_size = 2
            elif layer_group_size % 2 != 0:
                new_size = layer_group_size + 1
                logger.warning("stream_attention: auto-adjusting layer_group_size %d → %d (must be even for double-buffering)", layer_group_size, new_size)
                layer_group_size = new_size
        self.layer_group_size = layer_group_size

        hybrid_info = ""
        if self.cfg.is_hybrid:
            hybrid_info = f", hybrid={self.cfg.num_full_attention_layers} full + {self.cfg.num_hidden_layers - self.cfg.num_full_attention_layers} linear"
        logger.info(
            "KrasisModel: %d layers, PP=%s, %d GPUs, attn=%s%s",
            self.cfg.num_hidden_layers, pp_partition, len(self.ranks),
            "rust", hybrid_info,
        )

        # Will be populated by load()
        self.layers: List[TransformerLayer] = []
        self.embedding: Optional[torch.Tensor] = None
        self.final_norm: Optional[torch.Tensor] = None
        self.lm_head_data = None  # (int8, scale) tuple or plain BF16 tensor
        self.kv_caches: List[PagedKVCache] = []
        self.krasis_engine = None
        self.gpu_prefill_managers: dict = {}  # device -> GpuPrefillManager
        self.tokenizer: Optional[Tokenizer] = None
        self._loaded = False

        # Per-device state (streaming attention: GPU0 only).
        self._device_state: dict = {}
        self._active_device: Optional[str] = None  # current device for use_device()

        # Streaming attention: all layers on GPU0.
        # _layer_split = [(0, L)] — single entry covering all layers.
        # _get_gpu_for_layer(i) always returns 0.
        self._layer_split: List[Tuple[int, int]] = []

        # HCS device: GPU where ALL MoE decode happens (most free VRAM, typically GPU1).
        # Set by server.py after HCS allocation. None = use layer's own device.
        self._hcs_device: Optional[torch.device] = None
        # Multi-GPU HCS: experts pinned on ALL GPUs, GPU0 manager dispatches internally.
        # When True, _hcs_device is None and decode calls GPU0 manager directly.
        self._multi_gpu_hcs: bool = False

        # Decode mode: "gpu" (default) or "cpu".
        # GPU decode runs model.forward() per token on GPU.
        # GPU decode uses GpuDecodeStore in Rust (GIL-free).
        self.decode_mode: str = "gpu"

        # Streaming attention decode: weights on CPU pinned memory, DMA'd per-layer.
        # When enabled, attention weights are NOT permanently on GPU.
        self._stream_attn_enabled = False  # set True after offload
        self._stream_attn_cpu: dict = {}   # layer_idx -> {attr_name: pinned_tensor}
        self._stream_attn_gpu_bufs: list = [{}, {}]  # [buf0, buf1] ping-pong GPU buffers
        self._stream_attn_loaded: dict = {}  # {buf_idx: layer_idx} — tracks what's in each buffer
        self._stream_attn_dma_stream: Optional[torch.cuda.Stream] = None

        # For hybrid models: maps absolute layer idx → KV cache layer offset
        # within its device's cache. Linear attention layers get -1 (no KV).
        self._kv_layer_offsets: dict = {}  # layer_idx -> offset or -1

        # Maps absolute layer index → 0-based MoE sequential index.
        # For standard models: moe_idx = abs_layer - first_k_dense_replace
        # For hybrid (Nemotron): MoE layers are scattered, this provides the mapping.
        self._abs_to_moe_idx: dict = {}
        moe_seq = 0
        for i in range(self.cfg.num_hidden_layers):
            if self.cfg.is_moe_layer(i):
                self._abs_to_moe_idx[i] = moe_seq
                moe_seq += 1

        # Detect shared_expert_gate from weight map (Qwen3-Next)
        self._has_shared_expert_gate = self._detect_shared_expert_gate()
        self._hqq_attention_cache_bytes = 0
        self._hqq_attention_runtime = {}
        self._hqq_attention_runtime_nbits: Optional[int] = None
        self._hqq_attention_loaded_tensors = 0
        self._qwen_vision_processor = None
        self._qwen_vision_model = None
        self._qwen_vision_config = None
        self._step_vision_modules = None
        self._step_vision_processor = None
        self._step_vision_model = None
        self._step_vision_projector = None
        self._step_vision_config = None
        self._step_vision_quant_mode = None
        self._step_vision_quant_stats = None
        self._gemma_vision_processor = None
        self._gemma_vision_model = None
        self._gemma_vision_embedder = None
        self._gemma_vision_config = None
        self._gemma_vision_raw_config = None
        self._gemma_vision_quant_mode = None
        self._gemma_vision_quant_stats = None
        self._glm53_vision_processor = None
        self._glm53_vision_model = None
        self._glm53_vision_config = None
        self._glm53_vision_raw_config = None
        self._glm53_vision_quant_mode = None
        self._glm53_vision_quant_stats = None
        self._deepseek_v4_vision_processor = None
        self._deepseek_v4_vision_model = None
        self._deepseek_v4_vision_aligner = None
        self._deepseek_v4_vision_special = None
        self._deepseek_v4_vision_quant_mode = None
        self._last_multimodal_prefill_tensors = None

    def supports_image_inputs(self) -> bool:
        """Return True when the loaded model directory exposes a supported vision path."""
        return (
            self.supports_qwen_image_inputs()
            or self.supports_step_image_inputs()
            or self.supports_gemma_image_inputs()
            or self.supports_glm53_image_inputs()
            or self.supports_deepseek_v4_image_inputs()
        )

    def supports_deepseek_v4_image_inputs(self) -> bool:
        """Return True for a complete DeepSeek-V4-Flash-Vision checkpoint."""
        if self.cfg.model_type != "deepseek_v4" or self.cfg.vision_n_layers <= 0:
            return False
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        try:
            with open(index_path, encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            required = {
                "vision.patch_embed.proj.weight",
                "vision.norm.weight",
                "aligner.w1.weight",
                "aligner.w2.weight",
                "image_start",
                "image_pad",
                "image_newline",
                "image_end",
            }
            required.update(
                f"vision.blocks.{idx}.attn.wqkv.weight"
                for idx in range(self.cfg.vision_n_layers)
            )
            required.update(
                f"{self.cfg.layer_tensor_prefix(idx)}.ffn.gate.bias_vl"
                for idx in range(self.cfg.num_hidden_layers)
                if self.cfg.is_moe_layer(idx)
            )
            return required.issubset(weight_map)
        except (OSError, ValueError, KeyError):
            return False

    def supports_qwen_image_inputs(self) -> bool:
        """Return True when the loaded model directory contains Qwen image assets."""
        config_path = os.path.join(self.cfg.model_path, "config.json")
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        processor_path = os.path.join(self.cfg.model_path, "preprocessor_config.json")
        if not (os.path.exists(config_path) and os.path.exists(index_path) and os.path.exists(processor_path)):
            return False
        try:
            with open(config_path) as f:
                raw = json.load(f)
            if not isinstance(raw.get("vision_config"), dict) or not isinstance(raw.get("image_token_id"), int):
                return False
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            return any(key.startswith("model.visual.") for key in weight_map)
        except Exception:
            return False

    def supports_step_image_inputs(self) -> bool:
        """Return True when the loaded model directory contains Step-3.7 image assets."""
        config_path = os.path.join(self.cfg.model_path, "config.json")
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        required_code = (
            "configuration_step3p7.py",
            "processing_step3.py",
            "vision_encoder.py",
        )
        if not (os.path.exists(config_path) and os.path.exists(index_path)):
            return False
        if not all(os.path.exists(os.path.join(self.cfg.model_path, name)) for name in required_code):
            return False
        try:
            with open(config_path) as f:
                raw = json.load(f)
            if raw.get("model_type") != "step3p7":
                return False
            if not isinstance(raw.get("vision_config"), dict) or not isinstance(raw.get("image_token_id"), int):
                return False
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            return (
                any(key.startswith("vision_model.") for key in weight_map)
                and "vit_large_projector.weight" in weight_map
            )
        except Exception:
            return False

    def supports_gemma_image_inputs(self) -> bool:
        """Return True when the loaded model directory contains Gemma4 image assets."""
        config_path = os.path.join(self.cfg.model_path, "config.json")
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        processor_path = os.path.join(self.cfg.model_path, "processor_config.json")
        if not (os.path.exists(config_path) and os.path.exists(index_path) and os.path.exists(processor_path)):
            return False
        try:
            with open(config_path) as f:
                raw = json.load(f)
            if raw.get("model_type") != "gemma4":
                return False
            if not isinstance(raw.get("vision_config"), dict) or not isinstance(raw.get("image_token_id"), int):
                return False
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            return (
                any(key.startswith("model.vision_tower.") for key in weight_map)
                and "model.embed_vision.embedding_projection.weight" in weight_map
            )
        except Exception:
            return False

    def supports_glm53_image_inputs(self) -> bool:
        """Return True only for the qualified GLM-5.3 BF16 image path."""
        vision_quant = str(
            getattr(getattr(self, "quant_cfg", None), "step_vision_quant", "int4") or "int4"
        ).lower()
        if vision_quant != "bf16":
            return False
        config_path = os.path.join(self.cfg.model_path, "config.json")
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        processor_path = os.path.join(self.cfg.model_path, "processor_config.json")
        if not (os.path.exists(config_path) and os.path.exists(index_path) and os.path.exists(processor_path)):
            return False
        try:
            with open(config_path, encoding="utf-8") as f:
                raw = json.load(f)
            text_cfg = raw.get("text_config") or {}
            if raw.get("model_type") != "glm5_next" or text_cfg.get("model_type") != "glm5_next_text":
                return False
            if not isinstance(raw.get("vision_config"), dict) or not isinstance(raw.get("image_token_id"), int):
                return False
            with open(processor_path, encoding="utf-8") as f:
                processor_raw = json.load(f)
            image_cfg = processor_raw.get("image_processor")
            if not isinstance(image_cfg, dict):
                return False
            vision_cfg = Glm53VisionConfig.from_dict(raw["vision_config"])
            Glm53ImagePreprocessor.from_checkpoint_config(image_cfg, vision_cfg)
            with open(index_path, encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            required = {
                "model.visual.patch_embed.proj.weight",
                "model.visual.blocks.0.attn.qkv.weight",
                "model.visual.merger.proj.weight",
                "model.visual.downsample.weight",
            }
            return required.issubset(weight_map)
        except Exception:
            return False

    def _detect_shared_expert_gate(self) -> bool:
        """Check if model has shared_expert_gate weights (Qwen3-Next sigmoid gate)."""
        import json
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        try:
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            # Check layer 0 for shared_expert_gate
            gate_key = f"{self.cfg.layers_prefix}.layers.0.mlp.shared_expert_gate.weight"
            return gate_key in weight_map
        except (OSError, KeyError):
            return False

    def _require_supported_runtime_features(self) -> None:
        """Fail closed for parsed model features that are not executed yet."""
        return None

    def _extract_openai_images(self, messages_json: str):
        messages = json.loads(messages_json)
        images = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("video", "input_video") or "video" in part:
                    raise ValueError("vision support is image-only; video inputs are not supported")
                if not (ptype in ("image", "image_url", "input_image") or "image" in part or "image_url" in part):
                    continue
                source = part.get("image") or part.get("image_url")
                if isinstance(source, dict):
                    source = source.get("url")
                if not isinstance(source, str) or not source:
                    raise ValueError("image content part must provide an image URL, data URL, or local path")
                images.append(self._load_openai_image_source(source))
        if not images:
            raise ValueError("multimodal request contained no image content parts")
        return images

    def _load_openai_image_source(self, source: str):
        from PIL import Image
        if source.startswith("data:"):
            _, _, data = source.partition(",")
            if not data:
                raise ValueError("invalid image data URL")
            raw = base64.b64decode(data)
            return Image.open(BytesIO(raw)).convert("RGB")
        if source.startswith("file://") or source.startswith("/") or source.startswith("./") or source.startswith("../"):
            if os.environ.get("KRASIS_ALLOW_LOCAL_IMAGE_PATHS") != "1":
                raise ValueError("local image paths are disabled; use a data URL or http(s) URL")
            path = source[7:] if source.startswith("file://") else source
            return Image.open(path).convert("RGB")
        if source.startswith("http://") or source.startswith("https://"):
            from urllib.request import urlopen
            with urlopen(source, timeout=20) as resp:
                raw = resp.read()
            return Image.open(BytesIO(raw)).convert("RGB")
        raise ValueError(f"unsupported image source: {source[:80]}")

    def _ensure_step_vision_modules(self):
        if self._step_vision_modules is not None:
            return self._step_vision_modules

        from transformers import AutoTokenizer

        shim_name = "transformers.tokenization_utils_tokenizers"
        if shim_name not in sys.modules:
            shim = types.ModuleType(shim_name)
            shim.TokenizersBackend = AutoTokenizer
            sys.modules[shim_name] = shim

        model_path = os.path.abspath(self.cfg.model_path)
        package_name = f"krasis_step3p7_{hashlib.sha1(model_path.encode()).hexdigest()[:12]}"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [model_path]
            sys.modules[package_name] = package

        config_mod = importlib.import_module(f"{package_name}.configuration_step3p7")
        vision_mod = importlib.import_module(f"{package_name}.vision_encoder")
        processor_mod = importlib.import_module(f"{package_name}.processing_step3")
        self._step_vision_modules = (config_mod, vision_mod, processor_mod)
        return self._step_vision_modules

    def _ensure_qwen_vision_model(self):
        if self._qwen_vision_processor is None:
            from transformers import AutoProcessor
            self._qwen_vision_processor = AutoProcessor.from_pretrained(
                self.cfg.model_path,
                trust_remote_code=True,
            )
        if self._qwen_vision_model is not None:
            return self._qwen_vision_model

        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        qwen_cfg = Qwen3VLConfig.from_pretrained(self.cfg.model_path)
        log_ram_ledger("before-qwen-vision-load")
        vision = Qwen3VLVisionModel(qwen_cfg.vision_config)
        vision.to(dtype=torch.bfloat16)
        state = {}
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise RuntimeError("Qwen vision loading requires model.safetensors.index.json")
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            if key.startswith("model.visual."):
                shard_to_keys.setdefault(shard, []).append(key)
        if not shard_to_keys:
            raise RuntimeError("No model.visual.* tensors found in safetensors index")
        for shard, keys in shard_to_keys.items():
            shard_path = os.path.join(self.cfg.model_path, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    state[key.removeprefix("model.visual.")] = f.get_tensor(key)
        state_bytes = sum(t.numel() * t.element_size() for t in state.values())
        state_dtypes = sorted({str(t.dtype) for t in state.values()})
        log_ram_ledger("after-qwen-vision-state-load", {"vision_state": state_bytes})
        missing, unexpected = vision.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Qwen vision state mismatch: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        del state
        gc.collect()
        vision.eval()
        vision.requires_grad_(False)
        self._qwen_vision_model = vision
        self._qwen_vision_config = qwen_cfg
        param_bytes = sum(p.numel() * p.element_size() for p in vision.parameters())
        param_dtypes = sorted({str(p.dtype) for p in vision.parameters()})
        buffer_bytes = sum(b.numel() * b.element_size() for b in vision.buffers())
        buffer_dtypes = sorted({str(b.dtype) for b in vision.buffers()})
        log_ram_ledger(
            "after-qwen-vision-load",
            {
                "vision_params": param_bytes,
                "vision_buffers": buffer_bytes,
            },
        )
        logger.info(
            "Loaded Qwen vision tower on CPU: params_mb=%.1f buffers_mb=%.1f "
            "state_mb=%.1f param_dtypes=%s buffer_dtypes=%s state_dtypes=%s",
            param_bytes / (1024 * 1024),
            buffer_bytes / (1024 * 1024),
            state_bytes / (1024 * 1024),
            param_dtypes,
            buffer_dtypes,
            state_dtypes,
        )
        return vision

    @staticmethod
    def _module_param_buffer_bytes(*modules) -> tuple[int, int]:
        param_bytes = 0
        buffer_bytes = 0
        for module in modules:
            if module is None:
                continue
            param_bytes += sum(p.numel() * p.element_size() for p in module.parameters())
            buffer_bytes += sum(b.numel() * b.element_size() for b in module.buffers())
        return int(param_bytes), int(buffer_bytes)

    def _ensure_step_vision_model(self):
        if self._step_vision_processor is None:
            _, _, processor_mod = self._ensure_step_vision_modules()
            if self.tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path, trust_remote_code=True)
                template_path = os.path.join(self.cfg.model_path, "chat_template.jinja")
                if not getattr(tokenizer, "chat_template", None) and os.path.isfile(template_path):
                    with open(template_path, "r", encoding="utf-8") as f:
                        tokenizer.chat_template = f.read()
            else:
                tokenizer = self.tokenizer.tokenizer
            self._step_vision_processor = processor_mod.Step3VLProcessor(tokenizer=tokenizer)

        if self._step_vision_model is not None and self._step_vision_projector is not None:
            return self._step_vision_model, self._step_vision_projector

        config_mod, vision_mod, _ = self._ensure_step_vision_modules()
        step_cfg = config_mod.Step3p7Config.from_pretrained(self.cfg.model_path)
        log_ram_ledger("before-step-vision-load")
        vision = vision_mod.StepRoboticsVisionEncoder(step_cfg.vision_config)
        text_hidden = int(getattr(step_cfg.text_config, "hidden_size", 0) or self.cfg.hidden_size)
        projector = torch.nn.Linear(
            int(step_cfg.vision_config.width) * 4,
            text_hidden,
            bias=bool(getattr(step_cfg, "projector_bias", False)),
        )
        vision.to(dtype=torch.bfloat16)
        projector.to(dtype=torch.bfloat16)

        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise RuntimeError("Step vision loading requires model.safetensors.index.json")
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            if key.startswith("vision_model.") or key.startswith("vit_large_projector."):
                shard_to_keys.setdefault(shard, []).append(key)
        if not shard_to_keys:
            raise RuntimeError("No Step vision/projector tensors found in safetensors index")

        vision_state = {}
        projector_state = {}
        for shard, keys in shard_to_keys.items():
            shard_path = os.path.join(self.cfg.model_path, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    if key.startswith("vision_model."):
                        vision_state[key.removeprefix("vision_model.")] = f.get_tensor(key)
                    elif key.startswith("vit_large_projector."):
                        projector_state[key.removeprefix("vit_large_projector.")] = f.get_tensor(key)
        state_bytes = (
            sum(t.numel() * t.element_size() for t in vision_state.values())
            + sum(t.numel() * t.element_size() for t in projector_state.values())
        )
        state_dtypes = sorted(
            {str(t.dtype) for t in list(vision_state.values()) + list(projector_state.values())}
        )
        log_ram_ledger("after-step-vision-state-load", {"vision_state": state_bytes})
        missing, unexpected = vision.load_state_dict(vision_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Step vision state mismatch: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        missing, unexpected = projector.load_state_dict(projector_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Step vision projector state mismatch: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        del vision_state, projector_state
        gc.collect()
        step_vision_quant = str(getattr(self.quant_cfg, "step_vision_quant", "int4") or "int4").lower()
        step_vision_group_size = int(getattr(self.quant_cfg, "step_vision_group_size", 128))
        quant_stats = None
        if step_vision_quant == "int4":
            vision, projector, quant_stats = quantize_step_vision_modules_int4(
                vision,
                projector,
                group_size=step_vision_group_size,
            )
            gc.collect()
        elif step_vision_quant != "bf16":
            raise RuntimeError(f"Unsupported Step vision quant mode: {step_vision_quant!r}")
        vision.eval()
        projector.eval()
        vision.requires_grad_(False)
        projector.requires_grad_(False)
        self._step_vision_model = vision
        self._step_vision_projector = projector
        self._step_vision_config = step_cfg
        self._step_vision_quant_mode = step_vision_quant
        self._step_vision_quant_stats = quant_stats
        param_bytes, buffer_bytes = self._module_param_buffer_bytes(vision, projector)
        param_dtypes = sorted(
            {str(p.dtype) for p in list(vision.parameters()) + list(projector.parameters())}
        )
        buffer_dtypes = sorted({str(b.dtype) for b in list(vision.buffers()) + list(projector.buffers())})
        log_ram_ledger(
            "after-step-vision-load",
            {
                "vision_params": param_bytes,
                "vision_buffers": buffer_bytes,
            },
        )
        quant_detail = ""
        if quant_stats is not None:
            quant_detail = (
                " int4_modules=%d int4_source_mb=%.1f int4_weight_mb=%.1f int4_bias_mb=%.1f"
                % (
                    quant_stats.modules,
                    quant_stats.source_weight_bytes / (1024 * 1024),
                    quant_stats.quantized_weight_bytes / (1024 * 1024),
                    quant_stats.bf16_bias_bytes / (1024 * 1024),
                )
            )
        logger.info(
            "Loaded Step vision tower on CPU: quant=%s group_size=%d params_mb=%.1f buffers_mb=%.1f "
            "state_mb=%.1f param_dtypes=%s buffer_dtypes=%s state_dtypes=%s%s",
            step_vision_quant,
            step_vision_group_size,
            param_bytes / (1024 * 1024),
            buffer_bytes / (1024 * 1024),
            state_bytes / (1024 * 1024),
            param_dtypes,
            buffer_dtypes,
            state_dtypes,
            quant_detail,
        )
        return vision, projector

    def _ensure_gemma_vision_model(self):
        if self._gemma_vision_processor is None:
            processor_path = os.path.join(self.cfg.model_path, "processor_config.json")
            with open(processor_path, "r", encoding="utf-8") as f:
                processor_cfg = json.load(f)
            image_cfg = processor_cfg.get("image_processor") or {}
            self._gemma_vision_processor = Gemma4ImagePreprocessor(
                patch_size=int(image_cfg.get("patch_size", 16)),
                pooling_kernel_size=int(image_cfg.get("pooling_kernel_size", 3)),
                max_soft_tokens=int(
                    image_cfg.get(
                        "max_soft_tokens",
                        processor_cfg.get("image_seq_length", 280),
                    )
                ),
                rescale_factor=float(image_cfg.get("rescale_factor", 1.0 / 255.0)),
            )

        if self._gemma_vision_model is not None and self._gemma_vision_embedder is not None:
            return self._gemma_vision_model, self._gemma_vision_embedder

        config_path = os.path.join(self.cfg.model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        vision_cfg = Gemma4VisionConfig.from_dict(raw_cfg["vision_config"])
        text_hidden = int((raw_cfg.get("text_config") or {}).get("hidden_size") or self.cfg.hidden_size)

        log_ram_ledger("before-gemma-vision-load")
        vision = Gemma4VisionModel(vision_cfg)
        embedder = Gemma4MultimodalEmbedder(vision_cfg, text_hidden)
        vision.to(dtype=torch.bfloat16)
        embedder.to(dtype=torch.bfloat16)

        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise RuntimeError("Gemma4 vision loading requires model.safetensors.index.json")
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            if key.startswith("model.vision_tower.") or key.startswith("model.embed_vision."):
                shard_to_keys.setdefault(shard, []).append(key)
        if not shard_to_keys:
            raise RuntimeError("No Gemma4 vision/embed_vision tensors found in safetensors index")

        vision_state = {}
        embedder_state = {}
        for shard, keys in shard_to_keys.items():
            shard_path = os.path.join(self.cfg.model_path, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    if key.startswith("model.vision_tower."):
                        vision_state[key.removeprefix("model.vision_tower.")] = f.get_tensor(key)
                    elif key.startswith("model.embed_vision."):
                        embedder_state[key.removeprefix("model.embed_vision.")] = f.get_tensor(key)

        state_bytes = (
            sum(t.numel() * t.element_size() for t in vision_state.values())
            + sum(t.numel() * t.element_size() for t in embedder_state.values())
        )
        state_dtypes = sorted(
            {str(t.dtype) for t in list(vision_state.values()) + list(embedder_state.values())}
        )
        log_ram_ledger("after-gemma-vision-state-load", {"vision_state": state_bytes})
        missing, unexpected = vision.load_state_dict(vision_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Gemma4 vision state mismatch: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        missing, unexpected = embedder.load_state_dict(embedder_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Gemma4 embed_vision state mismatch: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        del vision_state, embedder_state
        gc.collect()

        vision_quant = str(getattr(self.quant_cfg, "step_vision_quant", "int4") or "int4").lower()
        vision_group_size = int(getattr(self.quant_cfg, "step_vision_group_size", 128))
        quant_stats = None
        if vision_quant == "int4":
            (vision, embedder), quant_stats = quantize_vision_modules_int4(
                vision,
                embedder,
                group_size=vision_group_size,
            )
            gc.collect()
        elif vision_quant != "bf16":
            raise RuntimeError(f"Unsupported Gemma4 vision quant mode: {vision_quant!r}")

        vision.eval()
        embedder.eval()
        vision.requires_grad_(False)
        embedder.requires_grad_(False)
        self._gemma_vision_model = vision
        self._gemma_vision_embedder = embedder
        self._gemma_vision_config = vision_cfg
        self._gemma_vision_raw_config = raw_cfg
        self._gemma_vision_quant_mode = vision_quant
        self._gemma_vision_quant_stats = quant_stats

        param_bytes, buffer_bytes = self._module_param_buffer_bytes(vision, embedder)
        param_dtypes = sorted(
            {str(p.dtype) for p in list(vision.parameters()) + list(embedder.parameters())}
        )
        buffer_dtypes = sorted({str(b.dtype) for b in list(vision.buffers()) + list(embedder.buffers())})
        log_ram_ledger(
            "after-gemma-vision-load",
            {
                "vision_params": param_bytes,
                "vision_buffers": buffer_bytes,
            },
        )
        quant_detail = ""
        if quant_stats is not None:
            quant_detail = (
                " int4_modules=%d int4_source_mb=%.1f int4_weight_mb=%.1f int4_bias_mb=%.1f"
                % (
                    quant_stats.modules,
                    quant_stats.source_weight_bytes / (1024 * 1024),
                    quant_stats.quantized_weight_bytes / (1024 * 1024),
                    quant_stats.bf16_bias_bytes / (1024 * 1024),
                )
            )
        logger.info(
            "Loaded Gemma4 vision tower on CPU: quant=%s group_size=%d params_mb=%.1f buffers_mb=%.1f "
            "state_mb=%.1f param_dtypes=%s buffer_dtypes=%s state_dtypes=%s%s",
            vision_quant,
            vision_group_size,
            param_bytes / (1024 * 1024),
            buffer_bytes / (1024 * 1024),
            state_bytes / (1024 * 1024),
            param_dtypes,
            buffer_dtypes,
            state_dtypes,
            quant_detail,
        )
        print(
            "  \033[0;32mGemma4 vision loaded on CPU: "
            f"quant={vision_quant}, group={vision_group_size}, "
            f"params={param_bytes / (1024 * 1024):.1f} MB, "
            f"buffers={buffer_bytes / (1024 * 1024):.1f} MB, "
            f"state={state_bytes / (1024 * 1024):.1f} MB{quant_detail}\033[0m",
            flush=True,
        )
        return vision, embedder

    def _ensure_glm53_vision_model(self):
        vision_quant = str(
            getattr(self.quant_cfg, "step_vision_quant", "int4") or "int4"
        ).lower()
        if vision_quant != "bf16":
            raise RuntimeError(
                "GLM-5.3 image execution is accuracy-qualified only with "
                "CFG_VISION_QUANT=bf16; INT4 vision failed the native-resolution "
                "image-reading acceptance gate"
            )
        if self._glm53_vision_processor is None:
            config_path = os.path.join(self.cfg.model_path, "config.json")
            processor_path = os.path.join(self.cfg.model_path, "processor_config.json")
            with open(config_path, encoding="utf-8") as f:
                raw_cfg = json.load(f)
            with open(processor_path, encoding="utf-8") as f:
                processor_cfg = json.load(f)
            image_cfg = processor_cfg.get("image_processor")
            if not isinstance(image_cfg, dict):
                raise ValueError("GLM-5.3 processor config has no image_processor object")
            vision_cfg = Glm53VisionConfig.from_dict(raw_cfg["vision_config"])
            self._glm53_vision_processor = Glm53ImagePreprocessor.from_checkpoint_config(
                image_cfg,
                vision_cfg,
            )
            self._glm53_vision_config = vision_cfg
            self._glm53_vision_raw_config = raw_cfg

        if self._glm53_vision_model is not None:
            return self._glm53_vision_model

        vision_cfg = self._glm53_vision_config
        log_ram_ledger("before-glm53-vision-load")
        vision = Glm53VisionModel(vision_cfg).to(dtype=torch.bfloat16)
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            if key.startswith("model.visual."):
                shard_to_keys.setdefault(shard, []).append(key)
        if not shard_to_keys:
            raise RuntimeError("No GLM-5.3 model.visual.* tensors found in safetensors index")

        state = {}
        for shard, keys in shard_to_keys.items():
            shard_path = os.path.join(self.cfg.model_path, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    state[key.removeprefix("model.visual.")] = f.get_tensor(key)
        state_bytes = sum(t.numel() * t.element_size() for t in state.values())
        state_dtypes = sorted({str(t.dtype) for t in state.values()})
        missing, unexpected = vision.load_state_dict(state, strict=False)
        missing = [name for name in missing if name != "rotary_inv_freq"]
        if missing or unexpected:
            raise RuntimeError(
                f"GLM-5.3 vision state mismatch: missing={missing[:8]} unexpected={list(unexpected)[:8]}"
            )
        del state
        gc.collect()

        vision.eval()
        vision.requires_grad_(False)
        self._glm53_vision_model = vision
        self._glm53_vision_quant_mode = vision_quant
        self._glm53_vision_quant_stats = None
        param_bytes, buffer_bytes = self._module_param_buffer_bytes(vision)
        log_ram_ledger(
            "after-glm53-vision-load",
            {"vision_params": param_bytes, "vision_buffers": buffer_bytes},
        )
        logger.info(
            "Loaded GLM-5.3 vision tower on CPU: quant=%s params_mb=%.1f "
            "buffers_mb=%.1f state_mb=%.1f state_dtypes=%s",
            vision_quant,
            param_bytes / (1024 * 1024),
            buffer_bytes / (1024 * 1024),
            state_bytes / (1024 * 1024),
            state_dtypes,
        )
        return vision

    def _ensure_deepseek_v4_vision_model(self):
        vision_quant = str(
            getattr(self.quant_cfg, "step_vision_quant", "int4") or "int4"
        ).lower()
        if vision_quant != "bf16":
            raise RuntimeError(
                "DeepSeek-V4-Flash-Vision image execution is initially qualified only "
                "with CFG_VISION_QUANT=bf16"
            )
        if not self.supports_deepseek_v4_image_inputs():
            raise RuntimeError(
                "DeepSeek-V4 checkpoint does not contain a complete Krasis-supported vision inventory"
            )
        if self._deepseek_v4_vision_processor is None:
            self._deepseek_v4_vision_processor = DeepseekV4ImagePreprocessor(self.cfg)
        if self._deepseek_v4_vision_model is not None:
            return (
                self._deepseek_v4_vision_model,
                self._deepseek_v4_vision_aligner,
                self._deepseek_v4_vision_special,
            )

        log_ram_ledger("before-deepseek-v4-vision-load")
        vision = DeepseekV4VisionModel(self.cfg).to(dtype=torch.bfloat16)
        aligner = DeepseekV4VisionAligner(self.cfg).to(dtype=torch.bfloat16)
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        wanted = {
            key: shard
            for key, shard in weight_map.items()
            if key.startswith("vision.")
            or key.startswith("aligner.")
            or key in {"image_start", "image_pad", "image_newline", "image_end"}
        }
        shard_to_keys = {}
        for key, shard in wanted.items():
            shard_to_keys.setdefault(shard, []).append(key)
        tensors = {}
        for shard, keys in shard_to_keys.items():
            with safe_open(
                os.path.join(self.cfg.model_path, shard), framework="pt", device="cpu"
            ) as f:
                for key in keys:
                    tensors[key] = f.get_tensor(key)

        vision_state = {
            key.removeprefix("vision."): value
            for key, value in tensors.items()
            if key.startswith("vision.")
        }
        aligner_state = {
            key.removeprefix("aligner."): value
            for key, value in tensors.items()
            if key.startswith("aligner.")
        }
        vision.load_state_dict(vision_state, strict=True)
        aligner.load_state_dict(aligner_state, strict=True)
        special = torch.stack(
            [
                tensors["image_start"],
                tensors["image_pad"],
                tensors["image_pad"],
                tensors["image_newline"],
                tensors["image_end"],
            ]
        ).to(dtype=torch.bfloat16)
        if special.shape != (5, self.cfg.hidden_size):
            raise RuntimeError(
                f"DeepSeek-V4 vision special embedding shape mismatch: {tuple(special.shape)}"
            )
        state_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        del tensors, vision_state, aligner_state
        gc.collect()

        vision.eval().requires_grad_(False)
        aligner.eval().requires_grad_(False)
        self._deepseek_v4_vision_model = vision
        self._deepseek_v4_vision_aligner = aligner
        self._deepseek_v4_vision_special = special
        self._deepseek_v4_vision_quant_mode = vision_quant
        param_bytes, buffer_bytes = self._module_param_buffer_bytes(vision, aligner)
        log_ram_ledger(
            "after-deepseek-v4-vision-load",
            {"vision_params": param_bytes, "vision_buffers": buffer_bytes},
        )
        logger.info(
            "Loaded DeepSeek-V4 vision tower on CPU: quant=%s params_mb=%.1f "
            "buffers_mb=%.1f state_mb=%.1f",
            vision_quant,
            param_bytes / (1024 * 1024),
            buffer_bytes / (1024 * 1024),
            state_bytes / (1024 * 1024),
        )
        return vision, aligner, special

    def _release_qwen_vision_gpu(self, vision, device, label: str = "after-qwen-vision-release"):
        if getattr(device, "type", None) != "cuda":
            return
        try:
            vision.to("cpu")
        finally:
            torch.cuda.empty_cache()
            log_ram_ledger(label)
            if _vram_ledger_enabled():
                _vram_checkpoint(label, [device])

    def _release_glm53_vision_gpu(self, vision, device, label: str = "after-glm53-vision-release"):
        if getattr(device, "type", None) != "cuda":
            return
        try:
            vision.to("cpu")
        finally:
            torch.cuda.empty_cache()
            log_ram_ledger(label)
            if _vram_ledger_enabled():
                _vram_checkpoint(label, [device])

    def _release_deepseek_v4_vision_gpu(
        self, vision, aligner, special, device, label="after-deepseek-v4-vision-release"
    ):
        if getattr(device, "type", None) != "cuda":
            return
        try:
            vision.to("cpu")
            aligner.to("cpu")
            self._deepseek_v4_vision_special = special.to("cpu")
        finally:
            torch.cuda.empty_cache()
            log_ram_ledger(label)
            if _vram_ledger_enabled():
                _vram_checkpoint(label, [device])

    def _release_gemma_vision_gpu(self, vision, embedder, device, label: str = "after-gemma-vision-release"):
        if getattr(device, "type", None) != "cuda":
            return
        free_before = None
        total = None
        try:
            free_before, total = torch.cuda.mem_get_info(device)
        except Exception:
            pass
        try:
            embedder.to("cpu")
            vision.to("cpu")
        finally:
            torch.cuda.empty_cache()
            try:
                free_after, total_after = torch.cuda.mem_get_info(device)
                logger.info(
                    "Released Gemma4 vision tower from GPU: freed_mb=%d free_before_mb=%d free_after_mb=%d total_vram_mb=%d",
                    int(((free_after - free_before) if free_before is not None else 0) // (1024 * 1024)),
                    int((free_before or 0) // (1024 * 1024)),
                    int(free_after // (1024 * 1024)),
                    int((total or total_after) // (1024 * 1024)),
                )
                print(
                    "  \033[0;32mGemma4 vision released from GPU: "
                    f"freed={int(((free_after - free_before) if free_before is not None else 0) // (1024 * 1024))} MB, "
                    f"free={int((free_before or 0) // (1024 * 1024))}->{int(free_after // (1024 * 1024))} MB\033[0m",
                    flush=True,
                )
            except Exception:
                logger.info("Released Gemma4 vision tower from GPU")
                print("  \033[0;32mGemma4 vision released from GPU\033[0m", flush=True)
            log_ram_ledger(label)
            if _vram_ledger_enabled():
                _vram_checkpoint(label, [device])

    def _release_step_vision_gpu(self, vision, projector, device, label: str = "after-step-vision-release"):
        if getattr(device, "type", None) != "cuda":
            return
        free_before = None
        total = None
        try:
            free_before, total = torch.cuda.mem_get_info(device)
        except Exception:
            pass
        try:
            projector.to("cpu")
            vision.to("cpu")
        finally:
            torch.cuda.empty_cache()
            try:
                free_after, total_after = torch.cuda.mem_get_info(device)
                logger.info(
                    "Released Step vision tower from GPU: freed_mb=%d free_before_mb=%d free_after_mb=%d total_vram_mb=%d",
                    int(((free_after - free_before) if free_before is not None else 0) // (1024 * 1024)),
                    int((free_before or 0) // (1024 * 1024)),
                    int(free_after // (1024 * 1024)),
                    int((total or total_after) // (1024 * 1024)),
                )
            except Exception:
                logger.info("Released Step vision tower from GPU")
            log_ram_ledger(label)
            if _vram_ledger_enabled():
                _vram_checkpoint(label, [device])

    def _qwen_vl_position_ids_and_delta(self, input_ids: torch.Tensor, mm_token_type_ids: torch.Tensor, image_grid_thw: torch.Tensor):
        spatial_merge_size = int(self._qwen_vision_config.vision_config.spatial_merge_size)
        grid_iter = iter(image_grid_thw.tolist())
        groups = []
        types = mm_token_type_ids.tolist()
        if not types:
            raise ValueError("empty mm_token_type_ids")
        start = 0
        cur = types[0]
        for idx, val in enumerate(types[1:], 1):
            if val != cur:
                groups.append((cur, start, idx))
                start = idx
                cur = val
        groups.append((cur, start, len(types)))

        current_pos = 0
        chunks = []
        device = input_ids.device
        for modality, start_idx, end_idx in groups:
            if modality == 0:
                text_len = end_idx - start_idx
                pos = torch.arange(text_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + current_pos
                chunks.append(pos)
                current_pos += text_len
            elif modality == 1:
                t, h, w = next(grid_iter)
                llm_t = int(t)
                llm_h = int(h) // spatial_merge_size
                llm_w = int(w) // spatial_merge_size
                pos_t = torch.arange(llm_t, device=device, dtype=torch.long).repeat_interleave(llm_h * llm_w) + current_pos
                pos_h = torch.arange(llm_h, device=device, dtype=torch.long).repeat_interleave(llm_w).repeat(llm_t) + current_pos
                pos_w = torch.arange(llm_w, device=device, dtype=torch.long).repeat(llm_h * llm_t) + current_pos
                chunks.append(torch.stack([pos_t, pos_h, pos_w], dim=0))
                current_pos += max(int(h), int(w)) // spatial_merge_size
            else:
                raise ValueError("video token type ids are not supported")
        position_ids = torch.cat(chunks, dim=1)
        rope_delta = int(position_ids.max().item() + 1 - int(input_ids.numel()))
        return position_ids, rope_delta

    def _qwen_vl_mrope_cos_sin(self, position_ids: torch.Tensor):
        rope_half = self.cfg.rotary_dim // 2
        rope_dim = rope_half * 2
        theta = float(self.cfg.rope_theta)
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=position_ids.device, dtype=torch.float32) / rope_dim))
        freqs = position_ids.to(torch.float32)[:, :, None] * inv_freq[None, None, :]
        freqs_t = freqs[0].clone()
        rope_params = getattr(self.cfg, "rope_scaling", {}) or {}
        sections = rope_params.get("mrope_section") or [rope_half // 3, rope_half // 3, rope_half - 2 * (rope_half // 3)]
        for dim, offset in ((1, 1), (2, 2)):
            length = int(sections[dim]) * 3
            freqs_t[:, offset:length:3] = freqs[dim, :, offset:length:3]
        return freqs_t.cos().contiguous(), freqs_t.sin().contiguous()

    def _build_qwen_multimodal_prefill_inputs(self, messages_json: str, rendered_prompt: str):
        """Build GPU inputs_embeds for Qwen image prompts.

        This method is intentionally called only from the image request path.
        Text-only requests keep using Rust token-id prefill directly.
        """
        if self.embedding is None:
            raise RuntimeError("Model embedding is not loaded")
        images = self._extract_openai_images(messages_json)
        vision = self._ensure_qwen_vision_model()
        processor = self._qwen_vision_processor
        device = self.embedding.device
        dtype = torch.bfloat16

        batch = processor(
            text=[rendered_prompt],
            images=images,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
        if "mm_token_type_ids" not in batch:
            if not hasattr(processor, "create_mm_token_type_ids"):
                raise ValueError("Qwen processor did not return mm_token_type_ids")
            batch["mm_token_type_ids"] = processor.create_mm_token_type_ids(batch["input_ids"])
        merge_size = int(getattr(vision, "spatial_merge_size", 2))
        expected_image_tokens = int((batch["image_grid_thw"].prod(-1) // (merge_size * merge_size)).sum().item())
        vision_param_bytes = sum(p.numel() * p.element_size() for p in vision.parameters())
        vision_buffer_bytes = sum(b.numel() * b.element_size() for b in vision.buffers())

        try:
            if getattr(device, "type", None) == "cuda":
                free_before, total = torch.cuda.mem_get_info(device)
                logger.info(
                    "Qwen image request staging: images=%d image_tokens=%d free_vram_mb=%d total_vram_mb=%d vision_params_mb=%.1f vision_buffers_mb=%.1f",
                    len(images),
                    expected_image_tokens,
                    int(free_before // (1024 * 1024)),
                    int(total // (1024 * 1024)),
                    vision_param_bytes / (1024 * 1024),
                    vision_buffer_bytes / (1024 * 1024),
                )
            input_ids = batch["input_ids"][0].to(device=device, dtype=torch.long)
            mm_token_type_ids = batch["mm_token_type_ids"][0].to(device=device, dtype=torch.long)
            image_grid_thw = batch["image_grid_thw"].to(device=device, dtype=torch.long)
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)

            log_ram_ledger("before-qwen-vision-to-gpu")
            if _vram_ledger_enabled():
                _vram_checkpoint("before-qwen-vision-to-gpu", [device])
            vision = vision.to(device=device, dtype=dtype)
            if _vram_ledger_enabled():
                _vram_checkpoint("after-qwen-vision-to-gpu", [device])
            with torch.inference_mode():
                vision_output = vision(pixel_values, grid_thw=image_grid_thw, return_dict=True)
                candidates: List[torch.Tensor] = []
                if hasattr(vision_output, "pooler_output"):
                    pooler = vision_output.pooler_output
                    if isinstance(pooler, torch.Tensor):
                        candidates.append(pooler)
                    elif isinstance(pooler, (list, tuple)):
                        candidates.extend(t for t in pooler if isinstance(t, torch.Tensor))
                if isinstance(vision_output, (list, tuple)):
                    candidates.extend(t for t in vision_output if isinstance(t, torch.Tensor))
                    for item in vision_output:
                        if isinstance(item, (list, tuple)):
                            candidates.extend(t for t in item if isinstance(t, torch.Tensor))
                text_hidden = int(self.embedding.shape[1])
                image_embeds = next(
                    (
                        t for t in candidates
                        if t.ndim == 2 and int(t.shape[0]) == expected_image_tokens and int(t.shape[1]) == text_hidden
                    ),
                    None,
                )
                if image_embeds is None:
                    shapes = [tuple(t.shape) for t in candidates]
                    raise RuntimeError(
                        f"could not find Qwen vision pooled embeddings; expected "
                        f"({expected_image_tokens}, {text_hidden}), candidates={shapes}"
                    )
                image_embeds = image_embeds.to(device=device, dtype=self.embedding.dtype)
                inputs_embeds = self.embedding[input_ids].clone()
                image_mask = input_ids == int(self._qwen_vision_config.image_token_id)
                if int(image_mask.sum().item()) != int(image_embeds.shape[0]):
                    raise RuntimeError(
                        f"Image feature/token mismatch: tokens={int(image_mask.sum().item())} features={int(image_embeds.shape[0])}"
                    )
                inputs_embeds[image_mask] = image_embeds
                position_ids, rope_delta = self._qwen_vl_position_ids_and_delta(
                    input_ids,
                    mm_token_type_ids,
                    image_grid_thw,
                )
                mrope_cos, mrope_sin = self._qwen_vl_mrope_cos_sin(position_ids)
                if getattr(device, "type", None) == "cuda":
                    # Rust prefill consumes these CUDA pointers on its own stream.
                    # Make the PyTorch-produced tensors visible before handoff.
                    torch.cuda.synchronize(device)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            self._last_multimodal_prefill_tensors = None
            self._release_qwen_vision_gpu(vision, device, "after-qwen-vision-oom-release")
            free_mb = -1
            total_mb = -1
            if getattr(device, "type", None) == "cuda":
                try:
                    free_after, total = torch.cuda.mem_get_info(device)
                    free_mb = int(free_after // (1024 * 1024))
                    total_mb = int(total // (1024 * 1024))
                except Exception:
                    pass
            raise KrasisVisionVramError(
                "VRAM is too constrained for this Qwen image request. "
                f"Transient BF16 vision staging needs about {vision_param_bytes / (1024 * 1024):.1f} MB "
                f"for vision parameters plus image activations and multimodal prefill scratch; "
                f"free_vram_mb_after_cleanup={free_mb}, total_vram_mb={total_mb}, "
                f"images={len(images)}, image_tokens={expected_image_tokens}. "
                "Use a smaller/fewer images or run this model on a GPU with more free VRAM."
            ) from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._last_multimodal_prefill_tensors = None
                self._release_qwen_vision_gpu(vision, device, "after-qwen-vision-oom-release")
                raise KrasisVisionVramError(
                    "VRAM is too constrained for this Qwen image request. "
                    f"Transient BF16 vision staging needs about {vision_param_bytes / (1024 * 1024):.1f} MB "
                    f"for vision parameters plus image activations and multimodal prefill scratch; "
                    f"images={len(images)}, image_tokens={expected_image_tokens}. "
                    "Use a smaller/fewer images or run this model on a GPU with more free VRAM."
                ) from e
            self._release_qwen_vision_gpu(vision, device)
            raise
        except Exception:
            self._release_qwen_vision_gpu(vision, device)
            raise

        self._release_qwen_vision_gpu(vision, device)

        self._last_multimodal_prefill_tensors = (inputs_embeds, mrope_cos, mrope_sin)
        return {
            "token_ids": [int(x) for x in input_ids.detach().cpu().tolist()],
            "prompt_tokens": int(input_ids.numel()),
            "hidden_size": int(inputs_embeds.shape[-1]),
            "inputs_embeds_ptr": int(inputs_embeds.data_ptr()),
            "mrope_cos_ptr": int(mrope_cos.data_ptr()),
            "mrope_sin_ptr": int(mrope_sin.data_ptr()),
            "mrope_half_dim": int(mrope_cos.shape[-1]),
            "rope_delta": int(rope_delta),
            "vision_block_ids_ptr": 0,
            "vision_visible_left_ptr": 0,
            "vision_visible_right_ptr": 0,
            "vision_max_tokens": 0,
            "image_count": int(len(images)),
            "image_tokens": int((input_ids == int(self._qwen_vision_config.image_token_id)).sum().item()),
        }

    def _step_process_image_features(self, image_features: torch.Tensor, projector: torch.nn.Module) -> torch.Tensor:
        if image_features is None:
            return None
        bsz, patches = image_features.shape[:2]
        hw = int(patches ** 0.5)
        if hw * hw != int(patches):
            raise RuntimeError(f"Step vision features must be square, got {patches} patches")
        image_features = image_features.permute(0, 2, 1).contiguous().view(bsz, -1, hw, hw)
        image_features = self._step_vision_model.vit_downsampler1(image_features)
        image_features = self._step_vision_model.vit_downsampler2(image_features)
        bsz, channels, hw, _ = image_features.shape
        image_features = image_features.view(bsz, channels, hw * hw).permute(0, 2, 1).contiguous()
        return projector(image_features)

    def _build_step_multimodal_prefill_inputs(self, messages_json: str, rendered_prompt: str):
        """Build GPU inputs_embeds for Step image prompts.

        Step uses normal text RoPE, so this returns zero MRoPE pointers.
        """
        if self.embedding is None:
            raise RuntimeError("Model embedding is not loaded")
        images = self._extract_openai_images(messages_json)
        vision, projector = self._ensure_step_vision_model()
        processor = self._step_vision_processor
        device = self.embedding.device
        dtype = torch.bfloat16

        batch = processor(
            text=[rendered_prompt],
            images=images,
            return_tensors="pt",
        )
        image_token_id = int(self._step_vision_config.image_token_id)
        input_ids_cpu = batch["input_ids"][0].to(dtype=torch.long)
        expected_image_tokens = int((input_ids_cpu == image_token_id).sum().item())
        vision_param_bytes, vision_buffer_bytes = self._module_param_buffer_bytes(vision, projector)
        vision_resident_bytes = vision_param_bytes + vision_buffer_bytes
        vision_quant = str(getattr(self, "_step_vision_quant_mode", None) or getattr(self.quant_cfg, "step_vision_quant", "bf16"))
        num_patches = batch.get("num_patches", [])
        if hasattr(num_patches, "detach"):
            num_patches = [int(x) for x in num_patches.detach().cpu().view(-1).tolist()]
        else:
            num_patches = [int(x) for x in num_patches]

        try:
            if getattr(device, "type", None) == "cuda":
                free_before, total = torch.cuda.mem_get_info(device)
                logger.info(
                    "Step image request staging: quant=%s images=%d image_tokens=%d patches=%d free_vram_mb=%d total_vram_mb=%d vision_resident_mb=%.1f vision_params_mb=%.1f vision_buffers_mb=%.1f",
                    vision_quant,
                    len(images),
                    expected_image_tokens,
                    sum(num_patches),
                    int(free_before // (1024 * 1024)),
                    int(total // (1024 * 1024)),
                    vision_resident_bytes / (1024 * 1024),
                    vision_param_bytes / (1024 * 1024),
                    vision_buffer_bytes / (1024 * 1024),
                )
            input_ids = input_ids_cpu.to(device=device, dtype=torch.long)
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            patch_pixel_values = batch.get("patch_pixel_values")
            if patch_pixel_values is not None and int(patch_pixel_values.shape[0]) > 0:
                patch_pixel_values = patch_pixel_values.to(device=device, dtype=dtype)
            else:
                patch_pixel_values = None

            log_ram_ledger("before-step-vision-to-gpu")
            if _vram_ledger_enabled():
                _vram_checkpoint("before-step-vision-to-gpu", [device])
            vision = vision.to(device=device, dtype=dtype)
            projector = projector.to(device=device, dtype=dtype)
            if _vram_ledger_enabled():
                _vram_checkpoint("after-step-vision-to-gpu", [device])
            with torch.inference_mode():
                image_features = vision(pixel_values)
                image_embeds = self._step_process_image_features(image_features, projector)
                patch_embeds = None
                if patch_pixel_values is not None:
                    patch_features = vision(patch_pixel_values)
                    patch_embeds = self._step_process_image_features(patch_features, projector)

                merged = []
                cur_patch_idx = 0
                for image_idx, num_patch in enumerate(num_patches):
                    parts = []
                    if num_patch > 0:
                        if patch_embeds is None:
                            raise RuntimeError("Step processor returned patch count without patch pixels")
                        patch_slice = patch_embeds[cur_patch_idx:cur_patch_idx + num_patch]
                        parts.append(patch_slice.reshape(-1, patch_slice.shape[-1]))
                    parts.append(image_embeds[image_idx].reshape(-1, image_embeds.shape[-1]))
                    cur_patch_idx += num_patch
                    merged.append(torch.cat(parts, dim=0) if len(parts) > 1 else parts[0])
                image_embeds = torch.cat(merged, dim=0) if len(merged) > 1 else merged[0]
                image_embeds = image_embeds.to(device=device, dtype=self.embedding.dtype)

                inputs_embeds = self.embedding[input_ids].clone()
                image_mask = input_ids == image_token_id
                if int(image_mask.sum().item()) != int(image_embeds.shape[0]):
                    raise RuntimeError(
                        f"Step image feature/token mismatch: tokens={int(image_mask.sum().item())} "
                        f"features={int(image_embeds.shape[0])}"
                    )
                inputs_embeds[image_mask] = image_embeds
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.synchronize(device)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            self._last_multimodal_prefill_tensors = None
            self._release_step_vision_gpu(vision, projector, device, "after-step-vision-oom-release")
            free_mb = -1
            total_mb = -1
            if getattr(device, "type", None) == "cuda":
                try:
                    free_after, total = torch.cuda.mem_get_info(device)
                    free_mb = int(free_after // (1024 * 1024))
                    total_mb = int(total // (1024 * 1024))
                except Exception:
                    pass
            raise KrasisVisionVramError(
                "VRAM is too constrained for this Step image request. "
                f"Transient {vision_quant} vision staging needs about {vision_resident_bytes / (1024 * 1024):.1f} MB "
                f"for vision parameters plus image activations and multimodal prefill scratch; "
                f"free_vram_mb_after_cleanup={free_mb}, total_vram_mb={total_mb}, "
                f"images={len(images)}, image_tokens={expected_image_tokens}, patches={sum(num_patches)}. "
                "Use a smaller/fewer images or run this model on a GPU with more free VRAM."
            ) from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._last_multimodal_prefill_tensors = None
                self._release_step_vision_gpu(vision, projector, device, "after-step-vision-oom-release")
                raise KrasisVisionVramError(
                    "VRAM is too constrained for this Step image request. "
                    f"Transient {vision_quant} vision staging needs about {vision_resident_bytes / (1024 * 1024):.1f} MB "
                    f"for vision parameters plus image activations and multimodal prefill scratch; "
                    f"images={len(images)}, image_tokens={expected_image_tokens}, patches={sum(num_patches)}. "
                    "Use a smaller/fewer images or run this model on a GPU with more free VRAM."
                ) from e
            self._release_step_vision_gpu(vision, projector, device)
            raise
        except Exception:
            self._release_step_vision_gpu(vision, projector, device)
            raise

        self._release_step_vision_gpu(vision, projector, device)

        self._last_multimodal_prefill_tensors = (inputs_embeds,)
        return {
            "token_ids": [int(x) for x in input_ids.detach().cpu().tolist()],
            "prompt_tokens": int(input_ids.numel()),
            "hidden_size": int(inputs_embeds.shape[-1]),
            "inputs_embeds_ptr": int(inputs_embeds.data_ptr()),
            "mrope_cos_ptr": 0,
            "mrope_sin_ptr": 0,
            "mrope_half_dim": 0,
            "rope_delta": 0,
            "vision_block_ids_ptr": 0,
            "vision_visible_left_ptr": 0,
            "vision_visible_right_ptr": 0,
            "vision_max_tokens": 0,
            "image_count": int(len(images)),
            "image_tokens": int(expected_image_tokens),
        }

    def _gemma_expand_image_token_ids(self, token_ids: List[int], num_soft_tokens: List[int]):
        raw_cfg = self._gemma_vision_raw_config or {}
        image_token_id = int(raw_cfg.get("image_token_id", 258880))
        boi_token_id = int(raw_cfg.get("boi_token_id", 255999))
        eoi_token_id = int(raw_cfg.get("eoi_token_id", 258882))
        expanded = []
        vision_block_ids = []
        image_idx = 0
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id != image_token_id:
                expanded.append(token_id)
                vision_block_ids.append(-1)
                continue
            if image_idx >= len(num_soft_tokens):
                raise ValueError(
                    f"Gemma4 prompt contains more image placeholders than supplied images: "
                    f"placeholder_index={image_idx}, images={len(num_soft_tokens)}"
                )
            soft = int(num_soft_tokens[image_idx])
            if soft <= 0:
                raise ValueError(f"Gemma4 image {image_idx} produced no soft tokens")
            expanded.append(boi_token_id)
            vision_block_ids.append(-1)
            expanded.extend([image_token_id] * soft)
            vision_block_ids.extend([image_idx] * soft)
            expanded.append(eoi_token_id)
            vision_block_ids.append(-1)
            image_idx += 1
        if image_idx != len(num_soft_tokens):
            raise ValueError(
                f"Gemma4 prompt/image count mismatch: placeholders={image_idx}, images={len(num_soft_tokens)}"
            )
        return expanded, vision_block_ids

    def _build_gemma_multimodal_prefill_inputs(self, messages_json: str, rendered_prompt: str):
        """Build GPU inputs_embeds for Gemma4 image prompts.

        Gemma4 uses normal text RoPE, but its sliding-attention layers need a
        bidirectional overlay inside each image soft-token block. The returned
        block-id tensor lets the Rust prefill path apply that mask while keeping
        full-attention layers causal.
        """
        if self.embedding is None:
            raise RuntimeError("Model embedding is not loaded")
        images = self._extract_openai_images(messages_json)
        vision, embedder = self._ensure_gemma_vision_model()
        processor = self._gemma_vision_processor
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(self.cfg.model_path)

        device = self.embedding.device
        dtype = torch.bfloat16
        batch = processor(images)
        num_soft_tokens = [int(x) for x in batch["num_soft_tokens_per_image"]]
        raw_ids = self.tokenizer.encode(rendered_prompt, add_special_tokens=False)
        expanded_ids, vision_block_ids_cpu = self._gemma_expand_image_token_ids(raw_ids, num_soft_tokens)

        raw_cfg = self._gemma_vision_raw_config or {}
        image_token_id = int(raw_cfg.get("image_token_id", 258880))
        pad_token_id = int((raw_cfg.get("text_config") or {}).get("pad_token_id", 0))
        expected_image_tokens = int(sum(num_soft_tokens))
        vision_param_bytes, vision_buffer_bytes = self._module_param_buffer_bytes(vision, embedder)
        vision_resident_bytes = vision_param_bytes + vision_buffer_bytes
        vision_quant = str(getattr(self, "_gemma_vision_quant_mode", None) or getattr(self.quant_cfg, "step_vision_quant", "int4"))

        try:
            if getattr(device, "type", None) == "cuda":
                free_before, total = torch.cuda.mem_get_info(device)
                logger.info(
                    "Gemma4 image request staging: quant=%s images=%d image_tokens=%d raw_prompt_tokens=%d expanded_prompt_tokens=%d free_vram_mb=%d total_vram_mb=%d vision_resident_mb=%.1f vision_params_mb=%.1f vision_buffers_mb=%.1f",
                    vision_quant,
                    len(images),
                    expected_image_tokens,
                    len(raw_ids),
                    len(expanded_ids),
                    int(free_before // (1024 * 1024)),
                    int(total // (1024 * 1024)),
                    vision_resident_bytes / (1024 * 1024),
                    vision_param_bytes / (1024 * 1024),
                    vision_buffer_bytes / (1024 * 1024),
                )
                print(
                    "  \033[0;32mGemma4 vision staging: "
                    f"quant={vision_quant}, images={len(images)}, image_tokens={expected_image_tokens}, "
                    f"prompt={len(raw_ids)}->{len(expanded_ids)} tokens, "
                    f"free_vram={int(free_before // (1024 * 1024))} MB, "
                    f"resident={vision_resident_bytes / (1024 * 1024):.1f} MB\033[0m",
                    flush=True,
                )
            input_ids = torch.tensor(expanded_ids, device=device, dtype=torch.long)
            vision_block_ids = torch.tensor(vision_block_ids_cpu, device=device, dtype=torch.int32)
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            image_position_ids = batch["image_position_ids"].to(device=device, dtype=torch.long)

            log_ram_ledger("before-gemma-vision-to-gpu")
            if _vram_ledger_enabled():
                _vram_checkpoint("before-gemma-vision-to-gpu", [device])
            vision = vision.to(device=device, dtype=dtype)
            embedder = embedder.to(device=device, dtype=dtype)
            if _vram_ledger_enabled():
                _vram_checkpoint("after-gemma-vision-to-gpu", [device])
            with torch.inference_mode():
                image_features = vision(pixel_values, image_position_ids)
                image_embeds = embedder(image_features).to(device=device, dtype=self.embedding.dtype)

                llm_input_ids = input_ids.clone()
                image_mask = input_ids == image_token_id
                if int(image_mask.sum().item()) != int(image_embeds.shape[0]):
                    raise RuntimeError(
                        f"Gemma4 image feature/token mismatch: tokens={int(image_mask.sum().item())} "
                        f"features={int(image_embeds.shape[0])}"
                    )
                llm_input_ids[image_mask] = pad_token_id
                inputs_embeds = self.embedding[llm_input_ids].clone()
                embed_scale = float(getattr(self.cfg, "embedding_scale", 1.0) or 1.0)
                if embed_scale != 1.0:
                    inputs_embeds.mul_(embed_scale)
                inputs_embeds[image_mask] = image_embeds
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.synchronize(device)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            self._last_multimodal_prefill_tensors = None
            self._release_gemma_vision_gpu(vision, embedder, device, "after-gemma-vision-oom-release")
            free_mb = -1
            total_mb = -1
            if getattr(device, "type", None) == "cuda":
                try:
                    free_after, total = torch.cuda.mem_get_info(device)
                    free_mb = int(free_after // (1024 * 1024))
                    total_mb = int(total // (1024 * 1024))
                except Exception:
                    pass
            raise KrasisVisionVramError(
                "VRAM is too constrained for this Gemma4 image request. "
                f"Transient {vision_quant} vision staging needs about {vision_resident_bytes / (1024 * 1024):.1f} MB "
                f"for vision parameters plus image activations and multimodal prefill scratch; "
                f"free_vram_mb_after_cleanup={free_mb}, total_vram_mb={total_mb}, "
                f"images={len(images)}, image_tokens={expected_image_tokens}. "
                "Use a smaller/fewer images or run this model on a GPU with more free VRAM."
            ) from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._last_multimodal_prefill_tensors = None
                self._release_gemma_vision_gpu(vision, embedder, device, "after-gemma-vision-oom-release")
                raise KrasisVisionVramError(
                    "VRAM is too constrained for this Gemma4 image request. "
                    f"Transient {vision_quant} vision staging needs about {vision_resident_bytes / (1024 * 1024):.1f} MB "
                    f"for vision parameters plus image activations and multimodal prefill scratch; "
                    f"images={len(images)}, image_tokens={expected_image_tokens}. "
                    "Use a smaller/fewer images or run this model on a GPU with more free VRAM."
                ) from e
            self._release_gemma_vision_gpu(vision, embedder, device)
            raise
        except Exception:
            self._release_gemma_vision_gpu(vision, embedder, device)
            raise

        self._release_gemma_vision_gpu(vision, embedder, device)

        self._last_multimodal_prefill_tensors = (inputs_embeds, vision_block_ids)
        return {
            "token_ids": [int(x) for x in input_ids.detach().cpu().tolist()],
            "prompt_tokens": int(input_ids.numel()),
            "hidden_size": int(inputs_embeds.shape[-1]),
            "inputs_embeds_ptr": int(inputs_embeds.data_ptr()),
            "mrope_cos_ptr": 0,
            "mrope_sin_ptr": 0,
            "mrope_half_dim": 0,
            "rope_delta": 0,
            "vision_block_ids_ptr": int(vision_block_ids.data_ptr()),
            "vision_visible_left_ptr": 0,
            "vision_visible_right_ptr": 0,
            "vision_max_tokens": 0,
            "image_count": int(len(images)),
            "image_tokens": int(expected_image_tokens),
        }

    def _build_glm53_multimodal_prefill_inputs(self, messages_json: str, rendered_prompt: str):
        """Build text-width image embeddings for a GLM-5.3 image request."""
        if self.embedding is None:
            raise RuntimeError("Model embedding is not loaded")
        images = self._extract_openai_images(messages_json)
        vision = self._ensure_glm53_vision_model()
        processor = self._glm53_vision_processor
        raw_cfg = self._glm53_vision_raw_config or {}
        vision_cfg = self._glm53_vision_config
        device = self.embedding.device
        dtype = torch.bfloat16

        batch = processor(images)
        image_grid_thw = batch["image_grid_thw"]
        image_token_counts = [
            int(t * h * w) // (vision_cfg.spatial_merge_size**2)
            for t, h, w in image_grid_thw.tolist()
        ]
        placeholder = "<|image|>"
        placeholder_count = rendered_prompt.count(placeholder)
        if placeholder_count != len(images):
            raise ValueError(
                f"GLM-5.3 prompt/image count mismatch: placeholders={placeholder_count} images={len(images)}"
            )
        expanded_prompt = rendered_prompt
        for image_tokens in image_token_counts:
            expanded_prompt = expanded_prompt.replace(placeholder, placeholder * image_tokens, 1)

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(self.cfg.model_path)
        input_ids_cpu = self.tokenizer.encode(expanded_prompt, add_special_tokens=False)
        input_ids_cpu = torch.tensor(input_ids_cpu, dtype=torch.long)
        image_token_id = int(raw_cfg["image_token_id"])
        expected_image_tokens = int(sum(image_token_counts))
        actual_image_tokens = int((input_ids_cpu == image_token_id).sum().item())
        if actual_image_tokens != expected_image_tokens:
            raise RuntimeError(
                f"GLM-5.3 image placeholder expansion mismatch: tokens={actual_image_tokens} "
                f"expected={expected_image_tokens}"
            )

        vision_param_bytes, vision_buffer_bytes = self._module_param_buffer_bytes(vision)
        vision_resident_bytes = vision_param_bytes + vision_buffer_bytes
        vision_quant = str(
            getattr(self, "_glm53_vision_quant_mode", None)
            or getattr(self.quant_cfg, "step_vision_quant", "int4")
        )
        try:
            if getattr(device, "type", None) == "cuda":
                free_before, total = torch.cuda.mem_get_info(device)
                logger.info(
                    "GLM-5.3 image request staging: quant=%s images=%d image_tokens=%d "
                    "prompt_tokens=%d free_vram_mb=%d total_vram_mb=%d vision_resident_mb=%.1f",
                    vision_quant,
                    len(images),
                    expected_image_tokens,
                    int(input_ids_cpu.numel()),
                    int(free_before // (1024 * 1024)),
                    int(total // (1024 * 1024)),
                    vision_resident_bytes / (1024 * 1024),
                )
            input_ids = input_ids_cpu.to(device=device)
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            image_grid_thw = image_grid_thw.to(device=device)
            log_ram_ledger("before-glm53-vision-to-gpu")
            if _vram_ledger_enabled():
                _vram_checkpoint("before-glm53-vision-to-gpu", [device])
            vision = vision.to(device=device, dtype=dtype)
            if _vram_ledger_enabled():
                _vram_checkpoint("after-glm53-vision-to-gpu", [device])
            with torch.inference_mode():
                image_embeds = vision(pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(device=device, dtype=self.embedding.dtype)
                image_mask = input_ids == image_token_id
                if int(image_mask.sum().item()) != int(image_embeds.shape[0]):
                    raise RuntimeError(
                        f"GLM-5.3 image feature/token mismatch: tokens={int(image_mask.sum().item())} "
                        f"features={int(image_embeds.shape[0])}"
                    )
                inputs_embeds = self.embedding[input_ids].clone()
                inputs_embeds[image_mask] = image_embeds
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.synchronize(device)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            self._last_multimodal_prefill_tensors = None
            self._release_glm53_vision_gpu(vision, device, "after-glm53-vision-oom-release")
            raise KrasisVisionVramError(
                "VRAM is too constrained for this GLM-5.3 image request. "
                f"Transient {vision_quant} vision staging needs about "
                f"{vision_resident_bytes / (1024 * 1024):.1f} MB for vision parameters plus "
                f"image activations and multimodal prefill scratch; images={len(images)}, "
                f"image_tokens={expected_image_tokens}. Use a smaller/fewer images or a GPU "
                "with more free VRAM."
            ) from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._last_multimodal_prefill_tensors = None
                self._release_glm53_vision_gpu(vision, device, "after-glm53-vision-oom-release")
                raise KrasisVisionVramError(
                    "VRAM is too constrained for this GLM-5.3 image request. "
                    f"images={len(images)}, image_tokens={expected_image_tokens}. "
                    "Use a smaller/fewer images or a GPU with more free VRAM."
                ) from e
            self._release_glm53_vision_gpu(vision, device)
            raise
        except Exception:
            self._release_glm53_vision_gpu(vision, device)
            raise

        self._release_glm53_vision_gpu(vision, device)
        self._last_multimodal_prefill_tensors = (inputs_embeds,)
        return {
            "token_ids": [int(x) for x in input_ids.detach().cpu().tolist()],
            "prompt_tokens": int(input_ids.numel()),
            "hidden_size": int(inputs_embeds.shape[-1]),
            "inputs_embeds_ptr": int(inputs_embeds.data_ptr()),
            "mrope_cos_ptr": 0,
            "mrope_sin_ptr": 0,
            "mrope_half_dim": 0,
            "rope_delta": 0,
            "vision_block_ids_ptr": 0,
            "vision_visible_left_ptr": 0,
            "vision_visible_right_ptr": 0,
            "vision_max_tokens": 0,
            "image_count": int(len(images)),
            "image_tokens": expected_image_tokens,
        }

    @staticmethod
    def _validate_deepseek_v4_image_roles(messages_json: str) -> None:
        """Allow images only in turns rendered as DeepSeek user input.

        DeepSeek-V4 has no standalone tool role. The bundled chat template
        nests OpenAI ``tool`` messages in a user ``<tool_result>`` block, so
        image-bearing tool results are valid model input. Other non-user roles
        remain rejected fail-closed.
        """
        messages = json.loads(messages_json)
        allowed_image_roles = {"user", "tool"}
        for message_idx, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                is_image = (
                    part_type in ("image", "image_url", "input_image")
                    or "image" in part
                    or "image_url" in part
                )
                if is_image and message.get("role") not in allowed_image_roles:
                    raise ValueError(
                        "DeepSeek-V4-Flash-Vision-Exp accepts images only in "
                        "user messages or tool results; "
                        f"message {message_idx} has role "
                        f"{message.get('role')!r}"
                    )

    def _build_deepseek_v4_multimodal_prefill_inputs(
        self, messages_json: str, rendered_prompt: str
    ):
        """Build checkpoint-native image blocks and BF16 embeddings for V4 Vision."""
        self._validate_deepseek_v4_image_roles(messages_json)
        if self.embedding is None:
            raise RuntimeError("Model embedding is not loaded")
        images = self._extract_openai_images(messages_json)
        vision, aligner, special_cpu = self._ensure_deepseek_v4_vision_model()
        processor = self._deepseek_v4_vision_processor
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(self.cfg.model_path)
        placeholder_id = self.tokenizer.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if placeholder_id is None or placeholder_id == self.tokenizer.tokenizer.unk_token_id:
            raise ValueError(
                f"DeepSeek-V4 image placeholder is absent from tokenizer: {IMAGE_PLACEHOLDER}"
            )
        raw_ids = self.tokenizer.encode(rendered_prompt, add_special_tokens=False)
        if raw_ids.count(placeholder_id) != len(images):
            raise ValueError(
                "DeepSeek-V4 prompt/image count mismatch: "
                f"placeholders={raw_ids.count(placeholder_id)} images={len(images)}"
            )

        expanded_ids = []
        image_inputs = []
        image_visible_tokens = []
        image_iter = iter(images)
        for token_id in raw_ids:
            if token_id != placeholder_id:
                expanded_ids.append(int(token_id))
                continue
            prepared = processor.prepare(next(image_iter), len(expanded_ids))
            image_inputs.append(prepared)
            expanded_ids.extend(
                (self.cfg.vocab_size + prepared.types).tolist()
            )
        expected_image_tokens = sum(int(item.types.numel()) for item in image_inputs)
        device = self.embedding.device
        dtype = self.embedding.dtype
        param_bytes, buffer_bytes = self._module_param_buffer_bytes(vision, aligner)
        resident_bytes = param_bytes + buffer_bytes + special_cpu.numel() * special_cpu.element_size()

        try:
            if getattr(device, "type", None) == "cuda":
                free_before, total = torch.cuda.mem_get_info(device)
                logger.info(
                    "DeepSeek-V4 image request staging: images=%d image_tokens=%d "
                    "prompt_tokens=%d->%d free_vram_mb=%d total_vram_mb=%d resident_mb=%.1f",
                    len(images),
                    expected_image_tokens,
                    len(raw_ids),
                    len(expanded_ids),
                    int(free_before // (1024 * 1024)),
                    int(total // (1024 * 1024)),
                    resident_bytes / (1024 * 1024),
                )
            input_ids = torch.tensor(expanded_ids, device=device, dtype=torch.long)
            safe_ids = input_ids.clone()
            safe_ids[safe_ids >= self.cfg.vocab_size] = 0
            inputs_embeds = self.embedding[safe_ids].clone()
            block_ids_cpu = torch.full((len(expanded_ids),), -1, dtype=torch.int32)
            visible_left_cpu = torch.zeros((len(expanded_ids),), dtype=torch.int32)
            visible_right_cpu = torch.zeros((len(expanded_ids),), dtype=torch.int32)

            log_ram_ledger("before-deepseek-v4-vision-to-gpu")
            if _vram_ledger_enabled():
                _vram_checkpoint("before-deepseek-v4-vision-to-gpu", [device])
            vision = vision.to(device=device, dtype=torch.bfloat16)
            aligner = aligner.to(device=device, dtype=torch.bfloat16)
            special = special_cpu.to(device=device, dtype=dtype)
            if _vram_ledger_enabled():
                _vram_checkpoint("after-deepseek-v4-vision-to-gpu", [device])
            with torch.inference_mode():
                for image_idx, item in enumerate(image_inputs):
                    features = vision(
                        item.patches.to(device=device, dtype=torch.bfloat16),
                        item.n_vit_h,
                        item.n_vit_w,
                    )
                    aligned = aligner(features, item.n_vit_h, item.n_vit_w)
                    types = item.types.to(device=device)
                    block = special[types].clone()
                    image_mask = types == IMAGE
                    perm = item.perm.to(device=device)
                    image_features = aligned[perm].to(dtype=dtype)
                    if int(image_mask.sum().item()) != int(image_features.shape[0]):
                        raise RuntimeError(
                            "DeepSeek-V4 image feature/token mismatch: "
                            f"tokens={int(image_mask.sum().item())} features={image_features.shape[0]}"
                        )
                    block[image_mask] = image_features
                    start = item.start
                    end = start + int(types.numel())
                    inputs_embeds[start:end] = block
                    block_ids_cpu[start:end] = image_idx
                    start_offsets = torch.nonzero(
                        item.types == IMAGE_START, as_tuple=False
                    ).flatten()
                    end_offsets = torch.nonzero(
                        item.types == IMAGE_END, as_tuple=False
                    ).flatten()
                    if start_offsets.numel() != 1 or end_offsets.numel() != 1:
                        raise RuntimeError(
                            "DeepSeek-V4 image block must contain exactly one start and end sentinel"
                        )
                    visible_start = start + int(start_offsets.item())
                    visible_end = start + int(end_offsets.item())
                    if visible_end < visible_start:
                        raise RuntimeError("DeepSeek-V4 image end precedes image start")
                    visible_positions = torch.arange(
                        visible_end - visible_start + 1, dtype=torch.int32
                    )
                    visible_left_cpu[visible_start : visible_end + 1] = visible_positions
                    visible_right_cpu[visible_start : visible_end + 1] = (
                        visible_end - visible_start - visible_positions
                    )
                    image_visible_tokens.append(visible_end - visible_start + 1)
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.synchronize(device)
            vision_block_ids = block_ids_cpu.to(device=device)
            visible_left = visible_left_cpu.to(device=device)
            visible_right = visible_right_cpu.to(device=device)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            self._last_multimodal_prefill_tensors = None
            self._release_deepseek_v4_vision_gpu(
                vision, aligner, special_cpu, device, "after-deepseek-v4-vision-oom-release"
            )
            raise KrasisVisionVramError(
                "VRAM is too constrained for this DeepSeek-V4 image request. "
                f"Transient BF16 vision staging needs about {resident_bytes / (1024 * 1024):.1f} MB "
                f"plus activations; images={len(images)}, image_tokens={expected_image_tokens}."
            ) from e
        except Exception:
            self._release_deepseek_v4_vision_gpu(vision, aligner, special_cpu, device)
            raise

        self._release_deepseek_v4_vision_gpu(vision, aligner, special, device)
        self._last_multimodal_prefill_tensors = (
            inputs_embeds,
            vision_block_ids,
            visible_left,
            visible_right,
        )
        return {
            "token_ids": [int(x) for x in expanded_ids],
            "prompt_tokens": len(expanded_ids),
            "hidden_size": int(inputs_embeds.shape[-1]),
            "inputs_embeds_ptr": int(inputs_embeds.data_ptr()),
            "mrope_cos_ptr": 0,
            "mrope_sin_ptr": 0,
            "mrope_half_dim": 0,
            "rope_delta": 0,
            "vision_block_ids_ptr": int(vision_block_ids.data_ptr()),
            "vision_visible_left_ptr": int(visible_left.data_ptr()),
            "vision_visible_right_ptr": int(visible_right.data_ptr()),
            "vision_max_tokens": max(image_visible_tokens),
            "image_count": len(images),
            "image_tokens": expected_image_tokens,
        }

    def build_multimodal_prefill_inputs(self, messages_json: str, rendered_prompt: str):
        """Build GPU inputs_embeds for a supported image prompt."""
        if self.supports_deepseek_v4_image_inputs():
            return self._build_deepseek_v4_multimodal_prefill_inputs(messages_json, rendered_prompt)
        if self.supports_qwen_image_inputs():
            return self._build_qwen_multimodal_prefill_inputs(messages_json, rendered_prompt)
        if self.supports_step_image_inputs():
            return self._build_step_multimodal_prefill_inputs(messages_json, rendered_prompt)
        if self.supports_gemma_image_inputs():
            return self._build_gemma_multimodal_prefill_inputs(messages_json, rendered_prompt)
        if self.supports_glm53_image_inputs():
            return self._build_glm53_multimodal_prefill_inputs(messages_json, rendered_prompt)
        raise ValueError("loaded model does not support Krasis image inputs")

    def clear_multimodal_prefill_inputs(self):
        self._last_multimodal_prefill_tensors = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, gpu_only: bool = True):
        """Load all weights: GPU (streaming INT8) + CPU (Krasis INT4 experts).

        Args:
            gpu_only: If True, skip CPU expert cache and CPU decode weights.
                Saves ~40 GB RAM and ~60s load time. Only GPU prefill and
                GPU decode (with HCS) will work. CPU decode will not be available.
        """
        start = time.perf_counter()

        log_ram_ledger("before-load")
        _vram_checkpoint("before-load")

        # Phase 0a: GPU VRAM sanity check — abort early if GPU is occupied
        # (e.g. zombie process holding memory after a crash)
        for dev in self.all_devices:
            free_mb, total_mb = torch.cuda.mem_get_info(dev)
            free_mb //= (1024 * 1024)
            total_mb //= (1024 * 1024)
            used_pct = 100 * (1 - free_mb / total_mb)
            if used_pct > 50:
                msg = (f"GPU {dev} has only {free_mb} MB free ({used_pct:.0f}% used). "
                       f"A zombie process may be holding GPU memory. "
                       f"Try: sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) "
                       f"or reboot to clear stuck GPU memory.")
                logger.error(msg)
                raise RuntimeError(msg)

        # Phase 0b: System RAM budget check (skip in gpu_only — no CPU experts loaded)
        if self.cfg.n_routed_experts > 0 and not gpu_only:
            _check_system_ram(
                self.cfg,
                cpu_expert_bits=self.quant_cfg.cpu_expert_bits,
                group_size=self.quant_cfg.expert_group_size,
                force_load=self.force_load,
            )
            self._estimated_expert_ram_gb = _estimate_expert_ram_gb(
                self.cfg, self.quant_cfg.cpu_expert_bits, self.quant_cfg.expert_group_size,
            )
        else:
            self._estimated_expert_ram_gb = 0.0

        # Start RAM watchdog BEFORE loading — protects during the entire load
        self._start_ram_watchdog()

        # Phase 1: GPU weights
        print(f"\n\033[1m\033[36m▸ Loading GPU weights\033[0m", flush=True)
        logger.info("Phase 1: Loading GPU weights (streaming INT8)...")
        loader = WeightLoader(self.cfg, self.quant_cfg)
        self._load_gpu_weights(loader)
        loader.close()

        gpu_elapsed = time.perf_counter() - start
        for i, rank in enumerate(self.ranks):
            dev = torch.device(rank.device)
            alloc_mb = torch.cuda.memory_allocated(dev) / (1024**2)
            logger.info("GPU%d: %.0f MB allocated", i, alloc_mb)
        logger.info("GPU weights loaded in %.1fs", gpu_elapsed)
        log_ram_ledger(
            "after-phase1-gpu-weights",
            {
                "hqq_attention_cache_validated": int(getattr(self, "_hqq_attention_cache_bytes", 0)),
            },
        )
        _vram_checkpoint("after-phase1-gpu-weights")
        self.log_vram_ledger_residency("after-phase1-gpu-weights")

        # Phase 1b: Streaming attention offload (if enabled) or resident check
        if self.stream_attention:
            print(f"\n\033[1m\033[36m▸ Offloading attention for streaming decode\033[0m", flush=True)
            self._init_stream_attention()
        else:
            dev = self.all_devices[0]
            free_mb = torch.cuda.mem_get_info(dev)[0] >> 20
            if self.quant_cfg.attention == "awq":
                # AWQ: attention weights are on CPU, will be quantized to INT4/INT8
                # and uploaded to GPU in setup_gpu_decode_store(). BF16 never touches GPU.
                logger.info("Attention on CPU (AWQ: will quantize and upload in decode setup), GPU free: %d MB",
                            free_mb)
                print(f"  \033[0;32mAttention on CPU (AWQ pending), {free_mb} MB free\033[0m", flush=True)
            elif is_hqq_attention(self.quant_cfg.attention):
                attn_mb = self._hqq_attention_cache_bytes >> 20
                logger.info(
                    "HQQ attention artifacts validated from cache: %d MB, GPU free: %d MB "
                    "(runtime descriptors will be restored during decode-store setup)",
                    attn_mb, free_mb,
                )
                print(
                    f"  \033[0;32mHQQ attention artifacts validated ({attn_mb} MB cached), "
                    f"{free_mb} MB free\033[0m",
                    flush=True,
                )
            else:
                # BF16: attention weights permanently resident on GPU
                attn_mb = self._estimate_attention_vram() >> 20
                logger.info("Attention resident on GPU: %d MB, GPU free: %d MB",
                            attn_mb, free_mb)
                print(f"  \033[0;32mAttention resident on GPU ({attn_mb} MB), {free_mb} MB free\033[0m", flush=True)

        # Phase 2: Expert weights (Rust engine) — GPU Marlin cache only (CPU cache no longer used)
        # IMPORTANT: gpu_expert_bits == 16 is an unvalidated debug-only path.
        # It must not be used as a correctness oracle because loader/layout/math
        # bugs may still exist there independently of the quantized Rust serving
        # path we actually care about.
        cpu_start = time.perf_counter()
        gpu_bits = self.quant_cfg.gpu_expert_bits
        cache_dir = cache_dir_for_model(self.cfg.model_path)
        has_gpu_cache = False
        marlin_cache_bytes = 0
        if gpu_bits == 16:
            print(
                f"\n\033[1m\033[36m▸ Loading BF16 expert weights from safetensors "
                f"(UNVALIDATED debug path only; not for validation or production)\033[0m",
                flush=True,
            )
            logger.warning(
                "Phase 2: Loading GPU expert weights in UNVALIDATED BF16 debug mode; "
                "this path likely contains unknown bugs and must not be used for "
                "validation. Production and correctness work must use external HF "
                "BF16 reference data plus quantized Krasis runs."
            )
        elif gpu_bits == 3:
            tileq_path = getattr(self.quant_cfg, "tileq_cache", None)
            if not tileq_path:
                raise RuntimeError("TileQ GPU experts require an explicit tileq_cache artifact")
            tileq_path = os.path.abspath(os.path.expanduser(tileq_path))
            if not os.path.isfile(tileq_path):
                raise RuntimeError(f"TileQ artifact does not exist: {tileq_path}")
            has_gpu_cache = True
            marlin_cache_bytes = os.path.getsize(tileq_path)
            print(
                f"\n\033[1m\033[36m▸ Loading source-bound TileQ expert cache\033[0m",
                flush=True,
            )
            logger.info("Phase 2: Loading GPU TileQ expert weights from %s", tileq_path)
        else:
            has_gpu_cache = os.path.isfile(
                os.path.join(
                    cache_dir,
                    marlin_cache_basename(
                        gpu_bits,
                        self.quant_cfg.expert_group_size,
                        self.quant_cfg.gpu_expert_int4_calib,
                    ),
                )
            )
            marlin_cache_path = os.path.join(
                cache_dir,
                marlin_cache_basename(
                    gpu_bits,
                    self.quant_cfg.expert_group_size,
                    self.quant_cfg.gpu_expert_int4_calib,
                ),
            )
            if os.path.isfile(marlin_cache_path):
                try:
                    marlin_cache_bytes = os.path.getsize(marlin_cache_path)
                except OSError:
                    marlin_cache_bytes = 0
            if has_gpu_cache:
                print(f"\n\033[1m\033[36m▸ Loading GPU expert weights from cache\033[0m", flush=True)
            else:
                print(f"\n\033[1m\033[36m▸ Building GPU INT{gpu_bits} Marlin expert cache (one-time)\033[0m", flush=True)
                print(f"  \033[2mCache will be saved to {cache_dir} for instant loading next time.\033[0m", flush=True)
            logger.info("Phase 2: Loading GPU expert weights (INT%d)...", gpu_bits)
        self._load_cpu_experts(gpu_only=gpu_only)
        cpu_elapsed = time.perf_counter() - cpu_start
        logger.info("Expert weights loaded in %.1fs", cpu_elapsed)
        if gpu_bits == 16:
            print(
                f"  \033[0;32mUNVALIDATED BF16 debug weights loaded in {cpu_elapsed:.0f}s. "
                f"Do not use this path for validation; use HF BF16 as the oracle.\033[0m",
                flush=True,
            )
        elif has_gpu_cache:
            print(f"  \033[0;32mExpert weights loaded in {cpu_elapsed:.0f}s.\033[0m", flush=True)
        else:
            print(f"  \033[0;32mExpert cache built in {cpu_elapsed:.0f}s — next launch will be much faster.\033[0m", flush=True)

        log_ram_ledger(
            "after-phase2-expert-weights",
            {
                "marlin_cache_file": marlin_cache_bytes,
                "hqq_attention_runtime": int(getattr(self, "_hqq_attention_cache_bytes", 0)),
            },
        )
        _vram_checkpoint("after-phase2-expert-weights")
        self.log_vram_ledger_residency("after-phase2-expert-weights")

        # Post-load RSS check: verify RAM estimate accuracy
        if self._estimated_expert_ram_gb > 0:
            _check_actual_rss(self._estimated_expert_ram_gb)

        # Phase 3: GPU prefill managers (one per device)
        if self.gpu_prefill_enabled and self.cfg.n_routed_experts > 0:
            print(f"\n\033[1m\033[36m▸ Initializing GPU prefill managers\033[0m", flush=True)
            self._init_gpu_prefill()

        log_ram_ledger("after-phase3-prefill-managers")
        _vram_checkpoint("after-phase3-prefill-managers")
        self.log_vram_ledger_residency("after-phase3-prefill-managers")

        # Allocate KV caches
        self._init_kv_caches()
        log_ram_ledger("after-kv-cache-init")
        _vram_checkpoint("after-kv-cache-init")
        self.log_vram_ledger_residency("after-kv-cache-init")

        # Phase 4: CPU Hub — disabled.  Multi-GPU uses layer-split decode,
        # not Expert Parallelism.  Prefill runs on primary GPU only.
        self.cpu_hub = None

        # Load tokenizer — override eos_token_id from tokenizer (authoritative)
        # config.json may have wrong eos_token_id (e.g. Qwen3.5 has endoftext=248044
        # in config but im_end=248046 is the real stop token from tokenizer_config.json)
        self.tokenizer = Tokenizer(self.cfg.model_path)
        if self.tokenizer.eos_token_id and self.tokenizer.eos_token_id != self.cfg.eos_token_id:
            logger.info("Overriding eos_token_id: config=%d -> tokenizer=%d",
                        self.cfg.eos_token_id, self.tokenizer.eos_token_id)
            self.cfg.eos_token_id = self.tokenizer.eos_token_id

        # CPU decode has been removed — GPU decode via Rust GpuDecodeStore only.
        # CpuDecoder is preserved in frozen-oracle for verification.

        self._loaded = True
        total = time.perf_counter() - start
        log_ram_ledger("after-full-load")
        _vram_checkpoint("after-full-load")
        self.log_vram_ledger_residency("after-full-load")
        logger.info("Model fully loaded in %.1fs", total)

    def warmup_cuda_runtime(self, devices: List[torch.device]):
        """Trigger ALL lazy CUDA runtime allocations on ALL devices before HCS expert loading."""
        from krasis.marlin_utils import (
            get_scalar_type,
            gptq_marlin_gemm,
            marlin_make_workspace,
        )

        logger.info("Warming up CUDA runtime on all devices: %s", [str(d) for d in devices])
        free_before = {str(d): torch.cuda.mem_get_info(d)[0] for d in devices}

        for device in devices:
            torch.cuda.set_device(device)

            # ── 1. cuBLAS workspace (for int8 matmuls) ──
            try:
                # Find largest non-expert weight to stress cuBLAS
                max_k, max_n = 0, 0
                for layer in self.layers:
                    if layer.device != device: continue
                    for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                         w = getattr(layer.attention, key, None)
                         if w is None:
                             w = layer.shared_expert.get(key) if layer.shared_expert else None
                         if w is not None:
                            wt = w[0] if isinstance(w, (tuple, list)) else w
                            k, n = wt.shape if len(wt.shape) == 2 else (wt.shape[1], wt.shape[0])
                            if k * n > max_k * max_n: max_k, max_n = k, n

                if max_k > 0:
                    for m_size in (32, 512):
                        x = torch.randint(-127, 127, (m_size, max_k), dtype=torch.int8, device=device)
                        w = torch.randint(-127, 127, (max_n, max_k), dtype=torch.int8, device=device)
                        _ = torch._int_mm(x, w.t())
            except Exception as e:
                raise RuntimeError(
                    f"CUDA warmup (cuBLAS) on {device} failed: {e}\n"
                    "cuBLAS is required for INT8 matmuls (attention/gates). "
                    "Cannot start without working cuBLAS."
                ) from e

            # ── 2. Vendored Marlin kernel loading/workspace allocation ──
            try:
                K = self.cfg.moe_latent_size or self.cfg.hidden_size
                N, bits = self.cfg.moe_intermediate_size, 4
                # Start from the configured expert quantization group and apply
                # the same power-of-two dimensional adjustment used by the
                # Marlin cache builder. Do not assume one model's group size.
                gs = int(self.quant_cfg.expert_group_size)
                while gs > 32 and (K % gs != 0 or N % gs != 0):
                    gs //= 2
                if K % gs != 0 or N % gs != 0:
                    raise RuntimeError(
                        "Marlin warmup cannot derive a dimension-compatible "
                        f"expert group size from configured group_size="
                        f"{self.quant_cfg.expert_group_size}: k={K} n={N}"
                    )
                gated = self.cfg.mlp_hidden_act != "relu2"
                logger.info(
                    "Marlin warmup expert shape on %s: k=%d n=%d group_size=%d gated=%s latent_size=%d",
                    device, K, N, gs, gated, self.cfg.moe_latent_size,
                )

                # GPU prefill uses the vendored Marlin sidecar, while decode is
                # already Rust/CUDA through GpuDecodeStore. Warm the real Marlin
                # dependency on every selected device; there is no Python Triton
                # expert path in the runtime.
                workspace = marlin_make_workspace(device, max_blocks_per_sm=4)
                scalar_type = get_scalar_type(bits)
                dummy_a = torch.zeros(1, K, dtype=torch.bfloat16, device=device)
                dummy_b = torch.zeros(
                    K // 16,
                    N * (bits // 2),
                    dtype=torch.int32,
                    device=device,
                )
                dummy_s = torch.zeros(
                    K // gs,
                    N,
                    dtype=torch.bfloat16,
                    device=device,
                )
                gptq_marlin_gemm(
                    a=dummy_a,
                    c=None,
                    b_q_weight=dummy_b,
                    b_scales=dummy_s,
                    global_scale=None,
                    b_zeros=None,
                    g_idx=None,
                    perm=None,
                    workspace=workspace,
                    b_q_type=scalar_type,
                    size_m=1,
                    size_n=N,
                    size_k=K,
                    is_k_full=True,
                )
                torch.cuda.synchronize(device)

            except Exception as e:
                raise RuntimeError(
                    f"CUDA warmup (Marlin) on {device} failed: {e}\n"
                    "Marlin kernels are required for expert computation. "
                    "Cannot start without the packaged Marlin sidecar."
                ) from e
            
            torch.cuda.synchronize(device)

        # ── 3. Linear attention torch.compile warmup ──
        # Must set device back to primary BEFORE torch.compile — after the per-device
        # loop above, the current CUDA device may be a secondary GPU with a different
        # SM architecture. Inductor generates PTX for the current device's arch, so
        # compiling while cuda:1 (e.g. Ada sm_89) is current would produce kernels
        # that fail on cuda:0 (e.g. Blackwell sm_120).
        if (
            self.cfg.linear_num_value_heads > 0
            and self.cfg.linear_attention_family == "gated_deltanet"
        ):
            try:
                from krasis.linear_attention import warmup_compiled_chunk_step
                primary = devices[0]
                torch.cuda.set_device(primary)
                la_warmup_chunk = 64
                startup_diag = os.environ.get("KRASIS_STARTUP_DIAG", "") == "1"
                la_t0 = time.perf_counter() if startup_diag else 0.0
                if startup_diag:
                    logger.info(
                        "Linear attention compile warmup starting on %s: nv=%d dk=%d dv=%d chunk=%d",
                        primary,
                        self.cfg.linear_num_value_heads,
                        self.cfg.linear_key_head_dim,
                        self.cfg.linear_value_head_dim,
                        la_warmup_chunk,
                    )
                warmup_compiled_chunk_step(
                    primary,
                    nv=self.cfg.linear_num_value_heads,
                    dk=self.cfg.linear_key_head_dim,
                    dv=self.cfg.linear_value_head_dim,
                    chunk_size=la_warmup_chunk,
                )
                if startup_diag:
                    logger.info(
                        "Linear attention compile warmup finished on %s in %.3fs",
                        primary,
                        time.perf_counter() - la_t0,
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Linear attention torch.compile warmup failed: {e}\n"
                    "Linear attention layers require torch.compile. "
                    "Cannot start without working compiled LA kernels."
                ) from e

        # ── Clean up and measure ──
        gc.collect()
        torch.cuda.empty_cache()
        
        for device in devices:
            free_after = torch.cuda.mem_get_info(device)[0]
            consumed = free_before[str(device)] - free_after
            logger.info(
                "CUDA runtime warmup on %s complete: %.0f MB consumed "
                "(%.0f MB free before → %.0f MB free after)",
                device, consumed / 1e6, free_before[str(device)] / 1e6, free_after / 1e6,
            )

    @staticmethod
    def _move_weight(w, device):
        """Copy a weight (tensor or INT8 (tensor, scale) tuple) to device."""
        if w is None:
            return None
        if isinstance(w, tuple):
            return tuple(t.to(device) for t in w)
        if isinstance(w, torch.Tensor):
            return w.to(device)
        return w

    @staticmethod
    def _extract_layer_weights(layer, src_device) -> dict:
        """Extract the weights dict from an existing TransformerLayer.

        Reconstructs the dict format that TransformerLayer.__init__ expects,
        so we can copy it to another device and create a new layer from it.
        """
        mv = KrasisModel._move_weight
        result = {
            "norms": {
                "input_layernorm": layer.input_norm_weight,
                "post_attention_layernorm": layer.post_attn_norm_weight,
            },
            "is_moe": layer.is_moe,
            "layer_type": layer.layer_type,
        }
        for attr, key in (
            ("pre_ffn_norm_weight", "pre_feedforward_layernorm"),
            ("post_ffn_norm_weight", "post_feedforward_layernorm"),
            ("post_ffn_norm1_weight", "post_feedforward_layernorm_1"),
            ("post_ffn_norm2_weight", "post_feedforward_layernorm_2"),
            ("pre_ffn_norm2_weight", "pre_feedforward_layernorm_2"),
            ("layer_scalar", "layer_scalar"),
        ):
            value = getattr(layer, attr, None)
            if value is not None:
                result["norms"][key] = value

        # ── Attention weights ──
        attn = layer.attention
        if layer.layer_type == "mamba2":
            result["mamba2"] = layer.mamba2_weights
        elif layer.layer_type == "moe" and attn is None:
            pass  # MoE-only layer, no attention weights
        elif layer.layer_type == "linear_attention":
            if layer.cfg.is_kimi_delta_attention_layer(layer.layer_idx):
                result["kimi_delta_attention"] = dict(attn.weights)
            else:
                result["linear_attention"] = {
                    "in_proj_qkvz": attn.in_proj_qkvz,
                    "in_proj_ba": attn.in_proj_ba,
                    "out_proj": attn.out_proj,
                    "conv1d_weight": attn.conv1d_weight,
                    "A_log": attn.A_log,
                    "dt_bias": attn.dt_bias,
                    "norm_weight": attn.norm_weight,
                }
        elif layer.cfg.is_deepseek_v4:
            if attn is None or not hasattr(attn, "attention"):
                raise RuntimeError(
                    f"DeepSeek-V4 layer {layer.layer_idx} has no native attention weights"
                )
            result["attention"] = dict(attn.attention)
            result["hyper_connection"] = dict(attn.hyper_connection)
        elif hasattr(attn, "kv_a_proj"):
            # MLA
            attn_d = {
                "kv_a_proj_with_mqa": attn.kv_a_proj,
                "o_proj": attn.o_proj,
                "kv_a_layernorm": attn.kv_a_norm_weight,
                "w_kc": attn.w_kc,
                "w_vc": attn.w_vc,
            }
            if attn.has_q_lora:
                attn_d["q_a_proj"] = attn.q_a_proj
                attn_d["q_b_proj"] = attn.q_b_proj
                attn_d["q_a_layernorm"] = attn.q_a_norm_weight
            else:
                attn_d["q_proj"] = attn.q_proj
            result["attention"] = attn_d
        elif hasattr(layer, 'gqa_weights') and layer.gqa_weights:
            # GQA — weights stored directly on layer (no attention wrapper)
            result["attention"] = dict(layer.gqa_weights)
        else:
            # GQA with legacy attention object (shouldn't happen but safe fallback)
            attn_d = {
                "q_proj": attn.q_proj,
                "k_proj": attn.k_proj,
                "v_proj": attn.v_proj,
                "o_proj": attn.o_proj,
            }
            for name in ("q_proj_bias", "k_proj_bias", "v_proj_bias",
                         "o_proj_bias", "sinks", "q_norm", "k_norm"):
                val = getattr(attn, name, None)
                if val is not None:
                    attn_d[name] = val
            result["attention"] = attn_d

        # ── MoE weights ──
        if layer.is_moe:
            result["gate"] = {"weight": layer.gate_weight}
            if layer.gate_bias is not None:
                result["gate"]["bias"] = layer.gate_bias
            if layer.e_score_correction_bias is not None:
                result["gate"]["e_score_correction_bias"] = layer.e_score_correction_bias
            if getattr(layer, "router_vl_bias", None) is not None:
                result["gate"]["vl_bias"] = layer.router_vl_bias
            if getattr(layer, "router_tid2eid", None) is not None:
                result["gate"]["tid2eid"] = layer.router_tid2eid
            if getattr(layer, "router_input_scale", None) is not None:
                result["gate"]["input_scale"] = layer.router_input_scale
            if getattr(layer, "router_per_expert_scale", None) is not None:
                result["gate"]["per_expert_scale"] = layer.router_per_expert_scale
            if layer.shared_expert is not None:
                se = {}
                if "gate_up_proj" in layer.shared_expert:
                    # Standard MoE: reconstruct original gate_proj/up_proj from fused gate_up_proj
                    gate_up = layer.shared_expert["gate_up_proj"]
                    if isinstance(gate_up, tuple):
                        # INT8: split (weight, scale) along dim 0
                        mid_w = gate_up[0].shape[0] // 2
                        mid_s = gate_up[1].shape[0] // 2
                        se["gate_proj"] = (gate_up[0][:mid_w], gate_up[1][:mid_s])
                        se["up_proj"] = (gate_up[0][mid_w:], gate_up[1][mid_s:])
                    else:
                        mid = gate_up.shape[0] // 2
                        se["gate_proj"] = gate_up[:mid]
                        se["up_proj"] = gate_up[mid:]
                else:
                    # Nemotron MoE: up_proj + down_proj only (no gate_proj)
                    se["up_proj"] = layer.shared_expert["up_proj"]
                se["down_proj"] = layer.shared_expert["down_proj"]
                if layer.shared_expert_gate is not None:
                    se["shared_expert_gate"] = layer.shared_expert_gate
                result["shared_expert"] = se
            if layer.dense_mlp is not None:
                result["dense_mlp"] = dict(layer.dense_mlp)
        elif layer.dense_mlp is not None:
            result["dense_mlp"] = dict(layer.dense_mlp)

        # ── Latent MoE projections (Nemotron) ──
        if hasattr(layer, 'latent_proj') and layer.latent_proj is not None:
            result["latent_proj"] = dict(layer.latent_proj)

        return result

    @staticmethod
    def _copy_weights_dict(weights: dict, device) -> dict:
        """Deep-copy a layer weights dict, moving all tensors to device."""
        mv = KrasisModel._move_weight
        result = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                result[k] = KrasisModel._copy_weights_dict(v, device)
            elif isinstance(v, (torch.Tensor, tuple)):
                result[k] = mv(v, device)
            else:
                result[k] = v
        return result

    def _build_hqq_fused_qkv_artifact_weight(
        self,
        layer_type: str,
        weights: dict,
    ) -> Optional[torch.Tensor]:
        if (
            layer_type not in ("full_attention", "sliding_attention")
            or self.cfg.is_mla
        ):
            return None
        fused_qkv = weights.get("fused_qkv")
        if isinstance(fused_qkv, torch.Tensor):
            return fused_qkv

        q_proj = weights.get("q_proj")
        k_proj = weights.get("k_proj")
        v_proj = weights.get("v_proj")
        if not all(isinstance(t, torch.Tensor) for t in (q_proj, k_proj, v_proj)):
            return None
        if any(t.dim() != 2 for t in (q_proj, k_proj, v_proj)):
            raise RuntimeError(
                "HQQ fused_qkv artifact construction requires 2D q/k/v projection weights."
            )
        if q_proj.shape[1] != k_proj.shape[1] or q_proj.shape[1] != v_proj.shape[1]:
            raise RuntimeError(
                "HQQ fused_qkv artifact construction requires q/k/v projection weights "
                "to share the same input width."
            )
        if q_proj.dtype != k_proj.dtype or q_proj.dtype != v_proj.dtype:
            raise RuntimeError(
                "HQQ fused_qkv artifact construction requires q/k/v projection weights "
                "to share the same dtype."
            )
        if q_proj.device != k_proj.device or q_proj.device != v_proj.device:
            raise RuntimeError(
                "HQQ fused_qkv artifact construction requires q/k/v projection weights "
                "to reside on the same device."
            )
        return torch.cat((q_proj, k_proj, v_proj), dim=0).contiguous()

    def _hqq_attention_tensor_map(self, layer_type: str, weights: dict) -> dict:
        if getattr(self.cfg, "is_deepseek_v4", False):
            result = {}

            def _add(prefix: str, source: dict, names: tuple[str, ...]) -> None:
                if not isinstance(source, dict):
                    return
                for name in names:
                    tensor = source.get(name)
                    if isinstance(tensor, torch.Tensor):
                        result[f"{prefix}{name}"] = tensor

            _add("", weights, ("wq_a", "wq_b", "wkv", "wo_b"))
            return result
        if layer_type == "linear_attention":
            if self.cfg.linear_attention_family == "kimi_delta_attention":
                ordered = (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "f_a_proj",
                    "f_b_proj",
                    "b_proj",
                    "g_a_proj",
                    "g_b_proj",
                )
            else:
                ordered = ("in_proj_qkvz", "in_proj_ba", "out_proj")
        elif layer_type in ("full_attention", "sliding_attention"):
            if self.cfg.is_mla:
                ordered = ("q_a_proj", "q_b_proj", "q_proj", "kv_a_proj_with_mqa", "o_proj")
            else:
                ordered = ("q_proj", "k_proj", "v_proj", "o_proj")
        else:
            return {}

        result = {}
        for name in ordered:
            tensor = weights.get(name)
            if isinstance(tensor, torch.Tensor):
                result[name] = tensor
        if (
            layer_type in ("full_attention", "sliding_attention")
            and not self.cfg.is_mla
        ):
            fused_qkv = weights.get("fused_qkv")
            if isinstance(fused_qkv, torch.Tensor):
                result["fused_qkv"] = fused_qkv
        return result

    def _attention_weight_key(self, layer_type: str) -> str:
        if layer_type == "linear_attention":
            if self.cfg.linear_attention_family == "kimi_delta_attention":
                return "kimi_delta_attention"
            return "linear_attention"
        if layer_type == "mamba2":
            return "mamba2"
        return "attention"

    def _record_hqq_expected_attention_tensors(
        self,
        layer_idx: int,
        layer_type: str,
        weights: dict,
    ) -> None:
        expected = getattr(self, "_hqq_expected_tensors", None)
        expected_set = getattr(self, "_hqq_expected_tensor_set", None)
        if expected is None or expected_set is None:
            expected = []
            expected_set = set()
            self._hqq_expected_tensors = expected
            self._hqq_expected_tensor_set = expected_set

        tensor_names = list(self._hqq_attention_tensor_map(layer_type, weights))
        if (
            layer_type in ("full_attention", "sliding_attention")
            and not self.cfg.is_mla
            and "fused_qkv" not in tensor_names
            and all(isinstance(weights.get(name), torch.Tensor) for name in ("q_proj", "k_proj", "v_proj"))
        ):
            tensor_names.append("fused_qkv")
        for tensor_name in tensor_names:
            key = (int(layer_idx), tensor_name)
            if key not in expected_set:
                expected_set.add(key)
                expected.append(key)

    def _hqq4_cache_parallelism(self, tensor_count: int) -> tuple[int, int]:
        """Return bounded HQQ4 outer concurrency and per-artifact Rayon threads."""
        if tensor_count <= 1:
            return 1, max(1, int(self.krasis_threads))

        raw_workers = os.environ.get("KRASIS_HQQ4_CACHE_CONCURRENCY")
        if raw_workers:
            try:
                requested_workers = int(raw_workers)
            except ValueError as exc:
                raise RuntimeError(
                    f"KRASIS_HQQ4_CACHE_CONCURRENCY must be an integer, got {raw_workers!r}"
                ) from exc
        else:
            requested_workers = 1

        # Keep the first-time HQQ cache build memory bound: one BF16 layer is
        # materialized, and at most two independent tensor quantizers run inside it.
        workers = max(1, min(2, int(tensor_count), requested_workers))

        raw_inner = os.environ.get("KRASIS_HQQ4_INNER_THREADS")
        if raw_inner:
            try:
                inner_threads = int(raw_inner)
            except ValueError as exc:
                raise RuntimeError(
                    f"KRASIS_HQQ4_INNER_THREADS must be an integer, got {raw_inner!r}"
                ) from exc
        else:
            inner_threads = max(1, int(self.krasis_threads) // workers)

        return workers, max(1, inner_threads)

    def _prepare_hqq_attention_cache(self) -> None:
        nbits = attention_quant_cache_nbits(self.quant_cfg.attention)
        if nbits is None:
            raise RuntimeError(
                f"HQQ attention backend must declare nbits, got {self.quant_cfg.attention}"
            )
        validate_hqq_cache_nbits(nbits)
        self._hqq_cache_nbits = nbits
        cache_profile = self.quant_cfg.hqq_cache_profile
        hqq_group_size = self.quant_cfg.hqq_group_size
        cache_dir = hqq_attention_cache_dir(self.cfg.model_path, cache_profile, nbits, hqq_group_size)
        manifest = init_hqq_attention_manifest(
            self.cfg.model_path,
            num_hidden_layers=self.cfg.num_hidden_layers,
            nbits=nbits,
            group_size=hqq_group_size,
        )
        if is_hqq_auto_attention(self.quant_cfg.attention):
            policy = hqq_auto_promotion_policy(self.quant_cfg.attention)
            manifest["mixed_precision"]["budget_pct"] = (
                None if self.quant_cfg.hqq_auto_budget_pct is None else float(self.quant_cfg.hqq_auto_budget_pct)
            )
            manifest["mixed_precision"]["base_nbits"] = int(policy["base_nbits"])
            manifest["mixed_precision"]["promoted_nbits"] = int(policy["promoted_nbits"])
            manifest["planner"]["budget_pct"] = manifest["mixed_precision"]["budget_pct"]
            if self.quant_cfg.attention == "hqq46_auto" and self.quant_cfg.hqq46_auto_budget_mib is not None:
                budget_bytes = hqq46_auto_budget_bytes_from_mib(self.quant_cfg.hqq46_auto_budget_mib)
                manifest["mixed_precision"]["legacy_budget_mib"] = int(self.quant_cfg.hqq46_auto_budget_mib)
                manifest["mixed_precision"]["budget_bytes"] = int(budget_bytes)
                manifest["planner"]["legacy_budget_mib"] = int(self.quant_cfg.hqq46_auto_budget_mib)
                manifest["planner"]["budget_bytes"] = int(budget_bytes)
        if cache_profile != HQQ_CACHE_PROFILE_BASELINE:
            self._hqq_rebuild = False
            self._hqq_finalize_pending_manifest = False
            self._hqq_manifest = require_complete_hqq_attention_manifest(
                self.cfg.model_path,
                cache_profile=cache_profile,
                expected_nbits=nbits,
                expected_num_hidden_layers=self.cfg.num_hidden_layers,
                expected_group_size=hqq_group_size,
            )
            logger.info(
                "Using HQQ attention cache profile '%s' from %s",
                cache_profile,
                hqq_attention_manifest_path(self.cfg.model_path, cache_profile, nbits, hqq_group_size),
            )
            return

        existing = load_hqq_attention_manifest(self.cfg.model_path, cache_profile, nbits, hqq_group_size)
        pending = load_hqq_attention_pending_manifest(self.cfg.model_path, cache_profile, nbits, hqq_group_size)
        self._hqq_rebuild = True
        self._hqq_finalize_pending_manifest = False
        self._hqq_resume_pending_manifest = False

        def _compatible(candidate: dict) -> bool:
            compatible = (
                candidate.get("format_version") == manifest["format_version"]
                and candidate.get("backend") == manifest["backend"]
                and candidate.get("num_hidden_layers") == manifest["num_hidden_layers"]
                and candidate.get("group_size") == manifest["group_size"]
                and candidate.get("axis") == manifest["axis"]
                and candidate.get("layout") == manifest["layout"]
                and candidate.get("quantizer") == manifest.get("quantizer")
            )
            if compatible and is_hqq_auto_attention(self.quant_cfg.attention):
                candidate_mixed = candidate.get("mixed_precision", {})
                requested_pct = manifest.get("mixed_precision", {}).get("budget_pct")
                if requested_pct is not None:
                    candidate_pct = candidate_mixed.get("budget_pct")
                    compatible = candidate_pct is not None and abs(float(candidate_pct) - float(requested_pct)) < 1e-9
                else:
                    compatible = (
                        int(candidate_mixed.get("budget_bytes", -1))
                        == int(manifest["mixed_precision"]["budget_bytes"])
                    )
            return compatible

        def _pending_has_all_layer_artifacts(candidate: dict) -> bool:
            if is_hqq_auto_attention(self.quant_cfg.attention):
                candidates = candidate.get("planner", {}).get("candidates", [])
                if candidates:
                    entries = [
                        record
                        for candidate_entry in candidates
                        for record in (
                            candidate_entry.get("base_record"),
                            candidate_entry.get("promoted_record"),
                        )
                        if isinstance(record, dict)
                    ]
                else:
                    entries = candidate.get("tensors", [])
            else:
                entries = candidate.get("tensors", [])
            if not entries:
                return False
            seen_layers = set()
            for entry in entries:
                try:
                    layer_idx = int(entry["layer_idx"])
                except (KeyError, TypeError, ValueError):
                    return False
                if layer_idx < 0 or layer_idx >= self.cfg.num_hidden_layers:
                    return False
                file_name = entry.get("file")
                if not file_name:
                    return False
                artifact_path = os.path.join(cache_dir, file_name)
                if not os.path.isfile(artifact_path):
                    return False
                seen_layers.add(layer_idx)
            return len(seen_layers) == self.cfg.num_hidden_layers

        if existing:
            if _compatible(existing) and existing.get("complete"):
                manifest = existing
                self._hqq_rebuild = False
        if self._hqq_rebuild and pending and _compatible(pending):
            if _pending_has_all_layer_artifacts(pending):
                manifest = pending
                self._hqq_rebuild = False
                self._hqq_finalize_pending_manifest = True
                logger.info(
                    "Recovering complete pending HQQ attention manifest from %s",
                    hqq_attention_pending_manifest_path(self.cfg.model_path, cache_profile, nbits, hqq_group_size),
                )
            elif is_hqq_auto_attention(self.quant_cfg.attention) and pending.get("planner", {}).get("candidates"):
                manifest = pending
                self._hqq_resume_pending_manifest = True
                logger.info(
                    "Resuming partial %s pending manifest from %s",
                    self.quant_cfg.attention,
                    hqq_attention_pending_manifest_path(self.cfg.model_path, cache_profile, nbits, hqq_group_size),
                )
        if self._hqq_rebuild:
            if os.path.isdir(cache_dir) and not self._hqq_resume_pending_manifest:
                shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            incomplete_manifest_path = hqq_attention_manifest_path(self.cfg.model_path, cache_profile, nbits, hqq_group_size)
            if os.path.isfile(incomplete_manifest_path) and not self._hqq_resume_pending_manifest:
                os.remove(incomplete_manifest_path)
            if not self._hqq_resume_pending_manifest:
                delete_hqq_attention_pending_manifest(self.cfg.model_path, cache_profile, nbits, hqq_group_size)
            save_hqq_attention_pending_manifest(self.cfg.model_path, manifest, cache_profile, nbits, hqq_group_size)
        self._hqq_manifest = manifest
        self._hqq_expected_tensors = []
        self._hqq_expected_tensor_set = set()

    def _maybe_write_hqq_attention_artifacts(self, layer_idx: int, layer_type: str, weights: dict) -> None:
        self._record_hqq_expected_attention_tensors(layer_idx, layer_type, weights)
        if not getattr(self, "_hqq_rebuild", False):
            return
        cache_nbits = getattr(self, "_hqq_cache_nbits", attention_quant_cache_nbits(self.quant_cfg.attention))
        if cache_nbits is None:
            raise RuntimeError(
                f"HQQ attention backend must declare nbits, got {self.quant_cfg.attention}"
            )
        validate_hqq_cache_nbits(cache_nbits)
        def _artifact_nbits(name: str) -> int:
            if self.quant_cfg.attention == "hqq46":
                return hqq46_tensor_nbits(name)
            if is_hqq_auto_attention(self.quant_cfg.attention):
                raise RuntimeError(f"{self.quant_cfg.attention} artifacts are selected by the planner, not by tensor name.")
            actual = attention_quant_nbits(self.quant_cfg.attention)
            if actual is None:
                raise RuntimeError(
                    f"HQQ attention backend must declare tensor nbits, got {self.quant_cfg.attention}"
                )
            validate_hqq_nbits(actual)
            return actual
        timing_enabled = os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING") == "1"
        timing_entries = []
        timing_started = time.perf_counter() if timing_enabled else 0.0

        def _write_record(
            tensor_name: str,
            tensor: torch.Tensor,
            nbits: int,
            *,
            hqq4_inner_threads: Optional[int] = None,
        ) -> dict:
            return write_hqq_attention_artifact(
                self.cfg.model_path,
                layer_idx=layer_idx,
                layer_type=layer_type,
                tensor_name=tensor_name,
                weight=tensor,
                nbits=nbits,
                cache_profile=self.quant_cfg.hqq_cache_profile,
                group_size=self.quant_cfg.hqq_group_size,
                cache_nbits=cache_nbits,
                hqq4_inner_threads=hqq4_inner_threads if int(nbits) == 4 else None,
                hqq_search_device=getattr(self, "_hqq_search_cuda_device", None),
            )

        def _existing_auto_entry(tensor_name: str, entries_key: str) -> Optional[dict]:
            if not is_hqq_auto_attention(self.quant_cfg.attention):
                return None
            cache_dir = hqq_attention_cache_dir(
                self.cfg.model_path,
                self.quant_cfg.hqq_cache_profile,
                cache_nbits,
                self.quant_cfg.hqq_group_size,
            )
            if entries_key == "planner":
                entries = self._hqq_manifest.get("planner", {}).get("candidates", [])
            else:
                entries = self._hqq_manifest.get(entries_key, [])
            for entry in entries:
                if int(entry.get("layer_idx", -1)) != int(layer_idx):
                    continue
                if str(entry.get("tensor_name")) != tensor_name:
                    continue
                if entries_key == "tensors":
                    records = (entry,)
                else:
                    records = (entry.get("base_record"), entry.get("promoted_record"))
                if all(
                    isinstance(record, dict)
                    and os.path.isfile(os.path.join(cache_dir, record.get("file", "")))
                    for record in records
                ):
                    return entry
            return None

        def _update_auto_direct_edge_metadata(direct_nbits: int) -> None:
            policy = hqq_auto_promotion_policy(self.quant_cfg.attention)
            base_nbits = int(policy["base_nbits"])
            promoted_nbits = int(policy["promoted_nbits"])
            selected_count = (
                len(self._hqq_manifest.get("tensors", []))
                if int(direct_nbits) == promoted_nbits
                else 0
            )
            selection_mode = (
                "direct_full_budget"
                if int(direct_nbits) == promoted_nbits
                else "direct_zero_budget"
            )
            measured_budget_bytes = 0 if int(direct_nbits) == base_nbits else None
            tensor_count = len(self._hqq_manifest.get("tensors", []))
            total_bytes = int(self._hqq_manifest.get("totals", {}).get("tensor_bytes", 0))
            mixed = self._hqq_manifest.setdefault("mixed_precision", {})
            mixed["base_nbits"] = base_nbits
            mixed["promoted_nbits"] = promoted_nbits
            mixed["selected_promotions"] = selected_count
            mixed["candidate_count"] = tensor_count
            mixed["budget_used_bytes"] = measured_budget_bytes
            mixed["direct_edge_nbits"] = int(direct_nbits)
            mixed["selection_mode"] = selection_mode
            planner = self._hqq_manifest.setdefault("planner", {})
            planner["name"] = policy["name"]
            planner["budget_pct"] = mixed.get("budget_pct")
            planner["candidate_count"] = tensor_count
            planner["selected_count"] = selected_count
            planner["candidates"] = []
            planner["summary"] = {
                "budget_bytes": measured_budget_bytes,
                "budget_used_bytes": measured_budget_bytes,
                "budget_unit_bytes": 1,
                "selected_count": selected_count,
                "candidate_count": tensor_count,
                "relative_rmse_reduction": 0.0,
                "selection_mode": selection_mode,
                "direct_edge_nbits": int(direct_nbits),
                "tensor_bytes": total_bytes,
            }

        def _write_selected_or_candidate(
            tensor_name: str,
            tensor: torch.Tensor,
            *,
            hqq4_inner_threads: Optional[int] = None,
        ) -> tuple[int, dict]:
            if is_hqq_auto_attention(self.quant_cfg.attention):
                policy = hqq_auto_promotion_policy(self.quant_cfg.attention)
                direct_nbits = hqq_auto_direct_edge_nbits(
                    self.quant_cfg.attention,
                    self._hqq_manifest.get("mixed_precision", {}).get("budget_pct"),
                )
                if direct_nbits is not None:
                    existing = _existing_auto_entry(tensor_name, "tensors")
                    if existing is not None:
                        return 0, existing
                    record = _write_record(
                        tensor_name,
                        tensor,
                        direct_nbits,
                        hqq4_inner_threads=hqq4_inner_threads,
                    )
                    self._hqq_manifest["tensors"].append(record)
                    self._hqq_manifest["totals"]["tensor_bytes"] += record["tensor_bytes"]
                    self._hqq_manifest["totals"]["num_tensors"] += 1
                    _update_auto_direct_edge_metadata(direct_nbits)
                    return record["tensor_bytes"], record

                existing = _existing_auto_entry(tensor_name, "planner")
                if existing is not None:
                    return 0, existing
                base_record = _write_record(
                    tensor_name,
                    tensor,
                    int(policy["base_nbits"]),
                    hqq4_inner_threads=hqq4_inner_threads,
                )
                promoted_record = _write_record(
                    tensor_name,
                    tensor,
                    int(policy["promoted_nbits"]),
                    hqq4_inner_threads=hqq4_inner_threads,
                )
                candidate = hqq_auto_candidate_from_records(base_record, promoted_record)
                planner = self._hqq_manifest.setdefault(
                    "planner",
                    {"name": policy["name"], "candidate_count": 0, "candidates": []},
                )
                planner.setdefault("candidates", []).append(candidate)
                planner["candidate_count"] = len(planner["candidates"])
                return base_record["tensor_bytes"] + promoted_record["tensor_bytes"], candidate

            record = _write_record(
                tensor_name,
                tensor,
                _artifact_nbits(tensor_name),
                hqq4_inner_threads=hqq4_inner_threads,
            )
            self._hqq_manifest["tensors"].append(record)
            self._hqq_manifest["totals"]["tensor_bytes"] += record["tensor_bytes"]
            self._hqq_manifest["totals"]["num_tensors"] += 1
            return record["tensor_bytes"], record

        tensor_map = self._hqq_attention_tensor_map(layer_type, weights)
        records_by_tensor = {}
        fixed_hqq4_parallel = (
            self.quant_cfg.attention == "hqq4"
            and not is_hqq_auto_attention(self.quant_cfg.attention)
            and int(cache_nbits) == 4
        )
        hqq4_workers, hqq4_inner_threads = (
            self._hqq4_cache_parallelism(len(tensor_map)) if fixed_hqq4_parallel else (1, None)
        )
        if fixed_hqq4_parallel and layer_idx == 0:
            logger.info(
                "HQQ4 cache build concurrency: workers=%d inner_threads=%d total_threads=%d",
                hqq4_workers,
                hqq4_inner_threads,
                self.krasis_threads,
            )

        def _emit_tensor_start(tensor_name: str, extra: Optional[dict] = None) -> None:
            if not timing_enabled:
                return
            payload = {
                "phase": "artifact_write_tensor_start",
                "layer_idx": int(layer_idx),
                "layer_type": layer_type,
                "tensor_name": tensor_name,
            }
            if extra:
                payload.update(extra)
            print(json.dumps({"hqq_real_model_timing": payload}, sort_keys=True), flush=True)

        def _emit_tensor_done(
            tensor_name: str,
            elapsed_s: float,
            written_bytes: int,
            extra: Optional[dict] = None,
        ) -> None:
            if not timing_enabled:
                return
            entry = {
                "tensor_name": tensor_name,
                "elapsed_s": elapsed_s,
                "tensor_bytes": int(written_bytes),
            }
            if extra:
                entry.update(extra)
            timing_entries.append(entry)
            payload = {
                "phase": "artifact_write_tensor_done",
                "layer_idx": int(layer_idx),
                "layer_type": layer_type,
                "tensor_name": tensor_name,
                "elapsed_s": elapsed_s,
                "tensor_bytes": int(written_bytes),
            }
            if extra:
                payload.update(extra)
            print(json.dumps({"hqq_real_model_timing": payload}, sort_keys=True), flush=True)

        def _record_fixed_artifact(tensor_name: str, record: dict, written_bytes: int) -> None:
            self._hqq_manifest["tensors"].append(record)
            self._hqq_manifest["totals"]["tensor_bytes"] += int(written_bytes)
            self._hqq_manifest["totals"]["num_tensors"] += 1
            if isinstance(record, dict) and record.get("tensor_name") == tensor_name:
                records_by_tensor[tensor_name] = record

        if fixed_hqq4_parallel and hqq4_workers > 1 and len(tensor_map) > 1:
            def _write_record_timed(tensor_name: str, tensor: torch.Tensor) -> tuple[dict, float]:
                worker_started = time.perf_counter()
                record = _write_record(
                    tensor_name,
                    tensor,
                    4,
                    hqq4_inner_threads=hqq4_inner_threads,
                )
                return record, time.perf_counter() - worker_started

            future_records = []
            with ThreadPoolExecutor(max_workers=hqq4_workers, thread_name_prefix="hqq4-cache") as executor:
                for tensor_name, tensor in tensor_map.items():
                    tensor_started = time.perf_counter()
                    timing_extra = {
                        "concurrency_workers": hqq4_workers,
                        "hqq4_inner_threads": hqq4_inner_threads,
                    }
                    _emit_tensor_start(tensor_name, timing_extra)
                    future = executor.submit(_write_record_timed, tensor_name, tensor)
                    future_records.append((tensor_name, tensor_started, future, timing_extra))

                for tensor_name, tensor_started, future, timing_extra in future_records:
                    record, worker_elapsed_s = future.result()
                    written_bytes = int(record["tensor_bytes"])
                    _record_fixed_artifact(tensor_name, record, written_bytes)
                    timing_extra = {
                        **timing_extra,
                        "queued_elapsed_s": time.perf_counter() - tensor_started,
                    }
                    _emit_tensor_done(
                        tensor_name,
                        worker_elapsed_s,
                        written_bytes,
                        timing_extra,
                    )
        else:
            for tensor_name, tensor in tensor_map.items():
                tensor_started = time.perf_counter() if timing_enabled else 0.0
                _emit_tensor_start(tensor_name)
                written_bytes, record = _write_selected_or_candidate(
                    tensor_name,
                    tensor,
                    hqq4_inner_threads=hqq4_inner_threads,
                )
                if isinstance(record, dict) and record.get("tensor_name") == tensor_name:
                    records_by_tensor[tensor_name] = record
                if timing_enabled:
                    _emit_tensor_done(
                        tensor_name,
                        time.perf_counter() - tensor_started,
                        int(written_bytes),
                    )
        if "fused_qkv" not in tensor_map:
            fused_build_started = time.perf_counter() if timing_enabled else 0.0
            can_synthesize_fused_qkv = (
                not is_hqq_auto_attention(self.quant_cfg.attention)
                and self.quant_cfg.attention in ("hqq6", "hqq8")
                and layer_type in ("full_attention", "sliding_attention")
                and not self.cfg.is_mla
                and all(name in records_by_tensor for name in ("q_proj", "k_proj", "v_proj"))
            )
            if can_synthesize_fused_qkv:
                fused_nbits = _artifact_nbits("fused_qkv")
                if not all(int(records_by_tensor[name].get("nbits", -1)) == fused_nbits for name in ("q_proj", "k_proj", "v_proj")):
                    raise RuntimeError(
                        "Cannot synthesize fixed-HQQ fused_qkv artifact: split q/k/v nbits differ."
                    )
                if all(int(records_by_tensor[name].get("nbits", -1)) == fused_nbits for name in ("q_proj", "k_proj", "v_proj")):
                    fused_write_started = time.perf_counter() if timing_enabled else 0.0
                    if timing_enabled:
                        print(
                            json.dumps(
                                {
                                    "hqq_real_model_timing": {
                                        "phase": "artifact_write_tensor_start",
                                        "layer_idx": int(layer_idx),
                                        "layer_type": layer_type,
                                        "tensor_name": "fused_qkv",
                                        "build_elapsed_s": fused_write_started - fused_build_started,
                                        "source": "split_hqq_artifacts",
                                    }
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    record = synthesize_hqq_fused_qkv_artifact(
                        self.cfg.model_path,
                        layer_idx=layer_idx,
                        layer_type=layer_type,
                        records_by_tensor=records_by_tensor,
                        nbits=fused_nbits,
                        cache_profile=self.quant_cfg.hqq_cache_profile,
                        group_size=self.quant_cfg.hqq_group_size,
                        cache_nbits=cache_nbits,
                    )
                    self._hqq_manifest["tensors"].append(record)
                    self._hqq_manifest["totals"]["tensor_bytes"] += record["tensor_bytes"]
                    self._hqq_manifest["totals"]["num_tensors"] += 1
                    written_bytes = record["tensor_bytes"]
                    if timing_enabled:
                        elapsed_s = time.perf_counter() - fused_write_started
                        timing_entries.append(
                            {
                                "tensor_name": "fused_qkv",
                                "build_elapsed_s": fused_write_started - fused_build_started,
                                "elapsed_s": elapsed_s,
                                "tensor_bytes": int(written_bytes),
                                "source": "split_hqq_artifacts",
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    "hqq_real_model_timing": {
                                        "phase": "artifact_write_tensor_done",
                                        "layer_idx": int(layer_idx),
                                        "layer_type": layer_type,
                                        "tensor_name": "fused_qkv",
                                        "build_elapsed_s": fused_write_started - fused_build_started,
                                        "elapsed_s": elapsed_s,
                                        "tensor_bytes": int(written_bytes),
                                        "source": "split_hqq_artifacts",
                                    }
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            else:
                fused_qkv = self._build_hqq_fused_qkv_artifact_weight(layer_type, weights)
                if fused_qkv is not None:
                    fused_write_started = time.perf_counter() if timing_enabled else 0.0
                    if timing_enabled:
                        print(
                            json.dumps(
                                {
                                    "hqq_real_model_timing": {
                                        "phase": "artifact_write_tensor_start",
                                        "layer_idx": int(layer_idx),
                                        "layer_type": layer_type,
                                        "tensor_name": "fused_qkv",
                                        "build_elapsed_s": fused_write_started - fused_build_started,
                                        "source": "bf16_concat",
                                    }
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    written_bytes, record = _write_selected_or_candidate(
                        "fused_qkv",
                        fused_qkv,
                        hqq4_inner_threads=max(1, int(self.krasis_threads)) if fixed_hqq4_parallel else hqq4_inner_threads,
                    )
                    if timing_enabled:
                        elapsed_s = time.perf_counter() - fused_write_started
                        timing_entries.append(
                            {
                                "tensor_name": "fused_qkv",
                                "build_elapsed_s": fused_write_started - fused_build_started,
                                "elapsed_s": elapsed_s,
                                "tensor_bytes": int(written_bytes),
                                "source": "bf16_concat",
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    "hqq_real_model_timing": {
                                        "phase": "artifact_write_tensor_done",
                                        "layer_idx": int(layer_idx),
                                        "layer_type": layer_type,
                                        "tensor_name": "fused_qkv",
                                        "build_elapsed_s": fused_write_started - fused_build_started,
                                        "elapsed_s": elapsed_s,
                                        "tensor_bytes": int(written_bytes),
                                        "source": "bf16_concat",
                                    }
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        save_hqq_attention_pending_manifest(
            self.cfg.model_path,
            self._hqq_manifest,
            self.quant_cfg.hqq_cache_profile,
            cache_nbits,
            self.quant_cfg.hqq_group_size,
        )
        if timing_enabled:
            print(
                json.dumps(
                    {
                        "hqq_real_model_timing": {
                            "phase": "artifact_write",
                            "layer_idx": int(layer_idx),
                            "layer_type": layer_type,
                            "elapsed_s": time.perf_counter() - timing_started,
                            "entries": timing_entries,
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def _finalize_hqq_auto_manifest(self, manifest: dict, expected: list[tuple[int, str]], cache_dir: str) -> None:
        planner = manifest.get("planner")
        if not isinstance(planner, dict):
            raise RuntimeError(f"{self.quant_cfg.attention} manifest is missing planner metadata.")
        mixed = manifest.get("mixed_precision", {})
        budget_pct = mixed.get("budget_pct")
        budget_bytes = int(mixed.get("budget_bytes", 0) or 0)
        if budget_pct is None and budget_bytes <= 0 and self.quant_cfg.attention == "hqq46_auto":
            budget_bytes = hqq46_auto_budget_bytes_from_mib(self.quant_cfg.hqq46_auto_budget_mib)

        candidates = planner.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(f"{self.quant_cfg.attention} planner has no candidates to finalize.")

        candidate_by_key = {}
        for candidate in candidates:
            try:
                key = (int(candidate["layer_idx"]), str(candidate["tensor_name"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"{self.quant_cfg.attention} planner candidate has invalid identity: {candidate}") from exc
            if key in candidate_by_key:
                raise RuntimeError(f"{self.quant_cfg.attention} planner has duplicate candidate for layer {key[0]} {key[1]}")
            for record_key in ("base_record", "promoted_record"):
                record = candidate.get(record_key)
                if not isinstance(record, dict):
                    raise RuntimeError(f"{self.quant_cfg.attention} planner candidate {key} is missing {record_key}")
                file_name = record.get("file")
                if not file_name:
                    raise RuntimeError(f"{self.quant_cfg.attention} planner candidate {key} {record_key} is missing file")
                artifact_path = os.path.join(cache_dir, file_name)
                if not os.path.isfile(artifact_path):
                    raise RuntimeError(f"{self.quant_cfg.attention} planner candidate artifact is missing: {artifact_path}")
            candidate_by_key[key] = candidate

        expected_set = set(expected)
        candidate_set = set(candidate_by_key)
        missing = [f"layer {idx} {name}" for idx, name in expected if (idx, name) not in candidate_set]
        extras = [f"layer {idx} {name}" for idx, name in candidate_set if (idx, name) not in expected_set]
        if missing or extras:
            problems = []
            if missing:
                problems.append("missing " + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""))
            if extras:
                problems.append("extra " + ", ".join(extras[:8]) + ("..." if len(extras) > 8 else ""))
            raise RuntimeError(f"{self.quant_cfg.attention} planner candidate set does not match expected attention tensors: " + "; ".join(problems))

        ordered_candidates = [candidate_by_key[key] for key in expected]
        promotion_span_bytes = sum(int(candidate["extra_bytes"]) for candidate in ordered_candidates)
        if budget_pct is not None:
            budget_bytes = hqq_auto_budget_bytes_from_pct(
                float(budget_pct),
                promotion_span_bytes,
                self.quant_cfg.attention,
            )
        elif budget_bytes <= 0:
            raise RuntimeError(f"{self.quant_cfg.attention} planner has no valid budget.")
        selected_keys, summary = select_hqq_auto_promotions(ordered_candidates, budget_bytes)

        final_entries = []
        candidate_summaries = []
        base_total_bytes = 0
        promoted_total_bytes = 0
        all_promoted_total_bytes = 0
        selected_extra_bytes = 0
        for candidate in ordered_candidates:
            key = (int(candidate["layer_idx"]), str(candidate["tensor_name"]))
            base_record = candidate["base_record"]
            promoted_record = candidate["promoted_record"]
            selected = key in selected_keys
            selected_record = promoted_record if selected else base_record
            unused_record = base_record if selected else promoted_record
            final_entries.append(selected_record)
            base_total_bytes += int(base_record["tensor_bytes"])
            all_promoted_total_bytes += int(promoted_record["tensor_bytes"])
            promoted_total_bytes += int(selected_record["tensor_bytes"])
            if selected:
                selected_extra_bytes += int(candidate["extra_bytes"])
            unused_path = os.path.join(cache_dir, unused_record["file"])
            if os.path.isfile(unused_path):
                os.remove(unused_path)
            candidate_summaries.append(
                {
                    "layer_idx": int(candidate["layer_idx"]),
                    "layer_type": candidate["layer_type"],
                    "tensor_name": candidate["tensor_name"],
                    "selected": selected,
                    "selected_nbits": int(selected_record["nbits"]),
                    "base_bytes": int(candidate["base_bytes"]),
                    "promoted_bytes": int(candidate["promoted_bytes"]),
                    "extra_bytes": int(candidate["extra_bytes"]),
                    "base_relative_rmse": float(candidate["base_relative_rmse"]),
                    "promoted_relative_rmse": float(candidate["promoted_relative_rmse"]),
                    "relative_rmse_reduction": float(candidate["relative_rmse_reduction"]),
                    "score": float(candidate["score"]),
                }
            )

        manifest["tensors"] = final_entries
        manifest["totals"]["tensor_bytes"] = promoted_total_bytes
        manifest["totals"]["num_tensors"] = len(final_entries)
        base_nbits = int(manifest["mixed_precision"]["base_nbits"])
        promoted_nbits = int(manifest["mixed_precision"]["promoted_nbits"])
        manifest["totals"][f"base_hqq{base_nbits}_tensor_bytes"] = base_total_bytes
        manifest["totals"][f"all_hqq{promoted_nbits}_tensor_bytes"] = all_promoted_total_bytes
        manifest["totals"]["promotion_span_bytes"] = promotion_span_bytes
        manifest["totals"]["selected_extra_bytes"] = selected_extra_bytes
        manifest["mixed_precision"]["selected_promotions"] = len(selected_keys)
        manifest["mixed_precision"]["candidate_count"] = len(ordered_candidates)
        manifest["mixed_precision"]["budget_pct"] = None if budget_pct is None else float(budget_pct)
        manifest["mixed_precision"]["promotion_span_bytes"] = promotion_span_bytes
        manifest["mixed_precision"]["budget_bytes"] = budget_bytes
        manifest["mixed_precision"]["budget_used_bytes"] = selected_extra_bytes
        manifest["planner"] = {
            "name": planner.get("name", "weight_error_v1"),
            "budget_pct": manifest.get("mixed_precision", {}).get("budget_pct"),
            "legacy_budget_mib": manifest.get("mixed_precision", {}).get("legacy_budget_mib"),
            "budget_bytes": budget_bytes,
            "summary": summary,
            "candidate_count": len(ordered_candidates),
            "selected_count": len(selected_keys),
            "candidates": candidate_summaries,
        }
        logger.info(
            "%s planner selected %d/%d HQQ%d promotions using %.1f/%.1f MiB extra VRAM-equivalent cache bytes (budget_pct=%s, span=%.1f MiB).",
            self.quant_cfg.attention,
            len(selected_keys),
            len(ordered_candidates),
            promoted_nbits,
            selected_extra_bytes / (1024 * 1024),
            budget_bytes / (1024 * 1024),
            "legacy" if budget_pct is None else f"{float(budget_pct):.3f}",
            promotion_span_bytes / (1024 * 1024),
        )

    def _build_hqq_attention_cache_from_safetensors(self, loader, primary_dev: torch.device) -> None:
        if not getattr(self, "_hqq_rebuild", False):
            return

        logger.info(
            "Building HQQ attention cache in bounded safetensors pass: at most one BF16 layer materialized at a time"
        )
        cpu = torch.device("cpu")
        self._hqq_search_cuda_device = primary_dev if primary_dev.type == "cuda" else None
        started = time.perf_counter()
        for layer_idx in range(self.cfg.num_hidden_layers):
            weights = loader.load_layer(layer_idx, primary_dev, attn_device=cpu)
            layer_type = weights.get("layer_type", "full_attention")
            attn_key = self._attention_weight_key(layer_type)
            self._maybe_write_hqq_attention_artifacts(
                layer_idx,
                layer_type,
                weights.get(attn_key, {}),
            )
            del weights
            if (layer_idx + 1) % 2 == 0 or layer_idx == self.cfg.num_hidden_layers - 1:
                gc.collect()
                torch.cuda.empty_cache()

        self._validate_hqq_attention_cache()
        self._hqq_rebuild = False
        self._hqq_finalize_pending_manifest = False
        logger.info(
            "HQQ attention cache bounded build complete in %.1fs; continuing normal model load from cache",
            time.perf_counter() - started,
        )

    def _validate_hqq_attention_cache(self) -> None:
        cache_nbits = getattr(self, "_hqq_cache_nbits", attention_quant_cache_nbits(self.quant_cfg.attention))
        if cache_nbits is None:
            raise RuntimeError(
                f"HQQ attention backend must declare nbits, got {self.quant_cfg.attention}"
            )
        validate_hqq_cache_nbits(cache_nbits)
        rebuilding = getattr(self, "_hqq_rebuild", False)
        finalize_pending = getattr(self, "_hqq_finalize_pending_manifest", False)
        cache_profile = self.quant_cfg.hqq_cache_profile
        hqq_group_size = self.quant_cfg.hqq_group_size
        manifest = (
            load_hqq_attention_pending_manifest(self.cfg.model_path, cache_profile, cache_nbits, hqq_group_size)
            if rebuilding or finalize_pending
            else load_hqq_attention_manifest(self.cfg.model_path, cache_profile, cache_nbits, hqq_group_size)
        )
        if manifest is None:
            expected_path = (
                hqq_attention_pending_manifest_path(self.cfg.model_path, cache_profile, cache_nbits, hqq_group_size)
                if rebuilding or finalize_pending
                else hqq_attention_manifest_path(self.cfg.model_path, cache_profile, cache_nbits, hqq_group_size)
            )
            raise RuntimeError(
                f"attention_quant={self.quant_cfg.attention} requested but no HQQ attention manifest exists. "
                f"Expected {expected_path}"
            )
        expected_backend = hqq_backend_name(cache_nbits)
        if (
            manifest.get("format_version") != HQQ_ATTENTION_CACHE_VERSION
            or manifest.get("backend") != expected_backend
            or manifest.get("group_size") != hqq_group_size
            or manifest.get("quantizer") != hqq_cache_algorithm_for_nbits(cache_nbits)
        ):
            raise RuntimeError(
                "HQQ attention cache manifest is incompatible with this build. "
                f"Found format_version={manifest.get('format_version')} backend={manifest.get('backend')} "
                f"group_size={manifest.get('group_size')} expected_group_size={hqq_group_size} "
                f"quantizer={manifest.get('quantizer')} expected_quantizer={hqq_cache_algorithm_for_nbits(cache_nbits)}"
            )

        expected = list(getattr(self, "_hqq_expected_tensors", []) or [])
        if not expected:
            for layer_idx, layer in enumerate(self.layers):
                layer_weights = self._extract_layer_weights(layer, layer.device)
                layer_type = layer_weights.get("layer_type", "full_attention")
                attn_key = self._attention_weight_key(layer_type)
                attn_weights = layer_weights.get(attn_key, {})
                expected_tensor_names = list(self._hqq_attention_tensor_map(layer_type, attn_weights))
                if (
                    layer_type in ("full_attention", "sliding_attention")
                    and not self.cfg.is_mla
                    and "fused_qkv" not in expected_tensor_names
                    and all(isinstance(attn_weights.get(name), torch.Tensor) for name in ("q_proj", "k_proj", "v_proj"))
                ):
                    expected_tensor_names.append("fused_qkv")
                for tensor_name in expected_tensor_names:
                    expected.append((layer_idx, tensor_name))

        if is_hqq_auto_attention(self.quant_cfg.attention):
            cache_dir = hqq_attention_cache_dir(self.cfg.model_path, cache_profile, cache_nbits, hqq_group_size)
            if not manifest.get("tensors") or len(manifest.get("tensors", [])) != len(expected):
                self._finalize_hqq_auto_manifest(manifest, expected, cache_dir)

        entries = {}
        duplicates = []
        for entry in manifest.get("tensors", []):
            key = (entry["layer_idx"], entry["tensor_name"])
            if key in entries:
                duplicates.append(f"layer {key[0]} {key[1]}")
            entries[key] = entry
        if duplicates:
            raise RuntimeError(
                "HQQ attention cache manifest has duplicate artifacts: "
                + ", ".join(duplicates[:8])
                + ("..." if len(duplicates) > 8 else "")
            )
        expected_set = set(expected)
        extras = [f"layer {idx} {name}" for idx, name in entries if (idx, name) not in expected_set]
        if extras:
            raise RuntimeError(
                "HQQ attention cache manifest has unexpected artifacts: "
                + ", ".join(extras[:8])
                + ("..." if len(extras) > 8 else "")
            )
        missing = [f"layer {idx} {name}" for idx, name in expected if (idx, name) not in entries]
        if missing:
            raise RuntimeError(
                "HQQ attention cache is incomplete. Missing artifacts: "
                + ", ".join(missing[:8])
                + ("..." if len(missing) > 8 else "")
            )

        total_bytes = 0
        totals_by_nbits = {}
        for layer_idx, tensor_name in expected:
            entry = entries[(layer_idx, tensor_name)]
            if "structure" not in entry or "quality" not in entry:
                raise RuntimeError(
                    "HQQ attention cache is missing measured structure/quality metadata. "
                    f"Rebuild required for layer {layer_idx} {tensor_name}."
                )
            artifact = load_hqq_attention_artifact(
                self.cfg.model_path,
                entry,
                expected_nbits=int(entry["nbits"]),
                device="cpu",
                cache_profile=cache_profile,
                group_size=hqq_group_size,
                cache_nbits=cache_nbits,
            )
            if artifact["tensor_bytes"] != entry["tensor_bytes"]:
                raise RuntimeError(
                    f"HQQ artifact byte mismatch for layer {layer_idx} {tensor_name}: "
                    f"manifest={entry['tensor_bytes']} actual={artifact['tensor_bytes']}"
                )
            total_bytes += artifact["tensor_bytes"]
            entry_nbits = int(entry["nbits"])
            totals_by_nbits[str(entry_nbits)] = totals_by_nbits.get(str(entry_nbits), 0) + artifact["tensor_bytes"]
        manifest["complete"] = True
        manifest["totals"]["tensor_bytes"] = total_bytes
        manifest["totals"]["num_tensors"] = len(expected)
        if totals_by_nbits:
            manifest["totals"]["tensor_bytes_by_nbits"] = totals_by_nbits
        save_hqq_attention_manifest(self.cfg.model_path, manifest, cache_profile, cache_nbits, hqq_group_size)
        delete_hqq_attention_pending_manifest(self.cfg.model_path, cache_profile, cache_nbits, hqq_group_size)
        self._hqq_manifest = manifest
        self._hqq_attention_cache_bytes = total_bytes

    def _load_hqq_attention_runtime_state(self) -> None:
        cache_nbits = getattr(self, "_hqq_cache_nbits", attention_quant_cache_nbits(self.quant_cfg.attention))
        if cache_nbits is None:
            raise RuntimeError(
                f"HQQ attention backend must declare nbits, got {self.quant_cfg.attention}"
            )
        validate_hqq_cache_nbits(cache_nbits)
        manifest = getattr(self, "_hqq_manifest", None)
        if not manifest or not manifest.get("complete"):
            raise RuntimeError(
                "HQQ attention runtime load requires a complete validated manifest."
            )

        runtime_layers = {}
        loaded_tensors = 0
        loaded_bytes = 0
        for entry in manifest.get("tensors", []):
            layer_idx = entry["layer_idx"]
            tensor_name = entry["tensor_name"]
            artifact = load_hqq_attention_artifact(
                self.cfg.model_path,
                entry,
                expected_nbits=int(entry["nbits"]),
                device="cpu",
                cache_profile=self.quant_cfg.hqq_cache_profile,
                group_size=self.quant_cfg.hqq_group_size,
                cache_nbits=cache_nbits,
            )
            if artifact["tensor_bytes"] != entry["tensor_bytes"]:
                raise RuntimeError(
                    f"HQQ artifact byte mismatch for layer {layer_idx} {tensor_name}: "
                    f"manifest={entry['tensor_bytes']} actual={artifact['tensor_bytes']}"
                )
            if artifact["structure"] != entry.get("structure"):
                raise RuntimeError(
                    f"HQQ artifact structure mismatch for layer {layer_idx} {tensor_name}: "
                    f"manifest={entry.get('structure')} actual={artifact['structure']}"
                )

            tensors = artifact["tensors"]
            group_size = int(tensors["group_size"][0].item())
            axis = int(tensors["axis"][0].item())
            stored_nbits = int(tensors["nbits"][0].item())
            if stored_nbits != int(entry["nbits"]):
                raise RuntimeError(
                    f"HQQ artifact tensor nbits mismatch for layer {layer_idx} {tensor_name}: "
                    f"stored={stored_nbits} expected={entry['nbits']}"
                )

            runtime_entry = {
                "backend": manifest["backend"],
                "format_version": int(manifest["format_version"]),
                "nbits": stored_nbits,
                "layout": entry["layout"],
                "group_size": group_size,
                "axis": axis,
                "orig_shape": tuple(int(v) for v in tensors["orig_shape"].tolist()),
                "packed": tensors["packed"],
                "scales": tensors["scales"],
                "zeros": tensors["zeros"],
                "packed_dtype": str(tensors["packed"].dtype).replace("torch.", ""),
                "scales_dtype": str(tensors["scales"].dtype).replace("torch.", ""),
                "zeros_dtype": str(tensors["zeros"].dtype).replace("torch.", ""),
                "original_dtype": entry.get("original_dtype", artifact["metadata"].get("dtype", "")),
                "path": artifact["path"],
                "tensor_bytes": artifact["tensor_bytes"],
            }
            runtime_layers.setdefault(layer_idx, {})[tensor_name] = runtime_entry
            loaded_tensors += 1
            loaded_bytes += artifact["tensor_bytes"]

        if loaded_bytes != self._hqq_attention_cache_bytes:
            raise RuntimeError(
                f"HQQ runtime byte mismatch after load: loaded={loaded_bytes} "
                f"validated={self._hqq_attention_cache_bytes}"
            )

        self._hqq_attention_runtime = runtime_layers
        self._hqq_attention_runtime_nbits = cache_nbits
        self._hqq_attention_loaded_tensors = loaded_tensors
        self._hqq_prefill_sidecar_runtime = self._load_hqq_prefill_sidecar_runtime(runtime_layers)
        for layer_idx, layer in enumerate(self.layers):
            setattr(layer, "_hqq_attention_runtime", runtime_layers.get(layer_idx, {}))

    def _load_hqq_prefill_sidecar_runtime(self, runtime_layers: dict) -> dict:
        sidecar_path = getattr(self.quant_cfg, "hqq_sidecar_manifest", None)
        if not sidecar_path:
            return {}
        if self.quant_cfg.attention != "hqq4":
            raise RuntimeError(
                "HQQ sidecar/self-correction is only supported for attention_quant=hqq4. "
                f"Found attention_quant={self.quant_cfg.attention!r}; HQQ8 must run without sidecars."
            )
        manifest = require_complete_hqq_sidecar_manifest(
            sidecar_path,
            model_path=self.cfg.model_path,
            source_cache_profile=self.quant_cfg.hqq_cache_profile,
        )
        manifest_path = manifest["_manifest_path"]
        base_dir = os.path.dirname(manifest_path)
        mode = str(manifest["sidecar_mode"])
        variant_name = str(manifest.get("variant_name", ""))
        sidecars_by_layer = {}
        total_rows = 0
        total_bytes = 0

        def _required_tensor(handle, name: str, path: str) -> torch.Tensor:
            if name not in handle.keys():
                raise RuntimeError(f"HQQ sidecar artifact {path} is missing tensor {name}")
            return handle.get_tensor(name).contiguous()

        def _build_int8_exception_base_f32(
            runtime: dict,
            output_rows: torch.Tensor,
            groups: torch.Tensor,
            start_cols: torch.Tensor,
            widths: torch.Tensor,
            max_width: int,
            path: str,
        ) -> torch.Tensor:
            packed = runtime["packed"].contiguous()
            scales = runtime["scales"].contiguous()
            zeros = runtime["zeros"].contiguous()
            rows = int(runtime["orig_shape"][0])
            cols = int(runtime["orig_shape"][1])
            group_size = int(runtime["group_size"])
            row_group_count = int(output_rows.numel())
            if packed.ndim != 2 or scales.ndim != 2 or zeros.ndim != 2:
                raise RuntimeError(f"HQQ sidecar base build expects 2D HQQ tensors for {path}")
            if packed.dtype != torch.uint8 or scales.dtype != torch.float32 or zeros.dtype != torch.float32:
                raise RuntimeError(
                    f"HQQ sidecar base build dtype mismatch for {path}: "
                    f"packed={packed.dtype} scales={scales.dtype} zeros={zeros.dtype}"
                )
            if int(packed.shape[0]) != rows or int(scales.shape[0]) != rows or int(zeros.shape[0]) != rows:
                raise RuntimeError(f"HQQ sidecar base build row-shape mismatch for {path}")
            if int(packed.shape[1]) < (cols + 1) // 2:
                raise RuntimeError(f"HQQ sidecar base build packed width too small for {path}")
            if int(scales.shape[1]) <= int(groups.max().item()) or int(zeros.shape[1]) <= int(groups.max().item()):
                raise RuntimeError(f"HQQ sidecar base build group index out of bounds for {path}")

            local_cols = torch.arange(max_width, dtype=torch.long).unsqueeze(0)
            widths_l = widths.to(torch.long).unsqueeze(1)
            start_l = start_cols.to(torch.long).unsqueeze(1)
            mask = local_cols < widths_l
            cols_l = start_l + local_cols
            safe_cols = torch.where(mask, cols_l, start_l)
            rows_l = output_rows.to(torch.long).unsqueeze(1).expand(row_group_count, max_width)
            byte_idx = safe_cols // 2
            packed_vals = packed[rows_l, byte_idx]
            low = packed_vals & 0x0F
            high = packed_vals >> 4
            q = torch.where((safe_cols & 1) == 0, low, high).to(torch.float32)
            row_idx = output_rows.to(torch.long)
            group_idx = groups.to(torch.long)
            group_start = group_idx * group_size
            group_end = group_start + group_size
            if bool(((start_cols.to(torch.long) < group_start) | ((start_cols + widths).to(torch.long) > group_end)).any().item()):
                raise RuntimeError(
                    f"HQQ sidecar base build requires each entry to stay inside one HQQ group: {path}"
                )
            scale = scales[row_idx, group_idx].unsqueeze(1)
            zero = zeros[row_idx, group_idx].unsqueeze(1)
            base = (q - zero) * scale
            base = torch.where(mask, base, torch.zeros_like(base))
            return base.to(torch.float32).contiguous()

        for artifact in manifest.get("artifacts", []):
            layer_idx = int(artifact["layer"])
            tensor_name = str(artifact["tensor"])
            runtime = runtime_layers.get(layer_idx, {}).get(tensor_name)
            if runtime is None:
                raise RuntimeError(
                    f"HQQ sidecar artifact targets missing HQQ runtime tensor: layer={layer_idx} tensor={tensor_name}"
                )
            path = os.path.join(base_dir, artifact["file"])
            with safe_open(path, framework="pt", device="cpu") as handle:
                output_rows_raw = _required_tensor(handle, "output_rows", path)
                start_cols_raw = _required_tensor(handle, "start_cols", path)
                widths_raw = _required_tensor(handle, "widths", path)
                groups_raw = _required_tensor(handle, "groups", path)
                scales_raw = _required_tensor(handle, "scales", path)
                for tensor_label, tensor_value, expected_dtype in (
                    ("output_rows", output_rows_raw, torch.int32),
                    ("start_cols", start_cols_raw, torch.int32),
                    ("widths", widths_raw, torch.int32),
                    ("groups", groups_raw, torch.int32),
                    ("scales", scales_raw, torch.float32),
                ):
                    if tensor_value.dtype != expected_dtype:
                        raise RuntimeError(
                            f"HQQ sidecar artifact {path} tensor {tensor_label} has dtype "
                            f"{tensor_value.dtype}, expected {expected_dtype}"
                        )
                output_rows = output_rows_raw.contiguous()
                start_cols = start_cols_raw.contiguous()
                widths = widths_raw.contiguous()
                groups = groups_raw.contiguous()
                scales = scales_raw.contiguous()
                if mode == "int8_symmetric":
                    correction = _required_tensor(handle, "correction_qint8", path)
                    if correction.dtype != torch.int8:
                        raise RuntimeError(f"HQQ sidecar correction_qint8 dtype mismatch for {path}")
                    correction = correction.contiguous()
                elif mode == "int8_exception":
                    correction = _required_tensor(handle, "exception_qint8", path)
                    if correction.dtype != torch.int8:
                        raise RuntimeError(f"HQQ sidecar exception_qint8 dtype mismatch for {path}")
                    correction = correction.contiguous()
                elif mode == "exact_bf16":
                    correction = _required_tensor(handle, "correction_bf16", path)
                    if correction.dtype != torch.bfloat16:
                        raise RuntimeError(f"HQQ sidecar correction_bf16 dtype mismatch for {path}")
                    correction = correction.contiguous()
                else:
                    raise RuntimeError(f"Unsupported HQQ sidecar mode {mode!r}")
            row_group_count = int(output_rows.numel())
            if row_group_count <= 0:
                raise RuntimeError(f"HQQ sidecar artifact has no row/groups: {path}")
            if correction.ndim != 2:
                raise RuntimeError(f"HQQ sidecar payload tensor must be 2D: {path}")
            if correction.shape[0] != row_group_count:
                raise RuntimeError(
                    f"HQQ sidecar payload row count mismatch for {path}: "
                    f"{correction.shape[0]} vs {row_group_count}"
                )
            for tensor_name_check, tensor in (
                ("start_cols", start_cols),
                ("widths", widths),
                ("groups", groups),
                ("scales", scales),
            ):
                if int(tensor.numel()) != row_group_count:
                    raise RuntimeError(
                        f"HQQ sidecar {tensor_name_check} length mismatch for {path}: "
                        f"{tensor.numel()} vs {row_group_count}"
                    )
            rows = int(runtime["orig_shape"][0])
            cols = int(runtime["orig_shape"][1])
            if int(output_rows.min().item()) < 0 or int(output_rows.max().item()) >= rows:
                raise RuntimeError(f"HQQ sidecar output row out of bounds for {path}: rows={rows}")
            if int(start_cols.min().item()) < 0:
                raise RuntimeError(f"HQQ sidecar start_col out of bounds for {path}")
            if int(widths.min().item()) <= 0:
                raise RuntimeError(f"HQQ sidecar width must be positive for {path}")
            if int((start_cols + widths).max().item()) > cols:
                raise RuntimeError(f"HQQ sidecar column range out of bounds for {path}: cols={cols}")
            if int(widths.max().item()) > int(correction.shape[1]):
                raise RuntimeError(f"HQQ sidecar payload width is smaller than metadata widths for {path}")
            if mode == "int8_exception":
                base_f32 = _build_int8_exception_base_f32(
                    runtime,
                    output_rows,
                    groups,
                    start_cols,
                    widths,
                    int(correction.shape[1]),
                    path,
                )
            else:
                base_f32 = torch.empty((0,), dtype=torch.float32)
            sidecars_by_layer.setdefault(layer_idx, []).append(
                {
                    "tensor_name": tensor_name,
                    "mode": mode,
                    "variant_name": variant_name,
                    "path": path,
                    "correction": correction,
                    "scales": scales,
                    "output_rows": output_rows,
                    "groups": groups,
                    "start_cols": start_cols,
                    "widths": widths,
                    "base_f32": base_f32,
                    "row_group_count": row_group_count,
                    "max_width": int(correction.shape[1]),
                }
            )
            total_rows += row_group_count
            total_bytes += correction.numel() * correction.element_size()
            total_bytes += scales.numel() * scales.element_size()
            total_bytes += base_f32.numel() * base_f32.element_size()
            total_bytes += sum(t.numel() * t.element_size() for t in (output_rows, groups, start_cols, widths))

        logger.info(
            "Loaded HQQ prefill sidecar manifest %s: variant=%s mode=%s artifacts=%d row_groups=%d bytes=%d source_profile=%s",
            manifest_path,
            variant_name,
            mode,
            sum(len(v) for v in sidecars_by_layer.values()),
            total_rows,
            total_bytes,
            self.quant_cfg.hqq_cache_profile,
        )
        return sidecars_by_layer

    @staticmethod
    def _hqq_layer_kind(layer: TransformerLayer) -> str:
        if getattr(layer.cfg, "is_deepseek_v4", False):
            return "deepseek_v4"
        if layer.layer_type == "linear_attention":
            if layer.cfg.is_kimi_delta_attention_layer(layer.layer_idx):
                return "kimi_delta_attention"
            return "linear_attention"
        attn = layer.attention
        if attn is not None and hasattr(attn, "kv_a_proj"):
            return "mla"
        return "gqa"

    @staticmethod
    def _move_hqq_tensor_to_device(
        tensor: torch.Tensor,
        device: torch.device,
        keepalive: list,
        label: str = "hqq_attention_meta",
    ) -> torch.Tensor:
        if tensor.device != device:
            tensor = tensor.to(device, non_blocking=True)
        # HQQ metadata frequently passes same-device temporaries such as
        # `.float().contiguous()` into Rust by raw pointer. Keep every returned
        # tensor alive, even when no device copy was needed.
        if hasattr(keepalive, "keep"):
            keepalive.keep(label, tensor)
        else:
            keepalive.append(tensor)
        return tensor

    def _gqa_layer_rope_params(self, layer_idx: int) -> dict:
        rope_params = self.cfg.rope_scaling if isinstance(self.cfg.rope_scaling, dict) else {}
        if self.cfg.gemma4_text and isinstance(rope_params, dict):
            layer_type = "sliding_attention" if self.cfg.is_sliding_attention_layer(layer_idx) else "full_attention"
            return dict(rope_params.get(layer_type, {}) or {})
        if self.cfg.step3_text and isinstance(rope_params, dict):
            layer_type = "sliding_attention" if self.cfg.is_sliding_attention_layer(layer_idx) else "full_attention"
            yarn_only_types = getattr(self.cfg, "yarn_only_types", None)
            if yarn_only_types and layer_type not in yarn_only_types:
                return {}
            return dict(rope_params)
        return {}

    def _gqa_inv_freq_for_layer(
        self,
        layer_idx: int,
        head_dim_for_rope: int,
        rope_half: int,
        theta: float,
        layer_rope_params: dict,
    ) -> torch.Tensor:
        if self.cfg.gemma4_text and layer_rope_params.get("rope_type") == "proportional":
            rope_proportion = float(layer_rope_params.get("partial_rotary_factor", 1.0))
            rope_angles = int(rope_proportion * head_dim_for_rope // 2)
            inv_freq_rotated = 1.0 / (
                theta ** (
                    torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                    / head_dim_for_rope
                )
            )
            nope_angles = head_dim_for_rope // 2 - rope_angles
            if nope_angles > 0:
                return torch.cat(
                    (inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)),
                    dim=0,
                )
            return inv_freq_rotated

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, rope_half * 2, 2, dtype=torch.float32)
                / (rope_half * 2)
            )
        )
        rope_type = layer_rope_params.get("rope_type", layer_rope_params.get("type", "default"))
        if rope_type != "llama3":
            return inv_freq

        required = ("factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor")
        missing = [name for name in required if name not in layer_rope_params]
        if missing:
            raise ValueError(f"Llama3 RoPE parameters missing required fields: {missing}")

        factor = float(layer_rope_params["factor"])
        original_max_position_embeddings = float(layer_rope_params["original_max_position_embeddings"])
        low_freq_factor = float(layer_rope_params["low_freq_factor"])
        high_freq_factor = float(layer_rope_params["high_freq_factor"])
        if factor <= 0.0 or high_freq_factor == low_freq_factor:
            raise ValueError(f"Invalid Llama3 RoPE parameters for layer {layer_idx}: {layer_rope_params}")

        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor
        wavelen = (2.0 * pi) / inv_freq
        scaled_inv_freq = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (
            (original_max_position_embeddings / wavelen) - low_freq_factor
        ) / (high_freq_factor - low_freq_factor)
        smoothed = (1.0 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        is_medium = torch.logical_and(
            ~(wavelen < high_freq_wavelen),
            ~(wavelen > low_freq_wavelen),
        )
        return torch.where(is_medium, smoothed, scaled_inv_freq)

    def _gqa_rope_table_ptrs(
        self,
        layer_idx: int,
        max_seq: int,
        target_device: torch.device,
    ) -> tuple[int, int, int, torch.Tensor | None, torch.Tensor | None]:
        """Build or reuse the per-layer GQA RoPE tables on a target GPU."""
        layer_rope_params = self._gqa_layer_rope_params(layer_idx)
        head_dim_for_rope = self.cfg.gqa_head_dim_for_layer(layer_idx)
        if self.cfg.gemma4_text and layer_rope_params.get("rope_type") == "proportional":
            rope_half = head_dim_for_rope // 2
        else:
            rope_half = int(self.cfg.rotary_dim_for_layer(layer_idx) // 2)
        if rope_half <= 0:
            return 0, 0, 0, None, None

        theta = float(self.cfg.rope_theta_for_layer(layer_idx))
        key = (
            int(max_seq),
            rope_half,
            theta,
            str(target_device),
            json.dumps(layer_rope_params, sort_keys=True, default=str),
            int(head_dim_for_rope),
        )
        rope_tables = getattr(self, "_rust_layer_rope_tables", None)
        if rope_tables is None:
            rope_tables = {}
            self._rust_layer_rope_tables = rope_tables
        cached = rope_tables.get(key)
        if cached is None:
            inv_freq = self._gqa_inv_freq_for_layer(
                layer_idx,
                head_dim_for_rope,
                rope_half,
                theta,
                layer_rope_params,
            )
            t = torch.arange(max_seq, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            cos_f32 = freqs.cos().contiguous().to(target_device)
            sin_f32 = freqs.sin().contiguous().to(target_device)
            cached = (cos_f32, sin_f32)
            rope_tables[key] = cached
            self._keep_rust_decode_weight("gqa_rope_table", cos_f32, sin_f32)
        cos_f32, sin_f32 = cached
        return (
            int(cos_f32.data_ptr()),
            int(sin_f32.data_ptr()),
            rope_half,
            cos_f32,
            sin_f32,
        )

    def _deepseek_v4_rope_table_ptrs(
        self,
        max_seq: int,
        compress_ratio: int,
        target_device: torch.device,
    ) -> tuple[int, int, int, torch.Tensor, torch.Tensor]:
        """Build the exact shipped DeepSeek-V4 RoPE table for one layer mode.

        Pure sliding-window layers deliberately disable YaRN and use the base
        model theta. Compressed layers use the checkpoint compression theta and
        its YaRN parameters. These setup-owned FP32 tables are consumed only by
        Rust/CUDA at runtime.
        """
        rope_dim = int(self.cfg.qk_rope_head_dim)
        if max_seq <= 0 or rope_dim <= 0 or rope_dim % 2:
            raise ValueError(
                f"Invalid DeepSeek-V4 RoPE contract max_seq={max_seq}, dim={rope_dim}"
            )
        if compress_ratio not in (0, 4, 128):
            raise ValueError(
                f"Unsupported DeepSeek-V4 compression ratio {compress_ratio}"
            )

        rope_params = self.cfg.rope_scaling or {}
        if not isinstance(rope_params, dict):
            raise ValueError("DeepSeek-V4 rope_scaling must be an object")
        if compress_ratio:
            original_seq_len = int(
                rope_params.get("original_max_position_embeddings", 0) or 0
            )
            base = float(self.cfg.compress_rope_theta)
            factor = float(rope_params.get("factor", 1.0))
            beta_fast = float(rope_params.get("beta_fast", 32.0))
            beta_slow = float(rope_params.get("beta_slow", 1.0))
            if original_seq_len <= 0 or factor <= 0.0:
                raise ValueError(
                    "Compressed DeepSeek-V4 RoPE requires positive YaRN "
                    f"original length/factor, got {original_seq_len}/{factor}"
                )
        else:
            original_seq_len = 0
            base = float(self.cfg.rope_theta)
            factor = 1.0
            beta_fast = 32.0
            beta_slow = 1.0
        if base <= 0.0:
            raise ValueError(f"Invalid DeepSeek-V4 RoPE base {base}")

        cache = getattr(self, "_rust_deepseek_v4_rope_tables", None)
        if cache is None:
            cache = {}
            self._rust_deepseek_v4_rope_tables = cache
        key = (
            max_seq,
            compress_ratio,
            rope_dim,
            base,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
            str(target_device),
        )
        cached = cache.get(key)
        if cached is None:
            half = rope_dim // 2
            freqs = 1.0 / (
                base
                ** (
                    torch.arange(0, rope_dim, 2, dtype=torch.float32)
                    / float(rope_dim)
                )
            )
            if original_seq_len > 0:
                def _correction_dim(rotations: float) -> float:
                    return rope_dim * math.log(
                        original_seq_len / (rotations * 2.0 * math.pi)
                    ) / (2.0 * math.log(base))

                low = max(math.floor(_correction_dim(beta_fast)), 0)
                high = min(math.ceil(_correction_dim(beta_slow)), rope_dim - 1)
                ramp_high = float(high) if low != high else float(high) + 0.001
                ramp = torch.clamp(
                    (torch.arange(half, dtype=torch.float32) - float(low))
                    / (ramp_high - float(low)),
                    0.0,
                    1.0,
                )
                smooth = 1.0 - ramp
                freqs = freqs / factor * (1.0 - smooth) + freqs * smooth

            positions = torch.arange(max_seq, dtype=torch.float32)
            angles = torch.outer(positions, freqs)
            cos_f32 = angles.cos().contiguous().to(target_device)
            sin_f32 = angles.sin().contiguous().to(target_device)
            cached = (cos_f32, sin_f32)
            cache[key] = cached
            self._keep_rust_decode_weight(
                "deepseek_v4_rope_table", cos_f32, sin_f32
            )
        cos_f32, sin_f32 = cached
        return (
            int(cos_f32.data_ptr()),
            int(sin_f32.data_ptr()),
            int(cos_f32.shape[0]),
            cos_f32,
            sin_f32,
        )

    def _keep_rust_decode_weight(self, label: str, *values) -> None:
        keepalive = getattr(self, "_rust_decode_weights", None)
        if keepalive is None:
            return
        if hasattr(keepalive, "keep"):
            keepalive.keep(label, *values)
        else:
            keepalive.extend(values)

    def _release_hqq_bf16_attention_residency(self) -> None:
        """Drop BF16 attention projection tensors replaced by HQQ runtime descriptors."""
        released_bytes = 0
        released_tensors = 0

        def _release_attr(obj, attr_name: str) -> None:
            nonlocal released_bytes, released_tensors
            if obj is None or not hasattr(obj, attr_name):
                return
            value = getattr(obj, attr_name)
            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    released_bytes += value.numel() * value.element_size()
                released_tensors += 1
                setattr(obj, attr_name, None)

        def _release_dict_tensor(dct, key: str) -> None:
            nonlocal released_bytes, released_tensors
            if not isinstance(dct, dict) or key not in dct:
                return
            value = dct.get(key)
            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    released_bytes += value.numel() * value.element_size()
                released_tensors += 1
                dct[key] = None

        free_before_mb = None
        if torch.cuda.is_available():
            try:
                free_before_mb = torch.cuda.mem_get_info()[0] / (1024.0 * 1024.0)
            except Exception:
                free_before_mb = None

        for hqq_layer in self.layers:
            attn_obj = hqq_layer.attention
            if getattr(hqq_layer.cfg, "is_deepseek_v4", False):
                if attn_obj is None or not hasattr(attn_obj, "attention"):
                    continue
                v4 = attn_obj.attention
                for name in ("wq_a", "wq_b", "wkv", "wo_b"):
                    _release_dict_tensor(v4, name)
            elif hqq_layer.layer_type == "linear_attention":
                if attn_obj is None:
                    continue
                if hqq_layer.cfg.is_kimi_delta_attention_layer(
                    hqq_layer.layer_idx
                ):
                    for name in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "f_a_proj",
                        "f_b_proj",
                        "b_proj",
                        "g_a_proj",
                        "g_b_proj",
                    ):
                        _release_dict_tensor(attn_obj.weights, name)
                else:
                    for name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                        _release_attr(attn_obj, name)
            elif attn_obj is not None and hasattr(attn_obj, "kv_a_proj"):
                for name in (
                    "q_proj", "q_a_proj", "q_b_proj",
                    "kv_a_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                    "o_proj",
                ):
                    _release_attr(attn_obj, name)
            else:
                gqa_w = getattr(hqq_layer, "gqa_weights", None)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj", "fused_qkv"):
                    _release_dict_tensor(gqa_w, name)
                    if attn_obj is not None:
                        _release_attr(attn_obj, name)

        if released_tensors:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            free_after_mb = None
            if torch.cuda.is_available():
                try:
                    free_after_mb = torch.cuda.mem_get_info()[0] / (1024.0 * 1024.0)
                except Exception:
                    free_after_mb = None
            if free_before_mb is not None and free_after_mb is not None:
                logger.info(
                    "HQQ BF16 attention projection residency released before VMM registration: tensors=%d cuda_mb=%.2f free_before=%.0f MB free_after=%.0f MB",
                    released_tensors,
                    released_bytes / (1024.0 * 1024.0),
                    free_before_mb,
                    free_after_mb,
                )
            else:
                logger.info(
                    "HQQ BF16 attention projection residency released before VMM registration: tensors=%d cuda_mb=%.2f",
                    released_tensors,
                    released_bytes / (1024.0 * 1024.0),
                )

    def _mamba2_projection_int4_requested(self) -> bool:
        if self.cfg.model_type != "nemotron_h":
            return False
        raw = os.environ.get("KRASIS_MAMBA2_PROJECTION_INT4")
        if raw is not None:
            return raw.strip().lower() not in ("0", "false", "off", "no")
        defaults = os.environ.get("KRASIS_NEMOTRON_DEFAULT_OPTIMIZATIONS", "1").strip().lower()
        if defaults in ("0", "false", "off", "no"):
            return False
        return self.quant_cfg.attention == "hqq4"

    def _mamba2_projection_int4_group_size(self) -> int:
        return int(self.quant_cfg.hqq_group_size)

    def _mamba2_projection_int4_source_name(self, layer_idx: int, tensor_name: str) -> str:
        return f"{self.cfg.layers_prefix}.layers.{int(layer_idx)}.mixer.{tensor_name}.weight"

    def _mamba2_projection_int4_update_digest_from_file(self, digest, path: str) -> None:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)

    def _mamba2_projection_int4_source_signature(
        self,
        expected: list[tuple[int, str, list[int]]],
    ) -> dict:
        model_path = os.path.abspath(os.path.expanduser(self.cfg.model_path))
        digest = hashlib.sha256()
        digest.update(MAMBA2_PROJECTION_INT4_CACHE_FORMAT.encode("utf-8"))
        digest.update(str(MAMBA2_PROJECTION_INT4_CACHE_VERSION).encode("utf-8"))
        digest.update(str(model_path).encode("utf-8"))
        digest.update(str(self.cfg.model_type).encode("utf-8"))
        digest.update(str(self.cfg.num_hidden_layers).encode("utf-8"))
        digest.update(str(bool(self.cfg.rescale_prenorm_residual)).encode("utf-8"))

        source_files = []
        for rel_path in ("config.json", "model.safetensors.index.json"):
            path = os.path.join(model_path, rel_path)
            if not os.path.isfile(path):
                continue
            stat = os.stat(path)
            digest.update(rel_path.encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            self._mamba2_projection_int4_update_digest_from_file(digest, path)
            source_files.append({
                "file": rel_path,
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "role": "metadata",
            })

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        weight_map = {}
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map", {}) or {}

        if weight_map:
            for layer_idx, tensor_name, _shape in expected:
                source_name = self._mamba2_projection_int4_source_name(layer_idx, tensor_name)
                shard_name = weight_map.get(source_name)
                if not shard_name:
                    raise RuntimeError(
                        "Mamba2 projection INT4 source signature could not find source tensor "
                        f"{source_name} in model.safetensors.index.json."
                    )
                shard_path = os.path.join(model_path, shard_name)
                if not os.path.isfile(shard_path):
                    raise RuntimeError(
                        "Mamba2 projection INT4 source signature could not find shard "
                        f"{shard_path} for source tensor {source_name}."
                    )
                stat = os.stat(shard_path)
                digest.update(source_name.encode("utf-8"))
                digest.update(shard_name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("utf-8"))
                digest.update(str(stat.st_mtime_ns).encode("utf-8"))
                source_files.append({
                    "layer_idx": int(layer_idx),
                    "tensor_name": tensor_name,
                    "source_name": source_name,
                    "file": shard_name,
                    "bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "role": "projection_shard",
                })
        else:
            safetensors_files = sorted(
                name for name in os.listdir(model_path) if name.endswith(".safetensors")
            )
            if not safetensors_files:
                raise RuntimeError(
                    f"Mamba2 projection INT4 source signature found no safetensors files in {model_path}."
                )
            for file_name in safetensors_files:
                path = os.path.join(model_path, file_name)
                stat = os.stat(path)
                digest.update(file_name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("utf-8"))
                digest.update(str(stat.st_mtime_ns).encode("utf-8"))
                source_files.append({
                    "file": file_name,
                    "bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "role": "projection_shard_fallback",
                })

        return {
            "signature": digest.hexdigest(),
            "model_path": model_path,
            "files": source_files,
        }

    def _mamba2_projection_int4_cache_dir(self) -> str:
        group_size = self._mamba2_projection_int4_group_size()
        dirname = (
            f"mamba2_projection_marlin_int4_v{MAMBA2_PROJECTION_INT4_CACHE_VERSION}"
            f"_g{group_size}"
        )
        return os.path.join(cache_dir_for_model(self.cfg.model_path), dirname)

    def _mamba2_projection_int4_manifest_path(self) -> str:
        return os.path.join(self._mamba2_projection_int4_cache_dir(), "manifest.json")

    def _mamba2_projection_int4_expected(self) -> list[tuple[int, str, list[int]]]:
        expected = []
        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_type != "mamba2":
                continue
            weights = layer.mamba2_weights or {}
            for tensor_name in ("in_proj", "out_proj"):
                tensor = weights.get(tensor_name)
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(
                        f"Mamba2 projection INT4 cache requested but layer {layer_idx} "
                        f"{tensor_name} source tensor is unavailable."
                    )
                if tensor.dim() != 2:
                    raise RuntimeError(
                        f"Mamba2 projection INT4 cache requires 2D tensors; layer {layer_idx} "
                        f"{tensor_name} has shape {tuple(tensor.shape)}."
                    )
                expected.append((layer_idx, tensor_name, [int(tensor.shape[0]), int(tensor.shape[1])]))
        return expected

    def _mamba2_projection_int4_manifest_base(self, expected: list[tuple[int, str, list[int]]]) -> dict:
        group_size = self._mamba2_projection_int4_group_size()
        source_signature = self._mamba2_projection_int4_source_signature(expected)
        return {
            "format": MAMBA2_PROJECTION_INT4_CACHE_FORMAT,
            "format_version": MAMBA2_PROJECTION_INT4_CACHE_VERSION,
            "backend": "marlin_int4_single_slot",
            "nbits": 4,
            "group_size": group_size,
            "num_hidden_layers": int(self.cfg.num_hidden_layers),
            "model_type": str(self.cfg.model_type),
            "rescale_prenorm_residual": bool(self.cfg.rescale_prenorm_residual),
            "model_path": source_signature["model_path"],
            "source_signature": source_signature["signature"],
            "source_files": source_signature["files"],
            "complete": False,
            "expected": [
                {
                    "layer_idx": int(layer_idx),
                    "tensor_name": tensor_name,
                    "shape": shape,
                }
                for layer_idx, tensor_name, shape in expected
            ],
            "tensors": [],
            "totals": {
                "num_tensors": 0,
                "tensor_bytes": 0,
            },
        }

    def _load_mamba2_projection_int4_manifest(self) -> Optional[dict]:
        path = self._mamba2_projection_int4_manifest_path()
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_mamba2_projection_int4_manifest(self) -> None:
        manifest = getattr(self, "_mamba2_projection_int4_manifest", None)
        if not isinstance(manifest, dict):
            return
        path = self._mamba2_projection_int4_manifest_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)

    def _mamba2_projection_int4_manifest_compatible(
        self,
        manifest: Optional[dict],
        expected: list[tuple[int, str, list[int]]],
    ) -> bool:
        if not isinstance(manifest, dict):
            return False
        group_size = self._mamba2_projection_int4_group_size()
        if (
            manifest.get("format") != MAMBA2_PROJECTION_INT4_CACHE_FORMAT
            or manifest.get("format_version") != MAMBA2_PROJECTION_INT4_CACHE_VERSION
            or manifest.get("backend") != "marlin_int4_single_slot"
            or int(manifest.get("nbits", -1)) != 4
            or int(manifest.get("group_size", -1)) != group_size
            or int(manifest.get("num_hidden_layers", -1)) != int(self.cfg.num_hidden_layers)
            or manifest.get("model_type") != str(self.cfg.model_type)
            or bool(manifest.get("rescale_prenorm_residual", False)) != bool(self.cfg.rescale_prenorm_residual)
            or not manifest.get("complete")
        ):
            return False
        source_signature = self._mamba2_projection_int4_source_signature(expected)
        if (
            manifest.get("source_signature") != source_signature["signature"]
            or os.path.abspath(os.path.expanduser(str(manifest.get("model_path", "")))) != source_signature["model_path"]
        ):
            return False
        expected_map = {
            (int(layer_idx), tensor_name): shape
            for layer_idx, tensor_name, shape in expected
        }
        entries = manifest.get("tensors", [])
        if len(entries) != len(expected_map):
            return False
        cache_dir = self._mamba2_projection_int4_cache_dir()
        seen = set()
        for entry in entries:
            try:
                key = (int(entry["layer_idx"]), str(entry["tensor_name"]))
                shape = [int(v) for v in entry["shape"]]
            except (KeyError, TypeError, ValueError):
                return False
            if key not in expected_map or expected_map[key] != shape or key in seen:
                return False
            if int(entry.get("group_size", -1)) != group_size or int(entry.get("nbits", -1)) != 4:
                return False
            file_name = entry.get("file")
            if not file_name or not os.path.isfile(os.path.join(cache_dir, file_name)):
                return False
            seen.add(key)
        return len(seen) == len(expected_map)

    def _prepare_mamba2_projection_int4_cache(self) -> None:
        self._mamba2_projection_int4_enabled = self._mamba2_projection_int4_requested()
        self._mamba2_projection_int4_rebuild = False
        self._mamba2_projection_int4_entries = {}
        self._mamba2_projection_bf16_released_bytes = 0
        self._mamba2_projection_bf16_released_tensors = 0
        if not self._mamba2_projection_int4_enabled:
            self._mamba2_projection_int4_manifest = None
            return

        expected = self._mamba2_projection_int4_expected()
        if not expected:
            self._mamba2_projection_int4_manifest = None
            self._mamba2_projection_int4_enabled = False
            return

        group_size = self._mamba2_projection_int4_group_size()
        for layer_idx, tensor_name, shape in expected:
            rows, cols = shape
            if rows % 64 != 0 or cols % 16 != 0 or cols % group_size != 0:
                raise RuntimeError(
                    "Mamba2 projection INT4 cache requested but tensor is not Marlin-compatible: "
                    f"layer={layer_idx} tensor={tensor_name} shape={rows}x{cols} group_size={group_size}."
                )

        manifest = self._load_mamba2_projection_int4_manifest()
        if self._mamba2_projection_int4_manifest_compatible(manifest, expected):
            self._mamba2_projection_int4_manifest = manifest
            self._mamba2_projection_int4_entries = {
                (int(entry["layer_idx"]), str(entry["tensor_name"])): entry
                for entry in manifest.get("tensors", [])
            }
            logger.info(
                "Using cached Mamba2 projection INT4 artifacts: tensors=%d cache=%s",
                len(self._mamba2_projection_int4_entries),
                self._mamba2_projection_int4_cache_dir(),
            )
            return

        cache_dir = self._mamba2_projection_int4_cache_dir()
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        self._mamba2_projection_int4_rebuild = True
        self._mamba2_projection_int4_manifest = self._mamba2_projection_int4_manifest_base(expected)
        self._save_mamba2_projection_int4_manifest()
        logger.info(
            "Building Mamba2 projection INT4 cache: tensors=%d group_size=%d cache=%s",
            len(expected),
            group_size,
            cache_dir,
        )

    def _write_mamba2_projection_int4_artifact(
        self,
        store,
        layer_idx: int,
        tensor_name: str,
        weight: torch.Tensor,
    ) -> dict:
        group_size = self._mamba2_projection_int4_group_size()
        rows, cols = int(weight.shape[0]), int(weight.shape[1])
        weight_cpu = weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        (
            marlin_packed_bytes,
            marlin_scales_bytes,
            simple_packed_bytes,
            simple_scales_bytes,
            packed_rows,
            packed_cols,
        ) = store.repack_marlin_int4_cpu_full(
            weight_cpu.data_ptr(),
            rows,
            cols,
            group_size,
        )
        del weight_cpu
        if int(packed_rows) != rows or int(packed_cols) != cols:
            raise RuntimeError(
                f"Mamba2 projection INT4 pack shape mismatch for layer {layer_idx} "
                f"{tensor_name}: packed {packed_rows}x{packed_cols}, expected {rows}x{cols}."
            )

        marlin_packed = torch.from_numpy(
            np.frombuffer(marlin_packed_bytes, dtype=np.uint32).copy()
        ).to(torch.int32).reshape(cols // 16, 2 * rows).contiguous()
        marlin_scales = torch.from_numpy(
            np.frombuffer(marlin_scales_bytes, dtype=np.uint16).copy()
        ).to(torch.int16).reshape(cols // group_size, rows).contiguous()
        simple_packed = torch.from_numpy(
            np.frombuffer(simple_packed_bytes, dtype=np.uint8).copy()
        ).reshape(rows, cols // 2).contiguous()
        simple_scales = torch.from_numpy(
            np.frombuffer(simple_scales_bytes, dtype=np.float32).copy()
        ).reshape(rows, cols // group_size).contiguous()

        cache_dir = self._mamba2_projection_int4_cache_dir()
        file_name = f"layer_{layer_idx:03d}_{tensor_name}_marlin_int4.safetensors"
        path = os.path.join(cache_dir, file_name)
        from safetensors.torch import save_file
        save_file(
            {
                "marlin_packed": marlin_packed,
                "marlin_scales": marlin_scales,
                "simple_packed": simple_packed,
                "simple_scales": simple_scales,
            },
            path,
            metadata={
                "format": MAMBA2_PROJECTION_INT4_CACHE_FORMAT,
                "format_version": str(MAMBA2_PROJECTION_INT4_CACHE_VERSION),
                "backend": "marlin_int4_single_slot",
                "layer_idx": str(layer_idx),
                "tensor_name": tensor_name,
                "nbits": "4",
                "group_size": str(group_size),
                "rows": str(rows),
                "cols": str(cols),
                "original_dtype": str(weight.dtype).replace("torch.", ""),
            },
        )
        tensor_bytes = int(os.path.getsize(path))
        entry = {
            "layer_idx": int(layer_idx),
            "tensor_name": tensor_name,
            "file": file_name,
            "shape": [rows, cols],
            "nbits": 4,
            "group_size": group_size,
            "marlin_packed_shape": [cols // 16, 2 * rows],
            "marlin_scales_shape": [cols // group_size, rows],
            "simple_packed_shape": [rows, cols // 2],
            "simple_scales_shape": [rows, cols // group_size],
            "tensor_bytes": tensor_bytes,
            "original_dtype": str(weight.dtype).replace("torch.", ""),
        }
        manifest = self._mamba2_projection_int4_manifest
        manifest["tensors"].append(entry)
        manifest["totals"]["num_tensors"] = len(manifest["tensors"])
        manifest["totals"]["tensor_bytes"] = int(manifest["totals"].get("tensor_bytes", 0)) + tensor_bytes
        self._mamba2_projection_int4_entries[(int(layer_idx), tensor_name)] = entry
        self._save_mamba2_projection_int4_manifest()
        return entry

    def _register_mamba2_projection_int4(
        self,
        store,
        layer_idx: int,
        tensor_name: str,
        weight: torch.Tensor,
        device: torch.device,
    ) -> tuple[int, "MarlinWeight"]:
        if getattr(self, "_mamba2_projection_int4_rebuild", False):
            entry = self._write_mamba2_projection_int4_artifact(store, layer_idx, tensor_name, weight)
        else:
            entry = self._mamba2_projection_int4_entries.get((int(layer_idx), tensor_name))
            if entry is None:
                raise RuntimeError(
                    f"Mamba2 projection INT4 cache missing manifest entry for layer {layer_idx} {tensor_name}."
                )

        rows, cols = [int(v) for v in entry["shape"]]
        group_size = int(entry["group_size"])
        path = os.path.join(self._mamba2_projection_int4_cache_dir(), entry["file"])
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != MAMBA2_PROJECTION_INT4_CACHE_FORMAT:
                raise RuntimeError(f"Mamba2 projection INT4 artifact format mismatch: {path}")
            if metadata.get("backend") != "marlin_int4_single_slot":
                raise RuntimeError(f"Mamba2 projection INT4 artifact backend mismatch: {path}")
            marlin_packed = handle.get_tensor("marlin_packed").contiguous()
            marlin_scales = handle.get_tensor("marlin_scales").contiguous()
            simple_packed = handle.get_tensor("simple_packed").contiguous()
            simple_scales = handle.get_tensor("simple_scales").contiguous()

        expected_shapes = {
            "marlin_packed": [cols // 16, 2 * rows],
            "marlin_scales": [cols // group_size, rows],
            "simple_packed": [rows, cols // 2],
            "simple_scales": [rows, cols // group_size],
        }
        actual_shapes = {
            "marlin_packed": list(marlin_packed.shape),
            "marlin_scales": list(marlin_scales.shape),
            "simple_packed": list(simple_packed.shape),
            "simple_scales": list(simple_scales.shape),
        }
        for name, expected_shape in expected_shapes.items():
            if actual_shapes[name] != expected_shape:
                raise RuntimeError(
                    f"Mamba2 projection INT4 artifact shape mismatch for {path} {name}: "
                    f"got {actual_shapes[name]}, expected {expected_shape}."
                )

        repacked = marlin_packed.to(dtype=torch.int32, device=device, non_blocking=True)
        scale_raw = marlin_scales.reshape(-1).to(dtype=torch.int16)
        n_scale_elements = int(scale_raw.numel())
        scales_slot = torch.empty(n_scale_elements * 2, dtype=torch.int16, device=device)
        scales_slot[:n_scale_elements].copy_(scale_raw.to(device, non_blocking=True))
        scale_perm = scales_slot[:n_scale_elements].view(torch.bfloat16).reshape(cols // group_size, rows)
        simple_packed = simple_packed.to(dtype=torch.uint8).contiguous()
        simple_scales = simple_scales.to(dtype=torch.float32).contiguous()
        store.stage_simple_int4_cpu(
            simple_packed.data_ptr(),
            simple_packed.numel(),
            simple_scales.data_ptr(),
            simple_scales.numel(),
            rows,
            cols,
            group_size,
        )
        wid = store.register_marlin_int4_weight(
            repacked.data_ptr(),
            scales_slot.data_ptr(),
            rows,
            cols,
            group_size,
        )
        from krasis.attention import MarlinWeight
        from krasis.marlin_utils import get_scalar_type, marlin_make_workspace
        if not hasattr(self, "_mamba2_projection_marlin_workspace"):
            self._mamba2_projection_marlin_workspace = marlin_make_workspace(device)
            self._mamba2_projection_marlin_scalar_type = get_scalar_type(4, False)
        marlin_weight = MarlinWeight(
            repacked,
            scale_perm,
            self._mamba2_projection_marlin_workspace,
            self._mamba2_projection_marlin_scalar_type,
            rows,
            cols,
        )
        self._keep_rust_decode_weight("mamba2_projection_int4", repacked, scales_slot)
        return wid, marlin_weight

    def _finalize_mamba2_projection_int4_cache(self) -> None:
        if not getattr(self, "_mamba2_projection_int4_enabled", False):
            return
        if getattr(self, "_mamba2_projection_int4_rebuild", False):
            expected_count = len(getattr(self, "_mamba2_projection_int4_manifest", {}).get("expected", []))
            actual_count = len(getattr(self, "_mamba2_projection_int4_manifest", {}).get("tensors", []))
            if expected_count != actual_count:
                raise RuntimeError(
                    f"Mamba2 projection INT4 cache build incomplete: {actual_count}/{expected_count} tensors."
                )
            self._mamba2_projection_int4_manifest["complete"] = True
            self._save_mamba2_projection_int4_manifest()
            logger.info(
                "Mamba2 projection INT4 cache build complete: tensors=%d bytes=%.1f MB cache=%s",
                actual_count,
                int(self._mamba2_projection_int4_manifest["totals"]["tensor_bytes"]) / (1024.0 * 1024.0),
                self._mamba2_projection_int4_cache_dir(),
            )

        released_tensors = int(getattr(self, "_mamba2_projection_bf16_released_tensors", 0))
        released_bytes = int(getattr(self, "_mamba2_projection_bf16_released_bytes", 0))
        if released_tensors:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "Mamba2 BF16 projection residency released after INT4 registration: tensors=%d cuda_mb=%.1f",
                released_tensors,
                released_bytes / (1024.0 * 1024.0),
            )

    def _kv_cache_slot_for_layer(self, layer_idx: int):
        layer_offset = self._kv_layer_offsets.get(layer_idx, -1)
        if layer_offset < 0:
            raise RuntimeError(
                f"Layer {layer_idx} does not own a full-attention KV cache slot."
            )
        for cache_idx, (start, end) in enumerate(self._layer_split):
            if start <= layer_idx < end:
                cache = self.kv_caches[cache_idx]
                if cache is None:
                    raise RuntimeError(
                        f"Layer {layer_idx} maps to empty KV cache group {cache_idx}."
                    )
                if layer_offset >= cache.num_layers:
                    raise RuntimeError(
                        f"Layer {layer_idx} KV offset {layer_offset} exceeds cache group "
                        f"{cache_idx} layer count {cache.num_layers}."
                    )
                return cache, layer_offset
        raise RuntimeError(
            f"Layer {layer_idx} is not covered by the configured layer split {self._layer_split}."
        )

    def _hqq_layer_meta(
        self,
        layer_idx: int,
        layer: TransformerLayer,
        target_device: torch.device,
        keepalive: list,
        sequence_state_registrations: list | None = None,
    ) -> dict:
        layer_kind = self._hqq_layer_kind(layer)
        if layer_kind == "deepseek_v4":
            return {}
        if layer_kind == "gqa":
            gqa_w = layer.gqa_weights if hasattr(layer, "gqa_weights") else None
            q_norm_src = gqa_w.get("q_norm") if gqa_w else None
            k_norm_src = gqa_w.get("k_norm") if gqa_w else None
            if q_norm_src is not None:
                q_norm = self._move_hqq_tensor_to_device(
                    q_norm_src.float().contiguous(), target_device, keepalive
                )
                q_norm_ptr = q_norm.data_ptr()
            else:
                q_norm_ptr = 0
            if k_norm_src is not None:
                k_norm = self._move_hqq_tensor_to_device(
                    k_norm_src.float().contiguous(), target_device, keepalive
                )
                k_norm_ptr = k_norm.data_ptr()
            else:
                k_norm_ptr = 0
            head_dim = self.cfg.gqa_head_dim_for_layer(layer_idx)
            sm_scale = 1.0 if self.cfg.gemma4_text else 1.0 / (head_dim ** 0.5)
            gated = hasattr(self.cfg, "gated_attention") and self.cfg.gated_attention
            max_seq = max(
                c.max_context_tokens for c in self.kv_caches if c is not None
            )
            layer_rope_params = self._gqa_layer_rope_params(layer_idx)
            head_dim_for_rope = self.cfg.gqa_head_dim_for_layer(layer_idx)
            if self.cfg.gemma4_text and layer_rope_params.get("rope_type") == "proportional":
                rope_half = head_dim_for_rope // 2
            else:
                rope_half = int(self.cfg.rotary_dim_for_layer(layer_idx) // 2)
            rope_cos_ptr = 0
            rope_sin_ptr = 0
            if rope_half > 0:
                theta = float(self.cfg.rope_theta_for_layer(layer_idx))
                cache = getattr(self, "_hqq_gqa_rope_table_cache", None)
                if cache is None:
                    cache = {}
                    self._hqq_gqa_rope_table_cache = cache
                rope_param_key = json.dumps(layer_rope_params, sort_keys=True, default=str)
                cache_key = (
                    target_device.type,
                    target_device.index,
                    max_seq,
                    rope_half,
                    theta,
                    head_dim_for_rope,
                    bool(self.cfg.gemma4_text),
                    rope_param_key,
                )
                cached = cache.get(cache_key)
                if cached is None:
                    inv_freq = self._gqa_inv_freq_for_layer(
                        layer_idx,
                        head_dim_for_rope,
                        rope_half,
                        theta,
                        layer_rope_params,
                    )
                    t = torch.arange(max_seq, dtype=torch.float32)
                    freqs = torch.outer(t, inv_freq)
                    rope_cos = self._move_hqq_tensor_to_device(
                        freqs.cos().contiguous(), target_device, keepalive, "hqq_rope_tables"
                    )
                    rope_sin = self._move_hqq_tensor_to_device(
                        freqs.sin().contiguous(), target_device, keepalive, "hqq_rope_tables"
                    )
                    cache[cache_key] = (rope_cos, rope_sin)
                else:
                    rope_cos, rope_sin = cached
                rope_cos_ptr = rope_cos.data_ptr()
                rope_sin_ptr = rope_sin.data_ptr()
            return {
                "num_heads": int(self.cfg.gqa_num_heads_for_layer(layer_idx)),
                "num_kv_heads": int(self.cfg.gqa_num_kv_heads_for_layer(layer_idx)),
                "head_dim": int(head_dim),
                "rope_half_dim": rope_half,
                "rope_cos_ptr": int(rope_cos_ptr),
                "rope_sin_ptr": int(rope_sin_ptr),
                "sm_scale": float(sm_scale),
                "q_norm_ptr": int(q_norm_ptr),
                "k_norm_ptr": int(k_norm_ptr),
                "gated": bool(gated),
                "sliding_window": int(self.cfg.sliding_window)
                if self.cfg.is_sliding_attention_layer(layer_idx) and self.cfg.sliding_window
                else 0,
                "v_norm_no_scale": bool(
                    gqa_w.get("v_norm_no_scale", False) if isinstance(gqa_w, dict) else False
                ),
                "rope_half_split": bool(self.cfg.gemma4_text or self.cfg.step3_text),
            }

        attn = layer.attention
        if attn is None:
            raise RuntimeError(
                f"HQQ layer {layer_idx} has no attention object for kind {layer_kind}"
            )

        if layer_kind == "mla":
            attn._hqq_kv_a_norm = self._move_hqq_tensor_to_device(
                attn.kv_a_norm_weight.float().contiguous(), target_device, keepalive
            )
            attn._hqq_w_kc = self._move_hqq_tensor_to_device(
                attn.w_kc.contiguous(), target_device, keepalive
            )
            attn._hqq_w_vc = self._move_hqq_tensor_to_device(
                attn.w_vc.contiguous(), target_device, keepalive
            )
            cache, mla_offset = self._kv_cache_slot_for_layer(layer_idx)
            ckv_layer = cache.ckv_cache[mla_offset]
            kpe_layer = cache.kpe_cache[mla_offset]
            ckv_cache = self._move_hqq_tensor_to_device(
                ckv_layer, target_device, keepalive
            )
            kpe_cache = self._move_hqq_tensor_to_device(
                kpe_layer, target_device, keepalive
            )
            if sequence_state_registrations is not None:
                sequence_state_registrations.append(
                    (
                        f"layer{layer_idx}.mla.compressed",
                        "mla_compressed_kv",
                        layer_idx,
                        ckv_cache,
                        1,
                    )
                )
                if kpe_cache.numel() > 0:
                    sequence_state_registrations.append(
                        (
                            f"layer{layer_idx}.mla.position",
                            "mla_positional_k",
                            layer_idx,
                            kpe_cache,
                            1,
                        )
                    )
            q_a_norm_ptr = 0
            if attn.has_q_lora:
                attn._hqq_q_a_norm = self._move_hqq_tensor_to_device(
                    attn.q_a_norm_weight.float().contiguous(), target_device, keepalive
                )
                q_a_norm_ptr = attn._hqq_q_a_norm.data_ptr()
            metadata = {
                "num_heads": int(attn.num_heads),
                "kv_lora_rank": int(attn.kv_lora_rank),
                "ckv_cache_dim": int(attn.ckv_dim),
                "qk_nope_dim": int(attn.qk_nope_dim),
                "qk_rope_dim": int(attn.qk_rope_dim),
                "v_head_dim": int(attn.v_head_dim),
                "q_lora_rank": int(attn.q_lora_rank if attn.has_q_lora else 0),
                "sm_scale": float(attn.sm_scale),
                "rope_interleave": bool(getattr(self.cfg, "rope_interleave", True)),
                "kv_a_norm_ptr": int(attn._hqq_kv_a_norm.data_ptr()),
                "w_kc_ptr": int(attn._hqq_w_kc.data_ptr()),
                "w_vc_ptr": int(attn._hqq_w_vc.data_ptr()),
                "ckv_cache_ptr": int(ckv_cache.data_ptr()),
                "kpe_cache_ptr": int(kpe_cache.data_ptr()),
                "q_a_norm_ptr": int(q_a_norm_ptr),
            }
            return metadata

        if layer_kind == "kimi_delta_attention":
            weights = attn.weights
            qkv_dim = int(
                self.cfg.linear_num_key_heads * self.cfg.linear_key_head_dim
            )
            conv_weight = torch.cat(
                (
                    weights["q_conv1d"],
                    weights["k_conv1d"],
                    weights["v_conv1d"],
                ),
                dim=0,
            ).squeeze(1)
            attn._hqq_conv_weight = self._move_hqq_tensor_to_device(
                conv_weight.float().contiguous(), target_device, keepalive
            )
            attn._hqq_a_log = self._move_hqq_tensor_to_device(
                weights["A_log"].float().contiguous(), target_device, keepalive
            )
            attn._hqq_dt_bias = self._move_hqq_tensor_to_device(
                weights["dt_bias"].float().contiguous(), target_device, keepalive
            )
            attn._hqq_norm_weight = self._move_hqq_tensor_to_device(
                weights["o_norm"].float().contiguous(), target_device, keepalive
            )
            attn._hqq_conv_state = torch.zeros(
                3,
                qkv_dim,
                self.cfg.linear_conv_kernel_dim - 1,
                device=target_device,
                dtype=torch.float32,
            )
            attn._hqq_recur_state = torch.zeros(
                self.cfg.linear_num_key_heads,
                self.cfg.linear_key_head_dim,
                self.cfg.linear_value_head_dim,
                device=target_device,
                dtype=torch.float32,
            )
            keepalive.extend(
                [attn._hqq_conv_state, attn._hqq_recur_state]
            )
            if sequence_state_registrations is not None:
                sequence_state_registrations.extend(
                    [
                        (
                            f"layer{layer_idx}.kda.conv",
                            "kimi_delta_conv_state",
                            layer_idx,
                            attn._hqq_conv_state,
                            0,
                        ),
                        (
                            f"layer{layer_idx}.kda.recurrent",
                            "kimi_delta_recurrent_state",
                            layer_idx,
                            attn._hqq_recur_state,
                            0,
                        ),
                    ]
                )
            return {
                "num_heads": int(self.cfg.linear_num_key_heads),
                "head_dim": int(self.cfg.linear_key_head_dim),
                "kernel_dim": int(self.cfg.linear_conv_kernel_dim),
                "gate_lower_bound": float(self.cfg.linear_gate_lower_bound),
                "conv_weight_ptr": int(attn._hqq_conv_weight.data_ptr()),
                "a_log_ptr": int(attn._hqq_a_log.data_ptr()),
                "dt_bias_ptr": int(attn._hqq_dt_bias.data_ptr()),
                "norm_weight_ptr": int(attn._hqq_norm_weight.data_ptr()),
                "conv_state_ptr": int(attn._hqq_conv_state.data_ptr()),
                "recur_state_ptr": int(attn._hqq_recur_state.data_ptr()),
            }

        if layer_kind == "linear_attention":
            conv_weight = attn.conv1d_weight
            if conv_weight.dim() == 3:
                conv_weight = conv_weight.squeeze(1)
            attn._init_state(batch_size=1)
            attn._hqq_conv_weight = self._move_hqq_tensor_to_device(
                conv_weight.float().contiguous(), target_device, keepalive
            )
            attn._hqq_a_log = self._move_hqq_tensor_to_device(
                attn.A_log.float().contiguous(), target_device, keepalive
            )
            attn._hqq_dt_bias = self._move_hqq_tensor_to_device(
                attn.dt_bias.float().contiguous(), target_device, keepalive
            )
            attn._hqq_norm_weight = self._move_hqq_tensor_to_device(
                attn.norm_weight.float().contiguous(), target_device, keepalive
            )
            attn._hqq_conv_state = self._move_hqq_tensor_to_device(
                attn._conv_state.squeeze(0).float().contiguous(), target_device, keepalive
            )
            attn._hqq_recur_state = self._move_hqq_tensor_to_device(
                attn._recurrent_state.squeeze(0).float().contiguous(), target_device, keepalive
            )
            if sequence_state_registrations is not None:
                sequence_state_registrations.extend(
                    [
                        (
                            f"layer{layer_idx}.linear.conv",
                            "linear_attention_conv_state",
                            layer_idx,
                            attn._hqq_conv_state,
                            0,
                        ),
                        (
                            f"layer{layer_idx}.linear.recurrent",
                            "linear_attention_recurrent_state",
                            layer_idx,
                            attn._hqq_recur_state,
                            0,
                        ),
                    ]
                )
            return {
                "num_k_heads": int(attn.num_k_heads),
                "num_v_heads": int(attn.num_v_heads),
                "k_head_dim": int(attn.k_head_dim),
                "v_head_dim": int(attn.v_head_dim),
                "head_ratio": int(attn.head_ratio),
                "kernel_dim": int(attn.kernel_dim),
                "conv_dim": int(attn.conv_dim),
                "scale": float(attn.scale),
                "conv_weight_ptr": int(attn._hqq_conv_weight.data_ptr()),
                "a_log_ptr": int(attn._hqq_a_log.data_ptr()),
                "dt_bias_ptr": int(attn._hqq_dt_bias.data_ptr()),
                "norm_weight_ptr": int(attn._hqq_norm_weight.data_ptr()),
                "conv_state_ptr": int(attn._hqq_conv_state.data_ptr()),
                "recur_state_ptr": int(attn._hqq_recur_state.data_ptr()),
            }

        raise RuntimeError(f"Unsupported HQQ layer kind {layer_kind} for layer {layer_idx}")

    def _dsa_indexer_layer_metadata(self, layer_idx: int, attn) -> dict | None:
        """Return the checkpoint-derived native DSA contract for one MLA layer."""
        if not self.cfg.is_dsa_layer(layer_idx):
            return None
        owner_layer_idx = attn.dsa_indexer_owner_layer
        if owner_layer_idx is None:
            raise RuntimeError(
                f"DSA layer {layer_idx} has no validated IndexShare owner"
            )
        owner_weights_present = attn.dsa_indexer is not None
        if owner_weights_present != (owner_layer_idx == layer_idx):
            raise RuntimeError(
                f"DSA layer {layer_idx} owner={owner_layer_idx} has "
                f"owner_weights_present={owner_weights_present}"
            )
        return {
            "layer_idx": int(layer_idx),
            "owner_layer_idx": int(owner_layer_idx),
            "owner_weights_present": bool(owner_weights_present),
            "index_topk": int(self.cfg.index_topk),
            "index_head_dim": int(self.cfg.index_head_dim),
            "index_n_heads": int(self.cfg.index_n_heads),
            "qk_rope_head_dim": int(self.cfg.qk_rope_head_dim),
            "index_topk_freq": int(self.cfg.index_topk_freq),
            "index_skip_topk_offset": int(self.cfg.index_skip_topk_offset),
            "q_lora_rank": int(self.cfg.q_lora_rank),
            "hidden_size": int(self.cfg.hidden_size),
            "rope_interleave": bool(self.cfg.indexer_rope_interleave),
            "index_kpool": int(self.cfg.index_kpool),
            "index_kpool_compress": bool(self.cfg.index_kpool_compress),
            "index_kpool_always_select_tail": bool(
                self.cfg.index_kpool_always_select_tail
            ),
        }

    def _register_dsa_indexer_layer_on_store(
        self, store, layer_idx: int, attn
    ) -> bool:
        """Attach one validated DSA contract after its native MLA registration."""
        metadata = self._dsa_indexer_layer_metadata(layer_idx, attn)
        if metadata is None:
            return False
        store.register_dsa_indexer_layer(**metadata)
        return True

    def _stage_dsa_indexer_resources_on_store(
        self,
        store,
        target_device: torch.device,
        keepalive: list,
        layer_start: int,
        layer_end: int,
    ) -> int:
        """Stage one fixed-address native DSA resource per owner used by a segment."""
        local_owner_layers, replica_owner_layers = _dsa_resource_layers_for_segment(
            self.cfg, layer_start, layer_end
        )
        owner_layers = local_owner_layers + replica_owner_layers
        if not owner_layers:
            return 0

        segment_contexts = set()
        for layer_idx in range(layer_start, layer_end):
            if not self.cfg.is_dsa_layer(layer_idx):
                continue
            cache, _ = self._kv_cache_slot_for_layer(layer_idx)
            segment_contexts.add(int(cache.max_pages * cache.page_size))
        if len(segment_contexts) != 1:
            raise RuntimeError(
                f"DSA decode segment [{layer_start}, {layer_end}) has "
                f"inconsistent KV capacities: {sorted(segment_contexts)}"
            )
        max_context_tokens = segment_contexts.pop()
        if max_context_tokens <= 0:
            raise RuntimeError(
                f"DSA decode segment [{layer_start}, {layer_end}) has no "
                "positive runtime context capacity"
            )

        tensor_shapes = {
            "wq_b": (
                self.cfg.index_n_heads * self.cfg.index_head_dim,
                self.cfg.q_lora_rank,
            ),
            "wk": (self.cfg.index_head_dim, self.cfg.hidden_size),
            "weights_proj": (
                self.cfg.index_n_heads,
                self.cfg.hidden_size,
            ),
            "k_norm_weight": (1, self.cfg.index_head_dim),
            "k_norm_bias": (1, self.cfg.index_head_dim),
        }
        if self.cfg.index_kpool_compress:
            tensor_shapes.update(
                {
                    "index_kpool_compress_ape": (
                        self.cfg.index_kpool,
                        self.cfg.index_head_dim,
                    ),
                    "index_kpool_compress_gate": (
                        self.cfg.index_head_dim,
                        self.cfg.hidden_size,
                    ),
                }
            )
        staged = 0
        for owner_layer_idx in local_owner_layers:
            owner_attn = self.layers[owner_layer_idx].attention
            owner = getattr(owner_attn, "dsa_indexer", None)
            if owner is None or owner.layer_idx != owner_layer_idx:
                raise RuntimeError(
                    f"DSA segment [{layer_start}, {layer_end}) requires owner "
                    f"layer {owner_layer_idx}, but its validated tensors are absent"
                )

            weight_ids = {}
            for tensor_name, (rows, cols) in tensor_shapes.items():
                source = getattr(owner, tensor_name, None)
                if not isinstance(source, torch.Tensor):
                    raise RuntimeError(
                        f"DSA owner layer {owner_layer_idx} has no tensor "
                        f"{tensor_name}"
                    )
                tensor = self._move_hqq_tensor_to_device(
                    source.contiguous(),
                    target_device,
                    keepalive,
                    "dsa_indexer_owner",
                )
                if tensor.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"DSA owner layer {owner_layer_idx} tensor {tensor_name} "
                        f"dtype {tensor.dtype} != torch.bfloat16"
                    )
                if tensor.numel() != rows * cols:
                    raise RuntimeError(
                        f"DSA owner layer {owner_layer_idx} tensor {tensor_name} "
                        f"has {tensor.numel()} elements, expected {rows * cols}"
                    )
                weight_ids[tensor_name] = int(
                    store.register_weight(
                        tensor.data_ptr(),
                        rows,
                        cols,
                        _weight_dtype_code(tensor),
                    )
                )

            store.register_dsa_indexer_owner_resource(
                owner_layer_idx=int(owner_layer_idx),
                wq_b_wid=weight_ids["wq_b"],
                wk_wid=weight_ids["wk"],
                weights_proj_wid=weight_ids["weights_proj"],
                k_norm_weight_wid=weight_ids["k_norm_weight"],
                k_norm_bias_wid=weight_ids["k_norm_bias"],
                kpool_ape_wid=weight_ids.get("index_kpool_compress_ape"),
                kpool_gate_wid=weight_ids.get("index_kpool_compress_gate"),
                max_context_tokens=max_context_tokens,
            )
            staged += 1

        for owner_layer_idx in replica_owner_layers:
            store.register_dsa_indexshare_replica(
                owner_layer_idx=int(owner_layer_idx),
                max_context_tokens=max_context_tokens,
            )
            staged += 1

        store.finalize_dsa_indexer_resources()
        logger.info(
            "DSA resources staged on %s: local_owners=%s replicas=%s "
            "layers=[%d,%d) max_context=%d",
            target_device,
            local_owner_layers,
            replica_owner_layers,
            layer_start,
            layer_end,
            max_context_tokens,
        )
        return staged

    def _dsa_indexer_resource_bytes_for_segment(
        self,
        layer_start: int,
        layer_end: int,
    ) -> int:
        """Return exact persistent owner-weight and decode-state bytes for a segment."""
        local_owner_layers, replica_owner_layers = _dsa_resource_layers_for_segment(
            self.cfg, layer_start, layer_end
        )
        owner_layers = local_owner_layers + replica_owner_layers
        if not owner_layers:
            return 0
        contexts = set()
        for layer_idx in range(layer_start, layer_end):
            if not self.cfg.is_dsa_layer(layer_idx):
                continue
            cache, _ = self._kv_cache_slot_for_layer(layer_idx)
            contexts.add(int(cache.max_pages * cache.page_size))
        if len(contexts) != 1:
            raise RuntimeError(
                f"DSA decode segment [{layer_start}, {layer_end}) has "
                f"inconsistent KV capacities: {sorted(contexts)}"
            )
        max_context_tokens = contexts.pop()
        total = 0
        for owner_layer_idx in local_owner_layers:
            owner = getattr(
                self.layers[owner_layer_idx].attention,
                "dsa_indexer",
                None,
            )
            if owner is None:
                raise RuntimeError(
                    f"DSA resource sizing requires owner layer {owner_layer_idx}"
                )
            for tensor_name in (
                "wq_b",
                "wk",
                "weights_proj",
                "k_norm_weight",
                "k_norm_bias",
            ):
                tensor = getattr(owner, tensor_name, None)
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(
                        f"DSA resource sizing requires owner {owner_layer_idx} "
                        f"tensor {tensor_name}"
                    )
                total += tensor.numel() * tensor.element_size()
            if self.cfg.index_kpool_compress:
                for tensor_name in (
                    "index_kpool_compress_ape",
                    "index_kpool_compress_gate",
                ):
                    tensor = getattr(owner, tensor_name, None)
                    if not isinstance(tensor, torch.Tensor):
                        raise RuntimeError(
                            f"DSA resource sizing requires owner {owner_layer_idx} "
                            f"tensor {tensor_name}"
                        )
                    total += tensor.numel() * tensor.element_size()
            total += max_context_tokens * self.cfg.index_head_dim * 2
            if self.cfg.index_kpool_compress:
                pool_capacity = (
                    max_context_tokens + self.cfg.index_kpool - 1
                ) // self.cfg.index_kpool
                total += max_context_tokens * self.cfg.index_head_dim * 2
                total += pool_capacity * self.cfg.index_head_dim * 2
                total += min(
                    self.cfg.index_topk // self.cfg.index_kpool,
                    pool_capacity,
                ) * 4
                selected_capacity = min(
                    self.cfg.index_topk, max_context_tokens
                ) + self.cfg.index_kpool - 1
            else:
                selected_capacity = min(
                    self.cfg.index_topk, max_context_tokens
                )
            total += selected_capacity * 4
        total += (
            len(replica_owner_layers)
            * (
                min(self.cfg.index_topk, max_context_tokens)
                + (self.cfg.index_kpool - 1 if self.cfg.index_kpool_compress else 0)
            )
            * 4
        )
        if not local_owner_layers:
            return int(total)
        query_elems = self.cfg.index_n_heads * self.cfg.index_head_dim
        # One graph-stable workspace is reused sequentially by every owner on
        # this store rather than multiplied by the IndexShare owner count.
        total += self.cfg.index_head_dim * 2
        total += self.cfg.index_head_dim * 2
        total += 4
        total += query_elems * 2
        total += query_elems * 2
        total += self.cfg.index_n_heads * 2
        total += max_context_tokens * self.cfg.index_n_heads * 4
        total += max_context_tokens * 4
        # Two FP32-score and two int32-index arrays are ping-ponged across
        # hierarchical merge passes. They are store-shared, not per owner.
        topk_candidates = _dsa_topk_candidate_capacity(
            max_context_tokens,
            self.cfg.index_topk,
        )
        total += topk_candidates * (4 + 4) * 2
        return int(total)

    def _register_hqq_attention_layers_on_store(
        self,
        store,
        target_device: torch.device,
        keepalive: list,
        sequence_state_registrations: list | None = None,
    ) -> int:
        manifest = getattr(self, "_hqq_manifest", None)
        if not manifest or not manifest.get("complete"):
            raise RuntimeError("HQQ registration requires a complete validated manifest.")
        nbits = self._hqq_attention_runtime_nbits
        if nbits is None:
            raise RuntimeError("HQQ registration requires loaded runtime state.")
        validate_hqq_cache_nbits(nbits)
        layer_register_nbits = 4 if is_hqq_mixed_attention(self.quant_cfg.attention) else hqq_cache_storage_nbits(nbits)

        registered_layers = 0
        staged_runtime_tensors = 0
        staged_sidecars = 0
        staged_sidecar_row_groups = 0
        sidecar_runtime = getattr(self, "_hqq_prefill_sidecar_runtime", {})
        expected_sidecar_layers = sorted(sidecar_runtime.keys())
        pending_layers = []
        for layer_idx, layer in enumerate(self.layers):
            runtime = self._hqq_attention_runtime.get(layer_idx)
            if not runtime:
                continue

            inp_norm = self._move_hqq_tensor_to_device(
                layer.input_norm_weight, target_device, keepalive
            )
            post_norm = layer.post_attn_norm_weight
            if post_norm is not None:
                post_norm = self._move_hqq_tensor_to_device(
                    post_norm, target_device, keepalive
                )
                post_norm_ptr = post_norm.data_ptr()
                post_norm_size = post_norm.numel()
            else:
                post_norm_ptr = 0
                post_norm_size = 0

            layer_kind = self._hqq_layer_kind(layer)
            layer_meta = self._hqq_layer_meta(
                layer_idx,
                layer,
                target_device,
                keepalive,
                sequence_state_registrations,
            )

            tensor_names = sorted(runtime.keys())
            if (self.cfg.gemma4_text or self.cfg.step3_text) and layer_kind == "gqa":
                # Mixed full/sliding GQA models can vary query/KV geometry by
                # layer. The fused-QKV HQQ decode branch was built for uniform
                # Qwen-style GQA, so use split Q/K/V descriptors for these
                # shape-varying models.
                tensor_names = [name for name in tensor_names if name != "fused_qkv"]
            if layer_kind == "gqa" and self.cfg.head_wise_attention_gate:
                gqa_w = getattr(layer, "gqa_weights", None)
                gate_src = gqa_w.get("g_proj") if isinstance(gqa_w, dict) else None
                if not isinstance(gate_src, torch.Tensor):
                    raise RuntimeError(
                        f"HQQ GQA layer {layer_idx} requires BF16 g_proj for "
                        "head-wise attention gate registration."
                    )
                gate = self._move_hqq_tensor_to_device(
                    gate_src.contiguous(), target_device, keepalive, "hqq_gqa_head_gate"
                )
                layer_meta["head_gate_proj_wid"] = int(
                    store.register_weight(
                        gate.data_ptr(),
                        gate.shape[0],
                        gate.shape[1],
                        _weight_dtype_code(gate),
                    )
                )
            for tensor_name in tensor_names:
                desc = runtime[tensor_name]
                store.stage_hqq_runtime_tensor_formats(
                    layer_idx=layer_idx,
                    tensor_name=tensor_name,
                    backend=manifest["backend"],
                    deepseek_v4=bool(self.cfg.is_deepseek_v4),
                    prefill_materialize_default=bool(
                        self.cfg.is_deepseek_v4 or self.cfg.is_glm5_next
                    ),
                    nbits=int(desc["nbits"]),
                    format_version=int(manifest["format_version"]),
                    packed_ptr=int(desc["packed"].data_ptr()),
                    packed_bytes=int(desc["packed"].numel() * desc["packed"].element_size()),
                    scales_ptr=int(desc["scales"].data_ptr()),
                    scales_bytes=int(desc["scales"].numel() * desc["scales"].element_size()),
                    zeros_ptr=int(desc["zeros"].data_ptr()),
                    zeros_bytes=int(desc["zeros"].numel() * desc["zeros"].element_size()),
                    rows=int(desc["orig_shape"][0]),
                    cols=int(desc["orig_shape"][1]),
                    group_size=int(desc["group_size"]),
                    axis=int(desc["axis"]),
                    layout=desc["layout"],
                    packed_dtype=desc["packed_dtype"],
                    scales_dtype=desc["scales_dtype"],
                    zeros_dtype=desc["zeros_dtype"],
                )
                staged_runtime_tensors += 1
            for sidecar in sidecar_runtime.get(layer_idx, []):
                correction = self._move_hqq_tensor_to_device(
                    sidecar["correction"], target_device, keepalive, "hqq_prefill_sidecar"
                )
                scales = self._move_hqq_tensor_to_device(
                    sidecar["scales"], target_device, keepalive, "hqq_prefill_sidecar"
                )
                output_rows = self._move_hqq_tensor_to_device(
                    sidecar["output_rows"], target_device, keepalive, "hqq_prefill_sidecar"
                )
                groups = self._move_hqq_tensor_to_device(
                    sidecar["groups"], target_device, keepalive, "hqq_prefill_sidecar"
                )
                start_cols = self._move_hqq_tensor_to_device(
                    sidecar["start_cols"], target_device, keepalive, "hqq_prefill_sidecar"
                )
                widths = self._move_hqq_tensor_to_device(
                    sidecar["widths"], target_device, keepalive, "hqq_prefill_sidecar"
                )
                base_f32 = self._move_hqq_tensor_to_device(
                    sidecar["base_f32"], target_device, keepalive, "hqq_prefill_sidecar"
                )
                store.stage_hqq_prefill_sidecar_tensor(
                    layer_idx=layer_idx,
                    tensor_name=str(sidecar["tensor_name"]),
                    mode=str(sidecar["mode"]),
                    variant_name=str(sidecar["variant_name"]),
                    correction_ptr=int(correction.data_ptr()),
                    correction_bytes=int(correction.numel() * correction.element_size()),
                    scales_ptr=int(scales.data_ptr()),
                    scales_bytes=int(scales.numel() * scales.element_size()),
                    output_rows_ptr=int(output_rows.data_ptr()),
                    output_rows_bytes=int(output_rows.numel() * output_rows.element_size()),
                    groups_ptr=int(groups.data_ptr()),
                    groups_bytes=int(groups.numel() * groups.element_size()),
                    start_cols_ptr=int(start_cols.data_ptr()),
                    start_cols_bytes=int(start_cols.numel() * start_cols.element_size()),
                    widths_ptr=int(widths.data_ptr()),
                    widths_bytes=int(widths.numel() * widths.element_size()),
                    base_f32_ptr=int(base_f32.data_ptr()) if base_f32.numel() else 0,
                    base_f32_bytes=int(base_f32.numel() * base_f32.element_size()),
                    row_group_count=int(sidecar["row_group_count"]),
                    max_width=int(sidecar["max_width"]),
                )
                staged_sidecars += 1
                staged_sidecar_row_groups += int(sidecar["row_group_count"])
            pending_layers.append(
                dict(
                    layer_idx=layer_idx,
                    layer_kind=layer_kind,
                    layer_meta=layer_meta,
                    tensor_names=tensor_names,
                    common_args=dict(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(),
                        input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm_ptr,
                        post_attn_norm_size=post_norm_size,
                        backend=manifest["backend"],
                        nbits=layer_register_nbits,
                        format_version=int(manifest["format_version"]),
                        tensor_names=tensor_names,
                    ),
                )
            )

        if expected_sidecar_layers:
            logger.info(
                "HQQ sidecar staging requested: layers=%d sidecars=%d row_groups=%d",
                len(expected_sidecar_layers),
                staged_sidecars,
                staged_sidecar_row_groups,
            )
            if staged_sidecars <= 0:
                raise RuntimeError(
                    "HQQ sidecar manifest loaded but no sidecar tensors were staged on the decode store."
                )

        if staged_runtime_tensors > 0:
            removed = store.restrict_hqq_runtime_slots_to_decode_segment()
            if removed > 0:
                logger.info(
                    "HQQ attention runtime scoping removed %d staged tensors outside the active decode segment on this store.",
                    removed,
                )
            self._release_hqq_bf16_attention_residency()
            store.register_hqq_runtime_slots()
            store.swap_hqq_runtime_to_prefill()
            store.swap_hqq_runtime_to_decode()
        for pending in pending_layers:
            layer_kind = pending["layer_kind"]
            layer_meta = pending["layer_meta"]
            common_args = pending["common_args"]
            if layer_kind == "gqa":
                store.register_hqq_runtime_gqa_layer(
                    **common_args,
                    num_heads=int(layer_meta["num_heads"]),
                    num_kv_heads=int(layer_meta["num_kv_heads"]),
                    head_dim=int(layer_meta["head_dim"]),
                    rope_half_dim=int(layer_meta["rope_half_dim"]),
                    rope_cos_ptr=int(layer_meta["rope_cos_ptr"]),
                    rope_sin_ptr=int(layer_meta["rope_sin_ptr"]),
                    sm_scale=float(layer_meta["sm_scale"]),
                    q_norm_ptr=int(layer_meta["q_norm_ptr"]),
                    k_norm_ptr=int(layer_meta["k_norm_ptr"]),
                    gated=bool(layer_meta["gated"]),
                    head_gate_proj_wid=layer_meta.get("head_gate_proj_wid"),
                )
                if int(layer_meta.get("sliding_window", 0)) > 0:
                    store.set_gqa_sliding_window(
                        int(common_args["layer_idx"]),
                        int(layer_meta["sliding_window"]),
                    )
                if bool(layer_meta.get("v_norm_no_scale", False)):
                    store.set_gqa_v_norm_no_scale(int(common_args["layer_idx"]), True)
                if bool(layer_meta.get("rope_half_split", False)):
                    store.set_gqa_rope_half_split(int(common_args["layer_idx"]), True)
            elif layer_kind == "mla":
                store.register_hqq_runtime_mla_layer(
                    **common_args,
                    num_heads=int(layer_meta["num_heads"]),
                    kv_lora_rank=int(layer_meta["kv_lora_rank"]),
                    ckv_cache_dim=int(layer_meta["ckv_cache_dim"]),
                    qk_nope_dim=int(layer_meta["qk_nope_dim"]),
                    qk_rope_dim=int(layer_meta["qk_rope_dim"]),
                    v_head_dim=int(layer_meta["v_head_dim"]),
                    q_lora_rank=int(layer_meta["q_lora_rank"]),
                    sm_scale=float(layer_meta["sm_scale"]),
                    rope_interleave=bool(layer_meta["rope_interleave"]),
                    kv_a_norm_ptr=int(layer_meta["kv_a_norm_ptr"]),
                    w_kc_ptr=int(layer_meta["w_kc_ptr"]),
                    w_vc_ptr=int(layer_meta["w_vc_ptr"]),
                    ckv_cache_ptr=int(layer_meta["ckv_cache_ptr"]),
                    kpe_cache_ptr=int(layer_meta["kpe_cache_ptr"]),
                    q_a_norm_ptr=int(layer_meta["q_a_norm_ptr"]),
                )
                self._register_dsa_indexer_layer_on_store(
                    store,
                    int(common_args["layer_idx"]),
                    self.layers[int(common_args["layer_idx"])].attention,
                )
            elif layer_kind == "linear_attention":
                store.register_hqq_runtime_linear_attention_layer(
                    **common_args,
                    num_k_heads=int(layer_meta["num_k_heads"]),
                    num_v_heads=int(layer_meta["num_v_heads"]),
                    k_head_dim=int(layer_meta["k_head_dim"]),
                    v_head_dim=int(layer_meta["v_head_dim"]),
                    head_ratio=int(layer_meta["head_ratio"]),
                    kernel_dim=int(layer_meta["kernel_dim"]),
                    conv_dim=int(layer_meta["conv_dim"]),
                    scale=float(layer_meta["scale"]),
                    conv_weight_ptr=int(layer_meta["conv_weight_ptr"]),
                    a_log_ptr=int(layer_meta["a_log_ptr"]),
                    dt_bias_ptr=int(layer_meta["dt_bias_ptr"]),
                    norm_weight_ptr=int(layer_meta["norm_weight_ptr"]),
                    conv_state_ptr=int(layer_meta["conv_state_ptr"]),
                    recur_state_ptr=int(layer_meta["recur_state_ptr"]),
                )
            elif layer_kind == "kimi_delta_attention":
                store.register_hqq_runtime_kimi_delta_attention_layer(
                    **common_args,
                    num_heads=int(layer_meta["num_heads"]),
                    head_dim=int(layer_meta["head_dim"]),
                    kernel_dim=int(layer_meta["kernel_dim"]),
                    gate_lower_bound=float(layer_meta["gate_lower_bound"]),
                    conv_weight_ptr=int(layer_meta["conv_weight_ptr"]),
                    a_log_ptr=int(layer_meta["a_log_ptr"]),
                    dt_bias_ptr=int(layer_meta["dt_bias_ptr"]),
                    norm_weight_ptr=int(layer_meta["norm_weight_ptr"]),
                    conv_state_ptr=int(layer_meta["conv_state_ptr"]),
                    recur_state_ptr=int(layer_meta["recur_state_ptr"]),
                )
            elif layer_kind == "deepseek_v4":
                # DeepSeek owns a distinct attention graph. Its base,
                # compressor and indexer registrations below consume stable
                # HQQ runtime weight views, then attach these exact descriptors
                # after the complete architecture contract is registered.
                registered_layers += 1
                continue
            else:
                raise RuntimeError(
                    f"Unsupported HQQ layer kind {layer_kind} for registration"
                )
            registered_layers += 1

        if registered_layers == 0:
            raise RuntimeError("No HQQ attention layers were registered on the decode store.")
        if expected_sidecar_layers:
            decode_sidecar_count = 0
            decode_sidecar_row_groups = 0
            for layer_idx in expected_sidecar_layers:
                try:
                    execution = json.loads(store.hqq_execution_json(int(layer_idx)))
                except Exception as exc:
                    raise RuntimeError(
                        f"HQQ sidecar layer {layer_idx} did not produce an execution descriptor"
                    ) from exc
                for tensor in execution.get("tensors", []):
                    decode_sidecar_count += int(tensor.get("decode_sidecar_count", 0))
                    decode_sidecar_row_groups += int(tensor.get("decode_sidecar_row_groups", 0))
            logger.info(
                "HQQ decode sidecar attachment validated: tensors=%d row_groups=%d",
                decode_sidecar_count,
                decode_sidecar_row_groups,
            )
            if decode_sidecar_count <= 0 or decode_sidecar_row_groups <= 0:
                raise RuntimeError(
                    "HQQ sidecar manifest loaded but no decode execution descriptors received sidecars."
                )
        if staged_runtime_tensors > 0:
            staging = json.loads(store.hqq_runtime_staging_json())
            summary = staging["summary"]
            registration = staging["registration"]
            last_swap = staging["last_swap"]
            logger.info(
                "HQQ runtime staging prepared: tensors=%d host_mb=%.2f prefill_mb=%.2f decode_mb=%.2f slot_mb=(packed %.2f, scales %.2f, zeros %.2f) build_ms=%.3f reg_ms=%.3f device_mb=%.2f decode_swap_ms=%.3f decode_swap_mb=%.2f",
                summary["tensor_count"],
                summary["total_host_bytes"] / (1024.0 * 1024.0),
                summary["prefill_host_bytes"] / (1024.0 * 1024.0),
                summary["decode_host_bytes"] / (1024.0 * 1024.0),
                summary["packed_slot_bytes"] / (1024.0 * 1024.0),
                summary["scales_slot_bytes"] / (1024.0 * 1024.0),
                summary["zeros_slot_bytes"] / (1024.0 * 1024.0),
                summary["total_build_ms"],
                registration["total_registration_ms"],
                registration["total_device_slot_bytes"] / (1024.0 * 1024.0),
                last_swap["total_swap_ms"],
                last_swap["total_host_bytes"] / (1024.0 * 1024.0),
            )
            _python_trace(
                "weights",
                "phase=hqq_runtime_staging "
                f"tensors={summary['tensor_count']} "
                f"host_mb={summary['total_host_bytes'] / (1024.0 * 1024.0):.3f} "
                f"build_ms={summary['total_build_ms']:.3f} "
                f"reg_ms={registration['total_registration_ms']:.3f} "
                f"decode_swap_ms={last_swap['total_swap_ms']:.3f}",
            )
        return registered_layers

    def _load_gpu_weights(self, loader: WeightLoader):
        """Stream-load GPU weights: all attention on GPU0.

        Streaming attention architecture: ALL attention weights, norms, gate
        weights, shared expert weights, embedding, final_norm, and lm_head
        live on GPU0. GPU1+ are reserved entirely for EP expert parallelism
        during prefill and HCS expert cache during decode.

        When stream_attention is enabled, attention weights are loaded to GPU
        one layer at a time, immediately offloaded to CPU, then freed from GPU
        before loading the next layer. This bounds peak VRAM to ~1 layer of
        attention + permanent weights (norms, gates, shared experts, embedding,
        lm_head) instead of the full model.
        """
        self.layers = []
        primary_dev = self.all_devices[0]
        torch.cuda.set_device(primary_dev)
        L = self.cfg.num_hidden_layers

        # All attention on GPU0 — single split covering all layers
        self._layer_split = [(0, L)]
        if self.stream_attention:
            logger.info("Streaming attention: all %d layers on GPU0, %d GPUs for EP",
                         L, len(self.all_devices))
        else:
            logger.info("Resident attention: all %d layers permanently on GPU0, %d GPUs for EP",
                         L, len(self.all_devices))

        # ── Load layers ──
        logger.info("Loading full base model to %s...", primary_dev)
        self.embedding = loader.load_embedding(primary_dev)
        hqq_active = is_hqq_attention(self.quant_cfg.attention)
        if hqq_active:
            self._prepare_hqq_attention_cache()
            self._build_hqq_attention_cache_from_safetensors(loader, primary_dev)

        if self.stream_attention:
            # Incremental load: attention goes directly to CPU (never touches GPU),
            # while norms, gates, and shared experts go to GPU (small, permanent).
            # This avoids the GPU roundtrip and ensures no OOM from accumulating
            # attention weights on GPU during loading.
            self._attn_cpu_weights = {}
            import torch as _torch
            _cpu = _torch.device('cpu')
            for layer_idx in range(L):
                weights = loader.load_layer(layer_idx, primary_dev, attn_device=_cpu)
                if hqq_active:
                    layer_type = weights.get("layer_type", "full_attention")
                    attn_key = self._attention_weight_key(layer_type)
                    self._maybe_write_hqq_attention_artifacts(
                        layer_idx,
                        layer_type,
                        weights.get(attn_key, {}),
                    )
                layer = TransformerLayer(
                    self.cfg, layer_idx, weights, primary_dev,
                    krasis_engine=None,
                    gpu_prefill_manager=None,
                    gpu_prefill_threshold=self.gpu_prefill_threshold,
                )
                # Extract attention weights to CPU, null GPU refs
                w = self._extract_layer_weights(layer, layer.device)
                attn_key = self._attention_weight_key(layer.layer_type)
                attn_dict = w.get(attn_key, {})
                cpu_attn = self._copy_weights_dict(attn_dict, 'cpu') if attn_dict else {}
                # Store for _init_stream_attention to pin later
                self._attn_cpu_weights[layer_idx] = {
                    "norms": w["norms"],
                    attn_key: cpu_attn,
                    "is_moe": w["is_moe"],
                    "layer_type": w.get("layer_type", "full_attention"),
                }
                if "gate" in w:
                    self._attn_cpu_weights[layer_idx]["gate"] = w["gate"]
                if "shared_expert" in w:
                    self._attn_cpu_weights[layer_idx]["shared_expert"] = w["shared_expert"]
                if "dense_mlp" in w:
                    self._attn_cpu_weights[layer_idx]["dense_mlp"] = w["dense_mlp"]
                if "latent_proj" in w:
                    self._attn_cpu_weights[layer_idx]["latent_proj"] = w["latent_proj"]
                # Null attention on GPU (keep norms/gate/shared — already on GPU)
                attn = layer.attention
                if attn is not None:
                    for attr_name in list(vars(attn).keys()):
                        val = getattr(attn, attr_name, None)
                        if isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                            setattr(attn, attr_name, None)
                        elif isinstance(val, tuple) and len(val) == 2:
                            if all(isinstance(t, torch.Tensor) for t in val):
                                setattr(attn, attr_name, None)
                self.layers.append(layer)
                if (layer_idx + 1) % 10 == 0 or layer_idx == L - 1:
                    torch.cuda.empty_cache()
                    alloc_mb = torch.cuda.memory_allocated(primary_dev) / (1024**2)
                    logger.info("Layer %d/%d loaded+offloaded (GPU alloc: %.0f MB)",
                                layer_idx + 1, L, alloc_mb)
            self._attn_offloaded = True
            if hqq_active:
                self._validate_hqq_attention_cache()
                self._load_hqq_attention_runtime_state()
            logger.info("Incremental attention offload: %d layers, peak VRAM bounded to ~1 layer", L)
        else:
            # When AWQ is configured, load attention weights to CPU (not GPU).
            # AWQ quantizes BF16→INT4 on CPU then uploads only the packed INT4 to GPU
            # in setup_gpu_decode_store(). Loading BF16 to GPU first wastes VRAM and
            # risks OOM on large models (e.g. Q235B: ~12.8 GB BF16 attention).
            awq_active = self.quant_cfg.attention == "awq"
            _attn_dev = torch.device('cpu') if (awq_active or hqq_active) else None
            for layer_idx in range(L):
                weights = loader.load_layer(layer_idx, primary_dev, attn_device=_attn_dev)
                if hqq_active:
                    layer_type = weights.get("layer_type", "full_attention")
                    attn_key = self._attention_weight_key(layer_type)
                    self._maybe_write_hqq_attention_artifacts(
                        layer_idx,
                        layer_type,
                        weights.get(attn_key, {}),
                    )
                layer = TransformerLayer(
                    self.cfg, layer_idx, weights, primary_dev,
                    krasis_engine=None,
                    gpu_prefill_manager=None,
                    gpu_prefill_threshold=self.gpu_prefill_threshold,
                )
                self.layers.append(layer)
            if hqq_active:
                self._validate_hqq_attention_cache()
                self._load_hqq_attention_runtime_state()

        self.final_norm = loader.load_final_norm(primary_dev)
        self.lm_head_data = loader.load_lm_head(primary_dev)
        self._deepseek_v4_hc_head = (
            loader.load_deepseek_v4_hc_head(primary_dev)
            if self.cfg.is_deepseek_v4
            else None
        )

        # ── VRAM checkpoint ──
        def _vram_snap(label):
            for di, d in enumerate(self.all_devices):
                free, total = torch.cuda.mem_get_info(d)
                alloc = torch.cuda.memory_allocated(d)
                resv = torch.cuda.memory_reserved(d)
                logger.info(
                    "VRAM[%s] GPU%d: free=%d MB, alloc=%d MB, reserved=%d MB, "
                    "non-pytorch=%d MB",
                    label, di, free >> 20, alloc >> 20, resv >> 20,
                    (total - free - resv) >> 20,
                )
        _vram_snap("after-layers-loaded")

        # ── Build per-device state (GPU0 only) ──
        self._active_device = str(primary_dev)
        dev_str = str(primary_dev)
        self._device_state[dev_str] = {
            "layers": self.layers,
            "embedding": self.embedding,
            "final_norm": self.final_norm,
            "lm_head_data": self.lm_head_data,
        }

    # ── Attention streaming: offload/reload for large models ──

    def _estimate_attention_vram(self) -> int:
        """Estimate total GPU VRAM used by attention weights across all layers.

        Returns bytes consumed by all attention-related parameters:
        norms, attention projections, gate weights, shared experts.
        Does NOT include KV cache, embedding, or lm_head.
        """
        total = 0
        seen_storages: set[tuple[str, int, int]] = set()

        def tensor_bytes(value) -> int:
            if isinstance(value, torch.Tensor):
                if value.device.type != 'cuda':
                    return 0
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr(), storage.nbytes())
                if key in seen_storages:
                    return 0
                seen_storages.add(key)
                return storage.nbytes()
            if isinstance(value, dict):
                return sum(tensor_bytes(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return sum(tensor_bytes(item) for item in value)
            return 0
        for layer_idx, layer in enumerate(self.layers):
            # Norms
            if layer.input_norm_weight is not None:
                total += layer.input_norm_weight.nelement() * layer.input_norm_weight.element_size()
            if layer.post_attn_norm_weight is not None:
                total += layer.post_attn_norm_weight.nelement() * layer.post_attn_norm_weight.element_size()

            # Attention weights (skip Mamba2/MoE-only layers which have no attention)
            attn = layer.attention
            if attn is not None:
                total += tensor_bytes(vars(attn))

            # Gate weights + shared expert (MoE layers)
            if layer.is_moe:
                if layer.gate_weight is not None:
                    total += tensor_bytes(layer.gate_weight)
                if layer.gate_bias is not None:
                    total += tensor_bytes(layer.gate_bias)
                if layer.e_score_correction_bias is not None:
                    total += tensor_bytes(layer.e_score_correction_bias)
                if getattr(layer, "router_vl_bias", None) is not None:
                    total += tensor_bytes(layer.router_vl_bias)
                if getattr(layer, "router_tid2eid", None) is not None:
                    total += tensor_bytes(layer.router_tid2eid)
                if layer.shared_expert is not None:
                    total += tensor_bytes(layer.shared_expert)
                if layer.shared_expert_gate is not None:
                    total += tensor_bytes(layer.shared_expert_gate)

            # Dense MLP (non-MoE layers)
            if not layer.is_moe and hasattr(layer, 'mlp') and layer.mlp is not None:
                total += tensor_bytes(layer.mlp)

        return total

    def log_vram_ledger_residency(self, label: str) -> None:
        """Log opt-in categorized CUDA tensor residency for startup VRAM diagnosis."""
        if not _vram_ledger_enabled() or not torch.cuda.is_available():
            return

        categories_by_device: dict[int, dict[str, int]] = {}
        seen: set[tuple[int, int]] = set()
        duplicate_refs = 0
        duplicate_bytes = 0

        def _category(device_idx: int, category: str) -> None:
            categories_by_device.setdefault(device_idx, {}).setdefault(category, 0)

        def _storage_key_and_bytes(tensor: torch.Tensor) -> tuple[tuple[int, int], int] | None:
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
                return None
            device_idx = tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()
            try:
                storage = tensor.untyped_storage()
                ptr = int(storage.data_ptr())
                nbytes = int(storage.nbytes())
            except Exception:
                ptr = int(tensor.data_ptr())
                nbytes = int(tensor.nelement() * tensor.element_size())
            if ptr == 0 or nbytes <= 0:
                return None
            return (int(device_idx), ptr), nbytes

        def _add_tensor(category: str, tensor: torch.Tensor) -> None:
            nonlocal duplicate_refs, duplicate_bytes
            info = _storage_key_and_bytes(tensor)
            if info is None:
                return
            key, nbytes = info
            device_idx, _ = key
            _category(device_idx, category)
            if key in seen:
                duplicate_refs += 1
                duplicate_bytes += nbytes
                return
            seen.add(key)
            categories_by_device[device_idx][category] += nbytes

        def _add_any(category: str, value) -> None:
            if value is None:
                return
            if isinstance(value, torch.Tensor):
                _add_tensor(category, value)
                return
            if hasattr(value, "packed") and hasattr(value, "scales"):
                _add_any(category, getattr(value, "packed", None))
                _add_any(category, getattr(value, "scales", None))
                _add_any("marlin_workspace", getattr(value, "workspace", None))
                return
            if isinstance(value, dict):
                for item in value.values():
                    _add_any(category, item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    _add_any(category, item)

        def _safe_category_label(label: object) -> str:
            text = str(label or "unclassified")
            text = "".join(ch if ch.isalnum() else "_" for ch in text)
            text = "_".join(part for part in text.split("_") if part)
            return text or "unclassified"

        def _add_rust_decode_keepalive(value) -> None:
            if value is None:
                return
            labels = getattr(value, "labels", None)
            if labels is None or len(labels) != len(value):
                _add_any("rust_decode_keepalive", value)
                return
            for label, item in zip(labels, value):
                _add_any(f"rust_keepalive_{_safe_category_label(label)}", item)

        _add_any("top_embedding", self.embedding)
        _add_any("top_final_norm", self.final_norm)
        _add_any("top_lm_head", self.lm_head_data)
        _add_any("rust_lm_head_bf16", getattr(self, "_rust_lm_head", None))

        for cache in getattr(self, "kv_caches", []) or []:
            if cache is None:
                continue
            for name in (
                "k_cache", "v_cache", "k_radius_cache", "v_radius_cache",
                "k_angles_cache", "v_angles_cache", "ckv_cache", "kpe_cache", "kv_cache",
            ):
                _add_any("kv_cache", getattr(cache, name, None))

        _add_any("marlin_attention_weights", getattr(self, "_marlin_attn_weights", None))
        _add_rust_decode_keepalive(getattr(self, "_rust_decode_weights", None))
        _add_any("rust_rope_tables", getattr(self, "_rust_rope_cos", None))
        _add_any("rust_rope_tables", getattr(self, "_rust_rope_sin", None))
        _add_any("rust_tq4_signs", getattr(self, "_tq4_sign_refs", None))
        _add_any("rust_shared_gate_refs", getattr(self, "_rust_shared_gate_refs", None))
        _add_any("aux_decode_keepalive", getattr(self, "_aux_decode_weights", None))

        for layer in getattr(self, "layers", []) or []:
            _add_any("layer_norms", getattr(layer, "input_norm_weight", None))
            _add_any("layer_norms", getattr(layer, "post_attn_norm_weight", None))
            _add_any("router_weights", getattr(layer, "gate_weight", None))
            _add_any("router_weights", getattr(layer, "gate_bias", None))
            _add_any("router_weights", getattr(layer, "e_score_correction_bias", None))
            _add_any("router_weights", getattr(layer, "router_vl_bias", None))
            _add_any("router_fp32_mirrors", getattr(layer, "_gate_weight_f32", None))
            _add_any("router_fp32_mirrors", getattr(layer, "_gate_bias_f32", None))
            _add_any("router_fp32_mirrors", getattr(layer, "_e_score_correction_bias_f32", None))
            _add_any("router_fp32_mirrors", getattr(layer, "_router_vl_bias_f32", None))
            _add_any("shared_expert", getattr(layer, "shared_expert", None))
            _add_any("shared_expert_gate", getattr(layer, "shared_expert_gate", None))
            _add_any("dense_mlp", getattr(layer, "dense_mlp", None))
            _add_any("latent_proj", getattr(layer, "latent_proj", None))
            _add_any("gqa_weight_refs", getattr(layer, "gqa_weights", None))
            _add_any("mamba2_weight_refs", getattr(layer, "mamba2_weights", None))
            attn = getattr(layer, "attention", None)
            if attn is not None:
                try:
                    attn_items = vars(attn).items()
                except TypeError:
                    attn_items = (
                        (name, getattr(attn, name, None))
                        for name in dir(attn)
                        if not name.startswith("__")
                    )
                for attr_name, attr_value in attn_items:
                    if attr_name.startswith("_hqq_"):
                        _add_any("hqq_attention_meta", attr_value)
                    elif attr_name.startswith("_rust_"):
                        _add_any("rust_attention_meta", attr_value)
                    elif isinstance(attr_value, (torch.Tensor, tuple, list, dict)) or (
                        hasattr(attr_value, "packed") and hasattr(attr_value, "scales")
                    ):
                        _add_any("attention_weight_refs", attr_value)

        for device_idx in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx)
            reserved = torch.cuda.memory_reserved(device_idx)
            driver_used = total - free
            categories = categories_by_device.get(device_idx, {})
            inventoried = sum(categories.values())
            category_parts = " ".join(
                f"{name}_mb={bytes_ / (1024 * 1024):.1f}"
                for name, bytes_ in sorted(categories.items())
                if bytes_ > 0
            )
            message = (
                f"VRAM LEDGER residency label={label} device={device_idx} "
                f"free_mb={free >> 20} total_mb={total >> 20} "
                f"driver_used_mb={driver_used >> 20} "
                f"torch_allocated_mb={allocated >> 20} "
                f"torch_reserved_mb={reserved >> 20} "
                f"inventoried_cuda_mb={inventoried / (1024 * 1024):.1f} "
                f"untracked_driver_mb={(driver_used - inventoried) / (1024 * 1024):.1f} "
                f"untracked_reserved_mb={(reserved - inventoried) / (1024 * 1024):.1f} "
                f"duplicate_refs={duplicate_refs} "
                f"duplicate_storage_mb={duplicate_bytes / (1024 * 1024):.1f} "
                f"{category_parts}"
            )
            logger.info(message)
            print(message, flush=True)

    def release_redundant_gpu_execution_tensors(
        self,
        *,
        release_lm_head_source: bool = True,
        allow_multi_gpu_lm_head: bool = False,
        release_router_fp32_mirrors: bool = False,
    ) -> int:
        """Move GPU tensors no longer needed after Rust execution setup back to CPU.

        The Rust decode store needs a BF16 lm_head pointer for GEMV. When the
        configured lm_head is INT8, setup_gpu_decode_store creates an independent
        BF16 GPU copy in self._rust_lm_head. The original INT8 tuple can live on
        CPU after all stores that need source data have been configured.

        Router FP32 mirrors are Python-forward caches. Rust serving and
        benchmarking use Rust-registered routing weights, so those mirrors can be
        dropped only on Rust-only paths.
        """
        if not release_lm_head_source and not release_router_fp32_mirrors:
            return 0

        released_bytes = 0
        released_lm_head_bytes = 0
        released_router_bytes = 0

        def _cuda_bytes(value) -> int:
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                return int(value.nelement() * value.element_size())
            if isinstance(value, dict):
                return sum(_cuda_bytes(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(_cuda_bytes(item) for item in value)
            return 0

        def _to_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.cpu() if value.device.type == "cuda" else value
            if isinstance(value, tuple):
                return tuple(_to_cpu(item) for item in value)
            if isinstance(value, list):
                return [_to_cpu(item) for item in value]
            if isinstance(value, dict):
                return {key: _to_cpu(item) for key, item in value.items()}
            return value

        can_release_lm_head = (
            release_lm_head_source
            and (allow_multi_gpu_lm_head or len(getattr(self, "all_devices", []) or []) == 1)
            and not self.cfg.tie_word_embeddings
            and getattr(self, "_rust_lm_head", None) is not None
        )
        if can_release_lm_head:
            released_lm_head_bytes = _cuda_bytes(self.lm_head_data)
            released_bytes += released_lm_head_bytes

        router_mirrors = []
        if release_router_fp32_mirrors:
            for layer in self.layers:
                if not getattr(layer, "is_moe", False):
                    continue
                for attr in (
                    "_gate_weight_f32",
                    "_gate_bias_f32",
                    "_e_score_correction_bias_f32",
                    "_router_vl_bias_f32",
                ):
                    value = getattr(layer, attr, None)
                    bytes_ = _cuda_bytes(value)
                    if bytes_ > 0:
                        router_mirrors.append((layer, attr))
                        released_router_bytes += bytes_
            released_bytes += released_router_bytes

        if released_bytes <= 0:
            return 0

        free_before = None
        if torch.cuda.is_available():
            try:
                free_before = torch.cuda.mem_get_info(self.all_devices[0])[0] >> 20
            except Exception:
                free_before = None

        if released_lm_head_bytes > 0:
            self.lm_head_data = _to_cpu(self.lm_head_data)
            for state in getattr(self, "_device_state", {}).values():
                if isinstance(state, dict) and state.get("lm_head_data") is not None:
                    state["lm_head_data"] = self.lm_head_data
        for layer, attr in router_mirrors:
            setattr(layer, attr, None)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        free_after = None
        if torch.cuda.is_available():
            try:
                free_after = torch.cuda.mem_get_info(self.all_devices[0])[0] >> 20
            except Exception:
                free_after = None

        released_mb = released_bytes >> 20
        if free_before is not None and free_after is not None:
            logger.info(
                "Released redundant GPU execution tensors after Rust setup: total=%d MB, lm_head_source=%d MB, router_fp32_mirrors=%d MB, free_before=%d MB, free_after=%d MB",
                released_mb,
                released_lm_head_bytes >> 20,
                released_router_bytes >> 20,
                free_before,
                free_after,
            )
        else:
            logger.info(
                "Released redundant GPU execution tensors after Rust setup: total=%d MB, lm_head_source=%d MB, router_fp32_mirrors=%d MB",
                released_mb,
                released_lm_head_bytes >> 20,
                released_router_bytes >> 20,
            )
        return released_mb

    def _should_stream_attention(self, headroom_mb: int = 2000) -> bool:
        """Check if attention weights should be streamed (offloaded) during prefill.

        Returns True if total attention VRAM exceeds what GPU0 can hold after
        accounting for headroom needed for KV cache, activations, and DMA buffers.
        """
        if not self._loaded or not self.layers:
            return False
        dev = self.all_devices[0]
        free, total = torch.cuda.mem_get_info(dev)
        attn_vram = self._estimate_attention_vram()
        # Headroom for KV cache, activations, DMA buffers, CUDA overhead
        available = free + attn_vram  # If we offloaded, we'd reclaim attn_vram
        needed = attn_vram + headroom_mb * 1024 * 1024
        should_stream = needed > total
        if should_stream:
            logger.info(
                "Attention streaming ENABLED: %d MB attention, %d MB total VRAM, "
                "%d MB needed (with %d MB headroom)",
                attn_vram >> 20, total >> 20, needed >> 20, headroom_mb,
            )
        return should_stream

    def _offload_attention_to_ram(self):
        """Move all attention weights from GPU0 to pinned CPU RAM for streaming.

        After this, layers still exist as TransformerLayer objects but their
        attention weights are on CPU. Gate/shared expert weights also offloaded.
        """
        self._attn_offloaded = True
        self._attn_cpu_weights = {}  # layer_idx -> {attr: cpu_tensor}
        t0 = time.perf_counter()
        total_bytes = 0

        for layer_idx, layer in enumerate(self.layers):
            w = self._extract_layer_weights(layer, layer.device)
            cpu_w = self._copy_weights_dict(w, 'cpu')
            self._attn_cpu_weights[layer_idx] = cpu_w
            total_bytes += sum(
                t.nelement() * t.element_size()
                for t in self._flatten_weights(cpu_w)
            )
            # Null GPU refs to free VRAM (keep layer object for structure)
            self._null_layer_gpu_weights(layer)

        torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Attention offloaded: %d layers, %d MB to CPU RAM in %.1fs",
            len(self.layers), total_bytes >> 20, elapsed,
        )

    def _load_attention_group(self, layer_indices: List[int]):
        """Load attention weights for a group of layers from CPU RAM to GPU0.

        Call before processing a layer group during prefill when streaming.
        """
        if not getattr(self, '_attn_offloaded', False):
            return
        dev = self.all_devices[0]
        for layer_idx in layer_indices:
            cpu_w = self._attn_cpu_weights.get(layer_idx)
            if cpu_w is None:
                continue
            gpu_w = self._copy_weights_dict(cpu_w, dev)
            old_layer = self.layers[layer_idx]
            new_layer = TransformerLayer(
                self.cfg, layer_idx, gpu_w, dev,
                krasis_engine=old_layer.krasis_engine,
                gpu_prefill_manager=old_layer.gpu_prefill_manager,
                gpu_prefill_threshold=self.gpu_prefill_threshold,
            )
            # Preserve CPU decode buffers
            for buf_name in ('_cpu_act_buf', '_cpu_ids_buf', '_cpu_wts_buf',
                             '_cpu_out_buf', '_gpu_out_buf'):
                buf = getattr(old_layer, buf_name, None)
                if buf is not None:
                    setattr(new_layer, buf_name, buf)
            self.layers[layer_idx] = new_layer

    def _free_attention_group(self, layer_indices: List[int]):
        """Free GPU attention weights for a group of layers after processing.

        Nulls GPU refs so VRAM can be reclaimed for the next group's DMA.
        """
        if not getattr(self, '_attn_offloaded', False):
            return
        for layer_idx in layer_indices:
            layer = self.layers[layer_idx]
            self._null_layer_gpu_weights(layer)
        torch.cuda.empty_cache()

    def _reload_all_attention(self):
        """Reload all attention weights to GPU0 permanently (for decode after prefill).

        After prefill with streaming, decode needs all attention weights resident
        on GPU0 for the full layer loop.
        """
        if not getattr(self, '_attn_offloaded', False):
            return
        dev = self.all_devices[0]
        t0 = time.perf_counter()
        for layer_idx in range(len(self.layers)):
            cpu_w = self._attn_cpu_weights.get(layer_idx)
            if cpu_w is None:
                continue
            gpu_w = self._copy_weights_dict(cpu_w, dev)
            old_layer = self.layers[layer_idx]
            new_layer = TransformerLayer(
                self.cfg, layer_idx, gpu_w, dev,
                krasis_engine=old_layer.krasis_engine,
                gpu_prefill_manager=old_layer.gpu_prefill_manager,
                gpu_prefill_threshold=self.gpu_prefill_threshold,
            )
            for buf_name in ('_cpu_act_buf', '_cpu_ids_buf', '_cpu_wts_buf',
                             '_cpu_out_buf', '_gpu_out_buf'):
                buf = getattr(old_layer, buf_name, None)
                if buf is not None:
                    setattr(new_layer, buf_name, buf)
            self.layers[layer_idx] = new_layer
        self._attn_offloaded = False
        elapsed = time.perf_counter() - t0
        logger.info("Attention reloaded to GPU0: %d layers in %.1fs", len(self.layers), elapsed)

    # ── Streaming attention decode: per-layer DMA from CPU pinned memory ──

    def _init_stream_attention(self):
        """Offload attention weights to CPU pinned memory for streaming decode.

        Uses the existing _offload_attention_to_ram() for prefill compatibility,
        then creates pinned CPU copies and pre-allocated GPU buffers for fast
        per-layer decode DMA.

        When _load_gpu_weights already did incremental offloading (stream_attention
        was True during load), _attn_cpu_weights is pre-populated and norms/gate/
        shared are already on GPU. We skip Step 1 and Step 2.

        Prefill: uses _load_attention_group / _free_attention_group (existing)
        Decode: uses _stream_attn_load / _stream_attn_prefetch (new, fast)
        """
        dev = self.all_devices[0]
        t0 = time.perf_counter()

        already_offloaded = getattr(self, '_attn_offloaded', False)

        if not already_offloaded:
            # Step 1: Use existing offload to move weights to CPU (populates _attn_cpu_weights)
            self._offload_attention_to_ram()

            # Step 2: Reload non-attention components to GPU permanently
            # (norms, gate weights, shared expert, dense MLP — small, needed every decode)
            for layer_idx in range(len(self.layers)):
                cpu_w = self._attn_cpu_weights.get(layer_idx)
                if cpu_w is None:
                    continue
                layer = self.layers[layer_idx]

                # Norms (tiny, ~28 KB per layer)
                norms = cpu_w.get("norms", {})
                if "input_layernorm" in norms:
                    layer.input_norm_weight = norms["input_layernorm"].to(dev)
                if "post_attention_layernorm" in norms:
                    layer.post_attn_norm_weight = norms["post_attention_layernorm"].to(dev)
                for attr, key in (
                    ("pre_ffn_norm_weight", "pre_feedforward_layernorm"),
                    ("post_ffn_norm_weight", "post_feedforward_layernorm"),
                    ("post_ffn_norm1_weight", "post_feedforward_layernorm_1"),
                    ("post_ffn_norm2_weight", "post_feedforward_layernorm_2"),
                    ("pre_ffn_norm2_weight", "pre_feedforward_layernorm_2"),
                    ("layer_scalar", "layer_scalar"),
                ):
                    if key in norms:
                        setattr(layer, attr, norms[key].to(dev))

                # Gate weights (MoE routing)
                if layer.is_moe and "gate" in cpu_w:
                    gate_d = cpu_w["gate"]
                    layer.gate_weight = gate_d["weight"].to(dev)
                    layer._gate_weight_f32 = layer.gate_weight.float()
                    if "bias" in gate_d:
                        layer.gate_bias = gate_d["bias"].to(dev)
                        layer._gate_bias_f32 = layer.gate_bias.float()
                    if "e_score_correction_bias" in gate_d:
                        layer.e_score_correction_bias = gate_d["e_score_correction_bias"].to(dev)
                        layer._e_score_correction_bias_f32 = layer.e_score_correction_bias.float()
                    if "vl_bias" in gate_d:
                        layer.router_vl_bias = gate_d["vl_bias"].to(dev)
                        layer._router_vl_bias_f32 = layer.router_vl_bias.float()
                    if "input_scale" in gate_d:
                        layer.router_input_scale = gate_d["input_scale"].to(dev)
                    if "per_expert_scale" in gate_d:
                        layer.router_per_expert_scale = gate_d["per_expert_scale"].to(dev)

                # Shared expert (MoE)
                if layer.is_moe and "shared_expert" in cpu_w:
                    se = self._copy_weights_dict(cpu_w["shared_expert"], dev)
                    layer.shared_expert = se
                    # Re-fuse gate+up proj
                    gp = se.get("gate_proj")
                    up = se.get("up_proj")
                    if gp is not None and up is not None:
                        if isinstance(gp, tuple):
                            se["gate_up_proj"] = (
                                torch.cat([gp[0], up[0]], dim=0),
                                torch.cat([gp[1], up[1]], dim=0),
                            )
                        else:
                            se["gate_up_proj"] = torch.cat([gp, up], dim=0)
                        del se["gate_proj"], se["up_proj"]
                    if "shared_expert_gate" in se:
                        layer.shared_expert_gate = se["shared_expert_gate"]

                # Dense MLP (non-MoE layers, plus Gemma4 dense+MoE layers)
                if "dense_mlp" in cpu_w:
                    layer.dense_mlp = self._copy_weights_dict(cpu_w["dense_mlp"], dev)
        else:
            logger.info("Attention already offloaded during load — skipping Steps 1-2")

        # Step 3: Create pinned CPU copies of ATTENTION-ONLY weights for fast DMA
        total_bytes = 0
        self._stream_attn_keys = {}

        for layer_idx in range(len(self.layers)):
            cpu_w = self._attn_cpu_weights.get(layer_idx)
            if cpu_w is None:
                self._stream_attn_cpu[layer_idx] = {}
                self._stream_attn_keys[layer_idx] = []
                continue

            # Extract attention-only weights and pin them
            attn_dict = cpu_w.get("attention") or cpu_w.get("linear_attention", {})
            pinned = {}
            keys = []
            for attr_name, val in attn_dict.items():
                if isinstance(val, tuple) and len(val) == 2:
                    pw = val[0].pin_memory() if not val[0].is_pinned() else val[0]
                    ps = val[1].pin_memory() if not val[1].is_pinned() else val[1]
                    pinned[attr_name] = (pw, ps)
                    total_bytes += pw.nelement() * pw.element_size()
                    total_bytes += ps.nelement() * ps.element_size()
                    keys.append(attr_name)
                elif isinstance(val, torch.Tensor):
                    p = val.pin_memory() if not val.is_pinned() else val
                    pinned[attr_name] = p
                    total_bytes += p.nelement() * p.element_size()
                    keys.append(attr_name)

            self._stream_attn_cpu[layer_idx] = pinned
            self._stream_attn_keys[layer_idx] = keys

        # Step 3: Pre-allocate GPU buffers per layer type (DOUBLE-BUFFERED)
        # Hybrid models (e.g. QCN) have different attention architectures
        # per layer (linear attention vs GQA), so we need separate buffer
        # sets keyed by the set of attribute names.
        # We allocate TWO buffer sets (ping-pong) so DMA into buf (N+1)%2
        # can overlap with compute on buf N%2.
        self._stream_attn_gpu_bufs: list = [{}, {}]  # [buf0, buf1], each: frozenset -> {attr: gpu_buf}
        self._stream_attn_layer_key: dict = {}   # layer_idx -> frozenset(attr_names)
        gpu_buf_bytes = 0

        for layer_idx in range(len(self.layers)):
            cpu_tensors = self._stream_attn_cpu[layer_idx]
            if not cpu_tensors:
                self._stream_attn_layer_key[layer_idx] = frozenset()
                continue
            key = frozenset(cpu_tensors.keys())
            self._stream_attn_layer_key[layer_idx] = key
            for buf_idx in range(2):
                if key not in self._stream_attn_gpu_bufs[buf_idx]:
                    bufs = {}
                    for attr_name, val in cpu_tensors.items():
                        if isinstance(val, tuple):
                            bufs[attr_name] = (
                                torch.empty_like(val[0], device=dev),
                                torch.empty_like(val[1], device=dev),
                            )
                            gpu_buf_bytes += val[0].nelement() * val[0].element_size()
                            gpu_buf_bytes += val[1].nelement() * val[1].element_size()
                        else:
                            bufs[attr_name] = torch.empty_like(val, device=dev)
                            gpu_buf_bytes += val.nelement() * val.element_size()
                    self._stream_attn_gpu_bufs[buf_idx][key] = bufs

        num_types = len(self._stream_attn_gpu_bufs[0])
        logger.info("Streaming attention: %d buffer set(s) for %d layer type(s) (double-buffered)",
                     num_types, num_types)

        # Step 4: DMA stream for async prefetch
        self._stream_attn_dma_stream = torch.cuda.Stream(dev)
        self._stream_attn_enabled = True
        self._stream_attn_loaded: dict = {}  # {buf_idx: layer_idx} — tracks what's in each buffer

        torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        gpu_free = torch.cuda.mem_get_info(dev)[0] >> 20
        logger.info(
            "Streaming attention decode: %d layers, %d MB pinned, "
            "%d MB GPU buffers (%d type(s), 2x ping-pong), GPU free: %d MB, %.1fs",
            len(self.layers), total_bytes >> 20, gpu_buf_bytes >> 20,
            num_types, gpu_free, elapsed,
        )

    def _get_stream_bufs(self, layer_idx: int, buf_idx: int) -> dict:
        """Get the GPU staging buffers for this layer's attention type.

        buf_idx: 0 or 1 (ping-pong buffer index).
        """
        key = self._stream_attn_layer_key.get(layer_idx, frozenset())
        return self._stream_attn_gpu_bufs[buf_idx].get(key, {})

    def _stream_attn_load(self, layer_idx: int, buf_idx: int):
        """Copy one layer's attention weights from CPU pinned to GPU buffers.

        Sets the layer's attention attributes to point to the pre-allocated
        GPU buffers. Norms, gate, shared expert are permanently on GPU.

        buf_idx: 0 or 1 (ping-pong buffer index).
        """
        if self._stream_attn_loaded.get(buf_idx) == layer_idx:
            return
        cpu_tensors = self._stream_attn_cpu[layer_idx]
        gpu_bufs = self._get_stream_bufs(layer_idx, buf_idx)
        layer = self.layers[layer_idx]
        attn = layer.attention
        gqa_weights = getattr(layer, "gqa_weights", None)
        for dict_key, gpu_buf in gpu_bufs.items():
            src = cpu_tensors.get(dict_key)
            if src is None:
                continue
            if isinstance(gpu_buf, tuple):
                gpu_buf[0].copy_(src[0], non_blocking=True)
                gpu_buf[1].copy_(src[1], non_blocking=True)
            else:
                gpu_buf.copy_(src, non_blocking=True)
            # Map weight dict key to the actual attention attribute name
            attr_name = self._WEIGHT_KEY_TO_ATTN_ATTR.get(dict_key, dict_key)
            if attn is not None:
                setattr(attn, attr_name, gpu_buf)
            elif gqa_weights is not None:
                gqa_weights[attr_name] = gpu_buf
            else:
                raise RuntimeError(
                    f"stream attention load: layer {layer_idx} has no attention target for {attr_name}"
                )
        self._stream_attn_loaded[buf_idx] = layer_idx

    def _stream_attn_prefetch(self, layer_idx: int, buf_idx: int):
        """Start async DMA for a layer's attention weights on the DMA stream.

        Call this while CPU is busy with MoE experts. Before using the
        prefetched weights, sync the DMA stream.

        buf_idx: 0 or 1 (ping-pong buffer index).
        """
        if self._stream_attn_loaded.get(buf_idx) == layer_idx:
            return
        dma_stream = self._stream_attn_dma_stream
        cpu_tensors = self._stream_attn_cpu[layer_idx]
        gpu_bufs = self._get_stream_bufs(layer_idx, buf_idx)
        # The DMA stream must wait for the default stream before overwriting
        # a ping-pong buffer. The buffer we're writing to was read 2 layers
        # ago on the default stream — without this wait, the DMA copy can
        # race with those still-executing kernels.
        dma_stream.wait_stream(torch.cuda.current_stream(self.all_devices[0]))
        with torch.cuda.stream(dma_stream):
            for dict_key, gpu_buf in gpu_bufs.items():
                src = cpu_tensors.get(dict_key)
                if src is None:
                    continue
                if isinstance(gpu_buf, tuple):
                    gpu_buf[0].copy_(src[0], non_blocking=True)
                    gpu_buf[1].copy_(src[1], non_blocking=True)
                else:
                    gpu_buf.copy_(src, non_blocking=True)

    def _stream_attn_sync_prefetch(self, layer_idx: int, buf_idx: int):
        """Sync DMA stream and point layer's attention to GPU buffers.

        buf_idx: 0 or 1 (ping-pong buffer index).
        """
        torch.cuda.current_stream(self.all_devices[0]).wait_stream(self._stream_attn_dma_stream)
        cpu_tensors = self._stream_attn_cpu[layer_idx]
        gpu_bufs = self._get_stream_bufs(layer_idx, buf_idx)
        layer = self.layers[layer_idx]
        attn = layer.attention
        gqa_weights = getattr(layer, "gqa_weights", None)
        for dict_key, gpu_buf in gpu_bufs.items():
            if dict_key in cpu_tensors:
                attr_name = self._WEIGHT_KEY_TO_ATTN_ATTR.get(dict_key, dict_key)
                if attn is not None:
                    setattr(attn, attr_name, gpu_buf)
                elif gqa_weights is not None:
                    gqa_weights[attr_name] = gpu_buf
                else:
                    raise RuntimeError(
                        f"stream attention sync: layer {layer_idx} has no attention target for {attr_name}"
                    )
        self._stream_attn_loaded[buf_idx] = layer_idx

    @staticmethod
    def _flatten_weights(w_dict: dict) -> List[torch.Tensor]:
        """Flatten a nested weight dict into a list of tensors."""
        tensors = []
        for v in w_dict.values():
            if isinstance(v, dict):
                tensors.extend(KrasisModel._flatten_weights(v))
            elif isinstance(v, torch.Tensor):
                tensors.append(v)
            elif isinstance(v, tuple):
                for t in v:
                    if isinstance(t, torch.Tensor):
                        tensors.append(t)
        return tensors

    @staticmethod
    def _null_layer_gpu_weights(layer):
        """Null out GPU weight references on a layer to free VRAM."""
        layer.input_norm_weight = None
        layer.post_attn_norm_weight = None

        # Attention
        attn = layer.attention
        for attr_name in list(vars(attn).keys()):
            val = getattr(attn, attr_name, None)
            if isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                setattr(attn, attr_name, None)
            elif isinstance(val, tuple) and len(val) == 2:
                if all(isinstance(t, torch.Tensor) for t in val):
                    setattr(attn, attr_name, None)

        # Gate + shared expert
        if layer.is_moe:
            layer.gate_weight = None
            layer.gate_bias = None
            layer.e_score_correction_bias = None
            layer.router_vl_bias = None
            layer._router_vl_bias_f32 = None
            layer.shared_expert = None
            layer.shared_expert_gate = None

    def _load_cpu_experts(self, gpu_only: bool = False):
        """Load expert weights into Krasis Rust engine.

        Args:
            gpu_only: If True, skip CPU expert cache loading (saves ~40 GB RAM
                and ~45s load time). GPU Marlin experts are still loaded for
                prefill and HCS decode. CPU decode will not work in this mode.
        """
        from krasis import KrasisEngine

        cpu_bits = self.quant_cfg.cpu_expert_bits
        gpu_bits = self.quant_cfg.gpu_expert_bits
        tileq_cache = getattr(self.quant_cfg, "tileq_cache", None)
        if gpu_bits == 3:
            if not tileq_cache:
                raise RuntimeError("TileQ GPU experts require an explicit tileq_cache artifact")
            os.environ["KRASIS_TILEQ_CACHE"] = os.path.abspath(os.path.expanduser(tileq_cache))
            shared_bits = {"int8": 8, "bf16": 16}.get(self.quant_cfg.shared_expert)
            if shared_bits is None:
                raise RuntimeError(
                    "TileQ shared experts require shared_expert quantization int8 or bf16, "
                    f"got {self.quant_cfg.shared_expert!r}"
                )
            # TileQ replaces only the routed bank.  Keep a model's independent
            # shared-expert path at the precision selected by the normal
            # runtime config rather than implicitly changing it to INT3.
            os.environ["KRASIS_TILEQ_SHARED_EXPERT_BITS"] = str(shared_bits)
        elif tileq_cache:
            raise RuntimeError("tileq_cache is valid only when gpu_expert_bits=3")

        # If model has shared_expert_gate, Python/GPU handles shared expert with gate
        # → tell Rust engine to skip shared experts to avoid double-counting
        skip_shared = self._has_shared_expert_gate
        if skip_shared:
            logger.info("shared_expert_gate detected — Rust engine will skip shared experts (handled on GPU)")
        engine = KrasisEngine(parallel=True, num_threads=self.krasis_threads, skip_shared_experts=skip_shared)

        if self.gguf_path:
            logger.info("Loading CPU experts from GGUF: %s (native=%s)", self.gguf_path, self.gguf_native)
            engine.load(
                self.cfg.model_path,
                group_size=self.quant_cfg.expert_group_size,
                cpu_num_bits=cpu_bits,
                gpu_num_bits=gpu_bits,
                expert_int4_calib=self.quant_cfg.gpu_expert_int4_calib,
                gguf_path=self.gguf_path,
                gguf_native=self.gguf_native,
                expert_hqq_diagnostic_cache_spec=self.expert_hqq_diagnostic_cache_spec,
            )
        else:
            engine.load(
                self.cfg.model_path,
                group_size=self.quant_cfg.expert_group_size,
                cpu_num_bits=cpu_bits,
                gpu_num_bits=gpu_bits,
                expert_int4_calib=self.quant_cfg.gpu_expert_int4_calib,
                gpu_only=gpu_only,
                expert_hqq_diagnostic_cache_spec=self.expert_hqq_diagnostic_cache_spec,
            )

        self.krasis_engine = engine

        # Wire engine to MoE layers on ALL devices + allocate pinned buffers
        # (CPU MoE buffers only needed when NOT gpu_only — they're for CPU decode path)
        first_k = self.cfg.first_k_dense_replace
        for layer_idx, layer in enumerate(self.layers):
            if layer.is_moe:
                layer.krasis_engine = engine
                if not gpu_only and layer._cpu_act_buf is None:
                    layer._cpu_act_buf = torch.empty(1, self.cfg.hidden_size, dtype=torch.bfloat16, pin_memory=True)
                    layer._cpu_ids_buf = torch.empty(1, self.cfg.num_experts_per_tok, dtype=torch.int32, pin_memory=True)
                    layer._cpu_wts_buf = torch.empty(1, self.cfg.num_experts_per_tok, dtype=torch.float32, pin_memory=True)
                    layer._cpu_out_buf = torch.empty(1, self.cfg.hidden_size, dtype=torch.bfloat16, pin_memory=True)
                    layer._gpu_out_buf = torch.empty(1, self.cfg.hidden_size, dtype=torch.bfloat16, device=layer.device)

        logger.info(
            "Krasis engine: %d MoE layers, %d experts, hidden=%d",
            engine.num_moe_layers(), engine.num_experts(), engine.hidden_size(),
        )

        # ── Send routing weights to Rust for fused routing+MoE (M=1 decode) ──
        num_moe_layers = self.cfg.num_moe_layers
        engine.set_routing_config(
            scoring_func=self.cfg.scoring_func,
            norm_topk_prob=self.cfg.norm_topk_prob,
            topk=self.cfg.num_experts_per_tok,
            n_experts=self.cfg.n_routed_experts,
            hidden_size=self.cfg.hidden_size,
            num_moe_layers=num_moe_layers,
        )
        moe_idx = 0
        for layer in self.layers:
            if layer.is_moe:
                gw = layer.gate_weight.cpu().contiguous()
                gw_bytes = gw.view(torch.uint16).numpy().view(np.uint8).tobytes()
                bias_bytes = None
                if layer.e_score_correction_bias is not None:
                    bias = layer.e_score_correction_bias.cpu().float().contiguous()
                    bias_bytes = bias.numpy().view(np.uint8).tobytes()
                elif self.cfg.swiglu_limit > 0 and layer.gate_bias is not None:
                    # GPT OSS: send gate bias as correction_bias for Rust routing fallback
                    bias = layer.gate_bias.cpu().float().contiguous()
                    bias_bytes = bias.numpy().view(np.uint8).tobytes()
                engine.set_routing_weights(moe_idx, gw_bytes, bias_bytes)
                moe_idx += 1
        logger.info("Routing weights sent to Rust engine (%d MoE layers)", moe_idx)

    def _init_gpu_prefill(self):
        """No-op: Python GPU prefill has been replaced by Rust prefill engine."""
        self._require_supported_runtime_features()
        return

    def _start_ram_watchdog(self, floor_pct: float = 5.0):
        """Start daemon thread that monitors system RAM and exits if too low.

        Checks /proc/meminfo every second. If MemAvailable drops below
        floor_pct% of MemTotal, logs an error and calls os._exit() to
        prevent a full system OOM that kills desktop processes.

        Args:
            floor_pct: Minimum % free RAM before forced exit (default 5%)
        """
        def _watchdog():
            while True:
                time.sleep(1.0)
                meminfo = _read_meminfo()
                if not meminfo:
                    continue
                total_kb = meminfo.get("MemTotal", 0)
                avail_kb = meminfo.get("MemAvailable", 0)
                if total_kb == 0:
                    continue
                pct_free = 100.0 * avail_kb / total_kb
                if pct_free < floor_pct:
                    logger.error(
                        "RAM WATCHDOG: %.1f%% free (%.1f GB available / %.1f GB total) "
                        "— below %.1f%% floor. Exiting to prevent system OOM!",
                        pct_free, avail_kb / 1024 / 1024,
                        total_kb / 1024 / 1024, floor_pct,
                    )
                    os._exit(137)

        t = threading.Thread(target=_watchdog, daemon=True, name="ram-watchdog")
        t.start()
        logger.info("RAM watchdog started: will exit if < %.1f%% free", floor_pct)

    # ── Multi-GPU calibration: replicate weights + measure inference cost ──

    @staticmethod
    def _move_weight(w, device):
        """Move a weight (tensor or INT8 (tensor, scale) tuple) to device."""
        if w is None:
            return None
        if isinstance(w, tuple):
            return tuple(t.to(device) for t in w)
        if isinstance(w, torch.Tensor):
            return w.to(device)
        return w

    def _replicate_to_device(self, device: torch.device) -> dict:
        """Copy all model weights to `device`, return saved references.

        Handles top-level weights (embedding, final_norm, lm_head),
        per-layer norms, attention weights (MLA/GQA/linear), MoE gate
        + shared expert weights, and dense MLP weights.

        Device-bound caches (RoPE, CUDA graphs)
        are invalidated — they re-create lazily on the new device.
        """
        mv = self._move_weight
        saved = {"layers": []}

        # ── Top-level weights ──
        saved["embedding"] = self.embedding
        saved["final_norm"] = self.final_norm
        saved["lm_head_data"] = self.lm_head_data
        self.embedding = self.embedding.to(device)
        self.final_norm = self.final_norm.to(device)
        self.lm_head_data = mv(self.lm_head_data, device)

        # ── Per-layer weights ──
        for layer in self.layers:
            ls = {"device": layer.device}

            # Norms
            ls["input_norm_weight"] = layer.input_norm_weight
            ls["post_attn_norm_weight"] = layer.post_attn_norm_weight
            layer.input_norm_weight = layer.input_norm_weight.to(device)
            layer.post_attn_norm_weight = layer.post_attn_norm_weight.to(device)

            # GPU output buffer (MoE layers)
            if layer._gpu_out_buf is not None:
                ls["_gpu_out_buf"] = layer._gpu_out_buf
                layer._gpu_out_buf = torch.empty_like(layer._gpu_out_buf, device=device)

            # ── Attention weights ──
            attn = layer.attention
            attn_saved = {}

            if attn is None:
                # GQA attention handled by Rust prefill — no Python attention object
                attn_saved = {"type": "rust_prefill"}
            elif (
                layer.layer_type == "linear_attention"
                and self.cfg.is_kimi_delta_attention_layer(layer_idx)
            ):
                attn_saved["weights"] = dict(attn.weights)
                attn.weights = {
                    name: mv(tensor, device)
                    for name, tensor in attn.weights.items()
                }
            elif (
                layer.layer_type == "linear_attention"
                and self.cfg.is_kimi_delta_attention_layer(layer_idx)
            ):
                # KDA uses independent Q/K/V projections whose exact output
                # width is heads * head_dim.  Do not apply GatedDeltaNet's
                # fused QKVZ geometry to this distinct attention family.
                kda_width = (
                    self.cfg.linear_num_key_heads
                    * self.cfg.linear_key_head_dim
                )
                max_qkv = max(max_qkv, kda_width)
            elif layer.layer_type == "linear_attention":
                # GatedDeltaNetAttention
                attn_saved["device"] = attn.device
                for name in ("in_proj_qkvz", "in_proj_ba", "out_proj",
                             "conv1d_weight", "A_log", "dt_bias", "norm_weight"):
                    attn_saved[name] = getattr(attn, name)
                    setattr(attn, name, mv(getattr(attn, name), device))
                # Invalidate linear attention state and CUDA graph
                attn_saved["_conv_state"] = attn._conv_state
                attn_saved["_recurrent_state"] = attn._recurrent_state
                attn_saved["_la_graph"] = attn._la_graph
                attn_saved["_la_input"] = attn._la_input
                attn_saved["_la_output"] = attn._la_output
                attn_saved["_la_stream"] = attn._la_stream
                attn._conv_state = None
                attn._recurrent_state = None
                attn._la_graph = None
                attn._la_input = None
                attn._la_output = None
                attn._la_stream = torch.cuda.Stream(device=device)
                attn.device = device
            else:
                raise NotImplementedError(
                    f"MLA attention replication not supported. "
                    f"Layer {layer.layer_idx} has unsupported attention type."
                )

            ls["attention"] = attn_saved

            # ── MoE weights ──
            if layer.is_moe:
                moe_saved = {}
                for name in ("gate_weight", "gate_bias", "e_score_correction_bias", "router_vl_bias"):
                    val = getattr(layer, name, None)
                    if val is not None:
                        moe_saved[name] = val
                        setattr(layer, name, val.to(device))

                # Shared expert dict
                if layer.shared_expert is not None:
                    moe_saved["shared_expert"] = dict(layer.shared_expert)
                    layer.shared_expert = {
                        k: mv(v, device) for k, v in layer.shared_expert.items()
                    }
                    # shared_expert_gate lives in shared_expert dict AND layer attr
                    if layer.shared_expert_gate is not None:
                        moe_saved["shared_expert_gate"] = layer.shared_expert_gate
                        layer.shared_expert_gate = layer.shared_expert.get("shared_expert_gate")

                # Recompute f32 caches
                moe_saved["_gate_weight_f32"] = layer._gate_weight_f32
                moe_saved["_gate_bias_f32"] = layer._gate_bias_f32
                moe_saved["_e_score_correction_bias_f32"] = layer._e_score_correction_bias_f32
                moe_saved["_router_vl_bias_f32"] = layer._router_vl_bias_f32
                layer._gate_weight_f32 = layer.gate_weight.float() if layer.gate_weight is not None else None
                layer._gate_bias_f32 = layer.gate_bias.float() if layer.gate_bias is not None else None
                layer._e_score_correction_bias_f32 = (
                    layer.e_score_correction_bias.float()
                    if layer.e_score_correction_bias is not None else None
                )
                layer._router_vl_bias_f32 = (
                    layer.router_vl_bias.float()
                    if layer.router_vl_bias is not None else None
                )

                # Nullify CUDA graph (device-bound)
                moe_saved["_se_graph"] = layer._se_graph
                moe_saved["_se_input"] = layer._se_input
                moe_saved["_se_output"] = layer._se_output
                moe_saved["_shared_stream"] = layer._shared_stream
                layer._se_graph = None
                layer._se_input = None
                layer._se_output = None
                layer._shared_stream = (
                    torch.cuda.Stream(device=device)
                    if layer.shared_expert_gate is not None else None
                )

                ls["moe"] = moe_saved

            elif layer.dense_mlp is not None:
                ls["dense_mlp"] = dict(layer.dense_mlp)
                layer.dense_mlp = {
                    k: mv(v, device) for k, v in layer.dense_mlp.items()
                }

            layer.device = device
            saved["layers"].append(ls)

        return saved

    def _restore_from_device(self, saved: dict):
        """Restore all tensor references from saved state and free copies."""
        import gc

        # ── Top-level ──
        self.embedding = saved["embedding"]
        self.final_norm = saved["final_norm"]
        self.lm_head_data = saved["lm_head_data"]

        # ── Per-layer ──
        for layer, ls in zip(self.layers, saved["layers"]):
            layer.device = ls["device"]
            layer.input_norm_weight = ls["input_norm_weight"]
            layer.post_attn_norm_weight = ls["post_attn_norm_weight"]

            if "_gpu_out_buf" in ls:
                layer._gpu_out_buf = ls["_gpu_out_buf"]

            # Attention
            attn = layer.attention
            attn_saved = ls["attention"]

            if attn is None:
                # GQA handled by Rust prefill — nothing to restore
                pass
            elif layer.layer_type == "linear_attention":
                attn.device = attn_saved["device"]
                for name in ("in_proj_qkvz", "in_proj_ba", "out_proj",
                             "conv1d_weight", "A_log", "dt_bias", "norm_weight"):
                    setattr(attn, name, attn_saved[name])
                attn._conv_state = attn_saved["_conv_state"]
                attn._recurrent_state = attn_saved["_recurrent_state"]
                attn._la_graph = attn_saved["_la_graph"]
                attn._la_input = attn_saved["_la_input"]
                attn._la_output = attn_saved["_la_output"]
                attn._la_stream = attn_saved["_la_stream"]
            else:
                pass  # MLA not supported

            # MoE
            if "moe" in ls:
                ms = ls["moe"]
                for name in ("gate_weight", "gate_bias", "e_score_correction_bias", "router_vl_bias"):
                    if name in ms:
                        setattr(layer, name, ms[name])
                if "shared_expert" in ms:
                    layer.shared_expert = ms["shared_expert"]
                if "shared_expert_gate" in ms:
                    layer.shared_expert_gate = ms["shared_expert_gate"]
                layer._gate_weight_f32 = ms["_gate_weight_f32"]
                layer._gate_bias_f32 = ms["_gate_bias_f32"]
                layer._e_score_correction_bias_f32 = ms["_e_score_correction_bias_f32"]
                layer._router_vl_bias_f32 = ms["_router_vl_bias_f32"]
                layer._se_graph = ms["_se_graph"]
                layer._se_input = ms["_se_input"]
                layer._se_output = ms["_se_output"]
                layer._shared_stream = ms["_shared_stream"]

            elif "dense_mlp" in ls:
                layer.dense_mlp = ls["dense_mlp"]

        # Free any remaining copies
        del saved
        gc.collect()
        torch.cuda.empty_cache()

    def calibrate_on_device(self, device: torch.device, test_tokens) -> dict:
        """Temporarily replicate model to `device`, run inference, measure VRAM cost.

        Disables GPU prefill so we measure only non-expert VRAM
        (attention, norms, activations, KV cache). Expert VRAM is
        managed separately by HCS budgets.

        Returns dict with 'inference_cost_bytes'.
        """
        import gc
        from krasis.config import PPRankConfig

        logger.info("Calibrating inference cost on %s...", device)

        # ── Save current state ──
        saved_gpu_prefill_enabled = self.gpu_prefill_enabled
        saved_ranks = self.ranks
        saved_kv_caches = self.kv_caches
        saved_kv_layer_offsets = self._kv_layer_offsets
        saved_layer_split = self._layer_split
        saved_all_devices = self.all_devices
        saved_hcs_device = self._hcs_device
        saved_multi_gpu_hcs = self._multi_gpu_hcs

        # Save and null out per-layer gpu_prefill_manager
        saved_layer_managers = []
        for layer in self.layers:
            saved_layer_managers.append(layer.gpu_prefill_manager)
            layer.gpu_prefill_manager = None

        # Disable GPU prefill — forces CPU MoE path
        self.gpu_prefill_enabled = False

        # ── Replicate weights to target device ──
        saved_weights = self._replicate_to_device(device)

        # ── Temporary single-device config ──
        N = self.cfg.num_hidden_layers
        self._layer_split = [(0, N)]
        self.all_devices = [device]
        self._hcs_device = None
        self._multi_gpu_hcs = False
        orig_rank = saved_ranks[0]
        self.ranks = [PPRankConfig(
            rank=0,
            device=str(device),
            layer_start=0,
            layer_end=N,
            num_layers=N,
            has_embedding=True,
            has_lm_head=True,
        )]

        # Re-init KV caches on target device
        self._init_kv_caches()

        # ── Measure inference cost ──
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        inference_cost = 0

        try:
            self._oom_retry_enabled = False
            with torch.inference_mode():
                self.generate(test_tokens, max_new_tokens=1, temperature=0.6)

            peak = torch.cuda.max_memory_allocated(device)
            inference_cost = peak - baseline
            logger.info(
                "Calibration on %s: inference_cost=%d MB "
                "(peak=%d MB, baseline=%d MB)",
                device,
                inference_cost // (1024 * 1024),
                peak // (1024 * 1024),
                baseline // (1024 * 1024),
            )
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            logger.error("Calibration OOM on %s: %s", device, e)
            gc.collect()
            torch.cuda.empty_cache()
            raise
        finally:
            self._oom_retry_enabled = True

            # ── Restore everything ──
            self.gpu_prefill_enabled = saved_gpu_prefill_enabled
            self.ranks = saved_ranks
            self.kv_caches = saved_kv_caches
            self._kv_layer_offsets = saved_kv_layer_offsets
            self._layer_split = saved_layer_split
            self.all_devices = saved_all_devices
            self._hcs_device = saved_hcs_device
            self._multi_gpu_hcs = saved_multi_gpu_hcs

            for layer, mgr in zip(self.layers, saved_layer_managers):
                layer.gpu_prefill_manager = mgr

            self._restore_from_device(saved_weights)

        return {"inference_cost_bytes": inference_cost}

    def _init_kv_caches(self):
        """Allocate paged KV caches per GPU split group.

        Streaming attention: all attention on GPU0, so single KV cache on GPU0.
        For hybrid models, linear attention layers get -1 in _kv_layer_offsets.

        self.kv_caches[gpu_idx] = KV cache for the gpu_idx'th split group.
        """
        self.kv_caches = []
        self._kv_layer_offsets = {}
        kv_mb = self.kv_cache_mb

        for gpu_idx, (start, end) in enumerate(self._layer_split):
            dev = self.all_devices[gpu_idx]

            # Count full attention layers in this GPU's split
            kv_offset = 0
            kv_layer_indices = []
            for layer_idx in range(start, end):
                if self.cfg.is_full_attention_layer(layer_idx):
                    self._kv_layer_offsets[layer_idx] = kv_offset
                    kv_offset += 1
                    kv_layer_indices.append(layer_idx)
                else:
                    self._kv_layer_offsets[layer_idx] = -1

            num_kv_layers = kv_offset

            if num_kv_layers > 0:
                cache = PagedKVCache(
                    self.cfg,
                    num_layers=num_kv_layers,
                    device=dev,
                    kv_dtype=self.kv_dtype,
                    combined=False,
                    max_mb=kv_mb,
                    kv_format=self.quant_cfg.kv_cache_format,
                    layer_indices=kv_layer_indices,
                    enable_ring_window=self.quant_cfg.ring_window_kv,
                )
            else:
                cache = None
            self.kv_caches.append(cache)

        if self.cfg.is_hybrid:
            total_kv = sum(1 for v in self._kv_layer_offsets.values() if v >= 0)
            total_linear = sum(1 for v in self._kv_layer_offsets.values() if v < 0)
            logger.info(
                "Hybrid model: %d full attention layers (KV cache), %d linear attention layers",
                total_kv, total_linear,
            )

    def get_max_context_tokens(self) -> int:
        """Maximum prompt + generation tokens supported by the KV cache."""
        if not self.kv_caches:
            return 0
        # Bottleneck is the smallest KV cache across GPU splits
        return min(
            self.cfg.max_position_embeddings,
            min(c.max_context_tokens for c in self.kv_caches if c is not None),
        )

    def _get_rank_for_layer(self, global_layer_idx: int) -> int:
        """Get the PP rank index that owns a given layer."""
        offset = 0
        for i, rank in enumerate(self.ranks):
            if global_layer_idx < offset + rank.num_layers:
                return i
            offset += rank.num_layers
        raise ValueError(f"Layer {global_layer_idx} out of range")

    def _get_gpu_for_layer(self, layer_idx: int) -> int:
        """Get GPU index owning a given layer (streaming attention: always 0)."""
        for gpu_idx, (start, end) in enumerate(self._layer_split):
            if start <= layer_idx < end:
                return gpu_idx
        raise ValueError(f"Layer {layer_idx} not in any GPU split: {self._layer_split}")

    def _cross_device_moe(
        self,
        layer: TransformerLayer,
        hidden_on_layer_dev: torch.Tensor,
        moe_layer_idx: int,
    ) -> torch.Tensor:
        """Run MoE for a layer whose attention is on a different GPU than HCS.

        Routing + shared expert run on layer_dev (where gate/shared weights are).
        Routed experts run on HCS device (where Marlin experts + compact buffers are).
        Result transferred back to layer_dev.

        Args:
            layer: The transformer layer (on layer_dev)
            hidden_on_layer_dev: [M, hidden_size] post-attn-norm hidden state
            moe_layer_idx: 0-based MoE layer index

        Returns:
            MoE output [M, hidden_size] on layer_dev
        """
        layer_dev = layer.device
        hcs_dev = self._hcs_device
        timing = TIMING.decode

        if timing:
            torch.cuda.synchronize(layer_dev)
            t0 = time.perf_counter()

        # 1. Routing on layer_dev (tiny gate matmul)
        topk_ids, topk_weights = layer.compute_routing(hidden_on_layer_dev)

        if timing:
            torch.cuda.synchronize(layer_dev)
            t_routing = time.perf_counter()

        # 2. Shared expert on layer_dev (CUDA graph or direct)
        shared_output = None
        has_shared = layer._shared_stream is not None and layer.shared_expert is not None
        if has_shared:
            if layer._se_graph is None:
                layer._capture_shared_expert_graph()
            if layer._se_graph is not None:
                layer._se_input.copy_(hidden_on_layer_dev)
                layer._shared_stream.wait_stream(torch.cuda.current_stream(layer_dev))
                layer._se_graph.replay()
            else:
                layer._shared_stream.wait_stream(torch.cuda.current_stream(layer_dev))
                with torch.cuda.stream(layer._shared_stream):
                    shared_output = layer._shared_expert_forward(hidden_on_layer_dev)
        elif layer.shared_expert_gate is not None and layer.shared_expert is not None:
            shared_output = layer._shared_expert_forward(hidden_on_layer_dev)

        if timing:
            # Don't sync shared stream yet — it should overlap with routed experts
            t_shared_launched = time.perf_counter()

        # 3. Transfer to HCS device
        trace_xdev = _python_trace_enabled("py_moe")
        if trace_xdev:
            torch.cuda.synchronize(layer_dev)
            _xdev_t0 = time.perf_counter()
        hidden_hcs = _to_device(hidden_on_layer_dev, hcs_dev)
        topk_ids_hcs = _to_device(topk_ids, hcs_dev)
        topk_weights_hcs = _to_device(topk_weights, hcs_dev)

        if trace_xdev:
            torch.cuda.synchronize(hcs_dev)
            _xdev_xfer_ms = (time.perf_counter() - _xdev_t0) * 1000

        if timing:
            torch.cuda.synchronize(hcs_dev)
            t_xfer_to_hcs = time.perf_counter()

        # 4. Routed experts on HCS device via its manager
        hcs_mgr = self.gpu_prefill_managers.get(str(hcs_dev))
        if trace_xdev:
            _xdev_t1 = time.perf_counter()
        if hcs_mgr is not None:
            routed_output = hcs_mgr.forward(
                moe_layer_idx, hidden_hcs, topk_ids_hcs, topk_weights_hcs,
                routed_only=True,
            )
            if trace_xdev:
                torch.cuda.synchronize(hcs_dev)
                _xdev_fwd_ms = (time.perf_counter() - _xdev_t1) * 1000
        else:
            raise RuntimeError(
                f"HCS manager is None during GPU decode at MoE layer {moe_layer_idx}. "
                "GPU decode requires HCS to be initialized. This indicates a bug in "
                "server initialization -- HCS should always be set up before decode starts."
            )

        if timing:
            torch.cuda.synchronize(hcs_dev)
            t_hcs_done = time.perf_counter()

        # 5. Transfer routed output back to layer_dev
        if trace_xdev:
            _xdev_t2 = time.perf_counter()
        routed_on_layer = _to_device(routed_output, layer_dev)

        if trace_xdev:
            torch.cuda.synchronize(layer_dev)
            _xdev_back_ms = (time.perf_counter() - _xdev_t2) * 1000

        if timing:
            torch.cuda.synchronize(layer_dev)
            t_xfer_back = time.perf_counter()

        # 6. Apply scaling + combine with shared expert
        if trace_xdev:
            _xdev_t3 = time.perf_counter()
        rsf = self.cfg.routed_scaling_factor
        if rsf != 1.0:
            routed_on_layer = routed_on_layer * rsf
        if has_shared:
            torch.cuda.current_stream(layer_dev).wait_stream(layer._shared_stream)
            if layer._se_graph is not None:
                shared_output = layer._se_output
            routed_on_layer = routed_on_layer + shared_output
        elif shared_output is not None:
            routed_on_layer = routed_on_layer + shared_output

        if trace_xdev:
            torch.cuda.synchronize(layer_dev)
            _xdev_combine_ms = (time.perf_counter() - _xdev_t3) * 1000
            _python_trace(
                "py_moe",
                (
                    f"phase=cross_device_moe layer={moe_layer_idx} "
                    f"xfer_to_hcs_ms={_xdev_xfer_ms:.3f} "
                    f"hcs_forward_ms={_xdev_fwd_ms:.3f} "
                    f"xfer_back_ms={_xdev_back_ms:.3f} "
                    f"combine_ms={_xdev_combine_ms:.3f}"
                ),
            )

        if timing:
            torch.cuda.synchronize(layer_dev)
            t_combine = time.perf_counter()
            logger.info(
                "XDEV-MOE L%d: route=%.2f shared_launch=%.2f xfer_to=%.2f hcs_fwd=%.2f xfer_back=%.2f combine=%.2f total=%.2fms",
                moe_layer_idx,
                (t_routing - t0) * 1000,
                (t_shared_launched - t_routing) * 1000,
                (t_xfer_to_hcs - t_shared_launched) * 1000,
                (t_hcs_done - t_xfer_to_hcs) * 1000,
                (t_xfer_back - t_hcs_done) * 1000,
                (t_combine - t_xfer_back) * 1000,
                (t_combine - t0) * 1000,
            )

        return routed_on_layer

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_states: List[SequenceKVState],
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        """Full forward pass with streaming attention architecture.

        ALL attention runs on GPU0. During decode (M=1):
        - Attention runs on GPU0 (where all weights live)
        - MoE runs on HCS device via _cross_device_moe (8 KB hidden transfer)
        - No cross-GPU hidden state bouncing for attention

        Args:
            token_ids: [M] int64 token IDs
            positions: [M] int32 position indices
            seq_states: One SequenceKVState per GPU split group
            return_all_logits: If True, return logits for ALL positions [M, V]
                instead of just the last token [1, V]. Used for perplexity measurement.

        Returns:
            logits: [M, vocab_size] or [1, vocab_size] float32
        """
        assert self._loaded, "Model not loaded. Call load() first."
        M = token_ids.shape[0]
        timing = TIMING.decode and M == 1

        if timing:
            t_fwd_start = time.perf_counter()

        # ── Layer-grouped prefill routing ──
        if (
            M > 1
            and self.layer_group_size >= 1
            and M >= self.gpu_prefill_threshold
            and self.gpu_prefill_enabled
            and self.cfg.n_routed_experts > 0
        ):
            return self._forward_prefill_with_oom_retry(
                token_ids, positions, seq_states, return_all_logits=return_all_logits)

        # ── Embedding (GPU0) ──
        first_dev = self.all_devices[0]
        hidden = self.embedding[token_ids.to(first_dev)]  # [M, hidden_size]

        # Diagnostics
        diag = False
        if TIMING.diag:
            if not hasattr(self, '_diag_count'):
                self._diag_count = 0
            self._diag_count += 1
            diag = self._diag_count <= 12
            if diag:
                h = hidden[-1] if hidden.shape[0] > 1 else hidden[0]
                logger.info("DIAG[%d] embed: mean=%.4f std=%.4f max=%.4f nan=%d",
                            self._diag_count, h.float().mean(), h.float().std(),
                            h.float().abs().max(), h.isnan().sum().item())

        # ── Ensure KV capacity (once per GPU split group) ──
        for gpu_idx in range(len(self._layer_split)):
            ss = seq_states[gpu_idx]
            if ss is not None:
                ss.ensure_capacity(M)

        # ── Transformer layers ──
        residual = None
        first_k = self.cfg.first_k_dense_replace
        hcs_dev = self._hcs_device  # None if single GPU or HCS not set
        prev_layer_dev = first_dev
        positions_cache = {}  # device -> positions tensor (avoid repeated transfers)
        positions_cache[str(first_dev)] = positions.to(first_dev) if positions.device != first_dev else positions

        # Per-layer timing accumulators (only when timing enabled)
        if timing:
            _t_attn_total = 0.0  # total attention time (all layer types)
            _t_moe_total = 0.0   # total MoE time (cross-device or unified)
            _t_xfer_total = 0.0  # total cross-device hidden transfers
            _t_dma_total = 0.0   # total attention DMA time (streaming only)
            _t_lin_attn_total = 0.0  # linear attention time
            _t_gqa_attn_total = 0.0  # GQA attention time
            _t_lin_attn_layers = 0  # count of linear attention layers
            _t_gqa_layers = 0      # count of GQA attention layers
            _t_moe_total_gpu = 0.0  # GPU hot expert time within MoE
            _t_moe_layers = 0      # count of MoE layers
            _t_unified_layers = 0  # count of unified (same-device) layers

        # Streaming attention: double-buffered (ping-pong) DMA
        # Enable for ANY forward pass where attention weights are offloaded to CPU
        _stream_attn = self._stream_attn_enabled
        _prefetch_started = False  # whether an async prefetch is in flight
        _prefetch_buf_idx = -1     # which buffer the prefetch targets

        # Pre-load layer 0 synchronously into buf 0 before the loop
        if _stream_attn:
            self._stream_attn_load(0, buf_idx=0)
            _python_trace("py_stream", "phase=preload layer=0 buffer=0")

        for abs_layer_idx in range(self.cfg.num_hidden_layers):
            layer = self.layers[abs_layer_idx]
            layer_dev = torch.device(layer.device)
            gpu_idx = self._get_gpu_for_layer(abs_layer_idx)
            kv_cache = self.kv_caches[gpu_idx]
            seq_state = seq_states[gpu_idx]

            # MoE layer index for Krasis engine (0-based sequential)
            moe_layer_idx = self._abs_to_moe_idx.get(abs_layer_idx)
            kv_layer_offset = self._kv_layer_offsets.get(abs_layer_idx, 0)

            # Transfer hidden to layer device if needed
            if prev_layer_dev != layer_dev:
                if timing:
                    torch.cuda.synchronize(prev_layer_dev)
                    _t_xfer0 = time.perf_counter()
                hidden = _to_device(hidden, layer_dev)
                if residual is not None:
                    residual = _to_device(residual, layer_dev)
                prev_layer_dev = layer_dev
                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_xfer_total += time.perf_counter() - _t_xfer0

            # Cache positions per device
            dev_str = str(layer_dev)
            if dev_str not in positions_cache:
                positions_cache[dev_str] = _to_device(positions, layer_dev)
            layer_positions = positions_cache[dev_str]

            # Double-buffered streaming attention
            _dbg_stream_ms = 0.0  # default for non-streaming layers
            trace_stream = _python_trace_enabled("py_stream")
            if _stream_attn:
                buf_idx = abs_layer_idx % 2
                if trace_stream:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_t0 = time.perf_counter()

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_dma0 = time.perf_counter()

                # Sync prefetch if one was started for this layer
                if _prefetch_started and _prefetch_buf_idx == buf_idx:
                    self._stream_attn_sync_prefetch(abs_layer_idx, buf_idx)
                    _prefetch_started = False
                elif self._stream_attn_loaded.get(buf_idx) != abs_layer_idx:
                    # Fallback: synchronous load (first layer already loaded above)
                    self._stream_attn_load(abs_layer_idx, buf_idx)

                if trace_stream:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_stream_ms = (time.perf_counter() - _dbg_t0) * 1000

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_dma_total += time.perf_counter() - _t_dma0

                # Start prefetch for NEXT layer into opposite buffer BEFORE compute
                next_layer = abs_layer_idx + 1
                if next_layer < self.cfg.num_hidden_layers:
                    next_buf = next_layer % 2
                    self._stream_attn_prefetch(next_layer, next_buf)
                    _prefetch_started = True
                    _prefetch_buf_idx = next_buf

            # Decide execution path:
            # (a) Multi-GPU HCS: attention + routing + shared on layer_dev,
            #     routed experts dispatched to all GPUs via primary manager
            # (b) Single-GPU HCS: attention on layer_dev, MoE on HCS device
            # (c) Unified: everything on layer_dev
            use_multi_gpu_hcs = (
                M == 1
                and layer.is_moe
                and self._multi_gpu_hcs
            )
            use_cross_device_moe = (
                M == 1
                and layer.is_moe
                and not use_multi_gpu_hcs
                and hcs_dev is not None
                and torch.device(hcs_dev) != layer_dev
            )

            if use_multi_gpu_hcs:
                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la = time.perf_counter()

                # Attention on layer_dev (GPU0)
                if kv_layer_offset < 0:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        None, None, -1, num_new_tokens=M,
                    )
                else:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        kv_cache, seq_state, kv_layer_offset, num_new_tokens=M,
                    )

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la_done = time.perf_counter()
                    _dt = _t_la_done - _t_la
                    _t_attn_total += _dt
                    if layer.layer_type == "linear_attention":
                        _t_lin_attn_total += _dt
                        _t_lin_attn_layers += 1
                    else:
                        _t_gqa_attn_total += _dt
                        _t_gqa_layers += 1

                # Routing on layer_dev (tiny gate matmul)
                topk_ids, topk_weights = layer.compute_routing(hidden)

                # Shared expert on layer_dev (CUDA graph or direct)
                shared_output = None
                has_shared = layer._shared_stream is not None and layer.shared_expert is not None
                if has_shared:
                    if layer._se_graph is None:
                        layer._capture_shared_expert_graph()
                    if layer._se_graph is not None:
                        layer._se_input.copy_(hidden)
                        layer._shared_stream.wait_stream(torch.cuda.current_stream(layer_dev))
                        layer._se_graph.replay()
                    else:
                        layer._shared_stream.wait_stream(torch.cuda.current_stream(layer_dev))
                        with torch.cuda.stream(layer._shared_stream):
                            shared_output = layer._shared_expert_forward(hidden)
                elif layer.shared_expert_gate is not None and layer.shared_expert is not None:
                    shared_output = layer._shared_expert_forward(hidden)

                # Routed experts via primary manager — dispatches to all GPUs internally
                primary_mgr = self.gpu_prefill_managers.get(str(layer_dev))
                if primary_mgr is not None:
                    routed_output = primary_mgr.forward(
                        moe_layer_idx, hidden, topk_ids, topk_weights,
                        routed_only=True,
                    )
                else:
                    raise RuntimeError(
                        f"GPU prefill manager is None at MoE layer {moe_layer_idx} on {layer_dev}. "
                        "GPU prefill requires the prefill manager to be initialized. "
                        "This indicates a bug in server initialization."
                    )

                # Combine: apply routed scaling + shared expert
                rsf = self.cfg.routed_scaling_factor
                if rsf != 1.0:
                    routed_output = routed_output * rsf
                if has_shared:
                    torch.cuda.current_stream(layer_dev).wait_stream(layer._shared_stream)
                    if layer._se_graph is not None:
                        shared_output = layer._se_output
                    hidden = routed_output + shared_output
                elif shared_output is not None:
                    hidden = routed_output + shared_output
                else:
                    hidden = routed_output

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_moe_done = time.perf_counter()
                    _t_moe_total += _t_moe_done - _t_la_done
                    _t_moe_layers += 1

            elif use_cross_device_moe:
                trace_layer = _python_trace_enabled("py_layer")
                if trace_layer:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_attn_t0 = time.perf_counter()

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la = time.perf_counter()

                # Split path: attention on layer_dev, MoE on HCS device
                if kv_layer_offset < 0:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        None, None, -1, num_new_tokens=M,
                    )
                else:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        kv_cache, seq_state, kv_layer_offset, num_new_tokens=M,
                    )

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la_done = time.perf_counter()
                    _dt = _t_la_done - _t_la
                    _t_attn_total += _dt
                    if layer.layer_type == "linear_attention":
                        _t_lin_attn_total += _dt
                        _t_lin_attn_layers += 1
                    else:
                        _t_gqa_attn_total += _dt
                        _t_gqa_layers += 1

                if trace_layer:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_attn_ms = (time.perf_counter() - _dbg_attn_t0) * 1000
                    _dbg_moe_t0 = time.perf_counter()

                # MoE on HCS device, result back on layer_dev
                hidden = self._cross_device_moe(layer, hidden, moe_layer_idx)

                if trace_layer:
                    _dbg_moe_ms = (time.perf_counter() - _dbg_moe_t0) * 1000
                    _python_trace(
                        "py_layer",
                        (
                            f"phase=decode_layer layer={abs_layer_idx} mode=cross_device "
                            f"stream_ms={_dbg_stream_ms:.3f} attn_ms={_dbg_attn_ms:.3f} "
                            f"moe_ms={_dbg_moe_ms:.3f}"
                        ),
                    )

                if timing:
                    # _cross_device_moe already syncs internally when timing
                    _t_moe_done = time.perf_counter()
                    _t_moe_total += _t_moe_done - _t_la_done
                    _t_moe_layers += 1
            else:
                trace_layer = _python_trace_enabled("py_layer")
                if trace_layer:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_uni_t0 = time.perf_counter()

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la = time.perf_counter()

                # Unified path: attention first, then MLP/MoE
                # Split into forward_attn + MoE to get separate timing
                if kv_layer_offset < 0:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        None, None, -1, num_new_tokens=M,
                    )
                else:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        kv_cache, seq_state, kv_layer_offset, num_new_tokens=M,
                    )

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la_done = time.perf_counter()
                    _dt = _t_la_done - _t_la
                    _t_attn_total += _dt
                    if layer.layer_type == "linear_attention":
                        _t_lin_attn_total += _dt
                        _t_lin_attn_layers += 1
                    else:
                        _t_gqa_attn_total += _dt
                        _t_gqa_layers += 1

                # MLP / MoE
                if layer.is_moe:
                    hidden = layer._moe_forward(hidden, moe_layer_idx)
                else:
                    hidden = layer._dense_mlp_forward(hidden)

                if trace_layer:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_uni_ms = (time.perf_counter() - _dbg_uni_t0) * 1000
                    _python_trace(
                        "py_layer",
                        (
                            f"phase=decode_layer layer={abs_layer_idx} mode=unified "
                            f"stream_ms={_dbg_stream_ms:.3f} unified_ms={_dbg_uni_ms:.3f} "
                            f"moe={int(layer.is_moe)}"
                        ),
                    )

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_moe_done = time.perf_counter()
                    _t_unified_layers += 1
                    if layer.is_moe:
                        _t_moe_total += _t_moe_done - _t_la_done
                        _t_moe_layers += 1

            if TIMING.diag and diag and abs_layer_idx in (0, 1, self.cfg.num_hidden_layers // 2, self.cfg.num_hidden_layers - 1):
                h = hidden[-1] if hidden.shape[0] > 1 else hidden[0]
                r = residual[-1] if residual.shape[0] > 1 else residual[0]
                logger.info("DIAG[%d] L%d: hid std=%.4f max=%.4f | res std=%.4f max=%.4f",
                            self._diag_count, abs_layer_idx,
                            h.float().std(), h.float().abs().max(),
                            r.float().std(), r.float().abs().max())

        # Advance KV cache seq_len (once per GPU split group)
        for gpu_idx in range(len(self._layer_split)):
            ss = seq_states[gpu_idx]
            if ss is not None:
                ss.advance(M)

        if timing:
            torch.cuda.synchronize()
            t_after_layers = time.perf_counter()

        # ── Final norm + LM head (always on GPU0) ──
        hidden = _to_device(hidden, first_dev)
        if residual is not None:
            residual = _to_device(residual, first_dev)

        from krasis.layer import _fused_add_rmsnorm
        _fused_add_rmsnorm(
            hidden, residual, self.final_norm, self.cfg.rms_norm_eps
        )

        if M > 1 and not return_all_logits:
            hidden = hidden[-1:, :]
        logits = _linear(hidden, self.lm_head_data)
        logits = logits.float()

        if TIMING.diag and diag:
            last_logits = logits[-1]
            topk_vals, topk_ids = last_logits.topk(5)
            tok_strs = []
            if self.tokenizer:
                for tid in topk_ids.tolist():
                    tok_strs.append(repr(self.tokenizer.decode([tid])))
            logger.info("DIAG[%d] logits: std=%.2f top5=%s",
                        self._diag_count, last_logits.std(),
                        list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))

        if timing:
            torch.cuda.synchronize()
            t_fwd_end = time.perf_counter()
            total_ms = (t_fwd_end - t_fwd_start) * 1000
            layers_ms = (t_after_layers - t_fwd_start) * 1000
            post_ms = (t_fwd_end - t_after_layers) * 1000
            attn_ms = _t_attn_total * 1000
            moe_ms = _t_moe_total * 1000
            xfer_ms = _t_xfer_total * 1000
            dma_ms = _t_dma_total * 1000
            logger.info(
                "DECODE-TOKEN: total=%.1fms (layers=%.1fms post=%.1fms)",
                total_ms, layers_ms, post_ms,
            )
            la_ms = _t_lin_attn_total * 1000
            gqa_ms = _t_gqa_attn_total * 1000
            dma_str = f" attn_dma=%.1fms" % dma_ms if dma_ms > 0 else ""
            la_per = la_ms / _t_lin_attn_layers if _t_lin_attn_layers else 0
            gqa_per = gqa_ms / _t_gqa_layers if _t_gqa_layers else 0
            moe_per = moe_ms / _t_moe_layers if _t_moe_layers else 0
            logger.info(
                "  BREAKDOWN: attn=%.1fms [LA=%.1fms (%d×%.1f) GQA=%.1fms (%d×%.1f)] "
                "moe=%.1fms (%d×%.1f) xfer=%.1fms unified=%d%s",
                attn_ms, la_ms, _t_lin_attn_layers, la_per,
                gqa_ms, _t_gqa_layers, gqa_per,
                moe_ms, _t_moe_layers, moe_per,
                xfer_ms, _t_unified_layers, dma_str,
            )
            if layers_ms > 0:
                dma_pct = f" dma=%.0f%%" % (dma_ms / total_ms * 100) if dma_ms > 0 else ""
                logger.info(
                    "  PERCENT: LA=%.0f%% GQA=%.0f%% moe=%.0f%% xfer=%.0f%% post=%.0f%%%s unaccounted=%.0f%%",
                    la_ms / total_ms * 100,
                    gqa_ms / total_ms * 100,
                    moe_ms / total_ms * 100,
                    xfer_ms / total_ms * 100,
                    post_ms / total_ms * 100,
                    dma_pct,
                    (total_ms - attn_ms - moe_ms - xfer_ms - dma_ms - post_ms) / total_ms * 100,
                )

        return logits

    def _forward_prefill_with_oom_retry(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_states: List[SequenceKVState],
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        """Wrapper around forward_prefill_layer_grouped with OOM recovery.

        On CUDA OOM:
        1. Halve the max chunk size (forces smaller chunks on retry)
        2. Free CUDA graphs if present (reclaims ~1 GB)
        3. Reset KV state and retry
        4. If minimum chunk still OOMs, raise the error
        """
        max_chunk_override = None
        max_retries = 0 if not getattr(self, '_oom_retry_enabled', True) else 3
        freed_graphs = False

        # Defragment the PyTorch allocator before prefill.  Decode leaves many
        # small cached-but-free blocks that prevent contiguous DMA allocations.
        # This reclaims those blocks so layer group DMA can allocate freely.
        torch.cuda.empty_cache()

        for attempt in range(max_retries + 1):
            try:
                result = self.forward_prefill_layer_grouped(
                    token_ids, positions, seq_states,
                    max_chunk_override=max_chunk_override,
                    return_all_logits=return_all_logits,
                )

                # Re-capture CUDA graphs if we freed them during OOM recovery.
                # Prefill is done so the activation VRAM is free again.
                if freed_graphs:
                    for manager in self.gpu_prefill_managers.values():
                        if manager._hcs_initialized and (manager._hcs_buffers or manager._hcs_devices):
                            try:
                                torch.cuda.empty_cache()
                                if len(manager._hcs_devices) > 1:
                                    manager._init_cuda_graphs_multi_gpu()
                                else:
                                    manager._init_cuda_graphs()
                                logger.info("CUDA graphs re-captured after OOM recovery")
                            except torch.OutOfMemoryError as graph_e:
                                raise RuntimeError(
                                    "Could not re-capture CUDA graphs after OOM recovery. "
                                    "Decode requires CUDA graphs for acceptable performance. "
                                    "Reduce KV cache size or expert cache budget to free VRAM."
                                ) from graph_e

                return result
            except torch.OutOfMemoryError:
                if attempt >= max_retries:
                    raise

                # Determine new chunk size
                old_chunk = max_chunk_override or token_ids.shape[0]
                new_chunk = max(128, old_chunk // 2)

                if new_chunk == max_chunk_override:
                    # Already at minimum, can't shrink further
                    raise

                logger.warning(
                    "Prefill OOM (attempt %d/%d): reducing chunk_size %d → %d",
                    attempt + 1, max_retries, old_chunk, new_chunk,
                )

                # Free CUDA intermediates
                torch.cuda.empty_cache()

                # Free CUDA graphs if present (reclaims ~1 GB)
                if not freed_graphs:
                    for manager in self.gpu_prefill_managers.values():
                        # Legacy single-GPU graphs
                        if manager._hcs_cuda_graphs:
                            logger.warning(
                                "Freeing %d CUDA graphs to reclaim VRAM for prefill",
                                len(manager._hcs_cuda_graphs),
                            )
                            manager._hcs_cuda_graphs.clear()
                            manager._hcs_graph_io.clear()
                            manager._hcs_cuda_graphs_enabled = False
                            freed_graphs = True
                        # Multi-GPU per-device graphs
                        for hcs_dev in getattr(manager, '_hcs_devices', []):
                            if hcs_dev.cuda_graphs:
                                logger.warning(
                                    "Freeing %d multi-GPU CUDA graphs on %s for prefill",
                                    len(hcs_dev.cuda_graphs), hcs_dev.device,
                                )
                                hcs_dev.cuda_graphs.clear()
                                if hcs_dev.graph_io:
                                    hcs_dev.graph_io.clear()
                                manager._hcs_cuda_graphs_enabled = False
                                freed_graphs = True
                        if freed_graphs:
                            torch.cuda.empty_cache()

                # Reset KV state for retry (prefill will re-fill from scratch)
                for seq_state in seq_states:
                    if seq_state is not None:
                        seq_state.seq_len = 0

                max_chunk_override = new_chunk

    def forward_prefill_layer_grouped(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_states: List[SequenceKVState],
        max_chunk_override: int = None,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        """GPU prefill with layer-grouped expert loading.

        Loop structure (group-outer, chunk-inner):
            for each group → DMA experts once → for each chunk → for each layer → compute

        Streaming attention: ALL attention runs on GPU0. Expert Parallelism
        splits MoE experts across all GPUs for parallel compute. No cross-GPU
        hidden state transfers during attention (only for EP MoE dispatch).
        """
        assert self._loaded, "Model not loaded. Call load() first."
        M = token_ids.shape[0]
        num_gpus = len(self.all_devices)
        first_k = self.cfg.first_k_dense_replace
        # EP disabled: prefill runs entirely on primary GPU (GPU0).
        # Aux GPUs are decode-only (HCS layer split).  The decomposed EP path
        # (separate attn/routing/expert dispatch + cross-GPU transfers) adds
        # overhead with no benefit on PCIe — single layer.forward() is faster.
        use_ep = False

        # ── Embedding (GPU0) ──
        first_dev = self.all_devices[0]
        all_hidden = self.embedding[token_ids.to(first_dev)]

        # ── Pre-allocate KV pages for all tokens ──
        for seq_state in seq_states:
            if seq_state is not None:
                seq_state.ensure_capacity(M)

        # ── Token chunking ──
        chunk_size = min(M, max_chunk_override or 5000)
        num_chunks = ceil(M / chunk_size)

        chunk_hidden = []
        chunk_positions = []
        chunk_residual: List[Optional[torch.Tensor]] = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, M)
            chunk_hidden.append(all_hidden[start:end])
            chunk_positions.append(positions[start:end])
            chunk_residual.append(None)
        del all_hidden

        # ── Layer group computation ──
        # layer_group_size = how many MoE layers to load at once.
        # Convert to divisor for _build_layer_groups: divisor = ceil(N / group_size).
        first_mgr = next(iter(self.gpu_prefill_managers.values()), None)
        first_k = self.cfg.first_k_dense_replace
        rank = self.ranks[0]
        num_moe = sum(1 for l in range(rank.layer_start, rank.layer_end) if l >= first_k)
        group_size = max(1, self.layer_group_size)
        effective_divisor = max(1, -(-num_moe // group_size))  # ceil division

        # Use rank 0 as the reference for groups (PP=1: all layers in one rank).
        # Streaming attention: all layers on GPU0, groups for expert DMA chunking.
        rank = self.ranks[0]
        dev = torch.device(rank.device)  # initial device (GPU0)

        # Collect all managers ordered by device index
        all_managers = []
        for d in self.all_devices:
            mgr = self.gpu_prefill_managers.get(str(d))
            if mgr is not None:
                all_managers.append(mgr)

        manager = all_managers[0] if all_managers else None

        # Single-GPU: manager must load ALL experts, not just its EP slice.
        if not use_ep and manager and manager.num_local_experts < self.cfg.n_routed_experts:
            _saved_ep = (manager.expert_start, manager.expert_end, manager.num_local_experts)
            manager.expert_start = 0
            manager.expert_end = self.cfg.n_routed_experts
            manager.num_local_experts = self.cfg.n_routed_experts
            manager._dma_bufs_initialized = False
        else:
            _saved_ep = None

        groups = _compute_layer_groups(rank, self.cfg, effective_divisor)

        is_active_only = manager and manager._prefill_mode in ("active_only", "lru")

        # Thread pool for DMA operations (parallel EP loads + pipelined prefetch)
        _dma_managers = all_managers if use_ep else ([manager] if manager else [])
        dma_pool = ThreadPoolExecutor(max_workers=max(len(_dma_managers), 1)) if _dma_managers else None

        # Per-GPU CUDA streams for EP compute.
        # GPU0 runs attention on default stream, EP streams handle MoE dispatch.
        ep_streams = {}
        if use_ep:
            for mgr in all_managers:
                ep_streams[mgr.device] = torch.cuda.Stream(device=mgr.device)

        # ── Async EP transfer: pinned CPU bounce buffers + copy stream ──
        # When P2P is broken, cross-GPU transfers go through CPU.  Without
        # pinned buffers + a dedicated copy stream, h.cpu() blocks the host
        # thread until GPU0's default stream drains (serializing GPU0 forward
        # and GPU1 forward).  The copy stream runs D2H in parallel with GPU0's
        # forward, and event-based sync eliminates host-blocking synchronize().
        _ep_copy_stream = None
        _ep_pinned_h = None
        _ep_pinned_ids = None
        _ep_pinned_wts = None
        _ep_pinned_outs = {}       # device -> pinned output buffer
        _ep_input_read_event = None  # GPU1 done reading pinned input
        _ep_has_secondary = use_ep and any(m.device != dev for m in all_managers)
        if _ep_has_secondary and not _check_p2p():
            _ep_copy_stream = torch.cuda.Stream(device=dev)

        # ── Lightweight per-group timing (DMA vs compute breakdown) ──
        _layer_timing = use_ep and os.environ.get("KRASIS_LAYER_TIMING") == "1"
        if _layer_timing:
            _lt_dma_total = 0.0
            _lt_compute_total = 0.0
            _lt_free_total = 0.0
            _lt_count = 0

        # ── EP timing instrumentation (gated on TIMING.prefill) ──
        ep_timing = TIMING.prefill and use_ep
        if ep_timing:
            _ep_times = {
                "dma_preload": 0.0,
                "attention": 0.0,
                "attention_gqa": 0.0,
                "attention_linear": 0.0,
                "routing": 0.0,
                "shared_expert": 0.0,
                "output_transfer": 0.0,
                "free": 0.0,
                "layer_total": 0.0,
            }
            # Per-GPU timing entries
            for _gi, _mgr in enumerate(all_managers):
                _ep_times[f"gpu{_gi}_forward"] = 0.0
                if ep_streams.get(_mgr.device) is not None:
                    _ep_times[f"gpu{_gi}_input_copy"] = 0.0
            _ep_layer_count = 0
            _ep_dense_count = 0
            _ep_dense_time = 0.0

        # ── DMA pipelining: overlap next group's DMA with current compute ──
        # Disabled during active_only mode and validation sync.
        # EP timing now uses CUDA events (no sync overhead), so pipelining stays enabled.
        _validation = getattr(self, '_validation_sync', False)
        _no_pipeline = os.environ.get("KRASIS_NO_PIPELINE", "") == "1"
        _pipeline_enabled = bool(_dma_managers) and not is_active_only and not _validation and not _no_pipeline
        _has_prefetch = False
        _prefetch_futures = []
        if _pipeline_enabled:
            logger.info("DMA pipelining ENABLED (%d managers, %d groups)",
                        len(_dma_managers), len(groups))

        # Use heavy attention streaming (new TransformerLayer per group) only
        # if attention is offloaded AND decode streaming is NOT active.
        # When _stream_attn_enabled, the lightweight per-layer DMA handles attention.
        _attn_streaming = getattr(self, '_attn_offloaded', False) and not self._stream_attn_enabled

        # Double-buffered streaming attention prefill state
        if self._stream_attn_enabled:
            self._prefill_prefetch_started = False
            self._prefill_prefetch_buf = -1
            # Pre-load first layer into buf 0
            first_group_layers = groups[0][0] if groups else []
            if first_group_layers:
                self._stream_attn_load(first_group_layers[0], buf_idx=first_group_layers[0] % 2)

        for group_idx, (group_layers, group_moe_indices) in enumerate(groups):
            need_load = group_moe_indices and not is_active_only

            # 0. Load attention weights for this group (streaming mode)
            if _attn_streaming:
                self._load_attention_group(group_layers)

            # 1. Ensure current group's experts are loaded
            if need_load:
                if _layer_timing:
                    _lt_dma0 = time.perf_counter()

                if _has_prefetch:
                    # Wait for async prefetch started in previous iteration
                    for f in _prefetch_futures:
                        f.result()  # propagate exceptions
                    _prefetch_futures = []
                    for mgr in _dma_managers:
                        mgr.swap_prefetch()
                    _has_prefetch = False
                else:
                    # Synchronous load (first group, or pipelining disabled)
                    if ep_timing:
                        for d in self.all_devices:
                            torch.cuda.synchronize(d)
                        _t_dma0 = time.perf_counter()

                    futures = [dma_pool.submit(mgr.preload_layer_group, group_moe_indices)
                               for mgr in _dma_managers]
                    for f in futures:
                        f.result()  # propagate exceptions

                    if ep_timing and group_moe_indices:
                        for d in self.all_devices:
                            torch.cuda.synchronize(d)
                        _ep_times["dma_preload"] += time.perf_counter() - _t_dma0

                if _layer_timing:
                    _lt_dma_total += time.perf_counter() - _lt_dma0

            # 2. Start async prefetch for next group (overlaps with compute below)
            if _pipeline_enabled and need_load:
                next_idx = group_idx + 1
                next_moe = groups[next_idx][1] if next_idx < len(groups) else None
                if next_moe:
                    _prefetch_futures = [dma_pool.submit(mgr.preload_layer_group_async, next_moe)
                                         for mgr in _dma_managers]
                    _has_prefetch = True

            # Reset seq_state for this group (all attention on GPU0)
            if seq_states[0] is not None:
                seq_states[0].seq_len = 0
            if _layer_timing and need_load:
                _lt_compute0 = time.perf_counter()

            # 3. Process all chunks through this group's layers
            for c in range(num_chunks):
                # Use _to_device (CPU bounce) for cross-GPU transfers.
                # PyTorch's .to() with broken P2P silently produces garbage
                # data on the target device (proven by positions corruption).
                h = _to_device(chunk_hidden[c], dev)
                r = chunk_residual[c]
                if r is not None:
                    r = _to_device(r, dev)
                pos = _to_device(chunk_positions[c], dev)
                chunk_M = h.shape[0]

                for li, abs_layer_idx in enumerate(group_layers):
                    layer = self.layers[abs_layer_idx]
                    moe_layer_idx = self._abs_to_moe_idx.get(abs_layer_idx)
                    kv_layer_offset = self._kv_layer_offsets.get(abs_layer_idx, 0)

                    # Streaming attention: double-buffered load
                    if self._stream_attn_enabled:
                        buf_idx = abs_layer_idx % 2
                        if c == 0:
                            # First chunk: sync prefetch or synchronous load
                            if hasattr(self, '_prefill_prefetch_started') and self._prefill_prefetch_started and self._prefill_prefetch_buf == buf_idx:
                                self._stream_attn_sync_prefetch(abs_layer_idx, buf_idx)
                                self._prefill_prefetch_started = False
                            elif self._stream_attn_loaded.get(buf_idx) != abs_layer_idx:
                                self._stream_attn_load(abs_layer_idx, buf_idx)

                            # Prefetch next layer within group (safe: uses opposite buffer)
                            next_li = li + 1
                            if next_li < len(group_layers):
                                next_layer = group_layers[next_li]
                                next_buf = next_layer % 2
                                self._stream_attn_prefetch(next_layer, next_buf)
                                self._prefill_prefetch_started = True
                                self._prefill_prefetch_buf = next_buf

                        # Cross-group prefetch: only on the LAST chunk.
                        # Earlier chunks still need the current group's buffers,
                        # and the cross-group prefetch would overwrite them.
                        is_last_layer_in_group = (li == len(group_layers) - 1)
                        is_last_chunk = (c == num_chunks - 1)
                        if is_last_layer_in_group and is_last_chunk:
                            next_group_idx = group_idx + 1
                            if next_group_idx < len(groups):
                                next_layer = groups[next_group_idx][0][0]
                                next_buf = next_layer % 2
                                self._stream_attn_prefetch(next_layer, next_buf)
                                self._prefill_prefetch_started = True
                                self._prefill_prefetch_buf = next_buf

                    # Streaming attention: all layers on GPU0, no cross-GPU transfers
                    kv_cache = self.kv_caches[0]
                    seq_state = seq_states[0]

                    # Dense or linear attention layers, or single-GPU: use standard path
                    if not use_ep or moe_layer_idx is None:
                        if ep_timing:
                            torch.cuda.synchronize(dev)
                            _t_dense0 = time.perf_counter()
                        if kv_layer_offset < 0:
                            h, r = layer.forward(
                                h, r, pos,
                                None, None, -1,
                                moe_layer_idx, num_new_tokens=chunk_M,
                            )
                        else:
                            h, r = layer.forward(
                                h, r, pos,
                                kv_cache, seq_state, kv_layer_offset,
                                moe_layer_idx, num_new_tokens=chunk_M,
                            )
                        if ep_timing:
                            torch.cuda.synchronize(dev)
                            _ep_dense_time += time.perf_counter() - _t_dense0
                            _ep_dense_count += 1
                        # Per-layer sync for validation: catch CUDA errors precisely
                        if getattr(self, '_validation_sync', False):
                            try:
                                torch.cuda.synchronize(dev)
                            except Exception as e:
                                logger.error(
                                    "CUDA error at layer %d (moe_idx=%s, chunk=%d/%d, M=%d): %s",
                                    abs_layer_idx, moe_layer_idx, c+1, num_chunks, chunk_M, e,
                                )
                                raise
                        continue

                    # ── Multi-GPU EP path for MoE layers ──
                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _t_layer0 = time.perf_counter()

                    # ── Attention ──
                    if kv_layer_offset < 0:
                        h, r = layer.forward_attn(
                            h, r, pos, None, None, -1,
                            num_new_tokens=chunk_M,
                        )
                    else:
                        h, r = layer.forward_attn(
                            h, r, pos, kv_cache, seq_state, kv_layer_offset,
                            num_new_tokens=chunk_M,
                        )

                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _t_attn = time.perf_counter()
                        _attn_dt = _t_attn - _t_layer0
                        _ep_times["attention"] += _attn_dt
                        dev_key = f"attn_{dev}"
                        if dev_key not in _ep_times:
                            _ep_times[dev_key] = 0.0
                            _ep_times[f"attn_{dev}_count"] = 0
                        _ep_times[dev_key] += _attn_dt
                        _ep_times[f"attn_{dev}_count"] += 1
                        if layer.layer_type == "linear_attention":
                            _ep_times["attention_linear"] += _attn_dt
                        else:
                            _ep_times["attention_gqa"] += _attn_dt

                    # (b) Routing on GPU 0 (tiny: gate matmul)
                    topk_ids, topk_weights = layer.compute_routing(h)

                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _t_route = time.perf_counter()
                        _ep_times["routing"] += _t_route - _t_attn

                    # (c) Shared expert on layer_dev (async overlap with routed experts)
                    shared_output = None
                    has_shared_overlap = (layer._shared_stream is not None
                                          and layer.shared_expert is not None)
                    if has_shared_overlap:
                        layer._shared_stream.wait_stream(torch.cuda.current_stream(dev))
                        with torch.cuda.stream(layer._shared_stream):
                            shared_output = layer._shared_expert_forward(h)
                    elif layer.shared_expert is not None:
                        shared_output = layer._shared_expert_forward(h)

                    if ep_timing:
                        _t_shared = time.perf_counter()

                    # (d) Routed experts — ALWAYS parallel, with CUDA event timing
                    h_ready = torch.cuda.current_stream(dev).record_event()

                    # CUDA events for per-GPU kernel timing (no sync required)
                    if ep_timing:
                        _gpu_events = {}  # gpu_idx -> (start_event, end_event, device)

                    # ── Async pinned bounce: start D2H on copy stream ──
                    # The copy stream runs D2H in parallel with GPU0's forward
                    # on the default stream.  Without this, h.cpu() blocks the
                    # host until GPU0's default stream drains (serializing the
                    # two GPU forwards).
                    _ep_copy_done = None
                    if _ep_copy_stream is not None:
                        # Lazy-allocate pinned buffers on first use
                        _M = h.shape[0]
                        if _ep_pinned_h is None or _ep_pinned_h.shape[0] < _M:
                            _ep_pinned_h = torch.empty(
                                _M, h.shape[1], dtype=h.dtype,
                                device='cpu', pin_memory=True)
                            _ep_pinned_ids = torch.empty(
                                _M, topk_ids.shape[1], dtype=topk_ids.dtype,
                                device='cpu', pin_memory=True)
                            _ep_pinned_wts = torch.empty(
                                _M, topk_weights.shape[1], dtype=topk_weights.dtype,
                                device='cpu', pin_memory=True)
                        # Wait for any previous GPU1 H2D reads to finish
                        # before overwriting the pinned buffer.
                        if _ep_input_read_event is not None:
                            _ep_copy_stream.wait_event(_ep_input_read_event)
                        _ep_copy_stream.wait_event(h_ready)
                        with torch.cuda.stream(_ep_copy_stream):
                            _ep_pinned_h[:_M].copy_(h, non_blocking=True)
                            _ep_pinned_ids[:_M].copy_(topk_ids, non_blocking=True)
                            _ep_pinned_wts[:_M].copy_(topk_weights, non_blocking=True)
                        _ep_copy_done = _ep_copy_stream.record_event()

                    partial_outputs = []
                    for _gi, mgr in enumerate(all_managers):
                        if mgr.device == dev:
                            if ep_timing:
                                _ev_start = torch.cuda.Event(enable_timing=True)
                                _ev_end = torch.cuda.Event(enable_timing=True)
                                _ev_start.record()
                            out = mgr.forward(moe_layer_idx, h, topk_ids, topk_weights,
                                              routed_only=True)
                            if ep_timing:
                                _ev_end.record()
                                _gpu_events[_gi] = (_ev_start, _ev_end, dev)
                            partial_outputs.append((mgr.device, out))
                        elif _ep_copy_stream is not None:
                            # ── Async path: copy from pinned buffer ──
                            stream = ep_streams[mgr.device]
                            stream.wait_event(_ep_copy_done)
                            with torch.cuda.stream(stream):
                                if ep_timing:
                                    _ev_start = torch.cuda.Event(enable_timing=True)
                                    _ev_end = torch.cuda.Event(enable_timing=True)
                                    _ev_start.record(stream)
                                _M = h.shape[0]
                                h_dev = _ep_pinned_h[:_M].to(mgr.device, non_blocking=True)
                                ids_dev = _ep_pinned_ids[:_M].to(mgr.device, non_blocking=True)
                                wts_dev = _ep_pinned_wts[:_M].to(mgr.device, non_blocking=True)
                                # Mark pinned input buffer as consumed
                                _ep_input_read_event = stream.record_event()
                                out = mgr.forward(moe_layer_idx, h_dev, ids_dev, wts_dev,
                                                  routed_only=True)
                                if ep_timing:
                                    _ev_end.record(stream)
                                    _gpu_events[_gi] = (_ev_start, _ev_end, mgr.device)
                                partial_outputs.append((mgr.device, out))
                        else:
                            # ── Fallback: synchronous _to_device (P2P works) ──
                            stream = ep_streams[mgr.device]
                            stream.wait_event(h_ready)
                            with torch.cuda.stream(stream):
                                if ep_timing:
                                    _ev_start = torch.cuda.Event(enable_timing=True)
                                    _ev_end = torch.cuda.Event(enable_timing=True)
                                    _ev_start.record(stream)
                                h_dev = _to_device(h, mgr.device)
                                ids_dev = _to_device(topk_ids, mgr.device)
                                wts_dev = _to_device(topk_weights, mgr.device)
                                out = mgr.forward(moe_layer_idx, h_dev, ids_dev, wts_dev,
                                                  routed_only=True)
                                if ep_timing:
                                    _ev_end.record(stream)
                                    _gpu_events[_gi] = (_ev_start, _ev_end, mgr.device)
                                partial_outputs.append((mgr.device, out))

                    if ep_timing:
                        _t_dispatch_done = time.perf_counter()

                    if has_shared_overlap:
                        torch.cuda.current_stream(dev).wait_stream(layer._shared_stream)

                    routed_output = None
                    for po_device, po_out in partial_outputs:
                        if po_device == dev:
                            routed_output = po_out
                        elif _ep_copy_stream is not None:
                            # ── Async output gather via pinned buffer ──
                            _M = po_out.shape[0]
                            if po_device not in _ep_pinned_outs or _ep_pinned_outs[po_device].shape[0] < _M:
                                _ep_pinned_outs[po_device] = torch.empty(
                                    _M, po_out.shape[1], dtype=po_out.dtype,
                                    device='cpu', pin_memory=True)
                            po_pinned = _ep_pinned_outs[po_device]
                            # D2H on secondary GPU's ep_stream (sequenced after forward)
                            with torch.cuda.stream(ep_streams[po_device]):
                                po_pinned[:_M].copy_(po_out, non_blocking=True)
                            out_ready = ep_streams[po_device].record_event()
                            # H2D on GPU0's default stream (wait for D2H via event)
                            torch.cuda.current_stream(dev).wait_event(out_ready)
                            out_on_primary = po_pinned[:_M].to(dev, non_blocking=True)
                            if routed_output is None:
                                routed_output = out_on_primary
                            else:
                                routed_output = routed_output + out_on_primary
                        else:
                            ep_streams[po_device].synchronize()
                            out_on_primary = _to_device(po_out, dev)
                            if routed_output is None:
                                routed_output = out_on_primary
                            else:
                                routed_output = routed_output + out_on_primary

                    if ep_timing:
                        # Sync all devices to ensure kernels have completed
                        for _sync_dev in self.all_devices:
                            try:
                                torch.cuda.synchronize(_sync_dev)
                            except Exception as _sync_err:
                                logger.warning("EP timing: sync %s failed: %s", _sync_dev, _sync_err)
                        _t_sync_done = time.perf_counter()
                        # Collect CUDA event timings (GPU kernel times)
                        for _gi, (_ev_s, _ev_e, _ev_dev) in _gpu_events.items():
                            try:
                                _ev_e.synchronize()
                                _gpu_ms = _ev_s.elapsed_time(_ev_e)
                                _ep_times[f"gpu{_gi}_forward"] += _gpu_ms / 1000.0
                            except Exception as _ev_err:
                                logger.warning("EP timing: gpu%d event error: %s", _gi, _ev_err)
                        _ep_times["shared_expert"] += (_t_shared - _t_route) if has_shared_overlap else 0
                        _ep_times["output_transfer"] += _t_sync_done - _t_dispatch_done
                        _ep_layer_count += 1

                    # (f) Apply routed_scaling_factor + shared expert on GPU 0
                    rsf = self.cfg.routed_scaling_factor
                    if rsf != 1.0:
                        routed_output *= rsf
                    if shared_output is not None:
                        h = routed_output + shared_output
                    else:
                        h = routed_output

                    if ep_timing:
                        try:
                            for _sync_dev in self.all_devices:
                                torch.cuda.synchronize(_sync_dev)
                        except Exception:
                            pass
                        _ep_times["layer_total"] += time.perf_counter() - _t_layer0

                # Save for next group
                chunk_hidden[c] = h
                chunk_residual[c] = r

                # Advance KV seq_state (all attention on GPU0 → single cache)
                if seq_states[0] is not None:
                    seq_states[0].advance(chunk_M)
            # 4. Free experts for this group
            if _layer_timing and need_load:
                torch.cuda.synchronize(dev)
                _lt_compute_total += time.perf_counter() - _lt_compute0

            if _layer_timing and need_load:
                _lt_free0 = time.perf_counter()

            if need_load:
                _clear_cache = True

                # Sync default stream before freeing: empty_cache() can return
                # blocks to CUDA that pending kernels still reference.
                torch.cuda.synchronize(dev)

                if ep_timing:
                    for d in self.all_devices:
                        torch.cuda.synchronize(d)
                    _t_free0 = time.perf_counter()

                for mgr in _dma_managers:
                    mgr.free_layer_group(clear_cache=_clear_cache)

                if ep_timing and group_moe_indices:
                    for d in self.all_devices:
                        torch.cuda.synchronize(d)
                    _ep_times["free"] += time.perf_counter() - _t_free0

            if _layer_timing and need_load:
                _lt_free_total += time.perf_counter() - _lt_free0
                _lt_count += 1

            # 5. Free attention weights for this group (streaming mode)
            if _attn_streaming:
                self._free_attention_group(group_layers)

        # Reload all attention weights for decode (streaming mode)
        # Skip reload if streaming decode is enabled — weights stay on CPU
        if _attn_streaming and not self._stream_attn_enabled:
            self._reload_all_attention()

        # Restore EP slicing state (single-GPU only)
        if _saved_ep is not None:
            manager.expert_start, manager.expert_end, manager.num_local_experts = _saved_ep
            manager._dma_bufs_initialized = False

        if dma_pool is not None:
            dma_pool.shutdown(wait=False)

        # ── EP timing summary ──
        if ep_timing:
            _total_measured = sum(v for k, v in _ep_times.items()
                                 if k not in ("layer_total", "attention_gqa", "attention_linear"))
            logger.info("EP TIMING BREAKDOWN (%d MoE layers, %d dense layers, %d chunks)",
                        _ep_layer_count, _ep_dense_count, num_chunks)
            _skip_phases = {"layer_total", "attention_gqa", "attention_linear"}
            # Also skip per-device attn_* keys (shown separately below)
            _skip_phases.update(k for k in _ep_times if k.startswith("attn_"))
            for phase, t in _ep_times.items():
                if phase in _skip_phases:
                    continue
                per_layer = (t / _ep_layer_count * 1000) if _ep_layer_count > 0 else 0
                pct = (t / _ep_times["layer_total"] * 100) if _ep_times["layer_total"] > 0 else 0
                logger.info("  %-20s %8.1f ms total  %6.2f ms/layer  %5.1f%%",
                            phase, t * 1000, per_layer, pct)
            # Per-layer-type attention averages (multiply by num_chunks for layer-chunks)
            num_gqa_chunks = (self.cfg.num_full_attention_layers if self.cfg.is_hybrid else _ep_layer_count) * num_chunks
            num_linear_chunks = _ep_layer_count - num_gqa_chunks
            if num_gqa_chunks > 0:
                logger.info("  %-20s %6.2f ms/chunk  (%d layer-chunks)",
                            "  avg GQA attn", _ep_times["attention_gqa"] / num_gqa_chunks * 1000, num_gqa_chunks)
            if num_linear_chunks > 0:
                logger.info("  %-20s %6.2f ms/chunk  (%d layer-chunks)",
                            "  avg linear attn", _ep_times["attention_linear"] / num_linear_chunks * 1000, num_linear_chunks)
            # Per-device attention averages
            for d in self.all_devices:
                dk = f"attn_{d}"
                ck = f"attn_{d}_count"
                if dk in _ep_times and _ep_times[ck] > 0:
                    logger.info("  %-20s %6.2f ms/chunk  (%d chunks, %.0f ms total)",
                                f"  attn on {d}",
                                _ep_times[dk] / _ep_times[ck] * 1000,
                                int(_ep_times[ck]),
                                _ep_times[dk] * 1000)
            logger.info("  %-20s %8.1f ms total  %6.2f ms/layer",
                        "LAYER TOTAL", _ep_times["layer_total"] * 1000,
                        _ep_times["layer_total"] / _ep_layer_count * 1000 if _ep_layer_count else 0)
            logger.info("  %-20s %8.1f ms total  %6.2f ms/layer",
                        "DENSE LAYERS", _ep_dense_time * 1000,
                        _ep_dense_time / _ep_dense_count * 1000 if _ep_dense_count else 0)
            logger.info("  %-20s %8.1f ms total",
                        "SUM (measured)", _total_measured * 1000)

        # ── Layer timing summary (DMA vs compute vs free) ──
        if _layer_timing and _lt_count > 0:
            _lt_total = _lt_dma_total + _lt_compute_total + _lt_free_total
            logger.info("LAYER TIMING (%d groups, %d chunks)", _lt_count, num_chunks)
            logger.info("  DMA load:  %8.1f ms total  %6.2f ms/group  %5.1f%%",
                        _lt_dma_total * 1000, _lt_dma_total / _lt_count * 1000,
                        _lt_dma_total / _lt_total * 100 if _lt_total > 0 else 0)
            logger.info("  Compute:   %8.1f ms total  %6.2f ms/group  %5.1f%%",
                        _lt_compute_total * 1000, _lt_compute_total / _lt_count * 1000,
                        _lt_compute_total / _lt_total * 100 if _lt_total > 0 else 0)
            logger.info("  Free:      %8.1f ms total  %6.2f ms/group  %5.1f%%",
                        _lt_free_total * 1000, _lt_free_total / _lt_count * 1000,
                        _lt_free_total / _lt_total * 100 if _lt_total > 0 else 0)
            logger.info("  TOTAL:     %8.1f ms", _lt_total * 1000)
            per_chunk = _lt_compute_total / (_lt_count * num_chunks) * 1000
            logger.info("  Compute per chunk: %.2f ms", per_chunk)

        # ── Final result: logits ──
        # final_norm and lm_head are always on GPU0 (first device)
        from krasis.layer import _fused_add_rmsnorm
        first_dev = self.all_devices[0]

        if return_all_logits:
            # Concatenate ALL chunk hidden/residual states for perplexity measurement
            all_h = torch.cat(chunk_hidden, dim=0)      # [M, hidden_size]
            all_r = torch.cat(chunk_residual, dim=0)     # [M, hidden_size]
            all_h = _to_device(all_h, first_dev)
            all_r = _to_device(all_r, first_dev)
            _fused_add_rmsnorm(
                all_h, all_r, self.final_norm, self.cfg.rms_norm_eps
            )
            final_logits = _linear(all_h, self.lm_head_data).float()
        else:
            # Standard path: only last token's logits (for generation)
            last_h = chunk_hidden[-1][-1:]
            last_r = chunk_residual[-1][-1:]
            last_h = _to_device(last_h, first_dev)
            last_r = _to_device(last_r, first_dev)
            _fused_add_rmsnorm(
                last_h, last_r, self.final_norm, self.cfg.rms_norm_eps
            )
            final_logits = _linear(last_h, self.lm_head_data).float()

        return final_logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_token_ids: Optional[List[int]] = None,
        presence_penalty: float = 0.0,
    ) -> List[int]:
        """Generate tokens autoregressively.

        Args:
            prompt_tokens: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            stop_token_ids: Stop generation on these tokens
            presence_penalty: Penalty for already-generated tokens (0 = disabled)

        Returns:
            List of generated token IDs (excluding prompt)
        """
        if stop_token_ids is None:
            stop_token_ids = [self.cfg.eos_token_id] + list(self.cfg.extra_stop_token_ids)

        # Guard: ensure prompt_tokens is a list of ints
        if isinstance(prompt_tokens, str):
            prompt_tokens = self.tokenizer.encode(prompt_tokens)
        elif isinstance(prompt_tokens, dict):
            prompt_tokens = prompt_tokens["input_ids"]
        elif hasattr(prompt_tokens, "input_ids"):
            prompt_tokens = prompt_tokens.input_ids
        if isinstance(prompt_tokens, list) and prompt_tokens and not isinstance(prompt_tokens[0], int):
            prompt_tokens = [int(t) for t in prompt_tokens]

        # Create per-GPU-split sequence states (one per KV cache / GPU)
        seq_states_per_rank = [
            SequenceKVState(c, seq_id=0) if c is not None else None
            for c in self.kv_caches
        ]

        # Reset linear attention states for new sequence
        if self.cfg.is_hybrid:
            for layer in self.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        # Reset Mamba2 SSM states for new sequence (zero conv_state and ssm_state)
        # Prefill will write fresh final states, but zeroing ensures no stale state leaks
        decode_states = getattr(self, '_mamba2_decode_states', None)
        if decode_states:
            for buffers in decode_states.values():
                buffers['conv_state'].zero_()
                buffers['ssm_state'].zero_()

        device = torch.device(self.ranks[0].device)
        generated = []

        _t_gen_start = time.perf_counter()

        try:
            # Multi-GPU: re-upload prefill-only attention if previously freed
            self._upload_prefill_only_attention()

            # ── Prefill ── (Single-slot AWQ: Marlin already in GPU slots)
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

            logits = self.forward(prompt_tensor, positions, seq_states_per_rank)
            # Take last token's logits
            next_logits = logits[-1:, :]

            # Track generated tokens for presence penalty
            generated_set = set()
            next_token = sample(
                next_logits, temperature, top_k, top_p,
                presence_penalty=presence_penalty,
                generated_tokens=generated_set,
            ).item()
            generated.append(next_token)
            generated_set.add(next_token)

            torch.cuda.synchronize()
            self._last_ttft = time.perf_counter() - _t_gen_start

            if next_token in stop_token_ids:
                return generated

            # ── Decode (GPU) — entire loop in Rust, zero Python per token ──
            gpu_store = getattr(self, '_gpu_decode_store', None)
            if gpu_store is None:
                raise RuntimeError("GPU decode store not configured. Call setup_gpu_decode_store() first.")
            self._export_kv_to_rust(seq_states_per_rank, len(prompt_tokens))
            self._transfer_mamba2_states()
            self._update_la_state_ptrs()
            self._update_la_state_ptrs_aux()

            # Optional Python-side comparison against the first Rust decode step.
            if os.environ.get("KRASIS_TRACE_PY_COMPARE") == "1":
                logger.info(
                    "[KRASIS-TRACE] event=python_compare phase=begin token=%d pos=%d prompt_len=%d",
                    next_token, len(prompt_tokens), len(prompt_tokens),
                )
                # Snapshot recurrent states and KV caches before Python forward
                saved_seq = []
                for rank_states in seq_states_per_rank:
                    rank_snap = []
                    for s in rank_states:
                        if s is not None:
                            rank_snap.append({k: v.clone() if isinstance(v, torch.Tensor) else v
                                              for k, v in s.items()})
                        else:
                            rank_snap.append(None)
                    saved_seq.append(rank_snap)
                # Save KV cache state (FP8 cache positions)
                saved_kv = []
                for kvc in self.kv_caches:
                    if kvc is not None:
                        saved_kv.append((kvc.k_cache.clone(), kvc.v_cache.clone()))
                    else:
                        saved_kv.append(None)
                with torch.no_grad():
                    tok_t = torch.tensor([next_token], dtype=torch.long, device=device)
                    pos_t = torch.tensor([len(prompt_tokens)], dtype=torch.int32, device=device)
                    py_logits = self.forward(tok_t, pos_t, seq_states_per_rank)
                    py_top5 = torch.topk(py_logits[0], 5)
                    top5 = ",".join(
                        f"{py_top5.indices[i].item()}:{py_top5.values[i].item():.6f}"
                        for i in range(5)
                    )
                    lmin, lmax = py_logits[0].min().item(), py_logits[0].max().item()
                    logger.info(
                        "[KRASIS-TRACE] event=python_compare phase=python_step pos=%d token=%d top5=[%s] logit_min=%.6f logit_max=%.6f spread=%.6f",
                        len(prompt_tokens), next_token, top5, lmin, lmax, lmax - lmin,
                    )
                # Restore recurrent states
                for ri, rank_snap in enumerate(saved_seq):
                    for si, snap in enumerate(rank_snap):
                        if snap is not None:
                            for k, v in snap.items():
                                if isinstance(v, torch.Tensor):
                                    seq_states_per_rank[ri][si][k].copy_(v)
                # Restore KV caches
                for i, saved in enumerate(saved_kv):
                    if saved is not None:
                        self.kv_caches[i].k_cache.copy_(saved[0])
                        self.kv_caches[i].v_cache.copy_(saved[1])
                # Re-export state to Rust
                self._update_la_state_ptrs()
                self._update_la_state_ptrs_aux()
                logger.info("[KRASIS-TRACE] event=python_compare phase=restore_complete pos=%d", len(prompt_tokens))

            # Multi-GPU: free prefill-only attention before decode
            self._free_prefill_only_attention()
            # Single-slot AWQ: swap simple INT4 into GPU slots for decode
            gpu_store.swap_to_simple_int4()
            decode_tokens = gpu_store.gpu_generate_batch(
                first_token=next_token,
                start_position=len(prompt_tokens),
                max_tokens=max_new_tokens - 1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_ids=list(stop_token_ids),
                presence_penalty=presence_penalty,
            )
            generated.extend(decode_tokens)

        finally:
            n_decode = len(generated) - 1  # First token is from prefill
            if n_decode > 0:
                self._last_decode_time = gpu_store.last_decode_elapsed_s if gpu_store else 0.0
                self._last_decode_tok_s = n_decode / self._last_decode_time if self._last_decode_time > 0 else 0.0
            else:
                self._last_decode_time = 0.0
                self._last_decode_tok_s = 0.0
            # Single-slot AWQ: restore Marlin into GPU slots for next prefill
            if gpu_store is not None:
                gpu_store.swap_to_marlin()
            # Free KV cache pages
            for s in seq_states_per_rank:
                if s is not None:
                    s.free()

        return generated

    def chat(
        self,
        messages: List[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """Chat completion: format messages, generate, decode."""
        prompt_tokens = self.tokenizer.apply_chat_template(messages)
        logger.info("Prompt: %d tokens", len(prompt_tokens))

        generated = self.generate(
            prompt_tokens, max_new_tokens, temperature, top_k, top_p,
        )
        return self.tokenizer.decode(generated)

    # ──────────────────────────────────────────────────────
    # Rust server interface
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _dense_mlp_tensor(w) -> torch.Tensor:
        """Extract a BF16 tensor from a dense MLP weight value.

        Dense MLP weights are either plain BF16 tensors or (weight_int8, scale)
        tuples from per-channel INT8 quantization.  The Rust decode store needs
        plain BF16, so dequantize on the fly when necessary.
        """
        if isinstance(w, tuple):
            w_int8, scale = w
            return (w_int8.float() * scale.float().unsqueeze(1)).to(torch.bfloat16)
        return w

    def _register_deepseek_v4_hash_tables(
        self,
        store,
        device: torch.device,
        keepalive: list,
        layer_indices=None,
    ) -> None:
        if not self.cfg.is_deepseek_v4:
            return
        selected = range(len(self.layers)) if layer_indices is None else layer_indices
        for layer_idx in selected:
            layer = self.layers[layer_idx]
            table = getattr(layer, "router_tid2eid", None)
            vl_bias = getattr(layer, "router_vl_bias", None)
            if self.cfg.vision_n_layers > 0:
                if vl_bias is None or tuple(vl_bias.shape) != (self.cfg.n_routed_experts,):
                    raise RuntimeError(
                        f"DeepSeek-V4 vision router bias contract failed at layer {layer_idx}: "
                        f"shape={None if vl_bias is None else tuple(vl_bias.shape)}"
                    )
                vl_bias_f32 = vl_bias.to(
                    device=device, dtype=torch.float32, non_blocking=True
                ).contiguous()
                keepalive.append(vl_bias_f32)
                store.set_moe_deepseek_v4_vision_bias(
                    layer_idx=layer_idx,
                    bias_ptr=vl_bias_f32.data_ptr(),
                )
            elif vl_bias is not None:
                raise RuntimeError(
                    f"DeepSeek-V4 text-only layer {layer_idx} unexpectedly has bias_vl"
                )
            if layer_idx < self.cfg.num_hash_layers:
                if table is None:
                    raise RuntimeError(
                        f"DeepSeek-V4 hash layer {layer_idx} has no tid2eid table"
                    )
                expected = (self.cfg.vocab_size, self.cfg.num_experts_per_tok)
                if tuple(table.shape) != expected or table.dtype != torch.int64:
                    raise RuntimeError(
                        f"DeepSeek-V4 hash layer {layer_idx} table contract "
                        f"{tuple(table.shape)}/{table.dtype} != {expected}/torch.int64"
                    )
                minimum = int(table.min().item())
                maximum = int(table.max().item())
                if minimum < 0 or maximum >= self.cfg.n_routed_experts:
                    raise RuntimeError(
                        f"DeepSeek-V4 hash layer {layer_idx} expert range "
                        f"[{minimum}, {maximum}] outside [0, {self.cfg.n_routed_experts})"
                    )
                table_i32 = table.to(
                    device=device, dtype=torch.int32, non_blocking=True
                ).contiguous()
                keepalive.append(table_i32)
                store.set_moe_deepseek_v4_hash_table(
                    layer_idx=layer_idx,
                    table_ptr=table_i32.data_ptr(),
                    vocab_size=self.cfg.vocab_size,
                )
            elif table is not None:
                raise RuntimeError(
                    f"DeepSeek-V4 non-hash layer {layer_idx} unexpectedly has tid2eid"
                )

    @staticmethod
    def _register_sequence_state_tensor(
        store,
        *,
        name: str,
        kind: str,
        layer_idx: int,
        tensor: torch.Tensor,
        logical_tokens_per_row: int = 0,
    ) -> None:
        """Register metadata read from one real sequence-state tensor.

        This runs only during model setup. The registry and all request-time
        inventory/snapshot operations live in Rust.
        """
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise RuntimeError(f"Sequence-state tensor {name} must be a CUDA tensor")
        if tensor.numel() == 0:
            raise RuntimeError(
                f"Sequence-state tensor {name} is empty and has no CUDA allocation"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(
                f"Sequence-state tensor {name} must retain its registered contiguous layout; "
                f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}"
            )
        element_size = int(tensor.element_size())
        storage_bytes = int(tensor.numel()) * element_size
        if logical_tokens_per_row > 0:
            if tensor.dim() < 2:
                raise RuntimeError(
                    f"Token-growing sequence-state tensor {name} must have rank >= 2"
                )
            if tensor.dim() >= 3:
                capacity_rows = int(tensor.shape[0]) * int(tensor.shape[1])
            else:
                capacity_rows = int(tensor.shape[0])
            row_view = tensor.reshape(capacity_rows, -1)
            row_bytes = int(row_view.shape[1]) * element_size
            shape = [int(dim) for dim in row_view.shape]
            strides_bytes = [int(stride) * element_size for stride in row_view.stride()]
            growth_mode = "token_rows"
        else:
            capacity_rows = 0
            row_bytes = 0
            shape = [int(dim) for dim in tensor.shape]
            strides_bytes = [int(stride) * element_size for stride in tensor.stride()]
            growth_mode = "fixed"
        store.register_sequence_state_allocation(
            name=name,
            kind=kind,
            layer_idx=int(layer_idx),
            ptr=int(tensor.data_ptr()),
            storage_bytes=storage_bytes,
            dtype=str(tensor.dtype).removeprefix("torch."),
            element_size=element_size,
            shape=shape,
            strides_bytes=strides_bytes,
            growth_mode=growth_mode,
            logical_tokens_per_row=int(logical_tokens_per_row),
            capacity_rows=capacity_rows,
            row_bytes=row_bytes,
        )

    @staticmethod
    def _sequence_state_plane_layers(value, count: int):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            if value.shape[0] != count:
                raise RuntimeError(
                    f"Sequence-state plane has {value.shape[0]} layers, expected {count}"
                )
            return [value[offset] for offset in range(count)]
        if isinstance(value, list):
            if len(value) != count:
                raise RuntimeError(
                    f"Sequence-state plane has {len(value)} layers, expected {count}"
                )
            return value
        raise RuntimeError(f"Unsupported sequence-state plane type {type(value).__name__}")

    def _register_primary_sequence_state_inventory(self, store, device: torch.device) -> int:
        """Populate the primary store from actual loaded sequence allocations."""
        device_ordinal = int(device.index or 0)

        def on_store(tensor) -> bool:
            return (
                isinstance(tensor, torch.Tensor)
                and tensor.is_cuda
                and int(tensor.device.index or 0) == device_ordinal
            )

        registered = 0

        def register(name, kind, layer_idx, tensor, tokens_per_row=0):
            nonlocal registered
            if tensor is None or tensor.numel() == 0 or not on_store(tensor):
                return
            self._register_sequence_state_tensor(
                store,
                name=name,
                kind=kind,
                layer_idx=layer_idx,
                tensor=tensor,
                logical_tokens_per_row=tokens_per_row,
            )
            registered += 1

        for cache_idx, cache in enumerate(self.kv_caches):
            if cache is None:
                continue
            layer_indices = list(cache.layer_indices)
            layer_count = len(layer_indices)
            if cache.attention_type == "deepseek_v4":
                for offset, layer_idx in enumerate(layer_indices):
                    ratio = int(cache.dsv4_compress_ratios[offset])
                    register(
                        f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.raw",
                        "deepseek_v4_raw_ring",
                        layer_idx,
                        cache.dsv4_raw_cache[offset],
                    )
                    raw_native = cache.dsv4_raw_native[offset]
                    if raw_native is not None:
                        for suffix, tensor in raw_native.items():
                            register(
                                f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.raw_{suffix}",
                                f"deepseek_v4_native_raw_{suffix}",
                                layer_idx,
                                tensor,
                            )
                    if ratio == 0:
                        continue
                    register(
                        f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.compressed",
                        "deepseek_v4_compressed_kv",
                        layer_idx,
                        cache.dsv4_compressed_cache[offset],
                        ratio,
                    )
                    compressed_native = cache.dsv4_compressed_native[offset]
                    if compressed_native is not None:
                        for suffix, tensor in compressed_native.items():
                            register(
                                f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.compressed_{suffix}",
                                f"deepseek_v4_native_compressed_{suffix}",
                                layer_idx,
                                tensor,
                                ratio,
                            )
                    register(
                        f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.compressor_kv",
                        "deepseek_v4_compressor_kv_state",
                        layer_idx,
                        cache.dsv4_compressor_kv_state[offset],
                    )
                    register(
                        f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.compressor_score",
                        "deepseek_v4_compressor_score_state",
                        layer_idx,
                        cache.dsv4_compressor_score_state[offset],
                    )
                    if ratio == 4:
                        register(
                            f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.index",
                            "deepseek_v4_index_cache",
                            layer_idx,
                            cache.dsv4_index_cache[offset],
                            ratio,
                        )
                        index_native = cache.dsv4_index_native[offset]
                        if index_native is not None:
                            for suffix, tensor in index_native.items():
                                register(
                                    f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.index_{suffix}",
                                    f"deepseek_v4_native_index_{suffix}",
                                    layer_idx,
                                    tensor,
                                    ratio,
                                )
                        register(
                            f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.index_kv",
                            "deepseek_v4_index_kv_state",
                            layer_idx,
                            cache.dsv4_index_kv_state[offset],
                        )
                        register(
                            f"gpu{device_ordinal}.layer{layer_idx}.deepseek_v4.index_score",
                            "deepseek_v4_index_score_state",
                            layer_idx,
                            cache.dsv4_index_score_state[offset],
                        )
                continue

            planes = [
                ("k", "gqa_k", cache.k_cache),
                ("v", "gqa_v", cache.v_cache),
                ("k_scale", "gqa_k_scale", cache.k_radius_cache),
                ("k_packed", "gqa_k_packed", cache.k_angles_cache),
                ("v_scale", "gqa_v_scale", cache.v_radius_cache),
                ("v_packed", "gqa_v_packed", cache.v_angles_cache),
                ("compressed", "mla_compressed_kv", cache.ckv_cache),
                ("position", "mla_positional_k", cache.kpe_cache),
                ("combined", "mla_combined_kv", cache.kv_cache),
            ]
            for suffix, kind, plane in planes:
                if plane is None:
                    continue
                for offset, tensor in enumerate(
                    self._sequence_state_plane_layers(plane, layer_count)
                ):
                    if tensor is None:
                        continue
                    layer_idx = layer_indices[offset]
                    register(
                        f"gpu{device_ordinal}.layer{layer_idx}.{suffix}",
                        kind,
                        layer_idx,
                        tensor,
                        1,
                    )

        hqq_active = is_hqq_attention(self.quant_cfg.attention)
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attention
            if layer.layer_type == "linear_attention" and attn is not None:
                if hqq_active:
                    conv_state = getattr(attn, "_hqq_conv_state", None)
                    recur_state = getattr(attn, "_hqq_recur_state", None)
                else:
                    conv_state = getattr(attn, "_rust_conv_state", None)
                    recur_state = getattr(attn, "_rust_recur_state", None)
                register(
                    f"gpu{device_ordinal}.layer{layer_idx}.linear.conv",
                    "linear_attention_conv_state",
                    layer_idx,
                    conv_state,
                )
                register(
                    f"gpu{device_ordinal}.layer{layer_idx}.linear.recurrent",
                    "linear_attention_recurrent_state",
                    layer_idx,
                    recur_state,
                )
            elif layer.layer_type == "mamba2":
                states = getattr(self, "_mamba2_decode_states", {}).get(layer_idx, {})
                register(
                    f"gpu{device_ordinal}.layer{layer_idx}.mamba.conv",
                    "mamba2_conv_state",
                    layer_idx,
                    states.get("conv_state"),
                )
                register(
                    f"gpu{device_ordinal}.layer{layer_idx}.mamba.ssm",
                    "mamba2_ssm_state",
                    layer_idx,
                    states.get("ssm_state"),
                )

        required = int(store.finalize_sequence_state_inventory())
        logger.info(
            "Sequence-state inventory on cuda:%d: registered=%d required=%d",
            device_ordinal,
            registered,
            required,
        )
        return registered

    def setup_gpu_decode_store(self) -> "GpuDecodeStore":
        """Create and configure a GpuDecodeStore for Rust-native GPU decode.

        Registers all attention weights, MoE layers, RoPE tables, and KV cache.
        Called once at startup. Returns the configured store.
        """
        self._require_supported_runtime_features()
        from krasis import GpuDecodeStore

        device = torch.device(self.ranks[0].device)
        gpu_idx = device.index or 0

        deepseek_v4_hqq6_native_int4 = bool(
            self.cfg.is_deepseek_v4
            and self.quant_cfg.attention == "hqq6"
            and self.quant_cfg.kv_cache_format == "native"
            and self.quant_cfg.gpu_expert_bits == 4
            and self.quant_cfg.cpu_expert_bits == 4
        )
        store = GpuDecodeStore(gpu_idx, deepseek_v4_hqq6_native_int4)
        # Compute max QKV buffer size across all layer types.
        # Use dimension attributes (not weight tensors) since streaming may have
        # offloaded weights to CPU, leaving tensor attributes as None.
        max_qkv = self.cfg.hidden_size * 3  # default for standard GQA
        for layer_idx, layer in enumerate(self.layers):
            if layer.attention is None:
                # Mamba2 or MoE-only layers have no attention QKV buffers
                if layer.layer_type == "mamba2" and layer.mamba2_weights is not None:
                    # Mamba2 in_proj output is large: d_inner*2 + conv_dim + num_heads
                    m2 = self.cfg
                    in_proj_out = m2.mamba_d_inner * 2 + m2.mamba_conv_dim + m2.mamba_num_heads
                    max_qkv = max(max_qkv, in_proj_out)
            elif (
                layer.layer_type == "linear_attention"
                and self.cfg.is_kimi_delta_attention_layer(layer_idx)
            ):
                # KDA has independent Q/K/V projections.  Its exact projection
                # width comes from the checkpoint's linear-attention geometry;
                # the fused GatedDeltaNet QKVZ formula does not apply.
                kda_width = (
                    self.cfg.linear_num_key_heads
                    * self.cfg.linear_key_head_dim
                )
                max_qkv = max(max_qkv, kda_width)
            elif layer.layer_type == "linear_attention":
                attn = layer.attention
                # in_proj_qkvz output dim = nk*(dk + dk + hr*dv + hr*dv)
                qkvz_out = attn.num_k_heads * (2 * attn.k_head_dim + 2 * attn.head_ratio * attn.v_head_dim)
                max_qkv = max(max_qkv, qkvz_out)
            elif hasattr(layer.attention, 'kv_a_proj'):
                # MLA: q_absorbed [num_heads * ckv_dim] is the largest buffer
                ma = layer.attention
                q_out = ma.num_heads * (ma.qk_nope_dim + ma.qk_rope_dim)
                q_absorbed = ma.num_heads * ma.ckv_dim  # padded to 512 for MLA
                kv_out = ma.ckv_dim + ma.qk_rope_dim
                max_qkv = max(max_qkv, q_out, q_absorbed, kv_out)
            elif hasattr(layer, 'gqa_weights') and layer.gqa_weights:
                q_sz = layer.gqa_weights["q_proj"].shape[0]
                k_sz = layer.gqa_weights["k_proj"].shape[0]
                v_src = layer.gqa_weights.get("v_proj", layer.gqa_weights["k_proj"])
                v_sz = v_src.shape[0]
                max_qkv = max(max_qkv, q_sz + k_sz + v_sz)
            elif hasattr(layer.attention, 'num_heads'):
                ga = layer.attention
                q_sz = ga.num_heads * ga.head_dim * (2 if ga.gated_attention else 1)
                kv_sz = ga.num_kv_heads * ga.head_dim
                max_qkv = max(max_qkv, q_sz + kv_sz * 2)

        # Compute max intermediate from actual layer types (not unused config fields)
        has_dense_mlp = any(l.dense_mlp is not None for l in self.layers)
        max_inter = max(self.cfg.moe_intermediate_size, self.cfg.effective_shared_expert_intermediate)
        if has_dense_mlp:
            max_inter = max(max_inter, self.cfg.intermediate_size)

        store.configure(
            hidden_size=self.cfg.hidden_size,
            num_layers=len(self.layers),
            vocab_size=self.cfg.vocab_size,
            eps=self.cfg.rms_norm_eps,
            max_experts_per_tok=self.cfg.num_experts_per_tok,
            max_intermediate_size=max_inter,
            max_qkv_size=max_qkv,
            group_size=self.quant_cfg.expert_group_size,
            expert_bits=self.quant_cfg.gpu_expert_bits,
            moe_intermediate_size=self.cfg.moe_intermediate_size,
            shared_expert_intermediate_size=self.cfg.effective_shared_expert_intermediate,
        )
        # Register embedding
        store.set_embedding(self.embedding.data_ptr())
        store.set_embedding_scale(float(getattr(self.cfg, "embedding_scale", 1.0)))
        store.set_final_logit_softcap(float(getattr(self.cfg, "final_logit_softcapping", 0.0)))

        # Register final norm
        store.set_final_norm(self.final_norm.data_ptr(), self.cfg.hidden_size)

        # Register LM head (always as BF16 for cuBLAS GEMV compatibility)
        # Dequantize on CPU to avoid OOM on GPU0 (float32 intermediate can be ~3 GB for large vocabs)
        lm_head_w = self.lm_head_data
        if isinstance(lm_head_w, tuple):
            # INT8 (weight, scale) — dequantize to BF16 on CPU, then copy to GPU
            w_int8, scale = lm_head_w
            lm_head_bf16 = (w_int8.cpu().float() * scale.cpu().unsqueeze(1)).to(torch.bfloat16).contiguous().to(device)
            self._rust_lm_head = lm_head_bf16  # prevent GC
            lm_head_ptr = lm_head_bf16.data_ptr()
            lm_head_rows = lm_head_bf16.shape[0]
            lm_head_cols = lm_head_bf16.shape[1]
        else:
            lm_head_ptr = lm_head_w.data_ptr()
            lm_head_rows = lm_head_w.shape[0]
            lm_head_cols = lm_head_w.shape[1]
        store.set_lm_head(
            store.register_weight(lm_head_ptr, lm_head_rows, lm_head_cols, 0)  # 0 = BF16
        )

        # Norm bias flag (Qwen3 uses (1+w)*x)
        store.set_norm_bias_one(getattr(self.cfg, 'norm_bias_one', False))

        # Track weight IDs for LA layers (needed for re-registration after state reset)
        self._la_wids = {}
        self._kda_wids = {}
        rope_set = False

        class _DecodeKeepalive(list):
            def __init__(self):
                super().__init__()
                self.labels = []

            def append(self, value):
                super().append(value)
                self.labels.append("unclassified")

            def extend(self, values):
                for value in values:
                    self.append(value)

            def keep(self, label: str, *values):
                for value in values:
                    super().append(value)
                    self.labels.append(label)

        # When streaming attention is enabled, the layer's weight attributes point to
        # ping-pong GPU buffers that may contain a DIFFERENT layer's data.  We must
        # create permanent GPU copies from the CPU-pinned originals so Rust decode
        # always sees correct weights for every layer.
        self._rust_decode_weights = _DecodeKeepalive()  # prevent GC of permanent GPU copies

        attn_quant = self.quant_cfg.attention  # "bf16" or "awq"
        hqq_active = is_hqq_attention(attn_quant)
        marlin_gs = 128  # Marlin group size for both INT8 and INT4

        # AWQ template: per-layer channel scales from calibration
        _awq_template = None
        if attn_quant == "awq":
            from krasis.awq_calibrate import load_template
            template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "templates", "attention")
            _awq_template = load_template(template_dir, self.cfg.model_path)
            if _awq_template is not None:
                tmpl_version = _awq_template.get("version", 1)
                if tmpl_version >= 2:
                    summary = _awq_template["summary"]
                    logger.info("AWQ v2 template loaded: %d AWQ-scaled, %d plain INT4, %d BF16 tensors "
                                "(avg error reduction: %.1f%%)",
                                summary["awq_scaled_tensors"],
                                summary["plain_int4_tensors"],
                                summary["bf16_tensors"],
                                summary["avg_error_reduction"] * 100)
                else:
                    # v1 template (old per-tensor decisions, no per-channel scaling)
                    n_int4 = _awq_template["summary"]["int4"]
                    n_int8 = _awq_template["summary"]["int8"]
                    n_bf16 = _awq_template["summary"]["bf16"]
                    logger.info("AWQ v1 template loaded: %d INT4, %d INT8, %d BF16 tensors",
                                n_int4, n_int8, n_bf16)
            else:
                raise RuntimeError(
                    f"AWQ attention requested but no calibration template found for model "
                    f"{self.cfg.model_path}. No template available locally or from GitHub. "
                    f"Run './dev awq-calibrate <config>' to generate one, or use "
                    f"--attention-quant bf16 for unquantized attention."
                )

        # Marlin quantization helpers (lazy import, only when quantizing)
        _marlin_workspace = None
        _marlin_scalar_type_int4 = None
        _marlin_scalar_type_int8 = None
        if attn_quant == "awq":
            from krasis.marlin_utils import marlin_make_workspace, get_scalar_type
            _marlin_workspace = marlin_make_workspace(device)
            _marlin_scalar_type_int4 = get_scalar_type(4, False)
            _marlin_scalar_type_int8 = get_scalar_type(8, False)

        def _register_attn_weight(w: torch.Tensor, layer_idx: int = -1,
                                  layer_type: str = "", tensor_name: str = "",
                                  awq_scales: Optional[torch.Tensor] = None) -> int:
            """Register an attention weight as BF16, Marlin INT8, or Marlin INT4.

            For Marlin quantization: Rust does quantize + repack on CPU,
            Python uploads packed result to GPU as PyTorch tensors (shared
            between prefill GEMM and Rust decode GEMV). Same repack format
            for both paths. No BF16 ever touches VRAM.

            For BF16: weight must already be on GPU.

            Args:
                awq_scales: Optional per-channel scales [K] from AWQ calibration.
                    When provided, weight columns are scaled by s[j] before
                    quantization: W[:,j] *= s[j]. The caller is responsible
                    for folding 1/s[j] into the preceding RMSNorm weight.

            Returns weight ID in the Rust store. For quantized weights, also
            stores (packed, scales, workspace, scalar_type, N, K) on
            self._marlin_attn_weights[wid] for prefill GEMM."""
            # Stash BF16 on CPU for aux store quantization (only when streaming is off)
            if self._aux_bf16_stash is not None and layer_idx >= 0:
                w_stash = w.cpu().contiguous() if w.is_cuda else w.contiguous()
                self._aux_bf16_stash[(layer_idx, tensor_name)] = w_stash

            # Determine effective quantization for this tensor
            effective_quant = attn_quant
            if attn_quant == "awq":
                if _awq_template is not None:
                    from krasis.awq_calibrate import get_tensor_decision
                    effective_quant = get_tensor_decision(
                        _awq_template, layer_idx, layer_type, tensor_name)
                else:
                    # This should be unreachable — we raise at template load time above
                    raise RuntimeError(
                        "AWQ attention active but no template loaded — this is a bug"
                    )

            if effective_quant in ("int8", "int4") and w.dtype == torch.bfloat16:
                use_int4 = (effective_quant == "int4")
                num_bits = 4 if use_int4 else 8
                n, k = w.shape[0], w.shape[1]
                if n % 64 == 0 and k % 16 == 0 and k % marlin_gs == 0:
                    # Step 1: Get contiguous CPU BF16 data
                    w_cpu = w.cpu().contiguous() if w.is_cuda else w.contiguous()

                    # Step 1.5: AWQ per-channel scaling (if provided)
                    # Scale weight columns: W[:,j] *= s[j] to protect channels
                    # with high activation magnitude during quantization.
                    if awq_scales is not None:
                        assert awq_scales.shape[0] == k, \
                            f"AWQ scales size {awq_scales.shape[0]} != weight K dim {k}"
                        # Apply in float32 for precision, then convert back to BF16
                        w_cpu = (w_cpu.float() * awq_scales.float().unsqueeze(0)).to(torch.bfloat16)
                        w_cpu = w_cpu.contiguous()

                    # Step 2: Rust quantizes + repacks on CPU (same format as expert weights)
                    if use_int4:
                        packed_bytes, scales_bytes, rn, rk = store.repack_marlin_int4_cpu(
                            w_cpu.data_ptr(), n, k, marlin_gs)
                    else:
                        packed_bytes, scales_bytes, rn, rk = store.repack_marlin_int8_cpu(
                            w_cpu.data_ptr(), n, k, marlin_gs)
                    del w_cpu

                    # Step 3: Create GPU tensors from Rust-repacked data
                    # CRITICAL: scales are raw BF16 bits stored as u16 — must use
                    # .view(torch.bfloat16) to reinterpret bits, NOT .to(torch.bfloat16)
                    # which would numerically convert (e.g. 15890 -> BF16(15872.0) instead
                    # of BF16(0.1429)).
                    import numpy as np
                    packed_np = np.frombuffer(packed_bytes, dtype=np.uint32)
                    scales_np = np.frombuffer(scales_bytes, dtype=np.uint16)
                    repacked = torch.from_numpy(packed_np.copy()).to(
                        dtype=torch.int32, device=device)
                    scale_raw = torch.from_numpy(scales_np.copy()).to(torch.int16)

                    # Single-slot AWQ: allocate scales at FP32 size (2x BF16) so the
                    # same GPU buffer holds either BF16 Marlin scales or FP32 simple scales.
                    # BF16 scales occupy the first half; gptq_marlin_gemm gets a BF16 view.
                    n_scale_elements = (k // marlin_gs) * n
                    if use_int4:
                        # Allocate slot: 2x BF16 elements as int16 = FP32 byte capacity
                        scales_slot = torch.empty(n_scale_elements * 2, dtype=torch.int16, device=device)
                        # Fill first half with BF16 data
                        bf16_on_gpu = scale_raw.view(torch.bfloat16).to(device)
                        scales_slot[:n_scale_elements].copy_(bf16_on_gpu.view(torch.int16).reshape(-1))
                        # BF16 view for gptq_marlin_gemm (same underlying memory, first half)
                        scale_perm = scales_slot[:n_scale_elements].view(torch.bfloat16).reshape(k // marlin_gs, n)
                        del bf16_on_gpu
                    else:
                        # INT8: no single-slot swap, standard BF16 scales
                        scale_perm = scale_raw.view(torch.bfloat16).to(device)

                    # Reshape for gptq_marlin_gemm: packed [K/16, pack_factor*N], scales [K/gs, N]
                    if use_int4:
                        repacked = repacked.reshape(k // 16, 2 * n)
                    else:
                        repacked = repacked.reshape(k // 16, 4 * n)
                    if not use_int4:
                        scale_perm = scale_perm.reshape(k // marlin_gs, n)

                    # Step 4: Register GPU pointer with Rust store
                    # For INT4 single-slot: scales_ptr points to the full FP32-sized slot
                    scalar_type = _marlin_scalar_type_int4 if use_int4 else _marlin_scalar_type_int8
                    if use_int4:
                        scales_data_ptr = scales_slot.data_ptr() if use_int4 else scale_perm.data_ptr()
                        wid = store.register_marlin_int4_weight(
                            repacked.data_ptr(), scales_data_ptr,
                            n, k, marlin_gs)
                    else:
                        wid = store.register_marlin_int8_weight(
                            repacked.data_ptr(), scale_perm.data_ptr(),
                            n, k, marlin_gs)

                    # Step 5: Keep PyTorch tensors alive and accessible for prefill.
                    # Single-slot: tensors stay on GPU permanently (they ARE the slots).
                    # For INT4, also keep scales_slot alive to prevent GC.
                    self._marlin_attn_weights[wid] = (
                        repacked, scale_perm, _marlin_workspace,
                        scalar_type, n, k,
                    )
                    if use_int4:
                        # Keep the full slot tensor alive (scale_perm is a view into it)
                        if not hasattr(self, '_single_slot_scales'):
                            self._single_slot_scales = []
                        self._single_slot_scales.append(scales_slot)
                    return wid
                else:
                    logger.warning("Attention weight [%d×%d] not Marlin-compatible "
                                   "(need N%%64==0, K%%16==0, K%%gs==0), using BF16", n, k)
            # BF16 path: weight must be on GPU
            if not w.is_cuda:
                w = w.to(device, non_blocking=True)
                self._keep_rust_decode_weight("attention_bf16_gpu_copy", w)
            return store.register_weight(
                w.data_ptr(), w.shape[0], w.shape[1], _weight_dtype_code(w))

        self._marlin_attn_weights = {}  # wid -> (packed, scales, workspace, scalar_type, N, K)

        # Stash BF16 CPU copies of attention weights for aux store AWQ quantization.
        # Needed when streaming attention is not enabled and attrs get replaced by MarlinWeight.
        if not getattr(self, '_stream_attn_enabled', False):
            self._aux_bf16_stash = {}  # {(layer_idx, attr_name): cpu_tensor}
        else:
            self._aux_bf16_stash = None  # streaming has its own CPU copies

        hqq_registered_layers = 0
        hqq_cache_bytes = 0
        if hqq_active:
            hqq_registered_layers = self._register_hqq_attention_layers_on_store(
                store, device, self._rust_decode_weights
            )
            hqq_cache_bytes = hqq_attention_cache_total_bytes(
                self.cfg.model_path,
                self.quant_cfg.hqq_cache_profile,
                attention_quant_cache_nbits(self.quant_cfg.attention) or 4,
                self.quant_cfg.hqq_group_size,
            ) or 0
            logger.info(
                "HQQ attention registration completed on cuda:%d: %d layers, %d MB validated cache, %d tensors loaded. Runtime execution descriptors are active on the decode store.",
                gpu_idx,
                hqq_registered_layers,
                hqq_cache_bytes >> 20,
                self._hqq_attention_loaded_tensors,
            )

        self._prepare_mamba2_projection_int4_cache()

        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attention
            inp_norm = layer.input_norm_weight
            post_norm = layer.post_attn_norm_weight

            py_diag = os.environ.get("KRASIS_PY_DIAG") == "1"

            # DIAG: check raw norm values BEFORE AWQ fold
            if py_diag and layer_idx == 0:
                import sys
                _rv = inp_norm.data.cpu().float()[:10].tolist()
                _rl2 = inp_norm.data.cpu().float().norm().item()
                print(f"[PY-DIAG] layer0 inp_norm BEFORE_AWQ first10={_rv}", file=sys.stderr)
                print(f"[PY-DIAG] layer0 inp_norm BEFORE_AWQ L2={_rl2:.6f}", file=sys.stderr)

            # ── Nemotron Mamba2 layer registration ──
            if layer.layer_type == "mamba2":
                self._register_mamba2_layer(store, layer_idx, layer, device)
                store.register_mlp(layer_idx, "none")
                continue

            # ── Nemotron MoE-only layer (no attention) ──
            if layer.layer_type == "moe":
                self._register_nemotron_moe_only_layer(store, layer_idx, layer, device)
                continue

            # AWQ v2: load per-layer per-channel scales for input projections
            _layer_awq_scales = None
            if attn_quant == "awq" and _awq_template is not None:
                from krasis.awq_calibrate import (
                    get_layer_scales, is_awq_scaled_tensor)
                _layer_awq_scales = get_layer_scales(_awq_template, layer_idx)

            if self.cfg.is_deepseek_v4:
                if self._stream_attn_enabled:
                    raise RuntimeError(
                        "DeepSeek-V4 nested attention streaming is not implemented. "
                        "The legacy flat GQA/LA streamer cannot represent compressor and "
                        "indexer tensors, so startup stops instead of registering partial weights."
                    )
                v4 = attn.attention
                hc = attn.hyper_connection
                v4_hqq_runtime = (
                    self._hqq_attention_runtime.get(layer_idx, {})
                    if hqq_active
                    else {}
                )

                def _v4_register(weight, name):
                    if hqq_active:
                        if name in v4_hqq_runtime:
                            return store.register_hqq_runtime_weight_view(
                                layer_idx=layer_idx,
                                tensor_name=name,
                            )
                        if name in {
                            "wq_a", "wq_b", "wkv", "wo_b",
                        }:
                            raise RuntimeError(
                                f"DeepSeek-V4 HQQ layer {layer_idx} is missing "
                                f"required runtime tensor {name}"
                            )
                    return _register_attn_weight(
                        weight, layer_idx, "deepseek_v4", name
                    )

                v4_wids = {
                    name: _v4_register(v4[name], name)
                    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
                }
                cache, v4_offset = self._kv_cache_slot_for_layer(layer_idx)
                v4_cache = cache.get_deepseek_v4_layer_caches(v4_offset)
                raw_cache = v4_cache["raw"]

                def _v4_cache_registration(bf16_cache, native_cache, block_size):
                    if v4_cache["format"] == "native":
                        if bf16_cache is not None or not isinstance(native_cache, dict):
                            raise RuntimeError(
                                f"DeepSeek-V4 layer {layer_idx} Native cache allocation is inconsistent"
                            )
                        codes = native_cache["codes"]
                        scales = native_cache["scales"]
                        tail = native_cache.get("tail")
                        return {
                            "cache_ptr": 0,
                            "cache_elems": 0,
                            "cache_rows": int(codes.shape[0]),
                            "cache_format": "native",
                            "native_codes_ptr": int(codes.data_ptr()),
                            "native_codes_elems": int(codes.numel()),
                            "native_scale_exponents_ptr": int(scales.data_ptr()),
                            "native_scale_exponents_elems": int(scales.numel()),
                            "native_tail_ptr": int(tail.data_ptr()) if tail is not None else 0,
                            "native_tail_elems": int(tail.numel()) if tail is not None else 0,
                            "native_block_size": int(block_size),
                        }
                    if bf16_cache is None or native_cache is not None:
                        raise RuntimeError(
                            f"DeepSeek-V4 layer {layer_idx} BF16 cache allocation is inconsistent"
                        )
                    return {
                        "cache_ptr": int(bf16_cache.data_ptr()),
                        "cache_elems": int(bf16_cache.numel()),
                        "cache_rows": int(bf16_cache.shape[0]),
                        "cache_format": "bf16",
                        "native_codes_ptr": 0,
                        "native_codes_elems": 0,
                        "native_scale_exponents_ptr": 0,
                        "native_scale_exponents_elems": 0,
                        "native_tail_ptr": 0,
                        "native_tail_elems": 0,
                        "native_block_size": int(block_size),
                    }

                raw_registration = _v4_cache_registration(
                    raw_cache,
                    v4_cache["raw_native"],
                    v4_cache["fp8_block_size"],
                )
                v4_max_seq = int(cache.max_context_tokens)
                (
                    v4_rope_cos_ptr,
                    v4_rope_sin_ptr,
                    v4_rope_rows,
                    _v4_rope_cos,
                    _v4_rope_sin,
                ) = self._deepseek_v4_rope_table_ptrs(
                    v4_max_seq, int(v4_cache["ratio"]), device
                )
                store.register_deepseek_v4_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(),
                    input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm.data_ptr(),
                    post_attn_norm_size=post_norm.numel(),
                    wq_a_wid=v4_wids["wq_a"],
                    wq_b_wid=v4_wids["wq_b"],
                    wkv_wid=v4_wids["wkv"],
                    wo_a_wid=v4_wids["wo_a"],
                    wo_b_wid=v4_wids["wo_b"],
                    attn_sink_ptr=v4["attn_sink"].data_ptr(),
                    attn_sink_elems=v4["attn_sink"].numel(),
                    q_norm_ptr=v4["q_norm"].data_ptr(),
                    q_norm_elems=v4["q_norm"].numel(),
                    kv_norm_ptr=v4["kv_norm"].data_ptr(),
                    kv_norm_elems=v4["kv_norm"].numel(),
                    num_heads=self.cfg.num_attention_heads,
                    head_dim=self.cfg.attention_head_dim,
                    rope_dim=self.cfg.qk_rope_head_dim,
                    o_groups=self.cfg.o_groups,
                    sliding_window=self.cfg.sliding_window,
                    compress_ratio=int(v4_cache["ratio"]),
                    compress_rope_theta=self.cfg.compress_rope_theta,
                    raw_cache_ptr=raw_registration["cache_ptr"],
                    raw_cache_elems=raw_registration["cache_elems"],
                    cache_format=raw_registration["cache_format"],
                    raw_native_codes_ptr=raw_registration["native_codes_ptr"],
                    raw_native_codes_elems=raw_registration["native_codes_elems"],
                    raw_native_scale_exponents_ptr=raw_registration["native_scale_exponents_ptr"],
                    raw_native_scale_exponents_elems=raw_registration["native_scale_exponents_elems"],
                    raw_native_tail_ptr=raw_registration["native_tail_ptr"],
                    raw_native_tail_elems=raw_registration["native_tail_elems"],
                    native_block_size=raw_registration["native_block_size"],
                    rope_cos_ptr=v4_rope_cos_ptr,
                    rope_sin_ptr=v4_rope_sin_ptr,
                    rope_rows=v4_rope_rows,
                    logical_max_seq=v4_max_seq,
                )

                if v4_cache["ratio"] > 0:
                    compressor = v4["compressor"]
                    compressed = v4_cache["compressed"]
                    compressed_registration = _v4_cache_registration(
                        compressed,
                        v4_cache["compressed_native"],
                        v4_cache["fp8_block_size"],
                    )
                    kv_state = v4_cache["compressor_kv_state"]
                    score_state = v4_cache["compressor_score_state"]
                    store.register_deepseek_v4_compressor(
                        layer_idx=layer_idx,
                        ape_wid=_v4_register(compressor["ape"], "compressor.ape"),
                        wkv_wid=_v4_register(compressor["wkv"], "compressor.wkv"),
                        wgate_wid=_v4_register(compressor["wgate"], "compressor.wgate"),
                        norm_ptr=compressor["norm"].data_ptr(),
                        norm_elems=compressor["norm"].numel(),
                        **compressed_registration,
                        kv_state_ptr=kv_state.data_ptr(),
                        kv_state_elems=kv_state.numel(),
                        score_state_ptr=score_state.data_ptr(),
                        score_state_elems=score_state.numel(),
                    )

                if v4_cache["ratio"] == 4:
                    indexer = v4["indexer"]
                    index_compressor = indexer["compressor"]
                    index_cache = v4_cache["index"]
                    index_registration = _v4_cache_registration(
                        index_cache,
                        v4_cache["index_native"],
                        v4_cache["fp4_block_size"],
                    )
                    index_kv_state = v4_cache["index_kv_state"]
                    index_score_state = v4_cache["index_score_state"]
                    store.register_deepseek_v4_indexer(
                        layer_idx=layer_idx,
                        wq_b_wid=_v4_register(indexer["wq_b"], "indexer.wq_b"),
                        weights_proj_wid=_v4_register(
                            indexer["weights_proj"], "indexer.weights_proj"
                        ),
                        ape_wid=_v4_register(
                            index_compressor["ape"], "indexer.compressor.ape"
                        ),
                        compressor_wkv_wid=_v4_register(
                            index_compressor["wkv"], "indexer.compressor.wkv"
                        ),
                        compressor_wgate_wid=_v4_register(
                            index_compressor["wgate"], "indexer.compressor.wgate"
                        ),
                        norm_ptr=index_compressor["norm"].data_ptr(),
                        norm_elems=index_compressor["norm"].numel(),
                        **index_registration,
                        kv_state_ptr=index_kv_state.data_ptr(),
                        kv_state_elems=index_kv_state.numel(),
                        score_state_ptr=index_score_state.data_ptr(),
                        score_state_elems=index_score_state.numel(),
                        index_topk=self.cfg.index_topk,
                        index_n_heads=self.cfg.index_n_heads,
                        index_head_dim=self.cfg.index_head_dim,
                    )

                store.register_deepseek_v4_hyper_connection(
                    layer_idx=layer_idx,
                    attn_fn_wid=_v4_register(hc["hc_attn_fn"], "hc_attn_fn"),
                    attn_base_ptr=hc["hc_attn_base"].data_ptr(),
                    attn_base_elems=hc["hc_attn_base"].numel(),
                    attn_scale_ptr=hc["hc_attn_scale"].data_ptr(),
                    attn_scale_elems=hc["hc_attn_scale"].numel(),
                    ffn_fn_wid=_v4_register(hc["hc_ffn_fn"], "hc_ffn_fn"),
                    ffn_base_ptr=hc["hc_ffn_base"].data_ptr(),
                    ffn_base_elems=hc["hc_ffn_base"].numel(),
                    ffn_scale_ptr=hc["hc_ffn_scale"].data_ptr(),
                    ffn_scale_elems=hc["hc_ffn_scale"].numel(),
                    mult=self.cfg.hc_mult,
                    sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                    eps=self.cfg.hc_eps,
                )
                if hqq_active:
                    store.attach_hqq_runtime_deepseek_v4_layer(
                        layer_idx=layer_idx,
                        tensor_names=sorted(v4_hqq_runtime),
                    )
            elif (
                layer.layer_type == "linear_attention"
                and self.cfg.is_kimi_delta_attention_layer(layer_idx)
            ):
                projection_names = (
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "f_a_proj", "f_b_proj", "b_proj",
                    "g_a_proj", "g_b_proj",
                )
                if hqq_active:
                    projection_wids = {
                        name: store.register_hqq_runtime_weight_view(
                            layer_idx=layer_idx, tensor_name=name
                        )
                        for name in projection_names
                    }
                    conv_weight = attn._hqq_conv_weight
                    a_log = attn._hqq_a_log
                    dt_bias = attn._hqq_dt_bias
                    o_norm = attn._hqq_norm_weight
                    conv_state = attn._hqq_conv_state
                    recurrent_state = attn._hqq_recur_state
                else:
                    kda = attn.weights
                    cpu_kda = (
                        self._stream_attn_cpu.get(layer_idx, {})
                        if self._stream_attn_enabled
                        else {}
                    )
                    projection_wids = {
                        name: _register_attn_weight(
                            cpu_kda.get(name, kda[name]),
                            layer_idx,
                            "kimi_delta_attention",
                            name,
                        )
                        for name in projection_names
                    }
                    conv_weight = torch.cat(
                        tuple(
                            cpu_kda.get(name, kda[name])
                            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
                        ),
                        dim=0,
                    ).squeeze(1).float().contiguous().to(device)
                    a_log = cpu_kda.get(
                        "A_log", kda["A_log"]
                    ).float().contiguous().to(device)
                    dt_bias = cpu_kda.get(
                        "dt_bias", kda["dt_bias"]
                    ).float().contiguous().to(device)
                    o_norm = cpu_kda.get(
                        "o_norm", kda["o_norm"]
                    ).float().contiguous().to(device)
                    qkv_dim = (
                        self.cfg.linear_num_key_heads
                        * self.cfg.linear_key_head_dim
                    )
                    conv_state = torch.zeros(
                        3,
                        qkv_dim,
                        self.cfg.linear_conv_kernel_dim - 1,
                        device=device,
                        dtype=torch.float32,
                    )
                    recurrent_state = torch.zeros(
                        self.cfg.linear_num_key_heads,
                        self.cfg.linear_key_head_dim,
                        self.cfg.linear_value_head_dim,
                        device=device,
                        dtype=torch.float32,
                    )
                    self._rust_decode_weights.extend(
                        [conv_weight, a_log, dt_bias, o_norm,
                         conv_state, recurrent_state]
                    )
                    attn._rust_conv_weight = conv_weight
                    attn._rust_a_log = a_log
                    attn._rust_dt_bias = dt_bias
                    attn._rust_norm_weight = o_norm
                    attn._rust_conv_state = conv_state
                    attn._rust_recur_state = recurrent_state
                self._kda_wids[layer_idx] = projection_wids
                store.register_kda_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(),
                    input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm.data_ptr(),
                    post_attn_norm_size=post_norm.numel(),
                    q_proj_wid=projection_wids["q_proj"],
                    k_proj_wid=projection_wids["k_proj"],
                    v_proj_wid=projection_wids["v_proj"],
                    o_proj_wid=projection_wids["o_proj"],
                    f_a_proj_wid=projection_wids["f_a_proj"],
                    f_b_proj_wid=projection_wids["f_b_proj"],
                    b_proj_wid=projection_wids["b_proj"],
                    g_a_proj_wid=projection_wids["g_a_proj"],
                    g_b_proj_wid=projection_wids["g_b_proj"],
                    conv_weight_ptr=conv_weight.data_ptr(),
                    a_log_ptr=a_log.data_ptr(),
                    dt_bias_ptr=dt_bias.data_ptr(),
                    norm_weight_ptr=o_norm.data_ptr(),
                    conv_state_ptr=conv_state.data_ptr(),
                    recur_state_ptr=recurrent_state.data_ptr(),
                    num_heads=self.cfg.linear_num_key_heads,
                    head_dim=self.cfg.linear_key_head_dim,
                    kernel_dim=self.cfg.linear_conv_kernel_dim,
                    gate_lower_bound=self.cfg.linear_gate_lower_bound,
                )
            elif hqq_active:
                # HQQ registration below owns attention projection residency.
                # Do not route HQQ tensors through the normal BF16/Marlin
                # registration path, otherwise quantized attention becomes
                # additive over resident BF16 weights instead of replacing it.
                pass
            elif layer.layer_type == "linear_attention":
                # Source projection weights — when quantizing, keep on CPU to avoid
                # putting full BF16 in VRAM. Only upload to GPU for BF16 mode.
                if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                    cpu_w = self._stream_attn_cpu[layer_idx]
                    if attn_quant == "bf16":
                        qkvz_w = cpu_w.get("in_proj_qkvz", attn.in_proj_qkvz).to(device, non_blocking=True)
                        ba_w = cpu_w.get("in_proj_ba", attn.in_proj_ba).to(device, non_blocking=True)
                        out_w = cpu_w.get("out_proj", attn.out_proj).to(device, non_blocking=True)
                        self._rust_decode_weights.extend([qkvz_w, ba_w, out_w])
                    else:
                        # Quantizing: pass CPU tensors directly, never touch VRAM with BF16
                        qkvz_w = cpu_w.get("in_proj_qkvz", attn.in_proj_qkvz)
                        ba_w = cpu_w.get("in_proj_ba", attn.in_proj_ba)
                        out_w = cpu_w.get("out_proj", attn.out_proj)
                else:
                    qkvz_w = attn.in_proj_qkvz
                    ba_w = attn.in_proj_ba
                    out_w = attn.out_proj

                # AWQ v2: pass per-channel scales for input projections only.
                # Both LA input projections consume the same post-input-norm hidden
                # state, so the standard AWQ equivalent transform applies here too:
                # scale weight columns by s and fold 1/s into the preceding norm.
                _qkvz_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "in_proj_qkvz")
                ) else None
                _ba_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "in_proj_ba")
                ) else None

                qkvz_wid = _register_attn_weight(qkvz_w, layer_idx, "linear_attention",
                                                  "in_proj_qkvz", awq_scales=_qkvz_scales)
                ba_wid = _register_attn_weight(ba_w, layer_idx, "linear_attention",
                                               "in_proj_ba", awq_scales=_ba_scales)
                out_wid = _register_attn_weight(out_w, layer_idx, "linear_attention", "out_proj")
                self._la_wids[layer_idx] = (qkvz_wid, ba_wid, out_wid)

                # Replace attention weight attributes with MarlinWeight for prefill
                if qkvz_wid in self._marlin_attn_weights:
                    from krasis.attention import MarlinWeight
                    mw = self._marlin_attn_weights[qkvz_wid]
                    attn.in_proj_qkvz = MarlinWeight(*mw)
                if ba_wid in self._marlin_attn_weights:
                    from krasis.attention import MarlinWeight
                    mw = self._marlin_attn_weights[ba_wid]
                    attn.in_proj_ba = MarlinWeight(*mw)
                if out_wid in self._marlin_attn_weights:
                    from krasis.attention import MarlinWeight
                    mw = self._marlin_attn_weights[out_wid]
                    attn.out_proj = MarlinWeight(*mw)

                # AWQ v2: fold 1/s into input_norm_weight.
                # This preserves the LA projection outputs exactly:
                # (RMSNorm(x) / s) @ (W * s)^T == RMSNorm(x) @ W^T
                # for both input projections fed by the same post-input-norm hidden.
                if _layer_awq_scales is not None and (_qkvz_scales is not None
                                                       or _ba_scales is not None):
                    s = _layer_awq_scales.to(inp_norm.device)
                    inp_norm.data.copy_(
                        (inp_norm.float() / s.float()).to(inp_norm.dtype))
                    logger.debug("AWQ: folded scales into input_norm for LA layer %d "
                                 "(mean_scale=%.4f)", layer_idx, s.mean().item())

                # DIAG: trace input_norm pointer and values at registration time
                if py_diag and layer_idx == 0:
                    import sys
                    _nv = inp_norm.data.cpu().float()[:10].tolist()
                    _nl2 = inp_norm.data.cpu().float().norm().item()
                    print(f"[PY-DIAG] layer0 inp_norm ptr={inp_norm.data_ptr():#x} "
                          f"numel={inp_norm.numel()} dtype={inp_norm.dtype} "
                          f"device={inp_norm.device}", file=sys.stderr)
                    print(f"[PY-DIAG] layer0 inp_norm first10={_nv}", file=sys.stderr)
                    print(f"[PY-DIAG] layer0 inp_norm L2={_nl2:.6f}", file=sys.stderr)
                    # Also check post_norm
                    _pv = post_norm.data.cpu().float()[:10].tolist()
                    _pl2 = post_norm.data.cpu().float().norm().item()
                    print(f"[PY-DIAG] layer0 post_norm ptr={post_norm.data_ptr():#x} "
                          f"numel={post_norm.numel()} dtype={post_norm.dtype}", file=sys.stderr)
                    print(f"[PY-DIAG] layer0 post_norm first10={_pv}", file=sys.stderr)
                    print(f"[PY-DIAG] layer0 post_norm L2={_pl2:.6f}", file=sys.stderr)

                # Conv weight: [conv_dim, 1, kernel_dim] -> [conv_dim, kernel_dim]
                # Rust LA kernels work in FP32, so create FP32 copies of BF16 tensors.
                # When streaming, attn.* may be None or stale — use CPU-pinned source.
                if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                    cpu_la = self._stream_attn_cpu[layer_idx]
                    conv_src = cpu_la.get("conv1d_weight", attn.conv1d_weight)
                    a_log_src = cpu_la.get("A_log", attn.A_log)
                    dt_bias_src = cpu_la.get("dt_bias", attn.dt_bias)
                    norm_w_src = cpu_la.get("norm_weight", attn.norm_weight)
                else:
                    conv_src = attn.conv1d_weight
                    a_log_src = attn.A_log
                    dt_bias_src = attn.dt_bias
                    norm_w_src = attn.norm_weight
                conv_w = conv_src.squeeze(1).contiguous().float().to(device)
                attn._rust_conv_weight = conv_w

                attn._init_state(batch_size=1)
                # conv_state is BF16 from _init_state, Rust kernel needs FP32
                attn._rust_conv_state = attn._conv_state.squeeze(0).float().contiguous()
                # norm_weight is BF16, Rust gated_rmsnorm_silu kernel needs FP32
                attn._rust_norm_weight = norm_w_src.float().contiguous().to(device)
                # recurrent_state may need FP32 conversion for Rust kernel
                attn._rust_recur_state = attn._recurrent_state.squeeze(0).float().contiguous()
                # A_log and dt_bias are BF16 from weight loader, Rust kernel needs FP32
                attn._rust_a_log = a_log_src.float().contiguous().to(device)
                attn._rust_dt_bias = dt_bias_src.float().contiguous().to(device)

                store.register_la_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                    in_proj_qkvz_wid=qkvz_wid, in_proj_ba_wid=ba_wid, out_proj_wid=out_wid,
                    conv_weight_ptr=conv_w.data_ptr(),
                    a_log_ptr=attn._rust_a_log.data_ptr(),
                    dt_bias_ptr=attn._rust_dt_bias.data_ptr(),
                    norm_weight_ptr=attn._rust_norm_weight.data_ptr(),
                    conv_state_ptr=attn._rust_conv_state.data_ptr(),
                    recur_state_ptr=attn._rust_recur_state.data_ptr(),
                    nk=attn.num_k_heads, nv=attn.num_v_heads,
                    dk=attn.k_head_dim, dv=attn.v_head_dim,
                    hr=attn.head_ratio,
                    kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                    scale=attn.scale,
                )
            elif hasattr(attn, 'kv_a_proj'):
                # MLA attention — register projections and absorbed weights
                # AWQ v2: per-channel scales for MLA input projections
                _qa_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "q_a_proj")
                ) else None
                _qproj_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "q_proj")
                ) else None
                _kva_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "kv_a_proj")
                ) else None

                # Q projection weights
                if attn.has_q_lora:
                    qa_w = attn.q_a_proj
                    qb_w = attn.q_b_proj
                    qa_wid = _register_attn_weight(qa_w, layer_idx, "mla", "q_a_proj",
                                                    awq_scales=_qa_scales)
                    qb_wid = _register_attn_weight(qb_w, layer_idx, "mla", "q_b_proj")
                    # q_a layernorm: BF16 → FP32 for Rust per_head_rmsnorm
                    attn._rust_q_a_norm = attn.q_a_norm_weight.float().contiguous().to(device)
                    q_a_norm_ptr = attn._rust_q_a_norm.data_ptr()
                    self._rust_decode_weights.append(attn._rust_q_a_norm)
                    q_proj_wid = None
                else:
                    q_w = attn.q_proj
                    q_proj_wid = _register_attn_weight(q_w, layer_idx, "mla", "q_proj",
                                                        awq_scales=_qproj_scales)
                    qa_wid = None
                    qb_wid = None
                    q_a_norm_ptr = 0

                # KV projection
                kva_w = attn.kv_a_proj
                kva_wid = _register_attn_weight(kva_w, layer_idx, "mla", "kv_a_proj",
                                                 awq_scales=_kva_scales)

                # O projection
                o_w = attn.o_proj
                o_wid = _register_attn_weight(o_w, layer_idx, "mla", "o_proj")

                # Replace MLA attention weight attributes with MarlinWeight for prefill
                from krasis.attention import MarlinWeight
                if attn.has_q_lora:
                    if qa_wid in self._marlin_attn_weights:
                        attn.q_a_proj = MarlinWeight(*self._marlin_attn_weights[qa_wid])
                    if qb_wid in self._marlin_attn_weights:
                        attn.q_b_proj = MarlinWeight(*self._marlin_attn_weights[qb_wid])
                else:
                    if q_proj_wid is not None and q_proj_wid in self._marlin_attn_weights:
                        attn.q_proj = MarlinWeight(*self._marlin_attn_weights[q_proj_wid])
                if kva_wid in self._marlin_attn_weights:
                    attn.kv_a_proj = MarlinWeight(*self._marlin_attn_weights[kva_wid])
                if o_wid in self._marlin_attn_weights:
                    attn.o_proj = MarlinWeight(*self._marlin_attn_weights[o_wid])

                # AWQ v2: fold 1/s into input_norm_weight for MLA layers
                if _layer_awq_scales is not None and (
                    _qa_scales is not None or _qproj_scales is not None
                    or _kva_scales is not None
                ):
                    s = _layer_awq_scales.to(inp_norm.device)
                    inp_norm.data.copy_(
                        (inp_norm.float() / s.float()).to(inp_norm.dtype))
                    logger.debug("AWQ: folded scales into input_norm for MLA layer %d "
                                 "(mean_scale=%.4f)", layer_idx, s.mean().item())

                # kv_a layernorm: BF16 → FP32
                attn._rust_kv_a_norm = attn.kv_a_norm_weight.float().contiguous().to(device)
                self._rust_decode_weights.append(attn._rust_kv_a_norm)

                # w_kc and w_vc: keep as BF16 on GPU (read by CUDA kernels directly)
                # Shape: [num_heads, dim_a, dim_b] — ensure contiguous
                attn._rust_w_kc = attn.w_kc.contiguous().to(device)
                attn._rust_w_vc = attn.w_vc.contiguous().to(device)
                self._rust_decode_weights.extend([attn._rust_w_kc, attn._rust_w_vc])

                # MLA k4v4 KV caches: share the paged byte stores directly.
                # For single-sequence decode, [pages, page_size, row_bytes] is
                # identical to flat [position, row_bytes] memory.
                cache, mla_offset = self._kv_cache_slot_for_layer(layer_idx)
                ckv_layer = cache.ckv_cache[mla_offset]
                kpe_layer = cache.kpe_cache[mla_offset]
                max_seq = cache.max_pages * cache.page_size

                store.register_mla_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                    kv_a_proj_wid=kva_wid, o_proj_wid=o_wid,
                    kv_a_norm_ptr=attn._rust_kv_a_norm.data_ptr(),
                    w_kc_ptr=attn._rust_w_kc.data_ptr(),
                    w_vc_ptr=attn._rust_w_vc.data_ptr(),
                    num_heads=attn.num_heads,
                    kv_lora_rank=attn.kv_lora_rank,  # real kv_lora_rank (256 for Mistral)
                    qk_nope_dim=attn.qk_nope_dim,
                    qk_rope_dim=attn.qk_rope_dim,
                    v_head_dim=attn.v_head_dim,
                    sm_scale=attn.sm_scale,
                    rope_interleave=getattr(self.cfg, 'rope_interleave', True),
                    ckv_cache_ptr=ckv_layer.data_ptr(),
                    kpe_cache_ptr=kpe_layer.data_ptr(),
                    q_a_proj_wid=qa_wid,
                    q_b_proj_wid=qb_wid,
                    q_a_norm_ptr=q_a_norm_ptr,
                    q_proj_wid=q_proj_wid,
                    q_lora_rank=attn.q_lora_rank if attn.has_q_lora else 0,
                    ckv_cache_dim=attn.ckv_dim,  # padded to ≥512 for MLA decode
                )
                self._register_dsa_indexer_layer_on_store(
                    store, layer_idx, attn
                )

                # Set up RoPE tables from first MLA layer
                if not rope_set:
                    cos, sin = attn._get_rope_cos_sin(max_seq)
                    cos_f32 = cos.float().contiguous()
                    sin_f32 = sin.float().contiguous()
                    self._rust_rope_cos = cos_f32
                    self._rust_rope_sin = sin_f32
                    store.set_rope_tables(
                        cos_f32.data_ptr(), sin_f32.data_ptr(),
                        cos_f32.shape[1], max_seq,
                    )
                    rope_set = True

            else:
                # GQA attention — get weights from layer.gqa_weights dict
                # (attn object was removed when FlashInfer was dropped)
                gqa_w = layer.gqa_weights
                if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                    cpu_w = self._stream_attn_cpu[layer_idx]
                    if attn_quant == "bf16":
                        q_w = cpu_w.get("q_proj", gqa_w["q_proj"]).to(device, non_blocking=True)
                        k_w = cpu_w.get("k_proj", gqa_w["k_proj"]).to(device, non_blocking=True)
                        v_w = cpu_w.get("v_proj", gqa_w["v_proj"]).to(device, non_blocking=True)
                        o_w = cpu_w.get("o_proj", gqa_w["o_proj"]).to(device, non_blocking=True)
                        g_w = None
                        if self.cfg.head_wise_attention_gate:
                            g_w = cpu_w.get("g_proj", gqa_w["g_proj"]).to(device, non_blocking=True)
                        self._rust_decode_weights.extend(
                            [w for w in (q_w, k_w, v_w, o_w, g_w) if w is not None]
                        )
                    else:
                        # Quantizing: pass CPU tensors directly, never touch VRAM with BF16
                        q_w = cpu_w.get("q_proj", gqa_w["q_proj"])
                        k_w = cpu_w.get("k_proj", gqa_w["k_proj"])
                        v_w = cpu_w.get("v_proj", gqa_w["v_proj"])
                        o_w = cpu_w.get("o_proj", gqa_w["o_proj"])
                        g_w = cpu_w.get("g_proj", gqa_w["g_proj"]) if self.cfg.head_wise_attention_gate else None
                else:
                    q_w = gqa_w["q_proj"]
                    k_w = gqa_w["k_proj"]
                    v_w = gqa_w["v_proj"]
                    o_w = gqa_w["o_proj"]
                    g_w = gqa_w["g_proj"] if self.cfg.head_wise_attention_gate else None

                # AWQ v2: pass per-channel scales for input projections (q/k/v)
                _q_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "q_proj")
                ) else None
                _k_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "k_proj")
                ) else None
                _v_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "v_proj")
                ) else None

                q_wid = _register_attn_weight(q_w, layer_idx, "gqa", "q_proj",
                                              awq_scales=_q_scales)
                k_wid = _register_attn_weight(k_w, layer_idx, "gqa", "k_proj",
                                              awq_scales=_k_scales)
                v_wid = _register_attn_weight(v_w, layer_idx, "gqa", "v_proj",
                                              awq_scales=_v_scales)
                o_wid = _register_attn_weight(o_w, layer_idx, "gqa", "o_proj")
                g_wid = (
                    _register_attn_weight(g_w, layer_idx, "gqa", "g_proj")
                    if g_w is not None else None
                )

                # Replace attention weight references with MarlinWeight for prefill
                from krasis.attention import MarlinWeight
                gqa_w = layer.gqa_weights if hasattr(layer, 'gqa_weights') else None
                if q_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[q_wid])
                    if gqa_w is not None:
                        gqa_w["q_proj"] = mw
                    elif attn is not None:
                        attn.q_proj = mw
                if k_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[k_wid])
                    if gqa_w is not None:
                        gqa_w["k_proj"] = mw
                    elif attn is not None:
                        attn.k_proj = mw
                if v_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[v_wid])
                    if gqa_w is not None:
                        gqa_w["v_proj"] = mw
                    elif attn is not None:
                        attn.v_proj = mw
                if o_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[o_wid])
                    if gqa_w is not None:
                        gqa_w["o_proj"] = mw
                    elif attn is not None:
                        attn.o_proj = mw
                if g_wid is not None and g_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[g_wid])
                    if gqa_w is not None:
                        gqa_w["g_proj"] = mw

                # AWQ v2: fold 1/s into input_norm_weight for GQA layers
                if _layer_awq_scales is not None and (
                    _q_scales is not None or _k_scales is not None
                    or _v_scales is not None
                ):
                    s = _layer_awq_scales.to(inp_norm.device)
                    inp_norm.data.copy_(
                        (inp_norm.float() / s.float()).to(inp_norm.dtype))
                    logger.debug("AWQ: folded scales into input_norm for GQA layer %d "
                                 "(mean_scale=%.4f)", layer_idx, s.mean().item())

                # QK norm weights are BF16, Rust per_head_rmsnorm kernel needs FP32
                # When streaming, use CPU-pinned source for these small weights
                if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                    cpu_gqa = self._stream_attn_cpu[layer_idx]
                    q_norm_src = cpu_gqa.get("q_norm", gqa_w.get("q_norm"))
                    k_norm_src = cpu_gqa.get("k_norm", gqa_w.get("k_norm"))
                else:
                    q_norm_src = gqa_w.get("q_norm")
                    k_norm_src = gqa_w.get("k_norm")
                if q_norm_src is not None:
                    _rust_q_norm = q_norm_src.float().contiguous().to(device)
                    self._rust_decode_weights.append(_rust_q_norm)
                    q_norm_ptr = _rust_q_norm.data_ptr()
                else:
                    q_norm_ptr = 0
                if k_norm_src is not None:
                    _rust_k_norm = k_norm_src.float().contiguous().to(device)
                    self._rust_decode_weights.append(_rust_k_norm)
                    k_norm_ptr = _rust_k_norm.data_ptr()
                else:
                    k_norm_ptr = 0

                post_norm_ptr = post_norm.data_ptr() if post_norm is not None else 0
                post_norm_size = post_norm.numel() if post_norm is not None else 0
                _gqa_head_dim = self.cfg.gqa_head_dim_for_layer(layer_idx)
                _gqa_sm_scale = 1.0 if self.cfg.gemma4_text else 1.0 / (_gqa_head_dim ** 0.5)
                _gqa_gated = hasattr(self.cfg, 'gated_attention') and self.cfg.gated_attention
                _gqa_num_heads = self.cfg.gqa_num_heads_for_layer(layer_idx)
                _gqa_kv_heads = self.cfg.gqa_num_kv_heads_for_layer(layer_idx)
                max_seq = max(
                    c.max_context_tokens for c in self.kv_caches if c is not None
                )
                _rope_cos_ptr, _rope_sin_ptr, _rope_half, _rope_cos, _rope_sin = self._gqa_rope_table_ptrs(layer_idx, max_seq, device)
                store.register_gqa_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm_ptr, post_attn_norm_size=post_norm_size,
                    q_proj_wid=q_wid, k_proj_wid=k_wid,
                    v_proj_wid=v_wid, o_proj_wid=o_wid,
                    fused_qkv_wid=None,
                    num_heads=_gqa_num_heads,
                    num_kv_heads=_gqa_kv_heads,
                    head_dim=_gqa_head_dim, sm_scale=_gqa_sm_scale,
                    q_norm_ptr=q_norm_ptr, k_norm_ptr=k_norm_ptr,
                    gated=_gqa_gated,
                    head_gate_proj_wid=g_wid,
                    rope_half_dim=_rope_half,
                    rope_cos_ptr=_rope_cos_ptr,
                    rope_sin_ptr=_rope_sin_ptr,
                )
                if self.cfg.is_sliding_attention_layer(layer_idx) and self.cfg.sliding_window:
                    store.set_gqa_sliding_window(layer_idx, int(self.cfg.sliding_window))
                if gqa_w is not None and gqa_w.get("v_norm_no_scale", False):
                    store.set_gqa_v_norm_no_scale(layer_idx, True)
                if self.cfg.gemma4_text or self.cfg.step3_text:
                    store.set_gqa_rope_half_split(layer_idx, True)

                # Set up RoPE tables from first GQA layer
                if not rope_set:
                    if _rope_cos is None or _rope_sin is None:
                        raise RuntimeError(f"Missing RoPE table for GQA layer {layer_idx}")
                    self._rust_rope_cos = _rope_cos
                    self._rust_rope_sin = _rope_sin
                    store.set_rope_tables(
                        _rope_cos.data_ptr(), _rope_sin.data_ptr(),
                        _rope_cos.shape[1], max_seq,
                    )
                    rope_set = True

            if self.cfg.is_glm5_next:
                hc = layer.hyper_connection.tensors
                attn_fn = hc["hc_attn_fn"]
                ffn_fn = hc["hc_ffn_fn"]
                self._rust_decode_weights.extend(
                    [
                        attn_fn,
                        hc["hc_attn_base"],
                        hc["hc_attn_scale"],
                        ffn_fn,
                        hc["hc_ffn_base"],
                        hc["hc_ffn_scale"],
                    ]
                )
                store.register_glm5_hyper_connection(
                    layer_idx=layer_idx,
                    attn_fn_wid=store.register_weight(
                        attn_fn.data_ptr(),
                        attn_fn.shape[0],
                        attn_fn.shape[1],
                        1,
                    ),
                    attn_base_ptr=hc["hc_attn_base"].data_ptr(),
                    attn_base_elems=hc["hc_attn_base"].numel(),
                    attn_scale_ptr=hc["hc_attn_scale"].data_ptr(),
                    attn_scale_elems=hc["hc_attn_scale"].numel(),
                    ffn_fn_wid=store.register_weight(
                        ffn_fn.data_ptr(),
                        ffn_fn.shape[0],
                        ffn_fn.shape[1],
                        1,
                    ),
                    ffn_base_ptr=hc["hc_ffn_base"].data_ptr(),
                    ffn_base_elems=hc["hc_ffn_base"].numel(),
                    ffn_scale_ptr=hc["hc_ffn_scale"].data_ptr(),
                    ffn_scale_elems=hc["hc_ffn_scale"].numel(),
                    mult=self.cfg.hc_mult,
                    sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                    eps=self.cfg.hc_eps,
                )

            # Register MLP type
            if layer.is_moe and self.cfg.gemma4_text and layer.dense_mlp is not None:
                gp = self._dense_mlp_tensor(layer.dense_mlp["gate_proj"])
                up = self._dense_mlp_tensor(layer.dense_mlp["up_proj"])
                dp = self._dense_mlp_tensor(layer.dense_mlp["down_proj"])
                self._rust_decode_weights.extend([gp, up, dp])
                gp_wid = store.register_weight(gp.data_ptr(), gp.shape[0], gp.shape[1], 0)
                up_wid = store.register_weight(up.data_ptr(), up.shape[0], up.shape[1], 0)
                dp_wid = store.register_weight(dp.data_ptr(), dp.shape[0], dp.shape[1], 0)
                _gemma_refs = []
                def _ptr(t):
                    if t is None:
                        return 0
                    tt = t.contiguous().to(device)
                    _gemma_refs.append(tt)
                    return tt.data_ptr()
                store.register_gemma4_moe_layer(
                    layer_idx=layer_idx,
                    dense_gate_proj_wid=gp_wid,
                    dense_up_proj_wid=up_wid,
                    dense_down_proj_wid=dp_wid,
                    pre_ffn_norm_ptr=_ptr(getattr(layer, "pre_ffn_norm_weight", None)),
                    post_ffn_norm_ptr=_ptr(getattr(layer, "post_ffn_norm_weight", None)),
                    post_ffn_norm1_ptr=_ptr(getattr(layer, "post_ffn_norm1_weight", None)),
                    post_ffn_norm2_ptr=_ptr(getattr(layer, "post_ffn_norm2_weight", None)),
                    pre_ffn_norm2_ptr=_ptr(getattr(layer, "pre_ffn_norm2_weight", None)),
                    layer_scalar_ptr=_ptr(getattr(layer, "layer_scalar", None)),
                    router_input_scale_ptr=_ptr(getattr(layer, "router_input_scale", None)),
                    router_per_expert_scale_ptr=_ptr(getattr(layer, "router_per_expert_scale", None)),
                )
                self._rust_decode_weights.extend(_gemma_refs)
            elif layer.is_moe:
                store.register_mlp(layer_idx, "moe")
            elif layer.dense_mlp is not None:
                gp = self._dense_mlp_tensor(layer.dense_mlp["gate_proj"])
                up = self._dense_mlp_tensor(layer.dense_mlp["up_proj"])
                dp = self._dense_mlp_tensor(layer.dense_mlp["down_proj"])
                self._rust_decode_weights.extend([gp, up, dp])
                gp_wid = store.register_weight(gp.data_ptr(), gp.shape[0], gp.shape[1], 0)
                up_wid = store.register_weight(up.data_ptr(), up.shape[0], up.shape[1], 0)
                dp_wid = store.register_weight(dp.data_ptr(), dp.shape[0], dp.shape[1], 0)
                store.register_mlp(
                    layer_idx,
                    "dense",
                    gate_proj_wid=gp_wid,
                    up_proj_wid=up_wid,
                    down_proj_wid=dp_wid,
                    swiglu_limit=float(self.cfg.swiglu_limit_for_layer(layer_idx)),
                    deepseek_clamp=bool(self.cfg.is_glm5_next),
                )
            else:
                store.register_mlp(layer_idx, "none")

        if self.cfg.is_deepseek_v4:
            hc_head = self._deepseek_v4_hc_head
            if hc_head is None:
                raise RuntimeError("DeepSeek-V4 final hyper-connection head was not loaded")
            store.register_deepseek_v4_head(
                fn_wid=_register_attn_weight(
                    hc_head["hc_head_fn"], -1, "deepseek_v4", "hc_head_fn"
                ),
                base_ptr=hc_head["hc_head_base"].data_ptr(),
                base_elems=hc_head["hc_head_base"].numel(),
                scale_ptr=hc_head["hc_head_scale"].data_ptr(),
                scale_elems=hc_head["hc_head_scale"].numel(),
                mult=self.cfg.hc_mult,
                eps=self.cfg.hc_eps,
            )
            registered_v4_layers = store.finalize_deepseek_v4_layers()
            if registered_v4_layers != self.cfg.num_hidden_layers:
                raise RuntimeError(
                    "DeepSeek-V4 runtime registration count mismatch: "
                    f"{registered_v4_layers}/{self.cfg.num_hidden_layers}"
                )
            logger.info(
                "DeepSeek-V4 runtime registration complete: %d/%d layers",
                registered_v4_layers,
                self.cfg.num_hidden_layers,
            )
        elif self.cfg.is_glm5_next:
            store.register_glm5_hyper_head_mean(mult=self.cfg.hc_mult)
            registered_glm5_layers = store.finalize_glm5_layers()
            if registered_glm5_layers != self.cfg.num_hidden_layers:
                raise RuntimeError(
                    "GLM-5.3 runtime registration count mismatch: "
                    f"{registered_glm5_layers}/{self.cfg.num_hidden_layers}"
                )
            logger.info(
                "GLM-5.3 runtime registration complete: %d/%d layers",
                registered_glm5_layers,
                self.cfg.num_hidden_layers,
            )

        self._finalize_mamba2_projection_int4_cache()

        # Register MoE expert data from engine (all MoE layers at once)
        if self.krasis_engine is not None:
            store.setup_from_engine(self.krasis_engine)
            self._register_deepseek_v4_hash_tables(
                store, device, self._rust_decode_weights
            )
            if self.cfg.swiglu_limits or self.cfg.swiglu_limits_shared:
                for layer_idx, layer in enumerate(self.layers):
                    if layer.is_moe:
                        routed_limit = float(self.cfg.swiglu_limit_for_layer(layer_idx))
                        shared_limit = float(self.cfg.shared_swiglu_limit_for_layer(layer_idx))
                        if routed_limit or shared_limit:
                            store.set_moe_swiglu_limits(
                                layer_idx=layer_idx,
                                swiglu_limit=routed_limit,
                                shared_swiglu_limit=shared_limit,
                            )

        # Register Nemotron MoE config (relu2, ungated, latent projections) — must come
        # after setup_from_engine which populates moe_layers[abs_layer_idx].
        if self.cfg.model_type == "nemotron_h":
            for layer_idx, layer in enumerate(self.layers):
                if layer.layer_type == "moe":
                    latent = layer.latent_proj
                    if latent is not None:
                        fc1_down = latent["fc1_latent_proj"]
                        fc2_up = latent["fc2_latent_proj"]
                        self._keep_rust_decode_weight(
                            "nemotron_latent_projection_bf16", fc1_down, fc2_up
                        )
                        ld_wid = store.register_weight(
                            fc1_down.data_ptr(), fc1_down.shape[0], fc1_down.shape[1], 0)
                        lu_wid = store.register_weight(
                            fc2_up.data_ptr(), fc2_up.shape[0], fc2_up.shape[1], 0)
                    else:
                        ld_wid = None
                        lu_wid = None
                    store.set_moe_nemotron_config(
                        layer_idx=layer_idx,
                        activation_type=1,
                        gated_experts=False,
                        latent_down_wid=ld_wid,
                        latent_up_wid=lu_wid,
                        moe_input_size=self.cfg.moe_latent_size,
                    )
                    logger.info("Set Nemotron MoE config for layer %d (latent_down=%s, latent_up=%s)",
                                layer_idx, ld_wid, lu_wid)
        elif self.cfg.gemma4_text:
            for layer_idx, layer in enumerate(self.layers):
                if layer.is_moe:
                    store.set_moe_nemotron_config(
                        layer_idx=layer_idx,
                        activation_type=2,
                        gated_experts=True,
                        latent_down_wid=None,
                        latent_up_wid=None,
                        moe_input_size=0,
                    )
            logger.info("Set Gemma4 MoE config for %d layers (gelu_tanh gated experts)", sum(1 for l in self.layers if l.is_moe))

        # Register shared_expert_gate weights (sigmoid gate for shared expert output)
        # These are BF16 tensors loaded by Python, not in the Rust engine cache
        self._rust_shared_gate_refs = []
        for layer_idx, layer in enumerate(self.layers):
            if layer.is_moe and layer.shared_expert_gate is not None:
                sg = layer.shared_expert_gate
                sg_wid = store.register_weight(sg.data_ptr(), sg.shape[0], sg.shape[1], 0)
                store.set_moe_shared_gate_wid(layer_idx, sg_wid)
                self._rust_shared_gate_refs.append(sg)  # prevent GC
                logger.info("Registered shared_expert_gate for layer %d: wid=%d shape=%s",
                            layer_idx, sg_wid, sg.shape)

        # Register shared KV cache pointers — Rust decode reads/writes the
        # same GPU buffers that Rust prefill writes. No separate
        # allocation, no D2D export copy between prefill and decode.
        cache = self.kv_caches[0]
        if cache is not None and cache.kv_format == 4 and cache.k_radius_cache is not None:
            # tq4 KV cache: K norm + packed Lloyd-Max indices, V scale/zero + packed uniform indices.
            tq4_ptrs = []
            self._tq4_sign_refs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    kn = cache.k_radius_cache[gqa_cache_idx]
                    ki = cache.k_angles_cache[gqa_cache_idx]
                    vm = cache.v_radius_cache[gqa_cache_idx]
                    vi = cache.v_angles_cache[gqa_cache_idx]
                    signs = _tq4_wht_signs(layer_idx, cache.gqa_head_dim, cache.device)
                    self._tq4_sign_refs.append(signs)
                    gqa_cache_idx += 1
                    tq4_ptrs.append((layer_idx,
                                     kn.data_ptr(), ki.data_ptr(),
                                     vm.data_ptr(), vi.data_ptr(),
                                     signs.data_ptr()))
            max_seq = cache.max_pages * cache.page_size
            store.set_kv_cache_ptrs_tq4(tq4_ptrs, max_seq, cache.num_kv_heads, cache.gqa_head_dim)
            logger.info("Shared tq4 KV cache: %d GQA layers, max_seq=%d (%d pages × %d), heads=%d head_dim=%d",
                        len(tq4_ptrs), max_seq, cache.max_pages, cache.page_size,
                        cache.num_kv_heads, cache.gqa_head_dim)
        elif cache is not None and cache.kv_format in (5, 6, 7, 8, 9) and cache.k_radius_cache is not None:
            # k4v4/k6v4/k7v4/k6v6/k8v6 KV cache: K is blockwise integer + BF16
            # scale. k4v4/k6v4/k7v4 use Polar4 V; k6v6/k8v6 use integer V.
            kintv4_ptrs = []
            max_seq_by_layer = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    kr = cache.k_radius_cache[gqa_cache_idx]
                    ka = cache.k_angles_cache[gqa_cache_idx]
                    vr = cache.v_radius_cache[gqa_cache_idx]
                    va = cache.v_angles_cache[gqa_cache_idx]
                    gqa_cache_idx += 1
                    kintv4_ptrs.append((layer_idx,
                                        kr.data_ptr(), ka.data_ptr(),
                                        vr.data_ptr(), va.data_ptr()))
                    max_seq_by_layer.append((layer_idx, int(kr.shape[0] * kr.shape[1])))
            max_seq = cache.max_pages * cache.page_size
            num_blocks = cache.max_num_blocks()
            if cache.kv_format == 8:
                store.set_kv_cache_ptrs_k8v6(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k8v6"
            elif cache.kv_format == 7:
                store.set_kv_cache_ptrs_k6v6(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k6v6"
            elif cache.kv_format == 6:
                store.set_kv_cache_ptrs_k7v4(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k7v4"
            elif cache.kv_format == 9:
                store.set_kv_cache_ptrs_k4v4(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k4v4"
            else:
                store.set_kv_cache_ptrs_k6v4(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k6v4"
            store.set_kv_cache_max_seq_by_layer(max_seq_by_layer)
            logger.info("Shared %s KV cache: %d GQA layers, max_seq=%d (%d pages × %d), %d blocks",
                        fmt, len(kintv4_ptrs), max_seq, cache.max_pages, cache.page_size, num_blocks)
        elif cache is not None and cache.kv_format == 3 and cache.k_cache is not None and cache.v_radius_cache is not None:
            # k8v4 KV cache: K is FP8, V is Polar4 radius + angle.
            k8v4_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    k_layer = cache.k_cache[gqa_cache_idx]
                    vr = cache.v_radius_cache[gqa_cache_idx]
                    va = cache.v_angles_cache[gqa_cache_idx]
                    gqa_cache_idx += 1
                    k8v4_ptrs.append((layer_idx,
                                      k_layer.data_ptr(),
                                      vr.data_ptr(), va.data_ptr()))
            max_seq = cache.max_pages * cache.page_size
            num_blocks = cache.max_num_blocks()
            store.set_kv_cache_ptrs_k8v4(k8v4_ptrs, max_seq, num_blocks)
            logger.info("Shared k8v4 KV cache: %d GQA layers, max_seq=%d (%d pages × %d), %d V blocks",
                        len(k8v4_ptrs), max_seq, cache.max_pages, cache.page_size, num_blocks)
        elif cache is not None and cache.kv_format == 2 and cache.k_radius_cache is not None:
            # Polar4 KV cache: 4 pointer sets (radius + angles for K and V)
            polar4_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    kr = cache.k_radius_cache[gqa_cache_idx]
                    vr = cache.v_radius_cache[gqa_cache_idx]
                    ka = cache.k_angles_cache[gqa_cache_idx]
                    va = cache.v_angles_cache[gqa_cache_idx]
                    gqa_cache_idx += 1
                    polar4_ptrs.append((layer_idx,
                                        kr.data_ptr(), vr.data_ptr(),
                                        ka.data_ptr(), va.data_ptr()))
            max_seq = cache.max_pages * cache.page_size
            num_blocks = cache.max_num_blocks()
            store.set_kv_cache_ptrs_polar4(polar4_ptrs, max_seq, num_blocks)
            logger.info("Shared Polar4 KV cache: %d GQA layers, max_seq=%d (%d pages × %d), %d blocks",
                        len(polar4_ptrs), max_seq, cache.max_pages, cache.page_size, num_blocks)
        elif cache is not None and cache.k_cache is not None:
            kv_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                # Only full GQA attention layers have KV cache slots.
                # Skip: linear_attention (no KV), MLA (has kv_a_proj), mamba2 (SSM state), moe-only (no attention).
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    k_layer = cache.k_cache[gqa_cache_idx]  # [max_pages, page_size, nkv, hd]
                    v_layer = cache.v_cache[gqa_cache_idx]
                    gqa_cache_idx += 1
                    kv_ptrs.append((layer_idx, k_layer.data_ptr(), v_layer.data_ptr()))
            max_seq = cache.max_pages * cache.page_size
            if cache.kv_format == 0:
                store.set_kv_cache_ptrs_bf16(kv_ptrs, max_seq)
                cache_label = "BF16"
            else:
                store.set_kv_cache_ptrs(kv_ptrs, max_seq)
                cache_label = "FP8"
            logger.info("Shared %s KV cache: %d GQA layers, max_seq=%d (%d pages × %d)",
                        cache_label, len(kv_ptrs), max_seq, cache.max_pages, cache.page_size)
        elif cache is not None and cache.ckv_cache is not None:
            # Native MLA layers own their compact cache pointers in their Rust
            # layer descriptors. Finalize the shared format/capacity contract
            # without mislabelling those byte stores as legacy FP8 GQA cache.
            max_seq = cache.max_pages * cache.page_size
            store.set_native_mla_kv_cache(cache.kv_format, max_seq)
            logger.info("Native MLA %s KV cache: max_seq=%d (%d pages × %d)",
                        cache.kv_format_str, max_seq, cache.max_pages, cache.page_size)

        if hqq_active and not rope_set:
            max_seq = max(
                c.max_context_tokens for c in self.kv_caches if c is not None
            )
            rope_half = self.cfg.rotary_dim // 2
            cos_f32 = None
            sin_f32 = None
            reused_hqq_gqa_rope = False
            if not self.cfg.gemma4_text:
                cache = getattr(self, "_hqq_gqa_rope_table_cache", {}) or {}
                expected_cache_key = (
                    device.type,
                    device.index,
                    max_seq,
                    rope_half,
                    float(self.cfg.rope_theta),
                    self.cfg.gqa_head_dim,
                    False,
                    "{}",
                )
                cached = cache.get(expected_cache_key)
                if cached is not None:
                    cos_f32, sin_f32 = cached
                    reused_hqq_gqa_rope = True
            if cos_f32 is None or sin_f32 is None:
                inv_freq = 1.0 / (self.cfg.rope_theta ** (
                    torch.arange(0, rope_half * 2, 2, dtype=torch.float32) / (rope_half * 2)))
                t = torch.arange(max_seq, dtype=torch.float32)
                freqs = torch.outer(t, inv_freq)
                cos_f32 = freqs.cos().contiguous().to(device)
                sin_f32 = freqs.sin().contiguous().to(device)
            self._rust_rope_cos = cos_f32
            self._rust_rope_sin = sin_f32
            store.set_rope_tables(
                cos_f32.data_ptr(), sin_f32.data_ptr(),
                cos_f32.shape[1], max_seq,
            )
            rope_set = True
            logger.info(
                "HQQ attention RoPE tables registered: max_seq=%d rope_half=%d reused_hqq_gqa_cache=%s",
                max_seq,
                rope_half,
                reused_hqq_gqa_rope,
            )

        if hqq_active:
            logger.info(
                "HQQ attention execution descriptors retained after shared decode setup on cuda:%d: %d layers registered, %d MB validated cache, %d tensors loaded.",
                gpu_idx,
                hqq_registered_layers,
                hqq_cache_bytes >> 20,
                self._hqq_attention_loaded_tensors,
            )

        if self.cfg.is_dsa:
            staged_dsa = self._stage_dsa_indexer_resources_on_store(
                store,
                device,
                self._rust_decode_weights,
                0,
                len(self.layers),
            )
            logger.info(
                "Primary full-prefill DSA resources staged on cuda:%d: %d "
                "owner/replica resources for layers [0,%d)",
                gpu_idx,
                staged_dsa,
                len(self.layers),
            )

        self._gpu_decode_store = store

        # Enable per-component timing if KRASIS_DECODE_TIMING=1
        if os.environ.get("KRASIS_DECODE_TIMING", "") == "1":
            store.set_timing(True)
            logger.info("GPU decode timing enabled (KRASIS_DECODE_TIMING=1)")

        # Log actual attention VRAM after AWQ quantization (if applicable)
        if self.quant_cfg.attention == "awq":
            attn_mb = self._estimate_attention_vram() >> 20
            dev = torch.device(self.ranks[0].device)
            free_mb = torch.cuda.mem_get_info(dev)[0] >> 20
            logger.info("Attention after AWQ quantization: %d MB, GPU free: %d MB", attn_mb, free_mb)
            print(f"  \033[0;32mAttention after AWQ: {attn_mb} MB, {free_mb} MB free\033[0m", flush=True)

        logger.info("GPU decode store configured: %d layers, store_addr=%d",
                     len(self.layers), store.gpu_store_addr())

        self._register_primary_sequence_state_inventory(store, device)

        # Single-slot AWQ: Marlin data is in the GPU slots from registration.
        # PyTorch tensors stay on GPU permanently (they ARE the slots).
        # Rust handles swapping slot contents between Marlin and simple INT4.

        return store

    def setup_dspark_gpu_store(self, store, mode: str) -> dict:
        """Load and register the checkpoint-owned D-Spark dense graph.

        This is startup-only Python orchestration. All request-time draft and
        verification execution remains in Rust/CUDA. D-Spark uses its own
        short BF16 attention rings while the target retains its configured
        Native sequence-state format.
        """
        if not self.cfg.is_deepseek_v4:
            raise RuntimeError("D-Spark setup requires DeepSeek-V4")
        targets = tuple(self.cfg.dspark_target_layer_ids or ())
        if not targets:
            raise RuntimeError("Checkpoint has no validated D-Spark target layers")
        if mode not in ("resident", "shared"):
            raise ValueError(f"Unsupported D-Spark mode {mode!r}")
        if store is not getattr(self, "_gpu_decode_store", None):
            raise RuntimeError("D-Spark must attach to the configured primary decode store")

        device = torch.device(self.ranks[0].device)
        loader = WeightLoader(self.cfg, self.quant_cfg)
        stage_data = []
        try:
            for stage_idx in range(len(targets)):
                stage_data.append(loader.load_dspark_stage(stage_idx, device))
        finally:
            loader.close()

        keepalive = self._rust_decode_weights

        def register(weight: torch.Tensor, label: str, dtype: int = 0) -> int:
            if not isinstance(weight, torch.Tensor) or not weight.is_cuda:
                raise RuntimeError(f"D-Spark tensor {label} is not GPU resident")
            if not weight.is_contiguous():
                weight = weight.contiguous()
            keepalive.keep(f"dspark.{label}", weight)
            return store.register_weight(
                weight.data_ptr(), int(weight.shape[0]), int(weight.shape[1]), dtype
            )

        target_cache, target_offset = self._kv_cache_slot_for_layer(0)
        target_v4_cache = target_cache.get_deepseek_v4_layer_caches(target_offset)
        logical_max_seq = int(target_cache.max_context_tokens)
        rope_cos_ptr, rope_sin_ptr, rope_rows, rope_cos, rope_sin = (
            self._deepseek_v4_rope_table_ptrs(logical_max_seq, 0, device)
        )
        keepalive.keep("dspark.rope", rope_cos, rope_sin)

        dspark_caches = []
        main_proj_wid = None
        main_norm = None
        final_norm = None
        head_hc = None
        markov_w1_wid = None
        markov_w2_wid = None
        confidence_wid = None
        for stage_idx, data in enumerate(stage_data):
            layer_idx = self.cfg.num_hidden_layers + stage_idx
            attention = data["attention"]
            hc = data["hyper_connection"]
            cache = torch.zeros(
                (self.cfg.sliding_window, self.cfg.attention_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            dspark_caches.append(cache)
            keepalive.keep(
                f"dspark.stage{stage_idx}.vectors",
                data["input_norm"],
                data["post_attn_norm"],
                data["gate_bias"],
                attention["attn_sink"],
                attention["q_norm"],
                attention["kv_norm"],
                hc["hc_attn_base"],
                hc["hc_attn_scale"],
                hc["hc_ffn_base"],
                hc["hc_ffn_scale"],
                cache,
            )
            wids = {
                name: register(
                    attention[name], f"stage{stage_idx}.attention.{name}"
                )
                for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
            }
            store.register_deepseek_v4_layer(
                layer_idx=layer_idx,
                input_norm_ptr=data["input_norm"].data_ptr(),
                input_norm_size=data["input_norm"].numel(),
                post_attn_norm_ptr=data["post_attn_norm"].data_ptr(),
                post_attn_norm_size=data["post_attn_norm"].numel(),
                wq_a_wid=wids["wq_a"],
                wq_b_wid=wids["wq_b"],
                wkv_wid=wids["wkv"],
                wo_a_wid=wids["wo_a"],
                wo_b_wid=wids["wo_b"],
                attn_sink_ptr=attention["attn_sink"].data_ptr(),
                attn_sink_elems=attention["attn_sink"].numel(),
                q_norm_ptr=attention["q_norm"].data_ptr(),
                q_norm_elems=attention["q_norm"].numel(),
                kv_norm_ptr=attention["kv_norm"].data_ptr(),
                kv_norm_elems=attention["kv_norm"].numel(),
                num_heads=self.cfg.num_attention_heads,
                head_dim=self.cfg.attention_head_dim,
                rope_dim=self.cfg.qk_rope_head_dim,
                o_groups=self.cfg.o_groups,
                sliding_window=self.cfg.sliding_window,
                compress_ratio=0,
                compress_rope_theta=self.cfg.compress_rope_theta,
                raw_cache_ptr=cache.data_ptr(),
                raw_cache_elems=cache.numel(),
                cache_format="bf16",
                raw_native_codes_ptr=0,
                raw_native_codes_elems=0,
                raw_native_scale_exponents_ptr=0,
                raw_native_scale_exponents_elems=0,
                raw_native_tail_ptr=0,
                raw_native_tail_elems=0,
                native_block_size=int(target_v4_cache["fp8_block_size"]),
                rope_cos_ptr=rope_cos_ptr,
                rope_sin_ptr=rope_sin_ptr,
                rope_rows=rope_rows,
                logical_max_seq=logical_max_seq,
            )
            store.register_sequence_state_allocation(
                name=f"dspark.stage{stage_idx}.raw_ring",
                kind="dspark_raw_ring",
                layer_idx=layer_idx,
                ptr=cache.data_ptr(),
                storage_bytes=cache.numel() * cache.element_size(),
                dtype="bfloat16",
                element_size=cache.element_size(),
                shape=list(cache.shape),
                strides_bytes=[
                    int(stride) * cache.element_size() for stride in cache.stride()
                ],
                growth_mode="fixed",
            )
            store.register_deepseek_v4_hyper_connection(
                layer_idx=layer_idx,
                attn_fn_wid=register(
                    hc["hc_attn_fn"], f"stage{stage_idx}.hc_attn_fn", 1
                ),
                attn_base_ptr=hc["hc_attn_base"].data_ptr(),
                attn_base_elems=hc["hc_attn_base"].numel(),
                attn_scale_ptr=hc["hc_attn_scale"].data_ptr(),
                attn_scale_elems=hc["hc_attn_scale"].numel(),
                ffn_fn_wid=register(
                    hc["hc_ffn_fn"], f"stage{stage_idx}.hc_ffn_fn", 1
                ),
                ffn_base_ptr=hc["hc_ffn_base"].data_ptr(),
                ffn_base_elems=hc["hc_ffn_base"].numel(),
                ffn_scale_ptr=hc["hc_ffn_scale"].data_ptr(),
                ffn_scale_elems=hc["hc_ffn_scale"].numel(),
                mult=self.cfg.hc_mult,
                sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                eps=self.cfg.hc_eps,
            )
            auxiliary_layer_idx = store.register_dspark_moe_stage(
                stage_idx=stage_idx,
                gate_ptr=data["gate"].data_ptr(),
                gate_rows=int(data["gate"].shape[0]),
                gate_cols=int(data["gate"].shape[1]),
                correction_bias_ptr=data["gate_bias"].data_ptr(),
                correction_bias_elems=data["gate_bias"].numel(),
            )
            if auxiliary_layer_idx != layer_idx:
                raise RuntimeError(
                    f"D-Spark stage {stage_idx} registered at {auxiliary_layer_idx}, "
                    f"expected {layer_idx}"
                )
            keepalive.keep(f"dspark.stage{stage_idx}.router", data["gate"])

            if stage_idx == 0:
                main_proj_wid = register(data["main_proj"], "main_proj")
                main_norm = data["main_norm"]
                keepalive.keep("dspark.main_norm", main_norm)
            if stage_idx == len(stage_data) - 1:
                final_norm = data["final_norm"]
                head_hc = data["head_hyper_connection"]
                markov_w1_wid = register(data["markov_w1"], "markov_w1")
                markov_w2_wid = register(data["markov_w2"], "markov_w2")
                confidence_wid = register(data["confidence"], "confidence")
                keepalive.keep(
                    "dspark.final_vectors",
                    final_norm,
                    head_hc["hc_head_base"],
                    head_hc["hc_head_scale"],
                )

        if any(
            value is None
            for value in (
                main_proj_wid,
                main_norm,
                final_norm,
                head_hc,
                markov_w1_wid,
                markov_w2_wid,
                confidence_wid,
            )
        ):
            raise RuntimeError("D-Spark checkpoint graph did not yield all required heads")
        registration = store.finalize_dspark_graph(
            main_proj_wid=main_proj_wid,
            main_norm_ptr=main_norm.data_ptr(),
            main_norm_elems=main_norm.numel(),
            final_norm_ptr=final_norm.data_ptr(),
            final_norm_elems=final_norm.numel(),
            head_fn_wid=register(head_hc["hc_head_fn"], "hc_head_fn", 1),
            head_base_ptr=head_hc["hc_head_base"].data_ptr(),
            head_base_elems=head_hc["hc_head_base"].numel(),
            head_scale_ptr=head_hc["hc_head_scale"].data_ptr(),
            head_scale_elems=head_hc["hc_head_scale"].numel(),
            head_mult=self.cfg.hc_mult,
            head_eps=self.cfg.hc_eps,
            markov_w1_wid=markov_w1_wid,
            markov_w2_wid=markov_w2_wid,
            confidence_wid=confidence_wid,
        )
        required_sequence_state = int(store.finalize_sequence_state_inventory())
        if required_sequence_state <= 0:
            raise RuntimeError("D-Spark sequence-state inventory is empty")
        self._dspark_caches = dspark_caches
        parsed = json.loads(registration)
        logger.info("D-Spark GPU graph setup complete: %s", parsed)
        return parsed

    def _register_mamba2_layer(self, store, layer_idx, layer, device):
        """Register a Mamba2 SSM layer with the Rust decode store."""
        inp_norm = layer.input_norm_weight
        m2 = layer.mamba2_weights
        cfg = self.cfg

        # Mamba2 projection weights: in_proj [in_proj_out, hidden_size], out_proj [hidden_size, d_inner]
        # Register as cached INT4 Marlin/simple single-slot weights when enabled.
        in_proj = m2["in_proj"]
        out_proj = m2["out_proj"]
        if getattr(self, "_mamba2_projection_int4_enabled", False):
            in_proj_wid, in_proj_marlin = self._register_mamba2_projection_int4(
                store, layer_idx, "in_proj", in_proj, device
            )
            out_proj_wid, out_proj_marlin = self._register_mamba2_projection_int4(
                store, layer_idx, "out_proj", out_proj, device
            )
            for tensor_name, marlin_weight in (
                ("in_proj", in_proj_marlin),
                ("out_proj", out_proj_marlin),
            ):
                tensor = in_proj if tensor_name == "in_proj" else out_proj
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    self._mamba2_projection_bf16_released_bytes += tensor.numel() * tensor.element_size()
                    self._mamba2_projection_bf16_released_tensors += 1
                m2[tensor_name] = marlin_weight
            del in_proj, out_proj
        else:
            self._keep_rust_decode_weight("mamba2_projection_bf16", in_proj, out_proj)
            in_proj_wid = store.register_weight(
                in_proj.data_ptr(), in_proj.shape[0], in_proj.shape[1], 0)
            out_proj_wid = store.register_weight(
                out_proj.data_ptr(), out_proj.shape[0], out_proj.shape[1], 0)

        # Conv1d weight: [conv_dim, conv_kernel] FP32 on GPU
        conv_w = m2["conv1d_weight"].float().contiguous().to(device)
        self._keep_rust_decode_weight("mamba2_conv_fp32", conv_w)

        # Conv1d bias: [conv_dim] FP32 on GPU (optional)
        conv_bias = m2.get("conv1d_bias")
        if conv_bias is not None:
            conv_bias = conv_bias.float().contiguous().to(device)
            self._keep_rust_decode_weight("mamba2_conv_bias_fp32", conv_bias)
            store.set_mamba2_conv_bias(layer_idx, conv_bias.data_ptr())

        # A_log, D, dt_bias, norm: all FP32 on GPU
        a_log = m2["A_log"].float().contiguous().to(device)
        d_param = m2["D"].float().contiguous().to(device)
        dt_bias = m2["dt_bias"].float().contiguous().to(device)
        norm_w = m2["norm_weight"].float().contiguous().to(device)
        self._keep_rust_decode_weight(
            "mamba2_params_fp32", a_log, d_param, dt_bias, norm_w
        )

        # Allocate SSM state buffers on GPU (persistent across tokens)
        # Conv state: [conv_dim, conv_kernel] FP32
        # SSM state: [num_heads, head_dim, state_size] FP32
        conv_dim = cfg.mamba_conv_dim
        conv_state = torch.zeros(conv_dim, cfg.mamba_conv_kernel,
                                 dtype=torch.float32, device=device)
        ssm_state = torch.zeros(cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size,
                                dtype=torch.float32, device=device)
        self._keep_rust_decode_weight("mamba2_state_fp32", conv_state, ssm_state)
        # Save references for prefill -> decode state transfer
        if not hasattr(self, '_mamba2_decode_states'):
            self._mamba2_decode_states = {}
        self._mamba2_decode_states[layer_idx] = {
            'conv_state': conv_state,
            'ssm_state': ssm_state,
        }

        store.register_mamba2_layer(
            layer_idx=layer_idx,
            input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
            post_attn_norm_ptr=0, post_attn_norm_size=0,  # Nemotron: no post_attn_norm
            in_proj_wid=in_proj_wid, out_proj_wid=out_proj_wid,
            conv_weight_ptr=conv_w.data_ptr(),
            a_ptr=a_log.data_ptr(), d_ptr=d_param.data_ptr(),
            dt_bias_ptr=dt_bias.data_ptr(), norm_weight_ptr=norm_w.data_ptr(),
            conv_state_ptr=conv_state.data_ptr(), ssm_state_ptr=ssm_state.data_ptr(),
            num_heads=cfg.mamba_num_heads, head_dim=cfg.mamba_head_dim,
            state_size=cfg.ssm_state_size,
            expand=cfg.mamba_expand, conv_kernel=cfg.mamba_conv_kernel,
            conv_dim=cfg.mamba_conv_dim,
        )
        # Set n_groups for B/C sharing (same for all Mamba2 layers)
        store.set_mamba2_n_groups(cfg.mamba_n_groups)
        store.set_mamba2_chunk_size(cfg.mamba_chunk_size)

        logger.info("Registered Mamba2 layer %d (conv_dim=%d, heads=%d, state=%d, chunk=%d)",
                     layer_idx, conv_dim, cfg.mamba_num_heads, cfg.ssm_state_size,
                     cfg.mamba_chunk_size)

    def _register_nemotron_moe_only_layer(self, store, layer_idx, layer, device):
        """Register a Nemotron MoE-only layer (no attention) with the Rust decode store.

        These layers have: input_norm → LatentMoE → residual (no attention, no post_attn_norm).
        Since the Rust decode loop expects input_norm + attn + post_attn_norm + MLP, we register
        the layer with a 'None' attention type and the MoE as the MLP.
        """
        inp_norm = layer.input_norm_weight

        # Register as GQA with 0 heads (skips attention computation).
        # post_attn_norm ptr=0/size=0 tells Rust to skip post-norm (Nemotron has no post_attn_norm).
        store.register_gqa_layer(
            layer_idx=layer_idx,
            input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
            post_attn_norm_ptr=0, post_attn_norm_size=0,
            q_proj_wid=0, k_proj_wid=0, v_proj_wid=0, o_proj_wid=0,
            fused_qkv_wid=None,
            num_heads=0, num_kv_heads=0, head_dim=0, sm_scale=0.0,
            q_norm_ptr=0, k_norm_ptr=0, gated=False,
            rope_half_dim=0,
        )

        # Register MoE placeholder — actual expert data is wired by setup_from_engine,
        # and Nemotron config (relu2, latent projections) is set after that.
        store.register_mlp(layer_idx, "moe")

        logger.info("Registered Nemotron MoE-only layer %d (MoE config deferred to post-engine setup)",
                     layer_idx)

    def setup_gpu_peer_expert_store(self, gpu_idx: int) -> "GpuDecodeStore":
        """Create a routed-expert-only store on an auxiliary GPU.

        The peer has no attention, KV, embedding, LM-head, or routing role.  It
        duplicates only the heat-ranked routed experts loaded into its own HCS;
        canonical expert bytes remain owned by the primary engine in host RAM.
        """
        self._require_supported_runtime_features()
        from krasis import GpuDecodeStore

        if self.krasis_engine is None:
            raise RuntimeError("Peer expert setup requires a loaded Krasis engine")
        store = GpuDecodeStore(int(gpu_idx))
        store.setup_peer_from_engine(self.krasis_engine)
        if self.cfg.swiglu_limits or self.cfg.swiglu_limits_shared:
            for layer_idx, layer in enumerate(self.layers):
                if not layer.is_moe:
                    continue
                routed_limit = float(self.cfg.swiglu_limit_for_layer(layer_idx))
                shared_limit = float(self.cfg.shared_swiglu_limit_for_layer(layer_idx))
                if routed_limit or shared_limit:
                    store.set_moe_swiglu_limits(
                        layer_idx=layer_idx,
                        swiglu_limit=routed_limit,
                        shared_swiglu_limit=shared_limit,
                    )
        return store

    def setup_gpu_decode_store_aux(self, gpu_idx: int, split_layer: int, layer_end: int = 0) -> "GpuDecodeStore":
        """Create an auxiliary GpuDecodeStore for multi-GPU decode.

        Sets up a store on gpu_idx for layers [split_layer..layer_end).
        If layer_end=0, defaults to num_layers (last GPU in pipeline).
        Registers attention weights (copied to aux GPU), KV cache (allocated on
        aux GPU), MoE layers, final norm, LM head, and RoPE on the aux GPU.
        Only the last segment (layer_end == num_layers) gets real final norm + LM head.
        """
        self._require_supported_runtime_features()
        if self.cfg.is_deepseek_v4:
            raise RuntimeError(
                "DeepSeek-V4 serial layer-split decode is not implemented. Its custom "
                "compressed-attention graph and Native sequence-state planes must not be "
                "registered as ordinary GQA on an auxiliary store. Use one primary GPU "
                "or the explicitly selected peer-expert mode."
            )
        from krasis import GpuDecodeStore
        import torch

        num_layers = len(self.layers)
        if layer_end <= 0:
            layer_end = num_layers
        is_last_segment = (layer_end == num_layers)

        aux_device = torch.device(f"cuda:{gpu_idx}")
        primary_device = torch.device(self.ranks[0].device)

        store = GpuDecodeStore(gpu_idx)
        # Sequence-state metadata is collected from the exact tensors allocated
        # for this pipeline segment, then transferred once into the Rust-owned
        # registry after every attention backend has finalized its pointers.
        # Tuple: (local name, semantic kind, layer, tensor, logical tokens/row).
        aux_sequence_state = []

        def remember_sequence_state(name, kind, layer_idx, tensor, tokens_per_row=0):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                raise RuntimeError(
                    f"Aux sequence-state tensor {name} must be a CUDA tensor"
                )
            if int(tensor.device.index or 0) != int(aux_device.index or 0):
                raise RuntimeError(
                    f"Aux sequence-state tensor {name} is on {tensor.device}, "
                    f"expected {aux_device}"
                )
            if tensor.numel() == 0:
                return
            aux_sequence_state.append(
                (name, kind, int(layer_idx), tensor, int(tokens_per_row))
            )

        # Same configure as primary
        max_qkv = self.cfg.hidden_size * 3
        for layer_idx, layer in enumerate(self.layers):
            if (
                layer.layer_type == "linear_attention"
                and self.cfg.is_kimi_delta_attention_layer(layer_idx)
            ):
                kda_width = (
                    self.cfg.linear_num_key_heads
                    * self.cfg.linear_key_head_dim
                )
                max_qkv = max(max_qkv, kda_width)
            elif layer.layer_type == "linear_attention":
                attn = layer.attention
                qkvz_out = attn.num_k_heads * (2 * attn.k_head_dim + 2 * attn.head_ratio * attn.v_head_dim)
                max_qkv = max(max_qkv, qkvz_out)
            elif hasattr(layer.attention, 'kv_a_proj'):
                # MLA: same calculation as primary store
                ma = layer.attention
                q_out = ma.num_heads * (ma.qk_nope_dim + ma.qk_rope_dim)
                q_absorbed = ma.num_heads * ma.ckv_dim
                kv_out = ma.ckv_dim + ma.qk_rope_dim
                max_qkv = max(max_qkv, q_out, q_absorbed, kv_out)
            elif hasattr(layer.attention, 'num_heads'):
                ga = layer.attention
                q_sz = ga.num_heads * ga.head_dim * (2 if ga.gated_attention else 1)
                kv_sz = ga.num_kv_heads * ga.head_dim
                max_qkv = max(max_qkv, q_sz + kv_sz * 2)

        has_dense_mlp = any(l.dense_mlp is not None for l in self.layers)
        max_inter = max(self.cfg.moe_intermediate_size, self.cfg.effective_shared_expert_intermediate)
        if has_dense_mlp:
            max_inter = max(max_inter, self.cfg.intermediate_size)

        store.configure(
            hidden_size=self.cfg.hidden_size,
            num_layers=len(self.layers),
            vocab_size=self.cfg.vocab_size,
            eps=self.cfg.rms_norm_eps,
            max_experts_per_tok=self.cfg.num_experts_per_tok,
            max_intermediate_size=max_inter,
            max_qkv_size=max_qkv,
            group_size=self.quant_cfg.expert_group_size,
            expert_bits=self.quant_cfg.gpu_expert_bits,
            moe_intermediate_size=self.cfg.moe_intermediate_size,
            shared_expert_intermediate_size=self.cfg.effective_shared_expert_intermediate,
        )
        store.set_decode_segment(split_layer, layer_end)

        hqq_active = is_hqq_attention(self.quant_cfg.attention)
        if hqq_active:
            if not hasattr(self, '_aux_decode_weights_all'):
                self._aux_decode_weights_all = []
            self._aux_decode_weights = []

        # Embedding — not needed on aux (segment_skip_embedding=true), but register
        # a dummy so configure doesn't complain. Use primary embedding ptr (not accessed).
        store.set_embedding(self.embedding.data_ptr())
        store.set_embedding_scale(float(getattr(self.cfg, "embedding_scale", 1.0)))
        store.set_final_logit_softcap(float(getattr(self.cfg, "final_logit_softcapping", 0.0)))

        if is_last_segment:
            # Final norm — copy to aux GPU (only needed on last segment)
            fn_aux = self.final_norm.to(aux_device).contiguous()
            if not hasattr(self, '_aux_final_norms'):
                self._aux_final_norms = []
            self._aux_final_norms.append(fn_aux)
            store.set_final_norm(fn_aux.data_ptr(), self.cfg.hidden_size)

            # LM head — copy to aux GPU (needed for final segment only)
            # Dequantize on CPU to avoid OOM on GPU0 (LM head is ~1.2 GB in BF16)
            lm_head_w = self.lm_head_data
            if isinstance(lm_head_w, tuple):
                w_int8, scale = lm_head_w
                lm_head_bf16 = (w_int8.cpu().float() * scale.cpu().unsqueeze(1)).to(torch.bfloat16).contiguous().to(aux_device)
            else:
                lm_head_bf16 = lm_head_w.to(aux_device).contiguous()
            if not hasattr(self, '_aux_lm_heads'):
                self._aux_lm_heads = []
            self._aux_lm_heads.append(lm_head_bf16)
            lm_head_wid = store.register_weight(lm_head_bf16.data_ptr(), lm_head_bf16.shape[0], lm_head_bf16.shape[1], 0)
            store.set_lm_head(lm_head_wid)
        else:
            # Intermediate segment — placeholder is never accessed because the
            # decode segment skips final logits here. Use a dummy instead of
            # retaining or registering the primary GPU lm-head source.
            store.set_final_norm(self.final_norm.data_ptr(), self.cfg.hidden_size)
            lm_head_wid = store.register_weight(0, 1, 1, 0)
            store.set_lm_head(lm_head_wid)

        store.set_norm_bias_one(getattr(self.cfg, 'norm_bias_one', False))

        # ── AWQ support for aux stores ──
        # Aux GPUs never do prefill, so they only need simple INT4 (no Marlin, no swap).
        attn_quant = self.quant_cfg.attention  # "bf16" or "awq"
        marlin_gs = 128
        _awq_template = None
        if attn_quant == "awq":
            from krasis.awq_calibrate import load_template
            template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "templates", "attention")
            _awq_template = load_template(template_dir, self.cfg.model_path)
            if _awq_template is None:
                raise RuntimeError(
                    "AWQ attention: template loaded on primary store but missing for aux")
            logger.info("Aux store cuda:%d: AWQ simple INT4 attention (no Marlin swap needed)",
                        gpu_idx)

        def _get_bf16_source(attn, attr_name, layer_idx):
            """Get BF16 source tensor, even if primary store replaced attr with MarlinWeight."""
            # Check streaming attention CPU copies first
            if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                cpu_w = self._stream_attn_cpu[layer_idx]
                if attr_name in cpu_w:
                    return cpu_w[attr_name]
            # Check BF16 stash (populated during primary store setup)
            stash = getattr(self, '_aux_bf16_stash', None)
            if stash is not None and (layer_idx, attr_name) in stash:
                return stash[(layer_idx, attr_name)]
            # Direct attribute or dict lookup (if still a tensor)
            if isinstance(attn, dict):
                w = attn.get(attr_name)
            else:
                w = getattr(attn, attr_name, None)
            if w is not None and isinstance(w, torch.Tensor):
                return w
            raise RuntimeError(
                f"Cannot get BF16 source for {attr_name} on layer {layer_idx} "
                f"(type={type(w).__name__}). No streaming attention or BF16 stash available.")

        def _register_aux_attn(w_bf16, layer_idx, layer_type, tensor_name,
                               awq_scales=None):
            """Register attention weight on aux store.
            AWQ → quantize to simple INT4 on aux GPU (Rust alloc + upload).
            BF16 → copy tensor to aux GPU.
            """
            if attn_quant == "awq" and _awq_template is not None:
                from krasis.awq_calibrate import get_tensor_decision
                effective_quant = get_tensor_decision(
                    _awq_template, layer_idx, layer_type, tensor_name)
            else:
                effective_quant = "bf16"

            if effective_quant in ("int4", "int8") and w_bf16.dtype == torch.bfloat16:
                use_int4 = (effective_quant == "int4")
                if use_int4:
                    n, k = w_bf16.shape
                    if n % 64 == 0 and k % 16 == 0 and k % marlin_gs == 0:
                        w_cpu = w_bf16.cpu().contiguous() if w_bf16.is_cuda else w_bf16.contiguous()
                        if awq_scales is not None:
                            w_cpu = (w_cpu.float() * awq_scales.float().unsqueeze(0)).to(
                                torch.bfloat16).contiguous()
                        store.repack_marlin_int4_cpu(w_cpu.data_ptr(), n, k, marlin_gs)
                        del w_cpu
                        return store.register_simple_int4_only(n, k, marlin_gs)
                    else:
                        logger.warning("Aux attn [%d×%d] not Marlin-compatible, using BF16",
                                       n, k)
                # INT8 or non-compatible: fall through to BF16

            # BF16 path
            w_gpu = w_bf16.to(aux_device, non_blocking=True)
            self._aux_decode_weights.append(w_gpu)
            return store.register_weight(
                w_gpu.data_ptr(), w_gpu.shape[0], w_gpu.shape[1],
                _weight_dtype_code(w_gpu))

        # Keep references to aux GPU weight copies
        if not hasattr(self, '_aux_decode_weights_all'):
            self._aux_decode_weights_all = []
        self._aux_decode_weights = []
        aux_rope_set = False
        gqa_count_before_split = 0

        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attention
            inp_norm = layer.input_norm_weight
            post_norm = layer.post_attn_norm_weight

            # Count GQA layers before split for gqa_cache_offset
            if layer_idx < split_layer and layer.layer_type != "linear_attention" \
                    and not hasattr(attn, 'kv_a_proj'):
                gqa_count_before_split += 1

            # Determine if this layer is in this aux GPU's segment
            in_segment = split_layer <= layer_idx < layer_end

            if hqq_active:
                # Native HQQ attention is registered after shared MLP/KV setup using
                # the aux store's decode segment. Do not try to recover BF16
                # projection sources here; primary HQQ setup has already released
                # redundant BF16 residency by design.
                dummy_wid = store.register_weight(0, 1, 1, 0)
                if layer.layer_type == "linear_attention":
                    store.register_la_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                        in_proj_qkvz_wid=dummy_wid, in_proj_ba_wid=dummy_wid,
                        out_proj_wid=dummy_wid,
                        conv_weight_ptr=0, a_log_ptr=0, dt_bias_ptr=0,
                        norm_weight_ptr=0, conv_state_ptr=0, recur_state_ptr=0,
                        nk=attn.num_k_heads, nv=attn.num_v_heads,
                        dk=attn.k_head_dim, dv=attn.v_head_dim,
                        hr=attn.head_ratio,
                        kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                        scale=attn.scale,
                    )
                elif hasattr(attn, 'kv_a_proj'):
                    cache, mla_offset = self._kv_cache_slot_for_layer(layer_idx)
                    max_seq = cache.max_pages * cache.page_size
                    store.register_mla_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(),
                        input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(),
                        post_attn_norm_size=post_norm.numel(),
                        kv_a_proj_wid=dummy_wid, o_proj_wid=dummy_wid,
                        kv_a_norm_ptr=0, w_kc_ptr=0, w_vc_ptr=0,
                        num_heads=attn.num_heads,
                        kv_lora_rank=attn.kv_lora_rank,
                        qk_nope_dim=attn.qk_nope_dim,
                        qk_rope_dim=attn.qk_rope_dim,
                        v_head_dim=attn.v_head_dim,
                        sm_scale=attn.sm_scale,
                        rope_interleave=getattr(self.cfg, 'rope_interleave', True),
                        ckv_cache_ptr=cache.ckv_cache[mla_offset].data_ptr(),
                        kpe_cache_ptr=cache.kpe_cache[mla_offset].data_ptr(),
                        q_a_proj_wid=None, q_b_proj_wid=None, q_a_norm_ptr=0,
                        q_proj_wid=None, q_lora_rank=0,
                        ckv_cache_dim=attn.ckv_dim,
                    )
                    self._register_dsa_indexer_layer_on_store(
                        store, layer_idx, attn
                    )
                else:
                    _gqa_head_dim = self.cfg.gqa_head_dim_for_layer(layer_idx)
                    _gqa_sm_scale = 1.0 if self.cfg.gemma4_text else 1.0 / (_gqa_head_dim ** 0.5)
                    _gqa_gated = hasattr(self.cfg, 'gated_attention') and self.cfg.gated_attention
                    _gqa_num_heads = self.cfg.gqa_num_heads_for_layer(layer_idx)
                    _gqa_kv_heads = self.cfg.gqa_num_kv_heads_for_layer(layer_idx)
                    max_seq = max(
                        c.max_context_tokens for c in self.kv_caches if c is not None
                    )
                    _rope_cos_ptr, _rope_sin_ptr, _rope_half, _rope_cos, _rope_sin = self._gqa_rope_table_ptrs(layer_idx, max_seq, aux_device)
                    store.register_gqa_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                        q_proj_wid=dummy_wid, k_proj_wid=dummy_wid,
                        v_proj_wid=dummy_wid, o_proj_wid=dummy_wid,
                        fused_qkv_wid=None,
                        num_heads=_gqa_num_heads,
                        num_kv_heads=_gqa_kv_heads,
                        head_dim=_gqa_head_dim, sm_scale=_gqa_sm_scale,
                        q_norm_ptr=0, k_norm_ptr=0,
                        gated=_gqa_gated,
                        rope_half_dim=_rope_half,
                        rope_cos_ptr=_rope_cos_ptr,
                        rope_sin_ptr=_rope_sin_ptr,
                    )
                    if self.cfg.gemma4_text or self.cfg.step3_text:
                        store.set_gqa_rope_half_split(layer_idx, True)
            elif layer.layer_type == "linear_attention":
                if in_segment:
                    # Get BF16 source weights (may have been replaced by MarlinWeight on primary)
                    qkvz_src = _get_bf16_source(attn, "in_proj_qkvz", layer_idx)
                    ba_src = _get_bf16_source(attn, "in_proj_ba", layer_idx)
                    out_src = _get_bf16_source(attn, "out_proj", layer_idx)

                    # AWQ scales for input projections
                    _layer_awq_scales = None
                    _qkvz_scales = None
                    _ba_scales = None
                    if attn_quant == "awq" and _awq_template is not None:
                        from krasis.awq_calibrate import (
                            get_layer_scales, is_awq_scaled_tensor)
                        _layer_awq_scales = get_layer_scales(_awq_template, layer_idx)
                        _qkvz_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "in_proj_qkvz")
                        ) else None
                        _ba_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "in_proj_ba")
                        ) else None

                    qkvz_wid = _register_aux_attn(qkvz_src, layer_idx,
                                                   "linear_attention", "in_proj_qkvz",
                                                   awq_scales=_qkvz_scales)
                    ba_wid = _register_aux_attn(ba_src, layer_idx,
                                                 "linear_attention", "in_proj_ba",
                                                 awq_scales=_ba_scales)
                    out_wid = _register_aux_attn(out_src, layer_idx,
                                                  "linear_attention", "out_proj")

                    # AWQ scales already folded into inp_norm in-place by primary store setup.
                    # Just copy the already-folded norm to aux GPU (no second fold).
                    inp_norm_aux = inp_norm.to(aux_device, non_blocking=True)
                    post_norm_aux = post_norm.to(aux_device, non_blocking=True)
                    self._aux_decode_weights.extend([inp_norm_aux, post_norm_aux])

                    # Copy LA-specific weights
                    conv_w = attn.conv1d_weight.squeeze(1).contiguous().float().to(aux_device)
                    attn._aux_conv_state = attn._conv_state.squeeze(0).float().contiguous().to(aux_device) if attn._conv_state is not None else torch.zeros(attn.conv_dim, attn.kernel_dim, device=aux_device, dtype=torch.float32)
                    attn._aux_norm_weight = attn.norm_weight.float().contiguous().to(aux_device)
                    attn._aux_recur_state = attn._recurrent_state.squeeze(0).float().contiguous().to(aux_device) if attn._recurrent_state is not None else torch.zeros(attn.num_k_heads, attn.k_head_dim, attn.head_ratio * attn.v_head_dim, device=aux_device, dtype=torch.float32)
                    attn._aux_a_log = attn.A_log.float().contiguous().to(aux_device)
                    attn._aux_dt_bias = attn.dt_bias.float().contiguous().to(aux_device)
                    self._aux_decode_weights.extend([conv_w, attn._aux_conv_state, attn._aux_norm_weight, attn._aux_recur_state, attn._aux_a_log, attn._aux_dt_bias])
                    remember_sequence_state(
                        f"layer{layer_idx}.linear.conv",
                        "linear_attention_conv_state",
                        layer_idx,
                        attn._aux_conv_state,
                    )
                    remember_sequence_state(
                        f"layer{layer_idx}.linear.recurrent",
                        "linear_attention_recurrent_state",
                        layer_idx,
                        attn._aux_recur_state,
                    )

                    store.register_la_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm_aux.data_ptr(), input_norm_size=inp_norm_aux.numel(),
                        post_attn_norm_ptr=post_norm_aux.data_ptr(), post_attn_norm_size=post_norm_aux.numel(),
                        in_proj_qkvz_wid=qkvz_wid, in_proj_ba_wid=ba_wid, out_proj_wid=out_wid,
                        conv_weight_ptr=conv_w.data_ptr(),
                        a_log_ptr=attn._aux_a_log.data_ptr(),
                        dt_bias_ptr=attn._aux_dt_bias.data_ptr(),
                        norm_weight_ptr=attn._aux_norm_weight.data_ptr(),
                        conv_state_ptr=attn._aux_conv_state.data_ptr(),
                        recur_state_ptr=attn._aux_recur_state.data_ptr(),
                        nk=attn.num_k_heads, nv=attn.num_v_heads,
                        dk=attn.k_head_dim, dv=attn.v_head_dim,
                        hr=attn.head_ratio,
                        kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                        scale=attn.scale,
                    )
                else:
                    # Placeholder for non-segment LA layers (never executed on this GPU).
                    # Use dummy weights — ptr=0 is safe since decode segment skips these.
                    dummy_wid = store.register_weight(0, 1, 1, 0)
                    store.register_la_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                        in_proj_qkvz_wid=dummy_wid, in_proj_ba_wid=dummy_wid,
                        out_proj_wid=dummy_wid,
                        conv_weight_ptr=0, a_log_ptr=0, dt_bias_ptr=0,
                        norm_weight_ptr=0, conv_state_ptr=0, recur_state_ptr=0,
                        nk=attn.num_k_heads, nv=attn.num_v_heads,
                        dk=attn.k_head_dim, dv=attn.v_head_dim,
                        hr=attn.head_ratio,
                        kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                        scale=attn.scale,
                    )
            elif hasattr(attn, 'kv_a_proj'):
                # MLA attention
                if in_segment:
                    # Get BF16 sources for MLA projections
                    _layer_awq_scales = None
                    _qa_scales = None
                    _qproj_scales = None
                    _kva_scales = None
                    if attn_quant == "awq" and _awq_template is not None:
                        from krasis.awq_calibrate import (
                            get_layer_scales, is_awq_scaled_tensor)
                        _layer_awq_scales = get_layer_scales(_awq_template, layer_idx)
                        _qa_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "q_a_proj")
                        ) else None
                        _qproj_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "q_proj")
                        ) else None
                        _kva_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "kv_a_proj")
                        ) else None

                    # Q projection
                    if attn.has_q_lora:
                        qa_src = _get_bf16_source(attn, "q_a_proj", layer_idx)
                        qb_src = _get_bf16_source(attn, "q_b_proj", layer_idx)
                        qa_wid = _register_aux_attn(qa_src, layer_idx, "mla", "q_a_proj",
                                                     awq_scales=_qa_scales)
                        qb_wid = _register_aux_attn(qb_src, layer_idx, "mla", "q_b_proj")
                        attn._aux_q_a_norm = attn.q_a_norm_weight.float().contiguous().to(aux_device)
                        self._aux_decode_weights.append(attn._aux_q_a_norm)
                        q_a_norm_ptr = attn._aux_q_a_norm.data_ptr()
                        q_proj_wid = None
                    else:
                        q_src = _get_bf16_source(attn, "q_proj", layer_idx)
                        q_proj_wid = _register_aux_attn(q_src, layer_idx, "mla", "q_proj",
                                                         awq_scales=_qproj_scales)
                        qa_wid = None
                        qb_wid = None
                        q_a_norm_ptr = 0

                    # KV projection
                    kva_src = _get_bf16_source(attn, "kv_a_proj", layer_idx)
                    kva_wid = _register_aux_attn(kva_src, layer_idx, "mla", "kv_a_proj",
                                                  awq_scales=_kva_scales)

                    # O projection
                    o_src = _get_bf16_source(attn, "o_proj", layer_idx)
                    o_wid = _register_aux_attn(o_src, layer_idx, "mla", "o_proj")

                    # AWQ scales already folded into inp_norm in-place by primary store setup.
                    # Just copy the already-folded norm to aux GPU (no second fold).
                    inp_norm_aux = inp_norm.to(aux_device, non_blocking=True)
                    post_norm_aux = post_norm.to(aux_device, non_blocking=True)
                    self._aux_decode_weights.extend([inp_norm_aux, post_norm_aux])

                    # kv_a layernorm, w_kc, w_vc on aux GPU
                    attn._aux_kv_a_norm = attn.kv_a_norm_weight.float().contiguous().to(aux_device)
                    attn._aux_w_kc = attn.w_kc.contiguous().to(aux_device)
                    attn._aux_w_vc = attn.w_vc.contiguous().to(aux_device)
                    self._aux_decode_weights.extend([
                        attn._aux_kv_a_norm, attn._aux_w_kc, attn._aux_w_vc])

                    # MLA KV cache on aux GPU
                    cache, mla_offset = self._kv_cache_slot_for_layer(layer_idx)
                    ckv_src = cache.ckv_cache[mla_offset]
                    kpe_src = cache.kpe_cache[mla_offset]
                    ckv_aux = torch.empty_like(ckv_src, device=aux_device)
                    kpe_aux = torch.empty_like(kpe_src, device=aux_device)
                    if not hasattr(self, '_aux_mla_caches'):
                        self._aux_mla_caches = []
                    self._aux_mla_caches.extend([ckv_aux, kpe_aux])
                    remember_sequence_state(
                        f"layer{layer_idx}.mla.compressed",
                        "mla_compressed_kv",
                        layer_idx,
                        ckv_aux,
                        1,
                    )
                    remember_sequence_state(
                        f"layer{layer_idx}.mla.position",
                        "mla_positional_k",
                        layer_idx,
                        kpe_aux,
                        1,
                    )
                    max_seq = cache.max_pages * cache.page_size

                    store.register_mla_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm_aux.data_ptr(),
                        input_norm_size=inp_norm_aux.numel(),
                        post_attn_norm_ptr=post_norm_aux.data_ptr(),
                        post_attn_norm_size=post_norm_aux.numel(),
                        kv_a_proj_wid=kva_wid, o_proj_wid=o_wid,
                        kv_a_norm_ptr=attn._aux_kv_a_norm.data_ptr(),
                        w_kc_ptr=attn._aux_w_kc.data_ptr(),
                        w_vc_ptr=attn._aux_w_vc.data_ptr(),
                        num_heads=attn.num_heads,
                        kv_lora_rank=attn.kv_lora_rank,
                        qk_nope_dim=attn.qk_nope_dim,
                        qk_rope_dim=attn.qk_rope_dim,
                        v_head_dim=attn.v_head_dim,
                        sm_scale=attn.sm_scale,
                        rope_interleave=getattr(self.cfg, 'rope_interleave', True),
                        ckv_cache_ptr=ckv_aux.data_ptr(),
                        kpe_cache_ptr=kpe_aux.data_ptr(),
                        q_a_proj_wid=qa_wid,
                        q_b_proj_wid=qb_wid,
                        q_a_norm_ptr=q_a_norm_ptr,
                        q_proj_wid=q_proj_wid,
                        q_lora_rank=attn.q_lora_rank if attn.has_q_lora else 0,
                        ckv_cache_dim=attn.ckv_dim,
                    )

                    # RoPE from first MLA layer in segment
                    if not aux_rope_set:
                        cos, sin = attn._get_rope_cos_sin(max_seq)
                        cos_f32 = cos.float().contiguous().to(aux_device)
                        sin_f32 = sin.float().contiguous().to(aux_device)
                        self._aux_rope_cos = cos_f32
                        self._aux_rope_sin = sin_f32
                        store.set_rope_tables(
                            cos_f32.data_ptr(), sin_f32.data_ptr(),
                            cos_f32.shape[1], max_seq,
                        )
                        aux_rope_set = True
                else:
                    # Placeholder MLA (never executed on this GPU)
                    dummy_wid = store.register_weight(0, 1, 1, 0)
                    cache, mla_offset = self._kv_cache_slot_for_layer(layer_idx)
                    max_seq = cache.max_pages * cache.page_size
                    store.register_mla_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(),
                        input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(),
                        post_attn_norm_size=post_norm.numel(),
                        kv_a_proj_wid=dummy_wid, o_proj_wid=dummy_wid,
                        kv_a_norm_ptr=0, w_kc_ptr=0, w_vc_ptr=0,
                        num_heads=attn.num_heads,
                        kv_lora_rank=attn.kv_lora_rank,
                        qk_nope_dim=attn.qk_nope_dim,
                        qk_rope_dim=attn.qk_rope_dim,
                        v_head_dim=attn.v_head_dim,
                        sm_scale=attn.sm_scale,
                        rope_interleave=getattr(self.cfg, 'rope_interleave', True),
                        ckv_cache_ptr=cache.ckv_cache[mla_offset].data_ptr(),
                        kpe_cache_ptr=cache.kpe_cache[mla_offset].data_ptr(),
                        q_a_proj_wid=None, q_b_proj_wid=None, q_a_norm_ptr=0,
                        q_proj_wid=None, q_lora_rank=0,
                        ckv_cache_dim=attn.ckv_dim,
                    )
            else:
                # GQA attention — attn object is None for GQA models (weights in layer.gqa_weights)
                _gqa_head_dim = self.cfg.gqa_head_dim_for_layer(layer_idx)
                _gqa_sm_scale = 1.0 if self.cfg.gemma4_text else 1.0 / (_gqa_head_dim ** 0.5)
                _gqa_gated = hasattr(self.cfg, 'gated_attention') and self.cfg.gated_attention
                _gqa_num_heads = self.cfg.gqa_num_heads_for_layer(layer_idx)
                _gqa_kv_heads = self.cfg.gqa_num_kv_heads_for_layer(layer_idx)
                gqa_w = layer.gqa_weights if hasattr(layer, 'gqa_weights') else None

                if in_segment:
                    # Get BF16 source weights from gqa_weights dict or stash
                    q_src = _get_bf16_source(gqa_w, "q_proj", layer_idx)
                    k_src = _get_bf16_source(gqa_w, "k_proj", layer_idx)
                    v_src = _get_bf16_source(gqa_w, "v_proj", layer_idx)
                    o_src = _get_bf16_source(gqa_w, "o_proj", layer_idx)
                    g_src = (
                        _get_bf16_source(gqa_w, "g_proj", layer_idx)
                        if self.cfg.head_wise_attention_gate else None
                    )

                    # AWQ scales
                    _layer_awq_scales = None
                    _q_scales = None
                    _k_scales = None
                    _v_scales = None
                    if attn_quant == "awq" and _awq_template is not None:
                        from krasis.awq_calibrate import (
                            get_layer_scales, is_awq_scaled_tensor)
                        _layer_awq_scales = get_layer_scales(_awq_template, layer_idx)
                        _q_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "q_proj")
                        ) else None
                        _k_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "k_proj")
                        ) else None
                        _v_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "v_proj")
                        ) else None

                    q_wid = _register_aux_attn(q_src, layer_idx, "gqa", "q_proj",
                                               awq_scales=_q_scales)
                    k_wid = _register_aux_attn(k_src, layer_idx, "gqa", "k_proj",
                                               awq_scales=_k_scales)
                    v_wid = _register_aux_attn(v_src, layer_idx, "gqa", "v_proj",
                                               awq_scales=_v_scales)
                    o_wid = _register_aux_attn(o_src, layer_idx, "gqa", "o_proj")
                    g_wid = (
                        _register_aux_attn(g_src, layer_idx, "gqa", "g_proj")
                        if g_src is not None else None
                    )

                    # AWQ scales already folded into inp_norm in-place by primary store setup.
                    # Just copy the already-folded norm to aux GPU (no second fold).
                    inp_norm_aux = inp_norm.to(aux_device, non_blocking=True)
                    post_norm_aux = post_norm.to(aux_device, non_blocking=True)
                    self._aux_decode_weights.extend([inp_norm_aux, post_norm_aux])

                    # QK norms from gqa_weights dict
                    q_norm_src = gqa_w.get("q_norm") if gqa_w else None
                    k_norm_src = gqa_w.get("k_norm") if gqa_w else None
                    if q_norm_src is not None:
                        q_norm_aux = q_norm_src.float().contiguous().to(aux_device)
                        self._aux_decode_weights.append(q_norm_aux)
                        q_norm_ptr = q_norm_aux.data_ptr()
                    else:
                        q_norm_ptr = 0
                    if k_norm_src is not None:
                        k_norm_aux = k_norm_src.float().contiguous().to(aux_device)
                        self._aux_decode_weights.append(k_norm_aux)
                        k_norm_ptr = k_norm_aux.data_ptr()
                    else:
                        k_norm_ptr = 0

                    max_seq = max(
                        c.max_context_tokens for c in self.kv_caches if c is not None
                    )
                    _rope_cos_ptr, _rope_sin_ptr, _rope_half, _rope_cos, _rope_sin = self._gqa_rope_table_ptrs(
                        layer_idx, max_seq, aux_device
                    )

                    store.register_gqa_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm_aux.data_ptr(), input_norm_size=inp_norm_aux.numel(),
                        post_attn_norm_ptr=post_norm_aux.data_ptr(), post_attn_norm_size=post_norm_aux.numel(),
                        q_proj_wid=q_wid, k_proj_wid=k_wid,
                        v_proj_wid=v_wid, o_proj_wid=o_wid,
                        fused_qkv_wid=None,
                        num_heads=_gqa_num_heads,
                        num_kv_heads=_gqa_kv_heads,
                        head_dim=_gqa_head_dim, sm_scale=_gqa_sm_scale,
                        q_norm_ptr=q_norm_ptr, k_norm_ptr=k_norm_ptr,
                        gated=_gqa_gated,
                        head_gate_proj_wid=g_wid,
                        rope_half_dim=_rope_half,
                        rope_cos_ptr=_rope_cos_ptr,
                        rope_sin_ptr=_rope_sin_ptr,
                    )
                    if self.cfg.is_sliding_attention_layer(layer_idx) and self.cfg.sliding_window:
                        store.set_gqa_sliding_window(layer_idx, int(self.cfg.sliding_window))
                    if gqa_w is not None and gqa_w.get("v_norm_no_scale", False):
                        store.set_gqa_v_norm_no_scale(layer_idx, True)
                    if self.cfg.gemma4_text or self.cfg.step3_text:
                        store.set_gqa_rope_half_split(layer_idx, True)

                    # RoPE tables on aux GPU (from first GQA layer in segment)
                    if not aux_rope_set:
                        if _rope_cos is None or _rope_sin is None:
                            raise RuntimeError(f"Missing aux RoPE table for GQA layer {layer_idx}")
                        self._aux_rope_cos = _rope_cos
                        self._aux_rope_sin = _rope_sin
                        store.set_rope_tables(
                            _rope_cos.data_ptr(), _rope_sin.data_ptr(),
                            _rope_cos.shape[1], max_seq,
                        )
                        aux_rope_set = True
                else:
                    # Placeholder GQA (never executed on this GPU)
                    dummy_wid = store.register_weight(0, 1, 1, 0)
                    max_seq = max(
                        c.max_context_tokens for c in self.kv_caches if c is not None
                    )
                    _rope_cos_ptr, _rope_sin_ptr, _rope_half, _rope_cos, _rope_sin = self._gqa_rope_table_ptrs(
                        layer_idx, max_seq, aux_device
                    )
                    store.register_gqa_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                        q_proj_wid=dummy_wid, k_proj_wid=dummy_wid,
                        v_proj_wid=dummy_wid, o_proj_wid=dummy_wid,
                        fused_qkv_wid=None,
                        num_heads=_gqa_num_heads,
                        num_kv_heads=_gqa_kv_heads,
                        head_dim=_gqa_head_dim, sm_scale=_gqa_sm_scale,
                        q_norm_ptr=0, k_norm_ptr=0,
                        gated=_gqa_gated,
                        head_gate_proj_wid=None,
                        rope_half_dim=_rope_half,
                        rope_cos_ptr=_rope_cos_ptr,
                        rope_sin_ptr=_rope_sin_ptr,
                    )
                    if self.cfg.gemma4_text or self.cfg.step3_text:
                        store.set_gqa_rope_half_split(layer_idx, True)

            # Register MLP type
            if layer.is_moe:
                store.register_mlp(layer_idx, "moe")
            elif layer.dense_mlp is not None:
                if in_segment:
                    gp = self._dense_mlp_tensor(layer.dense_mlp["gate_proj"]).to(aux_device, non_blocking=True)
                    up = self._dense_mlp_tensor(layer.dense_mlp["up_proj"]).to(aux_device, non_blocking=True)
                    dp = self._dense_mlp_tensor(layer.dense_mlp["down_proj"]).to(aux_device, non_blocking=True)
                    self._aux_decode_weights.extend([gp, up, dp])
                    gp_wid = store.register_weight(gp.data_ptr(), gp.shape[0], gp.shape[1], 0)
                    up_wid = store.register_weight(up.data_ptr(), up.shape[0], up.shape[1], 0)
                    dp_wid = store.register_weight(dp.data_ptr(), dp.shape[0], dp.shape[1], 0)
                else:
                    gp = self._dense_mlp_tensor(layer.dense_mlp["gate_proj"])
                    up = self._dense_mlp_tensor(layer.dense_mlp["up_proj"])
                    dp = self._dense_mlp_tensor(layer.dense_mlp["down_proj"])
                    self._rust_decode_weights.extend([gp, up, dp])
                    gp_wid = store.register_weight(gp.data_ptr(), gp.shape[0], gp.shape[1], 0)
                    up_wid = store.register_weight(up.data_ptr(), up.shape[0], up.shape[1], 0)
                    dp_wid = store.register_weight(dp.data_ptr(), dp.shape[0], dp.shape[1], 0)
                store.register_mlp(layer_idx, "dense",
                                   gate_proj_wid=gp_wid, up_proj_wid=up_wid, down_proj_wid=dp_wid)
            else:
                store.register_mlp(layer_idx, "none")

        # Register MoE expert data from engine (same engine, same CPU RAM data)
        if self.krasis_engine is not None:
            store.setup_from_engine(self.krasis_engine)
            self._register_deepseek_v4_hash_tables(
                store,
                aux_device,
                self._aux_decode_weights,
                range(split_layer, layer_end),
            )
            if self.cfg.swiglu_limits or self.cfg.swiglu_limits_shared:
                for layer_idx, layer in enumerate(self.layers):
                    if layer.is_moe:
                        routed_limit = float(self.cfg.swiglu_limit_for_layer(layer_idx))
                        shared_limit = float(self.cfg.shared_swiglu_limit_for_layer(layer_idx))
                        if routed_limit or shared_limit:
                            store.set_moe_swiglu_limits(
                                layer_idx=layer_idx,
                                swiglu_limit=routed_limit,
                                shared_swiglu_limit=shared_limit,
                            )

        # Shared expert gates for aux layers
        if not hasattr(self, '_aux_shared_gate_refs_all'):
            self._aux_shared_gate_refs_all = []
        self._aux_shared_gate_refs = []
        for layer_idx, layer in enumerate(self.layers):
            if layer.is_moe and layer.shared_expert_gate is not None:
                if split_layer <= layer_idx < layer_end:
                    sg = layer.shared_expert_gate.to(aux_device, non_blocking=True)
                    self._aux_shared_gate_refs.append(sg)
                    sg_wid = store.register_weight(sg.data_ptr(), sg.shape[0], sg.shape[1], 0)
                else:
                    sg = layer.shared_expert_gate
                    sg_wid = store.register_weight(sg.data_ptr(), sg.shape[0], sg.shape[1], 0)
                store.set_moe_shared_gate_wid(layer_idx, sg_wid)

        # Allocate KV cache on aux GPU for GQA layers in this segment [split_layer..layer_end)
        cache = self.kv_caches[0]
        if cache is not None and cache.kv_format == 4 and cache.k_radius_cache is not None:
            # tq4 KV cache: K norm/indices plus V scale-zero/indices.
            tq4_ptrs = []
            self._aux_tq4_sign_refs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    if split_layer <= layer_idx < layer_end:
                        kn_src = cache.k_radius_cache[gqa_cache_idx]
                        ki_src = cache.k_angles_cache[gqa_cache_idx]
                        vm_src = cache.v_radius_cache[gqa_cache_idx]
                        vi_src = cache.v_angles_cache[gqa_cache_idx]
                        kn_aux = torch.empty_like(kn_src, device=aux_device)
                        ki_aux = torch.empty_like(ki_src, device=aux_device)
                        vm_aux = torch.empty_like(vm_src, device=aux_device)
                        vi_aux = torch.empty_like(vi_src, device=aux_device)
                        signs_aux = _tq4_wht_signs(layer_idx, cache.gqa_head_dim, aux_device)
                        if not hasattr(self, '_aux_kv_caches'):
                            self._aux_kv_caches = []
                        self._aux_kv_caches.extend([kn_aux, ki_aux, vm_aux, vi_aux, signs_aux])
                        self._aux_tq4_sign_refs.append(signs_aux)
                        for suffix, kind, tensor in (
                            ("k_scale", "gqa_k_scale", kn_aux),
                            ("k_packed", "gqa_k_packed", ki_aux),
                            ("v_scale", "gqa_v_scale", vm_aux),
                            ("v_packed", "gqa_v_packed", vi_aux),
                        ):
                            remember_sequence_state(
                                f"layer{layer_idx}.{suffix}", kind, layer_idx, tensor, 1
                            )
                        tq4_ptrs.append((layer_idx,
                                         kn_aux.data_ptr(), ki_aux.data_ptr(),
                                         vm_aux.data_ptr(), vi_aux.data_ptr(),
                                         signs_aux.data_ptr()))
                    else:
                        kn = cache.k_radius_cache[gqa_cache_idx]
                        ki = cache.k_angles_cache[gqa_cache_idx]
                        vm = cache.v_radius_cache[gqa_cache_idx]
                        vi = cache.v_angles_cache[gqa_cache_idx]
                        signs = _tq4_wht_signs(layer_idx, cache.gqa_head_dim, cache.device)
                        self._aux_tq4_sign_refs.append(signs)
                        tq4_ptrs.append((layer_idx,
                                         kn.data_ptr(), ki.data_ptr(),
                                         vm.data_ptr(), vi.data_ptr(),
                                         signs.data_ptr()))
                    gqa_cache_idx += 1
            max_seq = cache.max_pages * cache.page_size
            store.set_kv_cache_ptrs_tq4(tq4_ptrs, max_seq, cache.num_kv_heads, cache.gqa_head_dim)
            logger.info("Aux tq4 KV cache on cuda:%d: %d GQA layers for [%d..%d), max_seq=%d, heads=%d head_dim=%d",
                        gpu_idx, len(tq4_ptrs), split_layer, layer_end, max_seq,
                        cache.num_kv_heads, cache.gqa_head_dim)
        elif cache is not None and cache.kv_format in (5, 6, 7, 8, 9) and cache.k_radius_cache is not None:
            # k4v4/k6v4/k7v4/k6v6/k8v6 KV cache: K is blockwise integer + BF16
            # scale. k4v4/k6v4/k7v4 use Polar4 V; k6v6/k8v6 use integer V.
            kintv4_ptrs = []
            max_seq_by_layer = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    if split_layer <= layer_idx < layer_end:
                        kr_src = cache.k_radius_cache[gqa_cache_idx]
                        ka_src = cache.k_angles_cache[gqa_cache_idx]
                        vr_src = cache.v_radius_cache[gqa_cache_idx]
                        va_src = cache.v_angles_cache[gqa_cache_idx]
                        kr_aux = torch.empty_like(kr_src, device=aux_device)
                        ka_aux = torch.empty_like(ka_src, device=aux_device)
                        vr_aux = torch.empty_like(vr_src, device=aux_device)
                        va_aux = torch.empty_like(va_src, device=aux_device)
                        if not hasattr(self, '_aux_kv_caches'):
                            self._aux_kv_caches = []
                        self._aux_kv_caches.extend([kr_aux, ka_aux, vr_aux, va_aux])
                        for suffix, kind, tensor in (
                            ("k_scale", "gqa_k_scale", kr_aux),
                            ("k_packed", "gqa_k_packed", ka_aux),
                            ("v_scale", "gqa_v_scale", vr_aux),
                            ("v_packed", "gqa_v_packed", va_aux),
                        ):
                            remember_sequence_state(
                                f"layer{layer_idx}.{suffix}", kind, layer_idx, tensor, 1
                            )
                        kintv4_ptrs.append((layer_idx,
                                            kr_aux.data_ptr(), ka_aux.data_ptr(),
                                            vr_aux.data_ptr(), va_aux.data_ptr()))
                        max_seq_by_layer.append((layer_idx, int(kr_aux.shape[0] * kr_aux.shape[1])))
                    else:
                        kr = cache.k_radius_cache[gqa_cache_idx]
                        ka = cache.k_angles_cache[gqa_cache_idx]
                        vr = cache.v_radius_cache[gqa_cache_idx]
                        va = cache.v_angles_cache[gqa_cache_idx]
                        kintv4_ptrs.append((layer_idx,
                                            kr.data_ptr(), ka.data_ptr(),
                                            vr.data_ptr(), va.data_ptr()))
                        max_seq_by_layer.append((layer_idx, int(kr.shape[0] * kr.shape[1])))
                    gqa_cache_idx += 1
            max_seq = cache.max_pages * cache.page_size
            num_blocks = cache.max_num_blocks()
            if cache.kv_format == 8:
                store.set_kv_cache_ptrs_k8v6(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k8v6"
            elif cache.kv_format == 7:
                store.set_kv_cache_ptrs_k6v6(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k6v6"
            elif cache.kv_format == 6:
                store.set_kv_cache_ptrs_k7v4(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k7v4"
            elif cache.kv_format == 9:
                store.set_kv_cache_ptrs_k4v4(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k4v4"
            else:
                store.set_kv_cache_ptrs_k6v4(kintv4_ptrs, max_seq, num_blocks)
                fmt = "k6v4"
            store.set_kv_cache_max_seq_by_layer(max_seq_by_layer)
            logger.info("Aux %s KV cache on cuda:%d: %d GQA layers for [%d..%d), max_seq=%d, blocks=%d",
                        fmt, gpu_idx, len(kintv4_ptrs), split_layer, layer_end, max_seq, num_blocks)
        elif cache is not None and cache.kv_format == 3 and cache.k_cache is not None and cache.v_radius_cache is not None:
            # k8v4 KV cache: K is FP8, V is Polar4 radius + angle.
            k8v4_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    if split_layer <= layer_idx < layer_end:
                        k_src = cache.k_cache[gqa_cache_idx]
                        vr_src = cache.v_radius_cache[gqa_cache_idx]
                        va_src = cache.v_angles_cache[gqa_cache_idx]
                        k_aux = torch.empty_like(k_src, device=aux_device)
                        vr_aux = torch.empty_like(vr_src, device=aux_device)
                        va_aux = torch.empty_like(va_src, device=aux_device)
                        if not hasattr(self, '_aux_kv_caches'):
                            self._aux_kv_caches = []
                        self._aux_kv_caches.extend([k_aux, vr_aux, va_aux])
                        for suffix, kind, tensor in (
                            ("k", "gqa_k", k_aux),
                            ("v_scale", "gqa_v_scale", vr_aux),
                            ("v_packed", "gqa_v_packed", va_aux),
                        ):
                            remember_sequence_state(
                                f"layer{layer_idx}.{suffix}", kind, layer_idx, tensor, 1
                            )
                        k8v4_ptrs.append((layer_idx,
                                          k_aux.data_ptr(),
                                          vr_aux.data_ptr(), va_aux.data_ptr()))
                    else:
                        k_layer = cache.k_cache[gqa_cache_idx]
                        vr = cache.v_radius_cache[gqa_cache_idx]
                        va = cache.v_angles_cache[gqa_cache_idx]
                        k8v4_ptrs.append((layer_idx,
                                          k_layer.data_ptr(),
                                          vr.data_ptr(), va.data_ptr()))
                    gqa_cache_idx += 1
            max_seq = cache.max_pages * cache.page_size
            num_blocks = (cache.num_kv_heads * cache.gqa_head_dim) // 16
            store.set_kv_cache_ptrs_k8v4(k8v4_ptrs, max_seq, num_blocks)
            logger.info("Aux k8v4 KV cache on cuda:%d: %d GQA layers for [%d..%d), max_seq=%d, V blocks=%d",
                        gpu_idx, len(k8v4_ptrs), split_layer, layer_end, max_seq, num_blocks)
        elif cache is not None and cache.kv_format == 2 and cache.k_radius_cache is not None:
            # Polar4 KV cache: 4 tensors per layer (radius BF16 + angles uint8 for K and V)
            polar4_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    if split_layer <= layer_idx < layer_end:
                        kr_src = cache.k_radius_cache[gqa_cache_idx]
                        vr_src = cache.v_radius_cache[gqa_cache_idx]
                        ka_src = cache.k_angles_cache[gqa_cache_idx]
                        va_src = cache.v_angles_cache[gqa_cache_idx]
                        kr_aux = torch.empty_like(kr_src, device=aux_device)
                        vr_aux = torch.empty_like(vr_src, device=aux_device)
                        ka_aux = torch.empty_like(ka_src, device=aux_device)
                        va_aux = torch.empty_like(va_src, device=aux_device)
                        if not hasattr(self, '_aux_kv_caches'):
                            self._aux_kv_caches = []
                        self._aux_kv_caches.extend([kr_aux, vr_aux, ka_aux, va_aux])
                        for suffix, kind, tensor in (
                            ("k_scale", "gqa_k_scale", kr_aux),
                            ("k_packed", "gqa_k_packed", ka_aux),
                            ("v_scale", "gqa_v_scale", vr_aux),
                            ("v_packed", "gqa_v_packed", va_aux),
                        ):
                            remember_sequence_state(
                                f"layer{layer_idx}.{suffix}", kind, layer_idx, tensor, 1
                            )
                        polar4_ptrs.append((layer_idx,
                                            kr_aux.data_ptr(), vr_aux.data_ptr(),
                                            ka_aux.data_ptr(), va_aux.data_ptr()))
                    else:
                        kr = cache.k_radius_cache[gqa_cache_idx]
                        vr = cache.v_radius_cache[gqa_cache_idx]
                        ka = cache.k_angles_cache[gqa_cache_idx]
                        va = cache.v_angles_cache[gqa_cache_idx]
                        polar4_ptrs.append((layer_idx,
                                            kr.data_ptr(), vr.data_ptr(),
                                            ka.data_ptr(), va.data_ptr()))
                    gqa_cache_idx += 1
            max_seq = cache.max_pages * cache.page_size
            num_blocks = (cache.num_kv_heads * cache.gqa_head_dim) // 16
            store.set_kv_cache_ptrs_polar4(polar4_ptrs, max_seq, num_blocks)
            logger.info("Aux Polar4 KV cache on cuda:%d: %d GQA layers for [%d..%d), max_seq=%d, blocks=%d",
                        gpu_idx, len(polar4_ptrs), split_layer, layer_end, max_seq, num_blocks)
        elif cache is not None and cache.k_cache is not None:
            # FP8/BF16 KV cache: 2 tensors per layer (K and V)
            kv_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    if split_layer <= layer_idx < layer_end:
                        k_src = cache.k_cache[gqa_cache_idx]
                        v_src = cache.v_cache[gqa_cache_idx]
                        k_aux = torch.empty_like(k_src, device=aux_device)
                        v_aux = torch.empty_like(v_src, device=aux_device)
                        if not hasattr(self, '_aux_kv_caches'):
                            self._aux_kv_caches = []
                        self._aux_kv_caches.extend([k_aux, v_aux])
                        remember_sequence_state(
                            f"layer{layer_idx}.k", "gqa_k", layer_idx, k_aux, 1
                        )
                        remember_sequence_state(
                            f"layer{layer_idx}.v", "gqa_v", layer_idx, v_aux, 1
                        )
                        kv_ptrs.append((layer_idx, k_aux.data_ptr(), v_aux.data_ptr()))
                    else:
                        k_layer = cache.k_cache[gqa_cache_idx]
                        v_layer = cache.v_cache[gqa_cache_idx]
                        kv_ptrs.append((layer_idx, k_layer.data_ptr(), v_layer.data_ptr()))
                    gqa_cache_idx += 1
            max_seq = cache.max_pages * cache.page_size
            if cache.kv_format == 0:
                store.set_kv_cache_ptrs_bf16(kv_ptrs, max_seq)
                cache_label = "BF16"
            else:
                store.set_kv_cache_ptrs(kv_ptrs, max_seq)
                cache_label = "FP8"
            logger.info("Aux %s KV cache on cuda:%d: %d GQA layers for [%d..%d), max_seq=%d",
                        cache_label, gpu_idx, len(kv_ptrs), split_layer, layer_end, max_seq)

        torch.cuda.synchronize(aux_device)

        # Store references for GC protection and multi-GPU bookkeeping
        if not hasattr(self, '_aux_gpu_decode_stores'):
            self._aux_gpu_decode_stores = []
        self._aux_gpu_decode_stores.append(store)
        if hqq_active:
            registered_layers = self._register_hqq_attention_layers_on_store(
                store,
                aux_device,
                self._aux_decode_weights,
                aux_sequence_state,
            )
            logger.info(
                "HQQ aux execution descriptors restored after shared decode setup on cuda:%d: %d layers registered.",
                gpu_idx,
                registered_layers,
            )
        if self.cfg.is_dsa:
            staged_dsa = self._stage_dsa_indexer_resources_on_store(
                store,
                aux_device,
                self._aux_decode_weights,
                split_layer,
                layer_end,
            )
            logger.info(
                "Aux decode DSA resources staged on cuda:%d: %d "
                "owner/replica resources for layers [%d,%d)",
                gpu_idx,
                staged_dsa,
                split_layer,
                layer_end,
            )

        registered_sequence_state = 0
        for name, kind, layer_idx, tensor, tokens_per_row in aux_sequence_state:
            if not (split_layer <= layer_idx < layer_end):
                continue
            self._register_sequence_state_tensor(
                store,
                name=f"gpu{gpu_idx}.{name}",
                kind=kind,
                layer_idx=layer_idx,
                tensor=tensor,
                logical_tokens_per_row=tokens_per_row,
            )
            registered_sequence_state += 1
        required_sequence_state = int(store.finalize_sequence_state_inventory())
        logger.info(
            "Aux sequence-state inventory on cuda:%d for layers [%d,%d): "
            "registered=%d required=%d",
            gpu_idx,
            split_layer,
            layer_end,
            registered_sequence_state,
            required_sequence_state,
        )

        self._aux_decode_weights_all.extend(self._aux_decode_weights)
        self._aux_shared_gate_refs_all.extend(self._aux_shared_gate_refs)

        # Legacy single-aux-store attributes (still set for backward compat)
        self._aux_gpu_decode_store = store
        self._aux_gpu_idx = gpu_idx
        self._multi_gpu_split_layer = split_layer
        self._multi_gpu_gqa_offset = gqa_count_before_split

        if os.environ.get("KRASIS_DECODE_TIMING", "") == "1":
            store.set_timing(True)

        logger.info("Aux GPU decode store configured: gpu=%d, layers=[%d..%d), gqa_offset=%d, store_addr=%d",
                     gpu_idx, split_layer, layer_end, gqa_count_before_split, store.gpu_store_addr())
        return store

    def _export_kv_to_rust(self, seq_states, prompt_len: int):
        """Set KV cache position for Rust decode after prefill.

        The KV cache is shared — Rust prefill already wrote FP8 data into the
        same GPU buffers that Rust decode reads. No data copy needed, just
        tell Rust how many tokens are valid.
        """
        store = self._gpu_decode_store

        # Guard: don't overflow shared KV buffer
        rust_max_seq = getattr(store, 'kv_max_seq', 0)
        if rust_max_seq > 0 and prompt_len > rust_max_seq:
            logger.warning(
                "Prompt length %d exceeds KV cache max_seq (%d), "
                "skipping decode for this request", prompt_len, rust_max_seq)
            return

        store.set_kv_position(prompt_len)

    def _transfer_mamba2_states(self):
        """Copy Mamba2 conv_state and ssm_state from Python prefill into Rust decode buffers.

        After prefill, each Mamba2 layer has saved its final conv_state and ssm_state
        on the layer object. These must be copied into the pre-allocated GPU buffers
        that the Rust decode engine uses. The buffers have fixed addresses (registered
        during _register_mamba2_layer), so we copy in-place to preserve pointers.
        """
        decode_states = getattr(self, '_mamba2_decode_states', None)
        if not decode_states:
            return

        for layer_idx, buffers in decode_states.items():
            layer = self.layers[layer_idx]

            # Conv state: prefill produces [1, conv_dim, conv_kernel], Rust expects [conv_dim, conv_kernel] FP32
            if hasattr(layer, '_mamba2_conv_state') and layer._mamba2_conv_state is not None:
                conv_src = layer._mamba2_conv_state.squeeze(0).float().contiguous()
                buffers['conv_state'].copy_(conv_src)
                layer._mamba2_conv_state = None  # Free prefill tensor

            # SSM state: prefill produces [1, num_heads, head_dim, state_size], Rust expects [num_heads, head_dim, state_size] FP32
            if hasattr(layer, '_mamba2_ssm_state') and layer._mamba2_ssm_state is not None:
                ssm_src = layer._mamba2_ssm_state.squeeze(0).float().contiguous()
                buffers['ssm_state'].copy_(ssm_src)
                layer._mamba2_ssm_state = None  # Free prefill tensor

    def _update_la_state_ptrs(self):
        """Re-register LA state pointers after prefill (states may have been reallocated).
        Prefill may reassign _conv_state and _recurrent_state tensors, so we
        need to create fresh FP32 copies and re-register the new pointers."""
        store = self._gpu_decode_store
        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_type == "linear_attention":
                attn = layer.attention
                if attn._conv_state is None or attn._recurrent_state is None:
                    continue
                wids = self._la_wids.get(layer_idx)
                if wids is None:
                    continue
                inp_norm = layer.input_norm_weight
                post_norm = layer.post_attn_norm_weight
                conv_w = attn._rust_conv_weight

                # Prefill may have updated conv_state and recurrent_state (BF16) -- convert to FP32 for Rust.
                # IMPORTANT: copy INTO existing buffers to preserve fixed GPU addresses for CUDA graph replay.
                # Only allocate on first call; subsequent calls copy data in-place.
                new_conv = attn._conv_state.squeeze(0).float().contiguous()
                new_recur = attn._recurrent_state.squeeze(0).float().contiguous()
                if hasattr(attn, '_rust_conv_state') and attn._rust_conv_state is not None \
                        and attn._rust_conv_state.shape == new_conv.shape:
                    attn._rust_conv_state.copy_(new_conv)
                    attn._rust_recur_state.copy_(new_recur)
                else:
                    attn._rust_conv_state = new_conv
                    attn._rust_recur_state = new_recur

                store.register_la_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                    in_proj_qkvz_wid=wids[0], in_proj_ba_wid=wids[1], out_proj_wid=wids[2],
                    conv_weight_ptr=conv_w.data_ptr(),
                    a_log_ptr=attn._rust_a_log.data_ptr(),
                    dt_bias_ptr=attn._rust_dt_bias.data_ptr(),
                    norm_weight_ptr=attn._rust_norm_weight.data_ptr(),
                    conv_state_ptr=attn._rust_conv_state.data_ptr(),
                    recur_state_ptr=attn._rust_recur_state.data_ptr(),
                    nk=attn.num_k_heads, nv=attn.num_v_heads,
                    dk=attn.k_head_dim, dv=attn.v_head_dim,
                    hr=attn.head_ratio,
                    kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                    scale=attn.scale,
                )

    def _update_la_state_ptrs_aux(self):
        """Re-register LA state pointers on all aux stores after prefill.
        Copies post-prefill conv/recur states from GPU0 to each aux GPU."""
        aux_stores = getattr(self, '_aux_gpu_decode_stores', None)
        if not aux_stores:
            # Fallback to legacy single-store attribute
            single = getattr(self, '_aux_gpu_decode_store', None)
            if single is None:
                return
            aux_stores = [single]

        import torch

        # For each aux store, update LA state pointers for layers in its segment.
        # We need to know each store's layer range. We can infer this from the store's
        # graph config (segment_layer_start/end were set during setup). For simplicity,
        # iterate all LA layers >= first split and update any aux store that has them.
        split_layer = getattr(self, '_multi_gpu_split_layer', 0)

        for aux_store in aux_stores:
            aux_gpu_idx = aux_store.gpu_index() if hasattr(aux_store, 'gpu_index') else 0
            aux_dev = torch.device(f"cuda:{aux_gpu_idx}")

            for layer_idx, layer in enumerate(self.layers):
                if layer_idx < split_layer:
                    continue
                if layer.layer_type != "linear_attention":
                    continue
                attn = layer.attention
                if attn._conv_state is None or attn._recurrent_state is None:
                    continue
                # Copy updated states to aux GPU -- preserve fixed addresses for CUDA graph replay.
                # Use per-store attribute names to avoid collisions between aux stores.
                attr_conv = f'_aux_conv_state_{aux_gpu_idx}'
                attr_recur = f'_aux_recur_state_{aux_gpu_idx}'
                new_conv_aux = attn._conv_state.squeeze(0).float().contiguous().to(aux_dev)
                new_recur_aux = attn._recurrent_state.squeeze(0).float().contiguous().to(aux_dev)
                existing_conv = getattr(attn, attr_conv, None)
                if existing_conv is not None and existing_conv.shape == new_conv_aux.shape:
                    existing_conv.copy_(new_conv_aux)
                    getattr(attn, attr_recur).copy_(new_recur_aux)
                else:
                    setattr(attn, attr_conv, new_conv_aux)
                    setattr(attn, attr_recur, new_recur_aux)
                # Also update the legacy attributes for backward compat
                if not hasattr(attn, '_aux_conv_state') or attn._aux_conv_state is None:
                    attn._aux_conv_state = getattr(attn, attr_conv)
                    attn._aux_recur_state = getattr(attn, attr_recur)
                aux_store.update_la_state_ptrs(
                    layer_idx,
                    getattr(attn, attr_conv).data_ptr(),
                    getattr(attn, attr_recur).data_ptr(),
                )

    def restrict_to_decode_segment(self, decode_start: int, decode_end: int):
        """Multi-GPU: restrict GPU 0's single-slot AWQ swaps to its decode segment only.

        Called after set_decode_segment(). Removes swap entries for layers outside
        [decode_start, decode_end) so swap_to_simple_int4/swap_to_marlin only touch
        decode-segment weights. Stashes prefill-only layer info for free/upload cycle.
        """
        gpu_store = getattr(self, '_gpu_decode_store', None)
        if gpu_store is None:
            return

        # Tell Rust to remove swap entries for non-decode layers
        removed = gpu_store.restrict_swaps_to_decode_segment()

        # Track which MarlinWeight attributes are "prefill-only" (outside decode segment)
        # These can be freed after prefill and re-uploaded before next prefill.
        from krasis.attention import MarlinWeight
        self._prefill_only_attn = []  # [(layer_idx, attn, attr_name, MarlinWeight)]
        self._prefill_only_freed = False

        for layer_idx, layer in enumerate(self.layers):
            if decode_start <= layer_idx < decode_end:
                continue  # In decode segment — managed by single-slot swap
            attn = layer.attention
            if layer.layer_type == "linear_attention":
                for attr_name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                    mw = getattr(attn, attr_name, None)
                    if isinstance(mw, MarlinWeight):
                        self._prefill_only_attn.append((layer_idx, attn, attr_name, mw))
            elif hasattr(attn, 'kv_a_proj'):
                for attr_name in ("q_a_proj", "q_b_proj", "q_proj", "kv_a_proj", "o_proj"):
                    mw = getattr(attn, attr_name, None)
                    if isinstance(mw, MarlinWeight):
                        self._prefill_only_attn.append((layer_idx, attn, attr_name, mw))
            else:
                for attr_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    mw = getattr(attn, attr_name, None)
                    if isinstance(mw, MarlinWeight):
                        self._prefill_only_attn.append((layer_idx, attn, attr_name, mw))

        n_tensors = len(self._prefill_only_attn)
        if n_tensors > 0:
            # Estimate VRAM that will be freed after prefill
            total_bytes = 0
            for _, _, _, mw in self._prefill_only_attn:
                total_bytes += mw.packed.nelement() * mw.packed.element_size()
                total_bytes += mw.scales.nelement() * mw.scales.element_size()
            logger.info("Prefill-only attention: %d tensors for layers outside [%d, %d), "
                        "~%.0f MB freeable after prefill",
                        n_tensors, decode_start, decode_end, total_bytes / 1024 / 1024)

    def _free_prefill_only_attention(self):
        """Free prefill-only attention tensors from GPU after prefill.

        Moves packed/scales to CPU and clears GPU references so PyTorch can
        reclaim VRAM. The CPU copies are kept for re-upload before next prefill.
        """
        entries = getattr(self, '_prefill_only_attn', None)
        if not entries or getattr(self, '_prefill_only_freed', False):
            return

        import gc
        from krasis.attention import MarlinWeight
        freed_bytes = 0
        new_entries = []

        for layer_idx, attn, attr_name, mw in entries:
            packed = mw.packed
            scales = mw.scales
            freed_bytes += packed.nelement() * packed.element_size()
            freed_bytes += scales.nelement() * scales.element_size()
            # Move to CPU (creates CPU copy, releases GPU tensor on next GC)
            packed_cpu = packed.cpu()
            scales_cpu = scales.cpu()
            cpu_mw = MarlinWeight(packed_cpu, scales_cpu, mw.workspace, mw.scalar_type, mw.n, mw.k)
            # Clear GPU reference on attention module
            setattr(attn, attr_name, None)
            # Also remove from _marlin_attn_weights if tracked there
            marlin_weights = getattr(self, '_marlin_attn_weights', {})
            to_remove = [wid for wid, info in marlin_weights.items()
                         if info[0] is packed]
            for wid in to_remove:
                del marlin_weights[wid]
            # Release scales_slot from _single_slot_scales if it backs this scales view
            slot_scales = getattr(self, '_single_slot_scales', [])
            scales_base = scales.storage().data_ptr() if scales.is_cuda else 0
            self._single_slot_scales = [
                s for s in slot_scales
                if s.data_ptr() != scales_base
            ]
            # Keep CPU copy for re-upload
            new_entries.append((layer_idx, attn, attr_name, cpu_mw))

        self._prefill_only_attn = new_entries
        gc.collect()
        torch.cuda.empty_cache()
        self._prefill_only_freed = True
        logger.info("Freed prefill-only attention: ~%.0f MB", freed_bytes / 1024 / 1024)

    def _upload_prefill_only_attention(self):
        """Re-upload prefill-only attention tensors to GPU before prefill.

        Restores MarlinWeight references on attention modules. Called before prefill
        in multi-GPU mode when the tensors were previously freed.
        """
        entries = getattr(self, '_prefill_only_attn', None)
        if not entries or not getattr(self, '_prefill_only_freed', False):
            return  # Nothing freed, nothing to upload

        device = torch.device(self.ranks[0].device)
        uploaded_bytes = 0
        new_entries = []

        from krasis.attention import MarlinWeight
        for layer_idx, attn, attr_name, mw in entries:
            # Re-upload to GPU
            if mw.packed.is_cuda:
                # Already on GPU (shouldn't happen if freed, but be safe)
                new_entries.append((layer_idx, attn, attr_name, mw))
                continue
            packed_gpu = mw.packed.to(device, non_blocking=True)
            scales_gpu = mw.scales.to(device, non_blocking=True)
            uploaded_bytes += packed_gpu.nelement() * packed_gpu.element_size()
            uploaded_bytes += scales_gpu.nelement() * scales_gpu.element_size()

            new_mw = MarlinWeight(packed_gpu, scales_gpu, mw.workspace, mw.scalar_type, mw.n, mw.k)
            setattr(attn, attr_name, new_mw)
            new_entries.append((layer_idx, attn, attr_name, new_mw))

        self._prefill_only_attn = new_entries
        self._prefill_only_freed = False
        torch.cuda.synchronize(device)
        logger.info("Uploaded prefill-only attention: ~%.0f MB", uploaded_bytes / 1024 / 1024)

    def server_cleanup(self):
        """Free server request state (KV cache pages, etc.).

        Also handles single-slot AWQ: restore Marlin data into GPU slots
        for instant prefill on the next request.
        """
        # Single-slot AWQ: restore Marlin into slots
        gpu_store = getattr(self, '_gpu_decode_store', None)
        if gpu_store is not None:
            gpu_store.swap_to_marlin()

        states = getattr(self, '_server_seq_states', None)
        if states:
            for s in states:
                if s is not None:
                    s.free()
            self._server_seq_states = None

        # Request-scoped recurrent state must be reset alongside KV cleanup.
        # Otherwise an internal test request can leak LA/Mamba decode state into
        # the next chat request even though the KV pages were freed.
        if self.cfg.is_hybrid:
            for layer in self.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        decode_states = getattr(self, '_mamba2_decode_states', None)
        if decode_states:
            for buffers in decode_states.values():
                buffers['conv_state'].zero_()
                buffers['ssm_state'].zero_()

        self._rust_kv_refs = None
