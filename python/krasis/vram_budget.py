"""VRAM budget calculator for Krasis + SGLang.

The user passes a context length hint. Krasis calculates the actual weight
footprint per PP rank, determines free VRAM, and allocates the LOWER of:
  (A) KV cache needed for the requested context, or
  (B) Maximum KV cache that fits minus headroom.

This way we never waste VRAM on KV cache we don't need, and always leave
room for GPU prefill workspace + temporary buffers.

Usage (CLI):
    python -m krasis.vram_budget \\
        --model-path /path/to/Kimi-K2.5 \\
        --pp-partition 20,21,20 \\
        --kv-cache-dtype fp8_e4m3 \\
        --quantization w8a8_int8 \\
        --context-length 0

Usage (Python):
    from krasis.vram_budget import compute_vram_budget
    budget = compute_vram_budget(
        model_path="/path/to/Kimi-K2.5",
        pp_partition=[20, 21, 20],
        kv_cache_dtype="fp8_e4m3",
        quantization="w8a8_int8",
        requested_context=0,
    )
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from krasis.attention_backend import (
    HQQ_ATTENTION_CACHE_VERSION,
    HQQ_CACHE_PROFILE_BASELINE,
    HQQ_DEFAULT_GROUP_SIZE,
    attention_quant_cache_nbits,
    cache_dir_for_model,
    hqq46_tensor_nbits,
    hqq_attention_cache_layer_bytes,
    hqq_attention_cache_dir,
    hqq_auto_budget_bytes_from_pct,
    hqq_auto_candidate_from_records,
    hqq_auto_direct_edge_nbits,
    hqq_auto_promotion_policy,
    is_hqq_auto_attention,
    load_hqq_attention_manifest,
    select_hqq_auto_promotions,
)
from krasis.config import ModelConfig
from krasis.kv_cache import MLA_CKV_KERNEL_MIN_DIM, PAGE_SIZE

logger = logging.getLogger(__name__)

# Default CUDA/PyTorch runtime overhead estimate for pre-launch VRAM budgets.
# This is a conservative estimate used ONLY in the launcher TUI (before the model
# is loaded). Actual VRAM availability is measured at runtime via 4-point calibration
# in server.py, which supersedes this estimate entirely.
# Can be overridden via compute_launcher_budget(cuda_overhead_mb=...) or
# compute_vram_budget(overhead_mb=...).
DEFAULT_CUDA_OVERHEAD_MB = 2000


HQQ_ATTENTION_QUANTS = ("hqq4", "hqq46", "hqq46_auto", "hqq6", "hqq68_auto", "hqq8")

# These are cache/kernel format contracts, not hardware measurements. They
# mirror src/weights/mod.rs and src/gpu_decode.rs so the pre-load launcher
# accounts for D-Spark without depending on a cache built on this host.
MARLIN_CACHE_HEADER_BYTES = 64
MARLIN_DEVICE_ALIGNMENT_BYTES = 512
DSPARK_MODES = ("off", "resident", "shared")


def _read_model_config(model_path: str) -> Dict[str, Any]:
    """Read and normalize model config.json."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    # Kimi K2.5 nests under text_config; Qwen3 is flat
    cfg = raw.get("text_config", raw)

    # Normalize tie_word_embeddings (may be at top level)
    if "tie_word_embeddings" not in cfg:
        cfg["tie_word_embeddings"] = raw.get("tie_word_embeddings", True)

    return cfg


def _normalized_launcher_config(model_path: str) -> tuple[Dict[str, Any], ModelConfig]:
    """Return budget input normalized by the runtime's architecture parser.

    Launcher budgeting must use the same aliases and layer schedules as model
    loading. Reading raw Transformers keys independently made Step appear dense
    and treated every Nemotron block as both attention and MoE.
    """
    cfg = _read_model_config(model_path)
    model_cfg = ModelConfig.from_model_path(model_path)
    cfg.update({
        "model_type": model_cfg.model_type,
        "hidden_size": model_cfg.hidden_size,
        "intermediate_size": model_cfg.intermediate_size,
        "moe_intermediate_size": model_cfg.moe_intermediate_size,
        "num_hidden_layers": model_cfg.num_hidden_layers,
        "num_attention_heads": model_cfg.num_attention_heads,
        "num_key_value_heads": model_cfg.num_key_value_heads,
        "head_dim": model_cfg.gqa_head_dim or model_cfg.attention_head_dim or model_cfg.head_dim,
        "n_routed_experts": model_cfg.n_routed_experts,
        "num_experts_per_tok": model_cfg.num_experts_per_tok,
        "n_shared_experts": model_cfg.n_shared_experts,
        "shared_expert_intermediate_size": model_cfg.effective_shared_expert_intermediate,
        "first_k_dense_replace": model_cfg.first_k_dense_replace,
        "max_position_embeddings": model_cfg.max_position_embeddings,
        "moe_latent_size": model_cfg.moe_latent_size,
        "mlp_hidden_act": model_cfg.mlp_hidden_act,
    })
    if model_cfg.layer_types is not None:
        cfg["layer_types"] = list(model_cfg.layer_types)
    if model_cfg.moe_layer_indices is not None:
        cfg["moe_layer_indices"] = list(model_cfg.moe_layer_indices)
    return cfg, model_cfg


def _model_kv_bytes_per_token_for_layer(
    model_cfg: ModelConfig,
    cfg: Dict[str, Any],
    layer_idx: int,
    kv_dtype: str,
) -> int:
    if not model_cfg.is_full_attention_layer(layer_idx):
        return 0
    if model_cfg.is_mla:
        return int(
            (model_cfg.kv_lora_rank + model_cfg.qk_rope_head_dim)
            * _kv_dtype_bytes(kv_dtype, cfg)
        )
    dtype_bytes = _kv_dtype_bytes(kv_dtype, cfg)
    return int(
        2
        * model_cfg.gqa_num_kv_heads_for_layer(layer_idx)
        * model_cfg.gqa_head_dim_for_layer(layer_idx)
        * dtype_bytes
    )


def _persistent_recurrent_state_bytes(model_cfg: ModelConfig, start: int, end: int) -> int:
    """Runtime FP32 recurrent state resident on a rank for hybrid layers."""
    total = 0
    for layer_idx in range(start, end):
        if model_cfg.is_linear_attention_layer(layer_idx):
            nk = model_cfg.linear_num_key_heads
            nv = model_cfg.linear_num_value_heads
            dk = model_cfg.linear_key_head_dim
            dv = model_cfg.linear_value_head_dim
            conv_dim = nk * dk * 2 + nv * dv
            total += conv_dim * model_cfg.linear_conv_kernel_dim * 4
            total += nv * dk * dv * 4
        elif model_cfg.is_mamba2_layer(layer_idx):
            total += model_cfg.mamba_conv_dim * model_cfg.mamba_conv_kernel * 4
            total += (
                model_cfg.mamba_num_heads
                * model_cfg.mamba_head_dim
                * model_cfg.ssm_state_size
                * 4
            )
    return total


def _detect_gpu_vram_bytes() -> int:
    """Auto-detect per-GPU VRAM in bytes via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip().split("\n")[0].strip())
            return mb * 1024 * 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    logger.warning("Could not detect GPU VRAM, assuming 16 GB")
    return 16 * 1024**3


def _detect_gpu_sm_count() -> int:
    """Detect GPU SM count via torch.cuda. Falls back to 170 (safe overestimate)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).multi_processor_count
    except (ImportError, RuntimeError):
        pass
    logger.warning("Could not detect GPU SM count, assuming 170")
    return 170  # 5090 SM count — overestimates workspace for smaller GPUs (safe direction)


def _is_mla(cfg: Dict[str, Any]) -> bool:
    return "kv_lora_rank" in cfg and cfg["kv_lora_rank"] > 0


def _weight_bytes(params: int, quantization: str) -> int:
    if quantization == "w8a8_int8":
        return params  # 1 byte per param
    else:
        return params * 2  # BF16


def _mla_attention_bytes_per_layer(cfg: Dict[str, Any], quantization: str) -> int:
    hidden = cfg["hidden_size"]
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    n_heads = cfg["num_attention_heads"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]

    q_a_params = hidden * q_lora
    q_b_params = q_lora * (n_heads * (qk_nope + qk_rope))
    kv_a_params = hidden * (kv_lora + qk_rope)
    kv_b_params = kv_lora * (n_heads * (qk_nope + v_head))
    o_params = (n_heads * v_head) * hidden

    total_params = q_a_params + q_b_params + kv_a_params + kv_b_params + o_params
    return _component_weight_bytes(total_params, quantization)


def _gqa_attention_bytes_per_layer(cfg: Dict[str, Any], quantization: str) -> int:
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", hidden // n_heads)

    q_params = hidden * (n_heads * head_dim)
    k_params = hidden * (n_kv_heads * head_dim)
    v_params = hidden * (n_kv_heads * head_dim)
    o_params = (n_heads * head_dim) * hidden

    total_params = q_params + k_params + v_params + o_params
    return _component_weight_bytes(total_params, quantization)


def _dense_mlp_bytes_per_layer(cfg: Dict[str, Any], quantization: str) -> int:
    hidden = cfg["hidden_size"]
    intermediate = cfg.get("intermediate_size", cfg.get("moe_intermediate_size", 0))
    total_params = 3 * hidden * intermediate  # gate + up + down
    return _weight_bytes(total_params, quantization)


def _gate_bytes_per_moe_layer(cfg: Dict[str, Any]) -> int:
    hidden = cfg["hidden_size"]
    n_experts = cfg.get("n_routed_experts", cfg.get("num_experts", 0))
    return hidden * n_experts * 2  # BF16


def _shared_expert_bytes_per_moe_layer(cfg: Dict[str, Any], quantization: str) -> int:
    n_shared = cfg.get("n_shared_experts", 0)
    if n_shared == 0:
        return 0
    hidden = cfg["hidden_size"]
    intermediate = cfg.get("moe_intermediate_size", 0) * n_shared
    total_params = 3 * hidden * intermediate
    return _weight_bytes(total_params, quantization)


def _layernorm_bytes_per_layer(cfg: Dict[str, Any]) -> int:
    hidden = cfg["hidden_size"]
    return hidden * 2 * 2  # 2 norms × hidden_size × 2 bytes (BF16)


def _embedding_bytes(cfg: Dict[str, Any]) -> int:
    return cfg["vocab_size"] * cfg["hidden_size"] * 2


def _lm_head_bytes(cfg: Dict[str, Any]) -> int:
    if cfg.get("tie_word_embeddings", True):
        return 0
    return cfg["vocab_size"] * cfg["hidden_size"] * 2


def _kv_dtype_bytes(kv_cache_dtype: str, cfg: Optional[Dict[str, Any]] = None) -> float:
    if kv_cache_dtype == "tq4":
        # TQ4 stores two 4-bit payload streams plus per-KV-head metadata:
        # K norm (2 bytes) + V scale/zero (4 bytes). Return bytes per KV
        # element, where the caller multiplies by K+V element count.
        if cfg is not None and not _is_mla(cfg):
            head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
            return 0.5 + (3.0 / head_dim)
        return 0.5234375  # exact for head_dim=128
    dtypes = {
        "fp8_e4m3": 1, "fp8_e5m2": 1, "fp8": 1,
        "polar4": 0.625,
        "k8v4": 0.8125,
        "k8v6": 1.0,
        "k7v4": 0.8125,
        "k6v6": 0.875,
        "k6v4": 0.75,
        "k4v4": 0.625,
        "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2,
        "auto": 2,
    }
    return dtypes.get(kv_cache_dtype, 2)


def _kv_bytes_per_token_per_layer(cfg: Dict[str, Any], kv_cache_dtype: str) -> int:
    dtype_bytes = _kv_dtype_bytes(kv_cache_dtype, cfg)
    if _is_mla(cfg):
        kv_lora = cfg["kv_lora_rank"]
        qk_rope = cfg["qk_rope_head_dim"]
        if kv_cache_dtype in ("k4v4", "k6v6"):
            ckv_dim = max(kv_lora, MLA_CKV_KERNEL_MIN_DIM)
            if ckv_dim % 16 != 0 or qk_rope % 16 != 0:
                raise ValueError(
                    "MLA compact-cache dimensions must be divisible by 16; "
                    f"got compressed_dim={ckv_dim}, positional_dim={qk_rope}."
                )
            block_bytes = 10 if kv_cache_dtype == "k4v4" else 14
            return (ckv_dim // 16) * block_bytes + (qk_rope // 16) * block_bytes
        return (kv_lora + qk_rope) * dtype_bytes
    else:
        n_kv_heads = cfg["num_key_value_heads"]
        head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        return 2 * n_kv_heads * head_dim * dtype_bytes


def _is_hybrid(cfg: Dict[str, Any]) -> bool:
    """True if model has a mix of linear and full attention layers."""
    return cfg.get("full_attention_interval", 0) > 0


def _num_full_attention_layers(cfg: Dict[str, Any]) -> int:
    """Number of layers that use full attention (need KV cache)."""
    interval = cfg.get("full_attention_interval", 0)
    if interval <= 0:
        return cfg["num_hidden_layers"]
    return sum(
        1 for i in range(cfg["num_hidden_layers"])
        if (i + 1) % interval == 0
    )


def _linear_attention_bytes_per_layer(cfg: Dict[str, Any], quantization: str) -> int:
    """Weight bytes for one linear attention (Gated DeltaNet) layer."""
    hidden = cfg["hidden_size"]
    nk = cfg.get("linear_num_key_heads", 16)
    nv = cfg.get("linear_num_value_heads", 32)
    dk = cfg.get("linear_key_head_dim", 128)
    dv = cfg.get("linear_value_head_dim", 128)
    kernel = cfg.get("linear_conv_kernel_dim", 4)

    q_dim = nk * dk
    k_dim = nk * dk
    v_dim = nv * dv
    z_dim = nv * dv
    conv_dim = q_dim + k_dim + v_dim

    # Quantizable: in_proj_qkvz, in_proj_ba, out_proj
    qkvz_params = hidden * (q_dim + k_dim + v_dim + z_dim)
    ba_params = hidden * (nv + nv)  # beta + alpha, one per value head each
    out_params = (nv * dv) * hidden
    quantizable_params = qkvz_params + ba_params + out_params

    # Always BF16: conv1d.weight, A_log, dt_bias, norm.weight
    bf16_params = conv_dim * kernel + nv + nv + dv

    return _component_weight_bytes(quantizable_params, quantization) + bf16_params * 2


def _expert_bytes_per_expert(cfg: Dict[str, Any], bits: int = 4, group_size: int = 128) -> int:
    """Expert buffer size for Marlin INT4/INT8 on GPU."""
    hidden = cfg.get("moe_latent_size", 0) or cfg["hidden_size"]
    intermediate = cfg.get("moe_intermediate_size", 0)
    matrix_count = 2 if cfg.get("mlp_hidden_act") == "relu2" else 3
    total_params = matrix_count * hidden * intermediate
    if bits == 4:
        packed = total_params // 2
        scales = (total_params // group_size) * 2
    else:  # INT8
        packed = total_params
        scales = (total_params // group_size) * 2
    return packed + scales


def estimate_int4_expert_cache_bytes(
    model_cfg: ModelConfig,
    group_size: int = 128,
) -> int:
    """Exact Marlin INT4 routed-expert cache bytes from normalized geometry."""
    if model_cfg.n_routed_experts <= 0 or model_cfg.num_moe_layers <= 0:
        return 0
    cfg = {
        "hidden_size": model_cfg.hidden_size,
        "moe_latent_size": model_cfg.moe_latent_size,
        "moe_intermediate_size": model_cfg.moe_intermediate_size,
        "mlp_hidden_act": model_cfg.mlp_hidden_act,
    }
    return (
        _expert_bytes_per_expert(cfg, 4, group_size)
        * model_cfg.n_routed_experts
        * model_cfg.num_moe_layers
    )


def _align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError(
            f"alignment inputs must be positive, got value={value} alignment={alignment}"
        )
    return ((value + alignment - 1) // alignment) * alignment


def _effective_marlin_group_size(
    hidden_size: int,
    intermediate_size: int,
    requested_group_size: int,
) -> int:
    """Mirror the source-bound Marlin group selection in weights/mod.rs."""
    if hidden_size <= 0 or intermediate_size <= 0 or requested_group_size <= 0:
        raise ValueError(
            "Marlin dimensions and requested group size must be positive: "
            f"hidden={hidden_size} intermediate={intermediate_size} "
            f"group={requested_group_size}"
        )
    group_size = requested_group_size
    minimum_dimension = min(hidden_size, intermediate_size)
    while group_size > 32 and minimum_dimension % group_size != 0:
        group_size //= 2
    return group_size


def _marlin_int4_expert_bytes(
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    *,
    gated: bool,
    pad_routed_w2: bool,
) -> int:
    """Exact INT4 payload bytes for one Marlin expert.

    Routed experts use the runtime's W2 padding contract. The combined shared
    expert uses the unpadded layout written by expected_marlin_cache_size().
    """
    if hidden_size % 8 != 0 or intermediate_size % 8 != 0:
        raise ValueError(
            "Marlin INT4 expert dimensions must be divisible by 8: "
            f"hidden={hidden_size} intermediate={intermediate_size}"
        )
    w13_width = intermediate_size * (2 if gated else 1)
    w2_width = hidden_size
    if (
        pad_routed_w2
        and hidden_size == intermediate_size
        and hidden_size % 256 != 0
    ):
        # Kernel-format contract mirrored from marlin_w2_padded_n().
        w2_width += 64
    w13_packed = (hidden_size // 8) * w13_width * 4
    w13_scales = (hidden_size // group_size) * w13_width * 2
    w2_packed = (intermediate_size // 8) * w2_width * 4
    w2_scales = math.ceil(intermediate_size / group_size) * w2_width * 2
    return w13_packed + w13_scales + w2_packed + w2_scales


def _dspark_required_tensor_storage(
    stage_count: int,
) -> Dict[str, int]:
    """Return D-Spark dense tensor names and their runtime bytes per element."""
    if stage_count <= 0:
        raise ValueError("D-Spark requires at least one checkpoint stage")
    required: Dict[str, int] = {}
    for stage_idx in range(stage_count):
        prefix = f"mtp.{stage_idx}"
        for name in (
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn.gate.weight",
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
        ):
            required[f"{prefix}.{name}"] = 2
        for name in (
            "ffn.gate.bias",
            "attn.attn_sink",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ):
            required[f"{prefix}.{name}"] = 4
        if stage_idx == 0:
            required[f"{prefix}.main_proj.weight"] = 2
            required[f"{prefix}.main_norm.weight"] = 2
        if stage_idx == stage_count - 1:
            for name in (
                "norm.weight",
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
                "confidence_head.proj.weight",
            ):
                required[f"{prefix}.{name}"] = 2
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                required[f"{prefix}.{name}"] = 4
    return required


def _safetensors_shapes_for_names(
    model_path: str,
    names: List[str],
) -> Dict[str, tuple[int, ...]]:
    """Read tensor shapes from safetensors metadata without loading payloads."""
    from safetensors import safe_open

    root = Path(model_path).resolve()
    index_path = root / "model.safetensors.index.json"
    weight_map: Dict[str, str]
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            parsed = json.load(handle)
        raw_weight_map = parsed.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise ValueError("D-Spark safetensors index has no usable weight_map")
        weight_map = {str(name): str(shard) for name, shard in raw_weight_map.items()}
    else:
        single = root / "model.safetensors"
        if not single.is_file():
            raise ValueError(
                "D-Spark budget requires model.safetensors.index.json or model.safetensors"
            )
        weight_map = {name: single.name for name in names}

    missing = [name for name in names if name not in weight_map]
    if missing:
        raise ValueError(
            "D-Spark checkpoint tensor inventory is incomplete: " + ", ".join(missing)
        )
    by_shard: Dict[str, List[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)

    shapes: Dict[str, tuple[int, ...]] = {}
    for shard_name, shard_names in by_shard.items():
        shard_path = (root / shard_name).resolve()
        if root != shard_path and root not in shard_path.parents:
            raise ValueError(
                f"D-Spark safetensors shard escapes model directory: {shard_name!r}"
            )
        if not shard_path.is_file():
            raise ValueError(f"D-Spark safetensors shard is missing: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in shard_names:
                if name not in available:
                    raise ValueError(
                        f"D-Spark tensor {name!r} is absent from indexed shard {shard_name!r}"
                    )
                shape = tuple(int(value) for value in handle.get_slice(name).get_shape())
                if not shape or any(value <= 0 for value in shape):
                    raise ValueError(f"D-Spark tensor {name!r} has invalid shape {shape}")
                shapes[name] = shape
    return shapes


def _dspark_dense_graph_bytes(model_path: str, stage_count: int) -> int:
    storage = _dspark_required_tensor_storage(stage_count)
    shapes = _safetensors_shapes_for_names(model_path, list(storage))
    return sum(math.prod(shapes[name]) * bytes_per_element for name, bytes_per_element in storage.items())


def _dspark_runtime_buffer_bytes(
    cfg: Dict[str, Any],
    model_cfg: ModelConfig,
    kv_dtype: str,
) -> int:
    """Persistent D-Spark buffers and transactional snapshots from dimensions."""
    targets = tuple(model_cfg.dspark_target_layer_ids or ())
    stages = len(targets)
    block_size = int(model_cfg.dspark_block_size)
    markov_rank = int(model_cfg.dspark_markov_rank)
    hidden = int(model_cfg.hidden_size)
    vocab = int(cfg["vocab_size"])
    head_dim = int(cfg["head_dim"])
    index_dim = int(cfg["index_head_dim"])
    sliding_window = int(cfg["sliding_window"])
    hc_mult = int(cfg["hc_mult"])
    topk = int(model_cfg.num_experts_per_tok)
    n_experts = int(model_cfg.n_routed_experts)
    if min(
        stages,
        block_size,
        markov_rank,
        hidden,
        vocab,
        head_dim,
        index_dim,
        sliding_window,
        hc_mult,
        topk,
        n_experts,
    ) <= 0:
        raise ValueError("D-Spark runtime buffer geometry is incomplete")

    target_batch = block_size + 1
    snapshot_rows = target_batch + 1

    # Three short auxiliary rings plus the draft/verification workspaces in
    # setup_dspark_gpu_store() and finalize_dspark_graph().
    total = stages * sliding_window * head_dim * 2
    target_history = stages * sliding_window * hc_mult * hidden * 2
    total += target_history
    total += stages * hidden * 2  # main input
    total += hidden * 2  # main hidden
    total += block_size * hc_mult * hidden * 2  # query states
    total += block_size * hidden * 2  # head hidden
    total += markov_rank * 2 + vocab * 4
    total += (hidden + markov_rank) * 2
    total += 2 * target_batch * hc_mult * hidden * 2
    total += target_batch * hc_mult * 4
    total += target_batch * hc_mult * hc_mult * 4
    total += (target_batch + 2) * 4

    # allocate_batch_buffers(block_size + 1); DeepSeek-V4 has no LA or GQA
    # projection buffers in this path.
    total += target_batch * hidden * 2 * 3
    total += target_batch * vocab * 4
    total += target_batch * topk * 4 * 2
    total += target_batch * n_experts * 4

    ratios = [int(value) for value in cfg.get("compress_ratios", [])]
    if len(ratios) < model_cfg.num_hidden_layers:
        raise ValueError(
            "DeepSeek-V4 compress_ratios must cover every target layer for D-Spark budgeting"
        )
    main_row_bytes, index_row_bytes = _deepseek_v4_cache_row_bytes(cfg, kv_dtype)
    recurrent_state_bytes = 0
    target_cache_backup_bytes = 0
    for ratio in ratios[: model_cfg.num_hidden_layers]:
        target_cache_backup_bytes += target_batch * main_row_bytes
        if ratio <= 0:
            continue
        copies = 2 if ratio == 4 else 1
        state_rows = copies * ratio
        state_cols = copies * head_dim
        recurrent_state_bytes += state_rows * state_cols * 4 * 2
        target_cache_backup_bytes += target_batch * main_row_bytes
        if ratio == 4:
            index_state_cols = copies * index_dim
            recurrent_state_bytes += state_rows * index_state_cols * 4 * 2
            target_cache_backup_bytes += target_batch * index_row_bytes

    total += recurrent_state_bytes * snapshot_rows
    # Transactionally back up the touched target cache rows and each D-Spark
    # target-history row. Source cache/state storage is already in the target budget.
    total += target_cache_backup_bytes
    total += stages * target_batch * hc_mult * hidden * 2
    return total


def _dspark_budget_components(
    model_path: str,
    cfg: Dict[str, Any],
    model_cfg: ModelConfig,
    group_size: int,
    kv_dtype: str,
    mode: str,
) -> Dict[str, int | str]:
    if mode not in DSPARK_MODES or mode == "off":
        raise ValueError(f"D-Spark budget mode must be resident or shared, got {mode!r}")
    targets = tuple(model_cfg.dspark_target_layer_ids or ())
    if not model_cfg.is_deepseek_v4 or not targets:
        raise ValueError("D-Spark budgeting requires checkpoint-owned DeepSeek-V4 metadata")
    stages = len(targets)
    effective_group = _effective_marlin_group_size(
        model_cfg.hidden_size,
        model_cfg.moe_intermediate_size,
        group_size,
    )
    routed_hidden = model_cfg.moe_latent_size or model_cfg.hidden_size
    routed_bytes = _marlin_int4_expert_bytes(
        routed_hidden,
        model_cfg.moe_intermediate_size,
        effective_group,
        gated=model_cfg.mlp_hidden_act != "relu2",
        pad_routed_w2=True,
    )
    shared_bytes = 0
    if model_cfg.n_shared_experts > 0:
        shared_bytes = _marlin_int4_expert_bytes(
            model_cfg.hidden_size,
            model_cfg.effective_shared_expert_intermediate,
            effective_group,
            gated=model_cfg.mlp_hidden_act != "relu2",
            pad_routed_w2=False,
        )
    routed_count = stages * model_cfg.n_routed_experts
    routed_cache_bytes = routed_count * routed_bytes
    shared_cache_bytes = stages * shared_bytes
    resident_routed_bytes = (
        routed_count * _align_up(routed_bytes, MARLIN_DEVICE_ALIGNMENT_BYTES)
        if mode == "resident"
        else 0
    )
    return {
        "dense_bytes": _dspark_dense_graph_bytes(model_path, stages),
        "runtime_bytes": _dspark_runtime_buffer_bytes(cfg, model_cfg, kv_dtype),
        "shared_expert_bytes": stages
        * _align_up(shared_bytes, MARLIN_DEVICE_ALIGNMENT_BYTES),
        "resident_expert_bytes": resident_routed_bytes,
        "host_cache_bytes": MARLIN_CACHE_HEADER_BYTES
        + routed_cache_bytes
        + shared_cache_bytes,
        "hcs_cacheable_bytes": 0
        if mode == "resident"
        else routed_count * _align_up(routed_bytes, MARLIN_DEVICE_ALIGNMENT_BYTES),
        "effective_group_size": effective_group,
        "source": "checkpoint safetensors shapes + normalized dimensions + runtime buffer geometry",
    }


def estimate_int4_runtime_ram_bytes(
    model_path: str,
    attention_quant: str,
    hqq_group_size: int = HQQ_DEFAULT_GROUP_SIZE,
    hqq_auto_budget_pct: Optional[float] = None,
) -> tuple[int, str]:
    """Estimate persistent INT4 runtime RAM from checkpoint dimensions.

    This is the same expert-cache plus dual HQQ host-staging contract shown by
    the launcher's detailed budget. It intentionally excludes optional prefix
    cache and operating-system headroom, which are separately user-controlled.
    """
    _cfg, model_cfg = _normalized_launcher_config(model_path)
    expert_bytes = estimate_int4_expert_cache_bytes(model_cfg, hqq_group_size)
    hqq_bytes = 0
    source = "normalized model dimensions"
    if attention_quant in HQQ_ATTENTION_QUANTS:
        layer_bytes, hqq_source = _hqq_attention_layer_bytes_for_budget(
            model_path,
            HQQ_CACHE_PROFILE_BASELINE,
            attention_quant,
            hqq_group_size,
            hqq_auto_budget_pct,
        )
        hqq_bytes = sum(layer_bytes.values()) * 2
        source = f"expert cache + dual HQQ staging ({hqq_source})"
    return expert_bytes + hqq_bytes, source


def _component_weight_bytes(params: int, quant: str, group_size: int = 128) -> int:
    """Weight bytes for per-component quant ("int4", "int8", "awq", or "bf16")."""
    if quant.startswith("hqq"):
        raise ValueError(
            f"HQQ attention bytes must use the dedicated HQQ layout estimator ({quant})."
        )
    if quant in ("int4", "awq"):
        # Marlin INT4 / AWQ estimate.
        return params // 2 + (params // group_size) * 2
    if quant == "int8":
        # Marlin INT8: packed = params, scales = params/group_size * 2
        return params + (params // group_size) * 2
    return params * 2  # BF16


def _manifest_records_by_key(manifest: Dict[str, Any]) -> Dict[tuple[int, str], Dict[str, Any]]:
    records: Dict[tuple[int, str], Dict[str, Any]] = {}
    for record in manifest.get("tensors", []):
        key = (int(record["layer_idx"]), str(record["tensor_name"]))
        if key in records:
            raise RuntimeError(f"HQQ manifest has duplicate tensor record for layer {key[0]} {key[1]}")
        records[key] = record
    return records


def _legacy_hqq_cache_dir(
    model_path: str,
    cache_profile: str,
    nbits: int,
    group_size: int,
) -> Optional[str]:
    legacy_names = {
        4: f"attention_hqq_v{HQQ_ATTENTION_CACHE_VERSION}",
        46: f"attention_hqq46_v{HQQ_ATTENTION_CACHE_VERSION}",
        460: f"attention_hqq46_auto_v{HQQ_ATTENTION_CACHE_VERSION}",
    }
    dirname = legacy_names.get(int(nbits))
    if dirname is None:
        return None
    if cache_profile != HQQ_CACHE_PROFILE_BASELINE:
        dirname = f"{dirname}_calib_selfcal_v1"
    if int(group_size) != HQQ_DEFAULT_GROUP_SIZE:
        dirname = f"{dirname}_g{int(group_size)}"
    return os.path.join(cache_dir_for_model(model_path), dirname)


def _layer_bytes_from_manifest_dir(cache_dir: Optional[str]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[int, int]]]:
    if not cache_dir:
        return None, None
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None, None
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not manifest.get("complete", False):
        return None, None
    per_layer: Dict[int, int] = {}
    for entry in manifest.get("tensors", []):
        file_name = entry.get("file")
        if not file_name or not os.path.isfile(os.path.join(cache_dir, file_name)):
            return None, None
        layer_idx = int(entry["layer_idx"])
        per_layer[layer_idx] = per_layer.get(layer_idx, 0) + int(entry["tensor_bytes"])
    return manifest, per_layer


def _hqq_cache_manifest_and_layer_bytes(
    model_path: str,
    cache_profile: str,
    nbits: int,
    group_size: int,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[int, int]]]:
    manifest = load_hqq_attention_manifest(model_path, cache_profile, nbits, group_size)
    layer_bytes = hqq_attention_cache_layer_bytes(model_path, cache_profile, nbits=nbits, group_size=group_size)
    if manifest is not None and layer_bytes is not None:
        return manifest, layer_bytes
    legacy_manifest, legacy_bytes = _layer_bytes_from_manifest_dir(
        _legacy_hqq_cache_dir(model_path, cache_profile, nbits, group_size)
    )
    if legacy_manifest is not None and legacy_bytes is not None:
        return legacy_manifest, legacy_bytes
    return manifest, layer_bytes


def _hqq_payload_bytes_for_shape(rows: int, cols: int, nbits: int, group_size: int) -> int:
    rows = int(rows)
    cols = int(cols)
    nbits = int(nbits)
    group_size = int(group_size)
    groups = (cols + group_size - 1) // group_size
    padded_cols = groups * group_size
    if nbits == 4:
        packed_cols = (padded_cols + 1) // 2
    elif nbits == 6:
        packed_cols = ((padded_cols + 3) // 4) * 3
    elif nbits == 8:
        packed_cols = padded_cols
    else:
        raise RuntimeError(f"Unsupported HQQ estimate nbits={nbits}")

    packed = rows * packed_cols
    scales = rows * groups * 4
    zeros = rows * groups * 4
    metadata = 5 * 4  # orig_shape[2], group_size, axis, nbits: all int32 tensors
    return packed + scales + zeros + metadata


def _hqq_attention_tensor_shapes_for_layer(
    cfg: Dict[str, Any],
    layer_type: str,
    layer_idx: Optional[int] = None,
) -> list[tuple[str, int, int]]:
    hidden = int(cfg["hidden_size"])
    if str(cfg.get("model_type", "")).lower().replace("-", "_") == "deepseek_v4":
        heads = int(cfg["num_attention_heads"])
        head_dim = int(cfg["head_dim"])
        q_rank = int(cfg["q_lora_rank"])
        o_rank = int(cfg["o_lora_rank"])
        return [
            ("wq_a", q_rank, hidden),
            ("wq_b", heads * head_dim, q_rank),
            ("wkv", head_dim, hidden),
            ("wo_b", hidden, o_rank),
        ]
    if layer_type == "linear_attention":
        nk = int(cfg.get("linear_num_key_heads", 16))
        nv = int(cfg.get("linear_num_value_heads", 32))
        dk = int(cfg.get("linear_key_head_dim", 128))
        dv = int(cfg.get("linear_value_head_dim", 128))
        q_dim = nk * dk
        k_dim = nk * dk
        v_dim = nv * dv
        z_dim = nv * dv
        return [
            ("in_proj_qkvz", q_dim + k_dim + v_dim + z_dim, hidden),
            ("in_proj_ba", nv + nv, hidden),
            ("out_proj", hidden, v_dim),
        ]

    if layer_type not in ("full_attention", "sliding_attention"):
        return []

    if _is_mla(cfg):
        n_heads = int(cfg["num_attention_heads"])
        q_lora = int(cfg.get("q_lora_rank", 0))
        kv_lora = int(cfg["kv_lora_rank"])
        qk_nope = int(cfg["qk_nope_head_dim"])
        qk_rope = int(cfg["qk_rope_head_dim"])
        v_head = int(cfg["v_head_dim"])
        q_rows = n_heads * (qk_nope + qk_rope)
        shapes: list[tuple[str, int, int]] = []
        if q_lora:
            shapes.append(("q_a_proj", q_lora, hidden))
            shapes.append(("q_b_proj", q_rows, q_lora))
        else:
            shapes.append(("q_proj", q_rows, hidden))
        shapes.append(("kv_a_proj_with_mqa", kv_lora + qk_rope, hidden))
        shapes.append(("o_proj", hidden, n_heads * v_head))
        return shapes

    other = cfg.get("attention_other_setting") if layer_type == "sliding_attention" else None
    if isinstance(other, dict) and str(other.get("attention_type", "")) == "sliding_attention":
        n_heads = int(other.get("num_attention_heads", cfg["num_attention_heads"]))
        n_kv_heads = int(
            other.get(
                "num_key_value_heads",
                other.get("num_attention_groups", cfg.get("num_key_value_heads", cfg.get("num_attention_groups", n_heads))),
            )
        )
        head_dim = int(other.get("head_dim", cfg.get("head_dim", hidden // n_heads)))
    else:
        n_heads = int(cfg["num_attention_heads"])
        n_kv_heads = int(cfg.get("num_key_value_heads", cfg.get("num_attention_groups", n_heads)))
        head_dim = int(cfg.get("head_dim", hidden // n_heads))
    gated_attention = bool(
        cfg.get(
            "attn_output_gate",
            cfg.get("model_type") in ("qwen3_next", "qwen3_5_moe_text"),
        )
    )
    q_rows = n_heads * head_dim * (2 if gated_attention else 1)
    kv_rows = n_kv_heads * head_dim
    o_cols = n_heads * head_dim
    return [
        ("q_proj", q_rows, hidden),
        ("k_proj", kv_rows, hidden),
        ("v_proj", kv_rows, hidden),
        ("o_proj", hidden, o_cols),
        ("fused_qkv", q_rows + kv_rows + kv_rows, hidden),
    ]


def _layer_type_for_hqq_budget(cfg: Dict[str, Any], layer_idx: int) -> str:
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_idx < len(layer_types):
        layer_type = str(layer_types[layer_idx])
        if layer_type in ("full_attention", "sliding_attention", "linear_attention", "mamba2", "moe"):
            return layer_type
        if layer_type == "attention":
            return "full_attention"

    interval = int(cfg.get("full_attention_interval", 0) or 0)
    if interval > 0:
        return "full_attention" if (layer_idx + 1) % interval == 0 else "linear_attention"
    return "full_attention"


def _estimate_hqq_attention_layer_bytes(
    cfg: Dict[str, Any],
    attention_quant: str,
    group_size: int,
    budget_pct: Optional[float],
) -> tuple[Dict[int, int], str]:
    hqq_nbits = attention_quant_cache_nbits(attention_quant)
    if hqq_nbits is None:
        raise RuntimeError(f"attention_quant={attention_quant} is not an HQQ attention mode")

    def tensor_bytes(shapes: list[tuple[str, int, int]], nbits_for_name) -> int:
        total = 0
        for tensor_name, rows, cols in shapes:
            total += _hqq_payload_bytes_for_shape(rows, cols, int(nbits_for_name(tensor_name)), group_size)
        return total

    per_layer: Dict[int, int] = {}
    num_layers = int(cfg["num_hidden_layers"])
    for layer_idx in range(num_layers):
        shapes = _hqq_attention_tensor_shapes_for_layer(
            cfg,
            _layer_type_for_hqq_budget(cfg, layer_idx),
            layer_idx,
        )
        if not shapes:
            per_layer[layer_idx] = 0
            continue

        if attention_quant == "hqq46":
            per_layer[layer_idx] = tensor_bytes(shapes, hqq46_tensor_nbits)
        elif is_hqq_auto_attention(attention_quant):
            direct_nbits = hqq_auto_direct_edge_nbits(attention_quant, budget_pct)
            if direct_nbits is not None:
                per_layer[layer_idx] = tensor_bytes(shapes, lambda _name: direct_nbits)
            else:
                policy = hqq_auto_promotion_policy(attention_quant)
                base_nbits = int(policy["base_nbits"])
                promoted_nbits = int(policy["promoted_nbits"])
                base_bytes = tensor_bytes(shapes, lambda _name: base_nbits)
                promoted_bytes = tensor_bytes(shapes, lambda _name: promoted_nbits)
                pct = float(budget_pct if budget_pct is not None else 0.0) / 100.0
                per_layer[layer_idx] = base_bytes + int((promoted_bytes - base_bytes) * pct)
        else:
            per_layer[layer_idx] = tensor_bytes(shapes, lambda _name: hqq_nbits)

    return per_layer, f"estimated from model dimensions ({attention_quant}, group_size={int(group_size)})"


def _derive_hqq46_layer_bytes_from_edges(
    model_path: str,
    cache_profile: str,
    group_size: int,
) -> Optional[Dict[int, int]]:
    base_manifest, _ = _hqq_cache_manifest_and_layer_bytes(model_path, cache_profile, 4, group_size)
    promoted_manifest, _ = _hqq_cache_manifest_and_layer_bytes(model_path, cache_profile, 6, group_size)
    if base_manifest is None or promoted_manifest is None:
        return None

    promoted_by_key = _manifest_records_by_key(promoted_manifest)
    per_layer: Dict[int, int] = {}
    for base_record in base_manifest.get("tensors", []):
        key = (int(base_record["layer_idx"]), str(base_record["tensor_name"]))
        selected_record = promoted_by_key.get(key) if hqq46_tensor_nbits(key[1]) == 6 else base_record
        if selected_record is None:
            return None
        per_layer[key[0]] = per_layer.get(key[0], 0) + int(selected_record["tensor_bytes"])
    return per_layer


def _derive_hqq_auto_layer_bytes_from_edges(
    model_path: str,
    cache_profile: str,
    attention_quant: str,
    group_size: int,
    budget_pct: Optional[float],
) -> Optional[Dict[int, int]]:
    direct_nbits = hqq_auto_direct_edge_nbits(attention_quant, budget_pct)
    if direct_nbits is not None:
        _manifest, layer_bytes = _hqq_cache_manifest_and_layer_bytes(
            model_path,
            cache_profile,
            direct_nbits,
            group_size,
        )
        return layer_bytes

    policy = hqq_auto_promotion_policy(attention_quant)
    base_nbits = int(policy["base_nbits"])
    promoted_nbits = int(policy["promoted_nbits"])
    base_manifest, _ = _hqq_cache_manifest_and_layer_bytes(model_path, cache_profile, base_nbits, group_size)
    promoted_manifest, _ = _hqq_cache_manifest_and_layer_bytes(model_path, cache_profile, promoted_nbits, group_size)
    if base_manifest is None or promoted_manifest is None:
        return None

    promoted_by_key = _manifest_records_by_key(promoted_manifest)
    ordered_candidates = []
    base_records = []
    for base_record in base_manifest.get("tensors", []):
        key = (int(base_record["layer_idx"]), str(base_record["tensor_name"]))
        promoted_record = promoted_by_key.get(key)
        if promoted_record is None:
            return None
        ordered_candidates.append(hqq_auto_candidate_from_records(base_record, promoted_record))
        base_records.append(base_record)

    promotion_span_bytes = sum(int(candidate["extra_bytes"]) for candidate in ordered_candidates)
    budget_bytes = hqq_auto_budget_bytes_from_pct(
        float(budget_pct),
        promotion_span_bytes,
        attention_quant,
    )
    selected_keys, _summary = select_hqq_auto_promotions(ordered_candidates, budget_bytes)
    selected_extra_bytes = sum(
        int(candidate["extra_bytes"])
        for candidate in ordered_candidates
        if (int(candidate["layer_idx"]), str(candidate["tensor_name"])) in selected_keys
    )
    if selected_extra_bytes < budget_bytes:
        for candidate in sorted(ordered_candidates, key=lambda item: int(item["extra_bytes"])):
            key = (int(candidate["layer_idx"]), str(candidate["tensor_name"]))
            if key in selected_keys:
                continue
            extra_bytes = int(candidate["extra_bytes"])
            if selected_extra_bytes + extra_bytes > budget_bytes:
                continue
            selected_keys.add(key)
            selected_extra_bytes += extra_bytes
            if selected_extra_bytes >= budget_bytes:
                break

    per_layer: Dict[int, int] = {}
    for base_record in base_records:
        key = (int(base_record["layer_idx"]), str(base_record["tensor_name"]))
        selected_record = promoted_by_key[key] if key in selected_keys else base_record
        per_layer[key[0]] = per_layer.get(key[0], 0) + int(selected_record["tensor_bytes"])
    return per_layer


def _hqq_attention_layer_bytes_for_budget(
    model_path: str,
    cache_profile: str,
    attention_quant: str,
    group_size: int,
    budget_pct: Optional[float],
) -> tuple[Dict[int, int], str]:
    hqq_nbits = attention_quant_cache_nbits(attention_quant)
    if hqq_nbits is None:
        raise RuntimeError(f"attention_quant={attention_quant} is not an HQQ attention mode")

    exact_bytes = hqq_attention_cache_layer_bytes(
        model_path,
        cache_profile,
        nbits=hqq_nbits,
        group_size=group_size,
    )
    if exact_bytes is not None:
        return exact_bytes, hqq_attention_cache_dir(model_path, cache_profile, nbits=hqq_nbits, group_size=group_size)

    legacy_manifest, legacy_bytes = _layer_bytes_from_manifest_dir(
        _legacy_hqq_cache_dir(model_path, cache_profile, hqq_nbits, group_size)
    )
    if legacy_manifest is not None and legacy_bytes is not None:
        source = _legacy_hqq_cache_dir(model_path, cache_profile, hqq_nbits, group_size)
        return legacy_bytes, f"legacy artifact-size estimate from {source}"

    derived_bytes = None
    if attention_quant == "hqq46":
        derived_bytes = _derive_hqq46_layer_bytes_from_edges(model_path, cache_profile, group_size)
    elif is_hqq_auto_attention(attention_quant):
        derived_bytes = _derive_hqq_auto_layer_bytes_from_edges(
            model_path,
            cache_profile,
            attention_quant,
            group_size,
            budget_pct,
        )
    if derived_bytes is not None:
        if is_hqq_auto_attention(attention_quant):
            policy = hqq_auto_promotion_policy(attention_quant)
            source = f"derived from validated HQQ{policy['base_nbits']} + HQQ{policy['promoted_nbits']} artifacts"
        else:
            source = "derived from validated HQQ4 + HQQ6 artifacts"
        return derived_bytes, source

    cfg = _read_model_config(model_path)
    return _estimate_hqq_attention_layer_bytes(
        cfg,
        attention_quant,
        group_size,
        budget_pct,
    )


def _cpu_expert_bytes_per_expert(cfg: Dict[str, Any], bits: int = 4, group_size: int = 128) -> int:
    """CPU expert size (INT4 or INT8 with group scales)."""
    hidden = cfg.get("moe_latent_size", 0) or cfg["hidden_size"]
    intermediate = cfg.get("moe_intermediate_size", 0)
    matrix_count = 2 if cfg.get("mlp_hidden_act") == "relu2" else 3
    total_params = matrix_count * hidden * intermediate
    if bits == 4:
        return total_params // 2 + (total_params // group_size) * 2
    else:  # INT8
        return total_params + (total_params // group_size) * 2


def _detect_windows_total_ram_gb() -> int:
    """Read native-Windows physical RAM through the OS-owned API."""
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys // (1024**3))
    except (AttributeError, OSError, ValueError):
        pass
    return 0


def _detect_total_ram_gb() -> int:
    """Auto-detect physical system RAM in GiB on Unix, WSL, and Windows."""
    if os.name == "nt":
        total = _detect_windows_total_ram_gb()
        if total > 0:
            return total
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return kb // (1024 * 1024)
    except (FileNotFoundError, ValueError):
        pass
    return 0


detect_total_ram_gb = _detect_total_ram_gb


def _deepseek_v4_cache_row_bytes(cfg: Dict[str, Any], kv_dtype: str) -> tuple[int, int]:
    head_dim = int(cfg["head_dim"])
    rope_dim = int(cfg["qk_rope_head_dim"])
    index_dim = int(cfg["index_head_dim"])
    if head_dim <= rope_dim or rope_dim <= 0 or index_dim <= 0:
        raise ValueError("DeepSeek-V4 cache geometry is invalid")
    if kv_dtype == "bf16":
        return head_dim * 2, index_dim * 2
    if kv_dtype != "native":
        raise ValueError(
            f"DeepSeek-V4 cache mode must be native or bf16, got {kv_dtype!r}"
        )
    quant_cols = head_dim - rope_dim
    main_row = quant_cols + math.ceil(quant_cols / 64) + rope_dim * 2
    index_row = math.ceil(index_dim / 2) + math.ceil(index_dim / 32)
    return main_row, index_row


def _deepseek_v4_rank_cache_bytes(
    cfg: Dict[str, Any],
    start_layer: int,
    end_layer: int,
    tokens: int,
    kv_dtype: str,
) -> int:
    if tokens <= 0:
        return 0
    ratios = [int(value) for value in cfg.get("compress_ratios", [])]
    runtime_layers = int(cfg["num_hidden_layers"])
    if len(ratios) < runtime_layers:
        raise ValueError(
            "DeepSeek-V4 compress_ratios must contain at least one entry per runtime layer"
        )
    ratios = ratios[:runtime_layers]
    head_dim = int(cfg["head_dim"])
    index_dim = int(cfg["index_head_dim"])
    window = int(cfg["sliding_window"])
    main_row_bytes, index_row_bytes = _deepseek_v4_cache_row_bytes(cfg, kv_dtype)
    total = 0
    for ratio in ratios[start_layer:end_layer]:
        total += window * main_row_bytes
        if ratio <= 0:
            continue
        compressed_rows = max(1, math.ceil(tokens / ratio))
        total += compressed_rows * main_row_bytes
        overlap_width = 2 if ratio == 4 else 1
        state_rows = overlap_width * ratio
        state_cols = overlap_width * head_dim
        total += state_rows * state_cols * 4 * 2
        if ratio == 4:
            total += compressed_rows * index_row_bytes
            index_state_cols = overlap_width * index_dim
            total += state_rows * index_state_cols * 4 * 2
    return total


def _deepseek_v4_tokens_for_rank_budget(
    cfg: Dict[str, Any],
    start_layer: int,
    end_layer: int,
    budget_bytes: int,
    token_limit: int,
    kv_dtype: str,
) -> int:
    if budget_bytes <= 0 or token_limit <= 0:
        return 0
    if _deepseek_v4_rank_cache_bytes(
        cfg, start_layer, end_layer, 1, kv_dtype
    ) > budget_bytes:
        return 0
    if _deepseek_v4_rank_cache_bytes(
        cfg, start_layer, end_layer, token_limit, kv_dtype
    ) <= budget_bytes:
        return token_limit
    low, high = 1, token_limit
    while low < high:
        middle = (low + high + 1) // 2
        if _deepseek_v4_rank_cache_bytes(
            cfg, start_layer, end_layer, middle, kv_dtype
        ) <= budget_bytes:
            low = middle
        else:
            high = middle - 1
    return low


def _deepseek_v4_attention_weight_bytes(
    cfg: Dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Return per-layer BF16 total and BF16 residual beside phase-one HQQ."""
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["head_dim"])
    q_rank = int(cfg["q_lora_rank"])
    o_rank = int(cfg["o_lora_rank"])
    o_groups = int(cfg["o_groups"])
    index_dim = int(cfg["index_head_dim"])
    index_heads = int(cfg["index_n_heads"])
    ratios = [int(value) for value in cfg.get("compress_ratios", [])]
    runtime_layers = int(cfg["num_hidden_layers"])
    if len(ratios) < runtime_layers or heads % o_groups != 0:
        raise ValueError("DeepSeek-V4 attention-weight geometry is invalid")
    ratios = ratios[:runtime_layers]

    quantized_params = (
        hidden * q_rank
        + q_rank * heads * head_dim
        + hidden * head_dim
        + hidden * o_rank
    )
    residual_base_bytes = (
        o_rank * (heads // o_groups) * head_dim * 2
        + (q_rank + head_dim + heads) * 2
    )
    totals: list[int] = []
    residuals: list[int] = []
    for ratio in ratios:
        residual = residual_base_bytes
        if ratio > 0:
            copies = 2 if ratio == 4 else 1
            projected = copies * head_dim
            residual += (2 * hidden * projected + ratio * projected * 2 + head_dim) * 2
            if ratio == 4:
                index_projected = 2 * index_dim
                residual += (
                    2 * hidden * index_projected
                    + ratio * index_projected * 2
                    + index_dim
                    + q_rank * index_heads * index_dim
                    + hidden * index_heads
                ) * 2
        residuals.append(residual)
        totals.append(residual + quantized_params * 2)
    return totals, residuals


def compute_launcher_budget(
    model_path: str,
    pp_partition: List[int],
    layer_group_size: int = 1,
    kv_dtype: str = "k6v6",
    gpu_expert_bits: int = 4,
    expert_group_size: int = 128,
    attention_quant: str = "bf16",
    hqq_cache_profile: str = "baseline",
    hqq_group_size: int = 128,
    hqq_auto_budget_pct: Optional[float] = None,
    shared_expert_quant: str = "int8",
    dense_mlp_quant: str = "int8",
    lm_head_quant: str = "int8",
    gpu_vram_mb: int = 0,
    total_ram_gb: int = 0,
    kv_cache_mb: int = 1000,
    max_context_tokens: int = 0,
    prefill_chunk_size: int = 10000,
    dspark_mode: str = "off",
) -> Dict[str, Any]:
    """Compute VRAM + RAM budget for the launcher TUI.

    Supports per-component quantization and layer_group_size modes.
    Returns a dict with worst-case rank breakdown for display.
    """
    import math

    cfg, model_cfg = _normalized_launcher_config(model_path)
    model_context_limit = int(model_cfg.max_position_embeddings)
    if model_context_limit <= 0:
        raise ValueError(
            f"Model has invalid max_position_embeddings={model_context_limit}"
        )
    if max_context_tokens < 0:
        raise ValueError(
            f"max_context_tokens must be non-negative, got {max_context_tokens}"
        )
    if max_context_tokens > model_context_limit:
        raise ValueError(
            f"max_context_tokens {max_context_tokens} exceeds model limit "
            f"{model_context_limit}"
        )
    effective_context_limit = max_context_tokens or model_context_limit
    effective_kv_capacity_tokens = (
        math.ceil(effective_context_limit / PAGE_SIZE) * PAGE_SIZE
    )

    if gpu_vram_mb <= 0:
        gpu_vram_mb = _detect_gpu_vram_bytes() // (1024 * 1024)
    if total_ram_gb <= 0:
        total_ram_gb = _detect_total_ram_gb()

    num_ranks = len(pp_partition)
    if dspark_mode not in DSPARK_MODES:
        raise ValueError(
            f"D-Spark mode must be one of {DSPARK_MODES}, got {dspark_mode!r}"
        )
    if dspark_mode != "off" and num_ranks != 1:
        raise ValueError(
            "D-Spark launcher budgeting requires exactly one pipeline rank"
        )
    if dspark_mode != "off" and gpu_expert_bits != 4:
        raise ValueError(
            f"D-Spark requires the INT4 production expert path, got INT{gpu_expert_bits}"
        )
    total_layers = cfg["num_hidden_layers"]
    is_deepseek_v4 = str(cfg.get("model_type", "")).lower().replace("-", "_") == "deepseek_v4"
    is_mla = _is_mla(cfg)
    hybrid = model_cfg.is_hybrid
    n_experts = model_cfg.n_routed_experts
    first_k_dense = model_cfg.first_k_dense_replace
    moe_layer_indices = {
        layer_idx
        for layer_idx in range(total_layers)
        if model_cfg.is_moe_layer(layer_idx)
    }

    # Linear attention estimates are only used by non-HQQ component estimates.
    # HQQ uses the dedicated artifact-layout estimator below so packed payload
    # and scale/zero metadata are counted correctly.
    linear_attn_bpl = (
        _linear_attention_bytes_per_layer(cfg, attention_quant)
        if hybrid and not attention_quant.startswith("hqq")
        else 0
    )

    # Attention params per layer
    if is_mla:
        q_lora = cfg.get("q_lora_rank", 0)
        kv_lora = cfg["kv_lora_rank"]
        n_heads = cfg["num_attention_heads"]
        qk_nope = cfg["qk_nope_head_dim"]
        qk_rope = cfg["qk_rope_head_dim"]
        v_head = cfg["v_head_dim"]
        if q_lora:
            q_params = cfg["hidden_size"] * q_lora + q_lora * n_heads * (qk_nope + qk_rope)
        else:
            q_params = cfg["hidden_size"] * n_heads * (qk_nope + qk_rope)
        kv_a_params = cfg["hidden_size"] * (kv_lora + qk_rope)
        kv_b_params = kv_lora * n_heads * (qk_nope + v_head)
        o_params = n_heads * v_head * cfg["hidden_size"]
        attn_params_per_layer = q_params + kv_a_params + kv_b_params + o_params
    else:
        hidden = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        n_kv_heads = cfg.get("num_key_value_heads", n_heads)
        head_dim = cfg.get("head_dim", hidden // n_heads)
        attn_params_per_layer = (
            hidden * n_heads * head_dim +          # Q
            hidden * n_kv_heads * head_dim +        # K
            hidden * n_kv_heads * head_dim +        # V
            n_heads * head_dim * hidden             # O
        )

    hqq_layer_bytes = None
    hqq_budget_source = None
    if attention_quant in HQQ_ATTENTION_QUANTS:
        hqq_layer_bytes, hqq_budget_source = _hqq_attention_layer_bytes_for_budget(
            model_path,
            hqq_cache_profile,
            attention_quant,
            hqq_group_size,
            hqq_auto_budget_pct,
        )
    deepseek_v4_attention_total = None
    deepseek_v4_attention_residual = None
    if is_deepseek_v4:
        (
            deepseek_v4_attention_total,
            deepseek_v4_attention_residual,
        ) = _deepseek_v4_attention_weight_bytes(cfg)

    dspark = None
    if dspark_mode != "off":
        dspark = _dspark_budget_components(
            model_path,
            cfg,
            model_cfg,
            expert_group_size,
            kv_dtype,
            dspark_mode,
        )

    # Expert buffer bytes per expert (GPU Marlin format)
    expert_buf_bytes = _expert_bytes_per_expert(
        cfg, gpu_expert_bits, expert_group_size,
    ) if n_experts > 0 else 0

    # Shared expert params per MoE layer
    hidden = cfg["hidden_size"]
    n_shared = cfg.get("n_shared_experts", 0)
    shared_inter = cfg.get("shared_expert_intermediate_size", 0)
    if not shared_inter and n_shared:
        shared_inter = n_shared * cfg.get("moe_intermediate_size", 0)
    shared_params_per_moe = 3 * hidden * shared_inter if shared_inter else 0

    # Dense MLP params per dense layer
    dense_inter = cfg.get("intermediate_size", cfg.get("moe_intermediate_size", 0))
    dense_mlp_params_per_layer = 3 * hidden * dense_inter

    # KV bytes per token per layer
    kv_b = 0 if is_deepseek_v4 else _kv_dtype_bytes(kv_dtype, cfg)
    if is_deepseek_v4:
        _deepseek_v4_cache_row_bytes(cfg, kv_dtype)
        kv_ptl = 0
    elif is_mla:
        kv_ptl = (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * kv_b
    else:
        kv_ptl = 0
    kv_bytes_by_layer = [
        0
        if is_deepseek_v4
        else _model_kv_bytes_per_token_for_layer(model_cfg, cfg, layer_idx, kv_dtype)
        for layer_idx in range(total_layers)
    ]

    cuda_overhead = DEFAULT_CUDA_OVERHEAD_MB

    # Per-rank budget
    ranks = []
    total_model_layers = cfg["num_hidden_layers"]

    # Replicated on every GPU: embeddings and lm_head
    base_embed_bytes = cfg["vocab_size"] * hidden * 2      # always BF16
    base_lmhead_bytes = 0
    if not cfg.get("tie_word_embeddings", True):
        base_lmhead_bytes = _component_weight_bytes(
            cfg["vocab_size"] * hidden, lm_head_quant, expert_group_size,
        )

    # Gate weights and norms are PERMANENT on GPU for ALL layers (not streamed).
    # Additionally, f32 copies of gate weights are kept for routing precision.
    total_moe_layers_all = len(moe_layer_indices)
    gate_bytes_all_layers = hidden * n_experts * 2 * total_moe_layers_all  # BF16
    gate_f32_bytes_all_layers = hidden * n_experts * 4 * total_moe_layers_all  # F32 routing copy
    norm_bytes_all_layers = 2 * hidden * 2 * total_layers  # BF16

    # Rust prefill workspace: scratch buffers for intermediates during prefill.
    # Sized from model dimensions.
    # Main cost: MoE intermediate buffers + attention scratch + LA scratch.
    top_k = cfg.get("num_experts_per_tok", cfg.get("num_selected_experts", 8))
    moe_inter = cfg.get("moe_intermediate_size", 0)
    # Rust prefill scratch: max(hidden, inter*2) * max_tokens * 4 (FP32) + expert buffers
    rust_prefill_workspace_bytes = max(hidden, (moe_inter * 2 if moe_inter > 0 else 0)) * 5000 * 4

    # Prefill workspace: intermediate tensors during MoE forward.
    # fused_marlin_moe allocates intermediate_cache1/3 and intermediate_cache2.
    # Estimate up to 10k tokens, but cap to what the KV cache can actually hold.
    top_k = cfg.get("num_experts_per_tok", cfg.get("num_selected_experts", 8))
    moe_inter = cfg.get("moe_intermediate_size", 0)
    total_full_attn = model_cfg.num_full_attention_layers
    kv_total_per_token = sum(kv_bytes_by_layer)
    max_kv_tokens = (
        effective_context_limit
        if is_deepseek_v4
        else (
            min(
                int(kv_cache_mb * 1024 * 1024 // kv_total_per_token),
                effective_context_limit,
            )
            if kv_total_per_token > 0
            else effective_context_limit
        )
    )
    prefill_chunk = min(prefill_chunk_size, max_kv_tokens)
    prefill_workspace_bytes = 0
    if moe_inter > 0 and top_k > 0:
        # intermediate_cache13: [M * topk, 2*N] bf16
        # intermediate_cache2: [M * topk, hidden] bf16
        prefill_workspace_bytes = (
            prefill_chunk * top_k * 2 * moe_inter * 2 +  # cache13
            prefill_chunk * top_k * hidden * 2             # cache2
        )
    # Dense MLP workspace: gate [M, inter] + up [M, inter] + cat [M, 2*inter], all bf16
    # Peak when all 3 coexist before silu_and_mul: M * intermediate_size * 8 bytes
    if dense_inter > 0:
        dense_mlp_workspace_bytes = prefill_chunk * dense_inter * 8
        prefill_workspace_bytes = max(prefill_workspace_bytes, dense_mlp_workspace_bytes)

    for rank_idx in range(num_ranks):
        rank_start = sum(pp_partition[:rank_idx])
        rank_end = rank_start + pp_partition[rank_idx]
        n_layers = pp_partition[rank_idx]

        # Count full attention vs linear attention layers IN THIS RANK
        rank_full_attn = sum(
            1 for i in range(rank_start, rank_end)
            if model_cfg.is_full_attention_layer(i)
        )
        rank_linear_attn = sum(
            1 for i in range(rank_start, rank_end)
            if model_cfg.is_linear_attention_layer(i)
        )

        # Expert related components (per-rank)
        mn = sum(1 for i in range(rank_start, rank_end) if i in moe_layer_indices)
        dn = n_layers - mn

        # layer_group_size controls expert DMA pipelining (HCS), NOT attention/shared
        # expert streaming. Attention and shared experts are permanently on GPU for
        # ALL layers (offloaded only for very large models, decided at runtime).
        # The budget shows the peak (non-streaming) footprint so the user sees the
        # real VRAM impact of their quant choices.
        if layer_group_size >= 1:
            group_size = min(layer_group_size, max(mn, 1))
            streaming = True
        else:
            group_size = 0  # not used
            streaming = False

        # Attention weights: ALL layers permanently on GPU (not capped by group_size)
        if is_deepseek_v4 and hqq_layer_bytes is not None:
            rank_attn_bytes = sum(
                hqq_layer_bytes.get(i, 0) + deepseek_v4_attention_residual[i]
                for i in range(rank_start, rank_end)
            )
        elif is_deepseek_v4:
            rank_attn_bytes = sum(
                deepseek_v4_attention_total[i] for i in range(rank_start, rank_end)
            )
        elif hqq_layer_bytes is not None:
            rank_attn_bytes = sum(hqq_layer_bytes.get(i, 0) for i in range(rank_start, rank_end))
        else:
            rank_attn_bytes = _component_weight_bytes(attn_params_per_layer, attention_quant) * rank_full_attn
            rank_attn_bytes += linear_attn_bpl * rank_linear_attn

        # Norms and gates are PERMANENT on GPU for ALL layers (not streamed)
        rank_norm_bytes = norm_bytes_all_layers
        gate_bytes = gate_bytes_all_layers + gate_f32_bytes_all_layers

        # Shared experts: ALL layers permanently on GPU (not capped by group_size)
        shared_bytes = _component_weight_bytes(
            shared_params_per_moe, shared_expert_quant, expert_group_size,
        ) * mn if shared_params_per_moe else 0
        dense_mlp_bytes = _component_weight_bytes(
            dense_mlp_params_per_layer, dense_mlp_quant, expert_group_size,
        ) * dn

        # Expert buffers (GPU side)
        # layer_group_size: 0 = persistent (all layers), >=1 = N layers at a time
        # When streaming with DMA pipelining, TWO groups are resident
        # simultaneously: current group computing + next group prefetching.
        if mn > 0 and n_experts > 0:
            if layer_group_size == 0:
                ebuf_bytes = expert_buf_bytes * n_experts * mn
                emode = "persistent"
            elif layer_group_size >= 1:
                # Pipeline doubles the buffer: current + prefetched group
                ebuf_bytes = expert_buf_bytes * n_experts * group_size * 2
                emode = f"grouped({layer_group_size})"
            else:
                ebuf_bytes = expert_buf_bytes * n_experts * 2
                emode = "grouped(1)"
        else:
            ebuf_bytes = 0
            emode = "n/a"

        recurrent_state_bytes = _persistent_recurrent_state_bytes(
            model_cfg, rank_start, rank_end
        )
        dspark_dense_bytes = int(dspark["dense_bytes"]) if dspark is not None else 0
        dspark_runtime_bytes = int(dspark["runtime_bytes"]) if dspark is not None else 0
        dspark_shared_expert_bytes = (
            int(dspark["shared_expert_bytes"]) if dspark is not None else 0
        )
        dspark_resident_expert_bytes = (
            int(dspark["resident_expert_bytes"]) if dspark is not None else 0
        )
        total_bytes = (
            rank_attn_bytes + rank_norm_bytes +
            base_embed_bytes + base_lmhead_bytes +
            shared_bytes + dense_mlp_bytes +
            gate_bytes +
            ebuf_bytes +
            recurrent_state_bytes +
            rust_prefill_workspace_bytes +
            prefill_workspace_bytes +
            dspark_dense_bytes +
            dspark_runtime_bytes +
            dspark_shared_expert_bytes +
            dspark_resident_expert_bytes +
            cuda_overhead * 1024 * 1024
        )
        free_bytes = gpu_vram_mb * 1024 * 1024 - total_bytes
        # KV cache only for layers in this rank's partition. DeepSeek-V4 has a
        # fixed raw ring plus ratio-dependent compressed/index state, so its
        # exact nonlinear model is inverted rather than approximated per token.
        kv_per_rank = sum(kv_bytes_by_layer[rank_start:rank_end])
        if is_deepseek_v4:
            kv_tokens = _deepseek_v4_tokens_for_rank_budget(
                cfg,
                rank_start,
                rank_end,
                max(0, int(free_bytes)),
                effective_kv_capacity_tokens,
                kv_dtype,
            )
            allocation_budget = min(
                kv_cache_mb * 1024 * 1024,
                max(0, int(free_bytes)),
            )
            kv_alloc_tokens = _deepseek_v4_tokens_for_rank_budget(
                cfg,
                rank_start,
                rank_end,
                allocation_budget,
                effective_kv_capacity_tokens,
                kv_dtype,
            )
            kv_alloc_bytes = (
                _deepseek_v4_rank_cache_bytes(
                    cfg, rank_start, rank_end, kv_alloc_tokens, kv_dtype
                )
                if kv_alloc_tokens > 0
                else 0
            )
        else:
            kv_tokens = (
                min(
                    max(0, int(free_bytes // kv_per_rank)),
                    effective_kv_capacity_tokens,
                )
                if kv_per_rank > 0 and free_bytes > 0
                else 0
            )
            kv_alloc_bytes = min(
                kv_cache_mb * 1024 * 1024,
                effective_kv_capacity_tokens * kv_per_rank,
            )
            kv_alloc_tokens = (
                max(0, int(kv_alloc_bytes // kv_per_rank)) if kv_per_rank > 0 else 0
            )
        total_with_kv = total_bytes + min(kv_alloc_bytes, max(0, free_bytes))
        free_after_kv = gpu_vram_mb * 1024 * 1024 - total_with_kv

        MB = 1024 * 1024
        ranks.append({
            "rank": rank_idx,
            "n_layers": n_layers,
            "moe_layers": mn,
            "dense_layers": dn,
            "attention_mb": rank_attn_bytes / MB,
            "shared_expert_mb": shared_bytes / MB,
            "dense_mlp_mb": dense_mlp_bytes / MB,
            "expert_buffer_mb": ebuf_bytes / MB,
            "expert_mode": emode,
            "embedding_mb": base_embed_bytes / MB,
            "lm_head_mb": base_lmhead_bytes / MB,
            "norms_gates_mb": (gate_bytes + rank_norm_bytes) / MB,
            "cuda_overhead_mb": cuda_overhead,
            "prefill_scratch_mb": rust_prefill_workspace_bytes / MB,
            "prefill_workspace_mb": prefill_workspace_bytes / MB,
            "recurrent_state_mb": recurrent_state_bytes / MB,
            "dspark_dense_mb": dspark_dense_bytes / MB,
            "dspark_runtime_mb": dspark_runtime_bytes / MB,
            "dspark_shared_expert_mb": dspark_shared_expert_bytes / MB,
            "dspark_resident_experts_mb": dspark_resident_expert_bytes / MB,
            "total_mb": total_bytes / MB,
            "free_mb": free_bytes / MB,
            "kv_tokens": kv_tokens,
            "kv_alloc_tokens": kv_alloc_tokens,
            "total_with_kv_mb": total_with_kv / MB,
            "free_after_kv_mb": free_after_kv / MB,
        })

    # Find worst-case rank
    worst = max(ranks, key=lambda r: r["total_mb"])
    over_budget = worst["total_mb"] > gpu_vram_mb

    total_moe_layers = len(moe_layer_indices)
    MB = 1024 * 1024

    # ── System RAM: mmap'd GPU Marlin expert cache plus HQQ host staging ──
    # Decode is 100% GPU (gpu_only=True). No CPU decode store, no CPU KV cache,
    # no separate CPU expert format. Non-expert weights (attention, norms, etc.)
    # are loaded directly into VRAM and don't persist in system RAM. HQQ runtime
    # keeps prefill and decode host formats staged so VMM slots can be refreshed
    # without re-reading safetensors.
    target_expert_cache_bytes = (
        expert_buf_bytes * n_experts * total_moe_layers if n_experts > 0 else 0
    )
    dspark_host_cache_bytes = int(dspark["host_cache_bytes"]) if dspark is not None else 0
    ram_gpu_experts_bytes = target_expert_cache_bytes + dspark_host_cache_bytes
    hqq_host_staging_bytes = 0
    if hqq_layer_bytes is not None:
        hqq_host_staging_bytes = sum(hqq_layer_bytes.values()) * 2
    ram_gpu_experts_mb = ram_gpu_experts_bytes / MB
    ram_hqq_host_staging_mb = hqq_host_staging_bytes / MB
    ram_total_mb = ram_gpu_experts_mb + ram_hqq_host_staging_mb

    peak_vram_mb = max(r["total_with_kv_mb"] for r in ranks) if ranks else 0

    arch = "DeepSeek-V4 compressed attention" if is_deepseek_v4 else "MLA" if is_mla else "GQA"
    if model_cfg.is_nemotron_h:
        arch += "+Mamba2"
    elif hybrid:
        arch += "+DeltaNet"

    return {
        "model_type": cfg.get("model_type", "unknown"),
        "architecture": arch,
        "num_layers": total_layers,
        "n_experts": n_experts,
        "n_shared_experts": n_shared,
        "first_k_dense": first_k_dense,
        "gpu_vram_mb": gpu_vram_mb,
        "total_ram_gb": total_ram_gb,
        "ranks": ranks,
        "worst_rank": worst["rank"],
        "over_budget": over_budget,
        "kv_dtype": kv_dtype,
        "max_context_tokens": effective_context_limit,
        "hybrid": hybrid,
        "num_full_attention_layers": model_cfg.num_full_attention_layers,
        "num_moe_layers": total_moe_layers,
        "ram_gpu_experts_mb": ram_gpu_experts_mb,
        "ram_hqq_host_staging_mb": ram_hqq_host_staging_mb,
        "ram_total_mb": ram_total_mb,
        "ram_dspark_experts_mb": dspark_host_cache_bytes / MB,
        "hcs_cacheable_experts_mb": (
            target_expert_cache_bytes
            + (int(dspark["hcs_cacheable_bytes"]) if dspark is not None else 0)
        ) / MB,
        "dspark_mode": dspark_mode,
        "dspark_budget_source": dspark["source"] if dspark is not None else None,
        "dspark_effective_group_size": (
            int(dspark["effective_group_size"]) if dspark is not None else None
        ),
        "peak_vram_mb": peak_vram_mb,
        "peak_system_ram_mb": ram_total_mb,
        "hqq_budget_source": hqq_budget_source,
    }


def compute_vram_budget(
    model_path: str,
    pp_partition: List[int],
    kv_cache_dtype: str = "auto",
    quantization: str = "none",
    gpu_vram_bytes: Optional[int] = None,
    headroom_mb: int = 500,
    num_gpu_experts: int = 0,
    requested_context: int = 0,
    layer_group_size: int = 2,
    expert_group_size: int = 128,
) -> Dict[str, Any]:
    """Compute VRAM budget and recommended SGLang parameters.

    The user passes requested_context as a hint. We allocate the lower of:
      (A) KV cache for the requested context, or
      (B) Maximum KV cache that fits minus headroom.

    Args:
        model_path: Path to HF model directory.
        pp_partition: Layer counts per PP rank (e.g. [20, 21, 20]).
        kv_cache_dtype: KV cache dtype ("fp8_e4m3", "bf16", etc.).
        quantization: Non-expert weight quantization ("w8a8_int8" or "none").
        gpu_vram_bytes: Per-GPU total VRAM in bytes (auto-detected if None).
        headroom_mb: Reserved VRAM for GPU prefill workspace + temporaries.
        num_gpu_experts: Number of pinned experts on GPU (default 0).
        requested_context: Context length hint from user (tokens); zero uses
            the checkpoint-declared model limit.
        layer_group_size: Layer group size for streaming (0=persistent, >=1=streaming).
        expert_group_size: INT4/INT8 expert quantization group size.
    """
    cfg, model_cfg = _normalized_launcher_config(model_path)

    if gpu_vram_bytes is None:
        gpu_vram_bytes = _detect_gpu_vram_bytes()

    num_ranks = len(pp_partition)
    total_layers = cfg["num_hidden_layers"]
    is_mla = _is_mla(cfg)
    hybrid = _is_hybrid(cfg)
    full_attn_interval = cfg.get("full_attention_interval", 0)

    if "first_k_dense_replace" in cfg:
        first_k_dense = cfg["first_k_dense_replace"]
    elif "decoder_sparse_step" in cfg:
        step = cfg["decoder_sparse_step"]
        first_k_dense = 0 if step <= 1 else step
    else:
        first_k_dense = 0

    max_position_embeddings = int(model_cfg.max_position_embeddings)
    if requested_context < 0:
        raise ValueError(
            f"requested_context must be non-negative, got {requested_context}"
        )
    if requested_context > max_position_embeddings:
        raise ValueError(
            f"requested_context {requested_context} exceeds model limit "
            f"{max_position_embeddings}"
        )
    effective_requested_context = requested_context or max_position_embeddings
    headroom_bytes = headroom_mb * 1024 * 1024
    overhead_bytes = DEFAULT_CUDA_OVERHEAD_MB * 1024 * 1024

    expert_bytes = _expert_bytes_per_expert(
        cfg, group_size=expert_group_size,
    ) if num_gpu_experts > 0 else 0
    linear_attn_bpl = _linear_attention_bytes_per_layer(cfg, quantization) if hybrid else 0

    total_model_layers = cfg["num_hidden_layers"]
    if is_mla:
        attn_per_layer = _mla_attention_bytes_per_layer(cfg, quantization)
    else:
        attn_per_layer = _gqa_attention_bytes_per_layer(cfg, quantization)

    norm_bytes_per_layer = _layernorm_bytes_per_layer(cfg)

    # Replicated on every GPU: embeddings and lm_head
    base_embed_total = _embedding_bytes(cfg)
    base_lm_head_total = _lm_head_bytes(cfg)

    ranks = []
    for rank_idx in range(num_ranks):
        rank_start = sum(pp_partition[:rank_idx])
        rank_end = rank_start + pp_partition[rank_idx]
        num_layers = pp_partition[rank_idx]

        # Count full attention vs linear attention layers IN THIS RANK
        if hybrid:
            rank_full_attn = sum(1 for i in range(rank_start, rank_end) if (i + 1) % full_attn_interval == 0)
            rank_linear_attn = num_layers - rank_full_attn
        else:
            rank_full_attn = num_layers
            rank_linear_attn = 0

        dense_layers_in_rank = max(0, min(rank_end, first_k_dense) - rank_start)
        moe_layers_in_rank = num_layers - dense_layers_in_rank

        # Attention and shared experts are permanently on GPU for ALL layers.
        # layer_group_size only controls expert DMA pipelining (HCS).
        rank_attn_total = attn_per_layer * rank_full_attn + linear_attn_bpl * rank_linear_attn
        # Norms and gates are permanent for ALL layers (not streamed)
        rank_norm_total = norm_bytes_per_layer * num_layers
        hidden = cfg["hidden_size"]
        n_experts_cli = cfg.get("n_routed_experts", cfg.get("num_experts", 0))
        gate_total = _gate_bytes_per_moe_layer(cfg) * moe_layers_in_rank
        # F32 routing copies of gate weights
        gate_f32_total = hidden * n_experts_cli * 4 * moe_layers_in_rank

        dense_mlp_total = _dense_mlp_bytes_per_layer(cfg, quantization) * dense_layers_in_rank
        shared_expert_total = _shared_expert_bytes_per_moe_layer(cfg, quantization) * moe_layers_in_rank
        pinned_expert_total = expert_bytes * num_gpu_experts if num_gpu_experts > 0 else 0

        weight_total = (
            rank_attn_total + rank_norm_total +
            base_embed_total + base_lm_head_total +
            dense_mlp_total + shared_expert_total +
            gate_total + gate_f32_total + pinned_expert_total
        )

        # KV cache only for full attention layers in THIS rank's partition
        kv_per_token = _kv_bytes_per_token_per_layer(cfg, kv_cache_dtype) * rank_full_attn

        # Free VRAM = total - weights - SGLang overhead - headroom
        free_bytes = gpu_vram_bytes - weight_total - overhead_bytes - headroom_bytes
        if free_bytes <= 0:
            max_tokens = 0
        else:
            max_tokens = free_bytes // kv_per_token if kv_per_token > 0 else 0

        ranks.append({
            "rank": rank_idx,
            "layers": f"{rank_start}-{rank_end - 1}",
            "num_layers": num_layers,
            "dense_layers": dense_layers_in_rank,
            "moe_layers": moe_layers_in_rank,
            "attention_mb": rank_attn_total / (1024**2),
            "dense_mlp_mb": dense_mlp_total / (1024**2),
            "shared_expert_mb": shared_expert_total / (1024**2),
            "gate_mb": gate_total / (1024**2),
            "norm_mb": rank_norm_total / (1024**2),
            "embedding_mb": base_embed_total / (1024**2),
            "lm_head_mb": base_lm_head_total / (1024**2),
            "pinned_experts_mb": pinned_expert_total / (1024**2),
            "weight_total_mb": weight_total / (1024**2),
            "weight_total_bytes": weight_total,
            "kv_bytes_per_token": kv_per_token,
            "free_for_kv_mb": max(0, free_bytes) / (1024**2),
            "max_tokens": max_tokens,
        })

    # Bottleneck rank determines max capacity
    bottleneck_rank = min(ranks, key=lambda r: r["max_tokens"])
    bottleneck_max = bottleneck_rank["max_tokens"]

    # Final context = min(user request, bottleneck capacity, model max)
    context_length = min(
        effective_requested_context, bottleneck_max, max_position_embeddings
    )

    # Was the user request satisfied?
    if effective_requested_context <= bottleneck_max:
        context_note = "user request fits"
    else:
        context_note = (
            f"limited by rank {bottleneck_rank['rank']}, requested "
            f"{effective_requested_context:,d}"
        )

    # Compute mem_fraction_static from actual weights + chosen KV allocation
    # Find the rank that needs the most VRAM (weights + its KV share)
    max_used_bytes = 0
    for r in ranks:
        kv_for_context = context_length * r["kv_bytes_per_token"]
        used = r["weight_total_bytes"] + overhead_bytes + kv_for_context
        if used > max_used_bytes:
            max_used_bytes = used
    mem_fraction = round(max_used_bytes / gpu_vram_bytes, 4)
    # Clamp to [0.1, 0.95] — never reserve more than 95%
    mem_fraction = max(0.1, min(0.95, mem_fraction))

    arch = "MLA" if is_mla else "GQA"
    if hybrid:
        arch += "+DeltaNet"

    return {
        "model_path": model_path,
        "model_type": cfg.get("model_type", "unknown"),
        "architecture": arch,
        "num_hidden_layers": total_layers,
        "first_k_dense": first_k_dense,
        "pp_partition": pp_partition,
        "gpu_vram_mb": gpu_vram_bytes / (1024**2),
        "headroom_mb": headroom_mb,
        "overhead_mb": DEFAULT_CUDA_OVERHEAD_MB,
        "quantization": quantization,
        "kv_cache_dtype": kv_cache_dtype,
        "max_position_embeddings": max_position_embeddings,
        "num_gpu_experts": num_gpu_experts,
        "requested_context": effective_requested_context,
        "ranks": ranks,
        "bottleneck_rank": bottleneck_rank["rank"],
        "bottleneck_max_tokens": bottleneck_max,
        "context_length": context_length,
        "context_note": context_note,
        "mem_fraction": mem_fraction,
    }


def print_budget_summary(budget: Dict[str, Any], file=sys.stderr) -> None:
    """Print a human-readable budget summary to stderr."""
    print(f"\n=== VRAM Budget: {budget['model_type']} ({budget['architecture']}) ===", file=file)
    print(f"PP partition: {budget['pp_partition']} ({budget['num_hidden_layers']} layers)", file=file)
    print(f"Quantization: {budget['quantization']}, KV dtype: {budget['kv_cache_dtype']}", file=file)
    print(f"GPU VRAM: {budget['gpu_vram_mb']:.0f} MB", file=file)
    print(f"Headroom: {budget['headroom_mb']} MB (GPU prefill + temporaries)", file=file)
    print(f"SGLang overhead: {budget['overhead_mb']} MB (PyTorch/CUDA/NCCL)", file=file)
    print(f"Requested context: {budget['requested_context']:,d} tokens", file=file)
    if budget['num_gpu_experts'] > 0:
        print(f"GPU-pinned experts: {budget['num_gpu_experts']}", file=file)
    print(file=file)

    headers = [f"PP{r['rank']}" for r in budget["ranks"]]
    print(f"{'':>20s}  " + "  ".join(f"{h:>10s}" for h in headers), file=file)
    print(f"{'':>20s}  " + "  ".join(f"{'─'*10:>10s}" for _ in headers), file=file)

    fields = [
        ("Layers", "layers", None),
        ("Attention", "attention_mb", "MB"),
        ("Dense MLP", "dense_mlp_mb", "MB"),
        ("Shared expert", "shared_expert_mb", "MB"),
        ("Gate+norms", None, "MB"),
        ("Embedding", "embedding_mb", "MB"),
        ("LM head", "lm_head_mb", "MB"),
        ("Pinned experts", "pinned_experts_mb", "MB"),
        ("Weight total", "weight_total_mb", "MB"),
        ("Free for KV", "free_for_kv_mb", "MB"),
        ("KV/token", "kv_bytes_per_token", "B"),
        ("Max tokens", "max_tokens", None),
    ]

    for label, key, unit in fields:
        vals = []
        for r in budget["ranks"]:
            if key is None and label == "Gate+norms":
                v = r["gate_mb"] + r["norm_mb"]
            elif key is not None:
                v = r[key]
            else:
                v = r.get(key, 0)

            if isinstance(v, str):
                vals.append(f"{v:>10s}")
            elif unit == "MB":
                vals.append(f"{v:>8.0f} MB")
            elif unit == "B":
                vals.append(f"{v:>8,d} B")
            else:
                if isinstance(v, int) and v > 1000:
                    vals.append(f"{v:>10,d}")
                else:
                    vals.append(f"{v!s:>10s}")

        print(f"{label:>20s}  {'  '.join(vals)}", file=file)

    print(file=file)
    bn = budget["bottleneck_rank"]
    print(f"Bottleneck: rank {bn} (max {budget['bottleneck_max_tokens']:,d} tokens)", file=file)
    print(f"Estimated context: {budget['context_length']:,d} tokens ({budget['context_note']})", file=file)
    print(f"mem_fraction_static: {budget['mem_fraction']}", file=file)
    print(file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Compute VRAM budget for Krasis + SGLang",
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to HF model directory")
    parser.add_argument("--pp-partition", required=True,
                        help="Comma-separated layer counts per rank (e.g. 20,21,20)")
    parser.add_argument("--kv-cache-dtype", default="k6v6",
                        help="KV cache format (default: k6v6 Quality; use k4v4 Ultra Compact or bf16 Full Precision)")
    parser.add_argument("--quantization", default="none",
                        help="Non-expert quantization (w8a8_int8 or none)")
    parser.add_argument("--gpu-vram-mb", type=int, default=None,
                        help="Per-GPU VRAM in MB (auto-detected if omitted)")
    parser.add_argument("--headroom-mb", type=int, default=500,
                        help="Reserved VRAM for GPU prefill + temporaries (default: 500)")
    parser.add_argument("--gpu-experts", type=int, default=0,
                        help="Number of pinned experts on GPU")
    parser.add_argument("--context-length", type=int, default=0,
                        help="Requested context length hint in tokens (default: 0, checkpoint-declared model limit)")
    parser.add_argument("--layer-group-size", type=int, default=2,
                        help="Layer group size for streaming (0=persistent, default: 2)")
    parser.add_argument("--expert-group-size", type=int, default=128, choices=[32, 64, 128],
                        help="INT4/INT8 expert quantization group size (default: 128)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress summary to stderr")
    args = parser.parse_args()

    partition = [int(x.strip()) for x in args.pp_partition.split(",")]
    gpu_vram_bytes = args.gpu_vram_mb * 1024 * 1024 if args.gpu_vram_mb else None

    budget = compute_vram_budget(
        model_path=args.model_path,
        pp_partition=partition,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization=args.quantization,
        gpu_vram_bytes=gpu_vram_bytes,
        headroom_mb=args.headroom_mb,
        num_gpu_experts=args.gpu_experts,
        requested_context=args.context_length,
        layer_group_size=args.layer_group_size,
        expert_group_size=args.expert_group_size,
    )

    if not args.quiet:
        print_budget_summary(budget)

    # Output JSON to stdout for script consumption
    json.dump(budget, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
