"""Streaming BF16→INT8 weight loader for Krasis standalone server.

Loads non-expert weights from HF safetensors one tensor at a time,
quantizes to INT8 per-channel symmetric, and places on target GPU.
Peak transient RAM: ~50 MB (one tensor buffer at a time).

Expert weights are loaded by the Krasis Rust engine (INT4, CPU RAM).
"""

import json
import logging
import os
import struct
import time
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

from krasis.config import ModelConfig, PPRankConfig, QuantConfig

logger = logging.getLogger(__name__)


def _emit_real_model_timing(payload: dict) -> None:
    if os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING") == "1":
        print(json.dumps({"hqq_real_model_timing": payload}, sort_keys=True), flush=True)


def _timed_bf16_load(
    loader: "WeightLoader",
    *,
    layer_idx: int,
    tensor_name: str,
    device: torch.device,
    step: str,
) -> torch.Tensor:
    started = time.perf_counter()
    tensor = loader._load_bf16(tensor_name, device)
    _emit_real_model_timing(
        {
            "phase": "load_tensor",
            "layer_idx": int(layer_idx),
            "step": step,
            "tensor_name": tensor_name,
            "device": str(device),
            "elapsed_s": time.perf_counter() - started,
        }
    )
    return tensor


def quantize_to_int8(
    weight_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 weight to INT8 per-channel symmetric.

    Args:
        weight_bf16: [out_features, in_features] BF16 tensor

    Returns:
        (weight_int8, scale) where:
        - weight_int8: [out_features, in_features] torch.int8
        - scale: [out_features] torch.bfloat16 (per-channel)
    """
    w = weight_bf16.float()
    # Per-channel (per-row) max absolute value
    amax = w.abs().amax(dim=1).clamp(min=1e-10)
    scale = amax / 127.0
    w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.to(torch.bfloat16)


def int8_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W8A8 INT8 matmul with on-the-fly activation quantization.

    Both weights and activations are INT8, matmul accumulates in INT32 via
    torch._int_mm for maximum precision. Works well for MLP/shared_expert/lm_head
    but NOT suitable for attention projections (use attention="bf16" instead).

    Args:
        x: [..., in_features] BF16 input
        weight_int8: [out_features, in_features] INT8
        scale: [out_features] BF16 per-channel scale

    Returns:
        [..., out_features] BF16
    """
    orig_shape = x.shape[:-1]
    in_features = x.shape[-1]
    x_2d = x.reshape(-1, in_features)
    M = x_2d.shape[0]

    # Quantize activation to INT8 per-token
    x_float = x_2d.float()
    x_amax = x_float.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    x_scale = x_amax / 127.0
    x_int8 = (x_float / x_scale).round().clamp(-128, 127).to(torch.int8)

    # _int_mm requires M >= 17 and M to be a multiple of 8 on SM89
    padded_M = max(M, 17)
    padded_M = (padded_M + 7) & ~7  # round up to multiple of 8
    if padded_M != M:
        pad = padded_M - M
        x_int8 = torch.nn.functional.pad(x_int8, (0, 0, 0, pad))
        x_scale = torch.nn.functional.pad(x_scale, (0, 0, 0, pad))

    # INT8 matmul: [M, K] @ [K, N] -> [M, N] INT32
    out_int32 = torch._int_mm(x_int8, weight_int8.t())

    if padded_M != M:
        out_int32 = out_int32[:M]
        x_scale = x_scale[:M]

    # Dequantize: multiply by x_scale * w_scale
    out = out_int32.float() * (x_scale * scale.float().unsqueeze(0))
    out = out.to(torch.bfloat16).reshape(*orig_shape, -1)

    if bias is not None:
        out = out + bias

    return out


class WeightLoader:
    """Streaming weight loader — loads one tensor at a time from safetensors.

    Uses safe_open with framework="pt" for random-access tensor reading.
    Each tensor is read, quantized to INT8, placed on GPU, then the CPU
    buffer is freed. Peak transient RAM: ~50 MB.
    """

    def __init__(self, cfg: ModelConfig, quant_cfg: QuantConfig = None):
        self.cfg = cfg
        self.quant_cfg = quant_cfg or QuantConfig()
        self.model_path = cfg.model_path

        # Load safetensors index
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                index = json.load(f)
            self._weight_map: Dict[str, str] = index["weight_map"]
        else:
            single_path = os.path.join(self.model_path, "model.safetensors")
            if not os.path.isfile(single_path):
                raise FileNotFoundError(
                    f"Expected either {index_path} or {single_path}"
                )
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                self._weight_map = {name: "model.safetensors" for name in handle.keys()}
        if self.cfg.is_deepseek_v4:
            self._validate_deepseek_v4_checkpoint_inventory()
        self._handles: Dict[str, object] = {}

    def _validate_deepseek_v4_checkpoint_inventory(self) -> None:
        """Fail closed on unknown V4 namespaces and inventory skipped MTP blocks.

        DeepSeek-V4 main-model tensors live at the checkpoint root while its
        speculative D-Spark/MTP blocks live under ``mtp.*``. The auxiliary
        namespace must remain explicit and disjoint from the configured
        main-layer range whether D-Spark is disabled or loaded later.
        """
        root_main = {
            "embed.weight",
            "norm.weight",
            "head.weight",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
        }
        main_layer_count = 0
        main_layer_ids = set()
        auxiliary_count = 0
        auxiliary_block_ids = set()
        unexpected = []

        for name in self._weight_map:
            parts = name.split(".")
            if name.startswith("layers.") and len(parts) >= 3:
                try:
                    layer_idx = int(parts[1])
                except ValueError:
                    unexpected.append(name)
                    continue
                if not 0 <= layer_idx < self.cfg.num_hidden_layers:
                    unexpected.append(name)
                    continue
                main_layer_count += 1
                main_layer_ids.add(layer_idx)
            elif name.startswith("mtp.") and len(parts) >= 3:
                try:
                    block_idx = int(parts[1])
                except ValueError:
                    unexpected.append(name)
                    continue
                if block_idx < 0:
                    unexpected.append(name)
                    continue
                auxiliary_count += 1
                auxiliary_block_ids.add(block_idx)
            elif name not in root_main:
                unexpected.append(name)

        if unexpected:
            preview = ", ".join(sorted(unexpected)[:8])
            raise ValueError(
                "DeepSeek-V4 checkpoint contains unexpected tensor namespace(s): "
                f"{preview} (unexpected_count={len(unexpected)})"
            )

        expected_layers = set(range(self.cfg.num_hidden_layers))
        if main_layer_ids != expected_layers:
            missing = sorted(expected_layers - main_layer_ids)
            extra = sorted(main_layer_ids - expected_layers)
            raise ValueError(
                "DeepSeek-V4 main-layer inventory mismatch: "
                f"missing={missing}, extra={extra}"
            )

        if auxiliary_block_ids:
            expected_blocks = set(range(max(auxiliary_block_ids) + 1))
            if auxiliary_block_ids != expected_blocks:
                raise ValueError(
                    "DeepSeek-V4 MTP block inventory must be contiguous from zero: "
                    f"found={sorted(auxiliary_block_ids)}"
                )
            targets = self.cfg.dspark_target_layer_ids
            if targets is not None and len(auxiliary_block_ids) != len(targets):
                raise ValueError(
                    "DeepSeek-V4 D-Spark/MTP block count does not match target-layer "
                    f"count: blocks={len(auxiliary_block_ids)}, targets={len(targets)}"
                )

        inventory_message = (
            "DeepSeek-V4 checkpoint namespace inventory: "
            f"{main_layer_count} layers.* main tensors across "
            f"{len(main_layer_ids)} layers; {auxiliary_count} mtp.* auxiliary "
            f"tensors across blocks {sorted(auxiliary_block_ids)} inventoried; "
            f"{sum(name in root_main for name in self._weight_map)} "
            "root main tensors; unexpected=0"
        )
        logger.info(inventory_message)
        # The normal launcher does not enable Python INFO logging. Keep this
        # fail-closed checkpoint audit visible in every built-command log.
        print(inventory_message, flush=True)

    @staticmethod
    def _memory_allocated_mb(device: torch.device) -> float:
        if device.type != "cuda":
            return 0.0
        return torch.cuda.memory_allocated(device) / (1024**2)

    def _get_handle(self, shard_name: str):
        """Get or open a safetensors file handle (cached)."""
        if shard_name not in self._handles:
            path = os.path.join(self.model_path, shard_name)
            self._handles[shard_name] = safe_open(path, framework="pt", device="cpu")
        return self._handles[shard_name]

    def _read_tensor(self, name: str) -> torch.Tensor:
        """Read a single tensor from safetensors by name."""
        shard_name = self._weight_map[name]
        handle = self._get_handle(shard_name)
        return handle.get_tensor(name)

    def _read_and_dequant(self, name: str) -> torch.Tensor:
        """Read a tensor, dequanting FP8 to BF16 if needed.

        FP8 models store weights with a companion scale tensor. Scalar scales
        are applied directly; block scales are expanded using the checkpoint's
        declared block geometry. Unknown or missing scale contracts fail closed.
        Non-FP8 tensors are returned as-is converted to BF16.
        """
        w = self._read_tensor(name)
        if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                        torch.float8_e5m2, torch.float8_e5m2fnuz):
            if self.cfg.is_deepseek_v4:
                if not name.endswith(".weight"):
                    raise ValueError(
                        f"DeepSeek-V4 FP8 tensor is not a weight: {name}"
                    )
                scale_name = name.removesuffix(".weight") + ".scale"
                if scale_name not in self._weight_map:
                    raise KeyError(
                        f"DeepSeek-V4 FP8 tensor {name} has no block scale {scale_name}"
                    )
            else:
                scale_name = f"{name}_scale_inv"
                if scale_name not in self._weight_map:
                    raise KeyError(
                        f"FP8 tensor {name} has no required scale {scale_name}"
                    )

            scale = self._read_tensor(scale_name).float()
            if scale.numel() == 1:
                return (w.float() * scale.reshape(())).to(torch.bfloat16)
            if w.ndim != 2 or scale.ndim != 2:
                raise ValueError(
                    f"FP8 block dequant requires rank-2 weight/scale; got "
                    f"{name} {tuple(w.shape)} and {scale_name} {tuple(scale.shape)}"
                )
            block_size = self.cfg.source_fp8_block_size
            if block_size is None:
                raise ValueError(
                    f"FP8 block scale {scale_name} is present but the checkpoint "
                    "does not declare quantization_config.weight_block_size"
                )
            block_rows, block_cols = block_size
            expected_shape = (
                (w.shape[0] + block_rows - 1) // block_rows,
                (w.shape[1] + block_cols - 1) // block_cols,
            )
            if tuple(scale.shape) != expected_shape:
                raise ValueError(
                    f"FP8 scale {scale_name} shape {tuple(scale.shape)} != expected "
                    f"{expected_shape} for {tuple(w.shape)} and blocks {block_size}"
                )
            expanded = scale.repeat_interleave(block_rows, 0)
            expanded = expanded.repeat_interleave(block_cols, 1)
            expanded = expanded[: w.shape[0], : w.shape[1]]
            return (w.float() * expanded).to(torch.bfloat16)
        return w.to(torch.bfloat16)

    def _load_and_quantize(
        self, name: str, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a weight tensor, quantize to INT8, place on GPU.

        Returns (weight_int8, scale) both on device.
        """
        w = self._read_and_dequant(name)
        w_int8, scale = quantize_to_int8(w)
        del w
        return w_int8.to(device), scale.to(device)

    def _load_bf16(self, name: str, device: torch.device) -> torch.Tensor:
        """Read a weight tensor and place on GPU as BF16."""
        w = self._read_and_dequant(name)
        return w.to(device)

    def _load_f32(self, name: str, device: torch.device) -> torch.Tensor:
        """Load a checkpoint tensor that is semantically FP32."""
        w = self._read_tensor(name)
        if w.dtype != torch.float32:
            raise ValueError(f"Expected FP32 tensor {name}, got {w.dtype}")
        return w.to(device)

    def _load_numeric_f32(self, name: str, device: torch.device) -> torch.Tensor:
        """Load a numeric checkpoint tensor and materialize FP32 on the target.

        This is used for operations whose published reference explicitly casts
        learned BF16 parameters to FP32 before applying them, such as GLM-5.3
        mHC projections. It is distinct from `_load_f32`, which verifies that
        the source tensor itself is stored as FP32.
        """
        return self._read_and_dequant(name).float().to(device)

    def _load_i64(self, name: str, device: torch.device) -> torch.Tensor:
        """Load an integer lookup table without changing its values."""
        w = self._read_tensor(name)
        if w.dtype != torch.int64:
            raise ValueError(f"Expected I64 tensor {name}, got {w.dtype}")
        return w.to(device)

    def load_embedding(self, device: torch.device) -> torch.Tensor:
        """Load embedding table (BF16, ~2.2 GB for Kimi K2.5)."""
        if self.cfg.is_deepseek_v4:
            return self._load_bf16("embed.weight", device)
        # Nemotron uses "embeddings" instead of "embed_tokens"
        name = f"{self.cfg.layers_prefix}.embed_tokens.weight"
        if name not in self._weight_map:
            name = f"{self.cfg.layers_prefix}.embeddings.weight"
        logger.info("Loading embedding: %s", name)
        return self._load_bf16(name, device)

    def load_final_norm(self, device: torch.device) -> torch.Tensor:
        """Load final RMSNorm weight (BF16, tiny)."""
        if self.cfg.is_deepseek_v4:
            return self._load_bf16("norm.weight", device)
        # Nemotron uses "norm_f" instead of "norm"
        name = f"{self.cfg.layers_prefix}.norm.weight"
        if name not in self._weight_map:
            name = f"{self.cfg.layers_prefix}.norm_f.weight"
        logger.info("Loading final norm: %s", name)
        w = self._load_bf16(name, device)
        return self._maybe_apply_delta_rms_bias(w, name)

    def _norm_uses_delta_rms_bias(self, label: str) -> bool:
        """Return whether a norm tensor uses the Qwen delta-RMS convention.

        Qwen3-Next and Qwen3.5 mix two norm contracts:
        - outer RMS norms store deltas and apply (1 + w) * x
        - linear-attention inner norm stores direct scales and applies w * x

        Use the tensor role rather than value heuristics so the rule stays
        stable across models within the family.
        """
        delta_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        if label.endswith(delta_suffixes):
            return True
        final_norm_suffixes = (
            ".norm.weight",
            ".norm_f.weight",
        )
        if label.endswith(final_norm_suffixes) and ".layers." not in label:
            return True
        return False

    def _maybe_apply_delta_rms_bias(self, weight: torch.Tensor, label: str) -> torch.Tensor:
        """Apply +1 only to norm roles that use the delta-RMS contract."""
        if not self.cfg.norm_bias_one:
            return weight

        if self._norm_uses_delta_rms_bias(label):
            logger.debug(
                "Applying +1 RMSNorm bias for %s (delta-RMS role)",
                label,
            )
            return weight + 1.0

        logger.debug(
            "Leaving RMSNorm weight unchanged for %s (direct-scale role)",
            label,
        )
        return weight

    def load_lm_head(self, device: torch.device):
        """Load LM head weight.

        Returns (weight_int8, scale) if INT8, or plain BF16 tensor if BF16.
        """
        # lm_head location depends on model — try multiple naming conventions
        name = "head.weight" if self.cfg.is_deepseek_v4 else "lm_head.weight"
        if name not in self._weight_map:
            if self.cfg.layers_prefix != "model":
                prefix = self.cfg.layers_prefix.rsplit(".", 1)[0]
                name = f"{prefix}.lm_head.weight"
        if name not in self._weight_map and self.cfg.gemma4_text:
            name = "model.language_model.lm_head.weight"
        if name not in self._weight_map:
            tied_name = f"{self.cfg.layers_prefix}.embed_tokens.weight"
            if tied_name in self._weight_map:
                logger.info("LM head is tied; using embedding weight: %s", tied_name)
                name = tied_name
            else:
                raise KeyError(f"Could not find LM head weight; tried lm_head.weight and {name}")
        logger.info("Loading LM head: %s (precision=%s)", name, self.quant_cfg.lm_head)
        if self.quant_cfg.lm_head == "bf16":
            return self._load_bf16(name, device)
        return self._load_and_quantize(name, device)

    def load_attention_weights(
        self, layer_idx: int, device: torch.device,
        proj_device: torch.device = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load attention weights for one layer (MLA or GQA).

        Args:
            device: Device for all weights by default (norms, biases, etc).
            proj_device: Device for large projection weights (q/k/v/o_proj).
                         Defaults to device. Pass CPU for AWQ mode where
                         projections stay on CPU for quantization, while
                         small weights (norms, biases) go directly to GPU.
        """
        if proj_device is None:
            proj_device = device
        if self.cfg.is_deepseek_v4:
            return self._load_deepseek_v4_attention(
                layer_idx, device, proj_device
            )
        if self.cfg.is_gqa:
            return self._load_gqa_attention(layer_idx, device, proj_device)
        return self._load_mla_attention(layer_idx, device)

    def _load_deepseek_v4_compressor(
        self,
        prefix: str,
        layer_idx: int,
        device: torch.device,
        proj_device: torch.device,
        step_prefix: str,
    ) -> Dict[str, torch.Tensor]:
        result = {
            "ape": self._load_f32(f"{prefix}.ape", device),
            "wkv": _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=f"{prefix}.wkv.weight",
                device=proj_device,
                step=f"{step_prefix}.wkv",
            ),
            "wgate": _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=f"{prefix}.wgate.weight",
                device=proj_device,
                step=f"{step_prefix}.wgate",
            ),
            "norm": self._load_bf16(f"{prefix}.norm.weight", device),
        }
        return result

    def _load_deepseek_v4_attention(
        self,
        layer_idx: int,
        device: torch.device,
        proj_device: torch.device,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        prefix = f"{self.cfg.layer_tensor_prefix(layer_idx)}.attn"
        result: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {
            "attn_sink": self._load_f32(f"{prefix}.attn_sink", device),
            "q_norm": self._load_bf16(f"{prefix}.q_norm.weight", device),
            "kv_norm": self._load_bf16(f"{prefix}.kv_norm.weight", device),
        }
        for projection in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            result[projection] = _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=f"{prefix}.{projection}.weight",
                device=proj_device,
                step=f"deepseek_v4.{projection}",
            )

        ratio = self.cfg.compress_ratios[layer_idx]
        if ratio > 0:
            result["compressor"] = self._load_deepseek_v4_compressor(
                f"{prefix}.compressor",
                layer_idx,
                device,
                proj_device,
                "deepseek_v4.compressor",
            )
        if ratio == 4:
            indexer_prefix = f"{prefix}.indexer"
            result["indexer"] = {
                "wq_b": _timed_bf16_load(
                    self,
                    layer_idx=layer_idx,
                    tensor_name=f"{indexer_prefix}.wq_b.weight",
                    device=proj_device,
                    step="deepseek_v4.indexer.wq_b",
                ),
                "weights_proj": self._load_bf16(
                    f"{indexer_prefix}.weights_proj.weight", device
                ),
                "compressor": self._load_deepseek_v4_compressor(
                    f"{indexer_prefix}.compressor",
                    layer_idx,
                    device,
                    proj_device,
                    "deepseek_v4.indexer.compressor",
                ),
            }
        return result

    def load_deepseek_v4_hyper_connection(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        prefix = self.cfg.layer_tensor_prefix(layer_idx)
        return {
            name: self._load_f32(f"{prefix}.{name}", device)
            for name in (
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
            )
        }

    def load_deepseek_v4_hc_head(
        self, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        if not self.cfg.is_deepseek_v4:
            raise ValueError("DeepSeek-V4 HC head requested for another architecture")
        return {
            name: self._load_f32(name, device)
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")
        }

    def load_dspark_stage(
        self, stage_idx: int, device: torch.device
    ) -> Dict[str, object]:
        """Load one checkpoint-owned D-Spark stage exactly into GPU tensors.

        Routed/shared experts are loaded by the Rust source-bound Marlin
        cache. This method owns only the small dense graph, router, norms,
        hyper-connections, and final speculative heads used by Rust/CUDA.
        """
        targets = tuple(self.cfg.dspark_target_layer_ids or ())
        if not self.cfg.is_deepseek_v4 or not targets:
            raise ValueError("D-Spark stage loading requires a D-Spark DeepSeek-V4 config")
        if not 0 <= stage_idx < len(targets):
            raise ValueError(
                f"D-Spark stage {stage_idx} outside [0, {len(targets)})"
            )
        prefix = f"mtp.{stage_idx}"
        required = [
            f"{prefix}.attn_norm.weight",
            f"{prefix}.ffn_norm.weight",
            f"{prefix}.ffn.gate.weight",
            f"{prefix}.ffn.gate.bias",
        ]
        for suffix in (
            "attn.attn_sink",
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ):
            required.append(f"{prefix}.{suffix}")
        if stage_idx == 0:
            required.extend(
                (f"{prefix}.main_proj.weight", f"{prefix}.main_norm.weight")
            )
        if stage_idx == len(targets) - 1:
            required.extend(
                (
                    f"{prefix}.norm.weight",
                    f"{prefix}.hc_head_fn",
                    f"{prefix}.hc_head_base",
                    f"{prefix}.hc_head_scale",
                    f"{prefix}.markov_head.markov_w1.weight",
                    f"{prefix}.markov_head.markov_w2.weight",
                    f"{prefix}.confidence_head.proj.weight",
                )
            )
        missing = [name for name in required if name not in self._weight_map]
        if missing:
            raise KeyError(
                f"D-Spark stage {stage_idx} is missing checkpoint tensors: "
                + ", ".join(missing)
            )

        attention = {
            "attn_sink": self._load_f32(f"{prefix}.attn.attn_sink", device),
            "q_norm": self._load_bf16(f"{prefix}.attn.q_norm.weight", device),
            "kv_norm": self._load_bf16(f"{prefix}.attn.kv_norm.weight", device),
        }
        for projection in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            attention[projection] = _timed_bf16_load(
                self,
                layer_idx=self.cfg.num_hidden_layers + stage_idx,
                tensor_name=f"{prefix}.attn.{projection}.weight",
                device=device,
                step=f"dspark.{projection}",
            )
        hyper_connection = {
            name: self._load_f32(f"{prefix}.{name}", device)
            for name in (
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
            )
        }
        result: Dict[str, object] = {
            "input_norm": self._load_bf16(f"{prefix}.attn_norm.weight", device),
            "post_attn_norm": self._load_bf16(f"{prefix}.ffn_norm.weight", device),
            "gate": self._load_bf16(f"{prefix}.ffn.gate.weight", device),
            "gate_bias": self._load_f32(f"{prefix}.ffn.gate.bias", device),
            "attention": attention,
            "hyper_connection": hyper_connection,
        }
        if stage_idx == 0:
            result["main_proj"] = self._load_bf16(
                f"{prefix}.main_proj.weight", device
            )
            result["main_norm"] = self._load_bf16(
                f"{prefix}.main_norm.weight", device
            )
        if stage_idx == len(targets) - 1:
            result["final_norm"] = self._load_bf16(f"{prefix}.norm.weight", device)
            result["head_hyper_connection"] = {
                name: self._load_f32(f"{prefix}.{name}", device)
                for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")
            }
            result["markov_w1"] = self._load_bf16(
                f"{prefix}.markov_head.markov_w1.weight", device
            )
            result["markov_w2"] = self._load_bf16(
                f"{prefix}.markov_head.markov_w2.weight", device
            )
            result["confidence"] = self._load_bf16(
                f"{prefix}.confidence_head.proj.weight", device
            )
        return result

    def _load_mla_attention(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load MLA attention weights for one layer.

        Handles both:
        - q_lora path (Kimi K2.5): q_a_proj + q_a_layernorm + q_b_proj
        - direct q path (V2-Lite): q_proj only

        kv_b_proj kept BF16 (split into w_kc, w_vc for quality).
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.self_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        if self.cfg.has_q_lora:
            for proj in ["q_a_proj", "q_b_proj"]:
                name = f"{prefix}.{proj}.weight"
                weights[proj] = _timed_bf16_load(
                    self,
                    layer_idx=layer_idx,
                    tensor_name=name,
                    device=device,
                    step=f"mla.{proj}",
                )
            weights["q_a_layernorm"] = _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=f"{prefix}.q_a_layernorm.weight",
                device=device,
                step="mla.q_a_layernorm",
            )
        else:
            weights["q_proj"] = _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=f"{prefix}.q_proj.weight",
                device=device,
                step="mla.q_proj",
            )

        weights["kv_a_proj_with_mqa"] = _timed_bf16_load(
            self,
            layer_idx=layer_idx,
            tensor_name=f"{prefix}.kv_a_proj_with_mqa.weight",
            device=device,
            step="mla.kv_a_proj_with_mqa",
        )
        weights["o_proj"] = _timed_bf16_load(
            self,
            layer_idx=layer_idx,
            tensor_name=f"{prefix}.o_proj.weight",
            device=device,
            step="mla.o_proj",
        )
        weights["kv_a_layernorm"] = _timed_bf16_load(
            self,
            layer_idx=layer_idx,
            tensor_name=f"{prefix}.kv_a_layernorm.weight",
            device=device,
            step="mla.kv_a_layernorm",
        )

        kv_b = _timed_bf16_load(
            self,
            layer_idx=layer_idx,
            tensor_name=f"{prefix}.kv_b_proj.weight",
            device=device,
            step="mla.kv_b_proj",
        )
        n_heads = self.cfg.num_attention_heads
        qk_nope = self.cfg.qk_nope_head_dim
        v_head = self.cfg.v_head_dim

        kv_b = kv_b.reshape(n_heads, qk_nope + v_head, self.cfg.kv_lora_rank)
        weights["w_kc"] = kv_b[:, :qk_nope, :].contiguous()
        weights["w_vc"] = kv_b[:, qk_nope:, :].contiguous()
        del kv_b

        if self.cfg.is_dsa and self.cfg.is_dsa_indexer_owner_layer(layer_idx):
            weights["dsa_indexer"] = self._load_dsa_indexer_weights(
                layer_idx,
                device,
            )

        return weights

    def _load_dsa_indexer_weights(
        self,
        layer_idx: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load the complete checkpoint tensor set owned by one DSA indexer."""
        if not self.cfg.is_dsa_indexer_owner_layer(layer_idx):
            owner_idx = self.cfg.dsa_indexer_owner_layer(layer_idx)
            raise ValueError(
                f"DSA layer {layer_idx} shares indexer owner {owner_idx}; "
                "shared layers must not load duplicate indexer tensors"
            )

        prefix = (
            f"{self.cfg.layers_prefix}.layers.{layer_idx}.self_attn.indexer"
        )
        tensor_names = {
            "wq_b": f"{prefix}.wq_b.weight",
            "wk": f"{prefix}.wk.weight",
            "weights_proj": f"{prefix}.weights_proj.weight",
            "k_norm_weight": f"{prefix}.k_norm.weight",
            "k_norm_bias": f"{prefix}.k_norm.bias",
        }
        if self.cfg.index_kpool_compress:
            tensor_names.update(
                {
                    "index_kpool_compress_ape": (
                        f"{prefix}.index_kpool_compress_ape"
                    ),
                    "index_kpool_compress_gate": (
                        f"{prefix}.index_kpool_compress_gate"
                    ),
                }
            )
        missing = [
            tensor_name
            for tensor_name in tensor_names.values()
            if tensor_name not in self._weight_map
        ]
        if missing:
            raise KeyError(
                f"DSA indexer owner layer {layer_idx} is missing checkpoint "
                f"tensors: {', '.join(missing)}"
            )

        return {
            key: _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=tensor_name,
                device=device,
                step=f"dsa_indexer.{key}",
            )
            for key, tensor_name in tensor_names.items()
        }

    def _load_gqa_attention(
        self, layer_idx: int, device: torch.device,
        proj_device: torch.device = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load GQA attention weights for one layer (Qwen3, GLM-4.7).

        Loads: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm.
        Also loads bias tensors when present (GLM-4.7 has attention_bias=true).

        Args:
            device: Device for small weights (norms, biases) — always GPU.
            proj_device: Device for large projection weights. Defaults to device.
                         CPU when AWQ will quantize them before GPU upload.
        """
        if proj_device is None:
            proj_device = device
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.self_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        projections = ["q_proj", "k_proj", "o_proj"]
        if self.cfg.gqa_has_v_proj_for_layer(layer_idx):
            projections.insert(2, "v_proj")
        if self.cfg.head_wise_attention_gate:
            projections.append("g_proj")
        for proj in projections:
            weights[proj] = _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=f"{prefix}.{proj}.weight",
                device=proj_device,
                step=f"gqa.{proj}",
            )
            # Load bias if present (GLM-4.7) — biases are small, always on GPU
            bias_name = f"{prefix}.{proj}.bias"
            if bias_name in self._weight_map:
                weights[f"{proj}_bias"] = _timed_bf16_load(
                    self,
                    layer_idx=layer_idx,
                    tensor_name=bias_name,
                    device=device,
                    step=f"gqa.{proj}_bias",
                )
        if "v_proj" not in weights:
            weights["v_proj"] = weights["k_proj"]
            weights["v_proj_aliases_k_proj"] = True

        # QK-Norm (Qwen3 and GLM-4.7 use RMSNorm on Q and K)
        q_norm_name = f"{prefix}.q_norm.weight"
        k_norm_name = f"{prefix}.k_norm.weight"
        if q_norm_name in self._weight_map:
            w = _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=q_norm_name,
                device=device,
                step="gqa.q_norm",
            )
            w = self._maybe_apply_delta_rms_bias(w, q_norm_name)
            weights["q_norm"] = w
        if k_norm_name in self._weight_map:
            w = _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=k_norm_name,
                device=device,
                step="gqa.k_norm",
            )
            w = self._maybe_apply_delta_rms_bias(w, k_norm_name)
            weights["k_norm"] = w

        if self.cfg.gemma4_text:
            weights["v_norm_no_scale"] = True

        # Attention sinks (GPT OSS: learnable logits for attention normalization)
        sinks_name = f"{prefix}.sinks"
        if sinks_name in self._weight_map:
            weights["sinks"] = _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=sinks_name,
                device=device,
                step="gqa.sinks",
            )

        return weights

    def load_layer_norms(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Load input_layernorm and post_attention_layernorm (BF16)."""
        if self.cfg.is_deepseek_v4:
            prefix = self.cfg.layer_tensor_prefix(layer_idx)
            return {
                "input_layernorm": self._load_bf16(
                    f"{prefix}.attn_norm.weight", device
                ),
                "post_attention_layernorm": self._load_bf16(
                    f"{prefix}.ffn_norm.weight", device
                ),
            }
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}"
        norms = {
            "input_layernorm": self._load_bf16(
                f"{prefix}.input_layernorm.weight", device),
            "post_attention_layernorm": self._load_bf16(
                f"{prefix}.post_attention_layernorm.weight", device),
        }
        if self.cfg.gemma4_text:
            for name in (
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "pre_feedforward_layernorm_2",
            ):
                tensor_name = f"{prefix}.{name}.weight"
                if tensor_name in self._weight_map:
                    norms[name] = self._load_bf16(tensor_name, device)
            layer_scalar_name = f"{prefix}.layer_scalar"
            if layer_scalar_name in self._weight_map:
                norms["layer_scalar"] = self._load_bf16(layer_scalar_name, device)
        for key in norms:
            norms[key] = self._maybe_apply_delta_rms_bias(
                norms[key],
                f"{prefix}.{key}.weight",
            )
        return norms

    def load_dense_mlp(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load dense MLP weights for a non-MoE layer.

        Uses quant_cfg.dense_mlp ("int8" or "bf16").
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp"
        load = self._load_and_quantize if self.quant_cfg.dense_mlp == "int8" else self._load_bf16
        return {
            "gate_proj": load(f"{prefix}.gate_proj.weight", device),
            "up_proj": load(f"{prefix}.up_proj.weight", device),
            "down_proj": load(f"{prefix}.down_proj.weight", device),
        }

    def load_moe_gate(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Load MoE router gate weight and optional biases (BF16).

        Supports naming conventions:
        - "mlp.gate" (DeepSeek/Kimi/Qwen3)
        - "mlp.router" (GPT OSS)
        - "router.proj" (Gemma4 text)
        - "moe.gate" + "moe.router_bias" (Step)
        """
        if self.cfg.is_deepseek_v4:
            prefix = f"{self.cfg.layer_tensor_prefix(layer_idx)}.ffn.gate"
            result = {"weight": self._load_bf16(f"{prefix}.weight", device)}
            table_name = f"{prefix}.tid2eid"
            bias_name = f"{prefix}.bias"
            if layer_idx < self.cfg.num_hash_layers:
                if table_name not in self._weight_map:
                    raise KeyError(f"Missing DeepSeek-V4 hash table {table_name}")
                result["tid2eid"] = self._load_i64(table_name, device)
            else:
                if bias_name not in self._weight_map:
                    raise KeyError(f"Missing DeepSeek-V4 router bias {bias_name}")
                result["e_score_correction_bias"] = self._load_f32(
                    bias_name, device
                )
            return result
        # Detect naming: "gate" vs "router"
        prefix_gate = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.gate"
        prefix_router = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.router"
        prefix_gemma_router = f"{self.cfg.layers_prefix}.layers.{layer_idx}.router"
        prefix_step_gate = f"{self.cfg.layers_prefix}.layers.{layer_idx}.moe.gate"
        if f"{prefix_gate}.weight" in self._weight_map:
            prefix = prefix_gate
            weight_name = f"{prefix}.weight"
        elif f"{prefix_step_gate}.weight" in self._weight_map:
            prefix = prefix_step_gate
            weight_name = f"{prefix}.weight"
        elif f"{prefix_gemma_router}.proj.weight" in self._weight_map:
            prefix = prefix_gemma_router
            weight_name = f"{prefix}.proj.weight"
        else:
            prefix = prefix_router
            weight_name = f"{prefix}.weight"
        result = {
            "weight": self._load_bf16(weight_name, device),
        }
        # Router bias (GPT OSS)
        bias_name = f"{prefix}.bias"
        if bias_name in self._weight_map:
            result["bias"] = self._load_bf16(bias_name, device)
        # e_score_correction_bias (Kimi K2.5)
        corr_name = f"{prefix}.e_score_correction_bias"
        if corr_name in self._weight_map:
            result["e_score_correction_bias"] = self._load_bf16(corr_name, device)
        step_router_bias = f"{self.cfg.layers_prefix}.layers.{layer_idx}.moe.router_bias"
        if step_router_bias in self._weight_map:
            result["e_score_correction_bias"] = self._load_bf16(step_router_bias, device)
        if self.cfg.gemma4_text:
            scale_name = f"{prefix}.scale"
            per_expert_scale_name = f"{prefix}.per_expert_scale"
            if scale_name in self._weight_map:
                result["input_scale"] = self._load_bf16(scale_name, device)
            if per_expert_scale_name in self._weight_map:
                result["per_expert_scale"] = self._load_bf16(per_expert_scale_name, device)
        return result

    def load_shared_expert(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load shared expert MLP weights for a MoE layer.

        Uses quant_cfg.shared_expert ("int8" or "bf16").
        Handles both naming conventions:
        - "shared_experts" (plural, DeepSeek/Kimi)
        - "shared_expert" (singular, Qwen3-Next)
        - "share_expert" (Step sibling of "moe")
        """
        if self.cfg.is_deepseek_v4:
            prefix = f"{self.cfg.layer_tensor_prefix(layer_idx)}.ffn.shared_experts"
            load = (
                self._load_and_quantize
                if self.quant_cfg.shared_expert == "int8"
                else self._load_bf16
            )
            return {
                "gate_proj": load(f"{prefix}.w1.weight", device),
                "up_proj": load(f"{prefix}.w3.weight", device),
                "down_proj": load(f"{prefix}.w2.weight", device),
            }
        # Detect naming convention from weight map
        prefix_plural = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_experts"
        prefix_singular = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_expert"
        prefix_step = f"{self.cfg.layers_prefix}.layers.{layer_idx}.share_expert"
        if f"{prefix_plural}.gate_proj.weight" in self._weight_map:
            prefix = prefix_plural
        elif f"{prefix_step}.gate_proj.weight" in self._weight_map:
            prefix = prefix_step
        else:
            prefix = prefix_singular

        load = self._load_and_quantize if self.quant_cfg.shared_expert == "int8" else self._load_bf16
        result = {
            "gate_proj": load(f"{prefix}.gate_proj.weight", device),
            "up_proj": load(f"{prefix}.up_proj.weight", device),
            "down_proj": load(f"{prefix}.down_proj.weight", device),
        }

        # Shared expert gate (Qwen3-Next): sigmoid gate on shared expert output
        # Weight: [1, hidden_size] — projects hidden → scalar per token
        gate_name = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        if gate_name in self._weight_map:
            result["shared_expert_gate"] = self._load_bf16(gate_name, device)

        return result

    def load_linear_attention_weights(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load Gated DeltaNet linear attention weights for one layer.

        Loads: in_proj_qkvz, in_proj_ba, out_proj (quantizable),
               conv1d.weight, A_log, dt_bias, norm.weight (always BF16).
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.linear_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        # Quantizable projections — handle fused (QCN) or separate (Qwen3.5) format
        fused_qkvz = f"{prefix}.in_proj_qkvz.weight"
        if fused_qkvz in self._weight_map:
            # Fused format: in_proj_qkvz = [Q, K, V, Z], in_proj_ba = [B, A]
            weights["in_proj_qkvz"] = load_proj(fused_qkvz, device)
            weights["in_proj_ba"] = load_proj(f"{prefix}.in_proj_ba.weight", device)
        else:
            # Separate format (Qwen3.5): rearrange into fused interleaved format
            # QKV is flat [Q_all, K_all, V_all], Z/B/A are flat per-head
            # Must interleave per key-head group to match QCN's fused layout
            qkv_raw = self._load_bf16(f"{prefix}.in_proj_qkv.weight", device)
            z_raw = self._load_bf16(f"{prefix}.in_proj_z.weight", device)
            b_raw = self._load_bf16(f"{prefix}.in_proj_b.weight", device)
            a_raw = self._load_bf16(f"{prefix}.in_proj_a.weight", device)

            nk = self.cfg.linear_num_key_heads    # 16
            dk = self.cfg.linear_key_head_dim      # 128
            hr = self.cfg.linear_num_value_heads // nk  # 2
            dv = self.cfg.linear_value_head_dim    # 128
            key_dim = nk * dk   # 2048
            val_dim = self.cfg.linear_num_value_heads * dv  # 4096

            # Interleave QKVZ per key-head group: [q_i, k_i, v_i, z_i] for each group i
            qkvz_parts = []
            for i in range(nk):
                qkvz_parts.append(qkv_raw[i * dk : (i + 1) * dk])              # q group i
                qkvz_parts.append(qkv_raw[key_dim + i * dk : key_dim + (i + 1) * dk])  # k group i
                qkvz_parts.append(qkv_raw[key_dim * 2 + i * hr * dv : key_dim * 2 + (i + 1) * hr * dv])  # v group i
                qkvz_parts.append(z_raw[i * hr * dv : (i + 1) * hr * dv])      # z group i
            weights["in_proj_qkvz"] = torch.cat(qkvz_parts, dim=0)

            # Interleave BA per key-head group: [b_i, a_i] for each group i
            ba_parts = []
            for i in range(nk):
                ba_parts.append(b_raw[i * hr : (i + 1) * hr])
                ba_parts.append(a_raw[i * hr : (i + 1) * hr])
            weights["in_proj_ba"] = torch.cat(ba_parts, dim=0)
        weights["out_proj"] = load_proj(f"{prefix}.out_proj.weight", device)

        # Small/critical weights — always BF16
        weights["conv1d_weight"] = self._load_bf16(f"{prefix}.conv1d.weight", device)
        weights["A_log"] = self._load_bf16(f"{prefix}.A_log", device)
        weights["dt_bias"] = self._load_bf16(f"{prefix}.dt_bias", device)
        norm_weight = self._load_bf16(f"{prefix}.norm.weight", device)
        weights["norm_weight"] = self._maybe_apply_delta_rms_bias(
            norm_weight,
            f"{prefix}.norm.weight",
        )

        return weights

    def load_kimi_delta_attention_weights(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Load one GLM-5.3 Kimi Delta Attention tensor set."""
        if not self.cfg.is_kimi_delta_attention_layer(layer_idx):
            raise ValueError(
                f"KDA weights requested for non-KDA layer {layer_idx}"
            )
        prefix = f"{self.cfg.layer_tensor_prefix(layer_idx)}.self_attn"
        projection_names = (
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
        result = {
            name: _timed_bf16_load(
                self,
                layer_idx=layer_idx,
                tensor_name=f"{prefix}.{name}.weight",
                device=device,
                step=f"kimi_delta_attention.{name}",
            )
            for name in projection_names
        }
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            result[name] = self._load_bf16(f"{prefix}.{name}.weight", device)
        result["A_log"] = self._load_numeric_f32(f"{prefix}.A_log", device)
        result["dt_bias"] = self._load_numeric_f32(f"{prefix}.dt_bias", device)
        result["o_norm"] = self._load_numeric_f32(
            f"{prefix}.o_norm.weight", device
        )
        return result

    def load_glm5_next_hyper_connection(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Load GLM-5.3 mHC parameters in their FP32 execution format."""
        if not self.cfg.is_glm5_next:
            raise ValueError("GLM-5.3 mHC requested for another architecture")
        prefix = self.cfg.layer_tensor_prefix(layer_idx)
        return {
            name: self._load_numeric_f32(f"{prefix}.{name}", device)
            for name in (
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
            )
        }

    def load_mamba2_weights(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load Mamba2 SSM weights for one layer (Nemotron-H).

        Loads: in_proj, out_proj (quantizable projections),
               conv1d.weight, conv1d.bias, A_log, D, dt_bias, norm.weight (always BF16/FP32).
        """
        # Nemotron-H uses "mixer" instead of "self_attn"
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer"
        weights = {}

        # Large projections (quantizable via Marlin for decode)
        weights["in_proj"] = self._load_bf16(f"{prefix}.in_proj.weight", device)
        # rescale_prenorm_residual is an HF initialization rule. from_pretrained
        # loads trained checkpoint tensors after initialization, so runtime
        # checkpoint loads must not apply that scale again.
        weights["out_proj"] = self._load_bf16(f"{prefix}.out_proj.weight", device)

        # Small SSM parameters — always FP32 for numerical precision
        weights["conv1d_weight"] = self._load_bf16(f"{prefix}.conv1d.weight", device)
        conv_bias_name = f"{prefix}.conv1d.bias"
        if conv_bias_name in self._weight_map:
            weights["conv1d_bias"] = self._load_bf16(conv_bias_name, device)
        weights["A_log"] = self._load_bf16(f"{prefix}.A_log", device)
        weights["D"] = self._load_bf16(f"{prefix}.D", device)
        weights["dt_bias"] = self._load_bf16(f"{prefix}.dt_bias", device)
        weights["norm_weight"] = self._load_bf16(f"{prefix}.norm.weight", device)

        return weights

    def load_nemotron_attention_weights(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load GQA attention weights for a Nemotron-H attention layer.

        Same structure as standard GQA but uses 'mixer' prefix instead of 'self_attn'.
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer"
        weights = {}
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weights[proj] = self._load_bf16(f"{prefix}.{proj}.weight", device)
        return weights

    def load_latent_moe_projections(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load LatentMoE latent projections for a Nemotron-H MoE layer.

        Returns fc1_latent_proj (hidden -> latent) and fc2_latent_proj (latent -> hidden).
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer"
        return {
            "fc1_latent_proj": self._load_bf16(f"{prefix}.fc1_latent_proj.weight", device),
            "fc2_latent_proj": self._load_bf16(f"{prefix}.fc2_latent_proj.weight", device),
        }

    def load_nemotron_moe_gate(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load MoE gate for a Nemotron-H layer (uses 'mixer.gate' prefix)."""
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer.gate"
        result = {
            "weight": self._load_bf16(f"{prefix}.weight", device),
        }
        corr_name = f"{prefix}.e_score_correction_bias"
        if corr_name in self._weight_map:
            result["e_score_correction_bias"] = self._load_bf16(corr_name, device)
        return result

    def load_nemotron_shared_expert(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load shared expert for Nemotron-H (relu2 activation, no gate_proj)."""
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer.shared_experts"
        load = self._load_and_quantize if self.quant_cfg.shared_expert == "int8" else self._load_bf16
        return {
            "up_proj": load(f"{prefix}.up_proj.weight", device),
            "down_proj": load(f"{prefix}.down_proj.weight", device),
        }

    def load_nemotron_layer_norm(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load layer norm for Nemotron-H (single norm per layer, no post_attn_norm).

        Nemotron blocks have exactly one mixer (Mamba2/MoE/Attention) with pre-norm only.
        post_attention_layernorm is set to None to signal the decode loop to skip it.
        """
        name = f"{self.cfg.layers_prefix}.layers.{layer_idx}.norm.weight"
        return {
            "input_layernorm": self._load_bf16(name, device),
            "post_attention_layernorm": None,  # signal: skip post_attn_norm
        }

    def load_layer(
        self, layer_idx: int, device: torch.device,
        attn_device: torch.device = None,
    ) -> Dict[str, any]:
        """Load all GPU weights for one layer.

        Args:
            device: Device for norms, gate, shared expert, dense MLP.
            attn_device: Device for attention weights. Defaults to device.
                         Pass torch.device('cpu') to keep attention in system RAM.

        Returns a dict with:
        - "norms": {input_layernorm, post_attention_layernorm}
        - "attention" or "linear_attention" or "mamba2": attention/SSM weights
        - "layer_type": "linear_attention" | "full_attention" | "mamba2" | "moe"
        - "is_moe": bool
        """
        if attn_device is None:
            attn_device = device
        start = time.perf_counter()

        # Nemotron-H: dispatch by hybrid_override_pattern layer type
        if self.cfg.is_nemotron_h:
            return self._load_nemotron_layer(layer_idx, device, start)

        is_linear = self.cfg.is_linear_attention_layer(layer_idx)
        if is_linear:
            layer_type = "linear_attention"
        elif self.cfg.is_sliding_attention_layer(layer_idx):
            layer_type = "sliding_attention"
        else:
            layer_type = "full_attention"

        result = {
            "norms": self.load_layer_norms(layer_idx, device),
            "is_moe": self.cfg.is_moe_layer(layer_idx),
            "layer_type": layer_type,
        }
        if self.cfg.is_deepseek_v4:
            result["hyper_connection"] = self.load_deepseek_v4_hyper_connection(
                layer_idx, device
            )
        elif self.cfg.is_glm5_next:
            result["hyper_connection"] = self.load_glm5_next_hyper_connection(
                layer_idx, device
            )
        norms_elapsed = time.perf_counter() - start
        _emit_real_model_timing(
            {
                "phase": "load_layer_step",
                "layer_idx": int(layer_idx),
                "step": "norms",
                "elapsed_s": norms_elapsed,
            }
        )

        # Load attention weights based on layer type
        # Linear attention is NOT affected by AWQ (AWQ only applies to GQA layers),
        # so linear attention always loads to the primary GPU device.
        attention_started = time.perf_counter()
        if is_linear:
            if self.cfg.is_kimi_delta_attention_layer(layer_idx):
                result["kimi_delta_attention"] = (
                    self.load_kimi_delta_attention_weights(layer_idx, device)
                )
            else:
                result["linear_attention"] = self.load_linear_attention_weights(
                    layer_idx, device
                )
        else:
            result["attention"] = self.load_attention_weights(
                layer_idx, device, proj_device=attn_device)
        attention_elapsed = time.perf_counter() - attention_started
        _emit_real_model_timing(
            {
                "phase": "load_layer_step",
                "layer_idx": int(layer_idx),
                "step": "attention",
                "elapsed_s": attention_elapsed,
            }
        )

        mlp_started = time.perf_counter()
        if result["is_moe"]:
            result["gate"] = self.load_moe_gate(layer_idx, device)
            if self.cfg.gemma4_text:
                result["dense_mlp"] = self.load_dense_mlp(layer_idx, device)
            if self.cfg.n_shared_experts > 0:
                result["shared_expert"] = self.load_shared_expert(layer_idx, device)
        else:
            result["dense_mlp"] = self.load_dense_mlp(layer_idx, device)
        mlp_elapsed = time.perf_counter() - mlp_started
        _emit_real_model_timing(
            {
                "phase": "load_layer_step",
                "layer_idx": int(layer_idx),
                "step": "mlp_gate",
                "elapsed_s": mlp_elapsed,
            }
        )

        elapsed = time.perf_counter() - start
        alloc_mb = self._memory_allocated_mb(device)
        _emit_real_model_timing(
            {
                "phase": "load_layer",
                "layer_idx": int(layer_idx),
                "layer_type": layer_type,
                "is_moe": bool(result["is_moe"]),
                "norms_s": norms_elapsed,
                "attention_s": attention_elapsed,
                "mlp_gate_s": mlp_elapsed,
                "total_s": elapsed,
            }
        )
        logger.info(
            "Layer %d loaded in %.1fs (GPU alloc: %.0f MB, moe=%s, type=%s)",
            layer_idx, elapsed, alloc_mb, result["is_moe"], layer_type,
        )
        return result

    def _load_nemotron_layer(
        self, layer_idx: int, device: torch.device, start: float,
    ) -> Dict[str, any]:
        """Load a Nemotron-H layer based on the hybrid pattern type."""
        layer_type = self.cfg.layer_types[layer_idx]
        norms = self.load_nemotron_layer_norm(layer_idx, device)

        if layer_type == "mamba2":
            result = {
                "norms": norms,
                "is_moe": False,
                "layer_type": "mamba2",
                "mamba2": self.load_mamba2_weights(layer_idx, device),
            }
        elif layer_type == "moe":
            result = {
                "norms": norms,
                "is_moe": True,
                "layer_type": "moe",
                "gate": self.load_nemotron_moe_gate(layer_idx, device),
            }
            # LatentMoE projections: only if moe_latent_size > 0 (Ultra models)
            if self.cfg.moe_latent_size > 0:
                result["latent_proj"] = self.load_latent_moe_projections(layer_idx, device)
            if self.cfg.n_shared_experts > 0:
                result["shared_expert"] = self.load_nemotron_shared_expert(layer_idx, device)
        elif layer_type == "full_attention":
            # Nemotron attention layers are JUST attention (no MoE).
            # Each block has exactly one mixer: Mamba2, MoE, or Attention.
            result = {
                "norms": norms,
                "is_moe": False,
                "layer_type": "full_attention",
                "attention": self.load_nemotron_attention_weights(layer_idx, device),
            }
        else:
            raise ValueError(f"Unknown Nemotron layer type: {layer_type}")

        elapsed = time.perf_counter() - start
        alloc_mb = self._memory_allocated_mb(device)
        logger.info(
            "Layer %d loaded in %.1fs (GPU alloc: %.0f MB, type=%s)",
            layer_idx, elapsed, alloc_mb, layer_type,
        )
        return result

    def close(self):
        """Close all safetensors handles."""
        self._handles.clear()
