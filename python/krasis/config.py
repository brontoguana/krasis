"""Model config parsing and PP partition for Krasis standalone server."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional, Union

from krasis.checkpoint_identity import cache_namespace


ATTENTION_QUANT_CHOICES = (
    "bf16",
    "hqq4",
    "hqq46",
    "hqq46_auto",
    "hqq6",
    "hqq68_auto",
    "hqq8",
)
DEPRECATED_ATTENTION_QUANT_CHOICES = ("awq",)
KV_CACHE_FORMAT_CHOICES = ("bf16", "bfloat16", "native", "k8v4", "k8v6", "k7v4", "k6v6", "k6v4", "k4v4", "tq4")
DEPRECATED_KV_CACHE_FORMAT_CHOICES = ("fp8", "fp8_e4m3", "polar4")
GPU_EXPERT_INT4_CALIB_CHOICES = ("amax", "search_rmse")
HQQ_CACHE_PROFILE_BASELINE = "baseline"
HQQ_CACHE_PROFILE_SELFCAL_V1 = "selfcal_v1"
HQQ_CACHE_PROFILE_CHOICES = (HQQ_CACHE_PROFILE_BASELINE, HQQ_CACHE_PROFILE_SELFCAL_V1)
HQQ_ATTENTION_GROUP_SIZE_CHOICES = (32, 64, 128)
HQQ_ATTENTION_DEFAULT_GROUP_SIZE = 128
ADAPTIVE_COLD_MASS_PRUNING_CHOICES = ("off", "75/3", "75/5", "75/8", "75/10")

_ADAPTIVE_COLD_MASS_PRUNING_ENV_KEYS = (
    "KRASIS_ADAPTIVE_COLD_DROP",
    "KRASIS_ADAPTIVE_COLD_DROP_PROTECT_PCT",
    "KRASIS_ADAPTIVE_COLD_DROP_MASS_PCT",
)


def configure_adaptive_cold_mass_pruning(
    policy: Optional[str],
    environ: Optional[MutableMapping[str, str]] = None,
) -> Optional[str]:
    """Apply a launcher/config policy to the Rust adaptive cold-drop env contract."""
    if policy is None:
        return None
    normalized = str(policy).strip().lower()
    if normalized not in ADAPTIVE_COLD_MASS_PRUNING_CHOICES:
        raise ValueError(
            f"Unsupported adaptive cold-mass pruning policy {policy!r}. "
            f"Use one of: {', '.join(ADAPTIVE_COLD_MASS_PRUNING_CHOICES)}."
        )

    target = os.environ if environ is None else environ
    for key in _ADAPTIVE_COLD_MASS_PRUNING_ENV_KEYS:
        target.pop(key, None)

    if normalized != "off":
        protect_pct, mass_pct = normalized.split("/", 1)
        target["KRASIS_ADAPTIVE_COLD_DROP"] = "1"
        target["KRASIS_ADAPTIVE_COLD_DROP_PROTECT_PCT"] = protect_pct
        target["KRASIS_ADAPTIVE_COLD_DROP_MASS_PCT"] = mass_pct

    return normalized


def cache_dir_for_model(model_path: str) -> str:
    """Return the immutable-checkpoint-isolated cache directory for a model."""
    home = os.path.expanduser("~")
    return os.path.join(home, ".krasis", "cache", cache_namespace(model_path))


def marlin_cache_basename(gpu_bits: int, group_size: Union[int, str], gpu_expert_int4_calib: str = "amax") -> str:
    calib_suffix = ""
    if gpu_bits == 4:
        calib_suffix = f"_cal{gpu_expert_int4_calib.replace('_', '')}"
    return f"experts_marlin_int{gpu_bits}_g{group_size}{calib_suffix}.bin"


def _collect_eos_ids(raw: dict, cfg: dict, gen_cfg: dict) -> list:
    """Collect all unique EOS token IDs from config.json and generation_config.json.

    generation_config.json is authoritative for stop tokens (often has the full
    list while config.json only has one).  Merge both, preserving order.
    """
    ids = []
    seen = set()
    # generation_config.json first (authoritative for generation)
    text_cfg = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    for source in (gen_cfg, raw, text_cfg, cfg):
        eos = source.get("eos_token_id")
        if eos is None:
            continue
        items = eos if isinstance(eos, list) else [eos]
        for v in items:
            if isinstance(v, int) and v not in seen:
                ids.append(v)
                seen.add(v)
    return ids if ids else [0]


def _parse_eos_token_id(raw: dict, cfg: dict, gen_cfg: dict) -> int:
    """Primary EOS token ID (first from merged list)."""
    return _collect_eos_ids(raw, cfg, gen_cfg)[0]


def _parse_extra_stop_ids(raw: dict, cfg: dict, gen_cfg: dict) -> tuple:
    """Additional stop token IDs beyond the primary EOS."""
    ids = _collect_eos_ids(raw, cfg, gen_cfg)
    return tuple(ids[1:]) if len(ids) > 1 else ()


def _parse_int_list(value: Any, field_name: str, *, max_len: Optional[int] = None) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
        parsed = [int(part) for part in items]
    elif isinstance(value, list):
        parsed = [int(part) for part in value]
    else:
        raise ValueError(f"{field_name} must be a comma-separated string or list")
    if max_len is not None:
        for idx in parsed:
            if idx < 0 or idx >= max_len:
                raise ValueError(f"{field_name} contains out-of-range layer {idx} for {max_len} layers")
    return parsed


def _parse_float_list(value: Any, field_name: str, *, max_len: Optional[int] = None) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a scalar or list")
    parsed = [float(part) for part in value]
    if max_len is not None and len(parsed) < max_len:
        raise ValueError(f"{field_name} has {len(parsed)} entries, expected at least {max_len}")
    return parsed[:max_len] if max_len is not None else parsed


def _infer_from_weights(model_path: str, cfg: dict) -> dict:
    """Infer missing config fields from safetensors weight shapes.

    Some VL models (DeepSeek-VL2) have incomplete language_config that's
    missing num_hidden_layers, num_attention_heads, MLA dims, etc.
    We infer these from the actual weight tensor shapes.
    """
    needed = {"num_hidden_layers", "num_attention_heads"}
    # Also need MLA dims if model has kv_a_proj (MLA) but no kv_lora_rank in config
    if all(k in cfg for k in needed) and "kv_lora_rank" in cfg:
        return cfg  # nothing missing

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return cfg

    with open(index_path) as f:
        index = json.load(f)
    wmap = index.get("weight_map", {})

    # Detect prefix from weights — find the one with attention layers (not projector/vision)
    prefix = None
    for key in wmap:
        pos = key.find(".layers.")
        if pos > 0 and "self_attn" in key:
            prefix = key[:pos]
            break
    if not prefix:
        return cfg

    # Infer num_hidden_layers by counting layer indices
    if "num_hidden_layers" not in cfg:
        layers = set()
        for k in wmap:
            if k.startswith(f"{prefix}.layers."):
                rest = k[len(prefix) + 8:]  # skip ".layers."
                try:
                    layers.add(int(rest.split(".")[0]))
                except ValueError:
                    pass
        if layers:
            cfg = dict(cfg)
            cfg["num_hidden_layers"] = max(layers) + 1

    # Infer MLA dims from weight shapes if kv_a_proj_with_mqa exists
    kv_a_key = f"{prefix}.layers.0.self_attn.kv_a_proj_with_mqa.weight"
    if kv_a_key in wmap and "kv_lora_rank" not in cfg:
        import struct as _struct
        # Read shapes from safetensors header
        shapes = {}
        _header_cache = {}
        def _get_shape(tensor_name):
            shard = wmap.get(tensor_name)
            if not shard:
                return None
            if shard not in _header_cache:
                fpath = os.path.join(model_path, shard)
                with open(fpath, "rb") as f:
                    hlen = _struct.unpack("<Q", f.read(8))[0]
                    _header_cache[shard] = json.loads(f.read(hlen))
            info = _header_cache[shard].get(tensor_name)
            return info["shape"] if info else None

        cfg = dict(cfg)

        # kv_a_proj_with_mqa: [kv_lora_rank + qk_rope_head_dim, hidden_size]
        # kv_a_layernorm: [kv_lora_rank]
        # kv_b_proj: [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        # o_proj: [hidden_size, n_heads * v_head_dim]
        # q_proj: [n_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden_size]
        ln_shape = _get_shape(f"{prefix}.layers.0.self_attn.kv_a_layernorm.weight")
        kv_a_shape = _get_shape(kv_a_key)
        kv_b_shape = _get_shape(f"{prefix}.layers.0.self_attn.kv_b_proj.weight")
        o_shape = _get_shape(f"{prefix}.layers.0.self_attn.o_proj.weight")
        q_shape = _get_shape(f"{prefix}.layers.0.self_attn.q_proj.weight")

        if ln_shape and kv_a_shape and kv_b_shape and o_shape and q_shape:
            kv_lora_rank = ln_shape[0]
            qk_rope_head_dim = kv_a_shape[0] - kv_lora_rank
            hidden_size = cfg.get("hidden_size", o_shape[0])

            # o_proj: [hidden, n_heads * v_head_dim]
            total_v = o_shape[1]
            # kv_b: [n_heads * (nope + v), kv_lora_rank]
            total_kv_b = kv_b_shape[0]
            # q_proj: [n_heads * (nope + rope), hidden]
            total_q = q_shape[0]

            # Solve: n_heads * v_head_dim = total_v
            #        n_heads * (nope + v) = total_kv_b
            #        n_heads * (nope + rope) = total_q
            # From q: n_heads * nope = total_q - n_heads * rope
            # From kv_b: n_heads * nope + total_v = total_kv_b
            #   → total_q - n_heads * rope + total_v = total_kv_b
            #   → n_heads = (total_q + total_v - total_kv_b) / rope  ... doesn't simplify easily
            # Try common head dims: 128
            for v_head in (128, 64, 96, 256):
                if total_v % v_head == 0:
                    n_heads = total_v // v_head
                    nope_plus_v = total_kv_b // n_heads if total_kv_b % n_heads == 0 else 0
                    nope = nope_plus_v - v_head
                    if nope > 0 and total_q == n_heads * (nope + qk_rope_head_dim):
                        cfg.setdefault("kv_lora_rank", kv_lora_rank)
                        cfg.setdefault("qk_nope_head_dim", nope)
                        cfg.setdefault("qk_rope_head_dim", qk_rope_head_dim)
                        cfg.setdefault("v_head_dim", v_head)
                        cfg.setdefault("num_attention_heads", n_heads)
                        cfg.setdefault("num_key_value_heads", n_heads)
                        break

    return cfg



def _normalise_layers_block_type(cfg: dict) -> dict:
    """Derive layer count from Nemotron-H layers_block_type when present."""
    raw_layer_blocks = cfg.get("layers_block_type")
    if raw_layer_blocks is None:
        return cfg
    if not isinstance(raw_layer_blocks, list) or not raw_layer_blocks:
        raise ValueError("layers_block_type must be a non-empty list")
    for idx, label in enumerate(raw_layer_blocks):
        if not isinstance(label, str):
            raise ValueError(f"layers_block_type[{idx}] must be a string")
    if "num_hidden_layers" in cfg and int(cfg["num_hidden_layers"]) != len(raw_layer_blocks):
        raise ValueError(
            f"layers_block_type length {len(raw_layer_blocks)} != "
            f"num_hidden_layers {cfg['num_hidden_layers']}"
        )
    cfg = dict(cfg)
    cfg["num_hidden_layers"] = len(raw_layer_blocks)
    return cfg


def _detect_layers_prefix(model_path: str) -> str:
    """Auto-detect the tensor prefix by scanning the safetensors index.

    Looks for '.layers.' in weight names to determine the prefix:
      - "model.layers.0..." → "model"
      - "model.language_model.layers.0..." → "model.language_model"
      - "language_model.model.layers.0..." → "language_model.model"
      - "language.model.layers.0..." → "language.model"
    Falls back to heuristic if no index file found.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        def is_auxiliary_prefix(prefix: str) -> bool:
            return (
                prefix == "mtp"
                or prefix.endswith(".mtp")
                or prefix == "model.visual"
                or ".visual" in prefix
            )
        # Prefer keys with self_attn to disambiguate from projector/vision layers
        for key in index.get("weight_map", {}):
            pos = key.find(".layers.")
            if pos > 0 and "self_attn" in key:
                prefix = key[:pos]
                if not is_auxiliary_prefix(prefix):
                    return prefix
        # DeepSeek-V4 uses a root-level namespace: layers.0.attn.*, layers.0.ffn.*.
        # The empty prefix is intentional and is handled by tensor_name helpers.
        if any(key.startswith("layers.") for key in index.get("weight_map", {})):
            return ""
        # Fallback: any .layers. key
        for key in index.get("weight_map", {}):
            pos = key.find(".layers.")
            if pos > 0:
                prefix = key[:pos]
                if not is_auxiliary_prefix(prefix):
                    return prefix
    # Fallback: check for text_config in config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            raw = json.load(f)
        if "text_config" in raw:
            return "language_model.model"
    return "model"


@dataclass
class QuantConfig:
    """Per-component quantization config for GPU weights.

    Components NOT configurable (always BF16): embedding, kv_b_proj/w_kc/w_vc,
    layernorms, gate weight. These are either too quality-critical or too small.
    """
    lm_head: str = "int8"          # "bf16" or "int8" ("bf16" remains an unvalidated debug path)
    attention: str = "bf16" # "bf16", "hqq4", "hqq46", "hqq46_auto", "hqq6", "hqq68_auto", or "hqq8" (native HQQ attention; "bf16" is debug-oriented, not an oracle)
    shared_expert: str = "int8"    # "bf16" or "int8" ("bf16" remains an unvalidated debug path)
    dense_mlp: str = "int8"        # "bf16" or "int8" ("bf16" remains an unvalidated debug path)
    gpu_expert_bits: int = 4       # 3 (TileQ), 4/8 (Marlin), or 16 (UNVALIDATED BF16 debug-only path)
    tileq_cache: Optional[str] = None  # explicit source-bound KTQ1 artifact when gpu_expert_bits=3
    expert_group_size: int = 128   # routed expert quantization group size; 32 matches Q8_0-style block scale granularity
    gpu_expert_int4_calib: str = "amax"  # "amax" or "search_rmse" for routed-expert GPU INT4 cache build
    cpu_expert_bits: int = 4       # 4 or 8 for CPU expert quantization
    kv_cache_format: str = "k6v6"  # Generic modes: k6v6/k4v4/bf16; Native is architecture-owned
    ring_window_kv: bool = False    # Experimental: cap sliding-attention KV layers to their window
    hqq_cache_profile: str = HQQ_CACHE_PROFILE_BASELINE  # "baseline" or an explicit calibrated HQQ profile
    hqq_group_size: int = HQQ_ATTENTION_DEFAULT_GROUP_SIZE  # HQQ attention quantization group size
    hqq_auto_budget_pct: Optional[float] = None  # auto promotion budget as % of base-to-target attention span
    hqq46_auto_budget_mib: Optional[int] = None  # legacy HQQ4/6 auto promotion budget in MiB
    hqq_sidecar_manifest: Optional[str] = None  # explicit HQQ4-only sidecar manifest for switchable correction
    step_vision_quant: str = "int4"  # "bf16" or "int4"; lazy vision image paths; legacy field name
    step_vision_group_size: int = 128  # lazy vision INT4 row group size; legacy field name

    def __post_init__(self):
        kv_aliases = {
            "bf16": "bf16",
            "bfloat16": "bf16",
            "k8v4": "k8v4",
            "k8v6": "k8v6",
            "k7v4": "k7v4",
            "k6v6": "k6v6",
            "k6v4": "k6v4",
            "k4v4": "k4v4",
            "tq4": "tq4",
            "native": "native",
        }
        if self.kv_cache_format in DEPRECATED_KV_CACHE_FORMAT_CHOICES:
            raise ValueError(
                f"kv_cache_format='{self.kv_cache_format}' is deprecated and disabled. "
                "Use 'k6v6' for quality, 'k4v4' for compact KV, or 'bf16' for full precision."
            )
        if self.kv_cache_format not in kv_aliases:
            raise ValueError(
                f"Unsupported kv_cache_format '{self.kv_cache_format}'. "
                "Use generic modes 'k6v6', 'k4v4', or 'bf16', or an architecture-owned "
                "'native' mode where supported; internal modes include "
                "'k8v4', 'k8v6', 'k7v4', 'k6v4', and 'tq4'."
            )
        self.kv_cache_format = kv_aliases[self.kv_cache_format]

        if self.attention in ("int4", "int8"):
            raise ValueError(
                f"Unsupported attention quant '{self.attention}'. "
                "Naive int4/int8 attention has been removed; use 'hqq4', 'hqq46', 'hqq46_auto', 'hqq6', 'hqq68_auto', 'hqq8', or 'bf16'."
            )
        if self.attention in DEPRECATED_ATTENTION_QUANT_CHOICES:
            raise ValueError(
                f"attention='{self.attention}' is deprecated and disabled. "
                "Use HQQ attention modes: 'hqq4', 'hqq46', 'hqq46_auto', 'hqq6', 'hqq68_auto', or 'hqq8'."
            )
        if self.attention not in ATTENTION_QUANT_CHOICES:
            raise ValueError(
                f"Unsupported attention quant '{self.attention}'. "
                f"Use one of: {', '.join(ATTENTION_QUANT_CHOICES)}."
            )
        self.hqq_cache_profile = str(self.hqq_cache_profile or HQQ_CACHE_PROFILE_BASELINE).strip().lower()
        if self.hqq_cache_profile not in HQQ_CACHE_PROFILE_CHOICES:
            raise ValueError(
                f"Unsupported hqq_cache_profile '{self.hqq_cache_profile}'. "
                f"Use one of: {', '.join(HQQ_CACHE_PROFILE_CHOICES)}."
            )
        try:
            self.hqq_group_size = int(self.hqq_group_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"hqq_group_size must be an integer, got {self.hqq_group_size!r}"
            ) from exc
        if self.hqq_group_size not in HQQ_ATTENTION_GROUP_SIZE_CHOICES:
            raise ValueError(
                f"Unsupported hqq_group_size={self.hqq_group_size}. "
                "Use 32, 64, or 128."
            )
        if not self.attention.startswith("hqq") and self.hqq_cache_profile != HQQ_CACHE_PROFILE_BASELINE:
            raise ValueError(
                f"hqq_cache_profile={self.hqq_cache_profile} requires an HQQ attention mode. "
                "Non-HQQ attention backends must use hqq_cache_profile='baseline'."
            )
        if not self.attention.startswith("hqq") and self.hqq_group_size != HQQ_ATTENTION_DEFAULT_GROUP_SIZE:
            raise ValueError(
                f"hqq_group_size={self.hqq_group_size} requires an HQQ attention mode. "
                "Non-HQQ attention backends must use the default HQQ group size."
            )
        if isinstance(self.hqq_auto_budget_pct, str) and not self.hqq_auto_budget_pct.strip():
            self.hqq_auto_budget_pct = None
        if self.hqq_auto_budget_pct is not None:
            try:
                self.hqq_auto_budget_pct = float(self.hqq_auto_budget_pct)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"hqq_auto_budget_pct must be a numeric percentage, got {self.hqq_auto_budget_pct!r}"
                ) from exc
            if self.hqq_auto_budget_pct < 0.0 or self.hqq_auto_budget_pct > 100.0:
                raise ValueError(
                    f"hqq_auto_budget_pct must satisfy 0 <= pct <= 100, got {self.hqq_auto_budget_pct!r}"
                )
        if isinstance(self.hqq46_auto_budget_mib, str) and not self.hqq46_auto_budget_mib.strip():
            self.hqq46_auto_budget_mib = None
        if self.hqq46_auto_budget_mib is not None:
            try:
                self.hqq46_auto_budget_mib = int(self.hqq46_auto_budget_mib)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"hqq46_auto_budget_mib must be an integer MiB budget, got {self.hqq46_auto_budget_mib!r}"
                ) from exc
        if self.attention == "hqq46_auto":
            if self.hqq_auto_budget_pct is None and (
                self.hqq46_auto_budget_mib is None or self.hqq46_auto_budget_mib <= 0
            ):
                raise ValueError(
                    "attention='hqq46_auto' requires hqq_auto_budget_pct. "
                    "Legacy hqq46_auto_budget_mib remains accepted only for existing configs."
                )
        elif self.attention == "hqq68_auto":
            if self.hqq_auto_budget_pct is None:
                raise ValueError(
                    "attention='hqq68_auto' requires hqq_auto_budget_pct. "
                    "Auto planner budgets are percentages of the HQQ6-to-HQQ8 promotion span."
                )
            if self.hqq46_auto_budget_mib not in (None, 0):
                raise ValueError("hqq46_auto_budget_mib is not valid with attention='hqq68_auto'.")
        elif self.hqq46_auto_budget_mib not in (None, 0):
            raise ValueError("hqq46_auto_budget_mib is only valid with attention='hqq46_auto'.")
        if self.attention not in ("hqq46_auto", "hqq68_auto") and self.hqq_auto_budget_pct is not None:
            raise ValueError("hqq_auto_budget_pct is only valid with attention='hqq46_auto' or attention='hqq68_auto'.")
        if self.hqq_sidecar_manifest is not None:
            sidecar_path = str(self.hqq_sidecar_manifest).strip()
            self.hqq_sidecar_manifest = os.path.expanduser(sidecar_path) if sidecar_path else None
        if self.hqq_sidecar_manifest is not None and self.attention != "hqq4":
            raise ValueError(
                "hqq_sidecar_manifest requires attention='hqq4'. "
                "HQQ4/6, HQQ6, HQQ6/8, and HQQ8 are clean higher-precision attention modes and do not support sidecar/self-correction."
            )
        if self.hqq_sidecar_manifest is not None and self.hqq_group_size != HQQ_ATTENTION_DEFAULT_GROUP_SIZE:
            raise ValueError(
                "hqq_sidecar_manifest currently requires hqq_group_size=128 because sidecar manifests "
                "are tied to source HQQ group boundaries."
            )
        self.step_vision_quant = str(self.step_vision_quant or "int4").strip().lower()
        if self.step_vision_quant not in ("bf16", "int4"):
            raise ValueError(
                f"Unsupported vision_quant/step_vision_quant '{self.step_vision_quant}'. Use 'bf16' or 'int4'."
            )
        try:
            self.step_vision_group_size = int(self.step_vision_group_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"vision_group_size/step_vision_group_size must be an integer, got {self.step_vision_group_size!r}"
            ) from exc
        if self.step_vision_group_size not in (32, 64, 128):
            raise ValueError(
                f"Unsupported vision_group_size/step_vision_group_size={self.step_vision_group_size}. "
                "Use 32, 64, or 128."
            )
        self.gpu_expert_int4_calib = self.gpu_expert_int4_calib.lower()
        if self.gpu_expert_int4_calib not in GPU_EXPERT_INT4_CALIB_CHOICES:
            raise ValueError(
                f"Unsupported gpu_expert_int4_calib '{self.gpu_expert_int4_calib}'. "
                f"Use one of: {', '.join(GPU_EXPERT_INT4_CALIB_CHOICES)}."
            )
        try:
            self.expert_group_size = int(self.expert_group_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"expert_group_size must be an integer, got {self.expert_group_size!r}"
            ) from exc
        if self.expert_group_size not in (32, 64, 128):
            raise ValueError(
                f"Unsupported expert_group_size={self.expert_group_size}. "
                "Use 32, 64, or 128."
            )


@dataclass
class ModelConfig:
    """Parsed model configuration for MLA (Kimi/DeepSeek) and GQA (Qwen3) models."""

    model_path: str
    hidden_size: int
    intermediate_size: int       # dense MLP intermediate (layer 0)
    moe_intermediate_size: int   # per-expert intermediate
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int

    # MLA dimensions (all None for GQA models)
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # DeepSeek Sparse Attention / IndexShare dimensions (GLM-MoE-DSA).
    # Zero/None means the model does not use DSA.
    index_topk: int = 0
    index_head_dim: int = 0
    index_n_heads: int = 0
    index_topk_freq: int = 0
    index_skip_topk_offset: int = 0
    indexer_types: Optional[List[str]] = None
    indexer_rope_interleave: bool = False
    index_share_for_mtp_iteration: bool = False
    index_kpool: int = 0
    index_kpool_compress: bool = False
    index_kpool_always_select_tail: bool = False

    # DeepSeek-V4 compressed sparse attention. These fields are deliberately
    # separate from MLA/GLM-DSA because the cache and indexer semantics differ.
    attention_head_dim: int = 0
    o_lora_rank: int = 0
    o_groups: int = 0
    compress_ratios: Optional[List[int]] = None
    compress_rope_theta: float = 0.0
    num_hash_layers: int = 0
    hc_mult: int = 0
    hc_sinkhorn_iters: int = 0
    hc_eps: float = 0.0
    expert_dtype: str = ""
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: Optional[List[int]] = None
    dspark_markov_rank: int = 0

    # GQA dimensions (None for MLA models)
    gqa_head_dim: Optional[int] = None    # per-head dim (e.g. 128 for Qwen3)
    global_head_dim: int = 0              # Gemma4 full-attention head dim
    num_global_key_value_heads: int = 0   # Gemma4 full-attention KV heads
    attention_k_eq_v: bool = False        # Gemma4 full-attention value uses k_proj source
    gqa_other_num_attention_heads: int = 0  # Step sliding-attention query heads
    gqa_other_num_key_value_heads: int = 0  # Step sliding-attention KV groups
    gqa_other_head_dim: int = 0             # Step sliding-attention head dim

    # Hybrid model: linear attention (Gated DeltaNet) + full attention
    full_attention_interval: int = 0   # 0 = all full attention; N = every Nth layer is full
    layer_types: Optional[List[str]] = None  # computed: ["linear_attention", ..., "full_attention", ...]
    linear_conv_kernel_dim: int = 4    # conv1d kernel size for linear attention
    linear_key_head_dim: int = 128     # per-head dim for keys in linear attention
    linear_num_key_heads: int = 16     # number of key heads in linear attention
    linear_value_head_dim: int = 128   # per-head dim for values in linear attention
    linear_num_value_heads: int = 32   # number of value heads in linear attention
    linear_attention_family: str = "gated_deltanet"
    linear_gate_lower_bound: float = 0.0

    # MoE
    n_routed_experts: int = 0
    num_experts_per_tok: int = 0
    n_shared_experts: int = 0
    shared_expert_intermediate_size: int = 0  # explicit shared expert size (Qwen3-Next)
    first_k_dense_replace: int = 0
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"         # "sigmoid" or "softmax"
    topk_method: str = "greedy"           # "noaux_tc"
    norm_topk_prob: bool = True
    moe_layer_indices: Optional[List[int]] = None
    use_moe_router_bias: bool = False
    need_fp32_gate: bool = False

    # Norm / activation
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"              # "silu"

    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: Dict[str, Any] = field(default_factory=dict)
    max_position_embeddings: int = 262144
    partial_rotary_factor: float = 1.0  # GLM-4.7 uses 0.5 (only half of head_dim gets RoPE)
    partial_rotary_factors: Optional[List[float]] = None  # Step per-layer partial rotary factors
    rope_theta_layers: Optional[List[float]] = None       # Step per-layer theta values
    yarn_only_types: Optional[List[str]] = None            # Step applies scaled RoPE only to these layer types
    rope_interleave: bool = True  # MLA models: True means q_pe/k_pe need de-interleaving before RoPE

    # Attention
    attention_bias: bool = False       # GLM-4.7, GPT OSS have bias on Q/K/V projections
    sliding_window: int = 0            # GPT OSS: 128 tokens for sliding_attention layers

    # Pre-quantized experts
    expert_quant_method: str = ""      # "mxfp4" for GPT OSS, "" for standard BF16
    source_fp8_block_size: Optional[tuple[int, int]] = None
    swiglu_limit: float = 0.0         # GPT OSS: 7.0 — clamp SwiGLU output to [-limit, limit]
    swiglu_limits: Optional[List[float]] = None          # Step per-layer routed clamp
    swiglu_limits_shared: Optional[List[float]] = None   # Step per-layer shared clamp
    gemma4_text: bool = False          # Gemma4 text tower semantics
    step3_text: bool = False           # Step text tower semantics
    embedding_scale: float = 1.0       # Gemma text embeddings multiply by sqrt(hidden_size)
    final_logit_softcapping: float = 0.0

    # Nemotron-H (hybrid Mamba2 + MoE + Attention)
    model_type: str = ""               # e.g. "nemotron_h", "qwen3_next", etc.
    hybrid_override_pattern: str = ""  # e.g. "MEMEMEM*E..." — M=Mamba2, E=MoE, *=Attention
    mamba_num_heads: int = 0           # Mamba2 SSM heads (e.g. 128 for Super, 64 for Nano)
    mamba_head_dim: int = 0            # Mamba2 SSM per-head dim (e.g. 64)
    ssm_state_size: int = 0            # Mamba2 SSM state size (e.g. 128)
    mamba_expand: int = 0              # Mamba2 expansion factor (e.g. 2)
    mamba_conv_kernel: int = 0         # Mamba2 conv1d kernel size (e.g. 4)
    mamba_n_groups: int = 1            # Mamba2 number of groups for B/C (e.g. 8)
    mamba_chunk_size: int = 128        # Mamba2 SSD chunk size (must be power of 2)
    # Nemotron-H initialization metadata. Loaded checkpoint tensors must remain verbatim.
    rescale_prenorm_residual: bool = False
    moe_latent_size: int = 0           # LatentMoE: latent projection dim (e.g. 1024)
    moe_shared_expert_intermediate_size: int = 0  # LatentMoE shared expert intermediate
    mlp_hidden_act: str = "silu"       # MLP activation: "silu" or "relu2"

    # GQA output gating: q_proj outputs [query, gate], apply sigmoid(gate) before o_proj
    # Qwen3.5 calls this attn_output_gate, Qwen3Next uses it implicitly
    gated_attention: bool = False
    head_wise_attention_gate: bool = False  # Step separate g_proj attention gate

    # Norm convention: Qwen3NextRMSNorm uses (1 + weight) * x, stored weights are ~0
    # Standard RMSNorm uses weight * x, stored weights are ~1
    norm_bias_one: bool = False  # True for qwen3_next models

    # Misc
    tie_word_embeddings: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0
    extra_stop_token_ids: tuple = ()  # additional stop tokens (e.g. from array eos_token_id)

    # Tensor prefix in safetensors (auto-detected)
    layers_prefix: str = "language_model.model"

    @classmethod
    def from_model_path(cls, model_path: str) -> "ModelConfig":
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            raw = json.load(f)

        # generation_config.json has the authoritative eos_token_id list
        gen_cfg_path = os.path.join(model_path, "generation_config.json")
        gen_cfg = {}
        if os.path.exists(gen_cfg_path):
            with open(gen_cfg_path) as f:
                gen_cfg = json.load(f)

        # Some models nest config: Kimi K2.5 → text_config, DeepSeek-VL2 → language_config
        cfg = raw.get("text_config", raw.get("language_config", raw))

        # Infer missing fields from weight shapes (VL models with incomplete config)
        cfg = _infer_from_weights(model_path, cfg)
        cfg = _normalise_layers_block_type(cfg)

        # tie_word_embeddings may be at top level; infer from weight presence if not set
        tie_default = True
        if "tie_word_embeddings" not in cfg and "tie_word_embeddings" not in raw:
            # Check if lm_head.weight exists in safetensors index — if so, not tied
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    idx_data = json.load(f)
                for k in idx_data.get("weight_map", {}):
                    if "lm_head.weight" in k:
                        tie_default = False
                        break
        tie = cfg.get("tie_word_embeddings",
                       raw.get("tie_word_embeddings", tie_default))

        # Model architecture type
        arch = cfg.get("model_type", "")
        is_deepseek_v4 = arch == "deepseek_v4"
        is_glm5_next = arch == "glm5_next_text"
        # DeepSeek-V4 is neither legacy MLA nor GQA: it has a single latent KV
        # vector plus compressed/indexed sparse attention and a low-rank output.
        is_mla = "kv_lora_rank" in cfg and not is_deepseek_v4
        step3_text = arch in ("step3p5", "step3p7")

        # Handle first_k_dense_replace from either field or decoder_sparse_step
        if "first_k_dense_replace" in cfg:
            first_k_dense = int(cfg["first_k_dense_replace"] or 0)
        elif "decoder_sparse_step" in cfg:
            step = cfg["decoder_sparse_step"]
            first_k_dense = 0 if step <= 1 else step
        else:
            first_k_dense = 0

        # Hybrid model: compute layer_types
        full_attn_interval = cfg.get("full_attention_interval", 0)
        num_layers = cfg["num_hidden_layers"]
        index_topk = int(cfg.get("index_topk", 0) or 0)
        index_head_dim = int(cfg.get("index_head_dim", 0) or 0)
        index_n_heads = int(cfg.get("index_n_heads", 0) or 0)
        index_topk_freq = int(cfg.get("index_topk_freq", 0) or 0)
        index_skip_topk_offset = int(cfg.get("index_skip_topk_offset", 0) or 0)
        raw_indexer_types = cfg.get("indexer_types")
        if raw_indexer_types is None:
            indexer_types = None
        elif isinstance(raw_indexer_types, list):
            indexer_types = [str(value) for value in raw_indexer_types]
        else:
            raise ValueError("indexer_types must be an array when present")
        # GLM-MoE-DSA defines interleaved RoPE for its indexer projections.
        # Released checkpoints may omit this redundant architecture semantic.
        indexer_rope_interleave = bool(
            cfg.get("indexer_rope_interleave", arch == "glm_moe_dsa")
        )
        index_share_for_mtp_iteration = bool(
            cfg.get("index_share_for_mtp_iteration", False)
        )
        index_kpool = int(cfg.get("index_kpool", 0) or 0)
        index_kpool_compress = bool(cfg.get("index_kpool_compress", False))
        index_kpool_always_select_tail = bool(
            cfg.get("index_kpool_always_select_tail", False)
        )
        attention_head_dim = int(cfg.get("head_dim", 0) or 0)
        o_lora_rank = int(cfg.get("o_lora_rank", 0) or 0)
        o_groups = int(cfg.get("o_groups", 0) or 0)
        raw_compress_ratios = cfg.get("compress_ratios")
        if raw_compress_ratios is None:
            compress_ratios = None
        elif isinstance(raw_compress_ratios, list):
            if any(not isinstance(value, int) for value in raw_compress_ratios):
                raise ValueError("compress_ratios must contain only integers")
            compress_ratios = list(raw_compress_ratios)
        else:
            raise ValueError("compress_ratios must be an array when present")
        compress_rope_theta = float(cfg.get("compress_rope_theta", 0.0) or 0.0)
        num_hash_layers = int(cfg.get("num_hash_layers", 0) or 0)
        hc_mult = int(cfg.get("hc_mult", 0) or 0)
        hc_sinkhorn_iters = int(cfg.get("hc_sinkhorn_iters", 0) or 0)
        hc_eps = float(cfg.get("hc_eps", 0.0) or 0.0)
        expert_dtype = str(cfg.get("expert_dtype", "") or "")
        dspark_block_size = int(cfg.get("dspark_block_size", 0) or 0)
        dspark_noise_token_id = int(cfg.get("dspark_noise_token_id", 0) or 0)
        raw_dspark_targets = cfg.get("dspark_target_layer_ids")
        if raw_dspark_targets is None:
            dspark_target_layer_ids = None
        elif isinstance(raw_dspark_targets, list):
            if any(not isinstance(value, int) for value in raw_dspark_targets):
                raise ValueError("dspark_target_layer_ids must contain only integers")
            dspark_target_layer_ids = list(raw_dspark_targets)
        else:
            raise ValueError("dspark_target_layer_ids must be an array when present")
        dspark_markov_rank = int(cfg.get("dspark_markov_rank", 0) or 0)

        linear_attn_cfg = cfg.get("linear_attn_config", {}) or {}
        if not isinstance(linear_attn_cfg, dict):
            raise ValueError("linear_attn_config must be an object when present")
        linear_attention_family = (
            "kimi_delta_attention" if is_glm5_next else "gated_deltanet"
        )
        if is_glm5_next:
            # KDA declares one head geometry shared by Q, K, and V.  Keep this
            # nested contract isolated from GatedDeltaNet checkpoints, whose
            # key and value head counts (and dimensions) are independent.
            linear_num_key_heads = int(linear_attn_cfg.get("num_heads", 0) or 0)
            linear_num_value_heads = linear_num_key_heads
            linear_key_head_dim = int(linear_attn_cfg.get("head_dim", 0) or 0)
            linear_value_head_dim = linear_key_head_dim
        else:
            linear_num_key_heads = int(cfg.get("linear_num_key_heads", 16) or 0)
            linear_num_value_heads = int(cfg.get("linear_num_value_heads", 32) or 0)
            linear_key_head_dim = int(cfg.get("linear_key_head_dim", 128) or 0)
            linear_value_head_dim = int(cfg.get("linear_value_head_dim", 128) or 0)
        linear_conv_kernel = int(
            linear_attn_cfg.get(
                "short_conv_kernel_size", cfg.get("linear_conv_kernel_dim", 4)
            ) or 0
        )
        linear_gate_lower_bound = float(
            linear_attn_cfg.get("gate_lower_bound", 0.0) or 0.0
        )

        if is_deepseek_v4:
            required_positive = {
                "q_lora_rank": int(cfg.get("q_lora_rank", 0) or 0),
                "head_dim": attention_head_dim,
                "qk_rope_head_dim": int(cfg.get("qk_rope_head_dim", 0) or 0),
                "o_lora_rank": o_lora_rank,
                "o_groups": o_groups,
                "index_topk": index_topk,
                "index_head_dim": index_head_dim,
                "index_n_heads": index_n_heads,
                "sliding_window": int(cfg.get("sliding_window", 0) or 0),
                "compress_rope_theta": int(compress_rope_theta),
                "hc_mult": hc_mult,
                "hc_sinkhorn_iters": hc_sinkhorn_iters,
            }
            for field_name, value in required_positive.items():
                if value <= 0:
                    raise ValueError(
                        f"deepseek_v4 requires positive {field_name}, got {value}"
                    )
            rope_dim = required_positive["qk_rope_head_dim"]
            if attention_head_dim <= rope_dim:
                raise ValueError(
                    "deepseek_v4 head_dim must exceed qk_rope_head_dim"
                )
            if rope_dim % 2 != 0 or index_head_dim % 2 != 0:
                raise ValueError(
                    "deepseek_v4 qk_rope_head_dim and index_head_dim must be even"
                )
            num_heads = int(cfg["num_attention_heads"])
            if num_heads % o_groups != 0:
                raise ValueError(
                    f"deepseek_v4 num_attention_heads {num_heads} is not divisible "
                    f"by o_groups {o_groups}"
                )
            if compress_ratios is None or len(compress_ratios) < num_layers:
                actual_len = 0 if compress_ratios is None else len(compress_ratios)
                raise ValueError(
                    "deepseek_v4 compress_ratios length "
                    f"{actual_len} < num_hidden_layers {num_layers}"
                )
            invalid_ratios = [value for value in compress_ratios if value < 0]
            if invalid_ratios:
                raise ValueError("deepseek_v4 compress_ratios must be non-negative")
            if num_hash_layers < 0 or num_hash_layers > num_layers:
                raise ValueError(
                    f"deepseek_v4 num_hash_layers {num_hash_layers} outside "
                    f"[0, {num_layers}]"
                )
            if hc_eps <= 0.0:
                raise ValueError(f"deepseek_v4 requires positive hc_eps, got {hc_eps}")
            if expert_dtype != "fp4":
                raise ValueError(
                    f"deepseek_v4 requires expert_dtype='fp4', got {expert_dtype!r}"
                )
            if cfg.get("scoring_func") != "sqrtsoftplus":
                raise ValueError("deepseek_v4 requires scoring_func='sqrtsoftplus'")
            if cfg.get("topk_method") != "noaux_tc":
                raise ValueError("deepseek_v4 requires topk_method='noaux_tc'")
            if int(cfg.get("n_shared_experts", 0) or 0) != 1:
                raise ValueError("deepseek_v4 requires exactly one shared expert")
            if dspark_target_layer_ids is not None:
                for layer_idx in dspark_target_layer_ids:
                    if layer_idx < 0 or layer_idx >= num_layers:
                        raise ValueError(
                            "deepseek_v4 dspark_target_layer_ids contains "
                            f"out-of-range layer {layer_idx}"
                        )
                if dspark_target_layer_ids and (
                    dspark_block_size <= 0
                    or dspark_markov_rank <= 0
                    or not 0 <= dspark_noise_token_id < int(cfg["vocab_size"])
                ):
                    raise ValueError(
                        "deepseek_v4 DSpark metadata is incomplete or invalid"
                    )
        if arch == "glm_moe_dsa":
            if not indexer_rope_interleave:
                raise ValueError(
                    "glm_moe_dsa indexer requires interleaved RoPE"
                )
            required_positive = {
                "q_lora_rank": int(cfg.get("q_lora_rank", 0) or 0),
                "kv_lora_rank": int(cfg.get("kv_lora_rank", 0) or 0),
                "qk_nope_head_dim": int(
                    cfg.get("qk_nope_head_dim", 0) or 0
                ),
                "qk_rope_head_dim": int(
                    cfg.get("qk_rope_head_dim", 0) or 0
                ),
                "v_head_dim": int(cfg.get("v_head_dim", 0) or 0),
                "index_topk": index_topk,
                "index_head_dim": index_head_dim,
                "index_n_heads": index_n_heads,
                "index_topk_freq": index_topk_freq,
            }
            for field_name, value in required_positive.items():
                if value <= 0:
                    raise ValueError(
                        f"{arch} requires positive {field_name}, got {value}"
                    )
            qk_rope_head_dim = required_positive["qk_rope_head_dim"]
            if index_head_dim < qk_rope_head_dim:
                raise ValueError(
                    "glm_moe_dsa index_head_dim "
                    f"{index_head_dim} is smaller than qk_rope_head_dim "
                    f"{qk_rope_head_dim}"
                )
            if index_head_dim % 2 != 0 or qk_rope_head_dim % 2 != 0:
                raise ValueError(
                    "glm_moe_dsa index_head_dim and qk_rope_head_dim "
                    "must both be even for RoPE"
                )
            if index_skip_topk_offset < 0:
                raise ValueError(
                    "glm_moe_dsa index_skip_topk_offset must be non-negative"
                )
            if indexer_types is None or len(indexer_types) != num_layers:
                actual_len = 0 if indexer_types is None else len(indexer_types)
                raise ValueError(
                    "glm_moe_dsa indexer_types length "
                    f"{actual_len} != num_hidden_layers {num_layers}"
                )
            invalid_indexer_types = sorted(
                {value for value in indexer_types if value not in ("full", "shared")}
            )
            if invalid_indexer_types:
                raise ValueError(
                    "glm_moe_dsa indexer_types contains unsupported values: "
                    + ", ".join(invalid_indexer_types)
                )
            full_index_seen = False
            for layer_idx, indexer_type in enumerate(indexer_types):
                if indexer_type == "full":
                    full_index_seen = True
                elif not full_index_seen:
                    raise ValueError(
                        "glm_moe_dsa shared indexer at layer "
                        f"{layer_idx} has no preceding full indexer"
                    )
        if is_glm5_next:
            required_positive = {
                "q_lora_rank": int(cfg.get("q_lora_rank", 0) or 0),
                "kv_lora_rank": int(cfg.get("kv_lora_rank", 0) or 0),
                "qk_nope_head_dim": int(cfg.get("qk_nope_head_dim", 0) or 0),
                "v_head_dim": int(cfg.get("v_head_dim", 0) or 0),
                "index_topk": index_topk,
                "index_head_dim": index_head_dim,
                "index_n_heads": index_n_heads,
                "index_kpool": index_kpool,
                "hc_mult": hc_mult,
                "hc_sinkhorn_iters": hc_sinkhorn_iters,
                "linear_attn_config.num_heads": linear_num_key_heads,
                "linear_attn_config.head_dim": linear_key_head_dim,
                "linear_attn_config.short_conv_kernel_size": linear_conv_kernel,
            }
            for field_name, value in required_positive.items():
                if value <= 0:
                    raise ValueError(
                        f"glm5_next_text requires positive {field_name}, got {value}"
                    )
            if int(cfg.get("qk_rope_head_dim", -1)) != 0:
                raise ValueError("glm5_next_text requires qk_rope_head_dim=0")
            if not bool(cfg.get("mhc", False)) or hc_eps <= 0.0:
                raise ValueError("glm5_next_text requires validated mHC parameters")
            if not index_kpool_compress or not index_kpool_always_select_tail:
                raise ValueError(
                    "glm5_next_text requires compressed index_kpool with visible-tail selection"
                )
            if index_topk % index_kpool != 0:
                raise ValueError(
                    f"glm5_next_text index_topk {index_topk} is not divisible by "
                    f"index_kpool {index_kpool}"
                )
            if indexer_types is None or len(indexer_types) != num_layers:
                actual_len = 0 if indexer_types is None else len(indexer_types)
                raise ValueError(
                    "glm5_next_text indexer_types length "
                    f"{actual_len} != num_hidden_layers {num_layers}"
                )
            invalid_indexers = sorted(set(indexer_types) - {"full", "shared"})
            if invalid_indexers:
                raise ValueError(
                    "glm5_next_text indexer_types contains unsupported values: "
                    + ", ".join(invalid_indexers)
                )
        layer_types = None
        moe_layer_indices = _parse_int_list(cfg.get("moe_layers_enum"), "moe_layers_enum", max_len=num_layers)
        raw_mlp_layer_types = cfg.get("mlp_layer_types")
        if raw_mlp_layer_types is not None:
            if (
                not isinstance(raw_mlp_layer_types, list)
                or len(raw_mlp_layer_types) != num_layers
            ):
                actual_len = (
                    0 if not isinstance(raw_mlp_layer_types, list)
                    else len(raw_mlp_layer_types)
                )
                raise ValueError(
                    f"mlp_layer_types length {actual_len} != num_hidden_layers {num_layers}"
                )
            invalid_mlp_types = sorted(
                set(raw_mlp_layer_types) - {"dense", "sparse"}
            )
            if invalid_mlp_types:
                raise ValueError(
                    "mlp_layer_types contains unsupported values: "
                    + ", ".join(invalid_mlp_types)
                )
            mlp_moe_layers = [
                i for i, value in enumerate(raw_mlp_layer_types)
                if value == "sparse"
            ]
            if moe_layer_indices is not None and moe_layer_indices != mlp_moe_layers:
                raise ValueError("moe_layers_enum disagrees with mlp_layer_types")
            moe_layer_indices = mlp_moe_layers
        if moe_layer_indices:
            first_k_dense = min(moe_layer_indices)
        hybrid_pattern = cfg.get("hybrid_override_pattern", "")
        layers_block_type = cfg.get("layers_block_type")
        if hybrid_pattern and arch == "nemotron_h":
            # Nemotron-H: parse M=mamba2, E=moe, *=attention from pattern
            type_map = {"M": "mamba2", "E": "moe", "*": "full_attention"}
            layer_types = [type_map.get(c, "full_attention") for c in hybrid_pattern]
            assert len(layer_types) == num_layers, (
                f"hybrid_override_pattern length {len(layer_types)} != num_hidden_layers {num_layers}")
        elif layers_block_type and arch == "nemotron_h":
            # Nemotron-H Ultra: explicit per-block labels.
            type_map = {
                "mamba": "mamba2",
                "mamba2": "mamba2",
                "moe": "moe",
                "attention": "full_attention",
                "full_attention": "full_attention",
            }
            layer_types = []
            for idx, label in enumerate(layers_block_type):
                if label not in type_map:
                    raise ValueError(f"Unsupported layers_block_type[{idx}]={label!r}")
                layer_types.append(type_map[label])
            assert len(layer_types) == num_layers, (
                f"layers_block_type length {len(layer_types)} != num_hidden_layers {num_layers}")
        elif "layer_types" in cfg:
            # GPT OSS: explicit layer_types array (sliding_attention / full_attention)
            layer_types = list(cfg["layer_types"])
            if len(layer_types) > num_layers and step3_text:
                # Step includes MTP/extra layers after the normal CausalLM text stack.
                layer_types = layer_types[:num_layers]
            if len(layer_types) < num_layers:
                raise ValueError(f"layer_types length {len(layer_types)} < num_hidden_layers {num_layers}")
            if is_glm5_next:
                allowed = {"linear_attention", "deepseek_sparse_attention"}
                invalid = sorted(set(layer_types) - allowed)
                if invalid:
                    raise ValueError(
                        "glm5_next_text layer_types contains unsupported values: "
                        + ", ".join(invalid)
                    )
                kda_layers = linear_attn_cfg.get("kda_layers")
                dsa_layers = linear_attn_cfg.get("full_attn_layers")
                if not isinstance(kda_layers, list) or not isinstance(dsa_layers, list):
                    raise ValueError(
                        "glm5_next_text linear_attn_config must declare "
                        "kda_layers and full_attn_layers"
                    )
                observed_kda = [
                    i for i, value in enumerate(layer_types)
                    if value == "linear_attention"
                ]
                observed_dsa = [
                    i for i, value in enumerate(layer_types)
                    if value == "deepseek_sparse_attention"
                ]
                if kda_layers != observed_kda or dsa_layers != observed_dsa:
                    raise ValueError(
                        "glm5_next_text layer_types disagree with "
                        "linear_attn_config schedules"
                    )
        elif full_attn_interval > 0:
            # Qwen3-Next: compute from full_attention_interval
            layer_types = [
                "full_attention" if (i + 1) % full_attn_interval == 0
                else "linear_attention"
                for i in range(num_layers)
            ]

        # Norm convention: Qwen3NextRMSNorm uses (1 + weight) * x
        # with weight initialized to zeros, while standard models use weight * x
        # with weight initialized to ones. We add 1.0 to stored weights at load time.
        norm_bias_one = arch in ("qwen3_next", "qwen3_5_moe_text") or step3_text

        # Gated attention: q_proj outputs [query, gate], apply sigmoid(gate) before o_proj
        # Qwen3.5 uses explicit attn_output_gate flag; Qwen3Next always uses it
        gated_attention = cfg.get("attn_output_gate", arch in ("qwen3_next", "qwen3_5_moe_text"))

        # Nemotron-H and Step use separate shared expert field names.
        nemotron_shared_inter = cfg.get("moe_shared_expert_intermediate_size", 0)
        step_shared_inter = cfg.get("share_expert_dim", 0)

        # Shared experts: n_shared_experts or infer from shared_expert_intermediate_size
        n_shared = cfg.get("n_shared_experts", 0)
        shared_inter = cfg.get("shared_expert_intermediate_size", nemotron_shared_inter or step_shared_inter)
        if n_shared == 0 and shared_inter > 0:
            n_shared = 1  # infer single shared expert

        # Expert count: n_routed_experts (DeepSeek) / num_experts (Qwen3) /
        # num_local_experts (GPT OSS/Nemotron) / moe_num_experts (Step).
        n_experts = cfg.get("n_routed_experts",
                           cfg.get("num_experts",
                                  cfg.get("num_local_experts",
                                          cfg.get("moe_num_experts", 0))))
        # Experts per token: num_experts_per_tok (DeepSeek/Qwen3) /
        # experts_per_token (GPT OSS) / moe_top_k (Step).
        experts_per_tok = cfg.get("num_experts_per_tok",
                                 cfg.get("experts_per_token",
                                         cfg.get("top_k_experts",
                                                 cfg.get("moe_top_k", 0))))

        # MoE intermediate size: moe_intermediate_size (Qwen3) / intermediate_size (GPT OSS)
        moe_inter = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 0))

        # Sliding window (GPT OSS: 128 tokens for sliding_attention layers)
        sliding_window = cfg.get("sliding_window", 0)

        # Pre-quantized expert format (GPT OSS uses MXFP4)
        quant_config = cfg.get("quantization_config", raw.get("quantization_config", {}))
        raw_fp8_block_size = quant_config.get("weight_block_size")
        source_fp8_block_size = None
        if raw_fp8_block_size is not None:
            if (
                not isinstance(raw_fp8_block_size, list)
                or len(raw_fp8_block_size) != 2
                or any(
                    not isinstance(value, int) or value <= 0
                    for value in raw_fp8_block_size
                )
            ):
                raise ValueError(
                    "quantization_config.weight_block_size must contain two "
                    "positive integers"
                )
            source_fp8_block_size = tuple(raw_fp8_block_size)
        expert_quant_method = (
            expert_dtype if is_deepseek_v4 else quant_config.get("quant_method", "")
        )

        # SwiGLU activation limit (GPT OSS scalar; Step has per-layer arrays).
        swiglu_limit = cfg.get("swiglu_limit", 0.0)
        swiglu_limits = _parse_float_list(cfg.get("swiglu_limits"), "swiglu_limits", max_len=num_layers)
        swiglu_limits_shared = _parse_float_list(
            cfg.get("swiglu_limits_shared"), "swiglu_limits_shared", max_len=num_layers
        )

        # RoPE: some models (Qwen3.5) nest rope_theta/partial_rotary_factor inside rope_parameters
        rope_params = cfg.get("rope_parameters", {}) or {}
        if arch == "gemma4_text" and isinstance(rope_params, dict):
            rope_default = rope_params.get("sliding_attention", {})
        else:
            rope_default = rope_params
        raw_rope_theta = cfg.get("rope_theta", rope_params.get("rope_theta", 10000.0))
        rope_theta_layers = _parse_float_list(raw_rope_theta, "rope_theta", max_len=num_layers)
        rope_theta = rope_theta_layers[0] if rope_theta_layers else raw_rope_theta
        if arch == "gemma4_text":
            rope_theta = cfg.get("rope_theta", rope_default.get("rope_theta", 10000.0))
            partial_rotary = rope_default.get("partial_rotary_factor", 1.0)
            partial_rotary_factors = None
        else:
            raw_partial_rotary = cfg.get("partial_rotary_factor",
                                         cfg.get("partial_rotary_factors",
                                                 rope_params.get("partial_rotary_factor", 1.0)))
            partial_rotary_factors = _parse_float_list(
                raw_partial_rotary, "partial_rotary_factors", max_len=num_layers
            )
            partial_rotary = partial_rotary_factors[0] if partial_rotary_factors else raw_partial_rotary
        raw_yarn_only_types = cfg.get("yarn_only_types")
        yarn_only_types = list(raw_yarn_only_types) if isinstance(raw_yarn_only_types, list) else None

        attention_other = cfg.get("attention_other_setting", {}) or {}
        other_heads = int(attention_other.get("num_attention_heads", 0) or 0)
        other_kv_heads = int(
            attention_other.get("num_key_value_heads",
                                attention_other.get("num_attention_groups", 0)) or 0
        )
        other_head_dim = int(attention_other.get("head_dim", 0) or 0)
        base_kv_heads = int(cfg.get("num_key_value_heads",
                                    cfg.get("num_attention_groups", cfg["num_attention_heads"])))
        router_activation = cfg.get("moe_router_activation", "")
        scoring_func = cfg.get("scoring_func")
        if scoring_func is None:
            scoring_func = "sigmoid" if arch == "nemotron_h" or router_activation == "sigmoid" else "softmax"
        rms_norm_eps = cfg.get("rms_norm_eps")
        if rms_norm_eps is None:
            rms_norm_eps = cfg.get("norm_eps", cfg.get("layer_norm_epsilon"))
        if rms_norm_eps is None:
            # Step ships rms_norm_eps as null in config.json, but its local
            # configuration class defaults this field to 1e-5.
            rms_norm_eps = 1e-5 if step3_text else 1e-6

        return cls(
            model_path=model_path,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg.get("intermediate_size") or cfg.get("moe_intermediate_size", 0),
            moe_intermediate_size=moe_inter,
            num_hidden_layers=num_layers,
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=base_kv_heads,
            vocab_size=cfg["vocab_size"],
            # MLA fields (None for GQA)
            q_lora_rank=cfg.get("q_lora_rank") if (is_mla or is_deepseek_v4) else None,
            kv_lora_rank=cfg.get("kv_lora_rank") if is_mla else None,
            qk_nope_head_dim=cfg.get("qk_nope_head_dim") if is_mla else None,
            qk_rope_head_dim=cfg.get("qk_rope_head_dim") if (is_mla or is_deepseek_v4) else None,
            v_head_dim=cfg.get("v_head_dim") if is_mla else None,
            # DSA / IndexShare fields
            index_topk=index_topk,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk_freq=index_topk_freq,
            index_skip_topk_offset=index_skip_topk_offset,
            indexer_types=indexer_types,
            indexer_rope_interleave=indexer_rope_interleave,
            index_share_for_mtp_iteration=index_share_for_mtp_iteration,
            index_kpool=index_kpool,
            index_kpool_compress=index_kpool_compress,
            index_kpool_always_select_tail=index_kpool_always_select_tail,
            attention_head_dim=attention_head_dim,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
            compress_ratios=compress_ratios,
            compress_rope_theta=compress_rope_theta,
            num_hash_layers=num_hash_layers,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            hc_eps=hc_eps,
            expert_dtype=expert_dtype,
            dspark_block_size=dspark_block_size,
            dspark_noise_token_id=dspark_noise_token_id,
            dspark_target_layer_ids=dspark_target_layer_ids,
            dspark_markov_rank=dspark_markov_rank,
            # GQA fields (None for MLA)
            gqa_head_dim=cfg.get("head_dim") if not (is_mla or is_deepseek_v4) else None,
            global_head_dim=cfg.get("global_head_dim", 0),
            num_global_key_value_heads=cfg.get("num_global_key_value_heads", 0),
            attention_k_eq_v=bool(cfg.get("attention_k_eq_v", False)),
            gqa_other_num_attention_heads=other_heads,
            gqa_other_num_key_value_heads=other_kv_heads,
            gqa_other_head_dim=other_head_dim,
            # Hybrid model
            full_attention_interval=full_attn_interval,
            layer_types=layer_types,
            linear_conv_kernel_dim=linear_conv_kernel,
            linear_key_head_dim=linear_key_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_value_heads=linear_num_value_heads,
            linear_attention_family=linear_attention_family,
            linear_gate_lower_bound=linear_gate_lower_bound,
            # MoE
            n_routed_experts=n_experts,
            num_experts_per_tok=experts_per_tok,
            n_shared_experts=n_shared,
            shared_expert_intermediate_size=shared_inter,
            first_k_dense_replace=first_k_dense,
            routed_scaling_factor=cfg.get("routed_scaling_factor",
                                          cfg.get("moe_router_scaling_factor", 1.0)),
            scoring_func=cfg.get("scoring_func",
                                 scoring_func),
            topk_method=cfg.get("topk_method", "greedy"),
            norm_topk_prob=cfg.get("norm_topk_prob", cfg.get("norm_expert_weight", True)),
            moe_layer_indices=moe_layer_indices,
            use_moe_router_bias=bool(cfg.get("use_moe_router_bias", False)),
            need_fp32_gate=bool(cfg.get("need_fp32_gate", False)),
            rms_norm_eps=float(rms_norm_eps),
            hidden_act=cfg.get("hidden_activation", cfg.get("hidden_act", "silu")),
            rope_theta=rope_theta,
            rope_scaling=cfg.get("rope_scaling") or rope_params or {},
            max_position_embeddings=cfg.get("max_position_embeddings", 131072),
            partial_rotary_factor=partial_rotary,
            partial_rotary_factors=partial_rotary_factors,
            rope_theta_layers=rope_theta_layers,
            yarn_only_types=yarn_only_types,
            rope_interleave=cfg.get("rope_interleave", True),
            attention_bias=cfg.get("attention_bias", False),
            sliding_window=sliding_window,
            expert_quant_method=expert_quant_method,
            source_fp8_block_size=source_fp8_block_size,
            swiglu_limit=swiglu_limit,
            swiglu_limits=swiglu_limits,
            swiglu_limits_shared=swiglu_limits_shared,
            gemma4_text=arch == "gemma4_text",
            step3_text=step3_text,
            embedding_scale=(cfg["hidden_size"] ** 0.5) if arch == "gemma4_text" else 1.0,
            final_logit_softcapping=float(cfg.get("final_logit_softcapping") or 0.0),
            # Nemotron-H fields
            model_type=arch,
            hybrid_override_pattern=hybrid_pattern,
            mamba_num_heads=cfg.get("mamba_num_heads", 0),
            mamba_head_dim=cfg.get("mamba_head_dim", 0),
            ssm_state_size=cfg.get("ssm_state_size", 0),
            mamba_expand=cfg.get("expand", 0),
            mamba_conv_kernel=cfg.get("conv_kernel", 0),
            mamba_n_groups=cfg.get("n_groups", 1),
            mamba_chunk_size=cfg.get("chunk_size", 128),
            rescale_prenorm_residual=bool(cfg.get("rescale_prenorm_residual", False)),
            moe_latent_size=cfg.get("moe_latent_size", 0),
            moe_shared_expert_intermediate_size=nemotron_shared_inter,
            mlp_hidden_act=cfg.get("mlp_hidden_act",
                                   cfg.get("hidden_activation", cfg.get("hidden_act", "silu"))),
            gated_attention=gated_attention,
            head_wise_attention_gate=bool(cfg.get("use_head_wise_attn_gate", False)),
            norm_bias_one=norm_bias_one,
            tie_word_embeddings=tie,
            bos_token_id=raw.get("bos_token_id", cfg.get("bos_token_id", 0)),
            eos_token_id=_parse_eos_token_id(raw, cfg, gen_cfg),
            extra_stop_token_ids=_parse_extra_stop_ids(raw, cfg, gen_cfg),
            layers_prefix=_detect_layers_prefix(model_path),
        )

    @property
    def attention_type(self) -> str:
        """Runtime attention family selected by the model architecture."""
        if self.is_deepseek_v4:
            return "deepseek_v4"
        if self.is_glm5_next:
            return "hybrid_kda_dsa"
        return "mla" if self.kv_lora_rank is not None else "gqa"

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def is_gqa(self) -> bool:
        return (
            self.kv_lora_rank is None
            and not self.is_deepseek_v4
            and not self.is_glm5_next
        )

    @property
    def is_deepseek_v4(self) -> bool:
        return self.model_type == "deepseek_v4"

    @property
    def is_glm5_next(self) -> bool:
        return self.model_type == "glm5_next_text"

    @property
    def has_hyper_connection(self) -> bool:
        return self.is_deepseek_v4 or self.is_glm5_next

    @property
    def is_dsa(self) -> bool:
        """True when the model uses DSA sparse attention and IndexShare."""
        return self.model_type in ("glm_moe_dsa", "glm5_next_text")

    def is_dsa_layer(self, layer_idx: int) -> bool:
        """True only for layers whose attention is DSA."""
        if not self.is_dsa:
            return False
        if self.is_glm5_next:
            return self.layer_types[layer_idx] == "deepseek_sparse_attention"
        return True

    def dsa_indexer_owner_layer(self, layer_idx: int) -> Optional[int]:
        """Return the full-indexer layer that owns this layer's IndexShare state."""
        if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
            raise IndexError(
                f"layer_idx {layer_idx} outside [0, {self.num_hidden_layers})"
            )
        if not self.is_dsa:
            return None
        if not self.is_dsa_layer(layer_idx):
            return None
        if self.indexer_types is None:
            raise RuntimeError("DSA model has no validated indexer_types schedule")
        for owner_idx in range(layer_idx, -1, -1):
            if (
                self.is_dsa_layer(owner_idx)
                and self.indexer_types[owner_idx] == "full"
            ):
                return owner_idx
        raise RuntimeError(
            f"DSA layer {layer_idx} has no full indexer owner in its schedule"
        )

    def is_dsa_indexer_owner_layer(self, layer_idx: int) -> bool:
        """True when this layer owns indexer weights instead of sharing them."""
        owner_idx = self.dsa_indexer_owner_layer(layer_idx)
        return owner_idx is not None and owner_idx == layer_idx

    @property
    def num_moe_layers(self) -> int:
        if self.moe_layer_indices is not None:
            return len(self.moe_layer_indices)
        if self.hybrid_override_pattern:
            return self.hybrid_override_pattern.count('E')
        return self.num_hidden_layers - self.first_k_dense_replace

    @property
    def head_dim(self) -> int:
        """Full head dim for q: nope + rope (MLA) or gqa_head_dim (GQA)."""
        if self.is_deepseek_v4:
            return self.attention_head_dim
        if self.is_mla:
            return self.qk_nope_head_dim + self.qk_rope_head_dim
        return self.gqa_head_dim

    def gqa_head_dim_for_layer(self, layer_idx: int) -> int:
        if self.step3_text and self.is_sliding_attention_layer(layer_idx) and self.gqa_other_head_dim:
            return self.gqa_other_head_dim
        if self.gemma4_text and self.is_full_attention_layer(layer_idx) and not self.is_sliding_attention_layer(layer_idx):
            return self.global_head_dim or self.gqa_head_dim
        return self.gqa_head_dim

    def gqa_num_heads_for_layer(self, layer_idx: int) -> int:
        if self.step3_text and self.is_sliding_attention_layer(layer_idx) and self.gqa_other_num_attention_heads:
            return self.gqa_other_num_attention_heads
        return self.num_attention_heads

    def gqa_num_kv_heads_for_layer(self, layer_idx: int) -> int:
        if self.step3_text and self.is_sliding_attention_layer(layer_idx) and self.gqa_other_num_key_value_heads:
            return self.gqa_other_num_key_value_heads
        if self.gemma4_text and self.is_full_attention_layer(layer_idx) and not self.is_sliding_attention_layer(layer_idx):
            return self.num_global_key_value_heads or self.num_key_value_heads
        return self.num_key_value_heads

    def gqa_has_v_proj_for_layer(self, layer_idx: int) -> bool:
        return not (
            self.gemma4_text
            and self.attention_k_eq_v
            and self.is_full_attention_layer(layer_idx)
            and not self.is_sliding_attention_layer(layer_idx)
        )

    def rope_theta_for_layer(self, layer_idx: int) -> float:
        if self.rope_theta_layers is not None:
            return float(self.rope_theta_layers[layer_idx])
        if self.gemma4_text and isinstance(self.rope_scaling, dict):
            key = "sliding_attention" if self.is_sliding_attention_layer(layer_idx) else "full_attention"
            params = self.rope_scaling.get(key, {})
            if isinstance(params, dict):
                return float(params.get("rope_theta", self.rope_theta))
        return self.rope_theta

    def rotary_dim_for_layer(self, layer_idx: int) -> int:
        head_dim = self.gqa_head_dim_for_layer(layer_idx)
        if self.partial_rotary_factors is not None:
            return int(head_dim * float(self.partial_rotary_factors[layer_idx]))
        if self.gemma4_text and isinstance(self.rope_scaling, dict):
            key = "sliding_attention" if self.is_sliding_attention_layer(layer_idx) else "full_attention"
            params = self.rope_scaling.get(key, {})
            if isinstance(params, dict):
                return int(head_dim * float(params.get("partial_rotary_factor", 1.0)))
        return int(head_dim * self.partial_rotary_factor)

    @property
    def q_head_dim(self) -> int:
        """Total query head dim."""
        return self.head_dim

    @property
    def rotary_dim(self) -> int:
        """Number of head dimensions that get RoPE (partial_rotary_factor * head_dim)."""
        if self.is_deepseek_v4:
            return self.qk_rope_head_dim
        if self.is_mla:
            return self.qk_rope_head_dim
        return int(self.gqa_head_dim * self.partial_rotary_factor)

    def tensor_name(self, suffix: str) -> str:
        """Join a checkpoint tensor suffix to an optional root namespace."""
        return f"{self.layers_prefix}.{suffix}" if self.layers_prefix else suffix

    def layer_tensor_prefix(self, layer_idx: int) -> str:
        """Return the checkpoint prefix for a main-model layer."""
        if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
            raise IndexError(
                f"layer_idx {layer_idx} outside [0, {self.num_hidden_layers})"
            )
        return self.tensor_name(f"layers.{layer_idx}")

    @property
    def kv_compressed_dim(self) -> int:
        """Compressed KV dimension stored in cache (MLA only)."""
        assert self.is_mla, "kv_compressed_dim only valid for MLA models"
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def has_q_lora(self) -> bool:
        """Whether this model uses q_a_proj + q_b_proj (True) or direct q_proj (False)."""
        return self.q_lora_rank is not None and self.q_lora_rank > 0

    @property
    def is_nemotron_h(self) -> bool:
        """True for Nemotron-H hybrid models (Mamba2 + MoE + Attention)."""
        return self.model_type == "nemotron_h"

    @property
    def is_hybrid(self) -> bool:
        """True if model has a mix of layer types (LA+GQA, or Mamba2+MoE+Attention)."""
        return self.layer_types is not None

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """True if this layer uses linear attention (Gated DeltaNet)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "linear_attention"

    def is_kimi_delta_attention_layer(self, layer_idx: int) -> bool:
        """True for GLM-5.3 Kimi Delta Attention layers."""
        return self.is_glm5_next and self.is_linear_attention_layer(layer_idx)

    def is_mamba2_layer(self, layer_idx: int) -> bool:
        """True if this layer uses Mamba2 SSM (Nemotron-H)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "mamba2"

    def is_moe_only_layer(self, layer_idx: int) -> bool:
        """True if this layer is MoE-only (no attention, Nemotron-H 'E' layers)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "moe"

    def is_sliding_attention_layer(self, layer_idx: int) -> bool:
        """True if this layer uses sliding window attention (GPT OSS)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "sliding_attention"

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """True if this layer uses standard full attention (GQA/MLA)."""
        if self.layer_types is None:
            return True
        return self.layer_types[layer_idx] in (
            "full_attention",
            "sliding_attention",
            "deepseek_sparse_attention",
        )

    @property
    def num_full_attention_layers(self) -> int:
        """Number of layers that need KV cache (full + sliding attention)."""
        if self.layer_types is None:
            return self.num_hidden_layers
        return sum(
            1
            for layer_type in self.layer_types
            if layer_type in (
                "full_attention",
                "sliding_attention",
                "deepseek_sparse_attention",
            )
        )

    @property
    def effective_shared_expert_intermediate(self) -> int:
        """Shared expert intermediate size, handling both naming conventions."""
        if self.shared_expert_intermediate_size > 0:
            return self.shared_expert_intermediate_size
        if self.n_shared_experts > 0:
            return self.n_shared_experts * self.moe_intermediate_size
        return 0

    @property
    def mamba_d_inner(self) -> int:
        """Mamba2 inner dimension = num_heads * head_dim."""
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int:
        """Mamba2 conv1d dimension = d_inner + 2 * n_groups * state_size."""
        return self.mamba_d_inner + 2 * self.mamba_n_groups * self.ssm_state_size

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.moe_layer_indices is not None:
            return layer_idx in self.moe_layer_indices
        if self.is_nemotron_h:
            # Nemotron: only MoE layers have experts. Attention and Mamba2 layers do not.
            return self.layer_types[layer_idx] == "moe"
        if self.n_routed_experts <= 0:
            return False
        return layer_idx >= self.first_k_dense_replace

    def swiglu_limit_for_layer(self, layer_idx: int) -> float:
        if self.swiglu_limits is not None:
            return float(self.swiglu_limits[layer_idx])
        return float(self.swiglu_limit)

    def shared_swiglu_limit_for_layer(self, layer_idx: int) -> float:
        if self.swiglu_limits_shared is not None:
            return float(self.swiglu_limits_shared[layer_idx])
        return self.swiglu_limit_for_layer(layer_idx)


def compute_pp_partition(
    num_layers: int,
    num_gpus: int,
) -> List[int]:
    """Compute PP partition — always PP=1 (all layers on primary GPU).

    Multi-GPU uses Expert Parallelism (EP) instead of Pipeline Parallelism.
    PP>1 (splitting layers across GPUs) is not supported as it provides
    zero parallelism (sequential pipeline with idle GPUs).
    """
    return [num_layers]


@dataclass
class PPRankConfig:
    """Per-rank configuration for pipeline parallelism."""
    rank: int
    device: str                  # e.g. "cuda:0"
    layer_start: int             # first layer index (absolute)
    layer_end: int               # exclusive end
    num_layers: int
    has_embedding: bool
    has_lm_head: bool

    @property
    def layer_range(self) -> range:
        return range(self.layer_start, self.layer_end)


def build_pp_ranks(
    cfg: ModelConfig,
    pp_partition: List[int],
    devices: Optional[List[str]] = None,
) -> List[PPRankConfig]:
    """Build per-rank configs from PP partition."""
    num_ranks = len(pp_partition)
    if devices is None:
        devices = [f"cuda:{i}" for i in range(num_ranks)]

    ranks = []
    offset = 0
    for i, count in enumerate(pp_partition):
        ranks.append(PPRankConfig(
            rank=i,
            device=devices[i],
            layer_start=offset,
            layer_end=offset + count,
            num_layers=count,
            has_embedding=(i == 0),
            has_lm_head=(i == num_ranks - 1),
        ))
        offset += count
    return ranks
