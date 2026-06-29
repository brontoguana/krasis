#!/usr/bin/env python3
"""Generate reference greedy decode outputs for deterministic model validation.

Loads a model via HuggingFace transformers in BF16, runs the sanity prompts
with greedy decoding (do_sample=False), and stores output token IDs and text
as JSON. This provides the ground truth for validating Krasis decode correctness.

Usage:
    ./dev generate-reference <model-name> [--max-tokens N]

This script must be run via ./dev generate-reference, not directly.
"""

import argparse
import hashlib
import inspect
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import re
import shlex
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from reference_contract import (
        REFERENCE_PROFILES,
        apply_capture_template,
        build_contract,
        build_reference_sanity_report,
        canonical_profile_id,
        collect_capture_stop_ids,
        capture_settings_for_profile,
        emit_reference_generation_trace,
        load_tokenizer_with_compat,
        profile_filename,
    )
except ModuleNotFoundError:
    from tests.reference_contract import (
        REFERENCE_PROFILES,
        apply_capture_template,
        build_contract,
        build_reference_sanity_report,
        canonical_profile_id,
        collect_capture_stop_ids,
        capture_settings_for_profile,
        emit_reference_generation_trace,
        load_tokenizer_with_compat,
        profile_filename,
    )

# Guard: must be run via ./dev
if not os.environ.get("KRASIS_DEV_SCRIPT"):
    print("ERROR: This script must be run via ./dev generate-reference, not directly.")
    print("  Usage: ./dev generate-reference <model-name>")
    sys.exit(1)

MODELS_DIR = os.environ.get("KRASIS_REFERENCE_CAPTURE_MODELS_DIR", os.path.expanduser("~/.krasis/models"))
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
REFERENCE_DIR = SCRIPT_DIR / "reference_outputs"
PROMPTS_FILE = REPO_DIR / "benchmarks" / "sanity_test_prompts.txt"
DEFAULT_DIAGNOSTIC_STEPS = 4
DEFAULT_DIAGNOSTIC_TOPK = 10

MODEL_ALIASES = {
    "qcn": "Qwen3-Coder-Next",
    "qwen3-coder-next": "Qwen3-Coder-Next",
    "qwen35": "Qwen3.5-35B-A3B",
    "q35b": "Qwen3.5-35B-A3B",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
    "q122b": "Qwen3.5-122B-A10B",
    "qwen122": "Qwen3.5-122B-A10B",
    "qwen3.5-122b-a10b": "Qwen3.5-122B-A10B",
    "gemma": "gemma-4-26B-A4B-it",
    "gemma26": "gemma-4-26B-A4B-it",
    "gemma-4-26b-a4b-it": "gemma-4-26B-A4B-it",
    "minimax": "MiniMax-M2.5",
    "minimax25": "MiniMax-M2.5",
    "minimax-m2.5": "MiniMax-M2.5",
    "nemotron-nano": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotronnano": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia-nemotron-3-nano-30b-a3b-bf16": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotron-super": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nemotronsuper": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia-nemotron-3-super-120b-a12b-bf16": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "q235": "Qwen3-235B-A22B",
    "qwen235": "Qwen3-235B-A22B",
    "qwen3-235b-a22b": "Qwen3-235B-A22B",
    "q397": "Qwen3.5-397B-A17B",
    "qwen397": "Qwen3.5-397B-A17B",
    "qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
}

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
NC = "\033[0m"


def info(msg: str):
    print(f"{CYAN}{BOLD}=>{NC} {msg}")

def ok(msg: str):
    print(f"{GREEN}{BOLD}OK{NC} {msg}")

def warn(msg: str):
    print(f"{YELLOW}{BOLD}!!{NC} {msg}")

def die(msg: str):
    print(f"{RED}{BOLD}ERROR{NC} {msg}", file=sys.stderr)
    sys.exit(1)


def _package_version(package_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_file_audit(model_path: str) -> Dict[str, Any]:
    root = Path(model_path)
    filenames = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    files: Dict[str, Any] = {}
    for name in filenames:
        path = root / name
        if path.is_file():
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    remote_code_files = []
    for path in sorted(root.glob("*.py")):
        remote_code_files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "path": str(root.resolve()),
        "files": files,
        "remote_code_files": remote_code_files,
    }


def build_invocation_metadata(model_name: str, profile_id: str, max_new_tokens: int) -> Dict[str, Any]:
    run_dir = os.environ.get("KRASIS_RUN_DIR")
    script_argv = [str(Path(arg)) if idx == 0 else arg for idx, arg in enumerate(sys.argv)]
    dev_command = os.environ.get("KRASIS_DEV_CMD")
    if not dev_command:
        dev_command = " ".join(shlex.quote(arg) for arg in (["./dev", "generate-reference"] + sys.argv[1:]))
    metadata: Dict[str, Any] = {
        "captured_at": datetime.now().isoformat(),
        "dev_command": dev_command,
        "script_argv": script_argv,
        "cwd": str(Path.cwd().resolve()),
        "run_type": os.environ.get("KRASIS_RUN_TYPE"),
        "model": model_name,
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
    }
    if run_dir:
        metadata["run_dir"] = str(Path(run_dir).resolve())
    return metadata


def resolve_model_name(raw_model: str) -> str:
    raw = raw_model.strip()
    if not raw:
        die("Model name cannot be empty")

    candidate = Path(raw)
    if candidate.name != raw:
        raw = candidate.name

    if os.path.isdir(os.path.join(MODELS_DIR, raw)):
        return raw

    lowered = raw.lower()
    resolved = MODEL_ALIASES.get(lowered)
    if resolved and os.path.isdir(os.path.join(MODELS_DIR, resolved)):
        return resolved

    if "/" in raw_model:
        basename = Path(raw_model).name
        if os.path.isdir(os.path.join(MODELS_DIR, basename)):
            return basename

    if resolved:
        return resolved
    return raw


def write_run_manifest(
    invocation: Dict[str, Any],
    output_path: Path,
    model_name: str,
    profile_id: str,
    max_new_tokens: int,
    diagnostic_steps: int,
    diagnostic_top_k: int,
    prompt_subset: Optional[Dict[str, Any]] = None,
    attn_implementation: Optional[str] = None,
    sanity: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "generate_reference_manifest.json"
    manifest = {
        "kind": "generate_reference_manifest",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": model_name,
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
        "decode_diagnostics": {
            "captured_steps_per_turn": diagnostic_steps,
            "top_k": diagnostic_top_k,
        },
        "prompt_subset": prompt_subset,
        "attn_implementation": attn_implementation,
        "reference_output_path": str(output_path.resolve()),
        "sanity": sanity,
        "follow_up_commands": [
            "./dev reference-inventory",
            "./dev validate <config> --max-prompts 0 --no-server --port 1",
            "./dev reference-test <config>",
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    ok(f"Run manifest saved: {manifest_path}")


def write_generation_failure_manifest(
    invocation: Dict[str, Any],
    *,
    intended_output_path: Path,
    model_name: str,
    profile_id: str,
    sanity: Dict[str, Any],
    prompt_subset: Optional[Dict[str, Any]] = None,
    attn_implementation: Optional[str] = None,
    turn_summaries: Optional[List[Dict[str, Any]]] = None,
    diagnostic_controls: Optional[Dict[str, Any]] = None,
    environment_audit: Optional[Dict[str, Any]] = None,
    token_audit_turns: Optional[List[Dict[str, Any]]] = None,
) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "generate_reference_failure.json"
    manifest = {
        "kind": "generate_reference_failure",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": model_name,
        "profile_id": profile_id,
        "prompt_subset": prompt_subset,
        "attn_implementation": attn_implementation,
        "diagnostic_controls": diagnostic_controls or {},
        "intended_reference_output_path": str(intended_output_path.resolve()),
        "sanity": sanity,
        "environment_audit": environment_audit or {},
        "turn_summaries": turn_summaries or [],
        "token_audit_turns": token_audit_turns or [],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    warn(f"Failure manifest saved: {manifest_path}")


def write_generation_diagnostic_manifest(
    invocation: Dict[str, Any],
    *,
    intended_output_path: Path,
    model_name: str,
    profile_id: str,
    sanity: Dict[str, Any],
    prompt_subset: Optional[Dict[str, Any]] = None,
    attn_implementation: Optional[str] = None,
    turn_summaries: Optional[List[Dict[str, Any]]] = None,
    diagnostic_controls: Optional[Dict[str, Any]] = None,
    environment_audit: Optional[Dict[str, Any]] = None,
    token_audit_turns: Optional[List[Dict[str, Any]]] = None,
) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "generate_reference_diagnostic.json"
    manifest = {
        "kind": "generate_reference_diagnostic",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": model_name,
        "profile_id": profile_id,
        "prompt_subset": prompt_subset,
        "attn_implementation": attn_implementation,
        "diagnostic_controls": diagnostic_controls or {},
        "intended_reference_output_path": str(intended_output_path.resolve()),
        "reference_artifact_written": False,
        "sanity": sanity,
        "environment_audit": environment_audit or {},
        "turn_summaries": turn_summaries or [],
        "token_audit_turns": token_audit_turns or [],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    ok(f"Diagnostic manifest saved: {manifest_path}")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fnv1a64_token_hash(tokens: List[int]) -> str:
    h = 0xCBF29CE484222325
    for token in tokens:
        value = int(token) & 0xFFFFFFFF
        for shift in range(0, 32, 8):
            h ^= (value >> shift) & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"0x{h:016x}"


def _finite_float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _tensor_finiteness_stats(tensor: Any) -> Dict[str, Any]:
    import torch

    detached = tensor.detach()
    flat = detached.reshape(-1)
    finite_mask = torch.isfinite(flat)
    nan_mask = torch.isnan(flat)
    inf_mask = torch.isinf(flat)
    finite_values = flat[finite_mask]
    stats: Dict[str, Any] = {
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "shape": [int(dim) for dim in detached.shape],
        "numel": int(flat.numel()),
        "finite_count": int(finite_mask.sum().item()),
        "nan_count": int(nan_mask.sum().item()),
        "inf_count": int(inf_mask.sum().item()),
        "all_finite": bool(finite_mask.all().item()) if flat.numel() else True,
    }
    if finite_values.numel() > 0:
        finite_float = finite_values.float()
        stats["finite_min"] = float(finite_float.min().item())
        stats["finite_max"] = float(finite_float.max().item())
    else:
        stats["finite_min"] = None
        stats["finite_max"] = None
    return stats


def _tensor_last_token_summary(
    tensor: Any,
    index: int,
    *,
    label: Optional[str] = None,
    source: str = "hf_forward_hidden_states_last_token",
    layer: Optional[int] = None,
) -> Dict[str, Any]:
    import torch

    detached = tensor.detach()
    if detached.ndim >= 3:
        row = detached[0, -1, :]
    elif detached.ndim == 2:
        row = detached[-1, :]
    else:
        row = detached.reshape(-1)

    row_f32 = row.float().contiguous()
    finite_mask = torch.isfinite(row_f32)
    finite_values = row_f32[finite_mask]
    row_cpu = row_f32.cpu()
    row_hash = hashlib.sha256(row_cpu.numpy().tobytes()).hexdigest()
    summary: Dict[str, Any] = {
        "index": int(index),
        "layer": int(index - 1 if layer is None else layer),
        "label": label if label is not None else ("embedding_output" if index == 0 else f"layer_{index - 1}_output"),
        "source": source,
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "shape": [int(dim) for dim in detached.shape],
        "row_width": int(row.numel()),
        "sha256_f32": row_hash,
        "finite_count": int(finite_mask.sum().item()),
        "nan_count": int(torch.isnan(row_f32).sum().item()),
        "inf_count": int(torch.isinf(row_f32).sum().item()),
    }
    if finite_values.numel() > 0:
        summary.update(
            {
                "mean": float(finite_values.mean().item()),
                "l2": float(torch.linalg.vector_norm(finite_values).item()),
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
            }
        )
    else:
        summary.update({"mean": None, "l2": None, "min": None, "max": None})
    return summary


def _tensor_flat_summary(
    tensor: Any,
    index: int,
    *,
    label: str,
    source: str,
    layer: int,
) -> Dict[str, Any]:
    import torch

    detached = tensor.detach()
    row_f32 = detached.reshape(-1).float().contiguous()
    finite_mask = torch.isfinite(row_f32)
    finite_values = row_f32[finite_mask]
    row_cpu = row_f32.cpu()
    row_hash = hashlib.sha256(row_cpu.numpy().tobytes()).hexdigest()
    summary: Dict[str, Any] = {
        "index": int(index),
        "layer": int(layer),
        "label": label,
        "source": source,
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "shape": [int(dim) for dim in detached.shape],
        "row_width": int(row_f32.numel()),
        "sha256_f32": row_hash,
        "finite_count": int(finite_mask.sum().item()),
        "nan_count": int(torch.isnan(row_f32).sum().item()),
        "inf_count": int(torch.isinf(row_f32).sum().item()),
    }
    if finite_values.numel() > 0:
        summary.update(
            {
                "mean": float(finite_values.mean().item()),
                "l2": float(torch.linalg.vector_norm(finite_values).item()),
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
            }
        )
    else:
        summary.update({"mean": None, "l2": None, "min": None, "max": None})
    return summary


def _first_tensor(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if hasattr(value, "last_hidden_state"):
        return _first_tensor(value.last_hidden_state)
    return None


def _find_hf_block(model: Any, layer_idx: int) -> Any:
    base_model = getattr(model, "model", None) or getattr(model, "backbone", None)
    layers = getattr(base_model, "layers", None) if base_model is not None else None
    if layers is None or len(layers) <= layer_idx:
        raise RuntimeError(
            f"HF model does not expose model/backbone.layers[{layer_idx}] for layer diagnostics"
        )
    return layers[layer_idx]


def _find_hf_layer0_block(model: Any) -> Any:
    return _find_hf_block(model, 0)


def _augment_dt_bias_summary(summary: Dict[str, Any], tensor: Any) -> Dict[str, Any]:
    import torch

    flat = tensor.detach().reshape(-1).float().cpu()
    sample_indices = [0, 1, 2, 16, 20, 31, 63]
    sample_values = []
    for idx in sample_indices:
        if idx < int(flat.numel()):
            sample_values.append({"index": int(idx), "value_f32": float(flat[idx].item())})
    summary["sample_values_f32"] = sample_values
    summary["value_count"] = int(flat.numel())
    if tensor.detach().dtype == torch.bfloat16:
        bits = tensor.detach().reshape(-1).cpu().view(torch.uint16)
        summary["sample_values_bf16_u16"] = [
            {"index": item["index"], "bits_u16": int(bits[item["index"]].item())}
            for item in sample_values
        ]
    return summary


def _dt_bias_flat_summary(
    tensor: Any,
    *,
    index: int,
    label: str,
    source: str,
) -> Dict[str, Any]:
    return _augment_dt_bias_summary(
        _tensor_flat_summary(tensor, index, label=label, source=source, layer=0),
        tensor,
    )


def _find_layer_dt_bias_parameter(model: Any, layer_idx: int) -> Tuple[Optional[str], Any]:
    for name, param in model.named_parameters():
        if name.endswith(f"layers.{layer_idx}.mixer.dt_bias"):
            return name, param
    try:
        block = _find_hf_block(model, layer_idx)
        mixer = getattr(block, "mixer", None)
        param = getattr(mixer, "dt_bias", None) if mixer is not None else None
        if param is not None:
            return f"backbone/model.layers.{layer_idx}.mixer.dt_bias", param
    except Exception:
        pass
    return None, None


def _find_layer0_dt_bias_parameter(model: Any) -> Tuple[Optional[str], Any]:
    return _find_layer_dt_bias_parameter(model, 0)


def _read_raw_safetensor_tensor(model_path: str, tensor_name: str) -> Tuple[Optional[Path], Any]:
    import torch
    from safetensors import safe_open

    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    candidate_files: List[Path] = []
    if index_path.is_file():
        with index_path.open("r") as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_name = weight_map.get(tensor_name)
        if shard_name:
            candidate_files.append(root / shard_name)
    candidate_files.extend(sorted(root.glob("*.safetensors")))

    seen = set()
    for shard_path in candidate_files:
        if shard_path in seen or not shard_path.is_file():
            continue
        seen.add(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            if tensor_name in shard.keys():
                tensor = shard.get_tensor(tensor_name)
                if not torch.is_tensor(tensor):
                    continue
                return shard_path, tensor
    return None, None


def capture_hf_layer0_dt_bias_source_diagnostics(model: Any, model_path: str) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    param_name, param = _find_layer0_dt_bias_parameter(model)
    if param is None:
        summaries.append(
            {
                "index": 0,
                "layer": 0,
                "label": "layer0_mamba2_dt_bias_after_from_pretrained",
                "source": "hf_model_parameter_after_from_pretrained",
                "available": False,
                "reason": "could not locate layer-0 mixer dt_bias parameter",
            }
        )
        return summaries

    loaded_summary = _dt_bias_flat_summary(
        param.detach(),
        index=len(summaries),
        label="layer0_mamba2_dt_bias_after_from_pretrained",
        source="hf_model_parameter_after_from_pretrained",
    )
    loaded_summary["parameter_name"] = param_name
    summaries.append(loaded_summary)

    raw_tensor_names = [
        "backbone.layers.0.mixer.dt_bias",
        "model.layers.0.mixer.dt_bias",
        str(param_name) if param_name else "",
    ]
    raw_seen = set()
    raw_found = False
    for tensor_name in raw_tensor_names:
        if not tensor_name or tensor_name in raw_seen:
            continue
        raw_seen.add(tensor_name)
        shard_path, raw_tensor = _read_raw_safetensor_tensor(model_path, tensor_name)
        if raw_tensor is None:
            continue
        raw_summary = _dt_bias_flat_summary(
            raw_tensor,
            index=len(summaries),
            label="layer0_mamba2_dt_bias_raw_safetensor",
            source="hf_checkpoint_safetensor",
        )
        raw_summary["tensor_name"] = tensor_name
        raw_summary["shard_path"] = str(shard_path) if shard_path is not None else None
        summaries.append(raw_summary)
        raw_found = True
        break

    if not raw_found:
        summaries.append(
            {
                "index": len(summaries),
                "layer": 0,
                "label": "layer0_mamba2_dt_bias_raw_safetensor",
                "source": "hf_checkpoint_safetensor",
                "available": False,
                "reason": f"raw safetensor tensor not found for candidates {raw_tensor_names}",
                "parameter_name": param_name,
            }
        )
    return summaries


def _is_nemotron_h_reference_model(model: Any, config: Any) -> bool:
    names = {
        type(model).__name__,
        type(config).__name__,
        str(getattr(config, "model_type", "")),
    }
    names.update(str(arch) for arch in (getattr(config, "architectures", []) or []))
    return any("NemotronH" in name or "nemotron_h" in name for name in names)


def restore_nemotron_dt_bias_from_safetensors(
    model: Any,
    config: Any,
    model_path: str,
) -> Dict[str, Any]:
    import torch
    from accelerate.utils.modeling import set_module_tensor_to_device

    if not _is_nemotron_h_reference_model(model, config):
        return {
            "label": "nemotron_dt_bias_safetensor_restore",
            "applied": False,
            "reason": "not a Nemotron-H reference model",
        }

    rows = []
    restored = 0
    max_abs_before = 0.0
    device_map = getattr(model, "hf_device_map", {}) or {}

    def target_device_for_param(param_name: str, param_value: Any) -> Any:
        if param_value.device.type != "meta":
            return param_value.device
        parts = param_name.split(".")
        for end in range(len(parts), 0, -1):
            prefix = ".".join(parts[:end])
            if prefix in device_map:
                mapped = device_map[prefix]
                if isinstance(mapped, int):
                    return torch.device(f"cuda:{mapped}")
                return torch.device(str(mapped))
        raise RuntimeError(f"Cannot determine device-map target for meta parameter {param_name}")

    for name, param in model.named_parameters():
        if not name.endswith(".mixer.dt_bias"):
            continue
        shard_path, raw_tensor = _read_raw_safetensor_tensor(model_path, name)
        if raw_tensor is None:
            raise RuntimeError(f"Missing raw safetensor dt_bias for HF reference parameter {name}")
        if list(raw_tensor.shape) != list(param.shape):
            raise RuntimeError(
                f"Raw safetensor dt_bias shape mismatch for {name}: "
                f"raw={list(raw_tensor.shape)} param={list(param.shape)}"
            )
        target_device = target_device_for_param(name, param)
        before_was_meta = param.device.type == "meta"
        before = None if before_was_meta else param.detach().float().cpu()
        raw_f32 = raw_tensor.detach().float().cpu()
        if before is not None:
            diff = (before - raw_f32).abs()
            row_max_abs = float(diff.max().item()) if diff.numel() else 0.0
            row_l2 = float(torch.linalg.vector_norm(diff).item()) if diff.numel() else 0.0
        else:
            row_max_abs = None
            row_l2 = None
        if row_max_abs is not None:
            max_abs_before = max(max_abs_before, row_max_abs)
        raw_for_param = raw_tensor.to(device=target_device, dtype=param.dtype)
        if before_was_meta:
            set_module_tensor_to_device(model, name, target_device, value=raw_for_param)
        else:
            with torch.no_grad():
                param.copy_(raw_for_param)
        restored += 1
        if len(rows) < 8 or row_max_abs is None or row_max_abs != 0.0:
            rows.append(
                {
                    "parameter_name": name,
                    "shard_path": str(shard_path) if shard_path is not None else None,
                    "shape": [int(v) for v in param.shape],
                    "device": str(target_device),
                    "dtype": str(param.dtype),
                    "before_was_meta": bool(before_was_meta),
                    "max_abs_before_restore": row_max_abs,
                    "l2_before_restore": row_l2,
                }
            )

    if restored == 0:
        raise RuntimeError("Nemotron-H reference model exposed no *.mixer.dt_bias parameters to restore")

    return {
        "label": "nemotron_dt_bias_safetensor_restore",
        "applied": True,
        "restored_parameters": int(restored),
        "max_abs_before_restore": float(max_abs_before),
        "sample_rows": rows[:24],
    }


def _capture_hf_layer0_mamba2_internal_summaries(
    model: Any,
    block: Any,
    prompt_input_ids: Any,
    forward_kwargs: Dict[str, Any],
    start_index: int,
    element_dims: Optional[List[int]] = None,
    row_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    import sys
    import torch

    block_type = getattr(block, "block_type", None)
    mixer = getattr(block, "mixer", None)
    summaries: List[Dict[str, Any]] = []
    selected_element_dims = sorted({int(dim) for dim in (element_dims or []) if int(dim) >= 0})
    selected_row_indices = sorted({int(row) for row in (row_indices or []) if int(row) >= 0})

    def unavailable(label: str, source: str, reason: str) -> None:
        summaries.append(
            {
                "index": start_index + len(summaries),
                "layer": 0,
                "label": label,
                "source": source,
                "available": False,
                "reason": reason,
                "block_type": block_type,
            }
        )

    if block_type != "mamba" or mixer is None:
        unavailable(
            "layer0_mamba2_internals",
            "hf_layer0_mamba2_manual_diagnostic",
            f"layer0 block_type is {block_type}; no Mamba2 mixer",
        )
        return summaries

    module = sys.modules.get(mixer.__class__.__module__)
    scan_fn = getattr(module, "mamba_chunk_scan_combined", None) if module is not None else None
    if scan_fn is None:
        unavailable(
            "layer0_mamba2_ssd_out",
            "hf_layer0_mamba2_manual_diagnostic",
            "mamba_chunk_scan_combined is unavailable",
        )
        return summaries
    causal_conv1d_fn = getattr(module, "causal_conv1d_fn", None) if module is not None else None

    def record(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable(label, source, "no tensor value")
            return
        summary = _tensor_last_token_summary(
            tensor,
            start_index + len(summaries),
            label=label,
            source=source,
            layer=0,
        )
        summary["block_type"] = block_type
        summaries.append(summary)

    def record_flat(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable(label, source, "no tensor value")
            return
        summary = _tensor_flat_summary(
            tensor,
            start_index + len(summaries),
            label=label,
            source=source,
            layer=0,
        )
        summary["block_type"] = block_type
        summaries.append(summary)

    def fnv1a_u16(values: Any) -> str:
        h = 0xCBF29CE484222325
        for raw in values:
            v = int(raw) & 0xFFFF
            h ^= v & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            h ^= (v >> 8) & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f"0x{h:016x}"

    def bf16_u16_values(tensor_value: Any) -> List[int]:
        return [
            int(v) & 0xFFFF
            for v in tensor_value.detach()
            .contiguous()
            .view(torch.int16)
            .cpu()
            .reshape(-1)
            .tolist()
        ]

    def f32_round(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]

    def f32_bits(value: float) -> int:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]

    def f32_from_bits(bits: int) -> float:
        return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]

    def bf16_value_from_bits(bits: int) -> float:
        return f32_from_bits((int(bits) & 0xFFFF) << 16)

    def bf16_bits_from_f32_bits(bits: int) -> int:
        value = f32_from_bits(bits)
        tensor = torch.tensor([value], dtype=torch.float32).to(torch.bfloat16)
        return int(bf16_u16_values(tensor)[0]) & 0xFFFF

    def bf16_rounding_metadata_from_f32_bits(bits: int) -> Dict[str, Any]:
        raw = int(bits) & 0xFFFFFFFF
        truncated = (raw >> 16) & 0xFFFF
        next_bits = (truncated + 1) & 0xFFFF
        midpoint = (bf16_value_from_bits(truncated) + bf16_value_from_bits(next_bits)) * 0.5
        midpoint_bits = f32_bits(midpoint)
        tie_rounds_to = next_bits if (truncated & 1) else truncated
        value = f32_from_bits(raw)
        if value < midpoint:
            side = "below_midpoint"
        elif value > midpoint:
            side = "above_midpoint"
        else:
            side = "at_midpoint"
        return {
            "f32_bits_hex": f"0x{raw:08x}",
            "truncated_bf16_bits": int(truncated),
            "truncated_bf16_bits_hex": f"0x{truncated:04x}",
            "truncated_bf16_value": bf16_value_from_bits(truncated),
            "next_bf16_bits": int(next_bits),
            "next_bf16_bits_hex": f"0x{next_bits:04x}",
            "next_bf16_value": bf16_value_from_bits(next_bits),
            "midpoint_f32_bits": int(midpoint_bits),
            "midpoint_f32_bits_hex": f"0x{midpoint_bits:08x}",
            "midpoint_value": midpoint,
            "value_vs_midpoint": side,
            "tie_rounds_to_bf16_bits": int(tie_rounds_to),
            "tie_rounds_to_bf16_bits_hex": f"0x{tie_rounds_to:04x}",
        }

    def f32_seq_dot(lhs: List[float], rhs: List[float]) -> float:
        acc = 0.0
        for a, b in zip(lhs, rhs):
            acc = f32_round(acc + f32_round(a * b))
        return acc

    def append_gated_norm_element_details(
        norm_input: Any,
        gate_value: Any,
        stored_output: Any,
    ) -> None:
        if not selected_element_dims:
            return
        norm_module = getattr(mixer, "norm", None)
        weight = getattr(norm_module, "weight", None) if norm_module is not None else None
        group_size = getattr(norm_module, "group_size", None) if norm_module is not None else None
        eps = getattr(norm_module, "variance_epsilon", None) if norm_module is not None else None
        tensors = {
            "norm_input": _first_tensor(norm_input),
            "gate": _first_tensor(gate_value),
            "stored_output": _first_tensor(stored_output),
            "weight": weight,
        }
        missing = [name for name, tensor in tensors.items() if tensor is None]
        if missing or group_size is None or eps is None:
            summaries.append(
                {
                    "index": start_index + len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_gated_norm_element_details",
                    "source": "hf_layer0_mamba2_selected_dim_gated_norm_provenance",
                    "available": False,
                    "reason": (
                        f"missing tensors/params: tensors={missing}, "
                        f"group_size={group_size}, eps={eps}"
                    ),
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )
            return

        norm_input_rows = tensors["norm_input"].reshape(-1, tensors["norm_input"].shape[-1])
        gate_rows = tensors["gate"].reshape(-1, tensors["gate"].shape[-1])
        stored_rows = tensors["stored_output"].reshape(-1, tensors["stored_output"].shape[-1])
        norm_input_row = norm_input_rows[-1].detach().contiguous()
        gate_row = gate_rows[-1].detach().contiguous()
        stored_row = stored_rows[-1].detach().contiguous()
        weight_row = tensors["weight"].detach().contiguous()
        rows = {
            "norm_input": norm_input_row,
            "gate": gate_row,
            "stored_output": stored_row,
        }
        non_bf16 = {
            name: str(row.dtype)
            for name, row in rows.items()
            if row.dtype != torch.bfloat16
        }
        if non_bf16:
            summaries.append(
                {
                    "index": start_index + len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_gated_norm_element_details",
                    "source": "hf_layer0_mamba2_selected_dim_gated_norm_provenance",
                    "available": False,
                    "reason": f"expected BF16 tensors, got {non_bf16}",
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )
            return

        width = int(norm_input_row.numel())
        group_size = int(group_size)
        if group_size <= 0 or width % group_size != 0:
            summaries.append(
                {
                    "index": start_index + len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_gated_norm_element_details",
                    "source": "hf_layer0_mamba2_selected_dim_gated_norm_provenance",
                    "available": False,
                    "reason": f"invalid group_size={group_size} for width={width}",
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )
            return

        weight_f32 = weight_row.to(torch.float32)
        producer_module_type = type(norm_module)
        producer_module = f"{producer_module_type.__module__}.{producer_module_type.__qualname__}"
        producer_op = "mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn"
        producer_call = "mixer.norm(scan_output_flat, gate)"
        producer_kwargs = {
            "bias": None,
            "eps": float(eps),
            "group_size": int(group_size),
            "norm_before_gate": False,
        }
        actual_producer_fp32_rows = None
        actual_producer_fp32_rstd_rows = None
        actual_producer_fp32_error = None
        try:
            from mamba_ssm.ops.triton import layernorm_gated as layernorm_gated_module

            if hasattr(layernorm_gated_module, "_layer_norm_fwd"):
                replay_x = norm_input_rows.detach().contiguous()
                replay_z = gate_rows.detach().contiguous()
                replay_out = torch.empty(
                    replay_x.shape,
                    dtype=torch.float32,
                    device=replay_x.device,
                )
                replay_y, _replay_mean, replay_rstd = layernorm_gated_module._layer_norm_fwd(
                    replay_x,
                    weight_row.detach().contiguous(),
                    None,
                    float(eps),
                    z=replay_z,
                    out=replay_out,
                    group_size=group_size,
                    norm_before_gate=False,
                    is_rms_norm=True,
                )
                actual_producer_fp32_rows = replay_y.detach().contiguous()
                actual_producer_fp32_rstd_rows = replay_rstd.detach().contiguous()
            else:
                actual_producer_fp32_error = (
                    "mamba_ssm.ops.triton.layernorm_gated._layer_norm_fwd is unavailable"
                )
        except Exception as exc:
            actual_producer_fp32_error = f"{type(exc).__name__}: {exc}"

        def emit_gated_norm_row(
            label: str,
            source: str,
            norm_row: Any,
            gate_row_value: Any,
            stored_row_value: Any,
            row_index: Optional[int] = None,
        ) -> None:
            norm_input_bits = bf16_u16_values(norm_row)
            gate_bits = bf16_u16_values(gate_row_value)
            stored_bits = bf16_u16_values(stored_row_value)
            norm_input_f32 = norm_row.to(torch.float32)
            gate_f32 = gate_row_value.to(torch.float32)
            stored_f32 = stored_row_value.to(torch.float32)
            actual_fp32_row = None
            actual_fp32_rstd_row = None
            if actual_producer_fp32_rows is not None:
                actual_row_index = int(row_index) if row_index is not None else int(norm_input_rows.shape[0]) - 1
                if 0 <= actual_row_index < int(actual_producer_fp32_rows.shape[0]):
                    actual_fp32_row = actual_producer_fp32_rows[actual_row_index].detach().contiguous()
                    if actual_producer_fp32_rstd_rows is not None:
                        actual_fp32_rstd_row = actual_producer_fp32_rstd_rows.detach().reshape(-1)
            details = []
            for idx in selected_element_dims:
                if idx < 0 or idx >= width:
                    details.append({"dim_index": int(idx), "available": False, "reason": "dim out of range"})
                    continue
                group = int(idx // group_size)
                group_base = group * group_size
                group_end = group_base + group_size
                group_input = norm_input_f32[group_base:group_end]
                group_gate = gate_f32[group_base:group_end]
                group_silu = group_gate / (1.0 + torch.exp(-group_gate))
                group_gated = group_input * group_silu
                mean_square = float((group_gated * group_gated).mean().item())
                eps_value = float(eps)
                mean_square_plus_eps = float(
                    torch.tensor(mean_square + eps_value, dtype=torch.float32).item()
                )
                mean_square_plus_eps_tensor = torch.tensor(mean_square_plus_eps, dtype=torch.float32)
                sqrt_value = float(torch.sqrt(mean_square_plus_eps_tensor).item())
                rms_inv = float(torch.rsqrt(mean_square_plus_eps_tensor).item())
                one_over_sqrt = float(
                    (torch.tensor(1.0, dtype=torch.float32) / torch.sqrt(mean_square_plus_eps_tensor)).item()
                )
                double_promoted_rstd = f32_round(1.0 / math.sqrt(float(mean_square_plus_eps)))
                norm_input_value = float(norm_input_f32[idx].item())
                gate_item = float(gate_f32[idx].item())
                silu_gate = float((gate_f32[idx] / (1.0 + torch.exp(-gate_f32[idx]))).item())
                gated_product = float((norm_input_f32[idx] * (gate_f32[idx] / (1.0 + torch.exp(-gate_f32[idx])))).item())
                normalized = float(gated_product * rms_inv)
                weight_value = float(weight_f32[idx].item())
                pre_store = float(normalized * weight_value)
                norm_input_value_bits = f32_bits(norm_input_value)
                gate_item_bits = f32_bits(gate_item)
                silu_gate_bits = f32_bits(silu_gate)
                gated_product_bits = f32_bits(gated_product)
                mean_square_bits = f32_bits(mean_square)
                eps_bits = f32_bits(eps_value)
                mean_square_plus_eps_bits = f32_bits(mean_square_plus_eps)
                sqrt_value_bits = f32_bits(sqrt_value)
                rms_inv_bits = f32_bits(rms_inv)
                one_over_sqrt_bits = f32_bits(one_over_sqrt)
                double_promoted_rstd_bits = f32_bits(double_promoted_rstd)
                normalized_bits = f32_bits(normalized)
                weight_value_bits = f32_bits(weight_value)
                pre_store_bits = f32_bits(pre_store)
                pre_store_candidate_bits = bf16_bits_from_f32_bits(pre_store_bits)
                actual_producer_output = None
                actual_producer_output_bits = None
                actual_producer_candidate_bits = None
                actual_producer_rstd = None
                actual_producer_rstd_bits = None
                if actual_fp32_row is not None:
                    actual_producer_output = float(actual_fp32_row[idx].item())
                    actual_producer_output_bits = f32_bits(actual_producer_output)
                    actual_producer_candidate_bits = bf16_bits_from_f32_bits(actual_producer_output_bits)
                if actual_fp32_rstd_row is not None:
                    group_rstd_index = (
                        group * int(norm_input_rows.shape[0])
                        + (int(row_index) if row_index is not None else int(norm_input_rows.shape[0]) - 1)
                    )
                    if 0 <= group_rstd_index < int(actual_fp32_rstd_row.numel()):
                        actual_producer_rstd = float(actual_fp32_rstd_row[group_rstd_index].item())
                        actual_producer_rstd_bits = f32_bits(actual_producer_rstd)
                stored_bits_idx = int(stored_bits[idx])
                details.append(
                    {
                        "dim_index": int(idx),
                        "group_index": group,
                        "group_size": group_size,
                        "norm_input_bits": int(norm_input_bits[idx]),
                        "gate_bits": int(gate_bits[idx]),
                        "stored_pre_out_proj_bits": stored_bits_idx,
                        "stored_pre_out_proj_bits_hex": f"0x{stored_bits_idx:04x}",
                        "norm_input_value": norm_input_value,
                        "norm_input_value_f32_bits": int(norm_input_value_bits),
                        "norm_input_value_f32_bits_hex": f"0x{norm_input_value_bits:08x}",
                        "gate_value": gate_item,
                        "gate_value_f32_bits": int(gate_item_bits),
                        "gate_value_f32_bits_hex": f"0x{gate_item_bits:08x}",
                        "silu_gate_value": silu_gate,
                        "silu_gate_value_f32_bits": int(silu_gate_bits),
                        "silu_gate_value_f32_bits_hex": f"0x{silu_gate_bits:08x}",
                        "gated_product": gated_product,
                        "gated_product_f32_bits": int(gated_product_bits),
                        "gated_product_f32_bits_hex": f"0x{gated_product_bits:08x}",
                        "mean_square": mean_square,
                        "mean_square_f32_bits": int(mean_square_bits),
                        "mean_square_f32_bits_hex": f"0x{mean_square_bits:08x}",
                        "eps": eps_value,
                        "eps_f32_bits": int(eps_bits),
                        "eps_f32_bits_hex": f"0x{eps_bits:08x}",
                        "mean_square_plus_eps": mean_square_plus_eps,
                        "mean_square_plus_eps_f32_bits": int(mean_square_plus_eps_bits),
                        "mean_square_plus_eps_f32_bits_hex": f"0x{mean_square_plus_eps_bits:08x}",
                        "sqrt_mean_square_plus_eps": sqrt_value,
                        "sqrt_mean_square_plus_eps_f32_bits": int(sqrt_value_bits),
                        "sqrt_mean_square_plus_eps_f32_bits_hex": f"0x{sqrt_value_bits:08x}",
                        "rms_inv": rms_inv,
                        "rms_inv_f32_bits": int(rms_inv_bits),
                        "rms_inv_f32_bits_hex": f"0x{rms_inv_bits:08x}",
                        "rstd_torch_rsqrt": rms_inv,
                        "rstd_torch_rsqrt_f32_bits": int(rms_inv_bits),
                        "rstd_torch_rsqrt_f32_bits_hex": f"0x{rms_inv_bits:08x}",
                        "rstd_one_over_torch_sqrt": one_over_sqrt,
                        "rstd_one_over_torch_sqrt_f32_bits": int(one_over_sqrt_bits),
                        "rstd_one_over_torch_sqrt_f32_bits_hex": f"0x{one_over_sqrt_bits:08x}",
                        "rstd_double_promoted": double_promoted_rstd,
                        "rstd_double_promoted_f32_bits": int(double_promoted_rstd_bits),
                        "rstd_double_promoted_f32_bits_hex": f"0x{double_promoted_rstd_bits:08x}",
                        "normalized_value": normalized,
                        "normalized_value_f32_bits": int(normalized_bits),
                        "normalized_value_f32_bits_hex": f"0x{normalized_bits:08x}",
                        "weight_value": weight_value,
                        "weight_f32_bits": int(weight_value_bits),
                        "weight_f32_bits_hex": f"0x{weight_value_bits:08x}",
                        "pre_store_output": pre_store,
                        "pre_store_source": "manual_component_reconstruction",
                        "pre_store_output_f32_bits": int(pre_store_bits),
                        "pre_store_output_f32_bits_hex": f"0x{pre_store_bits:08x}",
                        "pre_store_bf16_candidate_bits": int(pre_store_candidate_bits),
                        "pre_store_bf16_candidate_bits_hex": f"0x{pre_store_candidate_bits:04x}",
                        "pre_store_bf16_candidate_value": bf16_value_from_bits(pre_store_candidate_bits),
                        "pre_store_bf16_rounding": bf16_rounding_metadata_from_f32_bits(pre_store_bits),
                        "actual_producer_op": producer_op,
                        "actual_producer_call": producer_call,
                        "actual_producer_output_source": (
                            "same _layer_norm_fwd Triton kernel replayed with FP32 output buffer"
                        ),
                        "actual_producer_fp32_output_available": actual_producer_output_bits is not None,
                        "actual_producer_fp32_output_unavailable_reason": actual_producer_fp32_error,
                        "actual_producer_fp32_output": actual_producer_output,
                        "actual_producer_fp32_output_bits": actual_producer_output_bits,
                        "actual_producer_fp32_output_bits_hex": (
                            f"0x{actual_producer_output_bits:08x}"
                            if actual_producer_output_bits is not None
                            else None
                        ),
                        "actual_producer_bf16_candidate_bits": actual_producer_candidate_bits,
                        "actual_producer_bf16_candidate_bits_hex": (
                            f"0x{actual_producer_candidate_bits:04x}"
                            if actual_producer_candidate_bits is not None
                            else None
                        ),
                        "actual_producer_bf16_candidate_value": (
                            bf16_value_from_bits(actual_producer_candidate_bits)
                            if actual_producer_candidate_bits is not None
                            else None
                        ),
                        "actual_producer_bf16_rounding": (
                            bf16_rounding_metadata_from_f32_bits(actual_producer_output_bits)
                            if actual_producer_output_bits is not None
                            else None
                        ),
                        "actual_producer_rstd": actual_producer_rstd,
                        "actual_producer_rstd_f32_bits": actual_producer_rstd_bits,
                        "actual_producer_rstd_f32_bits_hex": (
                            f"0x{actual_producer_rstd_bits:08x}"
                            if actual_producer_rstd_bits is not None
                            else None
                        ),
                        "stored_pre_out_proj_value": float(stored_f32[idx].item()),
                        "stored_matches_pre_store_bf16_candidate": bool(
                            stored_bits_idx == pre_store_candidate_bits
                        ),
                        "stored_matches_actual_producer_bf16_candidate": (
                            bool(stored_bits_idx == actual_producer_candidate_bits)
                            if actual_producer_candidate_bits is not None
                            else None
                        ),
                        "manual_reconstruction_matches_actual_producer_fp32_bits": (
                            bool(pre_store_bits == actual_producer_output_bits)
                            if actual_producer_output_bits is not None
                            else None
                        ),
                    }
                )

            summary = {
                "index": start_index + len(summaries),
                "layer": 0,
                "label": label,
                "source": source,
                "dtype": "torch.bfloat16",
                "weight_dtype": str(weight_row.dtype),
                "selected_dims": selected_element_dims,
                "detail_count": len(details),
                "width": width,
                "group_size": group_size,
                "n_groups": int(width // group_size),
                "eps": float(eps),
                "producer_module": producer_module,
                "producer_op": producer_op,
                "producer_call": producer_call,
                "producer_kwargs": producer_kwargs,
                "actual_producer_fp32_replay_available": actual_producer_fp32_rows is not None,
                "actual_producer_fp32_replay_error": actual_producer_fp32_error,
                "stored_output_source": "actual mixer.norm/rmsnorm_fn BF16 output tensor",
                "manual_pre_store_source": "diagnostic component reconstruction from stored BF16 inputs",
                "norm_input_hash_fnv1a_u16": fnv1a_u16(norm_input_bits),
                "gate_hash_fnv1a_u16": fnv1a_u16(gate_bits),
                "stored_pre_out_proj_hash_fnv1a_u16": fnv1a_u16(stored_bits),
                "semantics": (
                    "actual stored output comes from HF MambaRMSNormGated/rmsnorm_fn; "
                    "manual pre_store_output fields are diagnostic reconstruction unless "
                    "actual_producer_fp32_output is present"
                ),
                "manual_reconstruction_operation_order": [
                    "bf16 norm_input/gate -> fp32",
                    "silu = gate / (1 + exp(-gate))",
                    "gated = norm_input * silu",
                    "mean_square = mean(gated * gated) within group",
                    "rstd = rsqrt(mean_square + eps)",
                    "pre_store = gated * rstd * weight",
                    "bf16 store uses round-to-nearest-even candidate",
                ],
                "details": details,
                "block_type": block_type,
            }
            if row_index is not None:
                summary["row_index"] = int(row_index)
                summary["row_count"] = int(norm_input_rows.shape[0])
            summaries.append(summary)

        emit_gated_norm_row(
            "layer0_mamba2_gated_norm_element_details",
            "hf_layer0_mamba2_selected_dim_gated_norm_provenance",
            norm_input_row,
            gate_row,
            stored_row,
        )

        for row_index in selected_row_indices:
            if row_index < 0 or row_index >= int(norm_input_rows.shape[0]):
                summaries.append(
                    {
                        "index": start_index + len(summaries),
                        "layer": 0,
                        "label": f"layer0_mamba2_gated_norm_row{row_index}_element_details",
                        "source": "hf_layer0_mamba2_selected_row_gated_norm_provenance",
                        "available": False,
                        "reason": f"row_index {row_index} out of range for row_count {int(norm_input_rows.shape[0])}",
                        "row_index": int(row_index),
                        "row_count": int(norm_input_rows.shape[0]),
                        "selected_dims": selected_element_dims,
                        "block_type": block_type,
                    }
                )
                continue
            emit_gated_norm_row(
                f"layer0_mamba2_gated_norm_row{row_index}_element_details",
                "hf_layer0_mamba2_selected_row_gated_norm_provenance",
                norm_input_rows[row_index].detach().contiguous(),
                gate_rows[row_index].detach().contiguous(),
                stored_rows[row_index].detach().contiguous(),
                row_index=row_index,
            )

    def append_branch_element_details(
        mixer_input: Any,
        pre_out_proj: Any,
        out_proj_value: Any,
    ) -> None:
        if not selected_element_dims:
            return
        out_proj_module = getattr(mixer, "out_proj", None)
        weight = getattr(out_proj_module, "weight", None) if out_proj_module is not None else None
        tensors = {
            "mixer_input": _first_tensor(mixer_input),
            "pre_out_proj": _first_tensor(pre_out_proj),
            "out_proj": _first_tensor(out_proj_value),
            "out_proj_weight": weight,
        }
        missing = [name for name, tensor in tensors.items() if tensor is None]
        if missing:
            summaries.append(
                {
                    "index": start_index + len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_branch_element_details",
                    "source": "hf_layer0_mamba2_selected_dim_branch_provenance",
                    "available": False,
                    "reason": f"missing tensors: {missing}",
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )
            return

        mixer_input_row = tensors["mixer_input"].reshape(-1, tensors["mixer_input"].shape[-1])[-1].detach().contiguous()
        pre_out_proj_row = tensors["pre_out_proj"].reshape(-1, tensors["pre_out_proj"].shape[-1])[-1].detach().contiguous()
        out_proj_row = tensors["out_proj"].reshape(-1, tensors["out_proj"].shape[-1])[-1].detach().contiguous()
        weight_tensor = tensors["out_proj_weight"].detach().contiguous()
        rows = {
            "mixer_input": mixer_input_row,
            "pre_out_proj": pre_out_proj_row,
            "out_proj": out_proj_row,
            "out_proj_weight": weight_tensor,
        }
        non_bf16 = {
            name: str(row.dtype)
            for name, row in rows.items()
            if row.dtype != torch.bfloat16
        }
        if non_bf16:
            summaries.append(
                {
                    "index": start_index + len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_branch_element_details",
                    "source": "hf_layer0_mamba2_selected_dim_branch_provenance",
                    "available": False,
                    "reason": f"expected BF16 tensors, got {non_bf16}",
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )
            return

        mixer_input_bits = bf16_u16_values(mixer_input_row)
        pre_out_proj_bits = bf16_u16_values(pre_out_proj_row)
        out_proj_bits = bf16_u16_values(out_proj_row)
        pre_out_proj_values = pre_out_proj_row.to(torch.float32).cpu().reshape(-1).tolist()
        details = []
        for idx in selected_element_dims:
            item: Dict[str, Any] = {"dim_index": int(idx)}
            if idx < len(mixer_input_bits):
                item.update(
                    {
                        "mixer_input_bits": int(mixer_input_bits[idx]),
                        "mixer_input_value": float(mixer_input_row[idx].to(torch.float32).item()),
                    }
                )
            if idx < len(pre_out_proj_bits):
                item.update(
                    {
                        "pre_out_proj_bits": int(pre_out_proj_bits[idx]),
                        "pre_out_proj_value": float(pre_out_proj_row[idx].to(torch.float32).item()),
                    }
                )
            if idx < len(out_proj_bits):
                weight_row = weight_tensor[idx].detach().contiguous()
                weight_bits = bf16_u16_values(weight_row)
                weight_values = weight_row.to(torch.float32).cpu().reshape(-1).tolist()
                manual_dot = f32_seq_dot(pre_out_proj_values, weight_values)
                production_value = float(out_proj_row[idx].to(torch.float32).item())
                max_abs_contrib_index = 0
                max_abs_contrib = -1.0
                for contrib_idx, (hidden_v, weight_v) in enumerate(zip(pre_out_proj_values, weight_values)):
                    contrib = abs(hidden_v * weight_v)
                    if contrib > max_abs_contrib:
                        max_abs_contrib = contrib
                        max_abs_contrib_index = contrib_idx
                item.update(
                    {
                        "out_proj_output_bits": int(out_proj_bits[idx]),
                        "out_proj_output_value": production_value,
                        "out_proj_weight_row_hash_fnv1a_u16": fnv1a_u16(weight_bits),
                        "out_proj_manual_seq_fp32_dot": float(manual_dot),
                        "out_proj_manual_minus_production": float(manual_dot - production_value),
                        "out_proj_max_abs_contrib_index": int(max_abs_contrib_index),
                        "out_proj_max_abs_contrib_input": float(pre_out_proj_values[max_abs_contrib_index]),
                        "out_proj_max_abs_contrib_weight": float(weight_values[max_abs_contrib_index]),
                    }
                )
            details.append(item)

        summaries.append(
            {
                "index": start_index + len(summaries),
                "layer": 0,
                "label": "layer0_mamba2_branch_element_details",
                "source": "hf_layer0_mamba2_selected_dim_branch_provenance",
                "dtype": "torch.bfloat16",
                "selected_dims": selected_element_dims,
                "detail_count": len(details),
                "mixer_input_width": int(mixer_input_row.numel()),
                "pre_out_proj_width": int(pre_out_proj_row.numel()),
                "out_proj_width": int(out_proj_row.numel()),
                "out_proj_weight_shape": [int(v) for v in weight_tensor.shape],
                "mixer_input_hash_fnv1a_u16": fnv1a_u16(mixer_input_bits),
                "pre_out_proj_hash_fnv1a_u16": fnv1a_u16(pre_out_proj_bits),
                "out_proj_hash_fnv1a_u16": fnv1a_u16(out_proj_bits),
                "manual_dot_semantics": "selected output rows, BF16 input/weight values, sequential FP32 multiply-add",
                "details": details,
                "block_type": block_type,
            }
        )

        contribution_rows = []
        for idx in selected_element_dims:
            if idx >= len(out_proj_bits) or idx >= weight_tensor.shape[0]:
                continue
            weight_row = weight_tensor[idx].detach().contiguous()
            weight_bits = bf16_u16_values(weight_row)
            weight_values = weight_row.to(torch.float32).cpu().reshape(-1).tolist()
            manual_dot = f32_seq_dot(pre_out_proj_values, weight_values)
            contribution_rows.append(
                {
                    "output_dim": int(idx),
                    "out_proj_output_bits": int(out_proj_bits[idx]),
                    "out_proj_output_value": float(out_proj_row[idx].to(torch.float32).item()),
                    "out_proj_weight_row_hash_fnv1a_u16": fnv1a_u16(weight_bits),
                    "out_proj_weight_bits_u16": [int(v) & 0xFFFF for v in weight_bits],
                    "out_proj_manual_seq_fp32_dot": float(manual_dot),
                    "out_proj_manual_minus_production": float(
                        manual_dot - float(out_proj_row[idx].to(torch.float32).item())
                    ),
                }
            )

        summaries.append(
            {
                "index": start_index + len(summaries),
                "layer": 0,
                "label": "layer0_mamba2_out_proj_contribution_rows",
                "source": "hf_layer0_mamba2_full_pre_out_proj_selected_weight_rows",
                "dtype": "torch.bfloat16",
                "selected_output_dims": selected_element_dims,
                "pre_out_proj_width": int(pre_out_proj_row.numel()),
                "out_proj_width": int(out_proj_row.numel()),
                "out_proj_weight_shape": [int(v) for v in weight_tensor.shape],
                "pre_out_proj_hash_fnv1a_u16": fnv1a_u16(pre_out_proj_bits),
                "pre_out_proj_bits_u16": [int(v) & 0xFFFF for v in pre_out_proj_bits],
                "manual_dot_semantics": "selected output rows, BF16 input/weight values, sequential FP32 multiply-add",
                "rows": contribution_rows,
                "row_count": len(contribution_rows),
                "block_type": block_type,
            }
        )

        if selected_row_indices:
            pre_rows = tensors["pre_out_proj"].reshape(-1, tensors["pre_out_proj"].shape[-1])
            out_rows = tensors["out_proj"].reshape(-1, tensors["out_proj"].shape[-1])
            row_count = int(pre_rows.shape[0])
            weight_row_hashes = [
                fnv1a_u16(bf16_u16_values(weight_tensor[out_idx].detach().contiguous()))
                for out_idx in range(int(weight_tensor.shape[0]))
            ]
            selected_rows = []
            for row_index in selected_row_indices:
                if row_index >= row_count:
                    selected_rows.append(
                        {
                            "available": False,
                            "row_index": int(row_index),
                            "row_count": row_count,
                            "reason": "row index out of range",
                        }
                    )
                    continue
                pre_row = pre_rows[row_index].detach().contiguous()
                out_row = out_rows[row_index].detach().contiguous()
                pre_bits = bf16_u16_values(pre_row)
                out_bits = bf16_u16_values(out_row)
                pre_values = pre_row.to(torch.float32).cpu().reshape(-1).tolist()
                projection_details = []
                for idx in selected_element_dims:
                    if idx >= len(out_bits) or idx >= weight_tensor.shape[0]:
                        continue
                    weight_row = weight_tensor[idx].detach().contiguous()
                    weight_bits = bf16_u16_values(weight_row)
                    weight_values = weight_row.to(torch.float32).cpu().reshape(-1).tolist()
                    manual_dot = f32_seq_dot(pre_values, weight_values)
                    max_abs_contrib_index = 0
                    max_abs_contrib = -1.0
                    for contrib_idx, (hidden_v, weight_v) in enumerate(zip(pre_values, weight_values)):
                        contrib = abs(hidden_v * weight_v)
                        if contrib > max_abs_contrib:
                            max_abs_contrib = contrib
                            max_abs_contrib_index = contrib_idx
                    projection_details.append(
                        {
                            "output_dim": int(idx),
                            "out_proj_output_bits": int(out_bits[idx]),
                            "out_proj_output_value": float(out_row[idx].to(torch.float32).item()),
                            "out_proj_weight_row_hash_fnv1a_u16": fnv1a_u16(weight_bits),
                            "out_proj_manual_seq_fp32_dot": float(manual_dot),
                            "out_proj_manual_minus_production": float(
                                manual_dot - float(out_row[idx].to(torch.float32).item())
                            ),
                            "out_proj_max_abs_contrib_index": int(max_abs_contrib_index),
                            "out_proj_max_abs_contrib_input": float(pre_values[max_abs_contrib_index]),
                            "out_proj_max_abs_contrib_weight": float(weight_values[max_abs_contrib_index]),
                        }
                    )
                selected_rows.append(
                    {
                        "available": True,
                        "row_index": int(row_index),
                        "row_count": row_count,
                        "pre_out_proj_hash_fnv1a_u16": fnv1a_u16(pre_bits),
                        "out_proj_hash_fnv1a_u16": fnv1a_u16(out_bits),
                        "pre_out_proj_bits_u16": [int(v) & 0xFFFF for v in pre_bits],
                        "out_proj_bits_u16": [int(v) & 0xFFFF for v in out_bits],
                        "projection_detail_count": len(projection_details),
                        "projection_details": projection_details,
                    }
                )
            summaries.append(
                {
                    "index": start_index + len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_out_proj_selected_row_full_details",
                    "source": "hf_layer0_mamba2_selected_rows_full_out_proj_provenance",
                    "dtype": "torch.bfloat16",
                    "selected_rows": selected_row_indices,
                    "selected_output_dims": selected_element_dims,
                    "pre_out_proj_width": int(pre_rows.shape[-1]),
                    "out_proj_width": int(out_rows.shape[-1]),
                    "out_proj_weight_shape": [int(v) for v in weight_tensor.shape],
                    "out_proj_weight_row_hashes_fnv1a_u16": weight_row_hashes,
                    "weight_layout": "bf16 row-major [out_dim,input_dim]",
                    "bias": None,
                    "manual_dot_semantics": "selected output rows, BF16 input/weight values, sequential FP32 multiply-add",
                    "rows": selected_rows,
                    "row_count": len(selected_rows),
                    "block_type": block_type,
                }
            )

    base_model = getattr(model, "model", None) or getattr(model, "backbone", None)
    embeddings = getattr(base_model, "embeddings", None) if base_model is not None else None
    if embeddings is None:
        unavailable(
            "layer0_mamba2_input",
            "hf_layer0_mamba2_manual_diagnostic",
            "base model does not expose embeddings",
        )
        return summaries

    with torch.no_grad():
        embed_device = next(embeddings.parameters()).device
        input_ids = prompt_input_ids.to(embed_device)
        hidden_states = embeddings(input_ids)
        norm_input = hidden_states.to(dtype=block.norm.weight.dtype)
        norm_output = block.norm(norm_input)
        attention_mask = forward_kwargs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(norm_output.device)
            if not torch.all(attention_mask == 1):
                norm_output = (norm_output * attention_mask[:, :, None]).to(norm_output.dtype)

        record("layer0_mamba2_input_hidden", "hf_layer0_mamba2_manual_norm_output", norm_output)
        projected_states = mixer.in_proj(norm_output)
        record("layer0_mamba2_in_proj", "hf_layer0_mamba2_manual_in_proj", projected_states)

        batch_size, seq_len, _ = norm_output.shape
        groups_time_state_size = mixer.n_groups * mixer.ssm_state_size
        d_to_remove = (
            2 * mixer.intermediate_size
            + 2 * mixer.n_groups * mixer.ssm_state_size
            + mixer.num_heads
        )
        d_mlp = (projected_states.shape[-1] - d_to_remove) // 2
        split_projection_dim = [
            d_mlp,
            d_mlp,
            mixer.intermediate_size,
            mixer.conv_dim,
            mixer.num_heads,
        ]
        _, _, gate, hidden_states_b_c, dt = torch.split(
            projected_states,
            split_projection_dim,
            dim=-1,
        )
        hidden_raw, b_raw, c_raw = torch.split(
            hidden_states_b_c,
            [mixer.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )
        record("layer0_mamba2_raw_gate", "hf_layer0_mamba2_manual_split", gate)
        record("layer0_mamba2_conv_input_xbc", "hf_layer0_mamba2_manual_split", hidden_states_b_c)
        record("layer0_mamba2_raw_x", "hf_layer0_mamba2_manual_split", hidden_raw)
        record("layer0_mamba2_raw_b", "hf_layer0_mamba2_manual_split", b_raw)
        record("layer0_mamba2_raw_c", "hf_layer0_mamba2_manual_split", c_raw)
        record("layer0_mamba2_raw_dt", "hf_layer0_mamba2_manual_split", dt)

        dt_bias_param = mixer.dt_bias.to(dt.device)
        dt_bias = dt_bias_param.to(dtype=dt.dtype)
        dt_plus_bias = dt + dt_bias
        dt_softplus = torch.nn.functional.softplus(dt_plus_bias)
        dt_plus_bias_fp32 = dt.float() + dt_bias_param.float()
        dt_softplus_fp32_log1p_exp = torch.log1p(torch.exp(dt_plus_bias_fp32))
        dt_softplus_fp32_torch = torch.nn.functional.softplus(dt_plus_bias_fp32)
        record("layer0_mamba2_dt_bias", "hf_layer0_mamba2_manual_dt_path", dt_bias)
        record("layer0_mamba2_dt_plus_bias", "hf_layer0_mamba2_manual_dt_path", dt_plus_bias)
        record("layer0_mamba2_dt_softplus", "hf_layer0_mamba2_manual_dt_path", dt_softplus)
        record(
            "layer0_mamba2_dt_plus_bias_fp32",
            "hf_layer0_mamba2_manual_dt_path_fp32",
            dt_plus_bias_fp32,
        )
        record(
            "layer0_mamba2_dt_softplus_fp32_log1p_exp",
            "hf_layer0_mamba2_manual_dt_path_fp32",
            dt_softplus_fp32_log1p_exp,
        )
        record(
            "layer0_mamba2_dt_softplus_fp32_torch",
            "hf_layer0_mamba2_manual_dt_path_fp32",
            dt_softplus_fp32_torch,
        )

        if causal_conv1d_fn is None or mixer.activation not in ["silu", "swish"]:
            hidden_states_b_c = mixer.act(
                mixer.conv1d(hidden_states_b_c.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
            )
            conv_source = "hf_layer0_mamba2_manual_torch_conv1d"
        else:
            hidden_states_b_c = causal_conv1d_fn(
                x=hidden_states_b_c.transpose(1, 2),
                weight=mixer.conv1d.weight.squeeze(1),
                bias=mixer.conv1d.bias,
                activation=mixer.activation,
            ).transpose(1, 2)[:, :seq_len]
            conv_source = "hf_layer0_mamba2_manual_causal_conv1d_fn"

        hidden_conv, b_conv, c_conv = torch.split(
            hidden_states_b_c,
            [mixer.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )
        record("layer0_mamba2_conv_xbc", conv_source, hidden_states_b_c)
        record("layer0_mamba2_conv_x", conv_source, hidden_conv)
        record("layer0_mamba2_conv_b", conv_source, b_conv)
        record("layer0_mamba2_conv_c", conv_source, c_conv)

        a = -torch.exp(mixer.A_log.float())
        record_flat("layer0_mamba2_a_val", "hf_layer0_mamba2_manual_scan_params", a)
        record_flat("layer0_mamba2_d_val", "hf_layer0_mamba2_manual_scan_params", mixer.D)
        dt_limit_kwargs = {} if mixer.time_step_limit is None else {"dt_limit": mixer.time_step_limit}
        dA_seq = dt_softplus_fp32_torch * a.view(1, 1, -1)
        dA_cumsum_seq = torch.cumsum(dA_seq, dim=1)
        record("layer0_mamba2_da_last", "hf_layer0_mamba2_manual_scan_cumsum", dA_seq)
        record(
            "layer0_mamba2_da_cumsum_last",
            "hf_layer0_mamba2_manual_scan_cumsum",
            dA_cumsum_seq,
        )
        record(
            "layer0_mamba2_decay_last",
            "hf_layer0_mamba2_manual_scan_cumsum",
            torch.exp(dA_seq),
        )
        initial_state = torch.zeros(
            (
                batch_size,
                mixer.num_heads,
                mixer.head_dim,
                mixer.ssm_state_size,
            ),
            device=norm_output.device,
            dtype=torch.float32,
        )
        record_flat(
            "layer0_mamba2_ssm_state_initial_zero",
            "hf_layer0_mamba2_manual_ssd_initial_state",
            initial_state,
        )
        scan_output, ssm_state = scan_fn(
            hidden_conv.view(batch_size, seq_len, -1, mixer.head_dim),
            dt,
            a,
            b_conv.view(batch_size, seq_len, mixer.n_groups, -1),
            c_conv.view(batch_size, seq_len, mixer.n_groups, -1),
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            z=None,
            seq_idx=None,
            return_final_states=True,
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
            **dt_limit_kwargs,
        )
        scan_output_flat = scan_output.view(batch_size, seq_len, -1)
        x_scan = hidden_conv.view(batch_size, seq_len, -1, mixer.head_dim)
        d_x = (x_scan.float() * mixer.D.float().view(1, 1, -1, 1)).reshape(
            batch_size,
            seq_len,
            -1,
        )
        record("layer0_mamba2_d_x_last", "hf_layer0_mamba2_manual_scan_components", d_x)
        c_state_contrib = scan_output_flat.float() - d_x
        record(
            "layer0_mamba2_c_state_contrib_last",
            "hf_layer0_mamba2_manual_scan_components",
            c_state_contrib,
        )
        record(
            "layer0_mamba2_y_pre_bf16_last",
            "hf_layer0_mamba2_manual_scan_components",
            scan_output_flat,
        )
        b_repeated = b_conv.view(batch_size, seq_len, mixer.n_groups, -1).repeat_interleave(
            mixer.num_heads // mixer.n_groups,
            dim=2,
        )
        selected_positions = sorted(
            {
                0,
                min(1, seq_len - 1),
                seq_len // 2,
                max(seq_len - 2, 0),
                seq_len - 1,
            }
        )
        x_scan_f32 = x_scan.float()
        b_repeated_f32 = b_repeated.float()
        dt_actual_f32 = dt_softplus_fp32_torch.float()
        state_fp32 = torch.zeros(
            (
                batch_size,
                mixer.num_heads,
                mixer.head_dim,
                mixer.ssm_state_size,
            ),
            device=norm_output.device,
            dtype=torch.float32,
        )
        final_da = dA_cumsum_seq[:, -1, :].float()
        chunk_formula_final_fp32 = torch.zeros_like(state_fp32)
        chunk_formula_final_bf16_bscale = torch.zeros_like(state_fp32)
        for pos in range(seq_len):
            decay = torch.exp(
                a.float().view(1, -1, 1, 1)
                * dt_actual_f32[:, pos, :, None, None]
            )
            update_fp32 = (
                b_repeated_f32[:, pos, :, None, :]
                * dt_actual_f32[:, pos, :, None, None]
                * x_scan_f32[:, pos, :, :, None]
            )
            update_bf16_bdt = (
                (
                    b_repeated_f32[:, pos, :, None, :]
                    * dt_actual_f32[:, pos, :, None, None]
                )
                .to(x_scan.dtype)
                .float()
                * x_scan_f32[:, pos, :, :, None]
            )
            post_decay = decay * state_fp32
            post_fp32 = post_decay + update_fp32
            post_bf16_update = post_decay + update_bf16_bdt

            scale_to_final = (
                torch.exp(
                    torch.minimum(
                        final_da - dA_cumsum_seq[:, pos, :].float(),
                        torch.zeros_like(final_da),
                    )
                )
                * dt_actual_f32[:, pos, :]
            )
            final_contrib_fp32 = (
                b_repeated_f32[:, pos, :, None, :]
                * scale_to_final[:, :, None, None]
                * x_scan_f32[:, pos, :, :, None]
            )
            final_contrib_bf16_bscale = (
                (
                    b_repeated_f32[:, pos, :, None, :]
                    * scale_to_final[:, :, None, None]
                )
                .to(x_scan.dtype)
                .float()
                * x_scan_f32[:, pos, :, :, None]
            )
            chunk_formula_final_fp32 = chunk_formula_final_fp32 + final_contrib_fp32
            chunk_formula_final_bf16_bscale = (
                chunk_formula_final_bf16_bscale + final_contrib_bf16_bscale
            )

            if pos in selected_positions:
                record_flat(
                    f"layer0_mamba2_state_pre_pos{pos}",
                    "hf_layer0_mamba2_manual_state_recurrence_fp32",
                    state_fp32,
                )
                record_flat(
                    f"layer0_mamba2_state_decay_pos{pos}",
                    "hf_layer0_mamba2_manual_state_recurrence_fp32",
                    decay.expand_as(state_fp32),
                )
                record_flat(
                    f"layer0_mamba2_state_post_decay_pos{pos}",
                    "hf_layer0_mamba2_manual_state_recurrence_fp32",
                    post_decay,
                )
                record_flat(
                    f"layer0_mamba2_state_update_fp32_pos{pos}",
                    "hf_layer0_mamba2_manual_state_recurrence_fp32",
                    update_fp32,
                )
                record_flat(
                    f"layer0_mamba2_state_update_bf16_bdt_pos{pos}",
                    "hf_layer0_mamba2_manual_state_cast_candidate",
                    update_bf16_bdt,
                )
                record_flat(
                    f"layer0_mamba2_state_post_fp32_pos{pos}",
                    "hf_layer0_mamba2_manual_state_recurrence_fp32",
                    post_fp32,
                )
                record_flat(
                    f"layer0_mamba2_state_post_bf16_update_pos{pos}",
                    "hf_layer0_mamba2_manual_state_cast_candidate",
                    post_bf16_update,
                )
                record_flat(
                    f"layer0_mamba2_final_contrib_fp32_pos{pos}",
                    "hf_layer0_mamba2_manual_chunk_formula_fp32",
                    final_contrib_fp32,
                )
                record_flat(
                    f"layer0_mamba2_final_contrib_bf16_bscale_pos{pos}",
                    "hf_layer0_mamba2_manual_chunk_formula_cast_candidate",
                    final_contrib_bf16_bscale,
                )

            state_fp32 = post_fp32

        record_flat(
            "layer0_mamba2_chunk_formula_final_fp32",
            "hf_layer0_mamba2_manual_chunk_formula_fp32",
            chunk_formula_final_fp32,
        )
        record_flat(
            "layer0_mamba2_chunk_formula_final_bf16_bscale",
            "hf_layer0_mamba2_manual_chunk_formula_cast_candidate",
            chunk_formula_final_bf16_bscale,
        )
        c_repeated = c_conv.view(batch_size, seq_len, mixer.n_groups, -1).repeat_interleave(
            mixer.num_heads // mixer.n_groups,
            dim=2,
        )
        c_repeated_last = c_repeated[:, -1, :, :]
        b_dt_x_update_last = (
            b_repeated[:, -1, :, None, :].float()
            * dt_softplus_fp32_torch[:, -1, :, None, None]
            * x_scan[:, -1, :, :, None].float()
        )
        record(
            "layer0_mamba2_b_dt_x_update_last",
            "hf_layer0_mamba2_manual_scan_components",
            b_dt_x_update_last.reshape(batch_size, 1, -1),
        )
        c_state_terms_last = c_repeated_last[:, :, None, :].float() * ssm_state.float()
        record(
            "layer0_mamba2_c_state_terms_last",
            "hf_layer0_mamba2_manual_scan_components",
            c_state_terms_last.reshape(batch_size, 1, -1),
        )
        c_last = c_repeated_last.float()
        dA_target = dA_cumsum_seq[:, -1, :].float()
        chunk_scan_c_state_fp32_cbscale = torch.zeros_like(x_scan_f32[:, -1, :, :])
        chunk_scan_c_state_bf16_cbscale = torch.zeros_like(x_scan_f32[:, -1, :, :])
        for pos in range(seq_len):
            cb = (c_last * b_repeated_f32[:, pos, :, :]).sum(dim=-1)
            scale = (
                torch.exp(
                    torch.minimum(
                        dA_target - dA_cumsum_seq[:, pos, :].float(),
                        torch.zeros_like(dA_target),
                    )
                )
                * dt_actual_f32[:, pos, :]
            )
            cb_scaled = cb * scale
            cb_scaled_bf16 = cb_scaled.to(x_scan.dtype).float()
            x_pos = x_scan_f32[:, pos, :, :]
            chunk_scan_c_state_fp32_cbscale = (
                chunk_scan_c_state_fp32_cbscale + cb_scaled[:, :, None] * x_pos
            )
            chunk_scan_c_state_bf16_cbscale = (
                chunk_scan_c_state_bf16_cbscale + cb_scaled_bf16[:, :, None] * x_pos
            )
        d_x_last = d_x.view(batch_size, seq_len, mixer.num_heads, mixer.head_dim)[:, -1, :, :]
        chunk_scan_y_fp32_cbscale = d_x_last + chunk_scan_c_state_fp32_cbscale
        chunk_scan_y_bf16_cbscale = d_x_last + chunk_scan_c_state_bf16_cbscale
        record(
            "layer0_mamba2_chunk_scan_c_state_fp32_cbscale_last",
            "hf_layer0_mamba2_manual_chunk_scan_emission_fp32_cbscale",
            chunk_scan_c_state_fp32_cbscale.reshape(batch_size, 1, -1),
        )
        record(
            "layer0_mamba2_chunk_scan_c_state_bf16_cbscale_last",
            "hf_layer0_mamba2_manual_chunk_scan_emission_bf16_cbscale",
            chunk_scan_c_state_bf16_cbscale.reshape(batch_size, 1, -1),
        )
        record(
            "layer0_mamba2_chunk_scan_y_fp32_cbscale_last",
            "hf_layer0_mamba2_manual_chunk_scan_emission_fp32_cbscale",
            chunk_scan_y_fp32_cbscale.reshape(batch_size, 1, -1),
        )
        record(
            "layer0_mamba2_chunk_scan_y_bf16_cbscale_last",
            "hf_layer0_mamba2_manual_chunk_scan_emission_bf16_cbscale",
            chunk_scan_y_bf16_cbscale.reshape(batch_size, 1, -1),
        )
        record(
            "layer0_mamba2_chunk_scan_y_bf16_cbscale_store_last",
            "hf_layer0_mamba2_manual_chunk_scan_emission_bf16_cbscale",
            chunk_scan_y_bf16_cbscale.to(x_scan.dtype).reshape(batch_size, 1, -1),
        )
        if selected_element_dims:
            scan_output_row = (
                scan_output_flat.reshape(-1, scan_output_flat.shape[-1])[-1]
                .detach()
                .contiguous()
            )
            x_last_flat = (
                x_scan.reshape(batch_size, seq_len, -1)[:, -1, :]
                .reshape(-1)
                .detach()
                .contiguous()
            )
            dt_last_row = dt[:, -1, :].reshape(-1).detach().contiguous()
            scan_output_bits = bf16_u16_values(scan_output_row)
            x_last_bits = bf16_u16_values(x_last_flat)
            dt_last_bits = bf16_u16_values(dt_last_row)

            width = int(scan_output_row.numel())
            effective_chunk_size = int(getattr(mixer, "chunk_size", 0) or seq_len)
            if effective_chunk_size <= 0:
                effective_chunk_size = seq_len
            row_index = seq_len - 1
            chunk_start = (row_index // effective_chunk_size) * effective_chunk_size
            details = []
            for idx in selected_element_dims:
                item: Dict[str, Any] = {"dim_index": int(idx)}
                if idx < 0 or idx >= width:
                    item.update({"available": False, "reason": "dim out of range"})
                    details.append(item)
                    continue

                head = int(idx // mixer.head_dim)
                head_dim_index = int(idx - head * mixer.head_dim)
                group = int(head // (mixer.num_heads // mixer.n_groups))
                x_last_value = float(x_scan_f32[0, row_index, head, head_dim_index].item())
                dt_raw_value = float(dt[0, row_index, head].to(torch.float32).item())
                dt_softplus_value = float(dt_actual_f32[0, row_index, head].item())
                d_x_value = float(d_x_last[0, head, head_dim_index].item())
                local_hf_chunk_scan = float(
                    chunk_scan_c_state_bf16_cbscale[0, head, head_dim_index].item()
                )
                y_pre_store = float(
                    chunk_scan_y_bf16_cbscale[0, head, head_dim_index].item()
                )
                y_pre_store_bf16_candidate = torch.tensor(
                    [y_pre_store], device=x_scan.device, dtype=torch.float32
                ).to(x_scan.dtype)
                y_pre_store_bf16_candidate_bits = int(
                    bf16_u16_values(y_pre_store_bf16_candidate)[0]
                )
                stored_value = float(scan_output_row[idx].to(torch.float32).item())
                c_vec_last = c_repeated[0, row_index, head, :].float()
                c_state_total = float(
                    (c_vec_last * ssm_state.float()[0, head, head_dim_index, :]).sum().item()
                )
                local_old_state = 0.0
                local_scan_forward = 0.0
                local_scan_kahan = 0.0
                local_scan_kahan_c = 0.0
                local_scan_fp32_cbscale = 0.0
                local_scan_token_details = []
                dA_target_value = float(dA_cumsum_seq[0, row_index, head].float().item())
                dA_chunk_base_value = (
                    float(dA_cumsum_seq[0, chunk_start - 1, head].float().item())
                    if chunk_start > 0
                    else 0.0
                )
                zero = torch.zeros((), device=dA_cumsum_seq.device, dtype=torch.float32)
                for pos in range(chunk_start, row_index + 1):
                    raw_dt_tensor = dt[0, pos, head].reshape(1)
                    raw_dt_value = float(raw_dt_tensor.float()[0].item())
                    dt_bias_value = float(dt_bias_param[head].float().item())
                    dt_plus_bias_fp32_value = raw_dt_value + dt_bias_value
                    dt_plus_bias_bf16_tensor = torch.tensor(
                        [dt_plus_bias_fp32_value],
                        device=x_scan.device,
                        dtype=torch.float32,
                    ).to(x_scan.dtype)
                    dt_plus_bias_bf16_value = float(
                        dt_plus_bias_bf16_tensor.float()[0].item()
                    )
                    softplus_bf16_plus_tensor = torch.nn.functional.softplus(
                        dt_plus_bias_bf16_tensor
                    )
                    softplus_bf16_plus_value = float(
                        softplus_bf16_plus_tensor.float()[0].item()
                    )
                    dt_softplus_log1p_exp_value = float(
                        dt_softplus_fp32_log1p_exp[0, pos, head].float().item()
                    )
                    dt_pos = float(dt_actual_f32[0, pos, head].item())
                    dA_pos = float(dA_cumsum_seq[0, pos, head].float().item())
                    decay = float(
                        torch.exp(
                            torch.minimum(
                                dA_cumsum_seq[0, row_index, head].float()
                                - dA_cumsum_seq[0, pos, head].float(),
                                zero,
                            )
                        ).item()
                    )
                    old_state_source = float(
                        (
                            c_vec_last
                            * (b_repeated_f32[0, pos, head, :] * dt_pos)
                            .to(x_scan.dtype)
                            .float()
                        )
                        .sum()
                        .item()
                    )
                    x_pos = float(x_scan_f32[0, pos, head, head_dim_index].item())
                    local_old_state += old_state_source * decay * x_pos
                    cb_terms = c_vec_last * b_repeated_f32[0, pos, head, :].float()
                    cb_forward = float(cb_terms.sum().item())
                    cb_reverse = float(torch.flip(cb_terms, dims=[0]).sum().item())
                    top_state_index = int(torch.argmax(torch.abs(cb_terms)).item())
                    top_state_contrib = float(cb_terms[top_state_index].item())
                    scale = decay * dt_pos
                    scale_log1p_exp = decay * dt_softplus_log1p_exp_value
                    scale_bf16_plus = decay * softplus_bf16_plus_value
                    cb_scaled_fp32 = cb_forward * scale
                    cb_scaled_bf16_tensor = torch.tensor(
                        [cb_scaled_fp32], device=x_scan.device, dtype=torch.float32
                    ).to(x_scan.dtype)
                    cb_scaled_bf16 = float(cb_scaled_bf16_tensor.float()[0].item())
                    cb_scaled_bf16_bits = int(bf16_u16_values(cb_scaled_bf16_tensor)[0])
                    cb_scaled_log1p_exp_bf16_tensor = torch.tensor(
                        [cb_forward * scale_log1p_exp],
                        device=x_scan.device,
                        dtype=torch.float32,
                    ).to(x_scan.dtype)
                    cb_scaled_bf16_plus_bf16_tensor = torch.tensor(
                        [cb_forward * scale_bf16_plus],
                        device=x_scan.device,
                        dtype=torch.float32,
                    ).to(x_scan.dtype)
                    term_bf16_cbscale = cb_scaled_bf16 * x_pos
                    term_fp32_cbscale = cb_scaled_fp32 * x_pos
                    local_scan_forward += term_bf16_cbscale
                    kahan_y = term_bf16_cbscale - local_scan_kahan_c
                    kahan_t = local_scan_kahan + kahan_y
                    local_scan_kahan_c = (kahan_t - local_scan_kahan) - kahan_y
                    local_scan_kahan = kahan_t
                    local_scan_fp32_cbscale += term_fp32_cbscale
                    local_scan_token_details.append(
                        {
                            "token_position": int(pos),
                            "x_bits": int(
                                bf16_u16_values(
                                    x_scan[0, pos, head, head_dim_index].reshape(1)
                                )[0]
                            ),
                            "dt_bits": int(bf16_u16_values(raw_dt_tensor)[0]),
                            "x_value": x_pos,
                            "raw_dt_value": raw_dt_value,
                            "dt_bias_value": dt_bias_value,
                            "dt_plus_bias_fp32": dt_plus_bias_fp32_value,
                            "dt_plus_bias_bf16_bits": int(
                                bf16_u16_values(dt_plus_bias_bf16_tensor)[0]
                            ),
                            "dt_plus_bias_bf16_value": dt_plus_bias_bf16_value,
                            "dt_softplus_value": dt_pos,
                            "dt_softplus_log1p_exp": dt_softplus_log1p_exp_value,
                            "dt_softplus_bf16_plus": softplus_bf16_plus_value,
                            "dA_cumsum": dA_pos,
                            "decay": decay,
                            "scale": scale,
                            "scale_log1p_exp": scale_log1p_exp,
                            "scale_bf16_plus": scale_bf16_plus,
                            "cb_forward": cb_forward,
                            "cb_reverse": cb_reverse,
                            "cb_scaled_fp32": cb_scaled_fp32,
                            "cb_scaled_bf16": cb_scaled_bf16,
                            "cb_scaled_bf16_bits": cb_scaled_bf16_bits,
                            "cb_scaled_log1p_exp_bf16_bits": int(
                                bf16_u16_values(cb_scaled_log1p_exp_bf16_tensor)[0]
                            ),
                            "cb_scaled_bf16_plus_bf16_bits": int(
                                bf16_u16_values(cb_scaled_bf16_plus_bf16_tensor)[0]
                            ),
                            "term_bf16_cbscale": term_bf16_cbscale,
                            "term_fp32_cbscale": term_fp32_cbscale,
                            "cumulative_after": local_scan_forward,
                            "top_state_index": top_state_index,
                            "top_state_contrib": top_state_contrib,
                        }
                    )
                prior_chunk_state = 0.0 if chunk_start == 0 else c_state_total - local_old_state

                c_bits = bf16_u16_values(
                    c_conv.view(batch_size, seq_len, mixer.n_groups, -1)[0, row_index, group, :]
                )
                b_bits = bf16_u16_values(
                    b_conv.view(batch_size, seq_len, mixer.n_groups, -1)[
                        0, chunk_start : row_index + 1, group, :
                    ]
                )
                x_chunk_bits = bf16_u16_values(
                    x_scan[0, chunk_start : row_index + 1, head, head_dim_index]
                )
                dt_chunk_bits = bf16_u16_values(dt[0, chunk_start : row_index + 1, head])

                item.update(
                    {
                        "available": True,
                        "head_index": head,
                        "head_dim_index": head_dim_index,
                        "group_index": group,
                        "width": width,
                        "stored_ssd_out_bits": int(scan_output_bits[idx]),
                        "stored_ssd_out_value": stored_value,
                        "x_last_bits": int(x_last_bits[idx]),
                        "x_last_value": x_last_value,
                        "dt_last_bits": int(dt_last_bits[head]),
                        "dt_raw_value": dt_raw_value,
                        "dt_softplus_value": dt_softplus_value,
                        "d_x": d_x_value,
                        "prior_chunk_state": float(prior_chunk_state),
                        "local_hf_chunk_scan": local_hf_chunk_scan,
                        "c_state_total": c_state_total,
                        "local_old_state": float(local_old_state),
                        "local_scan_summary": {
                            "local_scan_forward": float(local_scan_forward),
                            "local_scan_kahan": float(local_scan_kahan),
                            "local_scan_fp32_cbscale": float(local_scan_fp32_cbscale),
                        },
                        "local_scan_token_details": local_scan_token_details,
                        "a_value": float(a[head].float().item()),
                        "d_value": float(mixer.D[head].float().item()),
                        "y_pre_store": y_pre_store,
                        "y_pre_store_f32_bits": int(f32_bits(y_pre_store)),
                        "y_pre_store_bf16_candidate_bits": y_pre_store_bf16_candidate_bits,
                        "y_pre_store_bf16_candidate_value": float(
                            y_pre_store_bf16_candidate.float()[0].item()
                        ),
                        "stored_ssd_out_matches_y_pre_store_bf16_candidate": bool(
                            int(scan_output_bits[idx]) == y_pre_store_bf16_candidate_bits
                        ),
                        "chunk_start": int(chunk_start),
                        "row_index": int(row_index),
                        "effective_chunk_size": int(effective_chunk_size),
                        "dA_chunk_base": dA_chunk_base_value,
                        "dA_target": dA_target_value,
                        "c_last_group_hash_fnv1a_u16": fnv1a_u16(c_bits),
                        "b_chunk_group_hash_fnv1a_u16": fnv1a_u16(b_bits),
                        "x_chunk_dim_hash_fnv1a_u16": fnv1a_u16(x_chunk_bits),
                        "dt_chunk_head_hash_fnv1a_u16": fnv1a_u16(dt_chunk_bits),
                    }
                )
                details.append(item)

            summaries.append(
                {
                    "index": start_index + len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_ssd_output_element_details",
                    "source": "hf_layer0_mamba2_selected_dim_ssd_output_provenance",
                    "dtype": str(scan_output_row.dtype),
                    "selected_dims": selected_element_dims,
                    "detail_count": len(details),
                    "width": width,
                    "num_heads": int(mixer.num_heads),
                    "head_dim": int(mixer.head_dim),
                    "state_size": int(mixer.ssm_state_size),
                    "n_groups": int(mixer.n_groups),
                    "chunk_size": int(getattr(mixer, "chunk_size", 0) or 0),
                    "effective_chunk_size": int(effective_chunk_size),
                    "semantics": "HF chunk-scan SSD output selected dims before gated norm",
                    "details": details,
                    "block_type": block_type,
                }
            )
        record("layer0_mamba2_ssd_out", "hf_layer0_mamba2_manual_mamba_chunk_scan", scan_output_flat)
        record_flat(
            "layer0_mamba2_ssm_state_after_ssd",
            "hf_layer0_mamba2_manual_mamba_chunk_scan",
            ssm_state,
        )

        gated_norm = mixer.norm(scan_output_flat, gate)
        record("layer0_mamba2_gated_norm", "hf_layer0_mamba2_manual_gated_norm", gated_norm)
        out_proj_input = gated_norm.to(norm_output.dtype)
        append_gated_norm_element_details(scan_output_flat, gate, out_proj_input)
        out_proj = mixer.out_proj(out_proj_input)
        record("layer0_mamba2_out_proj", "hf_layer0_mamba2_manual_out_proj", out_proj)
        append_branch_element_details(norm_output, out_proj_input, out_proj)

        scan_output_bf16_dt, ssm_state_bf16_dt = scan_fn(
            hidden_conv.view(batch_size, seq_len, -1, mixer.head_dim),
            dt_softplus,
            a,
            b_conv.view(batch_size, seq_len, mixer.n_groups, -1),
            c_conv.view(batch_size, seq_len, mixer.n_groups, -1),
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            z=None,
            seq_idx=None,
            return_final_states=True,
            dt_bias=None,
            dt_softplus=False,
            **dt_limit_kwargs,
        )
        scan_output_bf16_dt = scan_output_bf16_dt.view(batch_size, seq_len, -1)
        record(
            "layer0_mamba2_ssd_out_bf16_dt_candidate",
            "hf_layer0_mamba2_manual_bf16_rounded_dt_candidate",
            scan_output_bf16_dt,
        )
        record_flat(
            "layer0_mamba2_ssm_state_after_ssd_bf16_dt_candidate",
            "hf_layer0_mamba2_manual_bf16_rounded_dt_candidate",
            ssm_state_bf16_dt,
        )
        gated_norm_bf16_dt = mixer.norm(scan_output_bf16_dt, gate)
        record(
            "layer0_mamba2_gated_norm_bf16_dt_candidate",
            "hf_layer0_mamba2_manual_bf16_rounded_dt_candidate",
            gated_norm_bf16_dt,
        )
        out_proj_bf16_dt = mixer.out_proj(gated_norm_bf16_dt.to(norm_output.dtype))
        record(
            "layer0_mamba2_out_proj_bf16_dt_candidate",
            "hf_layer0_mamba2_manual_bf16_rounded_dt_candidate",
            out_proj_bf16_dt,
        )

    return summaries


def _capture_hf_layer0_internal_summaries(
    model: Any,
    prompt_input_ids: Any,
    forward_kwargs: Dict[str, Any],
    element_dims: Optional[List[int]] = None,
    layer_idx: int = 0,
    row_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    import torch

    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative")
    layer_name = f"layer{layer_idx}"
    source_name = f"hf_layer{layer_idx}"

    def normalize_layer_string(value: str) -> str:
        if layer_idx == 0:
            return value
        return (
            value.replace("hf_layer0", source_name)
            .replace("layer-0", f"layer-{layer_idx}")
            .replace("layer 0", f"layer {layer_idx}")
            .replace("layer0", layer_name)
        )

    def normalize_layer_value(value: Any, key: Optional[str] = None) -> Any:
        if layer_idx == 0:
            return value
        if key == "layer" and value == 0:
            return layer_idx
        if isinstance(value, str):
            return normalize_layer_string(value)
        if isinstance(value, dict):
            return {
                item_key: normalize_layer_value(item_value, item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [normalize_layer_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(normalize_layer_value(item) for item in value)
        return value

    class LayerSummaryList(list):
        def append(self, item: Any) -> None:
            super().append(normalize_layer_value(item))

        def extend(self, items: Any) -> None:
            for item in items:
                self.append(item)

    block = _find_hf_block(model, layer_idx)
    summaries: List[Dict[str, Any]] = LayerSummaryList()
    handles: List[Any] = []
    block_type = getattr(block, "block_type", None)
    selected_element_dims = sorted({int(dim) for dim in (element_dims or []) if int(dim) >= 0})
    selected_row_indices = sorted({int(row) for row in (row_indices or []) if int(row) >= 0})
    layer0_element_cache: Dict[str, Any] = {}

    def fnv1a_u16(values: Any) -> str:
        h = 0xCBF29CE484222325
        for raw in values:
            v = int(raw) & 0xFFFF
            h ^= v & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            h ^= (v >> 8) & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f"0x{h:016x}"

    def bf16_u16_values(tensor_value: Any) -> List[int]:
        return [
            int(v) & 0xFFFF
            for v in tensor_value.detach()
            .contiguous()
            .view(torch.int16)
            .cpu()
            .reshape(-1)
            .tolist()
        ]

    def f32_bits(value: float) -> int:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]

    def cache_layer0_element_row(label: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        detached = tensor.detach()
        if detached.ndim >= 3:
            row = detached[0, -1, ...].reshape(-1).contiguous()
        elif detached.ndim == 2:
            row = detached[-1, :].reshape(-1).contiguous()
        else:
            row = detached.reshape(-1).contiguous()
        if row.dtype != torch.bfloat16:
            layer0_element_cache[label] = {
                "available": False,
                "reason": f"expected BF16 tensor, got {row.dtype}",
                "dtype": str(row.dtype),
            }
            return
        bits = bf16_u16_values(row)
        layer0_element_cache[label] = {
            "available": True,
            "dtype": str(row.dtype),
            "device": str(row.device),
            "width": int(row.numel()),
            "bits": bits,
            "values": row.to(torch.float32).cpu().reshape(-1).tolist(),
            "hash_fnv1a_u16": fnv1a_u16(bits),
        }

    def append_layer0_handoff_full_row_bits() -> None:
        residual = layer0_element_cache.get("layer0_input")
        branch = layer0_element_cache.get("layer0_mixer_output")
        output = layer0_element_cache.get("layer0_output")
        if not residual or not branch or not output:
            return
        for cached in (residual, branch, output):
            if not cached.get("available", False):
                return
        width = min(int(residual["width"]), int(branch["width"]), int(output["width"]))
        rounded_bits: List[int] = []
        for idx in range(width):
            sum_fp32 = struct.unpack(
                "<f",
                struct.pack(
                    "<f",
                    float(residual["values"][idx]) + float(branch["values"][idx]),
                ),
            )[0]
            rounded_bits.append(
                int(
                    bf16_u16_values(
                        torch.tensor([sum_fp32], dtype=torch.float32).to(torch.bfloat16)
                    )[0]
                )
            )
        summaries.append(
            {
                "index": len(summaries),
                "layer": 0,
                "label": "layer0_handoff_full_row_bits",
                "source": "hf_layer0_full_row_handoff_provenance",
                "dtype": str(output["dtype"]),
                "row_width": int(width),
                "residual_source": "hf_layer0_block_pre_hook",
                "branch_source": "hf_layer0_mixer_forward_hook",
                "output_source": "hf_layer0_block_forward_hook",
                "residual_hash_fnv1a_u16": residual["hash_fnv1a_u16"],
                "branch_hash_fnv1a_u16": branch["hash_fnv1a_u16"],
                "rounded_sum_hash_fnv1a_u16": fnv1a_u16(rounded_bits),
                "actual_output_hash_fnv1a_u16": output["hash_fnv1a_u16"],
                "residual_bits_u16": [int(v) & 0xFFFF for v in residual["bits"][:width]],
                "branch_bits_u16": [int(v) & 0xFFFF for v in branch["bits"][:width]],
                "rounded_sum_bits_u16": rounded_bits,
                "actual_output_bits_u16": [int(v) & 0xFFFF for v in output["bits"][:width]],
                "residual_in_fp32": bool(getattr(block, "residual_in_fp32", False)),
                "add_semantics": "hf_block_forward_residual_plus_mixer_output",
                "block_type": block_type,
            }
        )

    def append_layer0_element_details() -> None:
        if not selected_element_dims:
            return
        residual = layer0_element_cache.get("layer0_input")
        branch = layer0_element_cache.get("layer0_mixer_output")
        output = layer0_element_cache.get("layer0_output")
        if not residual or not branch or not output:
            missing = [
                label
                for label in ("layer0_input", "layer0_mixer_output", "layer0_output")
                if label not in layer0_element_cache
            ]
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": 0,
                    "label": "layer0_handoff_element_details",
                    "source": "hf_layer0_selected_dim_provenance",
                    "available": False,
                    "reason": f"missing cached rows: {missing}",
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )
            return
        for label, cached in (
            ("layer0_input", residual),
            ("layer0_mixer_output", branch),
            ("layer0_output", output),
        ):
            if not cached.get("available", False):
                summaries.append(
                    {
                        "index": len(summaries),
                        "layer": 0,
                        "label": "layer0_handoff_element_details",
                        "source": "hf_layer0_selected_dim_provenance",
                        "available": False,
                        "reason": f"{label}: {cached.get('reason', 'unavailable')}",
                        "selected_dims": selected_element_dims,
                        "block_type": block_type,
                    }
                )
                return
        width = min(int(residual["width"]), int(branch["width"]), int(output["width"]))
        details = []
        for idx in selected_element_dims:
            if idx >= width:
                continue
            residual_value = float(residual["values"][idx])
            branch_value = float(branch["values"][idx])
            sum_fp32 = struct.unpack("<f", struct.pack("<f", residual_value + branch_value))[0]
            rounded_value = (
                torch.tensor([sum_fp32], dtype=torch.float32)
                .to(torch.bfloat16)
                .to(torch.float32)
                .item()
            )
            rounded_bits = bf16_u16_values(torch.tensor([sum_fp32], dtype=torch.float32).to(torch.bfloat16))[0]
            actual_bits = int(output["bits"][idx])
            actual_value = float(output["values"][idx])
            details.append(
                {
                    "dim_index": int(idx),
                    "residual_bits": int(residual["bits"][idx]),
                    "residual_value": residual_value,
                    "branch_bits": int(branch["bits"][idx]),
                    "branch_value": branch_value,
                    "sum_fp32": float(sum_fp32),
                    "rounded_sum_bits": int(rounded_bits),
                    "rounded_sum_value": float(rounded_value),
                    "actual_output_bits": actual_bits,
                    "actual_output_value": actual_value,
                    "rounded_sum_matches_actual_output": rounded_bits == actual_bits,
                }
            )
        summaries.append(
            {
                "index": len(summaries),
                "layer": 0,
                "label": "layer0_handoff_element_details",
                "source": "hf_layer0_selected_dim_provenance",
                "dtype": str(output["dtype"]),
                "row_width": int(width),
                "selected_dims": selected_element_dims,
                "detail_count": len(details),
                "residual_source": "hf_layer0_block_pre_hook",
                "branch_source": "hf_layer0_mixer_forward_hook",
                "output_source": "hf_layer0_block_forward_hook",
                "residual_hash_fnv1a_u16": residual["hash_fnv1a_u16"],
                "branch_hash_fnv1a_u16": branch["hash_fnv1a_u16"],
                "actual_output_hash_fnv1a_u16": output["hash_fnv1a_u16"],
                "residual_in_fp32": bool(getattr(block, "residual_in_fp32", False)),
                "add_semantics": "hf_block_forward_residual_plus_mixer_output",
                "details": details,
                "block_type": block_type,
            }
        )

    def record(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": 0,
                    "label": label,
                    "source": source,
                    "available": False,
                    "reason": "no tensor value",
                    "block_type": block_type,
                }
            )
            return
        summary = _tensor_last_token_summary(
            tensor,
            len(summaries),
            label=label,
            source=source,
            layer=0,
        )
        summary["block_type"] = block_type
        summaries.append(summary)

    def record_flat(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable(label, source, "no tensor value")
            return
        summary = _tensor_flat_summary(
            tensor,
            len(summaries),
            label=label,
            source=source,
            layer=layer_idx,
        )
        summary["block_type"] = block_type
        summaries.append(summary)

    def record_flat(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable(label, source, "no tensor value")
            return
        summary = _tensor_flat_summary(
            tensor,
            len(summaries),
            label=label,
            source=source,
            layer=layer_idx,
        )
        summary["block_type"] = block_type
        summaries.append(summary)

    forward_chunk_cumsum_capture: Dict[str, Any] = {}
    mamba_forward_active = {"active": False}

    def append_actual_chunk_scan_store_cast_details(
        cb: Any,
        x: Any,
        dt: Any,
        dA_cumsum: Any,
        C: Any,
        states: Any,
        D: Any,
        z: Any,
        seq_idx: Any,
        out: Any,
    ) -> None:
        if not selected_element_dims or forward_original_chunk_scan is None:
            return
        source = "hf_layer0_mamba2_actual_chunk_scan_store_cast_boundary"
        kernel_globals = getattr(forward_original_chunk_scan, "__globals__", {})
        triton_mod = kernel_globals.get("triton")
        kernel = kernel_globals.get("_chunk_scan_fwd_kernel")
        triton_22 = bool(kernel_globals.get("TRITON_22", False))
        if triton_mod is None or kernel is None:
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_actual_chunk_scan_store_cast_details",
                    "source": source,
                    "available": False,
                    "reason": "missing Triton chunk-scan kernel globals",
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )
            return

        try:
            batch, seqlen, nheads, headdim = x.shape
            _, _, ngroups, dstate = C.shape
            _, _, nchunks, chunk_size = dt.shape
            out_fp32 = torch.empty(
                batch,
                seqlen,
                nheads,
                headdim,
                device=x.device,
                dtype=torch.float32,
            )
            out_x_fp32 = (
                torch.empty_like(out_fp32)
                if z is not None
                else None
            )
            grid = lambda meta: (  # noqa: E731 - mirrors upstream Triton launch style.
                triton_mod.cdiv(chunk_size, meta["BLOCK_SIZE_M"])
                * triton_mod.cdiv(headdim, meta["BLOCK_SIZE_N"]),
                batch * nchunks,
                nheads,
            )
            z_strides = (
                (z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                if z is not None
                else (0, 0, 0, 0)
            )
            kernel[grid](
                cb,
                x,
                z,
                out_fp32,
                out_x_fp32,
                dt,
                dA_cumsum,
                seq_idx,
                C,
                states,
                D,
                chunk_size,
                headdim,
                dstate,
                batch,
                seqlen,
                nheads // ngroups,
                cb.stride(0),
                cb.stride(1),
                cb.stride(2),
                cb.stride(3),
                cb.stride(4),
                x.stride(0),
                x.stride(1),
                x.stride(2),
                x.stride(3),
                z_strides[0],
                z_strides[1],
                z_strides[2],
                z_strides[3],
                out_fp32.stride(0),
                out_fp32.stride(1),
                out_fp32.stride(2),
                out_fp32.stride(3),
                dt.stride(0),
                dt.stride(2),
                dt.stride(1),
                dt.stride(3),
                dA_cumsum.stride(0),
                dA_cumsum.stride(2),
                dA_cumsum.stride(1),
                dA_cumsum.stride(3),
                *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
                C.stride(0),
                C.stride(1),
                C.stride(2),
                C.stride(3),
                states.stride(0),
                states.stride(1),
                states.stride(2),
                states.stride(3),
                states.stride(4),
                D.stride(0) if D is not None else 0,
                True,
                D is not None,
                D.dim() == 2 if D is not None else True,
                BLOCK_SIZE_DSTATE=max(triton_mod.next_power_of_2(dstate), 16),
                HAS_Z=z is not None,
                HAS_SEQ_IDX=seq_idx is not None,
                IS_TRITON_22=triton_22,
            )
        except Exception as exc:
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_actual_chunk_scan_store_cast_details",
                    "source": source,
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )
            return

        normal_row = out.reshape(batch, seqlen, -1)[0, -1, :].detach().contiguous()
        fp32_row = out_fp32.reshape(batch, seqlen, -1)[0, -1, :].detach().contiguous()
        candidate_row = fp32_row.to(torch.bfloat16).detach().contiguous()
        stored_bits = bf16_u16_values(normal_row)
        candidate_bits = bf16_u16_values(candidate_row)
        fp32_values = fp32_row.cpu().reshape(-1).tolist()
        stored_values = normal_row.float().cpu().reshape(-1).tolist()
        candidate_values = candidate_row.float().cpu().reshape(-1).tolist()
        width = int(normal_row.numel())
        details = []
        for idx in selected_element_dims:
            item: Dict[str, Any] = {"dim_index": int(idx)}
            if idx < 0 or idx >= width:
                item.update({"available": False, "reason": "dim out of range"})
                details.append(item)
                continue
            pre_store_value = float(fp32_values[idx])
            stored_bit = int(stored_bits[idx])
            candidate_bit = int(candidate_bits[idx])
            item.update(
                {
                    "available": True,
                    "actual_pre_store_value": pre_store_value,
                    "actual_pre_store_f32_bits": int(f32_bits(pre_store_value)),
                    "actual_bf16_candidate_bits": candidate_bit,
                    "actual_bf16_candidate_value": float(candidate_values[idx]),
                    "actual_stored_ssd_out_bits": stored_bit,
                    "actual_stored_ssd_out_value": float(stored_values[idx]),
                    "stored_matches_actual_candidate": bool(stored_bit == candidate_bit),
                }
            )
            details.append(item)
        summaries.append(
            {
                "index": len(summaries),
                "layer": 0,
                "label": "layer0_mamba2_actual_chunk_scan_store_cast_details",
                "source": source,
                "available": True,
                "selected_dims": selected_element_dims,
                "detail_count": len(details),
                "width": width,
                "normal_output_dtype": str(out.dtype),
                "actual_pre_store_dtype": str(out_fp32.dtype),
                "device": str(out.device),
                "runtime_op_path": "mamba_ssm.ops.triton.ssd_chunk_scan._chunk_scan_fwd_kernel",
                "capture_mode": "same actual forward kernel inputs replayed with an FP32 output buffer; normal BF16 forward output is unchanged",
                "store_dtype_control": "Triton tl.store casts acc to the dtype of out_ptr",
                "normal_output_hash_fnv1a_u16": fnv1a_u16(stored_bits),
                "candidate_output_hash_fnv1a_u16": fnv1a_u16(candidate_bits),
                "details": details,
                "block_type": block_type,
            }
        )

        def f32_round(value: float) -> float:
            return struct.unpack("<f", struct.pack("<f", float(value)))[0]

        def bf16_scalar_bits(tensor_value: Any) -> Optional[int]:
            tensor = _first_tensor(tensor_value)
            if tensor is None or tensor.dtype != torch.bfloat16:
                return None
            return int(bf16_u16_values(tensor.reshape(1))[0])

        try:
            ratio = nheads // ngroups
            row_index = seqlen - 1
            chunk_idx = row_index // chunk_size
            chunk_pos = row_index - chunk_idx * chunk_size
            chunk_limit = min(chunk_size, seqlen - chunk_idx * chunk_size)
            accumulator_details = []
            for idx in selected_element_dims:
                item = {"dim_index": int(idx)}
                if idx < 0 or idx >= width:
                    item.update({"available": False, "reason": "dim out of range"})
                    accumulator_details.append(item)
                    continue
                if chunk_idx < 0 or chunk_idx >= nchunks or chunk_pos >= chunk_limit:
                    item.update(
                        {
                            "available": False,
                            "reason": (
                                f"invalid chunk coordinates chunk_idx={chunk_idx}, "
                                f"chunk_pos={chunk_pos}, chunk_limit={chunk_limit}"
                            ),
                        }
                    )
                    accumulator_details.append(item)
                    continue

                head = int(idx // headdim)
                head_dim_index = int(idx - head * headdim)
                group = int(head // ratio)
                dA_target = float(dA_cumsum[0, head, chunk_idx, chunk_pos].float().item())
                d_x = 0.0
                if D is not None:
                    if D.dim() == 2:
                        d_val = float(D[head, head_dim_index].float().item())
                    else:
                        d_val = float(D[head].float().item())
                    x_last = float(x[0, row_index, head, head_dim_index].float().item())
                    d_x = f32_round(d_val * x_last)
                else:
                    d_val = 0.0
                    x_last = float(x[0, row_index, head, head_dim_index].float().item())

                prior_chunk_state = 0.0
                if chunk_idx > 0:
                    c_vec = C[0, row_index, group, :].float()
                    state_vec = states[0, chunk_idx, head, head_dim_index, :].to(C.dtype).float()
                    prior_chunk_state = f32_round(
                        float((c_vec * state_vec).sum().item()) * math.exp(dA_target)
                    )

                token_details = []
                local_scan_forward = 0.0
                local_scan_f32_seq = 0.0
                local_scan_kahan = 0.0
                local_scan_kahan_c = 0.0
                first_nonzero_term_token: Optional[int] = None
                max_abs_term_token: Optional[int] = None
                max_abs_term = -1.0

                def cuda_f32_unary(value: float, op_name: str) -> float:
                    tensor = torch.tensor([value], device=x.device, dtype=torch.float32)
                    if op_name == "softplus":
                        out = torch.nn.functional.softplus(tensor)
                    elif op_name == "log1p_exp":
                        out = torch.log1p(torch.exp(tensor))
                    elif op_name == "log_exp_add":
                        out = torch.log(1.0 + torch.exp(tensor))
                    elif op_name == "exp":
                        out = torch.exp(tensor)
                    else:
                        raise ValueError(f"unknown cuda_f32_unary op {op_name}")
                    return float(out.detach().cpu()[0].item())

                for pos in range(0, chunk_pos + 1):
                    abs_pos = chunk_idx * chunk_size + pos
                    cb_tensor = cb[0, chunk_idx, group, chunk_pos, pos].reshape(1)
                    cb_value = float(cb_tensor.float()[0].item())
                    dA_pos = float(dA_cumsum[0, head, chunk_idx, pos].float().item())
                    dA_prev = (
                        float(dA_cumsum[0, head, chunk_idx, pos - 1].float().item())
                        if pos > 0
                        else 0.0
                    )
                    dA_increment = f32_round(dA_pos - dA_prev)
                    decay_arg = min(dA_target - dA_pos, 0.0)
                    decay = f32_round(math.exp(decay_arg))
                    dt_value = float(dt[0, head, chunk_idx, pos].float().item())
                    scale = f32_round(decay * dt_value)
                    raw_dt_value: Optional[float] = None
                    raw_dt_bits: Optional[int] = None
                    dt_bias_value: Optional[float] = None
                    dt_plus_bias_fp32: Optional[float] = None
                    dt_plus_bias_fp32_bits: Optional[int] = None
                    dt_plus_bias_bf16_bits: Optional[int] = None
                    softplus_log1p_exp_value: Optional[float] = None
                    softplus_log_exp_fast_value: Optional[float] = None
                    softplus_torch_value: Optional[float] = None
                    softplus_cuda_log1p_exp_value: Optional[float] = None
                    softplus_cuda_log_exp_value: Optional[float] = None
                    softplus_log1p_exp_bits: Optional[int] = None
                    softplus_log_exp_fast_bits: Optional[int] = None
                    softplus_torch_bits: Optional[int] = None
                    softplus_cuda_log1p_exp_bits: Optional[int] = None
                    softplus_cuda_log_exp_bits: Optional[int] = None
                    a_value: Optional[float] = None
                    a_log_value: Optional[float] = None
                    negative_exp_a_log_cuda: Optional[float] = None
                    negative_exp_a_log_cuda_bits: Optional[int] = None
                    dA_increment_from_a_dt: Optional[float] = None
                    dA_increment_from_log1p_exp: Optional[float] = None
                    dA_increment_from_torch_softplus: Optional[float] = None
                    dA_increment_actual_minus_log1p_exp: Optional[float] = None
                    dA_cumsum_from_prev_log1p_exp: Optional[float] = None
                    dA_cumsum_actual_minus_prev_log1p_exp: Optional[float] = None
                    dA_cumsum_from_prev_torch_softplus: Optional[float] = None
                    dA_cumsum_actual_minus_prev_torch_softplus: Optional[float] = None
                    decay_from_prev_log1p_exp: Optional[float] = None
                    decay_from_prev_torch_softplus: Optional[float] = None
                    scale_log1p_exp: Optional[float] = None
                    scale_torch: Optional[float] = None
                    scale_prev_log1p_exp: Optional[float] = None
                    scale_prev_torch_softplus: Optional[float] = None
                    raw_dt_tensor = forward_chunk_cumsum_capture.get("raw_dt")
                    dt_bias_tensor = forward_chunk_cumsum_capture.get("dt_bias")
                    a_tensor = forward_chunk_cumsum_capture.get("A")
                    a_log_tensor = forward_chunk_cumsum_capture.get("A_log")
                    dt_softplus_enabled = forward_chunk_cumsum_capture.get("dt_softplus_enabled")
                    dt_limit_value = forward_chunk_cumsum_capture.get("dt_limit")
                    raw_dt_layout: Optional[str] = None
                    raw_dt_shape: Optional[List[int]] = None
                    dt_bias_shape: Optional[List[int]] = None
                    if raw_dt_tensor is not None:
                        raw_dt_shape = [int(v) for v in raw_dt_tensor.shape]
                        if raw_dt_tensor.ndim == 4:
                            raw_dt_layout = "batch,head,chunk,chunk_position"
                            raw_dt_scalar = raw_dt_tensor[0, head, chunk_idx, pos].reshape(1)
                        elif raw_dt_tensor.ndim == 3:
                            raw_dt_layout = "batch,sequence,head"
                            raw_dt_scalar = raw_dt_tensor[0, abs_pos, head].reshape(1)
                        else:
                            raw_dt_layout = f"unsupported_ndim_{raw_dt_tensor.ndim}"
                            raw_dt_scalar = None
                        if raw_dt_scalar is not None:
                            raw_dt_value = float(raw_dt_scalar.float()[0].item())
                            raw_dt_bits = bf16_scalar_bits(raw_dt_scalar)
                    if dt_bias_tensor is not None:
                        dt_bias_shape = [int(v) for v in dt_bias_tensor.shape]
                        dt_bias_value = float(dt_bias_tensor.reshape(-1)[head].float().item())
                    if a_tensor is not None:
                        a_value = float(a_tensor[head].float().item())
                    if a_log_tensor is not None:
                        a_log_value = float(a_log_tensor.reshape(-1)[head].float().item())
                        negative_exp_a_log_cuda = -cuda_f32_unary(a_log_value, "exp")
                        negative_exp_a_log_cuda_bits = int(f32_bits(negative_exp_a_log_cuda))
                    if raw_dt_value is not None and dt_bias_value is not None:
                        dt_plus_bias_fp32 = f32_round(raw_dt_value + dt_bias_value)
                        dt_plus_bias_fp32_bits = int(f32_bits(dt_plus_bias_fp32))
                        dt_plus_bias_bf16_tensor = torch.tensor(
                            [dt_plus_bias_fp32],
                            device=x.device,
                            dtype=torch.float32,
                        ).to(torch.bfloat16)
                        dt_plus_bias_bf16_bits = int(bf16_u16_values(dt_plus_bias_bf16_tensor)[0])
                        softplus_log1p_exp_value = f32_round(
                            math.log1p(math.exp(dt_plus_bias_fp32))
                        )
                        softplus_log_exp_fast_value = f32_round(
                            math.log(1.0 + math.exp(dt_plus_bias_fp32))
                        )
                        softplus_torch_value = float(
                            torch.nn.functional.softplus(
                                torch.tensor(
                                    [dt_plus_bias_fp32],
                                    device=x.device,
                                    dtype=torch.float32,
                                )
                            ).cpu()[0].item()
                        )
                        softplus_cuda_log1p_exp_value = cuda_f32_unary(
                            dt_plus_bias_fp32, "log1p_exp"
                        )
                        softplus_cuda_log_exp_value = cuda_f32_unary(
                            dt_plus_bias_fp32, "log_exp_add"
                        )
                        softplus_log1p_exp_bits = int(f32_bits(softplus_log1p_exp_value))
                        softplus_log_exp_fast_bits = int(f32_bits(softplus_log_exp_fast_value))
                        softplus_torch_bits = int(f32_bits(softplus_torch_value))
                        softplus_cuda_log1p_exp_bits = int(
                            f32_bits(softplus_cuda_log1p_exp_value)
                        )
                        softplus_cuda_log_exp_bits = int(
                            f32_bits(softplus_cuda_log_exp_value)
                        )
                        scale_log1p_exp = f32_round(decay * softplus_log1p_exp_value)
                        scale_torch = f32_round(decay * softplus_torch_value)
                    if a_value is not None:
                        dA_increment_from_a_dt = f32_round(a_value * dt_value)
                        if softplus_log1p_exp_value is not None:
                            dA_increment_from_log1p_exp = f32_round(
                                a_value * softplus_log1p_exp_value
                            )
                            dA_increment_actual_minus_log1p_exp = float(
                                dA_increment_from_a_dt - dA_increment_from_log1p_exp
                            )
                            dA_cumsum_from_prev_log1p_exp = f32_round(
                                dA_prev + dA_increment_from_log1p_exp
                            )
                            dA_cumsum_actual_minus_prev_log1p_exp = float(
                                dA_pos - dA_cumsum_from_prev_log1p_exp
                            )
                            decay_from_prev_log1p_exp = f32_round(
                                math.exp(min(dA_target - dA_cumsum_from_prev_log1p_exp, 0.0))
                            )
                            scale_prev_log1p_exp = f32_round(
                                decay_from_prev_log1p_exp * softplus_log1p_exp_value
                            )
                        if softplus_torch_value is not None:
                            dA_increment_from_torch_softplus = f32_round(
                                a_value * softplus_torch_value
                            )
                            dA_cumsum_from_prev_torch_softplus = f32_round(
                                dA_prev + dA_increment_from_torch_softplus
                            )
                            dA_cumsum_actual_minus_prev_torch_softplus = float(
                                dA_pos - dA_cumsum_from_prev_torch_softplus
                            )
                            decay_from_prev_torch_softplus = f32_round(
                                math.exp(min(dA_target - dA_cumsum_from_prev_torch_softplus, 0.0))
                            )
                            scale_prev_torch_softplus = f32_round(
                                decay_from_prev_torch_softplus * softplus_torch_value
                            )
                    cb_scaled_fp32 = f32_round(cb_value * scale)
                    cb_scaled_bf16_tensor = torch.tensor(
                        [cb_scaled_fp32],
                        device=x.device,
                        dtype=torch.float32,
                    ).to(x.dtype)
                    cb_scaled_bf16 = float(cb_scaled_bf16_tensor.float()[0].item())
                    x_tensor = x[0, abs_pos, head, head_dim_index].reshape(1)
                    x_value = float(x_tensor.float()[0].item())
                    term_bf16_cbscale = f32_round(cb_scaled_bf16 * x_value)
                    local_scan_forward += term_bf16_cbscale
                    local_scan_f32_seq = f32_round(local_scan_f32_seq + term_bf16_cbscale)
                    kahan_y = term_bf16_cbscale - local_scan_kahan_c
                    kahan_t = local_scan_kahan + kahan_y
                    local_scan_kahan_c = (kahan_t - local_scan_kahan) - kahan_y
                    local_scan_kahan = kahan_t
                    abs_term = abs(term_bf16_cbscale)
                    if first_nonzero_term_token is None and term_bf16_cbscale != 0.0:
                        first_nonzero_term_token = pos
                    if abs_term > max_abs_term:
                        max_abs_term = abs_term
                        max_abs_term_token = pos
                    token_details.append(
                        {
                            "token_position": int(abs_pos),
                            "chunk_token_position": int(pos),
                            "head_index": head,
                            "head_dim_index": head_dim_index,
                            "group_index": group,
                            "cb_dtype": str(cb.dtype),
                            "cb_bits": bf16_scalar_bits(cb_tensor),
                            "cb_value": cb_value,
                            "x_bits": bf16_scalar_bits(x_tensor),
                            "x_value": x_value,
                            "dt_dtype": str(dt.dtype),
                            "dt_value": dt_value,
                            "dt_value_f32_bits": int(f32_bits(dt_value)),
                            "dt_value_bf16_candidate_bits": int(
                                bf16_u16_values(
                                    torch.tensor(
                                        [dt_value],
                                        device=x.device,
                                        dtype=torch.float32,
                                    ).to(torch.bfloat16)
                                )[0]
                            ),
                            "dt_softplus_enabled": dt_softplus_enabled,
                            "dt_limit": dt_limit_value,
                            "raw_dt_bits": raw_dt_bits,
                            "raw_dt_value": raw_dt_value,
                            "raw_dt_layout": raw_dt_layout,
                            "raw_dt_shape": raw_dt_shape,
                            "dt_bias_shape": dt_bias_shape,
                            "dt_bias_value": dt_bias_value,
                            "dt_plus_bias_fp32": dt_plus_bias_fp32,
                            "dt_plus_bias_fp32_bits": dt_plus_bias_fp32_bits,
                            "dt_plus_bias_bf16_bits": dt_plus_bias_bf16_bits,
                            "softplus_log1p_exp": softplus_log1p_exp_value,
                            "softplus_log1p_exp_f32_bits": softplus_log1p_exp_bits,
                            "softplus_log_exp_fast": softplus_log_exp_fast_value,
                            "softplus_log_exp_fast_f32_bits": softplus_log_exp_fast_bits,
                            "softplus_torch": softplus_torch_value,
                            "softplus_torch_f32_bits": softplus_torch_bits,
                            "softplus_cuda_log1p_exp": softplus_cuda_log1p_exp_value,
                            "softplus_cuda_log1p_exp_f32_bits": softplus_cuda_log1p_exp_bits,
                            "softplus_cuda_log_exp": softplus_cuda_log_exp_value,
                            "softplus_cuda_log_exp_f32_bits": softplus_cuda_log_exp_bits,
                            "dt_out_minus_log1p_exp": (
                                float(dt_value - softplus_log1p_exp_value)
                                if softplus_log1p_exp_value is not None
                                else None
                            ),
                            "dt_out_minus_log_exp_fast": (
                                float(dt_value - softplus_log_exp_fast_value)
                                if softplus_log_exp_fast_value is not None
                                else None
                            ),
                            "dt_out_minus_torch_softplus": (
                                float(dt_value - softplus_torch_value)
                                if softplus_torch_value is not None
                                else None
                            ),
                            "dt_out_minus_cuda_log1p_exp": (
                                float(dt_value - softplus_cuda_log1p_exp_value)
                                if softplus_cuda_log1p_exp_value is not None
                                else None
                            ),
                            "dt_out_minus_cuda_log_exp": (
                                float(dt_value - softplus_cuda_log_exp_value)
                                if softplus_cuda_log_exp_value is not None
                                else None
                            ),
                            "dA_cumsum": dA_pos,
                            "dA_cumsum_f32_bits": int(f32_bits(dA_pos)),
                            "dA_cumsum_prev": dA_prev,
                            "dA_increment": dA_increment,
                            "dA_increment_f32_bits": int(f32_bits(dA_increment)),
                            "dA_increment_from_a_dt": dA_increment_from_a_dt,
                            "dA_increment_from_log1p_exp": dA_increment_from_log1p_exp,
                            "dA_increment_from_torch_softplus": dA_increment_from_torch_softplus,
                            "dA_increment_actual_minus_log1p_exp": (
                                dA_increment_actual_minus_log1p_exp
                            ),
                            "dA_cumsum_from_prev_log1p_exp": dA_cumsum_from_prev_log1p_exp,
                            "dA_cumsum_actual_minus_prev_log1p_exp": (
                                dA_cumsum_actual_minus_prev_log1p_exp
                            ),
                            "dA_cumsum_from_prev_torch_softplus": (
                                dA_cumsum_from_prev_torch_softplus
                            ),
                            "dA_cumsum_actual_minus_prev_torch_softplus": (
                                dA_cumsum_actual_minus_prev_torch_softplus
                            ),
                            "dA_target": dA_target,
                            "decay_arg": decay_arg,
                            "decay": decay,
                            "decay_from_prev_log1p_exp": decay_from_prev_log1p_exp,
                            "decay_from_prev_torch_softplus": decay_from_prev_torch_softplus,
                            "scale": scale,
                            "scale_log1p_exp": scale_log1p_exp,
                            "scale_torch": scale_torch,
                            "scale_prev_log1p_exp": scale_prev_log1p_exp,
                            "scale_prev_torch_softplus": scale_prev_torch_softplus,
                            "a_log_value": a_log_value,
                            "a_log_f32_bits": (
                                int(f32_bits(a_log_value)) if a_log_value is not None else None
                            ),
                            "negative_exp_a_log_cuda": negative_exp_a_log_cuda,
                            "negative_exp_a_log_cuda_f32_bits": negative_exp_a_log_cuda_bits,
                            "a_value": a_value,
                            "a_value_f32_bits": (
                                int(f32_bits(a_value)) if a_value is not None else None
                            ),
                            "cb_scaled_fp32": cb_scaled_fp32,
                            "cb_scaled_fp32_bits": int(f32_bits(cb_scaled_fp32)),
                            "cb_scaled_bf16_bits": int(bf16_u16_values(cb_scaled_bf16_tensor)[0]),
                            "cb_scaled_bf16": cb_scaled_bf16,
                            "term_bf16_cbscale": term_bf16_cbscale,
                            "term_bf16_cbscale_f32_bits": int(f32_bits(term_bf16_cbscale)),
                            "cumulative_forward": float(local_scan_forward),
                            "cumulative_f32_seq": float(local_scan_f32_seq),
                            "cumulative_kahan": float(local_scan_kahan),
                        }
                    )

                kernel_pre_store = float(fp32_values[idx])
                reconstructed_forward = float(d_x + prior_chunk_state + local_scan_forward)
                reconstructed_f32_seq = f32_round(
                    f32_round(d_x + prior_chunk_state) + local_scan_f32_seq
                )
                reconstructed_kahan = float(d_x + prior_chunk_state + local_scan_kahan)
                item.update(
                    {
                        "available": True,
                        "head_index": head,
                        "head_dim_index": head_dim_index,
                        "group_index": group,
                        "row_index": int(row_index),
                        "chunk_index": int(chunk_idx),
                        "chunk_position": int(chunk_pos),
                        "chunk_limit": int(chunk_limit),
                        "d_value": d_val,
                        "x_last_bits": bf16_scalar_bits(
                            x[0, row_index, head, head_dim_index].reshape(1)
                        ),
                        "x_last_value": x_last,
                        "d_x": d_x,
                        "prior_chunk_state": prior_chunk_state,
                        "local_scan_forward": float(local_scan_forward),
                        "local_scan_f32_seq": float(local_scan_f32_seq),
                        "local_scan_kahan": float(local_scan_kahan),
                        "kernel_pre_store": kernel_pre_store,
                        "kernel_pre_store_f32_bits": int(f32_bits(kernel_pre_store)),
                        "reconstructed_forward": reconstructed_forward,
                        "reconstructed_forward_f32_bits": int(f32_bits(reconstructed_forward)),
                        "reconstructed_f32_seq": reconstructed_f32_seq,
                        "reconstructed_f32_seq_f32_bits": int(f32_bits(reconstructed_f32_seq)),
                        "reconstructed_kahan": reconstructed_kahan,
                        "reconstructed_kahan_f32_bits": int(f32_bits(reconstructed_kahan)),
                        "kernel_minus_reconstructed_forward": float(
                            kernel_pre_store - reconstructed_forward
                        ),
                        "kernel_minus_reconstructed_f32_seq": float(
                            kernel_pre_store - reconstructed_f32_seq
                        ),
                        "first_nonzero_term_token": first_nonzero_term_token,
                        "max_abs_term_token": max_abs_term_token,
                        "max_abs_term": float(max_abs_term),
                        "token_detail_count": len(token_details),
                        "token_details": token_details,
                    }
                )
                accumulator_details.append(item)

            summaries.append(
                {
                    "index": len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_actual_chunk_scan_accumulator_details",
                    "source": "hf_layer0_mamba2_actual_chunk_scan_accumulator_pre_store",
                    "available": True,
                    "selected_dims": selected_element_dims,
                    "detail_count": len(accumulator_details),
                    "row_index": int(row_index),
                    "chunk_index": int(chunk_idx),
                    "chunk_position": int(chunk_pos),
                    "chunk_limit": int(chunk_limit),
                    "cb_dtype": str(cb.dtype),
                    "x_dtype": str(x.dtype),
                    "dt_dtype": str(dt.dtype),
                    "dA_cumsum_dtype": str(dA_cumsum.dtype),
                    "d_dtype": str(D.dtype) if D is not None else None,
                    "cb_shape": [int(v) for v in cb.shape],
                    "x_shape": [int(v) for v in x.shape],
                    "dt_shape": [int(v) for v in dt.shape],
                    "dA_cumsum_shape": [int(v) for v in dA_cumsum.shape],
                    "semantics": (
                        "selected-dim replay of the actual Triton chunk-scan local "
                        "accumulator inputs: precomputed CB, decay, dt scale, BF16 "
                        "scaled-CB cast, BF16 x, local scan terms, D*x, and final "
                        "pre-store reconstruction"
                    ),
                    "details": accumulator_details,
                    "block_type": block_type,
                }
            )
            selected_scan_rows = [
                int(row)
                for row in selected_row_indices
                if 0 <= int(row) < seqlen and int(row) != row_index
            ]
            for selected_row_index in selected_scan_rows:
                selected_chunk_idx = selected_row_index // chunk_size
                selected_chunk_pos = selected_row_index - selected_chunk_idx * chunk_size
                selected_chunk_limit = min(
                    chunk_size, seqlen - selected_chunk_idx * chunk_size
                )
                selected_fp32_row = (
                    out_fp32.reshape(batch, seqlen, -1)[0, selected_row_index, :]
                    .detach()
                    .contiguous()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )
                selected_accumulator_details = []
                for idx in selected_element_dims:
                    item = {"dim_index": int(idx)}
                    if idx < 0 or idx >= width:
                        item.update({"available": False, "reason": "dim out of range"})
                        selected_accumulator_details.append(item)
                        continue
                    if (
                        selected_chunk_idx < 0
                        or selected_chunk_idx >= nchunks
                        or selected_chunk_pos >= selected_chunk_limit
                    ):
                        item.update(
                            {
                                "available": False,
                                "reason": (
                                    "invalid chunk coordinates "
                                    f"chunk_idx={selected_chunk_idx}, "
                                    f"chunk_pos={selected_chunk_pos}, "
                                    f"chunk_limit={selected_chunk_limit}"
                                ),
                            }
                        )
                        selected_accumulator_details.append(item)
                        continue

                    head = int(idx // headdim)
                    head_dim_index = int(idx - head * headdim)
                    group = int(head // ratio)
                    dA_target = float(
                        dA_cumsum[
                            0, head, selected_chunk_idx, selected_chunk_pos
                        ].float().item()
                    )
                    if D is not None:
                        if D.dim() == 2:
                            d_val = float(D[head, head_dim_index].float().item())
                        else:
                            d_val = float(D[head].float().item())
                        x_last = float(
                            x[
                                0, selected_row_index, head, head_dim_index
                            ].float().item()
                        )
                        d_x = f32_round(d_val * x_last)
                    else:
                        d_val = 0.0
                        x_last = float(
                            x[
                                0, selected_row_index, head, head_dim_index
                            ].float().item()
                        )

                    prior_chunk_state = 0.0
                    if selected_chunk_idx > 0:
                        c_vec = C[0, selected_row_index, group, :].float()
                        state_vec = states[
                            0, selected_chunk_idx, head, head_dim_index, :
                        ].to(C.dtype).float()
                        prior_chunk_state = f32_round(
                            float((c_vec * state_vec).sum().item()) * math.exp(dA_target)
                        )

                    token_details = []
                    local_scan_forward = 0.0
                    local_scan_f32_seq = 0.0
                    first_nonzero_term_token: Optional[int] = None
                    max_abs_term_token: Optional[int] = None
                    max_abs_term = -1.0
                    raw_dt_tensor = forward_chunk_cumsum_capture.get("raw_dt")
                    dt_bias_tensor = forward_chunk_cumsum_capture.get("dt_bias")
                    a_tensor = forward_chunk_cumsum_capture.get("A")
                    a_log_tensor = forward_chunk_cumsum_capture.get("A_log")
                    dt_softplus_enabled = forward_chunk_cumsum_capture.get(
                        "dt_softplus_enabled"
                    )
                    dt_limit_value = forward_chunk_cumsum_capture.get("dt_limit")
                    for pos in range(0, selected_chunk_pos + 1):
                        abs_pos = selected_chunk_idx * chunk_size + pos
                        cb_tensor = cb[
                            0, selected_chunk_idx, group, selected_chunk_pos, pos
                        ].reshape(1)
                        cb_value = float(cb_tensor.float()[0].item())
                        dA_pos = float(
                            dA_cumsum[0, head, selected_chunk_idx, pos].float().item()
                        )
                        dA_prev = (
                            float(
                                dA_cumsum[
                                    0, head, selected_chunk_idx, pos - 1
                                ].float().item()
                            )
                            if pos > 0
                            else 0.0
                        )
                        dA_increment = f32_round(dA_pos - dA_prev)
                        decay = f32_round(math.exp(min(dA_target - dA_pos, 0.0)))
                        dt_value = float(
                            dt[0, head, selected_chunk_idx, pos].float().item()
                        )
                        scale = f32_round(decay * dt_value)
                        raw_dt_value: Optional[float] = None
                        raw_dt_bits: Optional[int] = None
                        raw_dt_layout: Optional[str] = None
                        raw_dt_shape: Optional[List[int]] = None
                        if raw_dt_tensor is not None:
                            raw_dt_shape = [int(v) for v in raw_dt_tensor.shape]
                            if raw_dt_tensor.ndim == 4:
                                raw_dt_layout = "batch,head,chunk,chunk_position"
                                raw_dt_scalar = raw_dt_tensor[
                                    0, head, selected_chunk_idx, pos
                                ].reshape(1)
                            elif raw_dt_tensor.ndim == 3:
                                raw_dt_layout = "batch,sequence,head"
                                raw_dt_scalar = raw_dt_tensor[0, abs_pos, head].reshape(1)
                            else:
                                raw_dt_layout = f"unsupported_ndim_{raw_dt_tensor.ndim}"
                                raw_dt_scalar = None
                            if raw_dt_scalar is not None:
                                raw_dt_value = float(raw_dt_scalar.float()[0].item())
                                raw_dt_bits = bf16_scalar_bits(raw_dt_scalar)
                        dt_bias_value: Optional[float] = None
                        if dt_bias_tensor is not None:
                            dt_bias_value = float(
                                dt_bias_tensor.reshape(-1)[head].float().item()
                            )
                        a_value: Optional[float] = None
                        a_log_value: Optional[float] = None
                        negative_exp_a_log_cuda: Optional[float] = None
                        negative_exp_a_log_cuda_bits: Optional[int] = None
                        if a_tensor is not None:
                            a_value = float(a_tensor[head].float().item())
                        if a_log_tensor is not None:
                            a_log_value = float(a_log_tensor.reshape(-1)[head].float().item())
                            negative_exp_tensor = -torch.exp(
                                torch.tensor(
                                    [a_log_value],
                                    device=x.device,
                                    dtype=torch.float32,
                                )
                            )
                            negative_exp_a_log_cuda = float(
                                negative_exp_tensor.detach().cpu()[0].item()
                            )
                            negative_exp_a_log_cuda_bits = int(
                                f32_bits(negative_exp_a_log_cuda)
                            )
                        cb_scaled_fp32 = f32_round(cb_value * scale)
                        cb_scaled_bf16_tensor = torch.tensor(
                            [cb_scaled_fp32],
                            device=x.device,
                            dtype=torch.float32,
                        ).to(x.dtype)
                        cb_scaled_bf16 = float(cb_scaled_bf16_tensor.float()[0].item())
                        x_tensor = x[0, abs_pos, head, head_dim_index].reshape(1)
                        x_value = float(x_tensor.float()[0].item())
                        term_bf16_cbscale = f32_round(cb_scaled_bf16 * x_value)
                        local_scan_forward += term_bf16_cbscale
                        local_scan_f32_seq = f32_round(
                            local_scan_f32_seq + term_bf16_cbscale
                        )
                        abs_term = abs(term_bf16_cbscale)
                        if first_nonzero_term_token is None and term_bf16_cbscale != 0.0:
                            first_nonzero_term_token = pos
                        if abs_term > max_abs_term:
                            max_abs_term = abs_term
                            max_abs_term_token = pos
                        token_details.append(
                            {
                                "token_position": int(abs_pos),
                                "chunk_token_position": int(pos),
                                "head_index": head,
                                "head_dim_index": head_dim_index,
                                "group_index": group,
                                "cb_value": cb_value,
                                "cb_bits": bf16_scalar_bits(cb_tensor),
                                "x_bits": bf16_scalar_bits(x_tensor),
                                "x_value": x_value,
                                "dt_value": dt_value,
                                "dt_value_f32_bits": int(f32_bits(dt_value)),
                                "dt_softplus_enabled": dt_softplus_enabled,
                                "dt_limit": dt_limit_value,
                                "raw_dt_bits": raw_dt_bits,
                                "raw_dt_value": raw_dt_value,
                                "raw_dt_layout": raw_dt_layout,
                                "raw_dt_shape": raw_dt_shape,
                                "dt_bias_value": dt_bias_value,
                                "a_log_value": a_log_value,
                                "a_log_f32_bits": (
                                    int(f32_bits(a_log_value))
                                    if a_log_value is not None
                                    else None
                                ),
                                "negative_exp_a_log_cuda": negative_exp_a_log_cuda,
                                "negative_exp_a_log_cuda_f32_bits": (
                                    negative_exp_a_log_cuda_bits
                                ),
                                "a_value": a_value,
                                "a_value_f32_bits": (
                                    int(f32_bits(a_value)) if a_value is not None else None
                                ),
                                "dA_cumsum": dA_pos,
                                "dA_cumsum_f32_bits": int(f32_bits(dA_pos)),
                                "dA_cumsum_prev": dA_prev,
                                "dA_increment": dA_increment,
                                "dA_increment_f32_bits": int(f32_bits(dA_increment)),
                                "decay": decay,
                                "scale": scale,
                                "cb_scaled_fp32": cb_scaled_fp32,
                                "cb_scaled_fp32_bits": int(f32_bits(cb_scaled_fp32)),
                                "cb_scaled_bf16_bits": int(
                                    bf16_u16_values(cb_scaled_bf16_tensor)[0]
                                ),
                                "cb_scaled_bf16": cb_scaled_bf16,
                                "term_bf16_cbscale": term_bf16_cbscale,
                                "term_bf16_cbscale_f32_bits": int(
                                    f32_bits(term_bf16_cbscale)
                                ),
                                "cumulative_forward": float(local_scan_forward),
                                "cumulative_f32_seq": float(local_scan_f32_seq),
                            }
                        )

                    kernel_pre_store = float(selected_fp32_row[idx])
                    reconstructed_forward = float(
                        d_x + prior_chunk_state + local_scan_forward
                    )
                    item.update(
                        {
                            "available": True,
                            "head_index": head,
                            "head_dim_index": head_dim_index,
                            "group_index": group,
                            "row_index": int(selected_row_index),
                            "chunk_index": int(selected_chunk_idx),
                            "chunk_position": int(selected_chunk_pos),
                            "chunk_limit": int(selected_chunk_limit),
                            "d_value": d_val,
                            "x_last_bits": bf16_scalar_bits(
                                x[
                                    0, selected_row_index, head, head_dim_index
                                ].reshape(1)
                            ),
                            "x_last_value": x_last,
                            "d_x": d_x,
                            "prior_chunk_state": prior_chunk_state,
                            "local_scan_forward": float(local_scan_forward),
                            "local_scan_f32_seq": float(local_scan_f32_seq),
                            "kernel_pre_store": kernel_pre_store,
                            "kernel_pre_store_f32_bits": int(f32_bits(kernel_pre_store)),
                            "reconstructed_forward": reconstructed_forward,
                            "reconstructed_forward_f32_bits": int(
                                f32_bits(reconstructed_forward)
                            ),
                            "kernel_minus_reconstructed_forward": float(
                                kernel_pre_store - reconstructed_forward
                            ),
                            "first_nonzero_term_token": first_nonzero_term_token,
                            "max_abs_term_token": max_abs_term_token,
                            "max_abs_term": float(max_abs_term),
                            "token_detail_count": len(token_details),
                            "token_details": token_details,
                        }
                    )
                    selected_accumulator_details.append(item)

                summaries.append(
                    {
                        "index": len(summaries),
                        "layer": 0,
                        "label": "layer0_mamba2_actual_chunk_scan_accumulator_details",
                        "source": "hf_layer0_mamba2_actual_chunk_scan_accumulator_pre_store",
                        "available": True,
                        "selected_dims": selected_element_dims,
                        "detail_count": len(selected_accumulator_details),
                        "row_index": int(selected_row_index),
                        "chunk_index": int(selected_chunk_idx),
                        "chunk_position": int(selected_chunk_pos),
                        "chunk_limit": int(selected_chunk_limit),
                        "cb_dtype": str(cb.dtype),
                        "x_dtype": str(x.dtype),
                        "dt_dtype": str(dt.dtype),
                        "dA_cumsum_dtype": str(dA_cumsum.dtype),
                        "d_dtype": str(D.dtype) if D is not None else None,
                        "cb_shape": [int(v) for v in cb.shape],
                        "x_shape": [int(v) for v in x.shape],
                        "dt_shape": [int(v) for v in dt.shape],
                        "dA_cumsum_shape": [int(v) for v in dA_cumsum.shape],
                        "semantics": (
                            "diagnostic-only selected-row replay of the actual Triton "
                            "chunk-scan local accumulator inputs"
                        ),
                        "details": selected_accumulator_details,
                        "block_type": block_type,
                    }
                )
        except Exception as exc:
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_actual_chunk_scan_accumulator_details",
                    "source": "hf_layer0_mamba2_actual_chunk_scan_accumulator_pre_store",
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "selected_dims": selected_element_dims,
                    "block_type": block_type,
                }
            )

    def pre_hook(label: str, source: str):
        def hook(_module: Any, inputs: Any) -> None:
            value = inputs[0] if inputs else None
            record(label, source, value)
            if label in ("layer0_input", "layer0_norm_input", "layer0_mixer_input"):
                append_bf16_selected_row_details(label, source, value)
            if label == "layer0_input":
                cache_layer0_element_row(label, value)

        return hook

    def post_hook(label: str, source: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            record(label, source, output)
            if label in ("layer0_norm_output", "layer0_mixer_output", "layer0_output"):
                append_bf16_selected_row_details(label, source, output)
            if label in ("layer0_mixer_output", "layer0_output"):
                cache_layer0_element_row(label, output)
                if label == "layer0_output":
                    append_layer0_handoff_full_row_bits()
                    append_layer0_element_details()

        return hook

    forward_scan_globals: Optional[Dict[str, Any]] = None
    forward_original_chunk_cumsum = None
    forward_original_chunk_state = None
    forward_original_state_passing = None
    forward_original_bmm_chunk = None
    forward_original_chunk_scan = None
    forward_chunk_capture = {"done": False}
    forward_state_capture = {
        "chunk_state_done": False,
        "state_passing_done": False,
        "bmm_chunk_done": False,
        "chunk_scan_done": False,
    }
    actual_preconv_split_capture = {"done": False}
    actual_conv_contract_capture = {"done": False}

    def append_bf16_exact_row_details(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable(f"{label}_selected_row_details", f"{source}_exact_last_row", "no tensor value")
            return
        detached = tensor.detach()
        if detached.ndim >= 3:
            row = detached[0, -1, ...].reshape(-1).contiguous()
        elif detached.ndim == 2:
            row = detached[-1, :].reshape(-1).contiguous()
        else:
            row = detached.reshape(-1).contiguous()
        width = int(row.numel())
        detail_indices = sorted(
            {
                0,
                1,
                2,
                3,
                max(0, width // 2),
                max(0, width - 1),
            }
            | {int(dim) for dim in selected_element_dims if 0 <= int(dim) < width}
        )
        details: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {
            "index": len(summaries),
            "layer": 0,
            "label": f"{label}_selected_row_details",
            "source": f"{source}_exact_last_row",
            "dtype": str(row.dtype),
            "device": str(row.device),
            "row_width": width,
            "selected_dims": detail_indices,
            "block_type": block_type,
        }
        row_f32 = row.to(torch.float32).detach().cpu().reshape(-1)
        if row.dtype == torch.bfloat16:
            bits = bf16_u16_values(row)
            summary["hash_fnv1a64_bf16_bits"] = fnv1a_u16(bits)
            for idx in detail_indices:
                details.append(
                    {
                        "dim_index": int(idx),
                        "bf16_bits": int(bits[idx]),
                        "bf16_bits_hex": f"0x{int(bits[idx]) & 0xFFFF:04x}",
                        "value": float(row_f32[idx].item()),
                    }
                )
        else:
            summary["sha256_f32"] = hashlib.sha256(row_f32.contiguous().numpy().tobytes()).hexdigest()
            for idx in detail_indices:
                value_f32 = float(row_f32[idx].item())
                details.append(
                    {
                        "dim_index": int(idx),
                        "f32_bits": int(f32_bits(value_f32)),
                        "f32_bits_hex": f"0x{int(f32_bits(value_f32)):08x}",
                        "value": value_f32,
                    }
                )
        summary["detail_count"] = len(details)
        summary["details"] = details
        summaries.append(summary)

    def append_bf16_selected_row_details(label: str, source: str, value: Any) -> None:
        if not selected_row_indices:
            return
        tensor = _first_tensor(value)
        if tensor is None:
            for row_index in selected_row_indices:
                unavailable(
                    f"{label}_row{row_index}_selected_row_details",
                    f"{source}_selected_row",
                    "no tensor value",
                )
            return
        detached = tensor.detach()
        if detached.ndim >= 3:
            rows = detached[0].reshape(detached.shape[1], -1)
        elif detached.ndim == 2:
            rows = detached.reshape(detached.shape[0], -1)
        else:
            rows = detached.reshape(1, -1)
        row_count = int(rows.shape[0])
        width = int(rows.shape[1])
        detail_indices = sorted(
            {
                0,
                1,
                2,
                3,
                max(0, width // 2),
                max(0, width - 1),
            }
            | {int(dim) for dim in selected_element_dims if 0 <= int(dim) < width}
        )
        for row_index in selected_row_indices:
            if row_index >= row_count:
                summaries.append(
                    {
                        "index": len(summaries),
                        "layer": 0,
                        "label": f"{label}_row{row_index}_selected_row_details",
                        "source": f"{source}_selected_row",
                        "available": False,
                        "reason": f"row_index {row_index} out of range for row_count {row_count}",
                        "row_index": int(row_index),
                        "row_count": row_count,
                        "block_type": block_type,
                    }
                )
                continue
            row = rows[row_index].contiguous()
            row_f32 = row.to(torch.float32).detach().cpu().reshape(-1)
            details: List[Dict[str, Any]] = []
            summary: Dict[str, Any] = {
                "index": len(summaries),
                "layer": 0,
                "label": f"{label}_row{row_index}_selected_row_details",
                "source": f"{source}_selected_row",
                "available": True,
                "dtype": str(row.dtype),
                "device": str(row.device),
                "row_index": int(row_index),
                "row_count": row_count,
                "row_width": width,
                "selected_dims": detail_indices,
                "block_type": block_type,
                "semantics": "diagnostic-only exact selected sequence row; model forward output is unchanged",
            }
            if row.dtype == torch.bfloat16:
                bits = bf16_u16_values(row)
                summary["hash_fnv1a64_bf16_bits"] = fnv1a_u16(bits)
                if label in {"layer0_output", "layer1_input", "layer1_norm_input"}:
                    summary["full_bits_u16"] = [int(v) & 0xFFFF for v in bits]
                    summary["full_bits_semantics"] = (
                        "diagnostic-only selected row full BF16 bits for layer0-to-layer1 "
                        "handoff provenance; enabled only by explicit row selectors"
                    )
                for idx in detail_indices:
                    details.append(
                        {
                            "dim_index": int(idx),
                            "bf16_bits": int(bits[idx]),
                            "bf16_bits_hex": f"0x{int(bits[idx]) & 0xFFFF:04x}",
                            "value": float(row_f32[idx].item()),
                        }
                    )
            else:
                summary["sha256_f32"] = hashlib.sha256(row_f32.contiguous().numpy().tobytes()).hexdigest()
                for idx in detail_indices:
                    value_f32 = float(row_f32[idx].item())
                    details.append(
                        {
                            "dim_index": int(idx),
                            "f32_bits": int(f32_bits(value_f32)),
                            "f32_bits_hex": f"0x{int(f32_bits(value_f32)):08x}",
                            "value": value_f32,
                        }
                    )
            summary["detail_count"] = len(details)
            summary["details"] = details
            summaries.append(summary)

    def append_actual_conv_contract_metadata() -> None:
        if actual_conv_contract_capture["done"]:
            return
        actual_conv_contract_capture["done"] = True
        source = "hf_layer0_mamba2_actual_forward_conv_boundary"
        if block_type != "mamba" or mixer is None:
            unavailable(
                "layer0_mamba2_actual_conv_boundary_metadata",
                source,
                f"layer0 block_type is {block_type}; no Mamba2 mixer",
            )
            return
        conv1d = getattr(mixer, "conv1d", None)
        if conv1d is None:
            unavailable("layer0_mamba2_actual_conv_boundary_metadata", source, "mixer has no conv1d")
            return
        weight = getattr(conv1d, "weight", None)
        bias = getattr(conv1d, "bias", None)
        if weight is None:
            unavailable("layer0_mamba2_actual_conv_boundary_metadata", source, "conv1d has no weight")
            return

        def tensor_contract_summary(tensor: Any) -> Dict[str, Any]:
            detached = tensor.detach().contiguous()
            result: Dict[str, Any] = {
                "dtype": str(detached.dtype),
                "device": str(detached.device),
                "shape": [int(v) for v in detached.shape],
                "stride": [int(v) for v in tensor.detach().stride()],
            }
            if detached.dtype == torch.bfloat16:
                result["hash_fnv1a64_bf16_bits"] = fnv1a_u16(bf16_u16_values(detached))
            else:
                flat_f32 = detached.to(torch.float32).cpu().reshape(-1).contiguous()
                result["sha256_f32"] = hashlib.sha256(flat_f32.numpy().tobytes()).hexdigest()
            return result

        metadata: Dict[str, Any] = {
            "index": len(summaries),
            "layer": 0,
            "label": "layer0_mamba2_actual_conv_boundary_metadata",
            "source": source,
            "block_type": block_type,
            "conv_module_class": conv1d.__class__.__name__,
            "conv_weight": tensor_contract_summary(weight),
            "conv_bias": tensor_contract_summary(bias) if bias is not None else None,
            "groups": int(getattr(conv1d, "groups", 0)),
            "padding": [int(v) for v in getattr(conv1d, "padding", ())],
            "stride": [int(v) for v in getattr(conv1d, "stride", ())],
            "dilation": [int(v) for v in getattr(conv1d, "dilation", ())],
            "kernel_size": [int(v) for v in getattr(conv1d, "kernel_size", ())],
            "activation": str(getattr(mixer, "activation", None)),
            "intermediate_size": int(getattr(mixer, "intermediate_size", 0)),
            "conv_dim": int(getattr(mixer, "conv_dim", 0)),
            "n_groups": int(getattr(mixer, "n_groups", 0)),
            "ssm_state_size": int(getattr(mixer, "ssm_state_size", 0)),
            "semantics": (
                "diagnostic metadata for the selected layer's actual HF Mamba2 "
                "depthwise conv boundary; model forward output is unchanged"
            ),
        }
        summaries.append(metadata)

    def append_actual_conv_window_details(hidden_states_b_c: Any) -> None:
        source = "hf_layer0_mamba2_actual_forward_conv_window"
        if block_type != "mamba" or mixer is None:
            unavailable(
                "layer0_mamba2_actual_conv_window_details",
                source,
                f"layer0 block_type is {block_type}; no Mamba2 mixer",
            )
            return
        tensor = _first_tensor(hidden_states_b_c)
        if tensor is None:
            unavailable("layer0_mamba2_actual_conv_window_details", source, "no pre-conv xBC tensor")
            return
        conv1d = getattr(mixer, "conv1d", None)
        if conv1d is None:
            unavailable("layer0_mamba2_actual_conv_window_details", source, "mixer has no conv1d")
            return
        weight = getattr(conv1d, "weight", None)
        bias = getattr(conv1d, "bias", None)
        if weight is None:
            unavailable("layer0_mamba2_actual_conv_window_details", source, "conv1d has no weight")
            return

        try:
            input_tensor = tensor.detach().contiguous()
            weight_tensor = weight.detach().contiguous()
            bias_tensor = bias.detach().contiguous() if bias is not None else None
            if input_tensor.ndim != 3:
                unavailable(
                    "layer0_mamba2_actual_conv_window_details",
                    source,
                    f"expected [batch, seq, conv_dim] xBC tensor, got shape {list(input_tensor.shape)}",
                )
                return
            batch = int(input_tensor.shape[0])
            seq_len = int(input_tensor.shape[1])
            conv_dim = int(input_tensor.shape[2])
            if batch < 1 or seq_len < 1:
                unavailable("layer0_mamba2_actual_conv_window_details", source, "empty xBC tensor")
                return
            kernel_width = int(weight_tensor.shape[-1])
            row_index = seq_len - 1
            groups_time_state_size = int(mixer.n_groups) * int(mixer.ssm_state_size)
            component_specs = [
                ("x", 0, int(mixer.intermediate_size)),
                ("b", int(mixer.intermediate_size), groups_time_state_size),
                ("c", int(mixer.intermediate_size) + groups_time_state_size, groups_time_state_size),
            ]

            def component_detail_indices(width: int) -> List[int]:
                candidates = {0, 1, 2, 3, max(0, width // 2), max(0, width - 1)}
                candidates.update(int(dim) for dim in selected_element_dims if 0 <= int(dim) < width)
                return sorted(idx for idx in candidates if 0 <= idx < width)

            def scalar_bf16_detail(value: Any) -> Dict[str, Any]:
                scalar = value.detach().reshape(1)
                if scalar.dtype != torch.bfloat16:
                    scalar = scalar.to(torch.bfloat16)
                bits = int(bf16_u16_values(scalar)[0])
                return {
                    "bf16_bits": bits,
                    "bf16_bits_hex": f"0x{bits & 0xFFFF:04x}",
                    "value": float(scalar.to(torch.float32).item()),
                }

            def scalar_f32_detail(value: float) -> Dict[str, Any]:
                value_f32 = struct.unpack("<f", struct.pack("<f", float(value)))[0]
                return {
                    "f32_bits": int(f32_bits(value_f32)),
                    "f32_bits_hex": f"0x{int(f32_bits(value_f32)):08x}",
                    "value": float(value_f32),
                    "bf16_rounded_bits": int(
                        bf16_u16_values(torch.tensor([value_f32], dtype=torch.float32).to(torch.bfloat16))[0]
                    ),
                }

            details: List[Dict[str, Any]] = []
            for component_name, component_offset, component_width in component_specs:
                for component_dim in component_detail_indices(component_width):
                    channel = component_offset + component_dim
                    if channel < 0 or channel >= conv_dim:
                        continue
                    if weight_tensor.ndim == 3:
                        weight_row = weight_tensor[channel, 0, :].reshape(-1)
                    else:
                        weight_row = weight_tensor[channel, :].reshape(-1)
                    if int(weight_row.numel()) != kernel_width:
                        unavailable(
                            "layer0_mamba2_actual_conv_window_details",
                            source,
                            (
                                f"unexpected weight row width for channel {channel}: "
                                f"{int(weight_row.numel())} vs kernel {kernel_width}"
                            ),
                        )
                        return
                    bias_value = (
                        bias_tensor[channel]
                        if bias_tensor is not None and channel < int(bias_tensor.numel())
                        else torch.tensor(0.0, dtype=torch.float32, device=input_tensor.device)
                    )
                    bias_detail = scalar_bf16_detail(bias_value)
                    acc_fp32 = struct.unpack("<f", struct.pack("<f", float(bias_detail["value"])))[0]
                    window = []
                    weight_details = []
                    for k in range(kernel_width):
                        src_pos = row_index + k - (kernel_width - 1)
                        if src_pos >= 0:
                            input_value = input_tensor[0, src_pos, channel]
                            input_detail = scalar_bf16_detail(input_value)
                            source_kind = "sequence_input"
                        else:
                            input_detail = {
                                "bf16_bits": 0,
                                "bf16_bits_hex": "0x0000",
                                "value": 0.0,
                            }
                            source_kind = "zero_padding"
                        weight_detail = scalar_bf16_detail(weight_row[k])
                        acc_fp32 = struct.unpack(
                            "<f",
                            struct.pack(
                                "<f",
                                float(acc_fp32)
                                + float(input_detail["value"]) * float(weight_detail["value"]),
                            ),
                        )[0]
                        window.append(
                            {
                                "kernel_index": int(k),
                                "source_position": int(src_pos),
                                "source_kind": source_kind,
                                "input": input_detail,
                            }
                        )
                        weight_details.append(
                            {
                                "kernel_index": int(k),
                                "weight": weight_detail,
                            }
                        )
                    silu_fp32 = acc_fp32 / (1.0 + math.exp(-acc_fp32))
                    silu_bf16_bits = int(
                        bf16_u16_values(torch.tensor([silu_fp32], dtype=torch.float32).to(torch.bfloat16))[0]
                    )
                    details.append(
                        {
                            "component": component_name,
                            "component_dim": int(component_dim),
                            "conv_channel": int(channel),
                            "row_index": int(row_index),
                            "kernel_width": int(kernel_width),
                            "padding": [int(v) for v in getattr(conv1d, "padding", ())],
                            "state_used_for_selected_row": any(item["source_kind"] != "sequence_input" for item in window),
                            "window": window,
                            "weights": weight_details,
                            "bias": bias_detail,
                            "manual_acc_fp32_from_bf16_inputs": scalar_f32_detail(acc_fp32),
                            "manual_silu_fp32_from_bf16_inputs": scalar_f32_detail(silu_fp32),
                            "manual_silu_bf16_bits": int(silu_bf16_bits),
                            "manual_silu_bf16_bits_hex": f"0x{silu_bf16_bits & 0xFFFF:04x}",
                            "manual_silu_bf16_value": float(
                                torch.tensor([silu_fp32], dtype=torch.float32)
                                .to(torch.bfloat16)
                                .to(torch.float32)
                                .item()
                            ),
                        }
                    )
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": 0,
                    "label": "layer0_mamba2_actual_conv_window_details",
                    "source": source,
                    "block_type": block_type,
                    "row_index": int(row_index),
                    "conv_dim": int(conv_dim),
                    "kernel_width": int(kernel_width),
                    "detail_count": len(details),
                    "selected_dims": selected_element_dims,
                    "semantics": (
                        "diagnostic-only selected-channel depthwise conv formula inputs "
                        "for the actual HF Mamba2 prefill path: source window, BF16 "
                        "weight/bias values, and BF16-input manual accumulation candidate"
                    ),
                    "details": details,
                }
            )
        except Exception as exc:
            unavailable("layer0_mamba2_actual_conv_window_details", source, f"{type(exc).__name__}: {exc}")

    def record_flat(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": 0,
                    "label": label,
                    "source": source,
                    "available": False,
                    "reason": "no tensor value",
                    "block_type": block_type,
                }
            )
            return
        summary = _tensor_flat_summary(
            tensor,
            len(summaries),
            label=label,
            source=source,
            layer=0,
        )
        summary["block_type"] = block_type
        summaries.append(summary)

    mixer = getattr(block, "mixer", None)
    mixer_module = sys.modules.get(mixer.__class__.__module__) if mixer is not None else None
    scan_fn = getattr(mixer_module, "mamba_chunk_scan_combined", None) if mixer_module is not None else None
    if block_type == "mamba" and scan_fn is not None:
        forward_scan_globals = getattr(scan_fn, "__globals__", None)
        if forward_scan_globals is not None:
            forward_original_chunk_cumsum = forward_scan_globals.get("_chunk_cumsum_fwd")
            forward_original_chunk_state = forward_scan_globals.get("_chunk_state_fwd")
            forward_original_state_passing = forward_scan_globals.get("_state_passing_fwd")
            forward_original_bmm_chunk = forward_scan_globals.get("_bmm_chunk_fwd")
            forward_original_chunk_scan = forward_scan_globals.get("_chunk_scan_fwd")

    def append_actual_preconv_split(value: Any) -> None:
        if actual_preconv_split_capture["done"]:
            return
        actual_preconv_split_capture["done"] = True
        source = "hf_layer0_mamba2_actual_forward_preconv_split"
        if block_type != "mamba" or mixer is None:
            unavailable(
                "layer0_mamba2_actual_preconv_split",
                source,
                f"layer0 block_type is {block_type}; no Mamba2 mixer",
            )
            return
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable("layer0_mamba2_actual_preconv_split", source, "no mixer input tensor")
            return
        in_proj = getattr(mixer, "in_proj", None)
        if in_proj is None:
            unavailable("layer0_mamba2_actual_preconv_split", source, "mixer has no in_proj")
            return

        try:
            with torch.no_grad():
                projected_states = in_proj(tensor)
                groups_time_state_size = int(mixer.n_groups) * int(mixer.ssm_state_size)
                d_to_remove = (
                    2 * int(mixer.intermediate_size)
                    + 2 * groups_time_state_size
                    + int(mixer.num_heads)
                )
                projected_width = int(projected_states.shape[-1])
                d_mlp = (projected_width - d_to_remove) // 2
                expected_width = 2 * d_mlp + d_to_remove
                if d_mlp < 0 or expected_width != projected_width:
                    unavailable(
                        "layer0_mamba2_actual_preconv_split",
                        source,
                        (
                            "invalid split dimensions: "
                            f"projected_width={projected_width}, d_mlp={d_mlp}, "
                            f"expected_width={expected_width}"
                        ),
                    )
                    return
                split_projection_dim = [
                    d_mlp,
                    d_mlp,
                    int(mixer.intermediate_size),
                    int(mixer.conv_dim),
                    int(mixer.num_heads),
                ]
                _, _, gate, hidden_states_b_c, dt = torch.split(
                    projected_states,
                    split_projection_dim,
                    dim=-1,
                )
                hidden_raw, b_raw, c_raw = torch.split(
                    hidden_states_b_c,
                    [
                        int(mixer.intermediate_size),
                        groups_time_state_size,
                        groups_time_state_size,
                    ],
                    dim=-1,
                )
        except Exception as exc:
            unavailable(
                "layer0_mamba2_actual_preconv_split",
                source,
                f"{type(exc).__name__}: {exc}",
            )
            return

        def append_preconv_exact_row_details(label: str, value: Any) -> None:
            tensor = _first_tensor(value)
            if tensor is None:
                unavailable(f"{label}_selected_row_details", f"{source}_exact_last_row", "no tensor value")
                return
            row = tensor.reshape(-1, tensor.shape[-1])[-1].detach().contiguous()
            width = int(row.numel())
            detail_indices = sorted(
                {
                    0,
                    1,
                    2,
                    3,
                    max(0, width // 2),
                    max(0, width - 1),
                }
                | {int(dim) for dim in selected_element_dims if 0 <= int(dim) < width}
            )
            details: List[Dict[str, Any]] = []
            summary: Dict[str, Any] = {
                "index": len(summaries),
                "layer": 0,
                "label": f"{label}_selected_row_details",
                "source": f"{source}_exact_last_row",
                "dtype": str(row.dtype),
                "device": str(row.device),
                "row_width": width,
                "selected_dims": detail_indices,
                "block_type": block_type,
            }
            row_f32 = row.to(torch.float32).detach().cpu().reshape(-1)
            if row.dtype == torch.bfloat16:
                bits = bf16_u16_values(row)
                summary["hash_fnv1a64_bf16_bits"] = fnv1a_u16(bits)
                for idx in detail_indices:
                    details.append(
                        {
                            "dim_index": int(idx),
                            "bf16_bits": int(bits[idx]),
                            "bf16_bits_hex": f"0x{int(bits[idx]) & 0xFFFF:04x}",
                            "value": float(row_f32[idx].item()),
                        }
                    )
            else:
                summary["sha256_f32"] = hashlib.sha256(row_f32.contiguous().numpy().tobytes()).hexdigest()
                for idx in detail_indices:
                    value_f32 = float(row_f32[idx].item())
                    details.append(
                        {
                            "dim_index": int(idx),
                            "f32_bits": int(f32_bits(value_f32)),
                            "f32_bits_hex": f"0x{int(f32_bits(value_f32)):08x}",
                            "value": value_f32,
                        }
                    )
            summary["detail_count"] = len(details)
            summary["details"] = details
            summaries.append(summary)

        summaries.append(
            {
                "index": len(summaries),
                "layer": 0,
                "label": "layer0_mamba2_actual_preconv_split_metadata",
                "source": source,
                "block_type": block_type,
                "projected_width": int(projected_width),
                "d_mlp": int(d_mlp),
                "intermediate_size": int(mixer.intermediate_size),
                "conv_dim": int(mixer.conv_dim),
                "num_heads": int(mixer.num_heads),
                "n_groups": int(mixer.n_groups),
                "ssm_state_size": int(mixer.ssm_state_size),
                "semantics": (
                    "diagnostic replay of the selected layer's actual mixer input "
                    "through HF mixer.in_proj, split before conv into gate, raw "
                    "pre-conv x/b/c, and raw dt; model forward output is unchanged"
                ),
            }
        )
        record("layer0_mamba2_actual_preconv_in_proj", source, projected_states)
        record("layer0_mamba2_actual_preconv_raw_gate", source, gate)
        record("layer0_mamba2_actual_preconv_xbc", source, hidden_states_b_c)
        record("layer0_mamba2_actual_preconv_raw_x", source, hidden_raw)
        record("layer0_mamba2_actual_preconv_raw_b", source, b_raw)
        record("layer0_mamba2_actual_preconv_raw_c", source, c_raw)
        record("layer0_mamba2_actual_preconv_raw_dt", source, dt)
        append_bf16_selected_row_details("layer0_mamba2_actual_preconv_in_proj", source, projected_states)
        append_bf16_selected_row_details("layer0_mamba2_actual_preconv_raw_x", source, hidden_raw)
        append_bf16_selected_row_details("layer0_mamba2_actual_preconv_raw_b", source, b_raw)
        append_bf16_selected_row_details("layer0_mamba2_actual_preconv_raw_c", source, c_raw)
        append_bf16_selected_row_details("layer0_mamba2_actual_preconv_raw_dt", source, dt)
        append_preconv_exact_row_details("layer0_mamba2_actual_preconv_raw_x", hidden_raw)
        append_preconv_exact_row_details("layer0_mamba2_actual_preconv_raw_b", b_raw)
        append_preconv_exact_row_details("layer0_mamba2_actual_preconv_raw_c", c_raw)
        append_preconv_exact_row_details("layer0_mamba2_actual_preconv_raw_dt", dt)
        append_actual_conv_contract_metadata()
        append_actual_conv_window_details(hidden_states_b_c)

    if forward_original_chunk_cumsum is not None:
        def wrapped_chunk_cumsum_fwd(
            dt: Any,
            A: Any,
            chunk_size: Any,
            dt_bias: Any = None,
            dt_softplus: bool = False,
            dt_limit: Any = (0.0, float("inf")),
        ) -> Any:
            capture_this_call = mamba_forward_active["active"] and not forward_chunk_capture["done"]
            if capture_this_call:
                forward_chunk_capture["done"] = True
                record(
                    "layer0_mamba2_forward_raw_dt",
                    "hf_layer0_mamba2_actual_forward_chunk_cumsum_input",
                    dt,
                )
                dt_bias_name, dt_bias_param = _find_layer_dt_bias_parameter(model, layer_idx)
                if dt_bias_param is not None:
                    summary = _dt_bias_flat_summary(
                        dt_bias_param.detach(),
                        index=len(summaries),
                        label="layer0_mamba2_dt_bias_pre_chunk_cumsum_param",
                        source="hf_model_parameter_pre_chunk_cumsum_call",
                    )
                    summary["block_type"] = block_type
                    summary["parameter_name"] = dt_bias_name
                    summaries.append(summary)
                else:
                    summaries.append(
                        {
                            "index": len(summaries),
                            "layer": 0,
                            "label": "layer0_mamba2_dt_bias_pre_chunk_cumsum_param",
                            "source": "hf_model_parameter_pre_chunk_cumsum_call",
                            "available": False,
                            "reason": "could not locate layer-0 mixer dt_bias parameter",
                            "block_type": block_type,
                        }
                    )
                if dt_bias is not None:
                    record_flat(
                        "layer0_mamba2_forward_dt_bias",
                        "hf_layer0_mamba2_actual_forward_chunk_cumsum_input",
                        dt_bias,
                    )
                    record(
                        "layer0_mamba2_forward_dt_plus_bias_fp32",
                        "hf_layer0_mamba2_actual_forward_chunk_cumsum_fp32_add",
                        dt.float() + dt_bias.float().view(1, 1, -1),
                    )
                else:
                    summaries.append(
                        {
                            "index": len(summaries),
                            "layer": 0,
                            "label": "layer0_mamba2_forward_dt_bias",
                            "source": "hf_layer0_mamba2_actual_forward_chunk_cumsum_input",
                            "available": False,
                            "reason": "actual forward call did not pass dt_bias",
                            "block_type": block_type,
                        }
                    )
                forward_chunk_cumsum_capture.clear()
                forward_chunk_cumsum_capture.update(
                    {
                        "raw_dt": dt.detach(),
                        "dt_bias": dt_bias.detach() if dt_bias is not None else None,
                        "A": A.detach(),
                        "A_log": mixer.A_log.detach().float(),
                        "dt_softplus_enabled": bool(dt_softplus),
                        "dt_limit": [str(v) for v in dt_limit],
                    }
                )
            dA_cumsum, dt_out = forward_original_chunk_cumsum(
                dt,
                A,
                chunk_size,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                dt_limit=dt_limit,
            )
            if capture_this_call:
                batch, seqlen, nheads = dt.shape
                dt_out_seq = (
                    dt_out.permute(0, 2, 3, 1)
                    .reshape(batch, -1, nheads)[:, :seqlen, :]
                    .contiguous()
                )
                dA_cumsum_seq = (
                    dA_cumsum.permute(0, 2, 3, 1)
                    .reshape(batch, -1, nheads)[:, :seqlen, :]
                    .contiguous()
                )
                forward_chunk_cumsum_capture.update(
                    {
                        "dt_out_seq": dt_out_seq.detach(),
                        "dA_cumsum_seq": dA_cumsum_seq.detach(),
                    }
                )
                record(
                    "layer0_mamba2_forward_da_last",
                    "hf_layer0_mamba2_actual_forward_chunk_cumsum_da",
                    dt_out_seq * A.float().view(1, 1, -1),
                )
                record(
                    "layer0_mamba2_forward_da_cumsum_last",
                    "hf_layer0_mamba2_actual_forward_chunk_cumsum_da",
                    dA_cumsum_seq,
                )
                record(
                    "layer0_mamba2_forward_dt_softplus",
                    "hf_layer0_mamba2_actual_forward_chunk_cumsum_dt_out",
                    dt_out_seq,
                )
            return dA_cumsum, dt_out

        forward_scan_globals["_chunk_cumsum_fwd"] = wrapped_chunk_cumsum_fwd

    if forward_original_chunk_state is not None:
        def wrapped_chunk_state_fwd(
            B: Any,
            x: Any,
            dt: Any,
            dA_cumsum: Any,
            seq_idx: Any = None,
            states: Any = None,
            states_in_fp32: bool = True,
        ) -> Any:
            capture_this_call = mamba_forward_active["active"] and not forward_state_capture["chunk_state_done"]
            if capture_this_call:
                forward_state_capture["chunk_state_done"] = True
                record(
                    "layer0_mamba2_forward_chunk_state_x",
                    "hf_layer0_mamba2_actual_forward_chunk_state_input",
                    x,
                )
                record(
                    "layer0_mamba2_forward_chunk_state_b",
                    "hf_layer0_mamba2_actual_forward_chunk_state_input",
                    B,
                )
                append_bf16_exact_row_details(
                    "layer0_mamba2_actual_postconv_x",
                    "hf_layer0_mamba2_actual_forward_conv_boundary",
                    x,
                )
                append_bf16_selected_row_details(
                    "layer0_mamba2_actual_postconv_x",
                    "hf_layer0_mamba2_actual_forward_conv_boundary",
                    x,
                )
                append_bf16_exact_row_details(
                    "layer0_mamba2_actual_postconv_b",
                    "hf_layer0_mamba2_actual_forward_conv_boundary",
                    B,
                )
                append_bf16_selected_row_details(
                    "layer0_mamba2_actual_postconv_b",
                    "hf_layer0_mamba2_actual_forward_conv_boundary",
                    B,
                )
                record(
                    "layer0_mamba2_forward_chunk_state_dt",
                    "hf_layer0_mamba2_actual_forward_chunk_state_input",
                    dt,
                )
                record(
                    "layer0_mamba2_forward_chunk_state_da_cumsum",
                    "hf_layer0_mamba2_actual_forward_chunk_state_input",
                    dA_cumsum,
                )
            out_states = forward_original_chunk_state(
                B,
                x,
                dt,
                dA_cumsum,
                seq_idx=seq_idx,
                states=states,
                states_in_fp32=states_in_fp32,
            )
            if capture_this_call:
                record_flat(
                    "layer0_mamba2_forward_chunk_state_out",
                    "hf_layer0_mamba2_actual_forward_chunk_state_output",
                    out_states,
                )
            return out_states

        forward_scan_globals["_chunk_state_fwd"] = wrapped_chunk_state_fwd

    if forward_original_state_passing is not None:
        def wrapped_state_passing_fwd(
            states: Any,
            dA_chunk_cumsum: Any,
            initial_states: Any = None,
            seq_idx: Any = None,
            chunk_size: Any = None,
            out_dtype: Any = None,
        ) -> Any:
            capture_this_call = mamba_forward_active["active"] and not forward_state_capture["state_passing_done"]
            if capture_this_call:
                forward_state_capture["state_passing_done"] = True
                record_flat(
                    "layer0_mamba2_forward_state_passing_input_states",
                    "hf_layer0_mamba2_actual_forward_state_passing_input",
                    states,
                )
                record(
                    "layer0_mamba2_forward_state_passing_dA_chunk_cumsum",
                    "hf_layer0_mamba2_actual_forward_state_passing_input",
                    dA_chunk_cumsum,
                )
            out_states, final_states = forward_original_state_passing(
                states,
                dA_chunk_cumsum,
                initial_states=initial_states,
                seq_idx=seq_idx,
                chunk_size=chunk_size,
                out_dtype=out_dtype,
            )
            if capture_this_call:
                record_flat(
                    "layer0_mamba2_forward_state_passing_out_states",
                    "hf_layer0_mamba2_actual_forward_state_passing_output",
                    out_states,
                )
                record_flat(
                    "layer0_mamba2_forward_state_passing_final_states",
                    "hf_layer0_mamba2_actual_forward_state_passing_output",
                    final_states,
                )
            return out_states, final_states

        forward_scan_globals["_state_passing_fwd"] = wrapped_state_passing_fwd

    if forward_original_bmm_chunk is not None:
        def wrapped_bmm_chunk_fwd(
            a: Any,
            b: Any,
            chunk_size: Any,
            seq_idx: Any = None,
            causal: bool = False,
            output_dtype: Any = None,
        ) -> Any:
            capture_this_call = mamba_forward_active["active"] and not forward_state_capture["bmm_chunk_done"]
            if capture_this_call:
                forward_state_capture["bmm_chunk_done"] = True
                record(
                    "layer0_mamba2_forward_bmm_chunk_c_input",
                    "hf_layer0_mamba2_actual_forward_bmm_chunk_input",
                    a,
                )
                append_bf16_exact_row_details(
                    "layer0_mamba2_actual_postconv_c",
                    "hf_layer0_mamba2_actual_forward_conv_boundary",
                    a,
                )
                append_bf16_selected_row_details(
                    "layer0_mamba2_actual_postconv_c",
                    "hf_layer0_mamba2_actual_forward_conv_boundary",
                    a,
                )
                record(
                    "layer0_mamba2_forward_bmm_chunk_b_input",
                    "hf_layer0_mamba2_actual_forward_bmm_chunk_input",
                    b,
                )
            out = forward_original_bmm_chunk(
                a,
                b,
                chunk_size,
                seq_idx=seq_idx,
                causal=causal,
                output_dtype=output_dtype,
            )
            if capture_this_call:
                record_flat(
                    "layer0_mamba2_forward_bmm_chunk_cb",
                    "hf_layer0_mamba2_actual_forward_bmm_chunk_output",
                    out,
                )
            return out

        forward_scan_globals["_bmm_chunk_fwd"] = wrapped_bmm_chunk_fwd

    if forward_original_chunk_scan is not None:
        def wrapped_chunk_scan_fwd(
            cb: Any,
            x: Any,
            dt: Any,
            dA_cumsum: Any,
            C: Any,
            states: Any,
            D: Any = None,
            z: Any = None,
            seq_idx: Any = None,
        ) -> Any:
            capture_this_call = mamba_forward_active["active"] and not forward_state_capture["chunk_scan_done"]
            if capture_this_call:
                forward_state_capture["chunk_scan_done"] = True
                record_flat(
                    "layer0_mamba2_forward_chunk_scan_cb_input",
                    "hf_layer0_mamba2_actual_forward_chunk_scan_input",
                    cb,
                )
                record(
                    "layer0_mamba2_forward_chunk_scan_x_input",
                    "hf_layer0_mamba2_actual_forward_chunk_scan_input",
                    x,
                )
                record(
                    "layer0_mamba2_forward_chunk_scan_dt_input",
                    "hf_layer0_mamba2_actual_forward_chunk_scan_input",
                    dt,
                )
                record(
                    "layer0_mamba2_forward_chunk_scan_da_cumsum_input",
                    "hf_layer0_mamba2_actual_forward_chunk_scan_input",
                    dA_cumsum,
                )
                record(
                    "layer0_mamba2_forward_chunk_scan_c_input",
                    "hf_layer0_mamba2_actual_forward_chunk_scan_input",
                    C,
                )
                record_flat(
                    "layer0_mamba2_forward_chunk_scan_states_input",
                    "hf_layer0_mamba2_actual_forward_chunk_scan_input",
                    states,
                )
            out, out_x = forward_original_chunk_scan(
                cb,
                x,
                dt,
                dA_cumsum,
                C,
                states,
                D=D,
                z=z,
                seq_idx=seq_idx,
            )
            if capture_this_call:
                record(
                    "layer0_mamba2_forward_chunk_scan_out",
                    "hf_layer0_mamba2_actual_forward_chunk_scan_output",
                    out,
                )
                append_actual_chunk_scan_store_cast_details(
                    cb,
                    x,
                    dt,
                    dA_cumsum,
                    C,
                    states,
                    D,
                    z,
                    seq_idx,
                    out,
                )
                if out_x is not None:
                    record(
                        "layer0_mamba2_forward_chunk_scan_out_x",
                        "hf_layer0_mamba2_actual_forward_chunk_scan_output",
                        out_x,
                    )
            return out, out_x

        forward_scan_globals["_chunk_scan_fwd"] = wrapped_chunk_scan_fwd

    handles.append(block.register_forward_pre_hook(pre_hook("layer0_input", "hf_layer0_block_pre_hook")))
    if hasattr(block, "norm"):
        handles.append(block.norm.register_forward_pre_hook(pre_hook("layer0_norm_input", "hf_layer0_norm_pre_hook")))
        handles.append(block.norm.register_forward_hook(post_hook("layer0_norm_output", "hf_layer0_norm_forward_hook")))
    if hasattr(block, "mixer"):
        mixer_pre = pre_hook("layer0_mixer_input", "hf_layer0_mixer_pre_hook")
        mixer_post = post_hook("layer0_mixer_output", "hf_layer0_mixer_forward_hook")

        def target_mixer_pre_hook(module: Any, inputs: Any) -> None:
            mamba_forward_active["active"] = True
            mixer_pre(module, inputs)
            append_actual_preconv_split(inputs[0] if inputs else None)

        def target_mixer_post_hook(module: Any, inputs: Any, output: Any) -> None:
            try:
                mixer_post(module, inputs, output)
            finally:
                mamba_forward_active["active"] = False

        handles.append(block.mixer.register_forward_pre_hook(target_mixer_pre_hook))
        handles.append(block.mixer.register_forward_hook(target_mixer_post_hook))
    handles.append(block.register_forward_hook(post_hook("layer0_output", "hf_layer0_block_forward_hook")))

    try:
        with torch.no_grad():
            model(prompt_input_ids, **forward_kwargs)
    finally:
        mamba_forward_active["active"] = False
        if forward_scan_globals is not None and forward_original_chunk_cumsum is not None:
            forward_scan_globals["_chunk_cumsum_fwd"] = forward_original_chunk_cumsum
        if forward_scan_globals is not None and forward_original_chunk_state is not None:
            forward_scan_globals["_chunk_state_fwd"] = forward_original_chunk_state
        if forward_scan_globals is not None and forward_original_state_passing is not None:
            forward_scan_globals["_state_passing_fwd"] = forward_original_state_passing
        if forward_scan_globals is not None and forward_original_bmm_chunk is not None:
            forward_scan_globals["_bmm_chunk_fwd"] = forward_original_bmm_chunk
        if forward_scan_globals is not None and forward_original_chunk_scan is not None:
            forward_scan_globals["_chunk_scan_fwd"] = forward_original_chunk_scan
        for handle in handles:
            handle.remove()

    if layer_idx == 0:
        summaries.extend(
            _capture_hf_layer0_mamba2_internal_summaries(
                model,
                block,
                prompt_input_ids,
                forward_kwargs,
                len(summaries),
                element_dims=selected_element_dims,
                row_indices=selected_row_indices,
            )
        )

    if block_type not in ("mlp", "moe"):
        summaries.append(
            {
                "index": len(summaries),
                "layer": 0,
                "label": "layer0_mlp_router",
                "source": "hf_layer0_metadata",
                "available": False,
                "reason": f"layer0 block_type is {block_type}; no layer-local MLP/router module",
                "block_type": block_type,
            }
        )
    return summaries


def _capture_hf_layer1_internal_summaries(
    model: Any,
    prompt_input_ids: Any,
    forward_kwargs: Dict[str, Any],
    element_dims: Optional[List[int]] = None,
    layer_idx: int = 1,
    row_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    import torch

    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative")
    layer_name = f"layer{layer_idx}"
    source_name = f"hf_layer{layer_idx}"

    def normalize_layer_string(value: str) -> str:
        if layer_idx == 1:
            return value
        return value.replace("hf_layer1", source_name).replace("layer1", layer_name)

    def normalize_layer_value(value: Any) -> Any:
        if layer_idx == 1:
            return value
        if isinstance(value, str):
            return normalize_layer_string(value)
        if isinstance(value, dict):
            return {key: normalize_layer_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize_layer_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(normalize_layer_value(item) for item in value)
        return value

    class LayerSummaryList(list):
        def append(self, item: Any) -> None:
            super().append(normalize_layer_value(item))

    block = _find_hf_block(model, layer_idx)
    summaries: List[Dict[str, Any]] = LayerSummaryList()
    handles: List[Any] = []
    block_type = getattr(block, "block_type", None)
    mixer = getattr(block, "mixer", None)
    norm_element_cache: Dict[str, Any] = {}
    layer1_handoff_cache: Dict[str, Any] = {}
    moe_route_cache: Dict[str, Any] = {}
    selected_element_dims = sorted({int(dim) for dim in (element_dims or []) if int(dim) >= 0})
    selected_row_indices = sorted({int(row) for row in (row_indices or []) if int(row) >= 0})

    layer1_moe_detail_labels = {
        "layer1_input",
        "layer1_norm_input",
        "layer1_norm_output",
        "layer1_mixer_input",
        "layer1_moe_router_input",
        "layer1_moe_e_score_correction_bias",
        "layer1_moe_router_logits",
        "layer1_moe_router_scores",
        "layer1_moe_router_scores_for_choice",
        "layer1_moe_router_scores_for_choice_masked",
        "layer1_moe_router_group_scores",
        "layer1_moe_router_group_idx",
        "layer1_moe_router_choice_topk_indices",
        "layer1_moe_router_choice_topk_scores",
        "layer1_moe_topk_sigmoid_scores",
        "layer1_moe_topk_weights_normalized",
        "layer1_moe_topk_weights_scaled_candidate",
        "layer1_moe_topk_indices",
        "layer1_moe_topk_weights",
        "layer1_moe_latent_input",
        "layer1_moe_routed_latent_output",
        "layer1_moe_routed_output",
        "layer1_moe_routed_output_pre_shared",
        "layer1_moe_shared_output",
        "layer1_moe_shared_add_input_routed",
        "layer1_moe_shared_add_input_shared",
        "layer1_shared_w1_input",
        "layer1_shared_w1_weight_flat",
        "layer1_shared_w1_output",
        "layer1_shared_activation_output",
        "layer1_shared_w2_input",
        "layer1_shared_w2_weight_flat",
        "layer1_shared_w2_output",
        "layer1_mixer_output",
        "layer1_output",
    }

    def fnv1a_u16(values: Any) -> str:
        h = 0xCBF29CE484222325
        for raw in values:
            v = int(raw) & 0xFFFF
            h ^= v & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            h ^= (v >> 8) & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f"0x{h:016x}"

    def bf16_u16_values(tensor_value: Any) -> List[int]:
        return [
            int(v) & 0xFFFF
            for v in tensor_value.detach()
            .contiguous()
            .view(torch.int16)
            .cpu()
            .reshape(-1)
            .tolist()
        ]

    def f32_bits(value: float) -> int:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]

    def detail_indices_for_width(label: str, width: int) -> List[int]:
        detail_indices = list(selected_element_dims)
        if width <= 64:
            detail_indices = list(range(width))
        if not detail_indices:
            detail_indices = [0, 1, 2, 3, width // 2, width - 1]
        if width == 512 and (
            "router" in label or "e_score_correction_bias" in label
        ):
            detail_indices.extend(
                [
                    28,
                    44,
                    69,
                    94,
                    98,
                    104,
                    119,
                    136,
                    165,
                    195,
                    202,
                    205,
                    216,
                    236,
                    250,
                    257,
                    290,
                    300,
                    301,
                    304,
                    368,
                    382,
                    461,
                    473,
                ]
            )
        return sorted({int(idx) for idx in detail_indices if 0 <= int(idx) < width})

    def append_layer1_moe_row_details(label: str, source: str, tensor: Any, row_index: Optional[int]) -> None:
        if label not in layer1_moe_detail_labels:
            return
        detached = tensor.detach()
        if row_index is None:
            if detached.ndim >= 3:
                row = detached[0, -1, :]
            elif detached.ndim == 2:
                row = detached[-1, :]
            else:
                row = detached.reshape(-1)
            output_label = f"{label}_selected_row_details"
            output_source = f"{source}_exact_last_row"
            absolute_row = None
        else:
            if detached.ndim >= 3:
                if row_index >= detached.shape[1]:
                    unavailable(
                        f"{label}_row{row_index}_selected_row_details",
                        f"{source}_selected_row",
                        f"row {row_index} out of range for sequence length {detached.shape[1]}",
                    )
                    return
                row = detached[0, row_index, :]
            elif detached.ndim == 2:
                if row_index >= detached.shape[0]:
                    unavailable(
                        f"{label}_row{row_index}_selected_row_details",
                        f"{source}_selected_row",
                        f"row {row_index} out of range for row count {detached.shape[0]}",
                    )
                    return
                row = detached[row_index, :]
            elif row_index == 0:
                row = detached.reshape(-1)
            else:
                unavailable(
                    f"{label}_row{row_index}_selected_row_details",
                    f"{source}_selected_row",
                    "scalar/flat tensor only has implicit row 0",
                )
                return
            output_label = f"{label}_row{row_index}_selected_row_details"
            output_source = f"{source}_selected_row"
            absolute_row = int(row_index)
        row = row.contiguous()
        width = int(row.numel())
        if width <= 0:
            return
        detail_indices = detail_indices_for_width(label, width)
        row_f32 = row.to(torch.float32).detach().cpu().reshape(-1)
        details: List[Dict[str, Any]] = []
        if row.dtype == torch.bfloat16:
            bits = bf16_u16_values(row)
            row_hash = fnv1a_u16(bits)
            for idx in detail_indices:
                details.append(
                    {
                        "index": int(idx),
                        "bf16_bits": int(bits[idx]),
                        "bf16_bits_hex": f"0x{int(bits[idx]) & 0xFFFF:04x}",
                        "value": float(row_f32[idx].item()),
                    }
                )
        elif row.dtype.is_floating_point:
            values = row_f32.tolist()
            row_hash = hashlib.sha256(row_f32.contiguous().numpy().tobytes()).hexdigest()
            for idx in detail_indices:
                value = float(values[idx])
                details.append(
                    {
                        "index": int(idx),
                        "f32_bits": int(f32_bits(value)),
                        "f32_bits_hex": f"0x{int(f32_bits(value)):08x}",
                        "value": value,
                    }
                )
        else:
            values = [int(v) for v in row.detach().cpu().reshape(-1).tolist()]
            row_hash = hashlib.sha256(
                torch.tensor(values, dtype=torch.int64).contiguous().numpy().tobytes()
            ).hexdigest()
            for idx in detail_indices:
                details.append({"index": int(idx), "value": int(values[idx])})

        top_values: List[Dict[str, Any]] = []
        if label in (
            "layer1_moe_router_logits",
            "layer1_moe_router_scores",
            "layer1_moe_router_scores_for_choice",
            "layer1_moe_router_scores_for_choice_masked",
        ) and width > 0:
            k = min(32, width)
            top = torch.topk(row_f32, k=k)
            top_values = [
                {
                    "rank": int(rank),
                    "index": int(idx),
                    "value": float(value),
                    "f32_bits": int(f32_bits(float(value))),
                    "f32_bits_hex": f"0x{int(f32_bits(float(value))):08x}",
                }
                for rank, (idx, value) in enumerate(zip(top.indices.tolist(), top.values.tolist()))
            ]

        full_values: Optional[List[Any]] = None
        if width <= 64:
            if row.dtype.is_floating_point:
                full_values = [float(v) for v in row_f32.tolist()]
            else:
                full_values = [int(v) for v in row.detach().cpu().reshape(-1).tolist()]

        summaries.append(
            {
                "index": len(summaries),
                "layer": layer_idx,
                "label": output_label,
                "source": output_source,
                "dtype": str(row.dtype),
                "device": str(tensor.device),
                "row_width": width,
                "row_index": absolute_row,
                "absolute_row": absolute_row,
                "selected_dims": detail_indices,
                "details": details,
                "full_values": full_values,
                "top_values": top_values,
                "row_hash": row_hash,
                "block_type": block_type,
            }
        )

    def append_layer1_moe_selected_row_details(label: str, source: str, tensor: Any) -> None:
        append_layer1_moe_row_details(label, source, tensor, None)
        for row_index in selected_row_indices:
            append_layer1_moe_row_details(label, source, tensor, row_index)

    def unavailable(label: str, source: str, reason: str) -> None:
        summaries.append(
            {
                "index": len(summaries),
                "layer": layer_idx,
                "label": label,
                "source": source,
                "available": False,
                "reason": reason,
                "block_type": block_type,
            }
        )

    def record(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable(label, source, "no tensor value")
            return
        summary = _tensor_last_token_summary(
            tensor,
            len(summaries),
            label=label,
            source=source,
            layer=layer_idx,
        )
        summary["block_type"] = block_type
        summaries.append(summary)
        append_layer1_moe_selected_row_details(label, source, tensor)

    def record_flat(label: str, source: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable(label, source, "no tensor value")
            return
        summary = _tensor_flat_summary(
            tensor,
            len(summaries),
            label=label,
            source=source,
            layer=layer_idx,
        )
        summary["block_type"] = block_type
        summaries.append(summary)

    def pre_hook(label: str, source: str):
        def hook(_module: Any, inputs: Any) -> None:
            value = inputs[0] if inputs else None
            record(label, source, value)
            if label == "layer1_input":
                cache_layer1_handoff_row(label, value)

        return hook

    def post_hook(label: str, source: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            record(label, source, output)
            if label in ("layer1_mixer_output", "layer1_output"):
                cache_layer1_handoff_row(label, output)
                if label == "layer1_output":
                    append_layer1_handoff_full_row_bits()

        return hook

    def append_layer1_shared_metadata(shared_module: Any, source: str) -> None:
        up = getattr(shared_module, "up_proj", None)
        down = getattr(shared_module, "down_proj", None)
        summaries.append(
            {
                "index": len(summaries),
                "layer": layer_idx,
                "label": "layer1_shared_expert_branch_metadata",
                "source": source,
                "block_type": block_type,
                "implementation": type(shared_module).__name__,
                "shared_gate_present": False,
                "activation": str(getattr(shared_module, "act_fn", None)),
                "up_proj_weight_shape": list(getattr(getattr(up, "weight", None), "shape", [])),
                "up_proj_weight_dtype": str(getattr(getattr(up, "weight", None), "dtype", None)),
                "down_proj_weight_shape": list(getattr(getattr(down, "weight", None), "shape", [])),
                "down_proj_weight_dtype": str(getattr(getattr(down, "weight", None), "dtype", None)),
                "add_semantics": "hf_moe_forward_hidden_states_plus_shared_experts_residuals",
            }
        )

    def shared_pre_hook(module: Any, inputs: Any) -> None:
        value = inputs[0] if inputs else None
        record("layer1_shared_w1_input", "hf_layer1_shared_experts_pre_hook", value)
        append_layer1_shared_metadata(module, "hf_layer1_shared_experts_pre_hook")

    def shared_post_hook(_module: Any, _inputs: Any, output: Any) -> None:
        record("layer1_moe_shared_output", "hf_layer1_moe_shared_forward_hook", output)
        record("layer1_moe_shared_add_input_shared", "hf_layer1_moe_shared_forward_hook", output)

    def shared_up_pre_hook(module: Any, inputs: Any) -> None:
        value = inputs[0] if inputs else None
        record("layer1_shared_w1_input", "hf_layer1_shared_up_proj_pre_hook", value)
        weight = getattr(module, "weight", None)
        if weight is not None:
            record_flat("layer1_shared_w1_weight_flat", "hf_layer1_shared_up_proj_pre_hook", weight)

    def shared_up_post_hook(_module: Any, _inputs: Any, output: Any) -> None:
        record("layer1_shared_w1_output", "hf_layer1_shared_up_proj_forward_hook", output)

    def shared_down_pre_hook(module: Any, inputs: Any) -> None:
        value = inputs[0] if inputs else None
        record("layer1_shared_activation_output", "hf_layer1_shared_down_proj_pre_hook", value)
        record("layer1_shared_w2_input", "hf_layer1_shared_down_proj_pre_hook", value)
        weight = getattr(module, "weight", None)
        if weight is not None:
            record_flat("layer1_shared_w2_weight_flat", "hf_layer1_shared_down_proj_pre_hook", weight)

    def shared_down_post_hook(_module: Any, _inputs: Any, output: Any) -> None:
        record("layer1_shared_w2_output", "hf_layer1_shared_down_proj_forward_hook", output)

    def append_layer1_moe_contribution_details(value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            unavailable("layer1_moe_expert_contribution_details", "hf_layer1_moe_manual_contribution_probe", "no latent input tensor")
            return
        if mixer is None or not hasattr(mixer, "experts"):
            unavailable("layer1_moe_expert_contribution_details", "hf_layer1_moe_manual_contribution_probe", "mixer has no experts")
            return
        experts = getattr(mixer, "experts", None)
        if experts is None:
            unavailable("layer1_moe_expert_contribution_details", "hf_layer1_moe_manual_contribution_probe", "experts list unavailable")
            return
        latent_row = tensor.reshape(-1, tensor.shape[-1])[-1].detach().contiguous()
        latent_width = int(latent_row.numel())
        latent_dims = sorted({0} | {int(dim) for dim in selected_element_dims if 0 <= int(dim) < latent_width})
        fc2 = getattr(mixer, "fc2_latent_proj", None)
        fc2_weight = getattr(fc2, "weight", None)
        fc2_bias = getattr(fc2, "bias", None)
        hidden_width = int(fc2_weight.shape[0]) if fc2_weight is not None else 0
        hidden_dims = sorted({0} | {int(dim) for dim in selected_element_dims if hidden_width and 0 <= int(dim) < hidden_width})
        route_specs = []
        for route_name in ("hf_corrected_actual", "raw_sigmoid_global"):
            ids = moe_route_cache.get(f"{route_name}_ids")
            weights = moe_route_cache.get(f"{route_name}_scaled_weights")
            normalized_weights = moe_route_cache.get(f"{route_name}_normalized_weights")
            if ids is not None and weights is not None:
                route_specs.append((route_name, ids, weights, normalized_weights))
        if not route_specs:
            unavailable(
                "layer1_moe_expert_contribution_details",
                "hf_layer1_moe_manual_contribution_probe",
                "missing cached route ids/weights",
            )
            return

        def scalar(value_tensor: Any) -> float:
            return float(value_tensor.detach().to(torch.float32).cpu().item())

        def sign(value_float: float) -> str:
            if value_float > 0.0:
                return "positive"
            if value_float < 0.0:
                return "negative"
            return "zero"

        def selected_row_summary(label: str, row: Any, selected_dims: List[int]) -> Dict[str, Any]:
            row = row.detach().contiguous().reshape(-1)
            width = int(row.numel())
            detail_indices = sorted({int(dim) for dim in selected_dims if 0 <= int(dim) < width})
            details: List[Dict[str, Any]] = []
            summary: Dict[str, Any] = {
                "label": label,
                "dtype": str(row.dtype),
                "width": width,
                "selected_dims": detail_indices,
            }
            if row.dtype == torch.bfloat16:
                bits = bf16_u16_values(row)
                row_f32 = row.to(torch.float32).detach().cpu().reshape(-1)
                summary["row_hash_fnv1a64_bf16_bits"] = fnv1a_u16(bits)
                for dim in detail_indices:
                    details.append(
                        {
                            "dim": int(dim),
                            "bf16_bits": int(bits[dim]),
                            "bf16_bits_hex": f"0x{int(bits[dim]) & 0xFFFF:04x}",
                            "value": float(row_f32[dim].item()),
                        }
                    )
            elif row.dtype.is_floating_point:
                row_f32 = row.to(torch.float32).detach().cpu().reshape(-1)
                values = row_f32.tolist()
                summary["row_hash_sha256_f32"] = hashlib.sha256(
                    row_f32.contiguous().numpy().tobytes()
                ).hexdigest()
                for dim in detail_indices:
                    value = float(values[dim])
                    details.append(
                        {
                            "dim": int(dim),
                            "f32_bits": int(f32_bits(value)),
                            "f32_bits_hex": f"0x{int(f32_bits(value)):08x}",
                            "value": value,
                        }
                    )
            else:
                values = [int(v) for v in row.detach().cpu().reshape(-1).tolist()]
                summary["row_hash_sha256_i64"] = hashlib.sha256(
                    torch.tensor(values, dtype=torch.int64).contiguous().numpy().tobytes()
                ).hexdigest()
                for dim in detail_indices:
                    details.append({"dim": int(dim), "value": int(values[dim])})
            summary["details"] = details
            return summary

        def f32_round(value: float) -> float:
            return struct.unpack("<f", struct.pack("<f", float(value)))[0]

        def bf16_cast_detail(value: float) -> Dict[str, Any]:
            tensor_value = torch.tensor([float(value)], dtype=torch.float32).to(dtype=torch.bfloat16)
            bits = int(bf16_u16_values(tensor_value)[0])
            return {
                "bf16_candidate_bits": bits,
                "bf16_candidate_bits_hex": f"0x{bits & 0xFFFF:04x}",
                "bf16_candidate_value": float(tensor_value.float()[0].item()),
            }

        def w1_up_proj_manual_dot_details(
            expert_id: int,
            stages: Dict[str, Any],
            selected_w1_dims: List[int],
        ) -> Dict[str, Any]:
            if not stages.get("available"):
                return {"available": False, "reason": stages.get("reason", "stage unavailable")}
            expert = experts[expert_id]
            up_proj = getattr(expert, "up_proj", None)
            weight = getattr(up_proj, "weight", None)
            if weight is None:
                return {"available": False, "reason": "expert up_proj has no weight"}
            weight_tensor = weight.detach().contiguous()
            if weight_tensor.ndim != 2:
                return {
                    "available": False,
                    "reason": f"expected 2D up_proj weight, got shape {list(weight_tensor.shape)}",
                }
            input_row = latent_row.detach().contiguous().reshape(-1)
            input_width = int(input_row.numel())
            out_width = int(weight_tensor.shape[0])
            in_width = int(weight_tensor.shape[1])
            detail_dims = sorted({int(dim) for dim in selected_w1_dims if 0 <= int(dim) < out_width})
            input_f32_cpu = input_row.to(torch.float32).detach().cpu().reshape(-1)
            actual_row = stages["w1_output"].detach().contiguous().reshape(-1)
            actual_bits = bf16_u16_values(actual_row) if actual_row.dtype == torch.bfloat16 else []
            input_bits = bf16_u16_values(input_row) if input_row.dtype == torch.bfloat16 else []

            dot_details: List[Dict[str, Any]] = []
            for out_dim in detail_dims:
                weight_row = weight_tensor[out_dim].detach().contiguous().reshape(-1)
                row = weight_row.to(torch.float32).detach().cpu()
                row_bits = bf16_u16_values(weight_row) if weight_row.dtype == torch.bfloat16 else []
                common_width = min(input_width, in_width)
                torch_dot = float(torch.sum(input_f32_cpu[:common_width] * row[:common_width]).item())
                sequential_exact = int(out_dim) == 0
                acc = f32_round(0.0)
                if sequential_exact:
                    input_values = input_f32_cpu[:common_width].tolist()
                    row_values = row[:common_width].tolist()
                    for input_idx in range(common_width):
                        product = f32_round(float(input_values[input_idx]) * float(row_values[input_idx]))
                        acc = f32_round(acc + product)
                else:
                    acc = f32_round(torch_dot)
                actual_value = float(actual_row.to(torch.float32).cpu()[out_dim].item())
                detail = {
                    "out_dim": int(out_dim),
                    "actual_dtype": str(actual_row.dtype),
                    "actual_value": actual_value,
                    "actual_bf16_bits": int(actual_bits[out_dim]) if actual_bits else None,
                    "actual_bf16_bits_hex": f"0x{int(actual_bits[out_dim]) & 0xFFFF:04x}" if actual_bits else None,
                    "sequential_fp32_accum_exact": bool(sequential_exact),
                    "sequential_fp32_accum_value": float(acc),
                    "sequential_fp32_accum_bits": int(f32_bits(acc)),
                    "sequential_fp32_accum_bits_hex": f"0x{int(f32_bits(acc)):08x}",
                    "torch_fp32_dot_value": float(torch_dot),
                    "torch_fp32_dot_bits": int(f32_bits(torch_dot)),
                    "torch_fp32_dot_bits_hex": f"0x{int(f32_bits(torch_dot)):08x}",
                }
                cast = bf16_cast_detail(acc)
                detail.update(cast)
                detail["sequential_candidate_matches_actual_bf16"] = (
                    bool(actual_bits) and int(actual_bits[out_dim]) == int(cast["bf16_candidate_bits"])
                )
                if row_bits:
                    detail["weight_row_hash_fnv1a64_bf16_bits"] = fnv1a_u16(row_bits)
                dot_details.append(detail)

            top_contrib: List[Dict[str, Any]] = []
            if out_width > 0 and input_width > 0 and in_width > 0:
                weight_row0 = weight_tensor[0].detach().contiguous().reshape(-1)
                row0 = weight_row0.to(torch.float32).detach().cpu()
                row0_bits = bf16_u16_values(weight_row0) if weight_row0.dtype == torch.bfloat16 else []
                contrib = input_f32_cpu[:in_width] * row0[:input_width]
                k = min(16, int(contrib.numel()))
                top = torch.topk(torch.abs(contrib), k=k) if k > 0 else None
                if top is not None:
                    for rank, input_idx in enumerate(top.indices.tolist()):
                        idx = int(input_idx)
                        input_value = float(input_f32_cpu[idx].item())
                        weight_value = float(row0[idx].item())
                        contribution = f32_round(input_value * weight_value)
                        top_contrib.append(
                            {
                                "rank": int(rank),
                                "input_dim": idx,
                                "input_value": input_value,
                                "input_bf16_bits": int(input_bits[idx]) if input_bits else None,
                                "input_bf16_bits_hex": f"0x{int(input_bits[idx]) & 0xFFFF:04x}" if input_bits else None,
                                "weight_value": weight_value,
                                "weight_bf16_bits": int(row0_bits[idx]) if row0_bits else None,
                                "weight_bf16_bits_hex": f"0x{int(row0_bits[idx]) & 0xFFFF:04x}" if row0_bits else None,
                                "contribution": float(contribution),
                                "contribution_bits": int(f32_bits(contribution)),
                                "contribution_bits_hex": f"0x{int(f32_bits(contribution)):08x}",
                                "contribution_sign": sign(float(contribution)),
                            }
                        )

            return {
                "available": True,
                "expert_id": int(expert_id),
                "input_dtype": str(input_row.dtype),
                "input_width": int(input_width),
                "weight_dtype": str(weight_tensor.dtype),
                "weight_shape": [int(v) for v in weight_tensor.shape],
                "weight_layout": "torch linear row-major [out_features, in_features]; output = input @ weight.T",
                "quant_dequant_scales": "not_applicable_hf_bf16_weight",
                "selected_output_dims": detail_dims,
                "selected_output_dim_details": dot_details,
                "dim0_top_input_contributions": top_contrib,
                "input_hash_fnv1a64_bf16_bits": fnv1a_u16(input_bits) if input_bits else None,
                "actual_w1_hash_fnv1a64_bf16_bits": fnv1a_u16(actual_bits) if actual_bits else None,
            }

        expert_stage_cache: Dict[int, Dict[str, Any]] = {}

        def get_expert_stages(expert_id: int) -> Dict[str, Any]:
            cached = expert_stage_cache.get(expert_id)
            if cached is not None:
                return cached
            expert = experts[expert_id]
            if not all(hasattr(expert, name) for name in ("up_proj", "act_fn", "down_proj")):
                output = expert(latent_row.unsqueeze(0)).reshape(-1).detach().contiguous()
                cached = {
                    "available": False,
                    "reason": "expert does not expose up_proj/act_fn/down_proj",
                    "w2_output": output,
                }
                expert_stage_cache[expert_id] = cached
                return cached
            w1_output = expert.up_proj(latent_row.unsqueeze(0)).reshape(-1).detach().contiguous()
            activation_output = expert.act_fn(w1_output).reshape(-1).detach().contiguous()
            w2_output = expert.down_proj(activation_output.unsqueeze(0)).reshape(-1).detach().contiguous()
            cached = {
                "available": True,
                "w1_output": w1_output,
                "activation_output": activation_output,
                "w2_output": w2_output,
            }
            expert_stage_cache[expert_id] = cached
            return cached

        def get_expert_output(expert_id: int) -> Any:
            return get_expert_stages(expert_id)["w2_output"]

        def w2_dim0_contribution_details(stages: Dict[str, Any], expert_id: int) -> List[Dict[str, Any]]:
            if not stages.get("available"):
                return []
            expert = experts[expert_id]
            down_proj = getattr(expert, "down_proj", None)
            weight = getattr(down_proj, "weight", None)
            if weight is None or int(weight.shape[0]) <= 0:
                return []
            activation = stages["activation_output"].detach().to(torch.float32).reshape(-1)
            weight_row = weight.detach().to(torch.float32)[0].reshape(-1)
            contrib = activation * weight_row
            k = min(16, int(contrib.numel()))
            if k <= 0:
                return []
            top = torch.topk(torch.abs(contrib), k=k)
            rows: List[Dict[str, Any]] = []
            for rank, dim in enumerate(top.indices.tolist()):
                dim_i = int(dim)
                activation_value = float(activation[dim_i].detach().cpu().item())
                weight_value = float(weight_row[dim_i].detach().cpu().item())
                contribution = float(contrib[dim_i].detach().cpu().item())
                rows.append(
                    {
                        "rank": int(rank),
                        "dim": dim_i,
                        "activation_value": activation_value,
                        "activation_f32_bits": int(f32_bits(activation_value)),
                        "activation_f32_bits_hex": f"0x{int(f32_bits(activation_value)):08x}",
                        "w2_weight_value": weight_value,
                        "w2_weight_f32_bits": int(f32_bits(weight_value)),
                        "w2_weight_f32_bits_hex": f"0x{int(f32_bits(weight_value)):08x}",
                        "contribution": contribution,
                        "contribution_sign": sign(contribution),
                    }
                )
            return rows

        routes: List[Dict[str, Any]] = []
        with torch.no_grad():
            fc2_weight_f32 = fc2_weight.detach().to(torch.float32) if fc2_weight is not None else None
            for route_name, route_ids_tensor, route_weights_tensor, route_norm_weights_tensor in route_specs:
                route_ids = [int(v) for v in route_ids_tensor.detach().cpu().reshape(-1).tolist()]
                scaled_weights = [
                    float(v) for v in route_weights_tensor.detach().to(torch.float32).cpu().reshape(-1).tolist()
                ]
                if route_norm_weights_tensor is not None:
                    normalized_weights = [
                        float(v)
                        for v in route_norm_weights_tensor.detach().to(torch.float32).cpu().reshape(-1).tolist()
                    ]
                else:
                    normalized_weights = [float("nan") for _ in scaled_weights]
                aggregate_f32 = torch.zeros(latent_width, dtype=torch.float32, device=latent_row.device)
                slot_details = []
                for slot, expert_id in enumerate(route_ids):
                    scaled_weight = float(scaled_weights[slot]) if slot < len(scaled_weights) else 0.0
                    normalized_weight = (
                        float(normalized_weights[slot]) if slot < len(normalized_weights) else float("nan")
                    )
                    expert_stages = get_expert_stages(expert_id)
                    expert_output = expert_stages["w2_output"].to(torch.float32)
                    weighted = expert_output * torch.tensor(
                        scaled_weight,
                        dtype=torch.float32,
                        device=expert_output.device,
                    )
                    aggregate_f32 = aggregate_f32 + weighted
                    latent_detail = []
                    for dim in latent_dims:
                        source_value = scalar(expert_output[dim])
                        contribution = scalar(weighted[dim])
                        latent_detail.append(
                            {
                                "dim": int(dim),
                                "expert_output_value": source_value,
                                "weighted_contribution": contribution,
                                "weighted_contribution_sign": sign(contribution),
                            }
                        )
                    hidden_detail = []
                    if fc2_weight_f32 is not None:
                        for dim in hidden_dims:
                            contribution = scalar(torch.sum(weighted * fc2_weight_f32[dim]))
                            hidden_detail.append(
                                {
                                    "dim": int(dim),
                                    "manual_pre_aggregate_cast_dot_contribution": contribution,
                                    "manual_pre_aggregate_cast_dot_contribution_sign": sign(contribution),
                                }
                            )
                    stage_details: Dict[str, Any] = {
                        "available": bool(expert_stages.get("available", False)),
                        "reason": expert_stages.get("reason"),
                    }
                    if expert_stages.get("available"):
                        w1_width = int(expert_stages["w1_output"].numel())
                        w1_dims = sorted(
                            {
                                0,
                                1,
                                2,
                                3,
                                w1_width // 2,
                                max(0, w1_width - 1),
                            }
                            | {int(dim) for dim in selected_element_dims if 0 <= int(dim) < w1_width}
                        )
                        top_contrib = w2_dim0_contribution_details(expert_stages, int(expert_id))
                        top_contrib_dims = [int(row["dim"]) for row in top_contrib]
                        activation_dims = sorted(set(w1_dims + top_contrib_dims))
                        stage_details.update(
                            {
                                "w1_output": selected_row_summary(
                                    "w1_output",
                                    expert_stages["w1_output"],
                                    activation_dims,
                                ),
                                "activation_output": selected_row_summary(
                                    "activation_output",
                                    expert_stages["activation_output"],
                                    activation_dims,
                                ),
                                "w2_output": selected_row_summary(
                                    "w2_output",
                                    expert_stages["w2_output"],
                                    latent_dims,
                                ),
                                "w2_dim0_top_activation_contributions": top_contrib,
                                "w1_up_proj_manual_dot_details": w1_up_proj_manual_dot_details(
                                    int(expert_id),
                                    expert_stages,
                                    w1_dims,
                                ),
                            }
                        )
                    slot_details.append(
                        {
                            "slot": int(slot),
                            "expert_id": int(expert_id),
                            "selected_by_route": True,
                            "normalized_uncorrected_sigmoid_weight": normalized_weight,
                            "scaled_weight": scaled_weight,
                            "is_boundary_expert_28": int(expert_id) == 28,
                            "is_boundary_expert_473": int(expert_id) == 473,
                            "latent_dim_details": latent_detail,
                            "hidden_dim_manual_dot_details": hidden_detail,
                            "expert_stage_details": stage_details,
                        }
                    )

                aggregate_bf16 = aggregate_f32.to(dtype=latent_row.dtype)
                aggregate_bits = bf16_u16_values(aggregate_bf16) if aggregate_bf16.dtype == torch.bfloat16 else []
                aggregate_details = []
                for dim in latent_dims:
                    aggregate_details.append(
                        {
                            "dim": int(dim),
                            "aggregate_fp32_pre_bf16": scalar(aggregate_f32[dim]),
                            "aggregate_bf16_bits": int(aggregate_bits[dim]) if aggregate_bits else None,
                            "aggregate_bf16_bits_hex": f"0x{int(aggregate_bits[dim]) & 0xFFFF:04x}" if aggregate_bits else None,
                            "aggregate_bf16_value": scalar(aggregate_bf16[dim]),
                        }
                    )
                hidden_candidate_details = []
                if fc2_weight is not None:
                    hidden_candidate = torch.nn.functional.linear(
                        aggregate_bf16.unsqueeze(0),
                        fc2_weight,
                        fc2_bias,
                    ).reshape(-1).detach().contiguous()
                    hidden_bits = (
                        bf16_u16_values(hidden_candidate)
                        if hidden_candidate.dtype == torch.bfloat16
                        else []
                    )
                    for dim in hidden_dims:
                        hidden_candidate_details.append(
                            {
                                "dim": int(dim),
                                "hidden_candidate_bits": int(hidden_bits[dim]) if hidden_bits else None,
                                "hidden_candidate_bits_hex": f"0x{int(hidden_bits[dim]) & 0xFFFF:04x}" if hidden_bits else None,
                                "hidden_candidate_value": scalar(hidden_candidate[dim]),
                            }
                        )

                routes.append(
                    {
                        "route_name": route_name,
                        "selected_expert_ids": route_ids,
                        "normalized_uncorrected_sigmoid_weights": normalized_weights,
                        "scaled_weights": scaled_weights,
                        "selected_expert_order_semantics": "torch.topk sorted=False order used by HF route" if route_name == "hf_corrected_actual" else "global raw-sigmoid torch.topk sorted=False diagnostic order",
                        "contains_expert_28": bool(28 in route_ids),
                        "contains_expert_473": bool(473 in route_ids),
                        "aggregate_latent_selected_dims": aggregate_details,
                        "hidden_candidate_selected_dims": hidden_candidate_details,
                        "slot_details": slot_details,
                    }
                )

        summaries.append(
            {
                "index": len(summaries),
                "layer": layer_idx,
                "label": "layer1_moe_expert_contribution_details",
                "source": "hf_layer1_moe_manual_contribution_probe",
                "dtype": "mixed",
                "device": str(latent_row.device),
                "latent_width": int(latent_width),
                "hidden_width": int(hidden_width),
                "selected_latent_dims": latent_dims,
                "selected_hidden_dims": hidden_dims,
                "route_count": len(routes),
                "routes": routes,
                "boundary_experts": [28, 473],
                "semantics": "diagnostic recompute of per-expert outputs; selected weights are uncorrected sigmoid scores normalized and scaled after route selection; expert_stage_details split up_proj, activation, and down_proj without changing the forward path",
                "block_type": block_type,
            }
        )

    def latent_input_post_hook(_module: Any, _inputs: Any, output: Any) -> None:
        record("layer1_moe_latent_input", "hf_layer1_moe_fc1_latent_forward_hook", output)
        append_layer1_moe_contribution_details(output)

    def cache_layer1_handoff_row(label: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        row = tensor.reshape(-1, tensor.shape[-1])[-1].detach().contiguous()
        if row.dtype != torch.bfloat16:
            layer1_handoff_cache[label] = {
                "available": False,
                "reason": f"expected BF16 tensor, got {row.dtype}",
                "dtype": str(row.dtype),
            }
            return
        bits = bf16_u16_values(row)
        layer1_handoff_cache[label] = {
            "available": True,
            "dtype": str(row.dtype),
            "device": str(row.device),
            "width": int(row.numel()),
            "bits": bits,
            "values": row.to(torch.float32).cpu().reshape(-1).tolist(),
            "hash_fnv1a_u16": fnv1a_u16(bits),
        }

    def append_layer1_handoff_full_row_bits() -> None:
        residual = layer1_handoff_cache.get("layer1_input")
        branch = layer1_handoff_cache.get("layer1_mixer_output")
        output = layer1_handoff_cache.get("layer1_output")
        if not residual or not branch or not output:
            return
        for cached in (residual, branch, output):
            if not cached.get("available", False):
                return
        width = min(int(residual["width"]), int(branch["width"]), int(output["width"]))
        rounded_bits: List[int] = []
        for idx in range(width):
            sum_fp32 = struct.unpack(
                "<f",
                struct.pack(
                    "<f",
                    float(residual["values"][idx]) + float(branch["values"][idx]),
                ),
            )[0]
            rounded_bits.append(
                int(
                    bf16_u16_values(
                        torch.tensor([sum_fp32], dtype=torch.float32).to(torch.bfloat16)
                    )[0]
                )
            )
        summaries.append(
            {
                "index": len(summaries),
                "layer": layer_idx,
                "label": "layer1_handoff_full_row_bits",
                "source": "hf_layer1_full_row_handoff_provenance",
                "dtype": str(output["dtype"]),
                "row_width": int(width),
                "residual_source": "hf_layer1_block_pre_hook",
                "branch_source": "hf_layer1_mixer_forward_hook",
                "output_source": "hf_layer1_block_forward_hook",
                "residual_hash_fnv1a_u16": residual["hash_fnv1a_u16"],
                "branch_hash_fnv1a_u16": branch["hash_fnv1a_u16"],
                "rounded_sum_hash_fnv1a_u16": fnv1a_u16(rounded_bits),
                "actual_output_hash_fnv1a_u16": output["hash_fnv1a_u16"],
                "residual_bits_u16": [int(v) & 0xFFFF for v in residual["bits"][:width]],
                "branch_bits_u16": [int(v) & 0xFFFF for v in branch["bits"][:width]],
                "rounded_sum_bits_u16": rounded_bits,
                "actual_output_bits_u16": [int(v) & 0xFFFF for v in output["bits"][:width]],
                "residual_in_fp32": bool(getattr(block, "residual_in_fp32", False)),
                "add_semantics": "hf_block_forward_residual_plus_mixer_output",
                "block_type": block_type,
            }
        )

    def norm_pre_hook(module: Any, inputs: Any) -> None:
        tensor = _first_tensor(inputs[0] if inputs else None)
        if tensor is None:
            unavailable("layer1_norm_input", "hf_layer1_norm_pre_hook", "no tensor value")
            return
        record("layer1_norm_input", "hf_layer1_norm_pre_hook", tensor)
        weight = getattr(module, "weight", None)
        if weight is None:
            unavailable("layer1_rmsnorm_weight", "hf_layer1_norm_manual", "module has no weight")
            return
        eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 0.0)))
        hidden_f32 = tensor.to(torch.float32)
        mean_square = hidden_f32.pow(2).mean(-1, keepdim=True)
        eps_tensor = torch.full_like(mean_square, eps, dtype=torch.float32)
        rsqrt = torch.rsqrt(mean_square + eps)
        scaled_input = hidden_f32 * rsqrt
        pre_store_output = weight.to(torch.float32) * scaled_input
        stored_output = pre_store_output.to(tensor.dtype)
        row_values = tensor[0, -1, :].detach().to(torch.float32).cpu().reshape(-1).tolist()
        weight_values = weight.detach().to(torch.float32).cpu().reshape(-1).tolist()

        def f32(value: float) -> float:
            return struct.unpack("<f", struct.pack("<f", float(value)))[0]

        def fnv1a_u16(values: Any) -> str:
            h = 0xCBF29CE484222325
            for raw in values:
                v = int(raw) & 0xFFFF
                h ^= v & 0xFF
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
                h ^= (v >> 8) & 0xFF
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            return f"0x{h:016x}"

        def bf16_u16_values(tensor_value: Any) -> List[int]:
            return [
                int(v) & 0xFFFF
                for v in tensor_value.detach()
                .contiguous()
                .view(torch.int16)
                .cpu()
                .reshape(-1)
                .tolist()
            ]

        seq_ss = f32(0.0)
        for value in row_values:
            v = f32(value)
            seq_ss = f32(seq_ss + f32(v * v))
        seq_mean_square_value = f32(seq_ss / f32(len(row_values)))
        seq_rsqrt_value = f32(1.0 / math.sqrt(f32(seq_mean_square_value + f32(eps))))
        seq_pre_store_values = [
            f32(f32(value) * seq_rsqrt_value * f32(w))
            for value, w in zip(row_values, weight_values)
        ]
        seq_stored_values = (
            torch.tensor(seq_pre_store_values, dtype=torch.float32)
            .to(tensor.dtype)
            .to(torch.float32)
            .tolist()
        )
        seq_mean_square = torch.tensor([[[seq_mean_square_value]]], dtype=torch.float32)
        seq_rsqrt = torch.tensor([[[seq_rsqrt_value]]], dtype=torch.float32)
        seq_pre_store_output = torch.tensor(seq_pre_store_values, dtype=torch.float32).reshape(1, 1, -1)
        seq_stored_output = torch.tensor(seq_stored_values, dtype=torch.float32).reshape(1, 1, -1).to(tensor.dtype)
        record_flat("layer1_rmsnorm_weight", "hf_layer1_norm_manual", weight)
        record("layer1_rmsnorm_mean_square", "hf_layer1_norm_manual", mean_square)
        record("layer1_rmsnorm_eps", "hf_layer1_norm_manual", eps_tensor)
        record("layer1_rmsnorm_rsqrt", "hf_layer1_norm_manual", rsqrt)
        record("layer1_rmsnorm_scaled_input", "hf_layer1_norm_manual", scaled_input)
        record("layer1_rmsnorm_pre_store_output", "hf_layer1_norm_manual", pre_store_output)
        record("layer1_rmsnorm_stored_output_candidate", "hf_layer1_norm_manual", stored_output)
        record("layer1_rmsnorm_seq_mean_square", "hf_layer1_norm_manual_seq_fp32", seq_mean_square)
        record("layer1_rmsnorm_seq_rsqrt", "hf_layer1_norm_manual_seq_fp32", seq_rsqrt)
        record("layer1_rmsnorm_seq_pre_store_output", "hf_layer1_norm_manual_seq_fp32", seq_pre_store_output)
        record("layer1_rmsnorm_seq_stored_output", "hf_layer1_norm_manual_seq_fp32", seq_stored_output)
        norm_input_row = tensor[0, -1, :].detach().contiguous()
        stored_output_row = stored_output.reshape(-1, stored_output.shape[-1])[-1].detach().contiguous()
        norm_element_cache.clear()
        norm_element_cache.update(
            {
                "width": int(norm_input_row.numel()),
                "norm_input_bits": bf16_u16_values(norm_input_row),
                "norm_input_values": norm_input_row.to(torch.float32).cpu().reshape(-1).tolist(),
                "weight_bits": bf16_u16_values(weight.detach().contiguous()),
                "weight_values": weight.detach().to(torch.float32).cpu().reshape(-1).tolist(),
                "mean_square": float(mean_square.reshape(-1)[-1].detach().cpu().item()),
                "rsqrt": float(rsqrt.reshape(-1)[-1].detach().cpu().item()),
                "pre_store_values": pre_store_output.reshape(-1, pre_store_output.shape[-1])[-1]
                .detach()
                .to(torch.float32)
                .cpu()
                .reshape(-1)
                .tolist(),
                "stored_candidate_bits": bf16_u16_values(stored_output_row),
                "stored_candidate_values": stored_output_row.to(torch.float32).cpu().reshape(-1).tolist(),
                "norm_input_hash_fnv1a_u16": fnv1a_u16(bf16_u16_values(norm_input_row)),
                "stored_candidate_hash_fnv1a_u16": fnv1a_u16(bf16_u16_values(stored_output_row)),
            }
        )

    def norm_forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
        record("layer1_norm_output", "hf_layer1_norm_forward_hook", output)
        tensor = _first_tensor(output)
        if tensor is None:
            unavailable("layer1_rmsnorm_element_details", "hf_layer1_norm_forward_hook", "no tensor value")
            return
        if not norm_element_cache:
            unavailable(
                "layer1_rmsnorm_element_details",
                "hf_layer1_norm_forward_hook",
                "missing pre-hook element cache",
            )
            return
        if tensor.dtype != torch.bfloat16:
            unavailable(
                "layer1_rmsnorm_element_details",
                "hf_layer1_norm_forward_hook",
                f"expected BF16 output, got {tensor.dtype}",
            )
            return

        def fnv1a_u16(values: Any) -> str:
            h = 0xCBF29CE484222325
            for raw in values:
                v = int(raw) & 0xFFFF
                h ^= v & 0xFF
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
                h ^= (v >> 8) & 0xFF
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            return f"0x{h:016x}"

        def bf16_u16_values(tensor_value: Any) -> List[int]:
            return [
                int(v) & 0xFFFF
                for v in tensor_value.detach()
                .contiguous()
                .view(torch.int16)
                .cpu()
                .reshape(-1)
                .tolist()
            ]

        actual_row = tensor.reshape(-1, tensor.shape[-1])[-1].detach().contiguous()
        actual_bits = bf16_u16_values(actual_row)
        actual_values = actual_row.to(torch.float32).cpu().reshape(-1).tolist()
        width = min(int(norm_element_cache.get("width", 0)), len(actual_bits))
        detail_indices = selected_element_dims or list(range(width))
        details = []
        for idx in detail_indices:
            if idx >= width:
                continue
            details.append(
                {
                    "dim_index": int(idx),
                    "norm_input_bits": int(norm_element_cache["norm_input_bits"][idx]),
                    "norm_input_value": float(norm_element_cache["norm_input_values"][idx]),
                    "weight_bits": int(norm_element_cache["weight_bits"][idx]),
                    "weight_value": float(norm_element_cache["weight_values"][idx]),
                    "mean_square": float(norm_element_cache["mean_square"]),
                    "rsqrt": float(norm_element_cache["rsqrt"]),
                    "pre_store_output": float(norm_element_cache["pre_store_values"][idx]),
                    "stored_candidate_bits": int(norm_element_cache["stored_candidate_bits"][idx]),
                    "stored_candidate_value": float(norm_element_cache["stored_candidate_values"][idx]),
                    "actual_output_bits": int(actual_bits[idx]),
                    "actual_output_value": float(actual_values[idx]),
                }
            )
        summaries.append(
            {
                "index": len(summaries),
                "layer": layer_idx,
                "label": "layer1_rmsnorm_element_details",
                "source": "hf_layer1_norm_forward_hook_exact_row",
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "row_width": int(width),
                "selected_dims": selected_element_dims,
                "detail_count": len(details),
                "norm_input_hash_fnv1a_u16": norm_element_cache["norm_input_hash_fnv1a_u16"],
                "stored_candidate_hash_fnv1a_u16": norm_element_cache[
                    "stored_candidate_hash_fnv1a_u16"
                ],
                "actual_output_hash_fnv1a_u16": fnv1a_u16(actual_bits[:width]),
                "details": details,
                "block_type": block_type,
            }
        )

    def gate_post_hook(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            record("layer1_moe_gate_output", "hf_layer1_moe_gate_forward_hook", output)
            return
        record("layer1_moe_topk_indices", "hf_layer1_moe_gate_forward_hook", output[0])
        record("layer1_moe_topk_weights", "hf_layer1_moe_gate_forward_hook", output[1])
        ids_tensor = _first_tensor(output[0])
        weights_tensor = _first_tensor(output[1])
        if ids_tensor is not None and weights_tensor is not None:
            moe_route_cache["hf_gate_actual_ids"] = (
                ids_tensor.reshape(-1, ids_tensor.shape[-1])[-1].detach().contiguous()
            )
            moe_route_cache["hf_gate_actual_scaled_weights"] = (
                weights_tensor.reshape(-1, weights_tensor.shape[-1])[-1].detach().contiguous()
            )

    def gate_pre_hook(module: Any, inputs: Any) -> None:
        tensor = _first_tensor(inputs[0] if inputs else None)
        if tensor is None:
            unavailable("layer1_moe_router_logits", "hf_layer1_moe_gate_manual_pre_hook", "no tensor value")
            return
        weight = getattr(module, "weight", None)
        if weight is None:
            unavailable("layer1_moe_router_logits", "hf_layer1_moe_gate_manual_pre_hook", "module has no weight")
            return
        record("layer1_moe_router_input", "hf_layer1_moe_gate_manual_pre_hook", tensor)
        record_flat("layer1_moe_router_weight_flat", "hf_layer1_moe_gate_manual_pre_hook", weight)
        hidden = tensor.reshape(-1, tensor.shape[-1]).to(torch.float32)
        weight_f32 = weight.to(torch.float32)
        router_logits = torch.nn.functional.linear(hidden, weight_f32)
        record("layer1_moe_router_logits", "hf_layer1_moe_gate_manual_pre_hook", router_logits)
        scores = router_logits.sigmoid()
        record("layer1_moe_router_scores", "hf_layer1_moe_gate_manual_pre_hook", scores)
        correction = getattr(module, "e_score_correction_bias", None)
        if correction is not None:
            record(
                "layer1_moe_e_score_correction_bias",
                "hf_layer1_moe_gate_manual_pre_hook",
                correction,
            )
            correction_for_choice = correction.detach().to(device=scores.device).reshape(1, -1)
            scores_for_choice = scores.view(-1, scores.shape[-1]) + correction_for_choice
            record(
                "layer1_moe_router_scores_for_choice",
                "hf_layer1_moe_gate_manual_pre_hook",
                scores_for_choice,
            )
            n_group = int(getattr(module, "n_group", 1) or 1)
            topk_group = int(getattr(module, "topk_group", 1) or 1)
            n_routed = int(getattr(module, "n_routed_experts", scores_for_choice.shape[-1]))
            top_k = int(getattr(module, "top_k", scores_for_choice.shape[-1]))
            if n_group > 0 and n_routed > 0 and n_routed % n_group == 0:
                grouped_scores = scores_for_choice.view(
                    -1, n_group, n_routed // n_group
                )
                group_scores = grouped_scores.topk(2, dim=-1)[0].sum(dim=-1)
                record(
                    "layer1_moe_router_group_scores",
                    "hf_layer1_moe_gate_manual_pre_hook",
                    group_scores,
                )
                group_idx = torch.topk(
                    group_scores,
                    k=min(topk_group, group_scores.shape[-1]),
                    dim=-1,
                    sorted=False,
                )[1]
                record(
                    "layer1_moe_router_group_idx",
                    "hf_layer1_moe_gate_manual_pre_hook",
                    group_idx,
                )
                group_mask = torch.zeros_like(group_scores)
                group_mask.scatter_(1, group_idx, 1)
                score_mask = (
                    group_mask.unsqueeze(-1)
                    .expand(-1, n_group, n_routed // n_group)
                    .reshape(-1, n_routed)
                )
                masked_scores_for_choice = scores_for_choice.masked_fill(
                    ~score_mask.bool(), 0.0
                )
                record(
                    "layer1_moe_router_scores_for_choice_masked",
                    "hf_layer1_moe_gate_manual_pre_hook",
                    masked_scores_for_choice,
                )
                choice_topk_indices = torch.topk(
                    masked_scores_for_choice,
                    k=min(top_k, masked_scores_for_choice.shape[-1]),
                    dim=-1,
                    sorted=False,
                )[1]
                choice_topk_scores = masked_scores_for_choice.gather(1, choice_topk_indices)
                record(
                    "layer1_moe_router_choice_topk_indices",
                    "hf_layer1_moe_gate_manual_pre_hook",
                    choice_topk_indices,
                )
                record(
                    "layer1_moe_router_choice_topk_scores",
                    "hf_layer1_moe_gate_manual_pre_hook",
                    choice_topk_scores,
                )

                def rank_map(values: List[float]) -> Dict[int, int]:
                    order = sorted(
                        range(len(values)),
                        key=lambda idx: (-float(values[idx]), idx),
                    )
                    return {int(idx): int(rank) for rank, idx in enumerate(order)}

                last_logits = router_logits.reshape(-1, router_logits.shape[-1])[-1].detach().cpu().reshape(-1)
                last_scores = scores.reshape(-1, scores.shape[-1])[-1].detach().cpu().reshape(-1)
                last_corr = correction_for_choice.reshape(-1).detach().cpu().reshape(-1)
                last_choice = scores_for_choice.reshape(-1, scores_for_choice.shape[-1])[-1].detach().cpu().reshape(-1)
                last_masked_choice = masked_scores_for_choice.reshape(
                    -1, masked_scores_for_choice.shape[-1]
                )[-1].detach().cpu().reshape(-1)
                actual_choice_ids = choice_topk_indices.reshape(-1, choice_topk_indices.shape[-1])[-1].detach().cpu().reshape(-1)
                actual_choice_set = {int(v) for v in actual_choice_ids.tolist()}
                raw_ranks = rank_map([float(v) for v in last_scores.tolist()])
                choice_ranks = rank_map([float(v) for v in last_choice.tolist()])
                masked_choice_ranks = rank_map([float(v) for v in last_masked_choice.tolist()])
                candidate_boundary_ids = (
                    set([28, 473, 195, 202, 250])
                    | set(int(v) for v in actual_choice_ids.tolist())
                    | set(
                        int(idx)
                        for idx in torch.topk(last_scores, k=min(28, last_scores.numel())).indices.tolist()
                    )
                    | set(
                        int(idx)
                        for idx in torch.topk(last_choice, k=min(28, last_choice.numel())).indices.tolist()
                    )
                )
                router_width = int(last_logits.numel())
                boundary_ids = sorted(
                    int(expert_id)
                    for expert_id in candidate_boundary_ids
                    if 0 <= int(expert_id) < router_width
                )
                boundary_details = []
                for expert_id in boundary_ids:
                    boundary_details.append(
                        {
                            "expert_id": int(expert_id),
                            "logit": float(last_logits[expert_id].item()),
                            "sigmoid_score": float(last_scores[expert_id].item()),
                            "e_score_correction_bias": float(last_corr[expert_id].item()),
                            "score_for_choice": float(last_choice[expert_id].item()),
                            "masked_score_for_choice": float(last_masked_choice[expert_id].item()),
                            "raw_sigmoid_rank": int(raw_ranks.get(expert_id, -1)),
                            "score_for_choice_rank": int(choice_ranks.get(expert_id, -1)),
                            "masked_score_for_choice_rank": int(masked_choice_ranks.get(expert_id, -1)),
                            "selected_by_choice_topk": bool(expert_id in actual_choice_set),
                        }
                    )
                summaries.append(
                    {
                        "index": len(summaries),
                        "layer": layer_idx,
                        "label": "layer1_moe_router_selection_semantics",
                        "source": "hf_layer1_moe_gate_manual_pre_hook",
                        "dtype": "mixed",
                        "device": str(scores.device),
                        "row_width": int(n_routed),
                        "top_k": int(top_k),
                        "n_group": int(n_group),
                        "topk_group": int(topk_group),
                        "norm_topk_prob": bool(getattr(module, "norm_topk_prob", False)),
                        "routed_scaling_factor": float(getattr(module, "routed_scaling_factor", 1.0)),
                        "e_score_correction_bias_dtype": str(correction.dtype),
                        "e_score_correction_bias_device": str(correction.device),
                        "choice_topk_indices_unsorted": [
                            int(v) for v in actual_choice_ids.tolist()
                        ],
                        "boundary_details": boundary_details,
                        "block_type": block_type,
                    }
                )
        if hasattr(module, "get_topk_indices"):
            topk_indices = module.get_topk_indices(scores)
            selected_scores = scores.gather(1, topk_indices)
            record("layer1_moe_topk_sigmoid_scores", "hf_layer1_moe_gate_manual_pre_hook", selected_scores)
            if bool(getattr(module, "norm_topk_prob", False)):
                denominator = selected_scores.sum(dim=-1, keepdim=True) + 1e-20
                normalized_scores = selected_scores / denominator
            else:
                normalized_scores = selected_scores
            record("layer1_moe_topk_weights_normalized", "hf_layer1_moe_gate_manual_pre_hook", normalized_scores)
            routed_scale = float(getattr(module, "routed_scaling_factor", 1.0))
            scaled_scores = normalized_scores * routed_scale
            record(
                "layer1_moe_topk_weights_scaled_candidate",
                "hf_layer1_moe_gate_manual_pre_hook",
                scaled_scores,
            )
            moe_route_cache["hf_corrected_actual_ids"] = (
                topk_indices.reshape(-1, topk_indices.shape[-1])[-1].detach().contiguous()
            )
            moe_route_cache["hf_corrected_actual_normalized_weights"] = (
                normalized_scores.reshape(-1, normalized_scores.shape[-1])[-1].detach().contiguous()
            )
            moe_route_cache["hf_corrected_actual_scaled_weights"] = (
                scaled_scores.reshape(-1, scaled_scores.shape[-1])[-1].detach().contiguous()
            )
            top_k_for_raw = int(getattr(module, "top_k", topk_indices.shape[-1]) or topk_indices.shape[-1])
            scores_flat = scores.reshape(-1, scores.shape[-1])
            raw_topk_indices = torch.topk(
                scores_flat,
                k=min(top_k_for_raw, scores_flat.shape[-1]),
                dim=-1,
                sorted=False,
            )[1]
            raw_selected_scores = scores_flat.gather(1, raw_topk_indices)
            if bool(getattr(module, "norm_topk_prob", False)):
                raw_normalized_scores = raw_selected_scores / (
                    raw_selected_scores.sum(dim=-1, keepdim=True) + 1e-20
                )
            else:
                raw_normalized_scores = raw_selected_scores
            raw_scaled_scores = raw_normalized_scores * routed_scale
            moe_route_cache["raw_sigmoid_global_ids"] = (
                raw_topk_indices.reshape(-1, raw_topk_indices.shape[-1])[-1].detach().contiguous()
            )
            moe_route_cache["raw_sigmoid_global_normalized_weights"] = (
                raw_normalized_scores.reshape(-1, raw_normalized_scores.shape[-1])[-1].detach().contiguous()
            )
            moe_route_cache["raw_sigmoid_global_scaled_weights"] = (
                raw_scaled_scores.reshape(-1, raw_scaled_scores.shape[-1])[-1].detach().contiguous()
            )

        def f32(value: float) -> float:
            return struct.unpack("<f", struct.pack("<f", float(value)))[0]

        def fnv1a_u16(values: Any) -> str:
            h = 0xCBF29CE484222325
            for raw in values:
                v = int(raw) & 0xFFFF
                h ^= v & 0xFF
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
                h ^= (v >> 8) & 0xFF
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            return f"0x{h:016x}"

        def bf16_u16_values(tensor_value: Any) -> List[int]:
            return [
                int(v) & 0xFFFF
                for v in tensor_value.detach()
                .contiguous()
                .view(torch.int16)
                .cpu()
                .reshape(-1)
                .tolist()
            ]

        last_hidden = hidden[-1].detach().cpu().reshape(-1).tolist()
        weight_rows = weight_f32.detach().cpu()
        seq_logits = []
        for row in weight_rows:
            acc = f32(0.0)
            for hidden_value, weight_value in zip(last_hidden, row.reshape(-1).tolist()):
                acc = f32(acc + f32(f32(hidden_value) * f32(weight_value)))
            seq_logits.append(acc)
        record(
            "layer1_moe_router_logits_seq_fp32",
            "hf_layer1_moe_gate_manual_pre_hook_seq_fp32",
            torch.tensor(seq_logits, dtype=torch.float32).reshape(1, -1),
        )
        if tensor.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
            hidden_bf16_row = tensor.reshape(-1, tensor.shape[-1])[-1].detach().contiguous()
            weight_bf16_rows = weight.detach().contiguous()
            hidden_bits = bf16_u16_values(hidden_bf16_row)
            hidden_f32_values = hidden_bf16_row.to(torch.float32).cpu().reshape(-1).tolist()
            weight_bits_rows = weight_bf16_rows.view(torch.int16).cpu().reshape(weight.shape[0], weight.shape[1])
            production_last = router_logits.reshape(-1, router_logits.shape[-1])[-1].detach().cpu().reshape(-1).tolist()
            details = []
            for expert_id, row in enumerate(weight_rows):
                row_bits = [int(v) & 0xFFFF for v in weight_bits_rows[expert_id].tolist()]
                weight_values = row.reshape(-1).tolist()
                max_idx = 0
                max_abs = -1.0
                for idx, (hidden_value, weight_value) in enumerate(zip(hidden_f32_values, weight_values)):
                    abs_prod = abs(float(hidden_value) * float(weight_value))
                    if abs_prod > max_abs:
                        max_abs = abs_prod
                        max_idx = idx
                details.append(
                    {
                        "expert_id": int(expert_id),
                        "width": int(len(hidden_bits)),
                        "router_input_hash_fnv1a_u16": fnv1a_u16(hidden_bits),
                        "router_weight_row_hash_fnv1a_u16": fnv1a_u16(row_bits),
                        "manual_seq_fp32_logit": float(seq_logits[expert_id]),
                        "production_logit": float(production_last[expert_id]),
                        "manual_minus_production": float(f32(seq_logits[expert_id] - production_last[expert_id])),
                        "max_abs_contrib_index": int(max_idx),
                        "max_abs_contrib_hidden": float(hidden_f32_values[max_idx]),
                        "max_abs_contrib_hidden_u16": int(hidden_bits[max_idx]),
                        "max_abs_contrib_weight": float(weight_values[max_idx]),
                        "max_abs_contrib_weight_u16": int(row_bits[max_idx]),
                    }
                )
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": layer_idx,
                    "label": "layer1_moe_router_logit_details",
                    "source": "hf_layer1_moe_gate_manual_pre_hook_exact_rows",
                    "dtype": "mixed",
                    "device": str(tensor.device),
                    "row_width": int(len(details)),
                    "router_input_hash_fnv1a_u16": fnv1a_u16(hidden_bits),
                    "router_input_sample_indices": [
                        int(idx)
                        for idx in torch.topk(hidden_bf16_row.to(torch.float32).abs(), k=min(16, hidden_bf16_row.numel())).indices.cpu().tolist()
                    ],
                    "details": details,
                    "block_type": block_type,
                }
            )
        else:
            summaries.append(
                {
                    "index": len(summaries),
                    "layer": layer_idx,
                    "label": "layer1_moe_router_logit_details",
                    "source": "hf_layer1_moe_gate_manual_pre_hook_exact_rows",
                    "available": False,
                    "reason": f"expected BF16 tensor/weight, got tensor={tensor.dtype} weight={weight.dtype}",
                    "block_type": block_type,
                }
            )

    original_moe_fn = getattr(mixer, "moe", None) if mixer is not None else None
    if callable(original_moe_fn):
        def wrapped_moe(hidden_states: Any, topk_indices: Any, topk_weights: Any) -> Any:
            output = original_moe_fn(hidden_states, topk_indices, topk_weights)
            record(
                "layer1_moe_routed_output_pre_shared",
                "hf_layer1_moe_method_wrapped_output",
                output,
            )
            record(
                "layer1_moe_shared_add_input_routed",
                "hf_layer1_moe_method_wrapped_output",
                output,
            )
            return output

        setattr(mixer, "moe", wrapped_moe)

    handles.append(block.register_forward_pre_hook(pre_hook("layer1_input", "hf_layer1_block_pre_hook")))
    if hasattr(block, "norm"):
        handles.append(block.norm.register_forward_pre_hook(norm_pre_hook))
        handles.append(block.norm.register_forward_hook(norm_forward_hook))
    if mixer is not None:
        handles.append(mixer.register_forward_pre_hook(pre_hook("layer1_mixer_input", "hf_layer1_mixer_pre_hook")))
        handles.append(mixer.register_forward_hook(post_hook("layer1_mixer_output", "hf_layer1_mixer_forward_hook")))
        if hasattr(mixer, "gate"):
            handles.append(mixer.gate.register_forward_pre_hook(gate_pre_hook))
            handles.append(mixer.gate.register_forward_hook(gate_post_hook))
        if hasattr(mixer, "fc1_latent_proj"):
            handles.append(
                mixer.fc1_latent_proj.register_forward_hook(
                    latent_input_post_hook
                )
            )
        if hasattr(mixer, "fc2_latent_proj"):
            handles.append(
                mixer.fc2_latent_proj.register_forward_pre_hook(
                    pre_hook("layer1_moe_routed_latent_output", "hf_layer1_moe_fc2_latent_pre_hook")
                )
            )
            handles.append(
                mixer.fc2_latent_proj.register_forward_hook(
                    post_hook("layer1_moe_routed_output", "hf_layer1_moe_fc2_latent_forward_hook")
                )
            )
        if hasattr(mixer, "shared_experts"):
            shared_experts = getattr(mixer, "shared_experts")
            handles.append(shared_experts.register_forward_pre_hook(shared_pre_hook))
            handles.append(
                shared_experts.register_forward_hook(shared_post_hook)
            )
            if hasattr(shared_experts, "up_proj"):
                handles.append(shared_experts.up_proj.register_forward_pre_hook(shared_up_pre_hook))
                handles.append(shared_experts.up_proj.register_forward_hook(shared_up_post_hook))
            if hasattr(shared_experts, "down_proj"):
                handles.append(shared_experts.down_proj.register_forward_pre_hook(shared_down_pre_hook))
                handles.append(shared_experts.down_proj.register_forward_hook(shared_down_post_hook))
    handles.append(block.register_forward_hook(post_hook("layer1_output", "hf_layer1_block_forward_hook")))

    try:
        with torch.no_grad():
            model(prompt_input_ids, **forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        if callable(original_moe_fn) and mixer is not None:
            setattr(mixer, "moe", original_moe_fn)

    if block_type != "moe":
        unavailable(
            "layer1_moe_router",
            "hf_layer1_metadata",
            f"layer1 block_type is {block_type}; expected moe for this diagnostic",
        )
    return summaries


def _cache_state_snapshot(model: Any, config: Any) -> Dict[str, Any]:
    generation_config = getattr(model, "generation_config", None)
    snapshot: Dict[str, Any] = {
        "config_use_cache": getattr(config, "use_cache", None),
        "generation_config_use_cache": getattr(generation_config, "use_cache", None),
        "model_training": bool(getattr(model, "training", False)),
        "model_eval": not bool(getattr(model, "training", False)),
        "known_cache_attrs": {},
    }
    for attr_name in (
        "past_key_values",
        "_past_key_values",
        "cache_params",
        "_cache",
        "cache",
        "_seen_tokens",
        "seen_tokens",
    ):
        if not hasattr(model, attr_name):
            continue
        try:
            value = getattr(model, attr_name)
        except Exception as exc:
            snapshot["known_cache_attrs"][attr_name] = {"present": True, "read_error": repr(exc)}
            continue
        entry: Dict[str, Any] = {
            "present": True,
            "is_none": value is None,
            "type": type(value).__name__,
        }
        try:
            entry["past_seen_tokens"] = _past_seen_tokens(value)
        except Exception:
            pass
        snapshot["known_cache_attrs"][attr_name] = entry
    return snapshot


def _cache_mode_to_value(cache_mode: str) -> Optional[bool]:
    if cache_mode not in ("auto", "on", "off"):
        die(f"Unknown cache mode: {cache_mode}")
    if cache_mode == "auto":
        return None
    return cache_mode == "on"


def _set_cache_mode(model: Any, config: Any, use_cache_mode: str) -> Dict[str, Any]:
    if use_cache_mode not in ("auto", "on", "off"):
        die(f"Unknown use-cache mode: {use_cache_mode}")
    generation_config = getattr(model, "generation_config", None)
    before = _cache_state_snapshot(model, config)
    if use_cache_mode == "auto":
        return {"mode": use_cache_mode, "before": before, "after": before}

    enabled = use_cache_mode == "on"
    if hasattr(config, "use_cache"):
        setattr(config, "use_cache", enabled)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        setattr(generation_config, "use_cache", enabled)
    after = _cache_state_snapshot(model, config)
    return {"mode": use_cache_mode, "before": before, "after": after}


def _cleanup_between_turns(model: Any, config: Any, mode: str) -> Dict[str, Any]:
    if mode not in ("none", "gc", "reset"):
        die(f"Unknown cleanup mode: {mode}")
    before = _cache_state_snapshot(model, config)
    actions: List[str] = []
    if mode in ("gc", "reset"):
        import gc
        import torch

        gc.collect()
        actions.append("gc.collect")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            actions.extend(["torch.cuda.empty_cache", "torch.cuda.synchronize"])
    if mode == "reset":
        model.eval()
        actions.append("model.eval")
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            try:
                generation_config.validate()
                actions.append("generation_config.validate")
            except Exception as exc:
                actions.append(f"generation_config.validate_error:{exc!r}")
        for attr_name in ("past_key_values", "_past_key_values", "cache_params", "_cache", "cache"):
            if hasattr(model, attr_name):
                try:
                    setattr(model, attr_name, None)
                    actions.append(f"clear_model_attr:{attr_name}")
                except Exception as exc:
                    actions.append(f"clear_model_attr_error:{attr_name}:{exc!r}")
    after = _cache_state_snapshot(model, config)
    return {"mode": mode, "actions": actions, "before": before, "after": after}


def summarize_reference_turns(reference: Dict[str, Any], expected_top_k: int) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for conv_idx, conv in enumerate(reference.get("conversations", [])):
        for turn_idx, turn in enumerate(conv.get("turns", [])):
            input_ids = turn.get("input_token_ids") if isinstance(turn.get("input_token_ids"), list) else []
            token_ids = turn.get("token_ids") if isinstance(turn.get("token_ids"), list) else []
            per_token = turn.get("per_token_data") if isinstance(turn.get("per_token_data"), list) else []
            first_diag = per_token[0] if per_token and isinstance(per_token[0], dict) else None
            top_k = first_diag.get("top_k") if first_diag and isinstance(first_diag.get("top_k"), list) else []
            top_k_log_probs = [
                entry.get("log_prob")
                for entry in top_k
                if isinstance(entry, dict)
            ]
            prompt = turn.get("prompt") if isinstance(turn.get("prompt"), str) else ""
            source_conv = turn.get("source_conversation_index", conv_idx)
            source_turn = turn.get("source_turn_index", turn_idx)
            diag_token_id = first_diag.get("token_id") if first_diag else None
            generated_first_token = token_ids[0] if token_ids else None
            top_k0 = top_k[0] if top_k and isinstance(top_k[0], dict) else None
            logits_stats = first_diag.get("logits_stats") if first_diag and isinstance(first_diag.get("logits_stats"), dict) else {}
            log_probs_stats = first_diag.get("log_probs_stats") if first_diag and isinstance(first_diag.get("log_probs_stats"), dict) else {}
            generation_state = turn.get("generation_state") if isinstance(turn.get("generation_state"), dict) else {}
            pre_generate = turn.get("pre_generate_diagnostic") if isinstance(turn.get("pre_generate_diagnostic"), list) else []
            post_generate = turn.get("post_generate_diagnostic") if isinstance(turn.get("post_generate_diagnostic"), list) else []
            postponed_prefill = (
                turn.get("postponed_prefill_diagnostic")
                if isinstance(turn.get("postponed_prefill_diagnostic"), list)
                else []
            )
            pre_first = pre_generate[0] if pre_generate and isinstance(pre_generate[0], dict) else None
            post_first = post_generate[0] if post_generate and isinstance(post_generate[0], dict) else None
            postponed_first = (
                postponed_prefill[0]
                if postponed_prefill and isinstance(postponed_prefill[0], dict)
                else None
            )
            summaries.append(
                {
                    "conversation_index": conv_idx,
                    "turn_index": turn_idx,
                    "source_conversation_index": source_conv,
                    "source_turn_index": source_turn,
                    "label": f"conv{source_conv}_t{source_turn}",
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_preview": prompt[:120].replace("\n", " "),
                    "input_token_count": len(input_ids),
                    "input_token_sha256": _sha256_json(input_ids),
                    "generated_token_count": len(token_ids),
                    "generated_first_token_id": generated_first_token,
                    "generated_first_token_text": turn.get("text", "")[:40] if isinstance(turn.get("text"), str) else "",
                    "diagnostic_present": first_diag is not None,
                    "diagnostic_token_id": diag_token_id,
                    "diagnostic_token_matches_generated_first": (
                        diag_token_id == generated_first_token
                        if first_diag is not None and generated_first_token is not None
                        else False
                    ),
                    "diagnostic_logit_finite": _finite_float_or_none(first_diag.get("logit") if first_diag else None) is not None,
                    "diagnostic_log_prob_finite": _finite_float_or_none(first_diag.get("log_prob") if first_diag else None) is not None,
                    "diagnostic_logit": _finite_float_or_none(first_diag.get("logit") if first_diag else None),
                    "diagnostic_log_prob": _finite_float_or_none(first_diag.get("log_prob") if first_diag else None),
                    "logits_dtype": logits_stats.get("dtype"),
                    "logits_device": logits_stats.get("device"),
                    "logits_finite_count": logits_stats.get("finite_count"),
                    "logits_numel": logits_stats.get("numel"),
                    "logits_nan_count": logits_stats.get("nan_count"),
                    "logits_inf_count": logits_stats.get("inf_count"),
                    "log_probs_finite_count": log_probs_stats.get("finite_count"),
                    "log_probs_numel": log_probs_stats.get("numel"),
                    "log_probs_nan_count": log_probs_stats.get("nan_count"),
                    "log_probs_inf_count": log_probs_stats.get("inf_count"),
                    "top_k_length": len(top_k),
                    "top_k_coverage_ok": len(top_k) == expected_top_k,
                    "top_k_log_probs_all_finite": all(
                        _finite_float_or_none(value) is not None for value in top_k_log_probs
                    ) if top_k else False,
                    "top_k0_token_id": top_k0.get("token_id") if top_k0 else None,
                    "top_k0_log_prob_finite": (
                        _finite_float_or_none(top_k0.get("log_prob")) is not None
                        if top_k0
                        else False
                    ),
                    "top_k0_log_prob": _finite_float_or_none(top_k0.get("log_prob") if top_k0 else None),
                    "model_training": generation_state.get("model_training"),
                    "model_eval": generation_state.get("model_eval"),
                    "attn_implementation": generation_state.get("attn_implementation"),
                    "use_cache_mode": generation_state.get("use_cache_mode"),
                    "generate_use_cache_mode": generation_state.get("generate_use_cache_mode"),
                    "diagnostic_use_cache_mode": generation_state.get("diagnostic_use_cache_mode"),
                    "diagnostic_scope": generation_state.get("diagnostic_scope"),
                    "use_cache_generate_arg": generation_state.get("use_cache_generate_arg"),
                    "use_cache_forward_arg": generation_state.get("use_cache_forward_arg"),
                    "config_use_cache": generation_state.get("config_use_cache"),
                    "generation_config_use_cache": generation_state.get("generation_config_use_cache"),
                    "cache_snapshot_before_turn": generation_state.get("cache_snapshot_before_turn"),
                    "cache_snapshot_before_generate": generation_state.get("cache_snapshot_before_generate"),
                    "cache_snapshot_after_generate": generation_state.get("cache_snapshot_after_generate"),
                    "cache_snapshot_after_diagnostics": generation_state.get("cache_snapshot_after_diagnostics"),
                    "between_turn_cleanup": generation_state.get("between_turn_cleanup"),
                    "reloaded_model_before_turn": generation_state.get("reloaded_model_before_turn"),
                    "explicit_past_key_values_supplied": generation_state.get("explicit_past_key_values_supplied"),
                    "explicit_cache_object_supplied": generation_state.get("explicit_cache_object_supplied"),
                    "past_key_values_reused_from_prior_turn": generation_state.get("past_key_values_reused_from_prior_turn"),
                    "diagnostics_captured": generation_state.get("diagnostics_captured"),
                    "diagnostics_postponed": generation_state.get("diagnostics_postponed"),
                    "diagnostics_skipped_reason": generation_state.get("diagnostics_skipped_reason"),
                    "pre_generate_diagnostic_present": pre_first is not None,
                    "pre_generate_logits_finite_count": (
                        pre_first.get("logits_stats", {}).get("finite_count") if pre_first else None
                    ),
                    "pre_generate_logits_numel": (
                        pre_first.get("logits_stats", {}).get("numel") if pre_first else None
                    ),
                    "pre_generate_logits_nan_count": (
                        pre_first.get("logits_stats", {}).get("nan_count") if pre_first else None
                    ),
                    "pre_generate_top_k_log_probs_all_finite": all(
                        _finite_float_or_none(entry.get("log_prob")) is not None
                        for entry in (pre_first.get("top_k", []) if pre_first else [])
                        if isinstance(entry, dict)
                    ) if pre_first else None,
                    "post_generate_diagnostic_present": post_first is not None,
                    "post_generate_logits_finite_count": (
                        post_first.get("logits_stats", {}).get("finite_count") if post_first else None
                    ),
                    "post_generate_logits_numel": (
                        post_first.get("logits_stats", {}).get("numel") if post_first else None
                    ),
                    "post_generate_logits_nan_count": (
                        post_first.get("logits_stats", {}).get("nan_count") if post_first else None
                    ),
                    "post_generate_top_k_log_probs_all_finite": all(
                        _finite_float_or_none(entry.get("log_prob")) is not None
                        for entry in (post_first.get("top_k", []) if post_first else [])
                        if isinstance(entry, dict)
                    ) if post_first else None,
                    "postponed_prefill_diagnostic_present": postponed_first is not None,
                    "postponed_prefill_logits_finite_count": (
                        postponed_first.get("logits_stats", {}).get("finite_count") if postponed_first else None
                    ),
                    "postponed_prefill_logits_numel": (
                        postponed_first.get("logits_stats", {}).get("numel") if postponed_first else None
                    ),
                    "postponed_prefill_logits_nan_count": (
                        postponed_first.get("logits_stats", {}).get("nan_count") if postponed_first else None
                    ),
                    "postponed_prefill_top_k_log_probs_all_finite": all(
                        _finite_float_or_none(entry.get("log_prob")) is not None
                        for entry in (postponed_first.get("top_k", []) if postponed_first else [])
                        if isinstance(entry, dict)
                    ) if postponed_first else None,
                }
            )
    return summaries


def build_token_audit_turns(reference: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for conv_idx, conv in enumerate(reference.get("conversations", [])):
        for turn_idx, turn in enumerate(conv.get("turns", [])):
            input_ids = turn.get("input_token_ids") if isinstance(turn.get("input_token_ids"), list) else []
            token_ids = turn.get("token_ids") if isinstance(turn.get("token_ids"), list) else []
            prompt = turn.get("prompt") if isinstance(turn.get("prompt"), str) else ""
            source_conv = turn.get("source_conversation_index", conv_idx)
            source_turn = turn.get("source_turn_index", turn_idx)
            turns.append(
                {
                    "conversation_index": conv_idx,
                    "turn_index": turn_idx,
                    "source_conversation_index": source_conv,
                    "source_turn_index": source_turn,
                    "label": f"conv{source_conv}_t{source_turn}",
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "input_token_ids": input_ids,
                    "input_token_count": len(input_ids),
                    "input_token_sha256": _sha256_json(input_ids),
                    "generated_token_ids": token_ids,
                    "generated_first_token_id": token_ids[0] if token_ids else None,
                    "generated_text": turn.get("text") if isinstance(turn.get("text"), str) else "",
                    "generation_state": turn.get("generation_state") if isinstance(turn.get("generation_state"), dict) else {},
                }
            )
    return turns


def parse_prompt_conversations(lines: List[str]) -> List[List[str]]:
    """Parse prompt lines into conversations.

    Lines starting with '- ' are continuations of the previous conversation.
    All other non-empty lines start a new conversation.
    """
    conversations: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            prompt_text = stripped[2:].strip()
            if not prompt_text:
                continue
            if conversations:
                conversations[-1].append(prompt_text)
            else:
                conversations.append([prompt_text])
        else:
            conversations.append([stripped])
    return conversations


def load_raw_input_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        root = json.load(f)

    if isinstance(root, dict) and "input_token_ids" in root:
        raw_cases = [root]
    elif isinstance(root, dict) and isinstance(root.get("cases"), list):
        raw_cases = root["cases"]
    elif isinstance(root, list):
        raw_cases = root
    else:
        die(
            "--raw-input-json must contain an input_token_ids array, a cases array, "
            "or be a list of case objects"
        )

    cases: List[Dict[str, Any]] = []
    for idx, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            die(f"--raw-input-json case {idx} is not an object")
        tokens = case.get("input_token_ids")
        if not isinstance(tokens, list) or not tokens:
            die(f"--raw-input-json case {idx} missing non-empty input_token_ids")
        try:
            input_token_ids = [int(token) for token in tokens]
        except Exception:
            die(f"--raw-input-json case {idx} input_token_ids must be integers")
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            prompt = f"raw_input_json:{path.name}:case{idx}"
        cases.append(
            {
                "prompt": prompt,
                "input_token_ids": input_token_ids,
                "source_case_index": idx,
                "input_token_count": len(input_token_ids),
                "input_token_hash_fnv1a64": _fnv1a64_token_hash(input_token_ids),
                "source_metadata": {
                    key: value
                    for key, value in case.items()
                    if key not in {"input_token_ids", "prompt"}
                },
            }
        )
    return cases


def _build_prompt_subset(
    conversations: List[List[str]],
    conversation_index: Optional[int],
    turn_index: Optional[int],
    turn_start_index: Optional[int],
    turn_end_index: Optional[int],
    first_stored_turns: Optional[int],
) -> Tuple[List[List[str]], Dict[str, Any]]:
    if first_stored_turns is not None and (
        conversation_index is not None
        or turn_index is not None
        or turn_start_index is not None
        or turn_end_index is not None
    ):
        die("--first-stored-turns cannot be combined with conversation/turn subset selectors")
    if first_stored_turns is not None:
        if first_stored_turns <= 0:
            die("--first-stored-turns must be > 0")
        selected: List[List[str]] = []
        source_indices: List[int] = []
        remaining = first_stored_turns
        total_turns = sum(len(conv) for conv in conversations)
        if first_stored_turns > total_turns:
            die(f"--first-stored-turns out of range: {first_stored_turns} (total={total_turns})")
        for source_conv_idx, conversation in enumerate(conversations):
            if remaining <= 0:
                break
            take = min(remaining, len(conversation))
            selected.append(conversation[:take])
            source_indices.append(source_conv_idx)
            remaining -= take
        return selected, {
            "enabled": True,
            "conversation_index": None,
            "turn_index": None,
            "turn_start_index": None,
            "turn_end_index": None,
            "first_stored_turns": first_stored_turns,
            "mode": "first_stored_turns",
            "source_conversation_indices": source_indices,
            "recorded_turns": first_stored_turns,
        }
    if turn_index is not None and (turn_start_index is not None or turn_end_index is not None):
        die("--turn-index cannot be combined with --turn-start-index or --turn-end-index")
    if turn_index is not None and conversation_index is None:
        die("--turn-index requires --conversation-index")
    if (turn_start_index is not None or turn_end_index is not None) and conversation_index is None:
        die("--turn-start-index/--turn-end-index require --conversation-index")
    if conversation_index is None:
        return conversations, {
            "enabled": False,
            "conversation_index": None,
            "turn_index": None,
            "turn_start_index": None,
            "turn_end_index": None,
            "first_stored_turns": None,
            "mode": "all",
        }
    if conversation_index < 0 or conversation_index >= len(conversations):
        die(f"--conversation-index out of range: {conversation_index} (count={len(conversations)})")

    selected_conversation = conversations[conversation_index]
    if turn_start_index is not None or turn_end_index is not None:
        start_idx = 0 if turn_start_index is None else turn_start_index
        end_idx = len(selected_conversation) - 1 if turn_end_index is None else turn_end_index
        if start_idx < 0 or start_idx >= len(selected_conversation):
            die(
                f"--turn-start-index out of range: {start_idx} "
                f"(conversation_index={conversation_index} turns={len(selected_conversation)})"
            )
        if end_idx < 0 or end_idx >= len(selected_conversation):
            die(
                f"--turn-end-index out of range: {end_idx} "
                f"(conversation_index={conversation_index} turns={len(selected_conversation)})"
            )
        if start_idx > end_idx:
            die(f"--turn-start-index must be <= --turn-end-index ({start_idx} > {end_idx})")
        return [selected_conversation[: end_idx + 1]], {
            "enabled": True,
            "conversation_index": conversation_index,
            "turn_index": None,
            "turn_start_index": start_idx,
            "turn_end_index": end_idx,
            "first_stored_turns": None,
            "mode": "turn_range_with_prior_history",
            "source_conversation_turns": len(selected_conversation),
            "recorded_turns": end_idx - start_idx + 1,
        }
    if turn_index is None:
        return [selected_conversation], {
            "enabled": True,
            "conversation_index": conversation_index,
            "turn_index": None,
            "turn_start_index": None,
            "turn_end_index": None,
            "first_stored_turns": None,
            "mode": "conversation",
            "source_conversation_turns": len(selected_conversation),
        }
    if turn_index < 0 or turn_index >= len(selected_conversation):
        die(
            f"--turn-index out of range: {turn_index} "
            f"(conversation_index={conversation_index} turns={len(selected_conversation)})"
        )
    return [selected_conversation[: turn_index + 1]], {
        "enabled": True,
        "conversation_index": conversation_index,
        "turn_index": turn_index,
        "turn_start_index": turn_index,
        "turn_end_index": turn_index,
        "first_stored_turns": None,
        "mode": "single_turn_with_prior_history",
        "source_conversation_turns": len(selected_conversation),
        "recorded_turns": 1,
    }


def _should_record_turn(prompt_subset: Dict[str, Any], turn_idx: int) -> bool:
    if not prompt_subset.get("enabled"):
        return True
    if prompt_subset.get("turn_index") is not None:
        return turn_idx == prompt_subset.get("turn_index")
    start_idx = prompt_subset.get("turn_start_index")
    end_idx = prompt_subset.get("turn_end_index")
    if start_idx is not None or end_idx is not None:
        start = int(start_idx or 0)
        end = int(end_idx if end_idx is not None else turn_idx)
        return start <= turn_idx <= end
    return True


def _is_final_recorded_turn(prompt_subset: Dict[str, Any], turn_idx: int) -> bool:
    if not _should_record_turn(prompt_subset, turn_idx):
        return False
    if prompt_subset.get("enabled"):
        if prompt_subset.get("turn_index") is not None:
            return turn_idx == int(prompt_subset["turn_index"])
        end_idx = prompt_subset.get("turn_end_index")
        if end_idx is not None:
            return turn_idx == int(end_idx)
    return True


def _should_capture_immediate_diagnostics(
    prompt_subset: Dict[str, Any],
    turn_idx: int,
    diagnostic_scope: str,
) -> bool:
    if diagnostic_scope == "all":
        return True
    if diagnostic_scope == "final":
        return _is_final_recorded_turn(prompt_subset, turn_idx)
    if diagnostic_scope == "postponed":
        return False
    die(f"Unknown diagnostic scope: {diagnostic_scope}")


def _diagnostic_output_path(
    model_name: str,
    profile_id: str,
    prompt_subset: Dict[str, Any],
    attn_implementation: Optional[str],
    explicit_output: Optional[str],
) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()

    base = profile_filename(profile_id)
    if not prompt_subset.get("enabled") and not attn_implementation:
        return REFERENCE_DIR / model_name / base

    stem = Path(base).stem
    suffix_parts: List[str] = []
    if prompt_subset.get("enabled"):
        conv_idx = prompt_subset.get("conversation_index")
        first_stored_turns = prompt_subset.get("first_stored_turns")
        turn_idx = prompt_subset.get("turn_index")
        turn_start_idx = prompt_subset.get("turn_start_index")
        turn_end_idx = prompt_subset.get("turn_end_index")
        if first_stored_turns is not None:
            suffix_parts.append(f"first{first_stored_turns}turns")
        else:
            suffix_parts.append(f"conv{conv_idx}" if conv_idx is not None else "subset")
        if turn_idx is not None:
            suffix_parts.append(f"turn{turn_idx}")
        elif turn_start_idx is not None or turn_end_idx is not None:
            suffix_parts.append(f"turns{turn_start_idx}-{turn_end_idx}")
    if attn_implementation:
        suffix_parts.append(f"attn-{attn_implementation}")
    suffix = "__" + "__".join(suffix_parts) if suffix_parts else ""
    return REFERENCE_DIR / model_name / f"{stem}{suffix}.json"


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def build_token_diagnostic_entry(
    tokenizer: Any,
    step_logits: Any,
    *,
    expected_token_id: int,
    logit_pos: int,
    prev_token_id: int,
    diagnostic_top_k: int,
) -> Dict[str, Any]:
    import torch

    logits_stats = _tensor_finiteness_stats(step_logits)
    log_probs = torch.log_softmax(step_logits.float(), dim=-1)
    log_probs_stats = _tensor_finiteness_stats(log_probs)
    expected_log_prob = float(log_probs[expected_token_id].item())
    expected_logit = float(step_logits[expected_token_id].float().item())
    expected_rank = int(torch.count_nonzero(step_logits > step_logits[expected_token_id]).item()) + 1
    vocab_size = step_logits.shape[-1]
    top_k = min(int(diagnostic_top_k), int(vocab_size))

    top_vals, top_ids = torch.topk(log_probs, k=top_k)
    top_k_entries = []
    for tok_id, log_prob in zip(top_ids.tolist(), top_vals.tolist()):
        top_k_entries.append(
            {
                "token_id": int(tok_id),
                "text": _decode_token_text(tokenizer, int(tok_id)),
                "log_prob": float(log_prob),
            }
        )

    return {
        "position": int(logit_pos),
        "previous_token_id": int(prev_token_id),
        "previous_token_text": _decode_token_text(tokenizer, int(prev_token_id)),
        "token_id": int(expected_token_id),
        "text": _decode_token_text(tokenizer, int(expected_token_id)),
        "log_prob": expected_log_prob,
        "logit": expected_logit,
        "rank": expected_rank,
        "logits_stats": logits_stats,
        "log_probs_stats": log_probs_stats,
        "top_k": top_k_entries,
    }


def build_teacher_forced_per_token_data(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Any,
    generated_token_ids: List[int],
    *,
    diagnostic_steps: int,
    diagnostic_top_k: int,
    use_cache: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    if diagnostic_steps <= 0 or diagnostic_top_k <= 0 or not generated_token_ids:
        return []

    import torch

    steps_to_capture = min(len(generated_token_ids), diagnostic_steps)
    teacher_suffix = generated_token_ids[: max(0, steps_to_capture - 1)]
    teacher_ids = prompt_input_ids[0].tolist() + teacher_suffix
    teacher_tensor = torch.tensor([teacher_ids], dtype=prompt_input_ids.dtype, device=model.device)

    forward_kwargs: Dict[str, Any] = {"attention_mask": torch.ones_like(teacher_tensor)}
    if use_cache is not None:
        forward_kwargs["use_cache"] = use_cache
    with torch.no_grad():
        logits = model(teacher_tensor, **forward_kwargs).logits[0].float()

    prompt_len = prompt_input_ids.shape[1]
    per_token_data: List[Dict[str, Any]] = []

    for step in range(steps_to_capture):
        logit_pos = prompt_len - 1 + step
        step_logits = logits[logit_pos]
        expected_token_id = int(generated_token_ids[step])
        prev_token_id = int(prompt_input_ids[0, -1].item()) if step == 0 else int(generated_token_ids[step - 1])
        per_token_data.append(
            build_token_diagnostic_entry(
                tokenizer,
                step_logits,
                expected_token_id=expected_token_id,
                logit_pos=logit_pos,
                prev_token_id=prev_token_id,
                diagnostic_top_k=diagnostic_top_k,
            )
        )

    return per_token_data


def capture_prefill_logits(model: Any, prompt_input_ids: Any, *, use_cache: Optional[bool] = None) -> Any:
    import torch

    forward_kwargs: Dict[str, Any] = {"attention_mask": torch.ones_like(prompt_input_ids)}
    if use_cache is not None:
        forward_kwargs["use_cache"] = use_cache
    with torch.no_grad():
        return model(prompt_input_ids, **forward_kwargs).logits[0, -1].detach()


def capture_prefill_forward_diagnostic(
    model: Any,
    prompt_input_ids: Any,
    *,
    use_cache: Optional[bool] = None,
    include_hidden_summaries: bool = False,
    include_layer0_internals: bool = False,
    include_layer1_internals: bool = False,
    include_layer2_internals: bool = False,
    layer0_element_dims: Optional[List[int]] = None,
    layer0_row_indices: Optional[List[int]] = None,
    layer1_element_dims: Optional[List[int]] = None,
    layer1_row_indices: Optional[List[int]] = None,
    layer2_element_dims: Optional[List[int]] = None,
    layer2_row_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    import torch

    forward_kwargs: Dict[str, Any] = {"attention_mask": torch.ones_like(prompt_input_ids)}
    if use_cache is not None:
        forward_kwargs["use_cache"] = use_cache
    if include_hidden_summaries:
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True
    with torch.no_grad():
        outputs = model(prompt_input_ids, **forward_kwargs)

    diagnostic: Dict[str, Any] = {
        "logits": outputs.logits[0, -1].detach(),
    }
    if include_hidden_summaries:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("HF diagnostic forward did not return hidden_states")
        diagnostic["hidden_state_summaries"] = [
            _tensor_last_token_summary(hidden, idx)
            for idx, hidden in enumerate(hidden_states)
        ]
    if include_layer0_internals:
        diagnostic["layer0_internal_summaries"] = _capture_hf_layer0_internal_summaries(
            model,
            prompt_input_ids,
            forward_kwargs,
            element_dims=layer0_element_dims,
            row_indices=layer0_row_indices,
        )
    if include_layer1_internals:
        diagnostic["layer1_internal_summaries"] = _capture_hf_layer1_internal_summaries(
            model,
            prompt_input_ids,
            forward_kwargs,
            element_dims=layer1_element_dims,
            row_indices=layer1_row_indices,
        )
    if include_layer2_internals:
        layer2_summaries = _capture_hf_layer1_internal_summaries(
            model,
            prompt_input_ids,
            forward_kwargs,
            element_dims=layer2_element_dims,
            layer_idx=2,
        )
        layer2_block = _find_hf_block(model, 2)
        if getattr(layer2_block, "block_type", None) == "mamba":
            layer2_summaries.extend(
                _capture_hf_layer0_internal_summaries(
                    model,
                    prompt_input_ids,
                    forward_kwargs,
                    element_dims=layer2_element_dims,
                    layer_idx=2,
                    row_indices=layer2_row_indices,
                )
            )
        diagnostic["layer2_internal_summaries"] = layer2_summaries
    return diagnostic


def build_prefill_diagnostic_from_logits(
    tokenizer: Any,
    prompt_input_ids: Any,
    generated_token_ids: List[int],
    step_logits: Any,
    *,
    diagnostic_top_k: int,
) -> List[Dict[str, Any]]:
    if diagnostic_top_k <= 0 or not generated_token_ids:
        return []
    return [
        build_token_diagnostic_entry(
            tokenizer,
            step_logits,
            expected_token_id=int(generated_token_ids[0]),
            logit_pos=int(prompt_input_ids.shape[1]) - 1,
            prev_token_id=int(prompt_input_ids[0, -1].item()),
            diagnostic_top_k=diagnostic_top_k,
        )
    ]


def build_prefill_next_token_data(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Any,
    generated_token_ids: List[int],
    *,
    diagnostic_top_k: int,
    use_cache: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Capture first-token diagnostics from HF generate scores for exact raw input ids."""
    if diagnostic_top_k <= 0 or not generated_token_ids:
        return []

    import torch

    generate_kwargs = {
        "max_new_tokens": 1,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "attention_mask": torch.ones_like(prompt_input_ids),
    }
    if use_cache is not None:
        generate_kwargs["use_cache"] = use_cache
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id

    with torch.no_grad():
        generated = model.generate(prompt_input_ids, **generate_kwargs)

    scores = getattr(generated, "scores", None)
    if not scores:
        raise RuntimeError("HF generate returned no scores for first-token diagnostic")

    prompt_len = prompt_input_ids.shape[1]
    return [
        build_token_diagnostic_entry(
            tokenizer,
            scores[0][0],
            expected_token_id=int(generated_token_ids[0]),
            logit_pos=prompt_len - 1,
            prev_token_id=int(prompt_input_ids[0, -1].item()),
            diagnostic_top_k=diagnostic_top_k,
        )
    ]


def _past_seen_tokens(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            return 0
    if isinstance(past_key_values, tuple) and past_key_values:
        try:
            return int(past_key_values[0][0].shape[2])
        except Exception:
            return 0
    return 0


def maybe_patch_legacy_remote_cache_position(model: Any) -> None:
    """Backfill cache_position for legacy remote-code generation hooks."""
    import torch
    from transformers.cache_utils import DynamicCache

    prepare = getattr(model, "prepare_inputs_for_generation", None)
    if prepare is None:
        return

    try:
        prepare_sig = inspect.signature(prepare)
        forward_sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return

    if "cache_position" not in prepare_sig.parameters:
        return
    if "cache_position" not in forward_sig.parameters:
        return

    module_name = getattr(model.__class__, "__module__", "")
    if "transformers_modules" not in module_name:
        return
    model_module = importlib.import_module(module_name)
    hybrid_cache_cls = (
        getattr(model_module, "HybridMambaAttentionDynamicCache", None)
        or getattr(model_module, "NemotronHHybridDynamicCache", None)
    )
    if hybrid_cache_cls is not None and not getattr(hybrid_cache_cls, "_krasis_nemotron_cache_compat", False):
        original_cache_init = hybrid_cache_cls.__init__

        class KrasisDeviceList(list):
            @property
            def device(self: Any) -> Any:
                for item in self:
                    if hasattr(item, "device"):
                        return item.device
                return torch.device("cpu")

        def compat_cache_init(self: Any, *cache_args: Any, **cache_kwargs: Any) -> None:
            original_cache_init(self, *cache_args, **cache_kwargs)
            config = cache_args[0] if cache_args else cache_kwargs.get("config")
            if config is not None and not hasattr(self, "conv_kernel_size"):
                self.conv_kernel_size = int(getattr(config, "conv_kernel", getattr(config, "mamba_conv_kernel", 0)))
            if isinstance(getattr(self, "ssm_states", None), list) and not hasattr(self.ssm_states, "device"):
                self.ssm_states = KrasisDeviceList(self.ssm_states)

        def compat_update_conv_state(
            self: Any,
            layer_idx: int,
            new_conv_state: Any,
            cache_init: bool = False,
        ) -> Any:
            current = self.conv_states[layer_idx]
            device = current.device if hasattr(current, "device") else new_conv_state.device
            if cache_init:
                self.conv_states[layer_idx] = new_conv_state.to(device)
            else:
                rolled = current.roll(shifts=-1, dims=-1)
                rolled[:, :, -1] = new_conv_state[:, 0, :].to(device)
                self.conv_states[layer_idx] = rolled
            return self.conv_states[layer_idx]

        def compat_update_ssm_state(self: Any, layer_idx: int, new_ssm_state: Any) -> Any:
            current = self.ssm_states[layer_idx]
            device = current.device if hasattr(current, "device") else new_ssm_state.device
            self.ssm_states[layer_idx] = new_ssm_state.to(device)
            return self.ssm_states[layer_idx]

        hybrid_cache_cls.__init__ = compat_cache_init
        hybrid_cache_cls.update_conv_state = compat_update_conv_state
        hybrid_cache_cls.update_ssm_state = compat_update_ssm_state
        hybrid_cache_cls._krasis_nemotron_cache_compat = True

    if os.environ.get("KRASIS_NEMOTRON_FORCE_TORCH_MAMBA") == "1":
        setattr(model_module, "is_fast_path_available", False)
        print(
            "Nemotron generation cache compat: forced remote Mamba torch path for reference generation",
            file=sys.stderr,
            flush=True,
        )

    original_causal_conv1d_update = getattr(model_module, "causal_conv1d_update", None)
    if original_causal_conv1d_update is not None and not getattr(
        original_causal_conv1d_update,
        "_krasis_nemotron_weight_compat",
        False,
    ):
        def compat_causal_conv1d_update(x: Any, conv_state: Any, weight: Any, *conv_args: Any, **conv_kwargs: Any) -> Any:
            before_weight_shape = tuple(weight.shape) if hasattr(weight, "shape") else None
            if hasattr(weight, "dim") and weight.dim() == 3:
                if weight.shape[1] == 1:
                    weight = weight.squeeze(1)
                elif weight.shape[2] == 1:
                    weight = weight.squeeze(2)
            if hasattr(weight, "contiguous"):
                weight = weight.contiguous()
            if hasattr(conv_state, "contiguous"):
                conv_state = conv_state.contiguous()
            if not getattr(compat_causal_conv1d_update, "_krasis_shape_logged", False):
                print(
                    "Nemotron causal_conv1d_update compat shapes: "
                    f"x={tuple(x.shape) if hasattr(x, 'shape') else None} "
                    f"conv_state={tuple(conv_state.shape) if hasattr(conv_state, 'shape') else None} "
                    f"weight_before={before_weight_shape} "
                    f"weight_after={tuple(weight.shape) if hasattr(weight, 'shape') else None}",
                    file=sys.stderr,
                    flush=True,
                )
                compat_causal_conv1d_update._krasis_shape_logged = True
            return original_causal_conv1d_update(x, conv_state, weight, *conv_args, **conv_kwargs)

        compat_causal_conv1d_update._krasis_nemotron_weight_compat = True
        model_module.causal_conv1d_update = compat_causal_conv1d_update

    original_prepare = prepare

    def patched_prepare_inputs_for_generation(*args: Any, **kwargs: Any) -> Any:
        arg_list = list(args)
        positional_cache = len(arg_list) > 1
        cache_state = kwargs.get("cache_params")
        if cache_state is None:
            cache_state = kwargs.get("past_key_values")
        if cache_state is None and positional_cache:
            cache_state = arg_list[1]

        cache_position = kwargs.get("cache_position")
        input_ids = kwargs.get("input_ids")
        if input_ids is None and arg_list:
            input_ids = arg_list[0]
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is not None:
            sequence_length = int(input_ids.shape[1])
            batch_size = int(input_ids.shape[0])
            device = input_ids.device
        elif inputs_embeds is not None:
            sequence_length = int(inputs_embeds.shape[1])
            batch_size = int(inputs_embeds.shape[0])
            device = inputs_embeds.device
        else:
            sequence_length = 0
            batch_size = 1
            device = model.device

        is_hybrid_cache = cache_state is not None and hasattr(cache_state, "conv_states") and hasattr(cache_state, "ssm_states")
        is_default_dynamic_cache = isinstance(cache_state, DynamicCache) and not is_hybrid_cache
        if hybrid_cache_cls is not None and is_default_dynamic_cache:
            cache_state = None
            if positional_cache:
                arg_list[1] = None
            kwargs.pop("past_key_values", None)
            kwargs.pop("cache_params", None)
        elif cache_state is not None and positional_cache:
            arg_list[1] = cache_state
        elif cache_state is not None:
            kwargs["past_key_values"] = cache_state
            kwargs.pop("cache_params", None)

        if is_hybrid_cache and input_ids is not None and sequence_length > 1:
            original_input_shape = tuple(input_ids.shape)
            trimmed_input_ids = input_ids[:, -1:]
            if arg_list and arg_list[0] is input_ids:
                arg_list[0] = trimmed_input_ids
            else:
                kwargs["input_ids"] = trimmed_input_ids
            position_ids = kwargs.get("position_ids")
            if position_ids is not None and hasattr(position_ids, "shape") and int(position_ids.shape[-1]) > 1:
                kwargs["position_ids"] = position_ids[:, -1:]
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None and hasattr(attention_mask, "shape") and len(attention_mask.shape) == 2:
                absolute_position = int(attention_mask.shape[-1]) - 1
                kwargs["cache_position"] = torch.tensor([absolute_position], device=device, dtype=torch.long)
                cache_position = kwargs["cache_position"]
            elif cache_position is not None and hasattr(cache_position, "shape") and int(cache_position.numel()) > 1:
                kwargs["cache_position"] = cache_position[-1:]
                cache_position = kwargs["cache_position"]
            input_ids = trimmed_input_ids
            sequence_length = int(trimmed_input_ids.shape[1])
            if not getattr(patched_prepare_inputs_for_generation, "_krasis_trim_logged", False):
                print(
                    "Nemotron generation cache compat: trimmed cached input_ids "
                    f"from {original_input_shape} to {tuple(trimmed_input_ids.shape)} "
                    f"cache_position={kwargs.get('cache_position')}",
                    file=sys.stderr,
                    flush=True,
                )
                patched_prepare_inputs_for_generation._krasis_trim_logged = True

        if cache_position is None and cache_state is not None:
            kwargs["cache_position"] = torch.arange(
                sequence_length,
                device=device,
                dtype=torch.long,
            ) + _past_seen_tokens(cache_state)

        model_inputs = original_prepare(*arg_list, **kwargs)
        if "cache_params" in forward_sig.parameters and "cache_params" not in model_inputs:
            cache_for_forward = model_inputs.pop("past_key_values", None)
            if cache_for_forward is not None:
                model_inputs["cache_params"] = cache_for_forward
        return model_inputs

    setattr(model, "prepare_inputs_for_generation", patched_prepare_inputs_for_generation)


def apply_model_config_compat(config: Any) -> None:
    """Backfill config aliases expected by newer HF integration code."""

    def _patch_one(target: Any) -> None:
        if target is None or hasattr(target, "num_experts"):
            return

        expert_count = None
        if hasattr(target, "num_local_experts"):
            expert_count = getattr(target, "num_local_experts")
        elif hasattr(target, "n_routed_experts"):
            expert_count = getattr(target, "n_routed_experts")

        if expert_count is not None:
            setattr(target, "num_experts", expert_count)

    _patch_one(config)
    for attr_name in ("text_config", "language_config", "llm_config"):
        _patch_one(getattr(config, attr_name, None))


def _target_pattern_matches_param_name(pattern: str, param_name: str) -> bool:
    regex = re.escape(pattern)
    regex = regex.replace(r"\.\*\.", r"\..*\.")
    regex = regex.replace(r"\*", r".*")
    return re.search(regex, param_name) is not None


def estimate_conversion_workspace_bytes(model_loader: Any, config: Any) -> int:
    """Estimate the largest temporary tensor created by checkpoint conversion ops."""
    import torch
    from accelerate import init_empty_weights

    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import Concatenate, WeightConverter
    except ImportError as exc:
        warn(f"Could not inspect conversion mapping on this transformers version: {exc}")
        return 0

    try:
        with init_empty_weights(include_buffers=True):
            empty_model = model_loader.from_config(config, trust_remote_code=True)
    except Exception as exc:
        warn(f"Could not build empty model for conversion workspace estimate: {exc}")
        return 0

    try:
        conversions = get_model_conversion_mapping(empty_model)
        max_bytes = 0
        seen_param_names = set()
        for conversion in conversions:
            if not isinstance(conversion, WeightConverter):
                continue
            if not any(isinstance(op, Concatenate) for op in conversion.operations):
                continue
            for param_name, param in empty_model.named_parameters():
                if param_name in seen_param_names:
                    continue
                if not any(
                    _target_pattern_matches_param_name(target_pattern, param_name)
                    for target_pattern in conversion.target_patterns
                ):
                    continue
                seen_param_names.add(param_name)
                max_bytes = max(max_bytes, int(param.numel()) * torch.empty((), dtype=param.dtype).element_size())
        return max_bytes
    finally:
        del empty_model


def load_reference_model_and_tokenizer(
    model_name: str,
    *,
    attn_implementation: Optional[str] = None,
    capture_dt_bias_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Load the BF16 HF reference model through the shared capture path."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    model_name = resolve_model_name(model_name)
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        die(f"Model not found: {model_path}")

    info(f"Model: {model_name}")
    info(f"Path: {model_path}")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    apply_model_config_compat(config)
    if attn_implementation:
        try:
            config._attn_implementation = attn_implementation
        except Exception:
            pass
    info(f"Config loaded ({time.time() - t0:.1f}s)")

    architectures = list(getattr(config, "architectures", []) or [])
    use_conditional_loader = any("ConditionalGeneration" in arch for arch in architectures)
    model_loader = AutoModelForImageTextToText if use_conditional_loader else AutoModelForCausalLM
    info(
        "HF model loader: "
        f"{model_loader.__name__} (architectures={architectures if architectures else ['unknown']})"
    )

    t0 = time.time()
    tokenizer = load_tokenizer_with_compat(model_path)
    info(f"Tokenizer loaded ({time.time() - t0:.1f}s)")

    conversion_workspace_bytes = estimate_conversion_workspace_bytes(model_loader, config)
    conversion_workspace_gib = conversion_workspace_bytes / (1024 ** 3)
    if conversion_workspace_bytes > 0:
        info(f"Estimated conversion workspace reserve: {conversion_workspace_gib:.2f}GiB")

    num_gpus = torch.cuda.device_count()
    max_memory = {"cpu": "200GiB"}
    base_headroom_mb = 2048
    conversion_headroom_mb = (conversion_workspace_bytes + (1024 * 1024 - 1)) // (1024 * 1024)
    total_headroom_mb = base_headroom_mb + conversion_headroom_mb
    for i in range(num_gpus):
        free_mb = torch.cuda.mem_get_info(i)[0] // (1024 * 1024)
        alloc_gb = max(1, (free_mb - total_headroom_mb)) / 1024
        max_memory[i] = f"{alloc_gb:.1f}GiB"
        info(
            f"GPU {i}: {free_mb}MB free, allocating {alloc_gb:.1f}GiB "
            f"(headroom {total_headroom_mb}MB)"
        )
    info(f"Loading model in BF16 ({num_gpus} GPUs + CPU offload)...")

    t0 = time.time()
    load_kwargs = {
        "config": config,
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "max_memory": max_memory,
        "trust_remote_code": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
        info(f"Requested HF attention implementation: {attn_implementation}")
    model = model_loader.from_pretrained(model_path, **load_kwargs)
    dt_bias_source_diagnostics = (
        capture_hf_layer0_dt_bias_source_diagnostics(model, model_path)
        if capture_dt_bias_diagnostics
        else None
    )
    dt_bias_restore_diagnostics = restore_nemotron_dt_bias_from_safetensors(
        model,
        config,
        model_path,
    )
    if dt_bias_restore_diagnostics.get("applied"):
        info(
            "Restored Nemotron dt_bias from safetensors: "
            f"{dt_bias_restore_diagnostics.get('restored_parameters')} tensors, "
            f"max_abs_before={dt_bias_restore_diagnostics.get('max_abs_before_restore')}"
        )
    if capture_dt_bias_diagnostics:
        post_restore = capture_hf_layer0_dt_bias_source_diagnostics(model, model_path)
        for row in post_restore:
            row["source_phase"] = "after_safetensor_restore"
            if row.get("label") == "layer0_mamba2_dt_bias_after_from_pretrained":
                row["label"] = "layer0_mamba2_dt_bias_after_safetensor_restore"
                row["source"] = "hf_model_parameter_after_safetensor_restore"
        if dt_bias_source_diagnostics is None:
            dt_bias_source_diagnostics = []
        dt_bias_source_diagnostics.append(dt_bias_restore_diagnostics)
        dt_bias_source_diagnostics.extend(post_restore)
    maybe_patch_legacy_remote_cache_position(model)
    model.eval()
    load_time = time.time() - t0
    info(f"Model loaded ({load_time:.1f}s)")

    runtime_version = "unknown"
    try:
        import transformers
        runtime_version = transformers.__version__
    except Exception:
        pass

    return {
        "model_name": model_name,
        "model_path": model_path,
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
        "dt_bias_source_diagnostics": dt_bias_source_diagnostics,
        "dt_bias_restore_diagnostics": dt_bias_restore_diagnostics,
        "runtime_version": runtime_version,
        "num_gpus": num_gpus,
        "load_kwargs_audit": {
            "dtype": str(load_kwargs.get("dtype")),
            "device_map": load_kwargs.get("device_map"),
            "max_memory": max_memory,
            "trust_remote_code": load_kwargs.get("trust_remote_code"),
            "attn_implementation": load_kwargs.get("attn_implementation"),
        },
    }


def build_environment_audit(
    loaded: Dict[str, Any],
    *,
    attn_implementation: Optional[str],
    diagnostic_controls: Dict[str, Any],
) -> Dict[str, Any]:
    import torch

    model = loaded["model"]
    config = loaded["config"]
    tokenizer = loaded["tokenizer"]
    generation_config = getattr(model, "generation_config", None)
    first_param_dtype = None
    first_param_device = None
    try:
        first_param = next(model.parameters())
        first_param_dtype = str(first_param.dtype)
        first_param_device = str(first_param.device)
    except Exception:
        pass

    cuda_devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            cuda_devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "capability": [props.major, props.minor],
                    "total_memory_bytes": int(total_bytes),
                    "free_memory_bytes": int(free_bytes),
                }
            )

    return {
        "python_executable": sys.executable,
        "packages": {
            "transformers": _package_version("transformers"),
            "torch": getattr(torch, "__version__", None),
            "accelerate": _package_version("accelerate"),
            "tokenizers": _package_version("tokenizers"),
            "safetensors": _package_version("safetensors"),
        },
        "cuda": {
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "devices": cuda_devices,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model": {
            "name": loaded.get("model_name"),
            "path": loaded.get("model_path"),
            "class": type(model).__name__,
            "config_class": type(config).__name__,
            "architectures": list(getattr(config, "architectures", []) or []),
            "hf_device_map": _json_safe(getattr(model, "hf_device_map", None)),
            "first_parameter_dtype": first_param_dtype,
            "first_parameter_device": first_param_device,
            "load_kwargs": loaded.get("load_kwargs_audit"),
            "dt_bias_source_diagnostics": loaded.get("dt_bias_source_diagnostics"),
            "dt_bias_restore_diagnostics": loaded.get("dt_bias_restore_diagnostics"),
            "files": _model_file_audit(str(loaded.get("model_path"))),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "chat_template_sha256": hashlib.sha256(
                str(getattr(tokenizer, "chat_template", "") or "").encode("utf-8")
            ).hexdigest(),
        },
        "generation": {
            "config": _json_safe(generation_config.to_dict() if hasattr(generation_config, "to_dict") else None),
            "config_use_cache": getattr(config, "use_cache", None),
            "generation_config_use_cache": getattr(generation_config, "use_cache", None),
            "attn_implementation_arg": attn_implementation,
            "config_attn_implementation": getattr(config, "_attn_implementation", None),
            "diagnostic_controls": diagnostic_controls,
        },
    }


def generate_reference(
    model_name: str,
    max_new_tokens: int = 200,
    profile: str = "auto",
    diagnostic_steps: int = DEFAULT_DIAGNOSTIC_STEPS,
    diagnostic_top_k: int = DEFAULT_DIAGNOSTIC_TOPK,
    conversation_index: Optional[int] = None,
    turn_index: Optional[int] = None,
    turn_start_index: Optional[int] = None,
    turn_end_index: Optional[int] = None,
    first_stored_turns: Optional[int] = None,
    attn_implementation: Optional[str] = None,
    output: Optional[str] = None,
    use_cache_mode: str = "auto",
    generate_use_cache_mode: Optional[str] = None,
    diagnostic_use_cache_mode: Optional[str] = None,
    diagnostic_scope: str = "all",
    diagnose_pre_post_forward: bool = False,
    diagnose_hidden_summaries: bool = False,
    diagnose_layer0_internals: bool = False,
    diagnose_layer1_internals: bool = False,
    diagnose_layer2_internals: bool = False,
    diagnose_layer0_element_dims: Optional[List[int]] = None,
    diagnose_layer0_row_indices: Optional[List[int]] = None,
    diagnose_layer1_element_dims: Optional[List[int]] = None,
    diagnose_layer1_row_indices: Optional[List[int]] = None,
    diagnose_layer2_element_dims: Optional[List[int]] = None,
    diagnose_layer2_row_indices: Optional[List[int]] = None,
    between_turn_cleanup: str = "none",
    reload_model_between_turns: bool = False,
    diagnostic_only: bool = False,
    prefill_only_first_token: bool = False,
    raw_input_json: Optional[str] = None,
):
    """Generate reference outputs using HuggingFace transformers."""
    import torch

    if diagnostic_scope not in ("all", "final", "postponed"):
        die("--diagnostic-scope must be one of: all, final, postponed")
    generate_cache_mode = generate_use_cache_mode or use_cache_mode
    diagnostic_cache_mode = diagnostic_use_cache_mode or use_cache_mode

    model_name = resolve_model_name(model_name)
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        die(f"Model not found: {model_path}")

    raw_input_cases: List[Dict[str, Any]] = []
    raw_input_path: Optional[Path] = None
    raw_input_by_conv_turn: Dict[Tuple[int, int], Dict[str, Any]] = {}

    if raw_input_json:
        if (
            conversation_index is not None
            or turn_index is not None
            or turn_start_index is not None
            or turn_end_index is not None
            or first_stored_turns is not None
        ):
            die("--raw-input-json cannot be combined with prompt subset selectors")
        raw_input_path = Path(raw_input_json).expanduser().resolve()
        if not raw_input_path.is_file():
            die(f"Raw input JSON not found: {raw_input_path}")
        raw_input_cases = load_raw_input_cases(raw_input_path)
        conversations = [[case["prompt"]] for case in raw_input_cases]
        raw_input_by_conv_turn = {(idx, 0): case for idx, case in enumerate(raw_input_cases)}
        prompt_subset = {
            "enabled": True,
            "conversation_index": None,
            "turn_index": None,
            "turn_start_index": None,
            "turn_end_index": None,
            "first_stored_turns": None,
            "mode": "raw_input_json",
            "source_path": str(raw_input_path),
            "case_count": len(raw_input_cases),
            "recorded_turns": len(raw_input_cases),
            "input_token_hashes_fnv1a64": [
                case["input_token_hash_fnv1a64"] for case in raw_input_cases
            ],
        }
    else:
        if not PROMPTS_FILE.is_file():
            die(f"Prompts file not found: {PROMPTS_FILE}")

        # Parse prompts
        with open(PROMPTS_FILE) as f:
            lines = f.readlines()
        source_conversations = parse_prompt_conversations(lines)
        conversations, prompt_subset = _build_prompt_subset(
            source_conversations,
            conversation_index,
            turn_index,
            turn_start_index,
            turn_end_index,
            first_stored_turns,
        )
    total_prompts = sum(len(c) for c in conversations)
    recorded_turns = int(prompt_subset.get("recorded_turns") or total_prompts)

    info(f"Model: {model_name}")
    info(f"Path: {model_path}")
    info(f"Conversations: {len(conversations)} ({total_prompts} prompts, {recorded_turns} recorded)")
    info(f"Max new tokens: {max_new_tokens}")
    if prompt_subset.get("enabled"):
        info(f"Prompt subset: {prompt_subset}")
    if attn_implementation:
        info(f"HF attention implementation override: {attn_implementation}")
    diagnostic_controls = {
        "use_cache_mode": use_cache_mode,
        "generate_use_cache_mode": generate_cache_mode,
        "diagnostic_use_cache_mode": diagnostic_cache_mode,
        "diagnostic_scope": diagnostic_scope,
        "diagnose_pre_post_forward": diagnose_pre_post_forward,
        "diagnose_hidden_summaries": diagnose_hidden_summaries,
        "diagnose_layer0_internals": diagnose_layer0_internals,
        "diagnose_layer1_internals": diagnose_layer1_internals,
        "diagnose_layer2_internals": diagnose_layer2_internals,
        "diagnose_layer0_element_dims": diagnose_layer0_element_dims or [],
        "diagnose_layer0_row_indices": diagnose_layer0_row_indices or [],
        "diagnose_layer1_element_dims": diagnose_layer1_element_dims or [],
        "diagnose_layer1_row_indices": diagnose_layer1_row_indices or [],
        "diagnose_layer2_element_dims": diagnose_layer2_element_dims or [],
        "diagnose_layer2_row_indices": diagnose_layer2_row_indices or [],
        "between_turn_cleanup": between_turn_cleanup,
        "reload_model_between_turns": reload_model_between_turns,
        "diagnostic_only": diagnostic_only,
        "prefill_only_first_token": prefill_only_first_token,
    }
    info(f"Diagnostic controls: {diagnostic_controls}")
    loaded = load_reference_model_and_tokenizer(
        model_name,
        attn_implementation=attn_implementation,
        capture_dt_bias_diagnostics=diagnose_layer0_internals,
    )
    model_name = loaded["model_name"]
    model_path = loaded["model_path"]
    config = loaded["config"]
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    cache_mode_state = _set_cache_mode(model, config, generate_cache_mode)
    generate_use_cache_value = _cache_mode_to_value(generate_cache_mode)
    diagnostic_use_cache_value = _cache_mode_to_value(diagnostic_cache_mode)
    environment_audit = build_environment_audit(
        loaded,
        attn_implementation=attn_implementation,
        diagnostic_controls=diagnostic_controls,
    )

    eos_token_ids = collect_capture_stop_ids(
        tokenizer,
        model_path=model_path,
        config_json=config.to_dict() if hasattr(config, "to_dict") else None,
    )
    info(f"EOS token IDs: {eos_token_ids}")

    profile_id = canonical_profile_id(tokenizer, profile)
    capture_settings = capture_settings_for_profile(profile_id)
    capture_settings["source"] = "local_generate_reference"
    if raw_input_path is not None:
        capture_settings["source"] = "local_generate_reference_raw_input_json"
        capture_settings["raw_input_json"] = str(raw_input_path)
    if prompt_subset.get("enabled"):
        capture_settings["prompt_subset"] = prompt_subset
    if attn_implementation:
        capture_settings["attn_implementation"] = attn_implementation
    capture_settings["diagnostic_controls"] = diagnostic_controls
    info(f"Reference profile: {profile_id}")
    invocation = build_invocation_metadata(model_name, profile_id, max_new_tokens)
    invocation["prompt_subset"] = prompt_subset
    if raw_input_path is not None:
        invocation["raw_input_json"] = str(raw_input_path)
    if attn_implementation:
        invocation["attn_implementation"] = attn_implementation
    invocation["diagnostic_controls"] = diagnostic_controls
    invocation["environment_audit"] = environment_audit

    # Generate reference outputs
    result = {
        "format_version": 5,
        "model": model_name,
        "model_path": model_path,
        "generated_at": datetime.now().isoformat(),
        "runtime": "transformers",
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
        "eos_token_ids": eos_token_ids,
        "capture_settings": capture_settings,
        "capture_invocation": invocation,
        "decode_diagnostics": {
            "schema_version": 1,
            "coverage": "teacher_forced_first_generated_steps",
            "captured_steps_per_turn": diagnostic_steps,
            "top_k": diagnostic_top_k,
            "log_prob_base": "natural_log",
        },
        "prompt_subset": prompt_subset,
        "attn_implementation": attn_implementation,
        "environment_audit": environment_audit,
        "conversations": [],
    }

    result["runtime_version"] = loaded["runtime_version"]
    if loaded.get("dt_bias_source_diagnostics") is not None:
        result["dt_bias_source_diagnostics"] = loaded["dt_bias_source_diagnostics"]

    prompt_num = 0
    for conv_idx, conversation in enumerate(conversations):
        conv_result: Dict[str, Any] = {"turns": []}
        messages: List[Dict[str, str]] = []
        is_multi = len(conversation) > 1

        if is_multi:
            info(f"Conversation {conv_idx + 1} ({len(conversation)} turns)")

        for turn_idx, prompt in enumerate(conversation):
            cleanup_state: Optional[Dict[str, Any]] = None
            reloaded_before_turn = False
            if turn_idx > 0:
                cleanup_state = _cleanup_between_turns(model, config, between_turn_cleanup)
                if reload_model_between_turns:
                    info(f"Reloading HF model before turn {turn_idx} as requested")
                    del model
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    loaded = load_reference_model_and_tokenizer(
                        model_name,
                        attn_implementation=attn_implementation,
                        capture_dt_bias_diagnostics=diagnose_layer0_internals,
                    )
                    model_name = loaded["model_name"]
                    model_path = loaded["model_path"]
                    config = loaded["config"]
                    model = loaded["model"]
                    tokenizer = loaded["tokenizer"]
                    cache_mode_state = _set_cache_mode(model, config, generate_cache_mode)
                    reloaded_before_turn = True

            prompt_num += 1
            turn_label = f" [turn {turn_idx + 1}/{len(conversation)}]" if is_multi else ""
            print(f"  {GREEN}[{prompt_num}/{total_prompts}]{NC} {prompt}{turn_label}")

            raw_case = raw_input_by_conv_turn.get((conv_idx, turn_idx))
            if raw_case is not None:
                input_ids = torch.tensor([raw_case["input_token_ids"]], dtype=torch.long)
            else:
                messages.append({"role": "user", "content": prompt})

                # Apply chat template — disable thinking if the model supports it
                # We want the "normal" response for reference, not thinking tokens
                try:
                    template_out = apply_capture_template(tokenizer, messages, capture_settings)
                except TypeError:
                    die(
                        "Requested reference profile is incompatible with this tokenizer/chat template: "
                        f"profile={profile_id}"
                    )

                # apply_chat_template may return a tensor or a BatchEncoding
                if hasattr(template_out, "input_ids"):
                    input_ids = template_out.input_ids
                elif isinstance(template_out, torch.Tensor):
                    input_ids = template_out
                else:
                    input_ids = torch.tensor([template_out], dtype=torch.long)

            input_ids = input_ids.to(model.device)
            input_len = input_ids.shape[1]
            cache_snapshot_before_turn = _cache_state_snapshot(model, config)
            pre_generate_forward_diagnostic: Optional[Dict[str, Any]] = None
            should_record = _should_record_turn(prompt_subset, turn_idx)
            should_capture_diagnostics = should_record and _should_capture_immediate_diagnostics(
                prompt_subset,
                turn_idx,
                diagnostic_scope,
            )
            if diagnose_pre_post_forward and should_capture_diagnostics:
                pre_generate_forward_diagnostic = capture_prefill_forward_diagnostic(
                    model,
                    input_ids,
                    use_cache=diagnostic_use_cache_value,
                    include_hidden_summaries=diagnose_hidden_summaries,
                    include_layer0_internals=diagnose_layer0_internals,
                    include_layer1_internals=diagnose_layer1_internals,
                    include_layer2_internals=diagnose_layer2_internals,
                    layer0_element_dims=diagnose_layer0_element_dims,
                    layer0_row_indices=diagnose_layer0_row_indices,
                    layer1_element_dims=diagnose_layer1_element_dims,
                    layer1_row_indices=diagnose_layer1_row_indices,
                    layer2_element_dims=diagnose_layer2_element_dims,
                    layer2_row_indices=diagnose_layer2_row_indices,
                )

            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,  # Greedy decoding
                "eos_token_id": eos_token_ids if eos_token_ids else None,
                "attention_mask": torch.ones_like(input_ids),
            }
            if generate_use_cache_value is not None:
                generate_kwargs["use_cache"] = generate_use_cache_value
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            cache_snapshot_before_generate = _cache_state_snapshot(model, config)
            if prefill_only_first_token:
                if pre_generate_forward_diagnostic is None:
                    die(
                        "--prefill-only-first-token requires "
                        "--diagnose-pre-post-forward so first-token logits are captured"
                    )
                first_token_logits = pre_generate_forward_diagnostic["logits"]
                new_token_ids = [int(torch.argmax(first_token_logits.reshape(-1)).item())]
                cache_snapshot_after_generate = _cache_state_snapshot(model, config)
                gen_time = 0.0
            else:
                t0 = time.time()
                with torch.no_grad():
                    output_ids = model.generate(input_ids, **generate_kwargs)
                cache_snapshot_after_generate = _cache_state_snapshot(model, config)
                gen_time = time.time() - t0

                # Extract only new tokens
                new_token_ids = output_ids[0][input_len:].tolist()

            # Strip trailing EOS/pad tokens for generated continuations. A
            # prefill-only first-token oracle must keep the selected token even
            # when it is a stop token.
            while (not prefill_only_first_token) and new_token_ids and new_token_ids[-1] in eos_token_ids:
                new_token_ids.pop()

            text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            tok_s = len(new_token_ids) / gen_time if gen_time > 0 else 0

            preview = text[:120].replace("\n", " ")
            print(f"    {DIM}{len(new_token_ids)} tokens, {gen_time:.1f}s ({tok_s:.1f} tok/s){NC}")
            print(f"    {DIM}{preview}{'...' if len(text) > 120 else ''}{NC}")

            if should_record:
                source_conversation_indices = prompt_subset.get("source_conversation_indices")
                if isinstance(source_conversation_indices, list) and conv_idx < len(source_conversation_indices):
                    source_conversation_index = source_conversation_indices[conv_idx]
                elif prompt_subset.get("conversation_index") is not None:
                    source_conversation_index = prompt_subset.get("conversation_index")
                else:
                    source_conversation_index = conv_idx
                turn_result = {
                    "prompt": prompt,
                    "source_conversation_index": source_conversation_index,
                    "source_turn_index": turn_idx,
                    "input_token_ids": input_ids[0].tolist(),
                    "input_token_hash_fnv1a64": _fnv1a64_token_hash(input_ids[0].tolist()),
                    "token_ids": new_token_ids,
                    "text": text,
                    "num_tokens": len(new_token_ids),
                    "generation_state": {
                        "model_training": bool(getattr(model, "training", False)),
                        "model_eval": not bool(getattr(model, "training", False)),
                        "attn_implementation": attn_implementation or getattr(config, "_attn_implementation", None),
                        "diagnostic_controls": diagnostic_controls,
                        "use_cache_mode": use_cache_mode,
                        "generate_use_cache_mode": generate_cache_mode,
                        "diagnostic_use_cache_mode": diagnostic_cache_mode,
                        "diagnostic_scope": diagnostic_scope,
                        "use_cache_generate_arg": generate_use_cache_value,
                        "use_cache_forward_arg": diagnostic_use_cache_value,
                        "use_cache_config": bool(getattr(config, "use_cache", True)),
                        "config_use_cache": getattr(config, "use_cache", None),
                        "generation_config_use_cache": getattr(getattr(model, "generation_config", None), "use_cache", None),
                        "cache_mode_state": cache_mode_state,
                        "between_turn_cleanup": cleanup_state,
                        "reloaded_model_before_turn": reloaded_before_turn,
                        "prefill_only_first_token": bool(prefill_only_first_token),
                        "generation_skipped_reason": (
                            "prefill_forward_first_token_only"
                            if prefill_only_first_token
                            else None
                        ),
                        "cache_snapshot_before_turn": cache_snapshot_before_turn,
                        "cache_snapshot_before_generate": cache_snapshot_before_generate,
                        "cache_snapshot_after_generate": cache_snapshot_after_generate,
                        "generate_kwargs_has_past_key_values": "past_key_values" in generate_kwargs,
                        "generate_kwargs_has_cache_params": "cache_params" in generate_kwargs,
                        "explicit_past_key_values_supplied": False,
                        "explicit_cache_object_supplied": False,
                        "past_key_values_reused_from_prior_turn": False,
                        "diagnostics_captured": should_capture_diagnostics,
                        "diagnostics_postponed": diagnostic_scope == "postponed",
                        "diagnostics_skipped_reason": (
                            None
                            if should_capture_diagnostics
                            else (
                                "postponed"
                                if diagnostic_scope == "postponed"
                                else "not_final_recorded_turn"
                            )
                        ),
                    },
                }
                if raw_case is not None:
                    turn_result["raw_input_json"] = {
                        "source_path": str(raw_input_path) if raw_input_path is not None else None,
                        "source_case_index": raw_case["source_case_index"],
                        "input_token_count": raw_case["input_token_count"],
                        "input_token_hash_fnv1a64": raw_case["input_token_hash_fnv1a64"],
                        "source_metadata": raw_case["source_metadata"],
                    }
                if pre_generate_forward_diagnostic is not None:
                    turn_result["pre_generate_diagnostic"] = build_prefill_diagnostic_from_logits(
                        tokenizer,
                        input_ids,
                        new_token_ids,
                        pre_generate_forward_diagnostic["logits"],
                        diagnostic_top_k=diagnostic_top_k,
                    )
                    hidden_summaries = pre_generate_forward_diagnostic.get("hidden_state_summaries")
                    if hidden_summaries is not None:
                        turn_result["pre_generate_hidden_state_summaries"] = hidden_summaries
                    layer0_summaries = pre_generate_forward_diagnostic.get("layer0_internal_summaries")
                    if layer0_summaries is not None:
                        turn_result["pre_generate_layer0_internal_summaries"] = layer0_summaries
                    layer1_summaries = pre_generate_forward_diagnostic.get("layer1_internal_summaries")
                    if layer1_summaries is not None:
                        turn_result["pre_generate_layer1_internal_summaries"] = layer1_summaries
                    layer2_summaries = pre_generate_forward_diagnostic.get("layer2_internal_summaries")
                    if layer2_summaries is not None:
                        turn_result["pre_generate_layer2_internal_summaries"] = layer2_summaries
                if should_capture_diagnostics:
                    per_token_data = build_teacher_forced_per_token_data(
                        model,
                        tokenizer,
                        input_ids,
                        new_token_ids,
                        diagnostic_steps=diagnostic_steps,
                        diagnostic_top_k=diagnostic_top_k,
                        use_cache=diagnostic_use_cache_value,
                    )
                    if per_token_data:
                        turn_result["per_token_data"] = per_token_data
                if diagnose_pre_post_forward and should_capture_diagnostics:
                    post_generate_forward_diagnostic = capture_prefill_forward_diagnostic(
                        model,
                        input_ids,
                        use_cache=diagnostic_use_cache_value,
                        include_hidden_summaries=diagnose_hidden_summaries,
                        include_layer0_internals=diagnose_layer0_internals,
                        include_layer1_internals=diagnose_layer1_internals,
                        include_layer2_internals=diagnose_layer2_internals,
                        layer0_element_dims=diagnose_layer0_element_dims,
                        layer0_row_indices=diagnose_layer0_row_indices,
                        layer1_element_dims=diagnose_layer1_element_dims,
                        layer1_row_indices=diagnose_layer1_row_indices,
                        layer2_element_dims=diagnose_layer2_element_dims,
                        layer2_row_indices=diagnose_layer2_row_indices,
                    )
                    turn_result["post_generate_diagnostic"] = build_prefill_diagnostic_from_logits(
                        tokenizer,
                        input_ids,
                        new_token_ids,
                        post_generate_forward_diagnostic["logits"],
                        diagnostic_top_k=diagnostic_top_k,
                    )
                    hidden_summaries = post_generate_forward_diagnostic.get("hidden_state_summaries")
                    if hidden_summaries is not None:
                        turn_result["post_generate_hidden_state_summaries"] = hidden_summaries
                    layer0_summaries = post_generate_forward_diagnostic.get("layer0_internal_summaries")
                    if layer0_summaries is not None:
                        turn_result["post_generate_layer0_internal_summaries"] = layer0_summaries
                    layer1_summaries = post_generate_forward_diagnostic.get("layer1_internal_summaries")
                    if layer1_summaries is not None:
                        turn_result["post_generate_layer1_internal_summaries"] = layer1_summaries
                    layer2_summaries = post_generate_forward_diagnostic.get("layer2_internal_summaries")
                    if layer2_summaries is not None:
                        turn_result["post_generate_layer2_internal_summaries"] = layer2_summaries
                    turn_result["generation_state"]["cache_snapshot_after_diagnostics"] = _cache_state_snapshot(model, config)
                conv_result["turns"].append(turn_result)

            # Add assistant response to history for multi-turn
            messages.append({"role": "assistant", "content": text})

        if conv_result["turns"]:
            result["conversations"].append(conv_result)

    if diagnostic_scope == "postponed":
        info("Running postponed diagnostics after all selected turns were generated")
        for conv in result["conversations"]:
            for turn in conv["turns"]:
                input_ids = torch.tensor([turn["input_token_ids"]], dtype=torch.long, device=model.device)
                new_token_ids = turn.get("token_ids", [])
                turn["generation_state"]["cache_snapshot_before_postponed_diagnostics"] = _cache_state_snapshot(model, config)
                per_token_data = build_teacher_forced_per_token_data(
                    model,
                    tokenizer,
                    input_ids,
                    new_token_ids,
                    diagnostic_steps=diagnostic_steps,
                    diagnostic_top_k=diagnostic_top_k,
                    use_cache=diagnostic_use_cache_value,
                )
                if per_token_data:
                    turn["per_token_data"] = per_token_data
                if diagnose_pre_post_forward:
                    postponed_forward_diagnostic = capture_prefill_forward_diagnostic(
                        model,
                        input_ids,
                        use_cache=diagnostic_use_cache_value,
                        include_hidden_summaries=diagnose_hidden_summaries,
                        include_layer0_internals=diagnose_layer0_internals,
                        include_layer1_internals=diagnose_layer1_internals,
                        include_layer2_internals=diagnose_layer2_internals,
                        layer0_element_dims=diagnose_layer0_element_dims,
                        layer0_row_indices=diagnose_layer0_row_indices,
                        layer1_element_dims=diagnose_layer1_element_dims,
                        layer1_row_indices=diagnose_layer1_row_indices,
                        layer2_element_dims=diagnose_layer2_element_dims,
                        layer2_row_indices=diagnose_layer2_row_indices,
                    )
                    turn["postponed_prefill_diagnostic"] = build_prefill_diagnostic_from_logits(
                        tokenizer,
                        input_ids,
                        new_token_ids,
                        postponed_forward_diagnostic["logits"],
                        diagnostic_top_k=diagnostic_top_k,
                    )
                    hidden_summaries = postponed_forward_diagnostic.get("hidden_state_summaries")
                    if hidden_summaries is not None:
                        turn["postponed_hidden_state_summaries"] = hidden_summaries
                    layer0_summaries = postponed_forward_diagnostic.get("layer0_internal_summaries")
                    if layer0_summaries is not None:
                        turn["postponed_layer0_internal_summaries"] = layer0_summaries
                    layer1_summaries = postponed_forward_diagnostic.get("layer1_internal_summaries")
                    if layer1_summaries is not None:
                        turn["postponed_layer1_internal_summaries"] = layer1_summaries
                    layer2_summaries = postponed_forward_diagnostic.get("layer2_internal_summaries")
                    if layer2_summaries is not None:
                        turn["postponed_layer2_internal_summaries"] = layer2_summaries
                turn["generation_state"]["diagnostics_captured"] = True
                turn["generation_state"]["diagnostics_postponed"] = True
                turn["generation_state"]["diagnostics_skipped_reason"] = None
                turn["generation_state"]["cache_snapshot_after_diagnostics"] = _cache_state_snapshot(model, config)

    result["contract"] = build_contract(
        model_name=model_name,
        model_path=model_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        add_generation_prompt=True,
        enable_thinking=capture_settings["enable_thinking"],
        profile_id=capture_settings["profile_id"],
        prompt_source_path=str(raw_input_path or PROMPTS_FILE),
        runtime_name="transformers",
        runtime_version=result.get("runtime_version"),
        torch_dtype="bfloat16",
        extra={
            "capture_invocation": invocation,
            "prompt_subset": prompt_subset,
            "attn_implementation": attn_implementation,
            "diagnostic_controls": diagnostic_controls,
        },
    )
    result["sanity"] = build_reference_sanity_report(
        result,
        expected_conversations=len(conversations),
        expected_turns=recorded_turns,
    )
    output_path = _diagnostic_output_path(model_name, profile_id, prompt_subset, attn_implementation, output)
    sanity = result["sanity"]
    if sanity.get("status") != "pass":
        warn(
            "Reference sanity gate: FAIL "
            + ", ".join(sanity.get("failed_checks", []))
        )
        for check in sanity.get("checks", []):
            if not check.get("ok"):
                warn(f"  {check.get('name')}: {check.get('detail')}")
        for sample in sanity.get("thought_leakage", {}).get("samples", []):
            warn(
                "  thought sample "
                f"conv={sample.get('conversation_index')} turn={sample.get('turn_index')}: "
                f"{sample.get('preview')}"
            )
        write_generation_failure_manifest(
            invocation,
            intended_output_path=output_path,
            model_name=model_name,
            profile_id=profile_id,
            sanity=sanity,
            prompt_subset=prompt_subset,
            attn_implementation=attn_implementation,
            turn_summaries=summarize_reference_turns(result, diagnostic_top_k),
            diagnostic_controls=diagnostic_controls,
            environment_audit=environment_audit,
            token_audit_turns=build_token_audit_turns(result),
        )
        die("Reference capture failed strict sanity gates; no reference artifact was written.")

    if diagnostic_only:
        write_generation_diagnostic_manifest(
            invocation,
            intended_output_path=output_path,
            model_name=model_name,
            profile_id=profile_id,
            sanity=sanity,
            prompt_subset=prompt_subset,
            attn_implementation=attn_implementation,
            turn_summaries=summarize_reference_turns(result, diagnostic_top_k),
            diagnostic_controls=diagnostic_controls,
            environment_audit=environment_audit,
            token_audit_turns=build_token_audit_turns(result),
        )
        ok("Diagnostic-only capture complete; no reference artifact was written.")
        return

    # Save reference only after strict sanity gates pass.
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    emit_reference_generation_trace(
        "generate_reference",
        result["contract"],
        str(output_path),
        max_new_tokens=max_new_tokens,
    )
    write_run_manifest(
        invocation,
        output_path,
        model_name,
        profile_id,
        max_new_tokens,
        diagnostic_steps,
        diagnostic_top_k,
        prompt_subset,
        attn_implementation,
        result["sanity"],
    )

    ok(f"Reference saved: {output_path}")
    ok(
        "Reference metadata: "
        f"profile={result['contract'].get('profile_id')} "
        f"template_hash={result['contract'].get('tokenizer', {}).get('chat_template_hash')} "
        f"prompt_sha256={result['contract'].get('prompt_source', {}).get('sha256')}"
    )
    ok(f"Total: {total_prompts} prompts, {sum(len(c['turns']) for c in result['conversations'])} recorded turns")
    ok("Reference sanity gate: PASS")


def main():
    if os.environ.get("KRASIS_ALLOW_ARCHIVED_HF_REFERENCE") != "1":
        print("ERROR: ./dev generate-reference is archived.")
        print("  HF/Transformers reference capture is no longer trusted by default.")
        print("  Use llama-witness for new reference authority.")
        print("  For forensic reruns only, set KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 and document the reason.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generate BF16 HuggingFace greedy reference outputs with stored contract metadata"
    )
    parser.add_argument("model", help="Model directory name under the active capture models dir")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max new tokens per turn (default: 200)")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=("auto",) + REFERENCE_PROFILES,
        help=(
            "Reference capture profile. auto chooses greedy_chat_thinking_off when "
            "the tokenizer chat template supports enable_thinking, otherwise greedy_chat_default"
        ),
    )
    parser.add_argument(
        "--diag-steps",
        type=int,
        default=DEFAULT_DIAGNOSTIC_STEPS,
        help=f"Teacher-forced decode diagnostic coverage per turn (default: {DEFAULT_DIAGNOSTIC_STEPS})",
    )
    parser.add_argument(
        "--diag-topk",
        type=int,
        default=DEFAULT_DIAGNOSTIC_TOPK,
        help=f"Top-k entries to store for each diagnostic step (default: {DEFAULT_DIAGNOSTIC_TOPK})",
    )
    parser.add_argument(
        "--conversation-index",
        type=int,
        default=None,
        help=(
            "Restrict capture to one zero-based conversation from benchmarks/sanity_test_prompts.txt. "
            "Use with --turn-index for a single recorded turn."
        ),
    )
    parser.add_argument(
        "--turn-index",
        type=int,
        default=None,
        help=(
            "Restrict capture to one zero-based turn within --conversation-index. "
            "Prior turns in that conversation are generated only to preserve chat history."
        ),
    )
    parser.add_argument(
        "--turn-start-index",
        type=int,
        default=None,
        help=(
            "Record a zero-based turn range within --conversation-index. "
            "Prior turns before the range are generated only to preserve chat history."
        ),
    )
    parser.add_argument(
        "--turn-end-index",
        type=int,
        default=None,
        help=(
            "End of the zero-based turn range recorded by --turn-start-index. "
            "If omitted with --turn-start-index, records through the end of the conversation."
        ),
    )
    parser.add_argument(
        "--first-stored-turns",
        type=int,
        default=None,
        help=(
            "Record the first N stored turns in global prompt order while preserving conversation grouping. "
            "Used for global-order diagnostics; cannot be combined with conversation/turn selectors."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="Optional HuggingFace attention implementation override for this reference capture",
    )
    parser.add_argument(
        "--use-cache",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Legacy cache mode applied to generate and diagnostics unless the split cache flags are supplied. "
            "auto leaves HF defaults unchanged; on/off explicitly sets model/generation config."
        ),
    )
    parser.add_argument(
        "--generate-use-cache",
        choices=("auto", "on", "off"),
        default=None,
        help="Cache mode passed to model.generate only. Defaults to --use-cache.",
    )
    parser.add_argument(
        "--diagnostic-use-cache",
        choices=("auto", "on", "off"),
        default=None,
        help="Cache mode passed to diagnostic forward calls only. Defaults to --use-cache.",
    )
    parser.add_argument(
        "--diagnostic-scope",
        choices=("all", "final", "postponed"),
        default="all",
        help=(
            "Diagnostic capture scope for recorded turns: all, final recorded turn only, "
            "or postponed until after all selected turns are generated."
        ),
    )
    parser.add_argument(
        "--diagnose-pre-post-forward",
        action="store_true",
        help="For recorded turns, capture first-token diagnostic forward before and after model.generate.",
    )
    parser.add_argument(
        "--diagnose-hidden-summaries",
        action="store_true",
        help=(
            "With --diagnose-pre-post-forward, capture per-hidden-state last-token "
            "summaries from diagnostic forward calls."
        ),
    )
    parser.add_argument(
        "--diagnose-layer0-internals",
        action="store_true",
        help=(
            "With --diagnose-pre-post-forward, capture layer-0 block input, norm, "
            "mixer, and output last-token summaries from diagnostic forward calls."
        ),
    )
    parser.add_argument(
        "--diagnose-layer1-internals",
        action="store_true",
        help=(
            "With --diagnose-pre-post-forward, capture layer-1 MoE input, norm, "
            "router, expert, shared, and output last-token summaries from diagnostic forward calls."
        ),
    )
    parser.add_argument(
        "--diagnose-layer2-internals",
        action="store_true",
        help=(
            "With --diagnose-pre-post-forward, capture layer-2 input, norm, "
            "Mamba2/MoE mixer internals including raw pre-conv and post-conv "
            "Mamba2 x/b/c rows plus conv metadata, and output last-token "
            "summaries from diagnostic forward calls."
        ),
    )
    parser.add_argument(
        "--diagnose-layer0-element-dims",
        default=None,
        help=(
            "Comma-separated non-negative hidden dimensions to include in exact "
            "layer-0 handoff residual/branch/output detail rows."
        ),
    )
    parser.add_argument(
        "--diagnose-layer0-row-indices",
        default=None,
        help=(
            "Comma-separated non-negative sequence row indices to include in "
            "diagnostic-only layer-0 producer row detail captures."
        ),
    )
    parser.add_argument(
        "--diagnose-layer1-element-dims",
        default=None,
        help=(
            "Comma-separated non-negative hidden dimensions to include in exact "
            "layer-1 RMSNorm element detail rows. Omit for full-row details."
        ),
    )
    parser.add_argument(
        "--diagnose-layer1-row-indices",
        default=None,
        help=(
            "Comma-separated non-negative sequence row indices to include in "
            "diagnostic-only layer-1 MoE producer row detail captures."
        ),
    )
    parser.add_argument(
        "--diagnose-layer2-element-dims",
        default=None,
        help=(
            "Comma-separated non-negative hidden dimensions to include in exact "
            "layer-2 RMSNorm and Mamba2 element detail rows. Omit for full-row details."
        ),
    )
    parser.add_argument(
        "--diagnose-layer2-row-indices",
        default=None,
        help=(
            "Comma-separated non-negative sequence row indices to include in "
            "diagnostic-only layer-2 Mamba2 row detail captures."
        ),
    )
    parser.add_argument(
        "--between-turn-cleanup",
        choices=("none", "gc", "reset"),
        default="none",
        help="Cleanup action before each turn after the first: none, gc/CUDA cache cleanup, or reset config/model cache attrs.",
    )
    parser.add_argument(
        "--reload-model-between-turns",
        action="store_true",
        help="Reload the HF model before each turn after the first while preserving generated chat history.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Run generation and strict gates but write only a diagnostic manifest, not a reference JSON artifact.",
    )
    parser.add_argument(
        "--prefill-only-first-token",
        action="store_true",
        help=(
            "Write a first-token oracle from the prompt forward logits and skip "
            "model.generate(). Requires --diagnose-pre-post-forward."
        ),
    )
    parser.add_argument(
        "--raw-input-json",
        default=None,
        help=(
            "Generate from exact input_token_ids in a JSON payload instead of "
            "benchmarks/sanity_test_prompts.txt. Intended for narrow prompt-oracle captures."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional explicit output JSON path. By default, subset or attention-override captures "
            "write a diagnostic non-canonical filename under tests/reference_outputs/<model>/."
        ),
    )
    args = parser.parse_args()

    if args.diag_steps < 0:
        die("--diag-steps must be >= 0")
    if args.diag_topk < 0:
        die("--diag-topk must be >= 0")
    diagnose_layer0_element_dims: Optional[List[int]] = None
    if args.diagnose_layer0_element_dims:
        diagnose_layer0_element_dims = []
        for raw_dim in args.diagnose_layer0_element_dims.split(","):
            raw_dim = raw_dim.strip()
            if not raw_dim:
                continue
            try:
                dim = int(raw_dim, 10)
            except ValueError:
                die("--diagnose-layer0-element-dims must contain only comma-separated integers")
            if dim < 0:
                die("--diagnose-layer0-element-dims must contain only non-negative integers")
            diagnose_layer0_element_dims.append(dim)
    diagnose_layer0_row_indices: Optional[List[int]] = None
    if args.diagnose_layer0_row_indices:
        diagnose_layer0_row_indices = []
        for raw_row in args.diagnose_layer0_row_indices.split(","):
            raw_row = raw_row.strip()
            if not raw_row:
                continue
            try:
                row = int(raw_row, 10)
            except ValueError:
                die("--diagnose-layer0-row-indices must contain only comma-separated integers")
            if row < 0:
                die("--diagnose-layer0-row-indices must contain only non-negative integers")
            diagnose_layer0_row_indices.append(row)
    diagnose_layer1_element_dims: Optional[List[int]] = None
    if args.diagnose_layer1_element_dims:
        diagnose_layer1_element_dims = []
        for raw_dim in args.diagnose_layer1_element_dims.split(","):
            raw_dim = raw_dim.strip()
            if not raw_dim:
                continue
            try:
                dim = int(raw_dim, 10)
            except ValueError:
                die("--diagnose-layer1-element-dims must contain only comma-separated integers")
            if dim < 0:
                die("--diagnose-layer1-element-dims must contain only non-negative integers")
            diagnose_layer1_element_dims.append(dim)
    diagnose_layer1_row_indices: Optional[List[int]] = None
    if args.diagnose_layer1_row_indices:
        diagnose_layer1_row_indices = []
        for raw_row in args.diagnose_layer1_row_indices.split(","):
            raw_row = raw_row.strip()
            if not raw_row:
                continue
            try:
                row = int(raw_row, 10)
            except ValueError:
                die("--diagnose-layer1-row-indices must contain only comma-separated integers")
            if row < 0:
                die("--diagnose-layer1-row-indices must contain only non-negative integers")
            diagnose_layer1_row_indices.append(row)
    diagnose_layer2_element_dims: Optional[List[int]] = None
    if args.diagnose_layer2_element_dims:
        diagnose_layer2_element_dims = []
        for raw_dim in args.diagnose_layer2_element_dims.split(","):
            raw_dim = raw_dim.strip()
            if not raw_dim:
                continue
            try:
                dim = int(raw_dim, 10)
            except ValueError:
                die("--diagnose-layer2-element-dims must contain only comma-separated integers")
            if dim < 0:
                die("--diagnose-layer2-element-dims must contain only non-negative integers")
            diagnose_layer2_element_dims.append(dim)
    diagnose_layer2_row_indices: Optional[List[int]] = None
    if args.diagnose_layer2_row_indices:
        diagnose_layer2_row_indices = []
        for raw_row in args.diagnose_layer2_row_indices.split(","):
            raw_row = raw_row.strip()
            if not raw_row:
                continue
            try:
                row = int(raw_row, 10)
            except ValueError:
                die("--diagnose-layer2-row-indices must contain only comma-separated integers")
            if row < 0:
                die("--diagnose-layer2-row-indices must contain only non-negative integers")
            diagnose_layer2_row_indices.append(row)

    generate_reference(
        args.model,
        args.max_tokens,
        args.profile,
        diagnostic_steps=args.diag_steps,
        diagnostic_top_k=args.diag_topk,
        conversation_index=args.conversation_index,
        turn_index=args.turn_index,
        turn_start_index=args.turn_start_index,
        turn_end_index=args.turn_end_index,
        first_stored_turns=args.first_stored_turns,
        attn_implementation=args.attn_implementation,
        output=args.output,
        use_cache_mode=args.use_cache,
        generate_use_cache_mode=args.generate_use_cache,
        diagnostic_use_cache_mode=args.diagnostic_use_cache,
        diagnostic_scope=args.diagnostic_scope,
        diagnose_pre_post_forward=args.diagnose_pre_post_forward,
        diagnose_hidden_summaries=args.diagnose_hidden_summaries,
        diagnose_layer0_internals=args.diagnose_layer0_internals,
        diagnose_layer1_internals=args.diagnose_layer1_internals,
        diagnose_layer2_internals=args.diagnose_layer2_internals,
        diagnose_layer0_element_dims=diagnose_layer0_element_dims,
        diagnose_layer0_row_indices=diagnose_layer0_row_indices,
        diagnose_layer1_element_dims=diagnose_layer1_element_dims,
        diagnose_layer1_row_indices=diagnose_layer1_row_indices,
        diagnose_layer2_element_dims=diagnose_layer2_element_dims,
        diagnose_layer2_row_indices=diagnose_layer2_row_indices,
        between_turn_cleanup=args.between_turn_cleanup,
        reload_model_between_turns=args.reload_model_between_turns,
        diagnostic_only=args.diagnostic_only,
        prefill_only_first_token=args.prefill_only_first_token,
        raw_input_json=args.raw_input_json,
    )


if __name__ == "__main__":
    main()
