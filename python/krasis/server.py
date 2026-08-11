"""Krasis LLM server — Rust HTTP server with Rust prefill/decode.

Usage:
    python -m krasis.server --model-path /path/to/model
"""

import argparse
import atexit
import gc
import hashlib
import json
import logging
import math
import os
import re
import select
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from krasis.attention_backend import (
    ATTENTION_QUANT_CHOICES,
    attention_quant_cache_nbits,
    attention_quant_label,
    hqq_attention_cache_dir,
)
from krasis.config import DEPRECATED_ATTENTION_QUANT_CHOICES, DEPRECATED_KV_CACHE_FORMAT_CHOICES, KV_CACHE_FORMAT_CHOICES
from krasis.config import HQQ_CACHE_PROFILE_BASELINE, HQQ_CACHE_PROFILE_CHOICES
from krasis.nvidia_smi import ensure_wsl_cuda_env, find_nvidia_smi
from krasis.run_paths import get_run_dir

_PCI_BUS_ID_RE = re.compile(
    r"^(?:(?:pci|bus):)?(?:(?P<domain>[0-9a-fA-F]{4,8}):)?"
    r"(?P<bus>[0-9a-fA-F]{2}):(?P<slot>[0-9a-fA-F]{2})\.(?P<func>[0-7])$"
)
_GPU_MEMORY_SELECTOR_RE = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g|gb|gib|m|mb|mib)$",
    re.IGNORECASE,
)


def _normalize_pci_bus_id(raw: str) -> Optional[str]:
    match = _PCI_BUS_ID_RE.match(raw.strip())
    if not match:
        return None
    domain = match.group("domain") or "0"
    return (
        f"{int(domain, 16):08X}:"
        f"{match.group('bus').upper()}:"
        f"{match.group('slot').upper()}."
        f"{match.group('func')}"
    )


def _gpu_alias_key(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", raw.lower())


def _gpu_memory_selector_matches(spec: str, gpu: dict[str, Any]) -> bool:
    match = _GPU_MEMORY_SELECTOR_RE.match(spec.strip())
    if not match:
        return False
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    target_mb = value * 1024.0 if unit.startswith("g") else value
    tolerance_mb = max(512.0 if unit.startswith("g") else 64.0, target_mb * 0.01)
    return abs(float(gpu.get("vram_mb", 0)) - target_mb) <= tolerance_mb


def _gpu_alias_matches(spec: str, gpu: dict[str, Any]) -> bool:
    if _gpu_memory_selector_matches(spec, gpu):
        return True
    needle = _gpu_alias_key(spec)
    if not needle:
        return False
    return needle in _gpu_alias_key(str(gpu.get("name", "")))


def _gpu_display(gpu: dict[str, Any]) -> str:
    ident = gpu.get("uuid") or gpu.get("pci_bus_id") or f"index {gpu.get('index', '?')}"
    return f"GPU {gpu.get('index', '?')} {gpu.get('name', 'unknown')} ({ident})"


def _unique_gpu_alias_match(spec: str, gpus: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    matches = [gpu for gpu in gpus if _gpu_alias_matches(spec, gpu)]
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def _nvidia_smi_gpu_inventory(source: str) -> list[dict[str, Any]]:
    try:
        ensure_wsl_cuda_env()
        nvidia_smi = find_nvidia_smi() or "nvidia-smi"
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,pci.bus_id,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        details = getattr(exc, "stderr", "") or str(exc)
        raise SystemExit(
            f"{source} selected GPU list uses a stable GPU selector, but nvidia-smi failed "
            f"while resolving it: {details.strip()}"
        ) from exc

    inventory: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) < 5:
            continue
        try:
            index = int(fields[0])
            vram_mb = int(fields[3])
        except ValueError:
            continue
        uuid = fields[1]
        pci_bus_id = _normalize_pci_bus_id(fields[2])
        if pci_bus_id and uuid.startswith(("GPU-", "MIG-")):
            inventory.append({
                "index": index,
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "vram_mb": vram_mb,
                "name": fields[4],
            })
    return inventory


def _normalize_selected_gpus(raw: Optional[str], source: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise SystemExit(f"{source} selected GPU list is empty")
    seen = set()
    values = []
    gpu_inventory: Optional[list[dict[str, Any]]] = None

    def inventory() -> list[dict[str, Any]]:
        nonlocal gpu_inventory
        if gpu_inventory is None:
            gpu_inventory = _nvidia_smi_gpu_inventory(source)
        return gpu_inventory

    for part in text.split(","):
        gpu = part.strip()
        if not gpu:
            continue
        if gpu.isdigit():
            inv = inventory()
            indexed = next((item for item in inv if item.get("index") == int(gpu)), None)
            if indexed and indexed.get("uuid"):
                visible_id = str(indexed["uuid"])
                duplicate_key = f"uuid:{visible_id}"
            elif len(gpu) >= 3:
                match, matches = _unique_gpu_alias_match(gpu, inv)
                if match:
                    visible_id = str(match["uuid"])
                    duplicate_key = f"uuid:{visible_id}"
                elif matches:
                    match_text = "; ".join(_gpu_display(item) for item in matches)
                    raise SystemExit(
                        f"{source} selected GPU selector {gpu!r} is ambiguous: {match_text}"
                    )
                else:
                    available = "; ".join(_gpu_display(item) for item in inv) or "none"
                    raise SystemExit(
                        f"{source} selected GPU selector {gpu!r} did not match any GPU. "
                        f"Available GPUs: {available}"
                    )
            else:
                available = "; ".join(_gpu_display(item) for item in inv) or "none"
                raise SystemExit(
                    f"{source} selected GPU index {gpu!r} did not match any GPU. "
                    f"Available GPUs: {available}"
                )
        elif gpu.startswith(("GPU-", "MIG-")):
            visible_id = gpu
            duplicate_key = f"uuid:{visible_id}"
        else:
            pci_bus_id = _normalize_pci_bus_id(gpu)
            if pci_bus_id:
                by_pci = {
                    str(item.get("pci_bus_id")): str(item.get("uuid"))
                    for item in inventory()
                    if item.get("pci_bus_id") and item.get("uuid")
                }
                visible_id = by_pci.get(pci_bus_id)
                if not visible_id:
                    available = ", ".join(sorted(by_pci)) or "none"
                    raise SystemExit(
                        f"{source} selected GPU PCI bus ID {pci_bus_id} was not found. "
                        f"Available PCI bus IDs: {available}"
                    )
                duplicate_key = f"uuid:{visible_id}"
            else:
                match, matches = _unique_gpu_alias_match(gpu, inventory())
                if match:
                    visible_id = str(match["uuid"])
                    duplicate_key = f"uuid:{visible_id}"
                elif matches:
                    match_text = "; ".join(_gpu_display(item) for item in matches)
                    raise SystemExit(
                        f"{source} selected GPU selector {gpu!r} is ambiguous: {match_text}"
                    )
                else:
                    available = "; ".join(_gpu_display(item) for item in inventory()) or "none"
                    raise SystemExit(
                        f"{source} selected GPU selector {gpu!r} did not match any GPU. "
                        f"Available GPUs: {available}"
                    )
            if not visible_id:
                raise SystemExit(
                    f"{source} selected GPU list contains unsupported entry: {gpu!r}. "
                    "Use physical GPU indices, GPU UUIDs, PCI bus IDs, or unique name/memory aliases."
                )
        if duplicate_key in seen:
            raise SystemExit(f"{source} selected GPU list contains duplicate GPU entry: {gpu}")
        seen.add(duplicate_key)
        values.append(visible_id)
    if not values:
        raise SystemExit(f"{source} selected GPU list is empty")
    if gpu_inventory is not None:
        index_to_uuid = {
            str(item.get("index")): str(item.get("uuid"))
            for item in gpu_inventory
            if item.get("uuid")
        }
        physical_seen = set()
        for value in values:
            physical_key = index_to_uuid.get(value, value)
            if physical_key in physical_seen:
                raise SystemExit(
                    f"{source} selected GPU list contains duplicate physical GPU: {value}"
                )
            physical_seen.add(physical_key)
    return ",".join(values)


def _argv_has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


# Pre-scan config file / CLI for selected GPUs to set CUDA_VISIBLE_DEVICES
# BEFORE any torch/CUDA imports, since CUDA init happens at import time.
def _prescan_selected_gpus():
    import argparse as _ap
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre.add_argument("--selected-gpus", default=None)
    _pre_args, _ = _pre.parse_known_args()
    if _pre_args.selected_gpus is not None:
        _gpus = _normalize_selected_gpus(_pre_args.selected_gpus, "--selected-gpus")
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
        print(f"Pre-scan: set CUDA_VISIBLE_DEVICES={_gpus} (--selected-gpus override)")
        return
    if _pre_args.config and os.path.isfile(_pre_args.config):
        with open(_pre_args.config, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("CFG_SELECTED_GPUS="):
                    _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    if _val.strip():
                        _gpus = _normalize_selected_gpus(_val, "CFG_SELECTED_GPUS")
                        os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
                        print(f"Pre-scan: set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
                    break
_prescan_selected_gpus()

from krasis.config import (
    ADAPTIVE_COLD_MASS_PRUNING_CHOICES,
    GPU_EXPERT_INT4_CALIB_CHOICES,
    QuantConfig,
    cache_dir_for_model,
    configure_adaptive_cold_mass_pruning,
    marlin_cache_basename,
)
from krasis.model import KrasisModel, log_ram_ledger

logger = logging.getLogger("krasis.server")

HEATMAP_FORMAT = "krasis_hcs_heatmap"
HEATMAP_FORMAT_VERSION = 2
APPROVED_HEATMAP_FORMAT = "krasis_approved_hcs_route_heatmap"
APPROVED_HEATMAP_FORMAT_VERSION = 1
APPROVED_HEATMAP_MANIFEST_FORMAT = "krasis_approved_hcs_route_heatmap_manifest"
APPROVED_HEATMAP_MANIFEST_FORMAT_VERSION = 1
APPROVED_HEATMAP_MODE_CHOICES = ("auto", "off", "require")
APPROVED_HEATMAP_DEFAULT_MANIFEST_URL = (
    "https://raw.githubusercontent.com/brontoguana/krasis/"
    "main/benchmarks/approved_heatmaps/manifest.json"
)


def _peer_expert_format_error(gpu_expert_bits: int) -> Optional[str]:
    """Return the explicit runtime incompatibility for peer expert serving."""
    if int(gpu_expert_bits) != 4:
        return "peer expert serving currently requires production INT4 experts"
    return None


HEATMAP_DEFAULT_TOP_K = 50
HEATMAP_DEFAULT_TOP_P = 0.95
HEATMAP_DEFAULT_PRESENCE_PENALTY = 0.0


class ApprovedHeatmapDownloadUnavailable(RuntimeError):
    """Raised when an approved heatmap artifact is absent or unreachable."""

# ANSI formatting for status output
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_NC = "\033[0m"


def _status(label: str) -> None:
    """Print a highlighted status section header (also logged)."""
    print(f"\n{_BOLD}{_CYAN}▸ {label}{_NC}", flush=True)
    logger.info("── %s ──", label)


def _detail(text: str) -> None:
    """Print a detail line under a status header (green, indented)."""
    print(f"  {_GREEN}{text}{_NC}", flush=True)


def _dim(text: str) -> None:
    """Print a dim info line (secondary details)."""
    print(f"  {_DIM}{text}{_NC}", flush=True)


def _warn(text: str) -> None:
    """Print a warning line (yellow, indented)."""
    print(f"  {_YELLOW}{text}{_NC}", flush=True)


def _abort_if_cuda_context_poisoned(context: str, exc: BaseException) -> None:
    text = str(exc)
    if "CUDA_ERROR_ILLEGAL_ADDRESS" in text or "illegal address" in text.lower():
        logger.critical("%s hit fatal CUDA context error: %s", context, text)
        print(
            f"{_RED}FATAL: {context} hit a CUDA illegal-address error; "
            f"exiting because the CUDA context is poisoned.{_NC}",
            flush=True,
        )
        os._exit(134)


def _headline(text: str, color: str = _CYAN) -> None:
    """Print a compact headline that stays readable when stdout is log-prefixed."""
    print(flush=True)
    print(f"{_BOLD}{color}{text}{_NC}", flush=True)


_model: Optional[KrasisModel] = None
_model_name: str = "unknown"

STARTUP_CALIBRATION_SHORT_TOKENS = int(os.environ.get("KRASIS_STARTUP_CALIBRATION_SHORT_TOKENS", "500"))
STARTUP_CALIBRATION_DECODE_TOKENS = int(os.environ.get("KRASIS_STARTUP_CALIBRATION_DECODE_TOKENS", "32"))
STARTUP_CALIBRATION_LONG_TOKENS_CAP = int(os.environ.get("KRASIS_STARTUP_CALIBRATION_LONG_TOKENS_CAP", "50000"))
STARTUP_CALIBRATION_LONG_INITIAL_MULTIPLIER = int(os.environ.get(
    "KRASIS_STARTUP_CALIBRATION_LONG_INITIAL_MULTIPLIER",
    "8",
))
VRAM_CALIBRATION_POLL_INTERVAL_MS = 1


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        _warn(f"Ignoring invalid {name}={raw!r}; using {default}")
        return default


def _startup_diag_enabled() -> bool:
    return os.environ.get("KRASIS_STARTUP_DIAG", "") == "1"


def _heatmap_substage_timing_enabled() -> bool:
    return os.environ.get("KRASIS_HEATMAP_SUBSTAGE_TIMING", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _vram_ledger_enabled() -> bool:
    return os.environ.get("KRASIS_VRAM_LEDGER", "").strip().lower() in ("1", "true", "yes", "on")


def _env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip() not in ("", "0", "false", "False")


def _env_disabled(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("0", "false", "no", "off")


def _nemotron_default_optimizations_enabled(model: KrasisModel) -> bool:
    return (
        getattr(getattr(model, "cfg", None), "model_type", None) == "nemotron_h"
        and not _env_disabled("KRASIS_NEMOTRON_DEFAULT_OPTIMIZATIONS")
    )


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _marlin_digest_cache_path(path: str) -> str:
    return f"{path}.sha256.json"


def _write_marlin_digest_cache(path: str, digest: str) -> None:
    """Persist a stat-bound digest so compressed launches do not re-read huge caches."""
    stat = os.stat(path)
    payload = {
        "format": "krasis_marlin_sha256_cache",
        "format_version": 1,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }
    cache_path = _marlin_digest_cache_path(path)
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, cache_path)


def _sha256_jsonable(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _krasis_runtime_code_hash() -> dict[str, Any]:
    """Hash first-party files that affect heatmap routing/validation behavior."""
    package_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        "pyproject.toml",
        "Cargo.toml",
        "build.rs",
        "python/krasis/__init__.py",
        "python/krasis/server.py",
        "python/krasis/benchmark.py",
        "python/krasis/model.py",
        "python/krasis/config.py",
        "python/krasis/tokenizer.py",
        "src/server.rs",
        "src/gpu_decode.rs",
        "src/gpu_prefill.rs",
        "src/cuda/prefill_kernels.cu",
    ]
    h = hashlib.sha256()
    hashed_files: list[str] = []
    for rel in candidates:
        path = repo_root / rel
        if not path.is_file() and rel.startswith("python/krasis/"):
            path = package_dir / rel.removeprefix("python/krasis/")
        if not path.is_file():
            continue
        hashed_files.append(rel)
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return {"sha256": h.hexdigest(), "files": hashed_files}


def _model_config_fingerprints(model_path: str) -> dict[str, Optional[str]]:
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    return {name: _sha256_file(os.path.join(model_path, name)) for name in names}


def _heatmap_route_signature_from_cfg(cfg, args) -> dict[str, Any]:
    """Return the routing identity an approved heatmap must match.

    Attention precision and KV-cache format are intentionally excluded: they can
    be listed as validated-compatible runtimes, but they are not part of the
    route-prior identity unless measurements prove they must be split.
    """
    layer_types = getattr(cfg, "layer_types", None)
    explicit_layer_types = [] if layer_types is None else list(layer_types)
    return {
        "model": {
            "model_name": os.path.basename(os.path.abspath(args.model_path)),
            "model_type": cfg.model_type,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_moe_layers": cfg.num_moe_layers,
            "n_routed_experts": cfg.n_routed_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "num_full_attention_layers": cfg.num_full_attention_layers,
            "config_fingerprints": _model_config_fingerprints(args.model_path),
        },
        "routing": {
            "scoring_func": getattr(cfg, "scoring_func", None),
            "routed_scaling_factor": getattr(cfg, "routed_scaling_factor", None),
            "norm_topk_prob": bool(getattr(cfg, "norm_topk_prob", False)),
            "use_moe_router_bias": bool(getattr(cfg, "use_moe_router_bias", False)),
            "need_fp32_gate": bool(getattr(cfg, "need_fp32_gate", False)),
            "norm_bias_one": bool(getattr(cfg, "norm_bias_one", False)),
            "layer_types_sha256": _sha256_jsonable(explicit_layer_types),
        },
        "schema": {
            "format": APPROVED_HEATMAP_FORMAT,
            "format_version": APPROVED_HEATMAP_FORMAT_VERSION,
        },
    }


def _heatmap_route_signature(model: KrasisModel, args) -> dict[str, Any]:
    return _heatmap_route_signature_from_cfg(model.cfg, args)


def _runtime_heatmap_capture_config(args) -> dict[str, Any]:
    sidecar_manifest = None
    if args.hqq_sidecar_manifest:
        manifest_path = os.path.expanduser(args.hqq_sidecar_manifest)
        sidecar_manifest = {
            "sha256": _sha256_file(manifest_path),
            "basename": os.path.basename(manifest_path),
        }
    config = {
        "gpu_expert_bits": int(args.gpu_expert_bits),
        "expert_group_size": int(args.expert_group_size),
        "gpu_expert_int4_calib": args.gpu_expert_int4_calib,
        "cpu_expert_bits": int(args.cpu_expert_bits),
        "attention_quant": args.attention_quant,
        "hqq_cache_profile": args.hqq_cache_profile,
        "hqq_group_size": int(args.hqq_group_size),
        "hqq_auto_budget_pct": args.hqq_auto_budget_pct,
        "hqq46_auto_budget_mib": args.hqq46_auto_budget_mib,
        "hqq_sidecar_manifest": sidecar_manifest,
        "shared_expert_quant": args.shared_expert_quant,
        "dense_mlp_quant": args.dense_mlp_quant,
        "lm_head_quant": args.lm_head_quant,
        "kv_dtype": args.kv_dtype,
        "layer_group_size": int(args.layer_group_size),
    }
    if int(args.max_context_tokens) > 0:
        config["max_context_tokens"] = int(args.max_context_tokens)
    return config


def _runtime_matches_approved_heatmap_policy(
    current: dict[str, Any],
    validated_runtimes: Any,
    policy: Any,
) -> tuple[bool, str]:
    if not isinstance(current, dict):
        return False, "current runtime compatibility metadata is not an object"
    if isinstance(validated_runtimes, list) and current in validated_runtimes:
        return True, "exact validated runtime"
    if not isinstance(policy, dict):
        return False, "no compatible runtime policy"

    accepted_attention = policy.get("accepted_attention_quants")
    if isinstance(accepted_attention, list):
        current_attention = current.get("attention_quant")
        if current_attention not in accepted_attention:
            return False, f"attention_quant={current_attention!r} is not accepted"

    accepted_kv = policy.get("accepted_kv_dtypes")
    if isinstance(accepted_kv, list):
        current_kv = current.get("kv_dtype")
        if current_kv not in accepted_kv:
            return False, f"kv_dtype={current_kv!r} is not accepted"

    ignored = policy.get("ignored_runtime_fields", [])
    if not isinstance(ignored, list) or not all(isinstance(item, str) for item in ignored):
        return False, "invalid ignored_runtime_fields policy"
    ignored_fields = set(ignored)

    comparable_current = {
        key: value for key, value in current.items() if key not in ignored_fields
    }
    for runtime in validated_runtimes if isinstance(validated_runtimes, list) else []:
        if not isinstance(runtime, dict):
            continue
        comparable_runtime = {
            key: value for key, value in runtime.items() if key not in ignored_fields
        }
        if comparable_current == comparable_runtime:
            return True, str(policy.get("reason") or "manifest-approved compatible runtime policy")

    return False, "no validated runtime matches after applying compatibility policy"


def _load_benchmark_decode_prompt_texts() -> dict[str, str]:
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    prompts: dict[str, str] = {}
    for i in range(1, 100):
        filename = f"decode_prompt_{i}"
        path = os.path.join(prompts_dir, filename)
        if not os.path.isfile(path):
            break
        with open(path) as f:
            prompts[filename] = f.read().strip()
    return prompts


def _assert_heatmap_prompts_are_held_out(prompts: list[str]) -> None:
    benchmark_prompts = _load_benchmark_decode_prompt_texts()
    normalized_benchmark = {
        " ".join(text.split()): name for name, text in benchmark_prompts.items()
    }
    for idx, prompt in enumerate(prompts, start=1):
        key = " ".join(prompt.split())
        if key in normalized_benchmark:
            raise RuntimeError(
                "Heatmap prompt set overlaps the benchmark decode prompt set: "
                f"heatmap prompt {idx} exactly matches {normalized_benchmark[key]}. "
                "Use held-out heatmap prompts so benchmark decode remains an unseen workload."
            )


def _heatmap_decode_params(args) -> dict[str, Any]:
    benchmark_mode = bool(getattr(args, "benchmark", False) or getattr(args, "benchmark_only", False))
    return {
        "mode": "benchmark" if benchmark_mode else "server-default",
        "temperature": 0.0 if benchmark_mode else float(args.temperature),
        "top_k": HEATMAP_DEFAULT_TOP_K,
        "top_p": HEATMAP_DEFAULT_TOP_P,
        "presence_penalty": HEATMAP_DEFAULT_PRESENCE_PENALTY,
        "enable_thinking": False if benchmark_mode else bool(args.enable_thinking),
        "decode_tokens_per_prompt": HEATMAP_DECODE_TOKENS,
    }


def _expected_heatmap_metadata(model: KrasisModel, args, prompts: list[str]) -> dict[str, Any]:
    from krasis import __version__ as krasis_version

    prompt_file = os.path.join(os.path.dirname(__file__), "prompts", "heatmap_prompts.txt")
    cfg = model.cfg
    quant_cfg = getattr(model, "quant_cfg", None)
    try:
        import torch
        resolved_num_gpus = int(args.num_gpus or torch.cuda.device_count())
    except Exception:
        resolved_num_gpus = int(args.num_gpus or 0)
    runtime_metadata = {
        "num_gpus": resolved_num_gpus,
        "selected_gpus": args.selected_gpus or "",
        "gpu_expert_bits": int(args.gpu_expert_bits),
        "expert_group_size": int(args.expert_group_size),
        "gpu_expert_int4_calib": args.gpu_expert_int4_calib,
        "cpu_expert_bits": int(args.cpu_expert_bits),
        "attention_quant": args.attention_quant,
        "hqq_cache_profile": args.hqq_cache_profile,
        "hqq_group_size": int(args.hqq_group_size),
        "hqq_auto_budget_pct": args.hqq_auto_budget_pct,
        "hqq46_auto_budget_mib": args.hqq46_auto_budget_mib,
        "hqq_sidecar_manifest": os.path.abspath(args.hqq_sidecar_manifest) if args.hqq_sidecar_manifest else None,
        "shared_expert_quant": args.shared_expert_quant,
        "dense_mlp_quant": args.dense_mlp_quant,
        "lm_head_quant": args.lm_head_quant,
        "kv_dtype": args.kv_dtype,
        "kv_cache_mb": int(args.kv_cache_mb),
        "layer_group_size": int(args.layer_group_size),
        "multi_gpu_hcs": bool(args.multi_gpu_hcs),
        "multi_gpu_mode": str(args.multi_gpu_mode),
        "hcs": bool(args.hcs),
        "quant_config": getattr(quant_cfg, "__dict__", {}) if quant_cfg is not None else {},
    }
    if int(args.max_context_tokens) > 0:
        runtime_metadata["max_context_tokens"] = int(args.max_context_tokens)

    metadata = {
        "format": HEATMAP_FORMAT,
        "format_version": HEATMAP_FORMAT_VERSION,
        "krasis": {
            "version": krasis_version,
            "runtime_code": _krasis_runtime_code_hash(),
        },
        "model": {
            "model_path": os.path.abspath(args.model_path),
            "model_name": os.path.basename(os.path.abspath(args.model_path)),
            "model_type": cfg.model_type,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_moe_layers": cfg.num_moe_layers,
            "n_routed_experts": cfg.n_routed_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "num_full_attention_layers": cfg.num_full_attention_layers,
            "config_fingerprints": _model_config_fingerprints(args.model_path),
        },
        "route_signature": _heatmap_route_signature(model, args),
        "runtime": runtime_metadata,
        "runtime_compat": _runtime_heatmap_capture_config(args),
        "heatmap_build": {
            "prompt_source": "python/krasis/prompts/heatmap_prompts.txt",
            "prompt_file_sha256": _sha256_file(prompt_file),
            "prompt_count": len(prompts),
            "prompt_set_sha256": _sha256_jsonable(prompts),
            "benchmark_prompt_overlap": False,
            "decode_params": _heatmap_decode_params(args),
        },
    }
    return metadata


def _metadata_mismatches(expected: Any, actual: Any, path: str = "") -> list[str]:
    mismatches: list[str] = []
    label = path or "metadata"
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{label}: expected object, found {type(actual).__name__}"]
        for key in sorted(expected):
            child_path = f"{label}.{key}" if path else key
            if key not in actual:
                mismatches.append(f"{child_path}: missing")
            else:
                mismatches.extend(_metadata_mismatches(expected[key], actual[key], child_path))
        for key in sorted(set(actual) - set(expected)):
            if key == "generated_at_utc":
                continue
            child_path = f"{label}.{key}" if path else key
            if child_path == "heatmap_build.total_decode_tokens":
                measured = actual[key]
                if (
                    isinstance(measured, bool)
                    or not isinstance(measured, int)
                    or measured <= 0
                ):
                    mismatches.append(
                        f"{child_path}: expected a positive measured integer, "
                        f"found {measured!r}"
                    )
                continue
            mismatches.append(f"{child_path}: unexpected")
        return mismatches
    if expected != actual:
        mismatches.append(f"{label}: expected {expected!r}, found {actual!r}")
    return mismatches


def _load_validated_heatmap(heatmap_path: str, expected_metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        with open(heatmap_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Heatmap file is not valid JSON: {heatmap_path}: {e}") from e
    except OSError as e:
        raise RuntimeError(f"Heatmap file is unreadable: {heatmap_path}: {e}") from e

    meta = data.get("_metadata")
    if not isinstance(meta, dict):
        raise RuntimeError(
            "Refusing to use heatmap without validation metadata: "
            f"{heatmap_path}. Rebuild it with the current Krasis server so the "
            "runtime params, heatmap params, prompt hash, and Krasis version can be verified."
        )
    fmt = meta.get("format")
    if fmt == APPROVED_HEATMAP_FORMAT:
        if meta.get("format_version") != APPROVED_HEATMAP_FORMAT_VERSION:
            raise RuntimeError(
                "Refusing approved heatmap with unsupported format version: "
                f"{meta.get('format_version')!r} in {heatmap_path}"
            )
        route_mismatches = _metadata_mismatches(
            expected_metadata.get("route_signature"),
            meta.get("route_signature"),
            "route_signature",
        )
        if route_mismatches:
            sample = "\n  - ".join(route_mismatches[:20])
            extra = "" if len(route_mismatches) <= 20 else f"\n  - ... {len(route_mismatches) - 20} more"
            raise RuntimeError(
                "Refusing approved heatmap because the model/router route signature does not match. "
                f"Path: {heatmap_path}\n"
                f"Mismatches:\n  - {sample}{extra}"
            )
        current_compat = expected_metadata.get("runtime_compat")
        compatible = meta.get("validated_compatible_runtimes", [])
        policy = expected_metadata.get("runtime_compat_policy")
        compatible_ok, compatible_reason = _runtime_matches_approved_heatmap_policy(
            current_compat,
            compatible,
            policy,
        )
        if not compatible_ok:
            raise RuntimeError(
                "Refusing approved heatmap because this HQQ/KV runtime has not been validated "
                "for the artifact. Build or approve a compatible heatmap first.\n"
                f"Path: {heatmap_path}\n"
                f"Current runtime: {current_compat!r}\n"
                f"Validated runtimes: {compatible!r}\n"
                f"Compatibility policy: {policy!r}\n"
                f"Reason: {compatible_reason}"
            )
        if compatible_reason != "exact validated runtime":
            logger.info(
                "APPROVED_HEATMAP runtime compatibility policy accepted artifact=%s reason=%s "
                "current_attention=%s current_kv=%s captured_runtime=%s",
                heatmap_path,
                compatible_reason,
                current_compat.get("attention_quant") if isinstance(current_compat, dict) else None,
                current_compat.get("kv_dtype") if isinstance(current_compat, dict) else None,
                compatible[0] if isinstance(compatible, list) and compatible else None,
            )
        return data
    if fmt != HEATMAP_FORMAT:
        raise RuntimeError(
            f"Refusing heatmap with unsupported format {fmt!r}: {heatmap_path}"
        )
    mismatches = _metadata_mismatches(expected_metadata, meta)
    if mismatches:
        sample = "\n  - ".join(mismatches[:20])
        extra = "" if len(mismatches) <= 20 else f"\n  - ... {len(mismatches) - 20} more"
        raise RuntimeError(
            "Refusing to use heatmap because it was not built for this exact runtime. "
            f"Path: {heatmap_path}\n"
            f"Mismatches:\n  - {sample}{extra}\n"
            "Rebuild the heatmap without --heatmap-path, or provide a heatmap built "
            "with the same Krasis version, runtime config, heatmap build params, "
            "and heatmap prompt set."
        )
    return data


def _download_url_bytes(url: str, *, timeout_s: float) -> tuple[Optional[bytes], str]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "krasis-approved-heatmap/1",
            "Accept": "application/json,*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return response.read(), ""
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, "not found"
        return None, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except OSError as e:
        return None, str(e)


def _approved_heatmap_download_timeout_s() -> float:
    raw = os.environ.get("KRASIS_APPROVED_HEATMAP_TIMEOUT_S", "").strip()
    if not raw:
        return 5.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        _warn(f"Ignoring invalid KRASIS_APPROVED_HEATMAP_TIMEOUT_S={raw!r}; using 5.0")
        return 5.0


def _load_approved_heatmap_manifest(manifest_url: str) -> tuple[Optional[dict[str, Any]], str]:
    payload, error = _download_url_bytes(
        manifest_url,
        timeout_s=_approved_heatmap_download_timeout_s(),
    )
    if payload is None:
        return None, error
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(manifest, dict):
        return None, "manifest root is not an object"
    if manifest.get("format") != APPROVED_HEATMAP_MANIFEST_FORMAT:
        return None, f"unsupported format {manifest.get('format')!r}"
    if manifest.get("format_version") != APPROVED_HEATMAP_MANIFEST_FORMAT_VERSION:
        return None, f"unsupported format_version {manifest.get('format_version')!r}"
    if not isinstance(manifest.get("artifacts"), list):
        return None, "manifest artifacts field is not a list"
    return manifest, ""


def _select_approved_heatmap_manifest_entry(
    manifest: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> Optional[dict[str, Any]]:
    route_hash = _sha256_jsonable(expected_metadata.get("route_signature"))
    runtime_hash = _sha256_jsonable(expected_metadata.get("runtime_compat"))
    current_compat = expected_metadata.get("runtime_compat")
    candidates = []
    for entry in manifest.get("artifacts", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "approved":
            continue
        if entry.get("route_signature_sha256") != route_hash:
            continue
        runtime_hashes = entry.get("validated_runtime_sha256s", [])
        runtime_matches = isinstance(runtime_hashes, list) and runtime_hash in runtime_hashes
        if not runtime_matches:
            runtime_matches, _ = _runtime_matches_approved_heatmap_policy(
                current_compat,
                entry.get("validated_compatible_runtimes", []),
                entry.get("runtime_compatibility"),
            )
        if not runtime_matches:
            continue
        candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            int(item.get("priority", 1000)),
            str(item.get("artifact_id", "")),
        )
    )
    return candidates[0]


def _approved_heatmap_cache_path(cache_dir: str, entry: dict[str, Any]) -> str:
    artifact_id = str(entry.get("artifact_id") or "approved_heatmap")
    filename = os.path.basename(str(entry.get("filename") or entry.get("download_url") or "heatmap.json"))
    sha = str(entry.get("sha256") or "")
    if not sha:
        raise RuntimeError(f"Approved heatmap manifest entry is missing sha256: {entry!r}")
    safe_artifact_id = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in artifact_id)
    safe_filename = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in filename)
    return os.path.join(
        cache_dir,
        "approved_heatmaps",
        f"{safe_artifact_id}.{sha[:16]}.{safe_filename}",
    )


def _verified_cached_approved_heatmap(cache_dir: str, entry: dict[str, Any]) -> str:
    download_url = str(entry.get("download_url") or "").strip()
    expected_sha = str(entry.get("sha256") or "").strip().lower()
    expected_bytes = entry.get("bytes")
    if not download_url:
        raise ApprovedHeatmapDownloadUnavailable(
            f"manifest entry is missing download_url: {entry.get('artifact_id', '<unknown>')}"
        )
    if len(expected_sha) != 64:
        raise RuntimeError(f"Approved heatmap manifest entry has invalid sha256: {expected_sha!r}")

    cache_path = _approved_heatmap_cache_path(cache_dir, entry)
    if os.path.isfile(cache_path) and _sha256_file(cache_path) == expected_sha:
        return cache_path

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload, error = _download_url_bytes(
        download_url,
        timeout_s=_approved_heatmap_download_timeout_s(),
    )
    if payload is None:
        raise ApprovedHeatmapDownloadUnavailable(
            "Approved heatmap was listed in the manifest but could not be downloaded. "
            f"URL: {download_url}; error: {error}"
        )
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            "Approved heatmap download failed checksum verification. "
            f"URL: {download_url}; expected {expected_sha}, got {actual_sha}"
        )
    if expected_bytes is not None and int(expected_bytes) != len(payload):
        raise RuntimeError(
            "Approved heatmap download failed size verification. "
            f"URL: {download_url}; expected {expected_bytes} bytes, got {len(payload)}"
        )
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(payload)
    os.replace(tmp_path, cache_path)
    return cache_path


def _try_load_auto_approved_heatmap(
    cache_dir: str,
    expected_metadata: dict[str, Any],
    args,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    mode = str(args.approved_heatmap_mode or "auto")
    if mode == "off":
        _dim("Approved route heatmap lookup disabled; falling back to quick startup heatmap generation")
        return None, None
    manifest_url = str(args.approved_heatmap_manifest_url or "").strip()
    if not manifest_url:
        if mode == "require":
            raise RuntimeError("--approved-heatmap-mode=require needs a manifest URL")
        _dim("No approved heatmap manifest URL configured; falling back to quick startup heatmap generation")
        return None, None

    manifest, error = _load_approved_heatmap_manifest(manifest_url)
    if manifest is None:
        message = f"Approved heatmap manifest unavailable from {manifest_url}: {error}"
        if mode == "require":
            raise RuntimeError(message)
        _warn(f"{message}; falling back to quick startup heatmap generation")
        return None, None

    entry = _select_approved_heatmap_manifest_entry(manifest, expected_metadata)
    if entry is None:
        message = "No approved heatmap artifact matches this model/router signature and validated runtime"
        if mode == "require":
            raise RuntimeError(f"{message}; manifest={manifest_url}")
        _dim(f"{message}; falling back to quick startup heatmap generation")
        return None, None

    try:
        heatmap_path = _verified_cached_approved_heatmap(cache_dir, entry)
    except ApprovedHeatmapDownloadUnavailable as e:
        message = f"Approved heatmap artifact unavailable: {e}"
        if mode == "require":
            raise RuntimeError(message) from e
        _warn(f"{message}; falling back to quick startup heatmap generation")
        return None, None
    validation_metadata = dict(expected_metadata)
    if entry.get("runtime_compatibility"):
        validation_metadata["runtime_compat_policy"] = entry.get("runtime_compatibility")
    validated = _load_validated_heatmap(heatmap_path, validation_metadata)
    policy_note = ""
    if entry.get("runtime_compatibility"):
        policy_note = " (manifest-approved runtime compatibility)"
    _detail(
        "Approved route heatmap loaded from cache: "
        f"{entry.get('artifact_id', os.path.basename(heatmap_path))}{policy_note}"
    )
    return heatmap_path, validated


def _load_heatmap_prompts(path: Optional[str] = None) -> list[str]:
    """Load heatmap calibration prompts from the prompts directory.

    Returns a list of prompt strings.  Users can edit heatmap_prompts.txt to
    match their typical workload.
    """
    if path is None:
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        path = os.path.join(prompts_dir, "heatmap_prompts.txt")
    else:
        path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Heatmap prompt file not found: {path}\n"
            "This file is required for HCS calibration.  "
            "See python/krasis/prompts/heatmap_prompts.txt in the repo."
        )
    prompts = []
    current = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped == "" and current:
                prompts.append(" ".join(current))
                current = []
            elif stripped:
                current.append(stripped)
    if current:
        prompts.append(" ".join(current))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _load_prompt_file(filename: str) -> str:
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    path = os.path.join(prompts_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        return f.read().strip()


def _discover_prefill_prompt_files() -> list[str]:
    files = []
    for i in range(1, 100):
        filename = f"prefill_prompt_{i}"
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        if os.path.isfile(os.path.join(prompts_dir, filename)):
            files.append(filename)
        else:
            break
    if not files:
        raise FileNotFoundError(
            "No prefill prompt files found. Expected prefill_prompt_1, prefill_prompt_2, etc."
        )
    return files


def _truncate_content_to_prompt_tokens(model: KrasisModel, content: str, max_tokens: int) -> tuple[str, list[int]]:
    tokens = _chat_prompt_tokens(model, content)
    if len(tokens) <= max_tokens:
        return content, tokens

    lo, hi = 0, len(content)
    best_content = ""
    best_tokens: list[int] = []
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = content[:mid]
        trial_tokens = _chat_prompt_tokens(model, trial)
        if len(trial_tokens) <= max_tokens:
            best_content = trial
            best_tokens = trial_tokens
            lo = mid + 1
        else:
            hi = mid - 1
    return best_content, best_tokens


def _kv_cache_max_tokens(model: KrasisModel) -> int:
    for cache in getattr(model, "kv_caches", []):
        if cache is not None:
            return cache.max_pages * cache.page_size
    return model.cfg.max_position_embeddings


def _make_startup_calibration_prompts(model: KrasisModel, lengths: list[int]) -> list[list[int]]:
    files = _discover_prefill_prompt_files()
    kv_limit = max(1, min(_kv_cache_max_tokens(model), model.cfg.max_position_embeddings) - 100)
    prompts: list[list[int]] = []
    for i, target in enumerate(lengths):
        content = _load_prompt_file(files[i % len(files)])
        _, tokens = _truncate_content_to_prompt_tokens(model, content, min(target, kv_limit))
        if not tokens:
            raise RuntimeError(f"Startup calibration prompt {files[i % len(files)]} produced no tokens")
        prompts.append(tokens)
    return prompts


def _startup_calibration_long_floor_mb(safety_margin_mb: int) -> int:
    # Startup calibration runs before HCS is loaded, so there is nothing to evict
    # if a long prefill transient underestimates VRAM. Keep an extra measured
    # safety margin during calibration; runtime HCS budgets still use the
    # configured margin after calibration completes.
    return max(1, safety_margin_mb * 2)


def _require_startup_vram_floor(
    label: str,
    min_free_mb: int,
    safety_margin_mb: int,
) -> None:
    if min_free_mb < safety_margin_mb:
        raise RuntimeError(
            f"{label} breached the configured VRAM safety floor: "
            f"min_free={min_free_mb} MB safety={safety_margin_mb} MB"
        )


def _startup_stage_exact_kv_mb_per_token(model: KrasisModel, kv_dtype: str) -> float:
    """Return compact-KV stage-exact prefill temp growth in MB/token."""
    if kv_dtype not in ("k6v6", "k4v4"):
        return 0.0
    cfg = model.cfg
    if not getattr(cfg, "is_gqa", False):
        return 0.0

    active_layers = int(getattr(cfg, "num_full_attention_layers", cfg.num_hidden_layers))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", 0))
    head_dim = int(getattr(cfg, "gqa_head_dim", None) or getattr(cfg, "head_dim", 0) or 0)
    if active_layers <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        return 0.0

    bytes_per_token = 2 * active_layers * num_kv_heads * head_dim
    return bytes_per_token / (1024.0 * 1024.0)


def _startup_calibration_estimated_prefill_mb_per_token(
    model: KrasisModel,
    gpu_store,
    kv_dtype: str,
    short_tokens: int,
    long_tokens: int,
) -> float:
    """Estimate long-prefill preparation growth from runtime model dimensions."""
    scratch_mb_per_token = 0.0
    if long_tokens > short_tokens and hasattr(gpu_store, "prefill_scratch_reservation_mb"):
        short_scratch_mb = int(gpu_store.prefill_scratch_reservation_mb(short_tokens))
        long_scratch_mb = int(gpu_store.prefill_scratch_reservation_mb(long_tokens))
        if long_scratch_mb > short_scratch_mb:
            scratch_mb_per_token = (long_scratch_mb - short_scratch_mb) / (long_tokens - short_tokens)

    return scratch_mb_per_token + _startup_stage_exact_kv_mb_per_token(model, kv_dtype)


def _project_startup_calibration_probe_target(
    *,
    current_tokens: int,
    current_min_free_mb: int,
    short_tokens: int,
    default_long_tokens: int,
    target_floor_mb: int,
    mb_per_token: float,
) -> tuple[Optional[int], str]:
    """Project a safe next calibration probe using a measured or model-derived slope."""
    if mb_per_token <= 0.0:
        return None, "no usable MB/token estimate"

    # Aim the next probe at one full calibration floor above the stop floor.
    # If the estimate is right, the probe validates useful long-context behavior
    # without running near the hard-exit floor during startup when HCS cannot evict.
    validation_floor_mb = target_floor_mb * 2
    if current_min_free_mb <= validation_floor_mb:
        return None, (
            f"current low-water {current_min_free_mb} MB leaves no validation reserve "
            f"above {validation_floor_mb} MB"
        )

    projected_extra_tokens = int((current_min_free_mb - validation_floor_mb) / mb_per_token)
    min_useful_extra = max(1, short_tokens // 2)
    if projected_extra_tokens <= min_useful_extra:
        return None, (
            f"projected next span {projected_extra_tokens} tokens is too close to "
            f"validation floor {validation_floor_mb} MB"
        )

    candidate = min(default_long_tokens, current_tokens + projected_extra_tokens)
    if candidate <= current_tokens:
        return None, "projected candidate would not grow"
    return candidate, (
        f"projected {mb_per_token:.4f} MB/token, "
        f"validation floor {validation_floor_mb} MB"
    )


def _next_startup_calibration_probe_target(
    *,
    short_tokens: int,
    default_long_tokens: int,
    observed_prefill_mins: list[tuple[int, int]],
    target_floor_mb: int,
    estimated_prefill_mb_per_token: float = 0.0,
    fail_closed_probe_tokens: Optional[int] = None,
    runtime_safety_floor_mb: Optional[int] = None,
) -> tuple[Optional[int], str]:
    """Choose the next startup long-calibration probe from observed low-water data."""
    if not observed_prefill_mins:
        return None, "no completed probes"

    current_tokens, current_min_free_mb = observed_prefill_mins[-1]
    if current_tokens >= default_long_tokens:
        return None, "default long target reached"
    fail_closed_probe = (
        fail_closed_probe_tokens is not None
        and fail_closed_probe_tokens > current_tokens
        and runtime_safety_floor_mb is not None
        and current_min_free_mb >= runtime_safety_floor_mb
    )
    if current_min_free_mb <= target_floor_mb and not fail_closed_probe:
        return None, (
            f"current low-water {current_min_free_mb} MB is at/below "
            f"adaptive floor {target_floor_mb} MB"
        )

    first_long = max(
        current_tokens + 1,
        short_tokens * max(1, STARTUP_CALIBRATION_LONG_INITIAL_MULTIPLIER),
    )
    if fail_closed_probe_tokens is not None:
        first_long = min(first_long, fail_closed_probe_tokens)
    raw_candidate = min(default_long_tokens, max(current_tokens * 2, first_long))
    if fail_closed_probe_tokens is not None:
        raw_candidate = min(raw_candidate, fail_closed_probe_tokens)
    if len(observed_prefill_mins) < 2:
        projected, reason = _project_startup_calibration_probe_target(
            current_tokens=current_tokens,
            current_min_free_mb=current_min_free_mb,
            short_tokens=short_tokens,
            default_long_tokens=default_long_tokens,
            target_floor_mb=target_floor_mb,
            mb_per_token=estimated_prefill_mb_per_token,
        )
        if projected is not None:
            return projected, f"model-estimated initial probe ({reason})"
        if fail_closed_probe and raw_candidate > current_tokens:
            return raw_candidate, (
                "runtime-derived fail-closed probe "
                f"(entry floor validated through {fail_closed_probe_tokens} tokens; "
                f"projection declined: {reason})"
            )
        if estimated_prefill_mb_per_token > 0.0:
            return None, reason

        initial_headroom_mb = current_min_free_mb - target_floor_mb
        if initial_headroom_mb <= max(1, target_floor_mb // 2):
            return None, (
                f"short-probe headroom {initial_headroom_mb} MB is too close to "
                f"adaptive floor {target_floor_mb} MB"
            )
        if raw_candidate <= current_tokens:
            return None, "no longer candidate above current probe"
        return raw_candidate, f"initial adaptive long probe ({reason})"

    worst_slope_mb_per_token = 0.0
    for (left_tokens, left_min), (right_tokens, right_min) in zip(
        observed_prefill_mins,
        observed_prefill_mins[1:],
    ):
        token_span = right_tokens - left_tokens
        min_drop_mb = left_min - right_min
        if token_span > 0 and min_drop_mb > 0:
            worst_slope_mb_per_token = max(
                worst_slope_mb_per_token,
                min_drop_mb / token_span,
            )

    slope_mb_per_token = max(worst_slope_mb_per_token, estimated_prefill_mb_per_token)
    if slope_mb_per_token <= 0.0:
        if raw_candidate <= current_tokens:
            return None, "no longer candidate above current probe"
        return raw_candidate, "no observed low-water decline yet"

    candidate, reason = _project_startup_calibration_probe_target(
        current_tokens=current_tokens,
        current_min_free_mb=current_min_free_mb,
        short_tokens=short_tokens,
        default_long_tokens=min(raw_candidate, default_long_tokens),
        target_floor_mb=target_floor_mb,
        mb_per_token=slope_mb_per_token,
    )
    if candidate is None:
        if fail_closed_probe and raw_candidate > current_tokens:
            return raw_candidate, (
                "runtime-derived fail-closed probe "
                f"(entry floor validated through {fail_closed_probe_tokens} tokens; "
                f"projection declined: {reason})"
            )
        return None, reason
    return candidate, (
        f"worst observed slope {worst_slope_mb_per_token:.4f} MB/token, "
        f"model estimate {estimated_prefill_mb_per_token:.4f} MB/token, {reason}"
    )


# Decode tokens per heatmap prompt.  This is deliberately high relative to the
# short prompt length so that the heatmap is dominated by decode routing, which
# is where HCS cache effectiveness matters.
HEATMAP_DECODE_TOKENS = 256


def _chat_prompt_tokens(
    model: KrasisModel,
    prompt_text: str,
    enable_thinking: Optional[bool] = None,
) -> list[int]:
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        **kwargs,
    )


def _default_stop_ids(model: KrasisModel) -> list[int]:
    return [model.cfg.eos_token_id] + list(model.cfg.extra_stop_token_ids)


def _build_heatmap(model: KrasisModel, save_path: str, args) -> str:
    """Build expert activation heatmap by running decode-heavy inference.

    Loads diverse short prompts from heatmap_prompts.txt, runs each with a
    long decode window (256 tokens), so the resulting heatmap reflects decode
    expert routing rather than prefill routing.  This runs on every startup
    to keep the heatmap current with the model and reference data.

    Uses the Rust decode engine's built-in heatmap collection — no
    GpuPrefillManager required.
    """
    import os, json

    prompts = _load_heatmap_prompts()
    _assert_heatmap_prompts_are_held_out(prompts)
    heatmap_metadata = _expected_heatmap_metadata(model, args, prompts)
    decode_params = heatmap_metadata["heatmap_build"]["decode_params"]
    timing_enabled = _heatmap_substage_timing_enabled()
    heatmap_t0 = time.perf_counter()

    gpu_store = getattr(model, '_gpu_decode_store', None)
    if gpu_store is None:
        raise RuntimeError("GPU decode store not configured — cannot build heatmap")

    cfg = model.cfg
    num_layers = cfg.num_hidden_layers
    num_experts = cfg.n_routed_experts

    heatmap_collection_started = False
    try:
        # Init lightweight HCS state for collection only (no VRAM allocation)
        collection_t0 = time.perf_counter()
        gpu_store.hcs_init_collection(num_layers, num_experts)
        gpu_store.hcs_start_collecting()
        heatmap_collection_started = True
        if timing_enabled:
            logger.info(
                "HEATMAP_TIMING collection_start prompts=%d layers=%d experts=%d collection_start_s=%.6f",
                len(prompts),
                num_layers,
                num_experts,
                time.perf_counter() - collection_t0,
            )

        # Run each prompt with long decode to build a decode-weighted heatmap
        total_decode_tokens = 0
        logger.info(
            "Building heatmap from %d held-out prompts (%d decode tokens each, "
            "temperature=%.3f top_k=%d top_p=%.3f enable_thinking=%s mode=%s)...",
            len(prompts),
            HEATMAP_DECODE_TOKENS,
            decode_params["temperature"],
            decode_params["top_k"],
            decode_params["top_p"],
            decode_params["enable_thinking"],
            decode_params["mode"],
        )
        stop_ids = _default_stop_ids(model)
        for i, prompt_text in enumerate(prompts):
            prompt_index = i + 1
            prompt_t0 = time.perf_counter()
            tokenize_t0 = time.perf_counter()
            tokens = _chat_prompt_tokens(
                model,
                prompt_text,
                enable_thinking=decode_params["enable_thinking"],
            )
            tokenize_s = time.perf_counter() - tokenize_t0
            _dim(
                f"Quick heatmap prompt {prompt_index}/{len(prompts)}: "
                f"{len(tokens):,} prefill tokens + {HEATMAP_DECODE_TOKENS:,} decode tokens"
            )
            logger.info(
                "HEATMAP prompt_start index=%d total=%d prompt_tokens=%d decode_tokens=%d "
                "tokenize_s=%.6f",
                prompt_index,
                len(prompts),
                len(tokens),
                HEATMAP_DECODE_TOKENS,
                tokenize_s,
            )
            prefill_t0 = time.perf_counter()
            first_token, prompt_len, kv_overflow = gpu_store.rust_prefill_tokens(
                tokens,
                temperature=decode_params["temperature"],
                disable_pinning=True,
            )
            prefill_s = time.perf_counter() - prefill_t0
            logger.info(
                "HEATMAP prompt_prefill_done index=%d total=%d prompt_len=%d first_token=%d "
                "kv_overflow=%s prefill_s=%.3f",
                prompt_index,
                len(prompts),
                prompt_len,
                first_token,
                bool(kv_overflow),
                prefill_s,
            )
            decode_s = 0.0
            generated_tokens = 0
            stopped_before_decode = bool(kv_overflow or first_token in stop_ids)
            if not kv_overflow and first_token not in stop_ids:
                logger.info(
                    "HEATMAP prompt_decode_start index=%d total=%d start_position=%d max_tokens=%d",
                    prompt_index,
                    len(prompts),
                    prompt_len,
                    HEATMAP_DECODE_TOKENS,
                )
                decode_t0 = time.perf_counter()
                generated = gpu_store.gpu_generate_batch(
                    first_token=first_token,
                    start_position=prompt_len,
                    max_tokens=HEATMAP_DECODE_TOKENS,
                    temperature=decode_params["temperature"],
                    top_k=decode_params["top_k"],
                    top_p=decode_params["top_p"],
                    stop_ids=stop_ids,
                    presence_penalty=decode_params["presence_penalty"],
                )
                decode_s = time.perf_counter() - decode_t0
                generated_tokens = len(generated)
                total_decode_tokens += 1 + len(generated)
            else:
                total_decode_tokens += 1
            prompt_s = time.perf_counter() - prompt_t0
            completed = prompt_index
            remaining = max(0, len(prompts) - completed)
            elapsed_s = time.perf_counter() - heatmap_t0
            avg_prompt_s = elapsed_s / completed
            eta_s = avg_prompt_s * remaining
            logger.info(
                "HEATMAP prompt_done index=%d total=%d prompt_len=%d first_token=%d "
                "kv_overflow=%s stopped_before_decode=%s generated_tokens=%d "
                "total_decode_tokens=%d tokenize_s=%.6f prefill_s=%.3f decode_s=%.3f "
                "prompt_s=%.3f elapsed_s=%.3f eta_s=%.3f",
                prompt_index,
                len(prompts),
                prompt_len,
                first_token,
                bool(kv_overflow),
                stopped_before_decode,
                generated_tokens,
                total_decode_tokens,
                tokenize_s,
                prefill_s,
                decode_s,
                prompt_s,
                elapsed_s,
                eta_s,
            )
            _dim(
                f"Quick heatmap prompt {prompt_index}/{len(prompts)} done: "
                f"{generated_tokens + 1:,} route tokens, {prompt_s:.1f}s "
                f"(elapsed {elapsed_s:.1f}s, eta {eta_s:.1f}s)"
            )
            if timing_enabled:
                logger.info(
                    "HEATMAP_TIMING prompt index=%d total=%d prompt_tokens=%d first_token=%d "
                    "prompt_len=%d kv_overflow=%s stopped_before_decode=%s generated_tokens=%d "
                    "tokenize_s=%.6f prefill_s=%.6f decode_s=%.6f prompt_s=%.6f",
                    prompt_index,
                    len(prompts),
                    len(tokens),
                    first_token,
                    prompt_len,
                    bool(kv_overflow),
                    stopped_before_decode,
                    generated_tokens,
                    tokenize_s,
                    prefill_s,
                    decode_s,
                    prompt_s,
                )

        logger.info("Heatmap collection complete: %d decode tokens across %d prompts",
                    total_decode_tokens, len(prompts))
        if timing_enabled:
            logger.info(
                "HEATMAP_TIMING prompt_loop prompts=%d decode_tokens=%d prompt_loop_s=%.6f",
                len(prompts),
                total_decode_tokens,
                time.perf_counter() - heatmap_t0,
            )

        # Export and save heatmap before tearing down the collection-only HCS.
        if total_decode_tokens <= 0:
            raise RuntimeError(
                "Quick heatmap collection produced no decode-route tokens"
            )
        export_t0 = time.perf_counter()
        heatmap_dict = gpu_store.hcs_export_heatmap()
        export_s = time.perf_counter() - export_t0
        # This is a measured build result, not part of the expected runtime
        # identity. Peer-plan route counts require the exact denominator.
        heatmap_metadata["heatmap_build"]["total_decode_tokens"] = int(
            total_decode_tokens
        )
        heatmap_metadata["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        heatmap_dict["_metadata"] = heatmap_metadata
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        write_t0 = time.perf_counter()
        with open(save_path, 'w') as f:
            json.dump(heatmap_dict, f, sort_keys=True)
        write_s = time.perf_counter() - write_t0
        logger.info("Heatmap saved to %s (%d entries)", save_path, len(heatmap_dict))
        if timing_enabled:
            logger.info(
                "HEATMAP_TIMING export entries=%d export_s=%.6f write_s=%.6f heatmap_total_pre_cleanup_s=%.6f",
                len(heatmap_dict),
                export_s,
                write_s,
                time.perf_counter() - heatmap_t0,
            )
        return save_path
    finally:
        reset_s = 0.0
        cleanup_s = 0.0
        try:
            if heatmap_collection_started:
                # Tear down collection-only HCS so normal startup can re-init with real budget.
                reset_t0 = time.perf_counter()
                gpu_store.hcs_reset()
                reset_s = time.perf_counter() - reset_t0
        finally:
            # Heatmap collection runs internal prefill/decode before the server is
            # ready. Clean it through the same lifecycle used after real requests so
            # request-scoped KV/recurrent state cannot leak into the first user call.
            cleanup_t0 = time.perf_counter()
            model.server_cleanup()
            cleanup_s = time.perf_counter() - cleanup_t0
            logger.info("Heatmap internal decode cleanup complete")
            if timing_enabled:
                logger.info(
                    "HEATMAP_TIMING cleanup reset_s=%.6f server_cleanup_s=%.6f heatmap_total_s=%.6f",
                    reset_s,
                    cleanup_s,
                    time.perf_counter() - heatmap_t0,
                )


def _approved_heatmap_checkpoint_path(save_path: str, prompts_processed: int) -> str:
    base = Path(save_path)
    suffix = base.suffix or ".json"
    stem = base.name[:-len(suffix)] if base.name.endswith(suffix) else base.name
    return str(base.with_name(f"{stem}.p{prompts_processed:05d}{suffix}"))


def _approved_heatmap_metadata(
    model: KrasisModel,
    args,
    prompts: list[str],
    prompt_path: str,
    decode_tokens: int,
    prompts_processed: int,
    total_decode_tokens: int,
    residency_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from krasis import __version__ as krasis_version

    return {
        "format": APPROVED_HEATMAP_FORMAT,
        "format_version": APPROVED_HEATMAP_FORMAT_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "krasis": {
            "version": krasis_version,
            "runtime_code": _krasis_runtime_code_hash(),
        },
        "route_signature": _heatmap_route_signature(model, args),
        "captured_runtime": _runtime_heatmap_capture_config(args),
        "validated_compatible_runtimes": [
            _runtime_heatmap_capture_config(args),
        ],
        "heatmap_build": {
            "prompt_source": os.path.abspath(prompt_path),
            "prompt_file_sha256": _sha256_file(prompt_path),
            "prompt_count_total": len(prompts),
            "prompt_count_processed": prompts_processed,
            "prompt_set_sha256": _sha256_jsonable(prompts),
            "prompt_prefix_sha256": _sha256_jsonable(prompts[:prompts_processed]),
            "benchmark_prompt_overlap": False,
            "decode_params": {
                **_heatmap_decode_params(args),
                "decode_tokens_per_prompt": int(decode_tokens),
            },
            "total_decode_tokens": int(total_decode_tokens),
            "score": "decode_route_topk_count",
            "collection_residency": residency_metadata,
        },
    }


def _write_heatmap_artifact(path: str, heatmap_dict: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(heatmap_dict, f, sort_keys=True)
        f.write("\n")


def _approved_heatmap_counts(data: dict[str, Any], path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in data.items():
        if key == "_metadata":
            continue
        if not isinstance(key, str) or "," not in key:
            raise RuntimeError(f"Approved heatmap contains invalid route key {key!r}: {path}")
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"Approved heatmap contains invalid route count for {key!r}: {path}")
        counts[key] = value
    return counts


def _merge_heatmap_counts(
    cumulative: dict[str, int],
    interval: dict[str, Any],
    source: str,
) -> int:
    merged_events = 0
    for key, value in _approved_heatmap_counts(interval, source).items():
        cumulative[key] = int(cumulative.get(key, 0)) + value
        merged_events += value
    return merged_events


def _full_heatmap_ranking(
    model: KrasisModel,
    counts: dict[str, int],
) -> list[tuple[int, int]]:
    ranked: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for key, count in counts.items():
        layer_text, expert_text = key.split(",", 1)
        layer_idx = int(layer_text)
        expert_idx = int(expert_text)
        if (
            layer_idx < 0
            or layer_idx >= len(model.layers)
            or not model.layers[layer_idx].is_moe
            or expert_idx < 0
            or expert_idx >= model.cfg.n_routed_experts
        ):
            raise RuntimeError(
                f"Heatmap ranking contains an invalid model route {key!r}"
            )
        route = (layer_idx, expert_idx)
        if route in seen:
            raise RuntimeError(f"Heatmap ranking contains duplicate route {key!r}")
        seen.add(route)
        ranked.append((int(count), layer_idx, expert_idx))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    full = [(layer_idx, expert_idx) for _, layer_idx, expert_idx in ranked]
    for layer_idx, layer in enumerate(model.layers):
        if not layer.is_moe:
            continue
        for expert_idx in range(model.cfg.n_routed_experts):
            route = (layer_idx, expert_idx)
            if route not in seen:
                full.append(route)
    return full


def _load_heatmap_residency_bootstrap(
    model: KrasisModel,
    args,
    path: str,
) -> tuple[dict[str, int], str]:
    try:
        with open(path) as f:
            unchecked = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Heatmap residency bootstrap is not valid JSON: {path}: {e}"
        ) from e
    except OSError as e:
        raise RuntimeError(
            f"Heatmap residency bootstrap is unreadable: {path}: {e}"
        ) from e
    metadata = unchecked.get("_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Heatmap residency bootstrap has no validation metadata: {path}"
        )
    fmt = metadata.get("format")
    if fmt == APPROVED_HEATMAP_FORMAT:
        expected = {
            "route_signature": _heatmap_route_signature(model, args),
            "runtime_compat": _runtime_heatmap_capture_config(args),
        }
    else:
        quick_prompts = _load_heatmap_prompts()
        _assert_heatmap_prompts_are_held_out(quick_prompts)
        expected = _expected_heatmap_metadata(model, args, quick_prompts)
    validated = _load_validated_heatmap(path, expected)
    return _approved_heatmap_counts(validated, path), str(fmt)


def _validate_approved_heatmap_resume_base(
    model: KrasisModel,
    args,
    resume_path: str,
    prompts: list[str],
    decode_tokens: int,
) -> tuple[dict[str, int], int, int]:
    expected_metadata = {
        "route_signature": _heatmap_route_signature(model, args),
        "runtime_compat": _runtime_heatmap_capture_config(args),
    }
    base_data = _load_validated_heatmap(resume_path, expected_metadata)
    meta = base_data.get("_metadata", {})
    if meta.get("format") != APPROVED_HEATMAP_FORMAT:
        raise RuntimeError(f"--approved-heatmap-resume-from must point to an approved heatmap artifact: {resume_path}")
    build = meta.get("heatmap_build")
    if not isinstance(build, dict):
        raise RuntimeError(f"Approved heatmap resume artifact is missing heatmap_build metadata: {resume_path}")

    try:
        base_prompt_count = int(build["prompt_count_processed"])
        base_decode_tokens = int(build["total_decode_tokens"])
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"Approved heatmap resume artifact has invalid prompt/token metadata: {resume_path}") from e
    if base_prompt_count <= 0:
        raise RuntimeError(f"Approved heatmap resume artifact has no processed prompts: {resume_path}")
    if base_prompt_count >= len(prompts):
        raise RuntimeError(
            "--approved-heatmap-resume-from has already processed all prompts in the current corpus: "
            f"{base_prompt_count} processed, {len(prompts)} available"
        )

    expected_prefix_sha = _sha256_jsonable(prompts[:base_prompt_count])
    actual_prefix_sha = build.get("prompt_prefix_sha256")
    if actual_prefix_sha != expected_prefix_sha:
        raise RuntimeError(
            "Refusing approved heatmap resume because the current prompt corpus no longer has "
            "the resume artifact's processed prefix. Append new prompts to the same corpus order; "
            "do not reorder or edit previously captured prompts."
        )

    expected_decode_params = {
        **_heatmap_decode_params(args),
        "decode_tokens_per_prompt": int(decode_tokens),
    }
    if build.get("decode_params") != expected_decode_params:
        raise RuntimeError(
            "Refusing approved heatmap resume because decode capture params changed.\n"
            f"Expected/current: {expected_decode_params!r}\n"
            f"Artifact: {build.get('decode_params')!r}"
        )

    return _approved_heatmap_counts(base_data, resume_path), base_prompt_count, base_decode_tokens


def _build_approved_heatmap(
    model: KrasisModel,
    save_path: str,
    args,
    residency_calibration: dict[str, int],
) -> str:
    prompt_path = os.path.expanduser(args.approved_heatmap_prompts) if args.approved_heatmap_prompts else os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "heatmap_prompts.txt",
    )
    prompts = _load_heatmap_prompts(prompt_path)
    _assert_heatmap_prompts_are_held_out(prompts)
    max_prompts = int(args.approved_heatmap_max_prompts or 0)
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise RuntimeError("Approved heatmap build has no prompts after filtering")

    decode_tokens = int(args.approved_heatmap_decode_tokens)
    if decode_tokens <= 0:
        raise RuntimeError("--approved-heatmap-decode-tokens must be positive")
    checkpoint_every = int(args.approved_heatmap_checkpoint_every or 0)
    if checkpoint_every < 0:
        raise RuntimeError("--approved-heatmap-checkpoint-every must be non-negative")
    residency_refresh_every = int(args.approved_heatmap_residency_refresh_every)
    if residency_refresh_every <= 0:
        raise RuntimeError(
            "--approved-heatmap-residency-refresh-every must be positive"
        )
    resume_path = os.path.expanduser(args.approved_heatmap_resume_from) if args.approved_heatmap_resume_from else None
    explicit_bootstrap_path = (
        os.path.expanduser(args.approved_heatmap_bootstrap_from)
        if args.approved_heatmap_bootstrap_from
        else None
    )
    if resume_path and explicit_bootstrap_path:
        raise RuntimeError(
            "--approved-heatmap-resume-from and --approved-heatmap-bootstrap-from "
            "cannot be combined; the validated resume artifact is already the bootstrap"
        )

    gpu_store = getattr(model, "_gpu_decode_store", None)
    if gpu_store is None:
        raise RuntimeError("GPU decode store not configured — cannot build approved heatmap")

    cfg = model.cfg
    stop_ids = _default_stop_ids(model)
    decode_params = _heatmap_decode_params(args)
    timing_enabled = _heatmap_substage_timing_enabled()
    build_t0 = time.perf_counter()
    base_counts: dict[str, int] = {}
    resume_start = 0
    total_decode_tokens = 0
    if resume_path:
        base_counts, resume_start, total_decode_tokens = _validate_approved_heatmap_resume_base(
            model,
            args,
            resume_path,
            prompts,
            decode_tokens,
        )
    bootstrap_path = resume_path or explicit_bootstrap_path or args.heatmap_path
    bootstrap_counts: dict[str, int] = {}
    bootstrap_format = ""
    bootstrap_source = ""
    if resume_path:
        bootstrap_counts = dict(base_counts)
        bootstrap_format = APPROVED_HEATMAP_FORMAT
        bootstrap_source = "resume"
    elif bootstrap_path:
        bootstrap_counts, bootstrap_format = _load_heatmap_residency_bootstrap(
            model,
            args,
            bootstrap_path,
        )
        if not bootstrap_counts:
            raise RuntimeError(
                "Heatmap residency bootstrap contains no route counts: "
                f"{bootstrap_path}"
            )
        bootstrap_source = (
            "explicit"
            if explicit_bootstrap_path
            else "config_heatmap_path"
        )
    residency_metadata = {
        "mode": "adaptive_calibrated_hcs",
        "refresh_every_prompts": residency_refresh_every,
        "cold_start_prompts": 0 if bootstrap_path else 1,
        "refresh_count": 0,
        "bootstrap": (
            {
                "source": bootstrap_source,
                "format": bootstrap_format,
                "basename": os.path.basename(bootstrap_path),
                "sha256": _sha256_file(bootstrap_path),
                "counts_included_in_output": bool(resume_path),
            }
            if bootstrap_path
            else None
        ),
    }
    heatmap_collection_started = False
    residency_active = False
    written_paths: list[str] = []

    try:
        calibration = residency_calibration
        cal_msg = gpu_store.set_vram_calibration(
            calibration["short_tokens"],
            calibration["long_tokens"],
            calibration["prefill_short_free_mb"],
            calibration["prefill_long_free_mb"],
            calibration["decode_short_free_mb"],
            calibration["decode_long_free_mb"],
            calibration["baseline_free_mb"],
            calibration["safety_margin_mb"],
            calibration["short_prefill_post_alloc_free_mb"],
            calibration["long_prefill_post_alloc_free_mb"],
        )
        logger.info("APPROVED_HEATMAP residency_calibration %s", cal_msg)
        if bootstrap_counts:
            ranking = _full_heatmap_ranking(model, bootstrap_counts)
            hcs_result = gpu_store.hcs_pool_init_tiered(
                ranking,
                hard_budget_mb=0,
                soft_budget_mb=calibration["decode_hcs_budget_mb"],
                safety_margin_mb=calibration["safety_margin_mb"],
            )
            evicted, freed_mb, free_mb = gpu_store.py_hcs_drain_vram_pressure(
                "approved_heatmap_bootstrap",
                True,
            )
            if free_mb < calibration["safety_margin_mb"]:
                raise RuntimeError(
                    "Approved heatmap bootstrap HCS could not restore the calibrated "
                    f"VRAM safety floor: free={free_mb} MB, "
                    f"safety={calibration['safety_margin_mb']} MB"
                )
            residency_active = True
            logger.info(
                "APPROVED_HEATMAP residency_bootstrap source=%s path=%s "
                "ranked_counts=%d full_ranking=%d evicted=%d freed_mb=%.1f "
                "free_mb=%d result=%s",
                bootstrap_source,
                bootstrap_path,
                len(bootstrap_counts),
                len(ranking),
                evicted,
                freed_mb,
                free_mb,
                hcs_result,
            )
            _detail(
                f"Initial HCS residency: {bootstrap_source} bootstrap "
                f"({len(bootstrap_counts):,} ranked routes)"
            )
        else:
            gpu_store.hcs_init_collection(
                cfg.num_hidden_layers,
                cfg.n_routed_experts,
            )
            _detail("Initial HCS residency: none; only the first prompt will run cold")
        gpu_store.hcs_start_collecting()
        heatmap_collection_started = True
        _status("Approved HCS route heatmap build")
        _detail(
            f"Prompts: {len(prompts):,} from {prompt_path}; "
            f"decode tokens/prompt: {decode_tokens:,}; checkpoint_every={checkpoint_every}; "
            f"residency_refresh_every={residency_refresh_every}"
        )
        if resume_path:
            _detail(
                f"Resuming from {resume_start:,} prompts and {total_decode_tokens:,} decode-route tokens: "
                f"{resume_path}"
            )
        logger.info(
            "APPROVED_HEATMAP build_start prompts=%d resume_start=%d base_decode_tokens=%d "
            "decode_tokens_per_prompt=%d checkpoint_every=%d residency_refresh_every=%d "
            "bootstrap_source=%s bootstrap_path=%s out=%s prompt_path=%s resume_from=%s",
            len(prompts),
            resume_start,
            total_decode_tokens,
            decode_tokens,
            checkpoint_every,
            residency_refresh_every,
            bootstrap_source,
            bootstrap_path or "",
            save_path,
            prompt_path,
            resume_path or "",
        )

        def export_checkpoint(prompts_processed: int, final: bool) -> str:
            export_t0 = time.perf_counter()
            heatmap_dict = dict(base_counts)
            heatmap_dict["_metadata"] = _approved_heatmap_metadata(
                model,
                args,
                prompts,
                prompt_path,
                decode_tokens,
                prompts_processed,
                total_decode_tokens,
                residency_metadata,
            )
            checkpoint_path = _approved_heatmap_checkpoint_path(save_path, prompts_processed)
            out_paths = [save_path] if final else [checkpoint_path]
            if final and checkpoint_path != save_path:
                out_paths.append(checkpoint_path)
            for out_path in out_paths:
                _write_heatmap_artifact(out_path, heatmap_dict)
                written_paths.append(out_path)
            elapsed = time.perf_counter() - build_t0
            logger.info(
                "APPROVED_HEATMAP checkpoint final=%s prompts=%d resume_start=%d entries=%d "
                "decode_tokens=%d elapsed_s=%.3f export_write_s=%.3f path=%s",
                final,
                prompts_processed,
                resume_start,
                len(heatmap_dict) - 1,
                total_decode_tokens,
                elapsed,
                time.perf_counter() - export_t0,
                ",".join(out_paths),
            )
            _detail(
                f"Checkpoint {'final' if final else prompts_processed}: "
                f"{len(heatmap_dict) - 1:,} ranked experts, "
                f"{total_decode_tokens:,} decode tokens, {elapsed:.1f}s -> {', '.join(out_paths)}"
            )
            return out_paths[0]

        for i, prompt_text in enumerate(prompts[resume_start:], start=resume_start + 1):
            prompt_t0 = time.perf_counter()
            tokens = _chat_prompt_tokens(
                model,
                prompt_text,
                enable_thinking=decode_params["enable_thinking"],
            )
            logger.info(
                "APPROVED_HEATMAP prompt_start index=%d total=%d prompt_tokens=%d",
                i,
                len(prompts),
                len(tokens),
            )
            first_token, prompt_len, kv_overflow = gpu_store.rust_prefill_tokens(
                tokens,
                temperature=decode_params["temperature"],
                disable_pinning=True,
            )
            stopped_before_decode = bool(kv_overflow or first_token in stop_ids)
            reload_count = 0
            reload_ms = 0.0
            if residency_active and not stopped_before_decode:
                reload_count, reload_ms = gpu_store.py_hcs_reload_after_prefill(
                    prompt_len
                )
            generated_tokens = 0
            if not stopped_before_decode:
                generated = gpu_store.gpu_generate_batch(
                    first_token=first_token,
                    start_position=prompt_len,
                    max_tokens=decode_tokens,
                    temperature=decode_params["temperature"],
                    top_k=decode_params["top_k"],
                    top_p=decode_params["top_p"],
                    stop_ids=stop_ids,
                    presence_penalty=decode_params["presence_penalty"],
                )
                generated_tokens = len(generated)
                total_decode_tokens += 1 + len(generated)
            else:
                total_decode_tokens += 1
            model.server_cleanup()
            interval = gpu_store.hcs_export_heatmap()
            interval_events = _merge_heatmap_counts(
                base_counts,
                interval,
                f"approved heatmap prompt {i}",
            )
            refresh_residency = (
                i < len(prompts)
                and (
                    not residency_active
                    or (i - resume_start) % residency_refresh_every == 0
                )
            )
            refresh_s = 0.0
            refresh_result = ""
            if refresh_residency:
                refresh_t0 = time.perf_counter()
                ranking = _full_heatmap_ranking(model, base_counts)
                gpu_store.hcs_reset()
                refresh_result = gpu_store.hcs_pool_init_tiered(
                    ranking,
                    hard_budget_mb=0,
                    soft_budget_mb=calibration["decode_hcs_budget_mb"],
                    safety_margin_mb=calibration["safety_margin_mb"],
                )
                evicted, freed_mb, refresh_free_mb = (
                    gpu_store.py_hcs_drain_vram_pressure(
                        f"approved_heatmap_refresh_{i}",
                        True,
                    )
                )
                if refresh_free_mb < calibration["safety_margin_mb"]:
                    raise RuntimeError(
                        "Approved heatmap adaptive HCS refresh could not restore the "
                        f"calibrated VRAM safety floor after prompt {i}: "
                        f"free={refresh_free_mb} MB, "
                        f"safety={calibration['safety_margin_mb']} MB"
                    )
                residency_metadata["refresh_count"] += 1
                residency_active = True
                gpu_store.hcs_start_collecting()
                refresh_s = time.perf_counter() - refresh_t0
                refresh_result = (
                    f"{refresh_result} | pressure_evicted={evicted} "
                    f"freed_mb={freed_mb:.1f} free_mb={refresh_free_mb}"
                )
            elif i < len(prompts):
                gpu_store.hcs_start_collecting()
            prompt_s = time.perf_counter() - prompt_t0
            logger.info(
                "APPROVED_HEATMAP prompt_done index=%d total=%d prompt_len=%d first_token=%d "
                "kv_overflow=%s stopped_before_decode=%s generated_tokens=%d total_decode_tokens=%d prompt_s=%.3f",
                i,
                len(prompts),
                prompt_len,
                first_token,
                bool(kv_overflow),
                stopped_before_decode,
                generated_tokens,
                total_decode_tokens,
                prompt_s,
            )
            logger.info(
                "APPROVED_HEATMAP residency_update index=%d interval_events=%d "
                "cumulative_entries=%d reload_count=%d reload_ms=%.3f refreshed=%s "
                "refresh_s=%.3f result=%s",
                i,
                interval_events,
                len(base_counts),
                reload_count,
                reload_ms,
                refresh_residency,
                refresh_s,
                refresh_result,
            )
            if refresh_residency:
                _detail(
                    f"Adaptive HCS refresh after prompt {i}: "
                    f"{len(base_counts):,} ranked experts, {refresh_s:.1f}s"
                )
            if timing_enabled:
                logger.info(
                    "HEATMAP_TIMING approved_prompt index=%d prompt_s=%.6f cumulative_s=%.6f",
                    i,
                    prompt_s,
                    time.perf_counter() - build_t0,
                )
            if checkpoint_every > 0 and i % checkpoint_every == 0 and i < len(prompts):
                export_checkpoint(i, final=False)

        final_path = export_checkpoint(len(prompts), final=True)
        logger.info(
            "APPROVED_HEATMAP build_complete prompts=%d decode_tokens=%d elapsed_s=%.3f final_path=%s written=%s",
            len(prompts),
            total_decode_tokens,
            time.perf_counter() - build_t0,
            final_path,
            ",".join(written_paths),
        )
        print("APPROVED HEATMAP BUILD COMPLETE", flush=True)
        return final_path
    finally:
        try:
            if heatmap_collection_started:
                gpu_store.hcs_reset()
        finally:
            model.server_cleanup()


_registry_file: Optional[Path] = None


def _write_registry(host: str, port: int, model_name: str) -> None:
    """Write a server registry entry to ~/.krasis/servers/{pid}.json."""
    global _registry_file
    registry_dir = Path.home() / ".krasis" / "servers"
    registry_dir.mkdir(parents=True, exist_ok=True)
    _registry_file = registry_dir / f"{os.getpid()}.json"
    entry = {
        "pid": os.getpid(),
        "port": port,
        "host": host,
        "model": model_name,
        "started": int(time.time()),
    }
    _registry_file.write_text(json.dumps(entry))
    logger.info("Registry entry written: %s", _registry_file)


def _remove_registry() -> None:
    """Remove the server registry entry on shutdown."""
    global _registry_file
    if _registry_file is not None:
        try:
            _registry_file.unlink(missing_ok=True)
        except Exception:
            pass
        _registry_file = None


def _cleanup_cuda():
    """Release all CUDA contexts to prevent zombie GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            torch.cuda.empty_cache()
    except Exception:
        pass


def _tileq_configuration_error(gpu_expert_bits: int, tileq_cache: Optional[str]) -> Optional[str]:
    """Return the fail-closed TileQ configuration error, if any."""
    if gpu_expert_bits == 3 and not tileq_cache:
        return "--gpu-expert-bits 3 requires --tileq-cache"
    if gpu_expert_bits != 3 and tileq_cache:
        return "--tileq-cache is valid only with --gpu-expert-bits 3"
    return None


def main():
    import os # Ensure os is in local scope
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" # Mitigate fragmentation

    # Note: WSL2 LD_LIBRARY_PATH fix for /usr/lib/wsl/lib is in launcher.py's
    # launch_server() — must be set BEFORE execvp because glibc caches
    # LD_LIBRARY_PATH at process startup (too late to set here for dlopen).

    # Register cleanup early to prevent CUDA zombie processes
    atexit.register(_cleanup_cuda)
    def _force_exit_handler(sig, frame):
        _cleanup_cuda()
        os._exit(1)
    signal.signal(signal.SIGTERM, _force_exit_handler)
    # ── Pre-parse --config to load defaults from file ──
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                     help="Path to config file (KEY=VALUE format). "
                          "CLI args override config file values.")
    pre.add_argument("--selected-gpus", default=None,
                     help="Explicit physical GPU identifiers to expose, overriding CFG_SELECTED_GPUS "
                          "(comma-separated indices, UUIDs, PCI bus IDs, or unique aliases like 6000/96GB).")
    pre_args, remaining_argv = pre.parse_known_args()
    explicit_selected_gpus = None
    if pre_args.selected_gpus is not None:
        explicit_selected_gpus = _normalize_selected_gpus(pre_args.selected_gpus, "--selected-gpus")
    explicit_num_gpus = _argv_has_option(remaining_argv, "--num-gpus")
    explicit_dynamic_tail_blocks = _argv_has_option(remaining_argv, "--dynamic-hcs-tail-blocks")

    config_defaults = {}
    config_selected_gpus = None
    if pre_args.config:
        config_path = pre_args.config
        if not os.path.isfile(config_path):
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        # Mapping from CFG_* keys (used in ~/.krasis/config) to argparse dests
        _CFG_KEY_MAP = {
            "MODEL_PATH": "model_path",
            "CFG_SELECTED_GPUS": "_selected_gpus",  # special: comma list -> CUDA_VISIBLE_DEVICES + num_gpus
            "CFG_PP_PARTITION": None,  # not used by server
            "CFG_LAYER_GROUP_SIZE": "layer_group_size",
            "CFG_KV_DTYPE": "kv_dtype",
            "CFG_GPU_EXPERT_BITS": "gpu_expert_bits",
            "CFG_TILEQ_CACHE": "tileq_cache",
            "CFG_EXPERT_GROUP_SIZE": "expert_group_size",
            "CFG_GPU_EXPERT_INT4_CALIB": "gpu_expert_int4_calib",
            "CFG_CPU_EXPERT_BITS": "cpu_expert_bits",
            "CFG_ATTENTION_QUANT": "attention_quant",
            "CFG_HQQ_CACHE_PROFILE": "hqq_cache_profile",
            "CFG_HQQ_GROUP_SIZE": "hqq_group_size",
            "CFG_HQQ_AUTO_BUDGET_PCT": "hqq_auto_budget_pct",
            "CFG_HQQ46_AUTO_BUDGET_MB": "hqq46_auto_budget_mib",
            "CFG_HQQ_SIDECAR_MANIFEST": "hqq_sidecar_manifest",
            "CFG_EXPERT_HQQ_DIAGNOSTIC_CACHE_SPEC": "expert_hqq_diagnostic_cache_spec",
            "CFG_SHARED_EXPERT_QUANT": "shared_expert_quant",
            "CFG_DENSE_MLP_QUANT": "dense_mlp_quant",
            "CFG_LM_HEAD_QUANT": "lm_head_quant",
            "CFG_VISION_QUANT": "step_vision_quant",
            "CFG_VISION_GROUP_SIZE": "step_vision_group_size",
            "CFG_STEP_VISION_QUANT": "step_vision_quant",
            "CFG_STEP_VISION_GROUP_SIZE": "step_vision_group_size",
            "CFG_KRASIS_THREADS": "krasis_threads",
            "CFG_HOST": "host",
            "CFG_PORT": "port",
            "CFG_SSH_TUNNEL": "ssh_tunnel",
            "CFG_SSH_KEY_PATH": "ssh_key_path",
            "CFG_GPU_PREFILL_THRESHOLD": "gpu_prefill_threshold",
            "CFG_GGUF_PATH": "gguf_path",
            "CFG_HEATMAP_PATH": "heatmap_path",
            "CFG_APPROVED_HEATMAP_MODE": "approved_heatmap_mode",
            "CFG_APPROVED_HEATMAP_MANIFEST_URL": "approved_heatmap_manifest_url",
            "CFG_APPROVED_HEATMAP_BOOTSTRAP_FROM": "approved_heatmap_bootstrap_from",
            "CFG_APPROVED_HEATMAP_RESIDENCY_REFRESH_EVERY": "approved_heatmap_residency_refresh_every",
            "CFG_FORCE_LOAD": "force_load",
            "CFG_FORCE_REBUILD_CACHE": "force_rebuild_cache",
            "CFG_FORCE_REBUILD_HQQ_CACHE": "force_rebuild_hqq_cache",
            "CFG_BUILD_CACHE": "build_cache",
            "CFG_HCS": "hcs",
            "CFG_MULTI_GPU_HCS": "multi_gpu_hcs",
            "CFG_MULTI_GPU_MODE": "multi_gpu_mode",
            "CFG_DYNAMIC_PEER": "dynamic_peer",
            "CFG_HCS_HOST_CACHE_MODE": "hcs_host_cache_mode",
            "CFG_KV_CACHE_MB": "kv_cache_mb",
            "CFG_MAX_CONTEXT_TOKENS": "max_context_tokens",
            "CFG_RING_WINDOW_KV": "ring_window_kv",
            "CFG_VRAM_SAFETY_MARGIN": "vram_safety_margin",
            "CFG_DYNAMIC_HCS": "dynamic_hcs",
            "CFG_DYNAMIC_HCS_TAIL_BLOCKS": "dynamic_hcs_tail_blocks",
            "CFG_ADAPTIVE_COLD_MASS_PRUNING": "adaptive_cold_mass_pruning",
            "CFG_EXPERT_COMPRESSION": "expert_compression",
            "CFG_EXPERT_COMPRESSION_SIDECAR": "expert_compression_sidecar",
            "CFG_EXPERT_COMPRESSION_PIPELINE": "expert_compression_pipeline",
            "CFG_STREAM_ATTENTION": "stream_attention",
            "CFG_DRAFT_MODEL": "draft_model",
            "CFG_DRAFT_K": "draft_k",
            "CFG_DRAFT_CONTEXT": "draft_context",
            "CFG_TEMPERATURE": "temperature",
            "CFG_ENABLE_THINKING": "enable_thinking",
            "CFG_PREFIX_CACHE": "prefix_cache",
            "CFG_PREFIX_CACHE_RAM_FRACTION": "prefix_cache_ram_fraction",
            "CFG_SESSION_ENABLED": None,  # Session messenger integration removed; ignore old saved configs
            "CFG_NUM_GPUS": "num_gpus",
            "CFG_CPU_DECODE": None,  # CPU decode removed, ignore config key
            "CFG_ATTN_SKIP_AFTER": "attn_skip_after",
        }
        _BOOL_CFG_KEYS = {
            "CFG_FORCE_LOAD",
            "CFG_FORCE_REBUILD_CACHE",
            "CFG_FORCE_REBUILD_HQQ_CACHE",
            "CFG_BUILD_CACHE",
            "CFG_ENABLE_THINKING",
            "CFG_PREFIX_CACHE",
            "CFG_HCS",
            "CFG_MULTI_GPU_HCS",
            "CFG_DYNAMIC_PEER",
            "CFG_DYNAMIC_HCS",
            "CFG_EXPERT_COMPRESSION",
            "CFG_RING_WINDOW_KV",
            "CFG_STREAM_ATTENTION",
            "CFG_CPU_DECODE",
        }
        with open(config_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Determine argparse dest: check CFG_ map first, then fall back
                if key in _CFG_KEY_MAP:
                    dest = _CFG_KEY_MAP[key]
                    if dest is None:
                        continue  # skip keys not used by server
                    if val == "" and key not in _BOOL_CFG_KEYS:
                        continue
                    # Handle special cases for CFG_ format
                    if key == "CFG_SELECTED_GPUS":
                        # Convert selected physical GPU identifiers to num_gpus count.
                        # CUDA_VISIBLE_DEVICES is set earlier in _prescan_selected_gpus()
                        selected = _normalize_selected_gpus(val, "CFG_SELECTED_GPUS")
                        config_selected_gpus = selected
                        gpu_list = [x.strip() for x in selected.split(",") if x.strip()]
                        if gpu_list:
                            config_defaults["selected_gpus"] = selected
                            config_defaults["num_gpus"] = len(gpu_list)
                        continue
                    if key in _BOOL_CFG_KEYS:
                        # CFG_ format uses "1"/"" for booleans
                        config_defaults[dest] = val == "1"
                        continue
                else:
                    # Plain key format (key-name or key_name)
                    dest = key.replace("-", "_").lower()
                # Convert "true"/"false" strings for store_true args
                if isinstance(val, str) and val.lower() == "true":
                    config_defaults[dest] = True
                elif isinstance(val, str) and val.lower() == "false":
                    config_defaults[dest] = False
                else:
                    # Try int, then float, then string
                    try:
                        config_defaults[dest] = int(val)
                    except ValueError:
                        try:
                            config_defaults[dest] = float(val)
                        except ValueError:
                            config_defaults[dest] = val
        if config_defaults.get("attention_quant") in ("int4", "int8"):
            raise ValueError(
                f"Unsupported attention_quant={config_defaults['attention_quant']} in {config_path}. "
                "Naive int4/int8 attention has been removed; use 'hqq4', 'hqq46', 'hqq46_auto', 'hqq6', 'hqq68_auto', 'hqq8', or 'bf16'."
            )
        if config_defaults.get("attention_quant") in DEPRECATED_ATTENTION_QUANT_CHOICES:
            raise ValueError(
                f"attention_quant={config_defaults['attention_quant']} in {config_path} is deprecated and disabled. "
                "Use HQQ attention modes: 'hqq4', 'hqq46', 'hqq46_auto', 'hqq6', 'hqq68_auto', or 'hqq8'."
            )
        if config_defaults.get("kv_dtype") in DEPRECATED_KV_CACHE_FORMAT_CHOICES:
            raise ValueError(
                f"kv_dtype={config_defaults['kv_dtype']} in {config_path} is deprecated and disabled. "
                "Use 'k6v6', 'k4v4', or 'bf16'."
            )
        # Expand ~ in model_path
        if "model_path" in config_defaults and isinstance(config_defaults["model_path"], str):
            config_defaults["model_path"] = os.path.expanduser(config_defaults["model_path"])
        if "heatmap_path" in config_defaults and isinstance(config_defaults["heatmap_path"], str):
            config_defaults["heatmap_path"] = os.path.expanduser(config_defaults["heatmap_path"])
        if "ssh_key_path" in config_defaults and isinstance(config_defaults["ssh_key_path"], str):
            config_defaults["ssh_key_path"] = os.path.expanduser(config_defaults["ssh_key_path"])
        if "draft_model" in config_defaults and isinstance(config_defaults["draft_model"], str):
            config_defaults["draft_model"] = os.path.expanduser(config_defaults["draft_model"])
        if explicit_selected_gpus is not None:
            config_defaults["selected_gpus"] = explicit_selected_gpus
            config_defaults["num_gpus"] = len(explicit_selected_gpus.split(","))
        print(f"Loaded config from {config_path}: {config_defaults}")
    elif explicit_selected_gpus is not None:
        config_defaults["selected_gpus"] = explicit_selected_gpus
        config_defaults["num_gpus"] = len(explicit_selected_gpus.split(","))
    config_dynamic_tail_blocks = "dynamic_hcs_tail_blocks" in config_defaults

    parser = argparse.ArgumentParser(description="Krasis standalone LLM server",
                                     parents=[pre])
    parser.add_argument("--model-path", required="model_path" not in config_defaults,
                        help="Path to HF model")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (auto-detected if omitted)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--ssh-tunnel", default="",
                        help="Reverse SSH tunnel target: user@host or user@host:port. "
                             "Remote 127.0.0.1:<server port> forwards to local Krasis.")
    parser.add_argument("--ssh-key-path", default="",
                        help="Optional SSH identity file for --ssh-tunnel; uses IdentitiesOnly=yes.")
    parser.add_argument("--krasis-threads", type=int, default=40,
                        help="CPU threads for expert computation")
    parser.add_argument("--kv-dtype", default="k6v6",
                        choices=list(KV_CACHE_FORMAT_CHOICES + DEPRECATED_KV_CACHE_FORMAT_CHOICES),
                        help="KV cache format: k6v6 Quality default, k4v4 Ultra Compact, bf16 Full Precision, or explicit internal formats; fp8/fp8_e4m3 are deprecated and disabled")
    parser.add_argument("--kv-cache-mb", type=int, default=1000,
                        help="KV cache size in MB (default: 1000)")
    parser.add_argument("--max-context-tokens", type=int, default=0,
                        help="Explicit runtime context cap; 0 uses the model limit")
    parser.add_argument("--ring-window-kv", action="store_true",
                        help="Experimental: cap sliding-attention KV layers to their physical window; requires correctness validation")
    parser.add_argument("--heatmap-path", default=None,
                        help="Path to expert_heatmap.json for HCS init")
    parser.add_argument("--approved-heatmap-mode",
                        default=os.environ.get("KRASIS_APPROVED_HEATMAP_MODE", "auto"),
                        choices=list(APPROVED_HEATMAP_MODE_CHOICES),
                        help="Approved route-heatmap lookup mode: auto downloads a matching GitHub artifact if available; off always builds the startup heatmap; require errors if no approved artifact matches")
    parser.add_argument("--approved-heatmap-manifest-url",
                        default=os.environ.get("KRASIS_APPROVED_HEATMAP_MANIFEST_URL", APPROVED_HEATMAP_DEFAULT_MANIFEST_URL),
                        help="Manifest URL for approved route-heatmap artifacts")
    parser.add_argument("--approved-heatmap-build-out", default=None,
                        help="Build an approved cumulative HCS route heatmap artifact at this path and exit")
    parser.add_argument("--approved-heatmap-resume-from", default=None,
                        help="Resume approved heatmap capture from an existing validated approved artifact")
    parser.add_argument("--approved-heatmap-bootstrap-from", default=None,
                        help="Use a compatible heatmap only to seed HCS residency during approved capture; its counts are not added to the output")
    parser.add_argument("--approved-heatmap-residency-refresh-every", type=int, default=1,
                        help="Rebuild calibrated HCS residency from cumulative captured routes every N new prompts")
    parser.add_argument("--approved-heatmap-prompts", default=None,
                        help="Prompt corpus for --approved-heatmap-build-out; same blank-line separated format as heatmap_prompts.txt")
    parser.add_argument("--approved-heatmap-decode-tokens", type=int, default=HEATMAP_DECODE_TOKENS,
                        help="Decode tokens per prompt for approved heatmap capture")
    parser.add_argument("--approved-heatmap-max-prompts", type=int, default=0,
                        help="Limit approved heatmap capture to the first N prompts; 0 means all")
    parser.add_argument("--approved-heatmap-checkpoint-every", type=int, default=0,
                        help="Write cumulative approved heatmap checkpoints every N prompts; 0 means final artifact only")
    parser.add_argument("--gpu-expert-bits", type=int, default=4, choices=[3, 4, 8, 16],
                        help="Expert weight bits: 3 uses an explicit TileQ cache, 4/8 use Marlin, 16 is direct BF16 debug mode")
    parser.add_argument("--tileq-cache", default=None,
                        help="Explicit source-bound KTQ1 routed-expert artifact; required for GPU expert bits=3")
    parser.add_argument("--expert-group-size", type=int, default=128, choices=[32, 64, 128],
                        help="Expert quantization group size for routed GPU/CPU expert caches")
    parser.add_argument("--gpu-expert-int4-calib", default="amax", choices=list(GPU_EXPERT_INT4_CALIB_CHOICES),
                        help="Offline calibration mode for GPU routed-expert INT4 cache build")
    parser.add_argument("--cpu-expert-bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bits for CPU decode experts")
    parser.add_argument("--attention-quant", default="bf16", choices=list(ATTENTION_QUANT_CHOICES),
                        help="Attention weight precision: hqq8 quality-first, hqq68_auto budget-planned mixed HQQ6/HQQ8, hqq6 packed middle-ground, hqq46_auto budget-planned mixed HQQ4/HQQ6, hqq46 fixed-policy mixed, hqq4 and bf16 remain explicit alternatives")
    parser.add_argument("--hqq-cache-profile", default="baseline", choices=list(HQQ_CACHE_PROFILE_CHOICES),
                        help="HQQ attention cache profile: baseline (default) or an explicit calibrated profile")
    parser.add_argument("--hqq-group-size", type=int, default=128, choices=[32, 64, 128],
                        help="HQQ attention quantization group size; default 128")
    parser.add_argument("--hqq-auto-budget-pct", type=float, default=None,
                        help="HQQ auto planner promotion budget as percentage of the base-to-target attention-memory span")
    parser.add_argument("--hqq46-auto-budget-mib", type=int, default=None,
                        help="Legacy HQQ4/6 auto planner HQQ6 promotion budget in MiB")
    parser.add_argument("--hqq-sidecar-manifest", default=None,
                        help="Explicit HQQ4-only sidecar manifest; HQQ8 rejects sidecar/self-correction")
    parser.add_argument("--expert-hqq-diagnostic-cache-spec", default=None,
                        help="Explicit diagnostic-only KRHQ cache spec for runtime metadata validation")
    parser.add_argument("--shared-expert-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for shared expert weights")
    parser.add_argument("--dense-mlp-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for dense MLP weights")
    parser.add_argument("--lm-head-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for lm_head weights")
    parser.add_argument("--step-vision-quant", "--vision-quant", dest="step_vision_quant",
                        default="int4", choices=["bf16", "int4"],
                        help="Lazy vision tower weight precision for image requests; INT4 is the default")
    parser.add_argument("--step-vision-group-size", "--vision-group-size", dest="step_vision_group_size",
                        type=int, default=128, choices=[32, 64, 128],
                        help="Lazy vision INT4 quantization group size")
    parser.add_argument("--gguf-path", default=None,
                        help="Path to GGUF file for CPU experts")
    parser.add_argument("--force-load", action="store_true",
                        help="Override RAM safety checks and load anyway")
    parser.add_argument("--force-rebuild-cache", action="store_true",
                        help="Delete existing expert caches and rebuild from safetensors")
    parser.add_argument("--force-rebuild-hqq-cache", action="store_true",
                        help="Delete the selected HQQ attention cache and rebuild from safetensors")
    parser.add_argument("--build-cache", action="store_true",
                        help="Build expert caches (if missing) and exit without starting server")
    parser.add_argument("--hcs", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Hot Cache Strategy (default: on for GPU decode, use --no-hcs to disable)")
    parser.add_argument("--multi-gpu-hcs", action="store_true", default=False,
                        help="Pin HCS experts across ALL GPUs (more capacity, but cross-device transfer)")
    parser.add_argument(
        "--multi-gpu-mode",
        default="auto",
        choices=["auto", "layer-split", "peer"],
        help=(
            "Multi-GPU decode planning: choose from measured startup inputs, "
            "force the serial layer split, or force peer expert serving"
        ),
    )
    parser.add_argument("--hcs-host-cache-mode", default="source", choices=["auto", "mirror", "source"],
                        help="Soft HCS host storage: source/lower-system-RAM is the default, "
                             "mirror keeps the fast duplicated host chunks, auto chooses "
                             "source only when system RAM is tight")
    parser.add_argument("--dynamic-hcs", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable dynamic HCS heatmap-prefix + recency-tail cache (default: on)")
    parser.add_argument("--dynamic-hcs-tail-blocks", type=int, default=2, choices=range(1, 6),
                        help="Advanced: recency tail size in activated-expert blocks (1-5, default: 2)")
    parser.add_argument(
        "--dynamic-peer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adapt peer hard-pool residency from live surviving cold routes",
    )
    parser.add_argument(
        "--adaptive-cold-mass-pruning",
        default=None,
        choices=list(ADAPTIVE_COLD_MASS_PRUNING_CHOICES),
        help=(
            "Approximate demand-cold expert pruning preset (launcher default: off; "
            "when omitted, existing low-level environment variables are unchanged)"
        ),
    )
    parser.add_argument(
        "--expert-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a versioned bit-exact compressed expert sidecar for demand-cold DMA",
    )
    parser.add_argument(
        "--expert-compression-sidecar",
        default=None,
        help="Exact .krec sidecar built for the loaded Marlin expert cache",
    )
    parser.add_argument(
        "--expert-compression-pipeline",
        default="grouped",
        choices=("grouped", "streaming", "auto"),
        help="Compressed expert pipeline: established grouped, persistent streaming, or measured auto",
    )
    # NOTE: --hcs-headroom-mb removed — HCS budget is computed from 4-point VRAM calibration, not a fixed headroom
    parser.add_argument("--vram-safety-margin", type=int, default=600,
                        help="VRAM safety margin in MB — reserved free VRAM for decode kernel intermediates "
                             "and CUDA allocator headroom (default: 600 MB)")
    parser.add_argument("--stream-attention", action="store_true",
                        help="Stream attention weights from CPU instead of keeping resident on GPU. "
                             "Use when attention weights don't fit in VRAM (e.g. very large models).")
    parser.add_argument("--no-stream-attention", action="store_true",
                        help="(deprecated, now the default) Attention is resident on GPU by default.")
    parser.add_argument("--layer-group-size", type=int, default=2,
                        help="Number of MoE layers to load per group during prefill (default: 2)")
    # GPU decode is the only mode — CPU decode has been removed.
    # Keep --gpu-decode as a no-op for config file compatibility.
    parser.add_argument("--gpu-decode", action="store_true", default=True,
                        help="(default, only mode) GPU decode via Rust GpuDecodeStore.")
    parser.add_argument("--draft-model", default=None,
                        help="Path to draft model for speculative decoding (e.g. ~/.krasis/models/Qwen3-0.6B)")
    parser.add_argument("--draft-k", type=int, default=3,
                        help="Number of tokens to draft per speculative round (default: 3)")
    parser.add_argument("--draft-context", type=int, default=512,
                        help="Context window for draft model warmup (default: 512)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run standardized benchmark via HTTP (same path as production)")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run benchmark via HTTP and exit (don't keep server running)")
    parser.add_argument("--timing", action="store_true",
                        help="Enable decode timing instrumentation (per-layer breakdown)")
    parser.add_argument("--vram-report", action="store_true",
                        help="Generate VRAM report CSV (periodic readings + events) in the current run directory")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run stress test (diverse prompts) and exit")
    parser.add_argument("--perplexity", action="store_true",
                        help="Run perplexity evaluation and exit")
    parser.add_argument("--note", default=None,
                        help="Description note written to the top of the log file for this run")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable thinking/reasoning mode (default: on)")
    parser.add_argument("--test-endpoints", action="store_true", default=False,
                        help="Enable test-only endpoints (/v1/internal/prefill_logits)")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable RAM-backed multi-conversation prefix-state caching (default: on)")
    parser.add_argument("--prefix-cache-ram-fraction", type=float, default=0.25,
                        help="Fraction of cgroup-aware available system RAM usable by prefix snapshots")
    # Apply config file defaults, then parse CLI (CLI wins over config file)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_argv)
    if not math.isfinite(args.prefix_cache_ram_fraction) or not 0.0 < args.prefix_cache_ram_fraction <= 1.0:
        parser.error("--prefix-cache-ram-fraction must be finite and in (0, 1]")
    try:
        args.dynamic_hcs_tail_blocks = int(args.dynamic_hcs_tail_blocks)
    except (TypeError, ValueError):
        parser.error("--dynamic-hcs-tail-blocks must be an integer in 1..5")
    if args.dynamic_hcs_tail_blocks < 1 or args.dynamic_hcs_tail_blocks > 5:
        parser.error("--dynamic-hcs-tail-blocks must be in 1..5")
    args.approved_heatmap_mode = str(args.approved_heatmap_mode or "auto").strip().lower()
    if args.approved_heatmap_mode not in APPROVED_HEATMAP_MODE_CHOICES:
        parser.error(
            "--approved-heatmap-mode must be one of: "
            + ", ".join(APPROVED_HEATMAP_MODE_CHOICES)
        )
    _hcs_host_aliases = {
        "1": "source",
        "true": "source",
        "yes": "source",
        "on": "source",
        "low_ram": "source",
        "low-ram": "source",
        "0": "mirror",
        "false": "mirror",
        "no": "mirror",
        "off": "mirror",
        "fast": "mirror",
    }
    args.hcs_host_cache_mode = _hcs_host_aliases.get(
        str(args.hcs_host_cache_mode or "source").strip().lower(),
        str(args.hcs_host_cache_mode or "source").strip().lower(),
    )
    if args.hcs_host_cache_mode not in ("auto", "mirror", "source"):
        parser.error("--hcs-host-cache-mode must be one of: auto, mirror, source")
    if args.adaptive_cold_mass_pruning is not None:
        try:
            args.adaptive_cold_mass_pruning = configure_adaptive_cold_mass_pruning(
                args.adaptive_cold_mass_pruning
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.expert_compression:
        if not str(args.expert_compression_sidecar or "").strip():
            parser.error("--expert-compression requires --expert-compression-sidecar")
        sidecar = os.path.expanduser(args.expert_compression_sidecar)
        if not os.path.isfile(sidecar):
            parser.error(f"expert compression sidecar not found: {sidecar}")
        os.environ["KRASIS_EXPERT_COMPRESSION_SIDECAR"] = sidecar
        os.environ["KRASIS_EXPERT_COMPRESSION_PIPELINE"] = (
            args.expert_compression_pipeline
        )
    else:
        os.environ.pop("KRASIS_EXPERT_COMPRESSION_SIDECAR", None)
        os.environ.pop("KRASIS_EXPERT_COMPRESSION_PIPELINE", None)
    if args.multi_gpu_mode == "peer" and not args.hcs:
        parser.error("--multi-gpu-mode peer requires HCS")
    peer_format_error = _peer_expert_format_error(args.gpu_expert_bits)
    if args.multi_gpu_mode == "peer" and peer_format_error is not None:
        parser.error(f"--multi-gpu-mode peer: {peer_format_error}")
    if args.dynamic_peer and not args.hcs:
        parser.error("--dynamic-peer requires HCS")
    if args.dynamic_peer and args.multi_gpu_mode == "layer-split":
        parser.error("--dynamic-peer requires auto or peer multi-GPU mode")
    os.environ["KRASIS_DYNAMIC_PEER"] = "1" if args.dynamic_peer else "0"
    if str(getattr(args, "ssh_key_path", "") or "").strip():
        args.ssh_key_path = os.path.expanduser(args.ssh_key_path.strip())
    if str(getattr(args, "ssh_tunnel", "") or "").strip():
        from krasis.ssh_tunnel import parse_ssh_tunnel_target

        try:
            parse_ssh_tunnel_target(args.ssh_tunnel)
        except ValueError as exc:
            parser.error(f"--ssh-tunnel: {exc}")
    if args.hcs and args.dynamic_hcs:
        os.environ["KRASIS_DYNAMIC_HCS"] = "1"
        if (
            explicit_dynamic_tail_blocks
            or config_dynamic_tail_blocks
            or "KRASIS_DYNAMIC_HCS_TAIL_BLOCKS" not in os.environ
        ):
            os.environ["KRASIS_DYNAMIC_HCS_TAIL_BLOCKS"] = str(args.dynamic_hcs_tail_blocks)
    else:
        os.environ["KRASIS_DYNAMIC_HCS"] = "0"
    os.environ["KRASIS_HCS_HOST_CACHE_MODE"] = str(args.hcs_host_cache_mode or "source").strip().lower()
    if explicit_selected_gpus is not None:
        selected_count = len(explicit_selected_gpus.split(","))
        if explicit_num_gpus and args.num_gpus is not None and args.num_gpus != selected_count:
            print(
                "Error: --selected-gpus implies "
                f"--num-gpus {selected_count}, but --num-gpus {args.num_gpus} was also supplied",
                file=sys.stderr,
            )
            sys.exit(1)
        args.selected_gpus = explicit_selected_gpus
        args.num_gpus = selected_count
        args.selected_gpus_source = "cli"
    elif getattr(args, "selected_gpus", None):
        args.selected_gpus = _normalize_selected_gpus(args.selected_gpus, "CFG_SELECTED_GPUS")
        args.num_gpus = len(args.selected_gpus.split(","))
        args.selected_gpus_source = "config"
    else:
        args.selected_gpus = None
        args.selected_gpus_source = "auto"
    _default_run_type = "server-run"
    if args.benchmark_only:
        _default_run_type = "server-benchmark"
    elif args.benchmark:
        _default_run_type = "server-run-benchmark"
    elif getattr(args, "stress_test", False):
        _default_run_type = "server-stress"
    _run_dir = get_run_dir(_default_run_type)

    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    _root_logger = logging.getLogger()
    _root_logger.setLevel(logging.INFO)
    for _handler in list(_root_logger.handlers):
        _root_logger.removeHandler(_handler)
        _handler.close()

    _log_file = os.path.join(_run_dir, "krasis.log")

    _file_handler = logging.FileHandler(_log_file, mode="w")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(log_format))
    _root_logger.addHandler(_file_handler)

    # Write run note to top of log file if provided
    if args.note:
        with open(_log_file, "w") as _nf:
            _nf.write(f"=== RUN NOTE: {args.note} ===\n\n")
        # Re-open handler in append mode so logging doesn't overwrite the note
        _root_logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = logging.FileHandler(_log_file, mode="a")
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(logging.Formatter(log_format))
        _root_logger.addHandler(_file_handler)

    # Capture uncaught exceptions to the log file (main thread)
    _original_excepthook = sys.excepthook
    def _log_excepthook(exc_type, exc_value, exc_tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        _original_excepthook(exc_type, exc_value, exc_tb)
    sys.excepthook = _log_excepthook

    # Capture thread exceptions to the log file too
    def _log_threading_excepthook(args):
        logger.critical("Exception in thread %s", args.thread.name if args.thread else "?",
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = _log_threading_excepthook

    # Tee stdout/stderr to the file logger while preserving clean terminal
    # output. The root logger intentionally has no console handler here: log
    # records get prefixed in krasis.log, while operator-visible print/eprintln
    # lines remain unprefixed in the terminal.
    class _StreamLogger:
        def __init__(self, original, log):
            self._original = original
            self._log = log
        def write(self, msg):
            if msg and msg.strip():
                self._log("%s", msg.rstrip())
            if self._original:
                self._original.write(msg)
        def flush(self):
            if self._original:
                self._original.flush()
        def fileno(self):
            return self._original.fileno() if self._original else -1
        def isatty(self):
            return self._original.isatty() if self._original else False
        def __getattr__(self, name):
            return getattr(self._original, name)

    sys.stdout = _StreamLogger(sys.stdout, logger.info)
    sys.stderr = _StreamLogger(sys.stderr, logger.error)

    logger.info("Logging to %s", _log_file)

    # Dump config to log for easier debugging
    if pre_args.config:
        logger.info("=== Config file: %s ===", pre_args.config)
        try:
            with open(pre_args.config, encoding="utf-8") as _cf:
                for _line in _cf:
                    logger.info("  %s", _line.rstrip())
        except Exception as _e:
            logger.warning("Could not read config file: %s", _e)
    else:
        logger.info("No config file — using CLI args / defaults")
    logger.info("=== Resolved arguments ===")
    for _k, _v in sorted(vars(args).items()):
        logger.info("  %s = %r", _k, _v)
    logger.info(
        "Effective selected GPUs: %s (source=%s, CUDA_VISIBLE_DEVICES=%s)",
        args.selected_gpus or "auto",
        args.selected_gpus_source,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    global _model, _model_name
    import torch

    if args.kv_dtype in DEPRECATED_KV_CACHE_FORMAT_CHOICES:
        raise ValueError(
            f"kv_dtype={args.kv_dtype} is deprecated and disabled. "
            "Use 'k6v6' for quality, 'k4v4' for compact KV, or 'bf16' for full precision."
        )

    kv_format_str = args.kv_dtype  # includes architecture-owned "native" and generic formats
    if args.kv_dtype in ("k8v4", "k8v6", "k7v4", "k6v6", "k6v4", "k4v4", "tq4"):
        kv_dtype = torch.float8_e4m3fn  # base dtype for size calc; custom formats allocate their own tensors
    else:
        kv_dtype = torch.bfloat16

    quant_cfg = QuantConfig(
        lm_head=args.lm_head_quant,
        attention=args.attention_quant,
        hqq_cache_profile=args.hqq_cache_profile,
        hqq_group_size=args.hqq_group_size,
        hqq_auto_budget_pct=args.hqq_auto_budget_pct,
        hqq46_auto_budget_mib=args.hqq46_auto_budget_mib,
        hqq_sidecar_manifest=args.hqq_sidecar_manifest,
        shared_expert=args.shared_expert_quant,
        dense_mlp=args.dense_mlp_quant,
        gpu_expert_bits=args.gpu_expert_bits,
        tileq_cache=args.tileq_cache,
        expert_group_size=args.expert_group_size,
        gpu_expert_int4_calib=args.gpu_expert_int4_calib,
        cpu_expert_bits=args.cpu_expert_bits,
        kv_cache_format=args.kv_dtype,
        ring_window_kv=args.ring_window_kv,
        step_vision_quant=args.step_vision_quant,
        step_vision_group_size=args.step_vision_group_size,
    )

    # Expand ~ in paths (config files use ~/.krasis/...)
    args.model_path = os.path.expanduser(args.model_path)
    if args.heatmap_path:
        args.heatmap_path = os.path.expanduser(args.heatmap_path)
    if args.approved_heatmap_build_out:
        args.approved_heatmap_build_out = os.path.expanduser(args.approved_heatmap_build_out)
    if args.approved_heatmap_resume_from:
        args.approved_heatmap_resume_from = os.path.expanduser(args.approved_heatmap_resume_from)
    if args.approved_heatmap_bootstrap_from:
        args.approved_heatmap_bootstrap_from = os.path.expanduser(args.approved_heatmap_bootstrap_from)
    if args.approved_heatmap_prompts:
        args.approved_heatmap_prompts = os.path.expanduser(args.approved_heatmap_prompts)
    if args.gguf_path:
        args.gguf_path = os.path.expanduser(args.gguf_path)
    if args.expert_hqq_diagnostic_cache_spec:
        args.expert_hqq_diagnostic_cache_spec = os.path.expanduser(args.expert_hqq_diagnostic_cache_spec)
    if args.tileq_cache:
        args.tileq_cache = os.path.expanduser(args.tileq_cache)
    tileq_error = _tileq_configuration_error(args.gpu_expert_bits, args.tileq_cache)
    if tileq_error:
        parser.error(tileq_error)
    if args.tileq_cache:
        os.environ["KRASIS_TILEQ_CACHE"] = args.tileq_cache

    if args.approved_heatmap_residency_refresh_every <= 0:
        parser.error("--approved-heatmap-residency-refresh-every must be positive")
    if args.approved_heatmap_resume_from and args.approved_heatmap_bootstrap_from:
        parser.error(
            "--approved-heatmap-resume-from and --approved-heatmap-bootstrap-from "
            "cannot be combined"
        )

    _model_name = args.model_path.rstrip("/").split("/")[-1]

    # ── Load model with HCS strategy ──
    import os, json
    from krasis.config import ModelConfig

    cfg = ModelConfig.from_model_path(args.model_path)
    num_layers = cfg.num_hidden_layers
    num_gpus_available = args.num_gpus or torch.cuda.device_count()
    if args.multi_gpu_mode == "peer" and num_gpus_available != 2:
        parser.error("--multi-gpu-mode peer currently requires exactly two selected GPUs")
    if args.dynamic_peer and num_gpus_available != 2:
        parser.error("--dynamic-peer currently requires exactly two selected GPUs")

    # GPU decode is the only supported serving mode here. Do not load CPU expert
    # weights or enable any CPU-side fallback path.
    gpu_only = True

    # ── Configuration summary ──
    _status(f"Krasis — {_model_name}")
    _detail(f"Decode: GPU  |  HCS: {'on' if args.hcs else 'off'}  |  GPUs: {num_gpus_available}")
    if num_gpus_available > 1:
        _detail(f"Multi-GPU mode: {args.multi_gpu_mode}")
    if args.hcs:
        _detail(f"HCS host cache: {args.hcs_host_cache_mode}")
    if args.selected_gpus:
        _detail(
            f"Selected physical GPUs: {args.selected_gpus} "
            f"({args.selected_gpus_source}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"
        )
    if args.gpu_expert_bits == 16:
        expert_detail = "Experts: GPU BF16"
    elif args.gpu_expert_bits == 3:
        expert_detail = f"Experts: TileQ INT3 residual g{args.expert_group_size}"
    else:
        expert_detail = f"Experts: GPU INT{args.gpu_expert_bits} g{args.expert_group_size}"
    if args.gpu_expert_bits == 4:
        expert_detail += f" ({args.gpu_expert_int4_calib})"
    _detail(f"{expert_detail}  |  Attention: {args.attention_quant}  |  KV: {args.kv_dtype}")
    _detail(f"Layer groups: {args.layer_group_size}  |  KV cache: {args.kv_cache_mb} MB  |  Threads: {args.krasis_threads}")
    _dim("GPU-only mode: CPU expert weights and CPU decoder skipped")
    validation_components = []
    if args.gpu_expert_bits == 16:
        validation_components.append("GPU experts=BF16")
    if args.shared_expert_quant == "bf16":
        validation_components.append("shared expert=BF16")
    if args.dense_mlp_quant == "bf16":
        validation_components.append("dense MLP=BF16")
    if args.lm_head_quant == "bf16":
        validation_components.append("lm head=BF16")
    if validation_components:
        joined = ", ".join(validation_components)
        _warn(
            "UNVALIDATED BF16 debug path enabled: "
            f"{joined}. This path likely contains unknown bugs and must not be "
            "used as a correctness oracle. Use HF Transformers BF16 reference "
            "data for validation; production runs must use quantized configs."
        )
        logger.warning(
            "UNVALIDATED BF16 debug path enabled (%s); this path likely contains "
            "unknown bugs and must not be used for validation. Use HF "
            "Transformers BF16 reference data as the oracle; production runs "
            "must use quantized configs.",
            joined,
        )

    # ── Force rebuild: delete existing expert caches before loading ──
    if getattr(args, 'force_rebuild_cache', False):
        _cache_dir = cache_dir_for_model(args.model_path)
        _deleted = []
        for pattern in ["experts_marlin_*.bin"]:
            import glob as _glob
            for f in _glob.glob(os.path.join(_cache_dir, pattern)):
                os.unlink(f)
                _deleted.append(os.path.basename(f))
        if _deleted:
            _status(f"Deleted {len(_deleted)} cache files: {', '.join(_deleted)}")
        else:
            _detail("No existing cache files to delete")

    # ── Force HQQ rebuild: delete selected HQQ attention cache before loading ──
    if getattr(args, 'force_rebuild_hqq_cache', False):
        cache_nbits = attention_quant_cache_nbits(quant_cfg.attention)
        if cache_nbits is None:
            _detail(f"No HQQ attention cache to rebuild for attention={quant_cfg.attention}")
        elif quant_cfg.hqq_cache_profile != HQQ_CACHE_PROFILE_BASELINE:
            raise RuntimeError(
                "Cannot rebuild calibrated HQQ attention cache profile "
                f"{quant_cfg.hqq_cache_profile!r} from normal server startup. "
                "Use the HQQ calibration/build command for that profile, or switch "
                "to the baseline profile before using --force-rebuild-hqq-cache."
            )
        else:
            hqq_cache_dir = hqq_attention_cache_dir(
                args.model_path,
                quant_cfg.hqq_cache_profile,
                cache_nbits,
                quant_cfg.hqq_group_size,
            )
            if os.path.isdir(hqq_cache_dir):
                shutil.rmtree(hqq_cache_dir)
                _status("Deleted HQQ attention cache")
                _detail(hqq_cache_dir)
            else:
                _detail(f"No HQQ attention cache directory to delete: {hqq_cache_dir}")

    pp_partition = [num_layers]  # PP=1: all layers on primary GPU
    logger.info("HCS strategy: PP=1, %d GPUs available", num_gpus_available)

    _model = KrasisModel(
        model_path=args.model_path,
        pp_partition=pp_partition,
        num_gpus=num_gpus_available,
        kv_dtype=kv_dtype,
        krasis_threads=args.krasis_threads,
        quant_cfg=quant_cfg,
        layer_group_size=args.layer_group_size,
        gguf_path=args.gguf_path,
        expert_hqq_diagnostic_cache_spec=args.expert_hqq_diagnostic_cache_spec,
        force_load=args.force_load,
        gpu_prefill_threshold=1 if args.hcs else getattr(args, 'gpu_prefill_threshold', int(os.environ.get("KRASIS_PREFILL_THRESHOLD", "500"))),
        kv_cache_mb=args.kv_cache_mb,
        stream_attention=args.stream_attention,
        max_context_tokens=(
            args.max_context_tokens if args.max_context_tokens != 0 else None
        ),
    )
    log_ram_ledger("after-model-object")

    # Set attention skip layer if configured
    attn_skip = getattr(args, 'attn_skip_after', None)
    if attn_skip is not None and attn_skip != '':
        from krasis.layer import TransformerLayer
        TransformerLayer.attn_skip_after = int(attn_skip)
        _detail(f"Attention skip: layers >= {attn_skip} will skip attention")

    _status("Loading model weights")
    if getattr(args, 'build_cache', False):
        # --build-cache: build GPU Marlin expert cache then exit (CPU cache no longer used)
        _detail("Build-cache mode: building/verifying GPU Marlin expert cache")
        _model.load(gpu_only=True)
        log_ram_ledger("after-build-cache-load")
        import glob as _glob
        _cache_dir = cache_dir_for_model(args.model_path)
        gpu_bits = args.gpu_expert_bits
        has_gpu = bool(
            _glob.glob(
                os.path.join(
                    _cache_dir,
                    marlin_cache_basename(gpu_bits, "*", args.gpu_expert_int4_calib),
                )
            )
        )
        _status("Cache build complete")
        _detail(f"GPU Marlin INT{gpu_bits}: {'exists' if has_gpu else 'MISSING'}")
        print("BUILD CACHE COMPLETE", flush=True)
        return
    _model.load(gpu_only=gpu_only)
    log_ram_ledger("after-model-load-return")

    _hf_tok = _model.tokenizer.tokenizer  # unwrap Tokenizer -> HF AutoTokenizer
    _template = getattr(_hf_tok, "chat_template", "") or ""
    template_supports_enable_thinking = "enable_thinking" in _template
    template_has_thinking = "<think>" in _template and "</think>" in _template
    if args.enable_thinking and not (template_supports_enable_thinking or template_has_thinking):
        logger.info(
            "Model template does not support enable_thinking; forcing server default thinking off"
        )
        args.enable_thinking = False

    # Resolve heatmap save path (rebuilt fresh on every startup unless --heatmap-path)
    cache_dir = cache_dir_for_model(args.model_path)
    heatmap_path = args.heatmap_path
    if not heatmap_path:
        heatmap_path = os.path.join(cache_dir, "auto_heatmap.json")

    # CUDA runtime warmup — triggers cuBLAS allocation and loads the vendored
    # Marlin kernel on every selected device before any VRAM measurements.
    num_gpus_available = args.num_gpus or torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus_available)]
    device_indices = list(range(num_gpus_available))
    _status("CUDA runtime warmup")
    _model.warmup_cuda_runtime(devices)
    _detail("cuBLAS + Marlin runtime warmup done")
    log_ram_ledger("after-cuda-runtime-warmup")

    # ── Set decode mode (GPU only) ──
    _model.decode_mode = "gpu"

    # ── GPU decode store setup (before warmup so decode warmup can use it) ──
    _status("Setting up GPU decode store")
    log_ram_ledger("before-decode-store-setup")
    gpu_store = _model.setup_gpu_decode_store()
    gpu_store_addr = gpu_store.gpu_store_addr()
    _detail(f"GPU decode store ready (addr={gpu_store_addr:#x})")
    log_ram_ledger("after-decode-store-setup-before-release")
    if _vram_ledger_enabled():
        _model.log_vram_ledger_residency("after-decode-store-setup-before-release")
    release_gpu_sources = not (args.stress_test or args.perplexity)
    if release_gpu_sources:
        released_mb = _model.release_redundant_gpu_execution_tensors(
            release_lm_head_source=True,
            allow_multi_gpu_lm_head=True,
            release_router_fp32_mirrors=True,
        )
        if released_mb > 0:
            _detail(f"Released redundant GPU execution source tensors ({released_mb:,} MB)")
            log_ram_ledger("after-redundant-gpu-source-release")
    elif _vram_ledger_enabled():
        logger.info(
            "VRAM LEDGER gpu_source_release_skipped stress_test=%s perplexity=%s",
            args.stress_test,
            args.perplexity,
        )
    if _vram_ledger_enabled():
        _model.log_vram_ledger_residency("after-decode-store-setup")

    # ── Load draft model BEFORE warmup/VRAM capture so HCS budget accounts for it ──
    if args.draft_model:
        import os
        draft_path = os.path.expanduser(args.draft_model)
        _status(f"Loading draft model from {draft_path}")
        gpu_store.load_draft_model(
            draft_path,
            max_seq=4096,
            draft_k=args.draft_k,
            context_window=args.draft_context,
        )
        _detail(f"Draft model loaded (k={args.draft_k}, context={args.draft_context})")

    # ── Start VRAM monitor before warmup for visibility ──
    from krasis import VramMonitor
    if args.vram_safety_margin > 0:
        SAFETY_MARGIN_MB = args.vram_safety_margin
    else:
        SAFETY_MARGIN_MB = 600
    _vram_poll_raw = os.environ.get("KRASIS_VRAM_MONITOR_POLL_MS")
    if _vram_poll_raw is None:
        _vram_poll_ms = 50
    else:
        try:
            _vram_poll_ms = int(_vram_poll_raw)
        except ValueError as exc:
            raise SystemExit(
                "KRASIS_VRAM_MONITOR_POLL_MS must be an integer number of milliseconds"
            ) from exc
        if not 1 <= _vram_poll_ms <= 1000:
            raise SystemExit(
                "KRASIS_VRAM_MONITOR_POLL_MS must be between 1 and 1000 milliseconds"
            )
        _dim(f"VRAM monitor poll override: {_vram_poll_ms} ms")
    _vram_startup_poll_ms = min(_vram_poll_ms, VRAM_CALIBRATION_POLL_INTERVAL_MS)
    vram_monitor = VramMonitor(
        device_indices,
        poll_interval_ms=_vram_startup_poll_ms,
        safety_margin_mb=SAFETY_MARGIN_MB,
    )
    vram_monitor.start()
    _dim(
        "VRAM monitor started "
        f"(startup measurement {_vram_startup_poll_ms} ms; runtime {_vram_poll_ms} ms)"
    )
    for idx in device_indices:
        total = vram_monitor.total_mb(idx)
        _dim(f"cuda:{idx}: {total:,} MB total")

    # ── Enable VRAM report if requested ──
    if getattr(args, 'vram_report', False):
        vram_monitor.enable_report()
        _dim("VRAM report enabled (periodic samples + events)")

    vram_monitor.report_event("model_loaded")

    # ── Phase 1: Warmup (trigger lazy Rust prefill allocations before HCS loading) ──
    # Run one real Rust prefill before budget measurement so first-request allocations
    # are not charged against the post-HCS steady-state budget.
    _model._hcs_device = None
    _model._multi_gpu_hcs = False
    _status("Pre-allocating Rust prefill engine")
    # Must happen BEFORE VRAM budget measurement and HCS pool loading.
    # The prefill engine allocates fixed scratch buffers on the GPU.
    # If we wait until after HCS, there may not be enough VRAM left.
    try:
        gpu_store.allocate_prefill_engine(_model.cfg.max_position_embeddings)
        _detail("Prefill engine scratch buffers allocated")
        log_ram_ledger("after-prefill-engine-allocate")
        if _vram_ledger_enabled():
            _model.log_vram_ledger_residency("after-prefill-engine-allocate")
    except Exception as e:
        logger.error("Failed to pre-allocate prefill engine: %s", e)
        raise

    _status("Warmup (Rust prefill engine)")
    vram_monitor.report_event("warmup_start")
    t_warmup = time.time()
    try:
        warmup_token = 0
        warmup_probe = _model.tokenizer.encode(" hello") if _model.tokenizer is not None else []
        if warmup_probe:
            warmup_token = int(warmup_probe[0])
        warmup_len_default = min(25000, max(256, int(getattr(args, "gpu_prefill_threshold", 300))))
        startup_diag = _startup_diag_enabled()
        warmup_len = (
            _env_int("KRASIS_STARTUP_WARMUP_TOKENS", warmup_len_default, minimum=1)
            if startup_diag else warmup_len_default
        )
        if startup_diag:
            prefill_debug_enabled = os.environ.get("KRASIS_PREFILL_DEBUG", "") == "1"
            prefill_timing_enabled = _env_flag("KRASIS_PREFILL_TIMING")
            exit_after_calibration = _env_flag("KRASIS_STARTUP_EXIT_AFTER_CALIBRATION")
            _detail(
                f"Startup diag: warmup tokens={warmup_len:,} "
                f"(default {warmup_len_default:,}, token={warmup_token})"
            )
            logger.info(
                "Startup diag warmup: tokens=%d default_tokens=%d token_id=%d",
                warmup_len, warmup_len_default, warmup_token,
            )
            no_graph = _env_flag("KRASIS_NO_GRAPH")
            mapped_reads = _env_flag("KRASIS_MAPPED_READS")
            _detail(
                "Startup diag env: "
                f"no_graph={no_graph if no_graph is not None else 'unset'}, "
                f"mapped_reads={mapped_reads if mapped_reads is not None else 'unset'}, "
                f"prefill_debug={prefill_debug_enabled}, "
                f"prefill_timing={prefill_timing_enabled if prefill_timing_enabled is not None else 'unset'}, "
                f"exit_after_calibration={exit_after_calibration if exit_after_calibration is not None else 'unset'}"
            )
            logger.info(
                "Startup diag env: no_graph=%s mapped_reads=%s prefill_debug=%s "
                "prefill_timing=%s exit_after_calibration=%s",
                "unset" if no_graph is None else int(no_graph),
                "unset" if mapped_reads is None else int(mapped_reads),
                int(prefill_debug_enabled),
                "unset" if prefill_timing_enabled is None else int(prefill_timing_enabled),
                "unset" if exit_after_calibration is None else int(exit_after_calibration),
            )
        gpu_store.rust_prefill_tokens(
            [warmup_token] * warmup_len,
            0.0,
            disable_pinning=True,
        )
        _model.server_cleanup()
        _detail(f"Rust prefill warmed with {warmup_len:,} tokens before HCS budgeting")
    except Exception as e:
        _abort_if_cuda_context_poisoned("Rust prefill warmup", e)
        if (
            _env_flag("KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V5")
            or _env_flag("KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V5_ORACLE")
            or _env_flag("KRASIS_MAMBA2_SSD_BLOCK_SCAN")
            or _env_flag("KRASIS_MAMBA2_SSD_BLOCK_SCAN_ORACLE")
            or _env_flag("KRASIS_MAMBA2_SSD_BLOCK_SCAN_OUTPUT_SUBLOOP_TIMING")
            or _env_flag("KRASIS_MAMBA2_SSD_BLOCK_SCAN_COEFF_TILE")
            or _env_flag("KRASIS_MAMBA2_SSD_BLOCK_SCAN_CANDIDATE_TIMING")
            or _env_flag("KRASIS_MAMBA2_SSD_BLOCK_SCAN_RECURRENT_SUBLOOP_TIMING")
            or _env_flag("KRASIS_MAMBA2_SSD_BLOCK_SCAN_STATE_PARALLEL_RECURRENT")
            or _env_flag("KRASIS_MAMBA2_SSD_PARALLEL_CHUNKED")
            or _nemotron_default_optimizations_enabled(_model)
        ):
            raise
        logger.warning("Rust prefill warmup failed, continuing without it: %s", e)
        _warn(f"Rust prefill warmup failed: {e}")
    warmup_elapsed = time.time() - t_warmup
    _detail(f"Warmup complete in {warmup_elapsed:.1f}s")
    vram_monitor.report_event("warmup_end")
    if _vram_ledger_enabled():
        _model.log_vram_ledger_residency("after-rust-prefill-warmup")

    # Log warmup VRAM impact before resetting
    for idx in device_indices:
        warmup_min_free = vram_monitor.min_free_mb(idx)
        warmup_peak_used = vram_monitor.peak_used_mb(idx)
        total = vram_monitor.total_mb(idx)
        _dim(f"cuda:{idx} warmup:  peak {warmup_peak_used:,} MB used / {total:,} MB total  (min free: {warmup_min_free:,} MB)")
        logger.info(
            "VRAM warmup cuda:%d: peak_used=%d MB, min_free=%d MB, total=%d MB",
            idx, warmup_peak_used, warmup_min_free, total,
        )
        _require_startup_vram_floor(
            f"Rust prefill warmup cuda:{idx}",
            int(warmup_min_free),
            SAFETY_MARGIN_MB,
        )

    # ── Phase 2: VRAM calibration ──
    # Measure real short/long prefill and decode VRAM minima with no HCS loaded,
    # then apply those transient deltas to the current post-calibration free VRAM.
    # This restores the measured decode model and avoids guessed headroom values.
    dev_idx = devices[0].index
    import torch
    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")

    _status("VRAM calibration")
    vram_monitor.report_event("calibration_start")
    _detail(f"Safety margin: {SAFETY_MARGIN_MB:,} MB")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    _free_mb = torch.cuda.mem_get_info(dev_idx)[0] // (1024 * 1024)
    _total_mb = torch.cuda.mem_get_info(dev_idx)[1] // (1024 * 1024)
    _detail(f"Free VRAM after startup: {_free_mb:,} MB / {_total_mb:,} MB")
    logger.info("VRAM calibration baseline: free=%d MB, total=%d MB", _free_mb, _total_mb)
    vram_ledger = _vram_ledger_enabled()
    if vram_ledger:
        _model.log_vram_ledger_residency("calibration-baseline")
        logger.info(
            "VRAM LEDGER startup_baseline device=%d free_mb=%d total_mb=%d torch_allocated_mb=%d torch_reserved_mb=%d safety_mb=%d kv_max_tokens=%d max_position_embeddings=%d",
            dev_idx,
            _free_mb,
            _total_mb,
            torch.cuda.memory_allocated(dev_idx) // (1024 * 1024),
            torch.cuda.memory_reserved(dev_idx) // (1024 * 1024),
            SAFETY_MARGIN_MB,
            _kv_cache_max_tokens(_model),
            _model.cfg.max_position_embeddings,
        )

    max_calibration_tokens = max(1, min(
        _kv_cache_max_tokens(_model),
        _model.cfg.max_position_embeddings,
        STARTUP_CALIBRATION_LONG_TOKENS_CAP,
    ) - 100)
    short_target_default = min(STARTUP_CALIBRATION_SHORT_TOKENS, max_calibration_tokens)
    long_target_default = max(short_target_default, min(max_calibration_tokens, int(max_calibration_tokens * 0.8)))
    startup_diag = _startup_diag_enabled()
    short_target = short_target_default
    long_target = long_target_default
    forced_long_target = False
    if startup_diag:
        short_target = min(max_calibration_tokens, _env_int(
            "KRASIS_STARTUP_CAL_SHORT_TOKENS", short_target_default, minimum=1
        ))
        if os.environ.get("KRASIS_STARTUP_CAL_LONG_TOKENS", "").strip():
            forced_long_target = True
            long_target = min(max_calibration_tokens, _env_int(
                "KRASIS_STARTUP_CAL_LONG_TOKENS", long_target_default, minimum=1
            ))
    long_target = max(short_target, long_target)
    if startup_diag:
        _detail(
            "Startup diag: calibration tokens "
            f"short={short_target:,} (default {short_target_default:,}), "
            f"long={long_target:,} (default {long_target_default:,}), "
            f"cap={max_calibration_tokens:,}, forced_long={forced_long_target}"
        )
        logger.info(
            "Startup diag calibration: short_tokens=%d default_short=%d long_tokens=%d "
            "default_long=%d cap=%d decode_tokens=%d forced_long=%s",
            short_target, short_target_default, long_target, long_target_default,
            max_calibration_tokens, STARTUP_CALIBRATION_DECODE_TOKENS, forced_long_target,
        )
    calibration_stop_ids: list[int] = []
    calibration_guard_margin_mb = _startup_calibration_long_floor_mb(SAFETY_MARGIN_MB)
    previous_prefill_margin_mb = int(
        gpu_store.set_prefill_safety_margin_mb(calibration_guard_margin_mb)
    )
    _detail(
        "Calibration prefill guard: "
        f"{calibration_guard_margin_mb:,} MB "
        f"(runtime safety remains {SAFETY_MARGIN_MB:,} MB)"
    )
    logger.info(
        "Startup calibration prefill guard: previous=%d MB guard=%d MB runtime=%d MB",
        previous_prefill_margin_mb,
        calibration_guard_margin_mb,
        SAFETY_MARGIN_MB,
    )

    def _measure_vram_probe(label: str, prompt_tokens: list[int]) -> tuple[int, int, int, int, int]:
        _detail(f"{label}: probing {len(prompt_tokens):,} prompt tokens + {STARTUP_CALIBRATION_DECODE_TOKENS} decode tokens")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        _model.server_cleanup()
        time.sleep(0.1)

        baseline_free = int(vram_monitor.current_free_mb(dev_idx))

        vram_monitor.reset_min_free()
        t_prefill = time.time()
        first_token, prompt_len, kv_overflow = gpu_store.rust_prefill_tokens(
            prompt_tokens,
            temperature=0.0,
            disable_pinning=True,
        )
        torch.cuda.synchronize()
        time.sleep(0.1)
        prefill_elapsed = time.time() - t_prefill
        prefill_min_free = int(vram_monitor.min_free_mb(dev_idx))
        prefill_post_alloc_free = int(gpu_store.get_last_prefill_post_alloc_free_mb())
        gpu_store.update_measured_prefill_runtime_overhead_mb(
            prefill_post_alloc_free, prefill_min_free
        )
        if kv_overflow:
            _model.server_cleanup()
            raise RuntimeError(f"{label} VRAM calibration overflowed KV cache at {prompt_len:,} tokens")

        vram_monitor.reset_min_free()
        t_decode = time.time()
        gpu_store.gpu_generate_stream_probe(
            tokenizer_path=tokenizer_path,
            first_token=first_token,
            start_position=prompt_len,
            max_tokens=STARTUP_CALIBRATION_DECODE_TOKENS,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            stop_ids=calibration_stop_ids,
            presence_penalty=0.0,
        )
        torch.cuda.synchronize()
        time.sleep(0.1)
        decode_elapsed = time.time() - t_decode
        decode_min_free = int(gpu_store.get_last_min_free_vram_mb())

        _model.server_cleanup()
        torch.cuda.synchronize()
        time.sleep(0.1)
        post_cleanup_free = int(vram_monitor.current_free_mb(dev_idx))

        _detail(
            f"{label}: baseline={baseline_free:,} MB, "
            f"prefill post-alloc={prefill_post_alloc_free:,} MB, "
            f"prefill min={prefill_min_free:,} MB, decode min={decode_min_free:,} MB, "
            f"post-cleanup={post_cleanup_free:,} MB"
        )
        _require_startup_vram_floor(
            f"{label} prefill",
            prefill_min_free,
            SAFETY_MARGIN_MB,
        )
        _require_startup_vram_floor(
            f"{label} decode",
            decode_min_free,
            SAFETY_MARGIN_MB,
        )
        if startup_diag:
            prefill_tps = prompt_len / prefill_elapsed if prefill_elapsed > 0 else 0.0
            decode_tps = STARTUP_CALIBRATION_DECODE_TOKENS / decode_elapsed if decode_elapsed > 0 else 0.0
            _detail(
                f"{label}: prefill {prefill_elapsed:.2f}s ({prefill_tps:.1f} tok/s), "
                f"decode {decode_elapsed:.2f}s ({decode_tps:.1f} tok/s)"
            )
            logger.info(
                "Startup diag probe %s: prompt_len=%d baseline_free=%d prefill_post_alloc=%d prefill_min=%d "
                "decode_min=%d post_cleanup=%d prefill_s=%.3f prefill_tps=%.2f "
                "decode_s=%.3f decode_tps=%.2f",
                label, prompt_len, baseline_free, prefill_post_alloc_free, prefill_min_free, decode_min_free,
                post_cleanup_free, prefill_elapsed, prefill_tps, decode_elapsed, decode_tps,
            )
        if vram_ledger:
            logger.info(
                "VRAM LEDGER calibration_probe label=%s prompt_len=%d baseline_free_mb=%d prefill_post_alloc_mb=%d prefill_min_mb=%d decode_min_mb=%d post_cleanup_mb=%d prefill_delta_mb=%d remaining_after_scratch_mb=%d decode_delta_mb=%d prefill_s=%.3f decode_s=%.3f",
                label,
                prompt_len,
                baseline_free,
                prefill_post_alloc_free,
                prefill_min_free,
                decode_min_free,
                post_cleanup_free,
                max(0, baseline_free - prefill_min_free),
                max(0, prefill_post_alloc_free - prefill_min_free),
                max(0, baseline_free - decode_min_free),
                prefill_elapsed,
                decode_elapsed,
            )
        return prompt_len, baseline_free, prefill_post_alloc_free, prefill_min_free, decode_min_free

    short_prompt = _make_startup_calibration_prompts(_model, [short_target])[0]
    short_tokens, short_baseline_free, short_prefill_post_alloc, short_prefill_min, short_decode_min = _measure_vram_probe(
        "Short calibration", short_prompt
    )

    adaptive_floor_mb = _startup_calibration_long_floor_mb(SAFETY_MARGIN_MB)
    estimated_prefill_mb_per_token = _startup_calibration_estimated_prefill_mb_per_token(
        _model,
        gpu_store,
        args.kv_dtype,
        short_tokens,
        long_target,
    )
    observed_prefill_mins: list[tuple[int, int]] = [(short_tokens, short_prefill_min)]
    long_tokens = short_tokens
    long_baseline_free = short_baseline_free
    long_prefill_post_alloc = short_prefill_post_alloc
    long_prefill_min = short_prefill_min
    long_decode_min = short_decode_min

    if forced_long_target:
        long_prompt = _make_startup_calibration_prompts(_model, [long_target])[0]
        long_tokens, long_baseline_free, long_prefill_post_alloc, long_prefill_min, long_decode_min = _measure_vram_probe(
            "Long calibration", long_prompt
        )
    else:
        _detail(
            f"Adaptive long calibration: default target={long_target:,} tokens, "
            f"target low-water floor={adaptive_floor_mb:,} MB, "
            f"estimated prefill growth={estimated_prefill_mb_per_token:.4f} MB/token"
        )
        logger.info(
            "Adaptive startup calibration: default_long=%d target_floor=%d MB estimated_prefill_mb_per_token=%.6f",
            long_target, adaptive_floor_mb, estimated_prefill_mb_per_token,
        )
        while True:
            current_probe_tokens = observed_prefill_mins[-1][0]
            bounded_probe_tokens = min(
                long_target,
                max(
                    current_probe_tokens + 1,
                    current_probe_tokens * 2,
                    short_tokens * max(1, STARTUP_CALIBRATION_LONG_INITIAL_MULTIPLIER),
                ),
            )
            fail_closed_probe_tokens: Optional[int] = None
            if bounded_probe_tokens > current_probe_tokens:
                probe_entry_floor_mb = int(
                    gpu_store.prefill_minimum_entry_free_mb(bounded_probe_tokens)
                )
                probe_available_mb = int(long_baseline_free)
                if probe_entry_floor_mb <= probe_available_mb:
                    fail_closed_probe_tokens = bounded_probe_tokens
                logger.info(
                    "Adaptive startup calibration fail-closed probe: current_tokens=%d "
                    "next_tokens=%d entry_floor_mb=%d available_mb=%d admitted=%s",
                    current_probe_tokens,
                    bounded_probe_tokens,
                    probe_entry_floor_mb,
                    probe_available_mb,
                    fail_closed_probe_tokens is not None,
                )
                _detail(
                    "Adaptive long calibration fail-closed entry check: "
                    f"{bounded_probe_tokens:,} tokens require "
                    f"{probe_entry_floor_mb:,} MB, "
                    f"{probe_available_mb:,} MB available — "
                    f"{'admitted' if fail_closed_probe_tokens is not None else 'not admitted'}"
                )
            next_target, reason = _next_startup_calibration_probe_target(
                short_tokens=short_tokens,
                default_long_tokens=long_target,
                observed_prefill_mins=observed_prefill_mins,
                target_floor_mb=adaptive_floor_mb,
                estimated_prefill_mb_per_token=estimated_prefill_mb_per_token,
                fail_closed_probe_tokens=fail_closed_probe_tokens,
                runtime_safety_floor_mb=SAFETY_MARGIN_MB,
            )
            if next_target is None:
                _detail(
                    f"Adaptive long calibration: using {long_tokens:,} tokens "
                    f"({reason})"
                )
                logger.info(
                    "Adaptive startup calibration stopped: long_tokens=%d reason=%s",
                    long_tokens, reason,
                )
                break

            _detail(
                f"Adaptive long calibration: next probe {next_target:,} tokens "
                f"({reason})"
            )
            logger.info(
                "Adaptive startup calibration probe: next_tokens=%d reason=%s",
                next_target, reason,
            )
            long_prompt = _make_startup_calibration_prompts(_model, [next_target])[0]
            previous_long_tokens = long_tokens
            long_tokens, long_baseline_free, long_prefill_post_alloc, long_prefill_min, long_decode_min = _measure_vram_probe(
                "Long calibration", long_prompt
            )
            observed_prefill_mins.append((long_tokens, long_prefill_min))

            if long_tokens <= previous_long_tokens:
                _detail(
                    f"Adaptive long calibration: prompt source stopped growing at "
                    f"{long_tokens:,} tokens"
                )
                logger.info(
                    "Adaptive startup calibration prompt source stopped growing: requested=%d actual=%d previous=%d",
                    next_target, long_tokens, previous_long_tokens,
                )
                break
            if long_tokens >= long_target:
                break
            if long_prefill_min <= adaptive_floor_mb and long_tokens >= long_target:
                _detail(
                    f"Adaptive long calibration: reached low-water floor at "
                    f"{long_prefill_min:,} MB; using {long_tokens:,} tokens"
                )
                logger.info(
                    "Adaptive startup calibration reached floor: tokens=%d min_free=%d floor=%d",
                    long_tokens, long_prefill_min, adaptive_floor_mb,
                )
                break

    restored_prefill_margin_mb = int(
        gpu_store.set_prefill_safety_margin_mb(SAFETY_MARGIN_MB)
    )
    logger.info(
        "Startup calibration prefill guard restored: previous=%d MB runtime=%d MB",
        restored_prefill_margin_mb,
        SAFETY_MARGIN_MB,
    )

    # Compute max scratch: worst case prompt (50K tokens).
    # This determines how much VRAM prefill can claim via soft eviction.
    max_scratch_tokens = min(50000, _model.cfg.max_position_embeddings)
    max_scratch_mb = gpu_store.prefill_scratch_reservation_mb(max_scratch_tokens)
    _detail(f"Scratch reservation: max={max_scratch_mb:,} MB (at {max_scratch_tokens:,} tokens)")
    if vram_ledger:
        logger.info(
            "VRAM LEDGER scratch_reservation max_scratch_tokens=%d max_scratch_mb=%d short_tokens=%d long_tokens=%d",
            max_scratch_tokens,
            max_scratch_mb,
            short_tokens,
            long_tokens,
        )

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    _model.server_cleanup()
    time.sleep(0.1)
    post_calibration_free_mb = int(vram_monitor.current_free_mb(dev_idx))

    short_prefill_delta = max(0, short_baseline_free - short_prefill_min)
    long_prefill_delta = max(0, long_baseline_free - long_prefill_min)
    short_decode_delta = max(0, short_baseline_free - short_decode_min)
    long_decode_delta = max(0, long_baseline_free - long_decode_min)

    prefill_short_free = max(0, post_calibration_free_mb - short_prefill_delta)
    prefill_long_free = max(0, post_calibration_free_mb - long_prefill_delta)
    decode_short_free = max(0, post_calibration_free_mb - short_decode_delta)
    decode_long_free = max(0, post_calibration_free_mb - long_decode_delta)

    # GPU0 HCS is fully reclaimable. Prefill and decode are separate runtime
    # stages: the decode-resident HCS pool is sized from measured decode VRAM,
    # while the prefill path evicts soft HCS back to the measured prefill floor
    # before allocating scratch for each request.
    decode_hcs_budget = max(0, decode_short_free - SAFETY_MARGIN_MB)
    prefill_short_hcs_budget = max(0, prefill_short_free - SAFETY_MARGIN_MB)
    prefill_long_hcs_budget = max(0, prefill_long_free - SAFETY_MARGIN_MB)

    vram_monitor.report_event("calibration_end")
    previous_poll_interval_ms = int(vram_monitor.set_poll_interval_ms(_vram_poll_ms))
    _detail(
        "VRAM monitor runtime cadence restored: "
        f"{previous_poll_interval_ms} ms -> {_vram_poll_ms} ms"
    )
    _status("VRAM calibration complete")
    _detail(f"Post-calibration free VRAM: {post_calibration_free_mb:,} MB")
    log_ram_ledger("after-vram-calibration")
    _detail(
        f"Transient deltas: short prefill={short_prefill_delta:,} MB, "
        f"long prefill={long_prefill_delta:,} MB, short decode={short_decode_delta:,} MB, "
        f"long decode={long_decode_delta:,} MB"
    )
    _detail(f"GPU0 decode HCS budget: {decode_hcs_budget:,} MB")
    _detail(
        f"Prefill HCS budgets: short={prefill_short_hcs_budget:,} MB, "
        f"long={prefill_long_hcs_budget:,} MB"
    )
    _detail(f"Worst-case prefill scratch reservation: {max_scratch_mb:,} MB at {max_scratch_tokens:,} tokens")
    logger.info(
        "VRAM calibration: post_free=%d MB, short=%dtok, long=%dtok, gpu0_decode_hcs=%d MB, prefill_short_hcs=%d MB, prefill_long_hcs=%d MB, max_scratch=%d MB",
        post_calibration_free_mb, short_tokens, long_tokens, decode_hcs_budget,
        prefill_short_hcs_budget, prefill_long_hcs_budget, max_scratch_mb,
    )
    if vram_ledger:
        logger.info(
            "VRAM LEDGER calibration_summary post_free_mb=%d short_tokens=%d long_tokens=%d short_prefill_delta_mb=%d long_prefill_delta_mb=%d short_decode_delta_mb=%d long_decode_delta_mb=%d prefill_short_free_mb=%d prefill_long_free_mb=%d decode_short_free_mb=%d decode_long_free_mb=%d decode_hcs_budget_mb=%d prefill_short_hcs_budget_mb=%d prefill_long_hcs_budget_mb=%d safety_mb=%d max_scratch_mb=%d",
            post_calibration_free_mb,
            short_tokens,
            long_tokens,
            short_prefill_delta,
            long_prefill_delta,
            short_decode_delta,
            long_decode_delta,
            prefill_short_free,
            prefill_long_free,
            decode_short_free,
            decode_long_free,
            decode_hcs_budget,
            prefill_short_hcs_budget,
            prefill_long_hcs_budget,
            SAFETY_MARGIN_MB,
            max_scratch_mb,
        )
    if startup_diag and _env_flag("KRASIS_STARTUP_EXIT_AFTER_CALIBRATION"):
        _status("Startup diagnostic exit")
        _detail("Exiting after VRAM calibration by request")
        logger.info("Startup diagnostic exit after VRAM calibration requested")
        return

    if args.approved_heatmap_build_out:
        approved_heatmap_residency_calibration = {
            "short_tokens": int(short_tokens),
            "long_tokens": int(long_tokens),
            "prefill_short_free_mb": int(prefill_short_free),
            "prefill_long_free_mb": int(prefill_long_free),
            "decode_short_free_mb": int(decode_short_free),
            "decode_long_free_mb": int(decode_long_free),
            "baseline_free_mb": int(post_calibration_free_mb),
            "safety_margin_mb": int(SAFETY_MARGIN_MB),
            "short_prefill_post_alloc_free_mb": int(short_prefill_post_alloc),
            "long_prefill_post_alloc_free_mb": int(long_prefill_post_alloc),
            "decode_hcs_budget_mb": int(decode_hcs_budget),
        }
        _build_approved_heatmap(
            _model,
            args.approved_heatmap_build_out,
            args,
            approved_heatmap_residency_calibration,
        )
        vram_monitor.report_event("approved_heatmap_build_complete")
        return

    # ── Pre-compute multi-GPU layer splits (before HCS, so we can filter rankings) ──
    # Splits are based on total HCS budget (hard+soft) on each GPU, so layers
    # are proportional to where experts can actually live.
    # _multi_gpu_splits: list of split points (one per aux GPU). Empty = single GPU.
    # _multi_gpu_gqa_offsets: GQA count before each split point.
    _multi_gpu_splits = []
    _multi_gpu_gqa_offsets = []
    # Legacy compat
    _multi_gpu_split = 0
    _multi_gpu_gqa_offset = 0
    _layer_split_plan = None
    _multi_gpu_selected_mode = "single"
    _peer_store = None
    _peer_startup = None
    _expert_compression_calibration = None
    if num_gpus_available > 1 and args.hcs:
        _status(f"Computing multi-GPU layer split ({num_gpus_available} GPUs)")
        num_layers = len(_model.layers)

        # Compute per-layer VRAM cost from actual loaded weights (not hardcoded estimates).
        _layer_vram_mb = []
        _layer_service_bytes = []
        kv_cache = _model.kv_caches[0] if _model.kv_caches else None
        kv_total_mb = 0
        if kv_cache is not None:
            # k_cache and v_cache are single tensors (not lists); compute sizes individually
            # to avoid FP8 tensor addition (ufunc_add not implemented for Float8_e4m3fn)
            kv_total_bytes = 0
            for cache_tensor in (kv_cache.k_cache, kv_cache.v_cache):
                if isinstance(cache_tensor, list):
                    for layer_tensor in cache_tensor:
                        kv_total_bytes += layer_tensor.nelement() * layer_tensor.element_size()
                elif cache_tensor is not None:
                    kv_total_bytes += cache_tensor.nelement() * cache_tensor.element_size()
            kv_total_mb = kv_total_bytes / (1024 * 1024)
            # num_layers is the first dim of the cache tensor
            if isinstance(kv_cache.k_cache, list):
                num_kv_layers = len(kv_cache.k_cache)
            else:
                num_kv_layers = kv_cache.k_cache.shape[0] if kv_cache.k_cache is not None else 0
            kv_per_layer_mb = kv_total_mb / num_kv_layers if num_kv_layers > 0 else 0
        else:
            kv_per_layer_mb = 0

        from krasis.attention import MarlinWeight as _MW
        for layer_index, layer in enumerate(_model.layers):
            layer_bytes = 0
            layer_bytes += layer.input_norm_weight.nelement() * layer.input_norm_weight.element_size()
            layer_bytes += layer.post_attn_norm_weight.nelement() * layer.post_attn_norm_weight.element_size()
            attn = layer.attention
            for attr_name in dir(attn):
                val = getattr(attn, attr_name, None)
                if isinstance(val, _MW):
                    # MarlinWeight: count packed + scales GPU tensors
                    if val.packed.is_cuda:
                        layer_bytes += val.packed.nelement() * val.packed.element_size()
                    if val.scales.is_cuda:
                        layer_bytes += val.scales.nelement() * val.scales.element_size()
                elif isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                    layer_bytes += val.nelement() * val.element_size()
                elif isinstance(val, tuple) and len(val) == 2:
                    for t in val:
                        if isinstance(t, torch.Tensor) and t.device.type == 'cuda':
                            layer_bytes += t.nelement() * t.element_size()
            if layer.is_moe:
                for w in [layer.gate_weight, layer.gate_bias, layer.e_score_correction_bias]:
                    if w is not None:
                        layer_bytes += w.nelement() * w.element_size()
                if layer.shared_expert is not None:
                    for v in layer.shared_expert.values():
                        if isinstance(v, torch.Tensor):
                            layer_bytes += v.nelement() * v.element_size()
                        elif isinstance(v, tuple):
                            for t in v:
                                if isinstance(t, torch.Tensor):
                                    layer_bytes += t.nelement() * t.element_size()
            layer_mb = layer_bytes / (1024 * 1024)
            if layer.layer_type != "linear_attention":
                layer_mb += kv_per_layer_mb
            _layer_vram_mb.append(layer_mb)
            layer_service_bytes = layer_bytes + (
                int(kv_per_layer_mb * 1024 * 1024)
                if layer.layer_type != "linear_attention"
                else 0
            )
            if cfg.is_dsa:
                layer_service_bytes += int(
                    _model._dsa_indexer_resource_bytes_for_segment(
                        layer_index, layer_index + 1
                    )
                )
            _layer_service_bytes.append(layer_service_bytes)

        # Compute base overhead for the last aux GPU (embedding + lm_head + final_norm)
        # Only the last GPU needs the LM head; intermediate GPUs only need attention/norms.
        last_gpu_base_overhead_bytes = 0
        last_gpu_base_overhead_bytes += _model.embedding.nelement() * _model.embedding.element_size()
        last_gpu_base_overhead_bytes += _model.final_norm.nelement() * _model.final_norm.element_size()
        if isinstance(_model.lm_head_data, tuple):
            for t in _model.lm_head_data:
                if isinstance(t, torch.Tensor):
                    last_gpu_base_overhead_bytes += t.nelement() * t.element_size()
        elif isinstance(_model.lm_head_data, torch.Tensor):
            last_gpu_base_overhead_bytes += _model.lm_head_data.nelement() * _model.lm_head_data.element_size()
        last_gpu_base_overhead = last_gpu_base_overhead_bytes / (1024 * 1024)

        # Compute HCS budget for each GPU.
        # GPU0: decode-resident HCS budget. Soft HCS is evicted per request
        # to satisfy measured prefill requirements.
        # Aux GPUs: total VRAM - attention cost - overhead - safety margin.
        # We iterate to find self-consistent splits where each GPU's layer assignment
        # matches its available HCS budget proportionally.
        gpu0_hcs_total = decode_hcs_budget
        num_aux = num_gpus_available - 1
        aux_totals = [vram_monitor.total_mb(device_indices[i + 1]) for i in range(num_aux)]

        # Initial guess: equal distribution
        initial_splits = [int(round(num_layers * (i + 1) / num_gpus_available))
                          for i in range(num_aux)]
        # Clamp each split to [2, num_layers - 2] and ensure monotonic
        for i in range(len(initial_splits)):
            initial_splits[i] = max(2, min(initial_splits[i], num_layers - 2))
        for i in range(1, len(initial_splits)):
            initial_splits[i] = max(initial_splits[i], initial_splits[i - 1] + 1)

        _multi_gpu_splits = list(initial_splits)
        gpu_hcs_budgets = [0.0] * num_gpus_available

        for _iter in range(5):
            prev_splits = list(_multi_gpu_splits)

            # Compute boundaries: [0, splits[0], splits[1], ..., num_layers]
            boundaries = [0] + _multi_gpu_splits + [num_layers]

            # Compute HCS budget for each GPU
            gpu_hcs_budgets[0] = gpu0_hcs_total
            for i in range(num_aux):
                gpu_idx_in_list = i + 1
                layer_start = boundaries[gpu_idx_in_list]
                layer_end_b = boundaries[gpu_idx_in_list + 1]
                attn_cost = sum(_layer_vram_mb[j] for j in range(layer_start, layer_end_b))
                dsa_resource_cost = (
                    _model._dsa_indexer_resource_bytes_for_segment(
                        layer_start, layer_end_b
                    )
                    / (1024 * 1024)
                    if cfg.is_dsa
                    else 0
                )
                # Last aux GPU has LM head overhead
                base_overhead = last_gpu_base_overhead if (i + 1 == num_aux) else 0
                gpu_hcs_budgets[gpu_idx_in_list] = max(0,
                    aux_totals[i] - base_overhead - attn_cost
                    - dsa_resource_cost - SAFETY_MARGIN_MB)

            total_hcs = sum(gpu_hcs_budgets)
            if total_hcs <= 0:
                break

            # Redistribute layers proportionally to HCS budgets
            cumulative = 0.0
            new_splits = []
            for i in range(num_aux):
                cumulative += gpu_hcs_budgets[i]
                split_pos = int(round(num_layers * cumulative / total_hcs))
                split_pos = max(2, min(split_pos, num_layers - 2))
                new_splits.append(split_pos)

            # Ensure monotonic increasing
            for i in range(1, len(new_splits)):
                new_splits[i] = max(new_splits[i], new_splits[i - 1] + 1)

            _multi_gpu_splits = new_splits
            if _multi_gpu_splits == prev_splits:
                break  # converged

        # The capacity-derived result above remains the reference assignment.
        # Measure the selected cards now and change it only when the predicted
        # serial-time reduction clears the observed calibration tail spread.
        from krasis.multi_gpu_planner import (
            measure_device_service_profiles,
            optimize_contiguous_splits,
        )

        expert_component_bytes = [int(value) for value in gpu_store.expert_component_bytes]
        expert_payload_bytes = sum(expert_component_bytes)
        if expert_payload_bytes <= 0:
            raise RuntimeError(
                "Multi-GPU service calibration could not obtain the loaded routed-expert payload size"
            )
        _status("Calibrating multi-GPU device service rates")
        _multi_gpu_service_profiles = measure_device_service_profiles(
            device_indices,
            expert_component_bytes,
            max(
                max(_layer_service_bytes),
                expert_payload_bytes * int(cfg.num_experts_per_tok),
            ),
        )
        for logical_gpu, profile in enumerate(_multi_gpu_service_profiles):
            _detail(
                f"  GPU{logical_gpu} cuda:{profile.gpu_index}: "
                f"H2D p50/p95={profile.h2d_p50_us:.1f}/{profile.h2d_p95_us:.1f} us "
                f"({1.0 / profile.h2d_seconds_per_byte / 1e9:.2f} GB/s), "
                f"D2D p50/p95={profile.d2d_p50_us:.1f}/{profile.d2d_p95_us:.1f} us "
                f"({1.0 / profile.d2d_seconds_per_byte / 1e9:.2f} GB/s)"
            )
            logger.info(
                "Multi-GPU service calibration gpu=%d cuda=%d probe_bytes=%d "
                "d2d_probe_bytes=%d h2d_p50_us=%.6f h2d_p95_us=%.6f h2d_gbps=%.6f "
                "d2d_p50_us=%.6f d2d_p95_us=%.6f d2d_gbps=%.6f uncertainty=%.6f",
                logical_gpu,
                profile.gpu_index,
                profile.probe_bytes,
                profile.d2d_probe_bytes,
                profile.h2d_p50_us,
                profile.h2d_p95_us,
                1.0 / profile.h2d_seconds_per_byte / 1e9,
                profile.d2d_p50_us,
                profile.d2d_p95_us,
                1.0 / profile.d2d_seconds_per_byte / 1e9,
                profile.relative_uncertainty,
            )

        preferred_splits = tuple(_multi_gpu_splits)

        def _candidate_hcs_budget_bytes(gpu_ordinal, layer_start, layer_end):
            if gpu_ordinal == 0:
                return int(gpu0_hcs_total * 1024 * 1024)
            aux_index = gpu_ordinal - 1
            attention_bytes = int(sum(_layer_vram_mb[layer_start:layer_end]) * 1024 * 1024)
            dsa_bytes = (
                int(_model._dsa_indexer_resource_bytes_for_segment(layer_start, layer_end))
                if cfg.is_dsa
                else 0
            )
            terminal_overhead_bytes = (
                last_gpu_base_overhead_bytes
                if gpu_ordinal == num_gpus_available - 1
                else 0
            )
            total_bytes = int(aux_totals[aux_index] * 1024 * 1024)
            safety_bytes = int(SAFETY_MARGIN_MB * 1024 * 1024)
            return max(
                0,
                total_bytes
                - attention_bytes
                - dsa_bytes
                - terminal_overhead_bytes
                - safety_bytes,
            )

        layer_is_moe = [bool(layer.is_moe) for layer in _model.layers]
        split_plan = optimize_contiguous_splits(
            preferred_splits=preferred_splits,
            layer_resident_bytes=_layer_service_bytes,
            layer_is_moe=layer_is_moe,
            profiles=_multi_gpu_service_profiles,
            hcs_budget_bytes=_candidate_hcs_budget_bytes,
            expert_bytes=expert_payload_bytes,
            experts_per_layer=int(cfg.n_routed_experts),
            experts_per_token=int(cfg.num_experts_per_tok),
            terminal_bytes=int(last_gpu_base_overhead_bytes),
        )
        _multi_gpu_splits = list(split_plan.splits)
        _layer_split_plan = split_plan
        _multi_gpu_selected_mode = "layer-split"
        predicted_improvement = (
            (split_plan.preferred_seconds_per_token - split_plan.predicted_seconds_per_token)
            / split_plan.preferred_seconds_per_token
            * 100.0
            if split_plan.preferred_seconds_per_token > 0.0
            else 0.0
        )
        decision = "admitted" if split_plan.admitted else "retained within measurement uncertainty"
        _detail(
            f"Speed-aware split: reference={list(preferred_splits)} "
            f"({split_plan.preferred_seconds_per_token * 1000.0:.2f} ms/token predicted), "
            f"selected={_multi_gpu_splits} "
            f"({split_plan.predicted_seconds_per_token * 1000.0:.2f} ms/token, "
            f"{predicted_improvement:.2f}% reduction, {decision}, "
            f"uncertainty={split_plan.uncertainty_seconds * 1000.0:.2f} ms)"
        )
        logger.info(
            "Multi-GPU speed-aware split reference=%s selected=%s reference_ms=%.6f "
            "selected_ms=%.6f improvement_pct=%.6f uncertainty_ms=%.6f admitted=%s",
            list(preferred_splits),
            _multi_gpu_splits,
            split_plan.preferred_seconds_per_token * 1000.0,
            split_plan.predicted_seconds_per_token * 1000.0,
            predicted_improvement,
            split_plan.uncertainty_seconds * 1000.0,
            split_plan.admitted,
        )

        # Compute GQA offsets for each split point
        _multi_gpu_gqa_offsets = []
        for split in _multi_gpu_splits:
            gqa_count = 0
            for i in range(min(split, num_layers)):
                if _model.layers[i].layer_type != "linear_attention":
                    gqa_count += 1
            _multi_gpu_gqa_offsets.append(gqa_count)

        # Recompute final boundaries and total HCS for display
        boundaries = [0] + _multi_gpu_splits + [num_layers]
        # Recompute HCS budgets for final splits (needed when loop was skipped)
        gpu_hcs_budgets[0] = gpu0_hcs_total
        for i in range(num_aux):
            gpu_idx_in_list = i + 1
            layer_start = boundaries[gpu_idx_in_list]
            layer_end_b = boundaries[gpu_idx_in_list + 1]
            attn_cost = sum(_layer_vram_mb[j] for j in range(layer_start, layer_end_b))
            dsa_resource_cost = (
                _model._dsa_indexer_resource_bytes_for_segment(
                    layer_start, layer_end_b
                )
                / (1024 * 1024)
                if cfg.is_dsa
                else 0
            )
            base_overhead = last_gpu_base_overhead if (i + 1 == num_aux) else 0
            gpu_hcs_budgets[gpu_idx_in_list] = max(0,
                aux_totals[i] - base_overhead - attn_cost
                - dsa_resource_cost - SAFETY_MARGIN_MB)
        total_hcs = sum(gpu_hcs_budgets)
        _detail(f"HCS budgets: " + ", ".join(
            f"GPU{i} {gpu_hcs_budgets[i]:,.0f} MB" for i in range(num_gpus_available)
        ) + f" = {total_hcs:,.0f} MB total")
        for i in range(num_gpus_available):
            n_layers = boundaries[i + 1] - boundaries[i]
            ratio = gpu_hcs_budgets[i] / total_hcs if total_hcs > 0 else 0
            _detail(f"  GPU{i}: layers [{boundaries[i]}..{boundaries[i+1]}) = {n_layers} layers ({ratio:.1%})")
        logger.info("Multi-GPU splits: %s, gqa_offsets=%s", _multi_gpu_splits, _multi_gpu_gqa_offsets)

        # Legacy compat (used by HCS ranking filter below)
        _multi_gpu_split = _multi_gpu_splits[0] if _multi_gpu_splits else 0
        _multi_gpu_gqa_offset = _multi_gpu_gqa_offsets[0] if _multi_gpu_gqa_offsets else 0

    heatmap_timing_enabled = _heatmap_substage_timing_enabled()
    heatmap_to_ready_start_s = None
    cpu_tail_transposed_requested = os.environ.get(
        "KRASIS_CPU_TAIL_TRANSPOSED", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if cpu_tail_transposed_requested and not args.hcs:
        raise RuntimeError(
            "KRASIS_CPU_TAIL_TRANSPOSED=1 requires HCS so the startup resident "
            "set is known before building the complete non-resident duplicate tier"
        )

    if not args.hcs:
        _status("GPU decode (no HCS)")
        _warn("All experts streamed via DMA per token (slow for decode)")
    else:
        _status("Calculating HCS budget")

        # ── Device selection ──
        primary_dev = devices[0]
        total_experts = cfg.n_routed_experts * cfg.num_moe_layers

        heuristic_hcs_init = os.environ.get("KRASIS_HCS_HEURISTIC_INIT", "").strip().lower() in (
            "1", "true", "yes", "on"
        )
        dynamic_hcs = os.environ.get("KRASIS_DYNAMIC_HCS", "").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if heuristic_hcs_init:
            if not dynamic_hcs:
                raise RuntimeError(
                    "KRASIS_HCS_HEURISTIC_INIT=1 requires KRASIS_DYNAMIC_HCS=1; "
                    "heuristic startup fill is only intended for the dynamic recency cache."
                )
            if args.heatmap_path:
                raise RuntimeError(
                    "KRASIS_HCS_HEURISTIC_INIT=1 cannot be combined with --heatmap-path; "
                    "choose heuristic startup or an explicit validated heatmap."
                )
            _warn("Experimental HCS heuristic init enabled: skipping global heatmap build")
            ranking = []
        else:
            # ── Load approved heatmap or build quick startup heatmap ──
            # Approved heatmaps are route-prior artifacts: they must match the
            # model/router signature and a validated runtime, but local VRAM
            # calibration still decides how many experts can be resident.
            heatmap_prompts = _load_heatmap_prompts()
            _assert_heatmap_prompts_are_held_out(heatmap_prompts)
            expected_heatmap_metadata = _expected_heatmap_metadata(_model, args, heatmap_prompts)
            validated_heatmap_data = None
            if args.heatmap_path:
                _dim(f"Using user-provided heatmap: {os.path.basename(heatmap_path)}")
                validated_heatmap_data = _load_validated_heatmap(heatmap_path, expected_heatmap_metadata)
                heatmap_meta = validated_heatmap_data.get("_metadata", {})
                if heatmap_meta.get("format") == APPROVED_HEATMAP_FORMAT:
                    _detail("Approved route heatmap validated against model/router signature and compatible runtime")
                else:
                    _detail("Heatmap metadata validated against current runtime and build params")
            else:
                _status("Checking approved route heatmap cache")
                approved_path, validated_heatmap_data = _try_load_auto_approved_heatmap(
                    cache_dir,
                    expected_heatmap_metadata,
                    args,
                )
                if approved_path:
                    heatmap_path = approved_path
                    _detail("Using approved route heatmap; quick startup heatmap collection skipped")
                    if heatmap_timing_enabled:
                        heatmap_to_ready_start_s = time.perf_counter()
                        logger.info(
                            "HEATMAP_TIMING approved_heatmap_loaded path=%s",
                            heatmap_path,
                        )
                else:
                    _status("Building expert heatmap (decode-weighted calibration)")
                    heatmap_path = _build_heatmap(_model, heatmap_path, args)
                    if heatmap_timing_enabled:
                        heatmap_to_ready_start_s = time.perf_counter()
                        logger.info(
                            "HEATMAP_TIMING post_heatmap_start path=%s",
                            heatmap_path,
                        )

            # ── Load heatmap and build sorted ranking ──
            ranking_t0 = time.perf_counter()
            if validated_heatmap_data is not None:
                raw_heatmap = validated_heatmap_data
            else:
                with open(heatmap_path) as f:
                    raw_heatmap = json.load(f)
            # Strip metadata before building ranking
            heatmap_metadata = raw_heatmap.pop("_metadata", {})
            sorted_ranking = sorted(raw_heatmap.items(), key=lambda x: x[1], reverse=True)
            ranking = [(int(k.split(",")[0]), int(k.split(",")[1])) for k, _ in sorted_ranking]
            _detail(f"Heatmap: {len(ranking):,} experts ranked from {len(raw_heatmap):,} entries")
            if heatmap_timing_enabled and heatmap_to_ready_start_s is not None:
                logger.info(
                    "HEATMAP_TIMING ranking entries=%d ranking_s=%.6f post_heatmap_elapsed_s=%.6f",
                    len(ranking),
                    time.perf_counter() - ranking_t0,
                    time.perf_counter() - heatmap_to_ready_start_s,
                )

        # Build full ranking for a layer range: heatmap-ranked experts first,
        # then unranked experts to fill remaining VRAM (better than empty slots).
        def _full_ranking_for_layers(base_ranking, layer_start, layer_end):
            """Return ranking with heatmap experts first, then unranked experts for [layer_start, layer_end)."""
            # Filter base ranking to this layer range
            filtered = [(l, e) for l, e in base_ranking if layer_start <= l < layer_end]
            ranked_set = set(filtered)
            # Append all unranked experts from this layer range
            for i in range(layer_start, layer_end):
                layer = _model.layers[i]
                if not layer.is_moe:
                    continue
                n_experts = cfg.n_routed_experts
                for e in range(n_experts):
                    if (i, e) not in ranked_set:
                        filtered.append((i, e))
            return filtered

        def _heuristic_ranking_for_layers(layer_start, layer_end):
            """Return a balanced deterministic per-layer ranking without heatmap collection."""
            filtered = []
            for e in range(cfg.n_routed_experts):
                for i in range(layer_start, layer_end):
                    layer = _model.layers[i]
                    if layer.is_moe:
                        filtered.append((i, e))
            return filtered

        if args.expert_compression:
            _status("Calibrating production expert compression path")
            _expert_compression_calibration = json.loads(
                gpu_store.expert_compression_calibration_json(3, 17)
            )
            if not _expert_compression_calibration.get("payload_bit_exact", False):
                raise RuntimeError(
                    "Expert compression startup calibration did not prove bit-identical reconstruction"
                )
            _detail(
                "Expert compression: "
                f"{_expert_compression_calibration['compressed_bytes']:,} / "
                f"{_expert_compression_calibration['raw_bytes']:,} bytes, "
                f"pipeline={_expert_compression_calibration['pipeline']}, "
                f"copies/expert={_expert_compression_calibration['copies_per_expert']}, "
                f"decoder-streams={_expert_compression_calibration['decoder_streams']}, "
                f"pipeline p50/p95="
                f"{_expert_compression_calibration['p50_us']:.2f}/"
                f"{_expert_compression_calibration['p95_us']:.2f} us, bit-exact"
            )
            logger.info(
                "Expert compression startup calibration: %s",
                json.dumps(_expert_compression_calibration, sort_keys=True),
            )

        peer_candidate = (
            num_gpus_available == 2
            and args.hcs
            and args.multi_gpu_mode in ("auto", "peer")
        )
        peer_format_error = _peer_expert_format_error(args.gpu_expert_bits)
        if peer_candidate and peer_format_error is not None:
            if args.multi_gpu_mode == "peer":
                raise RuntimeError(peer_format_error)
            _warn(
                "Multi-GPU auto selector retained layer-split: "
                f"{peer_format_error}"
            )
            peer_candidate = False
        if args.multi_gpu_mode == "peer" and heuristic_hcs_init:
            raise RuntimeError(
                "Peer expert mode requires a validated route heatmap; heuristic HCS init has no route-mass basis"
            )
        if peer_candidate and heuristic_hcs_init:
            _warn(
                "Multi-GPU auto selector retained layer-split: peer prediction requires a validated route heatmap"
            )
            peer_candidate = False

        if peer_candidate:
            if _layer_split_plan is None:
                raise RuntimeError("Peer selector has no measured layer-split comparison plan")
            _status("Calibrating peer-expert multi-GPU mode")
            peer_gpu_idx = device_indices[1]
            _peer_store = _model.setup_gpu_peer_expert_store(peer_gpu_idx)
            peer_store_addr = _peer_store.gpu_store_addr()

            rtt = json.loads(
                gpu_store.measure_peer_round_trip_json(
                    peer_store_addr,
                    16,
                    128,
                    1_000,
                )
            )
            _detail(
                "Peer host-bounce RTT: "
                f"p50/p95/p99={rtt['p50_us']:.3f}/{rtt['p95_us']:.3f}/"
                f"{rtt['p99_us']:.3f} us, max={rtt['max_us']:.3f} us, "
                f"payload={rtt['message_bytes_each_direction']:,} bytes each way"
            )
            logger.info("Peer RTT startup calibration: %s", json.dumps(rtt, sort_keys=True))
            peer_rtt_gate_us = 30.0
            if rtt["p95_us"] > peer_rtt_gate_us:
                message = (
                    "peer p95 RTT failed the user-defined 30 us topology gate: "
                    f"{rtt['p95_us']:.3f} us"
                )
                if args.multi_gpu_mode == "peer":
                    raise RuntimeError(message)
                _warn(f"Multi-GPU auto selector retained layer-split: {message}")
                _peer_store = None
                peer_candidate = False

        if peer_candidate:
            full_ranking = _full_ranking_for_layers(ranking, 0, len(_model.layers))
            by_layer = {}
            for pair in full_ranking:
                by_layer.setdefault(pair[0], []).append(pair)
            calibration_ranking = next(
                (
                    layer_ranking[: int(cfg.num_experts_per_tok)]
                    for layer_ranking in by_layer.values()
                    if len(layer_ranking) >= int(cfg.num_experts_per_tok)
                ),
                None,
            )
            if calibration_ranking is None:
                raise RuntimeError(
                    "Peer service calibration found no MoE layer with a complete routed top-k"
                )
            expert_payload_bytes = int(gpu_store.expert_payload_bytes)
            calibration_budget_mb = math.ceil(
                len(calibration_ranking) * expert_payload_bytes / (1024 * 1024)
            )
            calibration_load = _peer_store.hcs_pool_init_tiered(
                calibration_ranking,
                hard_budget_mb=calibration_budget_mb,
                soft_budget_mb=0,
                safety_margin_mb=SAFETY_MARGIN_MB,
            )
            _dim(f"Peer calibration HCS: {calibration_load}")
            torch.cuda.synchronize(peer_gpu_idx)
            vram_monitor.reset_min_free()
            peer_service_baseline_mb = int(vram_monitor.current_free_mb(peer_gpu_idx))
            service = json.loads(_peer_store.peer_service_calibration_json(3, 17))
            torch.cuda.synchronize(peer_gpu_idx)
            time.sleep(0.1)
            peer_service_min_mb = int(vram_monitor.min_free_mb(peer_gpu_idx))
            peer_service_transient_mb = max(
                0, peer_service_baseline_mb - peer_service_min_mb
            )
            service_curve_p95 = [
                float(point["p95_us"]) for point in service["curve"]
            ]
            _peer_store.hcs_reset()
            with torch.cuda.device(peer_gpu_idx):
                torch.cuda.empty_cache()
            peer_free_mb = int(vram_monitor.current_free_mb(peer_gpu_idx))
            peer_hcs_budget_mb = max(
                0,
                peer_free_mb - SAFETY_MARGIN_MB - peer_service_transient_mb,
            )
            peer_capacity_experts = (
                peer_hcs_budget_mb * 1024 * 1024 // expert_payload_bytes
            )
            primary_capacity_experts = (
                int(decode_hcs_budget) * 1024 * 1024 // expert_payload_bytes
            )
            if peer_capacity_experts <= 0:
                raise RuntimeError(
                    "Peer startup calibration left no measured HCS capacity after the safety margin"
                )

            total_decode_tokens = int(
                heatmap_metadata.get("heatmap_build", {}).get("total_decode_tokens", 0)
            )
            local_cold_p95_us = float(_multi_gpu_service_profiles[0].h2d_p95_us)
            local_cold_seconds = (
                float(_multi_gpu_service_profiles[0].h2d_seconds_per_byte)
                * expert_payload_bytes
            )
            if _expert_compression_calibration is not None:
                local_cold_p95_us = float(
                    _expert_compression_calibration["p95_us"]
                )
                local_cold_seconds = (
                    float(_expert_compression_calibration["p50_us"]) / 1_000_000.0
                )

            from krasis.multi_gpu_planner import predict_peer_expert_plan

            peer_plan = predict_peer_expert_plan(
                heatmap_counts=raw_heatmap,
                total_decode_tokens=total_decode_tokens,
                ranking=full_ranking,
                primary_capacity_experts=primary_capacity_experts,
                peer_capacity_experts=peer_capacity_experts,
                layer_resident_bytes=_layer_service_bytes,
                layer_is_moe=layer_is_moe,
                primary_profile=_multi_gpu_service_profiles[0],
                expert_bytes=expert_payload_bytes,
                service_p95_us_by_routes=service_curve_p95,
                rtt_p95_us=float(rtt["p95_us"]),
                terminal_bytes=int(last_gpu_base_overhead_bytes),
                local_cold_seconds_per_expert=local_cold_seconds,
            )
            admitted_route_counts = [
                float(rtt["p95_us"]) + service_curve_p95[count - 1]
                < local_cold_p95_us * count
                for count in range(1, int(cfg.num_experts_per_tok) + 1)
            ]
            peer_prediction_ms = peer_plan.predicted_seconds_per_token * 1_000.0
            split_prediction_ms = (
                _layer_split_plan.predicted_seconds_per_token * 1_000.0
            )
            selector_uncertainty_ms = _layer_split_plan.uncertainty_seconds * 1_000.0
            peer_is_faster = (
                peer_plan.predicted_seconds_per_token
                + _layer_split_plan.uncertainty_seconds
                < _layer_split_plan.predicted_seconds_per_token
                and any(admitted_route_counts)
            )
            _detail(
                "Multi-GPU mode predictions: "
                f"layer-split={split_prediction_ms:.3f} ms/token, "
                f"peer={peer_prediction_ms:.3f} ms/token, "
                f"uncertainty={selector_uncertainty_ms:.3f} ms, "
                f"peer_capacity={peer_capacity_experts:,} experts, "
                f"captured_cold={peer_plan.captured_cold_fraction:.2%} "
                f"({peer_plan.captured_routes_per_token:.3f} routes/token)"
            )
            logger.info(
                "Multi-GPU mode selector layer_split_ms=%.6f peer_ms=%.6f "
                "uncertainty_ms=%.6f peer_capacity=%d peer_budget_mb=%d "
                "peer_service_transient_mb=%d captured_routes_per_token=%.9f "
                "cold_routes_before=%.9f cold_routes_after=%.9f captured_cold_fraction=%.9f "
                "admitted_route_counts=%s",
                split_prediction_ms,
                peer_prediction_ms,
                selector_uncertainty_ms,
                peer_capacity_experts,
                peer_hcs_budget_mb,
                peer_service_transient_mb,
                peer_plan.captured_routes_per_token,
                peer_plan.cold_routes_before_per_token,
                peer_plan.cold_routes_after_per_token,
                peer_plan.captured_cold_fraction,
                admitted_route_counts,
            )
            if args.multi_gpu_mode == "peer" and not peer_is_faster:
                raise RuntimeError(
                    "Forced peer mode failed measured critical-path admission: "
                    f"peer={peer_prediction_ms:.3f} ms/token, "
                    f"layer-split={split_prediction_ms:.3f} ms/token, "
                    f"uncertainty={selector_uncertainty_ms:.3f} ms, "
                    f"admitted_routes={admitted_route_counts}"
                )
            if args.multi_gpu_mode == "peer" or peer_is_faster:
                _multi_gpu_selected_mode = "peer"
                _multi_gpu_splits = []
                _multi_gpu_gqa_offsets = []
                _multi_gpu_split = 0
                _multi_gpu_gqa_offset = 0
                _peer_startup = {
                    "store_addr": peer_store_addr,
                    "rtt": rtt,
                    "service": service,
                    "service_curve_p95": service_curve_p95,
                    "local_cold_p95_us": local_cold_p95_us,
                    "peer_h2d_p95_us": float(
                        _multi_gpu_service_profiles[1].h2d_p95_us
                    ),
                    "peer_hcs_budget_mb": peer_hcs_budget_mb,
                    "predicted_residents": peer_plan.peer_residents,
                    "peer_prediction_ms": peer_prediction_ms,
                    "split_prediction_ms": split_prediction_ms,
                    "selector_uncertainty_ms": selector_uncertainty_ms,
                }
                _status("Multi-GPU selector chose peer expert serving")
            else:
                _peer_store = None
                _status("Multi-GPU selector chose serial layer split")

        # ── Pass calibration data to Rust ──
        if hasattr(_model, '_gpu_decode_store'):
            store = _model._gpu_decode_store

            cal_msg = store.set_vram_calibration(
                short_tokens, long_tokens,
                prefill_short_free, prefill_long_free,
                decode_short_free, decode_long_free,
                post_calibration_free_mb,
                SAFETY_MARGIN_MB,
                short_prefill_post_alloc,
                long_prefill_post_alloc,
            )
            _dim(cal_msg)
            _model._benchmark_prefill_calibration = {
                "short_tokens": int(short_tokens),
                "long_tokens": int(long_tokens),
                "prefill_short_free_mb": int(prefill_short_free),
                "prefill_long_free_mb": int(prefill_long_free),
                "decode_short_free_mb": int(decode_short_free),
                "decode_long_free_mb": int(decode_long_free),
                "baseline_free_mb": int(post_calibration_free_mb),
                "safety_margin_mb": int(SAFETY_MARGIN_MB),
                "short_prefill_post_alloc_free_mb": int(short_prefill_post_alloc),
                "long_prefill_post_alloc_free_mb": int(long_prefill_post_alloc),
            }

            # ── Set decode segment on primary store (for accurate HCS% reporting) ──
            gpu0_layer_end = _multi_gpu_split if _multi_gpu_split > 0 else len(_model.layers)
            store.set_decode_segment(0, gpu0_layer_end)

            # Multi-GPU: ensure all layers are in Marlin format BEFORE restricting swaps.
            # After calibration, weights may be in simple INT4. restrict_to_decode_segment
            # removes swap entries for layers outside GPU0's segment, so those layers would
            # be permanently stuck in simple INT4 if not swapped back to Marlin first.
            # Prefill needs ALL layers in Marlin format on GPU0.
            if _multi_gpu_split > 0:
                store.swap_to_marlin()
                _model.restrict_to_decode_segment(0, gpu0_layer_end)

            # ── Initialize GPU0 HCS as fully reclaimable on the primary GPU ──
            # In multi-GPU mode, filter ranking to GPU0's layer segment only
            # and include unranked experts to fill remaining VRAM (better than empty slots).
            gpu0_ranking = ranking
            gpu0_hard = 0
            gpu0_soft = decode_hcs_budget
            if _multi_gpu_split > 0:
                gpu0_ranking = (
                    _heuristic_ranking_for_layers(0, _multi_gpu_split)
                    if heuristic_hcs_init
                    else _full_ranking_for_layers(ranking, 0, _multi_gpu_split)
                )
                num_ranked = sum(1 for l, e in gpu0_ranking if (l, e) in set(ranking))
                _dim(f"GPU0 HCS: {len(gpu0_ranking)} experts for layers [0..{_multi_gpu_split}) "
                     f"({num_ranked} ranked + {len(gpu0_ranking) - num_ranked} unranked), "
                     f"reclaimable: {gpu0_soft:,} MB")
            else:
                # Single GPU: include all unranked experts too
                gpu0_ranking = (
                    _heuristic_ranking_for_layers(0, len(_model.layers))
                    if heuristic_hcs_init
                    else _full_ranking_for_layers(ranking, 0, len(_model.layers))
                )
                num_ranked = len(ranking)
                _dim(f"GPU0 HCS: {len(gpu0_ranking)} experts "
                     f"({num_ranked} ranked + {len(gpu0_ranking) - num_ranked} unranked), "
                     f"reclaimable: {gpu0_soft:,} MB")

            t_hcs = time.time()
            vram_monitor.report_event("hcs_init_start")
            _status("Loading GPU0 HCS pool (fully reclaimable)")
            log_ram_ledger("before-hcs-init")
            if vram_ledger:
                logger.info(
                    "VRAM LEDGER hcs_init_request hard_budget_mb=%d soft_budget_mb=%d safety_mb=%d ranking_entries=%d free_before_mb=%d",
                    gpu0_hard,
                    gpu0_soft,
                    SAFETY_MARGIN_MB,
                    len(gpu0_ranking),
                    int(vram_monitor.current_free_mb(dev_idx)),
                )

            result = store.hcs_pool_init_tiered(
                gpu0_ranking,
                hard_budget_mb=gpu0_hard,
                soft_budget_mb=gpu0_soft,
                safety_margin_mb=SAFETY_MARGIN_MB,
            )
            clamp_soft_mb_raw = os.environ.get("KRASIS_HCS_CLAMP_SOFT_MB", "").strip()
            if clamp_soft_mb_raw:
                try:
                    clamp_soft_mb = int(clamp_soft_mb_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid KRASIS_HCS_CLAMP_SOFT_MB={clamp_soft_mb_raw!r}; "
                        "expected an integer MB value"
                    ) from exc
                if clamp_soft_mb < 0:
                    raise ValueError(
                        f"Invalid KRASIS_HCS_CLAMP_SOFT_MB={clamp_soft_mb}; "
                        "expected a non-negative integer MB value"
                    )
                applied_soft_mb, loaded_soft_mb = store.hcs_clamp_soft_budget_mb(clamp_soft_mb)
                _warn(
                    f"Diagnostic HCS soft clamp: requested {clamp_soft_mb:,} MB, "
                    f"loaded {loaded_soft_mb:,} MB, applied soft max {applied_soft_mb:,} MB"
                )
                logger.warning(
                    "Diagnostic HCS soft clamp: requested=%d MB loaded=%d MB applied=%d MB",
                    clamp_soft_mb, loaded_soft_mb, applied_soft_mb,
                )
            if cpu_tail_transposed_requested:
                _status("Building CPU-tail transposed non-resident tier")
                log_ram_ledger("before-cpu-tail-transposed-tier")
                tier_t0 = time.time()
                tier_result = store.cpu_tail_build_transposed_tier()
                tier_elapsed = time.time() - tier_t0
                _detail(tier_result)
                _dim(f"CPU-tail transposed tier built in {tier_elapsed:.1f}s")
                logger.warning(
                    "CPU-tail transposed tier: %s (%.3fs)",
                    tier_result,
                    tier_elapsed,
                )
                log_ram_ledger("after-cpu-tail-transposed-tier")
            hcs_elapsed = time.time() - t_hcs

            vram_monitor.report_event("hcs_init_end")
            _status("HCS pool loaded")
            _detail(result)
            _dim(f"Loaded in {hcs_elapsed:.1f}s")
            logger.info("HCS pool: %s (%.1fs)", result, hcs_elapsed)
            log_ram_ledger("after-hcs-init")
            if vram_ledger:
                logger.info(
                    "VRAM LEDGER hcs_init_result elapsed_s=%.3f free_after_mb=%d result=%s",
                    hcs_elapsed,
                    int(vram_monitor.current_free_mb(dev_idx)),
                    result,
                )
                _model.log_vram_ledger_residency("after-hcs-init")

            if _multi_gpu_selected_mode == "peer":
                if _peer_store is None or _peer_startup is None:
                    raise RuntimeError(
                        "Peer mode was selected without a calibrated peer store"
                    )
                primary_residents = set(store.hcs_resident_pairs())
                peer_ranking = [
                    pair for pair in full_ranking if pair not in primary_residents
                ]
                _status("Loading disjoint peer expert tier")
                peer_hcs_result = _peer_store.hcs_pool_init_tiered(
                    peer_ranking,
                    hard_budget_mb=int(_peer_startup["peer_hcs_budget_mb"]),
                    soft_budget_mb=0,
                    safety_margin_mb=SAFETY_MARGIN_MB,
                )
                actual_peer_residents = set(_peer_store.hcs_resident_pairs())
                overlap = primary_residents.intersection(actual_peer_residents)
                if overlap:
                    raise RuntimeError(
                        f"Peer HCS is not disjoint from primary HCS ({len(overlap)} duplicate experts)"
                    )
                captured_count = sum(
                    int(raw_heatmap.get(f"{layer},{expert}", 0))
                    for layer, expert in actual_peer_residents
                )
                total_decode_tokens = int(
                    heatmap_metadata.get("heatmap_build", {}).get(
                        "total_decode_tokens", 0
                    )
                )
                captured_routes_per_token = (
                    captured_count / total_decode_tokens
                    if total_decode_tokens > 0
                    else 0.0
                )
                _detail(f"Peer HCS: {peer_hcs_result}")
                _detail(
                    f"Peer residents: {len(actual_peer_residents):,}, "
                    f"primary overlap=0, approved-heatmap capture="
                    f"{captured_routes_per_token:.3f} routes/token"
                )
                attach_result = store.attach_peer_expert_store(
                    int(_peer_startup["store_addr"]),
                    list(_peer_startup["service_curve_p95"]),
                    float(_peer_startup["rtt"]["p95_us"]),
                    float(_peer_startup["local_cold_p95_us"]),
                    float(_peer_startup["peer_h2d_p95_us"]),
                    float(_peer_startup["peer_prediction_ms"]),
                    float(_peer_startup["split_prediction_ms"]),
                    float(_peer_startup["selector_uncertainty_ms"]),
                )
                _detail(attach_result)
                logger.info(
                    "Peer expert mode ready: primary_residents=%d peer_residents=%d "
                    "captured_routes_per_token=%.9f peer_budget_mb=%d attach=%s",
                    len(primary_residents),
                    len(actual_peer_residents),
                    captured_routes_per_token,
                    int(_peer_startup["peer_hcs_budget_mb"]),
                    attach_result,
                )

    # ── Decode validation ──
    # Rust prefill warmup already ran before HCS budgeting.
    vram_monitor.report_event("validation_start")
    _status("Decode validation")
    _detail("Skipped: Rust prefill already warmed before HCS budgeting")
    vram_monitor.report_event("validation_end")

    # ── Enable VRAM monitor runtime warnings ──
    # enable_warnings() resets min-free tracking so the first poll captures
    # the post-HCS state. If free VRAM is already below the safety margin
    # (i.e. HCS was too aggressive), we get an immediate warning.
    # During runtime, every new low below the margin triggers another warning.
    _status("VRAM monitor: runtime warnings enabled")
    _detail(f"Safety margin: {SAFETY_MARGIN_MB:,} MB — warnings on every new low below this")
    vram_monitor.enable_warnings()
    logger.info("VRAM monitor: runtime warnings enabled (safety margin: %d MB)", SAFETY_MARGIN_MB)

    def _verify_hcs_vram_floor(reason: str) -> int:
        """Gate readiness on measured free VRAM, not just planned HCS budget."""
        if not args.hcs:
            return 0
        final_free_mb = 0
        # The monitor polls every ~50ms. Use bounded rechecks so delayed CUDA
        # allocator/runtime lows are observed and drained before we publish ready.
        for attempt in range(4):
            evicted, freed_mb, final_free_mb = gpu_store.py_hcs_drain_vram_pressure(
                f"{reason}_check{attempt + 1}",
                True,
            )
            if evicted > 0:
                _warn(
                    f"VRAM pressure eviction at {reason}: evicted {evicted} soft HCS experts, "
                    f"freed {freed_mb:.1f} MB, final free {final_free_mb:,} MB"
                )
            if attempt >= 1 and final_free_mb >= SAFETY_MARGIN_MB:
                return final_free_mb
            time.sleep(0.12)
        if final_free_mb < SAFETY_MARGIN_MB:
            raise SystemExit(
                f"VRAM safety floor not restored at {reason}: "
                f"{final_free_mb} MB free, safety margin {SAFETY_MARGIN_MB} MB"
            )
        return final_free_mb

    if args.hcs:
        _verify_hcs_vram_floor("startup_ready")

    # Benchmark runs AFTER server starts (same HTTP path as production).
    # We set up the benchmark thread here; it launches after rust_server.run().
    _benchmark_requested = args.benchmark or args.benchmark_only
    _benchmark_only = args.benchmark_only

    # Run stress test if requested
    if args.stress_test:
        from krasis.stress_test import StressTest
        st = StressTest(_model)
        results = st.run()
        failed = sum(1 for r in results if r["status"].startswith("FAIL"))
        sys.exit(1 if failed > 0 else 0)

    # Run perplexity evaluation if requested
    if args.perplexity:
        _ppl_dir = os.path.join(os.path.dirname(__file__), "..", "..", "perplexity")
        sys.path.insert(0, os.path.dirname(_ppl_dir))
        from perplexity.measure_ppl import list_datasets, run_perplexity

        _status("Perplexity Evaluation")
        datasets = list_datasets()
        print("\nChoose dataset:")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['name']:20s} ({ds['tokens_approx']} tokens)")
        print(f"  {len(datasets) + 1}. All datasets")

        choice = input(f"\nSelection [1]: ").strip() or "1"
        try:
            choice_idx = int(choice)
        except ValueError:
            print(f"Invalid selection: {choice}")
            sys.exit(1)

        if choice_idx == len(datasets) + 1:
            # Run all datasets
            selected = [ds["name"] for ds in datasets]
        elif 1 <= choice_idx <= len(datasets):
            selected = [datasets[choice_idx - 1]["name"]]
        else:
            print(f"Invalid selection: {choice_idx}")
            sys.exit(1)

        config = {
            "model_path": args.model_path,
            "gpu_expert_bits": args.gpu_expert_bits,
            "expert_group_size": args.expert_group_size,
            "gpu_expert_int4_calib": args.gpu_expert_int4_calib,
            "cpu_expert_bits": args.cpu_expert_bits,
            "attention_quant": args.attention_quant,
            "lm_head_quant": args.lm_head_quant,
            "layer_group_size": args.layer_group_size,
            "krasis_threads": args.krasis_threads,
            "kv_cache_mb": args.kv_cache_mb,
        }

        all_results = []
        for ds_name in selected:
            result = run_perplexity(model=_model, dataset_name=ds_name, config=config)
            all_results.append(result)

        # Print summary table if multiple datasets
        if len(all_results) > 1:
            _headline("PERPLEXITY SUMMARY")
            print(f"  {'Dataset':20s} {'PPL':>10s} {'BPC':>8s} {'Tokens':>12s} {'Time':>8s}")
            for r in all_results:
                tok_s = r["num_tokens_scored"] / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
                print(
                    f"  {r['dataset']:20s} {r['perplexity']:10.2f} {r['bits_per_char']:8.2f} "
                    f"{r['num_tokens_scored']:>12,} {r['elapsed_s']:7.1f}s"
                )

        sys.exit(0)

    max_ctx = _model.get_max_context_tokens()

    # ── Multi-GPU decode setup ──
    # Lists passed to RustServer for N-GPU pipeline decode
    all_aux_gpu_store_addrs = []
    all_multi_gpu_split_layers = list(_multi_gpu_splits)
    all_multi_gpu_gqa_offsets = list(_multi_gpu_gqa_offsets)
    if _multi_gpu_splits and args.hcs:
        num_aux = len(_multi_gpu_splits)
        num_layers = len(_model.layers)
        boundaries = [0] + list(_multi_gpu_splits) + [num_layers]
        vram_monitor.report_event("multi_gpu_setup_start")
        _status(f"Multi-GPU decode setup ({num_aux + 1} GPUs)")
        gc.collect()
        torch.cuda.empty_cache()

        # Count layer types per GPU segment
        for gpu_i in range(num_aux + 1):
            seg_start, seg_end = boundaries[gpu_i], boundaries[gpu_i + 1]
            la, gqa, moe, dense = 0, 0, 0, 0
            for i in range(seg_start, seg_end):
                layer = _model.layers[i]
                if layer.layer_type == "linear_attention": la += 1
                else: gqa += 1
                if layer.is_moe: moe += 1
                elif layer.dense_mlp is not None: dense += 1
            _detail(f"  GPU{gpu_i}: layers [{seg_start}..{seg_end}) = {seg_end - seg_start} layers "
                    f"({la} LA + {gqa} GQA, {moe} MoE + {dense} dense)")
        _dim(f"  GQA cache offsets: {_multi_gpu_gqa_offsets}")
        _dim(f"  Attention layers are PERMANENT on each GPU (copied at setup, never evicted)")

        # Create aux stores for each aux GPU
        aux_stores = []
        for i in range(num_aux):
            seg_start = _multi_gpu_splits[i]
            seg_end = boundaries[i + 2]  # boundaries[i+1+1]
            gpu_idx = device_indices[i + 1]
            _dim(f"Creating aux decode store {i+1} on cuda:{gpu_idx} for layers [{seg_start}..{seg_end})...")
            aux_store = _model.setup_gpu_decode_store_aux(
                gpu_idx=gpu_idx,
                split_layer=seg_start,
                layer_end=seg_end,
            )
            aux_store.set_decode_segment(seg_start, seg_end)
            aux_stores.append(aux_store)
            all_aux_gpu_store_addrs.append(aux_store.gpu_store_addr())

        # Log post-setup VRAM
        for dev in devices:
            torch.cuda.synchronize(dev)
        gc.collect()
        torch.cuda.empty_cache()
        for idx in device_indices:
            post_free = vram_monitor.current_free_mb(idx)
            total_mb = vram_monitor.total_mb(idx)
            _dim(f"  cuda:{idx} after aux store setup: {post_free:,.0f} MB free / {total_mb:,} MB total")

        # Measure auxiliary decode runtime overhead before filling aux HCS. Aux GPUs
        # do not run prefill, but decode still has transient graph/kernel/runtime
        # allocations. Budget HCS against the measured low-water rather than
        # assuming all current free memory minus the safety margin is usable.
        aux_decode_overhead_mb = [0] * num_aux
        if num_aux > 0:
            _status("Calibrating multi-GPU aux decode overhead")
            vram_monitor.reset_min_free()
            aux_baseline_free = [
                int(vram_monitor.current_free_mb(device_indices[i + 1]))
                for i in range(num_aux)
            ]
            try:
                gpu_store = _model._gpu_decode_store
                evicted0, _freed0 = gpu_store.py_hcs_evict_for_prefill(500)
                if evicted0 > 0:
                    _dim(f"  Evicted {evicted0} soft experts for aux overhead prefill")

                prompt_tokens = _chat_prompt_tokens(_model, "Hi")
                stop_ids = _default_stop_ids(_model)
                first_token, prompt_len, _kv_overflow = gpu_store.rust_prefill_tokens(
                    prompt_tokens, temperature=0.6, disable_pinning=True
                )

                for i in range(num_aux):
                    seg_start = boundaries[i + 1]
                    seg_end = boundaries[i + 2]
                    gpu_store.py_copy_kv_to_aux(
                        all_aux_gpu_store_addrs[i], seg_start, seg_end,
                        _multi_gpu_gqa_offsets[i], prompt_len)
                    gpu_store.py_copy_la_states_to_aux(
                        all_aux_gpu_store_addrs[i], seg_start, seg_end)
                    gpu_store.py_copy_dsa_prompt_keys_to_aux(
                        all_aux_gpu_store_addrs[i], prompt_len)

                r0, _ = gpu_store.py_hcs_reload_after_prefill(prompt_len)
                if r0 > 0:
                    _dim(f"  Reloaded {r0} soft experts after aux overhead prefill")

                if first_token not in stop_ids:
                    tokens = gpu_store.gpu_generate_batch_multi(
                        aux_store_addrs=all_aux_gpu_store_addrs,
                        split_layers=all_multi_gpu_split_layers,
                        gqa_cache_offsets=all_multi_gpu_gqa_offsets,
                        first_token=first_token,
                        start_position=prompt_len,
                        max_tokens=32,
                        temperature=0.6,
                        top_k=50,
                        top_p=0.95,
                        stop_ids=stop_ids,
                        presence_penalty=0.0,
                    )
                    _detail(f"Aux decode overhead calibration: {len(tokens)} tokens generated OK")
                else:
                    _detail("Aux decode overhead calibration: prefill hit stop token (OK)")

                torch.cuda.synchronize()
                time.sleep(0.1)
                for i in range(num_aux):
                    gpu_idx = device_indices[i + 1]
                    aux_min = int(vram_monitor.min_free_mb(gpu_idx))
                    aux_current = int(vram_monitor.current_free_mb(gpu_idx))
                    measured = max(0, aux_baseline_free[i] - aux_min)
                    aux_decode_overhead_mb[i] = measured
                    _dim(
                        f"  GPU{i+1} decode transient: baseline={aux_baseline_free[i]:,} MB, "
                        f"min={aux_min:,} MB, current={aux_current:,} MB, overhead={measured:,} MB"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Multi-GPU aux decode overhead calibration failed: {e}\n"
                    "Cannot size aux HCS safely. Fix the underlying issue or disable multi-GPU."
                ) from e

        # Initialize HCS on each aux store if requested.
        if args.hcs and 'ranking' in locals():
            for i, aux_store in enumerate(aux_stores):
                seg_start = boundaries[i + 1]
                seg_end = boundaries[i + 2]
                gpu_idx = device_indices[i + 1]

                aux_ranking = (
                    _heuristic_ranking_for_layers(seg_start, seg_end)
                    if heuristic_hcs_init
                    else _full_ranking_for_layers(ranking, seg_start, seg_end)
                )
                num_aux_ranked = sum(1 for l, e in aux_ranking if (l, e) in set(ranking))
                _dim(f"  GPU{i+1} HCS: {len(aux_ranking)} experts for layers [{seg_start}..{seg_end}) "
                     f"({num_aux_ranked} ranked + {len(aux_ranking) - num_aux_ranked} unranked)")
                # Measure aux GPU free VRAM for HCS budget. Aux HCS is hard
                # resident and cannot be evicted for later decode/runtime
                # allocations, so reserve both the final safety floor and a
                # runtime transient band sized from measurement but never
                # smaller than the configured safety margin.
                aux_free_mb = vram_monitor.current_free_mb(gpu_idx)
                aux_runtime_reserve_mb = max(aux_decode_overhead_mb[i], SAFETY_MARGIN_MB)
                aux_hcs_budget = max(
                    0,
                    int(aux_free_mb) - SAFETY_MARGIN_MB - aux_runtime_reserve_mb,
                )
                # 100% hard tier — aux GPUs never do prefill so never need to evict
                aux_hard = aux_hcs_budget
                aux_soft = 0
                _detail(
                    f"  GPU{i+1} HCS budget: {aux_hard:,} MB hard "
                    f"(aux_free={aux_free_mb:,.0f} MB, decode_transient={aux_decode_overhead_mb[i]:,} MB, "
                    f"runtime_reserve={aux_runtime_reserve_mb:,} MB)"
                )

                if aux_ranking:
                    result = aux_store.hcs_pool_init_tiered(
                        aux_ranking,
                        hard_budget_mb=aux_hard,
                        soft_budget_mb=aux_soft,
                        safety_margin_mb=SAFETY_MARGIN_MB,
                    )
                    _detail(f"  GPU{i+1} HCS: {result}")

        # ── Multi-GPU decode validation ──
        _status("Validating multi-GPU decode")
        vram_monitor.reset_min_free()
        try:
            gpu_store = _model._gpu_decode_store
            evicted0, freed0 = gpu_store.py_hcs_evict_for_prefill(500)
            if evicted0 > 0:
                _dim(f"  Evicted {evicted0} soft experts for validation prefill")

            prompt_tokens = _chat_prompt_tokens(_model, "Hi")
            stop_ids = _default_stop_ids(_model)
            first_token, prompt_len, _kv_overflow = gpu_store.rust_prefill_tokens(
                prompt_tokens, temperature=0.6, disable_pinning=True
            )

            # Copy KV cache and LA state to each aux store
            for i in range(num_aux):
                seg_start = boundaries[i + 1]
                seg_end = boundaries[i + 2]
                gpu_store.py_copy_kv_to_aux(
                    all_aux_gpu_store_addrs[i], seg_start, seg_end,
                    _multi_gpu_gqa_offsets[i], prompt_len)
                gpu_store.py_copy_la_states_to_aux(
                    all_aux_gpu_store_addrs[i], seg_start, seg_end)
                gpu_store.py_copy_dsa_prompt_keys_to_aux(
                    all_aux_gpu_store_addrs[i], prompt_len)

            # Reload soft HCS on GPU0 only (aux GPUs have no soft tier)
            r0, _ = gpu_store.py_hcs_reload_after_prefill(prompt_len)
            if r0 > 0:
                _dim(f"  Reloaded {r0} soft experts after validation prefill")

            # Run multi-GPU decode
            if first_token not in stop_ids:
                tokens = gpu_store.gpu_generate_batch_multi(
                    aux_store_addrs=all_aux_gpu_store_addrs,
                    split_layers=all_multi_gpu_split_layers,
                    gqa_cache_offsets=all_multi_gpu_gqa_offsets,
                    first_token=first_token,
                    start_position=prompt_len,
                    max_tokens=32,
                    temperature=0.6,
                    top_k=50,
                    top_p=0.95,
                    stop_ids=stop_ids,
                    presence_penalty=0.0,
                )
                _detail(f"Multi-GPU decode validation: {len(tokens)} tokens generated OK")
            else:
                _detail("Multi-GPU decode validation: prefill hit stop token (OK)")

            # Log VRAM stats from all GPUs during multi-GPU decode
            torch.cuda.synchronize()
            time.sleep(0.1)  # let monitor poll
            for idx in device_indices:
                min_free = vram_monitor.min_free_mb(idx)
                current_free = vram_monitor.current_free_mb(idx)
                total_mb = vram_monitor.total_mb(idx)
                _dim(f"  cuda:{idx} during multi-GPU decode: min_free={min_free:,} MB, "
                     f"current_free={current_free:,.0f} MB / {total_mb:,} MB total")

            # ── Validate aux GPU VRAM safety during decode ──
            # Aux GPUs never run prefill, so decode transients are the only VRAM pressure.
            # The HCS budget must preserve the configured safety margin.
            for i in range(num_aux):
                gpu_idx = device_indices[i + 1]
                aux_min_free = vram_monitor.min_free_mb(gpu_idx)
                aux_current = vram_monitor.current_free_mb(gpu_idx)
                if aux_min_free < SAFETY_MARGIN_MB:
                    raise RuntimeError(
                        f"GPU{i+1} (cuda:{gpu_idx}) min_free={aux_min_free} MB during decode, "
                        f"below safety margin {SAFETY_MARGIN_MB} MB. Aux HCS budget is too aggressive."
                    )
                else:
                    _dim(f"  GPU{i+1} (cuda:{gpu_idx}): min_free={aux_min_free} MB during decode, "
                         f"current_free={aux_current:.0f} MB (budget safe)")

        except Exception as e:
            raise RuntimeError(
                f"Multi-GPU decode validation failed: {e}\n"
                "Cannot start server with broken multi-GPU decode. "
                "Fix the underlying issue or disable multi-GPU."
            ) from e

        vram_monitor.report_event("multi_gpu_setup_end")
        _status(f"Multi-GPU decode ready ({num_aux + 1} GPUs, splits={_multi_gpu_splits})")

    # ── Final VRAM summary (all GPUs) ──
    if num_gpus_available > 1 and args.hcs and _multi_gpu_splits:
        gc.collect()
        torch.cuda.empty_cache()
        _status("VRAM allocation summary")
        boundaries_final = [0] + list(_multi_gpu_splits) + [len(_model.layers)]
        for gpu_i in range(num_gpus_available):
            idx = device_indices[gpu_i]
            total_vram = vram_monitor.total_mb(idx)
            current_free = vram_monitor.current_free_mb(idx)
            used = total_vram - current_free
            seg_start = boundaries_final[gpu_i]
            seg_end = boundaries_final[gpu_i + 1]
            n_layers = seg_end - seg_start
            # Count layer types
            la_count = sum(1 for i in range(seg_start, seg_end) if _model.layers[i].layer_type == "linear_attention")
            gqa_count = n_layers - la_count
            moe_count = sum(1 for i in range(seg_start, seg_end) if _model.layers[i].is_moe)
            # Estimate attention VRAM for this segment
            seg_attn_mb = sum(_layer_vram_mb[j] for j in range(seg_start, seg_end))
            # HCS info
            if gpu_i == 0:
                hcs_reclaimable = gpu0_soft
                hcs_total = hcs_reclaimable
                hcs_type = f"reclaimable={hcs_reclaimable:,}MB"
                # Prefill-only info
                prefill_only_entries = getattr(_model, '_prefill_only_attn', [])
                if prefill_only_entries:
                    po_bytes = 0
                    for _, _, _, mw in prefill_only_entries:
                        po_bytes += mw.packed.nelement() * mw.packed.element_size()
                        po_bytes += mw.scales.nelement() * mw.scales.element_size()
                    po_mb = po_bytes / (1024 * 1024)
                    _detail(f"  GPU{gpu_i} (cuda:{idx}): layers [{seg_start}..{seg_end}) = {n_layers} ({la_count}LA+{gqa_count}GQA, {moe_count}MoE)")
                    attn_desc = attention_quant_label(args.attention_quant)
                    _detail(f"    Attention (decode segment): {seg_attn_mb:.0f} MB (permanent, {attn_desc})")
                    _detail(f"    Attention (prefill-only):   {po_mb:.0f} MB (freed after prefill, reclaimed for HCS)")
                    _detail(f"    HCS:   {hcs_type} = {hcs_total:,} MB total")
                    _detail(f"    VRAM:  {used:,.0f} MB used / {total_vram:,} MB total ({current_free:,.0f} MB free)")
                else:
                    _detail(f"  GPU{gpu_i} (cuda:{idx}): layers [{seg_start}..{seg_end}) = {n_layers} ({la_count}LA+{gqa_count}GQA, {moe_count}MoE)")
                    _detail(f"    Attention: {seg_attn_mb:.0f} MB")
                    _detail(f"    HCS:   {hcs_type} = {hcs_total:,} MB total")
                    _detail(f"    VRAM:  {used:,.0f} MB used / {total_vram:,} MB total ({current_free:,.0f} MB free)")
            else:
                aux_hcs_budget = gpu_hcs_budgets[gpu_i]
                attn_type = attention_quant_label(args.attention_quant)
                _detail(f"  GPU{gpu_i} (cuda:{idx}): layers [{seg_start}..{seg_end}) = {n_layers} ({la_count}LA+{gqa_count}GQA, {moe_count}MoE)")
                _detail(f"    Attention: {seg_attn_mb:.0f} MB ({attn_type}, permanent)")
                _detail(f"    HCS:   {aux_hcs_budget:,.0f} MB (100% hard, no prefill)")
                _detail(f"    VRAM:  {used:,.0f} MB used / {total_vram:,} MB total ({current_free:,.0f} MB free)")

    # ── Server registry: write entry + register cleanup ──
    _write_registry(args.host, args.port, _model_name)
    atexit.register(_remove_registry)

    # ── Rust HTTP server ──
    from krasis import RustServer

    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")

    # Look up </think> token ID for thinking budget tracking.
    # Some models expose an enable_thinking template switch; Step-style templates
    # always open a thinking block and rely on the serving layer to parse/close it.
    think_end_id = 0
    if template_supports_enable_thinking or template_has_thinking:
        _raw_id = _hf_tok.convert_tokens_to_ids("</think>")
        if isinstance(_raw_id, int) and _raw_id != _hf_tok.unk_token_id:
            think_end_id = _raw_id
            logger.info("Thinking end token: </think> = %d", think_end_id)
        else:
            logger.info("Template has thinking blocks but no </think> token")
    else:
        logger.info("Model template does not support enable_thinking — thinking budget disabled")

    # Look up turn-boundary tokens to suppress during generation.
    # Models sometimes generate <|im_start|> during multi-turn thinking,
    # creating phantom new turns. Suppressing these prevents the issue.
    _suppress_tokens = []
    for special_tok in ["<|im_start|>", "<|start_header_id|>", "<|begin_of_text|>"]:
        _raw_id = _hf_tok.convert_tokens_to_ids(special_tok)
        if isinstance(_raw_id, int) and _raw_id != _hf_tok.unk_token_id:
            _suppress_tokens.append(_raw_id)
    if _suppress_tokens:
        _model._gpu_decode_store.set_suppress_tokens(_suppress_tokens)
        _model._suppress_tokens = _suppress_tokens
        logger.info("Suppress tokens: %s", {tok: _hf_tok.convert_ids_to_tokens(tok) for tok in _suppress_tokens})

    logger.info(
        "Model loaded, starting server on %s:%d (max context: %d, decode: GPU%s)",
        args.host, args.port, max_ctx,
        f"s ({len(all_aux_gpu_store_addrs)+1}-GPU)" if all_aux_gpu_store_addrs else "",
    )
    _vision_supported = bool(getattr(_model, "supports_image_inputs", lambda: False)())
    _vision_model_name = f"{_model_name}-vision" if _vision_supported else ""

    rust_server = RustServer(
        _model,
        args.host,
        args.port,
        _model_name,
        tokenizer_path,
        max_ctx,
        args.enable_thinking,
        think_end_id,
        gpu_store_addr,
        all_aux_gpu_store_addrs,
        all_multi_gpu_split_layers,
        all_multi_gpu_gqa_offsets,
        _vision_supported,
        test_endpoints=getattr(args, 'test_endpoints', False),
        prefix_cache=bool(args.prefix_cache),
        prefix_cache_ram_fraction=float(args.prefix_cache_ram_fraction),
    )
    ssh_tunnel = None
    if str(getattr(args, "ssh_tunnel", "") or "").strip():
        from krasis.ssh_tunnel import SshReverseTunnel

        try:
            _status("Starting SSH reverse tunnel")
            ssh_tunnel = SshReverseTunnel(
                args.ssh_tunnel,
                local_port=args.port,
                key_path=args.ssh_key_path,
                logger=logger,
            )
            ssh_tunnel.start()
            _detail(
                f"remote 127.0.0.1:{args.port} -> local 127.0.0.1:{args.port} "
                f"via {args.ssh_tunnel}"
            )
        except Exception as exc:
            logger.error("SSH tunnel failed: %s", exc)
            rust_server.stop()
            raise SystemExit(f"SSH tunnel failed: {exc}") from exc

    def _handle_exit(sig, frame):
        if ssh_tunnel is not None:
            ssh_tunnel.stop()
        rust_server.stop()
        try:
            sys.stderr = open(os.devnull, "w")
            sys.stdout = open(os.devnull, "w")
            logging.disable(logging.CRITICAL)
        except Exception:
            pass
        os.write(1, f"\n{_BOLD}{_GREEN}Server stopped.{_NC}\n".encode())

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    # Final ready summary after all setup is done. Keep lines compact because
    # stdout is also written through the timestamped krasis.server logger.
    time.sleep(1.0)
    if args.hcs:
        _verify_hcs_vram_floor("server_ready")
    _decode_mode = f"{len(all_aux_gpu_store_addrs)+1}-GPU" if all_aux_gpu_store_addrs else "GPU"
    _hcs_str = "on" if args.hcs else "off"
    _think_str = "on" if args.enable_thinking else "off"
    _client_host = "127.0.0.1" if str(args.host).strip() in ("0.0.0.0", "::", "") else str(args.host).strip()
    _client_base_url = f"http://{_client_host}:{args.port}/v1"
    _client_chat_url = f"{_client_base_url}/chat/completions"
    _client_models_url = f"{_client_base_url}/models"
    _client_model_name = _vision_model_name or _model_name
    vram_monitor.report_event("server_ready")
    log_ram_ledger("server-ready")
    if heatmap_timing_enabled and heatmap_to_ready_start_s is not None:
        logger.info(
            "HEATMAP_TIMING post_heatmap_to_ready_s=%.6f",
            time.perf_counter() - heatmap_to_ready_start_s,
        )
    _headline("KRASIS SERVER READY", _GREEN)
    print(f"  {_GREEN}Model:{_NC}    {_BOLD}{_model_name}{_NC}", flush=True)
    print(f"  {_GREEN}Address:{_NC}  {_BOLD}{args.host}:{args.port}{_NC}", flush=True)
    if ssh_tunnel is not None:
        print(
            f"  {_GREEN}Tunnel:{_NC}   {_BOLD}{args.ssh_tunnel}{_NC} "
            f"{_DIM}remote 127.0.0.1:{args.port}{_NC}",
            flush=True,
        )
    print(f"  {_GREEN}Context:{_NC}  {max_ctx:,} tokens  |  KV cache: {args.kv_cache_mb:,} MB", flush=True)
    print(f"  {_GREEN}Decode:{_NC}   {_decode_mode}  |  HCS: {_hcs_str}  |  Think: {_think_str}", flush=True)
    print(f"  {_GREEN}Client setup:{_NC}", flush=True)
    print(f"    Base URL:      {_BOLD}{_client_base_url}{_NC}", flush=True)
    print(f"    Chat endpoint: {_client_chat_url}", flush=True)
    print(f"    Models:        {_client_models_url}", flush=True)
    print(f"    API key:       {_BOLD}X{_NC} {_DIM}(any value){_NC}", flush=True)
    print(f"    Model name:    {_BOLD}{_client_model_name}{_NC}", flush=True)
    if _vision_model_name:
        print(f"                   {_DIM}vision-capable; use this same model for text and images{_NC}", flush=True)
    print(f"  {_DIM}Press Q or Ctrl-C to stop{_NC}", flush=True)
    print(flush=True)

    # Q to quit (background thread)
    def _stdin_listener():
        try:
            while rust_server.is_running():
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ("q", "Q"):
                        _handle_exit(None, None)
                        break
        except (OSError, ValueError):
            pass
    if sys.stdin.isatty():
        t = threading.Thread(target=_stdin_listener, daemon=True)
        t.start()

    # Benchmark thread: engine benchmarks run immediately (no HTTP needed),
    # HTTP TTFT benchmark waits for server to be ready internally.
    if _benchmark_requested:
        def _run_benchmark():
            from krasis.benchmark import KrasisBenchmark
            bench = KrasisBenchmark(
                _model, rust_server=rust_server,
                host=args.host, port=args.port, timing=args.timing,
            )
            bench.run()

            if _benchmark_only:
                rust_server.stop()
            else:
                # Re-show readiness after benchmark output.
                _headline(f"SERVER READY: {args.host}:{args.port}", _GREEN)
                if ssh_tunnel is not None:
                    _dim(f"SSH tunnel: {args.ssh_tunnel} remote 127.0.0.1:{args.port}")
                _dim("Press Q or Ctrl-C to stop")

        bench_thread = threading.Thread(target=_run_benchmark, daemon=True)
        bench_thread.start()

    # run() releases the GIL and blocks until stop() is called
    rust_server.run()

    # ── Write VRAM report if enabled ──
    vram_monitor.report_event("server_shutdown")
    if getattr(args, 'vram_report', False):
        _report_path = os.path.join(_run_dir, "vram_report.csv")
        os.makedirs(os.path.dirname(_report_path), exist_ok=True)
        try:
            vram_monitor.write_report(_report_path)
            # Print summary of key events
            summary = vram_monitor.report_summary()
            if summary:
                _headline("VRAM REPORT SUMMARY")
                _hdr = f"  {'Event':<30s} {'Time':>8s}"
                for i in device_indices:
                    _hdr += f"  {'GPU' + str(i):>8s}"
                print(_hdr, flush=True)
                for event, ts_ms, gpu_free in summary:
                    _line = f"  {event:<30s} {ts_ms/1000:>7.1f}s"
                    for mb in gpu_free:
                        _line += f"  {mb:>6d}MB"
                    print(_line, flush=True)
                print(f"  {_DIM}Full report: {_report_path}{_NC}", flush=True)
        except Exception as e:
            logger.warning("Failed to write VRAM report: %s", e)

    # ── Clean exit before Python teardown triggers cascading errors ──
    vram_monitor.stop()
    if ssh_tunnel is not None:
        ssh_tunnel.stop()
    _remove_registry()
    _cleanup_cuda()
    os._exit(0)


if __name__ == "__main__":
    main()
