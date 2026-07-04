"""Krasis — hybrid LLM MoE runtime."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError
import os
from pathlib import Path

_DLL_DIRECTORY_HANDLES = []


def _configure_vendored_sidecars() -> None:
    pkg_dir = Path(__file__).resolve().parent
    search_roots = [
        pkg_dir,
        pkg_dir.parent / "krasis.libs",
    ]
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for root in search_roots:
            if root.is_dir():
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(root)))

    sidecars = {
        "KRASIS_MARLIN_SO": ("krasis_marlin.dll", "libkrasis_marlin.so"),
        "KRASIS_LIBKRASIS_FLASH_ATTN_SO": (
            "krasis_flash_attn.dll",
            "libkrasis_flash_attn.so",
        ),
        "KRASIS_LIBKRASIS_FLA_SO": ("krasis_fla.dll", "libkrasis_fla.so"),
    }
    for env_key, filenames in sidecars.items():
        if os.environ.get(env_key):
            continue
        for root in search_roots:
            for filename in filenames:
                candidate = root / filename
                if candidate.is_file():
                    os.environ[env_key] = str(candidate)
                    break
            if os.environ.get(env_key):
                break


_configure_vendored_sidecars()

try:
    __version__ = _pkg_version("krasis")
except PackageNotFoundError:
    __version__ = "dev"

try:
    from krasis.krasis import KrasisEngine, WeightStore, RustServer, system_check
    try:
        from krasis.krasis import hqq_search_cuda_tensor_ptr
    except ImportError:
        pass  # Built without HQQ CUDA search kernels
    try:
        from krasis.krasis import GpuDecodeStore
    except ImportError:
        pass  # Built without CUDA feature
    try:
        from krasis.krasis import VramMonitor
    except ImportError:
        pass  # Built without CUDA feature
except ImportError:
    # Native module not built yet
    pass

# GpuPrefillManager removed — Rust prefill engine replaces it

try:
    from krasis.vram_budget import compute_vram_budget, compute_launcher_budget
except ImportError:
    pass

# Standalone server modules
try:
    from krasis.config import ModelConfig, QuantConfig
    from krasis.model import KrasisModel
except ImportError:
    pass
