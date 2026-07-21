"""NVIDIA utility discovery shared by launch-time code."""

import os
import shutil
from typing import Optional


def is_wsl() -> bool:
    """Return True when running under WSL/WSL2."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def wsl_cuda_dir() -> str:
    return "/usr/lib/wsl/lib"


def ensure_wsl_cuda_env() -> None:
    """Expose WSL2 host driver binaries/libraries to child processes."""
    wsl_cuda = wsl_cuda_dir()
    if not os.path.isdir(wsl_cuda):
        return
    path = os.environ.get("PATH", "")
    if wsl_cuda not in path.split(":"):
        os.environ["PATH"] = f"{wsl_cuda}:{path}" if path else wsl_cuda
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if wsl_cuda not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{wsl_cuda}:{ld_path}" if ld_path else wsl_cuda


def find_nvidia_smi() -> Optional[str]:
    """Find nvidia-smi on Linux, WSL, or native Windows."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        return nvidia_smi
    if os.name == "nt":
        roots = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramW6432"),
            r"C:\Program Files",
        ]
        for root in roots:
            if not root:
                continue
            candidate = os.path.join(root, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
            if os.path.isfile(candidate):
                return candidate
    wsl_smi = os.path.join(wsl_cuda_dir(), "nvidia-smi")
    if os.path.isfile(wsl_smi):
        return wsl_smi
    return None
