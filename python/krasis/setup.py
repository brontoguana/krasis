"""Krasis Setup — one-command environment setup for CUDA + GPU dependencies.

Handles:
  1. CUDA toolkit installation (nvcc needed for kernel compilation)
  2. CUDA-enabled PyTorch
  3. GPU kernel compilation (Marlin GEMM vendored, no sgl-kernel needed)

Usage:
    krasis-setup          # auto-detect and install everything
    sudo krasis-setup     # if CUDA toolkit needs installing

The launcher calls lightweight checks on every start, but this script
handles the heavy one-time setup including system packages that need sudo.
"""

import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


def _run(cmd, check=True, **kwargs):
    """Run a command, printing it first."""
    print(f"  {DIM}$ {' '.join(cmd)}{NC}")
    return subprocess.run(cmd, check=check, **kwargs)


# Package approval mode — set during main(), used by _install_system_deps
_auto_approve = False


def _pkg_flag():
    """Return ['-y'] if auto-approve, else [] so the package manager prompts."""
    return ["-y"] if _auto_approve else []


def _has_nvidia_gpu():
    """Check if an NVIDIA GPU is present."""
    nvidia_smi = _find_nvidia_smi()
    if not nvidia_smi:
        return False
    try:
        subprocess.check_output([nvidia_smi], timeout=10, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _find_nvidia_smi():
    """Find nvidia-smi, including the WSL2 driver mount."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        return nvidia_smi
    wsl_smi = "/usr/lib/wsl/lib/nvidia-smi"
    if os.path.isfile(wsl_smi):
        return wsl_smi
    return None


def _ensure_wsl_cuda_env():
    """Expose the WSL2 host driver libraries to subprocesses."""
    wsl_cuda = "/usr/lib/wsl/lib"
    if not os.path.isdir(wsl_cuda):
        return
    path = os.environ.get("PATH", "")
    if wsl_cuda not in path.split(":"):
        os.environ["PATH"] = f"{wsl_cuda}:{path}" if path else wsl_cuda
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if wsl_cuda not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{wsl_cuda}:{ld_path}" if ld_path else wsl_cuda


def _get_cuda_version_from_driver():
    """Detect PyTorch CUDA wheel tag from the NVIDIA driver version.

    This is retained for compatibility with callers/tests that expect the old
    helper, but setup now uses _select_torch_cuda() so GPU architecture is part
    of the decision. Driver version alone is not enough: a modern driver can run
    both cu126 and cu128 wheels, but RTX 50-series GPUs need cu128+ binaries.
    """
    try:
        nvidia_smi = _find_nvidia_smi() or "nvidia-smi"
        out = subprocess.check_output(
            [nvidia_smi, "--query-gpu=driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().split("\n")[0]
        major = int(out.split(".")[0])
        if major >= 560:
            return "cu126", "12.6"
        elif major >= 545:
            return "cu124", "12.4"
        elif major >= 535:
            return "cu121", "12.1"
        elif major >= 525:
            return "cu118", "11.8"
    except Exception:
        pass
    return "cu126", "12.6"  # default


def _parse_version_tuple(value: str) -> Optional[Tuple[int, int]]:
    try:
        major, minor = value.strip().split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return None


def _cuda_tag(version: Tuple[int, int]) -> str:
    return f"cu{version[0]}{version[1]}"


def _cuda_version_string(version: Tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def _get_visible_gpu_compute_caps() -> List[Dict[str, object]]:
    """Return visible NVIDIA GPUs with compute capability from nvidia-smi."""
    try:
        nvidia_smi = _find_nvidia_smi() or "nvidia-smi"
        out = subprocess.check_output(
            [nvidia_smi, "--query-gpu=index,name,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip()
    except Exception:
        return []

    gpus: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",", 2)]
        if len(parts) != 3:
            continue
        cap = _parse_version_tuple(parts[2])
        if cap is None:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            index = len(gpus)
        gpus.append({"index": index, "name": parts[1], "capability": cap})
    return gpus


def _required_cuda_for_compute_cap(sm: Tuple[int, int]) -> Tuple[int, int]:
    """Minimum CUDA runtime/toolkit family required for a GPU architecture."""
    if sm >= (13, 0):
        return (13, 0)
    if sm >= (12, 0):
        return (12, 8)
    if sm >= (10, 0):
        return (12, 6)
    return (12, 6)


def _select_torch_cuda() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Select the PyTorch CUDA wheel index for the visible GPUs.

    Returns (tag, version, error). The selected wheel is the minimum official
    PyTorch CUDA wheel that satisfies the newest visible GPU architecture and
    does not exceed the driver CUDA runtime reported by nvidia-smi.
    """
    required = _get_required_cuda_version()
    driver = _get_driver_cuda_version()
    if driver is None:
        driver = required
    if driver < required:
        return (
            None,
            None,
            f"driver exposes CUDA {_cuda_version_string(driver)}, but visible GPUs "
            f"need CUDA {_cuda_version_string(required)}+",
        )

    candidates = [(11, 8), (12, 1), (12, 4), (12, 6), (12, 8), (13, 0)]
    for candidate in candidates:
        if candidate >= required and candidate <= driver:
            return _cuda_tag(candidate), _cuda_version_string(candidate), None

    return (
        None,
        None,
        f"no supported PyTorch CUDA wheel satisfies required "
        f"CUDA {_cuda_version_string(required)} with driver CUDA "
        f"{_cuda_version_string(driver)}",
    )


def _get_nvcc_version():
    """Get nvcc major.minor version, or None if not found."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        for path in [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.8/bin/nvcc",
            "/usr/local/cuda-12.6/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
            "/usr/local/cuda-12.1/bin/nvcc",
            "/usr/local/cuda-11.8/bin/nvcc",
            "/usr/bin/nvcc",
        ]:
            if os.path.isfile(path):
                nvcc = path
                break
    if not nvcc:
        return None
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True, timeout=5)
        # Parse "release 12.8, V12.8.89" or similar
        for line in out.split("\n"):
            if "release" in line:
                # e.g. "Cuda compilation tools, release 12.8, V12.8.89"
                parts = line.split("release")[-1].strip().split(",")[0].strip()
                major, minor = parts.split(".")[:2]
                return (int(major), int(minor))
    except Exception:
        pass
    return (0, 0)  # found but can't parse version


def _has_nvcc():
    """Check if nvcc is available (needed for CUDA kernel compilation)."""
    return _get_nvcc_version() is not None


def _detect_distro():
    """Detect Linux distro for package manager selection.

    Returns (family, version_id) e.g. ("debian", "24.04") or ("rhel", "39").
    """
    try:
        with open("/etc/os-release") as f:
            lines = f.read()
        version_id = ""
        for line in lines.split("\n"):
            if line.startswith("VERSION_ID="):
                version_id = line.split("=", 1)[1].strip('"')
        if "Ubuntu" in lines or "Debian" in lines:
            return "debian", version_id
        elif "Fedora" in lines or "Red Hat" in lines or "CentOS" in lines:
            return "rhel", version_id
        elif "Arch" in lines:
            return "arch", version_id
    except FileNotFoundError:
        pass
    return "unknown", ""


def _is_wsl():
    """Check if running in WSL."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def _get_driver_cuda_version():
    """Get the CUDA version supported by the installed driver.

    Returns (major, minor) tuple from nvidia-smi, e.g. (12, 8).
    Returns None if it can't be determined.
    """
    try:
        nvidia_smi = _find_nvidia_smi() or "nvidia-smi"
        out = subprocess.check_output(
            [nvidia_smi], text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        # Parse "CUDA Version: 12.8" from nvidia-smi header
        for line in out.split("\n"):
            if "CUDA Version:" in line:
                ver_str = line.split("CUDA Version:")[1].strip().split()[0]
                major, minor = ver_str.split(".")[:2]
                return (int(major), int(minor))
    except Exception:
        pass
    return None


def _get_required_cuda_version():
    """Determine which CUDA toolkit version to install.

    Uses the minimum toolkit version needed for the GPU architecture. The CUDA
    version printed by nvidia-smi is the driver's maximum supported runtime API,
    not the toolkit version Krasis needs to build or run. On WSL2 especially, a
    new Windows driver may advertise a newer CUDA version than NVIDIA's WSL
    toolkit repo provides.
    Returns (major, minor) tuple, e.g. (12, 8).
    """
    gpus = _get_visible_gpu_compute_caps()
    if gpus:
        required = (12, 6)
        for gpu in gpus:
            required = max(
                required,
                _required_cuda_for_compute_cap(gpu["capability"]),
            )
        return required
    return (12, 6)  # safe default


def _need_python_dev():
    """Check if Python development headers are installed (needed for pyo3)."""
    import sysconfig
    inc = sysconfig.get_path("include")
    if inc and os.path.isfile(os.path.join(inc, "Python.h")):
        return False
    return True


def _torch_arch_token(capability: str) -> str:
    major, minor = capability.split(".")[:2]
    return f"sm_{major}{minor}"


def _probe_torch_cuda() -> Dict[str, object]:
    """Probe torch in a subprocess so reinstall checks see the actual package."""
    code = r"""
import json
result = {"installed": False, "cuda_available": False, "devices": [], "arch_list": []}
try:
    import torch
    result["installed"] = True
    result["version"] = torch.__version__
    result["cuda_version"] = getattr(torch.version, "cuda", None)
    result["cuda_available"] = bool(torch.cuda.is_available())
    if hasattr(torch.cuda, "get_arch_list"):
        result["arch_list"] = list(torch.cuda.get_arch_list())
    if result["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            result["devices"].append({
                "index": i,
                "name": props.name,
                "capability": f"{cap[0]}.{cap[1]}",
            })
except Exception as exc:
    result["error"] = str(exc)
print(json.dumps(result))
"""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", code],
            text=True,
            timeout=30,
            stderr=subprocess.DEVNULL,
        )
        import json
        return json.loads(out)
    except Exception as exc:
        return {"installed": False, "cuda_available": False, "error": str(exc)}


def _unsupported_torch_devices(torch_probe: Dict[str, object]) -> List[str]:
    """Return visible devices whose SM arch is missing from installed torch.

    A GPU with compute capability X.Y can run any kernel compiled for an
    architecture <= X.Y (CUDA forward compatibility).  We only flag a device
    as unsupported when *no* arch in the torch binary is <= the device cap.
    """
    if not torch_probe.get("installed") or not torch_probe.get("cuda_available"):
        return []
    arch_list_raw = torch_probe.get("arch_list") or []
    devices = torch_probe.get("devices") or []
    if not arch_list_raw or not isinstance(devices, list):
        return []

    # Parse arch tokens like "sm_86" into (major, minor) tuples.
    compiled_caps: List[Tuple[int, int]] = []
    for tok in arch_list_raw:
        t = str(tok)
        if t.startswith("sm_") or t.startswith("compute_"):
            v = t.split("_", 1)[1]
            try:
                major, minor = int(v[0]), int(v[1]) if len(v) > 1 else 0
                compiled_caps.append((major, minor))
            except (ValueError, IndexError):
                pass

    unsupported = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        capability = str(device.get("capability") or "")
        if "." not in capability:
            continue
        try:
            parts = capability.split(".")
            dev_cap = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            continue
        # The device is supported if ANY compiled arch is <= device cap.
        if not any(ca <= dev_cap for ca in compiled_caps):
            token = _torch_arch_token(capability)
            unsupported.append(
                f"GPU {device.get('index')} {device.get('name')} ({token})"
            )
    return unsupported


def _show_required_packages():
    """Show what packages are needed without installing anything."""
    print(f"\n{BOLD}Required packages:{NC}\n")

    nvcc_ver = _get_nvcc_version()
    required_ver = _get_required_cuda_version()
    need_nvcc = nvcc_ver is None or nvcc_ver < required_ver
    need_ninja = not shutil.which("ninja") and not shutil.which("ninja-build")
    need_pydev = _need_python_dev()

    distro, _ = _detect_distro()

    if not need_nvcc and not need_ninja and not need_pydev:
        print(f"  {GREEN}All system packages already installed.{NC}")
    else:
        if need_nvcc:
            cuda_pkg = f"cuda-toolkit-{required_ver[0]}-{required_ver[1]}"
            if nvcc_ver:
                print(f"  • {cuda_pkg}  (have nvcc {nvcc_ver[0]}.{nvcc_ver[1]}, need {required_ver[0]}.{required_ver[1]}+)")
            else:
                print(f"  • {cuda_pkg}")
        if need_ninja:
            print(f"  • ninja-build")
        if need_pydev:
            py_ver_dot = f"{sys.version_info.major}.{sys.version_info.minor}"
            print(f"  • python{py_ver_dot}-dev")

    # Check Python packages
    torch_probe = _probe_torch_cuda()
    has_torch = bool(torch_probe.get("installed") and torch_probe.get("cuda_available"))
    unsupported = _unsupported_torch_devices(torch_probe)
    if not has_torch or unsupported:
        cu_tag, cu_ver, error = _select_torch_cuda()
        if error:
            print(f"  • torch — {error}")
        else:
            print(f"  • torch (CUDA {cu_ver}) — pip install torch --index-url https://download.pytorch.org/whl/{cu_tag}")
        for item in unsupported:
            print(f"    unsupported by installed torch: {item}")

    # sgl-kernel is no longer needed — Marlin GEMM kernels are vendored in libkrasis_marlin.so

    print(f"\n  Install these manually, then re-run {BOLD}krasis-setup{NC}.\n")


def _install_system_deps():
    """Install system packages: CUDA toolkit (nvcc), ninja, and python-dev headers."""
    print(f"\n{BOLD}Step 1: System Packages (nvcc, ninja, python-dev){NC}")

    need_ninja = not shutil.which("ninja") and not shutil.which("ninja-build")
    need_pydev = _need_python_dev()

    # Check nvcc: either missing or too old for this GPU
    nvcc_ver = _get_nvcc_version()
    required_ver = _get_required_cuda_version()
    need_nvcc = nvcc_ver is None
    nvcc_too_old = False
    if nvcc_ver is not None and nvcc_ver < required_ver:
        nvcc_too_old = True
        need_nvcc = True

    if not need_nvcc and not need_ninja and not need_pydev:
        print(f"  {GREEN}nvcc {nvcc_ver[0]}.{nvcc_ver[1]}, ninja, and python-dev already installed.{NC}")
        return True

    missing = []
    if need_nvcc:
        if nvcc_too_old:
            missing.append(f"nvcc (have {nvcc_ver[0]}.{nvcc_ver[1]}, need {required_ver[0]}.{required_ver[1]}+)")
        else:
            missing.append("nvcc")
    if need_ninja:
        missing.append("ninja")
    if need_pydev:
        missing.append("python-dev headers (Python.h)")
    print(f"  {YELLOW}{'Need upgrade' if nvcc_too_old else 'Missing'}: {', '.join(missing)}{NC}")
    print(f"  Installing (will ask for your password)...\n")

    distro, distro_ver = _detect_distro()
    is_wsl = _is_wsl()
    sudo = [] if os.geteuid() == 0 else ["sudo"]

    # Install ninja and python-dev headers (simple apt/dnf packages)
    apt_pkgs = []
    dnf_pkgs = []
    if need_ninja:
        apt_pkgs.append("ninja-build")
        dnf_pkgs.append("ninja-build")
    if need_pydev:
        py_ver_dot = f"{sys.version_info.major}.{sys.version_info.minor}"
        apt_pkgs.append(f"python{py_ver_dot}-dev")
        dnf_pkgs.append(f"python{sys.version_info.major}-devel")

    if apt_pkgs or dnf_pkgs:
        if distro == "debian":
            _run(sudo + ["apt-get", "update", "-qq"], check=False)
            ret = _run(sudo + ["apt-get", "install"] + _pkg_flag() + apt_pkgs, check=False)
        elif distro == "rhel":
            ret = _run(sudo + ["dnf", "install"] + _pkg_flag() + dnf_pkgs, check=False)
        else:
            ret = type("R", (), {"returncode": 1})()
        if ret.returncode != 0:
            print(f"  {RED}Failed to install {', '.join(apt_pkgs or dnf_pkgs)}.{NC}")

    # Install CUDA toolkit from NVIDIA repo (not the ancient distro package)
    if need_nvcc and distro == "debian":
        cuda_pkg = f"cuda-toolkit-{required_ver[0]}-{required_ver[1]}"
        if is_wsl:
            repo = "wsl-ubuntu"
        elif distro_ver.startswith("24."):
            repo = "ubuntu2404"
        else:
            repo = "ubuntu2204"  # works for 22.04 and most Debian
        keyring_url = (
            f"https://developer.download.nvidia.com/compute/cuda/repos/"
            f"{repo}/x86_64/cuda-keyring_1.1-1_all.deb"
        )
        print(f"  Adding NVIDIA CUDA {required_ver[0]}.{required_ver[1]} repository...")

        # Remove old distro nvcc if present — always prompt for removals
        if nvcc_too_old:
            print(f"  {YELLOW}Will remove old nvidia-cuda-toolkit before installing new version.{NC}")
            _run(sudo + ["apt-get", "remove"] + _pkg_flag() + ["nvidia-cuda-toolkit"], check=False)

        # Download and install keyring
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".deb", delete=False) as tmp:
            keyring_path = tmp.name
        ret = _run(["wget", "-q", "-O", keyring_path, keyring_url], check=False)
        if ret.returncode != 0:
            # Try curl as fallback
            ret = _run(["curl", "-sL", "-o", keyring_path, keyring_url], check=False)
        cuda_installed = False
        if ret.returncode != 0:
            print(f"  {YELLOW}Could not download CUDA keyring, checking if nvcc is already available...{NC}")
        else:
            cmds = [
                sudo + ["dpkg", "-i", keyring_path],
                sudo + ["apt-get", "update", "-qq"],
            ]
            repo_ok = True
            for cmd in cmds:
                ret = _run(cmd, check=False)
                if ret.returncode != 0:
                    repo_ok = False
                    break

            if repo_ok:
                # Try exact version first, fall back to major meta-package
                ret = _run(sudo + ["apt-get", "install"] + _pkg_flag() + [cuda_pkg], check=False)
                if ret.returncode != 0:
                    fallback_pkg = f"cuda-toolkit-{required_ver[0]}"
                    print(f"  {YELLOW}{cuda_pkg} not found, trying {fallback_pkg}...{NC}")
                    ret = _run(sudo + ["apt-get", "install"] + _pkg_flag() + [fallback_pkg], check=False)
                if ret.returncode == 0:
                    cuda_installed = True
            else:
                print(f"  {YELLOW}Could not set up CUDA repo, checking if nvcc is already available...{NC}")

            # Clean up keyring
            try:
                os.unlink(keyring_path)
            except OSError:
                pass

        # Add to PATH if needed
        cuda_bin = f"/usr/local/cuda-{required_ver[0]}.{required_ver[1]}/bin"
        if not os.path.isdir(cuda_bin):
            cuda_bin = "/usr/local/cuda/bin"
        if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
            print(f"  Added {cuda_bin} to PATH for this session.")
        # Set CUDA_HOME so PyTorch JIT finds the right toolkit
        cuda_home = os.path.dirname(cuda_bin)
        if os.path.isdir(cuda_home):
            os.environ["CUDA_HOME"] = cuda_home
            # Add to bashrc so it persists
            bashrc = os.path.expanduser("~/.bashrc")
            export_line = f'export PATH={cuda_bin}:$PATH'
            export_cuda_home = f'export CUDA_HOME={cuda_home}'
            try:
                existing = ""
                if os.path.isfile(bashrc):
                    with open(bashrc) as f:
                        existing = f.read()
                lines_to_add = []
                if export_line not in existing:
                    lines_to_add.append(export_line)
                if export_cuda_home not in existing:
                    lines_to_add.append(export_cuda_home)
                if lines_to_add:
                    with open(bashrc, "a") as f:
                        f.write(f"\n# Added by krasis-setup\n")
                        for line in lines_to_add:
                            f.write(f"{line}\n")
                    print(f"  Added to ~/.bashrc (will persist across sessions).")
            except OSError:
                print(f"  {DIM}Add to ~/.bashrc manually: {export_line}{NC}")

    elif need_nvcc and distro == "rhel":
        cuda_pkg = f"cuda-toolkit-{required_ver[0]}-{required_ver[1]}"
        ret = _run(sudo + ["dnf", "install"] + _pkg_flag() + [cuda_pkg], check=False)
        if ret.returncode != 0:
            fallback_pkg = f"cuda-toolkit-{required_ver[0]}"
            print(f"  {YELLOW}{cuda_pkg} not found, trying {fallback_pkg}...{NC}")
            ret = _run(sudo + ["dnf", "install"] + _pkg_flag() + [fallback_pkg], check=False)
            if ret.returncode != 0:
                print(f"  {YELLOW}Could not install {cuda_pkg} via dnf, checking if nvcc is already available...{NC}")

        # Add to PATH and set CUDA_HOME (same as debian path)
        cuda_bin = f"/usr/local/cuda-{required_ver[0]}.{required_ver[1]}/bin"
        if not os.path.isdir(cuda_bin):
            cuda_bin = "/usr/local/cuda/bin"
        if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
            print(f"  Added {cuda_bin} to PATH for this session.")
        cuda_home = os.path.dirname(cuda_bin)
        if os.path.isdir(cuda_home):
            os.environ["CUDA_HOME"] = cuda_home
            bashrc = os.path.expanduser("~/.bashrc")
            export_line = f'export PATH={cuda_bin}:$PATH'
            export_cuda_home = f'export CUDA_HOME={cuda_home}'
            try:
                existing = ""
                if os.path.isfile(bashrc):
                    with open(bashrc) as f:
                        existing = f.read()
                lines_to_add = []
                if export_line not in existing:
                    lines_to_add.append(export_line)
                if export_cuda_home not in existing:
                    lines_to_add.append(export_cuda_home)
                if lines_to_add:
                    with open(bashrc, "a") as f:
                        f.write(f"\n# Added by krasis-setup\n")
                        for line in lines_to_add:
                            f.write(f"{line}\n")
                    print(f"  Added to ~/.bashrc (will persist across sessions).")
            except OSError:
                print(f"  {DIM}Add to ~/.bashrc manually: {export_line}{NC}")
    elif need_nvcc:
        print(f"  {YELLOW}Unknown distro, checking if nvcc is already available...{NC}")

    # Verify
    ok = True
    new_nvcc = _get_nvcc_version()
    if need_nvcc and (new_nvcc is None or new_nvcc < required_ver):
        if new_nvcc:
            print(f"  {YELLOW}nvcc {new_nvcc[0]}.{new_nvcc[1]} still too old (need {required_ver[0]}.{required_ver[1]}+).{NC}")
        else:
            print(f"  {YELLOW}nvcc still not found on PATH.{NC}")
        print(f"  Try: export PATH=/usr/local/cuda/bin:$PATH")
        ok = False
    elif need_nvcc and new_nvcc:
        print(f"  {GREEN}nvcc {new_nvcc[0]}.{new_nvcc[1]} installed.{NC}")
    if need_ninja and not shutil.which("ninja") and not shutil.which("ninja-build"):
        print(f"  {YELLOW}ninja still not found on PATH.{NC}")
        ok = False
    if need_pydev and _need_python_dev():
        print(f"  {YELLOW}Python.h still not found. Install python3-dev manually.{NC}")
        ok = False
    if ok:
        print(f"  {GREEN}System packages installed successfully.{NC}")
    return ok


def _install_cuda_torch():
    """Install CUDA-enabled PyTorch."""
    print(f"\n{BOLD}Step 2: CUDA PyTorch{NC}")

    need_reinstall = False
    torch_probe = _probe_torch_cuda()
    unsupported = _unsupported_torch_devices(torch_probe)

    if torch_probe.get("installed") and torch_probe.get("cuda_available") and not unsupported:
        print(f"  {GREEN}CUDA torch already installed (v{torch_probe.get('version')}).{NC}")
        return True
    if torch_probe.get("installed") and not torch_probe.get("cuda_available"):
        print(f"  torch {torch_probe.get('version')} installed but CUDA not available.")
        need_reinstall = True
    elif unsupported:
        print(f"  {YELLOW}Installed torch does not support all visible GPUs.{NC}")
        for item in unsupported:
            print(f"    {item}")
        need_reinstall = True
    else:
        print(f"  torch not installed.")

    cu_tag, cu_ver, error = _select_torch_cuda()
    if error:
        print(f"  {RED}Cannot select a supported CUDA torch wheel: {error}.{NC}")
        print(f"  Update the NVIDIA driver or hide unsupported GPUs with CUDA_VISIBLE_DEVICES.")
        return False
    index_url = f"https://download.pytorch.org/whl/{cu_tag}"
    print(f"  Installing CUDA torch ({cu_tag})...")

    pip_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "--index-url", index_url,
        "--quiet", "--no-warn-conflicts",
    ]
    if need_reinstall:
        pip_cmd.append("--force-reinstall")
    ret = _run(pip_cmd, check=False)

    if ret.returncode != 0:
        print(f"  {RED}Failed. Install manually:{NC}")
        print(f"    pip install torch --index-url {index_url}")
        return False

    torch_probe = _probe_torch_cuda()
    unsupported = _unsupported_torch_devices(torch_probe)
    if not torch_probe.get("installed") or not torch_probe.get("cuda_available"):
        print(f"  {RED}Installed torch, but CUDA is still not available.{NC}")
        if torch_probe.get("error"):
            print(f"    {torch_probe.get('error')}")
        return False
    if unsupported:
        print(f"  {RED}Installed torch, but it still lacks support for visible GPUs:{NC}")
        for item in unsupported:
            print(f"    {item}")
        return False

    print(f"  {GREEN}CUDA torch installed.{NC}")
    return True


def _install_gpu_packages():
    """GPU packages check (Marlin GEMM kernels are now vendored)."""
    print(f"\n{BOLD}Step 3: GPU Kernels{NC}")
    print(f"  {GREEN}Marlin GEMM kernels are vendored in libkrasis_marlin.so (no pip install needed).{NC}")
    return True


def main():
    global _auto_approve

    _ensure_wsl_cuda_env()

    print(f"\n{BOLD}{CYAN}Krasis Setup{NC}")
    print(f"{DIM}{'─' * 50}{NC}\n")

    # Check platform
    if platform.system() != "Linux":
        print(f"{YELLOW}Krasis GPU support requires Linux (or WSL).{NC}")
        return

    # Check for NVIDIA GPU — Krasis requires GPU for prefill
    if not _has_nvidia_gpu():
        print(f"{RED}No NVIDIA GPU detected. Krasis requires at least one NVIDIA GPU.{NC}")
        if _is_wsl():
            print(f"  WSL2 must see the Windows NVIDIA driver through /usr/lib/wsl/lib.")
            print(f"  Check inside WSL:")
            print(f"    ls -l /usr/lib/wsl/lib/nvidia-smi /usr/lib/wsl/lib/libcuda.so.1")
            print(f"    /usr/lib/wsl/lib/nvidia-smi")
            print(f"  If those files are missing or nvidia-smi fails, install/update the")
            print(f"  Windows NVIDIA driver with WSL CUDA support, then run: wsl --shutdown")
            print(f"  Do not install a Linux nvidia-driver package inside WSL.")
        else:
            print(f"  If you have an NVIDIA GPU, install the driver first:")
            print(f"    sudo apt install nvidia-driver-560  # or newer")
        sys.exit(1)

    print(f"NVIDIA GPU detected.\n")

    # Ask about package approval mode
    print(f"  Krasis may need to install or upgrade system packages")
    print(f"  (CUDA toolkit, ninja, python-dev, etc).\n")
    print(f"  Options:")
    print(f"    {BOLD}[A]{NC} Auto-approve all package installations")
    print(f"    {BOLD}[R]{NC} Review each installation before it proceeds")
    print(f"    {BOLD}[M]{NC} Manual — show what's needed, install nothing\n")
    try:
        answer = input(f"  Package install mode? [A/r/m] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "a"

    if answer.startswith("m"):
        print(f"\n  {YELLOW}Manual mode — will show required packages but not install them.{NC}")
        print(f"  You can re-run krasis-setup after installing them yourself.\n")
        _auto_approve = False
        _show_required_packages()
        return
    elif answer.startswith("r"):
        print(f"\n  {CYAN}Review mode — each package install will show its plan for your approval.{NC}\n")
        _auto_approve = False
    else:
        print(f"\n  {CYAN}Auto-approve mode — packages will be installed without prompting.{NC}\n")
        _auto_approve = True

    results = {}

    # Step 1: CUDA toolkit
    results["system"] = _install_system_deps()

    # Step 2: CUDA torch
    results["torch"] = _install_cuda_torch()

    # Step 3: GPU packages
    results["packages"] = _install_gpu_packages()

    # Summary
    print(f"\n{DIM}{'─' * 50}{NC}")
    print(f"{BOLD}Summary:{NC}")
    for name, ok in results.items():
        if ok is True:
            print(f"  {GREEN}✓{NC} {name}")
        elif ok is False:
            print(f"  {RED}✗{NC} {name}")
        else:
            print(f"  {DIM}–{NC} {name} (skipped)")

    if all(v is True for v in results.values()):
        print(f"\n{GREEN}{BOLD}Setup complete! Run 'krasis' to start.{NC}\n")
    else:
        print(f"\n{YELLOW}Some steps failed. See above for details.{NC}\n")


if __name__ == "__main__":
    main()
