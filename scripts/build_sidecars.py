#!/usr/bin/env python3
"""Build and verify Krasis vendored CUDA sidecars.

Marlin and FlashAttention are runtime-loaded shared libraries, not Rust
compile inputs.  This script builds them before maturin packages the wheel and
records a manifest that is checked by local builds and release CI.

Must be run through ./dev or release CI with KRASIS_DEV_SCRIPT=1.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print("ERROR: scripts/build_sidecars.py must be run through ./dev, not directly.", file=sys.stderr)
    sys.exit(1)


REPO = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO / "python" / "krasis"
BUILD_ROOT = REPO / "target" / "sidecars"
BUNDLE_DIR = BUILD_ROOT / "bundles"
MANIFEST_PATH = PACKAGE_DIR / "sidecar_manifest.json"
FLA_CONTRACT_PATH = PACKAGE_DIR / "fla_sidecar_contract.json"
CUDA_GLIBC_COMPAT_HEADER = REPO / "src" / "cuda" / "cuda_glibc_compat.h"


def read_sidecar_abi_version() -> int:
    path = REPO / "sidecar_abi_version.txt"
    try:
        return int(path.read_text().strip())
    except Exception as exc:
        raise SystemExit(f"ERROR: invalid sidecar ABI version file {path}: {exc}") from exc


SIDECAR_ABI_VERSION = read_sidecar_abi_version()

IS_WINDOWS = os.name == "nt"
MARLIN_SO = "krasis_marlin.dll" if IS_WINDOWS else "libkrasis_marlin.so"
FLASH_ATTN_SO = "krasis_flash_attn.dll" if IS_WINDOWS else "libkrasis_flash_attn.so"


def read_fla_architectures() -> tuple[int, ...]:
    try:
        contract = json.loads(FLA_CONTRACT_PATH.read_text(encoding="utf-8"))
        architectures = tuple(int(arch) for arch in contract["architectures"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"ERROR: invalid FLA sidecar contract {FLA_CONTRACT_PATH}: {exc}"
        ) from exc
    if (
        contract.get("schema_version") != 1
        or not architectures
        or len(set(architectures)) != len(architectures)
        or any(arch <= 0 for arch in architectures)
    ):
        raise SystemExit(
            f"ERROR: invalid FLA architecture inventory in {FLA_CONTRACT_PATH}"
        )
    return architectures


FLA_ARCHS = read_fla_architectures()

MARLIN_SYMBOLS = [
    "krasis_sidecar_abi_version",
    "krasis_sidecar_build_id",
    "krasis_marlin_mm_bf16",
    "krasis_marlin_moe_mm_bf16",
    "krasis_moe_zero_and_scatter_weighted_bf16",
]
FLASH_ATTN_SYMBOLS = [
    "krasis_sidecar_abi_version",
    "krasis_sidecar_build_id",
    "krasis_flash_attn_fwd_bf16",
    "krasis_flash_attn_fwd_bf16_window",
    "krasis_flash_attn_fwd_bf16q_fp8kv",
]

DEFAULT_BUNDLE_PLATFORM = "windows-x86_64-cuda126" if IS_WINDOWS else "linux-x86_64-cuda126"
BUNDLE_PLATFORM = os.environ.get("KRASIS_SIDECAR_PLATFORM", DEFAULT_BUNDLE_PLATFORM)
GITHUB_TIMEOUT_SECONDS = int(os.environ.get("KRASIS_GITHUB_TIMEOUT_SECONDS", "15"))
GITHUB_RELEASE_SEARCH_LIMIT = int(os.environ.get("KRASIS_SIDECAR_RELEASE_SEARCH_LIMIT", "25"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def find_nvcc() -> str:
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root:
            for exe in ("nvcc.exe", "nvcc"):
                candidate = Path(root) / "bin" / exe
                if candidate.exists():
                    return str(candidate)
    for candidate in (
        Path("/usr/local/cuda/bin/nvcc"),
        Path("/usr/local/cuda-12.6/bin/nvcc"),
        Path("/usr/local/cuda-12/bin/nvcc"),
    ):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("nvcc")
    if found:
        return found
    raise SystemExit("ERROR: nvcc not found; cannot build Marlin/FlashAttention sidecars")


def command_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def nvcc_host_compiler_args() -> list[str]:
    ccbin = os.environ.get("KRASIS_NVCC_CCBIN", "").strip()
    args = ["-ccbin", ccbin] if ccbin else []
    # glibc 2.43 exposes rsqrt/rsqrtf under _GNU_SOURCE with noexcept
    # declarations that conflict with CUDA 13.0/13.1's math_functions.h.
    # Match build.rs so direct Marlin/FlashAttention sidecar builds work on
    # the same modern Linux hosts as the Rust-managed CUDA compilation.
    if not IS_WINDOWS:
        args.extend(
            [
                "-U_GNU_SOURCE",
                "-D_DEFAULT_SOURCE",
                "-include",
                rel(CUDA_GLIBC_COMPAT_HEADER),
            ]
        )
    return args


def nvcc_pic_args() -> list[str]:
    return [] if IS_WINDOWS else ["-Xcompiler", "-fPIC"]


def object_suffix() -> str:
    return ".obj" if IS_WINDOWS else ".o"


def timed_run(args: list[str], label: str) -> None:
    start = time.monotonic()
    print(f"[sidecars] {label}")
    proc = subprocess.run(args, cwd=REPO, text=True, capture_output=True)
    elapsed = time.monotonic() - start
    print(f"KRASIS_BUILD_TIMING phase=\"{label}\" duration_s={elapsed:.3f}")
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"ERROR: {label} failed with exit code {proc.returncode}")


def source_files(*roots: Path) -> list[Path]:
    suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inc", ".md"}
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                files.append(path)
    return sorted(set(files), key=lambda p: rel(p))


def flash_attn_cu_files() -> list[str]:
    fa_src = REPO / "src" / "cuda" / "flash_attn" / "fa2"
    cu_files = ["flash_attn_vendor.cu"]
    cu_files.extend(path.name for path in sorted(fa_src.glob("flash_fwd_*.cu")))
    if len(cu_files) <= 1:
        raise SystemExit(f"ERROR: no FlashAttention flash_fwd_*.cu files found in {fa_src}")
    return cu_files


def sidecar_inputs(nvcc: str) -> dict[str, dict[str, object]]:
    marlin_dir = REPO / "src" / "cuda" / "marlin"
    fa_dir = REPO / "src" / "cuda" / "flash_attn" / "fa2"
    cutlass_dir = REPO / "src" / "cuda" / "flash_attn" / "cutlass"
    compat_sources = [CUDA_GLIBC_COMPAT_HEADER] if not IS_WINDOWS else []

    marlin_flags = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
        "-I",
        "src/cuda/marlin",
        f"-DKRASIS_SIDECAR_ABI_VERSION={SIDECAR_ABI_VERSION}",
    ] + nvcc_pic_args() + nvcc_host_compiler_args()

    fa_common_flags = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-allow-unsupported-compiler",
        "-O3",
        "--use_fast_math",
        "-DKRASIS_FA_VENDOR",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_SOFTCAP",
        "-Isrc/cuda/flash_attn/fa2",
        "-Isrc/cuda/flash_attn/cutlass",
        f"-DKRASIS_SIDECAR_ABI_VERSION={SIDECAR_ABI_VERSION}",
    ] + nvcc_pic_args() + nvcc_host_compiler_args()

    env_contract = {
        "nvcc": nvcc,
        "nvcc_version": command_output([nvcc, "--version"]),
        "KRASIS_NVCC_CCBIN": os.environ.get("KRASIS_NVCC_CCBIN", ""),
        "KRASIS_NVCC_CCBIN_VERSION": command_output([os.environ["KRASIS_NVCC_CCBIN"], "--version"])
        if os.environ.get("KRASIS_NVCC_CCBIN")
        else "",
        "KRASIS_FA2_HDIM128_EXTRA_ARCHES": os.environ.get("KRASIS_FA2_HDIM128_EXTRA_ARCHES", ""),
        "KRASIS_FA2_ALL_ARCHES": os.environ.get("KRASIS_FA2_ALL_ARCHES", ""),
    }

    return {
        "marlin": {
            "sources": source_files(marlin_dir, *compat_sources),
            "flags": marlin_flags,
            "env": env_contract,
            "compiled_units": [
                "src/cuda/marlin/marlin_vendor.cu",
                "src/cuda/marlin/marlin_moe_vendor.cu",
            ],
            "symbols": MARLIN_SYMBOLS,
            "output": MARLIN_SO,
        },
        "flash_attn": {
            "sources": source_files(fa_dir, cutlass_dir, *compat_sources),
            "flags": fa_common_flags,
            "env": env_contract,
            "compiled_units": [f"src/cuda/flash_attn/fa2/{name}" for name in flash_attn_cu_files()],
            "symbols": FLASH_ATTN_SYMBOLS,
            "output": FLASH_ATTN_SO,
        },
    }


def input_hash(contract: dict[str, object]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(
            {
                "sidecar_abi": SIDECAR_ABI_VERSION,
                "flags": contract["flags"],
                "env": contract["env"],
                "compiled_units": contract["compiled_units"],
                "symbols": contract["symbols"],
            },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))
    for path in contract["sources"]:  # type: ignore[index]
        assert isinstance(path, Path)
        h.update(rel(path).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def fa2_arch_args(cu_file: str) -> list[str]:
    hdim128_extra = os.environ.get("KRASIS_FA2_HDIM128_EXTRA_ARCHES") == "1"
    all_arches = os.environ.get("KRASIS_FA2_ALL_ARCHES") == "1"
    if not all_arches and not (hdim128_extra and "hdim128" in cu_file):
        return ["-arch=sm_80"]

    archs = [80]
    if all_arches or (hdim128_extra and "hdim128" in cu_file):
        archs.extend([89, 90, 120])

    args: list[str] = []
    for arch in archs:
        args.append("-gencode")
        if arch == 120:
            args.append("arch=compute_120,code=[sm_120,compute_120]")
        else:
            args.append(f"arch=compute_{arch},code=sm_{arch}")
    return args


def build_marlin(nvcc: str, build_id: str, force: bool) -> Path:
    out = BUILD_ROOT / "marlin"
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    common_args = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
        "-I",
        "src/cuda/marlin",
        f"-DKRASIS_SIDECAR_ABI_VERSION={SIDECAR_ABI_VERSION}",
        f"-DKRASIS_SIDECAR_BUILD_ID=\\\"{build_id}\\\"",
    ] + nvcc_pic_args() + nvcc_host_compiler_args()

    suffix = object_suffix()
    obj_regular = out / f"marlin_vendor{suffix}"
    obj_moe = out / f"marlin_moe_vendor{suffix}"
    so_path = out / MARLIN_SO
    timed_run([nvcc, "-c", "-o", str(obj_regular), *common_args, "src/cuda/marlin/marlin_vendor.cu"], "sidecar Marlin regular compile")
    timed_run([nvcc, "-c", "-o", str(obj_moe), *common_args, "src/cuda/marlin/marlin_moe_vendor.cu"], "sidecar Marlin MoE compile")
    timed_run([nvcc, "-shared", "-o", str(so_path), str(obj_regular), str(obj_moe), "-Wno-deprecated-gpu-targets"], "sidecar Marlin link")
    return so_path


def build_flash_attn(nvcc: str, build_id: str, force: bool) -> Path:
    out = BUILD_ROOT / "flash_attn"
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    common_args = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-allow-unsupported-compiler",
        "-O3",
        "--use_fast_math",
        "-DKRASIS_FA_VENDOR",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_SOFTCAP",
        f"-DKRASIS_SIDECAR_ABI_VERSION={SIDECAR_ABI_VERSION}",
        f"-DKRASIS_SIDECAR_BUILD_ID=\\\"{build_id}\\\"",
        "-Isrc/cuda/flash_attn/fa2",
        "-Isrc/cuda/flash_attn/cutlass",
    ] + nvcc_pic_args() + nvcc_host_compiler_args()

    obj_files: list[Path] = []
    for cu_file in flash_attn_cu_files():
        obj_path = out / f"fa2_{cu_file.replace('.cu', object_suffix())}"
        timed_run(
            [
                nvcc,
                "-c",
                "-o",
                str(obj_path),
                *common_args,
                *fa2_arch_args(cu_file),
                f"src/cuda/flash_attn/fa2/{cu_file}",
            ],
            f"sidecar FlashAttention compile {cu_file}",
        )
        obj_files.append(obj_path)

    so_path = out / FLASH_ATTN_SO
    timed_run([nvcc, "-shared", "-o", str(so_path), *[str(p) for p in obj_files], "-Wno-deprecated-gpu-targets"], "sidecar FlashAttention link")
    return so_path


def read_manifest(path: Path = MANIFEST_PATH) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def nm_symbols(path: Path) -> set[str]:
    if IS_WINDOWS:
        return set()
    out = command_output(["nm", "-D", "--defined-only", str(path)])
    symbols: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if parts:
            symbols.add(parts[-1])
    return symbols


def verify_symbols(path: Path, required: list[str]) -> list[str]:
    if IS_WINDOWS:
        try:
            lib = ctypes.CDLL(str(path))
        except Exception as exc:
            return [f"<load failed: {exc}>"]
        return [sym for sym in required if not hasattr(lib, sym)]
    symbols = nm_symbols(path)
    return [sym for sym in required if sym not in symbols]


def verify_loaded_sidecar(path: Path, expected_build_id: str) -> str | None:
    try:
        lib = ctypes.CDLL(str(path))
        abi_fn = lib.krasis_sidecar_abi_version
        abi_fn.restype = ctypes.c_uint32
        actual_abi = int(abi_fn())
        if actual_abi != SIDECAR_ABI_VERSION:
            return f"ABI mismatch: expected {SIDECAR_ABI_VERSION}, got {actual_abi}"
        build_id_fn = lib.krasis_sidecar_build_id
        build_id_fn.restype = ctypes.c_char_p
        actual_build_id = build_id_fn().decode("utf-8")
        if actual_build_id != expected_build_id:
            return f"build_id mismatch: expected {expected_build_id}, got {actual_build_id}"
    except Exception as exc:
        return f"load/build-id check failed: {exc}"
    return None


def copy_to_package(path: Path) -> Path:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    dst = PACKAGE_DIR / path.name
    shutil.copy2(path, dst)
    return dst


def copy_windows_cuda_runtime(nvcc: str) -> list[str]:
    if not IS_WINDOWS:
        return []
    cuda_bin = Path(nvcc).resolve().parent
    candidates = sorted(cuda_bin.glob("cudart64*.dll"))
    if not candidates:
        raise SystemExit(f"ERROR: Windows CUDA runtime DLL not found next to nvcc: {cuda_bin}")
    copied: list[str] = []
    for src in candidates:
        dst = copy_to_package(src)
        copied.append(dst.name)
    return copied


def manifest_matches(manifest: dict[str, object], contracts: dict[str, dict[str, object]]) -> bool:
    if manifest.get("schema_version") != 1 or manifest.get("sidecar_abi") != SIDECAR_ABI_VERSION:
        return False
    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, dict):
        return False
    for name, contract in contracts.items():
        entry = sidecars.get(name)
        if not isinstance(entry, dict):
            return False
        output = PACKAGE_DIR / str(contract["output"])
        if not output.exists():
            return False
        if entry.get("input_hash") != input_hash(contract):
            return False
        if entry.get("sha256") != sha256_file(output):
            return False
        build_id = entry.get("build_id")
        if not isinstance(build_id, str):
            return False
        missing = verify_symbols(output, list(contract["symbols"]))  # type: ignore[arg-type]
        if missing:
            return False
        if verify_loaded_sidecar(output, build_id) is not None:
            return False
    return True


def sidecar_key(contracts: dict[str, dict[str, object]]) -> tuple[str, dict[str, str]]:
    hashes = {name: input_hash(contract) for name, contract in sorted(contracts.items())}
    payload = {
        "schema_version": 1,
        "platform": BUNDLE_PLATFORM,
        "sidecar_abi": SIDECAR_ABI_VERSION,
        "sidecars": hashes,
    }
    digest = sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest, hashes


def bundle_filename(bundle_hash: str) -> str:
    return f"krasis-sidecars-{BUNDLE_PLATFORM}-abi{SIDECAR_ABI_VERSION}-{bundle_hash}.tar.gz"


def bundle_path(contracts: dict[str, dict[str, object]]) -> Path:
    digest, _ = sidecar_key(contracts)
    return BUNDLE_DIR / bundle_filename(digest)


def extract_bundle(path: Path, contracts: dict[str, dict[str, object]]) -> None:
    expected_names = {
        f"krasis/{MARLIN_SO}",
        f"krasis/{FLASH_ATTN_SO}",
        "krasis/sidecar_manifest.json",
    }
    with tempfile.TemporaryDirectory(dir=path.parent) as tmpdir:
        tmp = Path(tmpdir)
        try:
            with tarfile.open(path, "r:gz") as tf:
                members = tf.getmembers()
                names = {member.name for member in members}
                missing = sorted(expected_names - names)
                if missing:
                    raise SystemExit(f"ERROR: sidecar bundle {path.name} missing {', '.join(missing)}")
                for member in members:
                    is_windows_runtime = IS_WINDOWS and member.name.startswith("krasis/cudart64") and member.name.endswith(".dll")
                    if member.name not in expected_names and not is_windows_runtime:
                        raise SystemExit(f"ERROR: sidecar bundle {path.name} contains unexpected entry {member.name}")
                    if not member.isfile():
                        raise SystemExit(f"ERROR: sidecar bundle {path.name} contains non-file entry {member.name}")
                tf.extractall(tmp)
        except tarfile.TarError as exc:
            raise SystemExit(f"ERROR: sidecar bundle {path.name} is not a valid tar.gz archive: {exc}") from exc

        PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        copy_names = list(expected_names)
        if IS_WINDOWS:
            copy_names.extend(name for name in names if name.startswith("krasis/cudart64") and name.endswith(".dll"))
        for name in copy_names:
            src = tmp / name
            dst = PACKAGE_DIR / Path(name).name
            shutil.copy2(src, dst)

    manifest = read_manifest()
    if manifest is None or not manifest_matches(manifest, contracts):
        raise SystemExit(f"ERROR: extracted sidecar bundle {path.name} failed manifest verification")


def create_bundle(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tarfile.open(tmp_path, "w:gz") as tf:
        for src, arcname in (
            (PACKAGE_DIR / MARLIN_SO, f"krasis/{MARLIN_SO}"),
            (PACKAGE_DIR / FLASH_ATTN_SO, f"krasis/{FLASH_ATTN_SO}"),
            (MANIFEST_PATH, "krasis/sidecar_manifest.json"),
        ):
            if not src.exists():
                raise SystemExit(f"ERROR: cannot bundle missing sidecar artifact {src}")
            tf.add(src, arcname=arcname)
        if IS_WINDOWS:
            for src in sorted(PACKAGE_DIR.glob("cudart64*.dll")):
                tf.add(src, arcname=f"krasis/{src.name}")
    tmp_path.replace(path)


def github_token_optional() -> str | None:
    return os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def github_token_required() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("ERROR: GH_TOKEN/GITHUB_TOKEN required for GitHub sidecar bundle access")
    return token


def github_repo() -> str:
    repo = os.environ.get("KRASIS_GITHUB_REPOSITORY") or os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        raise SystemExit("ERROR: KRASIS_GITHUB_REPOSITORY/GITHUB_REPOSITORY required for GitHub sidecar bundle access")
    return repo


def github_request(
    method: str,
    url: str,
    token: str | None,
    data: bytes | None = None,
    content_type: str = "application/json",
    accept: str = "application/vnd.github+json",
) -> tuple[int, bytes]:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=GITHUB_TIMEOUT_SECONDS) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()
    except urllib.error.URLError as exc:
        return 0, str(exc).encode("utf-8")


def github_release_by_tag(token: str | None, repo: str, tag: str) -> dict[str, object] | None:
    api = f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}"
    status, body = github_request("GET", api, token)
    if status == 200:
        return json.loads(body.decode("utf-8"))
    if status == 404:
        return None
    raise SystemExit(f"ERROR: failed to read GitHub release {tag}: HTTP {status}: {body.decode('utf-8', 'replace')}")


def github_releases(token: str | None, repo: str) -> list[dict[str, object]]:
    all_releases: list[dict[str, object]] = []
    page = 1
    while len(all_releases) < GITHUB_RELEASE_SEARCH_LIMIT:
        url = f"https://api.github.com/repos/{repo}/releases?" + urllib.parse.urlencode({"per_page": 100, "page": page})
        status, body = github_request("GET", url, token)
        if status != 200:
            raise SystemExit(f"ERROR: failed to list GitHub releases: HTTP {status}: {body.decode('utf-8', 'replace')}")
        batch = json.loads(body.decode("utf-8"))
        if not isinstance(batch, list) or not batch:
            break
        all_releases.extend(release for release in batch if isinstance(release, dict))
        if len(batch) < 100:
            break
        page += 1
    return all_releases[:GITHUB_RELEASE_SEARCH_LIMIT]


def github_release_assets(token: str | None, release: dict[str, object]) -> list[dict[str, object]]:
    assets = release.get("assets")
    if isinstance(assets, list):
        return [asset for asset in assets if isinstance(asset, dict)]
    assets_url = release.get("assets_url")
    if not isinstance(assets_url, str):
        return []
    all_assets: list[dict[str, object]] = []
    page = 1
    while True:
        url = assets_url + "?" + urllib.parse.urlencode({"per_page": 100, "page": page})
        status, body = github_request("GET", url, token)
        if status != 200:
            raise SystemExit(f"ERROR: failed to list sidecar bundle assets: HTTP {status}: {body.decode('utf-8', 'replace')}")
        batch = json.loads(body.decode("utf-8"))
        if not isinstance(batch, list) or not batch:
            break
        all_assets.extend(asset for asset in batch if isinstance(asset, dict))
        if len(batch) < 100:
            break
        page += 1
    return all_assets


def github_asset(token: str | None, release: dict[str, object], name: str) -> dict[str, object] | None:
    for asset in github_release_assets(token, release):
        if asset.get("name") == name:
            return asset
    return None


def download_github_bundle(name: str, dst: Path) -> bool:
    token = github_token_optional()
    repo = github_repo()
    asset = None
    for release in github_releases(token, repo):
        asset = github_asset(token, release, name)
        if asset is not None:
            tag = release.get("tag_name", "<unknown>")
            print(f"[sidecars] found GitHub sidecar bundle {name} on release {tag}")
            break
    if asset is None:
        print(f"[sidecars] GitHub sidecar bundle miss: {name}")
        return False
    url = asset.get("url")
    if not isinstance(url, str):
        raise SystemExit(f"ERROR: GitHub sidecar asset {name} has no API URL")
    status, body = github_request("GET", url, token, accept="application/octet-stream")
    if status != 200:
        raise SystemExit(f"ERROR: failed to download GitHub sidecar bundle {name}: HTTP {status}: {body.decode('utf-8', 'replace')}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(body)
    print(f"[sidecars] downloaded GitHub sidecar bundle {name}")
    return True


def upload_github_bundle(path: Path) -> None:
    token = github_token_required()
    repo = github_repo()
    tag = os.environ.get("KRASIS_GITHUB_UPLOAD_RELEASE_TAG") or os.environ.get("GITHUB_REF_NAME")
    if not tag:
        raise SystemExit("ERROR: KRASIS_GITHUB_UPLOAD_RELEASE_TAG/GITHUB_REF_NAME required to upload sidecar bundle")
    release = github_release_by_tag(token, repo, tag)
    if release is None:
        raise SystemExit(f"ERROR: upload release {tag} not found; publish the release before uploading sidecar bundles")
    if github_asset(token, release, path.name) is not None:
        print(f"[sidecars] GitHub sidecar bundle already exists on {tag}: {path.name}")
        return
    upload_url = release.get("upload_url")
    if not isinstance(upload_url, str):
        raise SystemExit("ERROR: sidecar bundle release has no upload_url")
    upload_url = upload_url.split("{", 1)[0] + "?" + urllib.parse.urlencode({"name": path.name})
    status, body = github_request("POST", upload_url, token, path.read_bytes(), content_type="application/gzip")
    if status not in (200, 201):
        raise SystemExit(f"ERROR: failed to upload GitHub sidecar bundle {path.name}: HTTP {status}: {body.decode('utf-8', 'replace')}")
    print(f"[sidecars] uploaded GitHub sidecar bundle {path.name}")


def build(args: argparse.Namespace) -> None:
    nvcc = find_nvcc()
    contracts = sidecar_inputs(nvcc)
    manifest = read_manifest()
    if not args.force and manifest is not None and manifest_matches(manifest, contracts):
        print("[sidecars] Marlin/FlashAttention sidecars are current")
        return
    if not args.force:
        cached_bundle = bundle_path(contracts)
        if cached_bundle.exists():
            extract_bundle(cached_bundle, contracts)
            print(f"[sidecars] restored local sidecar bundle {cached_bundle.name}")
            return

    start = time.monotonic()
    entries: dict[str, dict[str, object]] = {}

    marlin_hash = input_hash(contracts["marlin"])
    flash_hash = input_hash(contracts["flash_attn"])

    marlin_so = copy_to_package(build_marlin(nvcc, marlin_hash[:24], args.force))
    flash_so = copy_to_package(build_flash_attn(nvcc, flash_hash[:24], args.force))
    windows_runtime_dlls = copy_windows_cuda_runtime(nvcc)

    for name, path, contract, contract_hash in (
        ("marlin", marlin_so, contracts["marlin"], marlin_hash),
        ("flash_attn", flash_so, contracts["flash_attn"], flash_hash),
    ):
        missing = verify_symbols(path, list(contract["symbols"]))  # type: ignore[arg-type]
        if missing:
            raise SystemExit(f"ERROR: {path} is missing required symbols: {', '.join(missing)}")
        entries[name] = {
            "output": path.name,
            "sha256": sha256_file(path),
            "input_hash": contract_hash,
            "build_id": contract_hash[:24],
            "source_count": len(contract["sources"]),  # type: ignore[arg-type]
            "symbols": contract["symbols"],
        }

    payload = {
        "schema_version": 1,
        "sidecar_abi": SIDECAR_ABI_VERSION,
        "generated_at_unix": int(time.time()),
        "generator": "scripts/build_sidecars.py",
        "nvcc": contracts["marlin"]["env"],  # same env contract for both sidecars
        "sidecars": entries,
        "windows_runtime_dlls": windows_runtime_dlls,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    current_manifest = read_manifest()
    if current_manifest is None or not manifest_matches(current_manifest, contracts):
        raise SystemExit("ERROR: built sidecars failed post-build manifest verification")
    local_bundle = bundle_path(contracts)
    create_bundle(local_bundle)
    elapsed = time.monotonic() - start
    print(f"KRASIS_BUILD_TIMING phase=\"sidecar build total\" duration_s={elapsed:.3f}")
    print(f"[sidecars] wrote {rel(MANIFEST_PATH)}")
    print(f"[sidecars] packed {local_bundle}")


def verify(args: argparse.Namespace) -> None:
    nvcc = find_nvcc()
    contracts = sidecar_inputs(nvcc)
    manifest = read_manifest()
    if manifest is None:
        raise SystemExit(f"ERROR: sidecar manifest missing: {MANIFEST_PATH}\nRun ./dev build-sidecars")
    if not manifest_matches(manifest, contracts):
        raise SystemExit("ERROR: sidecars are missing, stale, or have invalid symbols. Run ./dev build-sidecars")
    missing_fla = [
        name
        for name in (
            f"krasis_fla_sm{arch}.dll" if IS_WINDOWS
            else f"libkrasis_fla_sm{arch}.so"
            for arch in FLA_ARCHS
        )
        if not (PACKAGE_DIR / name).is_file() or (PACKAGE_DIR / name).stat().st_size <= 0
    ]
    if missing_fla:
        raise SystemExit(
            "ERROR: package is missing required FLA sidecars: "
            + ", ".join(missing_fla)
        )
    print("[sidecars] verified package sidecars and manifest")


def probe_sidecar_library(path: Path, required_symbols: list[str]) -> dict[str, object]:
    """Load a sidecar in a child process so Windows can delete it afterwards."""
    probe_code = r"""
import ctypes
import json
import os
import sys

path = sys.argv[1]
required_symbols = json.loads(sys.argv[2])
dll_dir = os.path.dirname(path)
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(dll_dir)
if os.name == "nt":
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

lib = ctypes.CDLL(path)
abi_fn = lib.krasis_sidecar_abi_version
abi_fn.restype = ctypes.c_uint32
build_id_fn = lib.krasis_sidecar_build_id
build_id_fn.restype = ctypes.c_char_p
missing = []
for symbol in required_symbols:
    try:
        getattr(lib, symbol)
    except AttributeError:
        missing.append(symbol)
print(json.dumps({
    "abi": int(abi_fn()),
    "build_id": build_id_fn().decode("utf-8"),
    "missing": missing,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe_code, str(path), json.dumps(required_symbols)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"failed to load {path.name}: {detail}")
    return json.loads(result.stdout)


def verify_wheel(args: argparse.Namespace) -> None:
    wheel_dir = Path(args.wheel_dir)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"ERROR: no wheels found in {wheel_dir}")

    required = {
        f"krasis/{MARLIN_SO}",
        f"krasis/{FLASH_ATTN_SO}",
        "krasis/fla_sidecar_contract.json",
        "krasis/sidecar_manifest.json",
    }
    required.update(
        f"krasis/krasis_fla_sm{arch}.dll" if IS_WINDOWS
        else f"krasis/libkrasis_fla_sm{arch}.so"
        for arch in FLA_ARCHS
    )
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            names = set(zf.namelist())
            missing = sorted(required - names)
            if missing:
                raise SystemExit(f"ERROR: {wheel.name} missing {', '.join(missing)}")
            packaged_fla_contract = zf.read(
                "krasis/fla_sidecar_contract.json"
            )
            if packaged_fla_contract != FLA_CONTRACT_PATH.read_bytes():
                raise SystemExit(
                    f"ERROR: {wheel.name} FLA sidecar contract differs from source"
                )
            windows_runtime_names = sorted(
                name for name in names
                if name.startswith("krasis/cudart64") and name.endswith(".dll")
            )
            if IS_WINDOWS and not windows_runtime_names:
                raise SystemExit(f"ERROR: {wheel.name} missing bundled cudart64*.dll")
            manifest = json.loads(zf.read("krasis/sidecar_manifest.json").decode("utf-8"))
            if manifest.get("schema_version") != 1:
                raise SystemExit(f"ERROR: {wheel.name} manifest schema_version mismatch")
            if manifest.get("sidecar_abi") != SIDECAR_ABI_VERSION:
                raise SystemExit(
                    f"ERROR: {wheel.name} manifest sidecar_abi mismatch: "
                    f"expected {SIDECAR_ABI_VERSION}, got {manifest.get('sidecar_abi')}"
                )
            sidecars = manifest.get("sidecars", {})
            for sidecar_name, so_name in (("marlin", MARLIN_SO), ("flash_attn", FLASH_ATTN_SO)):
                entry = sidecars.get(sidecar_name)
                if not isinstance(entry, dict):
                    raise SystemExit(f"ERROR: {wheel.name} manifest missing {sidecar_name}")
                data = zf.read(f"krasis/{so_name}")
                digest = sha256_bytes(data)
                if entry.get("sha256") != digest:
                    raise SystemExit(f"ERROR: {wheel.name} {so_name} hash mismatch")
                with tempfile.TemporaryDirectory(dir=wheel_dir) as tmpdir:
                    extracted = Path(tmpdir) / so_name
                    extracted.write_bytes(data)
                    for runtime_name in windows_runtime_names:
                        runtime_path = Path(tmpdir) / Path(runtime_name).name
                        runtime_path.write_bytes(zf.read(runtime_name))
                    required_symbols = MARLIN_SYMBOLS if sidecar_name == "marlin" else FLASH_ATTN_SYMBOLS
                    probe = probe_sidecar_library(extracted, required_symbols)
                    actual_abi = int(probe["abi"])
                    if actual_abi != SIDECAR_ABI_VERSION:
                        raise SystemExit(
                            f"ERROR: {wheel.name} {so_name} ABI mismatch: "
                            f"expected {SIDECAR_ABI_VERSION}, got {actual_abi}"
                        )
                    actual_build_id = str(probe["build_id"])
                    if entry.get("build_id") != actual_build_id:
                        raise SystemExit(
                            f"ERROR: {wheel.name} {so_name} build_id mismatch: "
                            f"manifest={entry.get('build_id')} so={actual_build_id}"
                        )
                    missing_symbols = probe.get("missing", [])
                    if missing_symbols:
                        raise SystemExit(
                            f"ERROR: {wheel.name} {so_name} missing required symbol "
                            f"{', '.join(str(symbol) for symbol in missing_symbols)}"
                        )
        print(f"[sidecars] verified wheel {wheel.name}")


def print_bundle_key(args: argparse.Namespace) -> None:
    nvcc = find_nvcc()
    contracts = sidecar_inputs(nvcc)
    digest, hashes = sidecar_key(contracts)
    payload = {
        "bundle_hash": digest,
        "bundle_name": bundle_filename(digest),
        "bundle_path": str(bundle_path(contracts)),
        "platform": BUNDLE_PLATFORM,
        "sidecar_abi": SIDECAR_ABI_VERSION,
        "sidecars": hashes,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["bundle_name"])


def restore_bundle(args: argparse.Namespace) -> None:
    nvcc = find_nvcc()
    contracts = sidecar_inputs(nvcc)
    manifest = read_manifest()
    if manifest is not None and manifest_matches(manifest, contracts):
        print("[sidecars] package sidecars already match current bundle key")
        return

    path = bundle_path(contracts)
    if not path.exists() and args.github:
        download_github_bundle(path.name, path)
    if not path.exists():
        print(f"[sidecars] sidecar bundle miss: {path.name}")
        raise SystemExit(2)

    extract_bundle(path, contracts)
    print(f"[sidecars] restored sidecar bundle {path.name}")


def pack_bundle(args: argparse.Namespace) -> None:
    nvcc = find_nvcc()
    contracts = sidecar_inputs(nvcc)
    manifest = read_manifest()
    if manifest is None or not manifest_matches(manifest, contracts):
        raise SystemExit("ERROR: sidecars are missing or stale; run ./dev build-sidecars before packing a bundle")
    path = bundle_path(contracts)
    create_bundle(path)
    print(f"[sidecars] packed sidecar bundle {path}")


def upload_bundle(args: argparse.Namespace) -> None:
    nvcc = find_nvcc()
    contracts = sidecar_inputs(nvcc)
    manifest = read_manifest()
    if manifest is None or not manifest_matches(manifest, contracts):
        raise SystemExit("ERROR: sidecars are missing or stale; run ./dev build-sidecars before uploading a bundle")
    path = bundle_path(contracts)
    create_bundle(path)
    upload_github_bundle(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--force", action="store_true")
    p_build.set_defaults(func=build)
    p_verify = sub.add_parser("verify")
    p_verify.set_defaults(func=verify)
    p_wheel = sub.add_parser("verify-wheel")
    p_wheel.add_argument("--wheel-dir", required=True)
    p_wheel.set_defaults(func=verify_wheel)
    p_key = sub.add_parser("bundle-key")
    p_key.add_argument("--json", action="store_true")
    p_key.set_defaults(func=print_bundle_key)
    p_restore = sub.add_parser("restore-bundle")
    p_restore.add_argument("--github", action="store_true")
    p_restore.set_defaults(func=restore_bundle)
    p_pack = sub.add_parser("pack-bundle")
    p_pack.set_defaults(func=pack_bundle)
    p_upload = sub.add_parser("upload-bundle")
    p_upload.set_defaults(func=upload_bundle)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
