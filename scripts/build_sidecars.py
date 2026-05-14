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
import tempfile
import time
import zipfile


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print("ERROR: scripts/build_sidecars.py must be run through ./dev, not directly.", file=sys.stderr)
    sys.exit(1)


REPO = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO / "python" / "krasis"
BUILD_ROOT = REPO / "target" / "sidecars"
MANIFEST_PATH = PACKAGE_DIR / "sidecar_manifest.json"


def read_sidecar_abi_version() -> int:
    path = REPO / "sidecar_abi_version.txt"
    try:
        return int(path.read_text().strip())
    except Exception as exc:
        raise SystemExit(f"ERROR: invalid sidecar ABI version file {path}: {exc}") from exc


SIDECAR_ABI_VERSION = read_sidecar_abi_version()

MARLIN_SO = "libkrasis_marlin.so"
FLASH_ATTN_SO = "libkrasis_flash_attn.so"

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
    "krasis_flash_attn_fwd_bf16q_fp8kv",
]


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
            candidate = Path(root) / "bin" / "nvcc"
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
    return ["-ccbin", ccbin] if ccbin else []


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

    marlin_flags = [
        "--expt-relaxed-constexpr",
        "-allow-unsupported-compiler",
        "-Xcompiler",
        "-fPIC",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
        "-I",
        "src/cuda/marlin",
        f"-DKRASIS_SIDECAR_ABI_VERSION={SIDECAR_ABI_VERSION}",
    ] + nvcc_host_compiler_args()

    fa_common_flags = [
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-allow-unsupported-compiler",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "--use_fast_math",
        "-DKRASIS_FA_VENDOR",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_SOFTCAP",
        "-DFLASHATTENTION_DISABLE_LOCAL",
        "-Isrc/cuda/flash_attn/fa2",
        "-Isrc/cuda/flash_attn/cutlass",
        f"-DKRASIS_SIDECAR_ABI_VERSION={SIDECAR_ABI_VERSION}",
    ] + nvcc_host_compiler_args()

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
            "sources": source_files(marlin_dir),
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
            "sources": source_files(fa_dir, cutlass_dir),
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
        "--expt-relaxed-constexpr",
        "-allow-unsupported-compiler",
        "-Xcompiler",
        "-fPIC",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
        "-I",
        "src/cuda/marlin",
        f"-DKRASIS_SIDECAR_ABI_VERSION={SIDECAR_ABI_VERSION}",
        f"-DKRASIS_SIDECAR_BUILD_ID=\\\"{build_id}\\\"",
    ] + nvcc_host_compiler_args()

    obj_regular = out / "marlin_vendor.o"
    obj_moe = out / "marlin_moe_vendor.o"
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
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-allow-unsupported-compiler",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "--use_fast_math",
        "-DKRASIS_FA_VENDOR",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_SOFTCAP",
        "-DFLASHATTENTION_DISABLE_LOCAL",
        f"-DKRASIS_SIDECAR_ABI_VERSION={SIDECAR_ABI_VERSION}",
        f"-DKRASIS_SIDECAR_BUILD_ID=\\\"{build_id}\\\"",
        "-Isrc/cuda/flash_attn/fa2",
        "-Isrc/cuda/flash_attn/cutlass",
    ] + nvcc_host_compiler_args()

    obj_files: list[Path] = []
    for cu_file in flash_attn_cu_files():
        obj_path = out / f"fa2_{cu_file.replace('.cu', '.o')}"
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
    out = command_output(["nm", "-D", "--defined-only", str(path)])
    symbols: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if parts:
            symbols.add(parts[-1])
    return symbols


def verify_symbols(path: Path, required: list[str]) -> list[str]:
    symbols = nm_symbols(path)
    return [sym for sym in required if sym not in symbols]


def copy_to_package(path: Path) -> Path:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    dst = PACKAGE_DIR / path.name
    shutil.copy2(path, dst)
    return dst


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
        missing = verify_symbols(output, list(contract["symbols"]))  # type: ignore[arg-type]
        if missing:
            return False
    return True


def build(args: argparse.Namespace) -> None:
    nvcc = find_nvcc()
    contracts = sidecar_inputs(nvcc)
    manifest = read_manifest()
    if not args.force and manifest is not None and manifest_matches(manifest, contracts):
        print("[sidecars] Marlin/FlashAttention sidecars are current")
        return

    start = time.monotonic()
    entries: dict[str, dict[str, object]] = {}

    marlin_hash = input_hash(contracts["marlin"])
    flash_hash = input_hash(contracts["flash_attn"])

    marlin_so = copy_to_package(build_marlin(nvcc, marlin_hash[:24], args.force))
    flash_so = copy_to_package(build_flash_attn(nvcc, flash_hash[:24], args.force))

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
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elapsed = time.monotonic() - start
    print(f"KRASIS_BUILD_TIMING phase=\"sidecar build total\" duration_s={elapsed:.3f}")
    print(f"[sidecars] wrote {rel(MANIFEST_PATH)}")


def verify(args: argparse.Namespace) -> None:
    nvcc = find_nvcc()
    contracts = sidecar_inputs(nvcc)
    manifest = read_manifest()
    if manifest is None:
        raise SystemExit(f"ERROR: sidecar manifest missing: {MANIFEST_PATH}\nRun ./dev build-sidecars")
    if not manifest_matches(manifest, contracts):
        raise SystemExit("ERROR: sidecars are missing, stale, or have invalid symbols. Run ./dev build-sidecars")
    print("[sidecars] verified package sidecars and manifest")


def verify_wheel(args: argparse.Namespace) -> None:
    wheel_dir = Path(args.wheel_dir)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"ERROR: no wheels found in {wheel_dir}")

    required = {
        f"krasis/{MARLIN_SO}",
        f"krasis/{FLASH_ATTN_SO}",
        "krasis/sidecar_manifest.json",
    }
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            names = set(zf.namelist())
            missing = sorted(required - names)
            if missing:
                raise SystemExit(f"ERROR: {wheel.name} missing {', '.join(missing)}")
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
                    lib = ctypes.CDLL(str(extracted))
                    abi_fn = lib.krasis_sidecar_abi_version
                    abi_fn.restype = ctypes.c_uint32
                    actual_abi = int(abi_fn())
                    if actual_abi != SIDECAR_ABI_VERSION:
                        raise SystemExit(
                            f"ERROR: {wheel.name} {so_name} ABI mismatch: "
                            f"expected {SIDECAR_ABI_VERSION}, got {actual_abi}"
                        )
                    build_id_fn = lib.krasis_sidecar_build_id
                    build_id_fn.restype = ctypes.c_char_p
                    actual_build_id = build_id_fn().decode("utf-8")
                    if entry.get("build_id") != actual_build_id:
                        raise SystemExit(
                            f"ERROR: {wheel.name} {so_name} build_id mismatch: "
                            f"manifest={entry.get('build_id')} so={actual_build_id}"
                        )
                    required_symbols = MARLIN_SYMBOLS if sidecar_name == "marlin" else FLASH_ATTN_SYMBOLS
                    for symbol in required_symbols:
                        try:
                            getattr(lib, symbol)
                        except AttributeError as exc:
                            raise SystemExit(
                                f"ERROR: {wheel.name} {so_name} missing required symbol {symbol}"
                            ) from exc
        print(f"[sidecars] verified wheel {wheel.name}")


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
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
