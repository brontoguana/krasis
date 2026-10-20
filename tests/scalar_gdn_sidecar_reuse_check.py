"""Cold full-package source/metadata reuse checks, never an inference path."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Use ./dev scalar-gdn-observer for this cold check")
if len(sys.argv) != 3 or sys.argv[2] != "--check":
    raise SystemExit("Usage: ./dev scalar-gdn-observer <script> <output.json> --check")
output = Path(sys.argv[1])
if not output.is_absolute() or output.exists():
    raise SystemExit("A fresh absolute output path is required")
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("checked_sidecar_builder", root / "scripts/build_sidecars.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
nvcc = module.find_nvcc()
contracts = module.sidecar_inputs(nvcc)
original = module.read_manifest()
if original is None:
    raise SystemExit("Real current manifest is required")
flash_contract = {"flash_attn": contracts["flash_attn"]}
if not module.manifest_matches(original, flash_contract):
    raise SystemExit("Real retained FlashAttention failed exact current source/compiler/bytes/ABI/build-ID/symbol verification")

# Every negative uses the actual file and verifier. Only the declared contract
# is changed; no model/library/source/environment is altered or substituted.
cases = []
for label, mutate in [
    ("wrong_source_contract", lambda m: m["sidecars"]["flash_attn"].update(input_hash="0" * 64)),
    ("wrong_artifact_hash", lambda m: m["sidecars"]["flash_attn"].update(sha256="0" * 64)),
    ("wrong_exported_build_id", lambda m: m["sidecars"]["flash_attn"].update(build_id="invalid-build-id")),
    ("wrong_package_abi", lambda m: m.update(sidecar_abi=-1)),
    ("missing_artifact_entry", lambda m: m["sidecars"].pop("flash_attn")),
]:
    declared = copy.deepcopy(original)
    mutate(declared)
    if module.manifest_matches(declared, flash_contract):
        raise AssertionError(f"Invalid declared contract accepted: {label}")
    module.read_manifest = lambda declared=declared: declared
    module.sidecar_inputs = lambda _nvcc: contracts
    def prohibited_build(*args, **kwargs):
        raise AssertionError("Invalid FlashAttention must fail before either compiler runs")
    module.build_marlin = prohibited_build
    module.build_flash_attn = prohibited_build
    try:
        import argparse
        module.build(argparse.Namespace(force=False, marlin_only=True))
    except SystemExit as error:
        if "--marlin-only requires" not in str(error):
            raise
    else:
        raise AssertionError("Invalid retained artifact did not cause explicit failure")
    cases.append({"case": label, "rejected_before_build": True})
output.write_text(json.dumps({
    "passed": True,
    "real_flash_attention_source_compiler_bytes_abi_build_id_symbols_verified": True,
    "negative_cases": cases,
    "real_flash_attention_sha256": module.sha256_file(module.PACKAGE_DIR / module.FLASH_ATTN_SO),
    "scope": "Cold read-only actual artifact/metadata contract. No CUDA context, arithmetic, model, serving or performance claim; neither compiler ran.",
}, indent=2) + "\n")
print(f"PASS: real retained artifact and {len(cases)} invalid contracts checked")
