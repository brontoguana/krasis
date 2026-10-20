"""Build-only verification of an explicitly admitted historical diagnostic bank."""
import ctypes
import hashlib
import json
import os
import pathlib
import sys

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Use ./dev conditional-projection-build; direct execution is forbidden")
if len(sys.argv) != 3:
    raise SystemExit("Expected frozen bank JSON and repository directory")
repo = pathlib.Path(sys.argv[2]).resolve()
bank = json.loads(pathlib.Path(sys.argv[1]).read_text())
if bank.get("passed") is not True:
    raise SystemExit("Unadmitted sidecar bank")
package = repo / "python/krasis"
contract = json.loads((package / "fla_sidecar_contract.json").read_text())
expected = {"libkrasis_marlin.so", "libkrasis_flash_attn.so", "libkrasis_fla.so"}
expected.update(f"libkrasis_fla_sm{arch}.so" for arch in contract["architectures"])
rows = bank["ids"]
if len(rows) != len(expected) or {pathlib.Path(r["path"]).name for r in rows} != expected:
    raise SystemExit("Exact complete distinct historical sidecar set required")
by_name = {}
for row in rows:
    path = pathlib.Path(row["path"])
    if not path.is_absolute() or path.resolve() != (package / path.name).resolve():
        raise SystemExit("Sidecar path is not inside this diagnostic package")
    raw = path.read_bytes()
    if not raw or len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise SystemExit(f"Historical sidecar identity changed: {path.name}")
    by_name[path.name] = row
manifest = json.loads((package / "sidecar_manifest.json").read_text())
abi = int((repo / "sidecar_abi_version.txt").read_text())
if manifest["schema_version"] != 1 or manifest["sidecar_abi"] != abi:
    raise SystemExit("Historical sidecar ABI manifest mismatch")
for item in manifest["sidecars"].values():
    if by_name[item["output"]]["sha256"] != item["sha256"]:
        raise SystemExit("Historical build-id/SHA manifest mismatch")
    lib = ctypes.CDLL(str(package / item["output"]))
    for symbol in item["symbols"]:
        getattr(lib, symbol)
    lib.krasis_sidecar_abi_version.restype = ctypes.c_int
    lib.krasis_sidecar_build_id.restype = ctypes.c_char_p
    if lib.krasis_sidecar_abi_version() != abi or lib.krasis_sidecar_build_id().decode() != item["build_id"]:
        raise SystemExit("Historical loaded ABI/build-id mismatch")
print(json.dumps({"passed": True, "sidecars": len(rows), "abi": abi,
                  "scope": "explicit retained diagnostic bank, unchanged binary hashes and loaded symbols/build IDs"}))
