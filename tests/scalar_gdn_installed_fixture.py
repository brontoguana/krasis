"""One-time installed-owner setup only; all native test execution is Rust."""
import gc
import json
import os
from pathlib import Path
import sys
import weakref

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Use ./dev scalar-gdn-fixture owner|graph <plan> [fresh-output]")
if len(sys.argv) not in (3, 4) or sys.argv[1] not in ("owner", "owner-error", "graph"):
    raise SystemExit("Use ./dev scalar-gdn-fixture owner|graph <plan> [fresh-output]")
mode, plan_file = sys.argv[1:3]
plan = json.loads(Path(plan_file).read_text())
# Exact physical UUID selection precedes Torch import or any CUDA initialization.
os.environ["CUDA_VISIBLE_DEVICES"] = plan["gpu_uuid"]
import krasis.krasis as native
if Path(native.__file__).resolve().parent != Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]).resolve() / "krasis":
    raise RuntimeError("fixture imported a different compiled runtime")
native.scalar_gdn_fixture_preflight(mode, plan_file)
if len(sys.argv) == 3:
    print(json.dumps({"passed": True, "gpu_initialized": False, "model_loaded": False}))
    raise SystemExit(0)

import torch
references = []
def factory(initial, count):
    source = torch.frombuffer(bytearray(initial), dtype=torch.float32)
    tensors = [source.to(device="cuda") for _ in range(count)]
    if len({tensor.data_ptr() for tensor in tensors}) != count:
        raise RuntimeError("fixture tensors alias; require independent real allocations")
    references.extend(weakref.ref(tensor) for tensor in tensors)
    try:
        return tensors
    finally:
        del tensors

try:
    result = json.loads(native.scalar_gdn_fixture_installed(mode, plan_file, sys.argv[3], factory))
except ValueError as error:
    if mode != "owner-error" or str(error) != "injected installed scalar fixture error after actual GPU enqueue":
        raise
    result = {"passed": True, "expected_injected_post_enqueue_error": str(error), "model_loaded": False}
else:
    if mode == "owner-error":
        raise RuntimeError("expected post-enqueue failure was not observed")
gc.collect()
if not references or any(reference() is not None for reference in references):
    raise RuntimeError("installed native fixture retained actual tensors after completion")
result["actual_tensor_references_released_after_completed_native_suite"] = True
result["torch_allocated_after_actual_owner_release"] = torch.cuda.memory_allocated()
result["torch_reserved_after_actual_owner_release"] = torch.cuda.memory_reserved()
if result["torch_allocated_after_actual_owner_release"] != 0:
    raise RuntimeError("CUDA storage remains allocated after all actual tensor owners released")
# End-of-fixture cleanup only, after every native/control and ownership check.
torch.cuda.empty_cache()
torch.cuda.synchronize()
result["torch_allocated_after_cold_cleanup"] = torch.cuda.memory_allocated()
result["torch_reserved_after_cold_cleanup"] = torch.cuda.memory_reserved()
if result["torch_allocated_after_cold_cleanup"] != 0 or result["torch_reserved_after_cold_cleanup"] != 0:
    raise RuntimeError("cold fixture allocator teardown incomplete")
if mode == "owner-error":
    Path(sys.argv[3], "ERROR_RECEIPT.json").write_text(json.dumps(result, indent=2) + "\n")
Path(sys.argv[3], "COMPLETION_RECEIPT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
