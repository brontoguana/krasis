"""Cold real tensor setup; complete phase/thread/error/graph control is Rust."""
import gc
import json
import os
from pathlib import Path
import sys
import weakref

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Use ./dev scalar-gdn-observer <script> <plan> <fresh-output|--check>")
if len(sys.argv) != 3:
    raise SystemExit("Phase fixture needs an absolute plan and output path or --check")
plan_file = Path(sys.argv[1])
if not plan_file.is_absolute():
    raise SystemExit("Absolute frozen phase plan required")
plan = json.loads(plan_file.read_text())
os.environ["CUDA_VISIBLE_DEVICES"] = plan["gpu_uuid"]
import krasis.krasis as native
expected_package = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]).resolve() / "krasis"
if Path(native.__file__).resolve().parent != expected_package:
    raise RuntimeError("Phase fixture imported a different compiled runtime")
admission = json.loads(native.scalar_gdn_phase_preflight(str(plan_file)))
if sys.argv[2] == "--check":
    print(json.dumps(admission))
    raise SystemExit(0)
output = Path(sys.argv[2])
if not output.is_absolute() or output.exists():
    raise RuntimeError("Fresh absolute output directory required before Torch initialization")
import torch
references = []
def factory(initial, conv_shape, count):
    source = torch.frombuffer(bytearray(initial), dtype=torch.float32)
    tensors = [(source.to(device="cuda"), torch.zeros(conv_shape, device="cuda", dtype=torch.float32)) for _ in range(count)]
    if len({tensor.data_ptr() for pair in tensors for tensor in pair}) != count * 2:
        raise RuntimeError("Actual latest/convolution state allocations alias")
    references.extend(weakref.ref(tensor) for pair in tensors for tensor in pair)
    try:
        return tensors
    finally:
        del tensors
result = json.loads(native.scalar_gdn_phase_installed(str(plan_file), str(output), factory))
gc.collect()
if not references or any(reference() is not None for reference in references):
    raise RuntimeError("Actual phase tensor objects remain after completed native owner release")
result["actual_typed_tensor_objects_released"] = True
result["torch_allocated_before_cold_teardown"] = torch.cuda.memory_allocated()
result["torch_reserved_before_cold_teardown"] = torch.cuda.memory_reserved()
if result["torch_allocated_before_cold_teardown"] != 0:
    raise RuntimeError("Actual tensor storage remains allocated after owner release")
# Explicit fixture teardown follows all native assertions and ownership checks.
torch.cuda.empty_cache()
torch.cuda.synchronize()
result["torch_allocated_after_cold_teardown"] = torch.cuda.memory_allocated()
result["torch_reserved_after_cold_teardown"] = torch.cuda.memory_reserved()
if result["torch_allocated_after_cold_teardown"] != 0 or result["torch_reserved_after_cold_teardown"] != 0:
    raise RuntimeError("Phase fixture allocator teardown incomplete")
(output / "COMPLETION_RECEIPT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
