"""Setup-only Torch ownership validation through the linked native module.

No generation, model timing, per-token Python path or synthetic test prompts.
"""
import os
import sys

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Use ./dev model-owner-test <actual-bf16-source> <new-report> <physical-uuid>")

import hashlib
import json
from pathlib import Path
import traceback

import torch
import krasis.krasis as native


def main():
    if len(sys.argv) != 4:
        raise ValueError("Use ./dev model-owner-test <actual-bf16-source> <new-report> <physical-uuid>")
    source, output, uuid = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    if output.exists():
        raise ValueError("The owner report path must be fresh")
    data = source.read_bytes()
    if len(data) < 4096:
        raise ValueError("The admitted BF16 model operand must contain the complete 4096-byte test window")
    cpu = torch.frombuffer(bytearray(data[:4096]), dtype=torch.bfloat16).clone()
    device = torch.device("cuda", torch.cuda.current_device())
    index = device.index
    results = []
    report = {"passed": False, "physical_uuid": uuid, "controls": results,
              "source": {"path": str(source), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
              "scope": "Installed one-time setup owner admission/lifetime/type/window controls; no inference or CUDA numeric qualification"}
    try:
        if not hasattr(native, "model_tensor_owner_fixture"):
            raise ValueError("The linked installed selected-feature owner fixture is missing")
        factories = {
            "bf16": lambda: cpu.clone().to(device),
            "f32": lambda: cpu.float().to(device),
            "i32": lambda: torch.arange(cpu.numel(), dtype=torch.int32).to(device),
            "bytes": lambda: cpu.view(torch.uint8).clone().to(device),
        }
        sizes = {"bf16": 2, "f32": 4, "i32": 4, "bytes": 1}
        for name, factory in factories.items():
            result = json.loads(native.model_tensor_owner_fixture(factory, index, name, 32 * sizes[name]))
            if result["passed"] is not True or result["actual_device_uuid"] != uuid:
                raise ValueError("Native owner control did not admit the intended physical GPU")
            for field in ("real_owner_retained_after_original_reference_drop", "tensor_reference_released_after_owner_drop", "actual_cuda_context_rechecked"):
                if result[field] is not True:
                    raise ValueError(f"Native real-owner lifetime control failed: {field}")
            results.append({"case": name, "result": result})
        negatives = [
            ("not_tensor", lambda: object(), "bf16", 64, "actual Torch tensor"),
            ("cpu_owner", lambda: cpu.clone(), "bf16", 64, "not a CUDA tensor"),
            ("noncontiguous", lambda: cpu.clone().to(device).reshape(2, -1).transpose(0, 1), "bf16", 64, "not contiguous"),
            ("dtype_mismatch", factories["f32"], "bf16", 64, "differs from its native registration"),
            ("unsupported_dtype", lambda: cpu.double().to(device), "bf16", 64, "unsupported actual model tensor dtype"),
            ("empty_window", factories["bf16"], "bf16", 0, "window is empty"),
            ("unaligned_window", factories["bf16"], "bf16", 3, "window is empty, unaligned or exceeds"),
            ("oversized_window", factories["bf16"], "bf16", cpu.numel() * 2 + 2, "window is empty, unaligned or exceeds"),
        ]
        for name, factory, dtype, window, expected in negatives:
            try:
                native.model_tensor_owner_fixture(factory, index, dtype, window)
            except Exception as exc:
                if expected not in str(exc):
                    raise ValueError(f"Wrong native rejection for {name}: {exc}") from exc
                results.append({"case": name, "rejected": True, "reason": str(exc)})
            else:
                raise ValueError(f"Invalid actual owner/window was accepted: {name}")
        cold = json.loads(native.model_tensor_widen_fixture(factories["bf16"], index))
        expected_bits = [int.from_bytes(data[i:i + 2], "little") << 16 for i in range(0, 4096, 2)]
        if cold["passed"] is not True or cold["f32_bits"] != expected_bits:
            raise ValueError("Installed cold widening did not preserve every actual BF16 coefficient")
        for field in ("post_enqueue_error_completed", "post_enqueue_unwind_completed", "recovery_exact", "source_readonly_exact", "source_retained_through_error_unwind", "source_released_after_owner_drop", "real_gpu_buf_retained_context"):
            if cold[field] is not True:
                raise ValueError(f"Cold native ownership/completion control failed: {field}")
        results.append({"case": "actual_cold_widen_error_unwind_recovery", "result": cold})
        report["passed"] = True
        print(json.dumps({"passed": True, "controls": len(results), "physical_uuid": uuid}))
    except Exception:
        report["failure"] = traceback.format_exc()
        raise
    finally:
        with output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
