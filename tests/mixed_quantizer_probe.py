"""Read-only real-model quantizer equivalence probe through ./dev."""
import os
if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Run through KRASIS_DEV_SCRIPT=1 ./dev python")
import argparse
import json
import time
from pathlib import Path
import torch
from scripts import mixed_cache_rank as rank

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("calibration", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--group-size", type=int, required=True)
parser.add_argument("--cuda-device", type=int, default=0)
parser.add_argument("--serial-reference", type=Path)
args = parser.parse_args()
if args.output.exists():
    raise RuntimeError("refusing to replace probe evidence")
manifest, raw, offset = rank.load_kic1(args.calibration)
config = json.loads((args.model / "config.json").read_bytes())
config = config.get("text_config", config)
index = json.loads((args.model / "model.safetensors.index.json").read_bytes())["weight_map"]
layer = int(manifest["layers"][0]["model_layer"])
hidden, intermediate, experts = config["hidden_size"], config["moe_intermediate_size"], config["n_routed_experts"]
inputs, down = rank.kic1_layer_moments(manifest, raw, offset, layer, experts, hidden, intermediate)
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device(f"cuda:{args.cuda_device}")
serial = json.loads(args.serial_reference.read_bytes()) if args.serial_reference else None
if serial is not None and serial["calibration_sha256"] != rank.stable_sha256_file(args.calibration)[0]:
    raise RuntimeError("serial probe calibration identity mismatch")
serial_hashes = {item["projection"]: item["rust_projection_sha256"] for item in serial["results"]} if serial else {}
results = []
for name in ("gate_proj", "up_proj", "down_proj"):
    tensor = rank.unique_tensor_name(index, f".layers.{layer}.mlp.experts.0.{name}.weight")
    source, storage = rank.source_projection(args.model, index, tensor)
    moments = down[0] if name == "down_proj" else inputs[0]
    start = time.monotonic()
    exact, audit = rank.materialized_search_rmse(source, moments, args.group_size, device)
    exact_s = time.monotonic() - start
    start = time.monotonic()
    approximate, parallel_audit = rank.quantize_search_rmse(source, moments, args.group_size, device)
    approximate_s = time.monotonic() - start
    # Exercise deterministic CUDA matmul before the full population scan.
    # No synthetic inputs enter calibration or any quality verdict.
    fixture = torch.zeros((1, source.shape[1]), device=device, dtype=torch.bfloat16)
    product = torch.nn.functional.linear(fixture, exact.to(device))
    if not torch.isfinite(product).all():
        raise RuntimeError("nonfinite deterministic CUDA transport fixture")
    exact_bits = exact.view(torch.uint16)
    parallel_bits = approximate.view(torch.uint16)
    different = exact_bits != parallel_bits
    signed_zero = ((exact_bits == 0) & (parallel_bits == 32768)) | ((exact_bits == 32768) & (parallel_bits == 0))
    exact8 = rank.materialized_int8(source, args.group_size)
    proxy8 = rank.quantize_int8(source, args.group_size, device)
    exact_hash = rank.projection_sha256(exact)
    if serial is not None and exact_hash != serial_hashes[name]:
        raise RuntimeError(f"parallel Rust projection differs from serial reference: {name}")
    result = {"serial_rust_hash_equal": exact_hash == serial_hashes.get(name) if serial else None,
              "int8_bitwise_equal": rank.projection_sha256(exact8) == rank.projection_sha256(proxy8),
              "bit_mismatch_count": int(different.sum()),
              "signed_zero_mismatch_count": int((different & signed_zero).sum()),
              "nonzero_bit_mismatch_count": int((different & ~signed_zero).sum()),
              "model_layer": layer, "expert": 0, "projection": name, "storage": storage,
              "rust_materializer_seconds": exact_s, "parallel_proxy_seconds": approximate_s,
              "factor_mismatches": sum(a[1] != b[1] for a, b in zip(audit, parallel_audit)),
              "dequantized_bf16_equal": bool(torch.equal(exact, approximate)),
              "rust_projection_sha256": rank.projection_sha256(exact),
              "parallel_projection_sha256": rank.projection_sha256(approximate),
              "baseline_weighted_rmse_max": max(a[2] for a in audit),
              "optimized_weighted_rmse_max": max(a[3] for a in audit)}
    results.append(result)
    print(json.dumps(result, sort_keys=True), flush=True)
rank.atomic_write(args.output, rank.canonical_bytes({"calibration_sha256": rank.stable_sha256_file(args.calibration)[0], "results": results}))
