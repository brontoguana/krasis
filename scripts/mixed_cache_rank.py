#!/usr/bin/env python3
"""Measure routed-expert sensitivity and preregister compact mixed-cache plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import time
from functools import cmp_to_key
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from krasis.checkpoint_identity import cache_namespace, checkpoint_identity
try:
    from tileq_build import (
        _mixed_marlin_component_bytes,
        capture_binding_sha256,
        capture_inventory,
        capture_metadata,
        combine_capture_layers,
        combined_sha256,
        mixed_calibration_entry_hashes,
        read_capture_layer,
        stable_sha256_file,
    )
except ModuleNotFoundError:
    from scripts.tileq_build import (
        _mixed_marlin_component_bytes,
        capture_binding_sha256,
        capture_inventory,
        capture_metadata,
        combine_capture_layers,
        combined_sha256,
        mixed_calibration_entry_hashes,
        read_capture_layer,
        stable_sha256_file,
    )


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Run this calibration command through ./dev mixed-cache-rank")


FORMAT = "Krasis mixed routed-expert ranking evidence"
VERSION = 4
PLAN_FORMAT = "Krasis compact mixed Marlin build plan"
PLAN_VERSION = 1
MANIFEST_FORMAT = "Krasis routed mixed Marlin INT4/INT8"
MANIFEST_VERSION = 2
GAIN_EXPONENT = -12
FACTORS = (1.00, 0.98, 0.95, 0.92, 0.90, 0.87, 0.85, 0.82, 0.80, 0.75)
AUDIT_MAGIC = b"KSGA"
AUDIT_VERSION = 1
AUDIT_HEADER = struct.Struct("<4s7I")
AUDIT_ROW = struct.Struct("<IHBBHBBff")
KIC_HEADER = struct.Struct("<4sIQQQ32s")
KIC_MAGIC = b"KIC1"
KIC_VERSION = 1
ORDER_METHODS = {
    "expected": "full_expert_bf16_residual_expected_exposure_v3",
    "tail": "full_expert_bf16_residual_tail_v2",
    "blended": "full_expert_bf16_residual_blended_exposure_v3",
}
BUDGETS = (0, 500, 1000, 1500, 2000, 10_000)
SENSITIVITY_LAYER_FORMAT = "Krasis mixed sensitivity layer shard"
SENSITIVITY_LAYER_VERSION = 4


def coverage_required(counts: torch.Tensor, corpus_counts: list[torch.Tensor], minimum: int) -> torch.Tensor:
    if minimum < 1 or not corpus_counts or any(item.shape != counts.shape for item in corpus_counts):
        raise RuntimeError("invalid coverage policy geometry")
    if any(torch.any(item < 0) for item in corpus_counts) or not torch.equal(sum(corpus_counts), counts):
        raise RuntimeError("aggregate/per-corpus coverage counts differ")
    return (counts < minimum) | torch.stack(corpus_counts).eq(0).any(dim=0)


def select_with_reserve(candidates: list[dict], rank_key: str, score_key: str, cap: int, full: bool) -> tuple[set, int]:
    required = [item for item in candidates if item.get("coverage_required_int8")]
    selected = {(item["moe_layer"], item["expert"]) for item in required}
    used = sum(item["int8_bytes"] - item["int4_bytes"] for item in required)
    if used > cap:
        raise RuntimeError("coverage-required INT8 reserve exceeds requested byte budget")
    measured = sorted((item for item in candidates if not item.get("coverage_required_int8")), key=lambda item: item[rank_key])
    for item in measured:
        delta = item["int8_bytes"] - item["int4_bytes"]
        if (full or item[score_key] > 0) and used + delta <= cap:
            selected.add((item["moe_layer"], item["expert"]))
            used += delta
    return selected, used


def expected_exposure_gain(error4: float, error8: float, router_mass: float, rows: int) -> float:
    """Expected router-weighted relative squared-error reduction per input row."""
    if rows <= 0 or any(not math.isfinite(value) or value < 0
                        for value in (error4, error8, router_mass)):
        raise RuntimeError("expected-error exposure inputs must be finite and nonnegative")
    # Normalized per-expert RMSE alone discards empirical expert exposure.
    # Squared RMSE supplies relative squared error; measured squared router
    # mass supplies its actual contribution in the captured population.
    gain = max(0.0, error4 * error4 - error8 * error8) * (router_mass / rows)
    if not math.isfinite(gain):
        raise RuntimeError("expected-error exposure gain is nonfinite")
    return gain


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def proc_memory_snapshot() -> dict[str, int]:
    result: dict[str, int] = {}
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _unit = line.split()
                result[f"process_{key.removesuffix(':').lower()}_bytes"] = int(value) * 1024
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        wanted = {"MemAvailable", "Cached", "Buffers", "SReclaimable"}
        for line in meminfo_path.read_text().splitlines():
            key, rest = line.split(":", 1)
            if key in wanted:
                result[f"system_{key.lower()}_bytes"] = int(rest.split()[0]) * 1024
    return result


def directory_regular_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_json(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} root must be an object")
    return value


def sensitivity_layer_identity(common: dict, model_layer: int, moe_layer: int) -> dict:
    """Return the complete immutable identity for one resumable layer shard."""
    return {
        "common": common,
        "model_layer": model_layer,
        "moe_layer": moe_layer,
    }


def validate_sensitivity_layer_shard(
    value: dict,
    audit_path: Path,
    expected_identity: dict,
    expected_experts: int,
) -> None:
    """Fail closed before reusing any previously committed sensitivity work."""
    if (
        value.get("format") != SENSITIVITY_LAYER_FORMAT
        or value.get("schema_version") != SENSITIVITY_LAYER_VERSION
        or value.get("identity") != expected_identity
        or value.get("identity_sha256")
        != hashlib.sha256(canonical_bytes(expected_identity)).hexdigest()
    ):
        raise RuntimeError("sensitivity layer immutable identity mismatch")
    if value.get("audit_basename") != audit_path.name:
        raise RuntimeError("sensitivity layer audit basename mismatch")
    audit_raw = audit_path.read_bytes()
    if value.get("audit_sha256") != hashlib.sha256(audit_raw).hexdigest():
        raise RuntimeError("sensitivity layer audit identity mismatch")
    if not isinstance(value.get("memory"), dict):
        raise RuntimeError("sensitivity layer lacks memory evidence")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != expected_experts:
        raise RuntimeError("sensitivity layer candidate coverage mismatch")
    model_layer = expected_identity["model_layer"]
    moe_layer = expected_identity["moe_layer"]
    expected_ids = set(range(expected_experts))
    observed_ids = set()
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or candidate.get("model_layer") != model_layer
            or candidate.get("moe_layer") != moe_layer
            or not isinstance(candidate.get("expert"), int)
        ):
            raise RuntimeError("sensitivity layer candidate identity mismatch")
        observed_ids.add(candidate["expert"])
    if observed_ids != expected_ids:
        raise RuntimeError("sensitivity layer candidate identities are incomplete or duplicated")
    source_precisions = value.get("source_precisions")
    if not isinstance(source_precisions, list) or (not source_precisions and not all(item.get("coverage_required_int8") is True for item in candidates)):
        raise RuntimeError("sensitivity layer source provenance is missing")


def load_kic1(path: Path) -> tuple[dict, np.memmap, int]:
    """Verify KIC1 and retain its payload as a read-only memory map."""
    raw = np.memmap(path, mode="r", dtype=np.uint8)
    if raw.size < KIC_HEADER.size:
        raise RuntimeError(f"KIC1 artifact {path} is truncated")
    magic, version, manifest_len, payload_offset, payload_len, expected_sha = KIC_HEADER.unpack_from(raw, 0)
    manifest_end = KIC_HEADER.size + manifest_len
    payload_end = payload_offset + payload_len
    if magic != KIC_MAGIC or version != KIC_VERSION:
        raise RuntimeError(f"KIC1 artifact {path} has incompatible magic/version")
    if manifest_end > raw.size or payload_offset < manifest_end or payload_end != raw.size:
        raise RuntimeError(f"KIC1 artifact {path} has invalid ranges")
    manifest_bytes = bytes(raw[KIC_HEADER.size:manifest_end])
    if hashlib.sha256(manifest_bytes).digest() != expected_sha:
        raise RuntimeError(f"KIC1 artifact {path} manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != KIC_VERSION
        or manifest.get("format") != "Krasis expert INT4 activation moments"
        or manifest.get("objective")
        != "router_weight_squared_diagonal_activation_second_moment_v1"
        or manifest.get("payload_bytes") != payload_len
    ):
        raise RuntimeError(f"KIC1 artifact {path} has an invalid manifest contract")
    payload_sha = hashlib.sha256(raw[payload_offset:payload_end]).hexdigest()
    if payload_sha != manifest.get("payload_sha256"):
        raise RuntimeError(f"KIC1 artifact {path} payload SHA-256 mismatch")
    ranges = []
    for layer in manifest.get("layers", []):
        for key in ("input_second_moment_f32", "down_second_moment_f32"):
            descriptor = layer.get(key, {})
            ranges.append((int(descriptor.get("offset", -1)), int(descriptor.get("len", -1))))
    ranges.sort()
    cursor = 0
    for offset, length in ranges:
        if offset != cursor or length <= 0 or length % 4:
            raise RuntimeError(f"KIC1 payload is not exactly covered at byte {cursor}")
        cursor += length
    if cursor != payload_len:
        raise RuntimeError(f"KIC1 payload coverage ends at {cursor}, expected {payload_len}")
    return manifest, raw, int(payload_offset)


def kic1_layer_moments(
    manifest: dict,
    raw: np.memmap,
    payload_offset: int,
    model_layer: int,
    experts: int,
    hidden: int,
    intermediate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    matches = [item for item in manifest["layers"] if int(item["model_layer"]) == model_layer]
    if len(matches) != 1:
        raise RuntimeError(f"KIC1 has {len(matches)} records for model layer {model_layer}")
    layer = matches[0]
    tensors = []
    for key, shape in (
        ("input_second_moment_f32", (experts, hidden)),
        ("down_second_moment_f32", (experts, intermediate)),
    ):
        descriptor = layer[key]
        offset, length = int(descriptor["offset"]), int(descriptor["len"])
        expected = math.prod(shape) * 4
        if length != expected:
            raise RuntimeError(
                f"KIC1 {key} length mismatch at model layer {model_layer}: {length} != {expected}"
            )
        values = np.frombuffer(
            raw,
            dtype="<f4",
            count=math.prod(shape),
            offset=payload_offset + offset,
        ).reshape(shape)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise RuntimeError(f"KIC1 {key} has invalid moments at model layer {model_layer}")
        tensors.append(torch.from_numpy(values.copy()))
    return tensors[0], tensors[1]


def unique_tensor_name(weight_map: dict, suffix: str) -> str:
    matches = [name for name in weight_map if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one tensor ending {suffix!r}, found {matches}")
    return matches[0]


def source_projection(model_dir: Path, weight_map: dict, name: str) -> tuple[torch.Tensor, str]:
    shard = weight_map.get(name)
    if not isinstance(shard, str):
        raise RuntimeError(f"source tensor is absent from index: {name}")
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(name)
    if weight.ndim != 2:
        raise RuntimeError(f"source tensor {name} must be rank two, got {tuple(weight.shape)}")
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(torch.bfloat16).float(), f"{str(weight.dtype).removeprefix('torch.')}->bf16"
    scale_name = f"{name}_scale_inv"
    scale_shard = weight_map.get(scale_name)
    if not isinstance(scale_shard, str):
        raise RuntimeError(f"FP8 source tensor has no scale identity: {scale_name}")
    with safe_open(model_dir / scale_shard, framework="pt", device="cpu") as handle:
        scales = handle.get_tensor(scale_name).float()
    rows, cols = weight.shape
    if scales.numel() == 1:
        result = weight.float().mul_(float(scales.reshape(-1)[0]))
    elif scales.ndim == 2 and rows % scales.shape[0] == 0 and cols % scales.shape[1] == 0:
        row_block = rows // scales.shape[0]
        col_block = cols // scales.shape[1]
        expanded = scales.repeat_interleave(row_block, 0).repeat_interleave(col_block, 1)
        result = weight.float().mul_(expanded)
    else:
        raise RuntimeError(
            f"FP8 source scale {scale_name} shape {tuple(scales.shape)} does not tile {tuple(weight.shape)}"
        )
    # Match the Rust cache builder's bounded BF16 working representation.
    return result.to(torch.bfloat16).float(), f"{str(weight.dtype).removeprefix('torch.')}+scale_inv->bf16"


def rust_round(values: torch.Tensor) -> torch.Tensor:
    """Match f32::round (halfway away from zero), including adjacent floats."""
    truncated = values.trunc()
    correction = values.sub(truncated).abs().ge(0.5).to(values.dtype)
    return truncated.add(values.sign().mul(correction))


def rust_dequantize(
    grouped: torch.Tensor, scale: torch.Tensor, lower: int, upper: int
) -> torch.Tensor:
    # Rust multiplies by a FP32 reciprocal, rather than dividing each weight.
    # A zero persisted BF16 scale has zero reciprocal in the cache builder.
    inverse = torch.where(scale == 0, torch.zeros_like(scale), scale.reciprocal())
    quantized = rust_round(grouped.mul(inverse[..., None])).clamp_(lower, upper)
    # Rust stores integer zero and reconstructs +0; floating -0 has the same
    # numeric value but a different BF16 payload identity.
    quantized = torch.where(quantized == 0, torch.zeros_like(quantized), quantized)
    return quantized.mul_(scale[..., None])


def quantize_search_rmse(
    source_cpu: torch.Tensor,
    moments_cpu: torch.Tensor,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[int, int, float, float, bool]]]:
    rows, cols = source_cpu.shape
    if cols % group_size:
        raise RuntimeError(f"projection width {cols} is not divisible by group size {group_size}")
    groups = cols // group_size
    source = source_cpu.to(device=device, dtype=torch.float32)
    grouped = source.reshape(rows, groups, group_size)
    moments = moments_cpu.to(device=device, dtype=torch.float32).reshape(groups, group_size)
    base = grouped.abs().amax(dim=2).div_(7.0)
    base = torch.where(base == 0, torch.ones_like(base), base)
    denominator = grouped.square().mul_(moments[None]).sum(dim=(0, 2)).clamp_min_(1e-12)
    scores = []
    for factor in FACTORS:
        scale = base.mul(factor).clamp_min_(torch.finfo(torch.float32).eps)
        dequant = rust_dequantize(grouped, scale, -8, 7)
        scores.append(dequant.sub_(grouped).square_().mul_(moments[None]).sum(dim=(0, 2)))
    score_matrix = torch.stack(scores)
    best_index = score_matrix.argmin(dim=0)
    factor_tensor = torch.tensor(FACTORS, device=device).index_select(0, best_index)

    # Rust selects with FP32 scales, then persists BF16 scales. Audit the
    # actual persisted baseline and optimized representations.
    baseline_scale = base.to(torch.bfloat16).float()
    optimized_scale = base.mul(factor_tensor[None]).clamp_min_(torch.finfo(torch.float32).eps).to(torch.bfloat16).float()
    baseline = rust_dequantize(grouped, baseline_scale, -8, 7)
    optimized = rust_dequantize(grouped, optimized_scale, -8, 7)
    baseline_error = (
        baseline.sub(grouped).square().mul(moments[None]).sum(dim=(0, 2)).div(denominator).sqrt()
    )
    optimized_error = (
        optimized.sub(grouped).square().mul(moments[None]).sum(dim=(0, 2)).div(denominator).sqrt()
    )
    audit = [
        (
            group,
            int(best_index[group].item()),
            float(baseline_error[group].item()),
            float(optimized_error[group].item()),
            int(best_index[group].item()) == len(FACTORS) - 1,
        )
        for group in range(groups)
    ]
    return optimized.reshape(rows, cols).to(torch.bfloat16).cpu(), audit


def quantize_int8(source_cpu: torch.Tensor, group_size: int, device: torch.device) -> torch.Tensor:
    rows, cols = source_cpu.shape
    if cols % group_size:
        raise RuntimeError(f"projection width {cols} is not divisible by group size {group_size}")
    source = source_cpu.to(device=device, dtype=torch.float32)
    grouped = source.reshape(rows, cols // group_size, group_size)
    scale = grouped.abs().amax(dim=2).div_(127.0)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale).to(torch.bfloat16).float()
    dequant = rust_dequantize(grouped, scale, -128, 127)
    return dequant.reshape(rows, cols).to(torch.bfloat16).cpu()


def materialized_search_rmse(source: torch.Tensor, moments: torch.Tensor, group_size: int, device: torch.device):
    from krasis import WeightStore
    rows, cols = source.shape
    source_bytes = source.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    moment_bytes = moments.contiguous().numpy().astype("<f4", copy=False).tobytes()
    payload, factors = WeightStore.calibrated_projection(source_bytes, moment_bytes, rows, cols, group_size)
    optimized = torch.from_numpy(np.frombuffer(payload, dtype="<u2").copy()).view(torch.bfloat16).reshape(rows, cols)
    groups = cols // group_size
    if len(factors) != groups or any(item >= len(FACTORS) for item in factors):
        raise RuntimeError("Rust scale audit geometry mismatch")
    grouped = source.to(device=device, dtype=torch.float32).reshape(rows, groups, group_size)
    moment_device = moments.to(device).reshape(groups, group_size)
    base = grouped.abs().amax(dim=2).div_(7.0)
    base = torch.where(base == 0, torch.ones_like(base), base).to(torch.bfloat16).float()
    baseline = rust_dequantize(grouped, base, -8, 7).to(torch.bfloat16).float()
    actual = optimized.to(device=device, dtype=torch.float32).reshape(rows, groups, group_size)
    denominator = grouped.square().mul(moment_device[None]).sum(dim=(0, 2)).clamp_min_(1e-12)
    baseline_error = baseline.sub(grouped).square().mul(moment_device[None]).sum(dim=(0, 2)).div(denominator).sqrt().cpu()
    optimized_error = actual.sub(grouped).square().mul(moment_device[None]).sum(dim=(0, 2)).div(denominator).sqrt().cpu()
    audit = [(g, factors[g], float(baseline_error[g]), float(optimized_error[g]), factors[g] == len(FACTORS) - 1) for g in range(groups)]
    return optimized, audit


def projection_sha256(value: torch.Tensor) -> str:
    payload = value.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def materialized_int8(source: torch.Tensor, group_size: int) -> torch.Tensor:
    from krasis import WeightStore
    rows, cols = source.shape
    source_bytes = source.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    payload, factors = WeightStore.calibrated_projection(source_bytes, b"", rows, cols, group_size, 8)
    if factors:
        raise RuntimeError("INT8 materializer unexpectedly returned activation scale factors")
    return torch.from_numpy(np.frombuffer(payload, dtype="<u2").copy()).view(torch.bfloat16).reshape(rows, cols)


def observed_statistics(values: torch.Tensor) -> dict[str, float]:
    ordered = values.detach().to(device="cpu", dtype=torch.float64).sort().values
    if ordered.numel() == 0:
        raise RuntimeError("cannot score an expert without real routed observations")
    p99_index = max(0, math.ceil(0.99 * ordered.numel()) - 1)
    p99 = ordered[p99_index]
    tail = ordered[p99_index:]
    return {
        "maximum": float(ordered[-1].item()),
        "p99": float(p99.item()),
        "cvar99": float(tail.mean().item()),
    }


def treatment_outputs(
    x: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    swiglu_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_w, up_w, down_w = weights
    gate = torch.matmul(x, gate_w.T)
    up = torch.matmul(x, up_w.T)
    gate = gate.clamp(max=swiglu_limit)
    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    activated = F.silu(gate).mul_(up).to(torch.bfloat16)
    output = torch.matmul(activated, down_w.T)
    return gate, up, output


def full_expert_metrics(
    capture,
    expert: int,
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    int4: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    int8: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    swiglu_limit: float,
    device: torch.device,
) -> tuple[dict, int]:
    positions = torch.nonzero(capture.topk_ids.reshape(-1) == expert).flatten()
    if positions.numel() == 0:
        raise RuntimeError(f"expert {expert} has no real routed observations")
    token_rows = torch.div(positions, capture.topk, rounding_mode="floor")
    inputs = capture.expert_inputs.index_select(0, token_rows).to(torch.bfloat16)
    route_weights = capture.topk_weights.reshape(-1).index_select(0, positions).float()
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    per_row_bytes = max(
        1,
        4
        * (
            6 * capture.intermediate_size
            + 3 * capture.expert_input_size
        ),
    )
    # Geometry-derived temporary capacity bounded by currently measured free
    # device memory; no model/GPU-specific row constant is used.
    batch_rows = max(1, min(inputs.shape[0], free_bytes // (2 * per_row_bytes)))
    source_device = tuple(value.to(device=device, dtype=torch.bfloat16) for value in source)
    int4_device = tuple(value.to(device=device, dtype=torch.bfloat16) for value in int4)
    int8_device = tuple(value.to(device=device, dtype=torch.bfloat16) for value in int8)
    sample4, sample8 = [], []
    weighted_num4 = weighted_num8 = weighted_den = 0.0
    projection_num4 = [0.0, 0.0]
    projection_num8 = [0.0, 0.0]
    projection_den = [0.0, 0.0]
    for start in range(0, inputs.shape[0], batch_rows):
        end = min(inputs.shape[0], start + batch_rows)
        x = inputs[start:end].to(device)
        weights = route_weights[start:end].to(device).square()
        ref_gate, ref_up, reference = treatment_outputs(x, source_device, swiglu_limit)
        q4_gate, q4_up, output4 = treatment_outputs(x, int4_device, swiglu_limit)
        q8_gate, q8_up, output8 = treatment_outputs(x, int8_device, swiglu_limit)
        denominator = reference.float().square().sum(dim=1).clamp_min_(1e-12)
        numerator4 = output4.float().sub(reference.float()).square().sum(dim=1)
        numerator8 = output8.float().sub(reference.float()).square().sum(dim=1)
        sample4.append(numerator4.div(denominator).sqrt().cpu())
        sample8.append(numerator8.div(denominator).sqrt().cpu())
        weighted_num4 += float(numerator4.mul(weights).sum().item())
        weighted_num8 += float(numerator8.mul(weights).sum().item())
        weighted_den += float(denominator.mul(weights).sum().item())
        for slot, (ref_value, q4_value, q8_value) in enumerate(
            (
                (torch.cat((ref_gate, ref_up), 1), torch.cat((q4_gate, q4_up), 1), torch.cat((q8_gate, q8_up), 1)),
                (reference, output4, output8),
            )
        ):
            projection_den[slot] += float(ref_value.float().square().sum().item())
            projection_num4[slot] += float(q4_value.float().sub(ref_value.float()).square().sum().item())
            projection_num8[slot] += float(q8_value.float().sub(ref_value.float()).square().sum().item())
    errors4 = torch.cat(sample4)
    errors8 = torch.cat(sample8)
    stats4 = observed_statistics(errors4)
    stats8 = observed_statistics(errors8)
    stats4["router_weighted_relative_rmse"] = math.sqrt(weighted_num4 / max(weighted_den, 1e-30))
    stats8["router_weighted_relative_rmse"] = math.sqrt(weighted_num8 / max(weighted_den, 1e-30))
    stats4["w13_relative_rmse"] = math.sqrt(projection_num4[0] / max(projection_den[0], 1e-30))
    stats8["w13_relative_rmse"] = math.sqrt(projection_num8[0] / max(projection_den[0], 1e-30))
    stats4["w2_relative_rmse"] = math.sqrt(projection_num4[1] / max(projection_den[1], 1e-30))
    stats8["w2_relative_rmse"] = math.sqrt(projection_num8[1] / max(projection_den[1], 1e-30))
    del source_device, int4_device, int8_device
    torch.cuda.empty_cache()
    return {"search_rmse_int4": stats4, "int8": stats8}, int(batch_rows)


def compare_score(left: dict, right: dict, key: str) -> int:
    left_cross = left[key] * (right["int8_bytes"] - right["int4_bytes"])
    right_cross = right[key] * (left["int8_bytes"] - left["int4_bytes"])
    if left_cross != right_cross:
        return -1 if left_cross > right_cross else 1
    left_identity = (left["moe_layer"], left["expert"], left["model_layer"])
    right_identity = (right["moe_layer"], right["expert"], right["model_layer"])
    return (left_identity > right_identity) - (left_identity < right_identity)


def assign_quality_ranks(all_candidates: list[dict]) -> None:
    measured_candidates = [item for item in all_candidates if not item.get("coverage_required_int8")]
    for key in ("expected_gain_units", "tail_gain_units"):
        ranked = sorted(measured_candidates, key=cmp_to_key(lambda left, right, key=key: compare_score(left, right, key)))
        for rank, item in enumerate(ranked):
            item[f"{key.removesuffix('_gain_units')}_rank"] = rank
    max_expected = max((item["expected_gain_units"] for item in measured_candidates), default=0) or 1
    max_tail = max((item["tail_gain_units"] for item in measured_candidates), default=0) or 1
    for item in measured_candidates:
        item["blended_gain_units"] = round(
            10**12 * (item["expected_gain_units"] / max_expected + item["tail_gain_units"] / max_tail) / 2
        )
    ranked_blended = sorted(
        measured_candidates,
        key=cmp_to_key(lambda left, right: compare_score(left, right, "blended_gain_units")),
    )
    for rank, item in enumerate(ranked_blended):
        item["blended_rank"] = rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--extra", action="append", default=[], type=Path)
    parser.add_argument("--corpus-manifest", action="append", required=True, type=Path)
    parser.add_argument("--excluded-input-manifest", required=True, type=Path)
    parser.add_argument("--calibration-artifact", required=True, type=Path)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-tail-observations", type=int, default=5)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--coverage-policy", choices=("strict", "underobserved_int8_v1"), default="strict")
    parser.add_argument("--cublas-workspace-config", choices=(":4096:8", ":16:8"),
                        default=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"))
    args = parser.parse_args()
    if args.cublas_workspace_config not in (":4096:8", ":16:8"):
        raise RuntimeError("unsupported deterministic cuBLAS workspace configuration")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = args.cublas_workspace_config
    if not 0 < args.tail_fraction <= 0.1 or args.minimum_tail_observations < 1:
        raise RuntimeError("tail coverage parameters are invalid")
    minimum_routes = math.ceil(args.minimum_tail_observations / args.tail_fraction)
    reserve_policy = args.coverage_policy == "underobserved_int8_v1"
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    capture_dirs = [args.capture_dir.resolve(), *[path.resolve() for path in args.extra]]
    corpus_manifests = [path.resolve() for path in args.corpus_manifest]
    if len(capture_dirs) != len(corpus_manifests) or len(set(capture_dirs)) != len(capture_dirs):
        raise RuntimeError("provide one distinct corpus manifest for every distinct capture directory")
    phase_start = time.monotonic()
    print("[mixed-rank] verifying complete capture inventories", flush=True)
    captures = [capture_inventory(path) for path in capture_dirs]
    print(f"[mixed-rank] inventory_verified seconds={time.monotonic() - phase_start:.3f}", flush=True)
    capture_metas = [capture_metadata(path) for path in capture_dirs]
    layer_names = sorted(captures[0][0])
    if any(sorted(inventory) != layer_names for inventory, _ in captures[1:]):
        raise RuntimeError("capture layer inventories differ")
    phase_start = time.monotonic()
    capture_hashes = [combined_sha256(files) for _, files in captures]
    print(f"[mixed-rank] capture_hashes_verified seconds={time.monotonic() - phase_start:.3f}", flush=True)
    corpus_hashes = [meta["calibration_sha256"].lower() for meta in capture_metas]
    if len(set(corpus_hashes)) != len(corpus_hashes):
        raise RuntimeError("calibration corpora must have distinct identities")
    corpus_manifest_hashes = []
    calibration_input_hashes: set[str] = set()
    for expected, manifest_path in zip(corpus_hashes, corpus_manifests):
        raw = manifest_path.read_bytes()
        corpus_manifest_hashes.append(hashlib.sha256(raw).hexdigest())
        value = json.loads(raw)
        declared = value.pop("corpus_sha256", None)
        actual = hashlib.sha256(canonical_bytes(value)).hexdigest()
        if declared != actual or actual != expected:
            raise RuntimeError(f"corpus/capture identity mismatch: {manifest_path}")
        calibration_input_hashes.update(mixed_calibration_entry_hashes(value, manifest_path))
    exclusions_raw = args.excluded_input_manifest.resolve().read_bytes()
    exclusions = json.loads(exclusions_raw)
    excluded_hashes = set(exclusions.get("sha256s", [])) if isinstance(exclusions, dict) else set()
    if not excluded_hashes or (set(corpus_hashes) | calibration_input_hashes) & excluded_hashes:
        raise RuntimeError("calibration and excluded validation inputs overlap or exclusions are invalid")

    config_raw = (model_dir / "config.json").read_bytes()
    root_config = json.loads(config_raw)
    config = root_config.get("text_config", root_config)
    index_raw = (model_dir / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index_raw).get("weight_map")
    if not isinstance(config, dict) or not isinstance(weight_map, dict):
        raise RuntimeError("checkpoint config/index contract is incomplete")
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    experts = int(config["n_routed_experts"])
    topk = int(config["num_experts_per_tok"])
    swiglu_limit = float(config.get("swiglu_limit", 0.0))
    if swiglu_limit <= 0:
        raise RuntimeError("this sensitivity evaluator requires the measured clamped SwiGLU contract")
    experts_gated = str(config.get("mlp_hidden_act", "")).lower() != "relu2"
    if not experts_gated:
        raise RuntimeError("ungated expert sensitivity is not implemented")
    sizes4 = _mixed_marlin_component_bytes(hidden, intermediate, args.group_size, 4, True)
    sizes8 = _mixed_marlin_component_bytes(hidden, intermediate, args.group_size, 8, True)
    record4, record8 = sum(sizes4), sum(sizes8)
    calibration_path = args.calibration_artifact.resolve()
    calibration_sha, calibration_stat = stable_sha256_file(calibration_path)
    kic_manifest, kic_raw, kic_payload_offset = load_kic1(calibration_path)
    if (
        kic_manifest.get("capture_corpora_sha256") != corpus_hashes
        or kic_manifest.get("capture_artifact_sha256") != capture_hashes
        or kic_manifest.get("calibration_sha256") != capture_binding_sha256(corpus_hashes)
        or int(kic_manifest.get("hidden_size", -1)) != hidden
        or int(kic_manifest.get("intermediate_size", -1)) != intermediate
        or int(kic_manifest.get("routed_experts", -1)) != experts
        or int(kic_manifest.get("topk", -1)) != topk
        or int(kic_manifest.get("routed_layers", -1)) != len(layer_names)
    ):
        raise RuntimeError("KIC1 identity/geometry does not match the verified captures and checkpoint")
    checkpoint = checkpoint_identity(str(model_dir))["sha256"]
    namespace = cache_namespace(str(model_dir))
    config_sha = hashlib.sha256(config_raw).hexdigest()
    exclusion_sha = hashlib.sha256(exclusions_raw).hexdigest()
    properties = torch.cuda.get_device_properties(device)
    layer_identity_common = {
        "ranking_format": FORMAT,
        "ranking_schema_version": VERSION,
        "coverage_policy": args.coverage_policy,
        "ranking_methods": ORDER_METHODS,
        "quality_gain_scale_exponent": GAIN_EXPONENT,
        "checkpoint_sha256": checkpoint,
        "cache_namespace": namespace,
        "config_sha256": config_sha,
        "checkpoint_index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "calibration_corpora_sha256": corpus_hashes,
        "calibration_corpus_manifest_sha256": corpus_manifest_hashes,
        "calibration_input_sha256": sorted(calibration_input_hashes),
        "capture_artifact_sha256": capture_hashes,
        "search_rmse_calibration_sha256": calibration_sha,
        "search_rmse_calibration_bytes": calibration_stat.st_size,
        "search_rmse_payload_sha256": kic_manifest["payload_sha256"],
        "excluded_input_set_sha256": exclusion_sha,
        "search_scale_factors": list(FACTORS),
        "quantizer_contract": "rust_materializer_exact_f32_sequential_v1",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "group_size": args.group_size,
        "tail_fraction": args.tail_fraction,
        "minimum_tail_observations": args.minimum_tail_observations,
        "minimum_routes_per_expert": minimum_routes,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "routed_experts": experts,
        "topk": topk,
        "routed_layers": len(layer_names),
        "experts_gated": experts_gated,
        "swiglu_limit": swiglu_limit,
        "int4_record_bytes": record4,
        "int8_record_bytes": record8,
        "device": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "uuid": str(getattr(properties, "uuid", "")),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_start = proc_memory_snapshot()
    started = time.monotonic()
    all_candidates = []
    audit_files = []
    layer_memory = []
    source_precisions: set[str] = set()
    for moe_layer, layer_name in enumerate(layer_names):
        layer_json = output_dir / f"sensitivity-layer-{moe_layer:03d}.json"
        layer_audit = output_dir / f"scale-audit-layer-{moe_layer:03d}.bin"
        expected_layer_identity = sensitivity_layer_identity(
            layer_identity_common,
            int(kic_manifest["layers"][moe_layer]["model_layer"]),
            moe_layer,
        )
        if layer_json.exists() or layer_audit.exists():
            if not (layer_json.exists() and layer_audit.exists()):
                raise RuntimeError(f"partial sensitivity layer artifact exists for {moe_layer}")
            value = load_json(layer_json)
            validate_sensitivity_layer_shard(
                value, layer_audit, expected_layer_identity, experts
            )
            all_candidates.extend(value["candidates"])
            audit_files.append(layer_audit)
            layer_memory.append(value["memory"])
            source_precisions.update(value["source_precisions"])
            continue
        torch.cuda.reset_peak_memory_stats(device)
        layer_source_precisions: set[str] = set()
        # Request shards are the crash-resilient transaction unit.  Statistical
        # diversity is instead defined by independently prepared corpora, so first
        # combine the verified requests within each corpus and only then combine
        # the corpora for sensitivity evaluation.
        capture_corpora = [
            combine_capture_layers([read_capture_layer(path) for path in inventory[layer_name]])
            for inventory, _files in captures
        ]
        capture = combine_capture_layers(capture_corpora)
        if (
            capture.expert_input_size != hidden
            or capture.intermediate_size != intermediate
            or capture.expert_count != experts
            or capture.topk != topk
        ):
            raise RuntimeError(f"capture/model geometry mismatch at model layer {capture.model_layer}")
        flat_ids = capture.topk_ids.reshape(-1)
        flat_weights = capture.topk_weights.reshape(-1)
        route_counts = torch.bincount(flat_ids, minlength=experts)
        route_mass = torch.bincount(
            flat_ids, weights=flat_weights.double().square(), minlength=experts
        )
        if not reserve_policy and int(route_counts.min().item()) < minimum_routes:
            missing = torch.nonzero(route_counts < minimum_routes).flatten().tolist()
            raise RuntimeError(
                f"model layer {capture.model_layer} lacks statistically sufficient real routes "
                f"(required={minimum_routes}, experts={missing})"
            )
        diversity = torch.zeros(experts, dtype=torch.int64)
        for corpus_capture in capture_corpora:
            diversity += (
                torch.bincount(corpus_capture.topk_ids.reshape(-1), minlength=experts) > 0
            ).to(torch.int64)
        if not reserve_policy and int(diversity.min().item()) != len(capture_corpora):
            raise RuntimeError(f"model layer {capture.model_layer} lacks per-corpus expert coverage")
        corpus_counts = [torch.bincount(part.topk_ids.reshape(-1), minlength=experts) for part in capture_corpora]
        required = coverage_required(route_counts, corpus_counts, minimum_routes)
        # Consume the exact verified KIC1 moments used by the Rust materializer.
        # Never recompute a merely equivalent proxy for the scored ranking.
        input_moments, down_moments = kic1_layer_moments(
            kic_manifest,
            kic_raw,
            kic_payload_offset,
            capture.model_layer,
            experts,
            hidden,
            intermediate,
        )

        audit_payload = bytearray(AUDIT_HEADER.pack(
            AUDIT_MAGIC, AUDIT_VERSION, capture.model_layer, experts, args.group_size,
            hidden, intermediate, 0,
        ))
        layer_candidates = []
        for expert in range(experts):
            coverage = {
                "route_count": int(route_counts[expert]),
                "corpus_route_counts": [int(item[expert]) for item in corpus_counts],
            }
            if reserve_policy and bool(required[expert]):
                layer_candidates.append({
                    "model_layer": capture.model_layer, "moe_layer": moe_layer, "expert": expert,
                    "int4_bytes": record4, "int8_bytes": record8,
                    **coverage, "coverage_required_int8": True,
                })
                print(f"[mixed-rank] layer={capture.model_layer} expert={expert + 1}/{experts} coverage_required_int8 routes={coverage['route_count']}", flush=True)
                continue
            prefix = f".layers.{capture.model_layer}.mlp.experts.{expert}"
            names = {
                projection: unique_tensor_name(weight_map, f"{prefix}.{projection}.weight")
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            loaded_source = tuple(
                source_projection(model_dir, weight_map, names[name]) for name in names
            )
            source = tuple(value[0] for value in loaded_source)
            layer_source_precisions.update(value[1] for value in loaded_source)
            optimized = []
            int8 = []
            quantization_started = time.monotonic()
            worst_group4 = 0.0
            for projection_id, (name, source_weight) in enumerate(zip(names, source)):
                moments = down_moments[expert] if name == "down_proj" else input_moments[expert]
                q4, audit = materialized_search_rmse(source_weight, moments, args.group_size, device)
                q8 = materialized_int8(source_weight, args.group_size)
                optimized.append(q4)
                int8.append(q8)
                for group, factor_index, baseline_error, optimized_error, boundary in audit:
                    worst_group4 = max(worst_group4, optimized_error)
                    audit_payload.extend(AUDIT_ROW.pack(
                        capture.model_layer, expert, projection_id, 0, group,
                        factor_index, int(boundary), baseline_error, optimized_error,
                    ))
            quantization_seconds = time.monotonic() - quantization_started
            measurement_started = time.monotonic()
            metrics, batch_rows = full_expert_metrics(
                capture, expert, source, tuple(optimized), tuple(int8), swiglu_limit, device
            )
            error4 = metrics["search_rmse_int4"]
            error8 = metrics["int8"]
            expected_gain = expected_exposure_gain(
                error4["router_weighted_relative_rmse"],
                error8["router_weighted_relative_rmse"],
                float(route_mass[expert].item()),
                int(capture.expert_inputs.shape[0]),
            )
            tail_gain = max(0.0, error4["cvar99"] - error8["cvar99"])
            candidate = {
                "model_layer": capture.model_layer,
                "moe_layer": moe_layer,
                "expert": expert,
                "int4_bytes": record4,
                "int8_bytes": record8,
                **coverage,
                "coverage_required_int8": False,
                "squared_router_mass": format(float(route_mass[expert].item()), ".17g"),
                "corpus_diversity": int(diversity[expert].item()),
                "coverage_required": minimum_routes,
                "measurement_batch_rows": batch_rows,
                "quantization_seconds": quantization_seconds,
                "measurement_seconds": time.monotonic() - measurement_started,
                "int4_projection_sha256": [projection_sha256(value) for value in optimized],
                "int8_projection_sha256": [projection_sha256(value) for value in int8],
                "metrics": metrics,
                "worst_projection_int4": max(error4["w13_relative_rmse"], error4["w2_relative_rmse"]),
                "worst_projection_int8": max(error8["w13_relative_rmse"], error8["w2_relative_rmse"]),
                "worst_internal_group_int4": worst_group4,
                "expected_gain_units": max(0, round(expected_gain * 10**12)),
                "tail_gain_units": max(0, round(tail_gain * 10**12)),
            }
            layer_candidates.append(candidate)
            print(
                f"[mixed-rank] layer={capture.model_layer} expert={expert + 1}/{experts} "
                f"routes={candidate['route_count']} e4={error4['router_weighted_relative_rmse']:.6g} "
                f"e8={error8['router_weighted_relative_rmse']:.6g} "
                f"quantization_s={candidate['quantization_seconds']:.3f} measurement_s={candidate['measurement_seconds']:.3f}",
                flush=True,
            )
        audit_sha = hashlib.sha256(audit_payload).hexdigest()
        memory = proc_memory_snapshot()
        memory.update({
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "scratch_regular_bytes_before_commit": directory_regular_bytes(output_dir),
        })
        layer_value = {
            "format": SENSITIVITY_LAYER_FORMAT,
            "schema_version": SENSITIVITY_LAYER_VERSION,
            "model_layer": capture.model_layer,
            "moe_layer": moe_layer,
            "identity": expected_layer_identity,
            "identity_sha256": hashlib.sha256(
                canonical_bytes(expected_layer_identity)
            ).hexdigest(),
            "capture_artifact_sha256": capture_hashes,
            "calibration_sha256": calibration_sha,
            "audit_basename": layer_audit.name,
            "audit_sha256": audit_sha,
            "audit_rows": (len(audit_payload) - AUDIT_HEADER.size) // AUDIT_ROW.size,
            "source_precisions": sorted(layer_source_precisions),
            "memory": memory,
            "candidates": layer_candidates,
        }
        atomic_write(layer_audit, bytes(audit_payload))
        atomic_write(layer_json, canonical_bytes(layer_value))
        all_candidates.extend(layer_candidates)
        audit_files.append(layer_audit)
        layer_memory.append(memory)
        source_precisions.update(layer_source_precisions)
        print(
            f"[mixed-rank] committed routed layer {moe_layer + 1}/{len(layer_names)} "
            f"elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )

    if len(all_candidates) != len(layer_names) * experts:
        raise RuntimeError("sensitivity shards do not cover every routed expert")
    assign_quality_ranks(all_candidates)

    evidence = {
        "format": FORMAT,
        "schema_version": VERSION,
        "coverage_policy": args.coverage_policy,
        "ranking_methods": ORDER_METHODS,
        "quality_gain_scale_exponent": GAIN_EXPONENT,
        "checkpoint_sha256": checkpoint,
        "cache_namespace": namespace,
        "config_sha256": config_sha,
        "source_storage": sorted(source_precisions),
        "search_scale_factors": list(FACTORS),
        "tail_fraction": args.tail_fraction,
        "minimum_tail_observations": args.minimum_tail_observations,
        "minimum_routes_per_expert": minimum_routes,
        "calibration_corpora_sha256": corpus_hashes,
        "calibration_corpus_manifest_sha256": corpus_manifest_hashes,
        "calibration_input_sha256": sorted(calibration_input_hashes),
        "capture_artifact_sha256": capture_hashes,
        "search_rmse_calibration_sha256": calibration_sha,
        "excluded_input_set_sha256": exclusion_sha,
        "scale_audit": [
            {"basename": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in audit_files
        ],
        "device": {
            "logical": str(device),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "uuid": str(getattr(properties, "uuid", "")),
        },
        "memory": {
            "start": memory_start,
            "layers": layer_memory,
            "peak_process_vm_hwm_bytes": max(
                (item.get("process_vmhwm_bytes", 0) for item in layer_memory), default=0
            ),
            "minimum_system_memavailable_bytes": min(
                (item.get("system_memavailable_bytes", 0) for item in layer_memory), default=0
            ),
            "peak_cuda_allocated_bytes": max(
                (item.get("cuda_peak_allocated_bytes", 0) for item in layer_memory), default=0
            ),
            "peak_cuda_reserved_bytes": max(
                (item.get("cuda_peak_reserved_bytes", 0) for item in layer_memory), default=0
            ),
            "scratch_regular_bytes_before_publication": directory_regular_bytes(output_dir),
        },
        "candidates": sorted(all_candidates, key=lambda item: (item["moe_layer"], item["expert"])),
    }
    evidence_raw = canonical_bytes(evidence)
    evidence_path = output_dir / "mixed-expert-ranking.json"
    if evidence_path.exists():
        if evidence_path.read_bytes() != evidence_raw:
            raise RuntimeError("refusing to replace different mixed-expert ranking evidence")
    else:
        atomic_write(evidence_path, evidence_raw)
    evidence_sha = hashlib.sha256(evidence_raw).hexdigest()
    baseline = len(all_candidates) * record4
    delta = record8 - record4
    model_layers = [read_capture_layer(captures[0][0][name][0]).model_layer for name in layer_names]
    aggregate_capture = capture_binding_sha256(capture_hashes)
    for ordering, method in ORDER_METHODS.items():
        score_key = f"{ordering}_gain_units"
        rank_key = f"{ordering}_rank"
        reserve_bytes = sum(item["int8_bytes"] - item["int4_bytes"] for item in all_candidates if item.get("coverage_required_int8"))
        for budget in BUDGETS:
            cap = reserve_bytes if reserve_policy and budget == 0 else baseline * budget // 10_000
            selected, used = select_with_reserve(all_candidates, rank_key, score_key, cap, budget == 10_000)
            by_identity = {(item["moe_layer"], item["expert"]): item for item in all_candidates}
            regions = []
            for moe_layer, model_layer in enumerate(model_layers):
                for expert in range(experts):
                    item = by_identity[(moe_layer, expert)]
                    regions.append({
                        "model_layer": model_layer,
                        "moe_layer": moe_layer,
                        "expert": expert,
                        "bits": 8 if (moe_layer, expert) in selected else 4,
                        "int4_bytes": record4,
                        "int8_bytes": record8,
                        **({} if item.get("coverage_required_int8") else {
                            "quality_gain_units": item[score_key], "quality_rank": item[rank_key],
                        }),
                        **({"coverage": {
                            "route_count": item["route_count"],
                            "corpus_route_counts": item["corpus_route_counts"],
                            "selection_reason": "coverage_required_int8" if item.get("coverage_required_int8") else (
                                "measured_promotion" if (moe_layer, expert) in selected else "measured_int4"),
                        }} if reserve_policy else {}),
                    })
            manifest = {
                "format": MANIFEST_FORMAT,
                "schema_version": 3 if reserve_policy else MANIFEST_VERSION,
                "checkpoint_sha256": checkpoint,
                "cache_namespace": namespace,
                "config_sha256": config_sha,
                "group_size": args.group_size,
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "routed_layers": len(layer_names),
                "routed_experts_per_layer": experts,
                "experts_gated": True,
                "default_bits": 4,
                "calibration": {
                    "artifact_sha256": aggregate_capture,
                    "ranking_evidence_sha256": evidence_sha,
                    "ranking_method": method,
                    "quality_gain_scale_exponent": GAIN_EXPONENT,
                    "corpus_sha256": corpus_hashes,
                    "capture_artifact_sha256": capture_hashes,
                    "excluded_input_set_sha256": exclusion_sha,
                    "search_rmse_calibration_sha256": calibration_sha,
                    **({"coverage_policy": {"method": args.coverage_policy, "minimum_routes_per_expert": minimum_routes, "required_corpora": len(capture_dirs)}} if reserve_policy else {}),
                },
                "budget": {
                    "requested_basis_points": budget,
                    "baseline_int4_routed_bytes": baseline,
                    "maximum_added_bytes": cap,
                    "achieved_added_bytes": used,
                    "achieved_percent_millionths": used * 100_000_000 // baseline,
                },
                "regions": regions,
            }
            plan = {"format": PLAN_FORMAT, "schema_version": PLAN_VERSION, "manifest": manifest}
            label = "full-int8" if budget == 10_000 else ("minimum-reserve" if reserve_policy and budget == 0 else f"plus-{budget // 100}")
            path = output_dir / f"mixed-{ordering}-{label}.plan.json"
            raw = canonical_bytes(plan)
            if path.exists() and path.read_bytes() != raw:
                raise RuntimeError(f"refusing to replace different build plan {path}")
            if not path.exists():
                atomic_write(path, raw)
    if args.calibration_artifact.stat().st_size != calibration_stat.st_size or args.calibration_artifact.stat().st_mtime_ns != calibration_stat.st_mtime_ns:
        raise RuntimeError("search-RMSE calibration artifact changed during sensitivity measurement")
    print(json.dumps({
        "ranking_evidence": str(evidence_path),
        "ranking_sha256": evidence_sha,
        "experts": len(all_candidates),
        "plans": len(ORDER_METHODS) * len(BUDGETS),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
