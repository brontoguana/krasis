#!/usr/bin/env python3
"""Build and verify source-bound Krasis TileQ-S routed-expert caches.

This is an offline calibration/build tool. Model execution never imports this
module; the production hot path remains Rust/CUDA. Invoke it through
`./dev tileq-build` so the repository environment and direct-execution guard
are enforced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Run this tool through ./dev tileq-build; direct execution is disabled")

# CUDA's deterministic GEMM contract must be selected before the first cuBLAS
# handle is created. The artifact manifest records this exact build contract.
_DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"
_configured_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _configured_cublas_workspace not in (None, _DETERMINISTIC_CUBLAS_WORKSPACE):
    raise SystemExit(
        "TileQ build requires CUBLAS_WORKSPACE_CONFIG="
        f"{_DETERMINISTIC_CUBLAS_WORKSPACE}, got {_configured_cublas_workspace}"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _DETERMINISTIC_CUBLAS_WORKSPACE

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


MAGIC = b"KTQ1"
VERSION = 1
HEADER_BYTES = 64
PAYLOAD_OFFSET = 4 * 1024 * 1024
CAPTURE_MAGIC = b"KTC1"
CAPTURE_VERSION = 2


@dataclass
class CaptureLayer:
    model_layer: int
    expert_input_size: int
    intermediate_size: int
    topk: int
    expert_count: int
    expert_inputs: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    down_inputs: torch.Tensor


@dataclass
class ProxySamples:
    inputs: torch.Tensor
    importance: torch.Tensor
    counterfactual_experts: list[int]
    counterfactual_importance: float | None


def sha256_file(path: Path, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        encoded = path.name.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", path.stat().st_size))
        with path.open("rb", buffering=0) as handle:
            while True:
                chunk = handle.read(64 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read JSON {path}: {exc}") from exc


def text_config(model_dir: Path) -> tuple[dict, dict, dict]:
    root = load_json(model_dir / "config.json")
    config = root.get("text_config", root)
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model.safetensors.index.json has no weight_map")
    return root, config, weight_map


def discover_expert_tensors(weight_map: dict, layer_count: int) -> list[dict[str, str]]:
    discovered: list[dict[str, str]] = []
    for layer in range(layer_count):
        suffixes = {
            "gate_up": f"layers.{layer}.mlp.experts.gate_up_proj",
            "down": f"layers.{layer}.mlp.experts.down_proj",
            "gate": f"layers.{layer}.mlp.experts.gate_proj.weight",
            "up": f"layers.{layer}.mlp.experts.up_proj.weight",
            "down_separate": f"layers.{layer}.mlp.experts.down_proj.weight",
        }
        matches: dict[str, str] = {}
        for kind, suffix in suffixes.items():
            exact = [name for name in weight_map if name.endswith(suffix)]
            if len(exact) > 1:
                raise RuntimeError(f"layer {layer} has ambiguous {kind} tensors: {exact}")
            if exact:
                matches[kind] = exact[0]
        if "gate_up" in matches and "down" in matches:
            discovered.append({"gate_up": matches["gate_up"], "down": matches["down"]})
        elif all(key in matches for key in ("gate", "up", "down_separate")):
            discovered.append(
                {
                    "gate": matches["gate"],
                    "up": matches["up"],
                    "down": matches["down_separate"],
                }
            )
        else:
            raise RuntimeError(
                f"layer {layer} does not expose a supported fused or separate routed-expert tensor set"
            )
    return discovered


def read_capture_layer(path: Path) -> CaptureLayer:
    raw = np.memmap(path, mode="r", dtype=np.uint8)
    if raw.size < 32:
        raise RuntimeError(f"capture {path} is truncated")
    magic, version, layer, input_size, intermediate, topk, experts, reserved = struct.unpack_from(
        "<4s7I", raw, 0
    )
    if magic != CAPTURE_MAGIC or version != CAPTURE_VERSION or reserved != 0:
        raise RuntimeError(f"capture {path} has invalid header")
    offset = 32
    xs: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    downs: list[np.ndarray] = []
    while offset < raw.size:
        if offset + 4 > raw.size:
            raise RuntimeError(f"capture {path} has truncated chunk row count")
        rows = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if rows == 0:
            raise RuntimeError(f"capture {path} contains an empty chunk")
        x_count = rows * input_size
        id_count = rows * topk
        down_count = id_count * intermediate
        x_bytes = x_count * 2
        id_bytes = id_count * 4
        weight_bytes = id_count * 4
        down_bytes = down_count * 2
        end = offset + x_bytes + id_bytes + weight_bytes + down_bytes
        if end > raw.size:
            raise RuntimeError(f"capture {path} has truncated chunk payload")
        xs.append(np.frombuffer(raw, dtype="<u2", count=x_count, offset=offset).reshape(rows, input_size))
        offset += x_bytes
        ids.append(np.frombuffer(raw, dtype="<i4", count=id_count, offset=offset).reshape(rows, topk))
        offset += id_bytes
        weights.append(
            np.frombuffer(raw, dtype="<f4", count=id_count, offset=offset).reshape(rows, topk)
        )
        offset += weight_bytes
        downs.append(
            np.frombuffer(raw, dtype="<u2", count=down_count, offset=offset).reshape(
                rows, topk, intermediate
            )
        )
        offset += down_bytes
    if not xs:
        raise RuntimeError(f"capture {path} contains no rows")
    x_bits = torch.from_numpy(np.concatenate(xs, axis=0).copy())
    down_bits = torch.from_numpy(np.concatenate(downs, axis=0).copy())
    topk_ids = torch.from_numpy(np.concatenate(ids, axis=0).copy()).to(torch.int64)
    topk_weights = torch.from_numpy(np.concatenate(weights, axis=0).copy()).to(torch.float32)
    expert_inputs = x_bits.view(torch.bfloat16)
    down_inputs = down_bits.view(torch.bfloat16)
    if int(topk_ids.min()) < 0 or int(topk_ids.max()) >= experts:
        raise RuntimeError(f"capture {path} contains out-of-range expert IDs")
    if not torch.isfinite(topk_weights).all() or torch.any(topk_weights < 0):
        raise RuntimeError(f"capture {path} contains invalid router weights")
    return CaptureLayer(
        model_layer=layer,
        expert_input_size=input_size,
        intermediate_size=intermediate,
        topk=topk,
        expert_count=experts,
        expert_inputs=expert_inputs,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        down_inputs=down_inputs,
    )


def combine_capture_layers(layers: list[CaptureLayer]) -> CaptureLayer:
    if not layers:
        raise RuntimeError("cannot combine an empty capture-layer set")
    first = layers[0]
    for layer in layers[1:]:
        if (
            layer.model_layer != first.model_layer
            or layer.expert_input_size != first.expert_input_size
            or layer.intermediate_size != first.intermediate_size
            or layer.topk != first.topk
            or layer.expert_count != first.expert_count
        ):
            raise RuntimeError(
                f"capture layer geometry mismatch while combining model layer {first.model_layer}"
            )
    return CaptureLayer(
        model_layer=first.model_layer,
        expert_input_size=first.expert_input_size,
        intermediate_size=first.intermediate_size,
        topk=first.topk,
        expert_count=first.expert_count,
        expert_inputs=torch.cat([layer.expert_inputs for layer in layers], dim=0),
        topk_ids=torch.cat([layer.topk_ids for layer in layers], dim=0),
        topk_weights=torch.cat([layer.topk_weights for layer in layers], dim=0),
        down_inputs=torch.cat([layer.down_inputs for layer in layers], dim=0),
    )


def capture_binding_sha256(digests: list[str]) -> str:
    binding = json.dumps(digests, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(binding).hexdigest()


def capture_metadata(capture_dir: Path) -> dict:
    metadata = load_json(capture_dir / "capture.json")
    if metadata.get("schema_version") != CAPTURE_VERSION or metadata.get("format") != "KTC1":
        raise RuntimeError(f"capture metadata {capture_dir / 'capture.json'} is incompatible")
    if metadata.get("includes_router_weights") is not True:
        raise RuntimeError(f"capture metadata {capture_dir / 'capture.json'} omits router weights")
    digest = metadata.get("calibration_sha256", "")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise RuntimeError(f"capture metadata {capture_dir / 'capture.json'} has invalid corpus hash")
    return metadata


def bf16_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().to(torch.bfloat16).view(torch.uint16).cpu().numpy().tobytes()


def align(value: int, alignment: int = 4096) -> int:
    return (value + alignment - 1) // alignment * alignment


def allocate_range(cursor: int, length: int) -> tuple[dict[str, int], int]:
    start = align(cursor)
    return {"offset": start, "len": length}, start + length


def build_layout(
    model_id: str,
    architecture: str,
    hidden: int,
    intermediate: int,
    experts: int,
    model_layers: list[int],
    group_size: int,
    rank: int,
) -> tuple[dict, int]:
    grid_rows = max(1, round(math.sqrt(experts)))
    grid_cols = math.ceil(experts / grid_rows)
    cursor = 0
    layers = []
    for model_layer in model_layers:
        gate_packed = intermediate * hidden * 3 // 8
        gate_scales = intermediate * (hidden // group_size) * 2
        down_packed = hidden * intermediate * 3 // 8
        down_scales = hidden * (intermediate // group_size) * 2
        per_w13p = gate_packed * 2
        per_w13s = gate_scales * 2
        per_w2p = down_packed
        per_w2s = down_scales
        w13p, cursor = allocate_range(cursor, experts * per_w13p)
        w13s, cursor = allocate_range(cursor, experts * per_w13s)
        w2p, cursor = allocate_range(cursor, experts * per_w2p)
        w2s, cursor = allocate_range(cursor, experts * per_w2s)
        projections = {}
        for name, input_dim, output_dim in (
            ("gate", hidden, intermediate),
            ("up", hidden, intermediate),
            ("down", intermediate, hidden),
        ):
            tiles, cursor = allocate_range(cursor, experts * 4)
            inverse_scales, cursor = allocate_range(cursor, experts * input_dim * 2)
            left, cursor = allocate_range(cursor, grid_rows * input_dim * rank * 2)
            right, cursor = allocate_range(cursor, grid_cols * rank * output_dim * 2)
            projections[name] = {
                "name": name,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "rank": rank,
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "expert_tiles": tiles,
                "expert_inverse_scales_bf16": inverse_scales,
                "left_factors_bf16": left,
                "right_factors_bf16": right,
                "selected_scale_exponent": 0.0,
                "heldout_weighted_mse": 0.0,
            }
        layers.append(
            {
                "model_layer": model_layer,
                "expert_count": experts,
                "w13_packed": w13p,
                "w13_scales": w13s,
                "w2_packed": w2p,
                "w2_scales": w2s,
                "per_expert_w13_packed": per_w13p,
                "per_expert_w13_scales": per_w13s,
                "per_expert_w2_packed": per_w2p,
                "per_expert_w2_scales": per_w2s,
                "gate": projections["gate"],
                "up": projections["up"],
                "down": projections["down"],
            }
        )
    payload_bytes = align(cursor)
    manifest = {
        "schema_version": VERSION,
        "model_id": model_id,
        "architecture": architecture,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "routed_experts": experts,
        "routed_layers": len(layers),
        "residual_bits": 3,
        "group_size": group_size,
        "rank": rank,
        "source_routed_sha256": "0" * 64,
        "source_config_sha256": "0" * 64,
        "calibration_sha256": "0" * 64,
        "heldout_sha256": "0" * 64,
        "scale_exponent_candidates": [],
        "sketch_seed": 0,
        "sketch_iterations": 0,
        "clustering_seed": 0,
        "residual_quantizer": "tileq_s_int3_per_expert_diagonal_hessian_scale_search_v2",
        "scale_search_multipliers": [],
        "gptq_block_size": group_size,
        "payload_bytes": payload_bytes,
        "layers": layers,
    }
    return manifest, payload_bytes


def kmeans(features: torch.Tensor, clusters: int, seed: int, iterations: int = 20) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] < clusters:
        raise RuntimeError(f"invalid KMeans geometry {tuple(features.shape)} clusters={clusters}")
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    initial = torch.randperm(features.shape[0], generator=generator, device=features.device)[:clusters]
    centers = features[initial].clone()
    labels = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device)
    for _ in range(iterations):
        new_labels = torch.argmax(features @ centers.T, dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(clusters):
            selected = features[labels == cluster]
            if selected.numel() == 0:
                distances = 1.0 - torch.max(features @ centers.T, dim=1).values
                centers[cluster] = features[torch.argmax(distances)]
            else:
                center = selected.mean(dim=0)
                centers[cluster] = center / center.norm().clamp_min(1e-12)
    return labels.cpu()


def greedy_placement(row_labels: torch.Tensor, col_labels: torch.Tensor, rows: int, cols: int) -> list[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    placement: list[tuple[int, int]] = []
    for row_label, col_label in zip(row_labels.tolist(), col_labels.tolist()):
        candidates = [
            (abs(row - row_label) + abs(col - col_label), max(abs(row - row_label), abs(col - col_label)), row, col)
            for row in range(rows)
            for col in range(cols)
            if (row, col) not in occupied
        ]
        if not candidates:
            raise RuntimeError("TileQ placement ran out of cells")
        _, _, row, col = min(candidates)
        occupied.add((row, col))
        placement.append((row, col))
    return placement


def counterfactual_rows(capture: CaptureLayer, experts: list[int]) -> torch.Tensor:
    """Select deterministic, evenly spread real hidden states for rare experts."""
    if not experts:
        return torch.empty(0, dtype=torch.int64)
    rows = capture.expert_inputs.shape[0]
    if rows <= 0:
        raise RuntimeError("counterfactual calibration requires captured hidden states")
    denominator = max(capture.expert_count - 1, 1)
    return (
        torch.tensor(experts, dtype=torch.int64) * max(rows - 1, 0) // denominator
    ).clamp_max(rows - 1)


def counterfactual_hidden_sample(capture: CaptureLayer) -> torch.Tensor:
    """Return an expert-count-derived, evenly spaced sample of measured states."""
    rows = capture.expert_inputs.shape[0]
    sample_count = min(capture.expert_count, rows)
    if sample_count <= 0:
        raise RuntimeError("counterfactual calibration requires captured hidden states")
    if sample_count == 1:
        indices = torch.zeros(1, dtype=torch.int64)
    else:
        indices = (
            torch.arange(sample_count, dtype=torch.int64) * (rows - 1) // (sample_count - 1)
        )
    return capture.expert_inputs[indices]


def validate_counterfactual_activation(config: dict) -> str:
    activation = str(config.get("hidden_act", "")).lower()
    if activation != "silu":
        raise RuntimeError(
            "counterfactual down-projection calibration currently requires hidden_act=silu; "
            f"the model declares {activation or '<missing>'}"
        )
    swiglu_limit = config.get("swiglu_limit")
    if swiglu_limit is not None:
        raise RuntimeError(
            "counterfactual down-projection calibration does not yet implement swiglu_limit; "
            f"the model declares {swiglu_limit}"
        )
    return activation


def counterfactual_down_inputs(
    source_weights: dict[str, torch.Tensor],
    experts: list[int],
    hidden_inputs: torch.Tensor,
    activation: str,
    device: torch.device,
) -> torch.Tensor:
    """Execute selected BF16 experts on real hidden states outside model runtime."""
    if not experts:
        return torch.empty(
            (0, hidden_inputs.shape[1], source_weights["gate"].shape[1]),
            device=device,
            dtype=torch.bfloat16,
        )
    if activation != "silu":
        raise RuntimeError(f"unsupported counterfactual expert activation {activation}")
    if hidden_inputs.ndim != 3 or hidden_inputs.shape[0] != len(experts):
        raise RuntimeError(
            f"invalid counterfactual hidden input geometry {tuple(hidden_inputs.shape)} "
            f"for {len(experts)} experts"
        )
    expert_index = torch.tensor(experts, dtype=torch.int64)
    gate_weights = source_weights["gate"][expert_index].to(
        device=device, dtype=torch.bfloat16
    )
    up_weights = source_weights["up"][expert_index].to(
        device=device, dtype=torch.bfloat16
    )
    hidden = hidden_inputs.to(device=device, dtype=torch.bfloat16)
    gate = torch.bmm(hidden, gate_weights.transpose(1, 2))
    up = torch.bmm(hidden, up_weights.transpose(1, 2))
    result = F.silu(gate) * up
    del gate_weights, up_weights, hidden, gate, up
    return result


def activation_means(
    capture: CaptureLayer,
    projection: str,
    source_weights: dict[str, torch.Tensor],
    activation: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    experts = capture.expert_count
    if projection in ("gate", "up"):
        values = capture.expert_inputs.to(device=device, dtype=torch.float32)
        sums = torch.zeros((experts, values.shape[1]), device=device, dtype=torch.float32)
        counts = torch.zeros(experts, device=device, dtype=torch.float32)
        ids = capture.topk_ids.to(device)
        route_weights = capture.topk_weights.to(device=device, dtype=torch.float32).abs()
        absolute = values.abs()
        for slot in range(capture.topk):
            slot_ids = ids[:, slot]
            importance = route_weights[:, slot]
            sums.index_add_(0, slot_ids, absolute * importance[:, None])
            counts.index_add_(0, slot_ids, importance)
    else:
        values = capture.down_inputs.reshape(-1, capture.intermediate_size).to(device=device, dtype=torch.float32)
        ids = capture.topk_ids.reshape(-1).to(device)
        route_weights = capture.topk_weights.reshape(-1).to(device=device, dtype=torch.float32).abs()
        sums = torch.zeros((experts, values.shape[1]), device=device, dtype=torch.float32)
        counts = torch.zeros(experts, device=device, dtype=torch.float32)
        sums.index_add_(0, ids, values.abs() * route_weights[:, None])
        counts.index_add_(0, ids, route_weights)
    missing = torch.nonzero(counts == 0).flatten().cpu().tolist()
    means = sums / counts.clamp_min(1e-20)[:, None]
    if missing:
        if projection in ("gate", "up"):
            real_hidden = counterfactual_hidden_sample(capture).to(
                device=device, dtype=torch.float32
            )
            means[missing] = real_hidden.abs().mean(dim=0)
            del real_hidden
        else:
            real_hidden = counterfactual_hidden_sample(capture)
            expanded = real_hidden.unsqueeze(0).expand(len(missing), -1, -1)
            counterfactual = counterfactual_down_inputs(
                source_weights, missing, expanded, activation, device
            )
            means[missing] = counterfactual.float().abs().mean(dim=1)
            del real_hidden, expanded, counterfactual
    return means, missing


def expert_hessian_diagonal(
    capture: CaptureLayer,
    projection: str,
    source_weights: dict[str, torch.Tensor],
    activation: str,
    device: torch.device,
    row_chunk: int = 512,
) -> tuple[torch.Tensor, list[int]]:
    """Measure a route-specific diagonal Hessian for every expert.

    The v1 builder collapsed all routed activations into one global diagonal,
    even though experts see different input distributions.  Accumulate in
    bounded token chunks so this remains practical for 65K-row captures and
    complete genuinely unseen experts with the same explicit BF16
    counterfactual contract used by activation scaling.
    """
    experts = capture.expert_count
    input_dim = (
        capture.expert_input_size
        if projection in ("gate", "up")
        else capture.intermediate_size
    )
    sums = torch.zeros((experts, input_dim), device=device, dtype=torch.float32)
    importance = torch.zeros(experts, device=device, dtype=torch.float32)
    for start in range(0, capture.expert_inputs.shape[0], row_chunk):
        end = min(capture.expert_inputs.shape[0], start + row_chunk)
        ids = capture.topk_ids[start:end].reshape(-1).to(device=device)
        weights = (
            capture.topk_weights[start:end]
            .reshape(-1)
            .to(device=device, dtype=torch.float32)
            .abs()
        )
        if projection in ("gate", "up"):
            hidden = capture.expert_inputs[start:end].to(
                device=device, dtype=torch.float32
            )
            values = hidden[:, None, :].expand(-1, capture.topk, -1).reshape(
                -1, input_dim
            )
        else:
            values = capture.down_inputs[start:end].reshape(-1, input_dim).to(
                device=device, dtype=torch.float32
            )
        squared_weights = weights.square()
        sums.index_add_(0, ids, values.square() * squared_weights[:, None])
        importance.index_add_(0, ids, squared_weights)
        del ids, weights, values, squared_weights

    missing = torch.nonzero(importance == 0).flatten().cpu().tolist()
    diagonal = sums / importance.clamp_min(1e-20)[:, None]
    if missing:
        hidden = capture.expert_inputs[counterfactual_rows(capture, missing)]
        if projection in ("gate", "up"):
            counterfactual = hidden.to(device=device, dtype=torch.float32)
        else:
            generated = counterfactual_down_inputs(
                source_weights,
                missing,
                hidden.unsqueeze(1),
                activation,
                device,
            )
            counterfactual = generated[:, 0].to(dtype=torch.float32)
            del generated
        diagonal[missing] = counterfactual.square()
        del hidden, counterfactual
    return diagonal.clamp_min(1e-8), missing


def global_hessian_diagonal(
    capture: CaptureLayer,
    projection: str,
    row_chunk: int = 512,
) -> torch.Tensor:
    """Reproduce v1's shared diagonal from calibration data only.

    This control exists so held-out selection can score the exact global-
    diagonal quantization later published, without the v1 leakage where the
    held-out rows silently supplied a different quantizer Hessian.
    """
    input_dim = (
        capture.expert_input_size
        if projection in ("gate", "up")
        else capture.intermediate_size
    )
    sums = torch.zeros(input_dim, dtype=torch.float64)
    rows = 0
    if projection in ("gate", "up"):
        source = capture.expert_inputs
    else:
        source = capture.down_inputs.reshape(-1, input_dim)
    for start in range(0, source.shape[0], row_chunk):
        values = source[start : start + row_chunk].float()
        sums += values.square().sum(dim=0, dtype=torch.float64)
        rows += values.shape[0]
    if rows == 0:
        raise RuntimeError("cannot construct a global Hessian from an empty capture")
    return (sums / rows).float().clamp_min(1e-8)


def scaled_weights(
    weights: torch.Tensor, means: torch.Tensor, exponent: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if exponent == 0.0:
        normalized = torch.ones_like(means, dtype=torch.float32)
    else:
        log_mean = means.clamp_min(1e-8).log()
        normalized = torch.exp(exponent * (log_mean - log_mean.mean(dim=1, keepdim=True)))
    return weights.transpose(1, 2) * normalized[:, :, None], normalized.reciprocal()


def sketch_features(a_weights: torch.Tensor, rank0: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    experts, input_dim, output_dim = a_weights.shape
    generator = torch.Generator(device=a_weights.device)
    generator.manual_seed(seed)
    omega_out = torch.randn(output_dim, rank0, generator=generator, device=a_weights.device)
    omega_in = torch.randn(input_dim, rank0, generator=generator, device=a_weights.device)
    left = torch.matmul(a_weights, omega_out).reshape(experts, -1)
    right = torch.matmul(a_weights.transpose(1, 2), omega_in).reshape(experts, -1)
    left = left / left.norm(dim=1, keepdim=True).clamp_min(1e-12)
    right = right / right.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return left, right


def tileq_factors(
    weights: torch.Tensor,
    means: torch.Tensor,
    exponent: float,
    rank: int,
    grid_rows: int,
    grid_cols: int,
    sketch_seed: int,
    sketch_iterations: int,
    clustering_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    a_weights, inverse_scales = scaled_weights(weights, means, exponent)
    left_features, right_features = sketch_features(a_weights, rank // 2, sketch_seed)
    row_labels = kmeans(left_features, grid_rows, clustering_seed)
    col_labels = kmeans(right_features, grid_cols, clustering_seed + 1)
    placement = greedy_placement(row_labels, col_labels, grid_rows, grid_cols)
    input_dim = a_weights.shape[1]
    output_dim = a_weights.shape[2]
    tiled = torch.zeros(
        (grid_rows * input_dim, grid_cols * output_dim),
        device=a_weights.device,
        # torch.svd_lowrank uses an FP32 randomized sketch for FP32 source
        # weights. Keep the offline factorization in that same precision;
        # factors are converted to BF16 only when the artifact is written.
        dtype=a_weights.dtype,
    )
    for expert, (row, col) in enumerate(placement):
        tiled[
            row * input_dim : (row + 1) * input_dim,
            col * output_dim : (col + 1) * output_dim,
        ] = a_weights[expert]
    q = rank + 8
    rng_devices = (
        [a_weights.device.index if a_weights.device.index is not None else torch.cuda.current_device()]
        if a_weights.device.type == "cuda"
        else []
    )
    # torch.svd_lowrank creates its own random projection and does not accept a
    # Generator. Seed it inside a forked context so candidate order and prior
    # CUDA work cannot change an otherwise source-identical artifact.
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(sketch_seed)
        if a_weights.device.type == "cuda":
            torch.cuda.manual_seed(sketch_seed)
        u, singular, v = torch.svd_lowrank(
            tiled, q=q, niter=sketch_iterations
        )
    order = torch.argsort(singular, descending=True)[:rank]
    left = (u[:, order] * singular[order]).reshape(grid_rows, input_dim, rank).contiguous()
    right = v[:, order].T.reshape(rank, grid_cols, output_dim).permute(1, 0, 2).contiguous()
    return left, right, inverse_scales, placement


def correction_batch(
    left: torch.Tensor,
    right: torch.Tensor,
    inverse_scales: torch.Tensor,
    placement: list[tuple[int, int]],
    start: int,
    end: int,
) -> torch.Tensor:
    rows = torch.tensor([placement[index][0] for index in range(start, end)], device=left.device)
    cols = torch.tensor([placement[index][1] for index in range(start, end)], device=right.device)
    # Equation 12: the tiled approximation was built from W*s, therefore the
    # selected block must be multiplied by s^-1 before it is added to the
    # residual. [B,input,rank] @ [B,rank,output] -> [B,output,input].
    scaled = torch.bmm(left[rows], right[cols])
    return (scaled * inverse_scales[start:end, :, None]).transpose(1, 2).contiguous()


def diagonal_hessian_quantize(
    residual: torch.Tensor,
    hdiag: torch.Tensor,
    group_size: int,
    multipliers: torch.Tensor,
    row_chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts, output_dim, input_dim = residual.shape
    if input_dim % group_size:
        raise RuntimeError(f"input dimension {input_dim} is not divisible by group size {group_size}")
    groups = input_dim // group_size
    flat = residual.reshape(experts * output_dim, groups, group_size)
    q_result = torch.empty_like(flat, dtype=torch.int8, device="cpu")
    scales_result = torch.empty((flat.shape[0], groups), dtype=torch.bfloat16, device="cpu")
    if hdiag.ndim == 1:
        if hdiag.numel() != input_dim:
            raise RuntimeError(
                f"global Hessian diagonal has {hdiag.numel()} values, expected {input_dim}"
            )
        h = (
            hdiag.to(device=residual.device, dtype=torch.float32)
            .reshape(1, groups, group_size)
            .expand(experts, -1, -1)
        )
    elif hdiag.ndim == 2 and tuple(hdiag.shape) == (experts, input_dim):
        h = hdiag.to(device=residual.device, dtype=torch.float32).reshape(
            experts, groups, group_size
        )
    else:
        raise RuntimeError(
            f"invalid Hessian diagonal geometry {tuple(hdiag.shape)} for "
            f"experts={experts} input_dim={input_dim}"
        )
    for row_start in range(0, flat.shape[0], row_chunk):
        row_end = min(flat.shape[0], row_start + row_chunk)
        values = flat[row_start:row_end]
        expert_ids = torch.arange(
            row_start, row_end, device=residual.device, dtype=torch.int64
        ) // output_dim
        row_hessian = h[expert_ids]
        positive = values.amax(dim=-1) / 3.0
        negative = (-values.amin(dim=-1)) / 4.0
        base = torch.maximum(positive, negative).clamp_min(1e-8)
        candidate_scales = base[..., None] * multipliers.reshape(1, 1, -1)
        expanded = values[..., None] / candidate_scales[:, :, None, :]
        candidate_q = expanded.round().clamp(-4, 3)
        error = values[..., None] - candidate_q * candidate_scales[:, :, None, :]
        losses = (error.square() * row_hessian[..., None]).sum(dim=2)
        best = losses.argmin(dim=-1)
        scales = candidate_scales.gather(2, best[..., None]).squeeze(-1)
        q = (values / scales[..., None]).round().clamp(-4, 3).to(torch.int8)
        q_result[row_start:row_end].copy_(q.cpu())
        scales_result[row_start:row_end].copy_(scales.to(torch.bfloat16).cpu())
    return q_result.reshape(experts, output_dim, input_dim), scales_result.reshape(
        experts, output_dim, groups
    )


def pack_int3_rows(values: np.ndarray) -> bytes:
    if values.dtype != np.int8 or values.ndim != 2 or values.shape[1] % 32:
        raise RuntimeError(f"invalid INT3 pack input {values.shape} {values.dtype}")
    if values.min() < -4 or values.max() > 3:
        raise RuntimeError("INT3 pack input contains values outside -4..3")
    codes = np.bitwise_and(values.astype(np.int16), 7).astype(np.uint32)
    row_groups = codes.reshape(codes.shape[0], -1, 32)
    words = np.zeros((codes.shape[0], row_groups.shape[1], 3), dtype="<u4")
    for index in range(32):
        bit = index * 3
        word = bit // 32
        shift = bit % 32
        words[:, :, word] |= row_groups[:, :, index] << shift
        if shift > 29:
            words[:, :, word + 1] |= row_groups[:, :, index] >> (32 - shift)
    return words.reshape(codes.shape[0], -1).tobytes()


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.pwrite(fd, view[written:], offset + written)
        if count <= 0:
            raise RuntimeError(f"pwrite made no progress at offset {offset + written}")
        written += count


def load_layer_weights(
    model_dir: Path,
    weight_map: dict,
    tensor_names: dict[str, str],
    experts: int,
    hidden: int,
    intermediate: int,
) -> dict[str, torch.Tensor]:
    opened: dict[str, object] = {}

    def get_tensor(name: str) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in opened:
            opened[shard] = safe_open(model_dir / shard, framework="pt", device="cpu")
        tensor = opened[shard].get_tensor(name)
        return tensor

    if "gate_up" in tensor_names:
        fused = get_tensor(tensor_names["gate_up"])
        if tuple(fused.shape) != (experts, 2 * intermediate, hidden):
            raise RuntimeError(f"unexpected fused gate/up shape {tuple(fused.shape)}")
        gate = fused[:, :intermediate, :]
        up = fused[:, intermediate:, :]
    else:
        gate = get_tensor(tensor_names["gate"])
        up = get_tensor(tensor_names["up"])
    down = get_tensor(tensor_names["down"])
    expected = {
        "gate": (experts, intermediate, hidden),
        "up": (experts, intermediate, hidden),
        "down": (experts, hidden, intermediate),
    }
    actual = {"gate": gate, "up": up, "down": down}
    for name, tensor in actual.items():
        if tuple(tensor.shape) != expected[name]:
            raise RuntimeError(f"unexpected {name} shape {tuple(tensor.shape)}, expected {expected[name]}")
    return actual


def strongest_routes(capture: CaptureLayer) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Resolve the strongest measured route for every expert without tail sampling."""
    route_ids = capture.topk_ids.reshape(-1)
    route_weights = capture.topk_weights.reshape(-1).abs()
    experts = capture.expert_count
    best_weights = torch.full((experts,), -float("inf"), dtype=torch.float32)
    best_weights.scatter_reduce_(
        0, route_ids, route_weights, reduce="amax", include_self=True
    )
    positions = torch.arange(route_ids.numel(), dtype=torch.int64)
    sentinel = route_ids.numel()
    candidates = torch.where(
        route_weights == best_weights[route_ids],
        positions,
        torch.full_like(positions, sentinel),
    )
    best_positions = torch.full((experts,), sentinel, dtype=torch.int64)
    best_positions.scatter_reduce_(
        0, route_ids, candidates, reduce="amin", include_self=True
    )
    missing = torch.nonzero(best_positions == sentinel).flatten().tolist()
    return best_positions, best_weights, missing


def prepare_proxy_samples(
    capture: CaptureLayer,
    projection: str,
    source_weights: dict[str, torch.Tensor],
    activation: str,
    device: torch.device,
) -> ProxySamples:
    """Build an expert-balanced held-out set, completing only unseen experts."""
    best_positions, best_weights, missing = strongest_routes(capture)
    present_mask = torch.isfinite(best_weights)
    present_weights = best_weights[present_mask]
    if present_weights.numel() == 0:
        raise RuntimeError(
            f"held-out capture layer {capture.model_layer} contains no expert routes"
        )
    counterfactual_importance = (
        float(present_weights.median().item()) if missing else None
    )
    input_dim = (
        capture.expert_input_size
        if projection in ("gate", "up")
        else capture.intermediate_size
    )
    inputs = torch.empty(
        (capture.expert_count, input_dim), dtype=torch.bfloat16, device="cpu"
    )
    importance = best_weights.clone()
    present = torch.nonzero(present_mask).flatten()
    present_positions = best_positions[present]
    if projection in ("gate", "up"):
        token_rows = torch.div(
            present_positions, capture.topk, rounding_mode="floor"
        )
        inputs[present] = capture.expert_inputs[token_rows]
    else:
        inputs[present] = capture.down_inputs.reshape(
            -1, capture.intermediate_size
        )[present_positions]
    if missing:
        hidden = capture.expert_inputs[counterfactual_rows(capture, missing)]
        if projection in ("gate", "up"):
            inputs[missing] = hidden
        else:
            counterfactual = counterfactual_down_inputs(
                source_weights,
                missing,
                hidden.unsqueeze(1),
                activation,
                device,
            )
            inputs[missing] = counterfactual[:, 0].cpu()
            del counterfactual
        importance[missing] = counterfactual_importance
    if not torch.isfinite(inputs.float()).all() or not torch.isfinite(importance).all():
        raise RuntimeError(
            f"held-out proxy sample construction produced non-finite values for layer "
            f"{capture.model_layer} projection {projection}"
        )
    return ProxySamples(
        inputs=inputs,
        importance=importance,
        counterfactual_experts=missing,
        counterfactual_importance=counterfactual_importance,
    )


def proxy_loss(
    weights: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    inverse_scales: torch.Tensor,
    placement: list[tuple[int, int]],
    samples: ProxySamples,
    calibration_hdiag: torch.Tensor,
    group_size: int,
    multipliers: torch.Tensor,
    device: torch.device,
) -> float:
    total = 0.0
    elements = 0
    inputs = samples.inputs
    route_weights = samples.importance
    for expert in range(weights.shape[0]):
        x = inputs[expert : expert + 1].to(device=device, dtype=torch.float32)
        importance = route_weights[expert].to(device=device, dtype=torch.float32).abs()
        correction = correction_batch(
            left, right, inverse_scales, placement, expert, expert + 1
        )
        residual = weights[expert : expert + 1] - correction
        expert_hdiag = (
            calibration_hdiag
            if calibration_hdiag.ndim == 1
            else calibration_hdiag[expert : expert + 1]
        )
        q, scales = diagonal_hessian_quantize(
            residual,
            expert_hdiag,
            group_size,
            multipliers,
        )
        q_device = q.to(device=device, dtype=torch.float32)
        scale_device = scales.to(device=device, dtype=torch.float32)
        dequant = (q_device.reshape(1, residual.shape[1], -1, group_size) * scale_device[..., None]).reshape_as(residual)
        reconstructed = dequant + correction
        error = torch.matmul(x, (weights[expert] - reconstructed[0]).T) * importance
        total += float(error.square().sum().item())
        elements += error.numel()
    return total / max(elements, 1)


def write_projection(
    fd: int,
    layer_manifest: dict,
    projection: str,
    weights_cpu: torch.Tensor,
    source_weights: dict[str, torch.Tensor],
    calibration: CaptureLayer,
    heldout: CaptureLayer,
    activation: str,
    group_size: int,
    rank: int,
    exponents: list[float],
    scale_multipliers: list[float],
    sketch_seed: int,
    sketch_iterations: int,
    clustering_seed: int,
    expert_batch: int,
    hessian_scope: str,
    device: torch.device,
) -> None:
    descriptor = layer_manifest[projection]
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    means, calibration_counterfactual = activation_means(
        calibration, projection, source_weights, activation, device
    )
    proxy_samples = prepare_proxy_samples(
        heldout, projection, source_weights, activation, device
    )
    descriptor["counterfactual_calibration_experts"] = calibration_counterfactual
    descriptor["counterfactual_heldout_experts"] = proxy_samples.counterfactual_experts
    descriptor["counterfactual_heldout_importance"] = (
        proxy_samples.counterfactual_importance
    )
    if calibration_counterfactual or proxy_samples.counterfactual_experts:
        print(
            f"[tileq] layer={layer_manifest['model_layer']} projection={projection} "
            f"counterfactual_calibration={calibration_counterfactual} "
            f"counterfactual_heldout={proxy_samples.counterfactual_experts} "
            f"counterfactual_importance={proxy_samples.counterfactual_importance}",
            flush=True,
        )
    multipliers = torch.tensor(scale_multipliers, device=device, dtype=torch.float32)
    if hessian_scope == "expert":
        hdiag, hessian_counterfactual = expert_hessian_diagonal(
            calibration,
            projection,
            source_weights,
            activation,
            device,
        )
    elif hessian_scope == "global":
        hdiag = global_hessian_diagonal(calibration, projection)
        hessian_counterfactual = []
    else:
        raise RuntimeError(f"unsupported Hessian scope {hessian_scope!r}")
    descriptor["hessian_scope"] = hessian_scope
    descriptor["counterfactual_hessian_experts"] = hessian_counterfactual
    best = None
    for exponent in exponents:
        left, right, inverse_scales, placement = tileq_factors(
            weights,
            means,
            exponent,
            rank,
            descriptor["grid_rows"],
            descriptor["grid_cols"],
            sketch_seed,
            sketch_iterations,
            clustering_seed,
        )
        loss = proxy_loss(
            weights,
            left,
            right,
            inverse_scales,
            placement,
            proxy_samples,
            hdiag,
            group_size,
            multipliers,
            device,
        )
        print(
            f"[tileq] layer={layer_manifest['model_layer']} projection={projection} "
            f"scale_exponent={exponent} heldout_proxy_mse={loss:.9e}",
            flush=True,
        )
        if best is None or loss < best[0]:
            best = (loss, exponent, left, right, inverse_scales, placement)
        else:
            del left, right, inverse_scales
        torch.cuda.empty_cache()
    assert best is not None
    loss, exponent, left, right, inverse_scales, placement = best
    descriptor["selected_scale_exponent"] = float(exponent)
    descriptor["heldout_weighted_mse"] = float(loss)
    tile_bytes = b"".join(struct.pack("<HH", row, col) for row, col in placement)
    pwrite_all(fd, tile_bytes, PAYLOAD_OFFSET + descriptor["expert_tiles"]["offset"])
    pwrite_all(
        fd,
        bf16_bytes(inverse_scales),
        PAYLOAD_OFFSET + descriptor["expert_inverse_scales_bf16"]["offset"],
    )
    pwrite_all(fd, bf16_bytes(left), PAYLOAD_OFFSET + descriptor["left_factors_bf16"]["offset"])
    pwrite_all(fd, bf16_bytes(right), PAYLOAD_OFFSET + descriptor["right_factors_bf16"]["offset"])

    experts = weights.shape[0]
    gate_packed_bytes = layer_manifest["per_expert_w13_packed"] // 2
    gate_scale_bytes = layer_manifest["per_expert_w13_scales"] // 2
    for start in range(0, experts, expert_batch):
        end = min(experts, start + expert_batch)
        correction = correction_batch(left, right, inverse_scales, placement, start, end)
        residual = weights[start:end] - correction
        batch_hdiag = hdiag if hdiag.ndim == 1 else hdiag[start:end]
        q, scales = diagonal_hessian_quantize(
            residual, batch_hdiag, group_size, multipliers
        )
        for local, expert in enumerate(range(start, end)):
            packed = pack_int3_rows(q[local].numpy())
            scale_bytes = bf16_bytes(scales[local])
            if projection == "gate":
                packed_offset = layer_manifest["w13_packed"]["offset"] + expert * layer_manifest["per_expert_w13_packed"]
                scales_offset = layer_manifest["w13_scales"]["offset"] + expert * layer_manifest["per_expert_w13_scales"]
            elif projection == "up":
                packed_offset = (
                    layer_manifest["w13_packed"]["offset"]
                    + expert * layer_manifest["per_expert_w13_packed"]
                    + gate_packed_bytes
                )
                scales_offset = (
                    layer_manifest["w13_scales"]["offset"]
                    + expert * layer_manifest["per_expert_w13_scales"]
                    + gate_scale_bytes
                )
            else:
                packed_offset = layer_manifest["w2_packed"]["offset"] + expert * layer_manifest["per_expert_w2_packed"]
                scales_offset = layer_manifest["w2_scales"]["offset"] + expert * layer_manifest["per_expert_w2_scales"]
            pwrite_all(fd, packed, PAYLOAD_OFFSET + packed_offset)
            pwrite_all(fd, scale_bytes, PAYLOAD_OFFSET + scales_offset)
        print(
            f"[tileq] layer={layer_manifest['model_layer']} projection={projection} "
            f"quantized_experts={end}/{experts}",
            flush=True,
        )
        del correction, residual, q, scales
        torch.cuda.empty_cache()
    del weights, means, proxy_samples, left, right, inverse_scales, hdiag
    torch.cuda.empty_cache()


def publish_manifest(fd: int, manifest: dict) -> None:
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if HEADER_BYTES + len(manifest_bytes) > PAYLOAD_OFFSET:
        raise RuntimeError(
            f"TileQ manifest is {len(manifest_bytes)} bytes and exceeds reserved metadata area"
        )
    manifest_sha = hashlib.sha256(manifest_bytes).digest()
    header = struct.pack(
        "<4sIQQQ32s",
        MAGIC,
        VERSION,
        len(manifest_bytes),
        PAYLOAD_OFFSET,
        manifest["payload_bytes"],
        manifest_sha,
    )
    if len(header) != HEADER_BYTES:
        raise RuntimeError(f"internal TileQ header length is {len(header)}, expected {HEADER_BYTES}")
    pwrite_all(fd, header, 0)
    pwrite_all(fd, manifest_bytes, HEADER_BYTES)


def command_inspect_capture(args: argparse.Namespace) -> None:
    capture_dirs = [args.capture_dir.resolve(), *[path.resolve() for path in args.extra]]
    metadata = [capture_metadata(capture_dir) for capture_dir in capture_dirs]
    print(json.dumps({"captures": metadata}, indent=2, sort_keys=True))
    for path in sorted(capture_dirs[0].glob("layer_*.ktc")):
        layer = combine_capture_layers(
            [read_capture_layer(capture_dir / path.name) for capture_dir in capture_dirs]
        )
        counts = torch.bincount(layer.topk_ids.reshape(-1), minlength=layer.expert_count)
        routed = counts[counts > 0]
        print(
            f"{path.name}: tokens={layer.expert_inputs.shape[0]} input={layer.expert_input_size} "
            f"topk={layer.topk} intermediate={layer.intermediate_size} experts={layer.expert_count} "
            f"route_zero={int((counts == 0).sum())} "
            f"route_nonzero_min={int(routed.min()) if routed.numel() else 0} "
            f"route_max={int(counts.max())}"
        )


def command_prepare_corpus(args: argparse.Namespace) -> None:
    sources = [path.resolve() for path in args.sources]
    if not sources:
        raise RuntimeError("at least one corpus source is required")
    if len(set(sources)) != len(sources):
        raise RuntimeError("corpus source paths must be unique")
    output = args.output_dir.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite corpus directory or temporary directory: {output}")
    temporary.mkdir(parents=True)
    entries = []
    try:
        for index, source in enumerate(sources):
            raw = source.read_bytes()
            if not raw:
                raise RuntimeError(f"corpus source is empty: {source}")
            midpoint = len(raw) // 2
            boundary = raw.find(b"\n", midpoint)
            if boundary < 0:
                boundary = midpoint
            else:
                boundary += 1
            if args.half == "head":
                start, end = 0, boundary
            else:
                start, end = boundary, len(raw)
            payload = raw[start:end]
            if not payload:
                raise RuntimeError(f"selected {args.half} half is empty: {source}")
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"selected corpus slice is not UTF-8: {source}: {exc}") from exc
            destination = temporary / f"{index:02d}_{source.name}"
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            entries.append(
                {
                    "index": index,
                    "source": str(source),
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_bytes": len(raw),
                    "slice": args.half,
                    "slice_start": start,
                    "slice_end": end,
                    "slice_bytes": len(payload),
                    "slice_sha256": hashlib.sha256(payload).hexdigest(),
                    "file": destination.name,
                }
            )
        binding = {
            "schema_version": 1,
            "format": "Krasis TileQ calibration corpus",
            "half": args.half,
            "entries": entries,
        }
        canonical = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
        binding["corpus_sha256"] = hashlib.sha256(canonical).hexdigest()
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(binding, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        # Preserve a failed transactional directory for diagnosis without ever
        # publishing it under the requested canonical name.
        raise
    print(json.dumps(binding, indent=2, sort_keys=True))


def command_capture_corpus(args: argparse.Namespace) -> None:
    corpus_dir = args.corpus_dir.resolve()
    manifest = load_json(corpus_dir / "manifest.json")
    expected_digest = manifest.pop("corpus_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual_digest = hashlib.sha256(canonical).hexdigest()
    if expected_digest != actual_digest:
        raise RuntimeError(
            f"corpus manifest binding mismatch: expected={expected_digest} actual={actual_digest}"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("corpus manifest has no entries")
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    for expected_index, entry in enumerate(entries):
        if entry.get("index") != expected_index:
            raise RuntimeError(
                f"corpus entry ordering mismatch at {expected_index}: {entry.get('index')}"
            )
        source = Path(entry["source"])
        if sha256_file(source) != entry["source_sha256"]:
            raise RuntimeError(f"corpus source changed after preparation: {source}")
        prompt_path = corpus_dir / entry["file"]
        prompt_bytes = prompt_path.read_bytes()
        if len(prompt_bytes) != entry["slice_bytes"]:
            raise RuntimeError(f"corpus slice length mismatch: {prompt_path}")
        if hashlib.sha256(prompt_bytes).hexdigest() != entry["slice_sha256"]:
            raise RuntimeError(f"corpus slice SHA-256 mismatch: {prompt_path}")
        prompt = prompt_bytes.decode("utf-8")
        body = json.dumps(
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response_body = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"capture request failed for {prompt_path}: {exc}") from exc
        try:
            result = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"capture response is not JSON for {prompt_path}: {exc}") from exc
        usage = result.get("usage")
        if not isinstance(usage, dict) or int(usage.get("prompt_tokens", 0)) <= 0:
            raise RuntimeError(f"capture response omits valid token usage for {prompt_path}")
        print(
            json.dumps(
                {
                    "index": expected_index,
                    "file": entry["file"],
                    "slice_sha256": entry["slice_sha256"],
                    "usage": usage,
                    "response_id": result.get("id"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "corpus_sha256": actual_digest,
                "requests": len(entries),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def command_plan(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    root, config, weight_map = text_config(model_dir)
    layers = int(config["num_hidden_layers"])
    discovered = discover_expert_tensors(weight_map, layers)
    selected_layers = list(range(layers)) if args.layers is None else list(range(min(args.layers, layers)))
    manifest, payload = build_layout(
        model_dir.name,
        str(config.get("model_type", root.get("model_type", "unknown"))),
        int(config["hidden_size"]),
        int(config["moe_intermediate_size"]),
        int(config["num_experts"]),
        selected_layers,
        args.group_size,
        args.rank,
    )
    relevant_shards = sorted(
        {weight_map[name] for layer in discovered[: len(selected_layers)] for name in layer.values()}
    )
    print(
        json.dumps(
            {
                "model": model_dir.name,
                "layers": len(selected_layers),
                "experts": manifest["routed_experts"],
                "hidden_size": manifest["hidden_size"],
                "intermediate_size": manifest["intermediate_size"],
                "rank": args.rank,
                "grid": [manifest["layers"][0]["gate"]["grid_rows"], manifest["layers"][0]["gate"]["grid_cols"]],
                "payload_bytes": payload,
                "payload_gib": payload / 1024**3,
                "metadata_reserve_bytes": PAYLOAD_OFFSET,
                "relevant_shards": relevant_shards,
            },
            indent=2,
        )
    )


def command_counterfactual_test(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TileQ counterfactual contract test requires CUDA")
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    capture = CaptureLayer(
        model_layer=7,
        expert_input_size=4,
        intermediate_size=3,
        topk=2,
        expert_count=5,
        expert_inputs=torch.tensor(
            [[1.0, -2.0, 3.0, -4.0], [2.0, 3.0, -1.0, 0.5], [-3.0, 1.0, 2.0, 4.0]],
            dtype=torch.bfloat16,
        ),
        topk_ids=torch.tensor([[0, 1], [0, 1], [1, 2]], dtype=torch.int64),
        topk_weights=torch.tensor([[0.7, 0.2], [0.4, 0.9], [0.3, 0.6]], dtype=torch.float32),
        down_inputs=torch.arange(18, dtype=torch.float32).reshape(3, 2, 3).to(torch.bfloat16),
    )
    source_weights = {
        "gate": torch.arange(60, dtype=torch.float32).reshape(5, 3, 4).to(torch.bfloat16) / 17,
        "up": (torch.arange(60, dtype=torch.float32).reshape(5, 3, 4) - 29).to(torch.bfloat16) / 19,
        "down": torch.zeros((5, 4, 3), dtype=torch.bfloat16),
    }
    gate_means, gate_missing = activation_means(
        capture, "gate", source_weights, "silu", device
    )
    down_means, down_missing = activation_means(
        capture, "down", source_weights, "silu", device
    )
    if gate_missing != [3, 4] or down_missing != [3, 4]:
        raise RuntimeError(
            f"counterfactual activation missing-set mismatch: gate={gate_missing} down={down_missing}"
        )
    expected_gate = counterfactual_hidden_sample(capture).float().abs().mean(dim=0)
    if not torch.allclose(
        gate_means[[3, 4]].cpu(), expected_gate.expand(2, -1), atol=1e-6, rtol=0
    ):
        raise RuntimeError("counterfactual gate activation mean mismatch")
    if (
        not torch.isfinite(down_means[[3, 4]]).all()
        or torch.any(torch.count_nonzero(down_means[[3, 4]], dim=1) == 0)
    ):
        raise RuntimeError("counterfactual down activation mean is invalid")
    gate_proxy = prepare_proxy_samples(
        capture, "gate", source_weights, "silu", device
    )
    down_proxy = prepare_proxy_samples(
        capture, "down", source_weights, "silu", device
    )
    if (
        gate_proxy.counterfactual_experts != [3, 4]
        or down_proxy.counterfactual_experts != [3, 4]
    ):
        raise RuntimeError("counterfactual held-out missing-set mismatch")
    if gate_proxy.counterfactual_importance is None or not math.isclose(
        gate_proxy.counterfactual_importance, 0.7, rel_tol=0, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"counterfactual held-out importance mismatch: {gate_proxy.counterfactual_importance}"
        )
    expected_hidden = capture.expert_inputs[counterfactual_rows(capture, [3, 4])]
    if not torch.equal(gate_proxy.inputs[[3, 4]], expected_hidden):
        raise RuntimeError("counterfactual held-out hidden row mismatch")
    expected_down = counterfactual_down_inputs(
        source_weights,
        [3, 4],
        expected_hidden.reshape(2, 1, -1),
        "silu",
        device,
    )[:, 0].cpu()
    if not torch.equal(down_proxy.inputs[[3, 4]], expected_down):
        raise RuntimeError("counterfactual held-out down input mismatch")
    gate_hdiag, gate_hmissing = expert_hessian_diagonal(
        capture,
        "gate",
        source_weights,
        "silu",
        device,
        row_chunk=2,
    )
    down_hdiag, down_hmissing = expert_hessian_diagonal(
        capture,
        "down",
        source_weights,
        "silu",
        device,
        row_chunk=2,
    )
    if gate_hmissing != [3, 4] or down_hmissing != [3, 4]:
        raise RuntimeError(
            "counterfactual Hessian missing-set mismatch: "
            f"gate={gate_hmissing} down={down_hmissing}"
        )
    if tuple(gate_hdiag.shape) != (5, 4) or tuple(down_hdiag.shape) != (5, 3):
        raise RuntimeError(
            "counterfactual Hessian geometry mismatch: "
            f"gate={tuple(gate_hdiag.shape)} down={tuple(down_hdiag.shape)}"
        )
    expected_gate_hdiag = expected_hidden.float().square().clamp_min(1e-8)
    expected_down_hdiag = expected_down.float().square().clamp_min(1e-8)
    if not torch.allclose(
        gate_hdiag[[3, 4]].cpu(), expected_gate_hdiag, atol=1e-6, rtol=0
    ):
        raise RuntimeError("counterfactual gate Hessian diagonal mismatch")
    if not torch.allclose(
        down_hdiag[[3, 4]].cpu(), expected_down_hdiag, atol=1e-6, rtol=0
    ):
        raise RuntimeError("counterfactual down Hessian diagonal mismatch")
    if not torch.isfinite(gate_hdiag).all() or not torch.isfinite(down_hdiag).all():
        raise RuntimeError("counterfactual Hessian contains non-finite values")
    global_gate_hdiag = global_hessian_diagonal(capture, "gate", row_chunk=2)
    global_down_hdiag = global_hessian_diagonal(capture, "down", row_chunk=2)
    if not torch.allclose(
        global_gate_hdiag,
        capture.expert_inputs.float().square().mean(dim=0),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError("global gate Hessian diagonal mismatch")
    if not torch.allclose(
        global_down_hdiag,
        capture.down_inputs.reshape(-1, capture.intermediate_size)
        .float()
        .square()
        .mean(dim=0),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError("global down Hessian diagonal mismatch")
    try:
        validate_counterfactual_activation({"hidden_act": "gelu"})
    except RuntimeError:
        pass
    else:
        raise RuntimeError("unsupported counterfactual activation did not fail closed")
    print(
        json.dumps(
            {
                "status": "pass",
                "model_layer": capture.model_layer,
                "counterfactual_experts": gate_missing,
                "counterfactual_importance": gate_proxy.counterfactual_importance,
            },
            sort_keys=True,
        )
    )


def command_build(args: argparse.Namespace) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("TileQ build requires CUDA")
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    model_dir = args.model_dir.resolve()
    calibration_dirs = [
        args.calibration_capture.resolve(),
        *[path.resolve() for path in args.calibration_extra],
    ]
    heldout_dirs = [
        args.heldout_capture.resolve(),
        *[path.resolve() for path in args.heldout_extra],
    ]
    calibration_metas = [capture_metadata(path) for path in calibration_dirs]
    heldout_metas = [capture_metadata(path) for path in heldout_dirs]
    calibration_hashes = [meta["calibration_sha256"].lower() for meta in calibration_metas]
    heldout_hashes = [meta["calibration_sha256"].lower() for meta in heldout_metas]
    if set(calibration_hashes) & set(heldout_hashes):
        raise RuntimeError("calibration and held-out captures contain the same corpus hash")
    root, config, weight_map = text_config(model_dir)
    layer_count = int(config["num_hidden_layers"])
    experts = int(config["num_experts"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    discovered = discover_expert_tensors(weight_map, layer_count)
    if args.layers is not None and args.layer_list is not None:
        raise RuntimeError("--layers and --layer-list are mutually exclusive")
    if args.layer_list is not None:
        try:
            selected_layers = [int(value) for value in args.layer_list.split(",")]
        except ValueError as exc:
            raise RuntimeError("--layer-list must be comma-separated integer layer IDs") from exc
        if not selected_layers or len(selected_layers) != len(set(selected_layers)):
            raise RuntimeError("--layer-list must contain distinct layer IDs")
        if any(layer < 0 or layer >= layer_count for layer in selected_layers):
            raise RuntimeError(
                f"--layer-list contains a layer outside 0..{layer_count - 1}: {selected_layers}"
            )
        selected_layers.sort()
    else:
        selected_count = layer_count if args.layers is None else min(args.layers, layer_count)
        selected_layers = list(range(selected_count))
    selected_count = len(selected_layers)
    expert_biases = [
        name
        for name in weight_map
        if any(f"layers.{layer}.mlp.experts." in name for layer in selected_layers)
        and name.endswith(".bias")
    ]
    if expert_biases:
        raise RuntimeError(
            "TileQ counterfactual calibration does not support expert biases; "
            f"found {expert_biases[:8]} (total {len(expert_biases)})"
        )
    activation = validate_counterfactual_activation(config)
    manifest, payload_bytes = build_layout(
        model_dir.name,
        str(config.get("model_type", root.get("model_type", "unknown"))),
        hidden,
        intermediate,
        experts,
        selected_layers,
        args.group_size,
        args.rank,
    )
    manifest["residual_quantizer"] = (
        "tileq_s_int3_per_expert_diagonal_hessian_scale_search_v2"
        if args.hessian_scope == "expert"
        else "tileq_s_int3_global_diagonal_hessian_scale_search_v2"
    )
    exponents = [float(value) for value in args.scale_exponents.split(",")]
    multipliers = [float(value) for value in args.scale_multipliers.split(",")]
    if not exponents or any(not math.isfinite(value) for value in exponents):
        raise RuntimeError("scale exponent candidates must be finite")
    if not multipliers or any(not math.isfinite(value) or value <= 0 for value in multipliers):
        raise RuntimeError("scale multipliers must be finite and positive")
    manifest.update(
        {
            "source_config_sha256": combined_sha256(
                [model_dir / "config.json", model_dir / "model.safetensors.index.json"]
            ),
            "calibration_sha256": capture_binding_sha256(calibration_hashes),
            "calibration_corpora_sha256": calibration_hashes,
            "heldout_sha256": capture_binding_sha256(heldout_hashes),
            "heldout_corpora_sha256": heldout_hashes,
            "scale_exponent_candidates": exponents,
            "sketch_seed": args.sketch_seed,
            "sketch_iterations": args.sketch_iterations,
            "clustering_seed": args.clustering_seed,
            "scale_search_multipliers": multipliers,
            "deterministic_build": {
                "torch_deterministic_algorithms": True,
                "float32_matmul_precision": "highest",
                "allow_tf32": False,
                "cublas_workspace_config": _DETERMINISTIC_CUBLAS_WORKSPACE,
            },
            "counterfactual_calibration": {
                "schema_version": 1,
                "activation": activation,
                "hidden_state_sampling": "expert_count_evenly_spaced_real_capture_rows_v1",
                "heldout_route_selection": "strongest_measured_route_per_expert_v1",
                "unseen_heldout_importance": "median_strongest_measured_route_weight_v1",
            },
        }
    )
    relevant_shards = sorted(
        {
            model_dir / weight_map[name]
            for model_layer in selected_layers
            for name in discovered[model_layer].values()
        }
    )
    print(f"[tileq] hashing {len(relevant_shards)} source shard(s)", flush=True)
    manifest["source_routed_sha256"] = combined_sha256(relevant_shards)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite existing TileQ artifact or temporary file: {output}")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    started = time.monotonic()
    try:
        os.ftruncate(fd, PAYLOAD_OFFSET + payload_bytes)
        for sequence, model_layer in enumerate(selected_layers):
            layer_name = f"layer_{model_layer:05}.ktc"
            calibration = combine_capture_layers(
                [read_capture_layer(path / layer_name) for path in calibration_dirs]
            )
            heldout = combine_capture_layers(
                [read_capture_layer(path / layer_name) for path in heldout_dirs]
            )
            for capture, label in ((calibration, "calibration"), (heldout, "heldout")):
                if (
                    capture.expert_input_size != hidden
                    or capture.intermediate_size != intermediate
                    or capture.expert_count != experts
                ):
                    raise RuntimeError(
                        f"{label} layer {model_layer} geometry does not match model: "
                        f"input={capture.expert_input_size}/{hidden} intermediate={capture.intermediate_size}/{intermediate} "
                        f"experts={capture.expert_count}/{experts}"
                    )
            weights = load_layer_weights(
                model_dir,
                weight_map,
                discovered[model_layer],
                experts,
                hidden,
                intermediate,
            )
            for projection in ("gate", "up", "down"):
                write_projection(
                    fd,
                    manifest["layers"][sequence],
                    projection,
                    weights[projection],
                    weights,
                    calibration,
                    heldout,
                    activation,
                    args.group_size,
                    args.rank,
                    exponents,
                    multipliers,
                    args.sketch_seed + model_layer * 17 + {"gate": 0, "up": 1, "down": 2}[projection],
                    args.sketch_iterations,
                    args.clustering_seed + model_layer * 17 + {"gate": 0, "up": 1, "down": 2}[projection],
                    args.expert_batch,
                    args.hessian_scope,
                    device,
                )
            del weights, calibration, heldout
            torch.cuda.empty_cache()
            print(
                f"[tileq] completed layer {sequence + 1}/{selected_count} model_layer={model_layer} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
        publish_manifest(fd, manifest)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.replace(temporary, output)
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    actual_bytes = output.stat().st_size
    total_weights = selected_count * experts * 3 * hidden * intermediate
    effective_bpw = actual_bytes * 8 / total_weights
    print(
        f"[tileq] published {output} bytes={actual_bytes} effective_bpw={effective_bpw:.6f} "
        f"elapsed_s={time.monotonic() - started:.1f}",
        flush=True,
    )


def read_artifact_manifest(path: Path) -> tuple[dict, int, int]:
    with path.open("rb") as handle:
        header = handle.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            raise RuntimeError(f"TileQ artifact {path} is truncated")
        magic, version, manifest_len, payload_offset, payload_len, expected_sha = struct.unpack(
            "<4sIQQQ32s", header
        )
        if magic != MAGIC or version != VERSION:
            raise RuntimeError(f"TileQ artifact {path} has incompatible magic/version")
        manifest_bytes = handle.read(manifest_len)
        if hashlib.sha256(manifest_bytes).digest() != expected_sha:
            raise RuntimeError(f"TileQ artifact {path} manifest SHA-256 mismatch")
        manifest = json.loads(manifest_bytes)
    if payload_offset + payload_len != path.stat().st_size or manifest["payload_bytes"] != payload_len:
        raise RuntimeError(f"TileQ artifact {path} range/length mismatch")
    return manifest, payload_offset, payload_len


def command_verify(args: argparse.Namespace) -> None:
    path = args.artifact.resolve()
    manifest, payload_offset, payload_len = read_artifact_manifest(path)
    ranges = []
    for layer in manifest["layers"]:
        for key in ("w13_packed", "w13_scales", "w2_packed", "w2_scales"):
            ranges.append((f"layer{layer['model_layer']}.{key}", layer[key]))
        for projection in ("gate", "up", "down"):
            for key in (
                "expert_tiles",
                "expert_inverse_scales_bf16",
                "left_factors_bf16",
                "right_factors_bf16",
            ):
                ranges.append((f"layer{layer['model_layer']}.{projection}.{key}", layer[projection][key]))
    for label, value in ranges:
        start = int(value["offset"])
        length = int(value["len"])
        if start < 0 or length <= 0 or start + length > payload_len:
            raise RuntimeError(f"{label} range {start}+{length} exceeds payload {payload_len}")
    total_weights = (
        manifest["routed_layers"]
        * manifest["routed_experts"]
        * 3
        * manifest["hidden_size"]
        * manifest["intermediate_size"]
    )
    print(
        json.dumps(
            {
                "artifact": str(path),
                "bytes": path.stat().st_size,
                "payload_bytes": payload_len,
                "effective_bpw": path.stat().st_size * 8 / total_weights,
                "model_id": manifest["model_id"],
                "layers": manifest["routed_layers"],
                "experts": manifest["routed_experts"],
                "residual_quantizer": manifest["residual_quantizer"],
                "source_routed_sha256": manifest["source_routed_sha256"],
            },
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-capture", help="validate and summarize KTC1 capture files")
    inspect.add_argument("capture_dir", type=Path)
    inspect.add_argument("--extra", action="append", default=[], type=Path)
    inspect.set_defaults(func=command_inspect_capture)

    prepare = sub.add_parser(
        "prepare-corpus",
        help="build a source-bound head/tail corpus from real UTF-8 documents",
    )
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("sources", nargs="+", type=Path)
    prepare.add_argument("--half", choices=("head", "tail"), required=True)
    prepare.set_defaults(func=command_prepare_corpus)

    capture = sub.add_parser(
        "capture-corpus",
        help="submit every verified corpus slice to an armed Krasis server",
    )
    capture.add_argument("corpus_dir", type=Path)
    capture.add_argument("--base-url", default="http://127.0.0.1:18216")
    capture.add_argument(
        "--model",
        required=True,
        help="Exact model identifier exposed by the calibration server",
    )
    capture.add_argument("--timeout", type=float, default=1800.0)
    capture.set_defaults(func=command_capture_corpus)

    plan = sub.add_parser("plan", help="print exact TileQ artifact geometry without building")
    plan.add_argument("model_dir", type=Path)
    plan.add_argument("--layers", type=int)
    plan.add_argument("--group-size", type=int, default=128)
    plan.add_argument("--rank", type=int, default=32)
    plan.set_defaults(func=command_plan)

    counterfactual_test = sub.add_parser(
        "counterfactual-test",
        help="exercise rare-expert calibration and held-out completion contracts",
    )
    counterfactual_test.add_argument("--cuda-device", type=int, default=0)
    counterfactual_test.set_defaults(func=command_counterfactual_test)

    build = sub.add_parser("build", help="build a source-bound KTQ1 cache")
    build.add_argument("model_dir", type=Path)
    build.add_argument("calibration_capture", type=Path)
    build.add_argument("heldout_capture", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--calibration-extra", action="append", default=[], type=Path)
    build.add_argument("--heldout-extra", action="append", default=[], type=Path)
    build.add_argument("--layers", type=int)
    build.add_argument(
        "--layer-list",
        help="comma-separated model layer IDs for representative pilot artifacts",
    )
    build.add_argument("--cuda-device", type=int, default=0)
    build.add_argument("--group-size", type=int, default=128)
    build.add_argument("--rank", type=int, default=32)
    build.add_argument("--scale-exponents", default="0.0,0.25,0.5,0.75,1.0")
    build.add_argument("--scale-multipliers", default="0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20")
    build.add_argument("--sketch-seed", type=int, default=314159)
    build.add_argument("--sketch-iterations", type=int, default=2)
    build.add_argument("--clustering-seed", type=int, default=271828)
    build.add_argument("--expert-batch", type=int, default=2)
    build.add_argument(
        "--hessian-scope",
        choices=("expert", "global"),
        default="global",
        help="calibration Hessian geometry; global reproduces the v1 control without held-out leakage",
    )
    build.set_defaults(func=command_build)

    verify = sub.add_parser("verify", help="validate KTQ1 metadata and ranges")
    verify.add_argument("artifact", type=Path)
    verify.set_defaults(func=command_verify)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
