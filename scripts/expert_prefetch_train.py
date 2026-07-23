#!/usr/bin/env python3
"""Train offline expert-prefetch predictors from Krasis trace datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import struct
import sys
from dataclasses import dataclass
from typing import Any


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print(
        "expert_prefetch_train.py must be run through ./dev expert-prefetch-train",
        file=sys.stderr,
    )
    sys.exit(2)


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - environment gate
    print(f"torch is required for expert-prefetch training: {exc}", file=sys.stderr)
    sys.exit(2)


DATASET_SCHEMAS = {
    "krasis_expert_prefetch_dataset_v1",
    "krasis_expert_prefetch_dataset_v2",
}
DEFAULT_FEATURE_SET = "route_current_hcs"
FEATURE_SETS = {
    "route_current_hcs",
    "route_current",
    "route_history",
}


@dataclass
class LoadedDataset:
    meta: dict[str, Any]
    feature_set: str
    features: torch.Tensor
    targets: torch.Tensor
    label_bytes: torch.Tensor
    request_seq: torch.Tensor
    sample_id: torch.Tensor
    layer: torch.Tensor
    sources: list[str]


def parse_hidden_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"hidden size must be positive, got {value}")
        sizes.append(value)
    return sizes


def parse_sample_counts(raw: str | None) -> list[int | None]:
    if raw is None or not raw.strip():
        return [None]
    counts: list[int | None] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value in {"full", "all"}:
            counts.append(None)
            continue
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"sample count must be positive, got {parsed}")
        counts.append(parsed)
    if not counts:
        raise ValueError("--sample-counts did not contain any counts")
    return counts


def parse_positive_ints(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return []
    values: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"expected positive integer, got {parsed}")
        values.append(parsed)
    return values


def load_one_dataset(path: pathlib.Path, source_index: int, feature_set: str) -> LoadedDataset:
    meta: dict[str, Any] | None = None
    feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    byte_rows: list[list[float]] = []
    request_seq: list[int] = []
    sample_id: list[int] = []
    layers: list[int] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            schema = record.get("schema")
            if schema not in DATASET_SCHEMAS:
                raise ValueError(f"{path}:{line_no}: unsupported schema {schema!r}")
            event = record.get("event")
            if event == "meta":
                meta = record
                continue
            if event != "sample":
                raise ValueError(f"{path}:{line_no}: unknown event {event!r}")
            if meta is None:
                raise ValueError(f"{path}:{line_no}: sample before meta")
            features, targets, label_bytes = encode_sample(record, meta, feature_set)
            if not any(targets):
                continue
            feature_rows.append(features)
            target_rows.append(targets)
            byte_rows.append(label_bytes)
            request_seq.append((source_index << 48) | int(record["request_seq"]))
            sample_id.append(int(record["sample_id"]))
            layers.append(int(record["layer"]))

    if meta is None:
        raise ValueError(f"{path}: missing dataset meta record")
    if not feature_rows:
        raise ValueError(f"{path}: no trainable samples")

    return LoadedDataset(
        meta=meta,
        feature_set=feature_set,
        features=torch.tensor(feature_rows, dtype=torch.float32),
        targets=torch.tensor(target_rows, dtype=torch.float32),
        label_bytes=torch.tensor(byte_rows, dtype=torch.float32),
        request_seq=torch.tensor(request_seq, dtype=torch.long),
        sample_id=torch.tensor(sample_id, dtype=torch.long),
        layer=torch.tensor(layers, dtype=torch.long),
        sources=[str(path)],
    )


def load_datasets(paths: list[pathlib.Path], feature_set: str) -> LoadedDataset:
    if not paths:
        raise ValueError("at least one dataset path is required")

    loaded = [load_one_dataset(path, source_index, feature_set) for source_index, path in enumerate(paths)]
    base = loaded[0]
    for other in loaded[1:]:
        for key in ("schema", "num_layers", "num_experts", "topk", "lookahead_routes"):
            if base.meta.get(key) != other.meta.get(key):
                raise ValueError(
                    f"dataset metadata mismatch for {key}: "
                    f"{base.meta.get(key)!r} != {other.meta.get(key)!r}"
                )

    if len(loaded) == 1:
        return base

    return LoadedDataset(
        meta=base.meta,
        feature_set=feature_set,
        features=torch.cat([item.features for item in loaded], dim=0),
        targets=torch.cat([item.targets for item in loaded], dim=0),
        label_bytes=torch.cat([item.label_bytes for item in loaded], dim=0),
        request_seq=torch.cat([item.request_seq for item in loaded], dim=0),
        sample_id=torch.cat([item.sample_id for item in loaded], dim=0),
        layer=torch.cat([item.layer for item in loaded], dim=0),
        sources=[source for item in loaded for source in item.sources],
    )


def filter_layer_group(data: LoadedDataset, groups: int, group_index: int) -> tuple[LoadedDataset, dict[str, int] | None]:
    if groups <= 1:
        return data, None
    if group_index < 0 or group_index >= groups:
        raise ValueError(f"--layer-group-index must be in [0, {groups}), got {group_index}")

    num_layers = int(data.meta["num_layers"])
    start = (num_layers * group_index) // groups
    end = (num_layers * (group_index + 1)) // groups
    mask = (data.layer >= start) & (data.layer < end)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError(f"layer group {group_index}/{groups} produced no samples")

    filtered = LoadedDataset(
        meta=data.meta,
        feature_set=data.feature_set,
        features=data.features[indices],
        targets=data.targets[indices],
        label_bytes=data.label_bytes[indices],
        request_seq=data.request_seq[indices],
        sample_id=data.sample_id[indices],
        layer=data.layer[indices],
        sources=data.sources,
    )
    return filtered, {
        "groups": groups,
        "group_index": group_index,
        "start_layer": start,
        "end_layer_exclusive": end,
        "samples": int(indices.numel()),
    }


def encode_sample(
    record: dict[str, Any],
    meta: dict[str, Any],
    feature_set: str,
) -> tuple[list[float], list[float], list[float]]:
    num_layers = int(meta["num_layers"])
    num_experts = int(meta["num_experts"])
    lookahead = int(meta["lookahead_routes"])
    output_dim = lookahead * num_experts

    if feature_set == "route_current_hcs":
        features = encode_route_current_hcs(record, meta)
    elif feature_set == "route_current":
        features = encode_route_current(record, meta)
    elif feature_set == "route_history":
        features = encode_route_history(record, meta)
    else:
        raise ValueError(f"unknown feature set {feature_set!r}")

    targets = [0.0] * output_dim
    label_bytes = [0.0] * output_dim
    for label in record.get("future_cold", []):
        delta = int(label["delta"])
        expert = int(label["expert"])
        if 1 <= delta <= lookahead and 0 <= expert < num_experts:
            idx = (delta - 1) * num_experts + expert
            targets[idx] = 1.0
            label_bytes[idx] += float(label.get("bytes", 0))
    return features, targets, label_bytes


def encode_route_current_hcs(record: dict[str, Any], meta: dict[str, Any]) -> list[float]:
    num_layers = int(meta["num_layers"])
    num_experts = int(meta["num_experts"])
    feature_dim = num_layers + num_experts * 2 + 4
    features = [0.0] * feature_dim
    layer = int(record["layer"])
    if 0 <= layer < num_layers:
        features[layer] = 1.0

    expert_base = num_layers
    cold_base = num_layers + num_experts
    for expert in record.get("current_experts", []):
        expert = int(expert)
        if 0 <= expert < num_experts:
            features[expert_base + expert] = 1.0
    for expert in record.get("current_cold_experts", []):
        expert = int(expert)
        if 0 <= expert < num_experts:
            features[cold_base + expert] = 1.0

    scalar_base = num_layers + num_experts * 2
    denom_layers = max(1, num_layers - 1)
    features[scalar_base] = layer / denom_layers
    features[scalar_base + 1] = float(record.get("step", 0)) / 1024.0
    features[scalar_base + 2] = len(record.get("current_experts", [])) / max(1, int(meta["topk"]))
    features[scalar_base + 3] = len(record.get("current_cold_experts", [])) / max(1, int(meta["topk"]))
    return features


def encode_route_current(record: dict[str, Any], meta: dict[str, Any]) -> list[float]:
    num_layers = int(meta["num_layers"])
    num_experts = int(meta["num_experts"])
    scalar_count = 8
    feature_dim = num_layers + num_experts * 2 + scalar_count
    features = [0.0] * feature_dim
    layer = int(record["layer"])
    write_layer_and_scalars(features, record, meta, layer, num_layers, num_experts * 2, scalar_count)

    present_base = num_layers
    weight_base = num_layers + num_experts
    write_weighted_experts(features, present_base, weight_base, num_experts, record.get("current_weighted_experts", []))
    return features


def encode_route_history(record: dict[str, Any], meta: dict[str, Any]) -> list[float]:
    num_layers = int(meta["num_layers"])
    num_experts = int(meta["num_experts"])
    scalar_count = 10
    blocks = 8
    feature_dim = num_layers + num_experts * blocks + scalar_count
    features = [0.0] * feature_dim
    layer = int(record["layer"])
    write_layer_and_scalars(features, record, meta, layer, num_layers, num_experts * blocks, scalar_count)

    base = num_layers
    current_present = base
    current_weight = current_present + num_experts
    prev_present = current_weight + num_experts
    prev_weight = prev_present + num_experts
    recent_count = prev_weight + num_experts
    recent_weight = recent_count + num_experts
    prior_present = recent_weight + num_experts
    prior_weight = prior_present + num_experts

    write_weighted_experts(features, current_present, current_weight, num_experts, record.get("current_weighted_experts", []))

    previous = record.get("previous_same_layer")
    if isinstance(previous, dict):
        write_weighted_experts(features, prev_present, prev_weight, num_experts, previous.get("weighted_experts", []))

    history_tokens = max(1, int(meta.get("history_tokens", 4)))
    for item in record.get("recent_same_layer_counts", []):
        expert = int(item.get("expert", -1))
        if 0 <= expert < num_experts:
            features[recent_count + expert] = min(1.0, float(item.get("count", 0)) / history_tokens)
            features[recent_weight + expert] = float(item.get("weight_sum", 0.0)) / history_tokens

    prior_layers = max(1, int(meta.get("prior_layers", 4)))
    for snapshot in record.get("previous_layers_current_step", []):
        if not isinstance(snapshot, dict):
            continue
        scale = 1.0 / prior_layers
        for item in snapshot.get("weighted_experts", []):
            expert = int(item.get("expert", -1))
            if 0 <= expert < num_experts:
                features[prior_present + expert] = 1.0
                features[prior_weight + expert] += float(item.get("weight", 0.0)) * scale
    return features


def write_weighted_experts(
    features: list[float],
    present_base: int,
    weight_base: int,
    num_experts: int,
    weighted: Any,
) -> None:
    if not isinstance(weighted, list) or not weighted:
        return
    for item in weighted:
        if not isinstance(item, dict):
            continue
        expert = int(item.get("expert", -1))
        if 0 <= expert < num_experts:
            features[present_base + expert] = 1.0
            features[weight_base + expert] = float(item.get("weight", 0.0))


def write_layer_and_scalars(
    features: list[float],
    record: dict[str, Any],
    meta: dict[str, Any],
    layer: int,
    num_layers: int,
    expert_block_width: int,
    scalar_count: int,
) -> None:
    if 0 <= layer < num_layers:
        features[layer] = 1.0
    scalar_base = num_layers + expert_block_width
    denom_layers = max(1, num_layers - 1)
    topk = max(1, int(meta["topk"]))
    stats = record.get("current_weight_stats", {})
    if not isinstance(stats, dict):
        stats = {}
    scalars = [
        layer / denom_layers,
        float(record.get("step", 0)) / 1024.0,
        math.log1p(float(record.get("step", 0))) / math.log(1025.0),
        len(record.get("current_experts", [])) / topk,
        float(stats.get("weight_sum", sum(float(v) for v in record.get("current_weights", [])))),
        float(stats.get("weight_max", 0.0)),
        float(stats.get("top1_top2_margin", 0.0)),
        float(stats.get("entropy", 0.0)),
        float(stats.get("weight_min", 0.0)),
        float(stats.get("valid_experts", len(record.get("current_experts", [])))) / topk,
    ]
    for offset, value in enumerate(scalars[:scalar_count]):
        features[scalar_base + offset] = value


def split_indices(data: LoadedDataset, valid_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    n = data.features.shape[0]
    unique_requests = torch.unique(data.request_seq)
    if unique_requests.numel() >= 4:
        requests = unique_requests.tolist()
        rng = random.Random(seed)
        rng.shuffle(requests)
        valid_count = max(1, int(round(len(requests) * valid_ratio)))
        valid_requests = set(requests[:valid_count])
        mask = torch.tensor([int(value.item()) in valid_requests for value in data.request_seq])
        split_mode = "request_seq"
    else:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)
        valid_count = max(1, int(round(n * valid_ratio)))
        mask = torch.zeros(n, dtype=torch.bool)
        mask[perm[:valid_count]] = True
        split_mode = "sample_random"

    train_idx = torch.nonzero(~mask, as_tuple=False).flatten()
    valid_idx = torch.nonzero(mask, as_tuple=False).flatten()
    if train_idx.numel() == 0 or valid_idx.numel() == 0:
        raise ValueError("split produced empty train or validation set")
    return train_idx, valid_idx, split_mode


def sample_training_indices(train_idx: torch.Tensor, sample_count: int | None, seed: int) -> tuple[torch.Tensor, str]:
    if sample_count is None or sample_count >= train_idx.numel():
        return train_idx, "full"
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(train_idx.numel(), generator=generator)
    return train_idx[perm[:sample_count]], f"train_sample_count_{sample_count}"


class Predictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int | None):
        super().__init__()
        if hidden is None:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def export_nfp_model(
    path: pathlib.Path,
    model: nn.Module,
    data: LoadedDataset,
    hidden: int | None,
    args: argparse.Namespace,
) -> None:
    if hidden is None:
        raise ValueError("NFP export currently requires an MLP hidden size")
    if data.feature_set != "route_history":
        raise ValueError("NFP export currently supports route_history feature sets only")
    net = model.net
    if not isinstance(net, nn.Sequential) or len(net) != 3:
        raise ValueError("NFP export expected Predictor MLP architecture")

    w1 = net[0].weight.detach().cpu().to(torch.float32).contiguous()
    b1 = net[0].bias.detach().cpu().to(torch.float32).contiguous()
    w2_full = net[2].weight.detach().cpu().to(torch.float32).contiguous()
    b2_full = net[2].bias.detach().cpu().to(torch.float32).contiguous()
    num_experts = int(data.meta["num_experts"])
    delta = int(args.export_nfp_delta)
    lookahead = int(data.meta["lookahead_routes"])
    if delta < 1 or delta > lookahead:
        raise ValueError(f"--export-nfp-delta must be in [1, {lookahead}], got {delta}")
    start = (delta - 1) * num_experts
    end = start + num_experts

    # Runtime feature construction is sparse, so store W1 transposed as
    # [input_dim, hidden] for cache-friendly row additions in Rust.
    w1_by_input = w1.t().contiguous()
    w2 = w2_full[start:end, :].contiguous()
    b2 = b2_full[start:end].contiguous()

    header = {
        "schema": "krasis_nfp_model_v1",
        "feature_set": data.feature_set,
        "feature_schema": data.meta.get("feature_schema"),
        "dataset_schema": data.meta.get("dataset_schema"),
        "num_layers": int(data.meta["num_layers"]),
        "num_experts": num_experts,
        "topk": int(data.meta["topk"]),
        "history_tokens": int(data.meta.get("history_tokens", 4)),
        "prior_layers": int(data.meta.get("prior_layers", 4)),
        "input_dim": int(data.features.shape[1]),
        "hidden": hidden,
        "output_dim": num_experts,
        "delta": delta,
        "layer_group": args._layer_group_for_export,
        "arrays": [
            {"name": "w1_by_input", "dtype": "f32", "shape": list(w1_by_input.shape)},
            {"name": "b1", "dtype": "f32", "shape": list(b1.shape)},
            {"name": "w2", "dtype": "f32", "shape": list(w2.shape)},
            {"name": "b2", "dtype": "f32", "shape": list(b2.shape)},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(b"KRASIS_NFP_V1\0\0\0")
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for tensor in (w1_by_input, b1, w2, b2):
            handle.write(tensor.numpy().astype("<f4", copy=False).tobytes(order="C"))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: LoadedDataset,
    indices: torch.Tensor,
    budget: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    hits = 0
    total = 0
    byte_hits = 0
    byte_total = 0
    predicted_count = 0

    for batch_indices in indices.split(batch_size):
        x = data.features[batch_indices].to(device)
        targets = data.targets[batch_indices]
        label_bytes = data.label_bytes[batch_indices]
        scores = model(x).cpu()
        batch_budget = min(budget, scores.shape[1])
        top = torch.topk(scores, k=batch_budget, dim=1).indices
        predicted = torch.zeros_like(targets, dtype=torch.bool)
        predicted.scatter_(1, top, True)
        target_bool = targets > 0
        hit_mask = predicted & target_bool
        hits += hit_mask.sum().item()
        total += target_bool.sum().item()
        byte_hits += label_bytes[hit_mask].sum().item()
        byte_total += label_bytes[target_bool].sum().item()
        predicted_count += predicted.sum().item()

    return {
        "samples": int(indices.numel()),
        "cold_labels": int(total),
        "predicted": int(predicted_count),
        "cold_hits": int(hits),
        "cold_recall_pct": pct(hits, total),
        "cold_bytes_hit": int(byte_hits),
        "cold_bytes_total": int(byte_total),
        "cold_bytes_reduction_pct": pct(byte_hits, byte_total),
    }


def pct(num: float, denom: float) -> float:
    return 0.0 if denom == 0 else float(num) / float(denom) * 100.0


def infer_positive_label_bytes(data: LoadedDataset, indices: torch.Tensor) -> float:
    positives = data.label_bytes[indices][data.label_bytes[indices] > 0]
    if positives.numel() == 0:
        return 0.0
    return float(torch.median(positives).item())


@torch.no_grad()
def replay_delta_caps(
    model: nn.Module,
    data: LoadedDataset,
    indices: torch.Tensor,
    caps: list[int],
    deltas: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any] | None:
    if not caps or not deltas:
        return None

    model.eval()
    num_experts = int(data.meta["num_experts"])
    lookahead = int(data.meta["lookahead_routes"])
    expert_bytes = infer_positive_label_bytes(data, indices)
    cap_metrics: dict[tuple[int, int], dict[str, float]] = {
        (delta, cap): {
            "samples": 0,
            "cold_labels": 0,
            "predicted": 0,
            "cold_hits": 0,
            "cold_bytes_hit": 0,
            "cold_bytes_total": 0,
        }
        for delta in deltas
        if 1 <= delta <= lookahead
        for cap in caps
    }

    for batch_indices in indices.split(batch_size):
        x = data.features[batch_indices].to(device)
        scores = model(x).cpu()
        targets = data.targets[batch_indices] > 0
        label_bytes = data.label_bytes[batch_indices]
        batch_rows = int(batch_indices.numel())

        for delta in deltas:
            if delta < 1 or delta > lookahead:
                continue
            start = (delta - 1) * num_experts
            end = start + num_experts
            delta_scores = scores[:, start:end]
            delta_targets = targets[:, start:end]
            delta_bytes = label_bytes[:, start:end]
            cold_labels = int(delta_targets.sum().item())
            cold_bytes_total = int(delta_bytes[delta_targets].sum().item())

            for cap in caps:
                effective_cap = min(cap, num_experts)
                top = torch.topk(delta_scores, k=effective_cap, dim=1).indices
                predicted = torch.zeros_like(delta_targets, dtype=torch.bool)
                predicted.scatter_(1, top, True)
                hit_mask = predicted & delta_targets
                metrics = cap_metrics[(delta, cap)]
                metrics["samples"] += batch_rows
                metrics["cold_labels"] += cold_labels
                metrics["predicted"] += int(predicted.sum().item())
                metrics["cold_hits"] += int(hit_mask.sum().item())
                metrics["cold_bytes_hit"] += int(delta_bytes[hit_mask].sum().item())
                metrics["cold_bytes_total"] += cold_bytes_total

    result: dict[str, Any] = {
        "schema": "krasis_expert_prefetch_replay_caps_v1",
        "scope": "validation",
        "expert_bytes": int(expert_bytes),
        "deltas": {},
    }
    for delta in deltas:
        entries = []
        for cap in caps:
            metrics = cap_metrics.get((delta, cap))
            if metrics is None:
                continue
            predicted = int(metrics["predicted"])
            hit_bytes = int(metrics["cold_bytes_hit"])
            speculative_bytes = int(round(predicted * expert_bytes))
            entries.append(
                {
                    "cap": cap,
                    "samples": int(metrics["samples"]),
                    "cold_labels": int(metrics["cold_labels"]),
                    "predicted": predicted,
                    "cold_hits": int(metrics["cold_hits"]),
                    "precision_pct": pct(metrics["cold_hits"], predicted),
                    "cold_recall_pct": pct(metrics["cold_hits"], metrics["cold_labels"]),
                    "cold_bytes_hit": hit_bytes,
                    "cold_bytes_total": int(metrics["cold_bytes_total"]),
                    "cold_bytes_reduction_pct": pct(hit_bytes, metrics["cold_bytes_total"]),
                    "speculative_bytes": speculative_bytes,
                    "waste_bytes": max(0, speculative_bytes - hit_bytes),
                    "useful_byte_ratio_pct": pct(hit_bytes, speculative_bytes),
                }
            )
        result["deltas"][str(delta)] = entries
    return result


def train_one(
    data: LoadedDataset,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor,
    hidden: int | None,
    args: argparse.Namespace,
    device: torch.device,
    run_seed: int,
) -> dict[str, Any]:
    torch.manual_seed(run_seed)
    random.seed(run_seed)
    model = Predictor(data.features.shape[1], data.targets.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(data.features[train_idx], data.targets[train_idx])
    loader_generator = torch.Generator().manual_seed(run_seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=loader_generator)

    positive = data.targets[train_idx].sum()
    total = data.targets[train_idx].numel()
    negative = total - positive
    pos_weight_value = min(args.max_pos_weight, max(1.0, float(negative / positive.clamp_min(1.0))))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((data.targets.shape[1],), pos_weight_value, device=device))

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        valid = evaluate(model, data, valid_idx, args.budget, device, args.batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / max(1, seen),
                "valid_cold_recall_pct": valid["cold_recall_pct"],
                "valid_cold_bytes_reduction_pct": valid["cold_bytes_reduction_pct"],
            }
        )

    train_metrics = evaluate(model, data, train_idx, args.budget, device, args.batch_size)
    valid_metrics = evaluate(model, data, valid_idx, args.budget, device, args.batch_size)
    name = "linear" if hidden is None else f"mlp_h{hidden}"
    model_path = None
    if args.save_models:
        model_path = pathlib.Path(args.out_dir) / f"{args.model_prefix}_{name}.pt"
        torch.save(
            {
                "schema": "krasis_expert_prefetch_model_v1",
                "name": name,
                "meta": data.meta,
                "hidden": hidden,
                "state_dict": model.state_dict(),
                "input_dim": data.features.shape[1],
                "output_dim": data.targets.shape[1],
            },
            model_path,
        )
    nfp_model_path = None
    if args.export_nfp_models and hidden is not None:
        nfp_model_path = pathlib.Path(args.out_dir) / f"{args.model_prefix}_{name}.nfp"
        export_nfp_model(nfp_model_path, model, data, hidden, args)

    replay = replay_delta_caps(
        model,
        data,
        valid_idx,
        args.replay_caps,
        args.replay_deltas,
        device,
        args.batch_size,
    )

    result = {
        "name": name,
        "hidden": hidden,
        "parameters": parameter_count(model),
        "pos_weight": pos_weight_value,
        "train": train_metrics,
        "valid": valid_metrics,
        "history": history,
        "model_path": str(model_path) if model_path else None,
        "nfp_model_path": str(nfp_model_path) if nfp_model_path else None,
    }
    if replay is not None:
        result["replay"] = replay
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-prefix", default="expert_prefetch")
    parser.add_argument("--hidden-sizes", default="64,256,1024")
    parser.add_argument("--include-linear", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-counts")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default=DEFAULT_FEATURE_SET)
    parser.add_argument("--layer-groups", type=int, default=1)
    parser.add_argument("--layer-group-index", type=int, default=0)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--export-nfp-models", action="store_true")
    parser.add_argument("--export-nfp-delta", type=int, default=1)
    parser.add_argument("--replay-caps", default="")
    parser.add_argument("--replay-deltas", default="1")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not (0.0 < args.valid_ratio < 1.0):
        raise ValueError("--valid-ratio must be between 0 and 1")
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.layer_groups <= 0:
        raise ValueError("--layer-groups must be positive")
    if args.max_samples is not None and args.sample_counts is None:
        args.sample_counts = str(args.max_samples)
    args.replay_caps = parse_positive_ints(args.replay_caps)
    args.replay_deltas = parse_positive_ints(args.replay_deltas)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    data = load_datasets(args.datasets, args.feature_set)
    data, layer_group = filter_layer_group(data, args.layer_groups, args.layer_group_index)
    args._layer_group_for_export = layer_group
    train_idx, valid_idx, split_mode = split_indices(data, args.valid_ratio, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    configs: list[int | None] = []
    if args.include_linear:
        configs.append(None)
    configs.extend(hidden_sizes)

    sample_counts = parse_sample_counts(args.sample_counts)
    learning_curve = []
    for sample_index, sample_count in enumerate(sample_counts):
        curve_train_idx, curve_mode = sample_training_indices(train_idx, sample_count, args.seed)
        models = []
        for model_index, hidden in enumerate(configs):
            hidden_seed = 0 if hidden is None else hidden
            run_seed = args.seed + model_index * 1_000_003 + hidden_seed
            models.append(train_one(data, curve_train_idx, valid_idx, hidden, args, device, run_seed))
        learning_curve.append(
            {
                "sample_count": sample_count,
                "sample_index": sample_index,
                "effective_train_samples": int(curve_train_idx.numel()),
                "sample_mode": curve_mode,
                "models": models,
            }
        )
    summary = {
        "schema": "krasis_expert_prefetch_training_summary_v1",
        "datasets": [str(path) for path in args.datasets],
        "dataset": str(args.datasets[0]) if len(args.datasets) == 1 else None,
        "dataset_meta": data.meta,
        "feature_set": data.feature_set,
        "layer_group": layer_group,
        "samples": int(data.features.shape[0]),
        "feature_dim": int(data.features.shape[1]),
        "output_dim": int(data.targets.shape[1]),
        "budget": args.budget,
        "split_mode": split_mode,
        "train_samples": int(train_idx.numel()),
        "valid_samples": int(valid_idx.numel()),
        "device": str(device),
        "epochs": args.epochs,
        "sample_counts": sample_counts,
        "learning_curve": learning_curve,
        "models": learning_curve[-1]["models"],
    }
    out_path = pathlib.Path(args.out_dir) / f"{args.model_prefix}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
