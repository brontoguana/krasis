#!/usr/bin/env python3
"""Train and replay request-level HCS placement planners from route traces."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print(
        "hcs_planner_train.py must be run through ./dev hcs-planner-train",
        file=sys.stderr,
    )
    sys.exit(2)


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - environment gate
    print(f"torch is required for HCS planner training: {exc}", file=sys.stderr)
    sys.exit(2)


TRACE_SCHEMA = "krasis_expert_prefetch_trace_v1"
DATASET_SCHEMA = "krasis_hcs_request_heatmap_dataset_v1"


@dataclass
class RequestExample:
    source: str
    request_seq: int
    request_label: str
    feature: list[float]
    candidate_features: dict[str, dict[int, float]]
    layer_features: dict[str, dict[int, float]]
    target: list[float]
    all_counts: dict[int, int]
    all_bytes: dict[int, int]
    cold_counts: dict[int, int]
    cold_bytes: dict[int, int]
    recorded_hcs_hits: int
    recorded_hcs_bytes: int
    total_positions: int
    total_bytes: int
    cold_positions: int
    cold_total_bytes: int


@dataclass
class LoadedHeatmapDataset:
    examples: list[RequestExample]
    meta: dict[str, Any]
    sources: list[str]


class Planner(nn.Module):
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


class CandidateRanker(nn.Module):
    def __init__(self, input_dim: int, hidden: int | None):
        super().__init__()
        if hidden is None:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pct(num: float, denom: float) -> float:
    return 0.0 if denom == 0 else float(num) / float(denom) * 100.0


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"expected positive integer, got {value}")
        values.append(value)
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_hidden_sizes(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in {"linear", "none"}:
            values.append(None)
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"hidden size must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("--hidden-sizes must contain at least one value")
    return values


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("expected at least one float")
    return values


def route_matches(record: dict[str, Any], label: str | None, prefix: str | None) -> bool:
    request_label = normalize_request_label(str(record.get("request_label", "")))
    if label is not None and request_label != normalize_request_label(label):
        return False
    if prefix is not None and not request_label.startswith(prefix):
        return False
    return True


def normalize_request_label(label: str) -> str:
    for suffix in ("_nosse", "_sse"):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def read_trace_requests(
    paths: list[pathlib.Path],
    request_label: str | None,
    request_label_prefix: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, int, str], dict[str, Any]]]:
    routes: list[dict[str, Any]] = []
    predecode: dict[tuple[str, int, str], dict[str, Any]] = {}
    meta = {
        "sources": [str(path) for path in paths],
        "trace_routes": 0,
        "predecode_records": 0,
        "filtered_predecode_records": 0,
        "filtered_routes": 0,
        "bad_json_lines": 0,
        "num_layers": 0,
        "num_experts": 0,
        "topk": 0,
    }
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    meta["bad_json_lines"] += 1
                    continue
                event = record.get("event")
                if event == "meta":
                    if record.get("schema") != TRACE_SCHEMA:
                        raise ValueError(f"{path}:{line_no}: unsupported trace schema")
                    continue
                if event == "predecode":
                    meta["predecode_records"] += 1
                    if record.get("schema") != TRACE_SCHEMA:
                        raise ValueError(f"{path}:{line_no}: unsupported predecode schema")
                    if not route_matches(record, request_label, request_label_prefix):
                        continue
                    meta["filtered_predecode_records"] += 1
                    meta["num_layers"] = max(meta["num_layers"], int(record.get("count_layers", 0)))
                    meta["num_experts"] = max(meta["num_experts"], int(record.get("count_experts_per_layer", 0)))
                    record["_source"] = str(path)
                    record["_request_label_norm"] = normalize_request_label(str(record["request_label"]))
                    key = (str(path), int(record["request_seq"]), record["_request_label_norm"])
                    predecode[key] = record
                    continue
                if event != "route":
                    continue
                meta["trace_routes"] += 1
                if record.get("schema") != TRACE_SCHEMA:
                    raise ValueError(f"{path}:{line_no}: unsupported route schema")
                if not route_matches(record, request_label, request_label_prefix):
                    continue
                meta["filtered_routes"] += 1
                meta["num_layers"] = max(meta["num_layers"], int(record["layer"]) + 1)
                meta["num_experts"] = max(meta["num_experts"], int(record["num_experts"]))
                meta["topk"] = max(meta["topk"], int(record["topk"]))
                record["_source"] = str(path)
                record["_request_label_norm"] = normalize_request_label(str(record["request_label"]))
                routes.append(record)
    routes.sort(key=lambda r: (r["_source"], int(r["request_seq"]), str(r["_request_label_norm"]), int(r["step"]), int(r["layer"])))
    return routes, meta, predecode


def expert_bytes_for_route(route: dict[str, Any], default_expert_bytes: int) -> int:
    cold = int(route.get("cold_experts", 0))
    if cold > 0:
        return max(1, int(route.get("cold_bytes", 0)) // cold)
    return max(1, default_expert_bytes)


def infer_default_expert_bytes(routes: list[dict[str, Any]]) -> int:
    values = []
    for route in routes:
        cold = int(route.get("cold_experts", 0))
        if cold > 0:
            values.append(max(1, int(route.get("cold_bytes", 0)) // cold))
    if not values:
        return 1
    values.sort()
    return int(values[len(values) // 2])


def build_examples(
    routes: list[dict[str, Any]],
    meta: dict[str, Any],
    predecode: dict[tuple[str, int, str], dict[str, Any]],
    input_mode: str,
    input_steps: int,
    target_steps: int,
    target_start_step: int,
    target_cold_only: bool,
    token_hash_bins: int,
) -> list[RequestExample]:
    num_layers = int(meta["num_layers"])
    num_experts = int(meta["num_experts"])
    output_dim = num_layers * num_experts
    if input_mode == "predecode":
        feature_dim = 2 + output_dim + token_hash_bins * 2
    else:
        feature_dim = 1 if input_steps == 0 else 1 + output_dim * 2
    default_expert_bytes = infer_default_expert_bytes(routes)

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        grouped[(route["_source"], int(route["request_seq"]), str(route["_request_label_norm"]))].append(route)

    examples: list[RequestExample] = []
    target_end_step = target_start_step + target_steps - 1
    for (source, request_seq, request_label), req_routes in grouped.items():
        feature = [0.0] * feature_dim
        feature[0] = 1.0
        if input_mode == "predecode":
            pre = predecode.get((source, request_seq, request_label))
            if pre is None:
                continue
            prompt_tokens = max(1, int(pre.get("prompt_tokens", 0)))
            candidate_features: dict[str, dict[int, float]] = {
                "weighted": {},
                "final_presence": {},
                "final_weight": {},
                "final_rank": {},
                "head_presence": {},
                "head_weight": {},
                "head_rank": {},
                "tail_presence": {},
                "tail_weight": {},
                "tail_rank": {},
            }
            layer_features: dict[str, dict[int, float]] = {
                "mean_entropy": {},
                "final_entropy": {},
                "mean_top1": {},
                "mean_margin": {},
                "mean_topk_sum": {},
                "final_top1": {},
                "final_top2": {},
                "final_margin": {},
                "final_topk_sum": {},
            }
            for window in pre.get("recency_windows", []):
                window = int(window)
                candidate_features.setdefault(f"recency_count_{window}", {})
                candidate_features.setdefault(f"recency_weight_{window}", {})
                layer_features.setdefault(f"recency_entropy_{window}", {})
                layer_features.setdefault(f"recency_top1_{window}", {})
                layer_features.setdefault(f"recency_margin_{window}", {})
            bucket_denoms: dict[int, float] = {}
            for bucket in pre.get("position_buckets", []):
                bucket_idx = int(bucket.get("bucket", -1))
                start_token = int(bucket.get("start_token", 0))
                end_token = int(bucket.get("end_token", start_token))
                if bucket_idx >= 0:
                    bucket_denoms[bucket_idx] = float(max(1, end_token - start_token))
                    candidate_features.setdefault(f"bucket_count_{bucket_idx}", {})
                    candidate_features.setdefault(f"bucket_weight_{bucket_idx}", {})
            feature[1] = min(prompt_tokens, 65536) / 65536.0
            count_layers = int(pre.get("count_layers", 0))
            count_experts = int(pre.get("count_experts_per_layer", 0))
            count_norm = float(max(1, prompt_tokens))
            for item in pre.get("prompt_expert_counts", []):
                layer = int(item.get("layer", -1))
                expert = int(item.get("expert", -1))
                if 0 <= layer < num_layers and 0 <= expert < num_experts:
                    if layer < count_layers and expert < count_experts:
                        feature[2 + layer * num_experts + expert] = float(item.get("count", 0)) / count_norm
            for item in pre.get("prompt_expert_weight_sums", []):
                layer = int(item.get("layer", -1))
                expert = int(item.get("expert", -1))
                if 0 <= layer < num_layers and 0 <= expert < num_experts:
                    candidate_features["weighted"][layer * num_experts + expert] = float(item.get("weight_sum", 0.0)) / count_norm
            for item in pre.get("prompt_expert_recency_counts", []):
                window = int(item.get("window", 0))
                name = f"recency_count_{window}"
                candidate_features.setdefault(name, {})
                layer = int(item.get("layer", -1))
                expert = int(item.get("expert", -1))
                if 0 <= layer < num_layers and 0 <= expert < num_experts:
                    denom = float(max(1, min(window, prompt_tokens)))
                    candidate_features[name][layer * num_experts + expert] = float(item.get("count", 0)) / denom
            for item in pre.get("prompt_expert_recency_weight_sums", []):
                window = int(item.get("window", 0))
                name = f"recency_weight_{window}"
                candidate_features.setdefault(name, {})
                layer = int(item.get("layer", -1))
                expert = int(item.get("expert", -1))
                if 0 <= layer < num_layers and 0 <= expert < num_experts:
                    denom = float(max(1, min(window, prompt_tokens)))
                    candidate_features[name][layer * num_experts + expert] = float(item.get("weight_sum", 0.0)) / denom
            for item in pre.get("prompt_expert_bucket_counts", []):
                bucket = int(item.get("bucket", -1))
                name = f"bucket_count_{bucket}"
                candidate_features.setdefault(name, {})
                layer = int(item.get("layer", -1))
                expert = int(item.get("expert", -1))
                if bucket >= 0 and 0 <= layer < num_layers and 0 <= expert < num_experts:
                    denom = bucket_denoms.get(bucket, count_norm)
                    candidate_features[name][layer * num_experts + expert] = float(item.get("count", 0)) / denom
            for item in pre.get("prompt_expert_bucket_weight_sums", []):
                bucket = int(item.get("bucket", -1))
                name = f"bucket_weight_{bucket}"
                candidate_features.setdefault(name, {})
                layer = int(item.get("layer", -1))
                expert = int(item.get("expert", -1))
                if bucket >= 0 and 0 <= layer < num_layers and 0 <= expert < num_experts:
                    denom = bucket_denoms.get(bucket, count_norm)
                    candidate_features[name][layer * num_experts + expert] = float(item.get("weight_sum", 0.0)) / denom
            for item in pre.get("final_token_routes", []):
                layer = int(item.get("layer", -1))
                if not (0 <= layer < num_layers):
                    continue
                expert_ids = list(item.get("expert_ids", []))
                weights = list(item.get("weights", []))
                denom = float(max(1, len(expert_ids) - 1))
                for rank, expert in enumerate(expert_ids):
                    expert = int(expert)
                    if 0 <= expert < num_experts:
                        key = layer * num_experts + expert
                        candidate_features["final_presence"][key] = 1.0
                        candidate_features["final_weight"][key] = float(weights[rank]) if rank < len(weights) else 0.0
                        candidate_features["final_rank"][key] = 1.0 - (float(rank) / denom if denom > 0.0 else 0.0)
            for route_field, prefix in (("prompt_route_head", "head"), ("prompt_route_tail", "tail")):
                route_rows = list(pre.get(route_field, []))
                route_norm = float(max(1, len(route_rows)))
                for item in route_rows:
                    layer = int(item.get("layer", -1))
                    if not (0 <= layer < num_layers):
                        continue
                    expert_ids = list(item.get("expert_ids", []))
                    weights = list(item.get("weights", []))
                    rank_denom = float(max(1, len(expert_ids) - 1))
                    for rank, expert in enumerate(expert_ids):
                        expert = int(expert)
                        if 0 <= expert < num_experts:
                            key = layer * num_experts + expert
                            candidate_features[f"{prefix}_presence"][key] = (
                                candidate_features[f"{prefix}_presence"].get(key, 0.0) + 1.0 / route_norm
                            )
                            candidate_features[f"{prefix}_weight"][key] = (
                                candidate_features[f"{prefix}_weight"].get(key, 0.0)
                                + (float(weights[rank]) if rank < len(weights) else 0.0) / route_norm
                            )
                            candidate_features[f"{prefix}_rank"][key] = (
                                candidate_features[f"{prefix}_rank"].get(key, 0.0)
                                + (1.0 - (float(rank) / rank_denom if rank_denom > 0.0 else 0.0)) / route_norm
                            )
            for item in pre.get("prompt_route_entropy", []):
                layer = int(item.get("layer", -1))
                if 0 <= layer < num_layers:
                    layer_features["mean_entropy"][layer] = float(item.get("mean_topk_entropy", 0.0))
                    layer_features["final_entropy"][layer] = float(item.get("final_topk_entropy", 0.0))
            for item in pre.get("prompt_route_confidence", []):
                layer = int(item.get("layer", -1))
                if 0 <= layer < num_layers:
                    layer_features["mean_top1"][layer] = float(item.get("mean_top1_weight", 0.0))
                    layer_features["mean_margin"][layer] = float(item.get("mean_top1_top2_margin", 0.0))
                    layer_features["mean_topk_sum"][layer] = float(item.get("mean_topk_weight_sum", 0.0))
                    layer_features["final_top1"][layer] = float(item.get("final_top1_weight", 0.0))
                    layer_features["final_top2"][layer] = float(item.get("final_top2_weight", 0.0))
                    layer_features["final_margin"][layer] = float(item.get("final_top1_top2_margin", 0.0))
                    layer_features["final_topk_sum"][layer] = float(item.get("final_topk_weight_sum", 0.0))
            for item in pre.get("prompt_route_recency_confidence", []):
                window = int(item.get("window", 0))
                layer = int(item.get("layer", -1))
                if 0 <= layer < num_layers:
                    entropy_name = f"recency_entropy_{window}"
                    top1_name = f"recency_top1_{window}"
                    margin_name = f"recency_margin_{window}"
                    layer_features.setdefault(entropy_name, {})[layer] = float(item.get("mean_topk_entropy", 0.0))
                    layer_features.setdefault(top1_name, {})[layer] = float(item.get("mean_top1_weight", 0.0))
                    layer_features.setdefault(margin_name, {})[layer] = float(item.get("mean_top1_top2_margin", 0.0))
            first_base = 2 + output_dim
            last_base = first_base + token_hash_bins
            if token_hash_bins > 0:
                first = list(pre.get("first_token_ids", []))
                last = list(pre.get("last_token_ids", []))
                first_norm = float(max(1, len(first)))
                last_norm = float(max(1, len(last)))
                for tok in first:
                    feature[first_base + (int(tok) % token_hash_bins)] += 1.0 / first_norm
                for tok in last:
                    feature[last_base + (int(tok) % token_hash_bins)] += 1.0 / last_norm
        else:
            candidate_features = {}
            layer_features = {}
        target = [0.0] * output_dim
        all_counts: dict[int, int] = defaultdict(int)
        all_bytes: dict[int, int] = defaultdict(int)
        cold_counts: dict[int, int] = defaultdict(int)
        cold_bytes: dict[int, int] = defaultdict(int)
        recorded_hcs_hits = 0
        recorded_hcs_bytes = 0
        total_positions = 0
        total_bytes = 0
        cold_positions = 0
        cold_total_bytes = 0

        for route in req_routes:
            step = int(route["step"])
            layer = int(route["layer"])
            expert_bytes = expert_bytes_for_route(route, default_expert_bytes)
            expert_ids = list(route.get("expert_ids", []))[: int(route["topk"])]
            weights = list(route.get("weights", []))[: int(route["topk"])]
            hcs_hits = list(route.get("hcs_hits", []))[: int(route["topk"])]

            if input_mode != "predecode" and input_steps > 0 and step <= input_steps:
                for rank, expert in enumerate(expert_ids):
                    expert = int(expert)
                    if expert < 0 or expert >= num_experts:
                        continue
                    key = layer * num_experts + expert
                    count_base = 1
                    weight_base = 1 + output_dim
                    feature[count_base + key] += 1.0
                    feature[weight_base + key] += float(weights[rank]) if rank < len(weights) else 0.0

            if target_start_step <= step <= target_end_step:
                for rank, expert in enumerate(expert_ids):
                    expert = int(expert)
                    if expert < 0 or expert >= num_experts:
                        continue
                    key = layer * num_experts + expert
                    hit = bool(hcs_hits[rank]) if rank < len(hcs_hits) else False
                    total_positions += 1
                    total_bytes += expert_bytes
                    all_counts[key] += 1
                    all_bytes[key] += expert_bytes
                    if hit:
                        recorded_hcs_hits += 1
                        recorded_hcs_bytes += expert_bytes
                    else:
                        cold_positions += 1
                        cold_total_bytes += expert_bytes
                        cold_counts[key] += 1
                        cold_bytes[key] += expert_bytes
                    if (not target_cold_only) or (not hit):
                        target[key] += float(expert_bytes)

        if total_positions == 0 or not any(value > 0.0 for value in target):
            continue
        if input_steps > 0:
            scale = 1.0 / input_steps
            for idx in range(1, feature_dim):
                feature[idx] *= scale
        examples.append(
            RequestExample(
                source=source,
                request_seq=request_seq,
                request_label=request_label,
                feature=feature,
                candidate_features=candidate_features,
                layer_features=layer_features,
                target=target,
                all_counts=dict(all_counts),
                all_bytes=dict(all_bytes),
                cold_counts=dict(cold_counts),
                cold_bytes=dict(cold_bytes),
                recorded_hcs_hits=recorded_hcs_hits,
                recorded_hcs_bytes=recorded_hcs_bytes,
                total_positions=total_positions,
                total_bytes=total_bytes,
                cold_positions=cold_positions,
                cold_total_bytes=cold_total_bytes,
            )
        )
    return examples


def examples_to_tensors(examples: list[RequestExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor([example.feature for example in examples], dtype=torch.float32)
    values = torch.tensor([example.target for example in examples], dtype=torch.float32)
    labels = (values > 0).to(torch.float32)
    return features, labels, values


def _int_key_dict(raw: dict[Any, Any]) -> dict[int, Any]:
    return {int(key): value for key, value in raw.items()}


def _nested_int_key_dict(raw: dict[str, dict[Any, Any]]) -> dict[str, dict[int, float]]:
    return {
        str(name): {int(key): float(value) for key, value in values.items()}
        for name, values in raw.items()
    }


def request_example_to_record(example: RequestExample, sample_id: int) -> dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA,
        "event": "sample",
        "sample_id": sample_id,
        "source": example.source,
        "request_seq": example.request_seq,
        "request_label": example.request_label,
        "feature": example.feature,
        "candidate_features": example.candidate_features,
        "layer_features": example.layer_features,
        "target": example.target,
        "all_counts": example.all_counts,
        "all_bytes": example.all_bytes,
        "cold_counts": example.cold_counts,
        "cold_bytes": example.cold_bytes,
        "recorded_hcs_hits": example.recorded_hcs_hits,
        "recorded_hcs_bytes": example.recorded_hcs_bytes,
        "total_positions": example.total_positions,
        "total_bytes": example.total_bytes,
        "cold_positions": example.cold_positions,
        "cold_total_bytes": example.cold_total_bytes,
    }


def request_example_from_record(record: dict[str, Any]) -> RequestExample:
    if record.get("schema") != DATASET_SCHEMA or record.get("event") != "sample":
        raise ValueError(f"unsupported dataset sample record: {record.get('schema')!r}/{record.get('event')!r}")
    return RequestExample(
        source=str(record["source"]),
        request_seq=int(record["request_seq"]),
        request_label=str(record["request_label"]),
        feature=[float(value) for value in record["feature"]],
        candidate_features=_nested_int_key_dict(record.get("candidate_features", {})),
        layer_features=_nested_int_key_dict(record.get("layer_features", {})),
        target=[float(value) for value in record["target"]],
        all_counts={key: int(value) for key, value in _int_key_dict(record.get("all_counts", {})).items()},
        all_bytes={key: int(value) for key, value in _int_key_dict(record.get("all_bytes", {})).items()},
        cold_counts={key: int(value) for key, value in _int_key_dict(record.get("cold_counts", {})).items()},
        cold_bytes={key: int(value) for key, value in _int_key_dict(record.get("cold_bytes", {})).items()},
        recorded_hcs_hits=int(record["recorded_hcs_hits"]),
        recorded_hcs_bytes=int(record["recorded_hcs_bytes"]),
        total_positions=int(record["total_positions"]),
        total_bytes=int(record["total_bytes"]),
        cold_positions=int(record["cold_positions"]),
        cold_total_bytes=int(record["cold_total_bytes"]),
    )


def dataset_meta(
    *,
    trace_paths: list[pathlib.Path],
    filter_args: argparse.Namespace,
    trace_meta: dict[str, Any],
    examples: list[RequestExample],
) -> dict[str, Any]:
    feature_dim = len(examples[0].feature) if examples else 0
    output_dim = len(examples[0].target) if examples else 0
    return {
        "schema": DATASET_SCHEMA,
        "event": "meta",
        "sources": [str(path) for path in trace_paths],
        "filter": {
            "request_label": filter_args.request_label,
            "request_label_prefix": filter_args.request_label_prefix,
        },
        "input_mode": filter_args.input_mode,
        "input_steps": filter_args.input_steps,
        "token_hash_bins": filter_args.token_hash_bins,
        "target_start_step": filter_args.target_start_step,
        "target_steps": filter_args.target_steps,
        "target_cold_only": filter_args.target_cold_only,
        "trace_meta": trace_meta,
        "examples": len(examples),
        "feature_dim": feature_dim,
        "output_dim": output_dim,
    }


def write_dataset(path: pathlib.Path, meta: dict[str, Any], examples: list[RequestExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, sort_keys=True)
        handle.write("\n")
        for sample_id, example in enumerate(examples):
            json.dump(request_example_to_record(example, sample_id), handle, sort_keys=True)
            handle.write("\n")


def load_one_heatmap_dataset(path: pathlib.Path) -> LoadedHeatmapDataset:
    meta: dict[str, Any] | None = None
    examples: list[RequestExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("schema") != DATASET_SCHEMA:
                raise ValueError(f"{path}:{line_no}: unsupported dataset schema {record.get('schema')!r}")
            event = record.get("event")
            if event == "meta":
                if meta is not None:
                    raise ValueError(f"{path}:{line_no}: duplicate dataset meta")
                meta = record
            elif event == "sample":
                if meta is None:
                    raise ValueError(f"{path}:{line_no}: sample before meta")
                examples.append(request_example_from_record(record))
            else:
                raise ValueError(f"{path}:{line_no}: unknown dataset event {event!r}")
    if meta is None:
        raise ValueError(f"{path}: missing dataset meta")
    if not examples:
        raise ValueError(f"{path}: no examples")
    return LoadedHeatmapDataset(examples=examples, meta=meta, sources=[str(path)])


def load_heatmap_datasets(paths: list[pathlib.Path]) -> LoadedHeatmapDataset:
    if not paths:
        raise ValueError("at least one heatmap dataset path is required")
    loaded = [load_one_heatmap_dataset(path) for path in paths]
    base = loaded[0]
    for other in loaded[1:]:
        for key in ("input_mode", "input_steps", "target_start_step", "target_steps", "target_cold_only", "feature_dim", "output_dim"):
            if base.meta.get(key) != other.meta.get(key):
                raise ValueError(
                    f"dataset metadata mismatch for {key}: "
                    f"{base.meta.get(key)!r} != {other.meta.get(key)!r}"
                )
        base_trace = base.meta.get("trace_meta", {})
        other_trace = other.meta.get("trace_meta", {})
        for key in ("num_layers", "num_experts", "topk"):
            if base_trace.get(key) != other_trace.get(key):
                raise ValueError(
                    f"dataset trace metadata mismatch for {key}: "
                    f"{base_trace.get(key)!r} != {other_trace.get(key)!r}"
                )
    return LoadedHeatmapDataset(
        examples=[example for item in loaded for example in item.examples],
        meta=base.meta,
        sources=[source for item in loaded for source in item.sources],
    )


def split_examples(
    examples: list[RequestExample],
    valid_ratio: float,
    seed: int,
) -> tuple[list[RequestExample], list[RequestExample], str]:
    if len(examples) < 4:
        raise ValueError("need at least four request examples for train/valid split")
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_ratio)))
    return shuffled[valid_count:], shuffled[:valid_count], "request_random"


def select_top(scores: torch.Tensor, budget: int) -> list[set[int]]:
    effective = min(budget, int(scores.shape[1]))
    top = torch.topk(scores, k=effective, dim=1).indices.cpu().tolist()
    return [set(int(v) for v in row) for row in top]


def global_rank_from_examples(examples: list[RequestExample], attr: str) -> torch.Tensor:
    if not examples:
        raise ValueError("cannot rank from no examples")
    output_dim = len(examples[0].target)
    scores = torch.zeros(output_dim, dtype=torch.float32)
    for example in examples:
        values = getattr(example, attr)
        for key, value in values.items():
            scores[int(key)] += float(value)
    return scores


def static_prior_from_examples(
    examples: list[RequestExample],
    attr: str,
    smoothing: float,
) -> torch.Tensor:
    if smoothing < 0.0:
        raise ValueError("--prior-smoothing must be non-negative")
    scores = global_rank_from_examples(examples, attr)
    smoothed = scores + float(smoothing)
    if float(smoothed.sum()) <= 0.0:
        raise ValueError("cannot build static prior from zero scores without smoothing")
    probs = smoothed / smoothed.sum().clamp_min(1.0)
    return torch.log(probs.clamp_min(1e-30))


def make_layer_groups(num_layers: int, group_count: int) -> list[tuple[int, int]]:
    if group_count <= 0:
        raise ValueError("--layer-groups must be positive")
    if group_count > num_layers:
        raise ValueError(f"--layer-groups {group_count} exceeds layer count {num_layers}")
    groups = []
    for group_idx in range(group_count):
        start = (group_idx * num_layers) // group_count
        end = ((group_idx + 1) * num_layers) // group_count
        if start < end:
            groups.append((start, end))
    return groups


def candidate_feature_dim() -> int:
    return 40


def build_candidate_tensor(
    examples: list[RequestExample],
    keys: list[int],
    num_layers: int,
    num_experts: int,
    static_all_scores: torch.Tensor,
    static_cold_scores: torch.Tensor,
    prior_all_log: torch.Tensor,
    prior_cold_log: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_max = float(static_all_scores.max().clamp_min(1.0))
    cold_max = float(static_cold_scores.max().clamp_min(1.0))
    rows = torch.zeros((len(examples), len(keys), candidate_feature_dim()), dtype=torch.float32)
    values = torch.zeros((len(examples), len(keys)), dtype=torch.float32)
    denom_layers = float(max(1, num_layers - 1))
    denom_experts = float(max(1, num_experts - 1))
    max_prompt = 65536.0
    key_features = torch.zeros((len(keys), candidate_feature_dim()), dtype=torch.float32)
    for local_idx, key in enumerate(keys):
        layer = key // num_experts
        expert = key % num_experts
        key_features[local_idx, 0] = 1.0
        key_features[local_idx, 4] = float(prior_all_log[key] / 32.0)
        key_features[local_idx, 5] = float(prior_cold_log[key] / 32.0)
        key_features[local_idx, 6] = float(static_all_scores[key] / all_max)
        key_features[local_idx, 7] = float(static_cold_scores[key] / cold_max)
        key_features[local_idx, 8] = float(layer) / denom_layers
        key_features[local_idx, 9] = float(expert) / denom_experts
        key_features[local_idx, 10] = float((layer * 31 + expert * 17) % 997) / 996.0
    for request_idx, example in enumerate(examples):
        prompt_norm = float(example.feature[1]) if len(example.feature) > 1 else 0.0
        prompt_tokens = max(1.0, prompt_norm * max_prompt)
        rows[request_idx, :, :] = key_features
        rows[request_idx, :, 1] = prompt_norm
        for local_idx, key in enumerate(keys):
            count_norm = 0.0
            predecode_idx = 2 + key
            if predecode_idx < len(example.feature):
                count_norm = float(example.feature[predecode_idx])
            rows[request_idx, local_idx, 2] = count_norm
            rows[request_idx, local_idx, 3] = math.log1p(count_norm * prompt_tokens)
            rows[request_idx, local_idx, 11] = 1.0 if count_norm > 0.0 else 0.0
            rows[request_idx, local_idx, 12] = example.candidate_features.get("weighted", {}).get(key, 0.0)
            rows[request_idx, local_idx, 13] = example.candidate_features.get("recency_count_128", {}).get(key, 0.0)
            rows[request_idx, local_idx, 14] = example.candidate_features.get("recency_count_512", {}).get(key, 0.0)
            rows[request_idx, local_idx, 15] = example.candidate_features.get("recency_count_2048", {}).get(key, 0.0)
            rows[request_idx, local_idx, 16] = example.candidate_features.get("recency_weight_128", {}).get(key, 0.0)
            rows[request_idx, local_idx, 17] = example.candidate_features.get("recency_weight_512", {}).get(key, 0.0)
            rows[request_idx, local_idx, 18] = example.candidate_features.get("recency_weight_2048", {}).get(key, 0.0)
            rows[request_idx, local_idx, 19] = example.candidate_features.get("final_presence", {}).get(key, 0.0)
            rows[request_idx, local_idx, 20] = example.candidate_features.get("final_weight", {}).get(key, 0.0)
            rows[request_idx, local_idx, 21] = example.candidate_features.get("final_rank", {}).get(key, 0.0)
            layer = key // num_experts
            rows[request_idx, local_idx, 22] = example.layer_features.get("mean_entropy", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 23] = example.layer_features.get("final_entropy", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 24] = example.layer_features.get("mean_top1", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 25] = example.layer_features.get("mean_margin", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 26] = example.layer_features.get("mean_topk_sum", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 27] = example.layer_features.get("final_top1", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 28] = example.layer_features.get("final_top2", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 29] = example.layer_features.get("final_margin", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 30] = example.layer_features.get("final_topk_sum", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 31] = example.layer_features.get("recency_entropy_128", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 32] = example.layer_features.get("recency_entropy_512", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 33] = example.layer_features.get("recency_entropy_2048", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 34] = example.layer_features.get("recency_top1_128", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 35] = example.layer_features.get("recency_top1_512", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 36] = example.layer_features.get("recency_top1_2048", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 37] = example.layer_features.get("recency_margin_128", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 38] = example.layer_features.get("recency_margin_512", {}).get(layer, 0.0)
            rows[request_idx, local_idx, 39] = example.layer_features.get("recency_margin_2048", {}).get(layer, 0.0)
            values[request_idx, local_idx] = float(example.target[key])
    return rows, values


def evaluate_group_selected(
    name: str,
    examples: list[RequestExample],
    selected: list[set[int]],
    budget: int,
    group_keys: set[int],
) -> dict[str, Any]:
    all_hits = 0
    all_bytes_hit = 0
    total_positions = 0
    total_bytes = 0
    for example, selected_keys in zip(examples, selected, strict=True):
        selected_in_group = selected_keys & group_keys
        for key, count in example.all_counts.items():
            if key not in group_keys:
                continue
            total_positions += count
            total_bytes += example.all_bytes.get(key, 0)
        for key in selected_in_group:
            all_hits += example.all_counts.get(key, 0)
            all_bytes_hit += example.all_bytes.get(key, 0)
    return {
        "name": name,
        "budget": budget,
        "requests": len(examples),
        "all_hits": all_hits,
        "all_positions": total_positions,
        "all_hit_pct": pct(all_hits, total_positions),
        "all_bytes_hit": all_bytes_hit,
        "all_bytes_total": total_bytes,
        "all_bytes_hit_pct": pct(all_bytes_hit, total_bytes),
    }


def evaluate_selected(name: str, examples: list[RequestExample], selected: list[set[int]], budget: int) -> dict[str, Any]:
    all_hits = 0
    all_bytes_hit = 0
    cold_hits = 0
    cold_bytes_hit = 0
    total_positions = 0
    total_bytes = 0
    cold_positions = 0
    cold_total_bytes = 0
    recorded_hcs_hits = 0
    recorded_hcs_bytes = 0
    selected_entries = 0

    for example, selected_keys in zip(examples, selected, strict=True):
        selected_entries += len(selected_keys)
        total_positions += example.total_positions
        total_bytes += example.total_bytes
        cold_positions += example.cold_positions
        cold_total_bytes += example.cold_total_bytes
        recorded_hcs_hits += example.recorded_hcs_hits
        recorded_hcs_bytes += example.recorded_hcs_bytes
        for key in selected_keys:
            all_hits += example.all_counts.get(key, 0)
            all_bytes_hit += example.all_bytes.get(key, 0)
            cold_hits += example.cold_counts.get(key, 0)
            cold_bytes_hit += example.cold_bytes.get(key, 0)

    return {
        "name": name,
        "budget": budget,
        "requests": len(examples),
        "selected_entries": selected_entries,
        "all_hits": all_hits,
        "all_positions": total_positions,
        "all_hit_pct": pct(all_hits, total_positions),
        "all_bytes_hit": all_bytes_hit,
        "all_bytes_total": total_bytes,
        "all_bytes_hit_pct": pct(all_bytes_hit, total_bytes),
        "cold_hits": cold_hits,
        "cold_positions": cold_positions,
        "cold_hit_pct": pct(cold_hits, cold_positions),
        "cold_bytes_hit": cold_bytes_hit,
        "cold_bytes_total": cold_total_bytes,
        "cold_bytes_reduction_pct": pct(cold_bytes_hit, cold_total_bytes),
        "recorded_hcs_hits": recorded_hcs_hits,
        "recorded_hcs_hit_pct": pct(recorded_hcs_hits, total_positions),
        "recorded_hcs_bytes_hit": recorded_hcs_bytes,
        "recorded_hcs_bytes_hit_pct": pct(recorded_hcs_bytes, total_bytes),
    }


def evaluate_oracle(examples: list[RequestExample], budget: int, attr: str) -> dict[str, Any]:
    selected = []
    for example in examples:
        values = getattr(example, attr)
        ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        selected.append(set(int(key) for key, _ in ranked[:budget]))
    return evaluate_selected(f"oracle_{attr}", examples, selected, budget)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    examples: list[RequestExample],
    features: torch.Tensor,
    budgets: list[int],
    device: torch.device,
    batch_size: int,
    name: str,
    score_bias: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    scores = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].to(device)
        batch_scores = model(batch).cpu()
        if score_bias is not None:
            batch_scores = batch_scores + score_bias
        scores.append(batch_scores)
    all_scores = torch.cat(scores, dim=0)
    return [
        evaluate_selected(name, examples, select_top(all_scores, budget), budget)
        for budget in budgets
    ]


def train_model(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_values: torch.Tensor,
    hidden: int | None,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    score_bias: torch.Tensor | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = Planner(train_features.shape[1], train_labels.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == "bce":
        positives = train_labels.sum()
        total = train_labels.numel()
        negatives = total - positives
        pos_weight = min(args.max_pos_weight, max(1.0, float(negatives / positives.clamp_min(1.0))))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((train_labels.shape[1],), pos_weight, device=device))
        train_targets = train_labels
    elif args.loss == "value-kl":
        row_sums = train_values.sum(dim=1, keepdim=True).clamp_min(1.0)
        train_targets = train_values / row_sums
        criterion = None
    else:
        raise ValueError(f"unknown loss {args.loss!r}")
    dataset = TensorDataset(train_features, train_targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if score_bias is not None:
                logits = logits + score_bias.to(device)
            if args.loss == "value-kl":
                log_probs = torch.log_softmax(logits, dim=1)
                loss = -(yb * log_probs).sum(dim=1).mean()
            else:
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        history.append({"epoch": epoch, "train_loss": loss_total / max(1, seen)})
    return model, history


def train_candidate_ranker(
    train_candidates: torch.Tensor,
    train_values: torch.Tensor,
    hidden: int | None,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    score_bias: torch.Tensor | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = CandidateRanker(train_candidates.shape[2], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    history = []
    request_indices = torch.arange(train_candidates.shape[0])
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = request_indices[torch.randperm(len(request_indices), generator=generator)]
        loss_total = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            xb = train_candidates[batch_idx].to(device)
            yb = train_values[batch_idx].to(device)
            row_sums = yb.sum(dim=1, keepdim=True)
            keep = row_sums.squeeze(1) > 0
            if not bool(keep.any()):
                continue
            xb = xb[keep]
            yb = yb[keep] / row_sums[keep].clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape[0], xb.shape[1])
            if score_bias is not None:
                logits = logits + score_bias.to(device)
            loss = -(yb * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        history.append({"epoch": epoch, "train_loss": loss_total / max(1, seen)})
    return model, history


@torch.no_grad()
def score_candidate_ranker(
    model: nn.Module,
    candidates: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    rows = []
    flat = candidates.reshape(-1, candidates.shape[-1])
    for start in range(0, flat.shape[0], max(1, batch_size * candidates.shape[1])):
        batch = flat[start : start + max(1, batch_size * candidates.shape[1])].to(device)
        rows.append(model(batch).cpu())
    return torch.cat(rows, dim=0).reshape(candidates.shape[0], candidates.shape[1])


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def write_summary(path: pathlib.Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_model_checkpoint(
    path: pathlib.Path,
    model: nn.Module,
    *,
    name: str,
    hidden: int | None,
    args: argparse.Namespace,
    train_meta: dict[str, Any],
    parameter_count_value: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "krasis_hcs_planner_checkpoint_v1",
            "name": name,
            "hidden": hidden,
            "parameters": parameter_count_value,
            "state_dict": model.state_dict(),
            "input_mode": args.input_mode,
            "input_steps": args.input_steps,
            "token_hash_bins": args.token_hash_bins,
            "target_start_step": args.target_start_step,
            "target_steps": args.target_steps,
            "target_cold_only": args.target_cold_only,
            "loss": args.loss,
            "planner_kind": args.planner_kind,
            "train_meta": train_meta,
        },
        path,
    )


def build_examples_from_traces(
    traces: list[pathlib.Path],
    args: argparse.Namespace,
) -> tuple[list[RequestExample], dict[str, Any], dict[tuple[str, int, str], dict[str, Any]]]:
    routes, meta, predecode = read_trace_requests(traces, args.request_label, args.request_label_prefix)
    examples = build_examples(
        routes,
        meta,
        predecode,
        args.input_mode,
        args.input_steps,
        args.target_steps,
        args.target_start_step,
        args.target_cold_only,
        args.token_hash_bins,
    )
    return examples, meta, predecode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-traces", nargs="+", type=pathlib.Path)
    parser.add_argument("--valid-traces", nargs="+", type=pathlib.Path)
    parser.add_argument("--train-datasets", nargs="+", type=pathlib.Path)
    parser.add_argument("--valid-datasets", nargs="+", type=pathlib.Path)
    parser.add_argument("--dataset-out", type=pathlib.Path)
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-prefix", default="hcs_planner")
    parser.add_argument("--request-label")
    parser.add_argument("--request-label-prefix")
    parser.add_argument("--input-mode", choices=["decode_steps", "predecode"], default="decode_steps")
    parser.add_argument("--input-steps", type=int, default=0)
    parser.add_argument("--token-hash-bins", type=int, default=2048)
    parser.add_argument("--target-start-step", type=int, default=1)
    parser.add_argument("--target-steps", type=int, default=48)
    parser.add_argument("--target-cold-only", action="store_true")
    parser.add_argument("--budgets", default="426,852,1277,1703")
    parser.add_argument("--hidden-sizes", default="linear,512,2048")
    parser.add_argument("--planner-kind", choices=["dense", "candidate-ranker"], default="dense")
    parser.add_argument(
        "--layer-groups",
        type=int,
        default=8,
        help="Number of layer groups for --planner-kind candidate-ranker.",
    )
    parser.add_argument(
        "--ranker-prior-scale",
        type=float,
        default=1.0,
        help="Static all-bytes log-prior scale used to calibrate candidate-ranker scores across groups.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--loss", choices=["bce", "value-kl"], default="value-kl")
    parser.add_argument(
        "--prior",
        choices=["none", "all_bytes", "cold_bytes"],
        default="none",
        help="Train/evaluate neural residual scores on top of a static train-set prior.",
    )
    parser.add_argument(
        "--prior-scales",
        default="1.0",
        help="Comma-separated multipliers for the static-prior log-score bias.",
    )
    parser.add_argument(
        "--prior-smoothing",
        type=float,
        default=1.0,
        help="Non-negative additive smoothing used when converting static scores to log priors.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    if args.input_steps < 0:
        raise ValueError("--input-steps must be non-negative")
    if args.target_start_step <= 0 or args.target_steps <= 0:
        raise ValueError("--target-start-step and --target-steps must be positive")
    if args.request_label and args.request_label_prefix:
        raise ValueError("use only one of --request-label or --request-label-prefix")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.layer_groups <= 0:
        raise ValueError("--layer-groups must be positive")
    if args.input_mode == "predecode" and args.token_hash_bins <= 0:
        raise ValueError("--token-hash-bins must be positive for --input-mode predecode")
    if not (0.0 < args.valid_ratio < 1.0):
        raise ValueError("--valid-ratio must be between 0 and 1")
    if args.train_traces and args.train_datasets:
        raise ValueError("use either --train-traces or --train-datasets, not both")
    if args.valid_traces and args.valid_datasets:
        raise ValueError("use either --valid-traces or --valid-datasets, not both")
    if args.dataset_only and args.train_datasets:
        raise ValueError("--dataset-only requires --train-traces, not --train-datasets")
    if args.dataset_only and args.dataset_out is None:
        raise ValueError("--dataset-only requires --dataset-out")
    if not args.train_traces and not args.train_datasets:
        raise ValueError("one of --train-traces or --train-datasets is required")

    budgets = parse_int_list(args.budgets)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    prior_scales = parse_float_list(args.prior_scales)
    if args.prior == "none":
        prior_scales = [0.0]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    if args.train_traces:
        train_examples, train_meta, _train_predecode = build_examples_from_traces(args.train_traces, args)
        if args.dataset_out is not None:
            meta = dataset_meta(
                trace_paths=args.train_traces,
                filter_args=args,
                trace_meta=train_meta,
                examples=train_examples,
            )
            write_dataset(args.dataset_out, meta, train_examples)
        if args.dataset_only:
            print(json.dumps({"dataset": str(args.dataset_out), "examples": len(train_examples)}))
            return 0
    else:
        loaded_train = load_heatmap_datasets(args.train_datasets or [])
        train_examples = loaded_train.examples
        trace_meta = loaded_train.meta.get("trace_meta", {})
        train_meta = {
            "sources": loaded_train.sources,
            "trace_routes": trace_meta.get("trace_routes", 0),
            "predecode_records": trace_meta.get("predecode_records", 0),
            "filtered_predecode_records": trace_meta.get("filtered_predecode_records", 0),
            "filtered_routes": trace_meta.get("filtered_routes", 0),
            "bad_json_lines": trace_meta.get("bad_json_lines", 0),
            "num_layers": trace_meta.get("num_layers", 0),
            "num_experts": trace_meta.get("num_experts", 0),
            "topk": trace_meta.get("topk", 0),
        }

    split_mode = "explicit_valid_traces"
    if args.valid_datasets:
        loaded_valid = load_heatmap_datasets(args.valid_datasets)
        valid_examples = loaded_valid.examples
        valid_trace_meta = loaded_valid.meta.get("trace_meta", {})
        for key in ("num_layers", "num_experts", "topk"):
            if train_meta.get(key) != valid_trace_meta.get(key):
                raise ValueError(f"train/valid dataset metadata mismatch for {key}: {train_meta.get(key)} != {valid_trace_meta.get(key)}")
        split_mode = "explicit_valid_datasets"
    elif args.valid_traces:
        valid_examples, valid_meta, _valid_predecode = build_examples_from_traces(args.valid_traces, args)
        for key in ("num_layers", "num_experts", "topk"):
            if train_meta[key] != valid_meta[key]:
                raise ValueError(f"train/valid metadata mismatch for {key}: {train_meta[key]} != {valid_meta[key]}")
    else:
        train_examples, valid_examples, split_mode = split_examples(train_examples, args.valid_ratio, args.seed)

    if not train_examples or not valid_examples:
        raise ValueError("empty train or validation examples")

    train_features, train_labels, train_values = examples_to_tensors(train_examples)
    valid_features, _valid_labels, _valid_values = examples_to_tensors(valid_examples)

    static_scores = global_rank_from_examples(train_examples, "all_bytes")
    static_cold_scores = global_rank_from_examples(train_examples, "cold_bytes")
    prior_log_scores = None
    if args.prior != "none":
        prior_log_scores = static_prior_from_examples(train_examples, args.prior, args.prior_smoothing)
    baselines = []
    for budget in budgets:
        static_selected = [set(torch.topk(static_scores, k=min(budget, static_scores.numel())).indices.cpu().tolist()) for _ in valid_examples]
        static_cold_selected = [set(torch.topk(static_cold_scores, k=min(budget, static_cold_scores.numel())).indices.cpu().tolist()) for _ in valid_examples]
        baselines.append(evaluate_selected("static_train_all_bytes", valid_examples, static_selected, budget))
        baselines.append(evaluate_selected("static_train_cold_bytes", valid_examples, static_cold_selected, budget))
        baselines.append(evaluate_oracle(valid_examples, budget, "all_bytes"))
        baselines.append(evaluate_oracle(valid_examples, budget, "cold_bytes"))

    models = []
    layer_groups: list[dict[str, Any]] = []
    if args.planner_kind == "dense":
        for scale_index, prior_scale in enumerate(prior_scales):
            score_bias = None
            prior_label = "none"
            if prior_log_scores is not None:
                score_bias = prior_log_scores * float(prior_scale)
                prior_label = f"{args.prior}_x{prior_scale:g}"
            for model_index, hidden in enumerate(hidden_sizes):
                seed = args.seed + scale_index * 10_000_019 + model_index * 1_000_003 + (hidden or 0)
                model, history = train_model(
                    train_features,
                    train_labels,
                    train_values,
                    hidden,
                    args,
                    device,
                    seed,
                    score_bias=score_bias,
                )
                base_name = "linear" if hidden is None else f"mlp_h{hidden}"
                name = base_name if args.prior == "none" else f"{base_name}_prior_{prior_label}"
                model_metrics = evaluate_model(
                    model,
                    valid_examples,
                    valid_features,
                    budgets,
                    device,
                    args.batch_size,
                    name,
                    score_bias=score_bias,
                )
                model_path = None
                params = parameter_count(model)
                if args.save_models:
                    model_path = args.out_dir / f"{args.model_prefix}_{name}.pt"
                    save_model_checkpoint(
                        model_path,
                        model,
                        name=name,
                        hidden=hidden,
                        args=args,
                        train_meta=train_meta,
                        parameter_count_value=params,
                    )
                models.append(
                    {
                        "name": name,
                        "hidden": hidden,
                        "parameters": params,
                        "checkpoint": str(model_path) if model_path is not None else None,
                        "prior": args.prior,
                        "prior_scale": prior_scale,
                        "prior_smoothing": args.prior_smoothing,
                        "history": history,
                        "metrics": model_metrics,
                    }
                )
    else:
        num_layers = int(train_meta["num_layers"])
        num_experts = int(train_meta["num_experts"])
        output_dim = num_layers * num_experts
        groups = make_layer_groups(num_layers, args.layer_groups)
        prior_all_log = static_prior_from_examples(train_examples, "all_bytes", args.prior_smoothing)
        prior_cold_log = static_prior_from_examples(train_examples, "cold_bytes", args.prior_smoothing)
        max_budget = max(budgets)
        for model_index, hidden in enumerate(hidden_sizes):
            all_scores = torch.full((len(valid_examples), output_dim), -1.0e30, dtype=torch.float32)
            histories = []
            group_metrics = []
            total_params = 0
            for group_idx, (start_layer, end_layer) in enumerate(groups):
                keys = list(range(start_layer * num_experts, end_layer * num_experts))
                group_key_set = set(keys)
                train_candidates, train_group_values = build_candidate_tensor(
                    train_examples,
                    keys,
                    num_layers,
                    num_experts,
                    static_scores,
                    static_cold_scores,
                    prior_all_log,
                    prior_cold_log,
                )
                valid_candidates, _valid_group_values = build_candidate_tensor(
                    valid_examples,
                    keys,
                    num_layers,
                    num_experts,
                    static_scores,
                    static_cold_scores,
                    prior_all_log,
                    prior_cold_log,
                )
                group_bias = prior_all_log[keys] * float(args.ranker_prior_scale)
                seed = args.seed + model_index * 1_000_003 + group_idx * 1009 + (hidden or 0)
                model, history = train_candidate_ranker(
                    train_candidates,
                    train_group_values,
                    hidden,
                    args,
                    device,
                    seed,
                    score_bias=group_bias,
                )
                group_scores = score_candidate_ranker(model, valid_candidates, device, args.batch_size) + group_bias.reshape(1, len(keys))
                all_scores[:, keys] = group_scores
                total_params += parameter_count(model)
                histories.append(
                    {
                        "group": group_idx,
                        "layers": [start_layer, end_layer],
                        "history": history,
                    }
                )
                group_budget = max(1, min(len(keys), round(max_budget * len(keys) / output_dim)))
                static_group_scores = torch.full((len(valid_examples), output_dim), -1.0e30, dtype=torch.float32)
                static_group_scores[:, keys] = static_scores[keys].reshape(1, len(keys)).repeat(len(valid_examples), 1)
                neural_group_scores = torch.full((len(valid_examples), output_dim), -1.0e30, dtype=torch.float32)
                neural_group_scores[:, keys] = group_scores
                oracle_selected = []
                for example in valid_examples:
                    ranked = sorted(
                        ((key, example.all_bytes.get(key, 0)) for key in keys),
                        key=lambda item: (-item[1], item[0]),
                    )
                    oracle_selected.append(set(key for key, value in ranked[:group_budget] if value > 0))
                group_metrics.append(
                    {
                        "group": group_idx,
                        "layers": [start_layer, end_layer],
                        "budget": group_budget,
                        "static_train_all_bytes": evaluate_group_selected(
                            f"group_{group_idx}_static_train_all_bytes",
                            valid_examples,
                            select_top(static_group_scores, group_budget),
                            group_budget,
                            group_key_set,
                        ),
                        "candidate_ranker": evaluate_group_selected(
                            f"group_{group_idx}_candidate_ranker",
                            valid_examples,
                            select_top(neural_group_scores, group_budget),
                            group_budget,
                            group_key_set,
                        ),
                        "oracle_all_bytes": evaluate_group_selected(
                            f"group_{group_idx}_oracle_all_bytes",
                            valid_examples,
                            oracle_selected,
                            group_budget,
                            group_key_set,
                        ),
                    }
                )
            base_name = "linear" if hidden is None else f"mlp_h{hidden}"
            name = f"candidate_ranker_g{len(groups)}_{base_name}"
            model_metrics = [
                evaluate_selected(name, valid_examples, select_top(all_scores, budget), budget)
                for budget in budgets
            ]
            models.append(
                {
                    "name": name,
                    "hidden": hidden,
                    "parameters": total_params,
                    "candidate_feature_dim": candidate_feature_dim(),
                    "layer_group_count": len(groups),
                    "ranker_prior_scale": args.ranker_prior_scale,
                    "history": histories,
                    "metrics": model_metrics,
                    "layer_group_metrics": group_metrics,
                }
            )
        layer_groups = [
            {"group": idx, "layers": [start, end], "experts": (end - start) * num_experts}
            for idx, (start, end) in enumerate(groups)
        ]

    summary = {
        "schema": "krasis_hcs_planner_training_summary_v1",
        "train_traces": [str(path) for path in args.train_traces] if args.train_traces else None,
        "valid_traces": [str(path) for path in args.valid_traces] if args.valid_traces else None,
        "train_datasets": [str(path) for path in args.train_datasets] if args.train_datasets else None,
        "valid_datasets": [str(path) for path in args.valid_datasets] if args.valid_datasets else None,
        "filter": {
            "request_label": args.request_label,
            "request_label_prefix": args.request_label_prefix,
        },
        "input_mode": args.input_mode,
        "input_steps": args.input_steps,
        "token_hash_bins": args.token_hash_bins,
        "target_start_step": args.target_start_step,
        "target_steps": args.target_steps,
        "target_cold_only": args.target_cold_only,
        "budgets": budgets,
        "split_mode": split_mode,
        "train_meta": train_meta,
        "train_requests": len(train_examples),
        "valid_requests": len(valid_examples),
        "feature_dim": int(train_features.shape[1]),
        "output_dim": int(train_labels.shape[1]),
        "loss": args.loss,
        "planner_kind": args.planner_kind,
        "layer_groups": layer_groups,
        "ranker_prior_scale": args.ranker_prior_scale,
        "prior": args.prior,
        "prior_scales": prior_scales,
        "prior_smoothing": args.prior_smoothing,
        "recorded_current_hcs": evaluate_selected(
            "recorded_current_hcs",
            valid_examples,
            [set() for _ in valid_examples],
            0,
        ),
        "baselines": baselines,
        "models": models,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.model_prefix}_summary.json"
    write_summary(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "train_requests": len(train_examples), "valid_requests": len(valid_examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
