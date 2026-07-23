#!/usr/bin/env python3
"""Train one-token route predictors from Krasis predecode traces."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print(
        "route_mtp_train.py must be run through ./dev route-mtp-train",
        file=sys.stderr,
    )
    sys.exit(2)


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - environment gate
    print(f"torch is required for route-MTP training: {exc}", file=sys.stderr)
    sys.exit(2)


TRACE_SCHEMA = "krasis_expert_prefetch_trace_v1"


@dataclass
class RouteMtpExample:
    source: str
    request_seq: int
    request_label: str
    token_ids: list[int]
    route_events: list[tuple[int, int, int, int, int, int]]
    targets: list[list[int]]
    target_weights: list[list[float]]
    final_prompt_routes: list[list[int]]


SOURCE_FINAL = 1
SOURCE_HEAD = 2
SOURCE_TAIL = 3
SOURCE_RECENCY_COUNT = 4
SOURCE_RECENCY_WEIGHT = 5
SOURCE_BUCKET_COUNT = 6
SOURCE_BUCKET_WEIGHT = 7
ROUTE_SOURCE_TYPES = 8
ROUTE_RANK_BUCKETS = 16
ROUTE_POS_BUCKETS = 8192
ROUTE_VALUE_BUCKETS = 32


class RouteMtpModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_buckets: int,
        seq_len: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        num_layers: int,
        num_experts: int,
        dropout: float,
    ):
        super().__init__()
        self.vocab_buckets = vocab_buckets
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.token_emb = nn.Embedding(vocab_buckets + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.route_head = nn.Linear(d_model * 2, num_layers * num_experts)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        pos = torch.arange(seq_len, device=tokens.device).reshape(1, seq_len)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        padding_mask = ~mask.bool()
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        lengths = mask.sum(dim=1).clamp(min=1)
        positions = torch.arange(seq_len, device=tokens.device).reshape(1, seq_len)
        last_index = (mask.long() * positions).max(dim=1).values
        batch_index = torch.arange(batch, device=tokens.device)
        last = encoded[batch_index, last_index, :]
        mean = (encoded * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(-1).float()
        pooled = torch.cat([last, mean], dim=-1)
        return self.route_head(pooled).reshape(batch, self.num_layers, self.num_experts)


class RouteStateModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_buckets: int,
        token_seq_len: int,
        route_seq_len: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        num_layers: int,
        num_experts: int,
        dropout: float,
    ):
        super().__init__()
        self.vocab_buckets = vocab_buckets
        self.token_seq_len = token_seq_len
        self.route_seq_len = route_seq_len
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.token_emb = nn.Embedding(vocab_buckets + 1, d_model, padding_idx=0)
        self.token_pos_emb = nn.Embedding(token_seq_len, d_model)
        self.token_proj = nn.Linear(d_model * 2, d_model)

        self.route_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.source_emb = nn.Embedding(ROUTE_SOURCE_TYPES + 1, d_model, padding_idx=0)
        self.layer_emb = nn.Embedding(num_layers + 1, d_model, padding_idx=0)
        self.expert_emb = nn.Embedding(num_experts + 1, d_model, padding_idx=0)
        self.rank_emb = nn.Embedding(ROUTE_RANK_BUCKETS + 1, d_model, padding_idx=0)
        self.route_pos_emb = nn.Embedding(ROUTE_POS_BUCKETS + 1, d_model, padding_idx=0)
        self.value_emb = nn.Embedding(ROUTE_VALUE_BUCKETS + 1, d_model, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.route_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.teacher_layer_query = nn.Embedding(num_layers, d_model)
        self.expert_weight = nn.Parameter(torch.empty(num_layers, d_model, num_experts))
        self.expert_bias = nn.Parameter(torch.zeros(num_layers, num_experts))
        nn.init.trunc_normal_(self.expert_weight, std=0.02)

    def token_summary(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        pos = torch.arange(seq_len, device=tokens.device).reshape(1, seq_len)
        x = self.token_emb(tokens) + self.token_pos_emb(pos)
        lengths = mask.sum(dim=1).clamp(min=1)
        positions = torch.arange(seq_len, device=tokens.device).reshape(1, seq_len)
        last_index = (mask.long() * positions).max(dim=1).values
        batch_index = torch.arange(batch, device=tokens.device)
        last = x[batch_index, last_index, :]
        mean = (x * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(-1).float()
        return self.token_proj(torch.cat([last, mean], dim=-1))

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        route_events: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        route_events = route_events.long()
        source = route_events[..., 0].clamp(0, ROUTE_SOURCE_TYPES)
        layer = route_events[..., 1].clamp(0, self.num_layers)
        expert = route_events[..., 2].clamp(0, self.num_experts)
        rank = route_events[..., 3].clamp(0, ROUTE_RANK_BUCKETS)
        position = route_events[..., 4].clamp(0, ROUTE_POS_BUCKETS)
        value = route_events[..., 5].clamp(0, ROUTE_VALUE_BUCKETS)
        route_x = (
            self.source_emb(source)
            + self.layer_emb(layer)
            + self.expert_emb(expert)
            + self.rank_emb(rank)
            + self.route_pos_emb(position)
            + self.value_emb(value)
        )
        cls = self.route_cls.expand(batch, -1, -1)
        route_x = torch.cat([cls, route_x], dim=1)
        cls_mask = torch.ones((batch, 1), dtype=torch.bool, device=route_mask.device)
        route_mask_full = torch.cat([cls_mask, route_mask.bool()], dim=1)
        encoded = self.route_encoder(route_x, src_key_padding_mask=~route_mask_full)
        context = self.norm(encoded[:, 0, :] + self.token_summary(tokens, token_mask))
        layer_indices = torch.arange(self.num_layers, device=tokens.device)
        layer_states = context.unsqueeze(1) + self.teacher_layer_query(layer_indices).unsqueeze(0)
        return torch.einsum("bld,lde->ble", layer_states, self.expert_weight) + self.expert_bias


def normalize_request_label(label: str) -> str:
    for suffix in ("_nosse", "_sse"):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def route_matches(record: dict[str, Any], label: str | None, prefix: str | None) -> bool:
    request_label = normalize_request_label(str(record.get("request_label", "")))
    if label is not None and request_label != normalize_request_label(label):
        return False
    if prefix is not None and not request_label.startswith(prefix):
        return False
    return True


def trace_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("_source", "")),
        int(record.get("request_seq", 0)),
        normalize_request_label(str(record.get("request_label", ""))),
    )


def value_bucket_from_weight(weight: float) -> int:
    if not math.isfinite(weight):
        return 1
    return max(1, min(ROUTE_VALUE_BUCKETS, 1 + int(round(max(0.0, min(1.0, weight)) * 15.0))))


def value_bucket_from_count(value: float) -> int:
    if not math.isfinite(value) or value <= 0.0:
        return 1
    return max(1, min(ROUTE_VALUE_BUCKETS, 1 + int(math.log2(value + 1.0))))


def route_event(
    source: int,
    layer: int,
    expert: int,
    rank: int,
    position: int,
    value: int,
    *,
    num_layers: int,
    num_experts: int,
) -> tuple[int, int, int, int, int, int] | None:
    if not (0 <= layer < num_layers and 0 <= expert < num_experts):
        return None
    return (
        int(max(1, min(ROUTE_SOURCE_TYPES, source))),
        int(layer + 1),
        int(expert + 1),
        int(max(1, min(ROUTE_RANK_BUCKETS, rank + 1))),
        int(max(1, min(ROUTE_POS_BUCKETS, position + 1))),
        int(max(1, min(ROUTE_VALUE_BUCKETS, value))),
    )


def add_route_rows(
    events: list[tuple[int, int, int, int, int, int]],
    rows: list[dict[str, Any]],
    *,
    source: int,
    num_layers: int,
    num_experts: int,
    top_ranks: int,
    row_limit: int,
) -> None:
    if row_limit > 0:
        rows = rows[-row_limit:] if source == SOURCE_TAIL else rows[:row_limit]
    for row in rows:
        layer = int(row.get("layer", -1))
        token_index = int(row.get("token_index", 0))
        expert_ids = list(row.get("expert_ids", []))
        weights = list(row.get("weights", []))
        for rank, expert in enumerate(expert_ids[:top_ranks]):
            weight = float(weights[rank]) if rank < len(weights) else 1.0
            event = route_event(
                source,
                layer,
                int(expert),
                rank,
                token_index,
                value_bucket_from_weight(weight),
                num_layers=num_layers,
                num_experts=num_experts,
            )
            if event is not None:
                events.append(event)


def add_count_events(
    events: list[tuple[int, int, int, int, int, int]],
    rows: list[dict[str, Any]],
    *,
    source: int,
    group_field: str,
    value_field: str,
    num_layers: int,
    num_experts: int,
    topk_per_group: int,
) -> None:
    grouped: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        layer = int(row.get("layer", -1))
        expert = int(row.get("expert", -1))
        group = int(row.get(group_field, 0))
        value = float(row.get(value_field, 0.0))
        if 0 <= layer < num_layers and 0 <= expert < num_experts and value > 0.0:
            grouped[(group, layer)].append((value, expert))
    for (group, layer), values in grouped.items():
        for rank, (value, expert) in enumerate(sorted(values, key=lambda item: (-item[0], item[1]))[:topk_per_group]):
            event = route_event(
                source,
                layer,
                expert,
                rank,
                group,
                value_bucket_from_count(value),
                num_layers=num_layers,
                num_experts=num_experts,
            )
            if event is not None:
                events.append(event)


def build_route_state_events(
    pre: dict[str, Any],
    *,
    num_layers: int,
    num_experts: int,
    route_seq_len: int,
    route_event_top_ranks: int,
    final_route_top_ranks: int,
    route_head_rows: int,
    route_tail_rows: int,
    route_count_topk: int,
) -> list[tuple[int, int, int, int, int, int]]:
    summary_events: list[tuple[int, int, int, int, int, int]] = []
    head_events: list[tuple[int, int, int, int, int, int]] = []
    tail_events: list[tuple[int, int, int, int, int, int]] = []
    final_events: list[tuple[int, int, int, int, int, int]] = []

    if route_count_topk > 0:
        add_count_events(
            summary_events,
            list(pre.get("prompt_expert_recency_counts", [])),
            source=SOURCE_RECENCY_COUNT,
            group_field="window",
            value_field="count",
            num_layers=num_layers,
            num_experts=num_experts,
            topk_per_group=route_count_topk,
        )
        add_count_events(
            summary_events,
            list(pre.get("prompt_expert_recency_weight_sums", [])),
            source=SOURCE_RECENCY_WEIGHT,
            group_field="window",
            value_field="weight_sum",
            num_layers=num_layers,
            num_experts=num_experts,
            topk_per_group=route_count_topk,
        )
        add_count_events(
            summary_events,
            list(pre.get("prompt_expert_bucket_counts", [])),
            source=SOURCE_BUCKET_COUNT,
            group_field="bucket",
            value_field="count",
            num_layers=num_layers,
            num_experts=num_experts,
            topk_per_group=route_count_topk,
        )
        add_count_events(
            summary_events,
            list(pre.get("prompt_expert_bucket_weight_sums", [])),
            source=SOURCE_BUCKET_WEIGHT,
            group_field="bucket",
            value_field="weight_sum",
            num_layers=num_layers,
            num_experts=num_experts,
            topk_per_group=route_count_topk,
        )
    add_route_rows(
        head_events,
        list(pre.get("prompt_route_head", [])),
        source=SOURCE_HEAD,
        num_layers=num_layers,
        num_experts=num_experts,
        top_ranks=route_event_top_ranks,
        row_limit=route_head_rows,
    )
    add_route_rows(
        tail_events,
        list(pre.get("prompt_route_tail", [])),
        source=SOURCE_TAIL,
        num_layers=num_layers,
        num_experts=num_experts,
        top_ranks=route_event_top_ranks,
        row_limit=route_tail_rows,
    )
    add_route_rows(
        final_events,
        list(pre.get("final_token_routes", [])),
        source=SOURCE_FINAL,
        num_layers=num_layers,
        num_experts=num_experts,
        top_ranks=final_route_top_ranks,
        row_limit=0,
    )

    if route_seq_len <= 0:
        return []
    if len(summary_events) + len(head_events) + len(tail_events) + len(final_events) <= route_seq_len:
        return summary_events + head_events + tail_events + final_events
    final_keep = final_events[-route_seq_len:]
    remaining = max(0, route_seq_len - len(final_keep))
    tail_keep = tail_events[-remaining:] if remaining > 0 else []
    remaining = max(0, remaining - len(tail_keep))
    prefix = (summary_events + head_events)[:remaining] if remaining > 0 else []
    return prefix + tail_keep + final_keep


def read_route_mtp_examples(
    paths: list[pathlib.Path],
    *,
    request_label: str | None,
    request_label_prefix: str | None,
    target_step: int,
    seq_field: str,
    seq_len: int,
    token_stride: int,
    route_seq_len: int,
    route_event_top_ranks: int,
    final_route_top_ranks: int,
    route_head_rows: int,
    route_tail_rows: int,
    route_count_topk: int,
) -> tuple[list[RouteMtpExample], dict[str, Any]]:
    predecode: dict[tuple[str, int, str], dict[str, Any]] = {}
    route_rows: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    meta: dict[str, Any] = {
        "sources": [str(path) for path in paths],
        "predecode_records": 0,
        "filtered_predecode_records": 0,
        "route_records": 0,
        "filtered_route_records": 0,
        "target_route_records": 0,
        "bad_json_lines": 0,
        "num_layers": 0,
        "num_experts": 0,
        "topk": 0,
    }
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    meta["bad_json_lines"] += 1
                    continue
                if record.get("schema") != TRACE_SCHEMA:
                    continue
                record["_source"] = str(path)
                event = record.get("event")
                if event == "predecode":
                    meta["predecode_records"] += 1
                    if not route_matches(record, request_label, request_label_prefix):
                        continue
                    meta["filtered_predecode_records"] += 1
                    key = trace_key(record)
                    predecode[key] = record
                    meta["num_layers"] = max(meta["num_layers"], int(record.get("count_layers", 0)))
                    meta["num_experts"] = max(meta["num_experts"], int(record.get("count_experts_per_layer", 0)))
                    continue
                if event != "route":
                    continue
                meta["route_records"] += 1
                if not route_matches(record, request_label, request_label_prefix):
                    continue
                meta["filtered_route_records"] += 1
                if int(record.get("step", 0)) != target_step:
                    continue
                meta["target_route_records"] += 1
                key = trace_key(record)
                layer = int(record.get("layer", -1))
                if layer < 0:
                    continue
                route_rows[key][layer] = record
                meta["num_layers"] = max(meta["num_layers"], layer + 1)
                meta["num_experts"] = max(meta["num_experts"], int(record.get("num_experts", 0)))
                meta["topk"] = max(meta["topk"], int(record.get("topk", 0)))
    examples: list[RouteMtpExample] = []
    num_layers = int(meta["num_layers"])
    for key, pre in predecode.items():
        by_layer = route_rows.get(key)
        if not by_layer or len(by_layer) < num_layers:
            continue
        raw_tokens = [int(tok) for tok in list(pre.get(seq_field, []))]
        if token_stride > 1:
            raw_tokens = raw_tokens[::token_stride]
        tokens = raw_tokens[-seq_len:]
        route_events = build_route_state_events(
            pre,
            num_layers=num_layers,
            num_experts=int(meta["num_experts"]),
            route_seq_len=route_seq_len,
            route_event_top_ranks=route_event_top_ranks,
            final_route_top_ranks=final_route_top_ranks,
            route_head_rows=route_head_rows,
            route_tail_rows=route_tail_rows,
            route_count_topk=route_count_topk,
        )
        targets: list[list[int]] = []
        target_weights: list[list[float]] = []
        final_prompt_routes: list[list[int]] = [[] for _ in range(num_layers)]
        for item in pre.get("final_token_routes", []):
            layer = int(item.get("layer", -1))
            if 0 <= layer < num_layers:
                final_prompt_routes[layer] = [int(expert) for expert in list(item.get("expert_ids", []))]
        complete = True
        for layer in range(num_layers):
            row = by_layer.get(layer)
            if row is None:
                complete = False
                break
            topk = int(row.get("topk", 0))
            targets.append([int(expert) for expert in list(row.get("expert_ids", []))[:topk]])
            target_weights.append([float(weight) for weight in list(row.get("weights", []))[:topk]])
        if not complete:
            continue
        examples.append(
            RouteMtpExample(
                source=key[0],
                request_seq=key[1],
                request_label=key[2],
                token_ids=tokens,
                route_events=route_events,
                targets=targets,
                target_weights=target_weights,
                final_prompt_routes=final_prompt_routes,
            )
        )
    meta["examples"] = len(examples)
    if examples:
        event_lengths = [len(example.route_events) for example in examples]
        meta["route_event_min"] = min(event_lengths)
        meta["route_event_max"] = max(event_lengths)
        meta["route_event_mean"] = sum(event_lengths) / len(event_lengths)
    return examples, meta


def encode_examples(
    examples: list[RouteMtpExample],
    *,
    seq_len: int,
    vocab_buckets: int,
    num_layers: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.zeros((len(examples), seq_len), dtype=torch.long)
    mask = torch.zeros((len(examples), seq_len), dtype=torch.bool)
    labels = torch.zeros((len(examples), num_layers, num_experts), dtype=torch.float32)
    weights = torch.zeros((len(examples), num_layers, num_experts), dtype=torch.float32)
    for row, example in enumerate(examples):
        clipped = example.token_ids[-seq_len:]
        offset = seq_len - len(clipped)
        for idx, token_id in enumerate(clipped):
            tokens[row, offset + idx] = int(token_id) % vocab_buckets + 1
            mask[row, offset + idx] = True
        for layer, experts in enumerate(example.targets):
            for rank, expert in enumerate(experts):
                if 0 <= expert < num_experts:
                    labels[row, layer, expert] = 1.0
                    layer_weights = example.target_weights[layer]
                    weights[row, layer, expert] = float(layer_weights[rank]) if rank < len(layer_weights) else 1.0
    return tokens, mask, labels, weights


def encode_route_state_examples(
    examples: list[RouteMtpExample],
    *,
    token_seq_len: int,
    route_seq_len: int,
    vocab_buckets: int,
    num_layers: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.zeros((len(examples), token_seq_len), dtype=torch.long)
    token_mask = torch.zeros((len(examples), token_seq_len), dtype=torch.bool)
    route_events = torch.zeros((len(examples), route_seq_len, 6), dtype=torch.long)
    route_mask = torch.zeros((len(examples), route_seq_len), dtype=torch.bool)
    labels = torch.zeros((len(examples), num_layers, num_experts), dtype=torch.float32)
    weights = torch.zeros((len(examples), num_layers, num_experts), dtype=torch.float32)
    for row, example in enumerate(examples):
        clipped_tokens = example.token_ids[-token_seq_len:]
        token_offset = token_seq_len - len(clipped_tokens)
        for idx, token_id in enumerate(clipped_tokens):
            tokens[row, token_offset + idx] = int(token_id) % vocab_buckets + 1
            token_mask[row, token_offset + idx] = True
        clipped_events = example.route_events[-route_seq_len:]
        event_offset = route_seq_len - len(clipped_events)
        for idx, event in enumerate(clipped_events):
            route_events[row, event_offset + idx, :] = torch.tensor(event, dtype=torch.long)
            route_mask[row, event_offset + idx] = True
        for layer, experts in enumerate(example.targets):
            for rank, expert in enumerate(experts):
                if 0 <= expert < num_experts:
                    labels[row, layer, expert] = 1.0
                    layer_weights = example.target_weights[layer]
                    weights[row, layer, expert] = float(layer_weights[rank]) if rank < len(layer_weights) else 1.0
    return tokens, token_mask, route_events, route_mask, labels, weights


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def parse_arch(raw: str) -> dict[str, int | float | str]:
    parts = raw.split(":")
    if len(parts) != 7:
        raise ValueError("arch must be name:d_model:layers:heads:ff_mult:vocab_buckets:dropout")
    name, d_model, layers, heads, ff_mult, vocab_buckets, dropout = parts
    return {
        "name": name,
        "d_model": int(d_model),
        "layers": int(layers),
        "heads": int(heads),
        "ff_mult": int(ff_mult),
        "vocab_buckets": int(vocab_buckets),
        "dropout": float(dropout),
    }


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


def static_prior_logits(
    train_examples: list[RouteMtpExample],
    *,
    num_layers: int,
    num_experts: int,
) -> torch.Tensor:
    counts = torch.zeros((num_layers, num_experts), dtype=torch.float32)
    for example in train_examples:
        for layer, experts in enumerate(example.targets):
            for expert in experts:
                if 0 <= int(expert) < num_experts:
                    counts[layer, int(expert)] += 1.0
    probs = (counts + 1.0) / (counts.sum(dim=1, keepdim=True) + float(num_experts))
    return torch.log(probs)


def static_train_predictions(
    train_examples: list[RouteMtpExample],
    *,
    num_layers: int,
    num_experts: int,
    topk: int,
) -> list[list[int]]:
    counts: list[Counter[int]] = [Counter() for _ in range(num_layers)]
    for example in train_examples:
        for layer, experts in enumerate(example.targets):
            counts[layer].update(int(expert) for expert in experts if 0 <= expert < num_experts)
    predictions = []
    for layer in range(num_layers):
        ranked = sorted(counts[layer].items(), key=lambda item: (-item[1], item[0]))
        predictions.append([expert for expert, _count in ranked[:topk]])
    return predictions


def evaluate_prediction_sets(
    examples: list[RouteMtpExample],
    predictions: list[list[list[int]]],
    *,
    topk: int,
    name: str,
) -> dict[str, Any]:
    total_slots = 0
    hit_slots = 0
    exact_layers = 0
    total_layers = 0
    any_hit_layers = 0
    weighted_hit = 0.0
    weighted_total = 0.0
    for example, pred_layers in zip(examples, predictions, strict=True):
        for layer, actual in enumerate(example.targets):
            actual_set = set(int(expert) for expert in actual[:topk])
            predicted_set = set(int(expert) for expert in pred_layers[layer][:topk])
            total_slots += len(actual_set)
            hits = len(actual_set & predicted_set)
            hit_slots += hits
            total_layers += 1
            if hits > 0:
                any_hit_layers += 1
            if actual_set == predicted_set:
                exact_layers += 1
            for rank, expert in enumerate(actual[:topk]):
                layer_weights = example.target_weights[layer]
                weight = float(layer_weights[rank]) if rank < len(layer_weights) else 1.0
                weighted_total += weight
                if int(expert) in predicted_set:
                    weighted_hit += weight
    return {
        "name": name,
        "topk": topk,
        "route_recall_pct": 0.0 if total_slots == 0 else hit_slots / total_slots * 100.0,
        "weighted_route_recall_pct": 0.0 if weighted_total == 0.0 else weighted_hit / weighted_total * 100.0,
        "layer_exact_pct": 0.0 if total_layers == 0 else exact_layers / total_layers * 100.0,
        "layer_any_hit_pct": 0.0 if total_layers == 0 else any_hit_layers / total_layers * 100.0,
        "hit_slots": hit_slots,
        "total_slots": total_slots,
    }


def evaluate_static_baselines(
    train_examples: list[RouteMtpExample],
    valid_examples: list[RouteMtpExample],
    *,
    num_layers: int,
    num_experts: int,
    topk: int,
) -> list[dict[str, Any]]:
    static_layers = static_train_predictions(
        train_examples,
        num_layers=num_layers,
        num_experts=num_experts,
        topk=topk,
    )
    static_predictions = [static_layers for _ in valid_examples]
    final_predictions = [
        [
            [int(expert) for expert in example.final_prompt_routes[layer][:topk]]
            for layer in range(num_layers)
        ]
        for example in valid_examples
    ]
    return [
        evaluate_prediction_sets(valid_examples, static_predictions, topk=topk, name="static_train_step1_prior"),
        evaluate_prediction_sets(valid_examples, final_predictions, topk=topk, name="copy_final_prompt_routes"),
    ]


def model_predictions(
    model: RouteMtpModel,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    topk: int,
    prior_logits: torch.Tensor | None = None,
    prior_bias: float = 0.0,
) -> list[list[list[int]]]:
    model.eval()
    predictions: list[list[list[int]]] = []
    device_prior = prior_logits.to(device) if prior_logits is not None and prior_bias != 0.0 else None
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch_tokens = tokens[start : start + batch_size].to(device)
            batch_mask = mask[start : start + batch_size].to(device)
            logits = model(batch_tokens, batch_mask)
            if device_prior is not None:
                logits = logits + float(prior_bias) * device_prior.unsqueeze(0)
            top = torch.topk(logits, k=topk, dim=-1).indices.cpu().tolist()
            predictions.extend(top)
    return predictions


def route_state_model_predictions(
    model: RouteStateModel,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    route_events: torch.Tensor,
    route_mask: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    topk: int,
    prior_logits: torch.Tensor | None = None,
    prior_bias: float = 0.0,
) -> list[list[list[int]]]:
    model.eval()
    predictions: list[list[list[int]]] = []
    device_prior = prior_logits.to(device) if prior_logits is not None and prior_bias != 0.0 else None
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch_tokens = tokens[start : start + batch_size].to(device)
            batch_token_mask = token_mask[start : start + batch_size].to(device)
            batch_route_events = route_events[start : start + batch_size].to(device)
            batch_route_mask = route_mask[start : start + batch_size].to(device)
            logits = model(batch_tokens, batch_token_mask, batch_route_events, batch_route_mask)
            if device_prior is not None:
                logits = logits + float(prior_bias) * device_prior.unsqueeze(0)
            top = torch.topk(logits, k=topk, dim=-1).indices.cpu().tolist()
            predictions.extend(top)
    return predictions


def train_one_model(
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    valid_examples: list[RouteMtpExample],
    valid_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    arch: dict[str, int | float | str],
    seq_len: int,
    num_layers: int,
    num_experts: int,
    topk: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[RouteMtpModel, list[dict[str, float]], dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = RouteMtpModel(
        vocab_buckets=int(arch["vocab_buckets"]),
        seq_len=seq_len,
        d_model=int(arch["d_model"]),
        layers=int(arch["layers"]),
        heads=int(arch["heads"]),
        ff_mult=int(arch["ff_mult"]),
        num_layers=num_layers,
        num_experts=num_experts,
        dropout=float(arch["dropout"]),
    ).to(device)
    train_tokens, train_mask, train_labels, _train_weights = train_tensors
    positives = float(train_labels.sum().item())
    total = float(train_labels.numel())
    pos_weight_value = min(args.max_pos_weight, max(1.0, (total - positives) / max(1.0, positives)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = TensorDataset(train_tokens, train_mask, train_labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_tokens, batch_mask, batch_labels in loader:
            batch_tokens = batch_tokens.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_tokens, batch_mask)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch_tokens.shape[0])
            seen += int(batch_tokens.shape[0])
        if epoch == args.epochs - 1 or (epoch + 1) % args.eval_every == 0:
            predictions = model_predictions(
                model,
                valid_tokens,
                valid_mask,
                device=device,
                batch_size=args.batch_size,
                topk=topk,
            )
            metrics = evaluate_prediction_sets(
                valid_examples,
                predictions,
                topk=topk,
                name=str(arch["name"]),
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": 0.0 if seen == 0 else total_loss / float(seen),
                    "valid_route_recall_pct": float(metrics["route_recall_pct"]),
                    "valid_weighted_route_recall_pct": float(metrics["weighted_route_recall_pct"]),
                    "valid_layer_exact_pct": float(metrics["layer_exact_pct"]),
                    "valid_layer_any_hit_pct": float(metrics["layer_any_hit_pct"]),
                }
            )
    final_predictions = model_predictions(
        model,
        valid_tokens,
        valid_mask,
        device=device,
        batch_size=args.batch_size,
        topk=topk,
    )
    final_metrics = evaluate_prediction_sets(
        valid_examples,
        final_predictions,
        topk=topk,
        name=str(arch["name"]),
    )
    return model, history, final_metrics


def train_one_route_state_model(
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    valid_examples: list[RouteMtpExample],
    valid_tokens: torch.Tensor,
    valid_token_mask: torch.Tensor,
    valid_route_events: torch.Tensor,
    valid_route_mask: torch.Tensor,
    *,
    arch: dict[str, int | float | str],
    token_seq_len: int,
    route_seq_len: int,
    num_layers: int,
    num_experts: int,
    topk: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[RouteStateModel, list[dict[str, float]], dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = RouteStateModel(
        vocab_buckets=int(arch["vocab_buckets"]),
        token_seq_len=token_seq_len,
        route_seq_len=route_seq_len,
        d_model=int(arch["d_model"]),
        layers=int(arch["layers"]),
        heads=int(arch["heads"]),
        ff_mult=int(arch["ff_mult"]),
        num_layers=num_layers,
        num_experts=num_experts,
        dropout=float(arch["dropout"]),
    ).to(device)
    train_tokens, train_token_mask, train_route_events, train_route_mask, train_labels, _train_weights = train_tensors
    positives = float(train_labels.sum().item())
    total = float(train_labels.numel())
    pos_weight_value = min(args.max_pos_weight, max(1.0, (total - positives) / max(1.0, positives)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = TensorDataset(train_tokens, train_token_mask, train_route_events, train_route_mask, train_labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_tokens, batch_token_mask, batch_route_events, batch_route_mask, batch_labels in loader:
            batch_tokens = batch_tokens.to(device)
            batch_token_mask = batch_token_mask.to(device)
            batch_route_events = batch_route_events.to(device)
            batch_route_mask = batch_route_mask.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_tokens, batch_token_mask, batch_route_events, batch_route_mask)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch_tokens.shape[0])
            seen += int(batch_tokens.shape[0])
        if epoch == args.epochs - 1 or (epoch + 1) % args.eval_every == 0:
            predictions = route_state_model_predictions(
                model,
                valid_tokens,
                valid_token_mask,
                valid_route_events,
                valid_route_mask,
                device=device,
                batch_size=args.batch_size,
                topk=topk,
            )
            metrics = evaluate_prediction_sets(
                valid_examples,
                predictions,
                topk=topk,
                name=str(arch["name"]),
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": 0.0 if seen == 0 else total_loss / float(seen),
                    "valid_route_recall_pct": float(metrics["route_recall_pct"]),
                    "valid_weighted_route_recall_pct": float(metrics["weighted_route_recall_pct"]),
                    "valid_layer_exact_pct": float(metrics["layer_exact_pct"]),
                    "valid_layer_any_hit_pct": float(metrics["layer_any_hit_pct"]),
                }
            )
    final_predictions = route_state_model_predictions(
        model,
        valid_tokens,
        valid_token_mask,
        valid_route_events,
        valid_route_mask,
        device=device,
        batch_size=args.batch_size,
        topk=topk,
    )
    final_metrics = evaluate_prediction_sets(
        valid_examples,
        final_predictions,
        topk=topk,
        name=str(arch["name"]),
    )
    return model, history, final_metrics


def save_checkpoint(
    path: pathlib.Path,
    model: RouteMtpModel,
    *,
    arch: dict[str, int | float | str],
    args: argparse.Namespace,
    train_meta: dict[str, Any],
    parameter_count_value: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "krasis_route_mtp_checkpoint_v1",
            "state_dict": model.state_dict(),
            "model_kind": args.model_kind,
            "arch": arch,
            "seq_len": args.seq_len,
            "route_seq_len": args.route_seq_len,
            "target_step": args.target_step,
            "seq_field": args.seq_field,
            "train_meta": train_meta,
            "parameter_count": parameter_count_value,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one-token route-MTP predictors from v4 traces")
    parser.add_argument("--train-traces", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--valid-traces", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-prefix", default="route_mtp")
    parser.add_argument("--request-label")
    parser.add_argument("--request-label-prefix")
    parser.add_argument("--target-step", type=int, default=1)
    parser.add_argument("--model-kind", choices=["token", "route-state"], default="token")
    parser.add_argument("--seq-field", choices=["last_token_ids", "first_token_ids"], default="last_token_ids")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--token-stride", type=int, default=1)
    parser.add_argument("--route-seq-len", type=int, default=4096)
    parser.add_argument("--route-event-top-ranks", type=int, default=2)
    parser.add_argument("--final-route-top-ranks", type=int, default=8)
    parser.add_argument("--route-head-rows", type=int, default=256)
    parser.add_argument("--route-tail-rows", type=int, default=1024)
    parser.add_argument("--route-count-topk", type=int, default=4)
    parser.add_argument(
        "--archs",
        default="10m:160:6:5:4:32768:0.05,50m:384:8:8:4:65536:0.05,100m:512:16:8:4:65536:0.05,200m:768:20:12:4:65536:0.05",
        help="Comma-separated name:d_model:layers:heads:ff_mult:vocab_buckets:dropout specs.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--prior-biases",
        default="0.0",
        help="Comma-separated static train-prior log-prob biases to add at evaluation time.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    if args.target_step <= 0:
        raise ValueError("--target-step must be positive")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.route_seq_len <= 0:
        raise ValueError("--route-seq-len must be positive")
    if args.token_stride <= 0:
        raise ValueError("--token-stride must be positive")
    if args.route_event_top_ranks <= 0:
        raise ValueError("--route-event-top-ranks must be positive")
    if args.final_route_top_ranks <= 0:
        raise ValueError("--final-route-top-ranks must be positive")
    if args.route_head_rows < 0 or args.route_tail_rows < 0:
        raise ValueError("--route-head-rows and --route-tail-rows must be non-negative")
    if args.route_count_topk < 0:
        raise ValueError("--route-count-topk must be non-negative")
    if args.request_label and args.request_label_prefix:
        raise ValueError("use only one of --request-label or --request-label-prefix")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    archs = [parse_arch(raw.strip()) for raw in args.archs.split(",") if raw.strip()]
    if not archs:
        raise ValueError("--archs must contain at least one architecture")
    prior_biases = parse_float_list(args.prior_biases)

    train_examples, train_meta = read_route_mtp_examples(
        args.train_traces,
        request_label=args.request_label,
        request_label_prefix=args.request_label_prefix,
        target_step=args.target_step,
        seq_field=args.seq_field,
        seq_len=args.seq_len,
        token_stride=args.token_stride,
        route_seq_len=args.route_seq_len,
        route_event_top_ranks=args.route_event_top_ranks,
        final_route_top_ranks=args.final_route_top_ranks,
        route_head_rows=args.route_head_rows,
        route_tail_rows=args.route_tail_rows,
        route_count_topk=args.route_count_topk,
    )
    valid_examples, valid_meta = read_route_mtp_examples(
        args.valid_traces,
        request_label=args.request_label,
        request_label_prefix=args.request_label_prefix,
        target_step=args.target_step,
        seq_field=args.seq_field,
        seq_len=args.seq_len,
        token_stride=args.token_stride,
        route_seq_len=args.route_seq_len,
        route_event_top_ranks=args.route_event_top_ranks,
        final_route_top_ranks=args.final_route_top_ranks,
        route_head_rows=args.route_head_rows,
        route_tail_rows=args.route_tail_rows,
        route_count_topk=args.route_count_topk,
    )
    for key in ("num_layers", "num_experts", "topk"):
        if int(train_meta.get(key, 0)) != int(valid_meta.get(key, 0)):
            raise ValueError(f"train/valid metadata mismatch for {key}: {train_meta.get(key)} != {valid_meta.get(key)}")
    if not train_examples or not valid_examples:
        raise ValueError("empty train or validation examples")

    num_layers = int(train_meta["num_layers"])
    num_experts = int(train_meta["num_experts"])
    topk = int(train_meta["topk"])
    baselines = evaluate_static_baselines(
        train_examples,
        valid_examples,
        num_layers=num_layers,
        num_experts=num_experts,
        topk=topk,
    )
    prior_logits = static_prior_logits(
        train_examples,
        num_layers=num_layers,
        num_experts=num_experts,
    )
    models: list[dict[str, Any]] = []
    for index, arch in enumerate(archs):
        seed = args.seed + index * 1_000_003 + int(arch["d_model"]) * 1009 + int(arch["layers"])
        if args.model_kind == "route-state":
            train_tensors = encode_route_state_examples(
                train_examples,
                token_seq_len=args.seq_len,
                route_seq_len=args.route_seq_len,
                vocab_buckets=int(arch["vocab_buckets"]),
                num_layers=num_layers,
                num_experts=num_experts,
            )
            (
                valid_tokens,
                valid_token_mask,
                valid_route_events,
                valid_route_mask,
                _valid_labels,
                _valid_weights,
            ) = encode_route_state_examples(
                valid_examples,
                token_seq_len=args.seq_len,
                route_seq_len=args.route_seq_len,
                vocab_buckets=int(arch["vocab_buckets"]),
                num_layers=num_layers,
                num_experts=num_experts,
            )
            model, history, metrics = train_one_route_state_model(
                train_tensors,
                valid_examples,
                valid_tokens,
                valid_token_mask,
                valid_route_events,
                valid_route_mask,
                arch=arch,
                token_seq_len=args.seq_len,
                route_seq_len=args.route_seq_len,
                num_layers=num_layers,
                num_experts=num_experts,
                topk=topk,
                args=args,
                device=device,
                seed=seed,
            )
            prior_bias_metrics = []
            for prior_bias in prior_biases:
                predictions = route_state_model_predictions(
                    model,
                    valid_tokens,
                    valid_token_mask,
                    valid_route_events,
                    valid_route_mask,
                    device=device,
                    batch_size=args.batch_size,
                    topk=topk,
                    prior_logits=prior_logits,
                    prior_bias=prior_bias,
                )
                blend_metrics = evaluate_prediction_sets(
                    valid_examples,
                    predictions,
                    topk=topk,
                    name=f"{arch['name']}+prior_{prior_bias:g}",
                )
                blend_metrics["prior_bias"] = prior_bias
                prior_bias_metrics.append(blend_metrics)
        else:
            train_tensors = encode_examples(
                train_examples,
                seq_len=args.seq_len,
                vocab_buckets=int(arch["vocab_buckets"]),
                num_layers=num_layers,
                num_experts=num_experts,
            )
            valid_tokens, valid_mask, _valid_labels, _valid_weights = encode_examples(
                valid_examples,
                seq_len=args.seq_len,
                vocab_buckets=int(arch["vocab_buckets"]),
                num_layers=num_layers,
                num_experts=num_experts,
            )
            model, history, metrics = train_one_model(
                train_tensors,
                valid_examples,
                valid_tokens,
                valid_mask,
                arch=arch,
                seq_len=args.seq_len,
                num_layers=num_layers,
                num_experts=num_experts,
                topk=topk,
                args=args,
                device=device,
                seed=seed,
            )
            prior_bias_metrics = []
            for prior_bias in prior_biases:
                predictions = model_predictions(
                    model,
                    valid_tokens,
                    valid_mask,
                    device=device,
                    batch_size=args.batch_size,
                    topk=topk,
                    prior_logits=prior_logits,
                    prior_bias=prior_bias,
                )
                blend_metrics = evaluate_prediction_sets(
                    valid_examples,
                    predictions,
                    topk=topk,
                    name=f"{arch['name']}+prior_{prior_bias:g}",
                )
                blend_metrics["prior_bias"] = prior_bias
                prior_bias_metrics.append(blend_metrics)
        params = parameter_count(model)
        checkpoint = None
        if args.save_models:
            checkpoint_path = args.out_dir / f"{args.model_prefix}_{arch['name']}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                arch=arch,
                args=args,
                train_meta=train_meta,
                parameter_count_value=params,
            )
            checkpoint = str(checkpoint_path)
        models.append(
            {
                "name": arch["name"],
                "arch": arch,
                "parameters": params,
                "checkpoint": checkpoint,
                "history": history,
                "metrics": metrics,
                "prior_bias_metrics": prior_bias_metrics,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "schema": "krasis_route_mtp_training_summary_v1",
        "train_traces": [str(path) for path in args.train_traces],
        "valid_traces": [str(path) for path in args.valid_traces],
        "filter": {
            "request_label": args.request_label,
            "request_label_prefix": args.request_label_prefix,
        },
        "target_step": args.target_step,
        "model_kind": args.model_kind,
        "seq_field": args.seq_field,
        "seq_len": args.seq_len,
        "token_stride": args.token_stride,
        "route_seq_len": args.route_seq_len,
        "route_event_top_ranks": args.route_event_top_ranks,
        "final_route_top_ranks": args.final_route_top_ranks,
        "route_head_rows": args.route_head_rows,
        "route_tail_rows": args.route_tail_rows,
        "route_count_topk": args.route_count_topk,
        "prior_biases": prior_biases,
        "train_meta": train_meta,
        "valid_meta": valid_meta,
        "train_requests": len(train_examples),
        "valid_requests": len(valid_examples),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "topk": topk,
        "baselines": baselines,
        "models": models,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.model_prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"summary": str(summary_path), "train_requests": len(train_examples), "valid_requests": len(valid_examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
