#!/usr/bin/env python3
"""Build a static HCS decode heatmap prior from Krasis route traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from collections import defaultdict
from typing import Any


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print(
        "hcs_static_prior.py must be run through ./dev hcs-static-prior",
        file=sys.stderr,
    )
    sys.exit(2)


TRACE_SCHEMA = "krasis_expert_prefetch_trace_v1"
PRIOR_SCHEMA = "krasis_hcs_static_prior_v1"


try:
    from krasis.config import ModelConfig
except Exception as exc:  # pragma: no cover - environment gate
    print(f"krasis config import failed: {exc}", file=sys.stderr)
    sys.exit(2)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_fingerprints(model_path: pathlib.Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        path = model_path / name
        out[name] = sha256_file(path) if path.is_file() else None
    return out


def read_model_meta(model_path: pathlib.Path) -> dict[str, Any]:
    cfg = ModelConfig.from_model_path(str(model_path))
    return {
        "model_path": str(model_path.resolve()),
        "model_name": model_path.name,
        "model_type": cfg.model_type,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_moe_layers": cfg.num_moe_layers,
        "n_routed_experts": cfg.n_routed_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "num_full_attention_layers": cfg.num_full_attention_layers,
        "config_fingerprints": model_fingerprints(model_path),
    }


def route_label_ok(label: str, prefix: str | None) -> bool:
    if not prefix:
        return True
    return label.startswith(prefix)


def build_prior(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model_path.expanduser()
    model_meta = read_model_meta(model_path)
    expected_layers = int(model_meta["num_hidden_layers"])
    expected_experts = int(model_meta["n_routed_experts"])

    counts: dict[tuple[int, int], int] = defaultdict(int)
    weight_sums: dict[tuple[int, int], float] = defaultdict(float)
    per_trace: list[dict[str, Any]] = []
    request_keys: set[tuple[str, int, str]] = set()
    route_records = 0
    used_route_records = 0

    for trace in args.traces:
        trace = trace.expanduser()
        trace_records = 0
        trace_used = 0
        trace_requests: set[tuple[int, str]] = set()
        trace_predecode = 0
        with trace.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{trace}:{line_no}: invalid JSON: {exc}") from exc
                event = record.get("event")
                if event == "predecode":
                    if route_label_ok(str(record.get("request_label", "")), args.request_label_prefix):
                        trace_predecode += 1
                    continue
                if event != "route":
                    continue
                if record.get("schema") != TRACE_SCHEMA:
                    raise ValueError(f"{trace}:{line_no}: unsupported trace schema {record.get('schema')!r}")
                trace_records += 1
                route_records += 1
                label = str(record.get("request_label", ""))
                if not route_label_ok(label, args.request_label_prefix):
                    continue
                step = int(record.get("step", 0))
                if step < args.target_start_step:
                    continue
                if step >= args.target_start_step + args.target_steps:
                    continue
                layer = int(record["layer"])
                if layer < 0 or layer >= expected_layers:
                    raise ValueError(f"{trace}:{line_no}: layer {layer} outside 0..{expected_layers - 1}")
                experts = list(record.get("expert_ids", []))
                weights = list(record.get("weights", []))
                topk = int(record.get("topk", len(experts)))
                for rank, expert_raw in enumerate(experts[:topk]):
                    expert = int(expert_raw)
                    if expert < 0:
                        continue
                    if expert >= expected_experts:
                        raise ValueError(
                            f"{trace}:{line_no}: expert {expert} outside 0..{expected_experts - 1}"
                        )
                    key = (layer, expert)
                    counts[key] += 1
                    if rank < len(weights):
                        weight_sums[key] += float(weights[rank])
                trace_used += 1
                used_route_records += 1
                request_seq = int(record.get("request_seq", -1))
                trace_requests.add((request_seq, label))
                request_keys.add((str(trace), request_seq, label))
        per_trace.append(
            {
                "path": str(trace),
                "sha256": sha256_file(trace),
                "route_records": trace_records,
                "used_route_records": trace_used,
                "predecode_records": trace_predecode,
                "requests": len(trace_requests),
            }
        )

    if not counts:
        raise ValueError("no route records matched the requested filters")

    ranking = []
    for (layer, expert), count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        ranking.append(
            {
                "layer": layer,
                "expert": expert,
                "score": int(count),
                "count": int(count),
                "weight_sum": weight_sums.get((layer, expert), 0.0),
            }
        )

    return {
        "schema": PRIOR_SCHEMA,
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model_meta,
            "source": {
                "traces": per_trace,
                "request_label_prefix": args.request_label_prefix,
                "target_start_step": args.target_start_step,
                "target_steps": args.target_steps,
                "route_records": route_records,
                "used_route_records": used_route_records,
                "requests": len(request_keys),
                "score": "decode_route_topk_count",
            },
            "ranking_entries": len(ranking),
        },
        "ranking": ranking,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--traces", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--request-label-prefix", default="chat_")
    parser.add_argument("--target-start-step", type=int, default=1)
    parser.add_argument("--target-steps", type=int, default=48)
    args = parser.parse_args()

    if args.target_start_step <= 0 or args.target_steps <= 0:
        raise ValueError("--target-start-step and --target-steps must be positive")

    prior = build_prior(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(prior, handle, indent=2, sort_keys=True)
        handle.write("\n")

    meta = prior["metadata"]["source"]
    print(
        "Wrote HCS static prior: "
        f"{args.out} ranking_entries={prior['metadata']['ranking_entries']} "
        f"requests={meta['requests']} used_route_records={meta['used_route_records']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
