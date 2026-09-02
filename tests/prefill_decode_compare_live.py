#!/usr/bin/env python3
"""Compare batch-prefill and tokenwise Rust decode logits on one live server.

The server must be started with ``--test-endpoints``. The script first asks the
normal chat route to render and tokenize the prompt, then feeds those exact token
IDs to both diagnostic execution strategies. This is a parity diagnostic, not a
reference-quality judge and not a benchmark.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
from pathlib import Path
from typing import Any


def _post(port: int, route: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            route,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = response.read().decode("utf-8", errors="replace")
    finally:
        conn.close()
    if response.status != 200:
        raise RuntimeError(f"{route} returned HTTP {response.status}: {body}")
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{route} returned non-object JSON")
    return decoded


def _top_ids(position: dict[str, Any]) -> list[int]:
    rows = position.get("top_k", [])
    if not isinstance(rows, list):
        return []
    return [int(row["token_id"]) for row in rows if isinstance(row, dict) and "token_id" in row]


def main() -> int:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        print(
            "ERROR: run this diagnostic through ./dev prefill-decode-compare",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(
        description="Compare exact-token batch prefill and tokenwise Rust decode logits"
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--prompt",
        default="What is the capital of France? Answer with only the city name.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--debug-early", action="store_true")
    parser.add_argument(
        "--prefill-trace-layer",
        type=int,
        default=-1,
        help="also capture the established device-side prefill trace for this layer",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.top_k < 1 or args.top_k > 100:
        parser.error("--top-k must be in 1..=100")
    if args.max_input_tokens < 0:
        parser.error("--max-input-tokens must be non-negative")
    if args.prefill_trace_layer < -1:
        parser.error("--prefill-trace-layer must be non-negative when supplied")

    chat = _post(
        args.port,
        "/v1/chat/completions",
        {
            "model": "test",
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
            "debug_first_token_boundary": True,
            "top_logprobs": args.top_k,
        },
        args.timeout,
    )
    debug = chat.get("krasis_debug")
    if not isinstance(debug, dict):
        raise RuntimeError("chat response omitted krasis_debug tokenization data")
    input_token_ids = debug.get("input_token_ids")
    if not isinstance(input_token_ids, list) or not input_token_ids:
        raise RuntimeError("chat debug response omitted non-empty input_token_ids")
    tokens = [int(token) for token in input_token_ids]
    if args.max_input_tokens:
        tokens = tokens[: args.max_input_tokens]
    if not tokens:
        raise RuntimeError("token selection is empty")

    prefill = _post(
        args.port,
        "/v1/internal/prefill_logits",
        {"input_token_ids": tokens, "top_k": args.top_k, "sample_every": 1},
        args.timeout,
    )
    teacher = _post(
        args.port,
        "/v1/internal/teacher_forced_decode_logits",
        {
            "input_token_ids": tokens,
            "top_k": args.top_k,
            "debug_decode_early_trace": args.debug_early,
        },
        args.timeout,
    )
    prefill_device_trace = None
    if args.prefill_trace_layer >= 0:
        reference = _post(
            args.port,
            "/v1/internal/reference_test",
            {
                "input_token_ids": tokens,
                "max_tokens": 1,
                "top_logprobs": args.top_k,
                "debug_prefill_device_trace": True,
                "debug_prefill_device_trace_layer": args.prefill_trace_layer,
                "debug_prefill_device_trace_rows": [0],
                "debug_reference_trace": True,
            },
            args.timeout,
        )
        prefill_device_trace = reference.get("debug_prefill_device_trace")
        prefill_reference_trace = reference.get("debug_reference_trace")
    else:
        prefill_reference_trace = None

    prefill_by_position = {
        int(row["position"]): row
        for row in prefill.get("positions", [])
        if isinstance(row, dict) and "position" in row
    }
    decode_by_position = {
        int(row["position"]): row
        for row in teacher.get("positions", [])
        if isinstance(row, dict) and "position" in row
    }
    rows: list[dict[str, Any]] = []
    first_mismatch: int | None = None
    for position in range(len(tokens)):
        prefill_row = prefill_by_position.get(position)
        decode_row = decode_by_position.get(position)
        prefill_ids = _top_ids(prefill_row or {})
        decode_ids = _top_ids(decode_row or {})
        same_top1 = bool(prefill_ids and decode_ids and prefill_ids[0] == decode_ids[0])
        if not same_top1 and first_mismatch is None:
            first_mismatch = position
        rows.append(
            {
                "position": position,
                "input_token_id": tokens[position],
                "prefill_top1": prefill_ids[0] if prefill_ids else None,
                "decode_top1": decode_ids[0] if decode_ids else None,
                "same_top1": same_top1,
                "top_k_overlap": len(set(prefill_ids).intersection(decode_ids)),
                "prefill_top_k": prefill_ids,
                "decode_top_k": decode_ids,
            }
        )

    report = {
        "schema": "krasis_prefill_decode_live_compare_v1",
        "port": args.port,
        "prompt": args.prompt,
        "input_token_ids": tokens,
        "top_k": args.top_k,
        "positions_compared": len(rows),
        "top1_matches": sum(1 for row in rows if row["same_top1"]),
        "first_top1_mismatch": first_mismatch,
        "rows": rows,
        "chat_debug": debug,
        "teacher_debug_decode_early_trace": teacher.get("debug_decode_early_trace"),
        "prefill_device_trace": prefill_device_trace,
        "prefill_reference_trace": prefill_reference_trace,
        "teacher_reset_bytes": teacher.get("reset_bytes"),
        "teacher_cleanup_error": teacher.get("cleanup_error"),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    print(
        f"Compared {len(rows)} positions: top1={report['top1_matches']}/{len(rows)}, "
        f"first_mismatch={first_mismatch}"
    )
    for row in rows:
        marker = "MATCH" if row["same_top1"] else "MISMATCH"
        print(
            f"  pos={row['position']:>3} {marker:<8} "
            f"prefill={row['prefill_top1']} decode={row['decode_top1']} "
            f"top{args.top_k}_overlap={row['top_k_overlap']}"
        )
    if args.output is not None:
        print(f"Report: {args.output}")
    return 0 if first_mismatch is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
