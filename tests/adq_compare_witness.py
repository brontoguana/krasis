#!/usr/bin/env python3
"""Compare staged ADQ inputs with an authoritative llama-witness artifact."""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import math
from typing import Any


def _call(port: int, input_ids: list[int], max_tokens: int, stop_ids: list[int], timeout: int) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_token_ids": input_ids,
            "max_tokens": max_tokens,
            "top_logprobs": 10,
            "stop_token_ids": stop_ids,
        },
        separators=(",", ":"),
    )
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(
        "POST",
        "/v1/internal/reference_test",
        body=payload,
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    body = response.read().decode("utf-8")
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"reference endpoint returned HTTP {response.status}: {body[:2000]}")
    return json.loads(body)


def _top_ids(row: dict[str, Any]) -> list[int]:
    return [int(item["token_id"]) for item in row.get("top_k", [])]


def _selected_logprob(row: dict[str, Any], token_id: int) -> float | None:
    for item in row.get("top_k", []):
        if int(item.get("token_id", -1)) == token_id:
            value = item.get("log_prob", item.get("logprob"))
            return float(value) if isinstance(value, (int, float)) else None
    if int(row.get("token_id", -1)) == token_id:
        value = row.get("log_prob", row.get("logprob"))
        return float(value) if isinstance(value, (int, float)) else None
    return None


def _metrics(reference_turn: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_ids = [int(value) for value in reference_turn.get("token_ids", [])]
    actual_ids = [int(value) for value in actual.get("token_ids", [])]
    run = 0
    for expected, observed in zip(expected_ids, actual_ids):
        if expected != observed:
            break
        run += 1

    expected_rows = reference_turn.get("per_token_data", [])
    actual_rows = actual.get("per_token_data", [])
    # Row 0 is conditioned on the identical frozen prompt. If N generated
    # tokens match, rows 0..N are still conditioned on identical histories;
    # rows after the first divergent decision are not teacher-forced evidence.
    comparable = min(len(expected_ids), len(expected_rows), len(actual_rows), run + 1)
    containment = 0
    logprob_deltas: list[float] = []
    for index in range(comparable):
        expected = expected_ids[index]
        observed_top = _top_ids(actual_rows[index])
        if expected in observed_top:
            containment += 1
        if index < len(expected_rows):
            reference_logprob = _selected_logprob(expected_rows[index], expected)
            actual_logprob = _selected_logprob(actual_rows[index], expected)
            if (
                reference_logprob is not None
                and actual_logprob is not None
                and math.isfinite(reference_logprob)
                and math.isfinite(actual_logprob)
            ):
                logprob_deltas.append(abs(actual_logprob - reference_logprob))

    first_ref_top = set(_top_ids(expected_rows[0])) if expected_rows else set()
    first_actual_top = set(_top_ids(actual_rows[0])) if actual_rows else set()
    first_selected_delta = None
    if expected_ids and expected_rows and actual_rows:
        reference_value = _selected_logprob(expected_rows[0], expected_ids[0])
        actual_value = _selected_logprob(actual_rows[0], expected_ids[0])
        if (
            reference_value is not None
            and actual_value is not None
            and math.isfinite(reference_value)
            and math.isfinite(actual_value)
        ):
            first_selected_delta = abs(actual_value - reference_value)
    timing = actual.get("timing") or {}
    margin = timing.get("safety_margin_mb")
    low_water = timing.get("vram_low_water") or []
    vram_safe = bool(low_water) and isinstance(margin, (int, float)) and all(
        isinstance(row, dict)
        and isinstance(row.get("min_free_mb"), (int, float))
        and row["min_free_mb"] >= margin
        for row in low_water
    )
    return {
        "reference_tokens": len(expected_ids),
        "actual_tokens": len(actual_ids),
        "first_token_exact": bool(expected_ids and actual_ids and expected_ids[0] == actual_ids[0]),
        "exact_prefix_tokens": run,
        "identical_context_logit_steps": comparable,
        "witness_top10_hits": containment,
        "witness_top10_total": comparable,
        "witness_top10_rate": containment / comparable if comparable else None,
        "first_token_top10_overlap": len(first_ref_top & first_actual_top),
        "first_token_selected_logprob_abs_delta": first_selected_delta,
        "selected_logprob_abs_delta_mean": (
            sum(logprob_deltas) / len(logprob_deltas) if logprob_deltas else None
        ),
        "selected_logprob_abs_delta_max": max(logprob_deltas) if logprob_deltas else None,
        "vram_safe": vram_safe,
        "continuation_comparison_scope": "identical_history_only_through_first_divergent_decision",
        "timing": timing,
    }


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev adq-witness-compare; direct execution is unsupported.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--targets", default="")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference.get("runtime") != "llama-witness":
        raise SystemExit("reference artifact is not authored by llama-witness")
    selected = {
        int(value.strip()) for value in args.targets.split(",") if value.strip()
    }
    max_tokens = int(reference.get("max_new_tokens", 1))
    stop_ids = [int(value) for value in reference.get("eos_token_ids", [])]
    rows: list[dict[str, Any]] = []

    for conversation in reference.get("conversations", []):
        for turn in conversation.get("turns", []):
            input_ids = [int(value) for value in turn["input_token_ids"]]
            target = len(input_ids)
            if selected and target not in selected:
                continue
            print(f"Comparing exact {target:,}-token input", flush=True)
            try:
                actual = _call(args.port, input_ids, max_tokens, stop_ids, args.timeout)
                metrics = _metrics(turn, actual)
                status = "complete" if metrics["vram_safe"] else "runtime_failure"
                error = None if metrics["vram_safe"] else "measured VRAM low-water below safety margin"
            except Exception as exc:  # retained in the evidence artifact
                actual = None
                metrics = None
                status = "runtime_failure"
                error = str(exc)
            rows.append(
                {
                    "input_tokens": target,
                    "input_sha256": turn.get("input_sha256"),
                    "status": status,
                    "error": error,
                    "metrics": metrics,
                    "actual": actual,
                }
            )
            print(f"  {status}: {metrics or error}", flush=True)

    if not rows:
        raise SystemExit("no witness cases matched the requested targets")
    report = {
        "format": "krasis_adq_witness_comparison",
        "format_version": 1,
        "reference_path": str(args.reference.resolve()),
        "reference_profile_id": reference.get("profile_id"),
        "witness_model": reference.get("witness_model"),
        "max_new_tokens": max_tokens,
        "rows": rows,
        "runtime_complete": all(row["status"] == "complete" for row in rows),
        "quality_verdict": "pending_pre_registered_envelope_review",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    if not report["runtime_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
