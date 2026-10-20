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


def _build_request(
    input_ids: list[int],
    max_tokens: int,
    stop_ids: list[int],
    *,
    debug_reference_trace: bool = False,
    debug_prefill_device_trace_all_layers: bool = False,
    debug_prefill_device_trace_layer: int | None = None,
    debug_prefill_device_trace_full_pre_out_proj: bool = False,
    debug_prefill_device_trace_rows: list[int] | None = None,
    debug_prefill_device_trace_dims: list[int] | None = None,
    debug_prefill_device_trace_experts: list[int] | None = None,
    debug_prefill_device_trace_local_scan_token: int | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "input_token_ids": input_ids,
        "max_tokens": max_tokens,
        "top_logprobs": 10,
        "stop_token_ids": stop_ids,
        "debug_prompt_trace": True,
    }
    if debug_reference_trace:
        request["debug_reference_trace"] = True
    device_trace = (
        debug_prefill_device_trace_all_layers
        or debug_prefill_device_trace_layer is not None
        or debug_prefill_device_trace_full_pre_out_proj
        or bool(debug_prefill_device_trace_rows)
        or bool(debug_prefill_device_trace_dims)
        or bool(debug_prefill_device_trace_experts)
        or debug_prefill_device_trace_local_scan_token is not None
    )
    if device_trace:
        request["debug_prefill_device_trace"] = True
    if debug_prefill_device_trace_all_layers:
        request["debug_prefill_device_trace_all_layers"] = True
    if debug_prefill_device_trace_layer is not None:
        request["debug_prefill_device_trace_layer"] = debug_prefill_device_trace_layer
    if debug_prefill_device_trace_full_pre_out_proj:
        request["debug_prefill_device_trace_full_pre_out_proj"] = True
    if debug_prefill_device_trace_rows:
        request["debug_prefill_device_trace_rows"] = debug_prefill_device_trace_rows
    if debug_prefill_device_trace_dims:
        request["debug_prefill_device_trace_dims"] = debug_prefill_device_trace_dims
    if debug_prefill_device_trace_experts:
        request["debug_prefill_device_trace_experts"] = debug_prefill_device_trace_experts
    if debug_prefill_device_trace_local_scan_token is not None:
        request["debug_prefill_device_trace_local_scan_token"] = (
            debug_prefill_device_trace_local_scan_token
        )
    return request


def _call(
    port: int,
    input_ids: list[int],
    max_tokens: int,
    stop_ids: list[int],
    timeout: int,
    **trace_options: Any,
) -> dict[str, Any]:
    payload = json.dumps(
        _build_request(input_ids, max_tokens, stop_ids, **trace_options),
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


def _actual_prefill_top_k(actual: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_trace = actual.get("debug_prompt_trace")
    if not isinstance(prompt_trace, dict):
        return []
    first_token_logits = prompt_trace.get("first_token_logits")
    if not isinstance(first_token_logits, dict) or first_token_logits.get("available") is not True:
        return []
    values = first_token_logits.get("top_logits_before_logprob")
    return values if isinstance(values, list) else []


def _top_entry_logprob(row: dict[str, Any]) -> float | None:
    value = row.get("log_prob", row.get("logprob"))
    return float(value) if isinstance(value, (int, float)) else None


def _prefill_metrics(
    reference_turn: dict[str, Any], actual: dict[str, Any], expected_first_token: int | None
) -> dict[str, Any]:
    positions = (reference_turn.get("prefill_logits") or {}).get("positions") or []
    reference_row = positions[0] if len(positions) == 1 and isinstance(positions[0], dict) else None
    reference_top = reference_row.get("top_k", []) if reference_row else []
    actual_top = _actual_prefill_top_k(actual)
    prompt_trace = actual.get("debug_prompt_trace") or {}
    expected_position = len(reference_turn.get("input_token_ids", [])) - 1
    reference_position = reference_row.get("position") if reference_row else None
    actual_position = (
        int(prompt_trace["prompt_len"]) - 1
        if isinstance(prompt_trace, dict)
        and isinstance(prompt_trace.get("prompt_len"), int)
        else None
    )
    reference_ids = [int(row["token_id"]) for row in reference_top if "token_id" in row]
    actual_ids = [int(row["token_id"]) for row in actual_top if "token_id" in row]

    reference_selected = None
    actual_selected = None
    if expected_first_token is not None:
        for row in reference_top:
            if int(row.get("token_id", -1)) == expected_first_token:
                reference_selected = _top_entry_logprob(row)
                break
        for row in actual_top:
            if int(row.get("token_id", -1)) == expected_first_token:
                actual_selected = _top_entry_logprob(row)
                break
    selected_delta = None
    if (
        reference_selected is not None
        and actual_selected is not None
        and math.isfinite(reference_selected)
        and math.isfinite(actual_selected)
    ):
        selected_delta = abs(actual_selected - reference_selected)

    def finite_top(values: list[dict[str, Any]]) -> bool:
        return bool(values) and all(
            (value := _top_entry_logprob(row)) is not None and math.isfinite(value)
            for row in values
        )

    first_token_logits = (
        prompt_trace.get("first_token_logits") if isinstance(prompt_trace, dict) else None
    )
    actual_global_finite = bool(
        isinstance(first_token_logits, dict)
        and first_token_logits.get("available") is True
        and first_token_logits.get("nan_count") == 0
        and first_token_logits.get("pos_inf_count") == 0
        and first_token_logits.get("neg_inf_count") == 0
    )
    return {
        "expected_position": expected_position,
        "reference_position": reference_position,
        "actual_position": actual_position,
        "position_exact": (
            reference_position == expected_position and actual_position == expected_position
        ),
        "reference_top10_count": len(reference_ids),
        "actual_top10_count": len(actual_ids),
        "top10_overlap": len(set(reference_ids) & set(actual_ids)),
        "witness_selected_token_id": expected_first_token,
        "witness_selected_token_actual_rank": (
            actual_ids.index(expected_first_token) + 1
            if expected_first_token is not None and expected_first_token in actual_ids
            else None
        ),
        "witness_selected_token_in_actual_top10": (
            expected_first_token in actual_ids if expected_first_token is not None else False
        ),
        "selected_logprob_abs_delta": selected_delta,
        "reference_top10_finite": finite_top(reference_top),
        "actual_top10_finite": finite_top(actual_top),
        "actual_full_distribution_finite": actual_global_finite,
        "evidence_complete": bool(
            reference_row
            and reference_top
            and actual_top
            and reference_position == expected_position
            and actual_position == expected_position
            and finite_top(reference_top)
            and finite_top(actual_top)
            and actual_global_finite
        ),
    }


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
    prefill = _prefill_metrics(
        reference_turn,
        actual,
        expected_ids[0] if expected_ids else None,
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
        "prefill": prefill,
        "numerical_evidence_complete": prefill["evidence_complete"],
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
    parser.add_argument("--debug-reference-trace", action="store_true")
    parser.add_argument("--debug-prefill-device-trace-all-layers", action="store_true")
    parser.add_argument("--debug-prefill-device-trace-layer", type=int)
    parser.add_argument("--debug-prefill-device-trace-full-pre-out-proj", action="store_true")
    parser.add_argument("--debug-prefill-device-trace-rows", default="")
    parser.add_argument("--debug-prefill-device-trace-dims", default="")
    parser.add_argument("--debug-prefill-device-trace-experts", default="")
    parser.add_argument("--debug-prefill-device-trace-local-scan-token", type=int)
    args = parser.parse_args()

    def csv_nonnegative_ints(value: str, option: str) -> list[int]:
        try:
            parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as exc:
            raise SystemExit(f"{option} must be a comma-separated integer list") from exc
        if any(item < 0 for item in parsed):
            raise SystemExit(f"{option} values must be non-negative")
        return parsed

    trace_options = {
        "debug_reference_trace": args.debug_reference_trace,
        "debug_prefill_device_trace_all_layers": args.debug_prefill_device_trace_all_layers,
        "debug_prefill_device_trace_layer": args.debug_prefill_device_trace_layer,
        "debug_prefill_device_trace_full_pre_out_proj": (
            args.debug_prefill_device_trace_full_pre_out_proj
        ),
        "debug_prefill_device_trace_rows": csv_nonnegative_ints(
            args.debug_prefill_device_trace_rows,
            "--debug-prefill-device-trace-rows",
        ),
        "debug_prefill_device_trace_dims": csv_nonnegative_ints(
            args.debug_prefill_device_trace_dims,
            "--debug-prefill-device-trace-dims",
        ),
        "debug_prefill_device_trace_experts": csv_nonnegative_ints(
            args.debug_prefill_device_trace_experts,
            "--debug-prefill-device-trace-experts",
        ),
        "debug_prefill_device_trace_local_scan_token": (
            args.debug_prefill_device_trace_local_scan_token
        ),
    }

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
                actual = _call(
                    args.port,
                    input_ids,
                    max_tokens,
                    stop_ids,
                    args.timeout,
                    **trace_options,
                )
                metrics = _metrics(turn, actual)
                status = "complete"
                runtime_status = "complete"
                numerical_status = (
                    "complete" if metrics["numerical_evidence_complete"] else "incomplete"
                )
                vram_status = "pass" if metrics["vram_safe"] else "fail_separate"
                error = None
            except Exception as exc:  # retained in the evidence artifact
                actual = None
                metrics = None
                status = "runtime_failure"
                runtime_status = "failed"
                numerical_status = "unavailable"
                vram_status = "unavailable"
                error = str(exc)
            rows.append(
                {
                    "input_tokens": target,
                    "input_sha256": turn.get("input_sha256"),
                    "status": status,
                    "runtime_status": runtime_status,
                    "numerical_status": numerical_status,
                    "vram_status": vram_status,
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
        "trace_request": trace_options,
        "rows": rows,
        "runtime_complete": all(row["status"] == "complete" for row in rows),
        "numerical_evidence_complete": all(
            row["numerical_status"] == "complete" for row in rows
        ),
        "vram_safety_pass": all(row["vram_status"] == "pass" for row in rows),
        "verdict_separation": "runtime_numerical_and_vram_are_independent",
        "quality_verdict": "pending_pre_registered_envelope_review",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    if not report["runtime_complete"] or not report["numerical_evidence_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
