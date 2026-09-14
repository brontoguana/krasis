#!/usr/bin/env python3
"""Repeat a frozen raw-token ADQ state through the live Krasis diagnostic route."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
from threading import Barrier
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def post(port: int, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(
            "POST",
            "/v1/internal/reference_test",
            body=json.dumps(payload, separators=(",", ":")),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"reference_test returned HTTP {response.status}: {body[:4000]}")
    result = json.loads(body)
    if not isinstance(result, dict):
        raise RuntimeError("reference_test returned non-object JSON")
    return result


def first_divergence(expected: list[int], actual: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def numerical_signature(response: dict[str, Any]) -> dict[str, Any]:
    """Exact observable numerical result for a greedy reference replay."""
    return {
        "token_ids": response.get("token_ids"),
        "per_token_data": response.get("per_token_data"),
        "first_token_top_k": response.get("first_token_top_k"),
    }


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev adq-ledger-replay; direct execution is unsupported.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--probe-tokens", type=int)
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="submit all identical replay requests concurrently through the serialized model worker",
    )
    parser.add_argument("--debug-early", action="store_true")
    parser.add_argument("--debug-early-steps", type=int, default=2)
    parser.add_argument("--debug-reference", action="store_true")
    parser.add_argument("--debug-prefill-all-layers", action="store_true")
    parser.add_argument("--debug-prefill-layer", type=int)
    parser.add_argument("--debug-hcs-transitions", action="store_true")
    parser.add_argument("--debug-hcs-equiv-layer", type=int)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 for a determinism claim")
    if not 1 <= args.top_k <= 100:
        parser.error("--top-k must be in 1..=100")
    if args.probe_tokens is not None and args.probe_tokens < 2:
        parser.error("--probe-tokens must be at least 2 so decode is exercised")
    if args.debug_hcs_equiv_layer is not None and args.debug_hcs_equiv_layer < 0:
        parser.error("--debug-hcs-equiv-layer must be non-negative")
    if args.debug_prefill_layer is not None and args.debug_prefill_layer < 0:
        parser.error("--debug-prefill-layer must be non-negative")
    if args.debug_early_steps < 1:
        parser.error("--debug-early-steps must be positive")
    if args.concurrent and any(
        (
            args.debug_early,
            args.debug_reference,
            args.debug_prefill_all_layers,
            args.debug_prefill_layer is not None,
            args.debug_hcs_transitions,
            args.debug_hcs_equiv_layer is not None,
        )
    ):
        parser.error("--concurrent cannot be combined with tensor/debug tracing")
    if args.output.exists():
        raise SystemExit(f"replay output already exists; evidence is immutable: {args.output}")

    input_artifact = json.loads((args.state_dir / "preloop-input-token-ids.json").read_text())
    output_artifact = json.loads((args.state_dir / "accepted-output-token-ids.json").read_text())
    manifest = json.loads((args.state_dir / "manifest.json").read_text())
    input_ids = [int(token) for token in input_artifact["input_token_ids"]]
    accepted_text = str(output_artifact["visible_text"])
    accepted_count = int(output_artifact["recorded_generated_token_count"])
    accepted_first = int(output_artifact["retokenized_visible_candidate_token_ids"][0])
    stop_id = output_artifact.get("terminal_stop_token_id")
    stop_ids = [int(stop_id)] if isinstance(stop_id, int) else []
    diagnostic_probe = args.probe_tokens is not None

    def build_request() -> dict[str, Any]:
        request: dict[str, Any] = {
            "input_token_ids": input_ids,
            "max_tokens": args.probe_tokens if diagnostic_probe else accepted_count + 16,
            "top_logprobs": args.top_k,
            "stop_token_ids": [] if diagnostic_probe else stop_ids,
        }
        if args.debug_reference:
            request["debug_reference_trace"] = True
            request["debug_prompt_trace"] = True
        if args.debug_prefill_all_layers:
            request["debug_prefill_device_trace"] = True
            request["debug_prefill_device_trace_all_layers"] = True
        if args.debug_prefill_layer is not None:
            request["debug_prefill_device_trace"] = True
            request["debug_prefill_device_trace_layer"] = args.debug_prefill_layer
        if args.debug_early:
            request["debug_decode_early_trace"] = True
            request["debug_decode_early_trace_max_steps"] = args.debug_early_steps
        if args.debug_hcs_transitions:
            request["debug_hcs_transition_trace"] = True
        if args.debug_hcs_equiv_layer is not None:
            request["debug_decode_hcs_equiv_trace"] = True
            request["debug_decode_hcs_equiv_layer"] = args.debug_hcs_equiv_layer
        return request

    requests = [build_request() for _ in range(args.repeats)]
    if args.concurrent:
        start_barrier = Barrier(args.repeats)

        def submit(request: dict[str, Any]) -> dict[str, Any]:
            start_barrier.wait()
            return post(args.port, request, args.timeout)

        print(
            f"Submitting {args.repeats} identical exact-state requests concurrently",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.repeats) as executor:
            responses = list(executor.map(submit, requests))
    else:
        responses = []
        for repeat, request in enumerate(requests, start=1):
            print(f"Exact-state Krasis replay {repeat}/{args.repeats}", flush=True)
            responses.append(post(args.port, request, args.timeout))

    rows: list[dict[str, Any]] = []
    baseline_ids: list[int] | None = None
    baseline_numerical: dict[str, Any] | None = None
    for repeat, response in enumerate(responses, start=1):
        actual = [int(token) for token in response.get("token_ids", [])]
        if baseline_ids is None:
            baseline_ids = actual
        numerical = numerical_signature(response)
        if baseline_numerical is None:
            baseline_numerical = numerical
        baseline_divergence = first_divergence(baseline_ids, actual)
        numerical_exact = numerical == baseline_numerical
        text_exact = response.get("text") == accepted_text if not diagnostic_probe else None
        count_exact = len(actual) == accepted_count if not diagnostic_probe else None
        stop_exact = (
            bool(stop_ids) and bool(actual) and actual[-1] == stop_ids[0]
            if not diagnostic_probe
            else None
        )
        first_exact = bool(actual) and actual[0] == accepted_first
        finish_exact = response.get("finish_reason") == "stop" if not diagnostic_probe else None
        timing = response.get("timing") or {}
        low_water = timing.get("vram_low_water") or []
        margin = timing.get("safety_margin_mb")
        vram_safe = bool(low_water) and isinstance(margin, (int, float)) and all(
            isinstance(item, dict)
            and isinstance(item.get("min_free_mb"), (int, float))
            and item["min_free_mb"] >= margin
            for item in low_water
        )
        row = {
            "repeat": repeat,
            "token_count": len(actual),
            "token_ids_sha256": canonical_sha256(actual),
            "matches_accepted_observables": (
                text_exact and count_exact and stop_exact and first_exact and finish_exact
                if not diagnostic_probe
                else None
            ),
            "accepted_text_exact": text_exact,
            "accepted_count_exact": count_exact,
            "accepted_first_token_exact": first_exact,
            "accepted_terminal_stop_exact": stop_exact,
            "accepted_finish_reason_exact": finish_exact,
            "matches_first_replay": baseline_divergence is None,
            "numerically_matches_first_replay": numerical_exact,
            "first_divergence_from_first_replay": baseline_divergence,
            "finish_reason": response.get("finish_reason"),
            "vram_safe": vram_safe,
            "timing": timing,
            "response": response,
        }
        rows.append(row)
        print(
            f"  tokens={len(actual)} accepted_observables={row['matches_accepted_observables']} "
            f"token_divergence={baseline_divergence} numerical_match={numerical_exact} "
            f"vram_safe={vram_safe}",
            flush=True,
        )

    report = {
        "format": "krasis_adq_exact_state_replay",
        "format_version": 1,
        "mode": "short_decode_diagnostic_probe" if diagnostic_probe else "full_accepted_trajectory_replay",
        "source_state_manifest_sha256": canonical_sha256(manifest),
        "input_token_count": len(input_ids),
        "input_token_ids_sha256": canonical_sha256(input_ids),
        "accepted_output_token_count": accepted_count,
        "accepted_output_text_sha256": hashlib.sha256(accepted_text.encode("utf-8")).hexdigest(),
        "temperature": 0,
        "sampling": "greedy_internal_reference_test",
        "request_execution": "concurrent" if args.concurrent else "sequential",
        "repeats": args.repeats,
        "rows": rows,
        "token_deterministic": all(row["matches_first_replay"] for row in rows),
        "numerically_deterministic": all(
            row["numerically_matches_first_replay"] for row in rows
        ),
        "deterministic": all(
            row["numerically_matches_first_replay"] for row in rows
        ),
        "reproduces_accepted_observables": (
            None if diagnostic_probe else all(row["matches_accepted_observables"] for row in rows)
        ),
        "reconstructed_token_ids": None if diagnostic_probe else baseline_ids,
        "reconstructed_token_ids_sha256": None if diagnostic_probe else canonical_sha256(baseline_ids),
        "token_identity_basis": (
            "diagnostic probe only; not an accepted-trajectory reconstruction"
            if diagnostic_probe
            else "same frozen input/runtime; repeated greedy token sequence; exact accepted text/count/first-token/terminal-stop/finish"
        ),
        "vram_safe": all(row["vram_safe"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Exact-state replay complete: deterministic={report['deterministic']} "
        f"reproduces_accepted_observables={report['reproduces_accepted_observables']} "
        f"vram_safe={report['vram_safe']}"
    )
    if (
        not report["deterministic"]
        or (not diagnostic_probe and not report["reproduces_accepted_observables"])
        or not report["vram_safe"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
