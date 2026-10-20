#!/usr/bin/env python3
"""Run frozen provider-reference token cases through Krasis and classify them."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import time
from typing import Any


def _canonical_id_sha256(values: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in values).encode("ascii")
    ).hexdigest()


def _classify(krasis: str, provider: str, providers_disagree: bool = False) -> str:
    if krasis == "INCOMPLETE" or provider == "INCOMPLETE":
        return "INCOMPLETE"
    if providers_disagree:
        return "REFERENCE_DISAGREEMENT"
    if krasis == "PASS" and provider == "PASS":
        return "ALIGNED_PASS"
    if krasis == "FAIL" and provider == "PASS":
        return "KRASIS_DEFICIT"
    if krasis == "PASS" and provider == "FAIL":
        return "KRASIS_ONLY_PASS"
    if krasis == "FAIL" and provider == "FAIL":
        return "ALIGNED_FAIL"
    raise ValueError(f"unsupported result pair: krasis={krasis} provider={provider}")


def _call(port: int, payload: dict[str, Any], timeout: int) -> tuple[int, bytes, float]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    started = time.monotonic()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request(
        "POST",
        "/v1/internal/reference_test",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    connection.close()
    return status, raw, time.monotonic() - started


def _vram_status(timing: dict[str, Any]) -> tuple[str, int | None, int | None]:
    margin = timing.get("safety_margin_mb")
    lows = timing.get("vram_low_water")
    if not isinstance(margin, (int, float)) or not isinstance(lows, list) or not lows:
        return "INCOMPLETE", None, int(margin) if isinstance(margin, (int, float)) else None
    values = [
        int(row["min_free_mb"])
        for row in lows
        if isinstance(row, dict) and isinstance(row.get("min_free_mb"), (int, float))
    ]
    if len(values) != len(lows):
        return "INCOMPLETE", None, int(margin)
    minimum = min(values)
    return ("PASS" if minimum >= margin else "FAIL"), minimum, int(margin)


def main() -> int:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit(
            "Run through ./dev adq-provider-compare; direct execution is unsupported."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", required=True, type=Path)
    parser.add_argument("--provider-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-dir", required=True, type=Path)
    parser.add_argument("--targets", default="3000,16000,32000,64000,131072,262144,500000")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--timeout", type=int, default=7_200)
    args = parser.parse_args()

    targets = [int(value.strip()) for value in args.targets.split(",") if value.strip()]
    if not targets or len(set(targets)) != len(targets) or any(value <= 0 for value in targets):
        raise RuntimeError("targets must be unique positive integers")
    provider_results = json.loads(args.provider_results.read_text(encoding="utf-8"))
    if (
        provider_results.get("verdict") != "PASS"
        or provider_results.get("all_behavioral_oracles_pass") is not True
    ):
        raise RuntimeError("provider baseline is not independently verified PASS")
    provider_rows = provider_results.get("rows")
    if not isinstance(provider_rows, list):
        raise RuntimeError("provider result rows are missing")
    providers = sorted({str(row.get("provider")) for row in provider_rows})
    if len(providers) < 1:
        raise RuntimeError("provider result set is empty")

    args.rows_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    for target in targets:
        case_path = args.cases_dir / f"case-{target}.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        input_ids = [int(value) for value in case.get("input_token_ids", [])]
        expected = str(case.get("expected", ""))
        if int(case.get("target_tokens", -1)) != target or len(input_ids) != target:
            raise RuntimeError(f"case {target} has inconsistent token count")
        if _canonical_id_sha256(input_ids) != case.get("input_sha256"):
            raise RuntimeError(f"case {target} token identity mismatch")
        if not expected:
            raise RuntimeError(f"case {target} has no objective oracle")

        payload = {
            "input_token_ids": input_ids,
            "max_tokens": 64,
            "top_logprobs": 5,
        }
        request_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request_path = args.rows_dir / f"request-{target}.json"
        response_path = args.rows_dir / f"response-{target}.json"
        request_path.write_bytes(request_bytes)
        request_started = time.monotonic()
        try:
            status, raw, elapsed = _call(args.port, payload, args.timeout)
            response_path.write_bytes(raw + (b"" if raw.endswith(b"\n") else b"\n"))
            response = json.loads(raw)
            text = str(response.get("text", ""))
            timing = response.get("timing") or {}
            prompt_tokens = timing.get("prompt_tokens")
            oracle = "PASS" if status == 200 and text.strip() == expected else "FAIL"
            runtime = "PASS" if status == 200 else "FAIL"
            token_count = "PASS" if prompt_tokens == target else "FAIL"
            vram, minimum_vram, margin = _vram_status(timing)
            numerical_finite = all(
                isinstance(item.get("log_prob"), (int, float))
                and math.isfinite(float(item["log_prob"]))
                for token in response.get("per_token_data", [])
                for item in token.get("top_k", [])
            )
            if not response.get("per_token_data"):
                numerical_finite = False
        except Exception as exc:
            elapsed = time.monotonic() - request_started
            response = None
            status = None
            text = ""
            prompt_tokens = None
            oracle = "INCOMPLETE"
            runtime = "INCOMPLETE"
            token_count = "INCOMPLETE"
            vram = "INCOMPLETE"
            minimum_vram = None
            margin = None
            numerical_finite = False
            response_path.write_text(
                json.dumps({"error": str(exc)}, indent=2) + "\n", encoding="utf-8"
            )
        row = {
            "target_tokens": target,
            "case_path": str(case_path.resolve()),
            "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "input_sha256": case["input_sha256"],
            "request_path": str(request_path.resolve()),
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "response_path": str(response_path.resolve()),
            "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
            "http_status": status,
            "elapsed_seconds": elapsed,
            "expected": expected,
            "actual_text": text,
            "runtime_status": runtime,
            "behavioral_oracle": oracle,
            "prompt_token_count_status": token_count,
            "reported_prompt_tokens": prompt_tokens,
            "numerical_values_finite": numerical_finite,
            "vram_status": vram,
            "minimum_free_vram_mb": minimum_vram,
            "safety_margin_mb": margin,
            "response": response,
        }
        rows.append(row)

        statuses = {
            provider: next(
                (
                    str(value.get("oracle"))
                    for value in provider_rows
                    if value.get("provider") == provider
                    and int(value.get("target_tokens", -1)) == target
                ),
                "INCOMPLETE",
            )
            for provider in providers
        }
        providers_disagree = len(set(statuses.values())) > 1
        for provider, provider_status in statuses.items():
            classifications.append(
                {
                    "target_tokens": target,
                    "provider": provider,
                    "krasis": oracle,
                    "provider_result": provider_status,
                    "classification": _classify(
                        oracle, provider_status, providers_disagree
                    ),
                }
            )
        print(
            f"target={target} runtime={runtime} oracle={oracle} "
            f"prompt_tokens={prompt_tokens} vram={vram} minimum_free_mb={minimum_vram}",
            flush=True,
        )

    report = {
        "format": "krasis_adq_cross_runtime_provider_comparison",
        "format_version": 1,
        "provider_result_sha256": hashlib.sha256(args.provider_results.read_bytes()).hexdigest(),
        "targets": targets,
        "providers": providers,
        "attempts_per_target": 1,
        "retries": 0,
        "rows": rows,
        "classifications": classifications,
        "runtime_complete": all(row["runtime_status"] == "PASS" for row in rows),
        "behavioral_oracles_pass": all(row["behavioral_oracle"] == "PASS" for row in rows),
        "prompt_token_counts_exact": all(
            row["prompt_token_count_status"] == "PASS" for row in rows
        ),
        "numerical_values_finite": all(row["numerical_values_finite"] for row in rows),
        "vram_safety_pass": all(row["vram_status"] == "PASS" for row in rows),
        "cross_runtime_coverage": (
            "COMPLETE"
            if all(
                row["classification"] == "ALIGNED_PASS" for row in classifications
            )
            else "PARTIAL"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0 if all(
        (
            report["runtime_complete"],
            report["behavioral_oracles_pass"],
            report["prompt_token_counts_exact"],
            report["numerical_values_finite"],
            report["vram_safety_pass"],
            report["cross_runtime_coverage"] == "COMPLETE",
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
