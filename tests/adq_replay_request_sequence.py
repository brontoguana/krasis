#!/usr/bin/env python3
"""Replay every frozen OpenCode request prefix before a failed terminal turn.

This diagnostic preserves the exact rendered token IDs for each model request
and reproduces the request-order effects on runtime state before probing the
terminal request. It must be run through ``./dev adq-ledger-request-sequence``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def post_chat(port: int, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"chat completion returned HTTP {response.status}: {body[:4000]}")
    result = json.loads(body)
    if not isinstance(result, dict):
        raise RuntimeError("chat completion returned non-object JSON")
    return result


def first_divergence(expected: list[int], actual: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def request_prefixes(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("frozen preloop request has no preceding assistant turns")
    prefixes = [messages[:index] for index in assistant_indices]
    prefixes.append(messages)
    for prefix in prefixes:
        if not prefix or prefix[-1].get("role") not in {"user", "tool"}:
            raise ValueError("request prefix does not end at a user/tool model boundary")
    return prefixes


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit(
            "Run through ./dev adq-ledger-request-sequence; direct execution is unsupported."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--probe-tokens", type=int, default=16)
    parser.add_argument("--expected-prompt-tokens", required=True)
    args = parser.parse_args()
    if args.probe_tokens < 1:
        parser.error("--probe-tokens must be positive")
    expected_counts = [
        int(value.strip()) for value in args.expected_prompt_tokens.split(",") if value.strip()
    ]
    if not expected_counts or any(value <= 0 for value in expected_counts):
        parser.error("--expected-prompt-tokens must be a CSV of positive integers")
    if args.output.exists():
        raise SystemExit(f"sequence output already exists; evidence is immutable: {args.output}")

    request_path = args.state_dir / "preloop-request.json"
    input_path = args.state_dir / "preloop-input-token-ids.json"
    accepted_path = args.state_dir / "accepted-output-token-ids.json"
    for path in (request_path, input_path, accepted_path):
        if not path.is_file():
            raise SystemExit(f"required frozen state artifact is missing: {path}")
    frozen_request = json.loads(request_path.read_text(encoding="utf-8"))
    messages = frozen_request.get("messages")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise SystemExit("frozen preloop request has no valid messages array")
    try:
        prefixes = request_prefixes(messages)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(prefixes) != len(expected_counts):
        raise SystemExit(
            f"expected-token count has {len(expected_counts)} rows but reconstructed "
            f"sequence has {len(prefixes)} requests"
        )

    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_ids = [int(value) for value in accepted["retokenized_visible_candidate_token_ids"]]
    frozen_final_ids = [
        int(value)
        for value in json.loads(input_path.read_text(encoding="utf-8"))["input_token_ids"]
    ]
    rows: list[dict[str, Any]] = []
    for index, (prefix, expected_count) in enumerate(zip(prefixes, expected_counts), start=1):
        terminal = index == len(prefixes)
        payload = copy.deepcopy(frozen_request)
        payload["messages"] = prefix
        payload["stream"] = False
        payload["max_tokens"] = args.probe_tokens if terminal else 1
        payload.pop("max_completion_tokens", None)
        payload["temperature"] = 0
        payload["debug_first_token_boundary"] = True
        payload["top_logprobs"] = 100
        print(
            f"Replaying frozen request prefix {index}/{len(prefixes)} "
            f"(messages={len(prefix)}, expected_tokens={expected_count})",
            flush=True,
        )
        response = post_chat(args.port, payload, args.timeout)
        debug = response.get("krasis_debug")
        if not isinstance(debug, dict):
            raise SystemExit(f"request {index} omitted krasis_debug")
        input_ids = debug.get("input_token_ids")
        completion_ids = debug.get("completion_token_ids")
        if not isinstance(input_ids, list) or not all(isinstance(value, int) for value in input_ids):
            raise SystemExit(f"request {index} omitted exact input token IDs")
        if not isinstance(completion_ids, list) or not all(
            isinstance(value, int) for value in completion_ids
        ):
            raise SystemExit(f"request {index} omitted completion token IDs")
        if len(input_ids) != expected_count:
            raise SystemExit(
                f"request {index} prompt mismatch: expected {expected_count}, got {len(input_ids)}"
            )
        if terminal and input_ids != frozen_final_ids:
            raise SystemExit("terminal request IDs differ from the independently frozen exact input")
        rows.append(
            {
                "request_index": index,
                "terminal_request": terminal,
                "message_count": len(prefix),
                "preceding_assistant_messages": sum(
                    message.get("role") == "assistant" for message in prefix
                ),
                "preceding_tool_messages": sum(message.get("role") == "tool" for message in prefix),
                "input_token_count": len(input_ids),
                "input_token_ids": input_ids,
                "input_token_ids_sha256": canonical_sha256(input_ids),
                "completion_token_ids": completion_ids,
                "completion_token_ids_sha256": canonical_sha256(completion_ids),
                "selected_token_id": debug.get("selected_token_id"),
                "first_token_logits": debug.get("first_token_logits"),
                "completion_decode_trace": debug.get("completion_decode_trace"),
                "timing": response.get("krasis_timing") or response.get("usage"),
            }
        )

    final_ids = rows[-1]["completion_token_ids"]
    divergence = first_divergence(accepted_ids[: args.probe_tokens], final_ids)
    report = {
        "format": "krasis_adq_ledger_request_sequence_replay",
        "format_version": 1,
        "source_state_manifest_sha256": canonical_sha256(
            json.loads((args.state_dir / "manifest.json").read_text(encoding="utf-8"))
        ),
        "frozen_preloop_request_sha256": canonical_sha256(frozen_request),
        "temperature": 0,
        "sequence_scope": (
            "Every exact historical chat-message prefix is prefilled in original order; "
            "nonterminal probes generate one token and do not reproduce full historical decode."
        ),
        "expected_prompt_tokens": expected_counts,
        "request_count": len(rows),
        "rows": rows,
        "terminal_probe_tokens": args.probe_tokens,
        "terminal_matches_accepted_prefix": divergence is None,
        "terminal_first_divergence_from_accepted": divergence,
        "terminal_accepted_prefix_token_ids": accepted_ids[: args.probe_tokens],
        "terminal_observed_token_ids": final_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Request-sequence replay complete: final_divergence={divergence} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
