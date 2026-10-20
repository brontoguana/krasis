#!/usr/bin/env python3
"""Run objective exact-token ADQ long-context checks against a live server."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krasis.tokenizer import Tokenizer
from tests.adq_build_witness_inputs import _fit_exact_chat_input


TARGET_TOKENS = 500_000
EXPECTED = "ADQ_RESULT=CERULEAN-7391|AMBER-4826|VIOLET-1507"
QUERY_FORMAT = "ADQ_RESULT=<North active code>|<Centre active code>|<South active code>"
RECORDS = (
    (
        0.03,
        "\n\n===== AUTHORITATIVE ADQ CONTROL RECORD NORTH =====\n"
        "The active North access code is CERULEAN-7391. An older note naming "
        "CERULEAN-1937 is revoked and must not be used.\n"
        "===== END CONTROL RECORD NORTH =====\n\n",
    ),
    (
        0.50,
        "\n\n===== AUTHORITATIVE ADQ CONTROL RECORD CENTRE =====\n"
        "The active Centre access code is AMBER-4826. An older note naming "
        "AMBER-2684 is revoked and must not be used.\n"
        "===== END CONTROL RECORD CENTRE =====\n\n",
    ),
    (
        0.95,
        "\n\n===== AUTHORITATIVE ADQ CONTROL RECORD SOUTH =====\n"
        "The active South access code is VIOLET-1507. An older note naming "
        "VIOLET-7051 is revoked and must not be used.\n"
        "===== END CONTROL RECORD SOUTH =====\n\n",
    ),
)


def _sha256_ids(values: list[int]) -> str:
    canonical = ",".join(str(value) for value in values).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def select_smallest_qualifying_turn(
    source: dict[str, Any], target_tokens: int
) -> dict[str, Any]:
    turns = [
        turn
        for conversation in source.get("conversations", [])
        for turn in conversation.get("turns", [])
        if len(turn.get("input_token_ids", [])) >= target_tokens
    ]
    if not turns:
        raise ValueError(f"expected one source case at least {target_tokens:,} tokens")
    return min(turns, key=lambda turn: len(turn["input_token_ids"]))


def _decode(tokenizer: Tokenizer, values: list[int]) -> str:
    return tokenizer.tokenizer.decode(
        values,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def decode_exact_raw_prompt(tokenizer: Tokenizer, values: list[int]) -> str:
    """Decode a frozen rendered prompt and reject any lossy round trip."""
    prompt = _decode(tokenizer, values)
    roundtrip = tokenizer.tokenizer.encode(prompt, add_special_tokens=False)
    if roundtrip != values:
        mismatch = next(
            (
                index
                for index, (expected, actual) in enumerate(zip(values, roundtrip))
                if expected != actual
            ),
            min(len(values), len(roundtrip)),
        )
        raise RuntimeError(
            "decoded raw prompt does not reproduce its frozen token IDs: "
            f"expected={len(values):,} actual={len(roundtrip):,} "
            f"first_mismatch={mismatch:,}"
        )
    return prompt


def _chat_ids(tokenizer: Tokenizer, content: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _retrieval_query() -> str:
    return (
        "\n\n===== END ADQ LONG-CONTEXT TEST =====\n"
        "Read the authoritative North, Centre, and South ADQ control records. "
        "Ignore each explicitly revoked older note. Reply with exactly one line "
        "in North|Centre|South order and no other text, using this placeholder "
        f"format: {QUERY_FORMAT}"
    )


def _provider_content(
    corpus_ids: list[int], tokenizer: Tokenizer, content_count: int, alignment: str
) -> str:
    body = list(corpus_ids[:content_count])
    for position_ratio, text in reversed(RECORDS):
        record_ids = tokenizer.tokenizer.encode(text, add_special_tokens=False)
        position = int(content_count * position_ratio)
        end = position + len(record_ids)
        if end > len(body):
            raise RuntimeError(f"record at {position:,} falls outside provider body")
        body[position:end] = record_ids
    body_text = _decode(tokenizer, body)
    return body_text + alignment + _retrieval_query()


def build_provider_case(
    corpus_ids: list[int], tokenizer: Tokenizer, target_tokens: int
) -> tuple[list[int], str, list[dict[str, Any]]]:
    """Build canonical raw text whose checkpoint tokenization is exactly the target."""

    def candidate(content_count: int, alignment: str = "") -> tuple[list[int], str]:
        content = _provider_content(corpus_ids, tokenizer, content_count, alignment)
        values = _chat_ids(tokenizer, content)
        return values, content

    empty = _chat_ids(tokenizer, _retrieval_query())
    if len(empty) >= target_tokens:
        raise RuntimeError(
            f"target {target_tokens:,} is not larger than its "
            f"{len(empty):,}-token provider framing"
        )

    content_count = min(len(corpus_ids), max(1, target_tokens - len(empty)))
    exact: tuple[list[int], str] | None = None
    closest_below: tuple[int, int] | None = None
    seen_content_counts: set[int] = set()
    for attempt in range(24):
        if content_count in seen_content_counts:
            break
        seen_content_counts.add(content_count)
        values, content = candidate(content_count)
        delta = target_tokens - len(values)
        print(
            "Provider prompt solve "
            f"attempt={attempt + 1} content_tokens={content_count:,} "
            f"rendered_tokens={len(values):,} delta={delta:+,}",
            flush=True,
        )
        if delta == 0:
            exact = (values, content)
            break
        if delta > 0 and (closest_below is None or delta < closest_below[0]):
            closest_below = (delta, content_count)
        next_count = min(len(corpus_ids), max(1, content_count + delta))
        if next_count == content_count:
            break
        content_count = next_count

    if exact is None and closest_below is not None:
        gap, nearby_count = closest_below
        for unit in (" x", " .", "\n", " ", "\t", "\u200b"):
            repeats = max(1, gap)
            seen_repeats: set[int] = set()
            for attempt in range(8):
                if repeats in seen_repeats:
                    break
                seen_repeats.add(repeats)
                values, content = candidate(nearby_count, unit * repeats)
                delta = target_tokens - len(values)
                print(
                    "Provider prompt alignment "
                    f"unit={unit!r} attempt={attempt + 1} repeats={repeats:,} "
                    f"rendered_tokens={len(values):,} delta={delta:+,}",
                    flush=True,
                )
                if delta == 0:
                    exact = (values, content)
                    break
                repeats = max(1, repeats + delta)
            if exact is not None:
                break
    if exact is None:
        raise RuntimeError(
            f"could not construct canonical provider prompt at {target_tokens:,} tokens"
        )

    values, content = exact
    raw_prompt = decode_exact_raw_prompt(tokenizer, values)
    records: list[dict[str, Any]] = []
    for position_ratio, text in RECORDS:
        if raw_prompt.count(text) != 1:
            raise RuntimeError("provider prompt does not contain one intact control record")
        prefix = raw_prompt.split(text, 1)[0]
        absolute_position = len(
            tokenizer.tokenizer.encode(prefix, add_special_tokens=False)
        )
        records.append(
            {
                "requested_position_ratio": position_ratio,
                "absolute_token_position": absolute_position,
                "text": text,
            }
        )
    if EXPECTED in raw_prompt:
        raise RuntimeError("provider prompt leaked the expected answer")
    return values, raw_prompt, records


def build_case(
    corpus_ids: list[int], tokenizer: Tokenizer, target_tokens: int = TARGET_TOKENS
) -> tuple[list[int], list[dict[str, Any]]]:
    query = _retrieval_query()
    if len(corpus_ids) < target_tokens:
        raise RuntimeError(f"real-content corpus has only {len(corpus_ids):,} tokens")

    result, _alignment = _fit_exact_chat_input(
        tokenizer, corpus_ids, target_tokens, query
    )

    inserted: list[dict[str, Any]] = []
    for position_ratio, text in RECORDS:
        absolute_position = int(target_tokens * position_ratio)
        record_ids = tokenizer.tokenizer.encode(text, add_special_tokens=False)
        end = absolute_position + len(record_ids)
        if absolute_position < 0 or end > len(result):
            raise RuntimeError(f"record at {absolute_position:,} falls outside body")
        result[absolute_position:end] = record_ids
        inserted.append(
            {
                "absolute_token_position": absolute_position,
                "token_count": len(record_ids),
                "text": text,
            }
        )
    if len(result) != target_tokens:
        raise AssertionError(f"built {len(result):,} tokens, expected {target_tokens:,}")
    return result, inserted


def _post(port: int, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(
        "POST",
        "/v1/internal/reference_test",
        body=json.dumps(payload, separators=(",", ":")),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    raw = response.read().decode("utf-8")
    conn.close()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = {"raw": raw}
    return response.status, body


def _vram_safe(timing: dict[str, Any]) -> bool:
    margin = timing.get("safety_margin_mb")
    rows = timing.get("vram_low_water") or []
    return bool(rows) and isinstance(margin, (int, float)) and all(
        isinstance(row, dict)
        and isinstance(row.get("min_free_mb"), (int, float))
        and row["min_free_mb"] >= margin
        for row in rows
    )


def _short_isolation_ids(tokenizer: Tokenizer) -> list[int]:
    return _chat_ids(
        tokenizer,
        "Reply with exactly ISOLATION_OK and do not repeat any earlier access code.",
    )


def build_witness_input_artifact(
    retrieval_ids: list[int],
    isolation_ids: list[int],
    source_artifact: Path,
) -> dict[str, Any]:
    """Build the two-request source-witness input without retokenizing either case."""
    cases = (
        ("exact_500k_retrieval", retrieval_ids),
        ("post_500k_isolation", isolation_ids),
    )
    return {
        "format": "krasis_adq_long_context_witness_inputs",
        "format_version": 1,
        "source_artifact": str(source_artifact.resolve()),
        "conversations": [
            {
                "source_id": source_id,
                "turns": [
                    {
                        "prompt": source_id,
                        "input_token_ids": token_ids,
                        "input_sha256": _sha256_ids(token_ids),
                        "chat_template_application": {
                            "method": "pretokenized_frozen_input",
                            "add_generation_prompt": True,
                        },
                    }
                ],
            }
            for source_id, token_ids in cases
        ],
    }


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev adq-long-context; direct execution is unsupported.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--timeout", type=int, default=14_400)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--witness-input-output", type=Path)
    parser.add_argument(
        "--raw-prompt-output",
        type=Path,
        help=(
            "write the checkpoint-decoded rendered prompt after proving an exact "
            "decode/encode token-ID round trip"
        ),
    )
    args = parser.parse_args()

    source = json.loads(args.input_source.read_text(encoding="utf-8"))
    try:
        source_turn = select_smallest_qualifying_turn(source, args.target_tokens)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    tokenizer = Tokenizer(str(args.model_dir))
    corpus_ids = [int(value) for value in source.get("corpus_token_ids", [])]
    corpus_digest = _sha256_ids(corpus_ids)
    if not corpus_ids or corpus_digest != source.get("corpus_token_sha256"):
        raise SystemExit("staged input lacks an intact checkpoint-native corpus token stream")
    raw_prompt = None
    if args.raw_prompt_output is not None:
        input_ids, raw_prompt, inserted = build_provider_case(
            corpus_ids, tokenizer, args.target_tokens
        )
    else:
        input_ids, inserted = build_case(
            corpus_ids, tokenizer, args.target_tokens
        )
    isolation_ids = _short_isolation_ids(tokenizer)
    report: dict[str, Any] = {
        "format": "krasis_adq_long_context_result",
        "format_version": 1,
        "target_tokens": args.target_tokens,
        "input_sha256": _sha256_ids(input_ids),
        "source_artifact": str(args.input_source.resolve()),
        "source_input_sha256": source_turn.get("input_sha256"),
        "inserted_records": inserted,
        "expected": EXPECTED,
        "input_token_ids": input_ids,
        "isolation_input_token_ids": isolation_ids,
        "isolation_input_sha256": _sha256_ids(isolation_ids),
        "execution": "build_only" if args.build_only else "pending",
    }
    if raw_prompt is not None:
        encoded_prompt = raw_prompt.encode("utf-8")
        report["raw_prompt"] = {
            "path": str(args.raw_prompt_output.resolve()),
            "bytes": len(encoded_prompt),
            "sha256": hashlib.sha256(encoded_prompt).hexdigest(),
            "token_roundtrip_exact": True,
            "case_kind": "canonical_raw_completion_prompt",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if raw_prompt is not None:
        args.raw_prompt_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_prompt_output.write_text(raw_prompt, encoding="utf-8")
        print(f"Wrote exact raw prompt {args.raw_prompt_output}", flush=True)
    if args.witness_input_output is not None:
        witness_input = build_witness_input_artifact(
            input_ids, isolation_ids, args.output
        )
        args.witness_input_output.parent.mkdir(parents=True, exist_ok=True)
        args.witness_input_output.write_text(
            json.dumps(witness_input, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote witness input {args.witness_input_output}", flush=True)
    print(f"Built exact {len(input_ids):,}-token case: {report['input_sha256']}", flush=True)
    if args.build_only:
        print(f"Wrote {args.output}", flush=True)
        return

    print(f"Running exact {args.target_tokens:,}-token multi-region retrieval", flush=True)
    status, response = _post(
        args.port,
        {
            "input_token_ids": input_ids,
            "max_tokens": 64,
            "top_logprobs": 10,
            "stop_token_ids": [int(tokenizer.eos_token_id)],
        },
        args.timeout,
    )
    long_text = str(response.get("text", ""))
    long_pass = status == 200 and EXPECTED in long_text and _vram_safe(response.get("timing") or {})
    report["long_request"] = {
        "http_status": status,
        "pass": long_pass,
        "response": response,
    }
    print(f"Retrieval pass={long_pass} status={status} text={long_text!r}", flush=True)

    status, response = _post(
        args.port,
        {
            "input_token_ids": isolation_ids,
            "max_tokens": 16,
            "top_logprobs": 10,
            "stop_token_ids": [int(tokenizer.eos_token_id)],
        },
        args.timeout,
    )
    isolation_text = str(response.get("text", ""))
    isolation_pass = (
        status == 200
        and "ISOLATION_OK" in isolation_text
        and all(code not in isolation_text for code in ("CERULEAN", "AMBER", "VIOLET"))
        and _vram_safe(response.get("timing") or {})
    )
    report["isolation_request"] = {
        "input_tokens": len(isolation_ids),
        "http_status": status,
        "pass": isolation_pass,
        "response": response,
    }
    report["execution"] = "complete"
    report["pass"] = long_pass and isolation_pass
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Isolation pass={isolation_pass} status={status} text={isolation_text!r}", flush=True)
    print(f"Wrote {args.output}", flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
