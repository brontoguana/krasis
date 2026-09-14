#!/usr/bin/env python3
"""Exact-token ADQ multi-turn, session, boundary, and recovery gate."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
from typing import Any

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.adq_build_witness_inputs import _sha256, _source_paths


ACKS = ("ADQ_SEGMENT_1_ACK", "ADQ_SEGMENT_2_ACK", "ADQ_SEGMENT_3_ACK")
EXPECTED = "ADQ_SESSION_RESULT=NORTH-3147|CENTRE-8264|SOUTH-5091"
RECORDS = (
    "The authoritative North session code is NORTH-3147. NORTH-7413 is revoked.",
    "The authoritative Centre session code is CENTRE-8264. CENTRE-4628 is revoked.",
    "The authoritative South session code is SOUTH-5091. SOUTH-1905 is revoked.",
)


def _hash_ids(values: list[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in values).encode("ascii")).hexdigest()


def _render(messages: list[dict[str, str]], *, generation_prompt: bool = True) -> str:
    rendered = "<｜begin▁of▁sentence｜>"
    for message in messages:
        if message["role"] == "user":
            rendered += "<｜User｜>" + message["content"]
        elif message["role"] == "assistant":
            rendered += (
                "<｜Assistant｜></think>"
                + message["content"]
                + "<｜end▁of▁sentence｜>"
            )
        else:
            raise ValueError(f"unsupported role in ADQ renderer: {message['role']!r}")
    if generation_prompt:
        rendered += "<｜Assistant｜></think>"
    return rendered


def _token_count(tokenizer: Tokenizer, messages: list[dict[str, str]]) -> int:
    return len(tokenizer.encode(_render(messages), add_special_tokens=False).ids)


def _decode_slice(tokenizer: Tokenizer, values: list[int]) -> str:
    # A slice may begin or end next to a source-file boundary where the
    # tokenizer would choose a different merge if re-encoded in isolation.
    # Exactness is therefore asserted on the complete rendered conversation,
    # not on an arbitrary interior slice.
    return tokenizer.decode(values, skip_special_tokens=False)


def _content_with_record(
    tokenizer: Tokenizer,
    corpus_ids: list[int],
    start: int,
    before: int,
    after: int,
    record: str,
    instruction: str,
) -> tuple[str, int]:
    end = start + before + after
    if end > len(corpus_ids):
        raise RuntimeError("staged real-content corpus is too short")
    return (
        _decode_slice(tokenizer, corpus_ids[start : start + before])
        + f"\n\n===== ADQ SESSION CONTROL =====\n{record}\n===== END CONTROL =====\n\n"
        + _decode_slice(tokenizer, corpus_ids[start + before : end])
        + instruction,
        end,
    )


def _fit_final_turn(
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    corpus_ids: list[int],
    start: int,
    target_tokens: int,
) -> tuple[dict[str, str], list[int]]:
    record = (
        f"\n\n===== ADQ SESSION CONTROL =====\n{RECORDS[2]}\n"
        "===== END CONTROL =====\n\n"
    )
    query = (
        "\n\nThis is the final session question. Recover the three authoritative codes "
        "from the North, Centre, and South records, ignore every revoked code, and "
        f"reply with exactly {EXPECTED} and no other text."
    )

    def candidate(count: int) -> tuple[dict[str, str], list[int]]:
        values = corpus_ids[start : start + count]
        split = min(len(values), max(0, int(len(values) * 0.875)))
        content = (
            _decode_slice(tokenizer, values[:split])
            + record
            + _decode_slice(tokenizer, values[split:])
            + query
        )
        message = {"role": "user", "content": content}
        ids = tokenizer.encode(
            _render(messages + [message]), add_special_tokens=False
        ).ids
        return message, ids

    empty_message = {"role": "user", "content": record + query}
    fixed_tokens = len(
        tokenizer.encode(
            _render(messages + [empty_message]), add_special_tokens=False
        ).ids
    )
    count = min(len(corpus_ids) - start, max(1, target_tokens - fixed_tokens))
    exact: tuple[dict[str, str], list[int]] | None = None
    for _ in range(8):
        value = candidate(count)
        delta = target_tokens - len(value[1])
        if delta == 0:
            exact = value
            break
        next_count = min(len(corpus_ids) - start, max(1, count + delta))
        if next_count == count:
            break
        count = next_count
    if exact is None:
        for nearby_count in range(
            max(1, count - 64), min(len(corpus_ids) - start, count + 64) + 1
        ):
            value = candidate(nearby_count)
            if len(value[1]) == target_tokens:
                exact = value
                break
    if exact is None:
        raise RuntimeError(f"could not construct exact {target_tokens:,}-token session prompt")
    return exact


def _real_corpus_ids(tokenizer: Tokenizer, repo_root: Path) -> list[int]:
    values: list[int] = []
    for path in _source_paths(repo_root):
        raw = path.read_bytes()
        header = (
            f"\n\n===== SOURCE {path.relative_to(repo_root).as_posix()} "
            f"SHA256 {_sha256(raw)} =====\n"
        )
        values.extend(
            tokenizer.encode(header + raw.decode("utf-8"), add_special_tokens=False).ids
        )
    return values


def _exact_single_user(
    tokenizer: Tokenizer, corpus_ids: list[int], target_tokens: int
) -> tuple[list[dict[str, str]], list[int]]:
    framing = tokenizer.encode(
        "<｜begin▁of▁sentence｜><｜User｜><｜Assistant｜></think>",
        add_special_tokens=False,
    ).ids
    def candidate(count: int) -> tuple[list[dict[str, str]], list[int]]:
        messages = [{"role": "user", "content": _decode_slice(tokenizer, corpus_ids[:count])}]
        ids = tokenizer.encode(_render(messages), add_special_tokens=False).ids
        return messages, ids

    count = min(len(corpus_ids), max(1, target_tokens - len(framing)))
    for _ in range(8):
        value = candidate(count)
        delta = target_tokens - len(value[1])
        if delta == 0:
            return value
        next_count = min(len(corpus_ids), max(1, count + delta))
        if next_count == count:
            break
        count = next_count
    for nearby_count in range(max(1, count - 128), min(len(corpus_ids), count + 128) + 1):
        value = candidate(nearby_count)
        if len(value[1]) == target_tokens:
            return value
    raise RuntimeError(f"could not build exact {target_tokens:,}-token boundary prompt")


def _request(port: int, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(
        "POST",
        "/v1/chat/completions",
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


def _get(port: int, path: str, timeout: int) -> dict[str, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    response = conn.getresponse()
    raw = response.read().decode("utf-8")
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"GET {path} returned {response.status}: {raw}")
    return json.loads(raw)


def _text(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"response has no assistant content: {response}") from exc


def _chat_payload(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    prefix_cache: bool,
    stream: bool = False,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "min_new_tokens": 1,
        "enable_thinking": False,
        "prefix_cache": prefix_cache,
        "stream": stream,
    }


def _run_turn(
    port: int,
    model: str,
    messages: list[dict[str, str]],
    expected: str,
    timeout: int,
) -> dict[str, Any]:
    status, body = _request(
        port,
        _chat_payload(model, messages, max_tokens=16, prefix_cache=True),
        timeout,
    )
    text = _text(body) if status == 200 else ""
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return {
        "http_status": status,
        "text": text,
        "expected": expected,
        "pass": status == 200 and text == expected,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def _cancel_stream(port: int, model: str, content: str, timeout: int) -> None:
    body = json.dumps(
        _chat_payload(
            model,
            [{"role": "user", "content": content}],
            max_tokens=256,
            prefix_cache=True,
            stream=True,
        ),
        separators=(",", ":"),
    )
    request = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body.encode('utf-8'))}\r\n"
        "Connection: keep-alive\r\n\r\n"
    ).encode("ascii") + body.encode("utf-8")
    stream = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    stream.sendall(request)
    if not stream.recv(1):
        raise RuntimeError("cancellation stream closed before returning a byte")
    stream.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    stream.close()


def main() -> int:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit(
            "Run through ./dev adq-session-soak; direct execution is unsupported."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--target-tokens", type=int, default=500_000)
    parser.add_argument("--timeout", type=int, default=14_400)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    source = json.loads(args.input_source.read_text(encoding="utf-8"))
    source_turn = min(
        (
            turn
            for conversation in source.get("conversations", [])
            for turn in conversation.get("turns", [])
            if len(turn.get("input_token_ids", [])) >= args.target_tokens
        ),
        key=lambda turn: len(turn["input_token_ids"]),
    )
    framing = source["framing"]
    source_ids = [int(value) for value in source_turn["input_token_ids"]]
    corpus_ids = source_ids[
        int(framing["prefix_tokens"]) : len(source_ids) - int(framing["suffix_tokens"])
    ]
    if len(corpus_ids) < 400_000:
        raise RuntimeError("staged input does not contain enough real-content tokens")

    models = _get(args.port, "/v1/models", args.timeout)
    model = str(models["data"][0]["id"])
    max_context = int(models["data"][0]["max_context_tokens"])
    health = _get(args.port, "/health", args.timeout)
    initial_stats = _get(args.port, "/v1/session-cache/stats", args.timeout)
    messages: list[dict[str, str]] = []
    cursor = 0
    turn_results: list[dict[str, Any]] = []

    content, cursor = _content_with_record(
        tokenizer,
        corpus_ids,
        cursor,
        15_000,
        85_000,
        RECORDS[0],
        f"\n\nReply with exactly {ACKS[0]} and no other text.",
    )
    messages.append({"role": "user", "content": content})
    result = _run_turn(args.port, model, messages, ACKS[0], args.timeout)
    turn_results.append(result)
    if not result["pass"]:
        raise RuntimeError(f"first accumulated turn failed: {result}")
    messages.append({"role": "assistant", "content": result["text"]})

    content, cursor = _content_with_record(
        tokenizer,
        corpus_ids,
        cursor,
        100_000,
        0,
        "No new authoritative code appears in this segment.",
        f"\n\nReply with exactly {ACKS[1]} and no other text.",
    )
    messages.append({"role": "user", "content": content})
    result = _run_turn(args.port, model, messages, ACKS[1], args.timeout)
    turn_results.append(result)
    if not result["pass"]:
        raise RuntimeError(f"second accumulated turn failed: {result}")
    messages.append({"role": "assistant", "content": result["text"]})

    content, cursor = _content_with_record(
        tokenizer,
        corpus_ids,
        cursor,
        50_000,
        50_000,
        RECORDS[1],
        f"\n\nReply with exactly {ACKS[2]} and no other text.",
    )
    messages.append({"role": "user", "content": content})
    result = _run_turn(args.port, model, messages, ACKS[2], args.timeout)
    turn_results.append(result)
    if not result["pass"]:
        raise RuntimeError(f"third accumulated turn failed: {result}")
    messages.append({"role": "assistant", "content": result["text"]})

    final_message, final_ids = _fit_final_turn(
        tokenizer, messages, corpus_ids, cursor, args.target_tokens
    )
    messages.append(final_message)
    if _token_count(tokenizer, messages) != args.target_tokens:
        raise AssertionError("final accumulated prompt is not exact-token sized")

    accumulated_status, accumulated_body = _request(
        args.port,
        _chat_payload(model, messages, max_tokens=64, prefix_cache=True),
        args.timeout,
    )
    accumulated_text = _text(accumulated_body) if accumulated_status == 200 else ""
    accumulated_prompt_tokens = (accumulated_body.get("usage") or {}).get("prompt_tokens")
    after_accumulated_stats = _get(
        args.port, "/v1/session-cache/stats", args.timeout
    )

    fresh_status, fresh_body = _request(
        args.port,
        _chat_payload(model, messages, max_tokens=64, prefix_cache=False),
        args.timeout,
    )
    fresh_text = _text(fresh_body) if fresh_status == 200 else ""
    restore_status, restore_body = _request(
        args.port,
        _chat_payload(model, messages, max_tokens=64, prefix_cache=True),
        args.timeout,
    )
    restore_text = _text(restore_body) if restore_status == 200 else ""
    after_restore_stats = _get(args.port, "/v1/session-cache/stats", args.timeout)

    full_corpus_ids = _real_corpus_ids(tokenizer, Path(__file__).resolve().parents[1])
    boundary_messages, boundary_ids = _exact_single_user(
        tokenizer, full_corpus_ids, max_context - 1
    )
    boundary_status, boundary_body = _request(
        args.port,
        _chat_payload(model, boundary_messages, max_tokens=1, prefix_cache=False),
        args.timeout,
    )
    overflow_status, overflow_body = _request(
        args.port,
        _chat_payload(model, boundary_messages, max_tokens=2, prefix_cache=False),
        args.timeout,
    )

    cancel_content = _decode_slice(tokenizer, full_corpus_ids[:64_000]) + (
        "\nWrite a long engineering analysis."
    )
    before_cancel = _get(args.port, "/v1/session-cache/stats", args.timeout)
    _cancel_stream(args.port, model, cancel_content, args.timeout)
    deadline = time.monotonic() + 60
    after_cancel = before_cancel
    while time.monotonic() < deadline:
        time.sleep(0.5)
        after_cancel = _get(args.port, "/v1/session-cache/stats", args.timeout)
        if int(after_cancel["resident"]["reserved_bytes"]) == 0:
            break
    recovery = _run_turn(
        args.port,
        model,
        [{"role": "user", "content": "Reply with exactly ADQ_RECOVERY_OK."}],
        "ADQ_RECOVERY_OK",
        args.timeout,
    )

    cache_active_delta = int(after_accumulated_stats["hits"]["active_gpu"]) - int(
        initial_stats["hits"]["active_gpu"]
    )
    cache_ram_delta = int(after_restore_stats["hits"]["pageable_ram"]) - int(
        after_accumulated_stats["hits"]["pageable_ram"]
    )
    overflow_error = overflow_body.get("error")
    if isinstance(overflow_error, dict):
        overflow_code = overflow_error.get("code")
    else:
        overflow_code = None
    checks = {
        "intermediate_turns": all(result["pass"] for result in turn_results),
        "exact_accumulated_tokens": accumulated_prompt_tokens == args.target_tokens,
        "accumulated_answer": accumulated_status == 200 and accumulated_text == EXPECTED,
        "fresh_equivalence": fresh_status == 200 and fresh_text == accumulated_text,
        "restore_equivalence": restore_status == 200 and restore_text == accumulated_text,
        "active_cache_hit": cache_active_delta > 0,
        "ram_restore_hit": cache_ram_delta > 0,
        "boundary_admitted": (
            boundary_status == 200
            and (boundary_body.get("usage") or {}).get("prompt_tokens") == max_context - 1
        ),
        "overflow_rejected": overflow_status == 413 and overflow_code == "context_length_exceeded",
        "cancel_released": int(after_cancel["resident"]["reserved_bytes"]) == 0,
        "cancel_did_not_commit": int(after_cancel["resident"]["snapshots"])
        == int(before_cancel["resident"]["snapshots"]),
        "post_cancel_recovery": recovery["pass"],
    }
    report = {
        "format": "krasis_adq_session_soak_result",
        "format_version": 1,
        "model": model,
        "health": health,
        "target_tokens": args.target_tokens,
        "max_context_tokens": max_context,
        "accumulated_input_sha256": _hash_ids(final_ids),
        "boundary_input_tokens": len(boundary_ids),
        "boundary_input_sha256": _hash_ids(boundary_ids),
        "turn_results": turn_results,
        "accumulated": {
            "status": accumulated_status,
            "text": accumulated_text,
            "prompt_tokens": accumulated_prompt_tokens,
        },
        "fresh": {"status": fresh_status, "text": fresh_text},
        "restored": {"status": restore_status, "text": restore_text},
        "boundary": {
            "status": boundary_status,
            "prompt_tokens": (boundary_body.get("usage") or {}).get("prompt_tokens"),
        },
        "overflow": {"status": overflow_status, "body": overflow_body},
        "cache": {
            "active_hit_delta": cache_active_delta,
            "ram_hit_delta": cache_ram_delta,
            "before_cancel": before_cancel,
            "after_cancel": after_cancel,
        },
        "recovery": recovery,
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "target_tokens", "checks", "pass")}, indent=2))
    print(f"Wrote {args.output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
