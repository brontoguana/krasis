#!/usr/bin/env python3
"""Live in-flight prefill disconnect and worker-recovery contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krasis.tokenizer import Tokenizer
from tests.adq_session_soak_live import (
    _chat_payload,
    _exact_single_user,
    _get,
    _run_turn,
)


def _hash_ids(values: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in values).encode("ascii")
    ).hexdigest()


def main() -> int:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit(
            "Run through ./dev prefill-disconnect-test; direct execution is unsupported."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--target-tokens", type=int, default=64_000)
    parser.add_argument("--disconnect-delay", type=float, default=10.0)
    parser.add_argument("--timeout", type=int, default=1_800)
    args = parser.parse_args()

    if args.target_tokens <= 0 or args.disconnect_delay <= 0:
        raise RuntimeError("target tokens and disconnect delay must be positive")
    source = json.loads(args.input_source.read_text(encoding="utf-8"))
    corpus_ids = [int(value) for value in source.get("corpus_token_ids", [])]
    if not corpus_ids or _hash_ids(corpus_ids) != source.get("corpus_token_sha256"):
        raise RuntimeError("input source lacks an intact real-content token corpus")

    tokenizer = Tokenizer(str(args.model_dir))
    messages, input_ids = _exact_single_user(
        tokenizer, corpus_ids, args.target_tokens
    )
    if len(input_ids) != args.target_tokens:
        raise AssertionError("prefill cancellation request is not exact-token sized")

    models = _get(args.port, "/v1/models", args.timeout)
    model = str(models["data"][0]["id"])
    before = _get(args.port, "/v1/session-cache/stats", args.timeout)
    payload = _chat_payload(
        model,
        messages,
        max_tokens=128,
        prefix_cache=True,
        stream=True,
    )
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: keep-alive\r\n\r\n"
    ).encode("ascii") + body

    stream = socket.create_connection(("127.0.0.1", args.port), timeout=args.timeout)
    stream.sendall(request)
    sent_at = time.monotonic()
    time.sleep(args.disconnect_delay)
    stream.setblocking(False)
    response_bytes_before_disconnect = 0
    try:
        response_bytes_before_disconnect = len(stream.recv(1, socket.MSG_PEEK))
    except BlockingIOError:
        pass
    if response_bytes_before_disconnect != 0:
        stream.close()
        raise RuntimeError(
            "server returned response bytes before the preregistered prefill disconnect"
        )
    stream.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    stream.close()
    disconnected_at = time.monotonic()

    # This request queues behind the canceled model request. Its completion
    # therefore proves that the serialized model worker left prefill cleanup.
    after = _get(args.port, "/v1/session-cache/stats", args.timeout)
    recovered_at = time.monotonic()
    recovery = _run_turn(
        args.port,
        model,
        [{"role": "user", "content": "Reply with exactly ADQ_PREFILL_RECOVERY_OK."}],
        "ADQ_PREFILL_RECOVERY_OK",
        args.timeout,
    )
    checks = {
        "exact_request_tokens": len(input_ids) == args.target_tokens,
        "no_response_before_disconnect": response_bytes_before_disconnect == 0,
        "cancel_released": int(after["resident"]["reserved_bytes"]) == 0,
        "cancel_did_not_commit": int(after["resident"]["snapshots"])
        == int(before["resident"]["snapshots"]),
        "post_cancel_recovery": recovery["pass"],
    }
    result = {
        "format": "krasis_prefill_disconnect_test",
        "format_version": 1,
        "model": model,
        "input_tokens": len(input_ids),
        "input_sha256": _hash_ids(input_ids),
        "rendered_input_token_ids": input_ids,
        "rendered_input_token_ids_sha256": _hash_ids(input_ids),
        "request_payload": payload,
        "request_payload_sha256": hashlib.sha256(body).hexdigest(),
        "raw_http_request_sha256": hashlib.sha256(request).hexdigest(),
        "disconnect_delay_seconds": args.disconnect_delay,
        "response_bytes_before_disconnect": response_bytes_before_disconnect,
        "socket_open_seconds": disconnected_at - sent_at,
        "worker_recovery_wait_seconds": recovered_at - disconnected_at,
        "before": before,
        "after": after,
        "recovery": recovery,
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
