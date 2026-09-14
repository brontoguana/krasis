#!/usr/bin/env python3
"""Live SSE graceful-disconnect and model-worker recovery contract."""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import time


def _worker_stats(port: int, timeout: float) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request("GET", "/v1/session-cache/stats")
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, json.loads(body)


def main() -> int:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev sse-disconnect-test; direct execution is unsupported.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--cancel-timeout", type=float, default=5.0)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    payload["stream"] = True
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: keep-alive\r\n\r\n"
    ).encode("ascii") + body

    before_status, before_stats = _worker_stats(args.port, args.timeout)
    stream = socket.create_connection(("127.0.0.1", args.port), timeout=args.timeout)
    stream.sendall(request)
    first_byte = stream.recv(1)
    if not first_byte:
        raise RuntimeError("SSE connection closed before response headers")

    cancel_started = time.monotonic()
    # Reproduce an orderly peer write-half close. On the server this is a FIN /
    # CLOSE-WAIT event, not a failed server-to-client write.
    stream.shutdown(socket.SHUT_WR)
    stream.settimeout(args.cancel_timeout)
    response_bytes = bytearray(first_byte)
    closed = False
    try:
        while True:
            chunk = stream.recv(65536)
            if not chunk:
                closed = True
                break
            response_bytes.extend(chunk)
    except TimeoutError:
        pass
    finally:
        stream.close()
    cancel_elapsed = time.monotonic() - cancel_started

    recovery_started = time.monotonic()
    after_status, after_stats = _worker_stats(args.port, args.timeout)
    recovery_elapsed = time.monotonic() - recovery_started
    reserved_bytes = int(after_stats.get("resident", {}).get("reserved_bytes", -1))
    passed = (
        before_status == 200
        and closed
        and cancel_elapsed <= args.cancel_timeout
        and after_status == 200
        and reserved_bytes == 0
    )
    result = {
        "format": "krasis_sse_graceful_disconnect_test",
        "format_version": 1,
        "disconnect": "client_shutdown_write_fin",
        "payload": str(args.payload.resolve()),
        "payload_max_tokens": payload.get("max_tokens"),
        "response_bytes_before_close": len(response_bytes),
        "server_closed_within_limit": closed,
        "cancel_elapsed_seconds": cancel_elapsed,
        "cancel_timeout_seconds": args.cancel_timeout,
        "recovery_http_status": after_status,
        "recovery_elapsed_seconds": recovery_elapsed,
        "reserved_bytes_after_recovery": reserved_bytes,
        "committed_snapshots_before": before_stats.get("resident", {}).get("snapshots"),
        "committed_snapshots_after": after_stats.get("resident", {}).get("snapshots"),
        "pass": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
