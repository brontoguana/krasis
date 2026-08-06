#!/usr/bin/env python3
"""Live eviction, branching, multi-turn, cancellation and concurrency gate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_session_cache_live import _assistant_content, _completion_ids, _get, _request


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Run through ./dev session-cache-operations, not directly")


ROOT = Path(__file__).resolve().parents[1]
MOBY = (ROOT / "benchmarks/prompts/prompt1_moby_dick.txt").read_text()[:6000]
WAR = (ROOT / "benchmarks/prompts/prompt2_war_and_peace.txt").read_text()[:6000]


def payload(model: str, messages: list[dict], *, stream: bool = False, tokens: int = 4) -> dict:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_k": 1,
        "max_tokens": tokens,
        "min_new_tokens": tokens,
        "stream": stream,
        "prefix_cache": True,
        "debug_first_token_boundary": True,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    root = f"http://127.0.0.1:{args.port}/v1"
    model = _get(f"{root}/models", args.timeout)["data"][0]["id"]
    initial = _get(f"{root}/session-cache/stats", args.timeout)

    base_messages = [{"role": "user", "content": MOBY + "\nWho commands the Pequod?"}]
    base = _request(f"{root}/chat/completions", payload(model, base_messages), args.timeout)
    history = base_messages + [{"role": "assistant", "content": _assistant_content(base)}]
    turn2_messages = history + [{"role": "user", "content": "Name the ship."}]
    turn2 = _request(f"{root}/chat/completions", payload(model, turn2_messages), args.timeout)
    turn3_messages = turn2_messages + [
        {"role": "assistant", "content": _assistant_content(turn2)},
        {"role": "user", "content": "Answer with one nautical word."},
    ]
    turn3 = _request(f"{root}/chat/completions", payload(model, turn3_messages), args.timeout)

    branch_a = history + [{"role": "user", "content": "Answer only: whale or dolphin?"}]
    branch_b = history + [{"role": "user", "content": "Answer only: sea or land?"}]
    branch_a_result = _request(f"{root}/chat/completions", payload(model, branch_a), args.timeout)
    branch_b_result = _request(f"{root}/chat/completions", payload(model, branch_b), args.timeout)

    concurrent_payloads = [
        payload(model, [{"role": "user", "content": WAR + f"\nConcurrent branch {index}: name the novel."}])
        for index in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(executor.map(
            lambda request_payload: _request(
                f"{root}/chat/completions", request_payload, args.timeout
            ),
            concurrent_payloads,
        ))

    before_cancel = _get(f"{root}/session-cache/stats", args.timeout)
    cancel_body = json.dumps(payload(
        model,
        [{"role": "user", "content": MOBY + "\nWrite a long analysis."}],
        stream=True,
        tokens=250,
    ))
    cancel_request = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(cancel_body.encode('utf-8'))}\r\n"
        "Connection: keep-alive\r\n\r\n"
    ).encode("ascii") + cancel_body.encode("utf-8")
    cancel_socket = socket.create_connection(
        ("127.0.0.1", args.port), timeout=args.timeout
    )
    cancel_socket.sendall(cancel_request)
    if not cancel_socket.recv(1):
        raise RuntimeError("streaming cancellation connection closed before response")
    cancel_socket.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_LINGER,
        struct.pack("ii", 1, 0),
    )
    cancel_socket.close()
    deadline = time.monotonic() + 30
    after_cancel = before_cancel
    while time.monotonic() < deadline:
        time.sleep(0.25)
        after_cancel = _get(f"{root}/session-cache/stats", args.timeout)
        if int(after_cancel["resident"]["reserved_bytes"]) == 0:
            break

    # Force more committed conversations than the deliberately small live RAM
    # budget can retain. The server derives every snapshot's real size.
    before_pressure = after_cancel
    pressure_requests = 0
    while pressure_requests < 8:
        pressure_requests += 1
        text = (MOBY if pressure_requests % 2 else WAR)
        _request(
            f"{root}/chat/completions",
            payload(model, [{"role": "user", "content": text + f"\nUnique cache entry {pressure_requests}."}]),
            args.timeout,
        )
        current = _get(f"{root}/session-cache/stats", args.timeout)
        if int(current["evictions"]) > int(before_pressure["evictions"]):
            break
    final = _get(f"{root}/session-cache/stats", args.timeout)

    failures = []
    for name, value in {
        "base": base,
        "turn2": turn2,
        "turn3": turn3,
        "branch_a": branch_a_result,
        "branch_b": branch_b_result,
        "concurrent_0": concurrent[0],
        "concurrent_1": concurrent[1],
    }.items():
        if not _completion_ids(value):
            failures.append(f"{name} returned no completion token IDs")
    if int(final["hits"]["active_gpu"]) <= int(initial["hits"]["active_gpu"]):
        failures.append("multi-turn requests produced no active-GPU hit")
    if int(final["hits"]["pageable_ram"]) <= int(initial["hits"]["pageable_ram"]):
        failures.append("divergent branches produced no pageable-RAM hit")
    if int(after_cancel["resident"]["reserved_bytes"]) != 0:
        failures.append("cancelled request left reserved RAM bytes")
    if int(after_cancel["resident"]["snapshots"]) != int(
        before_cancel["resident"]["snapshots"]
    ):
        failures.append("cancelled request changed committed snapshot residency")
    if int(final["misses"]["restore_failed"]) != int(initial["misses"]["restore_failed"]):
        failures.append("restore_failed increased")
    if int(final["evictions"]) <= int(before_pressure["evictions"]):
        failures.append("RAM pressure did not evict an LRU snapshot")

    print(json.dumps({
        "model": model,
        "multi_turn_ids": [_completion_ids(base), _completion_ids(turn2), _completion_ids(turn3)],
        "branch_ids": [_completion_ids(branch_a_result), _completion_ids(branch_b_result)],
        "concurrent_ids": [_completion_ids(value) for value in concurrent],
        "cancel_reserved_bytes": after_cancel["resident"]["reserved_bytes"],
        "pressure_requests": pressure_requests,
        "active_hit_delta": int(final["hits"]["active_gpu"]) - int(initial["hits"]["active_gpu"]),
        "ram_hit_delta": int(final["hits"]["pageable_ram"]) - int(initial["hits"]["pageable_ram"]),
        "eviction_delta": int(final["evictions"]) - int(before_pressure["evictions"]),
        "final": final,
        "failures": failures,
    }, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
