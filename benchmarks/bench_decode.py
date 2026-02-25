#!/usr/bin/env python3
"""Benchmark decode speed at multiple context lengths.

Uses streaming API to measure TTFT (time-to-first-token) and ITL
(inter-token-latency) separately.

Usage:
    python3 bench_decode.py [--output results.json]
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

PORT = int(os.environ.get("PORT", 8080))
SERVER = f"http://localhost:{PORT}"
MODEL = os.environ.get("MODEL", "Qwen3-235B-A22B")

FILLER = "The quick brown fox jumps over the lazy dog. "

# Test configs: (label, approximate input tokens, max_output_tokens)
TESTS = [
    ("short",     50,   80),
    ("medium",   200,   80),
    ("long",     500,   80),
    ("1k",      1000,   80),
]


def check_server():
    try:
        req = urllib.request.Request(f"{SERVER}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def make_prompt(target_tokens):
    repeats = max(1, target_tokens // 9)
    filler = FILLER * repeats
    return f"Please summarize the following text in exactly 3 sentences:\n\n{filler}\n\nSummary:"


def run_streaming_test(prompt, max_tokens, temperature=0.7):
    """Run a streaming test and measure TTFT + ITL separately."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    token_count = 0

    with urllib.request.urlopen(req, timeout=3600) as resp:
        buffer = b""
        for chunk in iter(lambda: resp.read(1), b""):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    try:
                        obj = json.loads(line[6:])
                        delta = obj["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            now = time.perf_counter()
                            if t_first_token is None:
                                t_first_token = now
                            else:
                                token_times.append(now)
                            token_count += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

    t_end = time.perf_counter()

    ttft = (t_first_token - t_start) if t_first_token else 0
    total = t_end - t_start

    # Compute average ITL from second token onwards
    if len(token_times) >= 2:
        decode_duration = token_times[-1] - t_first_token
        decode_tokens = len(token_times)  # excludes first token
        avg_itl = decode_duration / decode_tokens
        decode_tok_s = decode_tokens / decode_duration
    elif t_first_token:
        decode_duration = t_end - t_first_token
        decode_tokens = max(token_count - 1, 1)
        avg_itl = decode_duration / decode_tokens
        decode_tok_s = decode_tokens / decode_duration
    else:
        avg_itl = 0
        decode_tok_s = 0

    return {
        "tokens": token_count,
        "ttft_s": round(ttft, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "avg_itl_ms": round(avg_itl * 1000, 1),
        "total_s": round(total, 2),
    }


def main():
    if not check_server():
        print(f"ERROR: Server not running at {SERVER}")
        sys.exit(1)

    print(f"Benchmarking {MODEL} at {SERVER}")
    print(f"{'Test':<10} {'TTFT(s)':<9} {'Decode tok/s':<14} {'ITL(ms)':<10} {'Tokens':<8} {'Total(s)'}")
    print("-" * 65)

    results = []

    # Warmup (two requests to ensure all caches are warm)
    print("Warming up...", end="", flush=True)
    run_streaming_test(make_prompt(50), 20)
    run_streaming_test(make_prompt(100), 20)
    print(" done\n")

    for label, target_tokens, max_output in TESTS:
        prompt = make_prompt(target_tokens)
        try:
            r = run_streaming_test(prompt, max_output)
            r["label"] = label
            r["target_input"] = target_tokens
            results.append(r)
            print(
                f"{label:<10} {r['ttft_s']:<9} {r['decode_tok_s']:<14} "
                f"{r['avg_itl_ms']:<10} {r['tokens']:<8} {r['total_s']}"
            )
        except Exception as e:
            print(f"{label:<10} ERROR: {e}")

    print()

    if results:
        rates = [r["decode_tok_s"] for r in results if r["decode_tok_s"] > 0]
        if rates:
            print(f"Decode tok/s: min={min(rates):.2f}, max={max(rates):.2f}, avg={sum(rates)/len(rates):.2f}")

    output_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--output" else None
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
