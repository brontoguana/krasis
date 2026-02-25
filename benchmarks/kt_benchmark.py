#!/usr/bin/env python3
"""Benchmark KTransformers/SGLang server with TTFT measurement."""
import sys
import time
import json
import urllib.request

SERVER_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
PROMPT_FILE = "/home/main/Documents/Claude/benchmark_prompt.txt"
MAX_TOKENS = 50

with open(PROMPT_FILE) as f:
    prompt_text = f.read()

print(f"Prompt: {len(prompt_text)} chars")

# Use /v1/chat/completions with streaming to measure TTFT
payload = json.dumps({
    "model": "default",
    "messages": [{"role": "user", "content": prompt_text}],
    "max_tokens": MAX_TOKENS,
    "temperature": 0.6,
    "stream": True,
}).encode()

req = urllib.request.Request(
    f"{SERVER_URL}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)

t_start = time.perf_counter()
first_token_time = None
token_count = 0
output_text = ""

try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        buffer = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buffer += chunk
            # Process complete SSE lines
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            output_text += content
                except json.JSONDecodeError:
                    pass
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

t_end = time.perf_counter()

if first_token_time is None:
    print("ERROR: No tokens received")
    sys.exit(1)

ttft = first_token_time - t_start
total_time = t_end - t_start
decode_time = t_end - first_token_time
decode_speed = (token_count - 1) / decode_time if token_count > 1 and decode_time > 0 else 0
# Approximate prefill speed: prompt tokens / TTFT
est_prompt_tokens = len(prompt_text) / 5.2  # rough estimate
prefill_speed = est_prompt_tokens / ttft if ttft > 0 else 0

print(f"\n=== Results ===")
print(f"TTFT: {ttft:.2f}s")
print(f"Tokens generated: {token_count}")
print(f"Decode speed: {decode_speed:.2f} tok/s")
print(f"Prefill speed (est): {prefill_speed:.1f} tok/s")
print(f"Total time: {total_time:.2f}s")
print(f"\nOutput preview: {output_text[:200]}...")
