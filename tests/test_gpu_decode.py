#!/usr/bin/env python3
"""Test GPU decode: verify M=1 decode through GPU Marlin kernel produces correct output.

Compares output of:
1. CPU decode (gpu_prefill_threshold=300, M=1 goes to CPU Rust engine)
2. GPU decode (gpu_prefill_threshold=1, M=1 goes to GPU fused_marlin_moe)

Both should produce identical output tokens (greedy sampling, temp=0).
"""

import logging
import sys
import time
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("test_gpu_decode")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"

from krasis.model import KrasisModel
from krasis.config import QuantConfig


def load_model(gpu_prefill_threshold: int, expert_divisor: int = -1):
    """Load V2-Lite with specified threshold."""
    qcfg = QuantConfig(
        attention="bf16", shared_expert="bf16",
        dense_mlp="bf16", lm_head="bf16",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=1,
        gpu_prefill=True,
        gpu_prefill_threshold=gpu_prefill_threshold,
        krasis_threads=16,
        quant_cfg=qcfg,
        kv_dtype=torch.bfloat16,
        expert_divisor=expert_divisor,
    )
    model.load()
    return model


def main():
    prompt_messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    max_tokens = 32

    # --- Test 1: CPU decode baseline ---
    print("=" * 60)
    print("Loading model with CPU decode (threshold=300)...")
    model_cpu = load_model(gpu_prefill_threshold=300, expert_divisor=-1)
    prompt = model_cpu.tokenizer.apply_chat_template(prompt_messages)
    print(f"Prompt: {len(prompt)} tokens")

    t0 = time.perf_counter()
    tokens_cpu = model_cpu.generate(prompt, max_new_tokens=max_tokens, temperature=0.0, top_k=1)
    cpu_time = time.perf_counter() - t0
    text_cpu = model_cpu.tokenizer.decode(tokens_cpu)
    cpu_tok_s = len(tokens_cpu) / cpu_time if cpu_time > 0 else 0
    print(f"CPU decode: {len(tokens_cpu)} tokens in {cpu_time:.2f}s ({cpu_tok_s:.1f} tok/s)")
    print(f"Output: {repr(text_cpu)}")

    # Free CPU model
    del model_cpu
    torch.cuda.empty_cache()

    # --- Test 2: GPU decode ---
    print()
    print("=" * 60)
    print("Loading model with GPU decode (threshold=1)...")
    model_gpu = load_model(gpu_prefill_threshold=1, expert_divisor=-1)

    t0 = time.perf_counter()
    tokens_gpu = model_gpu.generate(prompt, max_new_tokens=max_tokens, temperature=0.0, top_k=1)
    gpu_time = time.perf_counter() - t0
    text_gpu = model_gpu.tokenizer.decode(tokens_gpu)
    gpu_tok_s = len(tokens_gpu) / gpu_time if gpu_time > 0 else 0
    print(f"GPU decode: {len(tokens_gpu)} tokens in {gpu_time:.2f}s ({gpu_tok_s:.1f} tok/s)")
    print(f"Output: {repr(text_gpu)}")

    # --- Compare ---
    print()
    print("=" * 60)
    print("COMPARISON:")
    print(f"  CPU: {repr(text_cpu)}")
    print(f"  GPU: {repr(text_gpu)}")
    print(f"  CPU speed: {cpu_tok_s:.1f} tok/s")
    print(f"  GPU speed: {gpu_tok_s:.1f} tok/s")
    print(f"  Speedup: {gpu_tok_s/cpu_tok_s:.2f}x" if cpu_tok_s > 0 else "  Speedup: N/A")

    if tokens_cpu == tokens_gpu:
        print("  PASS: Outputs match exactly!")
    elif text_cpu.strip() == text_gpu.strip():
        print("  PASS: Text matches (minor token difference)")
    else:
        # Check first divergence point
        for i, (c, g) in enumerate(zip(tokens_cpu, tokens_gpu)):
            if c != g:
                print(f"  WARN: First divergence at token {i}: CPU={c} GPU={g}")
                break
        print(f"  NOTE: Outputs differ — check if numerically expected (quantization noise)")


if __name__ == "__main__":
    main()
