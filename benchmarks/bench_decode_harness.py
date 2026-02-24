#!/usr/bin/env python3
"""Synthetic decode benchmark — no model loading required.

Allocates fake weights matching the model's exact dimensions and runs
decode_step in a loop to measure throughput. Memory access patterns are
identical to real inference.

Usage:
    python benchmarks/bench_decode_harness.py [--steps N] [--warmup N] [--timing] [--bits 4|8]
    python benchmarks/bench_decode_harness.py --model ~/.krasis/models/Qwen3-Coder-Next
"""

import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Synthetic decode benchmark")
    parser.add_argument("--model", default=os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next"),
                        help="Path to model directory (needs config.json)")
    parser.add_argument("--steps", type=int, default=100, help="Number of timed decode steps")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--timing", action="store_true", help="Enable per-component timing")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Weight quantization bits")
    parser.add_argument("--max-experts", type=int, default=0,
                        help="Cap number of experts per layer (0=full, reduces allocation time)")
    parser.add_argument("--threads", type=int, default=20,
                        help="Rayon thread count (default 20, must match real model)")
    args = parser.parse_args()

    config_path = os.path.join(args.model, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    # Import krasis (triggers Rust library load)
    t0 = time.perf_counter()
    import krasis
    t1 = time.perf_counter()
    print(f"Library loaded in {t1-t0:.2f}s", file=sys.stderr)

    # Run benchmark (all work happens in Rust)
    krasis.bench_decode_synthetic(
        config_path,
        num_steps=args.steps,
        warmup=args.warmup,
        timing=args.timing,
        num_bits=args.bits,
        max_experts=args.max_experts,
        num_threads=args.threads,
    )

if __name__ == "__main__":
    main()
