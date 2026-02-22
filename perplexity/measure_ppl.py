#!/usr/bin/env python3
"""Perplexity measurement for Krasis models on WikiText-2.

Measures through the PRODUCTION forward path (GPU Marlin prefill or CPU-only)
to validate that quantization doesn't degrade model quality.

Usage:
    # GPU prefill path (production):
    python -m perplexity.measure_ppl --model-path ~/.krasis/DeepSeek-V2-Lite --num-gpus 1

    # CPU-only baseline (no GPU prefill):
    python -m perplexity.measure_ppl --model-path ~/.krasis/DeepSeek-V2-Lite --cpu-only

    # Quick test (first 5000 tokens):
    python -m perplexity.measure_ppl --model-path ~/.krasis/DeepSeek-V2-Lite --max-tokens 5000
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add krasis python path
_script_dir = Path(__file__).resolve().parent
_krasis_root = _script_dir.parent
sys.path.insert(0, str(_krasis_root / "python"))

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

DATASET_CACHE = _script_dir / "datasets" / "wikitext-2-raw-v1-test.txt"


def load_wikitext2_test() -> str:
    """Load WikiText-2-raw-v1 test split. Downloads and caches on first use."""
    if DATASET_CACHE.exists():
        logger.info("Loading cached WikiText-2 from %s", DATASET_CACHE)
        return DATASET_CACHE.read_text(encoding="utf-8")

    logger.info("Downloading WikiText-2-raw-v1 test split...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library required. Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate all lines with newlines (standard for WikiText PPL)
    text = "\n".join(ds["text"])

    DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DATASET_CACHE.write_text(text, encoding="utf-8")
    logger.info("Cached WikiText-2 test to %s (%d chars)", DATASET_CACHE, len(text))
    return text


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def evaluate_perplexity(
    model: KrasisModel,
    tokens: list,
    window_size: int,
    stride: int,
    max_tokens: int | None = None,
) -> dict:
    """Sliding-window perplexity evaluation.

    For each window of `window_size` tokens:
    - Run forward with return_all_logits=True to get [W, V] logits
    - Compute cross-entropy on the non-overlapping region (positions >= stride)
      to avoid double-counting, except for the first window which scores all

    Args:
        model: Loaded KrasisModel
        tokens: Full token list
        window_size: Context window per forward pass
        stride: Step size between windows (typically window_size // 2)
        max_tokens: If set, truncate tokens to this length

    Returns:
        Dict with ppl, bpc, mean_loss, num_tokens_scored, windows, elapsed_s
    """
    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    total_tokens = len(tokens)
    if total_tokens < 2:
        raise ValueError(f"Need at least 2 tokens, got {total_tokens}")

    device = torch.device(model.ranks[0].device)
    total_nll = 0.0
    total_scored = 0
    num_windows = 0

    t_start = time.perf_counter()

    # Compute window positions
    starts = list(range(0, total_tokens - 1, stride))
    total_windows = len(starts)

    for win_idx, begin in enumerate(starts):
        end = min(begin + window_size, total_tokens)
        win_len = end - begin

        if win_len < 2:
            break

        # Create fresh KV state for each window (no cross-window contamination)
        seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

        # Reset linear attention states for hybrid models
        if model.cfg.is_hybrid:
            for layer in model.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        try:
            token_tensor = torch.tensor(tokens[begin:end], dtype=torch.long, device=device)
            positions = torch.arange(win_len, dtype=torch.int32, device=device)

            with torch.inference_mode():
                logits = model.forward(
                    token_tensor, positions, seq_states,
                    return_all_logits=True,
                )  # [W, V]

            # Shift: logits[:-1] predicts tokens[1:]
            shift_logits = logits[:-1, :].float()  # [W-1, V]
            shift_labels = token_tensor[1:]          # [W-1]

            # Cross-entropy per position
            loss_per_pos = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="none"
            )  # [W-1]

            # Only score the non-overlapping region to avoid double-counting.
            # For the first window (begin == 0), score everything.
            # For subsequent windows, only score positions >= stride.
            if begin == 0:
                score_start = 0
            else:
                # In the shifted label space, position i corresponds to
                # predicting token at original position (begin + i + 1).
                # We want to score original positions >= begin + stride,
                # which means shifted positions where (begin + i + 1) >= (begin + stride)
                # i.e., i >= stride - 1
                score_start = stride - 1

            scored_loss = loss_per_pos[score_start:]
            n_scored = scored_loss.shape[0]

            if n_scored > 0:
                total_nll += scored_loss.sum().item()
                total_scored += n_scored

            num_windows += 1

            # Free logits immediately (can be ~1.2 GB for large vocab)
            del logits, shift_logits, loss_per_pos, scored_loss

        finally:
            for s in seq_states:
                s.free()

        # Progress
        elapsed = time.perf_counter() - t_start
        running_ppl = math.exp(total_nll / total_scored) if total_scored > 0 else float("inf")
        tok_per_s = total_scored / elapsed if elapsed > 0 else 0
        print(
            f"\r  Window {win_idx + 1}/{total_windows} | "
            f"scored {total_scored}/{total_tokens} tokens | "
            f"running PPL={running_ppl:.2f} | "
            f"{tok_per_s:.0f} tok/s",
            end="", flush=True,
        )

    elapsed_s = time.perf_counter() - t_start
    print()  # newline after progress

    if total_scored == 0:
        raise ValueError("No tokens scored — check window/stride settings")

    mean_loss = total_nll / total_scored
    ppl = math.exp(mean_loss)
    bpc = mean_loss / math.log(2)

    return {
        "perplexity": ppl,
        "bits_per_char": bpc,
        "mean_loss": mean_loss,
        "total_nll": total_nll,
        "num_tokens_scored": total_scored,
        "num_tokens_total": total_tokens,
        "num_windows": num_windows,
        "window_size": window_size,
        "stride": stride,
        "elapsed_s": elapsed_s,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(results: dict, config: dict, output_dir: Path):
    """Save results as JSON and human-readable log."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(config["model_path"]).name
    mode = "cpu" if config.get("cpu_only") else "gpu"
    gpu_bits = config.get("gpu_expert_bits", 4)
    cpu_bits = config.get("cpu_expert_bits", 4)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    basename = f"{model_name}_{mode}_g{gpu_bits}_c{cpu_bits}_{timestamp}"

    # JSON
    combined = {"config": config, "results": results}
    json_path = output_dir / f"{basename}.json"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    # Human-readable log
    log_path = output_dir / f"{basename}.log"
    lines = [
        f"Krasis Perplexity Measurement",
        f"{'=' * 50}",
        f"Date:          {datetime.now().isoformat()}",
        f"Model:         {config['model_path']}",
        f"Mode:          {'CPU-only' if config.get('cpu_only') else 'GPU prefill (Marlin)'}",
        f"GPUs:          {config.get('num_gpus', 'auto')}",
        f"GPU bits:      {gpu_bits}",
        f"CPU bits:      {cpu_bits}",
        f"Window size:   {results['window_size']}",
        f"Stride:        {results['stride']}",
        f"",
        f"Results",
        f"{'-' * 50}",
        f"Perplexity:    {results['perplexity']:.4f}",
        f"BPC:           {results['bits_per_char']:.4f}",
        f"Mean loss:     {results['mean_loss']:.6f}",
        f"Tokens scored: {results['num_tokens_scored']} / {results['num_tokens_total']}",
        f"Windows:       {results['num_windows']}",
        f"Elapsed:       {results['elapsed_s']:.1f}s",
        f"Throughput:    {results['num_tokens_scored'] / results['elapsed_s']:.0f} tok/s",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Measure WikiText-2 perplexity through Krasis production path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    p.add_argument("--model-path", required=True, help="Path to HF model directory")
    p.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (auto-detect if omitted)")

    # Eval params
    p.add_argument("--window-size", type=int, default=2048, help="Context window per forward pass (default: 2048)")
    p.add_argument("--stride", type=int, default=None, help="Stride between windows (default: window_size // 2)")
    p.add_argument("--max-tokens", type=int, default=None, help="Truncate dataset to N tokens (for quick tests)")

    # Quantization
    p.add_argument("--gpu-expert-bits", type=int, default=4, choices=[4, 8], help="GPU Marlin expert bits (default: 4)")
    p.add_argument("--cpu-expert-bits", type=int, default=4, choices=[4, 8], help="CPU expert bits (default: 4)")
    p.add_argument("--attention-quant", default="int8", choices=["bf16", "int8"], help="Attention quantization (default: int8)")
    p.add_argument("--lm-head-quant", default="int8", choices=["bf16", "int8"], help="LM head quantization (default: int8)")

    # Paths
    p.add_argument("--cpu-only", action="store_true", help="Disable GPU prefill — measure through CPU-only path")
    p.add_argument("--layer-group-size", type=int, default=2, help="Layers per group for prefill DMA (default: 2)")
    p.add_argument("--kv-cache-mb", type=int, default=2000, help="KV cache size in MB (default: 2000)")
    p.add_argument("--krasis-threads", type=int, default=48, help="CPU threads for Rust engine (default: 48)")

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stride = args.stride if args.stride is not None else args.window_size // 2

    print(f"Krasis Perplexity Measurement")
    print(f"{'=' * 50}")
    print(f"Model:       {args.model_path}")
    print(f"Mode:        {'CPU-only' if args.cpu_only else 'GPU prefill (Marlin)'}")
    print(f"GPU bits:    {args.gpu_expert_bits}")
    print(f"CPU bits:    {args.cpu_expert_bits}")
    print(f"Window:      {args.window_size}, stride={stride}")
    if args.max_tokens:
        print(f"Max tokens:  {args.max_tokens}")
    print()

    # ── Load dataset ──
    print("Loading WikiText-2 dataset...")
    text = load_wikitext2_test()
    print(f"  Dataset: {len(text)} chars")

    # ── Load model ──
    print("Loading model...")
    quant_cfg = QuantConfig(
        attention=args.attention_quant,
        shared_expert="int8",
        dense_mlp="int8",
        lm_head=args.lm_head_quant,
        gpu_expert_bits=args.gpu_expert_bits,
        cpu_expert_bits=args.cpu_expert_bits,
    )

    model = KrasisModel(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        layer_group_size=args.layer_group_size,
        kv_dtype=torch.float8_e4m3fn,
        quant_cfg=quant_cfg,
        krasis_threads=args.krasis_threads,
        gpu_prefill=not args.cpu_only,
        kv_cache_mb=args.kv_cache_mb,
    )
    model.load()

    # ── Tokenize (raw text, no chat template) ──
    print("Tokenizing...")
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    print(f"  Tokens: {len(tokens)}")
    if args.max_tokens:
        tokens = tokens[:args.max_tokens]
        print(f"  Truncated to: {len(tokens)}")
    print()

    # ── Evaluate ──
    print("Evaluating perplexity...")
    results = evaluate_perplexity(
        model=model,
        tokens=tokens,
        window_size=args.window_size,
        stride=stride,
    )

    # ── Print results ──
    print()
    print(f"Results")
    print(f"{'-' * 50}")
    print(f"  Perplexity:    {results['perplexity']:.4f}")
    print(f"  BPC:           {results['bits_per_char']:.4f}")
    print(f"  Mean loss:     {results['mean_loss']:.6f}")
    print(f"  Tokens scored: {results['num_tokens_scored']} / {results['num_tokens_total']}")
    print(f"  Windows:       {results['num_windows']}")
    print(f"  Elapsed:       {results['elapsed_s']:.1f}s")
    print(f"  Throughput:    {results['num_tokens_scored'] / results['elapsed_s']:.0f} tok/s")

    # ── Save ──
    config = {
        "model_path": args.model_path,
        "num_gpus": args.num_gpus,
        "cpu_only": args.cpu_only,
        "gpu_expert_bits": args.gpu_expert_bits,
        "cpu_expert_bits": args.cpu_expert_bits,
        "attention_quant": args.attention_quant,
        "lm_head_quant": args.lm_head_quant,
        "layer_group_size": args.layer_group_size,
        "window_size": args.window_size,
        "stride": stride,
        "max_tokens": args.max_tokens,
        "krasis_threads": args.krasis_threads,
        "kv_cache_mb": args.kv_cache_mb,
    }

    results_dir = _script_dir / "results"
    json_path, log_path = save_results(results, config, results_dir)
    print()
    print(f"  Saved: {json_path}")
    print(f"         {log_path}")


if __name__ == "__main__":
    main()
