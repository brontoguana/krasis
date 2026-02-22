"""Debug: isolate the GPU prefill corruption.

Test 1: Force validation sync (disables DMA pipelining)
Test 2: Normal mode (with DMA pipelining)
"""

import os
import sys
import time
import torch

os.environ.setdefault("KRASIS_LOG", "warn")

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample
from krasis.timing import TIMING

TIMING.decode = False
TIMING.prefill = False


def generate(model, prompt, max_new_tokens=64):
    tokenizer = model.tokenizer
    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(messages)

    device = torch.device(model.ranks[0].device)
    vocab_size = model.cfg.vocab_size
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    try:
        with torch.inference_mode():
            prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(tokens), dtype=torch.int32, device=device)
            t0 = time.perf_counter()
            logits = model.forward(prompt_tensor, positions, seq_states)
            t1 = time.perf_counter()

            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            if has_nan or has_inf:
                return f"[NaN={torch.isnan(logits).sum().item()} inf={torch.isinf(logits).sum().item()}]", len(tokens), t1-t0

            next_token = sample(logits[-1:, :], 0.6, 50, 0.95).item()
            generated_ids = [next_token]
            stop_ids = {model.cfg.eos_token_id}

            for step in range(max_new_tokens - 1):
                if next_token in stop_ids:
                    break
                pos = len(tokens) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
                logits = model.forward(token_tensor, pos_tensor, seq_states)
                if torch.isnan(logits).any():
                    break
                next_token = sample(logits, 0.6, 50, 0.95).item()
                if next_token < 0 or next_token >= vocab_size:
                    break
                generated_ids.append(next_token)

            text = tokenizer.decode(generated_ids)
            return text, len(tokens), t1 - t0

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[ERROR: {e}]", 0, 0
    finally:
        for s in seq_states:
            s.free()


def make_prompt(target_words):
    base = "Summarize this document: "
    n = max(1, target_words // 40)
    sections = [
        f"Section {i}: The experiment tested approach {i} and found {80+i%20}% accuracy "
        f"with latency of {i*10}ms under normal conditions."
        for i in range(1, n + 1)
    ]
    return base + " ".join(sections)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--layer-group-size", type=int, default=2)
    args = parser.parse_args()

    quant_cfg = QuantConfig(attention="int8", shared_expert="int8", dense_mlp="int8", lm_head="int8")

    model = KrasisModel(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        layer_group_size=args.layer_group_size,
        kv_dtype=torch.float8_e4m3fn,
        quant_cfg=quant_cfg,
        krasis_threads=12,
    )
    model.load()

    # Warmup
    generate(model, "Hello!", max_new_tokens=8)

    test_words = [200, 500, 1000, 1500, 2000]

    # ── TEST A: with _validation_sync=True (disables DMA pipelining + adds per-layer sync) ──
    print(f"\n{'='*70}")
    print(f"  TEST A: VALIDATION SYNC MODE (no DMA pipelining)")
    print(f"{'='*70}\n")

    model._validation_sync = True
    for words in test_words:
        prompt = make_prompt(words)
        text, n_tok, prefill_t = generate(model, prompt)
        preview = text.replace('\n', '\\n')[:120]
        print(f"  {words:>5} words ({n_tok:>5} tok, {n_tok/prefill_t:>7.0f} t/s): {preview}")
        sys.stdout.flush()

    # ── TEST B: normal mode (DMA pipelining enabled) ──
    print(f"\n{'='*70}")
    print(f"  TEST B: NORMAL MODE (DMA pipelining enabled)")
    print(f"{'='*70}\n")

    model._validation_sync = False
    for words in test_words:
        prompt = make_prompt(words)
        text, n_tok, prefill_t = generate(model, prompt)
        preview = text.replace('\n', '\\n')[:120]
        print(f"  {words:>5} words ({n_tok:>5} tok, {n_tok/prefill_t:>7.0f} t/s): {preview}")
        sys.stdout.flush()

    # ── TEST C: disable GPU prefill entirely (threshold = 999999) ──
    print(f"\n{'='*70}")
    print(f"  TEST C: GPU PREFILL DISABLED (all CPU decode path)")
    print(f"{'='*70}\n")

    old_threshold = model.gpu_prefill_threshold
    model.gpu_prefill_threshold = 999999
    for layer in model.layers:
        layer.gpu_prefill_threshold = 999999
    for words in test_words:
        prompt = make_prompt(words)
        text, n_tok, prefill_t = generate(model, prompt)
        preview = text.replace('\n', '\\n')[:120]
        print(f"  {words:>5} words ({n_tok:>5} tok, {n_tok/prefill_t:>7.0f} t/s): {preview}")
        sys.stdout.flush()
    model.gpu_prefill_threshold = old_threshold
    for layer in model.layers:
        layer.gpu_prefill_threshold = old_threshold

    print("\nDone.")
