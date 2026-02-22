"""Debug: run two prefill calls and check if the second one is corrupted."""

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


def generate_and_check(model, prompt, label, max_new_tokens=64):
    """Generate and print detailed diagnostics."""
    tokenizer = model.tokenizer
    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(messages)

    device = torch.device(model.ranks[0].device)
    vocab_size = model.cfg.vocab_size
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Prompt tokens: {len(tokens)}")
    print(f"{'='*60}")

    try:
        with torch.inference_mode():
            prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(tokens), dtype=torch.int32, device=device)

            # Check seq_state before
            print(f"  seq_state.seq_len before: {seq_states[0].seq_len}")
            print(f"  seq_state.pages: {len(seq_states[0].pages)}")

            t0 = time.perf_counter()
            logits = model.forward(prompt_tensor, positions, seq_states)
            t1 = time.perf_counter()

            print(f"  seq_state.seq_len after: {seq_states[0].seq_len}")
            print(f"  Prefill time: {t1-t0:.2f}s ({len(tokens)/(t1-t0):.0f} tok/s)")

            # Check logits
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            print(f"  Logits: shape={logits.shape}, NaN={nan_count}, inf={inf_count}")
            print(f"  Logits stats: mean={logits.float().mean():.4f}, std={logits.float().std():.4f}, "
                  f"max={logits.float().max():.4f}, min={logits.float().min():.4f}")

            if nan_count > 0 or inf_count > 0:
                print(f"  ERROR: NaN/inf in logits!")
                return

            # Top-5 predictions
            probs = torch.softmax(logits[-1:, :], dim=-1)
            top5 = torch.topk(probs, 5, dim=-1)
            print(f"  Top-5 predictions:")
            for i in range(5):
                tid = top5.indices[0, i].item()
                prob = top5.values[0, i].item()
                tok_str = tokenizer.decode([tid])
                print(f"    {i+1}. [{tid}] '{tok_str}' ({prob:.4f})")

            # Generate some tokens
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
                    print(f"  NaN at decode step {step}")
                    break
                next_token = sample(logits, 0.6, 50, 0.95).item()
                if next_token < 0 or next_token >= vocab_size:
                    break
                generated_ids.append(next_token)

            text = tokenizer.decode(generated_ids)
            print(f"\n  Generated ({len(generated_ids)} tokens):")
            print(f"  >>> {text[:200]}")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        for s in seq_states:
            s.free()


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

    # Short prompt that goes through CPU decode path (< threshold)
    generate_and_check(model, "What is the capital of France?", "TEST 1: Short (CPU path)")

    # Same short prompt through GPU prefill (force by lowering threshold)
    # Actually, use a medium prompt that's above threshold
    medium = (
        "Summarize this document: " +
        " ".join(f"Section {i}: The experiment tested approach {i} and found {80+i%20}% accuracy."
                 for i in range(1, 20))
    )
    generate_and_check(model, medium, "TEST 2: Medium (~250 words, FIRST GPU prefill)")

    # Second GPU prefill call with similar length - is this the one that breaks?
    medium2 = (
        "Summarize this report: " +
        " ".join(f"Part {i}: Researchers analyzed data set {i} with {90+i%10}% precision."
                 for i in range(1, 20))
    )
    generate_and_check(model, medium2, "TEST 3: Medium (~250 words, SECOND GPU prefill)")

    # Third call with longer prompt
    long = (
        "Summarize this document: " +
        " ".join(f"Section {i}: The experiment tested approach {i} and found {80+i%20}% accuracy. "
                 f"The latency was {i*10}ms under normal conditions."
                 for i in range(1, 50))
    )
    generate_and_check(model, long, "TEST 4: Long (~1000 words, THIRD GPU prefill)")

    # Repeat short prompt to see if CPU path still works after GPU prefill
    generate_and_check(model, "What is the capital of France?", "TEST 5: Short (CPU path after GPU prefill)")

    print("\n\nDone.")
