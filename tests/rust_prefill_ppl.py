#!/usr/bin/env python3
"""Perplexity measurement through Krasis Rust prefill test endpoint.

Run via `./dev quality-ppl <config>` while a matching `./dev run <config>
--test-endpoints` server is running.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perplexity.measure_ppl import DATASETS, load_dataset_text


def _require_dev_script() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit(
            "tests/rust_prefill_ppl.py must be run through ./dev quality-ppl"
        )


def _parse_conf(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip("\"'")
    return values


def _post_prefill_logits(
    port: int,
    input_token_ids: list[int],
    target_token_ids: list[int],
    timeout: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_token_ids": input_token_ids,
            "target_token_ids": target_token_ids,
            "top_k": 1,
            "sample_every": 1,
        },
        separators=(",", ":"),
    )
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/v1/internal/prefill_logits",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
    finally:
        conn.close()
    if resp.status != 200:
        raise RuntimeError(f"prefill_logits HTTP {resp.status}: {body[:1000]}")
    return json.loads(body)


def measure(args: argparse.Namespace) -> dict[str, Any]:
    conf = _parse_conf(args.config)
    model_path = Path(conf["MODEL_PATH"]).expanduser()
    port = int(conf.get("CFG_PORT", args.port or 18181))
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")

    print("Krasis Rust Prefill Perplexity")
    print("=" * 50)
    print(f"Config:      {args.config}")
    print(f"Model:       {model_path}")
    print(f"Dataset:     {args.dataset}")
    print(f"Port:        {port}")
    print(f"Attention:   {conf.get('CFG_ATTENTION_QUANT', 'unknown')}")
    print(f"KV dtype:    {conf.get('CFG_KV_DTYPE', 'unknown')}")
    print(f"Window:      {args.window_size}, stride={args.stride}")
    if args.max_tokens:
        print(f"Max tokens:  {args.max_tokens}")

    print("\n  Loading dataset...")
    text = load_dataset_text(args.dataset)
    if args.max_tokens and len(text) > args.max_tokens * 6:
        text = text[: args.max_tokens * 6]

    print("  Tokenizing...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    if args.max_tokens:
        tokens = tokens[: args.max_tokens]
    if len(tokens) < 2:
        raise ValueError(f"Need at least 2 tokens, got {len(tokens)}")
    print(f"  Tokens: {len(tokens):,}")

    stride = args.stride
    starts = list(range(0, len(tokens) - 1, stride))
    total_nll = 0.0
    total_scored = 0
    windows = 0
    t_start = time.perf_counter()

    for win_idx, begin in enumerate(starts):
        end = min(begin + args.window_size, len(tokens))
        window = tokens[begin:end]
        if len(window) < 2:
            break
        targets = tokens[begin + 1 : end] + [0]

        response = _post_prefill_logits(port, window, targets, args.request_timeout)
        positions = response.get("positions")
        if not isinstance(positions, list):
            raise RuntimeError(f"Bad prefill_logits response: {response}")

        score_start = 0 if begin == 0 else stride - 1
        scored_this = 0
        for row in positions:
            pos = int(row["position"])
            if pos < score_start or pos >= len(window) - 1:
                continue
            lp = row.get("target_logprob")
            if lp is None:
                raise RuntimeError(
                    f"Missing target_logprob for window={win_idx} position={pos}"
                )
            total_nll += -float(lp)
            total_scored += 1
            scored_this += 1

        windows += 1
        elapsed = time.perf_counter() - t_start
        ppl = math.exp(total_nll / total_scored) if total_scored else float("inf")
        tok_s = total_scored / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * (win_idx + 1) / len(starts)
        print(
            f"  [{pct:5.1f}%] Window {win_idx + 1}/{len(starts)} | "
            f"scored {scored_this} this, {total_scored}/{len(tokens)} total | "
            f"PPL={ppl:.4f} | {tok_s:.0f} tok/s",
            flush=True,
        )

    if total_scored == 0:
        raise RuntimeError("No tokens scored")

    elapsed = time.perf_counter() - t_start
    mean_loss = total_nll / total_scored
    results = {
        "schema": "krasis_rust_prefill_ppl_v1",
        "date": datetime.now().isoformat(),
        "config_path": str(args.config),
        "model_path": str(model_path),
        "dataset": args.dataset,
        "attention_quant": conf.get("CFG_ATTENTION_QUANT"),
        "kv_dtype": conf.get("CFG_KV_DTYPE"),
        "window_size": args.window_size,
        "stride": stride,
        "max_tokens": args.max_tokens,
        "num_tokens_total": len(tokens),
        "num_tokens_scored": total_scored,
        "num_windows": windows,
        "total_nll": total_nll,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "bits_per_char": mean_loss / math.log(2),
        "elapsed_s": elapsed,
        "throughput_tok_s": total_scored / elapsed,
    }

    out_dir = Path("perplexity/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_path.name
    quant = f"{results['attention_quant']}_{results['kv_dtype']}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{model_name}_{args.dataset}_{quant}_rust_prefill_{ts}.json"
    log_path = out_dir / f"{model_name}_{args.dataset}_{quant}_rust_prefill_{ts}.log"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log_path.write_text(
        "\n".join(
            [
                "Krasis Rust Prefill Perplexity",
                "=" * 50,
                f"Date:          {results['date']}",
                f"Config:        {results['config_path']}",
                f"Model:         {results['model_path']}",
                f"Dataset:       {results['dataset']}",
                f"Attention:     {results['attention_quant']}",
                f"KV dtype:      {results['kv_dtype']}",
                f"Window size:   {results['window_size']}",
                f"Stride:        {results['stride']}",
                f"Perplexity:    {results['perplexity']:.4f}",
                f"BPC:           {results['bits_per_char']:.4f}",
                f"Mean loss:     {results['mean_loss']:.6f}",
                f"Tokens scored: {results['num_tokens_scored']} / {results['num_tokens_total']}",
                f"Windows:       {results['num_windows']}",
                f"Elapsed:       {results['elapsed_s']:.1f}s",
                f"Throughput:    {results['throughput_tok_s']:.0f} tok/s",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    results["json_path"] = str(json_path)
    results["log_path"] = str(log_path)

    print()
    print("=" * 56)
    print(f"  RUST PREFILL PPL COMPLETE - {args.dataset}")
    print("=" * 56)
    print(f"  Perplexity:    {results['perplexity']:.4f}")
    print(f"  BPC:           {results['bits_per_char']:.4f}")
    print(f"  Tokens scored: {total_scored:,}")
    print(f"  Elapsed:       {elapsed:.1f}s ({results['throughput_tok_s']:.0f} tok/s)")
    print(f"  Log:           {log_path}")
    print("=" * 56)

    return results


def main() -> None:
    _require_dev_script()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--dataset", default="wikitext-2", choices=sorted(DATASETS))
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()
    measure(args)


if __name__ == "__main__":
    main()
