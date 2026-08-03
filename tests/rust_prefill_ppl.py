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


CHAT_PROMPT_SUITES: dict[str, list[str]] = {
    "quality-chat-v1": [
        "Hi",
        "What's your name?",
        "Who trained you?",
        "What is 2+2?",
        "Now multiply that by 10",
        "And divide the result by 5",
        "What is the largest animal in the world?",
        "What is the largest body of water in the world?",
        "Describe the binary chop algorithm in depth",
        (
            "If it takes 4 hours for 4 towels to dry on a clothesline in the sun, "
            "how long does it take for 20 towels to dry under the exact same conditions?"
        ),
        "Tell me facts about the blue whale",
        "Tell me more about whales in general",
        "Where do whales live geographically?",
        "Write me a quicksort implementation in Rust",
    ],
}


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
    top_k: int = 1,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_token_ids": input_token_ids,
            "target_token_ids": target_token_ids,
            "top_k": top_k,
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


def _post_reference_test(
    port: int,
    input_token_ids: list[int],
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_token_ids": input_token_ids,
            "max_tokens": max_tokens,
            "top_logprobs": 10,
        },
        separators=(",", ":"),
    )
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/v1/internal/reference_test",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
    finally:
        conn.close()
    if resp.status != 200:
        raise RuntimeError(f"reference_test HTTP {resp.status}: {body[:1000]}")
    return json.loads(body)


def _output_paths(
    model_path: Path,
    stem: str,
    attention_quant: str | None,
    kv_dtype: str | None,
) -> tuple[Path, Path]:
    out_dir = Path("perplexity/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_path.name
    quant = f"{attention_quant}_{kv_dtype}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{model_name}_{stem}_{quant}_{ts}.json"
    log_path = out_dir / f"{model_name}_{stem}_{quant}_{ts}.log"
    return json_path, log_path


def _load_chat_reference(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "krasis_chat_continuation_reference_v1":
        raise ValueError(
            f"{path} is not a krasis_chat_continuation_reference_v1 artifact"
        )
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} has no reference cases")
    for idx, case in enumerate(cases):
        if not isinstance(case.get("input_token_ids"), list):
            raise ValueError(f"{path} case {idx} missing input_token_ids")
        if not isinstance(case.get("continuation_token_ids"), list):
            raise ValueError(f"{path} case {idx} missing continuation_token_ids")
        if not case["continuation_token_ids"]:
            raise ValueError(f"{path} case {idx} has empty continuation_token_ids")
    return data


def _position_rows_by_index(response: dict[str, Any]) -> dict[int, dict[str, Any]]:
    positions = response.get("positions")
    if not isinstance(positions, list):
        raise RuntimeError(f"Bad prefill_logits response: {response}")
    return {int(row["position"]): row for row in positions}


def _normalize_top_k(top_k: Any) -> list[dict[str, float | int]]:
    if not isinstance(top_k, list):
        return []
    normalized: list[dict[str, float | int]] = []
    for rank, entry in enumerate(top_k, 1):
        if not isinstance(entry, dict):
            continue
        token_id = entry.get("token_id")
        logprob = entry.get("logprob")
        if token_id is None or logprob is None:
            continue
        normalized.append(
            {
                "rank": int(entry.get("rank", rank)),
                "token_id": int(token_id),
                "logprob": float(logprob),
            }
        )
    return normalized


def _top_k_distribution(top_k: list[dict[str, float | int]]) -> dict[int, float]:
    probs: dict[int, float] = {}
    for entry in top_k:
        token_id = int(entry["token_id"])
        probs[token_id] = math.exp(float(entry["logprob"]))
    return probs


def _normalized_js_divergence(
    left_top_k: list[dict[str, float | int]],
    right_top_k: list[dict[str, float | int]],
) -> float | None:
    """Return Jensen-Shannon divergence over the top-k union, normalized to [0, 1]."""
    left = _top_k_distribution(left_top_k)
    right = _top_k_distribution(right_top_k)
    ids = set(left) | set(right)
    if not ids:
        return None
    left_mass = sum(left.values())
    right_mass = sum(right.values())
    if left_mass <= 0.0 or right_mass <= 0.0:
        return None
    for token_id in list(left):
        left[token_id] /= left_mass
    for token_id in list(right):
        right[token_id] /= right_mass

    js = 0.0
    for token_id in ids:
        p = left.get(token_id, 0.0)
        q = right.get(token_id, 0.0)
        m = 0.5 * (p + q)
        if p > 0.0:
            js += 0.5 * p * math.log(p / m)
        if q > 0.0:
            js += 0.5 * q * math.log(q / m)
    return js / math.log(2.0)


def _extract_reference_token_diagnostics(
    port: int,
    prompt_ids: list[int],
    continuation: list[int],
    timeout: int,
) -> list[dict[str, Any]]:
    full = prompt_ids + continuation
    targets = full[1:] + [0]
    response = _post_prefill_logits(port, full, targets, timeout, top_k=10)
    rows = _position_rows_by_index(response)
    diagnostics: list[dict[str, Any]] = []
    first_pos = len(prompt_ids) - 1
    for rel_idx, token_id in enumerate(continuation):
        pos = first_pos + rel_idx
        row = rows.get(pos)
        if row is None:
            raise RuntimeError(f"Missing BF16 reference logprob row for position={pos}")
        top_k = _normalize_top_k(row.get("top_k", []))
        diagnostics.append(
            {
                "position": pos,
                "token_id": int(token_id),
                "target_logprob": float(row["target_logprob"]),
                "top_k": top_k,
            }
        )
    return diagnostics


def _build_chat_reference(
    args: argparse.Namespace,
    conf: dict[str, str],
    model_path: Path,
    port: int,
) -> dict[str, Any]:
    from krasis.tokenizer import Tokenizer as KrasisTokenizer

    prompts = CHAT_PROMPT_SUITES[args.prompt_suite]
    tokenizer = KrasisTokenizer(str(model_path))
    cases: list[dict[str, Any]] = []
    print("\n  Building chat-continuation reference...")
    for case_idx, prompt in enumerate(prompts, 1):
        input_token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response = _post_reference_test(
            port,
            input_token_ids,
            args.max_new_tokens,
            args.request_timeout,
        )
        continuation = [int(tid) for tid in response.get("token_ids", [])]
        if not continuation:
            raise RuntimeError(f"Reference generation case {case_idx} returned no tokens")
        token_diagnostics = _extract_reference_token_diagnostics(
            port,
            [int(tid) for tid in input_token_ids],
            continuation,
            args.request_timeout,
        )
        cases.append(
            {
                "case_index": case_idx,
                "prompt": prompt,
                "input_token_ids": [int(tid) for tid in input_token_ids],
                "continuation_token_ids": continuation,
                "continuation_text": response.get("text", ""),
                "reference_token_diagnostics": token_diagnostics,
            }
        )
        print(
            f"    reference {case_idx:02d}/{len(prompts)}: "
            f"{len(input_token_ids)} prompt tokens + {len(continuation)} continuation tokens",
            flush=True,
        )

    reference = {
        "schema": "krasis_chat_continuation_reference_v1",
        "date": datetime.now().isoformat(),
        "prompt_suite": args.prompt_suite,
        "model_path": str(model_path),
        "source_config_path": str(args.config),
        "source_attention_quant": conf.get("CFG_ATTENTION_QUANT"),
        "source_kv_dtype": conf.get("CFG_KV_DTYPE"),
        "max_new_tokens": args.max_new_tokens,
        "num_cases": len(cases),
        "total_continuation_tokens": sum(
            len(case["continuation_token_ids"]) for case in cases
        ),
        "cases": cases,
    }
    if args.write_reference_json:
        args.write_reference_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_reference_json.write_text(
            json.dumps(reference, indent=2),
            encoding="utf-8",
        )
        print(f"  Wrote reference: {args.write_reference_json}")
    return reference


def measure_chat_continuation(args: argparse.Namespace) -> dict[str, Any]:
    conf = _parse_conf(args.config)
    model_path = Path(conf["MODEL_PATH"]).expanduser()
    port = int(conf.get("CFG_PORT", args.port or 18181))
    if args.prompt_suite not in CHAT_PROMPT_SUITES:
        raise ValueError(
            f"Unknown prompt suite {args.prompt_suite!r}; "
            f"available: {', '.join(sorted(CHAT_PROMPT_SUITES))}"
        )
    if bool(args.reference_json) == bool(args.write_reference_json):
        raise ValueError(
            "chat-continuation mode requires exactly one of "
            "--reference-json or --write-reference-json"
        )

    print("Krasis Chat-Continuation Perplexity")
    print("=" * 50)
    print(f"Config:       {args.config}")
    print(f"Model:        {model_path}")
    print(f"Prompt suite: {args.prompt_suite}")
    print(f"Port:         {port}")
    print(f"Attention:    {conf.get('CFG_ATTENTION_QUANT', 'unknown')}")
    print(f"KV dtype:     {conf.get('CFG_KV_DTYPE', 'unknown')}")
    print(f"Max new toks: {args.max_new_tokens}")

    if args.reference_json:
        reference = _load_chat_reference(args.reference_json)
        print(f"Reference:    {args.reference_json}")
    else:
        reference = _build_chat_reference(args, conf, model_path, port)

    if reference.get("prompt_suite") != args.prompt_suite:
        raise ValueError(
            f"Reference prompt suite {reference.get('prompt_suite')!r} does not "
            f"match requested suite {args.prompt_suite!r}"
        )

    total_nll = 0.0
    total_scored = 0
    total_top1 = 0
    total_top10 = 0
    total_ref_top1 = 0
    total_ref_top10 = 0
    total_ref_compared = 0
    total_js = 0.0
    total_js_count = 0
    max_js: float | None = None
    case_results: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    cases = reference["cases"]
    has_reference_diagnostics = any(
        isinstance(case.get("reference_token_diagnostics"), list)
        and bool(case["reference_token_diagnostics"])
        for case in cases
        if isinstance(case, dict)
    )
    print("\n  Scoring chat continuations...")
    if not has_reference_diagnostics:
        print(
            "  WARNING: reference JSON has no BF16 top-k diagnostics; "
            "BF16 top-k comparison fields will be unavailable.",
            flush=True,
        )
    for case_idx, case in enumerate(cases, 1):
        prompt_ids = [int(tid) for tid in case["input_token_ids"]]
        continuation = [int(tid) for tid in case["continuation_token_ids"]]
        full = prompt_ids + continuation
        targets = full[1:] + [0]
        response = _post_prefill_logits(
            port,
            full,
            targets,
            args.request_timeout,
            top_k=10,
        )
        rows = _position_rows_by_index(response)
        ref_diagnostics = {
            int(row["position"]): row
            for row in case.get("reference_token_diagnostics", [])
            if isinstance(row, dict) and row.get("position") is not None
        }

        case_nll = 0.0
        case_scored = 0
        case_top1 = 0
        case_top10 = 0
        case_ref_top1 = 0
        case_ref_top10 = 0
        case_ref_compared = 0
        case_js_values: list[float] = []
        first_pos = len(prompt_ids) - 1
        for rel_idx, token_id in enumerate(continuation):
            pos = first_pos + rel_idx
            row = rows.get(pos)
            if row is None:
                raise RuntimeError(
                    f"Missing logprob row for case={case_idx} position={pos}"
                )
            lp = row.get("target_logprob")
            if lp is None:
                raise RuntimeError(
                    f"Missing target_logprob for case={case_idx} position={pos}"
                )
            top_ids = [int(entry["token_id"]) for entry in row.get("top_k", [])]
            case_nll += -float(lp)
            case_scored += 1
            case_top1 += int(bool(top_ids) and top_ids[0] == token_id)
            case_top10 += int(token_id in top_ids)

            ref_diag = ref_diagnostics.get(pos)
            if ref_diag is not None:
                ref_top_k = _normalize_top_k(ref_diag.get("top_k", []))
                our_top_k = _normalize_top_k(row.get("top_k", []))
                if ref_top_k and our_top_k:
                    ref_top_id = int(ref_top_k[0]["token_id"])
                    our_top_ids = [int(entry["token_id"]) for entry in our_top_k]
                    case_ref_top1 += int(our_top_ids[0] == ref_top_id)
                    case_ref_top10 += int(ref_top_id in our_top_ids)
                    case_ref_compared += 1
                    js = _normalized_js_divergence(ref_top_k, our_top_k)
                    if js is not None:
                        case_js_values.append(js)
                        total_js += js
                        total_js_count += 1
                        max_js = js if max_js is None else max(max_js, js)

        case_ppl = math.exp(case_nll / case_scored)
        total_nll += case_nll
        total_scored += case_scored
        total_top1 += case_top1
        total_top10 += case_top10
        total_ref_top1 += case_ref_top1
        total_ref_top10 += case_ref_top10
        total_ref_compared += case_ref_compared
        case_results.append(
            {
                "case_index": case_idx,
                "prompt": case.get("prompt"),
                "num_tokens_scored": case_scored,
                "total_nll": case_nll,
                "mean_loss": case_nll / case_scored,
                "perplexity": case_ppl,
                "top1": case_top1,
                "top10": case_top10,
                "bf16_top1": case_ref_top1,
                "bf16_top10": case_ref_top10,
                "bf16_positions_compared": case_ref_compared,
                "bf16_top_k_js_avg": (
                    sum(case_js_values) / len(case_js_values)
                    if case_js_values else None
                ),
                "bf16_top_k_js_max": max(case_js_values) if case_js_values else None,
            }
        )
        running_ppl = math.exp(total_nll / total_scored)
        ref_text = (
            f" | bf16_top1={case_ref_top1}/{case_ref_compared} "
            f"bf16_top10={case_ref_top10}/{case_ref_compared}"
            if case_ref_compared
            else ""
        )
        print(
            f"  Case {case_idx:02d}/{len(cases)} | scored {case_scored:3d} | "
            f"PPL={case_ppl:.4f} | top1={case_top1}/{case_scored} | "
            f"running={running_ppl:.4f}{ref_text}",
            flush=True,
        )

    elapsed = time.perf_counter() - t_start
    mean_loss = total_nll / total_scored
    results = {
        "schema": "krasis_chat_continuation_ppl_v1",
        "mode": "chat-continuation",
        "date": datetime.now().isoformat(),
        "config_path": str(args.config),
        "model_path": str(model_path),
        "prompt_suite": args.prompt_suite,
        "reference_json": str(args.reference_json) if args.reference_json else None,
        "written_reference_json": (
            str(args.write_reference_json) if args.write_reference_json else None
        ),
        "reference_source_attention_quant": reference.get("source_attention_quant"),
        "reference_source_kv_dtype": reference.get("source_kv_dtype"),
        "attention_quant": conf.get("CFG_ATTENTION_QUANT"),
        "kv_dtype": conf.get("CFG_KV_DTYPE"),
        "max_new_tokens": args.max_new_tokens,
        "num_cases": len(cases),
        "num_tokens_scored": total_scored,
        "total_nll": total_nll,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "bits_per_token": mean_loss / math.log(2),
        "top1": total_top1,
        "top10": total_top10,
        "bf16_top1": total_ref_top1,
        "bf16_top10": total_ref_top10,
        "bf16_positions_compared": total_ref_compared,
        "bf16_top_k_js_avg": (
            total_js / total_js_count if total_js_count else None
        ),
        "bf16_top_k_js_max": max_js,
        "bf16_reference_diagnostics": total_ref_compared > 0,
        "elapsed_s": elapsed,
        "throughput_tok_s": total_scored / elapsed,
        "cases": case_results,
    }

    json_path, log_path = _output_paths(
        model_path,
        f"{args.prompt_suite}_chat_continuation",
        results["attention_quant"],
        results["kv_dtype"],
    )
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log_path.write_text(
        "\n".join(
            [
                "Krasis Chat-Continuation Perplexity",
                "=" * 50,
                f"Date:          {results['date']}",
                f"Config:        {results['config_path']}",
                f"Model:         {results['model_path']}",
                f"Prompt suite:  {results['prompt_suite']}",
                f"Reference:     {results['reference_json'] or results['written_reference_json']}",
                f"Attention:     {results['attention_quant']}",
                f"KV dtype:      {results['kv_dtype']}",
                f"Perplexity:    {results['perplexity']:.4f}",
                f"Bits/token:    {results['bits_per_token']:.4f}",
                f"Mean loss:     {results['mean_loss']:.6f}",
                f"Tokens scored: {results['num_tokens_scored']}",
                f"Top-1:         {results['top1']} / {results['num_tokens_scored']}",
                f"Top-10:        {results['top10']} / {results['num_tokens_scored']}",
                f"BF16 top-1:    {results['bf16_top1']} / {results['bf16_positions_compared']}",
                f"BF16 top-10:   {results['bf16_top10']} / {results['bf16_positions_compared']}",
                (
                    f"BF16 top-k JS: avg {results['bf16_top_k_js_avg']:.6f}, "
                    f"max {results['bf16_top_k_js_max']:.6f}"
                    if results["bf16_top_k_js_avg"] is not None
                    else "BF16 top-k JS: n/a"
                ),
                f"Cases:         {results['num_cases']}",
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
    print(f"  CHAT-CONTINUATION PPL COMPLETE - {args.prompt_suite}")
    print("=" * 56)
    print(f"  Perplexity:    {results['perplexity']:.4f}")
    print(f"  Bits/token:    {results['bits_per_token']:.4f}")
    print(f"  Tokens scored: {total_scored:,}")
    print(f"  Top-1:         {total_top1:,} / {total_scored:,}")
    print(f"  Top-10:        {total_top10:,} / {total_scored:,}")
    if total_ref_compared:
        print(f"  BF16 top-1:    {total_ref_top1:,} / {total_ref_compared:,}")
        print(f"  BF16 top-10:   {total_ref_top10:,} / {total_ref_compared:,}")
        print(
            f"  BF16 top-k JS: avg {results['bf16_top_k_js_avg']:.6f}, "
            f"max {results['bf16_top_k_js_max']:.6f}"
        )
    print(f"  Elapsed:       {elapsed:.1f}s ({results['throughput_tok_s']:.0f} tok/s)")
    print(f"  Log:           {log_path}")
    print("=" * 56)

    return results


def measure_dataset(args: argparse.Namespace) -> dict[str, Any]:
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
    if args.max_windows is not None:
        if args.max_windows <= 0:
            raise ValueError(f"--max-windows must be positive, got {args.max_windows}")
        starts = starts[: args.max_windows]
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
        "mode": "dataset",
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

    json_path, log_path = _output_paths(
        model_path,
        f"{args.dataset}_rust_prefill",
        results["attention_quant"],
        results["kv_dtype"],
    )
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
    parser.add_argument(
        "--mode",
        choices=["dataset", "chat-continuation"],
        default="dataset",
        help="dataset scores raw benchmark text; chat-continuation scores fixed chat continuations from a reference artifact",
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--dataset", default="wikitext-2", choices=sorted(DATASETS))
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--prompt-suite", default="quality-chat-v1")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--reference-json", type=Path, default=None)
    parser.add_argument("--write-reference-json", type=Path, default=None)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.mode == "chat-continuation":
        measure_chat_continuation(args)
    else:
        measure_dataset(args)


if __name__ == "__main__":
    main()
