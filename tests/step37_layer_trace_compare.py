#!/usr/bin/env python3
"""Compare Step-3.7 Krasis prefill trace summaries with local BF16 layers.

This is a bring-up diagnostic, not a benchmark. It loads one official Step
decoder layer at a time from safetensors, runs a short prompt through the local
modeling code, and compares row summaries to a Krasis
debug_prefill_device_trace captured from /v1/internal/reference_test.

Run via:
  KRASIS_DEV_SCRIPT=1 ./dev python tests/step37_layer_trace_compare.py ...
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoConfig

from krasis.tokenizer import Tokenizer


def _require_dev_entrypoint() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        print(
            "Run via KRASIS_DEV_SCRIPT=1 ./dev python tests/step37_layer_trace_compare.py",
            file=sys.stderr,
        )
        sys.exit(1)


def _load_step_modeling(model_path: Path):
    package = types.ModuleType("step37_local")
    package.__path__ = [str(model_path)]
    sys.modules["step37_local"] = package
    return importlib.import_module("step37_local.modeling_step3p7")


def _load_index(model_path: Path) -> dict[str, str]:
    with (model_path / "model.safetensors.index.json").open() as f:
        return json.load(f)["weight_map"]


def _load_tensor(model_path: Path, index: dict[str, str], key: str, device: str) -> torch.Tensor:
    with safe_open(model_path / index[key], framework="pt", device=device) as f:
        return f.get_tensor(key)


def _load_layer_state(
    model_path: Path,
    index: dict[str, str],
    layer_idx: int,
    device: str,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}."
    keys = [key for key in index if key.startswith(prefix)]
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)

    state: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(model_path / shard, framework="pt", device=device) as f:
            for key in shard_keys:
                state[key[len(prefix) :]] = f.get_tensor(key)
    return state


def _row_summary(row: torch.Tensor) -> dict[str, float | int]:
    values = row.float().flatten()
    return {
        "l2": float(torch.linalg.vector_norm(values).item()),
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "finite_count": int(torch.isfinite(values).sum().item()),
    }


def _bf16_add(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return (lhs + rhs).to(torch.bfloat16)


def _causal_mask(length: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    mask = torch.full((length, length), torch.finfo(dtype).min, device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1).view(1, 1, length, length)


def _sliding_causal_mask(length: int, window: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    mask = _causal_mask(length, dtype, device).view(length, length)
    if window > 0:
        q_pos = torch.arange(length, device=device).view(length, 1)
        k_pos = torch.arange(length, device=device).view(1, length)
        too_old = k_pos < (q_pos - window + 1)
        mask = mask.masked_fill(too_old, torch.finfo(dtype).min)
    return mask.view(1, 1, length, length)


def _trace_lookup(trace_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    data = json.loads(trace_path.read_text())
    trace = data["debug_prefill_device_trace"]
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in trace["entries"]:
        stage = entry.get("stage", "")
        if not stage.startswith("all_layer_"):
            continue
        lookup[(int(entry["layer"]), stage.replace("all_layer_", ""))] = entry
    return lookup


def _make_layer(layer_cls: Any, cfg: Any, model_path: Path, index: dict[str, str], layer_idx: int, device: str):
    layer = layer_cls(cfg, layer_idx).to(device=device, dtype=torch.bfloat16).eval()
    state = _load_layer_state(model_path, index, layer_idx, device)
    missing, unexpected = layer.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"layer {layer_idx} state mismatch: missing={missing} unexpected={unexpected}"
        )
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return layer


def _run_layer(
    layer: Any,
    hidden: torch.Tensor,
    full_mask: torch.Tensor,
    sliding_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
):
    mask = (
        sliding_mask
        if getattr(layer, "attention_type", None) == "sliding_attention"
        and getattr(layer.self_attn, "sliding_window", None)
        else full_mask
    )
    residual = hidden
    normed = layer.input_layernorm(hidden)
    attn_out, _ = layer.self_attn(
        normed,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_value=None,
        cache_position=None,
    )
    post_attn = _bf16_add(residual, attn_out)
    mlp_in = layer.post_attention_layernorm(post_attn)
    if getattr(layer, "use_moe", False):
        mlp_out = (layer.share_expert(mlp_in) + layer.moe(mlp_in)).to(torch.bfloat16)
    else:
        mlp_out = layer.mlp(mlp_in).to(torch.bfloat16)
    output = _bf16_add(post_attn, mlp_out)
    return post_attn, mlp_out, output


def main() -> None:
    _require_dev_entrypoint()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/main/.krasis/models/Step-3.7-Flash")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Print exactly: Hello")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-layer", type=int, default=44)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    trace_path = Path(args.trace)
    output_path = Path(args.output)
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    modeling = _load_step_modeling(model_path)
    layer_cls = modeling.Step3p7DecoderLayer
    cfg_top = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    cfg = cfg_top.text_config
    cfg._attn_implementation = "eager"
    index = _load_index(model_path)
    lookup = _trace_lookup(trace_path)

    tokenizer = Tokenizer(str(model_path))
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids_tensor = torch.tensor([input_ids], device=args.device, dtype=torch.long)
    embed = _load_tensor(model_path, index, "model.embed_tokens.weight", args.device)
    hidden = (
        embed.index_select(0, ids_tensor.view(-1))
        .view(1, len(input_ids), cfg.hidden_size)
        .to(torch.bfloat16)
    )
    del embed
    gc.collect()
    torch.cuda.empty_cache()

    mask = _causal_mask(len(input_ids), hidden.dtype, args.device)
    sliding_mask = None
    if getattr(cfg, "sliding_window", None):
        sliding_mask = _sliding_causal_mask(
            len(input_ids),
            int(cfg.sliding_window),
            hidden.dtype,
            args.device,
        )
    position_ids = torch.arange(len(input_ids), device=args.device, dtype=torch.long).unsqueeze(0)

    comparisons: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    first_large_delta: dict[str, Any] | None = None
    stages = ("post_attn_residual_last", "post_mlp_hidden_last", "output_sum_last")

    for layer_idx in range(min(args.max_layer, cfg.num_hidden_layers - 1) + 1):
        t0 = time.time()
        layer = _make_layer(layer_cls, cfg, model_path, index, layer_idx, args.device)
        post_attn, mlp_out, output = _run_layer(layer, hidden, mask, sliding_mask, position_ids)
        reference = {
            "layer": layer_idx,
            "post_attn_residual_last": _row_summary(post_attn[0, -1]),
            "post_mlp_hidden_last": _row_summary(mlp_out[0, -1]),
            "output_sum_last": _row_summary(output[0, -1]),
            "elapsed_s": time.time() - t0,
        }
        references.append(reference)
        for stage in stages:
            kra = lookup.get((layer_idx, stage))
            if kra is None:
                raise RuntimeError(f"trace missing layer={layer_idx} stage={stage}")
            ref = reference[stage]
            l2_ratio = float(kra["l2"] / ref["l2"]) if ref["l2"] else None
            mean_delta = float(kra["mean"] - ref["mean"])
            comparison = {
                "layer": layer_idx,
                "stage": stage,
                "reference": ref,
                "krasis": {k: kra[k] for k in ("l2", "mean", "min", "max", "finite_count")},
                "l2_ratio": l2_ratio,
                "mean_delta": mean_delta,
            }
            comparisons.append(comparison)
            if (
                first_large_delta is None
                and l2_ratio is not None
                and (l2_ratio < 0.90 or l2_ratio > 1.10)
            ):
                first_large_delta = comparison
        print(
            json.dumps(
                {
                    "layer": layer_idx,
                    "ref_output_l2": reference["output_sum_last"]["l2"],
                    "krasis_output_l2": lookup[(layer_idx, "output_sum_last")]["l2"],
                    "output_l2_ratio": lookup[(layer_idx, "output_sum_last")]["l2"]
                    / reference["output_sum_last"]["l2"],
                    "elapsed_s": reference["elapsed_s"],
                    "free_total": torch.cuda.mem_get_info(),
                }
            ),
            flush=True,
        )
        hidden = output
        del layer, post_attn, mlp_out, output
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "schema": "step37_layer_trace_compare_v1",
        "model": str(model_path),
        "trace": str(trace_path),
        "prompt": prompt,
        "input_token_ids": input_ids,
        "references": references,
        "comparisons": comparisons,
        "first_large_delta": first_large_delta,
    }
    output_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {output_path}")
    if first_large_delta:
        print(f"first_large_delta={json.dumps(first_large_delta)}")
    else:
        print("first_large_delta=null")


if __name__ == "__main__":
    main()
