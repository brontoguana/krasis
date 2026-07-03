#!/usr/bin/env python3
"""Build Step-3.7 llama-witness input-token sources with Krasis tokenization.

This is intentionally Step-specific because Step's HF chat template opens an
initial <think> block but does not expose an enable_thinking variable. Krasis
closes that block for enable_thinking=false; witness capture must consume the
same input_token_ids rather than re-rendering the prompt independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from krasis.tokenizer import Tokenizer

try:
    from tests.reference_contract import build_contract
except ImportError:  # pragma: no cover - allows ./dev python from tests cwd
    from reference_contract import build_contract


def _require_dev_entry() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit(
            "ERROR: tests/step37_witness_inputs.py must be run through ./dev, "
            "not directly."
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_sha256(token_ids: Iterable[int]) -> str:
    canonical = ",".join(str(int(t)) for t in token_ids)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_prompt_source(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"prompt source must be a JSON object: {path}")
    conversations = data.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"prompt source has no conversations: {path}")
    return data


def _iter_prompt_turns(source: Dict[str, Any]) -> Iterable[Tuple[int, int, str]]:
    for ci, conv in enumerate(source.get("conversations", [])):
        turns = conv.get("turns") if isinstance(conv, dict) else None
        if not isinstance(turns, list):
            continue
        for ti, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            prompt = turn.get("prompt")
            if isinstance(prompt, str):
                yield ci, ti, prompt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step witness input-token source")
    parser.add_argument("--model-dir", required=True, help="Step HF model directory")
    parser.add_argument("--prompt-source", required=True, help="fixed prompt-source JSON")
    parser.add_argument("--output", required=True, help="output input-token JSON")
    parser.add_argument("--profile", default="greedy_chat_thinking_off", help="reference profile id")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("KRASIS_WITNESS_MAX_NEW_TOKENS", "1")),
        help="generation length recorded in the contract",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="optional prompt limit")
    return parser.parse_args()


def main() -> None:
    _require_dev_entry()
    args = _parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    prompt_source = Path(args.prompt_source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    if not prompt_source.is_file():
        raise FileNotFoundError(f"prompt source not found: {prompt_source}")

    source = _load_prompt_source(prompt_source)
    tokenizer = Tokenizer(str(model_dir))
    cases: List[Dict[str, Any]] = []
    max_cases = int(args.max_cases or 0)

    for ci, ti, prompt in _iter_prompt_turns(source):
        if max_cases and len(cases) >= max_cases:
            break
        messages = [{"role": "user", "content": prompt}]
        input_token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not input_token_ids:
            raise ValueError(f"empty tokenization for conversation={ci} turn={ti}")
        decoded = tokenizer.decode(input_token_ids, skip_special_tokens=False)
        closed_initial_think = decoded.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        turn = {
            "source_conversation_index": ci,
            "source_turn_index": ti,
            "prompt": prompt,
            "chat_template_application": {
                "method": "krasis.Tokenizer.apply_chat_template",
                "add_generation_prompt": True,
                "requested_enable_thinking": False,
                "krasis_closed_initial_think_block": closed_initial_think,
            },
            "input_token_ids": [int(t) for t in input_token_ids],
            "input_sha256": _input_sha256(input_token_ids),
            "token_ids": [],
        }
        cases.append(
            {
                "source_conversation_index": ci,
                "source_turn_index": ti,
                "turns": [turn],
            }
        )

    if not cases:
        raise ValueError(f"prompt source produced no prompt turns: {prompt_source}")

    contract = build_contract(
        model_name=model_dir.name,
        model_path=str(model_dir),
        tokenizer=tokenizer.tokenizer,
        max_new_tokens=int(args.max_new_tokens),
        add_generation_prompt=True,
        enable_thinking=False,
        profile_id=args.profile,
        prompt_source_path=str(prompt_source),
        runtime_name="krasis-tokenizer",
        runtime_version=None,
        torch_dtype=None,
        extra={
            "source_role": "input_token_source_only",
            "tokenizer_wrapper": "krasis.tokenizer.Tokenizer",
            "step_no_thinking_policy": "close_initial_think_block",
        },
    )
    contract["source_role"] = "input_token_source_only"

    root = {
        "format": "krasis_llama_witness_input_source",
        "format_version": 1,
        "runtime": "krasis-tokenizer",
        "profile_id": args.profile,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model_dir.name,
        "model_path": str(model_dir),
        "source_prompt_path": str(prompt_source),
        "source_prompt_sha256": _sha256_file(prompt_source),
        "max_new_tokens": int(args.max_new_tokens),
        "tokenizer": {
            "source": "krasis.tokenizer.Tokenizer",
            "chat_template_hash": contract.get("tokenizer", {}).get("chat_template_hash"),
            "chat_template_supports_enable_thinking": contract.get("tokenizer", {}).get(
                "chat_template_supports_enable_thinking"
            ),
        },
        "chat_template_application": {
            "method": "krasis.Tokenizer.apply_chat_template",
            "add_generation_prompt": True,
            "requested_enable_thinking": False,
            "step_no_thinking_policy": "close_initial_think_block",
        },
        "contract": contract,
        "conversations": cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(root, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(cases)} Step witness input cases to {output_path}")
    print(f"Prompt source: {prompt_source}")
    print(f"Profile: {args.profile}")


if __name__ == "__main__":
    main()
