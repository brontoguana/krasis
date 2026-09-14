#!/usr/bin/env python3
"""Review source-precision decisions along a frozen failed ADQ task trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def witness_input_sha256(token_ids: list[int]) -> str:
    """Match llama-witness's documented comma-separated token-ID hash."""
    return hashlib.sha256(
        ",".join(str(token_id) for token_id in token_ids).encode("utf-8")
    ).hexdigest()


def _top_k(turn: dict[str, Any]) -> list[dict[str, Any]]:
    rows = turn.get("per_token_data") or []
    if rows:
        return list(rows[0].get("top_k") or [])
    positions = (turn.get("prefill_logits") or {}).get("positions") or []
    return list(positions[0].get("top_k") or []) if positions else []


def build_review(inputs: dict[str, Any], witness: dict[str, Any]) -> dict[str, Any]:
    if inputs.get("format") not in {
        "krasis_adq_ledger_teacher_forced_inputs",
        "krasis_adq_task_teacher_forced_inputs",
    }:
        raise ValueError("input is not a frozen ADQ task teacher-forcing artifact")
    if witness.get("runtime") != "llama-witness":
        raise ValueError("reference artifact is not authored by llama-witness")
    if int(witness.get("max_new_tokens", 0)) != 1:
        raise ValueError("ADQ task checkpoint review requires exactly one witness token per case")

    input_turns = [
        turn
        for conversation in inputs.get("conversations", [])
        for turn in conversation.get("turns", [])
    ]
    witness_turns = [
        turn
        for conversation in witness.get("conversations", [])
        for turn in conversation.get("turns", [])
    ]
    if not input_turns:
        raise ValueError("teacher-forcing artifact has no turns")
    if len(witness_turns) != len(input_turns):
        raise ValueError(
            f"witness turn count {len(witness_turns)} does not match inputs {len(input_turns)}"
        )

    witness_by_prompt: dict[str, dict[str, Any]] = {}
    for turn in witness_turns:
        prompt = str(turn.get("prompt", ""))
        if not prompt or prompt in witness_by_prompt:
            raise ValueError("witness prompts must be present and unique")
        witness_by_prompt[prompt] = turn

    rows: list[dict[str, Any]] = []
    for source in input_turns:
        prompt = str(source.get("prompt", ""))
        observed = witness_by_prompt.get(prompt)
        if observed is None:
            raise ValueError(f"witness is missing checkpoint {prompt!r}")
        expected_ids = [int(value) for value in source.get("input_token_ids", [])]
        observed_ids = [int(value) for value in observed.get("input_token_ids", [])]
        if observed_ids != expected_ids:
            raise ValueError(f"witness input token IDs differ at {prompt!r}")
        expected_hash = source.get("input_sha256")
        if expected_hash != canonical_sha256(expected_ids):
            raise ValueError(f"teacher-forcing input hash is invalid at {prompt!r}")
        observed_hash = observed.get("input_sha256")
        expected_witness_hash = witness_input_sha256(expected_ids)
        if observed_hash != expected_witness_hash:
            raise ValueError(f"witness input hash is invalid at {prompt!r}")

        generated = [int(value) for value in observed.get("token_ids", [])]
        if len(generated) != 1:
            raise ValueError(f"witness did not produce exactly one token at {prompt!r}")
        accepted = int(source["accepted_next_token_id"])
        selected = generated[0]
        top_k = _top_k(observed)
        accepted_entry = next(
            (entry for entry in top_k if int(entry.get("token_id", -1)) == accepted),
            None,
        )
        selected_entry = next(
            (entry for entry in top_k if int(entry.get("token_id", -1)) == selected),
            None,
        )
        rows.append(
            {
                "checkpoint_output_tokens": int(source["checkpoint_output_tokens"]),
                "input_tokens": len(expected_ids),
                "input_sha256_canonical_json": expected_hash,
                "input_sha256_witness_csv": observed_hash,
                "accepted_next_token_id": accepted,
                "source_selected_token_id": selected,
                "source_selected_matches_accepted": selected == accepted,
                "accepted_in_source_top_k": accepted_entry is not None,
                "accepted_source_top_k_rank": (
                    top_k.index(accepted_entry) + 1 if accepted_entry is not None else None
                ),
                "accepted_source_log_prob": (
                    accepted_entry.get("log_prob", accepted_entry.get("logprob"))
                    if accepted_entry is not None
                    else None
                ),
                "source_selected_log_prob": (
                    selected_entry.get("log_prob", selected_entry.get("logprob"))
                    if selected_entry is not None
                    else None
                ),
                "source_top_k": top_k,
                "source_stopped_eos": bool(observed.get("stopped_eos")),
                "source_text": observed.get("text"),
            }
        )

    checkpoints = [row["checkpoint_output_tokens"] for row in rows]
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError("teacher-forcing checkpoints are not strictly increasing")
    terminal_stop = int(inputs["terminal_stop_token_id"])
    final_row = rows[-1]
    return {
        "format": "krasis_adq_task_source_trajectory_review",
        "format_version": 2,
        "source_task": inputs.get("source_task", "ledger"),
        "input_profile_id": inputs.get("profile_id"),
        "witness_profile_id": witness.get("profile_id"),
        "trajectory_source": inputs.get("trajectory_source"),
        "original_output_token_ids_retained": inputs.get(
            "original_output_token_ids_retained"
        ),
        "token_identity_basis": inputs.get("token_identity_basis"),
        "teacher_forcing_scope": (
            "Each row asks the source-precision witness for one next-token decision after "
            "forcing the frozen accepted visible-text prefix. Rows do not form a free-running "
            "source continuation."
        ),
        "terminal_stop_token_id": terminal_stop,
        "rows": rows,
        "checkpoint_count": len(rows),
        "source_selected_matches_accepted_count": sum(
            bool(row["source_selected_matches_accepted"]) for row in rows
        ),
        "source_selected_matches_accepted_rate": sum(
            bool(row["source_selected_matches_accepted"]) for row in rows
        )
        / len(rows),
        "source_matches_accepted_terminal_stop": (
            final_row["source_selected_token_id"] == terminal_stop
        ),
        "source_selected_terminal_token_id": final_row["source_selected_token_id"],
        "terminal_accepted_stop_source_rank": final_row["accepted_source_top_k_rank"],
        "terminal_accepted_stop_source_log_prob": final_row["accepted_source_log_prob"],
        "terminal_source_selected_log_prob": final_row["source_selected_log_prob"],
        "terminal_source_text": final_row["source_text"],
        "terminal_source_top_k": final_row["source_top_k"],
        "causal_verdict": "evidence_only_pending_runtime_root_cause",
    }


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit(
            "Run through ./dev adq-ledger-witness-review; direct execution is unsupported."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--witness", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"review output already exists; evidence is immutable: {args.output}")

    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    witness = json.loads(args.witness.read_text(encoding="utf-8"))
    try:
        review = build_review(inputs, witness)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    print(
        "Reviewed source-precision trajectory checkpoints: "
        f"matches={review['source_selected_matches_accepted_count']}/"
        f"{review['checkpoint_count']} "
        f"terminal_stop_match={review['source_matches_accepted_terminal_stop']}"
    )


if __name__ == "__main__":
    main()
