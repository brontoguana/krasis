#!/usr/bin/env python3
"""Build immutable teacher-forced witness cases from a frozen ADQ task state."""

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


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev adq-ledger-witness-cases; direct execution is unsupported.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument(
        "--trajectory-source",
        required=True,
        choices=(
            "deterministic-replay",
            "deterministic-current-replay",
            "accepted-retokenized-visible-text",
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--terminal-only",
        action="store_true",
        help="build only the final accepted boundary for an independently requested continuation",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"witness input already exists; evidence is immutable: {args.output}")

    manifest = json.loads((args.state_dir / "manifest.json").read_text())
    source_task = str(manifest.get("source_task", "ledger"))
    input_artifact = json.loads((args.state_dir / "preloop-input-token-ids.json").read_text())
    output_artifact = json.loads((args.state_dir / "accepted-output-token-ids.json").read_text())
    input_ids = [int(token) for token in input_artifact["input_token_ids"]]
    terminal_stop = output_artifact.get("terminal_stop_token_id")
    if not isinstance(terminal_stop, int):
        raise SystemExit("frozen accepted output has no integer terminal stop token")
    replay: dict[str, Any] | None = None
    if args.trajectory_source in {
        "deterministic-replay",
        "deterministic-current-replay",
    }:
        if args.replay_report is None:
            parser.error("--replay-report is required for a deterministic replay trajectory")
        replay = json.loads(args.replay_report.read_text())
        if not replay.get("deterministic"):
            raise SystemExit(
                "replay report does not establish a deterministic raw-token reconstruction"
            )
        if (
            args.trajectory_source == "deterministic-replay"
            and not replay.get("reproduces_accepted_observables")
        ):
            raise SystemExit(
                "replay report does not establish a deterministic accepted-observable reconstruction"
            )
        replay_ids = [int(token) for token in replay["reconstructed_token_ids"]]
        if not replay_ids or replay_ids[-1] != terminal_stop:
            raise SystemExit("reconstructed sequence does not end in the frozen terminal stop token")
        output_ids = replay_ids[:-1]
        token_identity_basis = (
            "direct deterministic raw-token replay of the accepted observable trajectory "
            "including terminal stop"
            if args.trajectory_source == "deterministic-replay"
            else "direct deterministic current-runtime raw-token replay including terminal stop; "
            "this is a post-failure diagnostic trajectory, not the accepted campaign output"
        )
        original_token_ids_retained = True
    else:
        if args.replay_report is not None:
            parser.error(
                "--replay-report is incompatible with accepted-retokenized-visible-text"
            )
        output_ids = [
            int(token)
            for token in output_artifact["retokenized_visible_candidate_token_ids"]
        ]
        expected_visible_count = int(output_artifact["retokenized_visible_token_count"])
        if len(output_ids) != expected_visible_count:
            raise SystemExit(
                "retokenized visible candidate length does not match its frozen count"
            )
        if canonical_sha256(output_ids) != output_artifact["retokenized_visible_candidate_sha256"]:
            raise SystemExit("retokenized visible candidate hash mismatch")
        token_identity_basis = (
            "aggregate accepted visible text retokenized with the frozen checkpoint tokenizer; "
            "original SSE token boundaries were not retained"
        )
        original_token_ids_retained = False
    visible_count = len(output_ids)
    checkpoints = [int(value.strip()) for value in args.checkpoints.split(",") if value.strip()]
    if not checkpoints or checkpoints != sorted(set(checkpoints)):
        parser.error("--checkpoints must be a non-empty strictly increasing CSV")
    if args.terminal_only and checkpoints != [visible_count]:
        parser.error(
            f"--terminal-only requires exactly the final visible boundary {visible_count}"
        )
    if not args.terminal_only and (checkpoints[0] != 0 or checkpoints[-1] != visible_count):
        parser.error(
            f"checkpoints must include 0 and the final visible token boundary {visible_count}"
        )
    if any(value < 0 or value > visible_count for value in checkpoints):
        parser.error(f"checkpoints must be within 0..{visible_count}")

    turns = []
    for checkpoint in checkpoints:
        ids = input_ids + output_ids[:checkpoint]
        turns.append(
            {
                "prompt": (
                    f"current-{source_task}-output-prefix-{checkpoint}"
                    if args.trajectory_source == "deterministic-current-replay"
                    else f"accepted-{source_task}-output-prefix-{checkpoint}"
                ),
                "input_token_ids": ids,
                "input_sha256": canonical_sha256(ids),
                "checkpoint_output_tokens": checkpoint,
                "accepted_next_token_id": (
                    output_ids[checkpoint] if checkpoint < visible_count else terminal_stop
                ),
                "chat_template_application": {
                    "mode": "exact_raw_token_ids",
                    "source_state_manifest_sha256": canonical_sha256(manifest),
                },
            }
        )

    artifact = {
        "format": "krasis_adq_task_teacher_forced_inputs",
        "format_version": 2,
        "source_task": source_task,
        "profile_id": args.profile,
        "source_state_manifest_sha256": canonical_sha256(manifest),
        "trajectory_source": args.trajectory_source,
        "source_replay_report_sha256": (
            canonical_sha256(replay) if replay is not None else None
        ),
        "original_output_token_ids_retained": original_token_ids_retained,
        "token_identity_basis": token_identity_basis,
        "base_input_token_count": len(input_ids),
        "accepted_visible_candidate_token_count": visible_count,
        "server_recorded_generated_token_count": int(
            output_artifact["recorded_generated_token_count"]
        ),
        "visible_output_token_count": visible_count,
        "terminal_stop_token_id": terminal_stop,
        "checkpoints": checkpoints,
        "checkpoint_policy": "frozen before source-precision outcomes",
        "terminal_only": args.terminal_only,
        "conversations": [{"turns": turns}],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"Built {len(turns)} immutable teacher-forced witness cases: "
        f"base={len(input_ids)} checkpoints={checkpoints}"
    )


if __name__ == "__main__":
    main()
