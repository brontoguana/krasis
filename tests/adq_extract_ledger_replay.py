#!/usr/bin/env python3
"""Freeze and verify an exact OpenCode ADQ task failure replay.

This is an offline/diagnostic artifact builder. It must be run through the
supported ``./dev adq-ledger-replay-capture`` command.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from tokenizers import Tokenizer


SENTINEL = "ADQ_EXACT_STATE_CAPTURE_SENTINEL_DO_NOT_USE_AS_MODEL_EVIDENCE"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(part.get("text", ""))
            for part in value
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def validate_terminal_text(loop_text: str, expected_sha256: str) -> str:
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected terminal-text SHA-256 must be 64 lowercase hexadecimal characters")
    actual_sha256 = sha256_bytes(loop_text.encode("utf-8"))
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "retained terminal assistant text does not match the caller-frozen SHA-256: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return actual_sha256


def load_session(db_path: Path, session_id: str) -> dict[str, Any]:
    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        session = connection.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            raise RuntimeError(f"session not found in retained database: {session_id}")
        messages = connection.execute(
            "SELECT * FROM message WHERE session_id = ? ORDER BY time_created, id", (session_id,)
        ).fetchall()
        parts = connection.execute(
            "SELECT * FROM part WHERE session_id = ? ORDER BY time_created, id", (session_id,)
        ).fetchall()
    finally:
        connection.close()

    by_message: dict[str, list[dict[str, Any]]] = {}
    for row in parts:
        by_message.setdefault(str(row["message_id"]), []).append(json.loads(row["data"]))
    return {
        "session": {key: session[key] for key in session.keys()},
        "messages": [
            {
                "id": row["id"],
                "time_created": row["time_created"],
                "time_updated": row["time_updated"],
                "data": json.loads(row["data"]),
                "parts": by_message.get(str(row["id"]), []),
            }
            for row in messages
        ],
    }


def post_json(port: int, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"{path} returned HTTP {response.status}: {body[:4000]}")
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{path} returned non-object JSON")
    return decoded


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev adq-ledger-replay-capture; direct execution is unsupported.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--task", default="ledger")
    parser.add_argument("--source-campaign-manifest", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--captured-request", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--template-source", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--native-extension", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--expected-prompt-tokens", required=True, type=int)
    parser.add_argument("--expected-generated-tokens", required=True, type=int)
    parser.add_argument("--expected-terminal-text-sha256", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not args.task or not args.task[0].islower() or any(
        not (character.islower() or character.isdigit() or character in "_-")
        for character in args.task
    ):
        parser.error("--task must be a lowercase identifier")

    db_root = args.source_run / f".runtime-{args.task}" / "home" / ".local" / "share" / "opencode"
    db_path = db_root / "opencode.db"
    transcript_path = args.source_run / f"{args.task}.transcript.jsonl"
    config_path = args.source_run / f".runtime-{args.task}" / "config" / "opencode.json"
    results_path = args.source_run / "results.jsonl"
    patch_path = args.source_run / f"{args.task}.patch"
    hidden_tests_path = args.source_run / f"{args.task}.hidden-tests.log"
    public_tests_path = args.source_run / f"{args.task}.public-tests.log"
    for path in (
        db_path,
        transcript_path,
        config_path,
        results_path,
        patch_path,
        hidden_tests_path,
        public_tests_path,
        args.source_campaign_manifest,
        args.captured_request,
        args.template_source,
        args.runtime_config,
        args.native_extension,
    ):
        if not path.is_file():
            raise SystemExit(f"required replay source is missing: {path}")
    if not args.model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {args.model_dir}")

    session = load_session(db_path, args.session_id)
    assistant_messages = [
        row for row in session["messages"] if row["data"].get("role") == "assistant"
    ]
    if not assistant_messages or assistant_messages[-1]["data"].get("finish") != "stop":
        raise SystemExit("retained session does not end in an assistant normal stop")
    loop_text = "".join(
        str(part.get("text", ""))
        for part in assistant_messages[-1]["parts"]
        if part.get("type") == "text"
    )
    try:
        terminal_text_sha256 = validate_terminal_text(
            loop_text, args.expected_terminal_text_sha256
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    source_campaign = json.loads(args.source_campaign_manifest.read_text(encoding="utf-8"))
    if source_campaign.get("status") != "FAILED":
        raise SystemExit("source campaign manifest is not immutably marked FAILED")
    campaign_result = source_campaign.get("productive_complex_work", {}).get("result", {})
    if campaign_result.get("task") != args.task:
        raise SystemExit(
            f"source campaign failed-task identity does not match --task {args.task!r}"
        )
    if campaign_result.get("transcript_sha256") != sha256_file(transcript_path):
        raise SystemExit("source campaign transcript identity does not match the retained run")
    if campaign_result.get("result_sha256") != sha256_file(results_path):
        raise SystemExit("source campaign result identity does not match the retained run")
    if campaign_result.get("patch_sha256") != sha256_file(patch_path):
        raise SystemExit("source campaign patch identity does not match the retained run")
    if int(campaign_result.get("generated_tokens_on_terminal_turn", -1)) != args.expected_generated_tokens:
        raise SystemExit("source campaign generated-token count does not match the requested freeze")
    runtime_config_sha256 = sha256_file(args.runtime_config)
    if source_campaign.get("krasis", {}).get("config_sha256") != runtime_config_sha256:
        raise SystemExit("source campaign runtime-config identity does not match the supplied config")

    captured_bytes = args.captured_request.read_bytes()
    captured = json.loads(captured_bytes)
    messages = captured.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise SystemExit("captured continuation request has no usable messages array")
    if messages[-1].get("role") != "user" or SENTINEL not in content_text(messages[-1].get("content")):
        raise SystemExit("captured continuation request does not end in the exact capture sentinel")
    if messages[-2].get("role") != "assistant" or content_text(messages[-2].get("content")) != loop_text:
        raise SystemExit("captured continuation does not preserve the terminal assistant loop byte-for-byte")

    preloop = copy.deepcopy(captured)
    preloop["messages"] = preloop["messages"][:-2]
    preloop_path = args.output_dir / "preloop-request.json"
    write_json(preloop_path, preloop)
    write_json(args.output_dir / "retained-session.json", session)
    (args.output_dir / "accepted-loop.txt").write_text(loop_text, encoding="utf-8")

    diagnostic = copy.deepcopy(preloop)
    diagnostic["stream"] = False
    diagnostic["max_tokens"] = 1
    diagnostic.pop("max_completion_tokens", None)
    diagnostic["temperature"] = 0
    diagnostic["debug_first_token_boundary"] = True
    diagnostic["top_logprobs"] = 100
    chat = post_json(args.port, "/v1/chat/completions", diagnostic, args.timeout)
    debug = chat.get("krasis_debug")
    if not isinstance(debug, dict):
        raise SystemExit("Krasis diagnostic response omitted krasis_debug")
    input_ids = debug.get("input_token_ids")
    if not isinstance(input_ids, list) or not all(isinstance(token, int) for token in input_ids):
        raise SystemExit("Krasis diagnostic response omitted exact input token IDs")
    if len(input_ids) != args.expected_prompt_tokens:
        raise SystemExit(
            f"exact prompt mismatch: expected {args.expected_prompt_tokens}, rendered {len(input_ids)}"
        )

    tokenizer_path = args.model_dir / "tokenizer.json"
    generation_path = args.model_dir / "generation_config.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    visible_output_ids = list(tokenizer.encode(loop_text, add_special_tokens=False).ids)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    eos_value = generation.get("eos_token_id")
    eos_ids = [int(eos_value)] if isinstance(eos_value, int) else [int(value) for value in eos_value or []]
    if len(eos_ids) != 1:
        raise SystemExit(f"exactly one checkpoint EOS token is required, found {eos_ids}")
    terminal_stop_id = eos_ids[0]
    if len(visible_output_ids) not in {
        args.expected_generated_tokens,
        args.expected_generated_tokens - 1,
    }:
        raise SystemExit(
            "retokenized terminal text is inconsistent with the server-recorded generated-token count: "
            f"visible={len(visible_output_ids)} recorded={args.expected_generated_tokens} eos={eos_ids}"
        )
    selected_first = debug.get("selected_token_id")
    if not visible_output_ids or selected_first != visible_output_ids[0]:
        raise SystemExit(
            f"exact first-token mismatch: accepted={visible_output_ids[:1]} replay={selected_first}"
        )

    input_artifact = {
        "format": "krasis_adq_exact_input_tokens",
        "format_version": 1,
        "session_id": args.session_id,
        "input_token_count": len(input_ids),
        "input_token_ids": input_ids,
        "input_token_ids_sha256": canonical_sha256(input_ids),
        "rendered_prompt": debug.get("rendered_prompt"),
        "rendered_prompt_sha256": sha256_bytes(str(debug.get("rendered_prompt", "")).encode("utf-8")),
        "krasis_input_token_hash_fnv1a64": debug.get("input_token_hash_fnv1a64"),
    }
    output_artifact = {
        "format": "krasis_adq_accepted_output_tokens",
        "format_version": 1,
        "session_id": args.session_id,
        "visible_text": loop_text,
        "visible_text_sha256": terminal_text_sha256,
        "retokenized_visible_token_count": len(visible_output_ids),
        "recorded_generated_token_count": args.expected_generated_tokens,
        "terminal_stop_token_id": terminal_stop_id,
        "retokenized_visible_candidate_token_ids": visible_output_ids,
        "retokenized_visible_candidate_sha256": canonical_sha256(visible_output_ids),
        "token_identity_status": "not_directly_retained_reconstruct_by_exact_deterministic_replay",
    }
    write_json(args.output_dir / "preloop-input-token-ids.json", input_artifact)
    write_json(args.output_dir / "accepted-output-token-ids.json", output_artifact)
    write_json(args.output_dir / "first-token-replay-response.json", chat)

    identity_names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    model_files = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name in identity_names
        for path in [args.model_dir / name]
        if path.is_file()
    }
    tools = preloop.get("tools", [])
    system_messages = [message for message in preloop["messages"] if message.get("role") == "system"]
    manifest = {
        "format": "krasis_adq_task_exact_state",
        "format_version": 2,
        "status": "FROZEN_REPLAY_SOURCE",
        "source_task": args.task,
        "source_campaign_status": "FAILED",
        "source_run": str(args.source_run.resolve()),
        "source_session_id": args.session_id,
        "source_hashes": {
            "opencode_db": sha256_file(db_path),
            "opencode_db_wal": sha256_file(db_root / "opencode.db-wal"),
            "transcript": sha256_file(transcript_path),
            "opencode_config": sha256_file(config_path),
            "results": sha256_file(results_path),
            "patch": sha256_file(patch_path),
            "hidden_tests": sha256_file(hidden_tests_path),
            "public_tests": sha256_file(public_tests_path),
            "source_campaign_manifest": sha256_file(args.source_campaign_manifest),
            "captured_continuation_request_raw": sha256_bytes(captured_bytes),
        },
        "source_campaign": {
            "campaign_id": source_campaign.get("campaign_id"),
            "status": source_campaign.get("status"),
            "failed_at": source_campaign.get("failed_at"),
            "declared_native_extension_sha256": source_campaign.get("krasis", {}).get(
                "native_extension_sha256"
            ),
            "runtime": source_campaign.get("runtime"),
        },
        "source_nonidentity_files": {
            "opencode_db_shm": {
                "path": str((db_root / "opencode.db-shm").resolve()),
                "sha256_at_extraction": sha256_file(db_root / "opencode.db-shm"),
                "identity_role": "excluded_ephemeral_sqlite_shared_memory",
            }
        },
        "opencode": {
            "version": session["session"]["version"],
            "model": session["session"]["model"],
            "agent": session["session"]["agent"],
            "system_messages_sha256": canonical_sha256(system_messages),
            "tool_schemas_sha256": canonical_sha256(tools),
            "tool_count": len(tools),
            "preloop_messages_sha256": canonical_sha256(preloop["messages"]),
            "preloop_request_sha256": canonical_sha256(preloop),
        },
        "checkpoint": {"model_dir": str(args.model_dir.resolve()), "files": model_files},
        "capture_runtime": {
            "config_path": str(args.runtime_config.resolve()),
            "config_sha256": runtime_config_sha256,
            "native_extension_path": str(args.native_extension.resolve()),
            "native_extension_sha256": sha256_file(args.native_extension),
            "identity_note": (
                "The capture runtime renders/verifies exact input tokens. The immutable source-"
                "campaign manifest above is authoritative for the binary that produced the failed turn."
            ),
        },
        "chat_template": {
            "selection": "bundled_deepseek_v4_fallback",
            "path": str(args.template_source.resolve()),
            "sha256": sha256_file(args.template_source),
        },
        "input": {
            "tokens": len(input_ids),
            "token_ids_sha256": input_artifact["input_token_ids_sha256"],
            "rendered_prompt_sha256": input_artifact["rendered_prompt_sha256"],
        },
        "accepted_output": {
            "server_recorded_generated_tokens": args.expected_generated_tokens,
            "retokenized_visible_tokens": len(visible_output_ids),
            "directly_retained_exact_token_ids": False,
            "retokenized_visible_candidate_sha256": output_artifact["retokenized_visible_candidate_sha256"],
            "visible_text_sha256": terminal_text_sha256,
            "terminal_stop_token_id": terminal_stop_id,
            "finish_reason": "stop",
        },
        "first_token_replay": {
            "accepted_token_id": visible_output_ids[0],
            "replayed_token_id": selected_first,
            "exact": True,
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    artifact_paths = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "artifact-sha256.json"
    )
    artifact_index = {
        "format": "krasis_adq_exact_state_artifact_hashes",
        "format_version": 1,
        "self_excluded": "artifact-sha256.json",
        "artifacts": [
            {
                "path": str(path.relative_to(args.output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(args.output_dir / "artifact-sha256.json", artifact_index)
    print(
        f"Frozen exact {args.task} state: input={len(input_ids)} recorded_output={args.expected_generated_tokens} "
        f"first_token={selected_first} tools={len(tools)}"
    )


if __name__ == "__main__":
    main()
