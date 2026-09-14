#!/usr/bin/env python3
"""Build exact-token, real-content ADQ witness inputs.

This is a test-artifact builder, not a model hot-path component. Run it only
through ``./dev adq-witness-inputs`` so repository and environment identity are
captured consistently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_TARGETS = (3_000, 16_000, 32_000, 64_000, 131_072, 262_144, 500_000)
ALLOWED_SOURCE_SUFFIXES = {".md", ".py", ".rs", ".toml", ".txt"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_targets(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("target lengths must be positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("target lengths must be unique and strictly increasing")
    return values


def _source_paths(repo_root: Path) -> list[Path]:
    books = sorted((repo_root / "benchmarks" / "prompts").glob("prompt*.txt"))
    if len(books) != 6:
        raise RuntimeError(f"expected six canonical prompt files, found {len(books)}")

    code_and_docs: list[Path] = []
    for root in (repo_root / "src", repo_root / "python" / "krasis"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in ALLOWED_SOURCE_SUFFIXES:
                continue
            if "prompts" in path.parts or "__pycache__" in path.parts:
                continue
            code_and_docs.append(path)
    for name in ("README.md", "ADVANCED.md", "STATS-QUALITY.md"):
        path = repo_root / name
        if path.is_file():
            code_and_docs.append(path)
    return books + sorted(set(code_and_docs))


def _token_id(tokenizer: Tokenizer, value: str) -> int:
    token_id = tokenizer.token_to_id(value)
    if token_id is None:
        raise RuntimeError(f"checkpoint tokenizer is missing required token {value!r}")
    return token_id


def build_artifact(
    *, model_dir: Path, repo_root: Path, targets: tuple[int, ...], profile: str
) -> dict[str, object]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_limit = config.get("max_position_embeddings")
    if not isinstance(model_limit, int) or model_limit <= 0:
        raise RuntimeError("checkpoint has no valid max_position_embeddings")
    if targets[-1] >= model_limit:
        raise RuntimeError(
            f"largest target {targets[-1]:,} must leave room below model limit {model_limit:,}"
        )

    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prefix_tokens = [
        _token_id(tokenizer, "<｜begin▁of▁sentence｜>"),
        _token_id(tokenizer, "<｜User｜>"),
    ]
    query = (
        "\n\n===== END ADQ CORPUS =====\n"
        "This is a frozen numerical-fidelity probe. Continue by stating the "
        "single most important engineering rule evident from the preceding "
        "real books, source code, and documentation."
    )
    suffix_tokens = tokenizer.encode(query, add_special_tokens=False).ids
    suffix_tokens.append(_token_id(tokenizer, "<｜Assistant｜>"))
    suffix_tokens.extend(tokenizer.encode("</think>", add_special_tokens=False).ids)

    corpus_tokens: list[int] = []
    sources: list[dict[str, object]] = []
    required_corpus_tokens = targets[-1] - len(prefix_tokens) - len(suffix_tokens)
    for path in _source_paths(repo_root):
        raw = path.read_bytes()
        relative = path.relative_to(repo_root).as_posix()
        digest = _sha256(raw)
        header = f"\n\n===== SOURCE {relative} SHA256 {digest} =====\n"
        text = header + raw.decode("utf-8", errors="strict")
        encoded = tokenizer.encode(text, add_special_tokens=False).ids
        corpus_tokens.extend(encoded)
        sources.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": len(raw),
                "token_count_with_header": len(encoded),
            }
        )
        if len(corpus_tokens) >= required_corpus_tokens:
            break

    if len(corpus_tokens) < required_corpus_tokens:
        raise RuntimeError(
            f"real-content corpus has {len(corpus_tokens):,} tokens but "
            f"{required_corpus_tokens:,} are required"
        )

    conversations: list[dict[str, object]] = []
    for target in targets:
        content_count = target - len(prefix_tokens) - len(suffix_tokens)
        if content_count <= 0:
            raise RuntimeError(f"target {target:,} is smaller than the chat framing")
        input_ids = prefix_tokens + corpus_tokens[:content_count] + suffix_tokens
        if len(input_ids) != target:
            raise AssertionError(f"built {len(input_ids)} tokens for target {target}")
        canonical = ",".join(str(token) for token in input_ids).encode("ascii")
        conversations.append(
            {
                "source_id": f"adq_prefix_{target}",
                "turns": [
                    {
                        "prompt": f"Frozen ADQ real-content prefix at exactly {target:,} tokens",
                        "input_token_ids": input_ids,
                        "input_sha256": _sha256(canonical),
                        "chat_template_application": {
                            "method": "krasis_adq_raw_token_builder",
                            "bos_token": "<｜begin▁of▁sentence｜>",
                            "user_token": "<｜User｜>",
                            "assistant_token": "<｜Assistant｜>",
                            "thinking": False,
                            "input_token_count": target,
                        },
                    }
                ],
            }
        )

    return {
        "format": "krasis_llama_witness_input_source",
        "format_version": 1,
        "runtime": "krasis-adq-input-builder",
        "model": model_dir.name,
        "model_path": str(model_dir.resolve()),
        "profile_id": profile,
        "source_role": "frozen_real_content_input_token_source",
        "model_max_position_embeddings": model_limit,
        "tokenizer": {
            "path": str(tokenizer_path.resolve()),
            "sha256": _sha256(tokenizer_path.read_bytes()),
        },
        "source_files": sources,
        "targets": list(targets),
        "framing": {
            "prefix_tokens": len(prefix_tokens),
            "suffix_tokens": len(suffix_tokens),
            "corpus_tokens_available": len(corpus_tokens),
        },
        "conversations": conversations,
    }


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev adq-witness-inputs; direct execution is unsupported.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--targets", default=",".join(str(v) for v in DEFAULT_TARGETS))
    parser.add_argument("--profile", default="dsv4_vision_adq500k_staged_witness")
    args = parser.parse_args()

    artifact = build_artifact(
        model_dir=args.model_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        targets=_parse_targets(args.targets),
        profile=args.profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print("Exact input lengths: " + ", ".join(f"{v:,}" for v in artifact["targets"]))
    print(f"Source files: {len(artifact['source_files'])}")


if __name__ == "__main__":
    main()
