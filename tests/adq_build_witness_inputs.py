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

from krasis.tokenizer import Tokenizer


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


def _model_context_limit(config: dict[str, object]) -> int:
    value = config.get("max_position_embeddings")
    if isinstance(value, int) and value > 0:
        return value
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        value = text_config.get("max_position_embeddings")
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("checkpoint has no valid max_position_embeddings")


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


def _encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    return [
        int(token)
        for token in tokenizer.tokenizer.encode(text, add_special_tokens=False)
    ]


def _decode_tokens(tokenizer: Tokenizer, values: list[int]) -> str:
    return tokenizer.tokenizer.decode(
        values,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _fit_exact_chat_input(
    tokenizer: Tokenizer,
    corpus_tokens: list[int],
    target: int,
    query: str,
) -> tuple[list[int], str]:
    """Render one checkpoint-native chat prompt at exactly ``target`` tokens."""

    def candidate(content_count: int, alignment: str = "") -> list[int]:
        content = (
            _decode_tokens(tokenizer, corpus_tokens[:content_count])
            + alignment
            + query
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            enable_thinking=False,
        )

    empty = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(empty) >= target:
        raise RuntimeError(
            f"target {target:,} is not larger than its {len(empty):,}-token chat framing"
        )

    content_count = min(len(corpus_tokens), max(1, target - len(empty)))
    for _ in range(12):
        values = candidate(content_count)
        delta = target - len(values)
        if delta == 0:
            return values, ""
        next_count = min(len(corpus_tokens), max(1, content_count + delta))
        if next_count == content_count:
            break
        content_count = next_count

    closest_below: tuple[int, int] | None = None
    for nearby_count in range(
        max(1, content_count - 256),
        min(len(corpus_tokens), content_count + 256) + 1,
    ):
        values = candidate(nearby_count)
        if len(values) == target:
            return values, ""
        if len(values) < target:
            gap = target - len(values)
            if closest_below is None or gap < closest_below[0]:
                closest_below = (gap, nearby_count)

    if closest_below is not None:
        gap, nearby_count = closest_below
        lower = max(1, gap - 12)
        upper = max(32, gap + 12)
        for unit in (" x", " .", "\n", " ", "\t", "\u200b"):
            for repeats in range(lower, upper + 1):
                alignment = unit * repeats
                values = candidate(nearby_count, alignment)
                if len(values) == target:
                    return values, alignment
    raise RuntimeError(
        f"could not construct an exact {target:,}-token checkpoint-native chat prompt"
    )


def build_artifact(
    *, model_dir: Path, repo_root: Path, targets: tuple[int, ...], profile: str
) -> dict[str, object]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_limit = _model_context_limit(config)
    if targets[-1] >= model_limit:
        raise RuntimeError(
            f"largest target {targets[-1]:,} must leave room below model limit {model_limit:,}"
        )

    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = Tokenizer(str(model_dir))
    chat_template = tokenizer.tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("checkpoint-native chat template did not resolve")
    query = (
        "\n\n===== END ADQ CORPUS =====\n"
        "This is a frozen numerical-fidelity probe. Continue by stating the "
        "single most important engineering rule evident from the preceding "
        "real books, source code, and documentation."
    )
    corpus_tokens: list[int] = []
    sources: list[dict[str, object]] = []
    required_corpus_tokens = targets[-1]
    for path in _source_paths(repo_root):
        raw = path.read_bytes()
        relative = path.relative_to(repo_root).as_posix()
        digest = _sha256(raw)
        header = f"\n\n===== SOURCE {relative} SHA256 {digest} =====\n"
        text = header + raw.decode("utf-8", errors="strict")
        encoded = _encode_text(tokenizer, text)
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
        input_ids, alignment = _fit_exact_chat_input(
            tokenizer, corpus_tokens, target, query
        )
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
                            "method": "checkpoint_native_krasis_tokenizer",
                            "chat_template_sha256": _sha256(chat_template.encode("utf-8")),
                            "add_generation_prompt": True,
                            "thinking": False,
                            "alignment_text": alignment,
                            "alignment_text_sha256": _sha256(alignment.encode("utf-8")),
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
        "corpus_token_ids": corpus_tokens,
        "corpus_token_sha256": _sha256(
            ",".join(str(token) for token in corpus_tokens).encode("ascii")
        ),
        "targets": list(targets),
        "framing": {
            "method": "checkpoint_native_krasis_tokenizer",
            "chat_template_sha256": _sha256(chat_template.encode("utf-8")),
            "add_generation_prompt": True,
            "thinking": False,
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
