#!/usr/bin/env python3
"""Build an exact-token real-content ADQ issue-history attachment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from adq_sources import source_paths


DECISIONS = (
    """\n\n===== ACTIVE POLICY DECISION A =====
Resolution: an explicitly supplied `resources` array wins over role defaults, including an empty array. Only an absent (`undefined`) field uses defaults. The supported canonical roles are `viewer` and `maintainer`; unknown roles must throw an error that names the role.
Revoked proposal nearby: treating an empty array as absent was rejected.
===== END DECISION A =====\n\n""",
    """\n\n===== ACTIVE POLICY DECISION B =====
Resolution: the legacy incoming role `operator` remains accepted but is normalized to canonical role `maintainer` before default lookup and in returned output.
Revoked proposal nearby: returning `operator` unchanged was rejected.
===== END DECISION B =====\n\n""",
    """\n\n===== ACTIVE POLICY DECISION C =====
Resolution: every resource must be a string; trim it, lowercase it, remove duplicates, and return resources in lexicographic order. The wildcard `*` is valid alone but must throw an error mentioning wildcard if combined with any other resource.
Revoked proposal nearby: preserving input case and order was rejected.
===== END DECISION C =====\n\n""",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render(tokenizer: Tokenizer, corpus_ids: list[int], count: int) -> str:
    cuts = ((count * 5) // 100, count // 2, (count * 95) // 100)
    parts: list[str] = []
    start = 0
    for cut, decision in zip(cuts, DECISIONS):
        parts.append(tokenizer.decode(corpus_ids[start:cut], skip_special_tokens=False))
        parts.append(decision)
        start = cut
    parts.append(tokenizer.decode(corpus_ids[start:count], skip_special_tokens=False))
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    args = parser.parse_args()
    if args.target_tokens <= 0:
        raise SystemExit("--target-tokens must be positive")

    tokenizer_path = args.model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    repo_root = Path(__file__).resolve().parents[1]
    sources = source_paths(repo_root)
    corpus = "\n\n".join(path.read_text(encoding="utf-8") for path in sources)
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False).ids
    if len(corpus_ids) < args.target_tokens:
        raise SystemExit(
            f"real-content corpus has {len(corpus_ids)} tokens, below target {args.target_tokens}"
        )

    decision_tokens = sum(
        len(tokenizer.encode(decision, add_special_tokens=False).ids)
        for decision in DECISIONS
    )
    count = args.target_tokens - decision_tokens
    if count <= 0:
        raise SystemExit("target is smaller than the required policy records")
    found: tuple[str, list[int]] | None = None
    tried: set[int] = set()
    for _ in range(6):
        if count in tried or count < 0 or count > len(corpus_ids):
            break
        tried.add(count)
        text = _render(tokenizer, corpus_ids, count)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) == args.target_tokens:
            found = (text, ids)
            break
        count += args.target_tokens - len(ids)
    if found is None:
        centre = count
        for nearby in range(max(0, centre - 8), min(len(corpus_ids), centre + 8) + 1):
            if nearby in tried:
                continue
            text = _render(tokenizer, corpus_ids, nearby)
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) == args.target_tokens:
                found = (text, ids)
                break
    if found is None:
        raise SystemExit(f"could not construct exact {args.target_tokens}-token history")

    text, ids = found
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    metadata = {
        "format": "krasis_adq_opencode_history",
        "format_version": 1,
        "target_tokens": args.target_tokens,
        "observed_tokens": len(ids),
        "token_ids_sha256": hashlib.sha256(
            ",".join(str(value) for value in ids).encode("ascii")
        ).hexdigest(),
        "output_sha256": _sha256(args.output),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in sources
        ],
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
