"""Shared immutable real-content source inventory for ADQ artifact builders."""

from __future__ import annotations

from pathlib import Path


ALLOWED_SOURCE_SUFFIXES = {".md", ".py", ".rs", ".toml", ".txt"}


def source_paths(repo_root: Path) -> list[Path]:
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
