"""Immutable source-checkpoint identity shared by caches and route artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_IDENTITY_FORMAT = "krasis_checkpoint_identity"
CHECKPOINT_IDENTITY_FORMAT_VERSION = 1
CHECKPOINT_IDENTITY_ENV = "KRASIS_CHECKPOINT_IDENTITIES"
_IDENTITY_DOMAIN = b"krasis-checkpoint-identity-v1"


@dataclass(frozen=True)
class _WeightFile:
    name: str
    size: int
    object_id: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _weight_names(model_dir: Path) -> tuple[list[str], Path | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read checkpoint index {index_path}: {exc}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Checkpoint index has no usable weight_map: {index_path}")
        names = sorted(set(weight_map.values()))
        if not all(
            isinstance(name, str)
            and name.endswith(".safetensors")
            and os.path.basename(name) == name
            for name in names
        ):
            raise RuntimeError(f"Checkpoint index contains unsafe shard paths: {index_path}")
        return names, index_path

    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single.name], None
    raise RuntimeError(f"No complete safetensors checkpoint inventory found in {model_dir}")


def _metadata_record(model_dir: Path, name: str) -> tuple[str, str, float] | None:
    path = model_dir / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    try:
        with path.open(encoding="utf-8") as handle:
            revision = handle.readline().strip()
            object_id = handle.readline().strip().strip('"')
            completed_at = float(handle.readline().strip())
    except (OSError, ValueError):
        return None
    if len(revision) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in revision):
        return None
    if len(object_id) not in (40, 64) or any(
        ch not in "0123456789abcdefABCDEF" for ch in object_id
    ):
        return None
    return revision.lower(), object_id.lower(), completed_at


def _feed_field(digest: Any, label: str, value: str | int) -> None:
    label_bytes = label.encode("utf-8")
    value_bytes = str(value).encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(value_bytes).to_bytes(8, "big"))
    digest.update(value_bytes)


def _identity_digest(
    weights: Iterable[_WeightFile],
) -> str:
    digest = hashlib.sha256()
    digest.update(_IDENTITY_DOMAIN)
    for weight in weights:
        _feed_field(digest, "weight_name", weight.name)
        _feed_field(digest, "weight_size", weight.size)
        _feed_field(digest, "weight_object_id", weight.object_id)
    return digest.hexdigest()


def _cache_identity_digest(
    checkpoint_sha256: str,
    controls: Iterable[tuple[str, int, str]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"krasis-cache-identity-v1")
    _feed_field(digest, "checkpoint_sha256", checkpoint_sha256)
    for name, size, sha256 in controls:
        _feed_field(digest, "control_name", name)
        _feed_field(digest, "control_size", size)
        _feed_field(digest, "control_sha256", sha256)
    return digest.hexdigest()


def _control_records(model_dir: Path, index_path: Path | None) -> list[tuple[str, int, str]]:
    paths = [model_dir / "config.json"]
    if index_path is not None:
        paths.append(index_path)
    records: list[tuple[str, int, str]] = []
    for path in sorted(paths, key=lambda item: item.name):
        if not path.is_file():
            raise RuntimeError(f"Checkpoint control file is missing: {path}")
        records.append((path.name, path.stat().st_size, _sha256_file(path)))
    return records


def _content_identity_cache_path(model_dir: Path) -> Path:
    path_key = hashlib.sha256(str(model_dir).encode("utf-8")).hexdigest()
    return (
        Path(os.path.expanduser("~"))
        / ".krasis"
        / "checkpoint-identities"
        / f"{path_key}.json"
    )


def _content_weight_records(model_dir: Path, names: list[str]) -> list[_WeightFile]:
    cache_path = _content_identity_cache_path(model_dir)
    current_stats = []
    for name in names:
        path = model_dir / name
        stat = path.stat()
        current_stats.append(
            {
                "name": name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        cached = None
    if (
        isinstance(cached, dict)
        and cached.get("format") == "krasis_content_checkpoint_hash_cache"
        and cached.get("format_version") == 1
        and cached.get("model_path") == str(model_dir)
        and cached.get("file_stats") == current_stats
        and isinstance(cached.get("sha256s"), dict)
        and all(
            isinstance(cached["sha256s"].get(name), str)
            and len(cached["sha256s"][name]) == 64
            and all(ch in "0123456789abcdef" for ch in cached["sha256s"][name])
            for name in names
        )
    ):
        return [
            _WeightFile(name, stat["size"], cached["sha256s"][name])
            for name, stat in zip(names, current_stats)
        ]

    records = [
        _WeightFile(name, stat["size"], _sha256_file(model_dir / name))
        for name, stat in zip(names, current_stats)
    ]
    cache_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "format": "krasis_content_checkpoint_hash_cache",
        "format_version": 1,
        "model_path": str(model_dir),
        "file_stats": current_stats,
        "sha256s": {record.name: record.object_id for record in records},
    }
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, sort_keys=True, separators=(",", ":"))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, cache_path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return records


def _register_for_rust(model_dir: Path, identity_sha256: str) -> None:
    try:
        existing = json.loads(os.environ.get(CHECKPOINT_IDENTITY_ENV, "{}"))
    except json.JSONDecodeError:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing[str(model_dir)] = identity_sha256
    os.environ[CHECKPOINT_IDENTITY_ENV] = json.dumps(
        existing, sort_keys=True, separators=(",", ":")
    )


@lru_cache(maxsize=32)
def checkpoint_identity(model_path: str) -> dict[str, Any]:
    """Return an immutable identity for the exact local source checkpoint.

    Complete Hugging Face local-dir metadata supplies the immutable revision and
    per-shard object IDs already verified by the downloader. Checkpoints without
    that provenance are hashed in full; no path, timestamp, or byte sampling is
    accepted as weight identity.
    """
    model_dir = Path(model_path).expanduser().resolve()
    names, index_path = _weight_names(model_dir)
    controls = _control_records(model_dir, index_path)

    hf_records: list[_WeightFile] = []
    revision: str | None = None
    complete_hf_metadata = True
    for name in names:
        path = model_dir / name
        if not path.is_file():
            raise RuntimeError(f"Checkpoint shard is missing: {path}")
        metadata = _metadata_record(model_dir, name)
        if metadata is None or len(metadata[1]) != 64:
            complete_hf_metadata = False
            break
        file_revision, object_id, completed_at = metadata
        stat = path.stat()
        if min(completed_at - stat.st_mtime, completed_at - stat.st_ctime) < -0.001:
            complete_hf_metadata = False
            break
        if revision is None:
            revision = file_revision
        elif revision != file_revision:
            complete_hf_metadata = False
            break
        hf_records.append(_WeightFile(name, stat.st_size, object_id))

    if complete_hf_metadata and revision is not None:
        source_kind = "huggingface_snapshot"
        weights = hf_records
    else:
        source_kind = "content_sha256"
        revision = None
        weights = _content_weight_records(model_dir, names)

    identity_sha256 = _identity_digest(weights)
    result = {
        "format": CHECKPOINT_IDENTITY_FORMAT,
        "format_version": CHECKPOINT_IDENTITY_FORMAT_VERSION,
        "sha256": identity_sha256,
        "source_kind": source_kind,
        "revision": revision,
        "weight_file_count": len(weights),
        "weight_bytes": sum(weight.size for weight in weights),
    }
    _register_for_rust(model_dir, identity_sha256)
    return result


def route_checkpoint_identity(model_path: str) -> dict[str, Any]:
    """Return only the source-independent checkpoint fields used for matching."""
    identity = checkpoint_identity(model_path)
    return {
        "format": identity["format"],
        "format_version": identity["format_version"],
        "sha256": identity["sha256"],
        "weight_file_count": identity["weight_file_count"],
        "weight_bytes": identity["weight_bytes"],
    }


def route_checkpoint_identity_from_records(
    weights: Iterable[tuple[str, int, str]],
) -> dict[str, Any]:
    """Build the portable route identity from verified remote/local records."""
    weight_records = sorted(
        [_WeightFile(str(name), int(size), str(sha256).lower()) for name, size, sha256 in weights],
        key=lambda record: record.name,
    )
    for record in weight_records:
        if len(record.object_id) != 64 or any(
            ch not in "0123456789abcdef" for ch in record.object_id
        ):
            raise ValueError(f"Weight {record.name!r} does not have a SHA-256 identity")
    return {
        "format": CHECKPOINT_IDENTITY_FORMAT,
        "format_version": CHECKPOINT_IDENTITY_FORMAT_VERSION,
        "sha256": _identity_digest(weight_records),
        "weight_file_count": len(weight_records),
        "weight_bytes": sum(record.size for record in weight_records),
    }


def cache_namespace(model_path: str) -> str:
    """Return the readable, checkpoint-isolated cache directory name."""
    model_dir = Path(model_path).expanduser().resolve()
    safe_name = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in "-_.") else "_"
        for ch in model_dir.name
    ) or "model"
    names, index_path = _weight_names(model_dir)
    del names
    identity = checkpoint_identity(str(model_dir))
    cache_sha256 = _cache_identity_digest(
        identity["sha256"], _control_records(model_dir, index_path)
    )
    return f"{safe_name}--{cache_sha256}"
