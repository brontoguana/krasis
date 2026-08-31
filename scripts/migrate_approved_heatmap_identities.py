#!/usr/bin/env python3
"""Audit and migrate approved heatmaps to immutable checkpoint identities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Run through ./dev approved-heatmap-migrate")

from huggingface_hub import HfApi, hf_hub_download

from krasis.checkpoint_identity import route_checkpoint_identity_from_records


MANIFEST_FORMAT = "krasis_approved_hcs_route_heatmap_manifest"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "benchmarks" / "approved_heatmaps"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_jsonable(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def sibling_fields(sibling: Any) -> tuple[str, int, str | None]:
    name = str(getattr(sibling, "rfilename", ""))
    size = int(getattr(sibling, "size", 0) or 0)
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        object_id = lfs.get("sha256")
    else:
        object_id = getattr(lfs, "sha256", None)
    return name, size, str(object_id).lower() if object_id else None


def download_small(repo_id: str, revision: str, name: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename=name,
        )
    )


def checkpoint_identity_for_source(
    repo_id: str,
    revision: str,
    expected_fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    resolved_revision = str(getattr(info, "sha", "") or "").lower()
    if resolved_revision != revision.lower():
        raise RuntimeError(
            f"{repo_id}: requested revision {revision}, resolved {resolved_revision or '<none>'}"
        )
    siblings = {}
    for sibling in getattr(info, "siblings", []) or []:
        name, size, object_id = sibling_fields(sibling)
        siblings[name] = {"size": size, "sha256": object_id}

    fingerprint_audit = {}
    for name, expected_sha in sorted(expected_fingerprints.items()):
        if name not in siblings:
            fingerprint_audit[name] = "not_present_in_source_revision"
            continue
        actual_sha = sha256_file(download_small(repo_id, revision, name))
        fingerprint_audit[name] = "match" if actual_sha == expected_sha else "different"

    config_path = download_small(repo_id, revision, "config.json")
    index_path = download_small(repo_id, revision, "model.safetensors.index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"{repo_id}@{revision}: no usable weight_map")
    weight_names = sorted(set(weight_map.values()))
    weights = []
    for name in weight_names:
        record = siblings.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"{repo_id}@{revision}: shard absent from Hub metadata: {name}")
        object_id = record.get("sha256")
        size = int(record.get("size") or 0)
        if not isinstance(object_id, str) or len(object_id) != 64 or size <= 0:
            raise RuntimeError(f"{repo_id}@{revision}: shard lacks LFS SHA-256/size: {name}")
        weights.append((name, size, object_id))
    identity = route_checkpoint_identity_from_records(weights)
    provenance = {
        "repo_id": repo_id,
        "revision": revision,
        "resolved_revision": resolved_revision,
        "verified_config_fingerprints": dict(sorted(expected_fingerprints.items())),
        "source_revision_fingerprint_audit": fingerprint_audit,
        "weight_file_count": len(weights),
        "weight_bytes": sum(item[1] for item in weights),
    }
    return identity, provenance


def load_catalog(directory: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    sources_path = directory / "checkpoint_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("format_version") != 1:
        raise RuntimeError("Unsupported approved heatmap manifest")
    if sources.get("format") != "krasis_approved_heatmap_checkpoint_sources":
        raise RuntimeError("Unsupported checkpoint source ledger")
    if not isinstance(sources.get("sources"), dict):
        raise RuntimeError("Checkpoint source ledger has no sources map")
    return manifest_path, manifest, sources


def audit_sources(directory: Path) -> dict[str, dict[str, Any]]:
    _manifest_path, manifest, source_ledger = load_catalog(directory)
    sources = source_ledger["sources"]
    legacy_entries = [
        entry for entry in manifest.get("artifacts", [])
        if entry.get("identity_generation") != "checkpoint_v1"
    ]
    route_groups: dict[str, list[dict[str, Any]]] = {}
    for entry in legacy_entries:
        route_groups.setdefault(str(entry.get("route_signature_sha256")), []).append(entry)
    if set(route_groups) != set(sources):
        missing = sorted(set(route_groups) - set(sources))
        extra = sorted(set(sources) - set(route_groups))
        raise RuntimeError(f"Checkpoint source coverage mismatch: missing={missing}, extra={extra}")

    audited = {}
    for route_hash, entries in sorted(route_groups.items()):
        first_path = ROOT / str(entries[0]["path"])
        artifact = json.loads(first_path.read_text(encoding="utf-8"))
        route_signature = artifact.get("_metadata", {}).get("route_signature")
        if sha256_jsonable(route_signature) != route_hash:
            raise RuntimeError(f"Legacy route hash mismatch: {first_path}")
        if route_signature.get("model", {}).get("checkpoint_identity") is not None:
            raise RuntimeError(f"Legacy artifact already contains checkpoint identity: {first_path}")
        fingerprints = route_signature.get("model", {}).get("config_fingerprints")
        if not isinstance(fingerprints, dict) or not fingerprints:
            raise RuntimeError(f"Legacy artifact lacks config fingerprints: {first_path}")
        source = sources[route_hash]
        identity, provenance = checkpoint_identity_for_source(
            str(source["repo_id"]), str(source["revision"]), fingerprints
        )
        audited[route_hash] = {
            "checkpoint_identity": identity,
            "provenance": provenance,
            "legacy_artifact_count": len(entries),
            "model_name": route_signature.get("model", {}).get("model_name"),
        }
        print(
            f"AUDITED {route_hash[:12]} {audited[route_hash]['model_name']}: "
            f"{identity['sha256'][:16]} {identity['weight_file_count']} shards"
        )
    return audited


def migrated_filename(filename: str) -> str:
    path = Path(filename)
    return f"{path.stem}.checkpoint-v1{path.suffix}"


def migrate(directory: Path, audited: dict[str, dict[str, Any]], dry_run: bool) -> None:
    manifest_path, manifest, _source_ledger = load_catalog(directory)
    legacy_entries = [
        copy.deepcopy(entry) for entry in manifest.get("artifacts", [])
        if entry.get("identity_generation") != "checkpoint_v1"
    ]
    strong_entries = []
    generated_paths = []
    for legacy in legacy_entries:
        route_hash = str(legacy["route_signature_sha256"])
        audit = audited[route_hash]
        source_path = ROOT / str(legacy["path"])
        original = json.loads(source_path.read_text(encoding="utf-8"))
        migrated = copy.deepcopy(original)
        route_signature = migrated["_metadata"]["route_signature"]
        route_signature["model"]["checkpoint_identity"] = audit["checkpoint_identity"]
        new_route_hash = sha256_jsonable(route_signature)
        filename = migrated_filename(str(legacy["filename"]))
        relative_path = f"benchmarks/approved_heatmaps/{filename}"
        target_path = ROOT / relative_path
        payload = (json.dumps(migrated, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        if {key: value for key, value in original.items() if key != "_metadata"} != {
            key: value for key, value in migrated.items() if key != "_metadata"
        }:
            raise RuntimeError(f"Routing counts changed during metadata migration: {source_path}")
        if not dry_run:
            target_path.write_bytes(payload)
        generated_paths.append(target_path)

        strong = copy.deepcopy(legacy)
        strong.update(
            {
                "artifact_id": f"{legacy['artifact_id']}__checkpoint_v1",
                "identity_generation": "checkpoint_v1",
                "legacy_artifact_id": legacy["artifact_id"],
                "path": relative_path,
                "filename": filename,
                "download_url": (
                    "https://raw.githubusercontent.com/brontoguana/krasis/main/"
                    f"{relative_path}"
                ),
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
                "route_signature_sha256": new_route_hash,
                "checkpoint_identity": audit["checkpoint_identity"],
                "checkpoint_provenance": audit["provenance"],
            }
        )
        strong_entries.append(strong)

    migrated_manifest = copy.deepcopy(manifest)
    migrated_manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    migrated_manifest["artifacts"] = legacy_entries + strong_entries
    if not dry_run:
        manifest_path.write_text(
            json.dumps(migrated_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"{'WOULD MIGRATE' if dry_run else 'MIGRATED'} {len(legacy_entries)} legacy + "
        f"{len(strong_entries)} checkpoint-v1 entries; generated={len(generated_paths)}"
    )


def verify(directory: Path) -> None:
    _manifest_path, manifest, source_ledger = load_catalog(directory)
    entries = manifest.get("artifacts", [])
    legacy = {
        entry["artifact_id"]: entry
        for entry in entries
        if entry.get("identity_generation") != "checkpoint_v1"
    }
    strong = [entry for entry in entries if entry.get("identity_generation") == "checkpoint_v1"]
    legacy_entries = [
        entry for entry in entries if entry.get("identity_generation") != "checkpoint_v1"
    ]
    expected_legacy_sha256 = source_ledger.get("legacy_entries_sha256")
    if sha256_jsonable(legacy_entries) != expected_legacy_sha256:
        raise RuntimeError("Legacy manifest entries differ from the preserved catalog")
    if len(strong) != len(legacy):
        raise RuntimeError(f"Expected one strong entry per legacy entry: {len(strong)} != {len(legacy)}")
    for entry in entries:
        path = ROOT / str(entry["path"])
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Manifest integrity mismatch: {path}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        route = artifact.get("_metadata", {}).get("route_signature")
        if sha256_jsonable(route) != entry["route_signature_sha256"]:
            raise RuntimeError(f"Route signature mismatch: {path}")
        if entry.get("identity_generation") == "checkpoint_v1":
            legacy_entry = legacy[entry["legacy_artifact_id"]]
            legacy_artifact = json.loads((ROOT / legacy_entry["path"]).read_text(encoding="utf-8"))
            embedded_identity = route.get("model", {}).get("checkpoint_identity")
            if embedded_identity != entry.get("checkpoint_identity"):
                raise RuntimeError(f"Checkpoint identity mismatch: {path}")
            if {key: value for key, value in artifact.items() if key != "_metadata"} != {
                key: value for key, value in legacy_artifact.items() if key != "_metadata"
            }:
                raise RuntimeError(f"Migrated routing counts differ: {path}")
    print(f"VERIFIED {len(legacy)} legacy + {len(strong)} checkpoint-v1 artifacts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "migrate", "verify"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.action == "verify":
        verify(directory)
        return
    audited = audit_sources(directory)
    if args.action == "migrate":
        migrate(directory, audited, args.dry_run)


if __name__ == "__main__":
    main()
