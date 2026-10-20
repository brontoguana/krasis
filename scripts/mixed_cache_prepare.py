#!/usr/bin/env python3
"""Supported offline entry point for compact mixed-cache materialization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Run this setup command through ./dev mixed-cache-prepare")


def canonical_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one crash-resilient compact INT4/INT8 routed-expert cache"
    )
    parser.add_argument("model_dir")
    parser.add_argument("plan")
    parser.add_argument("calibration")
    parser.add_argument("output_cache")
    parser.add_argument("output_manifest")
    parser.add_argument("--reuse-int4-manifest")
    parser.add_argument("--reuse-int8-manifest")
    args = parser.parse_args()
    if bool(args.reuse_int4_manifest) != bool(args.reuse_int8_manifest):
        parser.error("reuse requires both explicit INT4 and INT8 source manifests")

    from krasis import WeightStore

    inputs = (
        canonical_path(args.model_dir),
        canonical_path(args.plan),
        canonical_path(args.calibration),
        canonical_path(args.output_cache),
        canonical_path(args.output_manifest),
    )
    if args.reuse_int4_manifest:
        result = WeightStore.assemble_compact_mixed_cache(*inputs,
            canonical_path(args.reuse_int4_manifest), canonical_path(args.reuse_int8_manifest))
    else:
        result = WeightStore.prepare_compact_mixed_cache(*inputs)
    parsed = json.loads(result)
    print(json.dumps(parsed, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
