#!/bin/bash
set -euo pipefail

CONTAINER="opencode-test"

if ! podman container exists "$CONTAINER"; then
    echo "Container '$CONTAINER' does not exist. Nothing to remove."
    exit 0
fi

podman stop --time 10 "$CONTAINER" >/dev/null || true
podman rm "$CONTAINER" >/dev/null
echo "Removed container '$CONTAINER'. Persistent test data was retained."
