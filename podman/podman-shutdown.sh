#!/bin/bash
set -euo pipefail

CONTAINER="krasis-test"

if ! podman container exists "$CONTAINER" 2>/dev/null; then
    echo "Container '$CONTAINER' does not exist. Nothing to do."
    exit 0
fi

echo "Stopping container '$CONTAINER'..."
podman stop "$CONTAINER" 2>/dev/null || true

echo "Deleting container '$CONTAINER'..."
podman rm -f "$CONTAINER"

echo "Done. Container removed."
