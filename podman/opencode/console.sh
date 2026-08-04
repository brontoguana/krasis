#!/bin/bash
set -euo pipefail

CONTAINER="opencode-test"

if ! podman container exists "$CONTAINER"; then
    echo "Container '$CONTAINER' does not exist. Run ./startup.sh first." >&2
    exit 1
fi

if [[ "$(podman inspect "$CONTAINER" --format '{{.State.Status}}')" != "running" ]]; then
    podman start "$CONTAINER" >/dev/null
fi

exec podman exec -it --workdir /opencode-data/workspace "$CONTAINER" bash
