#!/bin/bash
set -euo pipefail

CONTAINER="opencode-test"
IMAGE="localhost/krasis-opencode-test:1.18.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${KRASIS_OPENCODE_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/krasis-opencode-test}"

if podman container exists "$CONTAINER"; then
    status=$(podman inspect "$CONTAINER" --format '{{.State.Status}}')
    if [[ "$status" != "running" ]]; then
        podman start "$CONTAINER" >/dev/null
        status="running"
    fi
    echo "Container '$CONTAINER' is already available ($status)."
    echo "Enter it with: $SCRIPT_DIR/console.sh"
    exit 0
fi

install -d "$DATA_DIR/home" "$DATA_DIR/config" "$DATA_DIR/workspace" "$DATA_DIR/transcripts"

podman build \
    --tag "$IMAGE" \
    --file "$SCRIPT_DIR/Containerfile" \
    "$SCRIPT_DIR"

podman create \
    --name "$CONTAINER" \
    --hostname "$CONTAINER" \
    --add-host=host.containers.internal:host-gateway \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    --env HOME=/opencode-data/home \
    --env OPENCODE_CONFIG=/opencode-data/config/opencode.json \
    --volume "$DATA_DIR:/opencode-data:Z" \
    --workdir /opencode-data/workspace \
    "$IMAGE" >/dev/null

podman start "$CONTAINER" >/dev/null

echo "Container '$CONTAINER' is ready."
echo "Persistent data: $DATA_DIR"
echo "Enter it with: $SCRIPT_DIR/console.sh"
