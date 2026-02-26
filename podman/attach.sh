#!/bin/bash
set -euo pipefail

CONTAINER="krasis-test"

if ! podman container exists "$CONTAINER" 2>/dev/null; then
    echo "Container '$CONTAINER' does not exist. Run ./podman-startup.sh first."
    exit 1
fi

STATUS=$(podman inspect "$CONTAINER" --format '{{.State.Status}}')
if [ "$STATUS" != "running" ]; then
    echo "Container is $STATUS, starting it..."
    podman start "$CONTAINER"
    sleep 1
fi

HOST_USER="$USER"

echo "Attaching to '$CONTAINER' as '$HOST_USER'..."
echo "(Type 'exit' to detach)"
echo ""
podman exec -it --user "$HOST_USER" -w "/home/$HOST_USER" "$CONTAINER" bash -l
