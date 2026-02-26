#!/bin/bash
set -euo pipefail

CONTAINER="krasis-test"
IMAGE="ubuntu:24.04"

echo "=== Krasis Test Container Setup (Podman) ==="

# Install podman if not present
if ! command -v podman &>/dev/null; then
    echo "Podman not found, installing..."
    sudo apt-get update
    sudo apt-get install -y podman
fi

# Remove existing container if present
if podman container exists "$CONTAINER" 2>/dev/null; then
    echo "Container '$CONTAINER' already exists. Removing it first..."
    podman rm -f "$CONTAINER" 2>/dev/null || true
fi

echo "Creating $IMAGE container '$CONTAINER'..."
podman create \
    --name "$CONTAINER" \
    --hostname "$CONTAINER" \
    -it \
    "$IMAGE" \
    bash

echo "Starting container..."
podman start "$CONTAINER"

# Minimal setup inside container
echo "Installing minimal essentials inside container..."
podman exec "$CONTAINER" bash -c "
    apt-get update -qq
    apt-get install -y -qq curl wget git sudo > /dev/null
"

# Create a regular user matching the host username (no UID pinning -- rootless podman remaps UIDs)
HOST_USER="$USER"
echo "Creating user '$HOST_USER' inside container..."
podman exec "$CONTAINER" bash -c "
    if ! id '$HOST_USER' &>/dev/null; then
        useradd -m -s /bin/bash '$HOST_USER'
    fi
    echo '$HOST_USER ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/$HOST_USER
"

echo ""
echo "=== Container '$CONTAINER' is ready ==="
echo ""
echo "Attach with:  ./podman-attach.sh"
echo "Shut down with: ./podman-shutdown.sh"
echo ""
echo "Clean Ubuntu -- no Python, no NVIDIA tooling preinstalled."
echo "To install krasis inside: follow the Krasis install instructions"
