#!/bin/bash
set -euo pipefail

CONTAINER="krasis-test"
IMAGE="ubuntu:24.04"
HOST_MODELS="/home/$USER/.krasis/models"

if [ ! -d "$HOST_MODELS" ]; then
    echo "Krasis model directory not found: $HOST_MODELS" >&2
    exit 1
fi

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
    --device nvidia.com/gpu=all \
    --volume "$HOST_MODELS:/krasis-host-models:ro" \
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
    install -d -o '$HOST_USER' -g '$HOST_USER' '/home/$HOST_USER/.krasis'
    ln -sfn /krasis-host-models '/home/$HOST_USER/.krasis/models'
"

echo ""
echo "=== Container '$CONTAINER' is ready ==="
echo ""
echo "Attach with:  ./console.sh"
echo "Shut down with: ./finish.sh"
echo ""
echo "Clean Ubuntu -- no Python, no NVIDIA tooling preinstalled."
echo "Host models are available read-only through ~/.krasis/models."
echo "To install krasis inside: follow the Krasis install instructions"
