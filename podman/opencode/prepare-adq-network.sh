#!/bin/bash
set -euo pipefail

[[ $# -eq 4 ]] || {
    echo "Usage: ./prepare-adq-network.sh <host-model-address> <host-model-port> <task-workspace> <task-runtime-dir>" >&2
    exit 1
}

UPSTREAM_HOST="$1"
UPSTREAM_PORT="$2"
WORKSPACE="$3"
RUNTIME_DIR="$4"
CONTAINER="opencode-adq-test"
RELAY="krasis-adq-model-relay"
NETWORK="krasis-adq-internal"
IMAGE="localhost/krasis-opencode-test:1.18.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ "$UPSTREAM_PORT" =~ ^[1-9][0-9]*$ ]] || {
    echo "Host model port must be a positive integer: $UPSTREAM_PORT" >&2
    exit 1
}
[[ -d "$WORKSPACE" && -d "$RUNTIME_DIR/config" && -f "$RUNTIME_DIR/config/opencode.json" ]] || {
    echo "Task workspace and configured task runtime must already exist" >&2
    exit 1
}
WORKSPACE="$(readlink -f "$WORKSPACE")"
RUNTIME_DIR="$(readlink -f "$RUNTIME_DIR")"
[[ "$WORKSPACE" != "/" && "$RUNTIME_DIR" != "/" && "$WORKSPACE" != "$RUNTIME_DIR" ]] || {
    echo "Refusing unsafe ADQ bind-mount roots" >&2
    exit 1
}
podman image exists "$IMAGE" || {
    echo "Image '$IMAGE' does not exist. Run ./startup.sh first." >&2
    exit 1
}

if ! podman network exists "$NETWORK"; then
    podman network create --internal "$NETWORK" >/dev/null
fi
[[ "$(podman network inspect "$NETWORK" --format '{{.Internal}}')" == "true" ]] || {
    echo "Refusing non-internal ADQ worker network: $NETWORK" >&2
    exit 1
}

if podman container exists "$RELAY"; then
    podman rm --force --time 0 "$RELAY" >/dev/null
fi
install -d "$RUNTIME_DIR/relay-capture"
podman create \
    --name "$RELAY" \
    --network podman \
    --add-host=host.containers.internal:host-gateway \
    --env ADq_UPSTREAM_HOST="$UPSTREAM_HOST" \
    --env ADq_UPSTREAM_PORT="$UPSTREAM_PORT" \
    --env ADq_LISTEN_PORT="$UPSTREAM_PORT" \
    --env ADq_CAPTURE_DIR=/adq-relay-capture \
    --volume "$SCRIPT_DIR/adq-model-relay.js:/opt/krasis/adq-model-relay.js:ro,Z" \
    --volume "$RUNTIME_DIR/relay-capture:/adq-relay-capture:Z" \
    "$IMAGE" node /opt/krasis/adq-model-relay.js >/dev/null
podman network connect --alias krasis-adq-model-relay "$NETWORK" "$RELAY"
podman start "$RELAY" >/dev/null

if podman container exists "$CONTAINER"; then
    podman rm --force --time 0 "$CONTAINER" >/dev/null
fi
install -d "$RUNTIME_DIR/home"
podman create \
    --name "$CONTAINER" \
    --hostname "$CONTAINER" \
    --network "$NETWORK" \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    --env HOME=/adq-runtime/home \
    --env OPENCODE_CONFIG=/adq-runtime/config/opencode.json \
    --volume "$WORKSPACE:/adq-workspace:Z" \
    --volume "$RUNTIME_DIR:/adq-runtime:Z" \
    --workdir /adq-workspace \
    "$IMAGE" >/dev/null
podman start "$CONTAINER" >/dev/null

for _ in $(seq 1 20); do
    if podman exec "$CONTAINER" curl --fail --silent --max-time 2 \
        "http://krasis-adq-model-relay:$UPSTREAM_PORT/v1/models" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
podman exec "$CONTAINER" curl --fail --silent --max-time 2 \
    "http://krasis-adq-model-relay:$UPSTREAM_PORT/v1/models" >/dev/null

if podman exec "$CONTAINER" curl --fail --silent --max-time 3 https://example.com/ >/dev/null 2>&1; then
    echo "ADQ worker unexpectedly has external network access" >&2
    exit 1
fi
if podman exec "$CONTAINER" curl --fail --silent --max-time 2 \
    "http://host.containers.internal:$UPSTREAM_PORT/v1/models" >/dev/null 2>&1; then
    echo "ADQ worker unexpectedly has direct host access" >&2
    exit 1
fi

echo "ADQ worker isolated on $NETWORK with only the current task mounted; fixed relay $RELAY:$UPSTREAM_PORT is reachable"
