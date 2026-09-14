#!/bin/bash
set -euo pipefail

[[ $# -eq 6 ]] || {
    echo "Usage: ./capture-adq-replay.sh <source-run> <task> <session-id> <model-id> <output-dir> <image>" >&2
    exit 1
}

SOURCE_RUN="$1"
TASK="$2"
SESSION_ID="$3"
MODEL_ID="$4"
OUTPUT_DIR="$5"
IMAGE="$6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "$TASK" =~ ^[a-z][a-z0-9_-]*$ ]] || {
    echo "Task must be a lowercase identifier: $TASK" >&2
    exit 1
}
SOURCE_RUNTIME="$SOURCE_RUN/.runtime-$TASK"
SOURCE_WORKSPACE="$SOURCE_RUN/$TASK"
TOKEN="$$-$(date -u '+%Y%m%d%H%M%S')"
NETWORK="adq-replay-capture-$TOKEN"
RELAY="adq-replay-capture-relay-$TOKEN"
WORKER="adq-replay-capture-worker-$TOKEN"
TEMP_ROOT="$(mktemp -d)"

[[ -d "$SOURCE_RUNTIME" && -f "$SOURCE_RUNTIME/config/opencode.json" ]] || {
    echo "Missing retained ledger runtime: $SOURCE_RUNTIME" >&2
    exit 1
}
[[ -d "$SOURCE_WORKSPACE" ]] || {
    echo "Missing retained ledger workspace: $SOURCE_WORKSPACE" >&2
    exit 1
}
[[ ! -e "$OUTPUT_DIR" ]] || {
    echo "Replay output directory already exists; evidence is immutable: $OUTPUT_DIR" >&2
    exit 1
}

cleanup() {
    podman rm --force --time 0 "$WORKER" "$RELAY" >/dev/null 2>&1 || true
    podman network rm --force "$NETWORK" >/dev/null 2>&1 || true
    rm -rf -- "$TEMP_ROOT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUTPUT_DIR/capture"
cp -a "$SOURCE_RUNTIME" "$TEMP_ROOT/runtime"
cp -a "$SOURCE_WORKSPACE" "$TEMP_ROOT/workspace"

CONFIG="$TEMP_ROOT/runtime/config/opencode.json"
CONFIG_TEMP="$TEMP_ROOT/opencode.json"
jq --arg url "http://$RELAY:8012/v1" \
    '.provider.krasis.options.baseURL = $url | .provider.krasis.options.timeout = false | del(.provider.krasis.options.chunkTimeout)' \
    "$CONFIG" > "$CONFIG_TEMP"
mv "$CONFIG_TEMP" "$CONFIG"

podman network create --internal "$NETWORK" >/dev/null
podman run -d --name "$RELAY" --network "$NETWORK" \
    --env ADq_CAPTURE_OUTPUT_DIR=/capture \
    --env ADq_CAPTURE_MODEL_ID="$MODEL_ID" \
    --volume "$OUTPUT_DIR/capture:/capture:Z" \
    --volume "$SCRIPT_DIR/adq-replay-capture-relay.js:/capture-relay.js:ro,Z" \
    "$IMAGE" node /capture-relay.js > "$OUTPUT_DIR/relay-container-id.txt"

for _ in $(seq 1 50); do
    if podman logs "$RELAY" 2>&1 | grep -q 'ADQ replay capture relay ready'; then
        break
    fi
    sleep 0.1
done
podman logs "$RELAY" > "$OUTPUT_DIR/relay.log" 2>&1
grep -q 'ADQ replay capture relay ready' "$OUTPUT_DIR/relay.log"

set +e
podman run --name "$WORKER" --network "$NETWORK" \
    --env HOME=/adq-runtime/home \
    --env OPENCODE_CONFIG=/adq-runtime/config/opencode.json \
    --volume "$TEMP_ROOT/runtime:/adq-runtime:Z" \
    --volume "$TEMP_ROOT/workspace:/adq-workspace:ro,Z" \
    --workdir /adq-workspace \
    "$IMAGE" opencode run --session "$SESSION_ID" \
        --model "krasis/$MODEL_ID" --agent krasis-adq --format json --auto \
        --title "ADQ exact-state capture" \
        "ADQ_EXACT_STATE_CAPTURE_SENTINEL_DO_NOT_USE_AS_MODEL_EVIDENCE" \
    > "$OUTPUT_DIR/opencode-capture.jsonl" 2> "$OUTPUT_DIR/opencode-capture.stderr"
MODEL_STATUS=$?
set -e
printf '%s\n' "$MODEL_STATUS" > "$OUTPUT_DIR/opencode-capture.status"
podman logs "$RELAY" > "$OUTPUT_DIR/relay.log" 2>&1

mapfile -t CAPTURED < <(find "$OUTPUT_DIR/capture" -maxdepth 1 -type f -name 'request-*.json' ! -name '*.headers.json' | sort)
[[ ${#CAPTURED[@]} -eq 1 ]] || {
    echo "Expected exactly one captured chat request, found ${#CAPTURED[@]}" >&2
    exit 1
}
[[ "$MODEL_STATUS" -eq 0 ]] || {
    echo "Copied-session capture exited with status $MODEL_STATUS" >&2
    exit 1
}
printf '%s\n' "${CAPTURED[0]}"
