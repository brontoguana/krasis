#!/bin/bash
set -euo pipefail

[[ $# -eq 2 ]] || {
    echo "Usage: ./run-tool-test.sh <test-name> <served-model-id>" >&2
    exit 1
}

TEST_NAME="$1"
MODEL_ID="$2"
CONTAINER="opencode-test"
DATA_DIR="${KRASIS_OPENCODE_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/krasis-opencode-test}"
PROOF_TOKEN="KRASIS_TOOL_PROOF_${TEST_NAME}"
TRANSCRIPT="$DATA_DIR/transcripts/${TEST_NAME}.jsonl"

[[ "$TEST_NAME" =~ ^[a-z0-9_-]+$ ]] || {
    echo "Test name may contain only lowercase letters, digits, underscores, and hyphens." >&2
    exit 1
}

podman container exists "$CONTAINER" || {
    echo "Container '$CONTAINER' does not exist. Run ./startup.sh first." >&2
    exit 1
}

printf '%s\n' "$PROOF_TOKEN" > "$DATA_DIR/workspace/tool-proof.txt"

podman exec \
    --workdir /opencode-data/workspace \
    "$CONTAINER" \
    opencode run \
    --model "krasis/$MODEL_ID" \
    --agent krasis-verify \
    --format json \
    --auto \
    "Use the read tool to read /opencode-data/workspace/tool-proof.txt. Do not guess or answer before using the tool. After the tool result, reply with exactly TOOL_RESULT=<file contents> and nothing else." \
    | tee "$TRANSCRIPT"

jq --exit-status --slurp --arg proof "$PROOF_TOKEN" '
    any(.[];
        .type == "tool_use"
        and .part.tool == "read"
        and .part.state.status == "completed"
        and (.part.state.output | contains($proof)))
    and any(.[];
        .type == "step_finish"
        and .part.reason == "tool-calls")
    and any(.[];
        .type == "text"
        and (.part.text | contains("TOOL_RESULT=" + $proof)))
' "$TRANSCRIPT" >/dev/null
echo "Transcript: $TRANSCRIPT"
