#!/bin/bash
set -euo pipefail

[[ $# -eq 4 ]] || {
    echo "Usage: ./configure-adq.sh <container-http-base-url/v1> <host-probe-http-base-url/v1> <served-model-id> <test-output-token-limit>" >&2
    exit 1
}

CONTAINER_BASE_URL="$1"
HOST_PROBE_BASE_URL="$2"
MODEL_ID="$3"
OUTPUT_LIMIT="$4"
DATA_DIR="${KRASIS_OPENCODE_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/krasis-opencode-test}"
CONFIG_PATH="$DATA_DIR/config/opencode.json"
CONFIG_TEMP="$(mktemp "$DATA_DIR/config/opencode.json.XXXXXX")"

case "$CONTAINER_BASE_URL" in
    http://*/v1) ;;
    *) echo "Container base URL must be explicit plain HTTP and end in /v1: $CONTAINER_BASE_URL" >&2; exit 1 ;;
esac
case "$HOST_PROBE_BASE_URL" in
    http://*/v1) ;;
    *) echo "Host probe base URL must be explicit plain HTTP and end in /v1: $HOST_PROBE_BASE_URL" >&2; exit 1 ;;
esac
[[ "$OUTPUT_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "Test output limit must be a positive integer: $OUTPUT_LIMIT" >&2
    exit 1
}

MODELS_JSON="$(curl --fail --silent --show-error --max-time 10 "${HOST_PROBE_BASE_URL%/}/models")"
CONTEXT_LIMIT="$(jq -er --arg model_id "$MODEL_ID" '
  .data[] | select(.id == $model_id) |
  (.max_context_tokens // .meta.n_ctx)
' <<< "$MODELS_JSON")"
[[ "$CONTEXT_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "Server did not publish a positive max_context_tokens for $MODEL_ID" >&2
    exit 1
}
(( OUTPUT_LIMIT < CONTEXT_LIMIT )) || {
    echo "Test output limit $OUTPUT_LIMIT must be below context limit $CONTEXT_LIMIT" >&2
    exit 1
}

jq -n \
    --arg base_url "$CONTAINER_BASE_URL" \
    --arg model_id "$MODEL_ID" \
    --argjson context_limit "$CONTEXT_LIMIT" \
    --argjson output_limit "$OUTPUT_LIMIT" \
    '{
      "$schema": "https://opencode.ai/config.json",
      "model": ("krasis/" + $model_id),
      "small_model": ("krasis/" + $model_id),
      "enabled_providers": ["krasis"],
      "snapshot": false,
      "share": "disabled",
      "agent": {
        "krasis-adq": {
          "description": "Isolated deterministic ADQ software-engineering agent",
          "mode": "primary",
          "model": ("krasis/" + $model_id),
          "prompt": "Work only inside the current disposable repository. Follow AGENTS.md. Inspect before editing, preserve unrelated changes, run the relevant tests, and report only work actually verified. Network access is forbidden.",
          "temperature": 0,
          "permission": {
            "*": "deny",
            "read": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "bash": "allow",
            "edit": "allow",
            "write": "allow"
          }
        }
      },
      "provider": {
        "krasis": {
          "npm": "@ai-sdk/openai-compatible",
          "name": "Krasis ADQ",
          "options": {
            "baseURL": $base_url,
            "apiKey": "krasis-local-adq",
            "timeout": false
          },
          "models": {
            ($model_id): {
              "name": $model_id,
              "limit": {"context": $context_limit, "output": $output_limit}
            }
          }
        }
      },
      "permission": {
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "bash": "allow",
        "edit": "allow",
        "write": "allow",
        "webfetch": "deny"
      }
    }' > "$CONFIG_TEMP"

mv "$CONFIG_TEMP" "$CONFIG_PATH"
echo "Configured ADQ agent krasis/$MODEL_ID at $CONTAINER_BASE_URL (context=$CONTEXT_LIMIT, output=$OUTPUT_LIMIT)"
