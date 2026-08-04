#!/bin/bash
set -euo pipefail

[[ $# -eq 3 ]] || {
    echo "Usage: ./configure.sh <http-base-url/v1> <served-model-id> <test-output-token-limit>" >&2
    exit 1
}

BASE_URL="$1"
MODEL_ID="$2"
OUTPUT_LIMIT="$3"
DATA_DIR="${KRASIS_OPENCODE_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/krasis-opencode-test}"
CONFIG_PATH="$DATA_DIR/config/opencode.json"
CONFIG_TEMP="$(mktemp "$DATA_DIR/config/opencode.json.XXXXXX")"

case "$BASE_URL" in
    http://*/v1) ;;
    *)
        echo "Base URL must be explicit plain HTTP and end in /v1: $BASE_URL" >&2
        exit 1
        ;;
esac

[[ "$OUTPUT_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "Test output limit must be a positive integer: $OUTPUT_LIMIT" >&2
    exit 1
}

MODELS_JSON="$(curl --fail --silent --show-error --max-time 10 "${BASE_URL%/}/models")"
CONTEXT_LIMIT="$(jq -er --arg model_id "$MODEL_ID" \
    '.data[] | select(.id == $model_id) | .max_context_tokens' <<< "$MODELS_JSON")"
[[ "$CONTEXT_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "Server did not publish a positive max_context_tokens for $MODEL_ID" >&2
    exit 1
}
(( OUTPUT_LIMIT < CONTEXT_LIMIT )) || {
    echo "Test output limit $OUTPUT_LIMIT must be below context limit $CONTEXT_LIMIT" >&2
    exit 1
}

jq -n \
    --arg base_url "$BASE_URL" \
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
        "krasis-verify": {
          "description": "Minimal read-only live tool-use verification",
          "mode": "primary",
          "model": ("krasis/" + $model_id),
          "prompt": "You are a concise coding assistant. Use the available tool to complete the user request before answering.",
          "temperature": 0,
          "permission": {
            "*": "deny",
            "read": "allow"
          }
        }
      },
      "provider": {
        "krasis": {
          "npm": "@ai-sdk/openai-compatible",
          "name": "Krasis live verification",
          "options": {
            "baseURL": $base_url,
            "apiKey": "krasis-local-test",
            "timeout": false,
            "chunkTimeout": 180000
          },
          "models": {
            ($model_id): {
              "name": $model_id,
              "limit": {
                "context": $context_limit,
                "output": $output_limit
              }
            }
          }
        }
      },
      "permission": {
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "bash": "deny",
        "edit": "deny",
        "write": "deny",
        "webfetch": "deny"
      }
    }' > "$CONFIG_TEMP"

mv "$CONFIG_TEMP" "$CONFIG_PATH"
echo "Configured krasis/$MODEL_ID at $BASE_URL (context=$CONTEXT_LIMIT, test output=$OUTPUT_LIMIT)"
