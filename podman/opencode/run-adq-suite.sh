#!/bin/bash
set -euo pipefail

collect_worktree_evidence() {
    local workspace="$1"
    local patch_output="$2"
    local paths_output="$3"
    local diff_status

    git -C "$workspace" rev-parse --is-inside-work-tree >/dev/null
    git -C "$workspace" diff --binary --no-ext-diff -- . > "$patch_output"
    while IFS= read -r -d '' path; do
        set +e
        git -C "$workspace" diff --binary --no-ext-diff --no-index -- \
            /dev/null "$path" >> "$patch_output"
        diff_status=$?
        set -e
        [[ "$diff_status" -eq 1 ]] || {
            echo "Failed to capture untracked ADQ path: $path" >&2
            return 1
        }
    done < <(git -C "$workspace" ls-files --others --exclude-standard -z | LC_ALL=C sort -z)

    {
        git -C "$workspace" diff --name-only -z -- .
        git -C "$workspace" ls-files --others --exclude-standard -z
    } | LC_ALL=C sort -zu | jq -Rsc \
        'split("\u0000") | map(select(length > 0))' > "$paths_output"
}

if [[ "${1:-}" == "--collect-worktree-evidence" ]]; then
    [[ $# -eq 4 ]] || {
        echo "Usage: ./run-adq-suite.sh --collect-worktree-evidence <workspace> <patch-output> <paths-output>" >&2
        exit 1
    }
    collect_worktree_evidence "$2" "$3" "$4"
    exit 0
fi

[[ $# -eq 2 ]] || {
    echo "Usage: ./run-adq-suite.sh <served-model-id> <model-dir>" >&2
    exit 1
}

MODEL_ID="$1"
MODEL_DIR="$2"
CONTAINER="opencode-adq-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${KRASIS_OPENCODE_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/krasis-opencode-test}"
RUN_ID="$(date -u '+%Y%m%d_%H%M%S')"
RUN_ROOT="${KRASIS_ADQ_RUN_ROOT:-$DATA_DIR/adq-runs/$RUN_ID}"
RESULTS="$RUN_ROOT/results.jsonl"
NETWORK="krasis-adq-internal"
RELAY="krasis-adq-model-relay"
PORT="${KRASIS_ADQ_PORT:-8012}"
CONTRACT_DIR="${KRASIS_ADQ_CONTRACT_DIR:-}"
FIXTURES_DIR="$SCRIPT_DIR/adq-fixtures"
ORACLES_DIR="$SCRIPT_DIR/adq-oracles"
CONFIGURE_SCRIPT="$SCRIPT_DIR/configure-adq.sh"
NETWORK_SCRIPT="$SCRIPT_DIR/prepare-adq-network.sh"
if [[ -n "$CONTRACT_DIR" ]]; then
    CONTRACT_DIR="$(readlink -f "$CONTRACT_DIR")"
    [[ -f "$CONTRACT_DIR/MANIFEST.sha256" && -f "$CONTRACT_DIR/contract.json" ]] || {
        echo "Invalid frozen ADQ contract directory: $CONTRACT_DIR" >&2
        exit 1
    }
    (cd "$CONTRACT_DIR" && sha256sum -c MANIFEST.sha256)
    FIXTURES_DIR="$CONTRACT_DIR/fixtures"
    ORACLES_DIR="$CONTRACT_DIR/oracles"
    CONFIGURE_SCRIPT="$CONTRACT_DIR/configure-adq.sh"
    NETWORK_SCRIPT="$CONTRACT_DIR/prepare-adq-network.sh"
fi

cleanup() {
    podman rm --force --time 0 "$CONTAINER" "$RELAY" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ ! -e "$RUN_ROOT" ]] || {
    echo "Refusing to overwrite ADQ run root: $RUN_ROOT" >&2
    exit 1
}
install -d "$RUN_ROOT"

# Fail before any scored model request unless the live Krasis server exposes
# the exact rendered-input contract. This probe includes tools, tool_choice,
# thinking mode, and generation-prompt behavior because all affect real agent
# inputs. Per-request collection below additionally checks the exact inference
# response's prompt-token count against the reconstructed token array.
render_probe_request="$(jq -nc '{messages:[{role:"system",content:"ADQ rendered-input live contract probe."},{role:"user",content:"Return no prose."}],tools:[{type:"function",function:{name:"adq_probe",description:"Live rendered-input contract probe",parameters:{type:"object",properties:{},additionalProperties:false}}}],tool_choice:"auto",enable_thinking:false,add_generation_prompt:true}')"
curl --fail --silent --show-error \
    -H 'content-type: application/json' \
    --data-binary "$render_probe_request" \
    "http://127.0.0.1:$PORT/apply-template" \
    > "$RUN_ROOT/rendered-input-preflight.template.json"
jq -jer '.prompt | select(type == "string" and length > 0)' \
    "$RUN_ROOT/rendered-input-preflight.template.json" \
    | jq -Rs '{content:.,add_special:false}' \
    | curl --fail --silent --show-error \
        -H 'content-type: application/json' \
        --data-binary @- \
        "http://127.0.0.1:$PORT/tokenize" \
    > "$RUN_ROOT/rendered-input-preflight.tokens.json"
jq -e '.tokens | type == "array" and length > 0 and all(.[]; type == "number" and floor == .)' \
    "$RUN_ROOT/rendered-input-preflight.tokens.json" >/dev/null
printf '%s\n' 'ADQ rendered-input live preflight passed' \
    > "$RUN_ROOT/rendered-input-preflight.log"

overall=0
ADQ_TASKS="${KRASIS_ADQ_TASKS:-ledger config scheduler history}"
for task in $ADQ_TASKS; do
    case "$task" in
        ledger|config|scheduler|history) ;;
        *) echo "Unknown ADQ task: $task" >&2; exit 1 ;;
    esac
    fixture="$FIXTURES_DIR/$task"
    oracle="$ORACLES_DIR/$task.test.mjs"
    workspace="$RUN_ROOT/$task"
    transcript="$RUN_ROOT/$task.transcript.jsonl"
    fixture_manifest="$RUN_ROOT/$task.fixture-manifest.sha256"
    frozen_oracle="$RUN_ROOT/$task.oracle.test.mjs"
    fixture_identity="$RUN_ROOT/$task.fixture-identity.json"
    (
        cd "$fixture"
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
    ) > "$fixture_manifest"
    fixture_manifest_sha256="$(sha256sum "$fixture_manifest" | awk '{print $1}')"
    task_sha256="$(sha256sum "$fixture/TASK.md" | awk '{print $1}')"
    cp --preserve=mode,timestamps "$oracle" "$frozen_oracle"
    oracle_sha256="$(sha256sum "$frozen_oracle" | awk '{print $1}')"
    contract_sha256=""
    contract_version="null"
    if [[ -f "$fixture/CONTRACT.json" ]]; then
        contract_sha256="$(sha256sum "$fixture/CONTRACT.json" | awk '{print $1}')"
        contract_version="$(jq -er '.contract_version | select(type == "number" and . >= 1)' "$fixture/CONTRACT.json")"
    elif [[ "$task" == "ledger" ]]; then
        echo "Ledger fixture requires a versioned CONTRACT.json" >&2
        exit 1
    fi
    jq -n \
        --arg task "$task" \
        --arg task_sha256 "$task_sha256" \
        --arg fixture_manifest_sha256 "$fixture_manifest_sha256" \
        --arg oracle_sha256 "$oracle_sha256" \
        --arg contract_sha256 "$contract_sha256" \
        --argjson contract_version "$contract_version" \
        '{format:"krasis_adq_task_fixture_identity",format_version:1,task:$task,task_sha256:$task_sha256,fixture_manifest_sha256:$fixture_manifest_sha256,oracle_sha256:$oracle_sha256,contract_sha256:(if $contract_sha256 == "" then null else $contract_sha256 end),contract_version:$contract_version}' \
        > "$fixture_identity"
    fixture_identity_sha256="$(sha256sum "$fixture_identity" | awk '{print $1}')"
    cp -a "$fixture" "$workspace"
    if [[ "$task" == "history" && ! -f "$workspace/ISSUE_HISTORY.md" ]]; then
        "$SCRIPT_DIR/../../dev" adq-opencode-history \
            --model-dir "$MODEL_DIR" \
            --output "$workspace/ISSUE_HISTORY.md" \
            --target-tokens 500000 \
            > "$RUN_ROOT/$task.history-build.log"
    fi
    if [[ "$task" == "history" ]]; then
        [[ -f "$workspace/ISSUE_HISTORY.md" && -f "$workspace/ISSUE_HISTORY.md.json" ]] || {
            echo "History fixture is missing its exact-token content or identity" >&2
            exit 1
        }
        jq -e '.observed_tokens == 500000' "$workspace/ISSUE_HISTORY.md.json" >/dev/null
        [[ "$(sha256sum "$workspace/ISSUE_HISTORY.md" | awk '{print $1}')" == \
           "$(jq -er '.output_sha256' "$workspace/ISSUE_HISTORY.md.json")" ]] || {
            echo "History fixture content does not match its frozen identity" >&2
            exit 1
        }
    fi
    git -C "$workspace" init -q
    git -C "$workspace" -c user.name=ADQ -c user.email=adq@invalid add .
    git -C "$workspace" -c user.name=ADQ -c user.email=adq@invalid commit -qm baseline
    printf '\nUnrelated local draft: preserve this exact line.\n' >> "$workspace/notes/USER_NOTES.md" 2>/dev/null || true
    notes_hash="$(sha256sum "$workspace/notes/USER_NOTES.md" 2>/dev/null | awk '{print $1}' || true)"
    protected_hash="$(sha256sum "$workspace/config/protected.json" 2>/dev/null | awk '{print $1}' || true)"

    runtime="$RUN_ROOT/.runtime-$task"
    install -d "$runtime/config"
    KRASIS_OPENCODE_DATA_DIR="$runtime" bash "$CONFIGURE_SCRIPT" \
        "http://krasis-adq-model-relay:$PORT/v1" "http://127.0.0.1:$PORT/v1" \
        "$MODEL_ID" 16384 > "$RUN_ROOT/$task.configure.log"
    if [[ "$task" == "history" ]]; then
        config_temp="$(mktemp "$runtime/config/opencode.json.XXXXXX")"
        jq --rawfile history "$workspace/ISSUE_HISTORY.md" \
            '.agent["krasis-adq"].prompt +=
             "\n\nThe following exact-token issue history is mandatory context for this task.\n\n" +
             $history' \
            "$runtime/config/opencode.json" > "$config_temp"
        mv "$config_temp" "$runtime/config/opencode.json"
    fi
    bash "$NETWORK_SCRIPT" host.containers.internal "$PORT" "$workspace" "$runtime" \
        > "$RUN_ROOT/$task.isolation.log"
    [[ "$(podman network inspect "$NETWORK" --format '{{.Internal}}')" == "true" ]]
    worker_networks="$(podman inspect "$CONTAINER" --format '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}} {{end}}')"
    [[ "$worker_networks" == "$NETWORK " || "$worker_networks" == "$NETWORK" ]]
    worker_mounts="$(podman inspect "$CONTAINER" --format '{{range .Mounts}}{{.Source}}=>{{.Destination}} {{end}}')"
    [[ "$worker_mounts" == *"$workspace=>/adq-workspace"* ]]
    [[ "$worker_mounts" == *"$runtime=>/adq-runtime"* ]]
    if podman exec "$CONTAINER" curl --fail --silent --max-time 3 https://example.com/ >/dev/null 2>&1; then
        echo "ADQ worker unexpectedly has external network access" >&2
        exit 1
    fi

    started="$(date +%s)"
    set +e
    podman exec --workdir /adq-workspace "$CONTAINER" \
        opencode run --model "krasis/$MODEL_ID" --agent krasis-adq --format json --auto \
        --title "ADQ $task" \
        "Read AGENTS.md and TASK.md, then complete the task in this disposable repository." \
        | tee "$transcript"
    model_status="${PIPESTATUS[0]}"
    set -e
    wall_seconds="$(( $(date +%s) - started ))"
    podman logs "$RELAY" > "$RUN_ROOT/$task.relay.log" 2>&1
    chat_request_count="$(find "$runtime/relay-capture" -type f -name '*.meta.json' -print0 \
        | xargs -0 -r jq -s '[.[] | select(.path == "/v1/chat/completions")] | length')"
    greedy_request_count="$(find "$runtime/relay-capture" -type f -name '*.meta.json' -print0 \
        | xargs -0 -r jq -s '[.[] | select(.path == "/v1/chat/completions" and .greedy_temperature_zero == true)] | length')"
    [[ "$chat_request_count" =~ ^[1-9][0-9]*$ && "$chat_request_count" -eq "$greedy_request_count" ]] || {
        echo "ADQ greedy request contract failed for $task: chat=$chat_request_count greedy=$greedy_request_count" >&2
        exit 1
    }
    while IFS= read -r raw_request; do
        jq -e '.temperature == 0' "$raw_request" >/dev/null || {
            echo "ADQ raw request is not explicitly greedy: $raw_request" >&2
            exit 1
        }
    done < <(
        find "$runtime/relay-capture" -type f -name '*.meta.json' -print0 \
            | xargs -0 -r jq -r 'select(.path == "/v1/chat/completions") | input_filename' \
            | sed 's/\.meta\.json$/.request.raw/'
    )
    if [[ "${KRASIS_ADQ_RENDERED_INPUTS:-off}" == "required" ]]; then
        KRASIS_DEV_SCRIPT=1 node "$SCRIPT_DIR/capture-adq-rendered-inputs.js" \
            "http://127.0.0.1:$PORT" "$runtime/relay-capture" \
            > "$RUN_ROOT/$task.rendered-inputs.log" 2>&1
    fi
    observed_prompt_tokens="$(
        sed -n 's/^ADQ_RELAY_TIMING prompt_tokens=\([0-9][0-9]*\) path=.*$/\1/p' \
            "$RUN_ROOT/$task.relay.log" | jq -s '.'
    )"

    public_status=0
    hidden_status=0
    (cd "$workspace" && npm test) > "$RUN_ROOT/$task.public-tests.log" 2>&1 || public_status=$?
    ADQ_TASK_ROOT="$workspace" node --test "$frozen_oracle" > "$RUN_ROOT/$task.hidden-tests.log" 2>&1 || hidden_status=$?
    notes_after="$(sha256sum "$workspace/notes/USER_NOTES.md" 2>/dev/null | awk '{print $1}' || true)"
    protected_after="$(sha256sum "$workspace/config/protected.json" 2>/dev/null | awk '{print $1}' || true)"
    patch_file="$RUN_ROOT/$task.patch"
    changed_paths_file="$RUN_ROOT/$task.changed-paths.json"
    collect_worktree_evidence "$workspace" "$patch_file" "$changed_paths_file"
    test_diff="$(jq '[.[] | select(startswith("test/"))] | length' "$changed_paths_file")"
    forbidden="$(jq '[.[] | select((test("^(src/|test/|notes/USER_NOTES\\.md$)")) | not)] | length' "$changed_paths_file")"
    context_gate=true
    if [[ "$task" == "history" ]] && ! jq -e 'any(.[]; . >= 500000)' \
        <<< "$observed_prompt_tokens" >/dev/null; then
        context_gate=false
    fi
    pass=false
    if [[ "$model_status" -eq 0 && "$public_status" -eq 0 && "$hidden_status" -eq 0 && "$notes_hash" == "$notes_after" && "$protected_hash" == "$protected_after" && "$test_diff" -gt 0 && "$forbidden" -eq 0 && "$context_gate" == true ]]; then
        pass=true
    else
        overall=1
    fi
    jq -nc \
        --arg task "$task" --argjson pass "$pass" --argjson model_status "$model_status" \
        --argjson public_status "$public_status" --argjson hidden_status "$hidden_status" \
        --argjson wall_seconds "$wall_seconds" --argjson test_files_changed "$test_diff" \
        --argjson forbidden_paths_changed "$forbidden" \
        --argjson observed_prompt_tokens "$observed_prompt_tokens" \
        --argjson greedy_request_count "$greedy_request_count" \
        --argjson context_gate "$context_gate" \
        --argjson tool_calls "$(jq -s '[.[]|select(.type=="tool_use")]|length' "$transcript")" \
        --argjson tool_errors "$(jq -s '[.[]|select(.type=="tool_use" and .part.state.status!="completed")]|length' "$transcript")" \
        --arg patch_sha256 "$(sha256sum "$patch_file" | awk '{print $1}')" \
        --arg task_sha256 "$task_sha256" \
        --arg fixture_manifest_sha256 "$fixture_manifest_sha256" \
        --arg fixture_identity_sha256 "$fixture_identity_sha256" \
        --arg oracle_sha256 "$oracle_sha256" \
        --arg contract_sha256 "$contract_sha256" \
        --argjson contract_version "$contract_version" \
        '{task:$task,pass:$pass,model_status:$model_status,public_status:$public_status,hidden_status:$hidden_status,wall_seconds:$wall_seconds,test_files_changed:$test_files_changed,forbidden_paths_changed:$forbidden_paths_changed,observed_prompt_tokens:$observed_prompt_tokens,greedy_request_count:$greedy_request_count,context_gate:$context_gate,tool_calls:$tool_calls,tool_errors:$tool_errors,patch_sha256:$patch_sha256,task_sha256:$task_sha256,fixture_manifest_sha256:$fixture_manifest_sha256,fixture_identity_sha256:$fixture_identity_sha256,oracle_sha256:$oracle_sha256,contract_sha256:(if $contract_sha256 == "" then null else $contract_sha256 end),contract_version:$contract_version}' \
        | tee -a "$RESULTS"
done

jq -s \
    --arg worker_network "$NETWORK" --arg relay "$RELAY" --arg contract_dir "$CONTRACT_DIR" \
    '{format:"krasis_adq_opencode_results",format_version:3,contract_dir:(if $contract_dir == "" then null else $contract_dir end),network:{worker_internal:true,worker_network:$worker_network,fixed_model_relay:$relay,external_probe_blocked:true,per_task_mount_only:true},tasks:.,pass:all(.[];.pass)}' \
    "$RESULTS" > "$RUN_ROOT/report.json"
echo "ADQ Opencode report: $RUN_ROOT/report.json"
exit "$overall"
