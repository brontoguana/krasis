#!/usr/bin/env bash
# Exercise the real dispatcher with an isolated shell worker, never model data.
set -euo pipefail
[[ ${KRASIS_DEV_SCRIPT:-} == 1 ]] || { printf 'Run ./dev dev-dispatch-test\n' >&2; exit 2; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch=$(mktemp -d "$repo/tests/.dispatch-fixture.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

run_case() {
    local label=$1 source=$2 worker_status=$3 expected=$4
    local root="$scratch/$label" pid result deadline
    mkdir "$root"
    awk '
      /^case "\$cmd" in$/ {
        print "do_mixed_cache_rank() {"
        print "  : > \"$KRASIS_DISPATCH_FIXTURE_ROOT/ready\""
        print "  while [[ ! -e $KRASIS_DISPATCH_FIXTURE_ROOT/release ]]; do sleep 0.05; done"
        print "  printf \"WORKER_FINISHED=%s\\n\" \"$KRASIS_DISPATCH_FIXTURE_STATUS\""
        print "  return \"$KRASIS_DISPATCH_FIXTURE_STATUS\""
        print "}"
      }
      {print}
    ' "$source" > "$root/dev"
    chmod +x "$root/dev"
    (
        cd "$root"
        KRASIS_DISPATCH_FIXTURE_ROOT="$root" KRASIS_DISPATCH_FIXTURE_STATUS="$worker_status" \
            ./dev mixed-cache-rank unused unused unused
    ) > "$root/output.log" 2>&1 &
    pid=$!
    deadline=$((SECONDS + 20))
    until [[ -e $root/ready ]]; do
        if ! kill -0 "$pid" 2>/dev/null || [[ $SECONDS -ge $deadline ]]; then
            cat "$root/output.log" >&2
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" || true
            return 1
        fi
        sleep 0.05
    done
    # In-place editing preserves the open inode, reproducing shifted Bash reads.
    awk 'NR==2 {print "# Dispatcher mutation deliberately shifts source offsets while worker runs."} {print}' \
        "$root/dev" > "$root/next"
    printf '\nexit 49\n' >> "$root/next"
    cat "$root/next" > "$root/dev"
    : > "$root/release"
    set +e
    wait "$pid"
    result=$?
    set -e
    rg -F "WORKER_FINISHED=$worker_status" "$root/output.log"
    if [[ $expected == failure ]]; then
        [[ $result != 0 ]]
    else
        [[ $result == "$expected" ]]
    fi
    printf '%s: worker_status=%s command_status=%s PASS\n' "$label" "$worker_status" "$result"
}

if [[ -n ${KRASIS_DISPATCH_TEST_ORIGINAL:-} ]]; then
    run_case original-control "$KRASIS_DISPATCH_TEST_ORIGINAL" 0 failure
fi
run_case successful-worker "$repo/dev" 0 0
run_case failed-worker "$repo/dev" 27 27
