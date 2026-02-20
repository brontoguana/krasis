#!/usr/bin/env bash
# Run a Krasis benchmark and exit (don't start the server).
# Usage: ./run_benchmark.sh <server.py args...>
#
# Runs server.py --benchmark, waits for benchmark completion,
# then cleanly shuts down (SIGTERM first, then SIGKILL).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi

LOGFILE=$(mktemp /tmp/krasis_bench_XXXXXX.log)
echo "Benchmark output: $LOGFILE"

# Cleanup function to ensure process is killed on script exit
cleanup() {
    if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
        echo "Cleaning up benchmark process $PID..."
        kill -TERM "$PID" 2>/dev/null || true
        # Wait up to 5 seconds for graceful shutdown
        for i in $(seq 1 10); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 0.5
        done
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
        wait "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Run server with --benchmark, capture output
"$PYTHON" -m krasis.server --benchmark "$@" > >(tee "$LOGFILE") 2>&1 &
PID=$!

echo "Server PID: $PID"

# Wait for benchmark to complete (look for "Benchmark archived" or server start)
while kill -0 "$PID" 2>/dev/null; do
    if grep -q "Benchmark archived to\|starting server on" "$LOGFILE" 2>/dev/null; then
        echo ""
        echo "Benchmark complete. Stopping server..."
        # SIGTERM triggers our cleanup handler in server.py
        kill -TERM "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=""  # Prevent cleanup trap from double-killing
        echo "Done. Full output in: $LOGFILE"
        exit 0
    fi
    sleep 2
done

# Process exited on its own (error or benchmark completed)
EXITCODE=$?
PID=""  # Prevent cleanup trap from killing dead process
echo "Process exited with code $EXITCODE"
echo "Full output in: $LOGFILE"
exit $EXITCODE
