#!/bin/bash
# Dump Python stack trace of krasis process
PID=$(pgrep -f "krasis.server" | head -1)
if [ -z "$PID" ]; then
    echo "No krasis.server process found"
    exit 1
fi
echo "=== Dumping stack trace for PID $PID ==="
echo ""

# Method 1: py-spy (best Python-level traces)
echo "--- py-spy dump ---"
/home/main/Documents/Claude/krasis/.venv/bin/py-spy dump --pid $PID 2>&1

echo ""
echo "--- gdb native backtrace ---"
gdb -batch -ex "thread apply all bt" -p $PID 2>&1 | head -200
