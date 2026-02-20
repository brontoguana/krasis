#!/usr/bin/env bash
# GPU zombie cleanup — kills stale CUDA processes and resets GPUs if needed.
# Usage: sudo ./gpu_cleanup.sh
#
# What it does:
# 1. Lists all processes holding GPU memory
# 2. Kills zombie/stale processes (skips Xorg/display server)
# 3. If memory is still held, attempts nvidia-smi --gpu-reset
#
# Run with sudo for full cleanup capability.

set -euo pipefail

echo "=== GPU Memory Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader

echo ""
echo "=== GPU Compute Processes ==="
# Get PIDs using GPUs (compute processes only)
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -v 'N/A' | sort -u || true)

if [[ -z "$PIDS" ]]; then
    echo "No compute processes found on GPUs."
    echo ""
    echo "Checking for zombie processes holding GPU memory..."
    # Check if any GPU has significant memory used (>500 MB) without compute processes
    NEED_RESET=false
    while IFS=, read -r idx used free total; do
        used_mb=$(echo "$used" | tr -d ' MiB')
        if [[ "$used_mb" -gt 500 ]]; then
            echo "  GPU $idx: ${used_mb} MiB used (no compute process — likely zombie)"
            NEED_RESET=true
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader)

    if $NEED_RESET; then
        echo ""
        echo "Attempting GPU reset to free zombie memory..."
        for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" | tr -d ' ')
            if [[ "$used" -gt 500 ]]; then
                echo "  Resetting GPU $i (${used} MiB used)..."
                nvidia-smi --gpu-reset -i "$i" 2>&1 || echo "  Failed to reset GPU $i (may need reboot)"
            fi
        done
    else
        echo "All GPUs clean."
    fi
    exit 0
fi

echo "Found GPU processes:"
for PID in $PIDS; do
    # Get process info
    STAT=$(ps -p "$PID" -o stat= 2>/dev/null || echo "gone")
    CMD=$(ps -p "$PID" -o comm= 2>/dev/null || echo "unknown")
    MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep "^$PID," | head -1 || true)

    # Skip display server processes
    if [[ "$CMD" == "Xorg" || "$CMD" == "Xwayland" || "$CMD" == "gnome-shell" ]]; then
        echo "  PID $PID ($CMD) — display server, skipping"
        continue
    fi

    echo "  PID $PID ($CMD) — state: $STAT, GPU mem: $MEM"

    # Kill zombies (Z state) or stale processes
    if [[ "$STAT" == *"Z"* ]]; then
        echo "    -> Zombie process, killing parent..."
        PPID=$(ps -p "$PID" -o ppid= 2>/dev/null | tr -d ' ')
        if [[ -n "$PPID" && "$PPID" != "1" ]]; then
            kill -9 "$PPID" 2>/dev/null || true
            sleep 1
        fi
        # Try to kill zombie itself (won't work but signals init to reap)
        kill -9 "$PID" 2>/dev/null || true
    elif [[ "$STAT" == "gone" ]]; then
        echo "    -> Process already gone, GPU memory may be orphaned"
    else
        echo "    -> Active process. Kill? [y/N] "
        read -r -t 10 REPLY || REPLY="n"
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            echo "    -> Sending SIGTERM..."
            kill "$PID" 2>/dev/null || true
            sleep 2
            if kill -0 "$PID" 2>/dev/null; then
                echo "    -> Still alive, sending SIGKILL..."
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
    fi
done

sleep 2
echo ""
echo "=== GPU Memory After Cleanup ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader

# Check if we need GPU reset
NEED_RESET=false
while IFS=, read -r idx used free total; do
    used_mb=$(echo "$used" | tr -d ' MiB')
    if [[ "$used_mb" -gt 500 ]]; then
        # Check if there's actually a process using it
        HAS_PROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$idx" 2>/dev/null | grep -v 'N/A' | head -1 || true)
        if [[ -z "$HAS_PROC" ]]; then
            echo "GPU $idx still has ${used_mb} MiB used with no process — needs reset"
            NEED_RESET=true
        fi
    fi
done < <(nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader)

if $NEED_RESET; then
    echo ""
    echo "Attempting GPU reset for orphaned memory..."
    for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" | tr -d ' ')
        has_proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" 2>/dev/null | grep -v 'N/A' | head -1 || true)
        if [[ "$used" -gt 500 && -z "$has_proc" ]]; then
            echo "  Resetting GPU $i..."
            nvidia-smi --gpu-reset -i "$i" 2>&1 || echo "  Failed (may need reboot)"
        fi
    done
fi

echo ""
echo "Done."
