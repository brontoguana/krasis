#!/usr/bin/env bash
# Full GPU driver reset — stops Xorg, reloads NVIDIA kernel modules, restarts Xorg.
# Usage: sudo ./gpu_reset.sh
#
# This fixes CUDA driver corruption (cuDeviceGetCount returning wrong values)
# without requiring a full system reboot.
#
# IMPORTANT: Display server is stopped during this process. You will briefly
# lose your desktop session.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo ./gpu_reset.sh)"
    exit 1
fi

echo "=== GPU Reset Script ==="
echo ""

# 1. Show current state
echo "Current GPU state:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
echo ""

# 2. Kill ALL processes using GPUs (including display server processes)
echo "Killing all GPU processes..."
PIDS=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | sort -un || true)
for PID in $PIDS; do
    CMD=$(ps -p "$PID" -o comm= 2>/dev/null || echo "unknown")
    echo "  Killing PID $PID ($CMD)..."
    kill -TERM "$PID" 2>/dev/null || true
done
if [[ -n "$PIDS" ]]; then
    sleep 2
    for PID in $PIDS; do
        kill -9 "$PID" 2>/dev/null || true
    done
    sleep 1
fi

# 3. Stop display server (must happen BEFORE module unload)
echo "Stopping display server..."
systemctl stop gdm 2>/dev/null || systemctl stop sddm 2>/dev/null || systemctl stop lightdm 2>/dev/null || echo "  No display manager found"
sleep 3

# 4. Kill any remaining GPU users (display server children etc.)
PIDS=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | sort -un || true)
for PID in $PIDS; do
    kill -9 "$PID" 2>/dev/null || true
done
sleep 1

# 5. Unload NVIDIA kernel modules (order matters — dependencies first)
echo "Unloading NVIDIA kernel modules..."
for mod in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
    if lsmod | grep -q "^$mod "; then
        if rmmod "$mod" 2>/dev/null; then
            echo "  $mod unloaded"
        else
            echo "  WARNING: Failed to unload $mod ($(lsmod | grep "^$mod " | awk '{print $3}') users)"
            # Try harder — kill anything still holding it
            PIDS=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | sort -un || true)
            for PID in $PIDS; do
                kill -9 "$PID" 2>/dev/null || true
            done
            sleep 1
            rmmod "$mod" 2>/dev/null && echo "  $mod unloaded (retry)" || echo "  FAILED: $mod still loaded"
        fi
    fi
done

sleep 1

# 6. Verify modules are unloaded
if lsmod | grep -q "^nvidia "; then
    echo ""
    echo "ERROR: nvidia module still loaded. Remaining users:"
    lsmod | grep nvidia
    echo ""
    echo "A full reboot is required."
    # Still try to restart display server so user has a desktop
    systemctl start gdm 2>/dev/null || true
    exit 1
fi

# 7. Reload NVIDIA kernel modules
echo "Loading NVIDIA kernel modules..."
modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_drm
modprobe nvidia_modeset
echo "  All modules loaded"

sleep 1

# 8. Restart display server
echo "Restarting display server..."
systemctl start gdm 2>/dev/null || systemctl start sddm 2>/dev/null || systemctl start lightdm 2>/dev/null || echo "  No display manager found"
sleep 3

# 9. Verify
echo ""
echo "=== Post-Reset GPU State ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed (driver may not have loaded)"

echo ""
echo "Done. CUDA should now see all GPUs."
