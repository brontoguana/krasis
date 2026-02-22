#!/bin/bash
# Krasis PCIe + GPU optimization script
# Run with: sudo bash setup_pcie.sh

set -e

echo "=== Krasis PCIe/GPU Optimization ==="
echo

# 1. Enable GPU persistence mode (prevents P8 idle, reduces wake latency)
echo "[1/4] Enabling GPU persistence mode..."
nvidia-smi -pm 1
echo "  Done. GPUs will stay in P0 state."
echo

# 2. Disable ASPM (prevents PCIe link speed drops between DMA bursts)
echo "[2/4] Disabling PCIe ASPM..."
if [ -f /sys/module/pcie_aspm/parameters/policy ]; then
    CURRENT=$(cat /sys/module/pcie_aspm/parameters/policy | grep -o '\[.*\]' | tr -d '[]')
    echo "  Current ASPM policy: $CURRENT"
    echo "performance" > /sys/module/pcie_aspm/parameters/policy
    NEW=$(cat /sys/module/pcie_aspm/parameters/policy | grep -o '\[.*\]' | tr -d '[]')
    echo "  New ASPM policy: $NEW"
else
    echo "  ASPM sysfs not available, skipping"
fi
echo

# 3. Set GPU power to max performance (prevent clock throttling)
echo "[3/4] Setting GPU power mode to max performance..."
for i in 0 1 2; do
    nvidia-smi -i $i -pl $(nvidia-smi -i $i --query-gpu=power.max_limit --format=csv,noheader,nounits | tr -d ' ') 2>/dev/null || true
done
echo "  Done."
echo

# 4. Verify PCIe link status
echo "[4/4] PCIe link status:"
for dev in 01:00.0 81:00.0 c4:00.0; do
    GPU_IDX=$(lspci -s $dev -v 2>/dev/null | head -1)
    CUR_SPEED=$(cat /sys/bus/pci/devices/0000:$dev/current_link_speed 2>/dev/null)
    MAX_SPEED=$(cat /sys/bus/pci/devices/0000:$dev/max_link_speed 2>/dev/null)
    CUR_WIDTH=$(cat /sys/bus/pci/devices/0000:$dev/current_link_width 2>/dev/null)
    MAX_WIDTH=$(cat /sys/bus/pci/devices/0000:$dev/max_link_width 2>/dev/null)
    echo "  $dev: ${CUR_SPEED} x${CUR_WIDTH} (max: ${MAX_SPEED} x${MAX_WIDTH})"
done
echo

# Summary
echo "=== Done ==="
echo "Persistence mode: ON (survives until reboot)"
echo "ASPM: performance (survives until reboot)"
echo ""
echo "To make permanent, add to /etc/default/grub GRUB_CMDLINE_LINUX:"
echo '  pcie_aspm=off nvidia.NVreg_EnablePCIeGen3=1'
echo "Then run: sudo update-grub && sudo reboot"
