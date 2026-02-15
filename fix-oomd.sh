#!/bin/bash
# Raise systemd-oomd memory pressure threshold from 50% to 95%
# Default kills at 50% pressure for 20s — way too aggressive for a 1TB RAM machine
# running large language models that legitimately use 500+ GB.

set -euo pipefail

echo "=== Fixing systemd-oomd memory pressure limits ==="

# 1. Global oomd.conf — raise default threshold to 95%
echo "Setting global DefaultMemoryPressureLimit=95% in /etc/systemd/oomd.conf..."
sudo mkdir -p /etc/systemd/oomd.conf.d
sudo tee /etc/systemd/oomd.conf.d/raise-limit.conf > /dev/null <<'EOF'
[OOM]
DefaultMemoryPressureLimit=95%
DefaultMemoryPressureDurationSec=20s
EOF

# 2. Per-user-slice drop-in — override ManagedOOMMemoryPressure threshold
# This is the slice that actually got killed (user@1000.service)
echo "Adding drop-in for user@.service to raise ManagedOOMMemoryPressureLimit=95%..."
sudo mkdir -p /etc/systemd/system/user@.service.d
sudo tee /etc/systemd/system/user@.service.d/oom-limit.conf > /dev/null <<'EOF'
[Service]
ManagedOOMMemoryPressureLimit=95%
EOF

# 3. Reload systemd and restart oomd
echo "Reloading systemd and restarting systemd-oomd..."
sudo systemctl daemon-reload
sudo systemctl restart systemd-oomd

# 4. Verify
echo ""
echo "=== Verification ==="
echo "Global oomd.conf.d:"
cat /etc/systemd/oomd.conf.d/raise-limit.conf
echo ""
echo "User slice drop-in:"
cat /etc/systemd/system/user@.service.d/oom-limit.conf
echo ""
echo "systemd-oomd status:"
systemctl status systemd-oomd --no-pager | head -5
echo ""
echo "Done. OOM killer will now only trigger at 95% memory pressure for 20s+."
