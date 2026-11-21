#!/bin/bash
# Runtime OS/system tunables matching MLPerf MI300X reference submission.
# Run as root/sudo BEFORE launching training to reduce system jitter.

set -euo pipefail

echo "[runtime_tunables] Applying MLPerf-aligned system tuning..."

# Drop page caches to start with a clean memory slate
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true

# Disable CPU idle states deeper than C1 (reduces wakeup latency)
sudo cpupower idle-set -d 2 > /dev/null 2>&1 || true

# Set CPU governor to performance (max frequency, no scaling)
sudo cpupower frequency-set -g performance > /dev/null 2>&1 || true

# Disable NMI watchdog (reduces periodic interrupts)
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog > /dev/null 2>&1 || true

# Disable NUMA balancing (prevents page migration during training)
echo 0 | sudo tee /proc/sys/kernel/numa_balancing > /dev/null 2>&1 || true

# Disable ASLR (deterministic memory layout)
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space > /dev/null 2>&1 || true

# Enable transparent huge pages (reduces TLB misses for large allocations)
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null 2>&1 || true
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/defrag > /dev/null 2>&1 || true

echo "[runtime_tunables] Done."
