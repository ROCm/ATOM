#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Monitor GPU memory usage during training and write to a CSV log.
# Works with AMD MI300X via rocm-smi.
#
# Usage:
#   bash monitor_gpu_memory.sh [interval_seconds] [output_csv]
#
# Defaults: interval=30s, output=./gpu_memory_lumen.csv

INTERVAL="${1:-30}"
OUTPUT="${2:-./gpu_memory_lumen.csv}"

echo "timestamp,gpu_id,vram_used_mb,vram_total_mb,utilization_pct" > "${OUTPUT}"

echo "[GPU Monitor] Writing to ${OUTPUT} every ${INTERVAL}s (Ctrl-C to stop)"

while true; do
    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    for gpu_id in 0 1 2 3 4 5 6 7; do
        USED=$(rocm-smi -d "${gpu_id}" --showmeminfo vram 2>/dev/null \
            | grep "Used Memory" | awk '{print $NF}')
        TOTAL=$(rocm-smi -d "${gpu_id}" --showmeminfo vram 2>/dev/null \
            | grep "Total Memory" | awk '{print $NF}')
        if [ -n "${USED}" ] && [ -n "${TOTAL}" ]; then
            USED_MB=$(echo "scale=2; ${USED}/1048576" | bc)
            TOTAL_MB=$(echo "scale=2; ${TOTAL}/1048576" | bc)
            UTIL=$(echo "scale=4; ${USED}*100/${TOTAL}" | bc)
            echo "${TS},${gpu_id},${USED_MB},${TOTAL_MB},${UTIL}" >> "${OUTPUT}"
        fi
    done
    sleep "${INTERVAL}"
done
