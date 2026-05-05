#!/usr/bin/env bash
# Collect GPU info from a (running) container or the local host and emit
# `gpu_name`, `gpu_vram_gb`, and `rocm_version` to $GITHUB_OUTPUT (when set).
#
# Prefers `amd-smi` because it correctly identifies recent ASICs (e.g. MI355X)
# whose marketing name is missing from `rocm-smi`'s product table. Falls back
# to `rocm-smi` and `rocminfo` for older container images that do not yet ship
# `amd-smi`.
#
# Usage:
#   collect_gpu_info.sh                              # run directly on the host
#   collect_gpu_info.sh <container>                  # docker exec <container>
#   collect_gpu_info.sh <container> <engine>         # custom engine, e.g. podman

set -uo pipefail

CONTAINER="${1:-}"
ENGINE="${2:-docker}"

if [ -n "$CONTAINER" ]; then
    exec_in() { "$ENGINE" exec "$CONTAINER" bash -lc "$1" 2>/dev/null; }
else
    exec_in() { bash -lc "$1" 2>/dev/null; }
fi

trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

# --- GPU name ---------------------------------------------------------------
# 1) amd-smi (preferred; covers MI300X / MI325X / MI355X+).
GPU_NAME=$(exec_in 'command -v amd-smi >/dev/null 2>&1 && amd-smi static -g 0 --asic 2>/dev/null' \
    | awk -F: '/MARKET_NAME/ {sub(/^[ \t]+/, "", $2); print $2; exit}' | trim)

# 2) rocm-smi (legacy product-name table).
if [ -z "${GPU_NAME:-}" ] || echo "$GPU_NAME" | grep -qi "Radeon Graphics"; then
    GPU_NAME=$(exec_in 'rocm-smi --showproductname' \
        | grep -i "Card Series" | head -1 | sed 's/.*:\s*//' | trim)
fi

# 3) rocminfo Marketing Name (last resort; matches device tree).
if [ -z "${GPU_NAME:-}" ] || echo "$GPU_NAME" | grep -qi "Radeon Graphics"; then
    GPU_NAME=$(exec_in 'rocminfo' \
        | grep -A1 "Uuid:.*GPU-" | grep "Marketing Name" | head -1 \
        | sed 's/.*:\s*//' | trim)
fi
GPU_NAME="${GPU_NAME:-unknown}"

# --- VRAM (GB) --------------------------------------------------------------
# 1) amd-smi via JSON (schema-tolerant).
GPU_VRAM_GB=$(exec_in 'command -v amd-smi >/dev/null 2>&1 && amd-smi static -g 0 --vram --json 2>/dev/null' \
    | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    entry = d[0] if isinstance(d, list) else d
    vram = entry.get("vram", entry)
    size = vram.get("size", vram.get("vram_size"))
    if isinstance(size, dict):
        value = size.get("value", 0)
        unit = (size.get("unit") or "MB").upper()
    else:
        value = size if size is not None else 0
        unit = (vram.get("size_unit") or "MB").upper()
    factor = {"B": 1.0/1024**3, "KB": 1.0/1024**2, "MB": 1.0/1024,
              "GB": 1.0, "TB": 1024.0}.get(unit, 1.0/1024)
    print(int(round(float(value) * factor)))
except Exception:
    pass
' 2>/dev/null)

# 2) rocm-smi (--showmeminfo vram reports bytes after the colon).
if [ -z "${GPU_VRAM_GB:-}" ] || [ "${GPU_VRAM_GB:-0}" = "0" ]; then
    GPU_VRAM_GB=$(exec_in 'rocm-smi --showmeminfo vram' \
        | grep -i "VRAM Total Memory" | head -1 \
        | awk -F: '{printf "%.0f", $NF/(1024*1024*1024)}')
fi
GPU_VRAM_GB="${GPU_VRAM_GB:-0}"

# --- ROCm version -----------------------------------------------------------
ROCM_VERSION=$(exec_in 'cat /opt/rocm/.info/version' | trim)
ROCM_VERSION="${ROCM_VERSION:-unknown}"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
    {
        echo "gpu_name=${GPU_NAME}"
        echo "gpu_vram_gb=${GPU_VRAM_GB}"
        echo "rocm_version=${ROCM_VERSION}"
    } >> "$GITHUB_OUTPUT"
fi

echo "GPU: ${GPU_NAME}, VRAM: ${GPU_VRAM_GB}GB, ROCm: ${ROCM_VERSION}"
