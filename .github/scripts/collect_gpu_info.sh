#!/usr/bin/env bash
# Collect GPU info from a (running) container or the local host and emit
# `gpu_name`, `gpu_vram_gb`, and `rocm_version` to $GITHUB_OUTPUT (when set).
#
# Probing order for the GPU marketing name:
#   1. `amd-smi static --asic` MARKET_NAME
#   2. `rocm-smi --showproductname` Card Series
#   3. `rocminfo` Marketing Name
#   4. <runner_hint> pattern match (mi355 / mi35x / mi325 / mi308 / mi300 / mi250)
#
# Step 4 is needed because on freshly-released ASICs (currently MI355X) every
# in-container SMI tool can still report "Radeon Graphics" until the
# marketing-name table is patched. The CI runner name is operator-asserted
# and reliable, so we use it as the final tie-breaker for the dashboard label.
#
# Note: `mi35x` is the family-style label used by accuracy-validation runners
# (`linux-atom-mi35x-1` / `-4` / `-8`). The MI35 series in ATOM CI currently
# only contains MI355X, so we map both `mi355` and `mi35x` to MI355X.
#
# Usage:
#   collect_gpu_info.sh                                          # local host
#   collect_gpu_info.sh <container>                              # docker exec
#   collect_gpu_info.sh <container> <engine>                     # custom engine
#   collect_gpu_info.sh <container> <engine> <runner_hint>       # + runner hint

set -uo pipefail

CONTAINER="${1:-}"
ENGINE="${2:-docker}"
RUNNER_HINT="${3:-${RUNNER_HINT:-}}"
COLLECT_VERBOSE_GPU_INFO="${COLLECT_VERBOSE_GPU_INFO:-0}"

if [ -n "$CONTAINER" ]; then
    exec_in() { "$ENGINE" exec "$CONTAINER" bash -lc "$1" 2>/dev/null; }
    exec_in_verbose() { "$ENGINE" exec "$CONTAINER" bash -lc "$1"; }
else
    exec_in() { bash -lc "$1" 2>/dev/null; }
    exec_in_verbose() { bash -lc "$1"; }
fi

trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

print_probe() {
    local title="$1"
    local command="$2"

    echo ""
    echo "========== ${title} =========="
    if ! exec_in_verbose "$command"; then
        echo "WARNING: ${title} failed"
    fi
}

# --- GPU name ---------------------------------------------------------------
# 1) amd-smi (preferred; covers MI300X / MI325X / MI355X+).
GPU_NAME=$(exec_in 'command -v amd-smi >/dev/null 2>&1 && amd-smi static -g 0 --asic 2>/dev/null' \
    | awk -F: '/MARKET_NAME/ {sub(/^[ \t]+/, "", $2); print $2; exit}' | trim)

# 2) rocm-smi (legacy product-name table).
if [ -z "${GPU_NAME:-}" ] || echo "$GPU_NAME" | grep -qi "Radeon Graphics"; then
    GPU_NAME=$(exec_in 'rocm-smi --showproductname' \
        | grep -i "Card Series" | head -1 | sed 's/.*:\s*//' | trim)
fi

# 3) rocminfo Marketing Name.
if [ -z "${GPU_NAME:-}" ] || echo "$GPU_NAME" | grep -qi "Radeon Graphics"; then
    GPU_NAME=$(exec_in 'rocminfo' \
        | grep -A1 "Uuid:.*GPU-" | grep "Marketing Name" | head -1 \
        | sed 's/.*:\s*//' | trim)
fi

# 4) Runner-name hint (last resort: every in-container SMI tool can still
#    return "Radeon Graphics" on freshly-released ASICs until the marketing
#    table is patched. The CI runner name encodes the chip family.)
if { [ -z "${GPU_NAME:-}" ] || echo "$GPU_NAME" | grep -qi "Radeon Graphics"; } \
    && [ -n "${RUNNER_HINT:-}" ]; then
    hint_lc=$(echo "$RUNNER_HINT" | tr '[:upper:]' '[:lower:]')
    case "$hint_lc" in
        *mi355*|*mi35x*) GPU_NAME="AMD Instinct MI355X" ;;
        *mi325*) GPU_NAME="AMD Instinct MI325X" ;;
        *mi308*) GPU_NAME="AMD Instinct MI308X" ;;
        *mi300x*|*mi300*) GPU_NAME="AMD Instinct MI300X" ;;
        *mi250x*|*mi250*) GPU_NAME="AMD Instinct MI250X" ;;
        *mi210*) GPU_NAME="AMD Instinct MI210" ;;
    esac
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

if [ "$COLLECT_VERBOSE_GPU_INFO" = "1" ]; then
    print_probe "ROCm version files" '
        set +e
        cat /opt/rocm/.info/version 2>/dev/null || true
        cat /opt/rocm/.info/version-dev 2>/dev/null || true
        cat /opt/rocm/.info/version-utils 2>/dev/null || true
    '

    print_probe "PyTorch HIP device mapping" '
        python3 - <<'"'"'PY'"'"'
import os

keys = [
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
]
for key in keys:
    print(f"{key}={os.environ.get(key)}")

try:
    import torch
except Exception as exc:
    print(f"torch import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

print(f"torch.version.hip={getattr(torch.version, '"'"'hip'"'"', None)}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
try:
    count = torch.cuda.device_count()
    print(f"torch.cuda.device_count={count}")
    for index in range(count):
        print(f"device[{index}]={torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch cuda probe failed: {type(exc).__name__}: {exc}")
PY
    '

    print_probe "rocminfo" '
        command -v rocminfo >/dev/null 2>&1 || { echo "rocminfo not found"; exit 127; }
        rocminfo
    '

    print_probe "rocm-smi" '
        command -v rocm-smi >/dev/null 2>&1 || { echo "rocm-smi not found"; exit 127; }
        rocm-smi
    '

    print_probe "rocm-smi product and driver details" '
        command -v rocm-smi >/dev/null 2>&1 || { echo "rocm-smi not found"; exit 127; }
        rocm-smi --showproductname || true
        rocm-smi --showdriverversion || true
        rocm-smi --showvbios || true
    '

    print_probe "rocm-smi topology" '
        command -v rocm-smi >/dev/null 2>&1 || { echo "rocm-smi not found"; exit 127; }
        rocm-smi --showtopo || true
    '

    print_probe "rocm-smi memory and processes" '
        command -v rocm-smi >/dev/null 2>&1 || { echo "rocm-smi not found"; exit 127; }
        rocm-smi --showmemuse || true
        rocm-smi --showpids || true
        rocm-smi --showpidgpus || true
    '

    print_probe "ROCm device nodes" '
        ls -l /dev/kfd /dev/dri 2>/dev/null || true
    '
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
    {
        echo "gpu_name=${GPU_NAME}"
        echo "gpu_vram_gb=${GPU_VRAM_GB}"
        echo "rocm_version=${ROCM_VERSION}"
    } >> "$GITHUB_OUTPUT"
fi

echo "GPU: ${GPU_NAME}, VRAM: ${GPU_VRAM_GB}GB, ROCm: ${ROCM_VERSION}"
