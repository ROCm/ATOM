#!/bin/bash
# Run ATOM offline simple_inference for any registered model.
# Usage: bash start_simple_inference.sh [MODEL_PATH] [TP_SIZE] [EXTRA_ARGS...]
#
# All engine flags (kv_cache_dtype, max_num_seqs, max_model_len, gpu mem util,
# spec method, …) go in EXTRA_ARGS as native CLI args — no env-var translation.
# Only LOG_FILE and true env vars (ATOM_*, AITER_*, HSA_*, HIP_*) remain
# overridable via env.
#
# Examples:
#   bash start_simple_inference.sh                                     # default DeepSeek-R1-0528 tp=8
#   bash start_simple_inference.sh /data/Llama-3.1-8B-Instruct-FP8-KV 1 --kv_cache_dtype fp8
#   bash start_simple_inference.sh /data/DeepSeek-V4-Pro 8 --kv_cache_dtype fp8 --enforce-eager --level 0
#   bash start_simple_inference.sh /data/DeepSeek-V4-Pro 8 \
#       --kv_cache_dtype fp8 --enforce-eager --level 0 \
#       --method mtp --num-speculative-tokens 3
#   MAX_MODEL_LEN=1024 ...  # ← no longer supported; pass --max-model-len 1024 instead
#
# DeepSeek-V4 must also export ATOM_USE_TRITON_MOE=1 (real env var, not a CLI flag).

set -euo pipefail

MODEL_PATH="${1:-/data/DeepSeek-R1-0528}"
TP_SIZE="${2:-8}"
shift 2 2>/dev/null || true
EXTRA_ARGS=("$@")

LOG_FILE="${LOG_FILE:-/app/logs_claude/simple_inference.log}"

export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-INFO}"

# === Pre-flight: ensure GPU is clean ===
echo "Pre-flight: cleaning up processes and GPU memory..."

pkill -f 'atom.entrypoints' 2>/dev/null || true
pkill -f 'atom.examples.simple_inference' 2>/dev/null || true
sleep 2

pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
sleep 3

# Verify GPU memory is actually free
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    USED_GPUS=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l)
    if [ "$USED_GPUS" -eq 0 ]; then
        echo "  All GPUs free."
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "  WARN: $USED_GPUS GPU(s) still showing memory after $MAX_WAIT s — proceeding anyway"
        break
    fi
    sleep 1
done

# Clear ATOM compile cache (stale cache after code change causes silent fails)
rm -rf /root/.cache/atom/* 2>/dev/null || true

# Clear ROCm core dumps (each ~85GB; HSA exceptions can fill the disk fast).
# Find from CWD + a few common roots — gpucore.PID files dropped in process CWD.
GPUCORE_DELETED=0
for d in . /app/ATOM /app /root /tmp /app/logs_claude; do
    [ -d "$d" ] || continue
    for f in "$d"/gpucore.* "$d"/core.[0-9]*; do
        if [ -e "$f" ]; then
            sz=$(du -BG --apparent-size "$f" 2>/dev/null | awk '{print $1}')
            rm -f "$f" 2>/dev/null && GPUCORE_DELETED=$((GPUCORE_DELETED+1))
            echo "  removed $f ($sz)"
        fi
    done
done
[ "$GPUCORE_DELETED" -gt 0 ] && echo "  Cleared $GPUCORE_DELETED ROCm core dump(s)"

# Disk-space guard — refuse to launch if root fs has < 50 GB free
# (each crashed worker can dump 85+GB; better to fail-fast than fill disk again).
AVAIL_GB=$(df -BG / 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}')
if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt 50 ]; then
    echo "ERROR: only ${AVAIL_GB}GB free on / — refusing to launch (would fill disk on crash)."
    echo "  Run: rm -f /app/ATOM/gpucore.* /app/logs_claude/**/gpucore.*"
    exit 2
fi

# Disable ROCm core dumps for this run (HSA crashes won't write 85GB files).
# Set HSA_ENABLE_COREDUMP_FILTER to 0x0 to skip all core regions.
export HSA_ENABLE_COREDUMP=0
export AMD_LOG_LEVEL=0  # quiet HSA error spam

# === Header to log ===
# Inherited env vars are dumped explicitly so missing ATOM_USE_TRITON_MOE etc.
# is visible at a glance instead of producing silent accuracy regressions.
{
echo "========================================"
echo " ATOM simple_inference Launcher"
echo "========================================"
echo " Model:           $MODEL_PATH"
echo " TP Size:         $TP_SIZE"
echo " Extra args:      ${EXTRA_ARGS[*]:-none}"
echo " Date:            $(date)"
echo "----------------------------------------"
echo " Inherited env vars (ATOM_*, V4_*, AITER_*, HSA_*, AMD_*, HIP_*):"
env | grep -E '^(ATOM_|V4_|AITER_|HSA_|AMD_|HIP_)' \
  | sort | sed 's/^/   /' || echo "   (none set)"
echo "========================================"
} | tee "$LOG_FILE"

set +e
python -m atom.examples.simple_inference \
    --model "$MODEL_PATH" \
    -tp "$TP_SIZE" \
    "${EXTRA_ARGS[@]}" \
    >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
set -e

echo ""
echo "========================================"
echo " Exit code: $EXIT_CODE"
echo " Log: $LOG_FILE"
echo "========================================"

# Show first error if any
if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "First error in log:"
    grep -nE "RuntimeError|AttributeError|TypeError|ValueError|^\[rank0\]:" "$LOG_FILE" \
      | grep -v "AsyncIOProc\|ModelRunner.*has no attribute" \
      | head -5
fi

exit $EXIT_CODE
