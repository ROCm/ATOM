#!/bin/bash
# Start ATOM OpenAI-compatible server
# Usage: bash start_atom_server.sh [MODEL_PATH] [TP_SIZE] [PORT] [EXTRA_ARGS...]
#
# Examples:
#   bash start_atom_server.sh                                    # DeepSeek-R1-0528, tp=8, port=8000
#   bash start_atom_server.sh /data/Llama-3.1-8B-Instruct-FP8-KV 1 8000
#   bash start_atom_server.sh /data/DeepSeek-R1-0528 8 8000 --method mtp --num-speculative-tokens 3

set -euo pipefail

MODEL_PATH="${1:-/data/DeepSeek-R1-0528}"
TP_SIZE="${2:-8}"
PORT="${3:-8000}"
shift 3 2>/dev/null || true
EXTRA_ARGS="$*"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
LOG_FILE="/app/logs_claude/atom_server.log"

export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-INFO}"
export KINETO_CONFIG="/home/ljin1/dk/libkineto.conf"

# === Pre-flight: ensure GPU is clean ===
echo "Pre-flight: cleaning up processes and GPU memory..."

# 1. Kill atom server processes
pkill -f 'atom.entrypoints' 2>/dev/null || true
sleep 2

# 2. Kill orphaned multiprocessing spawn/tracker (these hold GPU memory after server dies)
pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
sleep 3

# 3. Verify GPU memory is actually free
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    USED_GPUS=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l)
    if [ "$USED_GPUS" -eq 0 ]; then
        echo "GPU memory clear after ${i}s"
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "WARNING: GPU memory still in use after ${MAX_WAIT}s. Dumping GPU process info:"
        rocm-smi --showpidgpus 2>&1 | grep "PID.*is using" | grep -v "0 DRM" || true
        echo "Attempting force kill of GPU-holding processes..."
        rocm-smi --showpidgpus 2>&1 | grep -oP 'PID \K\d+' | while read pid; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 5
    fi
    sleep 1
done

# 4. Clear stale compile cache
rm -rf ~/.cache/atom/*
rm -rf ./gpucore.*

# Disable core dumps: a single ROCm fault dumps a 30-50 GB gpucore PER RANK,
# which on an 8-GPU TP run fills the disk in seconds (and triggers the
# apport "execvp failed" noise). The debug-agent wrapper already does this;
# the normal launcher must too, since faults can happen in production runs.
ulimit -c 0

# Write config header to log (truncates old content).
# Inherited env vars are dumped explicitly so you never have to wonder
# whether ATOM_USE_TRITON_MOE / V4_USE_REF_QUANT / etc. were set.
{
echo "========================================"
echo " ATOM Server Launcher"
echo "========================================"
echo " Model:          $MODEL_PATH"
echo " TP Size:        $TP_SIZE"
echo " Port:           $PORT"
echo " KV Cache dtype: $KV_CACHE_DTYPE"
echo " Max num seqs:   $MAX_NUM_SEQS"
echo " GPU mem util:   $GPU_MEM_UTIL"
echo " Extra args:     ${EXTRA_ARGS:-none}"
echo " Date:           $(date)"
echo "----------------------------------------"
echo " Inherited env vars (ATOM_*, V4_*, AITER_*, HSA_*, AMD_*, HIP_*):"
env | grep -E '^(ATOM_|V4_|AITER_|HSA_|AMD_|HIP_|KV_CACHE|MAX_NUM_SEQS|MAX_MODEL_LEN|MAX_BATCHED_TOKENS|GPU_MEM_UTIL)' \
  | sort | sed 's/^/   /' || echo "   (none set)"
echo "========================================"
} | tee "$LOG_FILE"

python -m atom.entrypoints.openai_server \
    --model "$MODEL_PATH" \
    --kv_cache_dtype "$KV_CACHE_DTYPE" \
    -tp "$TP_SIZE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --server-port "$PORT" \
    $EXTRA_ARGS \
    >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server started in background (PID: $SERVER_PID)"

# Wait for server ready with GPU verification
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        VRAM_COUNT=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l)
        if [ "$VRAM_COUNT" -gt 0 ]; then
            echo "Server is ready! (PID: $SERVER_PID, GPU VRAM loaded)"
            exit 0
        fi
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died. Check $LOG_FILE"
        exit 1
    fi
    if [ "$i" -eq 600 ]; then
        echo "ERROR: Server not ready after 600s (10 min)"
        exit 1
    fi
    sleep 1
done
