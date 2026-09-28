#!/bin/bash
# Performance benchmark for ATOM serving
# Usage: bash run_benchmark.sh [MODEL] [PORT] [ISL] [OSL] [CONCURRENCY] [PROMPT_MULTIPLIER] [PROFILE]
#
# Examples:
#   bash run_benchmark.sh /data/DeepSeek-R1-0528 8000 1024 1024 128          # normal
#   bash run_benchmark.sh /data/DeepSeek-R1-0528 8000 1024 1024 64 2 1       # profile trace (conc*1 requests)
#   bash run_benchmark.sh /data/DeepSeek-R1-0528 8000 1024 8192 32

set -euo pipefail

MODEL="${1:-/data/DeepSeek-R1-0528}"
PORT="${2:-8000}"
ISL="${3:-1024}"
OSL="${4:-1024}"
CONCURRENCY="${5:-128}"
PROMPT_MULTIPLIER="${6:-10}"
PROFILE="${7:-0}"
_n=$(( $# < 7 ? $# : 7 ))
shift "$_n" 2>/dev/null || true
EXTRA_ARGS="$*"
NUM_PROMPTS=$(( CONCURRENCY * PROMPT_MULTIPLIER ))
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
OUTPUT_DIR="/app/logs_claude"

echo "========================================"
echo " ATOM Performance Benchmark"
echo "========================================"
echo " Model:          $MODEL"
echo " Base URL:       http://localhost:${PORT}"
echo " ISL/OSL:        ${ISL}/${OSL}"
echo " Concurrency:    $CONCURRENCY"
echo " Num Prompts:    $NUM_PROMPTS (x${PROMPT_MULTIPLIER})"
echo " Range Ratio:    $RANDOM_RANGE_RATIO"
echo " Profile:        $PROFILE"
echo " Extra args:     ${EXTRA_ARGS:-none}"
echo "========================================"

# Wait for server ready (dual check: HTTP + GPU VRAM)
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        VRAM_COUNT=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l)
        if [ "$VRAM_COUNT" -gt 0 ]; then
            echo "Server is ready! (GPU VRAM loaded on $VRAM_COUNT GPUs)"
            break
        fi
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Server not ready after 120s"
        exit 1
    fi
    [ $((i % 10)) -eq 0 ] && echo "  Waiting... (${i}s)"
    sleep 1
done

LOG_FILE="${OUTPUT_DIR}/benchmark.log"
RESULT_FILE="${OUTPUT_DIR}/benchmark.json"

# Write config header to log
{
echo "========================================"
echo " ATOM Performance Benchmark"
echo "========================================"
echo " Model:          $MODEL"
echo " Base URL:       http://localhost:${PORT}"
echo " ISL/OSL:        ${ISL}/${OSL}"
echo " Concurrency:    $CONCURRENCY"
echo " Num Prompts:    $NUM_PROMPTS (x${PROMPT_MULTIPLIER})"
echo " Range Ratio:    $RANDOM_RANGE_RATIO"
echo " Profile:        $PROFILE"
echo " Extra args:     ${EXTRA_ARGS:-none}"
echo " Date:           $(date)"
echo "========================================"
} > "$LOG_FILE"

python -m atom.benchmarks.benchmark_serving \
    --model="$MODEL" --backend=vllm --base-url="http://localhost:${PORT}" \
    --dataset-name=random \
    --random-input-len="$ISL" --random-output-len="$OSL" \
    --random-range-ratio="$RANDOM_RANGE_RATIO" \
    --max-concurrency="$CONCURRENCY" \
    --num-prompts="$NUM_PROMPTS" \
    --trust-remote-code \
    --num-warmups=$((CONCURRENCY * 2)) \
    --request-rate=inf --ignore-eos \
    --save-result \
    --result-filename="$RESULT_FILE" \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    $([ "$PROFILE" = "1" ] && echo "--profile") \
    ${EXTRA_ARGS:-} \
    2>&1 | tee -a "$LOG_FILE"
