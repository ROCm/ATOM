#!/bin/bash
# GSM8K 精度测试 (使用 lm_eval)
# 用法: bash run_gsm8k_eval.sh [MODEL_NAME] [PORT] [NUM_FEWSHOT]
#
# 示例:
#   bash run_gsm8k_eval.sh /data/DeepSeek-R1-0528 8000 5
#   bash run_gsm8k_eval.sh meta-llama/Meta-Llama-3-8B 8000 5

set -euo pipefail

MODEL="${1:-/data/DeepSeek-R1-0528}"
PORT="${2:-8000}"
NUM_FEWSHOT="${3:-5}"
NUM_CONCURRENT="${NUM_CONCURRENT:-65}"
LIMIT="${LIMIT:-}"  # set LIMIT=50 to run only first 50 samples
BASE_URL="http://localhost:${PORT}/v1/completions"
OUTPUT_DIR="/app/logs_claude"
LOG_FILE="${OUTPUT_DIR}/gsm8k_eval.log"

{
echo "========================================"
echo " GSM8K Accuracy Evaluation (lm_eval)"
echo "========================================"
echo " Model:          $MODEL"
echo " Base URL:       $BASE_URL"
echo " Few-shot:       $NUM_FEWSHOT"
echo " Concurrency:    $NUM_CONCURRENT"
echo "========================================"
} > "$LOG_FILE"

# 等待服务就绪
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Server not ready after 120 retries (20 min)"
        exit 1
    fi
    echo "  Waiting... ($i/120)"
    sleep 10
done

LIMIT_ARG=()
if [ -n "$LIMIT" ]; then
    LIMIT_ARG=(--limit "$LIMIT")
fi

lm_eval --model local-completions \
    --model_args "model=${MODEL},base_url=${BASE_URL},num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot "$NUM_FEWSHOT" \
    "${LIMIT_ARG[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
