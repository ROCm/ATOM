#!/bin/bash
set -euo pipefail

ROUTER_URL="${ROUTER_URL:-http://10.24.112.168:8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/DeepSeek-V4-Pro/}"

pip install 'lm-eval[api]' 2>/dev/null || true

lm_eval --model local-completions \
    --model_args "model=${MODEL_PATH},base_url=${ROUTER_URL}/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 3 \
    --output_path /workspace/gsm8k_results
