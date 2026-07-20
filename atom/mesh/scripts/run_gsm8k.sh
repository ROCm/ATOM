#!/usr/bin/env bash
set -euo pipefail

# Run GSM8K accuracy evaluation against the mesh router endpoint.
# Run this script inside the container.
#
# Required env:
#   MODEL_PATH      - model path (used by lm_eval for tokenizer)
#
# Optional env (with defaults):
#   ROUTER_PORT=8000  LM_EVAL_LIMIT=100
#   LM_EVAL_TASK=gsm8k  LM_EVAL_NUM_FEWSHOT=3  LM_EVAL_NUM_CONCURRENT=65
#   RESULT_DIR=/workspace/gsm8k_results

: "${MODEL_PATH:?}"

ROUTER_PORT="${ROUTER_PORT:-8000}"
LM_EVAL_TASK="${LM_EVAL_TASK:-gsm8k}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-3}"
LM_EVAL_NUM_CONCURRENT="${LM_EVAL_NUM_CONCURRENT:-65}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"
RESULT_DIR="${RESULT_DIR:-/workspace/gsm8k_results}"

echo "[gsm8k] model=${MODEL_PATH} endpoint=http://127.0.0.1:${ROUTER_PORT}"
echo "[gsm8k] task=${LM_EVAL_TASK} fewshot=${LM_EVAL_NUM_FEWSHOT} concurrent=${LM_EVAL_NUM_CONCURRENT}"

if ! command -v lm_eval >/dev/null 2>&1; then
    echo '[gsm8k] installing lm-eval...'
    pip install 'lm-eval[api]'
fi

RUN_TAG="$(date +%Y%m%d%H%M%S)_gsm8k"
mkdir -p "${RESULT_DIR}"

echo '[gsm8k] running evaluation...'
lm_eval --model local-completions \
    --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/completions,num_concurrent=${LM_EVAL_NUM_CONCURRENT},max_retries=1,tokenized_requests=False,trust_remote_code=True" \
    --tasks "${LM_EVAL_TASK}" \
    --num_fewshot "${LM_EVAL_NUM_FEWSHOT}" \
    --limit "${LM_EVAL_LIMIT}" \
    --output_path "${RESULT_DIR}/${RUN_TAG}"

echo "[gsm8k] extracting results..."
python3 -c "
from pathlib import Path
import json

result_dir = Path('${RESULT_DIR}/${RUN_TAG}')
json_files = list(result_dir.rglob('*.json')) if result_dir.is_dir() else []
if not json_files:
    print('[gsm8k] ERROR: no result JSON found')
    exit(1)

result_file = max(json_files, key=lambda p: p.stat().st_mtime)
data = json.load(open(result_file))

score = data.get('results', {}).get('gsm8k', {}).get('exact_match,flexible-extract', 'N/A')
print(f'[gsm8k] exact_match,flexible-extract = {score}')
print(json.dumps(data.get('results', {}), indent=2))
"

echo "[gsm8k] results saved to ${RESULT_DIR}/${RUN_TAG}"
