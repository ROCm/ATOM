#!/bin/bash
set -euo pipefail

# Usage:
#   .github/scripts/atom_oot_test.sh launch <mode> [model_name]
#   .github/scripts/atom_oot_test.sh accuracy <mode> [model_name]
#
# TYPE:
#   launch   - launch vLLM server and wait until ready
#   accuracy - run gsm8k accuracy test (and threshold check)
#
# MODE:
#   ci    - only Kimi-K2
#   full  - all OOT-supported models
#
# Optional model_name can be used to run a single model in full mode.

TYPE=${1:-launch}
MODE=${2:-ci}
SELECTED_MODEL=${3:-}

if [[ "$TYPE" != "launch" && "$TYPE" != "accuracy" ]]; then
  echo "Invalid TYPE: $TYPE. Expected: launch or accuracy"
  exit 2
fi

if [[ "$MODE" != "ci" && "$MODE" != "full" ]]; then
  echo "Invalid MODE: $MODE. Expected: ci or full"
  exit 2
fi

MAX_WAIT_RETRIES=${MAX_WAIT_RETRIES:-60}
WAIT_INTERVAL_SEC=${WAIT_INTERVAL_SEC:-30}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PID_FILE=${VLLM_PID_FILE:-/tmp/vllm_oot.pid}
VLLM_LOG_FILE=${VLLM_LOG_FILE:-/tmp/vllm_oot.log}
RESULT_DIR=${RESULT_DIR:-/tmp/oot_accuracy_results}
ACCURACY_LOG_FILE=${ACCURACY_LOG_FILE:-/tmp/oot_accuracy_output.txt}

# Format:
#   MODEL_NAME|MODEL_PATH|EXTRA_ARGS|THRESHOLD
# Note: CI runs Kimi-K2 with TP=4 on an 8-GPU runner to reduce runtime and
# improve CI stability. Full mode uses TP=8 on the same class of runner for
# higher-fidelity validation.
CI_MODE_MODELS=(
  "Kimi-K2|amd/Kimi-K2-Thinking-MXFP4|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 4 --enable-expert-parallel|0.90"
)

FULL_MODE_MODELS=(
  "Qwen3 Dense|Qwen/Qwen3-8B|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 1|0.70"
  "Qwen3 MoE|Qwen/Qwen3-235B-A22B-Instruct-2507-FP8|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 8 --enable-expert-parallel|0.87"
  "DeepSeek-V3 family|deepseek-ai/DeepSeek-R1-0528|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 8|0.94"
  "GPT-OSS|openai/gpt-oss-120b|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 2 --enable-dp-attention --enable-expert-parallel --gpu-memory-utilization 0.3|0.38"
  "Kimi-K2|amd/Kimi-K2-Thinking-MXFP4|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 8 --enable-expert-parallel|0.90"
)

declare -a ACTIVE_MODELS=()
if [[ "$MODE" == "ci" ]]; then
  ACTIVE_MODELS=("${CI_MODE_MODELS[@]}")
else
  ACTIVE_MODELS=("${FULL_MODE_MODELS[@]}")
fi

resolve_model_path() {
  local model_path="$1"
  if [[ -f "/models/${model_path}/config.json" ]]; then
    echo "/models/${model_path}"
  else
    echo "${model_path}"
  fi
}

wait_server_ready() {
  local model_name="$1"
  echo ""
  echo "========== Waiting for vLLM server (${model_name}) =========="
  for ((i=1; i<=MAX_WAIT_RETRIES; i++)); do
    if curl -sS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
      echo "vLLM server is ready for ${model_name}."
      return 0
    fi

    if [[ -f "${VLLM_PID_FILE}" ]]; then
      local pid
      pid=$(cat "${VLLM_PID_FILE}")
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "vLLM process exited early for ${model_name}."
        tail -n 200 "${VLLM_LOG_FILE}" || true
        return 1
      fi
    fi

    echo "Waiting for vLLM server... (${i}/${MAX_WAIT_RETRIES})"
    sleep "${WAIT_INTERVAL_SEC}"
  done

  echo "vLLM server did not become ready in time for ${model_name}."
  tail -n 200 "${VLLM_LOG_FILE}" || true
  return 1
}

stop_server() {
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid
    pid=$(cat "${VLLM_PID_FILE}")
    kill "${pid}" 2>/dev/null || true
    rm -f "${VLLM_PID_FILE}" || true
  fi
}

launch_one_model() {
  local model_name="$1"
  local model_path="$2"
  local extra_args="$3"

  local resolved_model_path
  resolved_model_path=$(resolve_model_path "${model_path}")

  echo ""
  echo "========== Launching vLLM server =========="
  echo "Model name: ${model_name}"
  echo "Model path: ${resolved_model_path}"
  echo "Extra args: ${extra_args}"

  export SAFETENSORS_FAST_GPU=1
  export VLLM_ROCM_USE_AITER=1
  export VLLM_RPC_TIMEOUT=1800000
  export VLLM_CACHE_ROOT=/tmp/.cache/vllm
  export TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/inductor
  rm -rf /tmp/.cache

  rm -f "${VLLM_PID_FILE}" || true

  nohup vllm serve "${resolved_model_path}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --async-scheduling \
    --load-format fastsafetensors \
    --max-model-len 16384 \
    ${extra_args} \
    > "${VLLM_LOG_FILE}" 2>&1 &
  echo $! > "${VLLM_PID_FILE}"
  echo "Server PID: $(cat "${VLLM_PID_FILE}")"

  wait_server_ready "${model_name}"
}

accuracy_one_model() {
  local model_name="$1"
  local model_path="$2"
  local extra_args="$3"
  local threshold="$4"

  local resolved_model_path
  resolved_model_path=$(resolve_model_path "${model_path}")

  if ! command -v lm_eval >/dev/null 2>&1; then
    echo "========== Installing lm-eval =========="
    pip install 'lm-eval[api]'
  fi

  mkdir -p "${RESULT_DIR}"
  local result_file="${RESULT_DIR}/$(date +%Y%m%d%H%M%S)_${model_name// /_}.json"

  echo ""
  echo "========== Running OOT gsm8k accuracy =========="
  echo "Model name: ${model_name}"
  echo "Threshold: ${threshold}"

  lm_eval --model local-completions \
    --model_args model="${resolved_model_path}",base_url="http://127.0.0.1:${VLLM_PORT}/v1/completions",num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True \
    --tasks gsm8k \
    --num_fewshot 3 \
    --output_path "${result_file}" 2>&1 | tee -a "${ACCURACY_LOG_FILE}"

  local value
  value=$(python - <<PY
import json
with open("${result_file}", "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["results"]["gsm8k"]["exact_match,flexible-extract"])
PY
)

  echo "Result file: ${result_file}"
  echo "Flexible extract value: ${value}"
  echo "Accuracy threshold: ${threshold}"

  python - <<PY
value = float("${value}")
threshold = float("${threshold}")
assert value >= threshold, f"Accuracy failed: {value} < {threshold}"
print(f"Accuracy passed: {value} >= {threshold}")
PY
}

run_for_models() {
  local action="$1"
  local matched=0

  for entry in "${ACTIVE_MODELS[@]}"; do
    IFS='|' read -r model_name model_path extra_args threshold <<< "${entry}"

    if [[ -n "${SELECTED_MODEL}" && "${SELECTED_MODEL}" != "${model_name}" ]]; then
      continue
    fi
    matched=1

    if [[ "${action}" == "launch" ]]; then
      launch_one_model "${model_name}" "${model_path}" "${extra_args}"
      break
    fi

    # accuracy mode: launch + evaluate each selected model, then stop server.
    launch_one_model "${model_name}" "${model_path}" "${extra_args}"
    accuracy_one_model "${model_name}" "${model_path}" "${extra_args}" "${threshold}"
    stop_server
  done

  if [[ "${matched}" -eq 0 ]]; then
    echo "No model matched MODE=${MODE}, SELECTED_MODEL=${SELECTED_MODEL}"
    exit 2
  fi
}

trap 'stop_server' EXIT

if [[ "${TYPE}" == "launch" ]]; then
  run_for_models "launch"
else
  run_for_models "accuracy"
fi

