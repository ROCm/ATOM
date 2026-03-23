#!/bin/bash
set -euo pipefail

# Usage:
#   .github/scripts/atom_oot_test.sh launch <mode> [model_name]
#   .github/scripts/atom_oot_test.sh accuracy <mode> [model_name]
#
# Alternatively, pass a single model explicitly through environment variables:
#   OOT_MODEL_NAME
#   OOT_MODEL_PATH
#   OOT_EXTRA_ARGS
#
# TYPE:
#   launch   - launch vLLM server and wait until ready
#   accuracy - run gsm8k accuracy test and save result JSON
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
STREAM_VLLM_LOGS=${STREAM_VLLM_LOGS:-1}
KEEP_SERVER_ALIVE_ON_EXIT=${KEEP_SERVER_ALIVE_ON_EXIT:-0}
EXPLICIT_MODEL_NAME=${OOT_MODEL_NAME:-}
EXPLICIT_MODEL_PATH=${OOT_MODEL_PATH:-}
EXPLICIT_EXTRA_ARGS=${OOT_EXTRA_ARGS:-}
LAST_VLLM_LOG_LINE=0

# Default model format:
#   MODEL_NAME|MODEL_PATH|EXTRA_ARGS
# Default lists are kept for ad hoc/manual runs. CI workflows pass the exact
# current matrix entry via OOT_MODEL_* env vars so model coverage does not drift
# when the workflow matrix changes.
CI_MODE_MODELS=(
  "Kimi-K2-Thinking-MXFP4|amd/Kimi-K2-Thinking-MXFP4|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 4 --enable-expert-parallel"
)

FULL_MODE_MODELS=(
  "Qwen3 MoE|Qwen/Qwen3-235B-A22B-Instruct-2507-FP8|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 8 --enable-expert-parallel"
  "DeepSeek-R1 FP8|deepseek-ai/DeepSeek-R1-0528|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 8"
  "DeepSeek-R1 MXFP4|amd/DeepSeek-R1-0528-MXFP4|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 8"
  "GPT-OSS|openai/gpt-oss-120b|--trust-remote-code --kv-cache-dtype fp8 --gpu-memory-utilization 0.3"
  "Kimi-K2|amd/Kimi-K2-Thinking-MXFP4|--trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 8 --enable-expert-parallel"
)

declare -a ACTIVE_MODELS=()
if [[ -n "${EXPLICIT_MODEL_NAME}" || -n "${EXPLICIT_MODEL_PATH}" || -n "${EXPLICIT_EXTRA_ARGS}" ]]; then
  if [[ -z "${EXPLICIT_MODEL_NAME}" || -z "${EXPLICIT_MODEL_PATH}" ]]; then
    echo "OOT_MODEL_NAME and OOT_MODEL_PATH must both be set when using explicit model overrides."
    exit 2
  fi
  ACTIVE_MODELS=("${EXPLICIT_MODEL_NAME}|${EXPLICIT_MODEL_PATH}|${EXPLICIT_EXTRA_ARGS}")
elif [[ "$MODE" == "ci" ]]; then
  ACTIVE_MODELS=("${CI_MODE_MODELS[@]}")
else
  ACTIVE_MODELS=("${FULL_MODE_MODELS[@]}")
fi

resolve_model_path() {
  local model_path="$1"
  if [[ "${model_path}" = /* ]]; then
    echo "${model_path}"
  elif [[ -f "/models/${model_path}/config.json" ]]; then
    echo "/models/${model_path}"
  else
    echo "${model_path}"
  fi
}

emit_new_vllm_logs() {
  if [[ "${STREAM_VLLM_LOGS}" != "1" || ! -f "${VLLM_LOG_FILE}" ]]; then
    return 0
  fi

  local current_line_count
  current_line_count=$(wc -l < "${VLLM_LOG_FILE}")
  if (( current_line_count <= LAST_VLLM_LOG_LINE )); then
    return 0
  fi

  echo ""
  echo "========== New vLLM log output =========="
  sed -n "$((LAST_VLLM_LOG_LINE + 1)),${current_line_count}p" "${VLLM_LOG_FILE}" || true
  LAST_VLLM_LOG_LINE=${current_line_count}
}

wait_server_ready() {
  local model_name="$1"
  echo ""
  echo "========== Waiting for vLLM server (${model_name}) =========="
  for ((i=1; i<=MAX_WAIT_RETRIES; i++)); do
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      emit_new_vllm_logs
      echo "vLLM server is ready for ${model_name}."
      return 0
    fi

    emit_new_vllm_logs

    if [[ -f "${VLLM_PID_FILE}" ]]; then
      local pid
      pid=$(cat "${VLLM_PID_FILE}")
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "vLLM process exited early for ${model_name}."
        emit_new_vllm_logs
        tail -n 200 "${VLLM_LOG_FILE}" || true
        return 1
      fi
    fi

    echo "Waiting for vLLM server... (${i}/${MAX_WAIT_RETRIES})"
    sleep "${WAIT_INTERVAL_SEC}"
  done

  echo "vLLM server did not become ready in time for ${model_name}."
  emit_new_vllm_logs
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
  local -a extra_arg_array=()

  local resolved_model_path
  resolved_model_path=$(resolve_model_path "${model_path}")

  if [[ -n "${extra_args}" ]]; then
    read -r -a extra_arg_array <<< "${extra_args}"
  fi

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
  # FIXME here disable the dual stream in OOT CI for avoid the hang issue
  export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
  rm -rf /tmp/.cache

  rm -f "${VLLM_PID_FILE}" || true

  nohup vllm serve "${resolved_model_path}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --async-scheduling \
    --load-format fastsafetensors \
    --max-model-len 16384 \
    "${extra_arg_array[@]}" \
    > "${VLLM_LOG_FILE}" 2>&1 &
  echo $! > "${VLLM_PID_FILE}"
  echo "Server PID: $(cat "${VLLM_PID_FILE}")"

  wait_server_ready "${model_name}"
}

accuracy_one_model() {
  local model_name="$1"
  local model_path="$2"
  local extra_args="$3"
  local flat_result_file=""

  local resolved_model_path
  resolved_model_path=$(resolve_model_path "${model_path}")

  if ! command -v lm_eval >/dev/null 2>&1; then
    echo "========== Installing lm-eval =========="
    pip install 'lm-eval[api]'
  fi

  mkdir -p "${RESULT_DIR}"
  local run_tag
  run_tag="$(date +%Y%m%d%H%M%S)_${model_name// /_}"
  local output_path="${RESULT_DIR}/${run_tag}"
  flat_result_file="${RESULT_DIR}/${run_tag}.json"

  echo ""
  echo "========== Running OOT gsm8k accuracy =========="
  echo "Model name: ${model_name}"

  lm_eval --model local-completions \
    --model_args model="${resolved_model_path}",base_url="http://127.0.0.1:${VLLM_PORT}/v1/completions",num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True \
    --tasks gsm8k \
    --num_fewshot 3 \
    --output_path "${output_path}" 2>&1 | tee -a "${ACCURACY_LOG_FILE}"

  # lm-eval output layout differs across versions: output_path may be a file
  # or a directory containing one/more JSON files. Follow native CI style:
  # resolve the latest generated JSON first, then parse metrics from it.
  local result_file=""
  result_file=$(python - <<PY
from pathlib import Path

candidate_roots = [Path("${output_path}"), Path("${RESULT_DIR}")]
json_candidates = []
for root in candidate_roots:
    if root.is_file() and root.suffix == ".json":
        json_candidates.append(root)
    elif root.is_dir():
        for p in root.rglob("*.json"):
            if p.is_file():
                json_candidates.append(p)

if not json_candidates:
    print("")
else:
    latest = max(json_candidates, key=lambda p: p.stat().st_mtime)
    print(str(latest))
PY
)

  if [[ -z "${result_file}" || ! -f "${result_file}" ]]; then
    echo "ERROR: No results JSON file found under ${output_path} or ${RESULT_DIR}"
    return 2
  fi

  # Flatten the result into RESULT_DIR so workflow-side checks can use the
  # same simple `ls`-based lookup as atom-test without depending on Python.
  if [[ "${result_file}" != "${flat_result_file}" ]]; then
    cp -f "${result_file}" "${flat_result_file}"
    result_file="${flat_result_file}"
  fi

  local value
  if command -v jq >/dev/null 2>&1; then
    value=$(jq '.results.gsm8k["exact_match,flexible-extract"]' "${result_file}")
  else
    value=$(python - <<PY
import json
with open("${result_file}", "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["results"]["gsm8k"]["exact_match,flexible-extract"])
PY
)
  fi

  echo "Result file: ${result_file}"
  echo "Flexible extract value: ${value}"
}

run_for_models() {
  local action="$1"
  local matched=0

  for entry in "${ACTIVE_MODELS[@]}"; do
    IFS='|' read -r model_name model_path extra_args <<< "${entry}"

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
    accuracy_one_model "${model_name}" "${model_path}" "${extra_args}"
    stop_server
  done

  if [[ "${matched}" -eq 0 ]]; then
    echo "No model matched MODE=${MODE}, SELECTED_MODEL=${SELECTED_MODEL}"
    exit 2
  fi
}

cleanup_on_exit() {
  if [[ "${TYPE}" == "launch" && "${KEEP_SERVER_ALIVE_ON_EXIT}" == "1" ]]; then
    echo "Keeping vLLM server alive for follow-up steps."
    return 0
  fi
  stop_server
}

trap 'cleanup_on_exit' EXIT

if [[ "${TYPE}" == "launch" ]]; then
  run_for_models "launch"
else
  run_for_models "accuracy"
fi

