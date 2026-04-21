#!/bin/bash
set -euo pipefail

# Usage:
#   .github/scripts/atom_sglang_test.sh launch
#   .github/scripts/atom_sglang_test.sh accuracy
#
# Required environment variables:
#   SGLANG_MODEL_NAME
#   SGLANG_MODEL_PATH
#
# Optional environment variables:
#   SGLANG_EXTRA_ARGS
#   SGLANG_ENV_VARS
#   SGLANG_PORT
#   SGLANG_HOST
#   MAX_WAIT_RETRIES
#   WAIT_INTERVAL_SEC
#   STREAM_SGLANG_LOGS
#   KEEP_SERVER_ALIVE_ON_EXIT
#   SGLANG_PID_FILE
#   SGLANG_LOG_FILE
#   RESULT_DIR
#   ACCURACY_LOG_FILE
#   LM_EVAL_TASK
#   LM_EVAL_NUM_FEWSHOT
#   LM_EVAL_NUM_CONCURRENT

TYPE=${1:-launch}
if [[ "${TYPE}" != "launch" && "${TYPE}" != "accuracy" ]]; then
  echo "Invalid TYPE: ${TYPE}. Expected: launch or accuracy"
  exit 2
fi

MAX_WAIT_RETRIES=${MAX_WAIT_RETRIES:-60}
WAIT_INTERVAL_SEC=${WAIT_INTERVAL_SEC:-30}
SGLANG_PORT=${SGLANG_PORT:-8000}
SGLANG_HOST=${SGLANG_HOST:-localhost}
SGLANG_PID_FILE=${SGLANG_PID_FILE:-/tmp/atom_sglang.pid}
SGLANG_LOG_FILE=${SGLANG_LOG_FILE:-/tmp/atom_sglang.log}
RESULT_DIR=${RESULT_DIR:-/tmp/atom_sglang_accuracy_results}
ACCURACY_LOG_FILE=${ACCURACY_LOG_FILE:-/tmp/atom_sglang_accuracy_output.txt}
STREAM_SGLANG_LOGS=${STREAM_SGLANG_LOGS:-1}
KEEP_SERVER_ALIVE_ON_EXIT=${KEEP_SERVER_ALIVE_ON_EXIT:-0}
LM_EVAL_TASK=${LM_EVAL_TASK:-gsm8k}
LM_EVAL_NUM_FEWSHOT=${LM_EVAL_NUM_FEWSHOT:-3}
LM_EVAL_NUM_CONCURRENT=${LM_EVAL_NUM_CONCURRENT:-65}

MODEL_NAME=${SGLANG_MODEL_NAME:-}
MODEL_PATH=${SGLANG_MODEL_PATH:-}
MODEL_EXTRA_ARGS=${SGLANG_EXTRA_ARGS:-}
MODEL_ENV_VARS=${SGLANG_ENV_VARS:-}
SGLANG_DOCKER_IMAGE=${SGLANG_DOCKER_IMAGE:-}

LAST_SGLANG_LOG_LINE=0

if [[ -z "${MODEL_NAME}" || -z "${MODEL_PATH}" ]]; then
  echo "SGLANG_MODEL_NAME and SGLANG_MODEL_PATH must both be set."
  exit 2
fi

prepare_runtime_paths() {
  if [[ -d /app/sglang/python && -d /app/ATOM ]]; then
    local path_prefix="/app/sglang/python:/app/ATOM"
    if [[ -d /app/aiter-test ]]; then
      path_prefix="/app/aiter-test:${path_prefix}"
    fi
    export PYTHONPATH="${path_prefix}${PYTHONPATH:+:${PYTHONPATH}}"
    cd /app
  elif [[ -d /workspace ]]; then
    cd /workspace
  fi
}

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

emit_new_sglang_logs() {
  if [[ "${STREAM_SGLANG_LOGS}" != "1" || ! -f "${SGLANG_LOG_FILE}" ]]; then
    return 0
  fi

  local current_line_count
  current_line_count=$(wc -l < "${SGLANG_LOG_FILE}")
  if (( current_line_count <= LAST_SGLANG_LOG_LINE )); then
    return 0
  fi

  echo ""
  echo "========== New SGLang log output =========="
  sed -n "$((LAST_SGLANG_LOG_LINE + 1)),${current_line_count}p" "${SGLANG_LOG_FILE}" || true
  LAST_SGLANG_LOG_LINE=${current_line_count}
}

wait_server_ready() {
  echo ""
  echo "========== Waiting for SGLang server (${MODEL_NAME}) =========="
  for ((i=1; i<=MAX_WAIT_RETRIES; i++)); do
    if curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
      emit_new_sglang_logs
      echo "SGLang server is ready for ${MODEL_NAME}."
      return 0
    fi

    emit_new_sglang_logs

    if [[ -f "${SGLANG_PID_FILE}" ]]; then
      local pid
      pid=$(cat "${SGLANG_PID_FILE}")
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "SGLang process exited early for ${MODEL_NAME}."
        emit_new_sglang_logs
        tail -n 200 "${SGLANG_LOG_FILE}" || true
        return 1
      fi
    fi

    echo "Waiting for SGLang server... (${i}/${MAX_WAIT_RETRIES})"
    sleep "${WAIT_INTERVAL_SEC}"
  done

  echo "SGLang server did not become ready in time for ${MODEL_NAME}."
  emit_new_sglang_logs
  tail -n 200 "${SGLANG_LOG_FILE}" || true
  return 1
}

stop_server() {
  if [[ -f "${SGLANG_PID_FILE}" ]]; then
    local pid
    pid=$(cat "${SGLANG_PID_FILE}")
    kill "${pid}" 2>/dev/null || true
    rm -f "${SGLANG_PID_FILE}" || true
  fi
}

launch_server() {
  local resolved_model_path
  resolved_model_path=$(resolve_model_path "${MODEL_PATH}")

  local -a extra_arg_array=()
  if [[ -n "${MODEL_EXTRA_ARGS}" ]]; then
    read -r -a extra_arg_array <<< "${MODEL_EXTRA_ARGS}"
  fi

  prepare_runtime_paths

  export AITER_QUICK_REDUCE_QUANTIZATION="${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}"
  export SGLANG_AITER_FP8_PREFILL_ATTN="${SGLANG_AITER_FP8_PREFILL_ATTN:-0}"
  export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
  export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION="${ATOM_ENABLE_DS_QKNORM_QUANT_FUSION:-1}"
  export SGLANG_EXTERNAL_MODEL_PACKAGE="${SGLANG_EXTERNAL_MODEL_PACKAGE:-atom.plugin.sglang.models}"
  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-128}"

  if [[ -n "${MODEL_ENV_VARS}" ]]; then
    while IFS= read -r env_line; do
      [[ -n "${env_line}" ]] || continue
      export "${env_line}"
      echo "Exported: ${env_line}"
    done <<< "$(printf '%b' "${MODEL_ENV_VARS}")"
  fi

  rm -rf /root/.cache

  rm -f "${SGLANG_PID_FILE}" "${SGLANG_LOG_FILE}" || true

  echo ""
  echo "========== Launching SGLang server =========="
  echo "Model name: ${MODEL_NAME}"
  echo "Model path: ${resolved_model_path}"
  echo "Extra args: ${MODEL_EXTRA_ARGS}"

  nohup python3 -m sglang.launch_server \
    --model-path "${resolved_model_path}" \
    --host "${SGLANG_HOST}" \
    --port "${SGLANG_PORT}" \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache \
    "${extra_arg_array[@]}" \
    > "${SGLANG_LOG_FILE}" 2>&1 &

  echo $! > "${SGLANG_PID_FILE}"
  echo "Server PID: $(cat "${SGLANG_PID_FILE}")"

  wait_server_ready
}

run_accuracy() {
  local resolved_model_path
  resolved_model_path=$(resolve_model_path "${MODEL_PATH}")

  prepare_runtime_paths

  if ! command -v lm_eval >/dev/null 2>&1; then
    echo "========== Installing lm-eval =========="
    pip install 'lm-eval[api]'
  fi

  mkdir -p "${RESULT_DIR}"
  : > "${ACCURACY_LOG_FILE}"

  local run_tag
  run_tag="$(date +%Y%m%d%H%M%S)_${MODEL_NAME// /_}"
  local output_path="${RESULT_DIR}/${run_tag}"
  local flat_result_file="${RESULT_DIR}/${run_tag}.json"

  echo ""
  echo "========== Running SGLang accuracy =========="
  echo "Model name: ${MODEL_NAME}"

  lm_eval --model local-completions \
    --model_args model="${resolved_model_path}",base_url="http://127.0.0.1:${SGLANG_PORT}/v1/completions",num_concurrent="${LM_EVAL_NUM_CONCURRENT}",max_retries=1,tokenized_requests=False,trust_remote_code=True \
    --tasks "${LM_EVAL_TASK}" \
    --num_fewshot "${LM_EVAL_NUM_FEWSHOT}" \
    --output_path "${output_path}" 2>&1 | tee -a "${ACCURACY_LOG_FILE}"

  local result_file=""
  result_file=$(python3 - <<PY
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

  if [[ "${result_file}" != "${flat_result_file}" ]]; then
    cp -f "${result_file}" "${flat_result_file}"
    result_file="${flat_result_file}"
  fi

  if [[ -n "${SGLANG_DOCKER_IMAGE:-}" ]] || [[ -n "${GPU_NAME:-}" ]] || [[ -n "${GPU_VRAM_GB:-}" ]] || [[ -n "${ROCM_VERSION:-}" ]]; then
    RESULT_FILE="${result_file}" \
    SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-}" \
    GPU_NAME="${GPU_NAME:-}" \
    GPU_VRAM_GB="${GPU_VRAM_GB:-}" \
    ROCM_VERSION="${ROCM_VERSION:-}" \
    python3 - <<'PY'
import json
import os

result_file = os.environ["RESULT_FILE"]
with open(result_file, "r", encoding="utf-8") as f:
    data = json.load(f)

metadata = data.setdefault("atom_ci_metadata", {})
if os.environ.get("SGLANG_DOCKER_IMAGE"):
    metadata["docker_image"] = os.environ["SGLANG_DOCKER_IMAGE"]
if os.environ.get("GPU_NAME"):
    metadata["gpu_name"] = os.environ["GPU_NAME"]
if os.environ.get("GPU_VRAM_GB"):
    try:
        metadata["gpu_vram_gb"] = int(float(os.environ["GPU_VRAM_GB"]))
    except ValueError:
        pass
if os.environ.get("ROCM_VERSION"):
    metadata["rocm_version"] = os.environ["ROCM_VERSION"]

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PY
  fi

  local value
  if command -v jq >/dev/null 2>&1; then
    value=$(jq ".results.\"${LM_EVAL_TASK}\"[\"exact_match,flexible-extract\"]" "${result_file}")
  else
    value=$(python3 - <<PY
import json
with open("${result_file}", "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["results"]["${LM_EVAL_TASK}"]["exact_match,flexible-extract"])
PY
)
  fi

  echo "Result file: ${result_file}"
  echo "Flexible extract value: ${value}"
}

cleanup_on_exit() {
  if [[ "${TYPE}" == "launch" && "${KEEP_SERVER_ALIVE_ON_EXIT}" == "1" ]]; then
    echo "Keeping SGLang server alive for follow-up steps."
    return 0
  fi
  stop_server
}

trap 'cleanup_on_exit' EXIT

if [[ "${TYPE}" == "launch" ]]; then
  launch_server
else
  launch_server
  run_accuracy
fi
