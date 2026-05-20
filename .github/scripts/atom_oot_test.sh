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
#   LM_EVAL_NUM_FEWSHOT
#   ACCURACY_TASK     (gsm8k | gsm8k_cot, default: gsm8k)
#                     gsm8k_cot uses chat-completions endpoint + apply_chat_template
#                     and ignores LM_EVAL_NUM_FEWSHOT (defers to YAML 8-shot).
#                     This mirrors the standalone CF-PROGRESS-0427 refactor.
#   KEEP_SERVER_ALIVE_ON_EXIT=1  (used by both launch and accuracy phases when
#                     the same vLLM server should be reused across steps)
#
# TYPE:
#   launch   - launch vLLM server and wait until ready
#   accuracy - run gsm8k accuracy test and save result JSON
#
# MODE:
#   ci    - workflow-provided OOT CI model entry
#   full  - workflow-provided OOT full-validation model entry
#
# Optional model_name can be used to run a single model when a caller passes
# multiple explicit entries.

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
VLLM_HOST=${VLLM_HOST:-localhost}
VLLM_PID_FILE=${VLLM_PID_FILE:-/tmp/vllm_oot.pid}
VLLM_LOG_FILE=${VLLM_LOG_FILE:-/tmp/vllm_oot.log}
RESULT_DIR=${RESULT_DIR:-/tmp/oot_accuracy_results}
ACCURACY_LOG_FILE=${ACCURACY_LOG_FILE:-/tmp/oot_accuracy_output.txt}
STREAM_VLLM_LOGS=${STREAM_VLLM_LOGS:-1}
KEEP_SERVER_ALIVE_ON_EXIT=${KEEP_SERVER_ALIVE_ON_EXIT:-0}
EXPLICIT_MODEL_NAME=${OOT_MODEL_NAME:-}
EXPLICIT_MODEL_PATH=${OOT_MODEL_PATH:-}
EXPLICIT_EXTRA_ARGS=${OOT_EXTRA_ARGS:-}
OOT_DOCKER_IMAGE=${OOT_DOCKER_IMAGE:-}
LM_EVAL_NUM_FEWSHOT=${LM_EVAL_NUM_FEWSHOT:-3}
ACCURACY_TASK=${ACCURACY_TASK:-gsm8k}
LAST_VLLM_LOG_LINE=0

if ! [[ "${LM_EVAL_NUM_FEWSHOT}" =~ ^[0-9]+$ ]]; then
  echo "Invalid LM_EVAL_NUM_FEWSHOT: ${LM_EVAL_NUM_FEWSHOT}. Expected a non-negative integer."
  exit 2
fi

if [[ "${ACCURACY_TASK}" != "gsm8k" && "${ACCURACY_TASK}" != "gsm8k_cot" ]]; then
  echo "Invalid ACCURACY_TASK: ${ACCURACY_TASK}. Expected: gsm8k or gsm8k_cot"
  exit 2
fi

declare -a ACTIVE_MODELS=()
if [[ -n "${EXPLICIT_MODEL_NAME}" || -n "${EXPLICIT_MODEL_PATH}" || -n "${EXPLICIT_EXTRA_ARGS}" ]]; then
  if [[ -z "${EXPLICIT_MODEL_NAME}" || -z "${EXPLICIT_MODEL_PATH}" ]]; then
    echo "OOT_MODEL_NAME and OOT_MODEL_PATH must both be set when using explicit model overrides."
    exit 2
  fi
  ACTIVE_MODELS=("${EXPLICIT_MODEL_NAME}|${EXPLICIT_MODEL_PATH}|${EXPLICIT_EXTRA_ARGS}")
else
  echo "${MODE} mode requires OOT_MODEL_NAME and OOT_MODEL_PATH env vars from the workflow."
  exit 2
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

server_already_running() {
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid
    pid=$(cat "${VLLM_PID_FILE}" 2>/dev/null || echo "")
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        return 0
      fi
    fi
  fi
  return 1
}

# Return 0 if the already-running vLLM server is serving the model at
# ${1}. We compare on the vLLM `/v1/models` `id` field (which equals the
# model path the server was launched with), not the human-friendly
# OOT_MODEL_NAME, because vLLM only exposes the path.
server_running_model_matches() {
  local expected_path="$1"
  local served
  served=$(curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null \
           | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
ids = [m.get("id", "") for m in data.get("data", [])]
print("\n".join(ids))
' 2>/dev/null) || return 1
  while IFS= read -r served_id; do
    if [[ "${served_id}" == "${expected_path}" ]]; then
      return 0
    fi
  done <<< "${served}"
  echo "Running vLLM server is serving model(s): ${served}" >&2
  echo "Expected model path: ${expected_path}" >&2
  return 1
}

launch_one_model() {
  local model_name="$1"
  local model_path="$2"
  local extra_args="$3"
  local -a extra_arg_array=()

  local resolved_model_path
  resolved_model_path=$(resolve_model_path "${model_path}")

  if server_already_running; then
    # Don't blindly reuse: verify the running server is actually serving the
    # model we were asked to launch. Without this check, iterating multiple
    # models (or running with KEEP_SERVER_ALIVE_ON_EXIT=1 across phases that
    # change OOT_MODEL_PATH) would silently evaluate accuracy / benchmarks
    # against the *previous* model and report wrong numbers.
    if server_running_model_matches "${resolved_model_path}"; then
      echo "Reusing already-running vLLM server (PID $(cat "${VLLM_PID_FILE}"))."
      return 0
    fi
    echo "ERROR: an existing vLLM server (PID $(cat "${VLLM_PID_FILE}")) is " \
         "running on port ${VLLM_PORT} but serves a different model than " \
         "requested. Stop it (or unset KEEP_SERVER_ALIVE_ON_EXIT) before " \
         "launching ${resolved_model_path}." >&2
    return 2
  fi

  if [[ -n "${extra_args}" ]]; then
    while IFS= read -r -d '' token; do
      extra_arg_array+=("${token}")
    done < <(
      EXTRA_ARGS="${extra_args}" python3 - <<'PY'
import os
import shlex
import sys

for token in shlex.split(os.environ["EXTRA_ARGS"]):
    sys.stdout.write(token)
    sys.stdout.write("\0")
PY
    )
  fi

  echo ""
  echo "========== Launching vLLM server =========="
  echo "Model name: ${model_name}"
  echo "Model path: ${resolved_model_path}"
  echo "Extra args: ${extra_args}"

  export SAFETENSORS_FAST_GPU=1
  export VLLM_RPC_TIMEOUT=1800000
  export VLLM_CACHE_ROOT=/root/.cache/vllm
  export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

  if [[ -n "${OOT_ENV_VARS:-}" ]]; then
    while IFS= read -r _env_line; do
      [[ -n "${_env_line}" ]] && export "${_env_line}" && echo "Exported: ${_env_line}"
    done <<< "$(printf '%b' "${OOT_ENV_VARS}")"
  fi
  rm -rf /root/.cache

  rm -f "${VLLM_PID_FILE}" || true

  # Avoid importing a host-mounted source tree as a namespace package.
  cd /tmp
  nohup vllm serve "${resolved_model_path}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --async-scheduling \
    --load-format fastsafetensors \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    "${extra_arg_array[@]}" \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
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
  echo "========== Running OOT ${ACCURACY_TASK} accuracy =========="
  echo "Model name: ${model_name}"
  echo "Task: ${ACCURACY_TASK}"

  local eval_model
  local eval_base_url
  local apply_chat_template_arg=""
  local num_fewshot_arg=""

  if [[ "${ACCURACY_TASK}" == "gsm8k_cot" ]]; then
    eval_model="local-chat-completions"
    eval_base_url="http://127.0.0.1:${VLLM_PORT}/v1/chat/completions"
    apply_chat_template_arg="--apply_chat_template"
    echo "Few-shot: YAML default (8-shot CoT)"
  else
    eval_model="local-completions"
    eval_base_url="http://127.0.0.1:${VLLM_PORT}/v1/completions"
    num_fewshot_arg="--num_fewshot ${LM_EVAL_NUM_FEWSHOT}"
    echo "Few-shot: ${LM_EVAL_NUM_FEWSHOT}"
  fi

  lm_eval --model "${eval_model}" \
    --model_args model="${resolved_model_path}",base_url="${eval_base_url}",num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True \
    --tasks "${ACCURACY_TASK}" \
    ${apply_chat_template_arg} \
    ${num_fewshot_arg} \
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

  if [[ -n "${OOT_DOCKER_IMAGE:-}" ]] || [[ -n "${GPU_NAME:-}" ]] || [[ -n "${GPU_VRAM_GB:-}" ]] || [[ -n "${ROCM_VERSION:-}" ]]; then
    RESULT_FILE="${result_file}" \
    OOT_DOCKER_IMAGE="${OOT_DOCKER_IMAGE:-}" \
    GPU_NAME="${GPU_NAME:-}" \
    GPU_VRAM_GB="${GPU_VRAM_GB:-}" \
    ROCM_VERSION="${ROCM_VERSION:-}" \
    python - <<'PY'
import json
import os

result_file = os.environ["RESULT_FILE"]
with open(result_file, "r", encoding="utf-8") as f:
    data = json.load(f)

metadata = data.setdefault("atom_ci_metadata", {})
if os.environ.get("OOT_DOCKER_IMAGE"):
    metadata["docker_image"] = os.environ["OOT_DOCKER_IMAGE"]
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
    value=$(jq ".results.${ACCURACY_TASK}[\"exact_match,flexible-extract\"]" "${result_file}")
  else
    value=$(RESULT_FILE="${result_file}" ACC_TASK="${ACCURACY_TASK}" python - <<'PY'
import json
import os
with open(os.environ["RESULT_FILE"], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["results"][os.environ["ACC_TASK"]]["exact_match,flexible-extract"])
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

    # accuracy mode: launch + evaluate each selected model, then stop the
    # server unless KEEP_SERVER_ALIVE_ON_EXIT=1 (e.g. so a follow-up
    # benchmark step can reuse the same vLLM server).
    launch_one_model "${model_name}" "${model_path}" "${extra_args}"
    accuracy_one_model "${model_name}" "${model_path}" "${extra_args}"
    if [[ "${KEEP_SERVER_ALIVE_ON_EXIT}" != "1" ]]; then
      stop_server
    fi
  done

  if [[ "${matched}" -eq 0 ]]; then
    echo "No model matched MODE=${MODE}, SELECTED_MODEL=${SELECTED_MODEL}"
    exit 2
  fi
}

cleanup_on_exit() {
  if [[ "${KEEP_SERVER_ALIVE_ON_EXIT}" == "1" ]]; then
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
