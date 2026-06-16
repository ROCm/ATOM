#!/bin/bash
set -euo pipefail

# Usage:
#   .github/scripts/atom_oot_test.sh start <mode> [model_name]
#   .github/scripts/atom_oot_test.sh launch <mode> [model_name]
#   .github/scripts/atom_oot_test.sh client <mode> [model_name]
#   .github/scripts/atom_oot_test.sh accuracy <mode> [model_name]
#
# Alternatively, pass a single model explicitly through environment variables:
#   OOT_MODEL_NAME
#   OOT_MODEL_PATH
#   OOT_EXTRA_ARGS
#   LM_EVAL_NUM_FEWSHOT
#
# TYPE:
#   start    - launch vLLM server in the background and return immediately
#   launch   - launch vLLM server and wait until ready
#   client   - run gsm8k accuracy against an existing server
#   accuracy - launch server, run gsm8k accuracy, and save result JSON
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

if [[ "$TYPE" != "start" && "$TYPE" != "launch" && "$TYPE" != "client" && "$TYPE" != "accuracy" ]]; then
  echo "Invalid TYPE: $TYPE. Expected: start, launch, client, or accuracy"
  exit 2
fi

if [[ "$MODE" != "ci" && "$MODE" != "full" ]]; then
  echo "Invalid MODE: $MODE. Expected: ci or full"
  exit 2
fi

MAX_WAIT_RETRIES=${MAX_WAIT_RETRIES:-60}
WAIT_INTERVAL_SEC=${WAIT_INTERVAL_SEC:-30}
# Fatal server-log markers: if any appears while waiting for the server, abort
# immediately instead of burning the full MAX_WAIT_RETRIES budget (which keeps the
# GPU runner occupied long after init has already crashed). These are unambiguously
# terminal — e.g. NCCL "unhandled cuda error" corrupts the CUDA context and never
# recovers. The recoverable "tp_group_reuse failed ... will fall back" warning is
# intentionally NOT matched. Override via FATAL_LOG_PATTERNS; set empty to disable.
FATAL_LOG_PATTERNS=${FATAL_LOG_PATTERNS:-'unhandled cuda error|uncorrectable ECC|EngineCore[_ ][A-Za-z0-9]* died|Engine core proc.* died|EngineCore failed to start|Failed to initialize EngineCore'}
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
EXPLICIT_CLIENT_COMMAND=${OOT_CLIENT_COMMAND:-}
OOT_DOCKER_IMAGE=${OOT_DOCKER_IMAGE:-}
LM_EVAL_NUM_FEWSHOT=${LM_EVAL_NUM_FEWSHOT:-3}
LAST_VLLM_LOG_LINE=0

if ! [[ "${LM_EVAL_NUM_FEWSHOT}" =~ ^[0-9]+$ ]]; then
  echo "Invalid LM_EVAL_NUM_FEWSHOT: ${LM_EVAL_NUM_FEWSHOT}. Expected a non-negative integer."
  exit 2
fi

declare -a ACTIVE_MODELS=()
if [[ -n "${EXPLICIT_MODEL_NAME}" || -n "${EXPLICIT_MODEL_PATH}" || -n "${EXPLICIT_EXTRA_ARGS}" ]]; then
  if [[ -z "${EXPLICIT_MODEL_NAME}" || -z "${EXPLICIT_MODEL_PATH}" ]]; then
    echo "OOT_MODEL_NAME and OOT_MODEL_PATH must both be set when using explicit model overrides."
    exit 2
  fi
  ACTIVE_MODELS=("${EXPLICIT_MODEL_NAME}|${EXPLICIT_MODEL_PATH}|${EXPLICIT_EXTRA_ARGS}|${EXPLICIT_CLIENT_COMMAND}")
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

# Scan the server log for a fatal marker. Prints the first matching line and
# returns 0 when a fatal error is present, 1 otherwise.
detect_fatal_log() {
  [[ -n "${FATAL_LOG_PATTERNS}" && -f "${VLLM_LOG_FILE}" ]] || return 1
  grep -E -m1 "${FATAL_LOG_PATTERNS}" "${VLLM_LOG_FILE}" 2>/dev/null
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

    local fatal_line
    if fatal_line=$(detect_fatal_log); then
      echo "Detected fatal server error for ${model_name}; aborting wait early instead of retrying:"
      echo "  ${fatal_line}"
      emit_new_vllm_logs
      tail -n 200 "${VLLM_LOG_FILE}" || true
      return 1
    fi

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

# Scrape MTP/speculative-decode acceptance from the live vLLM /metrics endpoint
# and store overall + per-position acceptance into the result JSON. Must be
# called while the server is still running. No-op for non-speculative runs
# (the spec_decode counters are absent). The workflow's "Check OOT MTP
# acceptance rate" step reads these values to gate against regressions —
# gsm8k accuracy alone cannot, since spec decoding is lossless w.r.t. the
# target model and a broken draft head only craters acceptance/throughput.
record_mtp_acceptance() {
  local result_file="$1"
  local metrics_file="/tmp/oot_spec_metrics.txt"

  if ! curl -fsS "http://127.0.0.1:${VLLM_PORT}/metrics" -o "${metrics_file}" 2>/dev/null; then
    echo "MTP acceptance: /metrics not reachable (skipping)."
    return 0
  fi

  RESULT_FILE="${result_file}" METRICS_FILE="${metrics_file}" python3 - <<'PY'
import json, os, re

with open(os.environ["METRICS_FILE"], encoding="utf-8", errors="replace") as f:
    metrics = f.read()

def sum_counter(name):
    # Sum a Prometheus counter across all label series; tolerate the `_total`
    # suffix and optional `{labels}`. Anchored so e.g. num_accepted_tokens does
    # not also match num_accepted_tokens_per_pos.
    pat = rf'^{re.escape(name)}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9eE+.\-]+)\s*$'
    vals = [float(m.group(1)) for m in re.finditer(pat, metrics, re.M)]
    return sum(vals) if vals else None

accepted = sum_counter("vllm:spec_decode_num_accepted_tokens")
draft_tokens = sum_counter("vllm:spec_decode_num_draft_tokens")
num_drafts = sum_counter("vllm:spec_decode_num_drafts")

per_pos_counts = {}
for m in re.finditer(
    r'vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{([^}]*)\}\s+([0-9eE+.\-]+)',
    metrics,
):
    pm = re.search(r'position="(\d+)"', m.group(1))
    if pm:
        i = int(pm.group(1))
        per_pos_counts[i] = per_pos_counts.get(i, 0.0) + float(m.group(2))

if not draft_tokens:
    print("MTP acceptance: no spec-decode metrics found (non-MTP run).")
else:
    overall = accepted / draft_tokens
    per_pos = []
    if num_drafts and per_pos_counts:
        per_pos = [per_pos_counts[i] / num_drafts for i in sorted(per_pos_counts)]
    rf = os.environ["RESULT_FILE"]
    with open(rf, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.setdefault("atom_ci_metadata", {})
    meta["mtp_acceptance_overall"] = overall
    meta["mtp_per_pos_acceptance"] = per_pos
    with open(rf, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("MTP acceptance overall: %.4f, per-position: %s" % (
        overall, ", ".join("%.4f" % r for r in per_pos) if per_pos else "n/a"))
PY
}

launch_one_model() {
  local model_name="$1"
  local model_path="$2"
  local extra_args="$3"
  local wait_for_ready="${4:-1}"
  local -a extra_arg_array=()

  local resolved_model_path
  resolved_model_path=$(resolve_model_path "${model_path}")

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
    --no-enable-prefix-caching \
    > "${VLLM_LOG_FILE}" 2>&1 &
  echo $! > "${VLLM_PID_FILE}"
  echo "Server PID: $(cat "${VLLM_PID_FILE}")"

  if [[ "${wait_for_ready}" == "1" ]]; then
    wait_server_ready "${model_name}"
  fi
}

accuracy_one_model() {
  local model_name="$1"
  local model_path="$2"
  local extra_args="$3"
  local client_command="${4:-}"
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
  echo "Few-shot count: ${LM_EVAL_NUM_FEWSHOT}"

  if [[ "${client_command}" == "null" ]]; then
    client_command=""
  fi

  if [[ -n "${client_command}" ]]; then
    local -a client_command_args=()
    while IFS= read -r -d '' token; do
      client_command_args+=("${token}")
    done < <(
      CLIENT_COMMAND="${client_command}" \
      MODEL_PATH_VALUE="${resolved_model_path}" \
      OUTPUT_PATH_VALUE="${output_path}" \
      LM_EVAL_NUM_FEWSHOT_VALUE="${LM_EVAL_NUM_FEWSHOT}" \
      VLLM_PORT_VALUE="${VLLM_PORT}" \
      python3 - <<'PY'
import os
import shlex
import sys

client_command = os.environ["CLIENT_COMMAND"]
replacements = {
    "${MODEL_PATH}": os.environ["MODEL_PATH_VALUE"],
    "$MODEL_PATH": os.environ["MODEL_PATH_VALUE"],
    "${OUTPUT_PATH}": os.environ["OUTPUT_PATH_VALUE"],
    "$OUTPUT_PATH": os.environ["OUTPUT_PATH_VALUE"],
    "${LM_EVAL_NUM_FEWSHOT}": os.environ["LM_EVAL_NUM_FEWSHOT_VALUE"],
    "$LM_EVAL_NUM_FEWSHOT": os.environ["LM_EVAL_NUM_FEWSHOT_VALUE"],
    "${VLLM_PORT}": os.environ["VLLM_PORT_VALUE"],
    "$VLLM_PORT": os.environ["VLLM_PORT_VALUE"],
}
for src, dst in replacements.items():
    client_command = client_command.replace(src, dst)

for token in shlex.split(client_command):
    sys.stdout.write(token)
    sys.stdout.write("\0")
PY
    )

    if [[ ${#client_command_args[@]} -eq 0 ]]; then
      echo "ERROR: client_command is set but empty after parsing."
      return 2
    fi

    for arg in "${client_command_args[@]}"; do
      if [[ "${arg}" =~ \$\{[A-Z0-9_]+\} ]] || [[ "${arg}" =~ \$[A-Z_][A-Z0-9_]* ]]; then
        echo "ERROR: client_command contains unresolved placeholder after expansion: ${arg}"
        return 2
      fi
    done

    echo "Using custom lm-eval command from client_command: ${client_command}"
    "${client_command_args[@]}" 2>&1 | tee -a "${ACCURACY_LOG_FILE}"
  else
    echo "Using default lm-eval command."
    local lm_args=(
      --model_args
      model="${resolved_model_path}",base_url="http://127.0.0.1:${VLLM_PORT}/v1/completions",num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True
    )
    lm_eval --model local-completions \
      "${lm_args[@]}" \
      --tasks gsm8k \
      --num_fewshot "${LM_EVAL_NUM_FEWSHOT}" \
      --output_path "${output_path}" 2>&1 | tee -a "${ACCURACY_LOG_FILE}"
  fi

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

  # Capture MTP acceptance from /metrics while the server is still alive.
  record_mtp_acceptance "${result_file}"

  echo "Result file: ${result_file}"
  echo "Flexible extract value: ${value}"
}

run_for_models() {
  local action="$1"
  local matched=0

  for entry in "${ACTIVE_MODELS[@]}"; do
    IFS='|' read -r model_name model_path extra_args client_command <<< "${entry}"

    if [[ -n "${SELECTED_MODEL}" && "${SELECTED_MODEL}" != "${model_name}" ]]; then
      continue
    fi
    matched=1

    if [[ "${action}" == "start" ]]; then
      launch_one_model "${model_name}" "${model_path}" "${extra_args}" "0"
      break
    fi

    if [[ "${action}" == "launch" ]]; then
      launch_one_model "${model_name}" "${model_path}" "${extra_args}"
      break
    fi

    if [[ "${action}" == "client" ]]; then
      accuracy_one_model "${model_name}" "${model_path}" "${extra_args}" "${client_command}"
      break
    fi

    # accuracy mode: launch + evaluate each selected model, then stop server.
    launch_one_model "${model_name}" "${model_path}" "${extra_args}"
    accuracy_one_model "${model_name}" "${model_path}" "${extra_args}" "${client_command}"
    stop_server
  done

  if [[ "${matched}" -eq 0 ]]; then
    echo "No model matched MODE=${MODE}, SELECTED_MODEL=${SELECTED_MODEL}"
    exit 2
  fi
}

cleanup_on_exit() {
  if [[ "${TYPE}" == "start" || ( "${TYPE}" == "launch" && "${KEEP_SERVER_ALIVE_ON_EXIT}" == "1" ) ]]; then
    echo "Keeping vLLM server alive for follow-up steps."
    return 0
  fi
  stop_server
}

trap 'cleanup_on_exit' EXIT

if [[ "${TYPE}" == "start" ]]; then
  run_for_models "start"
elif [[ "${TYPE}" == "launch" ]]; then
  run_for_models "launch"
elif [[ "${TYPE}" == "client" ]]; then
  run_for_models "client"
else
  run_for_models "accuracy"
fi

