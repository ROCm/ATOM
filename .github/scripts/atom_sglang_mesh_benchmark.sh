#!/bin/bash
set -euo pipefail

# InferenceX-style single-node SGLang runner for mori+sglang images.
# This intentionally does not use atom_sglang_test.sh: the Mesh docker is
# expected to provide SGLang directly, and we benchmark it with the same command
# shape as InferenceX DeepSeek-R1 FP4 SGLang recipes.

MODEL=${MODEL:-}
MODEL_NAME=${SGLANG_MODEL_NAME:-${MODEL}}
TP=${TP:-}
DP_SIZE=${DP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
CONC=${CONC:-}
ISL=${ISL:-}
OSL=${OSL:-}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1}
RESULT_FILENAME=${RESULT_FILENAME:-}
RESULT_DIR=${RESULT_DIR:-/tmp/sglang-benchmark-results}
BENCH_SERVING_DIR=${BENCH_SERVING_DIR:-/tmp/sglang-benchmark/bench_serving}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
BENCH_EXTRA_ARGS=${BENCH_EXTRA_ARGS:-}
SPEC_MODE=${SPEC_MODE:-none}
PORT=${PORT:-8888}
SERVER_LOG=${SERVER_LOG:-/workspace/server.log}
SERVER_PID_FILE=${SERVER_PID_FILE:-/tmp/atom_sglang_mesh.pid}
MAX_WAIT_RETRIES=${MAX_WAIT_RETRIES:-120}
WAIT_INTERVAL_SEC=${WAIT_INTERVAL_SEC:-30}
STREAM_SGLANG_LOGS=${STREAM_SGLANG_LOGS:-1}

if [[ -z "${MODEL}" || -z "${TP}" || -z "${CONC}" || -z "${ISL}" || -z "${OSL}" || -z "${RESULT_FILENAME}" ]]; then
  echo "MODEL, TP, CONC, ISL, OSL, and RESULT_FILENAME must be set."
  exit 2
fi

prepare_runtime_paths() {
  if [[ -d /app/sglang/python ]]; then
    export PYTHONPATH="/app/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"
  fi
  if [[ -d /workspace ]]; then
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

shlex_split_to_array() {
  local source_text="$1"
  local array_name="$2"
  local token
  while IFS= read -r -d '' token; do
    eval "${array_name}+=(\"\${token}\")"
  done < <(
    SOURCE_TEXT="${source_text}" python3 - <<'PY'
import os
import shlex
import sys

for token in shlex.split(os.environ["SOURCE_TEXT"]):
    sys.stdout.write(token)
    sys.stdout.write("\0")
PY
  )
}

emit_new_logs() {
  if [[ "${STREAM_SGLANG_LOGS}" != "1" || ! -f "${SERVER_LOG}" ]]; then
    return 0
  fi
  echo ""
  echo "========== SGLang Mesh server log =========="
  tail -n 240 "${SERVER_LOG}" || true
}

wait_server_ready() {
  echo ""
  echo "========== Waiting for SGLang Mesh server (${MODEL_NAME}) =========="
  for ((i=1; i<=MAX_WAIT_RETRIES; i++)); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "SGLang Mesh server is ready for ${MODEL_NAME}."
      return 0
    fi
    if [[ -f "${SERVER_PID_FILE}" ]]; then
      local pid
      pid=$(cat "${SERVER_PID_FILE}")
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "SGLang Mesh server exited early for ${MODEL_NAME}."
        emit_new_logs
        return 1
      fi
    fi
    echo "Waiting for SGLang Mesh server... (${i}/${MAX_WAIT_RETRIES})"
    sleep "${WAIT_INTERVAL_SEC}"
  done
  echo "SGLang Mesh server did not become ready in time for ${MODEL_NAME}."
  emit_new_logs
  return 1
}

cleanup() {
  if [[ -f "${SERVER_PID_FILE}" ]]; then
    kill "$(cat "${SERVER_PID_FILE}")" 2>/dev/null || true
    rm -f "${SERVER_PID_FILE}" || true
  fi
}
trap cleanup EXIT

prepare_runtime_paths

resolved_model_path=$(resolve_model_path "${MODEL}")
if [[ "${MODEL}" != /* && "${resolved_model_path}" == "${MODEL}" ]] && command -v hf >/dev/null 2>&1; then
  hf download "${MODEL}" || true
fi

export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_AITER_MLA_PERSIST="${SGLANG_AITER_MLA_PERSIST:-1}"
export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export AITER_QUICK_REDUCE_QUANTIZATION="${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}"
export IBDEVICES="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"

prefill_size=196608
if [[ "${ISL}" == "8192" && "${OSL}" == "1024" && "${CONC}" -gt "32" ]]; then
  prefill_size=32768
fi

declare -a server_args=()
if [[ -n "${SERVER_EXTRA_ARGS}" ]]; then
  shlex_split_to_array "${SERVER_EXTRA_ARGS}" server_args
fi

if [[ "${SPEC_MODE}" == "mtp" ]]; then
  export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
  if [[ "${SERVER_EXTRA_ARGS}" != *"--speculative-algorithm"* ]]; then
    server_args+=(
      --speculative-algorithm NEXTN
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
    )
  fi
  if [[ "${SERVER_EXTRA_ARGS}" != *"--max-running-requests"* ]]; then
    server_args+=(--max-running-requests 256)
  fi
  if [[ "${SERVER_EXTRA_ARGS}" != *"--cuda-graph-bs"* ]]; then
    server_args+=(--cuda-graph-bs)
    for bs in $(seq 1 128) 160 192 224 256; do
      server_args+=("${bs}")
    done
  fi
fi

rm -rf "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"
rm -f "${SERVER_LOG}" "${SERVER_PID_FILE}" || true

echo ""
echo "========== Launching mori+sglang Mesh server =========="
echo "Model path: ${resolved_model_path}"
echo "TP=${TP} DP=${DP_SIZE} EP=${EP_SIZE} SPEC=${SPEC_MODE}"

set -x
python3 -m sglang.launch_server \
  --model-path="${resolved_model_path}" \
  --trust-remote-code \
  --host=0.0.0.0 \
  --port="${PORT}" \
  --tensor-parallel-size="${TP}" \
  --chunked-prefill-size="${prefill_size}" \
  --mem-fraction-static=0.8 \
  --disable-radix-cache \
  --num-continuous-decode-steps=4 \
  --max-prefill-tokens="${prefill_size}" \
  --cuda-graph-max-bs=256 \
  --attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  "${server_args[@]}" \
  > "${SERVER_LOG}" 2>&1 &
set +x

echo $! > "${SERVER_PID_FILE}"
wait_server_ready

declare -a bench_args=()
if [[ -n "${BENCH_EXTRA_ARGS}" ]]; then
  shlex_split_to_array "${BENCH_EXTRA_ARGS}" bench_args
fi

set -x
PYTHONDONTWRITEBYTECODE=1 python "${BENCH_SERVING_DIR}/benchmark_serving.py" \
  --model="${resolved_model_path}" \
  --backend=vllm \
  --base-url="http://127.0.0.1:${PORT}" \
  --dataset-name=random \
  --random-input-len="${ISL}" \
  --random-output-len="${OSL}" \
  --random-range-ratio "${RANDOM_RANGE_RATIO}" \
  --num-prompts="$(( CONC * 10 ))" \
  --max-concurrency="${CONC}" \
  --num-warmups="$(( 2 * CONC ))" \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --percentile-metrics="ttft,tpot,itl,e2el" \
  --result-dir="${RESULT_DIR}" \
  --result-filename="${RESULT_FILENAME}.json" \
  "${bench_args[@]}"
set +x
