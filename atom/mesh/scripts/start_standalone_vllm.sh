#!/usr/bin/env bash
set -euo pipefail

# Launch a standalone (non-disaggregated) vLLM server on a single node.
# For debugging ATOM vLLM plugin — no PD, no Mooncake.
#
# Required env:
#   MODEL_PATH      - model path (e.g. /it-share/models/deepseek-ai/DeepSeek-R1-0528-MXFP4)
#   TP              - tensor parallel size (e.g. 8)
#
# Optional env (with defaults):
#   PORT=8000  MEM_FRACTION=0.9  KV_CACHE_DTYPE=fp8
#   MAX_NUM_SEQS=128  MAX_NUM_BATCHED_TOKENS=16384
#   LOAD_FORMAT=fastsafetensors
#   ENFORCE_EAGER=              (set to 1 to disable torch.compile + CUDAGraph)
#   LOAD_DUMMY=                 (set to 1 to skip weight loading)
#
# Usage (inside container):
#   MODEL_PATH=/it-share/models/deepseek-ai/DeepSeek-R1-0528-MXFP4 TP=8 bash start_standalone_vllm.sh
#   MODEL_PATH=/it-share/models/deepseek-ai/DeepSeek-R1-0528 TP=8 LOAD_DUMMY=1 ENFORCE_EAGER=1 bash start_standalone_vllm.sh

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TP:?TP is required}"

PORT="${PORT:-8000}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
LOAD_DUMMY="${LOAD_DUMMY:-}"

GPU_IDS=$(seq -s, 0 $((TP - 1)))

echo "[standalone-vllm] MODEL=${MODEL_PATH}"
echo "[standalone-vllm] TP=${TP} GPUs=${GPU_IDS} port=${PORT}"
echo "[standalone-vllm] MEM=${MEM_FRACTION} KV_DTYPE=${KV_CACHE_DTYPE} LOAD_FORMAT=${LOAD_FORMAT}"
echo "[standalone-vllm] ENFORCE_EAGER=${ENFORCE_EAGER:-<off>} LOAD_DUMMY=${LOAD_DUMMY:-<off>}"

mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${GPU_IDS}
export SAFETENSORS_FAST_GPU=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache

declare -a EXTRA_ARGS=()

if [[ -n "${LOAD_DUMMY}" ]]; then
    EXTRA_ARGS+=(--load-format dummy)
    echo "[standalone-vllm] LOAD_DUMMY enabled — using dummy weights"
else
    EXTRA_ARGS+=(--load-format "${LOAD_FORMAT}")
fi

if [[ -n "${ENFORCE_EAGER}" ]]; then
    EXTRA_ARGS+=(--enforce-eager)
    echo "[standalone-vllm] enforce-eager mode (no torch.compile, no CUDAGraph)"
else
    EXTRA_ARGS+=(--async-scheduling)
    CUDAGRAPH_SIZES=$(seq -s, 1 256)
    EXTRA_ARGS+=(--compilation-config "{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_SIZES}]}")
    echo "[standalone-vllm] async-scheduling + CUDAGraph FULL_AND_PIECEWISE"
fi

vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --gpu-memory-utilization "${MEM_FRACTION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-chunked-prefill \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --no-enable-prefix-caching \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee /workspace/logs/standalone_vllm.log
