#!/usr/bin/env bash
set -euo pipefail

# Launch a standalone (non-disaggregated) SGLang server on a single node.
#
# Required env:
#   MODEL_PATH      - model path (e.g. /mnt/models/deepseek-ai/DeepSeek-R1)
#   TP              - tensor parallel size (e.g. 8)
#
# Optional env (with defaults):
#   PORT=8000  MEM_FRACTION=0.85  KV_CACHE_DTYPE=fp8_e4m3
#   CHUNKED_PREFILL_SIZE=16384  MAX_RUNNING_REQUESTS=128
#   CUDA_GRAPH_BS_START=1  CUDA_GRAPH_BS_END=64
#   LOAD_DUMMY=1            skip weight loading (for benchmarking infra)

: "${MODEL_PATH:?}"
: "${TP:?}"

PORT="${PORT:-8000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
CUDA_GRAPH_BS_START="${CUDA_GRAPH_BS_START:-1}"
CUDA_GRAPH_BS_END="${CUDA_GRAPH_BS_END:-64}"
LOAD_DUMMY="${LOAD_DUMMY:-}"

GPU_IDS=$(seq -s, 0 $((TP - 1)))

echo "[standalone] MODEL=${MODEL_PATH} TP=${TP} GPUs=${GPU_IDS} port=${PORT}"

mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${GPU_IDS}
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}

EXTRA_ARGS=()
if [[ -n "${LOAD_DUMMY}" ]]; then
    EXTRA_ARGS+=(--load-format dummy)
    export LOAD_DUMMY=1
    echo "[standalone] LOAD_DUMMY enabled — skipping weight loading"
fi

TORCHINDUCTOR_COMPILE_THREADS=128 python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 --port "${PORT}" \
    --trust-remote-code \
    --tp-size "${TP}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --attention-backend aiter \
    --mem-fraction-static "${MEM_FRACTION}" \
    --page-size 1 \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --cuda-graph-bs $(seq "${CUDA_GRAPH_BS_START}" "${CUDA_GRAPH_BS_END}") \
    --log-level info \
    --watchdog-timeout 3600 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee /workspace/logs/standalone.log
