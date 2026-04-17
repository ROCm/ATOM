#!/usr/bin/env bash
set -euo pipefail

# Required env:
#   CONTAINER       - docker container name
#   DECODE_IP       - this node's IP
#   MODEL_PATH      - model path inside container
#   DECODE_TP       - tensor parallel size
#
# Optional env (with defaults):
#   DECODE_PORT=8020  BOOTSTRAP_PORT=8998  MEM_FRACTION=0.85
#   KV_CACHE_DTYPE=fp8_e4m3  MAX_RUNNING_REQUESTS=128
#   CUDA_GRAPH_BS_START=1  CUDA_GRAPH_BS_END=64
#   IB_DEVICE=rdma0,...  WAIT_TIMEOUT=900

: "${CONTAINER:?}"
: "${DECODE_IP:?}"
: "${MODEL_PATH:?}"
: "${DECODE_TP:?}"

DECODE_PORT="${DECODE_PORT:-8020}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
CUDA_GRAPH_BS_START="${CUDA_GRAPH_BS_START:-1}"
CUDA_GRAPH_BS_END="${CUDA_GRAPH_BS_END:-64}"
IB_DEVICE="${IB_DEVICE:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"

GPU_IDS=$(seq -s, 0 $((DECODE_TP - 1)))

echo "[decode] IP=${DECODE_IP} TP=${DECODE_TP} GPUs=${GPU_IDS} port=${DECODE_PORT}"

docker exec "${CONTAINER}" bash -c "mkdir -p /workspace/logs"

docker exec -d "${CONTAINER}" bash -c "
    export HIP_VISIBLE_DEVICES=${GPU_IDS}
    export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
    export SGLANG_USE_AITER=1
    export SGLANG_AITER_FP8_PREFILL_ATTN=0
    export AITER_QUICK_REDUCE_QUANTIZATION=INT4
    export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
    export SGLANG_HOST_IP=${DECODE_IP}
    export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/mooncake:/opt/rocm/lib:\${LD_LIBRARY_PATH:-}

    TORCHINDUCTOR_COMPILE_THREADS=128 python3 -m sglang.launch_server \
        --model-path '${MODEL_PATH}' \
        --host 0.0.0.0 --port ${DECODE_PORT} \
        --trust-remote-code \
        --tp-size ${DECODE_TP} \
        --kv-cache-dtype ${KV_CACHE_DTYPE} \
        --mem-fraction-static ${MEM_FRACTION} \
        --page-size 1 \
        --max-running-requests ${MAX_RUNNING_REQUESTS} \
        --cuda-graph-bs \$(seq ${CUDA_GRAPH_BS_START} ${CUDA_GRAPH_BS_END}) \
        --disable-radix-cache \
        --log-level warning \
        --watchdog-timeout 3600 \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
        --disaggregation-ib-device ${IB_DEVICE} \
        >/workspace/logs/decode.log 2>&1
"

echo "[decode] waiting for server (timeout=${WAIT_TIMEOUT}s)..."
timeout "${WAIT_TIMEOUT}" bash -c "
    while ! docker exec '${CONTAINER}' curl -sf http://127.0.0.1:${DECODE_PORT}/v1/models >/dev/null 2>&1; do
        sleep 10
    done
"
echo "[decode] ready"
