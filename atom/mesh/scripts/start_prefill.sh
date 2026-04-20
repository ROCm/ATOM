#!/usr/bin/env bash
set -euo pipefail

# Launch the prefill server. Run this script inside the container.
#
# Required env:
#   PREFILL_IP      - this node's IP
#   MODEL_PATH      - model path
#   PREFILL_TP      - tensor parallel size
#
# Optional env (with defaults):
#   PREFILL_PORT=8010  BOOTSTRAP_PORT=8998  MEM_FRACTION=0.85
#   KV_CACHE_DTYPE=fp8_e4m3  CHUNKED_PREFILL_SIZE=16384
#   MAX_RUNNING_REQUESTS=128  IB_DEVICE=rdma0,...

: "${PREFILL_IP:?}"
: "${MODEL_PATH:?}"
: "${PREFILL_TP:?}"

PREFILL_PORT="${PREFILL_PORT:-8010}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
IB_DEVICE="${IB_DEVICE:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"

GPU_IDS=$(seq -s, 0 $((PREFILL_TP - 1)))

echo "[prefill] IP=${PREFILL_IP} TP=${PREFILL_TP} GPUs=${GPU_IDS} port=${PREFILL_PORT}"

mkdir -p /workspace/logs
cat > /workspace/mooncake_prefill.json <<EOF
{"prefill_url": "${PREFILL_IP}:${PREFILL_PORT}", "protocol": "rdma"}
EOF

export HIP_VISIBLE_DEVICES=${GPU_IDS}
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export SGLANG_HOST_IP=${PREFILL_IP}
export MOONCAKE_CONFIG_PATH=/workspace/mooncake_prefill.json
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 --port "${PREFILL_PORT}" \
    --trust-remote-code \
    --tp-size "${PREFILL_TP}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --attention-backend aiter \
    --mem-fraction-static "${MEM_FRACTION}" \
    --page-size 1 \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --disable-radix-cache \
    --log-level warning \
    --watchdog-timeout 3600 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --disaggregation-ib-device "${IB_DEVICE}" \
    2>&1 | tee /workspace/logs/prefill.log
