#!/bin/bash
set -euo pipefail

export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export ATOM_HOST_IP=10.24.112.184
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --enable-dp-attention \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"10.24.112.184","handshake_port":6301}'
