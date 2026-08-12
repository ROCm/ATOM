#!/bin/bash
# GLM-5.2 PP standalone (no PD) — parametrized for shared-indexer PP-split tests.
#
# The cross-PP-boundary sparse top-k transfer is unconditional; any PP partition
# is valid (a rank may start on a "shared" layer). This script just varies the
# partition/stage count for accuracy checks.
#
# Env overrides:
#   PP_SIZE    pipeline stages           (default 4)
#   PARTITION  VLLM_PP_LAYER_PARTITION   (default 18,20,20,20)
#   GPUS       HIP_VISIBLE_DEVICES       (default 0,1,2,3)
#   PORT       server port               (default 8010)
#
# Scenarios:
#   Full-start    : PP_SIZE=4 PARTITION=18,20,20,20              (every rank starts "full")
#   Shared-split  : PP_SIZE=4 PARTITION=20,20,20,18             (ranks start "shared")
#   PP8 balanced  : PP_SIZE=8 PARTITION=10,10,10,10,10,10,10,8 GPUS=0,1,2,3,4,5,6,7
#
# Usage (inside container atom_pp4pd_test):
#   PP_SIZE=8 PARTITION=10,10,10,10,10,10,10,8 GPUS=0,1,2,3,4,5,6,7 \
#     bash /it-share/yajizhan/code/ATOM/scripts/start_glm52_pp_indexer_test.sh

set -euo pipefail

MODEL=/mnt/models/GLM-5.2-MXFP4
PP_SIZE=${PP_SIZE:-4}
PARTITION=${PARTITION:-18,20,20,20}
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-8010}
LOG=/tmp/pp_indexer_test_pp${PP_SIZE}.log

# ── cleanup ──────────────────────────────────────────────────────────
pkill -f openai_server 2>/dev/null || true
sleep 2
rm -rf /root/.cache/atom/*

echo ">>> PP${PP_SIZE}xTP1 partition=${PARTITION} GPUS=${GPUS}"
AITER_LOG_LEVEL=WARNING \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
VLLM_PP_LAYER_PARTITION=${PARTITION} \
HIP_VISIBLE_DEVICES=${GPUS} \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$PORT" --trust-remote-code \
  -pp "$PP_SIZE" -tp 1 --level 3 --enforce-eager \
  --max-num-batched-tokens 8192 \
  --kv_cache_dtype fp8 --gpu-memory-utilization 0.85 \
  --enable_prefix_caching \
  > "$LOG" 2>&1 &

echo ">>> Waiting for server (port $PORT) ..."
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "    server READY"; break; }
  sleep 5
done

echo ""
echo "=== Server up ==="
echo "  log: $LOG"
echo "  grep 'PP shared-indexer transfer' $LOG   # confirm which ranks transfer"
echo "  API: http://127.0.0.1:$PORT/v1/completions"
