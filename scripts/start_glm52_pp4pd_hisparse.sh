#!/bin/bash
# GLM-5.2 PP4×TP1 Prefill + TP4 Decode (PD) with HiSparse ENABLED on decode.
#
# Same topology as start_glm52_pp4pd.sh, but the DECODE node runs:
#   - --level 3 CUDAGraph: the fused GPU hot path (ATOM_HISPARSE_FUSED=1) is
#     sync-free and reads fixed-address metadata, so it captures/replays.
#   - small --max-num-seqs / --max-model-len: the cold pool is sized
#     max_num_seqs × max_model_len × num_layers × 576B; defaults would OOM.
#   - ATOM_HISPARSE_ENABLE=1 (decode node only).
#
# Prefill is unchanged (HiSparse is a decode-side overlay). The full GPU KV cache
# is still allocated on decode (Phase 0), so correctness holds even where the
# overlay falls back.
#
# Usage:
#   docker exec -it atom_pp4pd_test bash scripts/start_glm52_pp4pd_hisparse.sh

set -euo pipefail

MODEL=/mnt/models/GLM-5.2-MXFP4
PREFILL_PORT=8010
DECODE_PORT=8020
MESH_PORT=30000
HANDSHAKE_PORT=6301

# ── HiSparse decode-node knobs (keep small to bound the CPU cold pool) ──
# Overridable via env: HS_MAX_NUM_SEQS / HS_MAX_MODEL_LEN / HS_HOT_BUFFER_SIZE.
HS_MAX_NUM_SEQS=${HS_MAX_NUM_SEQS:-16}
HS_MAX_MODEL_LEN=${HS_MAX_MODEL_LEN:-4096}
HS_HOT_BUFFER_SIZE=${HS_HOT_BUFFER_SIZE:-8192}   # must be >= index_topk (2048)
# Stage D: RDMA KV straight into the paged host cold pool (skip GPU->cold copy).
HS_RDMA_DIRECT=${HS_RDMA_DIRECT:-0}

# ── cleanup ──────────────────────────────────────────────────────────
pkill -f openai_server 2>/dev/null || true
pkill -f atomesh 2>/dev/null || true
sleep 2
rm -rf /root/.cache/atom/*

# ── prefill: PP4×TP1 on GPU 0-3 (unchanged) ─────────────────────────
echo ">>> Starting prefill (PP4×TP1) on GPU 0-3 ..."
AITER_LOG_LEVEL=WARNING \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
VLLM_PP_LAYER_PARTITION=18,20,20,20 \
HIP_VISIBLE_DEVICES=0,1,2,3 \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$PREFILL_PORT" --trust-remote-code \
  -pp 4 -tp 1 --level 3 --enforce-eager \
  --max-num-batched-tokens 8192 \
  --kv_cache_dtype fp8 --gpu-memory-utilization 0.85 \
  --enable_prefix_caching \
  --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"handshake_port\":$HANDSHAKE_PORT,\"proxy_ip\":\"127.0.0.1\"}" \
  > /tmp/prefill.log 2>&1 &

# ── decode: TP4 on GPU 4-7, CUDAGraph (level 3) + HiSparse ──────────
echo ">>> Starting decode (TP4, level 3 CUDAGraph, HiSparse) on GPU 4-7 ..."
AITER_LOG_LEVEL=WARNING \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
ATOM_HISPARSE_ENABLE=1 \
ATOM_HISPARSE_HOT_BUFFER_SIZE=$HS_HOT_BUFFER_SIZE \
ATOM_HISPARSE_RDMA_DIRECT=$HS_RDMA_DIRECT \
HIP_VISIBLE_DEVICES=4,5,6,7 \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$DECODE_PORT" --trust-remote-code \
  -tp 4 --level 3 \
  --max-num-seqs $HS_MAX_NUM_SEQS --max-model-len $HS_MAX_MODEL_LEN \
  --kv_cache_dtype fp8 --gpu-memory-utilization 0.85 \
  --enable_prefix_caching \
  --kv-transfer-config "{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"handshake_port\":$HANDSHAKE_PORT,\"proxy_ip\":\"127.0.0.1\"}" \
  > /tmp/decode.log 2>&1 &

# ── wait for both servers ────────────────────────────────────────────
echo ">>> Waiting for prefill server (port $PREFILL_PORT) ..."
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PREFILL_PORT/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "    prefill READY"; break; }
  sleep 5
done

echo ">>> Waiting for decode server (port $DECODE_PORT) ..."
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$DECODE_PORT/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "    decode READY"; break; }
  sleep 5
done

# ── mesh proxy ───────────────────────────────────────────────────────
echo ">>> Starting mesh proxy on port $MESH_PORT ..."
nohup atomesh launch --pd-disaggregation \
  --prefill "http://127.0.0.1:$PREFILL_PORT" "$HANDSHAKE_PORT" \
  --decode "http://127.0.0.1:$DECODE_PORT" \
  --backend atom --port "$MESH_PORT" --log-level info \
  > /tmp/mesh_dsa.log 2>&1 &

echo ">>> Waiting for mesh (port $MESH_PORT) ..."
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$MESH_PORT/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "    mesh READY"; break; }
  sleep 2
done

echo ""
echo "=== All services up (HiSparse ENABLED on decode) ==="
echo "  prefill log: /tmp/prefill.log"
echo "  decode  log: /tmp/decode.log   (grep HiSparseCoordinator to confirm alloc)"
echo "  mesh    log: /tmp/mesh_dsa.log"
echo "  mesh API:    http://127.0.0.1:$MESH_PORT/v1/chat/completions"
