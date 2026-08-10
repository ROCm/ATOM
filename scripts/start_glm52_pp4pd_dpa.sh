#!/bin/bash
# GLM-5.2 PP4×TP1 Prefill (CPP) + DP-Attention Decode (PD disaggregation via mooncake)
#
# Container: atom_pp4pd_test
# GPUs:      0-3 = prefill (PP4×TP1), 4-7 = decode (DPA: dp4 × tp1)
# Ports:     8010 = prefill API, 8020 = decode API, 30000 = mesh proxy
#
# Usage:
#   docker exec -it atom_pp4pd_test bash /it-share/yajizhan/code/ATOM/scripts/start_glm52_pp4pd_dpa.sh
#
# Derived from start_glm52_pp4pd.sh; the only structural change is that the
# decode node runs `-tp 4 --enable-dp-attention`, which CoreManager expands to
# dp=4 / tp=1 (four independent EngineCores, one request lives entirely on one
# rank).  Prefill is unchanged.
#
# Both nodes use the same legacy 576-wide MLA KV layout, so the mooncake
# transfer needs no changes: DPA decode (tp=1 per rank) and PP prefill (tp=1)
# register the same [kv_cache x L] + [index_cache x L] region groups with the
# same bytes-per-block.  RDMA traffic drops 4x versus the TP4 decode node,
# which needed one copy per TP rank.
#
# ── PRECONDITION ─────────────────────────────────────────────────────────
#
# atom/model_ops/attentions/aiter_mla.py still carries the builder half of the
# reverted DS32 KV format (use_ds32 at :171 plus a dangling import of
# _DS32_CACHE_BYTES at :33, which no longer exists anywhere).  Today that is an
# ImportError for the whole aiter MLA backend; and if only the import were
# patched, a DPA node would allocate the 3-tensor cache and register 3 region
# groups per layer while MLAAttention writes the 576-wide one — silent KV
# corruption over RDMA.  Strip those 6 sites first.
# See docs/pp4_prefill_dpa_decode_research.md §1.

set -euo pipefail

# ── precondition check ───────────────────────────────────────────────────
if python -c "import atom.model_ops.attentions.aiter_mla" 2>&1 | grep -q "_DS32_CACHE_BYTES"; then
  cat >&2 <<'EOF'
REFUSING TO START: atom/model_ops/attentions/aiter_mla.py still imports
_DS32_CACHE_BYTES, which was removed together with the DS32 KV format. The aiter
MLA backend cannot import, and the leftover use_ds32 branches would give the DPA
decode node a different KV cache layout than the PP prefill node.

Remove the DS32 leftovers first (see docs/pp4_prefill_dpa_decode_research.md §1).
EOF
  exit 1
fi

MODEL=/mnt/models/GLM-5.2-MXFP4
PREFILL_PORT=8010
DECODE_PORT=8020
MESH_PORT=30000
HANDSHAKE_PORT=6301

# ONLINE_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}'

# ── cleanup ──────────────────────────────────────────────────────────
pkill -f openai_server 2>/dev/null || true
pkill -f atomesh 2>/dev/null || true
sleep 2
rm -rf /root/.cache/atom/*

# ── prefill: PP4×TP1 on GPU 0-3 (unchanged from start_glm52_pp4pd.sh) ─
echo ">>> Starting prefill (PP4×TP1) on GPU 0-3 ..."
AITER_LOG_LEVEL=WARNING \
ATOM_LOG_PREFIX_CACHE_PER_REQ=1 \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
VLLM_PP_LAYER_PARTITION=18,20,20,20 \
HIP_VISIBLE_DEVICES=0,1,2,3 \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$PREFILL_PORT" --trust-remote-code \
  -pp 4 -tp 1 --level 3 --enforce-eager \
  --max-num-batched-tokens 8192 \
  --kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85 \
  --enable_prefix_caching \
  --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"handshake_port\":$HANDSHAKE_PORT,\"proxy_ip\":\"127.0.0.1\"}" \
  > /tmp/prefill.log 2>&1 &

# ── decode: DPA (dp4 × tp1) on GPU 4-7 ───────────────────────────────
#
# `-tp 4 --enable-dp-attention` => CoreManager sets dp_size=4, tp_size=1 and
# spawns 4 EngineCores (engine_core_mgr.py:104). Per-rank knobs below are
# therefore PER DP RANK, not per node: --max-num-seqs 128 means 512 in flight
# across the decode node. Same for the CUDA-graph capture sizes.
#
# --dp-load-balance least_requests is the default; kept explicit because request
# skew across DP ranks directly costs throughput (idle ranks still run dummy
# batches to keep the per-step all_reduce in lockstep).
#
# Not passing --enable-expert-parallel: under DPA the MoE is sharded across the
# flattened DP×TP device space (moe.py:141), which is what the GLM-5.2 recipe
# validated. Add it only if you want EP + mori all2all.
#
# Do NOT set ATOM_MLA_PAGE_SIZE > 1: is_persistent_mode() drops out of persistent
# mode when page_size > 1, and GLM-5.2 DPA decode needs persistent mode because
# AITER's fp8-Q/fp8-KV GQA64 kernel only exists there (attention_mla.py:224).
echo ">>> Starting decode (DPA dp4×tp1) on GPU 4-7 ..."
AITER_LOG_LEVEL=WARNING \
ATOM_LOG_PREFIX_CACHE_PER_REQ=1 \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
HIP_VISIBLE_DEVICES=4,5,6,7 \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$DECODE_PORT" --trust-remote-code \
  -tp 4 --enable-dp-attention \
  --level 3 --cudagraph-mode FULL \
  --cudagraph-capture-sizes "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]" \
  --max-num-seqs 128 \
  --dp-load-balance least_requests \
  --kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85 \
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
# --dp-aware makes mesh probe each worker's dp_size (/server_info, falling back
# to ATOM's /kv_transfer_info) and expand one URL into one logical worker per DP
# rank, addressed as <url>@<rank>. Forwarding then injects data_parallel_rank
# into the body, which api_server.py:450 turns into a hard pin onto that
# EngineCore. Decode reports dp_size=4, so it becomes @0..@3; prefill reports
# dp_size=1, so it stays a single @0 worker and the injected rank is ignored by
# the PP path (add_request sends to stage 0 whenever pp_size > 1).
#
# --decode-policy dp_sticky then pins each X-Session-ID to one decode rank, so a
# multi-turn session comes back to the rank that already holds its prefix (each
# DP rank has its own block pool and prefix tree). Prefill keeps the default
# policy — stickiness is meaningless for a PP node.
#
# The client MUST send the X-Session-ID header for this to do anything. Without
# it dp_sticky falls back to mesh's own lowest-load counter and also bypasses
# ATOM's least_requests, which is strictly worse than leaving both off.
#
# Not enabling AtomPdRankMappingPolicy::Idx2Idx (default is None): it maps
# prefill rank N to decode rank N, and our prefill has dp_size=1, so decode
# ranks 1-3 would just log a skip warning per request.
echo ">>> Starting mesh proxy on port $MESH_PORT (DP-aware + dp_sticky decode) ..."
nohup atomesh launch --pd-disaggregation \
  --prefill "http://127.0.0.1:$PREFILL_PORT" "$HANDSHAKE_PORT" \
  --decode "http://127.0.0.1:$DECODE_PORT" \
  --dp-aware --decode-policy dp_sticky \
  --backend atom --port "$MESH_PORT" --log-level info \
  > /tmp/mesh_dpa.log 2>&1 &

echo ">>> Waiting for mesh (port $MESH_PORT) ..."
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$MESH_PORT/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "    mesh READY"; break; }
  sleep 2
done

echo ""
echo "=== All services up ==="
echo "  prefill log: /tmp/prefill.log"
echo "  decode  log: /tmp/decode.log"
echo "  mesh    log: /tmp/mesh_dpa.log"
echo "  mesh API:    http://127.0.0.1:$MESH_PORT/v1/chat/completions"
