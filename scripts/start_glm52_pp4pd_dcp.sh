#!/bin/bash
# GLM-5.2 PP4×TP1 Prefill (CPP) + DCP Decode (PD disaggregation via mooncake)
#
# Container: atom_pp4pd_dcp
# GPUs:      0-3 = prefill (PP4×TP1), 4-7 = decode (TP4 × DCP4)
# Ports:     8010 = prefill API, 8020 = decode API, 30000 = mesh proxy
#
# Usage:
#   docker exec -it atom_pp4pd_dcp bash /it-share/yajizhan/code/ATOM/scripts/start_glm52_pp4pd_dcp.sh
#
# Derived from start_glm52_pp4pd_dpa.sh. The decode node swaps
# `--enable-dp-attention` (dp4 × tp1, one request wholly on one rank) for
# `-tp 4 -dcp 4` (one request's KV interleaved across all four ranks). DCP reuses
# the TP GPUs, so world size is unchanged and the two flags are mutually
# exclusive. Prefill is unchanged except for the MTP gating below.
#
# The KV a prefill rank holds is whole blocks; the KV a DCP decode rank holds is
# an interleaved shard of each block. The mooncake producer relayouts on the way
# out (dcp_plan.plan_sharded), driven by the three DCP fields the consumer ships
# in its write_request, so nothing here needs to configure the transfer itself.
#
# Layer split is 20,20,20,18 rather than the usual 18,20,20,20: with --method mtp
# the drafter is built only on the last PP rank, so stage 3 also carries MTP layer
# 78 and runs three draft forwards after its own slice.  Giving it the short slice
# keeps the per-stage times even, which is what the pipeline streams on.  Uneven
# splits stopped needing full-layer alignment as of efa37fc3 (DSA shared-indexer
# top-k now crosses PP boundaries).
set -euo pipefail

MODEL=/mnt/models/GLM-5.2-MXFP4
PREFILL_PORT=8010
DECODE_PORT=8020
MESH_PORT=30000
HANDSHAKE_PORT=6301

# Everything but the MoE experts, the router gate and the embeddings is
# quantized to per-token-per-channel fp8 while the checkpoint loads. Both nodes
# take it: the KV they exchange is fp8 either way, but the two model instances
# have to agree on weights or the decode continuation drifts from the prefill.
ONLINE_QUANT_CONFIG='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}'

# DCP KV interleave granularity S: global token i lives on decode rank
# (i // S) % dcp. S=1 is token-level round-robin; S=block_size makes each decode
# rank own contiguous 16-token runs, which collapses the producer's RDMA
# descriptors from one-per-token-run to one-per-block.
#
# S > 1 forbids speculative decode (config.py: the q>1 verify cprr MLA kernel
# assumes token-level interleave), so ENABLE_MTP only has an effect at S=1.
# MTP is all-or-nothing across the pair, never per node: the drafter adds a KV
# layer on prefill's last PP stage, so a node running it registers a different
# number of region groups and _consumer_region_map rejects every transfer.
DCP_SIZE="${DCP_SIZE:-4}"
DCP_INTERLEAVE="${DCP_INTERLEAVE:-1}"

# Keep the MLA KV sharded but give every decode rank the whole indexer page of
# each virtual block, which lets decode skip the indexer candidate all-gather
# and the global merge that follows it. Costs HBM: the index cache goes from
# 1/dcp to full on every rank, so the KV+index footprint per token rises about
# a fifth and the block pool shrinks to match.
#
# DECODE ONLY. The startup check reads the bare environment variable rather
# than the dcp>1-gated helper, so a prefill rank that saw this would abort with
# "decode context parallel size must be > 1". Never export it for both nodes.
REPLICATE_INDEX_CACHE="${REPLICATE_INDEX_CACHE:-1}"

# On by default: this script serves the throughput runs, and MTP is part of the
# configuration being measured. Set ENABLE_MTP=0 to isolate a DCP change from
# the draft KV layer and the q>1 verify path.
ENABLE_MTP="${ENABLE_MTP:-1}"
if [ "$ENABLE_MTP" -eq 1 ] && [ "$DCP_INTERLEAVE" -ne 1 ]; then
  echo "ENABLE_MTP=1 needs DCP_INTERLEAVE=1 (speculative decode requires token-level interleave)" >&2
  exit 1
fi

# Decode is one scheduler for the whole node now, not four. The DPA baseline ran
# --max-num-seqs 128 per DP rank = 512 in flight node-wide; keep that number so
# throughput comparisons are apples-to-apples. Graphs are captured up to 256 --
# the same ladder the CI case uses -- and batches above that run eager.
DECODE_MAX_SEQS="${DECODE_MAX_SEQS:-512}"
DECODE_CAPTURE_SIZES="${DECODE_CAPTURE_SIZES:-[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]}"

# Nominal per-stage CPU offload budget. The connector redistributes the total
# (pp_size * this) across stages by layer count so every stage caches the same
# number of tokens — see offload/config.py:scale_cpu_size_for_pp. At pp4 this
# is 4 * 256 = 1TB of host RAM, pre-allocated at startup.
LMCACHE_CPU_SIZE="${LMCACHE_CPU_SIZE:-256.0}"
LMCACHE_CHUNK="${LMCACHE_CHUNK:-256}"

# Force a fixed MTP acceptance rate so throughput numbers are comparable across
# runs instead of tracking whatever the draft head happens to agree on for the
# replayed traffic. The rejection sampler force-accepts each draft token with a
# position-decaying probability calibrated to this mean; equivalent accept
# length = 1 + 3 * rate. Set on both nodes to keep the speculative configs
# identical (the sampler only fires where decoding happens).
SPEC_ACCEPT_RATE="${SPEC_ACCEPT_RATE:-0.6633}"

# Set SPEC_ACCEPT_RATE=off to drop the flag and let the MTP head decide each
# draft token on its own. Accuracy runs need this: the synthetic rate accepts
# drafts the target never agreed with, so generated text is not the model's.
SPEC_ARGS=()
if [ "$ENABLE_MTP" -eq 1 ]; then
  SPEC_ARGS=(--method mtp --num-speculative-tokens 3)
  if [ -n "$SPEC_ACCEPT_RATE" ] && [ "$SPEC_ACCEPT_RATE" != "off" ]; then
    SPEC_ARGS+=(--spec-decode-acceptance-rate "$SPEC_ACCEPT_RATE")
  fi
fi

# One MiB under the 2048 default. The indexer skips row-chunking when the whole
# logits buffer already fits the budget, so a budget of exactly 2 GiB lets an
# 8192-row chunk against 65536 KV tokens allocate exactly 2 GiB. aiter's
# fp8_mqa_logits picks its store path with a strict `bytes < 2 GiB`, so that one
# shape falls to the plain global-store gluon kernel, which fails to compile on
# gfx950 (LLVM iota_range assertion) and aborts the PP stage. Only exact
# 8192x65536 hits it -- every other size chunks down and stays on the buffer
# path. 2047 keeps the buffer strictly under 2 GiB for all shapes.
SPARSE_INDEXER_LOGITS_BUDGET_MB="${SPARSE_INDEXER_LOGITS_BUDGET_MB:-2047}"

# ── cleanup ──────────────────────────────────────────────────────────
pkill -f openai_server 2>/dev/null || true
pkill -f atomesh 2>/dev/null || true
sleep 2
rm -rf /root/.cache/atom/*

# ── prefill: PP4×TP1 on GPU 0-3 (mooncake producer + lmcache offload) ─
echo ">>> Starting prefill (PP4×TP1) on GPU 0-3 [mooncake + lmcache] ..."
AITER_LOG_LEVEL=WARNING \
ATOM_LOG_PREFIX_CACHE_PER_REQ=1 \
ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB="$SPARSE_INDEXER_LOGITS_BUDGET_MB" \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
VLLM_PP_LAYER_PARTITION=20,20,20,18 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE="$LMCACHE_CPU_SIZE" \
LMCACHE_CHUNK_SIZE="$LMCACHE_CHUNK" \
HIP_VISIBLE_DEVICES=0,1,2,3 \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$PREFILL_PORT" --trust-remote-code \
  -pp 4 -tp 1 --level 3 --enforce-eager \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  --max-num-batched-tokens 8192 \
  --kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85 \
  --enable_prefix_caching \
  --online_quant_config "$ONLINE_QUANT_CONFIG" \
  --kv-transfer-config "{\"kv_connector\":\"multi\",\"connectors\":[{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_producer\",\"handshake_port\":$HANDSHAKE_PORT,\"proxy_ip\":\"127.0.0.1\"},{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}]}" \
  > /tmp/prefill.log 2>&1 &

# ── decode: TP4 × DCP4 on GPU 4-7 ────────────────────────────────────
#
# One EngineCore for the node (no DP), so --max-num-seqs and the capture sizes
# are NODE-WIDE here, unlike the per-rank numbers in the DPA script.
#
# Each rank stores 1/dcp of the MLA KV and 1/dcp of the indexer cache, and
# attends over its own shard returning an LSE that reduce-scatter merges. The
# saving is not a flat 1/4 of decode HBM: weights and activations are untouched,
# so expect roughly the §2 estimate of 1/2.5 on the KV+index footprint.
#
# Do NOT set ATOM_MLA_PAGE_SIZE > 1: is_persistent_mode() drops out of persistent
# mode when page_size > 1, and GLM-5.2 decode needs persistent mode because
# AITER's fp8-Q/fp8-KV GQA64 kernel only exists there (attention_mla.py:224).
# MTP under DCP additionally needs the persistent cprr MLA kernel, gfx950 only.
echo ">>> Starting decode (TP4 x DCP$DCP_SIZE, interleave_size=$DCP_INTERLEAVE) on GPU 4-7 ..."
AITER_LOG_LEVEL=WARNING \
ATOM_LOG_PREFIX_CACHE_PER_REQ=1 \
ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB="$SPARSE_INDEXER_LOGITS_BUDGET_MB" \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
ATOM_DCP_REPLICATE_INDEX_CACHE="$REPLICATE_INDEX_CACHE" \
HIP_VISIBLE_DEVICES=4,5,6,7 \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$DECODE_PORT" --trust-remote-code \
  -tp 4 -dcp "$DCP_SIZE" --dcp-config "{\"interleave_size\": $DCP_INTERLEAVE}" \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  --level 3 --cudagraph-mode FULL \
  --cudagraph-capture-sizes "$DECODE_CAPTURE_SIZES" \
  --max-num-seqs "$DECODE_MAX_SEQS" \
  --kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85 \
  --enable_prefix_caching \
  --online_quant_config "$ONLINE_QUANT_CONFIG" \
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
# No --dp-aware / --decode-policy dp_sticky here: decode is a single logical
# worker again (one EngineCore, one block pool, one prefix tree), so there is no
# per-rank pin for a session to stick to and nothing for mesh to expand.
echo ">>> Starting mesh proxy on port $MESH_PORT ..."
nohup atomesh launch --pd-disaggregation \
  --prefill "http://127.0.0.1:$PREFILL_PORT" "$HANDSHAKE_PORT" \
  --decode "http://127.0.0.1:$DECODE_PORT" \
  --backend atom --port "$MESH_PORT" --log-level info \
  > /tmp/mesh_dcp.log 2>&1 &

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
echo "  mesh    log: /tmp/mesh_dcp.log"
echo "  mesh API:    http://127.0.0.1:$MESH_PORT/v1/chat/completions"
echo "  DCP:         tp4 x dcp${DCP_SIZE}, interleave_size=${DCP_INTERLEAVE}, max-num-seqs=${DECODE_MAX_SEQS} (node-wide)"
echo "  LMCache:     CPU=${LMCACHE_CPU_SIZE}GB  chunk=${LMCACHE_CHUNK}"
echo "  Index cache: $([ "$REPLICATE_INDEX_CACHE" = "1" ] && echo "replicated on every DCP rank" || echo "sharded across DCP ranks")"
if [ "$ENABLE_MTP" -eq 1 ]; then
  echo "  MTP accept:  ${SPEC_ACCEPT_RATE}"
else
  echo "  MTP:         off"
fi
echo "  Verify:      grep -i 'lmcache\|offload' /tmp/prefill.log"
