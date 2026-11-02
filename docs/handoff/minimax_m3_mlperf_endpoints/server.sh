#!/bin/bash
# server.sh <log> (in the container): MiniMax-M3-MXFP4 TP4 + EAGLE3 for MLPerf Endpoints, as the agentic recipe
# (ATOM recipes/MiniMax-M3-Agentic-InferenceX.md) plus the MI455 (gfx1250) env of LLM-Booster #201. Differences from
# the AgentX server (../minimax_m3_prefill_token_graph/scripts/server.sh): EAGLE3 always runs its real acceptance -- a
# fixed rate changes the output, which MLPerf cannot use -- and there is no profiler and no LMCache tier.
# Env (run_point.sh passes them): CONC, required; API_PORT (8890); MODEL, DRAFT (directory names under /models);
#   MAX_SEQS (2 x CONC); GPU_MEM (0.95); KV (fp8); TG (0; 1 = MiniMax-M3 prefill token graphs); TG_SIZES (buckets).
set -u
CONC=${CONC:?set CONC}
LOG=${1:?usage: server.sh <log>}
MODEL=/models/${MODEL:-MiniMax-M3-MXFP4}
DRAFT=/models/${DRAFT:-MiniMax-M3-EAGLE3-GQA}
ARCH=$(rocminfo 2> /dev/null | grep -m1 -oE 'gfx[0-9a-f]+')

cd /app/ATOM
rm -rf /root/.cache/atom/*

envs=(NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1
  AITER_QUICK_REDUCE_QUANTIZATION=INT4 AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
  ATOM_FORCE_ATTN_TRITON=1 AITER_LOG_LEVEL=WARNING ATOM_GC_THRESHOLD=20000,50,50 ATOM_PA_FLYDSL=1)
[ "$ARCH" = gfx1250 ] && envs+=(PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 HSA_OVERRIDE_GFX_VERSION=12.5.0
  ATOM_MOE_GU_ITLV=1 AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1)
# The recipe's indexer context parallelism from c15.
[ "$CONC" -ge 15 ] && envs+=(ATOM_M3_INDEXER_CP=1)
[ "${TG:-0}" = 1 ] && envs+=(ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH=1)
[ -n "${TG_SIZES:-}" ] && envs+=(ATOM_TOKEN_GRAPH_SIZES=$TG_SIZES)

echo "ARCH=$ARCH CONC=$CONC KV=${KV:-fp8} TG=${TG:-0} env: ${envs[*]}" > "$LOG"
nohup env "${envs[@]}" python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 0.0.0.0 --port 8896 --server-port "${API_PORT:-8890}" \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --kv_cache_dtype "${KV:-fp8}" \
  --gpu-memory-utilization "${GPU_MEM:-0.95}" \
  --block-size 128 \
  --max-num-batched-tokens 32768 \
  --attn-prefill-chunk-size 16384 \
  --max-num-seqs "${MAX_SEQS:-$((2 * CONC))}" \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}' \
  --enable-prefix-caching \
  --method eagle3 \
  --draft-model "$DRAFT" \
  --num-speculative-tokens 3 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]' \
  >> "$LOG" 2>&1 &
