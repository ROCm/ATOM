#!/bin/bash
# MiniMax-M3-MXFP4 agentic server, TP4 + EAGLE3, per ATOM recipes/MiniMax-M3-Agentic-InferenceX.md
# (the MI355X recipe); on gfx1250 (MI455) adds the env from the MI455 baseline (LLM-Booster #201).
# Usage (in container): CONC=10 bash /scripts/agentic/server.sh <log>
#   ARCH=gfx950|gfx1250  (auto from rocminfo)   GPUS=0,1,4,5  (ROCR_VISIBLE_DEVICES; 8-GPU MI355X nodes)
#   MONO=0|1  (ATOM_MONO_ENABLE, gfx950 only; unset = ATOM default)   MAX_SEQS (default 2*CONC)   GPU_MEM (0.95)
#   API_PORT (8890)   ENGINE_PORT (8896, ATOM --port: the TP group's MASTER_PORT; a second server on the node needs others)
set -u
CONC=${CONC:?set CONC}
LOG=${1:-/logs/agentic/server_c${CONC}.log}
MODEL=${MODEL:-/models/MiniMax-M3-MXFP4}
DRAFT=${DRAFT:-/models/MiniMax-M3-EAGLE3-GQA}
KV=${KV:-fp8}
ARCH=${ARCH:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')}
case "$ARCH" in gfx950|gfx1250) ;; *) echo "unsupported ARCH='$ARCH', set ARCH=gfx950|gfx1250"; exit 1;; esac

cd /app/ATOM
rm -rf /root/.cache/atom/*

envs=()
[ "$ARCH" = gfx1250 ] && envs+=(
  PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 HSA_OVERRIDE_GFX_VERSION=12.5.0
  ATOM_MOE_GU_ITLV=1 AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1)
[ -n "${GPUS:-}" ] && envs+=(ROCR_VISIBLE_DEVICES=$GPUS)
[ -n "${MONO:-}" ] && envs+=(ATOM_MONO_ENABLE=$MONO)
# TG=1: MiniMax-M3 prefill token graphs; TG_SIZES overrides the bucket list (ATOM default 256,512,1024,2048,3072).
[ "${TG:-0}" = 1 ] && envs+=(ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH=1)
[ -n "${TG_SIZES:-}" ] && envs+=(ATOM_TOKEN_GRAPH_SIZES=$TG_SIZES)
envs+=(
  NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1
  AITER_QUICK_REDUCE_QUANTIZATION=${QR:-INT4} AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
  ATOM_FORCE_ATTN_TRITON=1 AITER_LOG_LEVEL=WARNING ATOM_GC_THRESHOLD=20000,50,50
  ATOM_PA_FLYDSL=${FLY:-1}
)
args=()
# ACC=1: accuracy run, real draft acceptance. The fixed rate only holds for perf runs.
[ "${ACC:-0}" = 1 ] || args+=(--spec-decode-acceptance-rate 0.5933)
# PROFILE_DIR set: torch profiler armed (driven by /start_profile, /stop_profile), step annotations
# carry bs/tok/ctx fields for atom-trace-layer-breakdown; no shapes or stacks, as in LLM-Booster #201.
if [ -n "${PROFILE_DIR:-}" ]; then
  envs+=(ATOM_ENABLE_DETAILED_ANNOTATION=1 ATOM_PROFILER_MORE=0 ATOM_TORCH_PROFILER_DIR="$PROFILE_DIR")
  args+=(--torch-profiler-dir "$PROFILE_DIR")
  # Nodes whose roctracer GPU timestamps run ahead of the host clock lose nearly all kernel records
  # in kineto; roctracer_shift.so moves them back (ROCTRACER_SHIFT=0 disables the check).
  if [ "${ROCTRACER_SHIFT:-auto}" != 0 ]; then
    shift_ns=$(CALIB_GPUS=${GPUS:-0,1,2,3} bash /scripts/agentic/roctracer_calib.sh)
    [ "${shift_ns:-0}" != 0 ] && envs+=(LD_PRELOAD=/scripts/agentic/libroctracer_shift.so ROCTRACER_TS_SHIFT_NS=$shift_ns)
  fi
fi
# Indexer CP from CONC >= 15, LMCache CPU tier + SLRU from CONC >= 40, as in the recipe.
# On 8-GPU MI355X nodes the recipe requires GPUS=0,1,4,5 with the CPU tier (NUMA split);
# a 4-GPU MI455 node needs nothing.
[ "$CONC" -ge 15 ] && envs+=(ATOM_M3_INDEXER_CP=1)
if [ "$CONC" -ge 40 ] && [ "${OFFLOAD:-1}" = 1 ]; then
  envs+=(PYTHONHASHSEED=0 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=256 LMCACHE_CHUNK_SIZE=256
    ATOM_PREFIX_CACHE_POLICY=slru ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5
    LMCACHE_CACHE_POLICY=ATOM_SLRU LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3)
  args+=(--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}')
fi

echo "ARCH=$ARCH CONC=$CONC KV=$KV ACC=${ACC:-0} TG=${TG:-0} env: ${envs[*]}" > "$LOG"
nohup env "${envs[@]}" python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "${ENGINE_PORT:-8896}" --server-port "${API_PORT:-8890}" \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --kv_cache_dtype "$KV" \
  --gpu-memory-utilization ${GPU_MEM:-0.95} \
  --block-size 128 \
  --max-num-batched-tokens 32768 \
  --attn-prefill-chunk-size 16384 \
  --max-num-seqs ${MAX_SEQS:-$((2 * CONC))} \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}' \
  --enable-prefix-caching \
  --method eagle3 \
  --draft-model "$DRAFT" \
  --num-speculative-tokens 3 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]' \
  "${args[@]}" >> "$LOG" 2>&1 &
