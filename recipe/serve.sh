#!/bin/bash
# MiniMax-M3.1-Flash TP4 server for the max-perf effort. Run inside a container, under gpurun.sh:
#   docker exec -d [-e VAR=..] atom-megamoe bash /shared_nfs/ygan/max/bin/gpurun.sh 4 bash /shared_nfs/ygan/max/bin/serve.sh
# env: TAG (required) CONC (default 8; max-num-seqs = 2*CONC unless MAX_SEQS) PORT (default 8890)
#      REPO (default /shared_nfs/ygan/work/ATOM-max)  AL (synthetic DSpark acceptance length, default 4.04;
#      AL=real -> real draft acceptance, required for accuracy runs)  MEM (gpu-memory-utilization, 0.85)
#      DSPARK_K (draft length, default 7; vLLM's MI355X recipe = 7 below CONC 64, 3 at CONC >= 64; synthetic AL is the
#      golden value for that k: 7 -> 4.04, 3 -> 2.88, a numeric AL that disagrees is replaced)
#      EXTRA (extra server args, word-split)  PYPATH_PRE (prepended to PYTHONPATH, e.g. an aiter overlay)
# Hard rules (do not change): native MXFP8 attention/shared-expert weights (no --online_quant_config),
#   unfused shared expert, NVFP4 KV + NVFP4 index cache, no low-precision all-reduce.
set -u
: "${TAG:?}"
CONC=${CONC:-8}; PORT=${PORT:-8890}; AL=${AL:-4.04}; MEM=${MEM:-0.85}
REPO=${REPO:-/shared_nfs/ygan/work/ATOM-max}
MAX_SEQS=${MAX_SEQS:-$((2 * CONC))}
MODEL=${MODEL:-/shared_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4}
D=/shared_nfs/ygan/max
LOG=${P1_OUT:-${HOME:-/tmp}/p1-repro-out}/logs/server_${TAG}.log; mkdir -p "$(dirname "$LOG")"
export PYTHONPATH=${PYPATH_PRE:+$PYPATH_PRE:}$REPO
export HOME=/mnt/m2m_nobackup/ygan/p1/milestone/home/$TAG; mkdir -p "$HOME"
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-/mnt/m2m_nobackup/ygan/p1/milestone/flydsl}; mkdir -p "$FLYDSL_RUNTIME_CACHE_DIR"
export USER=ygan LOGNAME=ygan ARCH=gfx950 HSA_ENABLE_IPC_MODE_LEGACY=1
export NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1 AITER_LOG_LEVEL=WARNING ATOM_FORCE_ATTN_TRITON=1 PYTHONHASHSEED=0
export ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=0
unset AITER_QUICK_REDUCE_QUANTIZATION
# default flips to "3 at CONC >= 64" once k=3 is validated on ATOM (orchestrator); until then pass DSPARK_K=3
DSPARK_K=${DSPARK_K:-7}
case "$DSPARK_K" in 7) AL_GOLD=4.04 ;; 3) AL_GOLD=2.88 ;; *) AL_GOLD="" ;; esac
if [ "$AL" != real ] && [ -n "$AL_GOLD" ] && [ "$AL" != "$AL_GOLD" ]; then
  echo "serve.sh: AL=$AL is not the golden acceptance for k=$DSPARK_K; using $AL_GOLD" >&2; AL=$AL_GOLD
fi
spec=(--method dspark --draft-model "$MODEL/dspark" --num-speculative-tokens "$DSPARK_K")
[ "$AL" != real ] && spec+=(--spec-decode-acceptance-length "$AL")
cd "$REPO" || exit 1
{ echo "### $(date -Is) tag=$TAG conc=$CONC max_seqs=$MAX_SEQS port=$PORT k=$DSPARK_K al=$AL mem=$MEM gpus=${HIP_VISIBLE_DEVICES:-?} repo=$REPO"
  git -c safe.directory='*' -C "$REPO" log -1 --format="### atom commit: %h %s"; echo "### extra: ${EXTRA:-}"; env | grep -E '^(ATOM_|AITER_|LMCACHE_)' | sort | sed 's/^/### env /'; } > "$LOG"
exec python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL" --served-model-name minimax-m3.1 \
  --host 0.0.0.0 --port $((PORT + 6)) --server-port "$PORT" \
  --tensor-parallel-size 4 --trust-remote-code \
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}' \
  --gpu-memory-utilization "$MEM" --block-size 128 \
  --max-num-batched-tokens 32768 --max-num-seqs "$MAX_SEQS" \
  --kv_cache_dtype nvfp4 --index-cache-dtype nvfp4 \
  --enable-prefix-caching \
  "${spec[@]}" ${EXTRA:-} >> "$LOG" 2>&1
