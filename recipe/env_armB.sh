# ARM B = p1/integrated5 with every p1 flag on (sourced).
# Tree: /shared_nfs/ygan/work/p1-integrated5 @ 5f26263a8 = p1/integrated4 e8d340aca + sched-stalls finish_hold.patch
#       + prefill-step draft pinned to the ladder. Server/client/gate otherwise identical to arm A (same serve.sh args).
unset MONO DUTY AK_MOE EXTRA_MORE EXTRA TIER CPU_GB MEM MAX_SEQS AL DSPARK_K
# 1. the int1 base on-set, unchanged (prefill fast/ak16/amdk insert, token graph, AK insert/decode/MoE, MXFP8 GEMM,
#    fused decode, tuned GEMM tables, amd-kernels snapshot); same byte copy as arm A.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/int1_recipe.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 2. amd-kernels snapshot first, then the draft-fusion overlay (df_draft_ar, needed by ATOM_M3_DRAFT_AK_AR_NORM)
export PYPATH_PRE=${PYPATH_PRE:-}  # amd_kernels + df_draft_ar both vendored at $REPO root; set PYPATH_PRE to your aiter checkout path before sourcing this chain
# 3. round-3 flags (p1/integrated)
export ATOM_MINIMAX_AK_AR_NORM=1 ATOM_P1_FRONT_QKV_CHAIN=1 ATOM_M3_AK_DECODE_QOUT=1 ATOM_M3_DRAFT_POS_MAJOR=1 \
       ATOM_M3_DRAFT_MARKOV_WIDE=1 ATOM_M3_VERIFY_HEAD_CHUNK_TUNED=1 ATOM_DG_LMHEAD_AK=1
# 4. integrated2/3: chain Y codes, MoE tiles, split4 (8-16 requests), draft AR+norm+MXFP8, fused draft ctx rope
export ATOM_P1_AR_QKV_CODES=1 ATOM_P1_MOE_TILES=1 ATOM_P1_INDEXER_SPLIT4=1 ATOM_M3_DRAFT_AK_AR_NORM=1 \
       ATOM_M3_DRAFT_FUSED_CTX_ROPE=1
# 5. integrated4: split4 from 17 requests; drafter pinned to the production ladder (decode AND, in int5, prefill steps)
export ATOM_P1_INDEXER_SPLIT4_WIDE=1
export ATOM_P1_DRAFT_BS_LADDER=1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64
# 6. mono for <= 8-row decode steps (k7 G1). Off by itself whenever the engine's GPU guard sees another process on
#    the TP GPUs or a CPU KV tier transfer is in flight (no CPU tier in this recipe).
export ATOM_MONO_ENABLE=1 ATOM_P1_MONO_MAX_ROWS=8
# 7. scheduler: int1_both's duty/cap/fill, interval 8 instead of 16, finish-aware prefill hold (sched-stalls arm B)
export ATOM_M3_PREFILL_TOKEN_GRAPH=1 ATOM_SCHED_PREFILL_DUTY=0.3 ATOM_SCHED_DECODE_PREFILL_CAP=8192 \
       ATOM_PREFILL_DECODE_INTERVAL=8 ATOM_PREFILL_DELAYER_TARGET_FILL=0.225 \
       ATOM_P1_SCHED_FINISH_HOLD_LEFT=128 ATOM_P1_SCHED_FINISH_HOLD_MAX_MS=500 ATOM_P1_SCHED_FINISH_HOLD_MAX_TOKENS=400
# 8. exact decode graphs for 1..24 requests plus the production sizes up to 64 (the ladder keeps the drafter on the
#    production sizes, so the exact target graphs are gate-free by the row-invariance argument of integrate4)
CG='--cudagraph-capture-sizes [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,28,32,40,48,56,64]'
DS='--dspark-config {"draft_kv_dtype":"fp8"}'
export EXTRA="$CG $DS"
# 9. same observability as arm A (point.sh)
export ATOM_MONO_STATS_EVERY=200 ATOM_ENABLE_METRICS_DEVICE_TIMER=1
ARM=B
ARM_REPO_EXPECT=5f26263a8
