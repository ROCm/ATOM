# ARM C15 = C14 + the p1/mbf-compose decode MoE block flags (sourced). Tree: /shared_nfs/ygan/work/p1-c15 @ 82e41951d
#       (branch p1/c15) = p1/c14 1121e1d52 + the p1/mbf-compose 1c4af4413 diff vs int8 a52b845cb (git apply --3way, clean;
#       only atom/utils/envs.py is touched by both, separate hunks). One commit, no trailers. Flags default off in code.
# Not double-applied: C14 (= C10 flags on int14) sets ATOM_P1_MOE_TILES=1 but none of IDROWS / MBF_ASM / MBF_BACK; int14
#   does not touch any MoE file (ak_moe*.py, mbf_back.py).
# All three are bit-identical to int8's MoE path (p1/mbf-verify RESULT SUMMARY: 21 row counts, routing extremes, graph
#   replay with changed inputs, TP rank shards; sim1 greedy tokens identical): no GSM8K gate.
# - ATOM_P1_MOE_IDROWS=1 (needs ATOM_P1_MOE_TILES=1, set by C3/C14): id-rows host at 97-128 rows, tuned 8/16-row tiles.
# - ATOM_P1_MBF_ASM=1 (needs IDROWS; the mbf-build-asm name ATOM_P1_MOE_ASM was renamed to this in p1/mbf-compose):
#   32-128 rows routed GEMM2 prefetch 3 + GEMM1 prologue priority.
# - ATOM_P1_MBF_BACK=1: GEMM2 + slot/shared sum in one launch (<= 128 rows), honours the tiling above (compose fix F1).
#   Hazard: one sentinel buffer set per (device, stream); safe for plain decode graphs, NOT with TBO/multi-stream decode
#   (TBO stays off: no ATOM_P1_SP_TBO / SCHED_BALANCE in C14, checked by check_recipe).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC14.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ATOM_P1_MOE_IDROWS=1 ATOM_P1_MBF_ASM=1 ATOM_P1_MBF_BACK=1
ARM=C15
ARM_REPO_EXPECT=82e41951d
# own node-local FlyDSL cache (content-keyed), merged seeds, first wins: C14's cache, the mbf lanes' caches (034/035),
# then C14's seeds. MoE variants not in any seed compile at warm-up (mbf_back.warm_up compiles 1..128 rows).
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC15
FLYDSL_SEED="$NLOC/flydsl_armC14 /mnt/m2m_nobackup/ygan/p1/mbf-verify/flydsl /mnt/m2m_nobackup/ygan/p1/moe-gemm-headroom/flydsl /mnt/m2m_nobackup/ygan/p1/mbf-build-gemm2/flydsl /mnt/m2m_nobackup/ygan/p1/mbf-build-asm/flydsl $FLYDSL_SEED"
