# ARM C3 = p1/integrated8 (sourced): arm C2's flags + the round-12/13 bit-identical changes below.
# Tree: /shared_nfs/ygan/work/p1-integrated8 = p1/integrated7 df75cb422 + sp-step-cost sp_moe_compact.patch
#       + decode-g9-16 aux_next.patch @ a52b845cb. One commit, no trailers.
# 1. everything arm C2 sets (C + SPLIT4_MIN8 6 + decode-prefill cap 16384 + ITL-risk max 150 ms)
_C3_FLY=${FLYDSL_RUNTIME_CACHE_DIR:-}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC2.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 2. sp-step-cost: SP prefill MoE hand-off gathers vLLM's lossless compact payload (MXFP8 codes+scales, MXFP4 scales,
#    correction bits, top-4 ids+weights; router K split forced to the full-M value) instead of bf16 rows, for SP steps
#    >= 4096 tokens. Bit-identical (slots, shared, RS output, payload == local quantization); -4..-20 ms per SP step.
export ATOM_P1_SP_MOE_COMPACT=1 ATOM_P1_SP_MOE_COMPACT_MIN_TOKENS=4096
# 3. decode-g9-16 + build-int8: DSpark taps take their aux from the next fused AR+add+norm tail's residual out on
#    pure-decode forwards >= 64 rows (removes 4 AITER all-reduces + adds per step). Bit-identical token streams
#    (build-int8 TP4 ABA: OFF1 == ON == OFF2 at G9/12/16); step -0.04/-0.04/-0.06 ms vs mean(OFF) at G9/12/16.
export ATOM_P1_DSPARK_AUX_NEXT=64
ARM=C3
ARM_REPO_EXPECT=a52b845cb
# new FlyDSL kernels (compact router/quant): own node-local cache, seeded from C's (seed_flydsl in common.sh)
export FLYDSL_RUNTIME_CACHE_DIR=${_C3_FLY:-$NLOC/flydsl_armC3}
FLYDSL_SEED=$NLOC/flydsl_armC
unset _C3_FLY
