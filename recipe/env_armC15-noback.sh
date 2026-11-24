# ARM C15-noback = C15 without ATOM_P1_MBF_BACK (sourced): C14 + ATOM_P1_MOE_IDROWS=1 + ATOM_P1_MBF_ASM=1, same tree.
# Separates MBF_BACK (microbench -0.0125 ms/step on top of ASM, carries the per-stream sentinel hazard) in an A/B.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC15.sh"
unset ATOM_P1_MBF_BACK
ARM=C15-noback
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC15nb
FLYDSL_SEED="$NLOC/flydsl_armC15 $FLYDSL_SEED"
