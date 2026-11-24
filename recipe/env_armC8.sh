# ARM C8 = p1/integrated12 (sourced): arm C3's flags + scheduler-owned midpoint SP TBO on every prefill-only step of
# 10240..24575 tokens (p1/balanced-split case (i), round 16; m5-smoke round 17).
# Tree: /shared_nfs/ygan/work/p1-integrated12 @ 30ba3fd20 = p1/integrated8 a52b845cb + balance_full.patch
#   (tbo-large SP TBO gen mode + tbo-pair-bench pair runner + tbo-pair-policy scheduler marks + balanced-split flags).
#   One commit, no trailers.
# - ATOM_P1_SP_TBO=1, MODE gen: SP prefill TBO runner (two micro-batches interleaved on one thread, comm stream).
# - ATOM_P1_SP_TBO_PAIR=1: each micro-batch uses its OWN rows' router K split => a micro-batch is bitwise a plain SP
#   step of its own tokens (tbo-pair-bench lease 1). The marked step == two plain steps (chunked-prefill semantics).
# - ATOM_P1_SCHED_BALANCE_TBO_MIN/MAX/SNAP_PCT: the scheduler marks every prefill-only step of 10240..24575 tokens
#   with its cut (request boundary within 10% of the midpoint, else the midpoint) in ScheduledBatch.p1_pair_cut; an
#   UNMARKED step never splits (TBO_MIN > 0 => the runner's own size vote is off, so ATOM_P1_SP_TBO_MIN_TOKENS is
#   set for consistency only). Steps > 24575 (idle steps up to max-num-batched-tokens 32768) run plain, as in C3
#   (pairs >= 24K hit allocator retries / engine death).
# NOT set on purpose: ATOM_P1_SCHED_BALANCE (step balancing, predicted 0 +- 0.05%) and ATOM_P1_SCHED_PAIR*.
_C8_FLY=${FLYDSL_RUNTIME_CACHE_DIR:-}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC3.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ATOM_P1_SP_TBO=1 ATOM_P1_SP_TBO_PAIR=1 ATOM_P1_SP_TBO_MODE=gen ATOM_P1_SP_TBO_MIN_TOKENS=10240
export ATOM_P1_SCHED_BALANCE_TBO_MIN=10240 ATOM_P1_SCHED_BALANCE_TBO_MAX=24575 ATOM_P1_SCHED_BALANCE_SNAP_PCT=10
ARM=C8
ARM_REPO_EXPECT=30ba3fd20
# own node-local FlyDSL cache seeded from C3's (seed_flydsl in common.sh), so a C8 start never compiles into C3's dir
export FLYDSL_RUNTIME_CACHE_DIR=${_C8_FLY:-$NLOC/flydsl_armC8}
FLYDSL_SEED=$NLOC/flydsl_armC3
# node 034 has only the milestone4 pair caches flydsl_armC3a/b (same kernels): use one when flydsl_armC3 is absent
[ -n "$(ls -A "$FLYDSL_SEED" 2>/dev/null)" ] || FLYDSL_SEED=$NLOC/flydsl_armC3a
unset _C8_FLY
