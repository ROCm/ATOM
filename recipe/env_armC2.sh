# ARM C2 = p1/integrated7 (sourced): arm C's flags + the round-11 validated changes.
# Tree: /shared_nfs/ygan/work/p1-integrated7 @ df75cb422 = p1/integrated6 f5c123c7b + decode-gap split4_min8.patch.
# 1. everything arm C sets (B flags, P2 scheduler, AK_WIDE, SP prefill, MEM 0.80, watchdog)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 2. decode-gap: 8-row (k7) split4 indexer from 6 requests instead of 8 (bit-identical routes; G6/G7 live gain)
export ATOM_P1_INDEXER_SPLIT4_MIN8=6
# 3. ttft-prefill-host arm C16R: decode-prefill cap 8192 -> 16384 (SP step linear past 8K) and ITL-risk hold <= 150 ms
export ATOM_SCHED_DECODE_PREFILL_CAP=16384 ATOM_P1_SCHED_ITL_RISK_MAX_MS=150
ARM=C2
ARM_REPO_EXPECT=df75cb422
# same kernels as C: share C's node-local flydsl cache
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-$NLOC/flydsl_armC}
