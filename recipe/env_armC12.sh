# ARM C12 = C3 + prefix-cache eviction policy (sourced). Tree: /shared_nfs/ygan/work/p1-prefix-cache @ 9bfa1c6c4
#       = p1/integrated8 a52b845cb + prefix_fix.patch (one commit, no trailers; both flags default off).
# 1. everything arm C3 sets (MEM 0.80, decode-prefill cap 16384, SP prefill, ...), unchanged
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC3.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 2. prefix-fix: a finished request's generated (decode-output) blocks go to the evict-first end of the free list
#    (after vacant blocks, before every other cached block; still hittable until spent). Prompt blocks keep the
#    tail-first LRU. Cache sim (official-3600 C3 034 trace, 80.5K effective blocks): uncached -1.8M tokens of 40.2M;
#    scheduler sim RS-like +0.29%. No numerics change (a hit returns the same KV blocks).
export ATOM_P1_PREFIX_DEMOTE_DECODE=1
ARM=C12
ARM_REPO_EXPECT=9bfa1c6c4
# same kernels as C3: own node-local cache seeded from C3's
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC12
FLYDSL_SEED=$NLOC/flydsl_armC3
