# ARM C14 = C13 + every C10 flag (sourced). Tree: /shared_nfs/ygan/work/p1-c14 @ 1121e1d52 (branch p1/c14)
#       = p1/integrated14 5bf0aff51 (= int8 a52b845cb + campaign kernel wins) + prefix_fix.patch (= the p1/prefix-cache
#       9bfa1c6c4 diff, byte-identical; disjoint files, no conflict). One commit, no trailers. Flags default off in code.
# 1. everything arm C13 sets (C3 flags + ATOM_P1_PREFIX_DEMOTE_DECODE=1 + MEM 0.85 + ATOM_P1_SCHED_IDLE_PREFILL_CAP=16384)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC13.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 2. campaign overlay (p1l2 = verify-L2-attn-decode copy, l4c_y = L4-comm copy) appended to PYPATH_PRE, as arm C10 does
export PYPATH_PRE=$PYPATH_PRE  # l4c_y/p1l2 vendored at $REPO root; no external overlay needed
# 3. the five C10 flags. Bit-identical (composed TP4 tokens == int8 at 16 points): ATOM_P1_L2_D2, ATOM_P1_L4_PULLQ,
#    ATOM_P1_FAST_CONCAT. fp32 summation order only: ATOM_P1_LMHEAD_RING, ATOM_P1_DRAFT_FC_AK (no GSM8K gate required by
#    the p1 rules: a replacement that differs only in fp32 summation order is allowed with its errors reported; L5 +
#    verifier: ring bit-identical to amd-kernels' kernel, rel L2 vs fp64 equal to hgemm; fc 0-161 elements 1 bf16 ulp apart,
#    rel L2 vs fp64 identical). NOT included: sp-local-shared (~3.1 GiB/GPU = ~1.7% of KV blocks at MEM 0.85).
export ATOM_P1_L2_D2=1 ATOM_P1_L4_PULLQ=1 ATOM_P1_FAST_CONCAT=1 ATOM_P1_LMHEAD_RING=1 ATOM_P1_DRAFT_FC_AK=1
ARM=C14
ARM_REPO_EXPECT=1121e1d52
# own node-local FlyDSL cache (content-keyed): merged seeds, first wins: C13's cache, the campaign's C10 prefill-bench
# cache (034 only), C3's caches. Kernels not in any seed (d2/ring/fc variants) compile at warm-up / graph capture.
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC14
FLYDSL_SEED="$NLOC/flydsl_armC12 /mnt/m2m_nobackup/ygan/p1/campaign/flydsl $NLOC/flydsl_armC3 $NLOC/flydsl_armC3a"
