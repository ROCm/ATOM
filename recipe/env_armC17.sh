# ARM C17 = C16 + sp-local-shared (the three C9/C11 flags + the p1_sls_shared overlay), sourced.
# Tree: /shared_nfs/ygan/work/p1-c17 @ 2d96c9954 (branch p1/c17) = p1/k2-compose 33786376b + /shared_nfs/ygan/p1/sp-local-shared/
#       sp_local_shared.patch (== p1/integrated13 4a46145aa - p1/integrated8 a52b845cb, md5 67607a8a; applied cleanly, hunk
#       offsets only). One commit, no trailers. Flags default off in code.
# Exactly as C9/C11 use it (combined-3600/recipe/env_armC9.sh): overlay /shared_nfs/ygan/p1/sp-local-shared/overlay appended
#   to PYPATH_PRE, ATOM_P1_SP_SHARED_LOCAL=1 ATOM_P1_SP_SHARED_LOCAL_MIN_TOKENS=2048 ATOM_P1_SP_SHARED_LOCAL_SIDE_ROWS=1.
#   Companions already set by C3 -> C16: ATOM_M3_SP_PREFILL=1, ATOM_P1_SP_MOE_SLOTS=1, ATOM_M3_AK_MOE=1,
#   ATOM_P1_SP_MOE_COMPACT=1 (reached only when shared_full is absent).
# Memory: +3.1 GiB/GPU resident (full shared expert, all-gathered TP shards at load; byte permutation, no re-quantization).
#   At MEM 0.85 (unchanged) the KV pool shrinks ~1.7-1.8% (C16 90957 blocks -> ~89.35-89.46K predicted). See C17b.
# Numerics: NOT bit-identical (one fp32 K=3072 down-GEMM sum instead of four K=768 partials; fp32 order only, sum5 RS with
#   bf16 partials): GSM8K gate required (mean of 2 >= 0.9621, real AL >= 4.608, needle 3/3).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC16.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYPATH_PRE=$PYPATH_PRE  # p1_sls_shared already vendored at $REPO root; no external overlay needed
export ATOM_P1_SP_SHARED_LOCAL=1 ATOM_P1_SP_SHARED_LOCAL_MIN_TOKENS=2048 ATOM_P1_SP_SHARED_LOCAL_SIDE_ROWS=1
ARM=C17
ARM_REPO_EXPECT=2d96c9954
# own node-local FlyDSL cache (content-keyed), merged seeds, first wins: C16's cache, sp-local-shared's flydsl_c3 and
# C11's / C9's caches (034: contain the overlay kernels shared_dense gate/up + down and sum5), then C16's seeds.
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC17
FLYDSL_SEED="$NLOC/flydsl_armC16 /mnt/m2m_nobackup/ygan/p1/sp-local-shared/flydsl_c3 $NLOC/flydsl_armC11 $NLOC/flydsl_armC9 $FLYDSL_SEED"
