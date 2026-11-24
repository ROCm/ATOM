# ARM C14b = C14 + the memory-neutral bit-identical items of arm C11 (combined-3600), sourced. Same tree as C14.
# (c) amd-kernels 3d62453 (PR #6 split4 half-run scorer): the amdk_snapshot entry of PYPATH_PRE is REPLACED IN PLACE by
#     /shared_nfs/ygan/p1/upstream-leads/amdk_3d62453 (= snapshot + 3d62453 decode_score.py only; bitwise per
#     p1/adopt-split4-3d62453) + ATOM_P1_INDEXER_SPLIT4_MIN8=4 (C3: 6; indexer route by request count, bitwise ids).
# (d) aiter #7772 tuned mxscale rows, bit-identical subset: AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE ->
#     p1/aiter-7772-bench/mx8_tuned_gemm_m31_pr7772_bitonly.csv.
# NOT included: sp-local-shared (C9/C11's ATOM_P1_SP_SHARED_LOCAL, ~3.1 GiB/GPU, not bitwise).
# Overlay check (CPU): c14/bin/import_check.py on this PYTHONPATH (logs/import_check_C14b.txt): campaign overlay p1l2 /
# l4c_y import only amd_kernels.common and the sparse-decode host, which are identical in snapshot and 3d62453.
# Interaction not measured: SPLIT4_MIN8=4 + 3d62453 with the C10 flags never ran a composed TP4 step.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC14.sh"
case ":$PYPATH_PRE:" in *:/shared_nfs/ygan/attn-prefill/amdk_snapshot:*) ;; *) echo "env_armC14b: amdk_snapshot not in PYPATH_PRE" >&2; return 2 ;; esac
export PYPATH_PRE=$(echo ":$PYPATH_PRE:" | sed 's#:/shared_nfs/ygan/attn-prefill/amdk_snapshot:#:/shared_nfs/ygan/p1/upstream-leads/amdk_3d62453:#; s#^:##; s#:$##')
export ATOM_P1_INDEXER_SPLIT4_MIN8=4
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE=/shared_nfs/ygan/p1/aiter-7772-bench/mx8_tuned_gemm_m31_pr7772_bitonly.csv
ARM=C14b
ARM_REPO_EXPECT=1121e1d52
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC14b
FLYDSL_SEED="$NLOC/flydsl_armC14 $NLOC/flydsl_armC12 /mnt/m2m_nobackup/ygan/p1/campaign/flydsl /mnt/m2m_nobackup/ygan/p1/adopt-split4-3d62453/flydsl $NLOC/flydsl_armC3 $NLOC/flydsl_armC3a"
