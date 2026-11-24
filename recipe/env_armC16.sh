# ARM C16 = C15 + the verified k2 decode-kernel flags (sourced). Tree: /shared_nfs/ygan/work/p1-k2-compose @ 33786376b
#       (branch p1/k2-compose) = p1/c15 82e41951d + p1/k2-scorer diff (ak_kernels.py, backends.py) + p1/k2-draft diff
#       (p1_markov_block.py, dspark.py, backends.py; WITHOUT the lane's harness-only model_runner.py layout fix).
#       One commit, no trailers. Flags default off in code.
# Not double-applied: C15/C14 set neither ATOM_P1_K2_SCORER nor ATOM_P1_DRAFT_MARKOV_PERSIST; c15 touches only MoE files.
# Both bit-identical (verify-k2-scorer: 68 cases x 4 ranks + graph replay, composed TP4 tok_sha equal; verify-k2-draft:
#   144 eager cases + graph replay, composed TP4 tokens equal) -> no GSM8K gate.
# - ATOM_P1_K2_SCORER=rot2_wr2_h1024: split4 fused scorer + top-16 (score_topk, <= 128 rows) runs overlay p1k2.s4k2
#   (/shared_nfs/ygan/p1/k2-compose/overlay = frozen copy of the k2-scorer overlay, sha256 in k2-compose/NOTES.md).
#   Arm chosen by verify-k2-scorer (better than the lane's lf_wr2_h1024 at every live point measured).
# - ATOM_P1_DRAFT_MARKOV_PERSIST=1: DSpark Markov block in one persistent launch per rank (needs TP 4, B <= 64).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC15.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYPATH_PRE=$PYPATH_PRE  # p1k2 vendored at $REPO root; no external overlay needed
export ATOM_P1_K2_SCORER=rot2_wr2_h1024 ATOM_P1_DRAFT_MARKOV_PERSIST=1
ARM=C16
ARM_REPO_EXPECT=33786376b
# own node-local FlyDSL cache (content-keyed), merged seeds, first wins: C15's cache, the verify-k2-scorer cache (034,
# has the s4k2 arms), then C15's seeds. The Markov block is Triton (TRITON cache under the per-TAG HOME).
export FLYDSL_RUNTIME_CACHE_DIR=$NLOC/flydsl_armC16
FLYDSL_SEED="$NLOC/flydsl_armC15 /mnt/m2m_nobackup/ygan/p1/verify-k2-scorer/cache/flydsl $FLYDSL_SEED"
