# ARM C4 = p1/integrated9 (sourced): arm C3's flags + the per-step SP prefill dispatch by token count N
# (p1/prefill-dispatch, round 15). Tree: /shared_nfs/ygan/work/p1-integrated9 @ b790686e8 = p1/integrated8 a52b845cb
#   + p1/pcg-small (SP piecewise graph) + p1/tbo-large (SP TBO, single-thread interleaved) + split guard. One commit, no trailers.
# Dispatch (thresholds here, not in code):
#   N <= 512              TP token graphs 256 / 512 (unchanged)
#   513 <= N <= 1792      SP piecewise graph, padded up to the next bucket of ATOM_TOKEN_GRAPH_SIZES (1024/1280/1536/1792)
#                         (was: TP token graph at 1024, TP eager 1025-2047). Graph replay == eager SP at the bucket bitwise
#                         (PCG_CHECK 1024/1280/1536/1792); SP numerics below 2048 = the gated SP path on fewer rows.
#   1793 <= N <= 2047     TP eager (unchanged)
#   2048 <= N < 10240     eager SP (+ compact MoE >= 4096), unchanged from C3
#   N >= 10240            SP TBO: two micro-batches, gen mode; bit-identical to the unsplit SP step (in-step dumps at
#                         10240/12288/14336/16384, P=102400, and 12288 at P=32768 x 2 prompts); splits with a micro-batch
#                         <= 4096 tokens are refused in code (MoE expert host changes there: not bit-identical)
_C4_FLY=${FLYDSL_RUNTIME_CACHE_DIR:-}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC3.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ATOM_P1_SP_PCG=1 ATOM_P1_SP_PCG_MIN=1024
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,1280,1536,1792
export ATOM_P1_SP_TBO=1 ATOM_P1_SP_TBO_MIN_TOKENS=10240 ATOM_P1_SP_TBO_MODE=gen
ARM=C4
ARM_REPO_EXPECT=b790686e8
# no new FlyDSL kernels beyond C3's: own node-local cache seeded from C3's (seed_flydsl in common.sh)
export FLYDSL_RUNTIME_CACHE_DIR=${_C4_FLY:-$NLOC/flydsl_armC4}
FLYDSL_SEED=$NLOC/flydsl_armC3
unset _C4_FLY
