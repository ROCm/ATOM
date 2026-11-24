# ARM C4T = arm C4 without the SP piecewise graph (sourced): only the bit-identical part of the dispatch (SP TBO >= 10240 tokens).
# Below 2048 tokens this keeps C3's TP path (token graphs 256/512/1024, TP eager 1025-2047). Use it if the < 2048 SP graph is not gated.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC4.sh"
export ATOM_P1_SP_PCG=0 ATOM_TOKEN_GRAPH_SIZES=256,512,1024
unset ATOM_P1_SP_PCG_MIN
ARM=C4T
