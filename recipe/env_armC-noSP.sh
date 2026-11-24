# ARM C-noSP = arm C without SP prefill (bisects a gate/perf failure of C): every SP flag and the token-graph cut
# removed (TP prefill with arm B's token graphs 256..3072). Everything else identical to env_armC.sh (incl. MEM 0.80).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC.sh"
unset ATOM_M3_SP_PREFILL ATOM_M3_SP_PREFILL_COMM ATOM_M3_SP_PREFILL_OPROJ ATOM_M3_SP_PREFILL_QGATHER \
      ATOM_P1_SP_HANDOFFS ATOM_P1_SP_QG_AK ATOM_P1_SP_A2A_UNPACK ATOM_P1_SP_OPROJ_AK ATOM_P1_SP_MOE_SLOTS \
      ATOM_M3_SP_PREFILL_MIN_TOKENS ATOM_TOKEN_GRAPH_SIZES
ARM=C-noSP
