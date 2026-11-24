# ARM C-noWIDE = arm C without ATOM_P1_AK_WIDE (bisects a gate/perf failure of C): above 16 requests decode takes
# int5's split4_wide -> q8 route again. Everything else identical to env_armC.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC.sh"
unset ATOM_P1_AK_WIDE
ARM=C-noWIDE
