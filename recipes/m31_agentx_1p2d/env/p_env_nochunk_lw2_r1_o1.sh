# shellcheck shell=bash
# p_env_nochunk_lw2_r1_o1.sh: the hour point's PREFILL role = p_env_nochunk_lw2.sh (chunked transfer off, two Store load
# threads) + ATOM_STORE_LOAD_PIPELINE=1 (Store R1) + ATOM_PD_SEND_PROCESS=1 (O1). Both flags need this branch's code
# commits, so run it with --commit at the NIC-fix commit (see ../README.md), not 76b83860b.
source "$(dirname "${BASH_SOURCE[0]}")/p_env_nochunk_lw2.sh"
export ATOM_STORE_LOAD_PIPELINE=1 ATOM_PD_SEND_PROCESS=1
for _n in ATOM_STORE_LOAD_PIPELINE ATOM_PD_SEND_PROCESS; do
  case " ${P_ENV_NAMES[*]} " in *" ${_n} "*) ;; *) P_ENV_NAMES+=("${_n}") ;; esac
done
unset _n
