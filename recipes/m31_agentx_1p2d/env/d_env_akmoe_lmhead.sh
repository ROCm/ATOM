# shellcheck shell=bash
# d_env_akmoe_lmhead.sh: d_env_akmoe.sh (AK MoE on) plus ganyi's C17 lm_head AK + ring (user decision 2026-11-25, for every
# 1P-multi-D run from now on): verify / draft lm_head at 7..64 rows (<= 8 requests at k=7) on a shuffled shard copy
# (+615 MB per rank) and 65..128 rows (9-16 requests) on the same copy. Numerics: fp32 summation order on the verify logits.
source "$(dirname "${BASH_SOURCE[0]}")/d_env_akmoe.sh"
export ATOM_DG_LMHEAD_AK=1 ATOM_P1_LMHEAD_RING=1
for _n in ATOM_DG_LMHEAD_AK ATOM_P1_LMHEAD_RING; do
  case " ${D_ENV_NAMES[*]} " in *" ${_n} "*) ;; *) D_ENV_NAMES+=("${_n}") ;; esac
done
unset _n
