# shellcheck shell=bash
# p_env_nochunk_lw2.sh: p_env_nochunk.sh (chunked transfer OFF) + two Store load threads per prefill rank
# (OFFLOAD_LOAD_WORKERS=2; OFFLOAD_LOAD_WORKERS is already in P_ENV_NAMES). The prefill max_num_seqs cap is the
# launcher's --p-seqs, not an env (USER DECISION 2026-11-25 08:37 UTC: chunked off + p-seqs 24 + load workers 2).
source "$(dirname "${BASH_SOURCE[0]}")/p_env_nochunk.sh"
export OFFLOAD_LOAD_WORKERS=2
