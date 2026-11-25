# shellcheck shell=bash
# p_env_nochunk.sh: the PREFILL role of recipe-audit/p_env.sh with chunked P/D transfer OFF (user decision 2026-11-25):
# the producer prefills a request without waiting for the decode's admission and sends its KV after the prefill;
# decode credit then covers only transfer + first forward. Everything else is p_env.sh as written (load workers 1).
source "$(dirname "${BASH_SOURCE[0]}")/p_env.sh"
P_KV_TRANSFER_CONFIG="${P_KV_TRANSFER_CONFIG/,\"enable_chunked_transfer\":true,\"chunked_admission_timeout_s\":1500/}"
export PREFILL_KV_TRANSFER_CONFIG="${P_KV_TRANSFER_CONFIG}"
