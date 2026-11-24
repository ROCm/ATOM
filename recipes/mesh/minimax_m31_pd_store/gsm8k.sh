#!/usr/bin/env bash
# GSM8K 20-shot (chat, multi-turn few-shot) through the router, real acceptance.
# The same lm_eval settings as the M3.1 P/D CI accuracy cells. Usage: gsm8k.sh LABEL [CONC] [LIMIT]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL=${1:?label}; CONC=${2:-32}; LIMIT=${3:-}
OUT=${RUN}/gsm8k-${LABEL}
mkdir -p "${OUT}"
lim=()
[[ -n "${LIMIT}" ]] && lim=(--limit "${LIMIT}")
curl -s "http://${NODE_IP}:${D_PORT}/debug/mtp_stats" > "${OUT}/mtp_before.json" || true
start=$(date +%s)
lm_eval --model local-chat-completions \
  --model_args "model=${MODEL},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions,num_concurrent=${CONC},max_retries=3,max_gen_toks=16384" \
  --tasks gsm8k --num_fewshot 20 --apply_chat_template --fewshot_as_multiturn \
  "${lim[@]}" --log_samples --output_path "${OUT}" > "${OUT}/lm_eval.log" 2>&1
echo "elapsed_s=$(( $(date +%s) - start ))" | tee -a "${OUT}/lm_eval.log"
curl -s "http://${NODE_IP}:${D_PORT}/debug/mtp_stats" > "${OUT}/mtp_after.json" || true
grep -E "gsm8k|exact_match" "${OUT}/lm_eval.log" | tail -4
