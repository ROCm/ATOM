#!/usr/bin/env bash
# gsm8k accuracy check for Kimi-K3 (Minimax-m3-xiaobing) served by ATOM.
#
# Validated result (full 1319, clean CUDA-graph server, tp8):
#   flexible-extract 0.9378 / strict-match 0.9371  (on par with Kimi-K2-Thinking 0.9363)
#
# IMPORTANT — matches how Kimi models are evaluated (see recipes/Kimi-K2*.md):
#   use `local-completions` (BASE completions, NOT chat) with few-shot. This
#   avoids the K3 thinking-template (which otherwise burns the token budget on
#   <|think|> and never emits the "#### N" answer, tanking the score).
#
# Run the server first, e.g.:
#   bash my_script/run_k3_mi355_tp8_cudagraph.sh
# then (inside the container):
#   bash my_script/eval_k3_gsm8k.sh              # full 1319
#   K3_LIMIT=50 bash my_script/eval_k3_gsm8k.sh  # quick 50-question smoke test
#
# NOTE: run on a CLEAN GPU set — a competing sglang CI job stealing VRAM mid-run
# corrupts the score (a one-off 0.53 was traced to exactly that). Verify no
# `atom_sglang_validation_*` container is running and the server stays up
# (grep the eval output for ServerDisconnected).
set -euo pipefail

MODEL="${K3_MODEL:-/shared/data/amd_int/models/xiaobing/Minimax-m3-xiaobing}"
PORT="${K3_PORT:-8000}"
NUM_FEWSHOT="${K3_NUM_FEWSHOT:-5}"
NUM_CONCURRENT="${K3_NUM_CONCURRENT:-16}"

LIMIT_ARG=()
if [[ -n "${K3_LIMIT:-}" ]]; then
  LIMIT_ARG+=(--limit "${K3_LIMIT}")
fi

SAMPLES_ARG=()
if [[ -n "${K3_OUTPUT:-}" ]]; then
  SAMPLES_ARG+=(--log_samples --output_path "${K3_OUTPUT}")
fi

exec /opt/venv/bin/lm_eval \
  --model local-completions \
  --model_args "model=${MODEL},base_url=http://localhost:${PORT}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot "${NUM_FEWSHOT}" \
  "${LIMIT_ARG[@]}" \
  "${SAMPLES_ARG[@]}"
