#!/bin/bash
# GLM-5.2 PP4×PD GSM8K accuracy evaluation via lm_eval
#
# Prerequisites: mesh proxy running on port 30000
#   (launch with: scripts/start_glm52_pp4pd.sh)
#
# Usage:
#   docker exec -it atom_pp4pd_test bash /it-share/yajizhan/code/ATOM/scripts/eval_glm52_pp4pd_gsm8k.sh
#
# Environment overrides:
#   LIMIT=50        — run only first 50 samples (default: full 1319)
#   NUM_FEWSHOT=5   — few-shot count (default: 5)
#   NUM_CONCURRENT=64 — parallel requests (default: 64)
#   MESH_PORT=30000 — mesh proxy port (default: 30000)
#
# The lm_eval invocation is the one recipes/GLM-5.md publishes its GLM-5.2
# MXFP4 reference against (TP4, no DPA/MTP: flexible 0.9742, strict 0.9727),
# so a PD number from this script is comparable to that row.

set -euo pipefail

MODEL=/mnt/models/GLM-5.2-MXFP4
MESH_PORT="${MESH_PORT:-30000}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
NUM_CONCURRENT="${NUM_CONCURRENT:-64}"
LIMIT="${LIMIT:-}"
RESULT_DIR=/tmp/results/glm52_pp4pd_gsm8k
BASE_URL="http://127.0.0.1:${MESH_PORT}/v1/chat/completions"

mkdir -p "$RESULT_DIR"

echo ""
echo "========================================"
echo " GLM-5.2 PP4×PD GSM8K Accuracy (lm_eval)"
echo "========================================"
echo " Model:       $MODEL"
echo " Base URL:    $BASE_URL"
echo " Few-shot:    $NUM_FEWSHOT"
echo " Concurrency: $NUM_CONCURRENT"
echo " Limit:       ${LIMIT:-full (1319)}"
echo " Result dir:  $RESULT_DIR"
echo "========================================"

# wait for mesh
echo ">>> Waiting for mesh (port $MESH_PORT) ..."
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${MESH_PORT}/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "    mesh READY"; break; }
  [ "$i" -eq 60 ] && { echo "ERROR: mesh not ready after 5 min"; exit 1; }
  sleep 5
done

LIMIT_ARG=()
if [ -n "$LIMIT" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

echo ""
echo ">>> Running GSM8K evaluation ..."
lm_eval --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE_URL},api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=${NUM_CONCURRENT},timeout=1800,tokenized_requests=False,max_length=1048576" \
  --apply_chat_template \
  --tasks gsm8k \
  --gen_kwargs max_tokens=16384,temperature=0,top_p=1 \
  --num_fewshot "$NUM_FEWSHOT" \
  "${LIMIT_ARG[@]}" \
  --output_path "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/gsm8k_eval.log"

echo ""
echo ">>> GSM8K evaluation done. Results in $RESULT_DIR/"
