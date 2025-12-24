set -euo pipefail

TYPE=${1:-launch}
MODEL_PATH=${2:-meta-llama/Meta-Llama-3-8B-Instruct}
EXTRA_ARGS="${@:3}"


if [ "$TYPE" == "launch" ]; then
  echo ""
  echo "========== Launching ATOM server =========="
  python -m atom.entrypoints.openai_server --model $MODEL_PATH $EXTRA_ARGS &
  atom_server_pid=$!

  echo ""
  echo "========== Waiting for ATOM server to start =========="
  max_retries=60
  retry_interval=60
  for ((i=1; i<=max_retries; i++)); do
      if curl -s http://localhost:8000/v1/completions -o /dev/null; then
          echo "ATOM server is up."
          break
      fi
      echo "Waiting for ATOM server to be ready... ($i/$max_retries)"
      sleep $retry_interval
  done
  if ! curl -s http://localhost:8000/v1/completions -o /dev/null; then
      echo "ATOM server did not start after $((max_retries * retry_interval)) seconds."
      kill $atom_server_pid
      exit 1
  fi
fi

if [ "$TYPE" == "accuracy" ]; then
echo ""
echo "========== Installing lm-eval =========="
pip install lm-eval[api]
echo ""
echo "========== Running accuracy test =========="
if [ "$MODEL_PATH" == "meta-llama/Meta-Llama-3-8B-Instruct" ]; then
   APPLY_CHAT_TEMPLATE="--apply_chat_template"
else
   APPLY_CHAT_TEMPLATE=""
fi
lm_eval --model local-completions \
        --model_args model=$MODEL_PATH,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False $APPLY_CHAT_TEMPLATE \
        --tasks gsm8k \
        $APPLY_CHAT_TEMPLATE \
        --num_fewshot 3
fi