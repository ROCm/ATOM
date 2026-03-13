#!/bin/bash
set -euo pipefail

TYPE=${1:-launch}
MODEL_PATH=${2:-meta-llama/Meta-Llama-3-8B-Instruct}
EXTRA_ARGS=("${@:3}")


if [ "$TYPE" == "launch" ]; then
  echo ""
  echo "========== Launching ATOM server =========="
  PROFILER_ARGS=""
  if [ "${ENABLE_TORCH_PROFILER:-0}" == "1" ]; then
    PROFILER_ARGS="--torch-profiler-dir /app/trace"
    echo "Torch profiler enabled, trace output: /app/trace"
  fi
  ATOM_SERVER_LOG="/tmp/atom_server.log"
  python -m atom.entrypoints.openai_server --model "$MODEL_PATH" $PROFILER_ARGS "${EXTRA_ARGS[@]}" 2>&1 | tee "$ATOM_SERVER_LOG" &
  atom_server_pid=$!

  echo ""
  echo "========== Waiting for ATOM server to start =========="
  # Phase 1: Wait for HTTP server to be up via /health endpoint
  max_retries=30
  retry_interval=60
  server_up=false
  for ((i=1; i<=max_retries; i++)); do
      if ! kill -0 $atom_server_pid 2>/dev/null; then
          echo "ATOM server process exited unexpectedly."
          echo "Last 50 lines of server log:"
          tail -50 "$ATOM_SERVER_LOG" 2>/dev/null || true
          exit 1
      fi
      if curl -sf http://localhost:8000/health -o /dev/null; then
          echo "ATOM server HTTP endpoint is up."
          server_up=true
          break
      fi
      echo "Waiting for ATOM server to be ready... ($i/$max_retries)"
      sleep $retry_interval
  done
  if [ "$server_up" = false ]; then
      echo "ATOM server did not start after $((max_retries * retry_interval)) seconds."
      kill $atom_server_pid
      exit 1
  fi

  # Phase 2: Warmup - send a real completion request to ensure model is fully ready
  # (CUDA graph capture, JIT compilation, etc. may still be in progress after /health returns OK)
  echo "========== Warming up ATOM server =========="
  warmup_retries=10
  warmup_interval=30
  warmup_done=false
  for ((i=1; i<=warmup_retries; i++)); do
      if ! kill -0 $atom_server_pid 2>/dev/null; then
          echo "ATOM server process exited unexpectedly during warmup."
          echo "Last 50 lines of server log:"
          tail -50 "$ATOM_SERVER_LOG" 2>/dev/null || true
          exit 1
      fi
      if curl -sf http://localhost:8000/v1/completions \
          -H "Content-Type: application/json" \
          -d '{"model":"'"$MODEL_PATH"'","prompt":"hi","max_tokens":1}' \
          -o /dev/null --max-time 120; then
          echo "ATOM server warmup completed successfully."
          warmup_done=true
          break
      fi
      echo "Warmup attempt $i/$warmup_retries failed, retrying in ${warmup_interval}s..."
      sleep $warmup_interval
  done
  if [ "$warmup_done" = false ]; then
      echo "ATOM server warmup failed after $((warmup_retries * warmup_interval)) seconds."
      kill $atom_server_pid
      exit 1
  fi
fi

if [ "$TYPE" == "accuracy" ]; then
  echo ""
  if ! command -v lm_eval >/dev/null 2>&1; then
    echo "========== Installing lm-eval =========="
    pip install lm-eval[api]
  else
    echo "========== lm-eval already installed; skipping installation =========="
  fi

  echo ""
  echo "========== Running accuracy test =========="
  mkdir -p accuracy_test_results
  RESULT_FILENAME=accuracy_test_results/$(date +%Y%m%d%H%M%S).json
  lm_eval --model local-completions \
          --model_args model="$MODEL_PATH",base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
          --tasks gsm8k \
          --num_fewshot 3 \
          --output_path "${RESULT_FILENAME}"
  echo "Accuracy test results saved to ${RESULT_FILENAME}"
  chmod -R 777 accuracy_test_results
fi

if [ "$TYPE" == "benchmark" ]; then
  echo ""
  echo "========== Cloning bench_serving =========="
  git clone https://github.com/kimbochen/bench_serving.git && chmod +x bench_serving/benchmark_serving.py
  echo "========== Running benchmark test =========="
  if [ "${ENABLE_TORCH_PROFILER:-0}" == "1" ]; then
    echo "Starting torch profiler..."
    curl -s -S -X POST http://127.0.0.1:8000/start_profile || echo "Warning: failed to start profiler"
  fi
  python bench_serving/benchmark_serving.py \
    --model=$MODEL_PATH --backend=vllm --base-url="http://localhost:8000" \
    --dataset-name=random \
    --random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
    --max-concurrency=$CONC \
    --num-prompts=$(( $CONC * 10 )) \
    --trust-remote-code \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=. --result-filename=${RESULT_FILENAME}.json

  if [ "${ENABLE_TORCH_PROFILER:-0}" == "1" ]; then
    echo "Stopping torch profiler..."
    curl -s -S -X POST http://127.0.0.1:8000/stop_profile || echo "Warning: failed to stop profiler"
    ATOM_SERVER_LOG="/tmp/atom_server.log"
    echo "Waiting for 'Profiler stopped.' in server log ..."
    profiler_done=false
    for i in $(seq 1 300); do
      if grep -q "Profiler stopped." "$ATOM_SERVER_LOG" 2>/dev/null; then
        echo "Profiler stopped after ${i}s"
        ls -lhR /app/trace/
        profiler_done=true
        break
      fi
      echo "Waiting for profiler to finish... ($i/300)"
      sleep 1
    done
    if [ "$profiler_done" = false ]; then
      echo "Warning: 'Profiler stopped.' not found in server log after 300s"
      ls -lhR /app/trace/ 2>/dev/null || true
    fi
  fi

  # Inject ISL/OSL into result JSON for summary table
  if [ -f "${RESULT_FILENAME}.json" ]; then
    jq --argjson isl "$ISL" --argjson osl "$OSL" \
      '. + {random_input_len: $isl, random_output_len: $osl}' \
      "${RESULT_FILENAME}.json" > "${RESULT_FILENAME}.tmp" && \
      mv "${RESULT_FILENAME}.tmp" "${RESULT_FILENAME}.json"
  fi
fi