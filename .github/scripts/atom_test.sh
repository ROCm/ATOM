#!/bin/bash
set -euo pipefail

TYPE=${1:-launch}
MODEL_PATH=${2:-meta-llama/Meta-Llama-3-8B-Instruct}
EXTRA_ARGS=("${@:3}")


if [ "$TYPE" == "launch" ]; then
  echo ""
  echo "========== Launching ATOM server =========="
  # Clear stale compile cache to avoid NameError from outdated generated code
  echo "Clearing compile cache..."
  rm -rf ~/.cache/atom/*
  PROFILER_ARGS=""
  if [ "${ENABLE_TORCH_PROFILER:-0}" == "1" ]; then
    PROFILER_ARGS="--torch-profiler-dir /app/trace --mark-trace"
    echo "Torch profiler enabled, trace output: /app/trace"
  fi

  # RTL (rocm-trace-lite) GPU kernel tracing
  RTL_CMD=""
  if [ "${ENABLE_RTL_PROFILER:-0}" == "1" ]; then
    RTL_TRACE_DIR="${ATOM_RTL_TRACE_DIR:-/app/rtl_traces}"
    mkdir -p "$RTL_TRACE_DIR"
    if command -v rtl &>/dev/null; then
      RTL_CMD="rtl trace -o ${RTL_TRACE_DIR}/trace.db --"
      echo "RTL profiler enabled, trace output: ${RTL_TRACE_DIR}"
    else
      echo "WARNING: RTL profiler requested but rtl command not found, skipping"
    fi
  fi

  ATOM_SERVER_LOG="/tmp/atom_server.log"
  PYTHONUNBUFFERED=1 $RTL_CMD python -m atom.entrypoints.openai_server --model "$MODEL_PATH" $PROFILER_ARGS "${EXTRA_ARGS[@]}" > "$ATOM_SERVER_LOG" 2>&1 &
  atom_server_pid=$!
  tail -f "$ATOM_SERVER_LOG" &
  _tail_launch_pid=$!
  trap 'kill $_tail_launch_pid 2>/dev/null || true' EXIT

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

  # Stop streaming server log now that launch is complete;
  # test phases (accuracy/benchmark) keep their output clean.
  # Full server log is available via the workflow "Dump server log" step.
  kill $_tail_launch_pid 2>/dev/null || true
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
  ATOM_CLIENT_LOG="${ATOM_CLIENT_LOG:-/tmp/atom_client.log}"
  # Set umask so files created by lm_eval are world-readable (container runs as root,
  # host runner user needs to read results via the shared volume mount)
  umask 0022
  mkdir -p accuracy_test_results
  RESULT_FILENAME=accuracy_test_results/$(date +%Y%m%d%H%M%S).json
  lm_eval --model local-completions \
          --model_args model="$MODEL_PATH",base_url=http://localhost:8000/v1/completions,num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True \
          --tasks gsm8k \
          --num_fewshot 3 \
          --output_path "${RESULT_FILENAME}" \
          2>&1 | tee "$ATOM_CLIENT_LOG"
  echo "Accuracy test results saved to ${RESULT_FILENAME}"
fi

if [ "$TYPE" == "stop" ]; then
  echo ""
  echo "========== Stopping ATOM server =========="

  # Generate RTL trace summary before killing the server
  RTL_TRACE_DIR="${ATOM_RTL_TRACE_DIR:-/app/rtl_traces}"
  if [ -d "$RTL_TRACE_DIR" ] && ls "$RTL_TRACE_DIR"/trace*.db 1>/dev/null 2>&1; then
    echo "Generating RTL trace summary..."
    for db in "$RTL_TRACE_DIR"/trace*.db; do
      rtl summary "$db" > "${db%.db}_summary.txt" 2>/dev/null || true
    done
    echo "RTL traces: $(ls "$RTL_TRACE_DIR"/*.db 2>/dev/null | wc -l) db files"
  fi

  # Wait for trace files to finish writing (before killing the server process)
  TRACE_DIR="${TORCH_PROFILER_DIR:-/app/trace}"
  if [ -d "$TRACE_DIR" ]; then
    echo "Waiting for trace files to finish writing..."
    for i in $(seq 1 120); do
      TMP_COUNT=$(find "$TRACE_DIR" -name '*.tmp' 2>/dev/null | wc -l)
      if [ "$TMP_COUNT" -eq 0 ]; then
        echo "Trace files ready after ${i}s"
        break
      fi
      [ "$i" -eq 120 ] && echo "WARNING: trace .tmp files still present after 120s"
      sleep 1
    done
  fi

  # Kill server processes
  pkill -f 'atom.entrypoints' || true
  sleep 2
  pkill -9 -f 'multiprocessing.spawn' || true
  pkill -9 -f 'multiprocessing.resource_tracker' || true

  # Wait for GPU memory to release
  echo "Waiting for GPU memory to release..."
  for i in $(seq 1 60); do
    USED_GPUS=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l 2>/dev/null || echo "0")
    if [ "$USED_GPUS" -eq 0 ]; then
      echo "GPU memory released after ${i}s"
      break
    fi
    if [ "$i" -eq 60 ]; then
      echo "WARNING: GPU memory still in use after 60s, force killing GPU processes"
      rocm-smi --showpidgpus 2>&1 | grep -oP 'PID \K\d+' | while read pid; do
        kill -9 "$pid" 2>/dev/null || true
      done
      sleep 5
    fi
    sleep 1
  done
  echo "Server stopped."
fi

if [ "$TYPE" == "benchmark" ]; then
  echo ""
  echo "========== Running benchmark test =========="
  ATOM_CLIENT_LOG="${ATOM_CLIENT_LOG:-/tmp/atom_client.log}"
  RESULT_FILENAME=${RESULT_FILENAME:-benchmark_result}
  PROFILE_ARG=""
  if [ "${ENABLE_TORCH_PROFILER:-0}" == "1" ]; then
    PROFILE_ARG="--profile"
    echo "Profiling enabled via --profile flag"
  fi
  python -m atom.benchmarks.benchmark_serving \
    --model=$MODEL_PATH --backend=vllm --base-url="http://localhost:8000" \
    --dataset-name=random \
    --random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
    --max-concurrency=$CONC \
    --num-prompts=${NUM_PROMPTS_OVERRIDE:-$(( $CONC * 10 ))} \
    --trust-remote-code \
    --num-warmups=$(( $CONC * 2 )) \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=. --result-filename=${RESULT_FILENAME}.json \
    $PROFILE_ARG ${BENCH_EXTRA_ARGS:-} \
    2>&1 | tee "$ATOM_CLIENT_LOG"

  # Inject ISL/OSL into result JSON for summary table
  if [ -f "${RESULT_FILENAME}.json" ]; then
    RESULT_PATH="${RESULT_FILENAME}.json" python3 - <<'PY'
import json
import os
import re

result_path = os.environ["RESULT_PATH"]
with open(result_path, encoding="utf-8") as f:
    d = json.load(f)

d["random_input_len"] = int(os.environ["ISL"])
d["random_output_len"] = int(os.environ["OSL"])
d["benchmark_backend"] = "ATOM"

tp_match = re.search(r"-tp\s+(\d+)", os.environ.get("SERVER_ARGS", ""))
if tp_match:
    d["tensor_parallel_size"] = int(tp_match.group(1))

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2)
PY
  fi
fi
