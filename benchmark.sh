Here's a sweep script matching the result-file convention already in your workspace (`vllm-infqps-concurrency256-DeepSeek-R1-0528-MXFP4-Preview-20260731-224512.json`):

```bash
#!/usr/bin/env bash
# Sweep ATOM serving benchmarks across concurrency levels.
set -uo pipefail   # deliberately not -e: one failed point shouldn't kill the sweep

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
PORT="${PORT:-8000}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
RESULT_DIR="${RESULT_DIR:-/workspace/}"
PROMPTS_PER_CONC="${PROMPTS_PER_CONC:-10}"
SETTLE_SEC="${SETTLE_SEC:-20}"

# Override by passing values as args: ./sweep.sh 1 4 16 64
CONCURRENCIES=("${@:-}")
[ -z "${CONCURRENCIES[0]:-}" ] && CONCURRENCIES=(4 8 16 32 64 128 256 512)
INPUT_LENS=(8192)

MODEL_SHORT="$(basename "$MODEL")"
LOG_DIR="${RESULT_DIR%/}/sweep-logs"
mkdir -p "$LOG_DIR"

wait_for_server() {
  for _ in $(seq 1 60); do
    if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' \
            "http://localhost:${PORT}/health")" = "200" ]; then
      return 0
    fi
    sleep 5
  done
  return 1
}

if ! wait_for_server; then
  echo "ERROR: server not healthy on port ${PORT} after 5 min" >&2
  exit 1
fi
# /health can return 200 with no model resident — confirm VRAM is actually in use.
if command -v rocm-smi >/dev/null 2>&1; then
  echo "GPU memory in use:"
  rocm-smi --showmemuse 2>/dev/null | grep -m2 "VRAM%"
fi

declare -a SUMMARY=()
for INPUT in "${INPUT_LENS[@]}"; do
  for CONC in "${CONCURRENCIES[@]}"; do
    TS="$(date +%Y%m%d-%H%M%S)"
    RESULT_FILENAME="vllm-infqps-concurrency${CONC}-${MODEL_SHORT}-${TS}.json"
    LOG="${LOG_DIR}/concurrency${CONC}-${TS}.log"

    echo "=============================================================="
    echo "concurrency=${CONC}  num-prompts=$((CONC * PROMPTS_PER_CONC))  isl=${INPUT} osl=${OSL}"
    echo "  -> ${RESULT_FILENAME}"
    echo "=============================================================="

    START=$(date +%s)
    python -m atom.benchmarks.benchmark_serving \
      --model "$MODEL" \
      --port "$PORT" \
      --backend vllm \
      --base-url=http://localhost:8000   --dataset-name=random \
      --random-input-len "$INPUT" \
      --random-output-len "$OSL" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --num-prompts "$((CONC * PROMPTS_PER_CONC))" \
      --max-concurrency "$CONC" \
      --request-rate=inf --ignore-eos  \
      --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
      --result-filename "$RESULT_FILENAME" \
      --result-dir "$RESULT_DIR" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - START ))

    if [ "$RC" -eq 0 ]; then
      SUMMARY+=("OK    conc=${CONC} ${ELAPSED}s ${RESULT_FILENAME}")
    else
      SUMMARY+=("FAIL  conc=${CONC} rc=${RC} (log: ${LOG})")
      echo "WARNING: concurrency=${CONC} failed (rc=${RC}); continuing sweep" >&2
      # Bail out if the server died rather than grinding through failures.
      wait_for_server || { echo "ERROR: server unhealthy, aborting" >&2; break; }
    fi

    sleep "$SETTLE_SEC"   # let in-flight work drain before the next point
  done
done

echo
echo "===================== SWEEP SUMMARY ====================="
printf '%s\n' "${SUMMARY[@]}"