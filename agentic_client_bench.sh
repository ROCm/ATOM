#!/usr/bin/env bash
# =============================================================================
# Agentic (Claude-Code trace replay) CLIENT benchmark — standalone
# -----------------------------------------------------------------------------
# Reproduces the InferenceX `agentic-coding` scenario CLIENT side only.
# It drives an ALREADY-RUNNING OpenAI-compatible server (ATOM / vLLM / SGLang)
# with the `inferencex-agentx-mvp` aiperf scenario, replaying real Claude-Code
# conversation traces (multi-turn, with sub-agents) from the SemiAnalysis Weka
# corpus.
#
# This script does NOT launch a server. Start your engine first, e.g.:
#   python -m atom.entrypoints.openai_server --model <M> -tp 8 \
#       --enable_prefix_caching --kv_cache_dtype fp8 --server-port 8889
#
# It handles, in order:
#   1. isolated Python venv (uv) with the pinned aiperf agentx fork + deps
#   2. dataset download into HF_HUB_CACHE (skipped if already cached)
#   3. the `aiperf profile` replay command
#   4. result aggregation directory
#
# Usage:
#   ./agentic_client_bench.sh                       # defaults below
#   MODEL=amd/MiniMax-M3-MXFP4 PORT=8889 CONC=16 ./agentic_client_bench.sh
#   CONC=32 DURATION=1800 ./agentic_client_bench.sh
# =============================================================================
set -euo pipefail

# ---- User-tunable config (env-overridable) ----------------------------------
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"   # must match --served-model-name on the server
HOST="${HOST:-localhost}"
PORT="${PORT:-8889}"                            # server port
CONC="${CONC:-16}"                              # concurrency = live session-trees (NOT raw requests)
DURATION="${DURATION:-1800}"                    # measurement window, seconds (>=900 for a "canonical" run)
MODEL_PREFIX="${MODEL_PREFIX:-}"                # dsv4* / minimaxm3* -> unfiltered corpus; else 256k-capped
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"             # optional: clamp replay to server context window
CACHE_WARMUP="${AIPERF_AGENTIC_CACHE_WARMUP_DURATION:-600}"  # KV/prefix-cache warmup before profiling
RESULT_DIR="${RESULT_DIR:-$(pwd)/agentic_results_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)}"

# aiperf fork (pinned to the InferenceX submodule commit)
AIPERF_REPO="${AIPERF_REPO:-https://github.com/SemiAnalysisAI/aiperf.git}"
AIPERF_BRANCH="${AIPERF_BRANCH:-cquil11/aiperf-agentx-v1.0}"
AIPERF_COMMIT="${AIPERF_COMMIT:-0d2aa0572ac685943d38c580675c4a61023581d3}"

# Isolated runtime (keep aiperf out of the server's site-packages!)
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/agentic-client-$$}"
AIPERF_SRC="${AIPERF_SRC:-$RUNTIME_DIR/aiperf}"
VENV="${VENV:-$RUNTIME_DIR/venv}"
PY="$VENV/bin/python"
AIPERF="$VENV/bin/aiperf"
HF="$VENV/bin/hf"

export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

# ---- Pick the trace corpus (mirrors resolve_trace_source) --------------------
# Long-context families (dsv4 / minimaxm3, native 1M ctx) replay the unfiltered
# corpus; everything else uses the 256k-capped variant so long traces don't 4xx.
case "$MODEL_PREFIX" in
  dsv4*|minimaxm3*) LOADER="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126}" ;;
  *)                LOADER="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126_256k}" ;;
esac
case "$LOADER" in
  semianalysis_cc_traces_weka_062126)       DATASET="semianalysisai/cc-traces-weka-062126" ;;
  semianalysis_cc_traces_weka_062126_256k)  DATASET="semianalysisai/cc-traces-weka-062126-256k" ;;
  semianalysis_cc_traces_weka_061526)       DATASET="semianalysisai/cc-traces-weka-061526" ;;
  semianalysis_cc_traces_weka_061526_256k)  DATASET="semianalysisai/cc-traces-weka-061526-256k" ;;
  *) echo "ERROR: unknown loader '$LOADER'. Set WEKA_LOADER_OVERRIDE to a known name." >&2; exit 1 ;;
esac

echo "======================================================================"
echo " Agentic client benchmark"
echo "   server      : http://$HOST:$PORT  (model=$MODEL)"
echo "   concurrency : $CONC   duration: ${DURATION}s   warmup: ${CACHE_WARMUP}s"
echo "   trace corpus: $LOADER  ($DATASET)"
echo "   result dir  : $RESULT_DIR"
echo "======================================================================"

mkdir -p "$RESULT_DIR" "$RUNTIME_DIR"

# ---- 1. Isolated venv + aiperf fork + deps ----------------------------------
if [ ! -x "$AIPERF" ] || [ ! -x "$HF" ]; then
  echo "[setup] building isolated aiperf environment ..."
  command -v git >/dev/null 2>&1 || { echo "git required"; exit 1; }

  # uv for a fast, isolated venv (fall back to the venv module if uv is absent)
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$RUNTIME_DIR/uv" sh
    export PATH="$RUNTIME_DIR/uv:$PATH"
  fi

  # clone the pinned aiperf agentx fork
  if [ ! -d "$AIPERF_SRC/.git" ]; then
    git clone --branch "$AIPERF_BRANCH" "$AIPERF_REPO" "$AIPERF_SRC"
    git -C "$AIPERF_SRC" checkout "$AIPERF_COMMIT"
  fi

  uv venv --python "$(command -v python3)" "$VENV"
  uv pip install --python "$PY" \
    -e "$AIPERF_SRC" \
    numpy pandas aiohttp "transformers>=4.46" xlsxwriter tqdm \
    "datasets>=4.7.0" tiktoken matplotlib \
    "huggingface_hub[cli]>=0.25.0" urllib3 requests \
    Pillow fastapi uvicorn

  [ -x "$AIPERF" ] || { echo "ERROR: aiperf not installed at $AIPERF" >&2; exit 1; }
fi

# ---- 2. Dataset download (idempotent; reads HF cache on repeat) --------------
echo "[data] ensuring trace corpus is cached: $DATASET"
"$HF" download --repo-type dataset "$DATASET"

# ---- 3. Build the replay command (mirrors build_replay_cmd) ------------------
# The inferencex-agentx-mvp scenario auto-injects: --cache-bust first_turn_prefix
# and --trace-idle-gap-cap-seconds 10, so we do NOT pass them here.
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"

CMD=( "$AIPERF" profile
  --scenario inferencex-agentx-mvp
  --url "http://$HOST:$PORT"
  --endpoint /v1/chat/completions
  --endpoint-type chat
  --streaming
  --model "$MODEL"
  --concurrency "$CONC"
  --benchmark-duration "$DURATION"
  --random-seed 42
  --failed-request-threshold "${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
  --trajectory-start-min-ratio 0.25
  --trajectory-start-max-ratio 0.75
  --use-server-token-count
  --tokenizer-trust-remote-code
  --num-dataset-entries 393
  --slice-duration 1.0
  --no-gpu-telemetry
  --output-artifact-dir "$RESULT_DIR/aiperf_artifacts"
  --public-dataset "$LOADER"
)

# Optional cache-pressure warmup (fills KV/prefix cache before measuring)
[ -n "$CACHE_WARMUP" ] && CMD+=( --agentic-cache-warmup-duration "$CACHE_WARMUP" )

# Keep replayed traces within the server's context window
[ -n "$MAX_MODEL_LEN" ] && [ "$MAX_MODEL_LEN" != "0" ] && CMD+=( --max-context-length "$MAX_MODEL_LEN" )

# The scenario enforces a 900s minimum; allow short smoke tests to opt out
if [ "$DURATION" -lt 900 ] || [ "${AIPERF_UNSAFE_OVERRIDE:-false}" = "true" ]; then
  CMD+=( --unsafe-override )
fi

# ---- 4. Run --------------------------------------------------------------------
printf '%q ' "${CMD[@]}" | tee "$RESULT_DIR/benchmark_command.txt"; echo | tee -a "$RESULT_DIR/benchmark_command.txt"

echo "[run] starting agentic trace replay ..."
"${CMD[@]}" 2>&1 | tee "$RESULT_DIR/benchmark.log"

echo "======================================================================"
echo " Done. Artifacts in: $RESULT_DIR/aiperf_artifacts"
echo "   - profile_export.json / .jsonl   (per-request + aggregate metrics)"
echo "   - benchmark.log                  (full replay log)"
echo "======================================================================"
