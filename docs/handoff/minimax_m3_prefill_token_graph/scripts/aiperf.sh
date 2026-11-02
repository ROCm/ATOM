#!/bin/bash
# aiperf AgentX replay, per ATOM recipes/MiniMax-M3-Agentic-InferenceX.md.
# Usage (in container): CONC=10 bash /scripts/agentic/aiperf.sh <artifact dir>
set -u
CONC=${CONC:?set CONC}
ART=${1:?artifact dir}
MODEL=${MODEL:-/models/MiniMax-M3-MXFP4}
mkdir -p "$ART"

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_FAILED_REQUEST_THRESHOLD=0.10
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10
export AIPERF_WARMUP_REQUESTS_PER_LANE=10
export AIPERF_BENCHMARK_GRACE_PERIOD=30
export AIPERF_SERVER_METRICS_URLS="http://localhost:8890/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8890 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model "$MODEL" --tokenizer "$MODEL" \
  --tokenizer-trust-remote-code \
  --apply-chat-template \
  --concurrency "$CONC" --benchmark-duration "${DURATION:-3600}" --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics http://localhost:8890/metrics \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir "$ART" ${AIPERF_EXTRA:-}
