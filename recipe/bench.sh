#!/bin/bash
# AgentX (InferenceX agentic) replay against a running server; run inside a container (any GPU-free one).
# env: CONC TAG (required) PORT (8890) DURATION (seconds, >= 900; default 900)
# results: /shared_nfs/ygan/max/results/$TAG ; then: python3 /shared_nfs/ygan/max/bin/summarize.py $TAG
set -u
: "${CONC:?}" "${TAG:?}"
PORT=${PORT:-8890}; DURATION=${DURATION:-900}
MODEL=/shared_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4
ART=/shared_nfs/ygan/p1/c17/results/$TAG; mkdir -p "$ART"
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000 AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0 AIPERF_UI_REALTIME_METRICS_ENABLED=false
export AIPERF_FAILED_REQUEST_THRESHOLD=0.10 AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10 AIPERF_WARMUP_REQUESTS_PER_LANE=10
export AIPERF_BENCHMARK_GRACE_PERIOD=30 AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics" AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"
export HF_HOME=/shared_nfs/ygan/hf_home
exec aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:$PORT --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model minimax-m3.1 --tokenizer "$MODEL" --tokenizer-trust-remote-code --apply-chat-template \
  --concurrency "$CONC" --benchmark-duration "$DURATION" --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics http://localhost:$PORT/metrics \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir "$ART" > "$ART/client.log" 2>&1
