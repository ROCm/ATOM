#!/bin/bash
set -euo pipefail

readonly CONCURRENCY=32
readonly BENCHMARK_DURATION=900
readonly WARMUP_REQUESTS_PER_LANE=3
RUN_NAME="${RUN_NAME:-agentx-c${CONCURRENCY}-$(date -u +%Y%m%d_%H%M%S)}"
AIPERF="${AIPERF:-/root/agentx/venv/bin/aiperf}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/agentx/artifacts-opt}"
SERVER_URL="${SERVER_URL:-http://10.210.11.14:8000}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

artifact_dir="${ARTIFACT_ROOT}/${RUN_NAME}"
mkdir -p "${artifact_dir}"

exec "${AIPERF}" profile \
  --scenario inferencex-agentx-mvp \
  --url "${SERVER_URL}" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model moonshotai/Kimi-K3 \
  --tokenizer moonshotai/Kimi-K3 \
  --tokenizer-trust-remote-code \
  --apply-chat-template \
  --concurrency "${CONCURRENCY}" \
  --benchmark-duration "${BENCHMARK_DURATION}" \
  --stats-interval 30 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane "${WARMUP_REQUESTS_PER_LANE}" \
  --warmup-grace-period 1800 \
  --trace-idle-gap-cap-seconds 300 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir "${artifact_dir}"
