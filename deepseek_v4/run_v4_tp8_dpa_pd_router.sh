#!/bin/bash
set -euo pipefail

PREFILL_IP="${PREFILL_IP:-10.24.112.168}"
DECODE_IP="${DECODE_IP:-10.24.112.184}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

/usr/local/bin/atom-mesh launch \
    --host 0.0.0.0 --port "${ROUTER_PORT}" \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:${PREFILL_PORT}" \
    --decode  "http://${DECODE_IP}:${DECODE_PORT}" \
    --policy random \
    --backend atom \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-health-check \
    --prometheus-port 29100
