#!/usr/bin/env bash
set -euo pipefail

# Required env:
#   CONTAINER       - docker container name
#   PREFILL_IP      - prefill node IP
#   DECODE_IP       - decode node IP
#
# Optional env (with defaults):
#   PREFILL_PORT=8010  DECODE_PORT=8020  ROUTER_PORT=8080
#   BOOTSTRAP_PORT=8998  POLICY=random
#   MESH_BIN=/usr/local/bin/atom-mesh  WAIT_TIMEOUT=120

: "${CONTAINER:?}"
: "${PREFILL_IP:?}"
: "${DECODE_IP:?}"

PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
POLICY="${POLICY:-random}"
MESH_BIN="${MESH_BIN:-/usr/local/bin/atom-mesh}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"

echo "[router] prefill=${PREFILL_IP}:${PREFILL_PORT} decode=${DECODE_IP}:${DECODE_PORT} router=0.0.0.0:${ROUTER_PORT}"

if docker exec "${CONTAINER}" curl -sf "http://127.0.0.1:${ROUTER_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[router] already running"
    exit 0
fi

docker exec "${CONTAINER}" bash -c "mkdir -p /workspace/logs"

docker exec -d "${CONTAINER}" bash -c "
    ${MESH_BIN} launch \
        --host 0.0.0.0 --port ${ROUTER_PORT} \
        --pd-disaggregation \
        --prefill http://${PREFILL_IP}:${PREFILL_PORT} ${BOOTSTRAP_PORT} \
        --decode  http://${DECODE_IP}:${DECODE_PORT} \
        --policy ${POLICY} \
        --backend sglang \
        --log-dir /workspace/logs \
        --log-level info \
        --disable-health-check \
        --prometheus-port 29100 \
        >/workspace/logs/router.log 2>&1
"

echo "[router] waiting (timeout=${WAIT_TIMEOUT}s)..."
timeout "${WAIT_TIMEOUT}" bash -c "
    while ! docker exec '${CONTAINER}' curl -sf http://127.0.0.1:${ROUTER_PORT}/v1/models >/dev/null 2>&1; do
        sleep 5
    done
"
echo "[router] ready"
