#!/bin/bash
# Serve a model 1P1D on this node for the ATOMesh accuracy test: an ATOM
# prefill and decode server, Mooncake moving the KV between them, and the
# ATOMesh PD router on ATOM_SERVER_PORT, where `atom_test.sh accuracy` sends
# its requests. The servers keep running after this script returns.
#
# Usage: atomesh_pd_accuracy.sh MODEL_PATH [SERVER_ARGS...]
set -euo pipefail

MODEL_PATH="$1"
SERVER_ARGS=("${@:2}")
ROUTER_PORT="${ATOM_SERVER_PORT:-8000}"
ROUTER_LOG="${ATOM_SERVER_LOG:-/tmp/atom_server.log}"
PREFILL_LOG="$(dirname "${ROUTER_LOG}")/atom_prefill.log"
DECODE_LOG="$(dirname "${ROUTER_LOG}")/atom_decode.log"
PREFILL_GPUS="${PREFILL_GPUS:-0}"
DECODE_GPUS="${DECODE_GPUS:-1}"
PREFILL_PORT=8010
DECODE_PORT=8020
HANDSHAKE_PORT=6301
MESH_BINARY="${ATOMESH_BINARY:-atom/mesh/target/release/atomesh}"
WAIT_SECONDS="${WAIT_SECONDS:-2700}"
HOST_IP="${ATOM_HOST_IP:-$(hostname -I | awk '{print $1}')}"
CACHE_ROOT="$(mktemp -d -t atomesh-pd-accuracy.XXXXXX)"

# Each server gets its own compile caches: both compile the same graphs at once.
# The CI container has no RDMA devices, so Mooncake moves the KV over TCP, and
# MC_DISABLE_HIP keeps it off HIP IPC between the two processes.
start_server() {
  local role="$1" gpus="$2" port="$3" handshake_port="$4" log="$5"
  local cache="${CACHE_ROOT}/${role}"
  mkdir -p "${cache}/home" "${cache}/torchinductor"
  HOME="${cache}/home" TORCHINDUCTOR_CACHE_DIR="${cache}/torchinductor" \
    HIP_VISIBLE_DEVICES="${gpus}" ATOM_HOST_IP="${HOST_IP}" \
    USE_ATOMESH_ENTRYPOINTS=0 MC_DISABLE_HIP=1 PYTHONUNBUFFERED=1 \
    python3 -m atom.entrypoints.openai_server --model "${MODEL_PATH}" \
    --server-port "${port}" "${SERVER_ARGS[@]}" \
    --kv-transfer-config "{\"kv_role\":\"${role}\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${HOST_IP}\",\"handshake_port\":${handshake_port},\"protocol\":\"tcp\"}" \
    > "${log}" 2>&1 &
}

wait_http() {
  local url="$1" pid="$2" log="$3"
  local deadline=$((SECONDS + WAIT_SECONDS))
  until curl -sf --max-time 10 "${url}" -o /dev/null; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Exited before ${url} answered. Last 50 lines of ${log}:"
      tail -n 50 "${log}" || true
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "No answer from ${url} in ${WAIT_SECONDS}s. Last 50 lines of ${log}:"
      tail -n 50 "${log}" || true
      exit 1
    fi
    sleep 10
  done
}

echo "1P1D: prefill on GPU ${PREFILL_GPUS}, decode on GPU ${DECODE_GPUS}, router on port ${ROUTER_PORT}"
start_server kv_producer "${PREFILL_GPUS}" "${PREFILL_PORT}" "${HANDSHAKE_PORT}" "${PREFILL_LOG}"
prefill_pid=$!
# Every prefill rank takes a handshake port of its own, from HANDSHAKE_PORT up.
prefill_tp="$(awk -F, '{print NF}' <<< "${PREFILL_GPUS}")"
start_server kv_consumer "${DECODE_GPUS}" "${DECODE_PORT}" "$((HANDSHAKE_PORT + prefill_tp))" "${DECODE_LOG}"
decode_pid=$!
wait_http "http://${HOST_IP}:${PREFILL_PORT}/health" "${prefill_pid}" "${PREFILL_LOG}"
wait_http "http://${HOST_IP}:${DECODE_PORT}/health" "${decode_pid}" "${DECODE_LOG}"

"${MESH_BINARY}" launch --host 0.0.0.0 --port "${ROUTER_PORT}" --pd-disaggregation \
  --prefill "http://${HOST_IP}:${PREFILL_PORT}" --decode "http://${HOST_IP}:${DECODE_PORT}" \
  --policy random --backend atom --log-level info --disable-circuit-breaker \
  > "${ROUTER_LOG}" 2>&1 &
router_pid=$!
wait_http "http://127.0.0.1:${ROUTER_PORT}/v1/models" "${router_pid}" "${ROUTER_LOG}"

# The first request through the router is the first KV transfer: make it here,
# where a failure shows the three logs, rather than in the accuracy run.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf --max-time 120 "http://127.0.0.1:${ROUTER_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"hi\",\"max_tokens\":8}" -o /dev/null; then
    echo "1P1D service answered on port ${ROUTER_PORT}"
    exit 0
  fi
  echo "Request ${attempt}/10 through the router failed; retrying in 30s"
  sleep 30
done
echo "No completion through the router. Last 30 lines of each log:"
tail -n 30 "${PREFILL_LOG}" "${DECODE_LOG}" "${ROUTER_LOG}" || true
exit 1
