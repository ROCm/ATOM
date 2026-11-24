#!/usr/bin/env bash
# Wait until an ATOM server answers /health and its GPUs hold memory, or its log
# shows a failure. Usage: wait_ready.sh PORT LOG [TIMEOUT_S]   (inside a container)
set -uo pipefail
PORT=$1; LOG=$2; TIMEOUT=${3:-3600}
deadline=$((SECONDS + TIMEOUT))
while (( SECONDS < deadline )); do
  if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" > /dev/null; then
    echo "[ready] port ${PORT} after ${SECONDS}s"; exit 0
  fi
  if grep -q -E "Traceback \(most recent call last\)|EngineCore failed|RuntimeError: |Memory access fault|HSA_STATUS_ERROR|NotImplementedError|exitcode=-|Engine core .* died" "${LOG}" 2>/dev/null; then
    echo "[fail] port ${PORT}: error in ${LOG}"; grep -n -E "Error|error|Traceback|fault" "${LOG}" | tail -20; exit 1
  fi
  sleep 10
done
echo "[timeout] port ${PORT} after ${TIMEOUT}s"; tail -20 "${LOG}"; exit 2
