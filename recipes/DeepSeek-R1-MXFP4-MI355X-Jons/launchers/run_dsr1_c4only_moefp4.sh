#!/bin/bash
# c4-only driver for MTP-MoEFP4 model — uses the correct model path in bench.
set -u
LAUNCHER="${1:?Usage: $0 <launcher_basename>}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="/projects/teamK/supreme-leader/runs/${TS}_c4_${LAUNCHER}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

LAUNCHER_FILE="/projects/teamK/supreme-leader/launch_atom_${LAUNCHER}.sh"
[ -f "${LAUNCHER_FILE}" ] || { echo "FATAL: launcher missing: ${LAUNCHER_FILE}"; exit 2; }

CONTAINER="atom-dsr1-dev"
IMAGE="rocm/atom:rocm7.2.1-ubuntu24.04-pytorch2.9.1-atom0.1.2"
PORT=8888
DOCKER="/usr/local/bin/docker-teamK-unrestricted"

if ! "${DOCKER}" ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  DRI_FLAGS=()
  for d in /dev/dri/*; do [ -c "$d" ] && DRI_FLAGS+=("--device=$d"); done
  mkdir -p /projects/teamK/supreme-leader/dsr1_aiter_cache
  chmod 777 /projects/teamK/supreme-leader/dsr1_aiter_cache
  "${DOCKER}" run -d --name "${CONTAINER}" --ipc=host --shm-size=16g \
    --device=/dev/kfd "${DRI_FLAGS[@]}" \
    -v /share4:/share4 -v /projects/teamK:/projects/teamK -v /projects/teamK:/workspace \
    -v /projects/teamK/supreme-leader/dsr1_aiter_cache:/root/.aiter \
    -p ${PORT}:${PORT} \
    "${IMAGE}" /bin/bash -c "sleep infinity"
fi

"${DOCKER}" exec "${CONTAINER}" bash -c '
  mkdir -p /workspace/amdgpu_bounty_optimization
  [ -e /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x ] || \
    ln -sfn /workspace/supreme-leader/bench_atom /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
  git config --global --add safe.directory /workspace/supreme-leader/bench_atom 2>/dev/null || true
'

LN="$(basename ${LAUNCHER_FILE})"
cp "${LAUNCHER_FILE}" "${RUN_DIR}/${LN}"
sed -i "s|tee /projects/teamK/server_.*\.log|tee /workspace/supreme-leader/runs/${TS}_c4_${LAUNCHER}/server.log|g" "${RUN_DIR}/${LN}"

echo "=== launching server"
"${DOCKER}" exec -d "${CONTAINER}" bash -c "
  cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
  bash /workspace/supreme-leader/runs/${TS}_c4_${LAUNCHER}/${LN}
"

SECONDS=0
HEALTHY=0
while [ $SECONDS -lt 1200 ]; do
  curl -fsS "http://0.0.0.0:${PORT}/health" >/dev/null 2>&1 && { HEALTHY=1; break; }
  if grep -qE "Out of symmetric heap|RuntimeError|proc died|All EngineCores shut down|memory access fault" "${RUN_DIR}/server.log" 2>/dev/null; then
    if ! "${DOCKER}" exec "${CONTAINER}" pgrep -f atom.entrypoints >/dev/null 2>&1; then
      echo "FATAL early. Last 20:"; tail -20 "${RUN_DIR}/server.log"
      "${DOCKER}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
      exit 6
    fi
  fi
  sleep 15
  [ $((SECONDS % 60)) -lt 15 ] && echo "  [${SECONDS}s] waiting"
done
[ "${HEALTHY}" = "1" ] || { echo "FATAL: not healthy in 20m"; tail -30 "${RUN_DIR}/server.log"; exit 4; }
echo "=== healthy after ${SECONDS}s"

BENCH_LOG="${LOG_DIR}/bench_${LAUNCHER}_c4.log"
"${DOCKER}" exec "${CONTAINER}" bash -c "
  cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
  export MODEL=/share4/teamK/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4
  export PORT=${PORT}; export TP=8; export ISL=8192; export OSL=1024; export CONC=4
  export RANDOM_RANGE_RATIO=1.0
  export NUM_PROMPTS=40
  export RESULT_FILENAME=c4_${LAUNCHER}_${TS}.json
  export EP_SIZE=1; export DP_ATTENTION=0
  export HF_HOME=/projects/teamK/hf_home
  ./dsr1_benchmark perf
" 2>&1 | tee "${BENCH_LOG}"

"${DOCKER}" exec "${CONTAINER}" bash -c "pkill -f 'atom.entrypoints.openai_server' || true"
sleep 5
"${DOCKER}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "=== DONE ${LAUNCHER} c4"
grep -E "Total Token throughput|gsm8k_metric|Mean TPOT" "${BENCH_LOG}" 2>&1 | head -10
