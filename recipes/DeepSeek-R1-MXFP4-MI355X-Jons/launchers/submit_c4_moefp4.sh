#!/bin/bash
# Submit c4 with MTP-MoEFP4 model + level=3 + cudagraph[1,2,4,8].
# Bench MUST use the MoEFP4 model path to match server.
set -u
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="/projects/teamK/supreme-leader/runs/${TS}_submit_c4_moefp4"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
LAUNCHER_FILE="/projects/teamK/supreme-leader/launch_atom_c4_level3_mtp_moefp4.sh"
CONTAINER="atom-dsr1-dev"
DOCKER="/usr/local/bin/docker-teamK-unrestricted"
PORT=8888

log() { echo "[$(date -u +%FT%TZ)] $*"; }

"${DOCKER}" rm -f "${CONTAINER}" 2>/dev/null || true
sleep 1

DRI_FLAGS=()
for d in /dev/dri/*; do [ -c "$d" ] && DRI_FLAGS+=("--device=$d"); done
mkdir -p /projects/teamK/supreme-leader/dsr1_aiter_cache
chmod 777 /projects/teamK/supreme-leader/dsr1_aiter_cache
"${DOCKER}" run -d --name "${CONTAINER}" --ipc=host --shm-size=16g \
  --device=/dev/kfd "${DRI_FLAGS[@]}" \
  -v /share4:/share4 -v /projects/teamK:/projects/teamK -v /projects/teamK:/workspace \
  -v /projects/teamK/supreme-leader/dsr1_aiter_cache:/root/.aiter \
  -p ${PORT}:${PORT} \
  rocm/atom:rocm7.2.1-ubuntu24.04-pytorch2.9.1-atom0.1.2 \
  /bin/bash -c "sleep infinity"

"${DOCKER}" exec "${CONTAINER}" bash -c '
  mkdir -p /workspace/amdgpu_bounty_optimization
  [ -e /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x ] || \
    ln -sfn /workspace/supreme-leader/bench_atom /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
  git config --global --add safe.directory /workspace/supreme-leader/bench_atom 2>/dev/null || true
'

LN="$(basename ${LAUNCHER_FILE})"
cp "${LAUNCHER_FILE}" "${RUN_DIR}/${LN}"
sed -i "s|tee /projects/teamK/server_.*\.log|tee /workspace/supreme-leader/runs/${TS}_submit_c4_moefp4/server.log|g" "${RUN_DIR}/${LN}"

log "launching server (MTP-MoEFP4)"
"${DOCKER}" exec -d "${CONTAINER}" bash -c "
  cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
  bash /workspace/supreme-leader/runs/${TS}_submit_c4_moefp4/${LN}
"

SECONDS=0
while [ $SECONDS -lt 900 ]; do
  curl -fsS "http://0.0.0.0:${PORT}/health" >/dev/null 2>&1 && break
  if grep -qE "Out of symmetric heap|RuntimeError|proc died|All EngineCores shut down" "${RUN_DIR}/server.log" 2>/dev/null; then
    log "FATAL early"; tail -30 "${RUN_DIR}/server.log"
    "${DOCKER}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    exit 6
  fi
  sleep 15
  [ $((SECONDS % 60)) -lt 15 ] && log "[${SECONDS}s] waiting"
done
log "=== healthy"

TEAM="${1:-Jons}"

log "=== submitting CONC=4 as ${TEAM} (MTP-MoEFP4 model)"
"${DOCKER}" exec "${CONTAINER}" bash -c "
  cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
  export MODEL=/share4/teamK/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4
  export PORT=${PORT}; export TP=8; export ISL=8192; export OSL=1024; export CONC=4
  export RANDOM_RANGE_RATIO=1.0
  export NUM_PROMPTS=40
  export RESULT_FILENAME=submit_c4_moefp4_${TS}.json
  export EP_SIZE=1; export DP_ATTENTION=0
  export HF_HOME=/projects/teamK/hf_home
  ./dsr1_benchmark submit ${TEAM}
" 2>&1 | tee "${LOG_DIR}/submit.log"

"${DOCKER}" exec "${CONTAINER}" bash -c "pkill -f 'atom.entrypoints.openai_server' || true"
sleep 5
"${DOCKER}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
log "=== DONE"
