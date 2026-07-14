#!/usr/bin/env bash
#SBATCH --job-name=atomesh-pd-bench
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
SCRIPT_PATH="${REPO_ROOT}/.github/scripts/atomesh/pd_server_atom.sh"
JOB_ID="${SLURM_JOB_ID:-${SPUR_JOB_ID:-local}}"
RUN_DIR="${LOG_ROOT}/slurm_job-${JOB_ID}"

mkdir -p "${RUN_DIR}"

write_env_file() {
  local env_file="$1"
  python3 - <<'PY' > "${env_file}"
import os

allow = (
    "ATOMESH_",
    "MODEL_",
    "BACKEND",
    "DOCKER_IMAGE",
    "PRECISION",
    "TOPOLOGY",
    "DISPLAY_TOPOLOGY",
    "ISL_LIST",
    "OSL",
    "CONC_LIST",
    "BENCH_",
    "RANDOM_RANGE_RATIO",
    "REQUEST_RATE",
    "WAIT_",
    "PREFILL_",
    "DECODE_",
    "ROUTER_",
    "PROMETHEUS_PORT",
    "KV_CACHE_DTYPE",
    "BLOCK_SIZE",
    "MEM_FRACTION",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "ONLINE_QUANT_CONFIG",
    "HF_OVERRIDES",
    # Preserve FlyDSL cache overrides for non-root Spur containers.
    "FLYDSL_",
    "SPEC_",
    "DRAFT_MODEL_PATH",
    "NUM_SPEC_TOKENS",
    "EXTRA_SERVER_ARGS",
    "RUN_EVAL",
    "EVAL_",
)
for key, value in sorted(os.environ.items()):
    if key.startswith(allow):
        print(f"{key}={value}")
PY
}

pre_cleanup_local() {
  echo "=== pre-cleanup: stop running containers on $(hostname) ==="
  set +e
  running=()
  while read -r id; do
    [[ -n "${id}" ]] && running+=("${id}")
  done < <(docker ps -q 2>/dev/null)

  if [[ "${#running[@]}" -gt 0 ]]; then
    docker ps --format "  {{.ID}} {{.Names}} {{.Status}}"
    docker stop -t 0 "${running[@]}" >/dev/null 2>&1 || true
  else
    echo "no running containers"
  fi
  set -e
}

run_container_rank() {
  local rank="$1"
  local env_file="$2"
  local container="atomesh-${ATOMESH_CELL_ID}-${JOB_ID}-${rank}"
  local rank_dir="${RUN_DIR}/rank-${rank}"
  local bin_dir="${RUN_DIR}/bin"
  local video_gid render_gid host_ionic

  mkdir -p "${rank_dir}"
  mkdir -p "${bin_dir}"
  # PyTorch Inductor may call `nvcc --version` while formatting compiler errors.
  # Spur requires containers to run as the Slurm user, and the image's CUDA nvcc
  # path is not executable for that uid. Provide a narrow ROCm-only shim for the
  # version probe without pretending to support CUDA compilation.
  cat > "${bin_dir}/nvcc" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  exec hipcc --version
fi
echo "nvcc shim is only available for --version on ROCm CI" >&2
exit 127
EOF
  chmod +x "${bin_dir}/nvcc"

  video_gid="$(getent group video 2>/dev/null | cut -d: -f3 || true)"
  render_gid="$(getent group render 2>/dev/null | cut -d: -f3 || true)"
  host_ionic="$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)"

  docker rm -f "${container}" >/dev/null 2>&1 || true
  docker pull "${DOCKER_IMAGE}"

  docker_args=(
    run --rm --name "${container}"
    --user "$(id -u):$(id -g)"
    --network host --ipc host
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband
    --cap-add=IPC_LOCK --cap-add=NET_ADMIN
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=65536:524288
    --shm-size=128G
    --env-file "${env_file}"
    -e SLURM_JOB_ID="${JOB_ID}"
    -e SPUR_JOB_ID="${SPUR_JOB_ID:-${JOB_ID}}"
    -e NODE_RANK="${rank}"
    -e NODE0_ADDR="${NODE0_ADDR}"
    -e IPADDRS="${IPADDRS}"
    -e xP="${PREFILL_WORKERS}"
    -e yD="${DECODE_WORKERS}"
    -e PREFILL_TP_SIZE="${PREFILL_TP}"
    -e DECODE_TP_SIZE="${DECODE_TP}"
    -e RUN_DIR="/run_logs/slurm_job-${JOB_ID}"
    -e USER="$(id -un)"
    -e LOGNAME="$(id -un)"
    -e HOME="/tmp/atomesh-home-${JOB_ID}-${rank}"
    -e XDG_CACHE_HOME="/tmp/atomesh-cache-${JOB_ID}-${rank}"
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/torchinductor"
    -e AITER_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/aiter"
    -e AITER_JIT_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/aiter/jit"
    # FlyDSL otherwise tries to create caches under /app/aiter-test, which is
    # read-only for the Slurm uid required by Spur's Docker template.
    -e FLYDSL_RUNTIME_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/flydsl"
    -e NCCL_NET_PLUGIN=none
    -e NCCL_SOCKET_IFNAME=eth1
    -e NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    -e NCCL_IB_GID_INDEX=1
    -e NCCL_CROSS_NIC=0
    -e NCCL_PXN_DISABLE=0
    -e NCCL_NET_DISABLE_INTRA=1
    -e NCCL_IB_TC=104
    -e NCCL_IB_FIFO_TC=192
    -e NCCL_IB_QPS_PER_CONNECTION=1
    -e NCCL_IB_TIMEOUT=22
    -e NCCL_IB_RETRY_CNT=12
    -e NCCL_DEBUG=WARN
    -v "${REPO_ROOT}":/workspace/ATOM:ro
    -v "${RUN_DIR}":/run_logs/slurm_job-"${JOB_ID}"
    -v /mnt:/mnt
    -v /data:/data
  )

  [[ -n "${video_gid}" ]] && docker_args+=(--group-add "${video_gid}")
  [[ -n "${render_gid}" ]] && docker_args+=(--group-add "${render_gid}")
  [[ -n "${host_ionic}" && -e "${host_ionic}" ]] && docker_args+=(-v "${host_ionic}:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
  [[ -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ]] && docker_args+=(-v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
  [[ -e /etc/libibverbs.d/ionic.driver ]] && docker_args+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)
  [[ -d /it-share ]] && docker_args+=(-v /it-share:/it-share)

  docker_args+=(
    "${DOCKER_IMAGE}"
    bash -lc "export PATH=/run_logs/slurm_job-${JOB_ID}/bin:\${PATH}; cd /workspace/ATOM && bash .github/scripts/atomesh/pd_server_atom.sh"
  )

  docker "${docker_args[@]}" 2>&1 | tee "${rank_dir}/container.log"
}

run_spur_job() {
  if [[ -z "${SPUR_TASK_OFFSET:-}" || -z "${SPUR_PEER_NODES:-}" ]]; then
    return 1
  fi

  local node_rank="${SPUR_TASK_OFFSET}"
  local env_file="${RUN_DIR}/docker-rank-${node_rank}.env"
  local peers=()
  IFS=',' read -r -a peers <<< "${SPUR_PEER_NODES}"
  IFS=',' read -r -a SELECTED_NODES <<< "${SPUR_NODELIST:-${NODE_LIST}}"

  IPS=()
  for peer in "${peers[@]}"; do
    IPS+=("${peer%%:*}")
  done

  if [[ "${#SELECTED_NODES[@]}" -eq 0 || -z "${SELECTED_NODES[0]:-}" ]]; then
    SELECTED_NODES=()
    for idx in "${!IPS[@]}"; do
      SELECTED_NODES+=("spur-node-${idx}")
    done
  fi

  if [[ "${#IPS[@]}" -lt "${NUM_NODES}" ]]; then
    echo "ERROR: SPUR_PEER_NODES has ${#IPS[@]} nodes, expected ${NUM_NODES}" >&2
    exit 1
  fi

  SELECTED_NODES=("${SELECTED_NODES[@]:0:${NUM_NODES}}")
  IPS=("${IPS[@]:0:${NUM_NODES}}")
  SELECTED_NODELIST="$(IFS=,; echo "${SELECTED_NODES[*]}")"
  IPADDRS="$(IFS=,; echo "${IPS[*]}")"
  NODE0_ADDR="${IPS[0]}"

  echo "=== ATOMesh Spur job ${JOB_ID} rank ${node_rank}/${NUM_NODES} ==="
  echo "nodes=${SELECTED_NODELIST}"
  echo "ips=${IPADDRS}"
  echo "run_dir=${RUN_DIR}"

  pre_cleanup_local
  write_env_file "${env_file}"
  if [[ "${node_rank}" -eq 0 ]]; then
    cat > "${RUN_DIR}/cell-metadata.json" <<EOF
{
  "cell_id": "${ATOMESH_CELL_ID}",
  "model": "${MODEL_NAME}",
  "backend": "${BACKEND}",
  "topology": "${TOPOLOGY}",
  "display_topology": "${DISPLAY_TOPOLOGY}",
  "nodes": "${SELECTED_NODELIST}",
  "ips": "${IPADDRS}",
  "slurm_job_id": "${JOB_ID}",
  "log_root": "${RUN_DIR}"
}
EOF
  fi

  SPUR_NODE_RANK_FOR_CLEANUP="${node_rank}"
  cleanup_spur() {
    local rc=$?
    echo "=== cleanup rank=${SPUR_NODE_RANK_FOR_CLEANUP} rc=${rc} ==="
    docker stop -t 0 "atomesh-${ATOMESH_CELL_ID}-${JOB_ID}-${SPUR_NODE_RANK_FOR_CLEANUP}" >/dev/null 2>&1 || true
    return "${rc}"
  }
  trap cleanup_spur EXIT

  local rc=0
  run_container_rank "${node_rank}" "${env_file}" || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    echo "=== Spur rank ${node_rank} failed rc=${rc} ==="
    return "${rc}"
  fi

  echo "=== Spur rank ${node_rank} completed ==="
  find "${RUN_DIR}" -maxdepth 3 -type f | sort
  return 0
}

if run_spur_job; then
  exit 0
fi

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#ALLOC_NODES[@]}" -lt "${NUM_NODES}" ]]; then
  echo "ERROR: allocation has ${#ALLOC_NODES[@]} nodes, expected ${NUM_NODES}" >&2
  exit 1
fi

SELECTED_NODES=("${ALLOC_NODES[@]:0:${NUM_NODES}}")
SELECTED_NODELIST="$(IFS=,; echo "${SELECTED_NODES[*]}")"

pre_cleanup_nodes() {
  echo "=== pre-cleanup: stop all running containers ==="
  for node in "${SELECTED_NODES[@]}"; do
    echo "[pre-cleanup] node=${node}"
    srun --nodes=1 --ntasks=1 --nodelist="${node}" bash -lc '
      set +e
      echo "host=$(hostname)"

      running=()
      while read -r id; do
        [[ -n "${id}" ]] && running+=("${id}")
      done < <(docker ps -q 2>/dev/null)

      if [[ "${#running[@]}" -gt 0 ]]; then
        echo "stopping running containers:"
        docker ps --format "  {{.ID}} {{.Names}} {{.Status}}"
        docker stop -t 0 "${running[@]}" >/dev/null 2>&1 || true
      else
        echo "no running containers"
      fi

      sleep 2
      if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showmemuse 2>/dev/null || true
      fi
    ' || true
  done
  echo "=== pre-cleanup done ==="
}

pre_cleanup_nodes

IPS=()
for node in "${SELECTED_NODES[@]}"; do
  ip="$(srun --nodes=1 --ntasks=1 --nodelist="${node}" bash -lc "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")"
  if [[ -z "${ip}" ]]; then
    echo "ERROR: failed to resolve IP for ${node}" >&2
    exit 1
  fi
  IPS+=("${ip}")
done

IPADDRS="$(IFS=,; echo "${IPS[*]}")"
NODE0_ADDR="${IPS[0]}"

cat > "${RUN_DIR}/cell-metadata.json" <<EOF
{
  "cell_id": "${ATOMESH_CELL_ID}",
  "model": "${MODEL_NAME}",
  "backend": "${BACKEND}",
  "topology": "${TOPOLOGY}",
  "display_topology": "${DISPLAY_TOPOLOGY}",
  "nodes": "$(IFS=,; echo "${SELECTED_NODES[*]}")",
  "ips": "${IPADDRS}",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "log_root": "${RUN_DIR}"
}
EOF

echo "=== ATOMesh Slurm job ${SLURM_JOB_ID} ==="
echo "nodes=${SELECTED_NODELIST}"
echo "ips=${IPADDRS}"
echo "run_dir=${RUN_DIR}"

ENV_FILE="${RUN_DIR}/docker.env"
write_env_file "${ENV_FILE}"

cleanup() {
  local rc=$?
  echo "=== cleanup rc=${rc} ==="
  for node in "${SELECTED_NODES[@]}"; do
    srun --nodes=1 --ntasks=1 --nodelist="${node}" bash -lc "
      docker stop -t 0 atomesh-${ATOMESH_CELL_ID}-${SLURM_JOB_ID}-\${SLURM_PROCID:-x} >/dev/null 2>&1 || true
    " || true
  done
}
trap cleanup EXIT

echo "=== docker.env (passed to container) ==="
cat "${ENV_FILE}"
echo "=== end docker.env ==="

srun \
  --nodes="${NUM_NODES}" \
  --ntasks="${NUM_NODES}" \
  --ntasks-per-node=1 \
  --nodelist="${SELECTED_NODELIST}" \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    rank="${SLURM_PROCID}"
    container="atomesh-'"${ATOMESH_CELL_ID}"'-'"${SLURM_JOB_ID}"'-${rank}"
    rank_dir="'"${RUN_DIR}"'/rank-${rank}"
    mkdir -p "${rank_dir}"
    docker rm -f "${container}" >/dev/null 2>&1 || true
    docker pull "'"${DOCKER_IMAGE}"'"
    docker run --rm --name "${container}" \
      --network host --ipc host --privileged \
      --device /dev/kfd --device /dev/dri --device /dev/infiniband \
      --group-add video --cap-add IPC_LOCK --cap-add NET_ADMIN \
      --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
      --shm-size 128G \
      --env-file "'"${ENV_FILE}"'" \
      -e SLURM_JOB_ID="'"${SLURM_JOB_ID}"'" \
      -e NODE_RANK="${rank}" \
      -e NODE0_ADDR="'"${NODE0_ADDR}"'" \
      -e IPADDRS="'"${IPADDRS}"'" \
      -e xP="'"${PREFILL_WORKERS}"'" \
      -e yD="'"${DECODE_WORKERS}"'" \
      -e PREFILL_TP_SIZE="'"${PREFILL_TP}"'" \
      -e DECODE_TP_SIZE="'"${DECODE_TP}"'" \
      -e RUN_DIR="/run_logs/slurm_job-'"${SLURM_JOB_ID}"'" \
      -v "'"${REPO_ROOT}"'":/workspace/ATOM:ro \
      -v "'"${RUN_DIR}"'":/run_logs/slurm_job-'"${SLURM_JOB_ID}"' \
      -v /mnt:/mnt \
      -v /data:/data \
      -v /it-share:/it-share \
      "'"${DOCKER_IMAGE}"'" \
      bash -lc "cd /workspace/ATOM && bash .github/scripts/atomesh/pd_server_atom.sh" \
      2>&1 | tee "${rank_dir}/container.log"
  '

echo "=== Slurm job completed ==="
find "${RUN_DIR}" -maxdepth 3 -type f | sort
