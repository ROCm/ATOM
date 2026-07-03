#!/usr/bin/env bash
#SBATCH --job-name=atomesh-pd-bench
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
SCRIPT_PATH="${REPO_ROOT}/.github/scripts/atomesh/pd_server_atom.sh"
RUN_DIR="${LOG_ROOT}/slurm_job-${SLURM_JOB_ID}"

mkdir -p "${RUN_DIR}"

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
python3 - <<'PY' > "${ENV_FILE}"
import os

allow = (
    "ATOMESH_",
    "MODEL_",
    "BACKEND",
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
    "MAX_NUM_SEQS",
    "EXTRA_SERVER_ARGS",
    "RUN_EVAL",
    "EVAL_",
)
for key, value in sorted(os.environ.items()):
    if key.startswith(allow):
        print(f"{key}={value}")
PY

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
