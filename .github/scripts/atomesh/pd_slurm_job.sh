#!/usr/bin/env bash
#SBATCH --job-name=atomesh-pd-bench
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
SCRIPT_PATH="${REPO_ROOT}/.github/scripts/atomesh/pd_server_atom.sh"
JOB_ID="${SLURM_JOB_ID:-${SPUR_JOB_ID:-local}}"
CURRENT_USER="$(id -un 2>/dev/null || id -u)"
RUN_DIR="${LOG_ROOT}/slurm_job-${JOB_ID}"

mkdir -p "${RUN_DIR}"

EXECUTION_PHASES=(combined)
if [[ "${BENCHMARK_KIND:-random}" == "aiperf_agentic" \
  && ( "${EVAL_TASK:-gsm8k}" == "swebench_lite" \
    || "${EVAL_TASK:-gsm8k}" == "gsm8k" ) \
  && ( "${RUN_EVAL:-false}" == "true" || "${RUN_EVAL:-false}" == "1" ) ]]; then
  EXECUTION_PHASES=(benchmark eval)
fi
ATOMESH_RESTART_PORT_OFFSET="${ATOMESH_RESTART_PORT_OFFSET:-1000}"
if [[ "${#EXECUTION_PHASES[@]}" -gt 1 && ! "${ATOMESH_RESTART_PORT_OFFSET}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ATOMESH_RESTART_PORT_OFFSET must be a positive integer" >&2
  exit 2
fi

execution_phase_port_offset() {
  if [[ "$1" == "eval" ]]; then
    printf '%s\n' "${ATOMESH_RESTART_PORT_OFFSET}"
  else
    printf '0\n'
  fi
}

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
    "BENCHMARK_KIND",
    "AIPERF_",
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
    "ENABLE_PREFIX_CACHING",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "DECODE_MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "DECODE_MAX_NUM_BATCHED_TOKENS",
    "ONLINE_QUANT_CONFIG",
    "HF_OVERRIDES",
    # Preserve FlyDSL cache overrides for non-root Spur containers.
    "FLYDSL_",
    "SPEC_",
    "STATE_CHECKPOINT_",
    "DRAFT_MODEL_PATH",
    "NUM_SPEC_TOKENS",
    "EXTRA_SERVER_ARGS",
    "RUN_EVAL",
    "EVAL_",
    "SWEBENCH_",
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
  local execution_phase="${3:-combined}"
  local service_port_offset="${4:-0}"
  local phase_suffix=""
  local container_log="container.log"
  if [[ "${execution_phase}" != "combined" ]]; then
    phase_suffix="-${execution_phase}"
    container_log="container-${execution_phase}.log"
  fi
  local container="atomesh-${ATOMESH_CELL_ID}-${JOB_ID}-${rank}${phase_suffix}"
  local rank_dir="${RUN_DIR}/rank-${rank}"
  local bin_dir="${RUN_DIR}/bin"
  local video_gid render_gid host_ionic nccl_socket_ifname
  local docker_socket_gid docker_cli docker_root

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
  nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-}"
  if [[ -z "${nccl_socket_ifname}" && -d /sys/class/net/eth1 ]]; then
    nccl_socket_ifname="eth1"
  fi

  docker rm -f "${container}" >/dev/null 2>&1 || true
  if [[ "${execution_phase}" != "eval" ]]; then
    docker pull "${DOCKER_IMAGE}"
  fi

  docker_args=(
    run --rm --name "${container}"
    --user "$(id -u):$(id -g)"
    --network host --ipc host
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband
    --cap-add=IPC_LOCK --cap-add=NET_ADMIN
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=65536:524288
    --shm-size=128G
    --env-file "${env_file}"
    -e ATOMESH_EXECUTION_PHASE="${execution_phase}"
    -e ATOMESH_SERVICE_PORT_OFFSET="${service_port_offset}"
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
    -e USER="${CURRENT_USER}"
    -e LOGNAME="${CURRENT_USER}"
    -e HOME="/tmp/atomesh-home-${JOB_ID}-${rank}"
    -e XDG_CACHE_HOME="/tmp/atomesh-cache-${JOB_ID}-${rank}"
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/torchinductor"
    -e AITER_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/aiter"
    -e AITER_JIT_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/aiter/jit"
    # FlyDSL otherwise tries to create caches under /app/aiter-test, which is
    # read-only for the Slurm uid required by Spur's Docker template.
    -e FLYDSL_RUNTIME_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/flydsl"
    -e NCCL_NET_PLUGIN=none
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

  if [[ "${rank}" -eq 0 \
    && "${EVAL_TASK:-}" == "swebench_lite" \
    && ( "${RUN_EVAL:-false}" == "true" || "${RUN_EVAL:-false}" == "1" ) ]]; then
    if [[ ! -S /var/run/docker.sock ]]; then
      echo "ERROR: local SWE-bench Lite requires /var/run/docker.sock" >&2
      return 2
    fi
    if [[ "${execution_phase}" != "benchmark" ]]; then
      docker_cli="$(readlink -f "$(command -v docker)")"
      docker_socket_gid="$(stat -c '%g' /var/run/docker.sock)"
      # Agent generation and official scoring create sibling containers through
      # the host daemon. This mount is intentionally limited to the rank-0
      # accuracy container.
      docker_args+=(
        -v /var/run/docker.sock:/var/run/docker.sock
        -v "${docker_cli}":/usr/local/bin/docker-host:ro
        -e SWEBENCH_DOCKER_EXECUTABLE=/usr/local/bin/docker-host
        --group-add "${docker_socket_gid}"
      )
      # The SWE-bench disk preflight df(1)s the path the daemon reports as its
      # root, but that path is in the *host* namespace. Bind it in at the same
      # path so it resolves to the same filesystem inside rank 0; without it the
      # check either finds nothing and skips, or measures the rootfs of the
      # container and reports a number for the wrong disk.
      docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
      if [[ -n "${docker_root}" && -d "${docker_root}" ]]; then
        docker_args+=(-v "${docker_root}:${docker_root}:ro")
      else
        echo "WARN: could not resolve the Docker root on this host; the" \
          "SWE-bench disk preflight check will be skipped" >&2
      fi
    fi
  fi

  [[ -n "${video_gid}" ]] && docker_args+=(--group-add "${video_gid}")
  [[ -n "${render_gid}" ]] && docker_args+=(--group-add "${render_gid}")
  [[ -n "${nccl_socket_ifname}" ]] && docker_args+=(-e NCCL_SOCKET_IFNAME="${nccl_socket_ifname}")
  [[ -n "${host_ionic}" && -e "${host_ionic}" ]] && docker_args+=(-v "${host_ionic}:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
  [[ -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ]] && docker_args+=(-v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
  [[ -e /etc/libibverbs.d/ionic.driver ]] && docker_args+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)
  [[ -d /it-share ]] && docker_args+=(-v /it-share:/it-share)
  [[ -d /shared_nfs ]] && docker_args+=(-v /shared_nfs:/shared_nfs)

  docker_args+=(
    "${DOCKER_IMAGE}"
    bash -lc "export PATH=/run_logs/slurm_job-${JOB_ID}/bin:\${PATH}; cd /workspace/ATOM && bash .github/scripts/atomesh/pd_server_atom.sh"
  )

  docker "${docker_args[@]}" 2>&1 | tee "${rank_dir}/${container_log}"
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
  SPUR_CLEANUP_DONE=0
  cleanup_spur() {
    local rc="${1:-$?}"
    local suffix
    if [[ "${SPUR_CLEANUP_DONE}" == "1" ]]; then
      return "${rc}"
    fi
    SPUR_CLEANUP_DONE=1
    echo "=== cleanup rank=${SPUR_NODE_RANK_FOR_CLEANUP} rc=${rc} ==="
    for suffix in "" "-benchmark" "-eval"; do
      docker rm -f \
        "atomesh-${ATOMESH_CELL_ID}-${JOB_ID}-${SPUR_NODE_RANK_FOR_CLEANUP}${suffix}" \
        >/dev/null 2>&1 || true
    done
    return "${rc}"
  }
  trap 'cleanup_spur $?' EXIT
  trap 'cleanup_spur 129; exit 129' HUP
  trap 'cleanup_spur 130; exit 130' INT
  trap 'cleanup_spur 143; exit 143' TERM

  local execution_phase service_port_offset
  local rc=0
  for execution_phase in "${EXECUTION_PHASES[@]}"; do
    service_port_offset="$(execution_phase_port_offset "${execution_phase}")"
    echo "=== Spur rank ${node_rank} phase=${execution_phase} service_port_offset=${service_port_offset} ==="
    run_container_rank \
      "${node_rank}" \
      "${env_file}" \
      "${execution_phase}" \
      "${service_port_offset}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      echo "=== Spur rank ${node_rank} phase=${execution_phase} failed rc=${rc} ==="
      return "${rc}"
    fi
  done

  echo "=== Spur rank ${node_rank} completed ==="
  find "${RUN_DIR}" -maxdepth 3 -type f | sort
  return 0
}

if [[ -n "${SPUR_TASK_OFFSET:-}" || -n "${SPUR_PEER_NODES:-}" ]]; then
  run_spur_job
  exit $?
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

CLEANUP_DONE=0
cleanup() {
  local rc="${1:-$?}"
  local idx node container suffix
  if [[ "${CLEANUP_DONE}" == "1" ]]; then
    return "${rc}"
  fi
  CLEANUP_DONE=1
  echo "=== cleanup rc=${rc} ==="
  for idx in "${!SELECTED_NODES[@]}"; do
    node="${SELECTED_NODES[$idx]}"
    for suffix in "" "-benchmark" "-eval"; do
      container="atomesh-${ATOMESH_CELL_ID}-${SLURM_JOB_ID}-${idx}${suffix}"
      srun --nodes=1 --ntasks=1 --nodelist="${node}" bash -lc "
        docker rm -f '${container}' >/dev/null 2>&1 || true
      " || true
    done
  done
  return "${rc}"
}
trap 'cleanup $?' EXIT
trap 'cleanup 129; exit 129' HUP
trap 'cleanup 130; exit 130' INT
trap 'cleanup 143; exit 143' TERM

echo "=== docker.env (passed to container) ==="
cat "${ENV_FILE}"
echo "=== end docker.env ==="

for execution_phase in "${EXECUTION_PHASES[@]}"; do
  service_port_offset="$(execution_phase_port_offset "${execution_phase}")"
  export ATOMESH_EXECUTION_PHASE="${execution_phase}"
  export ATOMESH_SERVICE_PORT_OFFSET="${service_port_offset}"
  echo "=== Slurm phase=${execution_phase} service_port_offset=${service_port_offset} ==="

  srun \
    --nodes="${NUM_NODES}" \
    --ntasks="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --nodelist="${SELECTED_NODELIST}" \
    --kill-on-bad-exit=1 \
    bash -lc '
      set -euo pipefail
      rank="${SLURM_PROCID}"
      execution_phase="${ATOMESH_EXECUTION_PHASE:-combined}"
      service_port_offset="${ATOMESH_SERVICE_PORT_OFFSET:-0}"
      phase_suffix=""
      container_log="container.log"
      if [[ "${execution_phase}" != "combined" ]]; then
        phase_suffix="-${execution_phase}"
        container_log="container-${execution_phase}.log"
      fi
      container="atomesh-'"${ATOMESH_CELL_ID}"'-'"${SLURM_JOB_ID}"'-${rank}${phase_suffix}"
      rank_dir="'"${RUN_DIR}"'/rank-${rank}"
      mkdir -p "${rank_dir}"
      docker rm -f "${container}" >/dev/null 2>&1 || true
      if [[ "${execution_phase}" != "eval" ]]; then
        docker pull "'"${DOCKER_IMAGE}"'"
      fi
      nested_docker_args=()
      if [[ "${rank}" -eq 0 \
        && "${EVAL_TASK:-}" == "swebench_lite" \
        && ( "${RUN_EVAL:-false}" == "true" || "${RUN_EVAL:-false}" == "1" ) ]]; then
        if [[ ! -S /var/run/docker.sock ]]; then
          echo "ERROR: local SWE-bench Lite requires /var/run/docker.sock" >&2
          exit 2
        fi
        if [[ "${execution_phase}" != "benchmark" ]]; then
          host_docker_cli="$(readlink -f "$(command -v docker)")"
          docker_socket_gid="$(stat -c "%g" /var/run/docker.sock)"
          nested_docker_args=(
            -v /var/run/docker.sock:/var/run/docker.sock
            -v "${host_docker_cli}:/usr/local/bin/docker-host:ro"
            -e SWEBENCH_DOCKER_EXECUTABLE=/usr/local/bin/docker-host
            --group-add "${docker_socket_gid}"
          )
          # The SWE-bench disk preflight df(1)s the path the daemon reports as
          # its root, but that path is in the *host* namespace. Bind it in at
          # the same path so it resolves to the same filesystem inside rank 0;
          # without it the check either finds nothing and skips, or measures the
          # rootfs of the container and reports a number for the wrong disk.
          # No apostrophes and no single quotes below: this whole block is one
          # single-quoted remote command string, and either would end it early.
          host_docker_root="$(docker info \
            --format "{{.DockerRootDir}}" 2>/dev/null || true)"
          if [[ -n "${host_docker_root}" && -d "${host_docker_root}" ]]; then
            nested_docker_args+=(
              -v "${host_docker_root}:${host_docker_root}:ro"
            )
          else
            echo "WARN: could not resolve the Docker root on this host; the" \
              "SWE-bench disk preflight check will be skipped" >&2
          fi
        fi
      fi
      docker run --rm --name "${container}" \
        --network host --ipc host --privileged \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --cap-add IPC_LOCK --cap-add NET_ADMIN \
        --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
        --shm-size 128G \
        --env-file "'"${ENV_FILE}"'" \
        -e ATOMESH_EXECUTION_PHASE="${execution_phase}" \
        -e ATOMESH_SERVICE_PORT_OFFSET="${service_port_offset}" \
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
        "${nested_docker_args[@]}" \
        "'"${DOCKER_IMAGE}"'" \
        bash -lc "cd /workspace/ATOM && bash .github/scripts/atomesh/pd_server_atom.sh" \
        2>&1 | tee "${rank_dir}/${container_log}"
    '
done
unset ATOMESH_EXECUTION_PHASE ATOMESH_SERVICE_PORT_OFFSET

echo "=== Slurm job completed ==="
find "${RUN_DIR}" -maxdepth 3 -type f | sort
