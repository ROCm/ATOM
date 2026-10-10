#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-}" == --container ]]; then
  export PIP_CACHE_DIR=/tmp/mooncake-store-smoke-pip-cache
  python3 -m venv --system-site-packages /tmp/mooncake-store-smoke-venv
  /tmp/mooncake-store-smoke-venv/bin/python -m pip install --no-deps \
    /scripts/mooncake-dist/*.whl
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export MC_GID_INDEX=1
  cat /proc/meminfo > /results/meminfo-before.txt
  ulimit -a > /results/ulimits.txt
  /tmp/mooncake-store-smoke-venv/bin/python \
    /scripts/pd_mooncake_store_smoke.py --output /results/cases
  exit
fi

if [[ "${1:-}" == --worker ]]; then
  image="$2"
  output="$3"
  job_id="${SLURM_JOB_ID:-${SPUR_JOB_ID}}"
  run_token="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
  hostname > "${output}/node.txt"
  scontrol show job "${job_id}" > "${output}/job.txt"
  ATOMESH_EXCLUSIVE_GPU_CLEANUP=1 python3 "${script_dir}/pd_gpu_cleanup.py" \
    --job-id "${job_id}" --run-token "${run_token}" \
    --node "$(hostname -s)" --rank 0 --cell-id mooncake-store-smoke \
    --out "${output}/gpu-preflight.json"
  container="atomesh-mooncake-store-smoke-${job_id}-0"
  cleanup() {
    local query_rc=0
    timeout 20s docker rm -f "${container}" > "${output}/cleanup.log" 2>&1 || true
    timeout 10s docker ps -a --format '{{.Names}}' \
      > "${output}/containers-after.txt" 2>&1 || query_rc=$?
    echo "${query_rc}" > "${output}/cleanup-query.rc"
  }
  trap cleanup EXIT
  docker pull "${image}"
  docker run --name "${container}" --user "$(id -u):$(id -g)" \
    -e STORE_SMOKE_OUTER_HOST_IP="$(hostname -I | awk '{print $1}')" \
    --group-add video --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --network host --ipc host \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --cap-add=IPC_LOCK --cap-add=NET_ADMIN --cap-add=SYS_NICE \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
    -v "${script_dir}:/scripts:ro" -v "${output}:/results" \
    --entrypoint /bin/bash "${image}" /scripts/pd_mooncake_store_smoke.sh --container
  exit
fi

nodes="${1:?candidate nodes required}"
image="${2:?image required}"
output="${3:?output directory required}"
IFS=',' read -r -a candidates <<< "${nodes}"
(( ${#candidates[@]} >= 1 && ${#candidates[@]} <= 4 ))
for node in "${candidates[@]}"; do
  [[ "${node}" =~ ^pit2-p03-g[0-9]+$ ]]
  [[ ",${ATOMESH_NODE_POOL}," == *",${node},"* ]]
done
mkdir -p "${output}"
output="$(cd "${output}" && pwd)"
name="mooncake-store-smoke-${GITHUB_RUN_ID}"
cancel_diagnostic() {
  while IFS='|' read -r job_id job_name; do
    if [[ "${job_id}" =~ ^[0-9]+$ && "${job_name}" == "${name}" ]]; then
      timeout 10s scancel "${job_id}" || true
    fi
  done < <(timeout 15s squeue --noheader --format='%i|%j')
}
trap cancel_diagnostic EXIT
timeout --kill-after=20s 21000s srun --job-name "${name}" \
  --account amd-frameworks --partition amd-spur --qos amd-frameworks-qos \
  --exclusive --nodes 1 --ntasks 1 --cpus-per-task 16 --gpus 8 \
  --time 00:20:00 --nodelist "${nodes}" \
  bash "${script_dir}/pd_mooncake_store_smoke.sh" --worker "${image}" "${output}" \
  > "${output}/slurm.log" 2>&1
