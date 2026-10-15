#!/usr/bin/env bash
# Invoked only by the ATOMesh benchmark Slurm job. No model downloads/builds.
set -euo pipefail
: "${REPRO_IMAGE:?}"
: "${REPRO_REPO_ROOT:?}"
: "${REPRO_OUT_DIR:?}"
: "${SLURM_JOB_ID:?Run through the ATOMesh benchmark CI Slurm allocation}"
container="atomesh-m3-repro-${SLURM_JOB_ID}-$$"
mkdir -p "${REPRO_OUT_DIR}"
exec > >(tee -a "${REPRO_OUT_DIR}/rank.log") 2>&1
cleanup() {
  local rc=$?
  trap - EXIT
  timeout 20 docker rm -f "${container}" >/dev/null 2>&1 || true
  printf '%s\n' "${rc}" > "${REPRO_OUT_DIR}/rank-rc.tmp"
  mv "${REPRO_OUT_DIR}/rank-rc.tmp" "${REPRO_OUT_DIR}/rank-rc"
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'node=%s slurm_job=%s image=%s\n' "$(hostname)" "${SLURM_JOB_ID}" "${REPRO_IMAGE}"
# Slurm selects the physical GPUs; never assume this job owns GPUs 0-3.
visible="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}}}"
if [[ -z "${visible}" ]]; then
  echo 'ERROR: Slurm did not supply allocated GPU visibility' >&2
  exit 1
fi
printf 'allocated_gpu_visibility=%s\n' "${visible}"
timeout --kill-after=20 300 docker pull "${REPRO_IMAGE}"
docker image inspect --format '{{json .RepoDigests}}' "${REPRO_IMAGE}" > "${REPRO_OUT_DIR}/image-digests.json"
# All four ranks share this node/network namespace. Pin bootstrap sockets to
# loopback: auto-detection chose eno1 on pit2-p03-g34, whose local address was
# unreachable (run 38703620681). GPU P2P transport is still chosen by RCCL.
docker run -d --name "${container}" \
  --network host --ipc host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_NICE --shm-size=16G \
  -e ROCR_VISIBLE_DEVICES="${visible}" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=1 \
  -e TRITON_CACHE_DIR=/tmp/m3-repro-triton \
  -e NCCL_SOCKET_IFNAME==lo -e NCCL_SOCKET_FAMILY=AF_INET \
  -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,COLL \
  -e TORCH_NCCL_TRACE_BUFFER_SIZE=20000 -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  -e TORCH_NCCL_DESYNC_DEBUG=1 -e TORCH_NCCL_ENABLE_TIMING=1 \
  -e TORCH_NCCL_TRACE_CPP_STACK=1 \
  -e TORCH_NCCL_ENABLE_MONITORING=1 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=90 \
  -v "${REPRO_REPO_ROOT}:/workspace/ATOM:ro" \
  -v "${REPRO_OUT_DIR}:/out" -w /workspace/ATOM \
  "${REPRO_IMAGE}" sleep infinity
# Each subcase has a fresh process group and a hard process-group timeout. A hung
# subcase ends the allocation; don't schedule more GPU work onto a stalled device.
for spec in none:0 two:0 single:0 two:244; do
  mode="${spec%:*}"
  resident="${spec#*:}"
  name="${mode}-resident${resident}"
  mkdir -p "${REPRO_OUT_DIR}/${name}"
  set +e
  docker exec -e "TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/out/${name}/nccl_trace_rank_" \
    -e "NCCL_DEBUG_FILE=/out/${name}/rccl.%h.%p.log" \
    "${container}" timeout --signal=TERM --kill-after=15 180 \
    python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    --log-dir "/out/${name}/torchrun" --redirects 3 --tee 3 \
    .github/scripts/atomesh/minimax_prefill_repro.py \
    --mode "${mode}" --resident-gib "${resident}" --cycles 64 --out "/out/${name}" \
    2>&1 | tee "${REPRO_OUT_DIR}/${name}/launcher.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "${rc}" > "${REPRO_OUT_DIR}/${name}/exit-code"
  if [[ "${rc}" != 0 ]]; then
    echo "Subcase ${name} failed rc=${rc}; inspect rank logs before classifying as the original hang."
    exit "${rc}"
  fi
  # Fail closed: all four ranks must finish and write their own results.
  python3 - "${REPRO_OUT_DIR}/${name}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for rank in range(4):
    result = json.loads((root / f"result_rank{rank}.json").read_text())
    assert result["rank"] == rank
    assert result["steps"] == 128
    assert result["result"] == "completed_without_reproducing_hang"
print(f"{root.name}: all 4 ranks completed; this reduction did not reproduce the hang.")
PY
done
