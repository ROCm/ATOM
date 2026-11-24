#!/usr/bin/env bash
# Start the recipe's three containers on this node (host, whole node):
# prefill (GPUs 0-3), decode (GPUs 4-7), store (no GPU). RUNS = run logs.
# Usage: containers.sh [TREE] [IMAGE]
#   TREE  ATOM tree mounted read-only at /workspace/ATOM and put on PYTHONPATH
#   IMAGE docker image
# HIP numbers the GPUs in KFD order, not PCI order, on pit2-p03 nodes, but HIP
# 0-3 are still the four NUMA0 GPUs and HIP 4-7 the four NUMA1 GPUs, so the
# roles select their GPUs with HIP_VISIBLE_DEVICES inside the containers.
set -euo pipefail
TREE=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
IMAGE=${2:-rocm/atom-dev:nightly_202611240650}
PFX=${PFX:-${USER%%@*}-m31pd}
RUNS=${RUNS:-${PWD}/runs}
mkdir -p "${RUNS}"

rdma=(--device /dev/infiniband --cap-add IPC_LOCK --cap-add NET_ADMIN
  --ulimit memlock=-1:-1 --ulimit nofile=65536:524288 --ulimit stack=67108864)
# The host's ionic provider, as the CI launcher mounts it (pd_slurm_job.sh).
host_ionic=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)
[[ -n "${host_ionic}" && -e "${host_ionic}" ]] && rdma+=(-v "${host_ionic}:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
[[ -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ]] && rdma+=(
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
[[ -e /etc/libibverbs.d/ionic.driver ]] && rdma+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)
common=(--network host --ipc host --shm-size 128g --cap-add SYS_NICE --cap-add SYS_PTRACE
  --security-opt seccomp=unconfined
  -v /share_nfs:/share_nfs -v "${TREE}:/workspace/ATOM:ro" -v "${RUNS}:/runs"
  -e PYTHONPATH=/workspace/ATOM --entrypoint sleep)
gpu=(--device /dev/kfd --device /dev/dri --group-add video --group-add render)

for role in prefill decode store; do
  docker rm -f "${PFX}-${role}" >/dev/null 2>&1 || true
done
docker run -d --name "${PFX}-prefill" "${gpu[@]}" "${rdma[@]}" "${common[@]}" "${IMAGE}" infinity
docker run -d --name "${PFX}-decode" "${gpu[@]}" "${rdma[@]}" "${common[@]}" "${IMAGE}" infinity
# The Store's master and owners see no GPU: Mooncake would probe every buffer through HIP.
docker run -d --name "${PFX}-store" "${rdma[@]}" "${common[@]}" "${IMAGE}" infinity
docker ps --filter "name=${PFX}" --format '{{.Names}} {{.Status}}'
