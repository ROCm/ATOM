#!/usr/bin/env bash
set -euo pipefail

# podman pull docker.io/rocm/atom-dev:vllm-latest
# podman pull docker.io/rocm/atom-dev:MiniMax-M3-20260623

# podman pull docker.io/rocm/atom-dev:vllm-v0.19.0-nightly_20260603-aws
# podman pull docker.io/rocm/atom-dev:nightly_202605111702
# podman pull docker.io/rocm/atom-dev:vllm-v0.19.0-nightly_20260508_perf_prebuild
# podman pull docker.io/rocm/atom-dev:vllm-v0.19.0-nightly_20260430-kimi-k2
# podman pull docker.io/vllm/vllm-openai-rocm:nightly
# podman stop zxb_atom_vllm_m3
# podman rm zxb_atom_vllm_m3
# podman pull  docker.io/rocm/sgl-dev:v0.5.11-rocm700-mi35x-20260514

# podman pull docker.io/rocm/atom-dev:vllm-v0.19.0-nightly_20260520-qwen3
# docker pull docker.io/rocm/atom-dev:nightly_202607171835

CONTAINER_NAME=zxb_kimi3
IMAGE=docker.io/rocm/fw-bringup:gfx1250-atom-dev-20260715-tp4_pro_flash
NFS_SOURCE=172.16.1.6:/helios_msft_shared
MODEL_DIR=/data/models/Kimi-K3
HOSTNAME_ALIAS="$(hostname)"

if ! mount | rg -q "${NFS_SOURCE} on /mnt "; then
    mount -t nfs "${NFS_SOURCE}" /mnt
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Model config not found: ${MODEL_DIR}/config.json" >&2
    exit 1
fi

if [[ ! -e /dev/kfd ]]; then
    modprobe amdgpu
    sleep 5
fi

if [[ -w /proc/sys/kernel/numa_balancing ]]; then
    echo 0 > /proc/sys/kernel/numa_balancing
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker rm -f "${CONTAINER_NAME}"
fi

docker run \
    -d \
    --privileged \
    --network=host \
    --ipc=host \
    --security-opt seccomp=unconfined \
    --add-host "${HOSTNAME_ALIAS}:127.0.0.1" \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864 \
    --device=/dev/kfd \
    --device=/dev/dri \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -e GLOO_SOCKET_IFNAME=lo \
    -e NCCL_SOCKET_IFNAME=lo \
    -v /home/xiaobing:/workdir \
    -v /mnt/models:/mnt/models \
    --workdir /workdir \
    --name "${CONTAINER_NAME}" \
    "${IMAGE}"

docker exec "${CONTAINER_NAME}" bash -lc '
python - <<'"'"'PY'"'"'
from pathlib import Path
import shutil
pth = Path("/opt/venv/lib/python3.12/site-packages/easy-install.pth")
if pth.exists():
    lines = [line for line in pth.read_text().splitlines() if line.strip() != "/app/aiter"]
    pth.write_text("\n".join(lines) + ("\n" if lines else ""))
shutil.rmtree("/opt/venv/lib/python3.12/site-packages/aiter", ignore_errors=True)
PY
python -m pip uninstall -y atom amd-aiter >/tmp/pip_uninstall.log 2>&1 || true
AITER_USE_SYSTEM_TRITON=1 python -m pip install --force-reinstall --no-deps -e /workdir/aiter-k3 >/tmp/aiter_install.log 2>&1
python -m pip install --force-reinstall --no-deps "git+https://github.com/fla-org/flash-linear-attention.git" >/tmp/fla_install.log 2>&1
python -m pip install --force-reinstall --no-deps -e /workdir/ATOM-K3 >/tmp/atom_install.log 2>&1
'
