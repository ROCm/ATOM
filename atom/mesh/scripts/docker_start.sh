#!/usr/bin/env bash
set -euo pipefail

# Start the atom-mesh Docker container.
#
# Optional env (with defaults):
#   CONTAINER=atom_sglang_mesh
#   DOCKER_IMAGE=rocm/atom-dev:mesh-sglang-latest

#CONTAINER="${CONTAINER:-atom_sglang_mesh}"
CONTAINER="${CONTAINER:-atom_mesh_dsv4}"
#DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:mesh-sglang-latest}"
#DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:nightly_202605271131-Jasen-atom_mesh}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:mesh_dsv4}"
echo "[docker] starting container=${CONTAINER} image=${DOCKER_IMAGE}"

docker rm -f "${CONTAINER}" 2>/dev/null || true

docker run -d --name "${CONTAINER}" \
    --network host --ipc host --privileged \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    --cap-add IPC_LOCK --cap-add NET_ADMIN \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
    -v /mnt:/mnt -v /it-share:/it-share \
    "${DOCKER_IMAGE}" sleep infinity

# Fix ionic RDMA ABI mismatch: host kernel module may be newer than the
# libionic bundled in the image.  Copy the host's matching .so into the
# container so ibv_devices can see the Pensando NICs.
HOST_IONIC="$(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.* 2>/dev/null \
              | grep -v '\.a$' | head -1 || true)"
if [[ -n "${HOST_IONIC}" && -f "${HOST_IONIC}" ]]; then
    IONIC_NAME="$(basename "${HOST_IONIC}")"
    docker cp "${HOST_IONIC}" "${CONTAINER}:/usr/lib/x86_64-linux-gnu/${IONIC_NAME}"
    docker exec "${CONTAINER}" bash -c "
        cd /usr/lib/x86_64-linux-gnu
        ln -sf '${IONIC_NAME}' libionic.so.1
        cp -f '${IONIC_NAME}' libibverbs/libionic-rdmav34.so 2>/dev/null || true
    "
    echo "[docker] patched libionic → ${IONIC_NAME}"
fi

echo "[docker] container ${CONTAINER} started"
