#!/usr/bin/env bash
set -euo pipefail

# Start the atom-mesh Docker container.
#
# Optional env (with defaults):
#   CONTAINER=atom_sglang_mesh
#   DOCKER_IMAGE=sglang-atom-mesh-20260420-v0.1:latest

CONTAINER="${CONTAINER:-atom_sglang_mesh}"
DOCKER_IMAGE="${DOCKER_IMAGE:-sglang-atom-mesh-20260420-v0.1:latest}"

echo "[docker] starting container=${CONTAINER} image=${DOCKER_IMAGE}"

docker rm -f "${CONTAINER}" 2>/dev/null || true

docker run -d --name "${CONTAINER}" \
    --network host --ipc host --privileged \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    --cap-add IPC_LOCK --cap-add NET_ADMIN \
    -v /mnt:/mnt -v /it-share:/it-share \
    "${DOCKER_IMAGE}" sleep infinity

echo "[docker] container ${CONTAINER} started"
