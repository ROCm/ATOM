#!/usr/bin/env bash
# Copy a locally built image from SRC_NODE to other nodes without any registry: docker save to node-local disk on
# SRC_NODE, copy it to $AGX_IMAGE_ART/<short>/image-<short>.tar once, docker load on each target. The archive is
# complete, so the load does not depend on the targets' base layers. Same image ID everywhere. Never pushes.
# Usage: AGX_IMAGE_ART=<dir> distribute_image.sh SRC_NODE TAG NODE...   Env: AGX_IMAGE_NODE_DIR, AGX_HOST_PREFIX.
set -euo pipefail
SRC=$1 TAG=$2; shift 2
SHORT=${TAG##*-}
P=${AGX_HOST_PREFIX-pit2-p03-}
L=${AGX_IMAGE_NODE_DIR:-/var/tmp/m31agx-aiter}
ART=${AGX_IMAGE_ART:?set AGX_IMAGE_ART to a directory that every node mounts at the same path}/${SHORT}
F=${ART}/image-${SHORT}.tar
mkdir -p "${ART}"
if [[ ! -s ${F} ]]; then
  ssh -o BatchMode=yes "${P}${SRC}" "mkdir -p ${L} && docker save -o ${L}/image-${SHORT}.tar ${TAG} && cp ${L}/image-${SHORT}.tar ${F}.part && mv ${F}.part ${F}"
fi
ls -la "${F}"
for n in "$@"; do
  ( ssh -o BatchMode=yes "${P}${n}" "docker load -q -i ${F} >/dev/null && echo ${n} ${TAG} \$(docker image inspect --format '{{.Id}}' ${TAG})" ) &
done
wait
echo "source ${SRC} ${TAG} $(ssh -o BatchMode=yes "${P}${SRC}" docker image inspect --format '{{.Id}}' "${TAG}")"
