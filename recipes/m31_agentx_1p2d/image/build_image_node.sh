#!/usr/bin/env bash
# Stage 2 (host side, per node, CPU only): build rocm/atom-dev:m31-agx-local-<short> on NODE from the stage-1 tar.
# Pulls the base if the node lacks it; copies the tar from $AGX_IMAGE_ART/<short>/ to node-local disk if needed. The
# context's aot/ holds this checkout's two decode GEMM tables, renamed for aiter's table matching, and
# $AGX_IMAGE_ART/<short>/c17-aot-<short>.tar as aot/prewarm.tar if present; without it the build runs that AOT replay.
# Never pushes. Prints "NODE TAG IMAGE_ID". DOCKER_CONFIG is node-local, so buildx state never lands in an NFS home.
# Usage: AGX_IMAGE_ART=<dir> build_image_node.sh NODE SHA [RETARGET_SHA RETARGET_BUNDLE]
#   SHA: the full commit stage 1 prebuilt (names the tar); RETARGET_SHA (same tree) names the image and its checkout.
# Env: AGX_BASE_IMAGE, AGX_IMAGE_TAG, AGX_IMAGE_NODE_DIR, AGX_HOST_PREFIX.
set -euo pipefail
NODE=$1 SHA=$2 RSHA=${3:-} RBUNDLE=${4:-}
SHORT=${SHA:0:8}
TSHORT=${RSHA:0:8}; TSHORT=${TSHORT:-${SHORT}}
BASE=${AGX_BASE_IMAGE:-rocm/atom-dev:nightly_202611240650}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CFG=$(cd "${HERE}/../../../recipe/configs" && pwd)
P=${AGX_HOST_PREFIX-pit2-p03-}
ART=${AGX_IMAGE_ART:?set AGX_IMAGE_ART to a directory that NODE mounts at the same path}/${SHORT}
TAG=${AGX_IMAGE_TAG:-rocm/atom-dev:m31-agx-local-${TSHORT}}
TAR=aiter-tree-${SHORT}.tar
ssh -o BatchMode=yes "${P}${NODE}" bash -s <<REMOTE
set -euo pipefail
docker image inspect ${BASE} >/dev/null 2>&1 || docker pull -q ${BASE}
L=${AGX_IMAGE_NODE_DIR:-/var/tmp/m31agx-aiter}
mkdir -p \${L}/ctx-${SHORT}
if [[ ! -s \${L}/${TAR} ]]; then cp ${ART}/${TAR} \${L}/${TAR}.part && mv \${L}/${TAR}.part \${L}/${TAR}; fi
ln -f \${L}/${TAR} \${L}/ctx-${SHORT}/${TAR} 2>/dev/null || cp \${L}/${TAR} \${L}/ctx-${SHORT}/${TAR}
cp ${HERE}/Dockerfile \${L}/ctx-${SHORT}/Dockerfile
mkdir -p \${L}/ctx-${SHORT}/aot
cp ${CFG}/mx8_tuned_gemm_m31.csv \${L}/ctx-${SHORT}/aot/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm_m31c17.csv
cp ${CFG}/bf16_tuned_gemm_m31.csv \${L}/ctx-${SHORT}/aot/bf16_tuned_gemm_m31c17.csv
printf '%s\n' "source: ROCm/ATOM origin/p1-c17-reproducible 759de87d9 recipe/configs/ (ganyi C17 decode recipe tables, set by recipe/int1_recipe.sh as AITER_CONFIG_GEMM_BF16 / AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE)" \
  "renamed so aiter.aot.flydsl.gemm's table_of() matches each file by table name: mx8_tuned_gemm_m31.csv -> a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm_m31c17.csv, bf16_tuned_gemm_m31.csv -> bf16_tuned_gemm_m31c17.csv" \
  > \${L}/ctx-${SHORT}/aot/SOURCE.txt
if [[ -s ${ART}/c17-aot-${SHORT}.tar ]]; then cp ${ART}/c17-aot-${SHORT}.tar \${L}/ctx-${SHORT}/aot/prewarm.tar; fi
if [[ -n "${RBUNDLE}" ]]; then cp ${RBUNDLE} \${L}/ctx-${SHORT}/aot/retarget.bundle; fi
mkdir -p \${L}/docker-config && DOCKER_CONFIG=\${L}/docker-config DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} --build-arg BASE_IMAGE=${BASE} --build-arg AITER_SHA=${SHA} --build-arg AITER_TAR=${TAR} --build-arg RETARGET_SHA=${RSHA} \${L}/ctx-${SHORT} > \${L}/build-${SHORT}.log 2>&1 || { tail -40 \${L}/build-${SHORT}.log; exit 1; }
echo "${NODE} ${TAG} \$(docker image inspect --format '{{.Id}}' ${TAG})"
REMOTE
