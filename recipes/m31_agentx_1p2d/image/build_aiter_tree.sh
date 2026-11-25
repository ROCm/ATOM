#!/usr/bin/env bash
# Stage 1 (host side, CPU only, no GPU): prebuild the combined aiter once on NODE, in a detached GPU-less container of
# the base image, niced and CPU-capped so the node's running jobs keep their CPUs.
# Output: <node dir>/aiter-tree-<short>.tar on NODE (AGX_IMAGE_NODE_DIR, default /var/tmp/m31agx-aiter; copy it to
# $AGX_IMAGE_ART/<short>/ to build the image on another node), built from the git bundle
# $AGX_IMAGE_ART/<short>/aiter-agx-<short>.bundle, which this script writes from AITER_REPO.
# Usage: AITER_REPO=<aiter clone> AGX_IMAGE_ART=<dir> build_aiter_tree.sh NODE BRANCH [CPUS] [SEED_DIR_ON_NODE]
#   BRANCH: a local branch of AITER_REPO; the bundle carries it as refs/heads/BRANCH.
#   AGX_IMAGE_ART: a directory that this host and NODE mount at the same path; so must this recipe be.
#   SEED_DIR_ON_NODE: dir with flydsl_cache/ + flydsl_aot/ of an earlier build (see aiter_prebuild_inner_v2.sh)
# Env: AGX_TOLERATE_AOT_8WAVE_OVERFLOW=1 passes through (see aiter_prebuild_inner_v2.sh); AGX_BASE_IMAGE, AGX_HOST_PREFIX.
set -euo pipefail
NODE=$1 BRANCH=$2 CPUS=${3:-48} SEED=${4:-}
BASE=${AGX_BASE_IMAGE:-rocm/atom-dev:nightly_202611240650}
BASE_AITER=82b6b4f19bf3e9b5fc1f33741d4a495b1d454993   # the base image's /app/aiter-test commit (bundle prerequisite)
REPO=${AITER_REPO:?set AITER_REPO to a clone of ROCm/aiter that has BRANCH and ${BASE_AITER}}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=${AGX_HOST_PREFIX-pit2-p03-}
L=${AGX_IMAGE_NODE_DIR:-/var/tmp/m31agx-aiter}
git -C "${REPO}" show-ref --verify -q "refs/heads/${BRANCH}" || { echo "${BRANCH} is not a local branch of ${REPO}" >&2; exit 2; }
SHA=$(git -C "${REPO}" rev-parse "${BRANCH}")
SHORT=${SHA:0:8}
ART=${AGX_IMAGE_ART:?set AGX_IMAGE_ART to a directory that NODE mounts at the same path}/${SHORT}
mkdir -p "${ART}"
BUNDLE=${ART}/aiter-agx-${SHORT}.bundle
[[ -s "${BUNDLE}" ]] || git -C "${REPO}" bundle create -q "${BUNDLE}" "${BASE_AITER}..${BRANCH}"
git -C "${REPO}" bundle list-heads "${BUNDLE}" | grep -q "^${SHA} " || { echo "bundle ${BUNDLE} does not carry ${SHA}" >&2; exit 2; }
U=$(id -un); C=${U%%@*}-agx-aiter-prebuild-${SHORT}
ssh -o BatchMode=yes "${P}${NODE}" "docker rm -f ${C} >/dev/null 2>&1; mkdir -p ${L} && docker run -d --name ${C} --network host --cpus ${CPUS} \
  -e AGX_TOLERATE_AOT_8WAVE_OVERFLOW=${AGX_TOLERATE_AOT_8WAVE_OVERFLOW:-0} \
  -v ${HERE}:${HERE}:ro -v ${ART}:${ART}:ro -v ${L}:/out ${SEED:+-v ${SEED}:/seed:ro} --entrypoint bash ${BASE} \
  ${HERE}/aiter_prebuild_inner_v2.sh ${SHA} ${BUNDLE} /out/aiter-tree-${SHORT}.tar ${CPUS} ${SEED:+/seed}"
echo "started ${C} on ${NODE}: sha ${SHA}; follow: ssh ${P}${NODE} docker logs -f ${C}"
