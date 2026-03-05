#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/build_vllm_atom_oot_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${LOG_DIR}"
# Mirror all stdout/stderr to terminal and log file.
exec > >(tee -a "${LOG_FILE}") 2>&1

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile_vllm_atom_oot"
IMAGE_TAG="${IMAGE_TAG:-rocm/atom-vllm-dev:nightly_202603040155}"
BASE_IMAGE="${BASE_IMAGE:-rocm/atom-dev:nightly_202603040155}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-f5d17400303149bbb480f6abfb6f7bb646c1d895}"
MAX_JOBS="${MAX_JOBS:-64}"
INSTALL_LM_EVAL="${INSTALL_LM_EVAL:-1}"
PULL_BASE_IMAGE="${PULL_BASE_IMAGE:-1}"

print_banner() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

print_banner "Build vLLM on top of ATOM base image"
echo "Log file        : ${LOG_FILE}"
echo "Dockerfile      : ${DOCKERFILE_PATH}"
echo "Build context   : ${REPO_ROOT}"
echo "Target image    : ${IMAGE_TAG}"
echo "Base image      : ${BASE_IMAGE}"
echo "vLLM repo       : ${VLLM_REPO}"
echo "vLLM commit     : ${VLLM_COMMIT}"
echo "MAX_JOBS        : ${MAX_JOBS}"
echo "INSTALL_LM_EVAL : ${INSTALL_LM_EVAL}"
echo
echo "Build plan:"
echo "  Step 1/3: (optional) pull base image"
echo "  Step 2/3: build image from Dockerfile_vllm_atom_oot"
echo "  Step 3/3: print final image info"
echo

if [[ "${PULL_BASE_IMAGE}" == "1" ]]; then
  print_banner "Step 1/3 - Pull base image: ${BASE_IMAGE}"
  docker pull "${BASE_IMAGE}"
else
  print_banner "Step 1/3 - Skip base image pull (PULL_BASE_IMAGE=${PULL_BASE_IMAGE})"
fi

print_banner "Step 2/3 - Build target image: ${IMAGE_TAG}"
DOCKER_BUILDKIT=1 docker build \
  -f "${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  --build-arg "INSTALL_LM_EVAL=${INSTALL_LM_EVAL}" \
  "$@" \
  "${REPO_ROOT}"

print_banner "Step 3/3 - Build completed"
docker image inspect "${IMAGE_TAG}" --format 'Image={{.RepoTags}}  ID={{.Id}}  Created={{.Created}}'
