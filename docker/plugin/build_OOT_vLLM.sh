#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/build_OOT_vLLM_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${LOG_DIR}"
# Mirror all stdout/stderr to terminal and log file.
exec > >(tee -a "${LOG_FILE}") 2>&1

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile_OOT_vLLM"
BASE_IMAGE="${BASE_IMAGE:-rocm/atom-dev:latest}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-b31e9326a7d9394aab8c767f8ebe225c65594b60}"
VLLM_VERSION="${VLLM_VERSION:-0.17}"
VLLM_COMMIT_SHORT="$(printf '%s' "${VLLM_COMMIT}" | cut -c1-6)"
IMAGE_REPO="${IMAGE_REPO:-rocm/atom-vllm-dev}"
IMAGE_TAG="${IMAGE_TAG:-${IMAGE_REPO}:v${VLLM_VERSION}-${VLLM_COMMIT_SHORT}}"
MAX_JOBS="${MAX_JOBS:-64}"
INSTALL_LM_EVAL="${INSTALL_LM_EVAL:-1}"
PULL_BASE_IMAGE="${PULL_BASE_IMAGE:-1}"
BUILD_NO_CACHE="${BUILD_NO_CACHE:-1}"

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
echo "vLLM version    : ${VLLM_VERSION}"
echo "vLLM commit     : ${VLLM_COMMIT}"
echo "commit short    : ${VLLM_COMMIT_SHORT}"
echo "MAX_JOBS        : ${MAX_JOBS}"
echo "INSTALL_LM_EVAL : ${INSTALL_LM_EVAL}"
echo "BUILD_NO_CACHE  : ${BUILD_NO_CACHE}"
echo
echo "Build plan:"
echo "  Step 1/4: (optional) pull base image"
echo "  Step 2/4: check/remove existing target image"
echo "  Step 3/4: build image from Dockerfile_OOT_vLLM"
echo "  Step 4/4: print final image info"
echo

if [[ "${PULL_BASE_IMAGE}" == "1" ]]; then
  print_banner "Step 1/4 - Pull base image: ${BASE_IMAGE}"
  docker pull "${BASE_IMAGE}"
else
  print_banner "Step 1/4 - Skip base image pull (PULL_BASE_IMAGE=${PULL_BASE_IMAGE})"
fi

print_banner "Step 2/4 - Check whether target image already exists"
if docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  echo "Target image already exists: ${IMAGE_TAG}"
  docker image inspect "${IMAGE_TAG}" --format 'Existing image -> ID={{.Id}}  Created={{.Created}}'
  echo "Removing existing target image: ${IMAGE_TAG}"
  docker image rm -f "${IMAGE_TAG}"
else
  echo "Target image does not exist yet: ${IMAGE_TAG}"
fi
echo

print_banner "Step 3/4 - Build target image: ${IMAGE_TAG}"
NO_CACHE_FLAG=""
if [[ "${BUILD_NO_CACHE}" == "1" ]]; then
  NO_CACHE_FLAG="--no-cache"
fi

DOCKER_BUILDKIT=1 docker build \
  ${NO_CACHE_FLAG} \
  -f "${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  --build-arg "INSTALL_LM_EVAL=${INSTALL_LM_EVAL}" \
  "$@" \
  "${REPO_ROOT}"

print_banner "Step 4/4 - Build completed"
docker image inspect "${IMAGE_TAG}" --format 'Image={{.RepoTags}}  ID={{.Id}}  Created={{.Created}}'
