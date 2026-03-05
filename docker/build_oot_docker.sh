#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile_For_OOT"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="f5d17400303149bbb480f6abfb6f7bb646c1d895"
VLLM_DOCKER_REF="${VLLM_DOCKER_REF:-${VLLM_COMMIT}}"
IMAGE_TAG="${IMAGE_TAG:-atom-vllm-oot}"
VLLM_ROCM_BASE_IMAGE="${VLLM_ROCM_BASE_IMAGE:-rocm/dev-ubuntu-22.04:7.0-complete}"
VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-rocm/vllm-dev:base}"
BASE_IMAGE="${BASE_IMAGE:-${VLLM_BASE_IMAGE}}"
BUILD_VLLM_BASE="${BUILD_VLLM_BASE:-1}"

echo "========================================"
echo "OOT Docker build config"
echo "  Base image       : ${BASE_IMAGE}"
echo "  VLLM commit      : ${VLLM_COMMIT}"
echo "  Final image name : ${IMAGE_TAG}"
echo "  Build vLLM base  : ${BUILD_VLLM_BASE}"
echo "========================================"

if [[ "${BUILD_VLLM_BASE}" == "1" ]]; then
  VLLM_TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${VLLM_TMP_DIR}"' EXIT

  echo "Step 1/2: build vLLM ROCm base image"
  echo "  vLLM repo   : ${VLLM_REPO}"
  echo "  vLLM ref    : ${VLLM_DOCKER_REF}"
  echo "  Base image  : ${VLLM_ROCM_BASE_IMAGE}"
  echo "  Image tag   : ${VLLM_BASE_IMAGE}"

  git clone "${VLLM_REPO}" "${VLLM_TMP_DIR}"
  git -C "${VLLM_TMP_DIR}" checkout "${VLLM_DOCKER_REF}"

  VLLM_BASE_DOCKERFILE="${VLLM_TMP_DIR}/docker/Dockerfile.rocm_base"
  if [[ ! -f "${VLLM_BASE_DOCKERFILE}" ]]; then
    echo "ERROR: cannot find ${VLLM_BASE_DOCKERFILE}"
    exit 1
  fi

  DOCKER_BUILDKIT=1 docker build \
    -f "${VLLM_BASE_DOCKERFILE}" \
    -t "${VLLM_BASE_IMAGE}" \
    --build-arg "BASE_IMAGE=${VLLM_ROCM_BASE_IMAGE}" \
    "${VLLM_TMP_DIR}"

  BASE_IMAGE="${VLLM_BASE_IMAGE}"
else
  echo "Step 1/2: skip vLLM base build (use existing ${VLLM_BASE_IMAGE})"
fi

echo "Step 2/2: build vLLM + ATOM OOT image"
echo "  Dockerfile : ${DOCKERFILE_PATH}"
echo "  Image tag  : ${IMAGE_TAG}"
echo "  Base image : ${BASE_IMAGE}"
echo "  VLLM commit: ${VLLM_COMMIT}"

DOCKER_BUILDKIT=1 docker build \
  -f "${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  "$@" \
  "${REPO_ROOT}"

echo "Build finished."
echo "  Final image name : ${IMAGE_TAG}"
echo "  Base image used  : ${BASE_IMAGE}"
echo "  VLLM commit used : ${VLLM_COMMIT}"

