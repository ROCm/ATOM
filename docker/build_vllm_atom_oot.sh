#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile_vllm_atom_oot"
IMAGE_TAG="${IMAGE_TAG:-atom-vllm-oot}"
BASE_IMAGE="${BASE_IMAGE:-rocm/vllm-dev:nightly_main_20260118}"
ATOM_COMMIT="${ATOM_COMMIT:-HEAD}"
AITER_COMMIT="${AITER_COMMIT:-HEAD}"
BUILD_MORI="${BUILD_MORI:-0}"

echo "========================================"
echo "Build vLLM + ATOM OOT image"
echo "  Dockerfile : ${DOCKERFILE_PATH}"
echo "  Image name : ${IMAGE_TAG}"
echo "  Base image : ${BASE_IMAGE}"
echo "  ATOM commit: ${ATOM_COMMIT}"
echo "  AITER commit: ${AITER_COMMIT}"
echo "  Build MORI : ${BUILD_MORI}"
echo "========================================"

DOCKER_BUILDKIT=1 docker build \
  -f "${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "ATOM_COMMIT=${ATOM_COMMIT}" \
  --build-arg "AITER_COMMIT=${AITER_COMMIT}" \
  --build-arg "BUILD_MORI=${BUILD_MORI}" \
  "$@" \
  "${REPO_ROOT}"

echo "Build finished: ${IMAGE_TAG}"
