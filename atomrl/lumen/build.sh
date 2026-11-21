#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-lumen/tests:latest}"

echo "==> Building test image: ${IMAGE_NAME}"
docker build \
    -f "${ATOM_ROOT}/atomrl/docker/Dockerfile.lumen" \
    -t "${IMAGE_NAME}" \
    "${ATOM_ROOT}"

echo "==> Running tests"
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=16g \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    "${IMAGE_NAME}" \
    pytest tests/ops/ -v --tb=short "$@"
