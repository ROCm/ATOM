#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build the AtomRL + vLLM + ATOM Docker image for MI350 (ROCm gfx950)
#
# Usage:
#   bash examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/docker/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-atomrl-vllm-mi350:latest}"
AITER_SHA="${AITER_SHA:-adbb6ac8b84110134e0cf4dc513b250d2cc0a75c}"
MORI_SHA="${MORI_SHA:-9c42e3240b33359331b2cf2141b28886055da583}"

echo "Building ${IMAGE_NAME}..."
echo "  ATOM checkout: ${ATOM_ROOT}"
echo "  Lumen: ${ATOM_ROOT}/atomrl/lumen"
echo "  AITER: ROCm/aiter@${AITER_SHA}"
echo "  MORI:  ROCm/mori@${MORI_SHA}"

docker buildx build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --build-arg "AITER_SHA=${AITER_SHA}" \
    --build-arg "MORI_SHA=${MORI_SHA}" \
    -t "${IMAGE_NAME}" \
    "${ATOM_ROOT}"

echo "Done: ${IMAGE_NAME}"
