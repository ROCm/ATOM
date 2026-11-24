#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build the Lumen — LLaMA 2 Docker image.
#
# Usage:
#   bash examples/llama2/build.sh
#   BASE_IMAGE=rocm/7.0:custom bash examples/llama2/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_NAME=${IMAGE_NAME:-"lumen/llama2:latest"}
BASE_IMAGE=${BASE_IMAGE:-"rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1"}

echo "============================================================"
echo " Building Lumen — LLaMA 2 Docker Image"
echo "============================================================"
echo "  Image:      ${IMAGE_NAME}"
echo "  Base:       ${BASE_IMAGE}"
echo "  Context:    ${REPO_ROOT}"
echo "============================================================"

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${REPO_ROOT}"
