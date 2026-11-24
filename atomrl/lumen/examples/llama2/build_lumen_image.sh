#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build a Docker image with the latest Lumen code baked in on top of the
# base lumen/llama2:latest image. Run from the Lumen repo root or from anywhere.
set -euo pipefail

LUMEN_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ATOM_ROOT="$(cd "${LUMEN_DIR}/../.." && pwd)"
TAG="${1:-dev}"
IMAGE="lumen/llama2:${TAG}"

echo "[build] Context: ${ATOM_ROOT}"
echo "[build] Target:  ${IMAGE}"

docker build \
    --file "${ATOM_ROOT}/atomrl/docker/Dockerfile.lumen" \
    --tag  "${IMAGE}" \
    "${ATOM_ROOT}"

echo "[build] Done: ${IMAGE}"
echo "Run training with:"
echo "  IMAGE=${IMAGE} HOST_MODEL=/path/to/model HOST_DATA=/path/to/data bash examples/llama2/run_fsdp_lora_7b.sh"
