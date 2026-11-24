#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build lumen/mxfp4-lumen:gfx950 from atomrl.lumen/tests:latest + baked Lumen source.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ATOM_ROOT="$(cd "${LUMEN_DIR}/../.." && pwd)"

LUMEN_IMAGE="${LUMEN_IMAGE:-lumen/tests:latest}"
IMAGE="${IMAGE:-lumen/mxfp4-lumen:gfx950}"

if ! docker image inspect "${LUMEN_IMAGE}" &>/dev/null; then
    if [[ "${BUILD_BASE:-0}" == "1" ]]; then
        # build.sh would follow the build with tests/ops on one GPU; that is a
        # separate gate and a failure there must not abort this build.
        echo "==> Base ${LUMEN_IMAGE} missing — building it first"
        docker build -f "${LUMEN_DIR}/Dockerfile" -t "${LUMEN_IMAGE}" "${ATOM_ROOT}"
    else
        echo "[ERROR] Lumen base image missing: ${LUMEN_IMAGE}"
        echo "        bash build.sh            (tags lumen/tests:latest, then runs tests/ops)"
        echo "        BUILD_BASE=1 bash examples/qwen3/build_mxfp4_lumen_image.sh"
        exit 1
    fi
fi

echo "==> Building ${IMAGE} (base=${LUMEN_IMAGE})"
docker build -f "${LUMEN_DIR}/examples/qwen3/Dockerfile" \
    --build-arg "LUMEN_IMAGE=${LUMEN_IMAGE}" \
    -t "${IMAGE}" \
    "${ATOM_ROOT}"

echo "==> Done: ${IMAGE}"
echo "Run: IMAGE=${IMAGE} bash examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh"
