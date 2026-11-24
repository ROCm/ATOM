#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build lumen/dsv4-lumen:mi308x from atomrl.lumen/tests:latest + bootstrap.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ATOM_ROOT="$(cd "${LUMEN_DIR}/../.." && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

LUMEN_IMAGE="${LUMEN_IMAGE:-lumen/tests:latest}"
IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"
STAGING="${LUMEN_DIR}/examples/dsv4/.bootstrap-build"

if ! docker image inspect "${LUMEN_IMAGE}" &>/dev/null; then
    if [[ "${LUMEN_IMAGE}" == "lumen/tests:latest" ]] \
        && docker image inspect "lumen/dsv4-lumen:mi308x" &>/dev/null; then
        echo "[WARN] ${LUMEN_IMAGE} missing — rebuilding from atomrl.lumen/dsv4-lumen:mi308x base"
        LUMEN_IMAGE="lumen/dsv4-lumen:mi308x"
    else
        echo "[ERROR] Lumen base image missing: ${LUMEN_IMAGE}"
        echo "        Build it with: bash build.sh   (tags lumen/tests:latest)"
        exit 1
    fi
fi

if [[ ! -f "${BOOTSTRAP_DIR}/.ready" ]]; then
    echo "[prepare] bootstrap missing — running prepare_bootstrap.sh"
    bash "${SCRIPT_DIR}/prepare_bootstrap.sh"
fi

echo "[staging] ${BOOTSTRAP_DIR} -> ${STAGING}"
rm -rf "${STAGING}"
mkdir -p "${STAGING}"
# Megatron is a symlink; the host tilelang tree is too large to stage.
rsync -a \
    --copy-links \
    --exclude 'tilelang/' \
    --exclude 'sglang-python/' \
    "${BOOTSTRAP_DIR}/" "${STAGING}/"

echo "==> Building ${IMAGE} (base=${LUMEN_IMAGE})"
docker build -f "${LUMEN_DIR}/examples/dsv4/Dockerfile" \
    --build-arg "LUMEN_IMAGE=${LUMEN_IMAGE}" \
    -t "${IMAGE}" \
    "${ATOM_ROOT}"

echo "==> Done: ${IMAGE}"
echo "Run: SKIP_PREPARE=1 GBS=256 NUM_ROLLOUT=10 IMAGE=${IMAGE} bash examples/dsv4/run_dsv4.sh"
