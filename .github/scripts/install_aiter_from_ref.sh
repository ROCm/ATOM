#!/usr/bin/env bash
# Reinstall amd-aiter inside a running benchmark container from a requested
# ROCm/aiter ref. Intended for manual benchmark runs; callers skip this script
# when no ref was provided so scheduled/default runs keep the image's package.
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${AITER_GIT_REF:?AITER_GIT_REF is required}"

docker exec \
  -e AITER_GIT_REF="${AITER_GIT_REF}" \
  "${CONTAINER_NAME}" \
  bash -lc '
    set -euo pipefail

    echo "=== AITER version BEFORE reinstall ==="
    python3 -m pip show amd-aiter || true

    echo "=== Reinstalling amd-aiter from ROCm/aiter ref: ${AITER_GIT_REF} ==="
    python3 -m pip uninstall -y amd-aiter || true
    python3 -m pip install --upgrade "pybind11>=3.0.1"
    python3 -m pip show pybind11

    rm -rf /app/aiter-test
    git clone --filter=blob:none --no-checkout https://github.com/ROCm/aiter.git /app/aiter-test
    cd /app/aiter-test
    git fetch --filter=blob:none --depth=1 origin "${AITER_GIT_REF}"
    git checkout --detach FETCH_HEAD
    git submodule sync
    git submodule update --init --recursive

    MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 python3 setup.py develop

    echo "=== AITER version AFTER reinstall ==="
    python3 -m pip show amd-aiter
  '
