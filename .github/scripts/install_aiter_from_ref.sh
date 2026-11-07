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
    python3 -m pip install "pybind11>=3.0.1" "setuptools_scm[toml]>=6.2" vcs_versioning wheel
    python3 -m pip show pybind11

    rm -rf /app/aiter-test
    git clone --filter=blob:none --no-checkout https://github.com/ROCm/aiter.git /app/aiter-test
    cd /app/aiter-test
    git fetch --filter=blob:none --depth=1 origin "${AITER_GIT_REF}"
    git checkout --detach FETCH_HEAD
    git submodule sync
    git submodule update --init --recursive

    # Keep the immutable image runtime. PEP 517 avoids the legacy
    # setup_requires resolver that fails on incomplete package metadata.
    AITER_USE_SYSTEM_TRITON=1 AITER_MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 \
      python3 -m pip install --no-build-isolation --no-deps -e .

    echo "=== AITER version AFTER reinstall ==="
    git rev-parse HEAD
    python3 -m pip show amd-aiter torch triton flydsl pybind11 setuptools setuptools_scm vcs_versioning
  '
