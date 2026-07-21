#!/usr/bin/env bash
# Build and install a requested ROCm/aiter git ref inside a running CI container.
# Inputs via env: CONTAINER_NAME (required), AITER_GIT_REF (required).
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME must be set}"
: "${AITER_GIT_REF:?AITER_GIT_REF must be set}"

if [ "$AITER_GIT_REF" = "main" ]; then
  echo "ERROR: install_aiter_branch.sh is only for non-main refs" >&2
  exit 1
fi

echo "=== Building ROCm/aiter ref ${AITER_GIT_REF} ==="
docker exec -e AITER_GIT_REF="$AITER_GIT_REF" "$CONTAINER_NAME" bash -lc '
  set -euo pipefail

  echo "=== Uninstalling existing amd-aiter ==="
  pip uninstall -y amd-aiter || true
  pip install --upgrade "pybind11>=3.0.1"

  rm -rf /app/aiter-test
  git clone --filter=blob:none --single-branch --branch "$AITER_GIT_REF" \
    https://github.com/ROCm/aiter.git /app/aiter-test
  cd /app/aiter-test
  git submodule sync
  git submodule update --init --recursive
  MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 python3 setup.py develop

  echo "=== Installed requested AITER revision ==="
  git rev-parse HEAD
  pip show amd-aiter
'
