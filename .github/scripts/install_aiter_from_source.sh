#!/usr/bin/env bash
# Build and install aiter from a git ref inside the running CI container. Backs
# the workflow_dispatch `aiter_branch` input, which exists to validate an
# unmerged aiter branch against ATOM. This replaces install_aiter_wheel.sh
# (whose wheel always comes from aiter main), so the caller must run exactly one
# of the two.
# Inputs via env: CONTAINER_NAME, AITER_GIT_REF (required); AITER_REPO_URL,
# AITER_SRC_DIR, MAX_JOBS (defaults below).
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME must be set}"
: "${AITER_GIT_REF:?AITER_GIT_REF must be set}"
if [[ "${AITER_GIT_REF}" == -* || "${AITER_GIT_REF}" =~ [[:space:]] ]]; then
  echo "ERROR: AITER_GIT_REF must be a git ref / tag / commit and must not start with '-' or contain whitespace (got: ${AITER_GIT_REF})" >&2
  exit 1
fi
AITER_REPO_URL="${AITER_REPO_URL:-https://github.com/ROCm/aiter.git}"
AITER_SRC_DIR="${AITER_SRC_DIR:-/app/aiter-test}"
MAX_JOBS="${MAX_JOBS:-64}"

echo "=== Installing aiter ${AITER_GIT_REF} from ${AITER_REPO_URL} ==="
docker exec \
  -e AITER_REPO_URL="${AITER_REPO_URL}" \
  -e AITER_GIT_REF="${AITER_GIT_REF}" \
  -e AITER_SRC_DIR="${AITER_SRC_DIR}" \
  -e MAX_JOBS="${MAX_JOBS}" \
  "$CONTAINER_NAME" bash -lc '
  set -euo pipefail

  echo "=== amd-aiter version BEFORE uninstall ==="
  pip show amd-aiter || true
  pip uninstall -y amd-aiter || true

  # The base image ships an editable install rooted at AITER_SRC_DIR; rebuild
  # in place so nothing keeps pointing at the old tree.
  rm -rf "${AITER_SRC_DIR}"
  # Clone every ref (blobless) and then check out, so a branch, tag or commit
  # all work -- same contract as the aiter_commit input in docker-release.yaml.
  # Not a depth-1 clone: setuptools_scm needs tags to derive a real version.
  git clone --filter=blob:none "${AITER_REPO_URL}" "${AITER_SRC_DIR}"
  cd "${AITER_SRC_DIR}"
  git checkout "${AITER_GIT_REF}"
  git submodule sync && git submodule update --init --recursive
  echo "=== Building aiter $(git rev-parse --short HEAD) (${AITER_GIT_REF}) ==="

  pip install --upgrade setuptools_scm
  # requirements.txt is intentionally skipped: the base image already satisfies
  # aiter, and its pins would move flydsl/pandas out from under ATOM. GPU_ARCHS
  # is left unset so aiter targets the runner GPU it detects.
  PREBUILD_KERNELS=0 python3 setup.py develop

  echo "=== amd-aiter version AFTER installation ==="
  pip show amd-aiter
'
