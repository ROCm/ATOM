#!/usr/bin/env bash
# Replace the image-baked AITER with a pinned pull-request revision.
#
# This file is sourced by pd_server_atom.sh so the job-local virtualenv and
# PYTHONPATH remain active for the benchmark servers.
set -euo pipefail

AITER_REPO="${ATOMESH_AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_REF="${ATOMESH_AITER_GIT_REF:-refs/pull/5066/head}"
AITER_SHA="${ATOMESH_AITER_GIT_SHA:?ATOMESH_AITER_GIT_SHA is required}"
AITER_MAX_JOBS="${ATOMESH_AITER_MAX_JOBS:-16}"
AITER_FLYDSL_VERSION="${ATOMESH_AITER_FLYDSL_VERSION:-0.3.2}"
AITER_WORK_ROOT="${ATOMESH_AITER_WORK_ROOT:-${XDG_CACHE_HOME:-/tmp}/atomesh-aiter-${AITER_SHA:0:12}}"
AITER_SOURCE_DIR="${AITER_WORK_ROOT}/src"
AITER_VENV_DIR="${AITER_WORK_ROOT}/venv"
AITER_BASE_PYTHON="$(command -v python3)"

echo "=== Replacing image AITER with ${AITER_REPO}@${AITER_SHA} ==="
echo "=== Image AITER before uninstall ==="
"${AITER_BASE_PYTHON}" -m pip show amd-aiter || true

image_aiter_root="$(
  "${AITER_BASE_PYTHON}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("aiter")
if spec and spec.submodule_search_locations:
    print(Path(next(iter(spec.submodule_search_locations))).resolve().parent)
PY
)"

# The container is ephemeral. Remove the image distribution when permissions
# allow it; non-root Spur containers still get a job-local virtualenv below,
# which shadows an image distribution that pip cannot physically remove.
if ! "${AITER_BASE_PYTHON}" -m pip uninstall -y amd-aiter; then
  echo "WARN: image amd-aiter could not be uninstalled; the job-local install will shadow it" >&2
fi

clean_aiter_artifacts() {
  local root="$1"
  local path
  local so

  [[ -n "${root}" && -d "${root}" ]] || return 0
  for path in "${root}/build" "${root}/aiter/jit/build"; do
    if [[ -e "${path}" ]]; then
      rm -rf "${path}" 2>/dev/null \
        || echo "WARN: could not remove stale AITER build directory ${path}" >&2
    fi
  done
  shopt -s nullglob
  for so in "${root}"/aiter/jit/*.so; do
    rm -f "${so}" 2>/dev/null \
      || echo "WARN: could not remove stale AITER module ${so}" >&2
  done
  shopt -u nullglob
}

clean_aiter_artifacts "${image_aiter_root}"
for cache_dir in "${AITER_JIT_DIR:-}" "${AITER_CACHE_DIR:-}"; do
  case "${cache_dir}" in
    ""|/) continue ;;
  esac
  rm -rf "${cache_dir}" 2>/dev/null \
    || echo "WARN: could not remove stale AITER cache ${cache_dir}" >&2
done

rm -rf "${AITER_WORK_ROOT}"
mkdir -p "${AITER_SOURCE_DIR}"
git -C "${AITER_SOURCE_DIR}" init
git -C "${AITER_SOURCE_DIR}" remote add origin "${AITER_REPO}"
git -C "${AITER_SOURCE_DIR}" fetch --depth 1 origin "${AITER_REF}"
if ! git -C "${AITER_SOURCE_DIR}" cat-file -e "${AITER_SHA}^{commit}" 2>/dev/null; then
  git -C "${AITER_SOURCE_DIR}" fetch --depth 1 origin "${AITER_SHA}"
fi
git -C "${AITER_SOURCE_DIR}" checkout --detach "${AITER_SHA}"

actual_sha="$(git -C "${AITER_SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_sha}" != "${AITER_SHA}" ]]; then
  echo "ERROR: ${AITER_REF} resolved to ${actual_sha}, expected ${AITER_SHA}" >&2
  exit 1
fi
git -C "${AITER_SOURCE_DIR}" submodule update --init --recursive --depth 1

"${AITER_BASE_PYTHON}" -m venv --system-site-packages "${AITER_VENV_DIR}"
# shellcheck disable=SC1091
source "${AITER_VENV_DIR}/bin/activate"
python3 -m pip install --upgrade --force-reinstall --no-deps \
  "flydsl==${AITER_FLYDSL_VERSION}"
AITER_USE_SYSTEM_TRITON=1 MAX_JOBS="${AITER_MAX_JOBS}" \
  python3 -m pip install -e "${AITER_SOURCE_DIR}" \
    --no-build-isolation --no-deps

AITER_PREBUILT_JIT_DIR="${AITER_SOURCE_DIR}/aiter/jit"
aiter_base_site_packages="$(
  "${AITER_BASE_PYTHON}" - <<'PY'
import site

print(":".join(site.getsitepackages()))
PY
)"

export PATH="${AITER_VENV_DIR}/bin:${PATH}"
export PYTHONPATH="${AITER_SOURCE_DIR}:${aiter_base_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
export ATOMESH_AITER_PREBUILT_JIT_DIR="${AITER_PREBUILT_JIT_DIR}"
hash -r

# Editable installation builds module_aiter_core, while module_cache remains a
# JIT module. Build and exercise the PR's DCP case once before any workers
# start, avoiding a multi-process first-use race and rejecting a stale 18-arg
# module_cache.so. PYTHONPATH above exposes the image venv (torch, pandas, …)
# to the job-local interpreter.
env -u AITER_CACHE_DIR -u AITER_JIT_DIR \
  AITER_REBUILD=1 AITER_USE_SYSTEM_TRITON=1 MAX_JOBS="${AITER_MAX_JOBS}" \
  python3 "${AITER_SOURCE_DIR}/op_tests/test_indexer_qk_rope_quant_and_cache.py" \
    -n 8 --num_heads 32 -d bf16 --valid_fraction 0.5 \
    --compute_all_q_rope 1 --is_neox 1

if [[ ! -f "${AITER_PREBUILT_JIT_DIR}/module_cache.so" ]]; then
  echo "ERROR: AITER module_cache.so was not produced" >&2
  exit 1
fi

env -u AITER_CACHE_DIR -u AITER_JIT_DIR python3 - <<'PY'
import importlib.metadata

import aiter
from aiter.jit.core import get_module

module_cache = get_module("module_cache")
doc = module_cache.indexer_qk_rope_quant_and_cache.__doc__.splitlines()[0]
if "compute_all_q_rope" not in doc:
    raise RuntimeError(f"module_cache has the old fused-indexer ABI: {doc}")
print(f"Installed amd-aiter: {importlib.metadata.version('amd-aiter')}")
print(f"Imported aiter from: {aiter.__file__}")
print(f"Verified module_cache ABI: {doc}")
PY
