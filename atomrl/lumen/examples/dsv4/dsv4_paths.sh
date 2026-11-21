#!/usr/bin/env bash
# Shared path defaults for examples/dsv4/*.sh — source, do not execute directly.
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/dsv4_paths.sh"
#
# Override via env: WORKSPACE_ROOT, AITER_DIR, DATA_ROOT,
# MODEL_DIR, LOG_DIR, TVM_CACHE_DIR, BOOTSTRAP_DIR, NFS_ROOT, MEGATRON_PATH,
# MEGATRON_ROCM_DIR, MEGATRON_ROCM_REF.

: "${SCRIPT_DIR:?dsv4_paths.sh: set SCRIPT_DIR before sourcing}"

# shellcheck source=examples/dsv4/dsv4_bf16_gemm_env.sh
source "${SCRIPT_DIR}/dsv4_bf16_gemm_env.sh"

LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${LUMEN_DIR}/.." && pwd)}"

MILES_DIR="${MILES_DIR:-${WORKSPACE_ROOT}/miles}"
AITER_DIR="${AITER_DIR:-${WORKSPACE_ROOT}/aiter}"

if [[ -z "${DATA_ROOT:-}" ]]; then
    for _dsv4_data_candidate in \
        "/nfs/data/${USER}" \
        "/mnt/data/${USER}" \
        "${WORKSPACE_ROOT}/dsv4-data"; do
        if [[ -d "${_dsv4_data_candidate}" ]]; then
            DATA_ROOT="${_dsv4_data_candidate}"
            break
        fi
    done
    DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/dsv4-data}"
fi

DATA_DIR="${DATA_DIR:-${DATA_ROOT}/datasets}"
NFS_ROOT="${NFS_ROOT:-/nfs/data}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-${DATA_ROOT}/lumen-dsv4-bootstrap}"
MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs}"
TVM_CACHE_DIR="${TVM_CACHE_DIR:-${DATA_ROOT}/tvm-cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_ROOT}/pip-cache}"
MEGATRON_ROCM_DIR="${MEGATRON_ROCM_DIR:-${DATA_ROOT}/Megatron-LM-rocm-dev}"
MEGATRON_ROCM_REF="${MEGATRON_ROCM_REF:-fb4552449f9b33c6f72207a80e80045eadf5267e}"
MEGATRON_PATH="${MEGATRON_PATH:-${MEGATRON_ROCM_DIR}}"
TILELANG_DIR="${TILELANG_DIR:-${DATA_ROOT}/tilelang}"
PYTHON_EXTRAS_DIR="${PYTHON_EXTRAS_DIR:-${DATA_ROOT}/python-extras}"

unset _dsv4_data_candidate
