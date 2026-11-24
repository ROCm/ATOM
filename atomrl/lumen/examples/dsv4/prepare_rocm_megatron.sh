#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Clone ROCm/Megatron-LM (rocm_dev @ fb45524) and apply Lumen DSV4 patch.
#
# Usage:
#   bash examples/dsv4/prepare_rocm_megatron.sh
#
# List SOURCE patches (no PyTorch required):
#   PYTHONPATH="${ATOMRL_DIR}" python3 examples/dsv4/patch_megatron_source.py --list
#   PYTHONPATH="${ATOMRL_DIR}" python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
#   PYTHONPATH="${ATOMRL_DIR}" python3 examples/dsv4/patch_megatron_source.py --list --tag rocm
#
# Output: ${MEGATRON_ROCM_DIR} (default ${DATA_ROOT}/Megatron-LM-rocm-dev)
#
# See examples/dsv4/PATCHES.md for the full patch registry guide.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/ROCm/Megatron-LM.git}"
MEGATRON_ROCM_REF="${MEGATRON_ROCM_REF:-fb4552449f9b33c6f72207a80e80045eadf5267e}"
MEGATRON_ROCM_DIR="${MEGATRON_ROCM_DIR:-${DATA_ROOT}/Megatron-LM-rocm-dev}"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [[ -d /workspace/ATOM/atomrl/atomrl ]]; then
    _DEFAULT_ATOMRL_DIR=/workspace/ATOM/atomrl
else
    _DEFAULT_ATOMRL_DIR="$(cd "${LUMEN_DIR}/.." && pwd)"
fi
ATOMRL_DIR="${ATOMRL_DIR:-${_DEFAULT_ATOMRL_DIR}}"
PATCH_SCRIPT="${LUMEN_DIR}/examples/dsv4/patch_megatron_source.py"
STAMP_FILE="${MEGATRON_ROCM_DIR}/.lumen_rocm_megatron_ref"

if [[ -f "${STAMP_FILE}" && "$(cat "${STAMP_FILE}")" == "${MEGATRON_ROCM_REF}" \
    && -f "${MEGATRON_ROCM_DIR}/megatron/core/__init__.py" ]]; then
    echo "[rocm-megatron] already prepared at ${MEGATRON_ROCM_DIR} (${MEGATRON_ROCM_REF})"
    exit 0
fi

if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git is required to clone ROCm/Megatron-LM"
    exit 1
fi

mkdir -p "$(dirname "${MEGATRON_ROCM_DIR}")"
tmp_dir="${MEGATRON_ROCM_DIR}.prepare.$$"
rm -rf "${tmp_dir}"

echo "[rocm-megatron] cloning ${MEGATRON_REPO} @ ${MEGATRON_ROCM_REF}"
git clone --filter=blob:none "${MEGATRON_REPO}" "${tmp_dir}"
git -C "${tmp_dir}" checkout "${MEGATRON_ROCM_REF}"
git -C "${tmp_dir}" submodule update --init --recursive

echo "[rocm-megatron] applying DSV4 patch"
PYTHONPATH="${ATOMRL_DIR}:${PYTHONPATH:-}" \
    python3 "${PATCH_SCRIPT}" "${tmp_dir}"

rm -rf "${MEGATRON_ROCM_DIR}"
mv "${tmp_dir}" "${MEGATRON_ROCM_DIR}"
echo "${MEGATRON_ROCM_REF}" > "${STAMP_FILE}"

echo "[rocm-megatron] done: ${MEGATRON_ROCM_DIR} ($(git -C "${MEGATRON_ROCM_DIR}" rev-parse --short HEAD))"
