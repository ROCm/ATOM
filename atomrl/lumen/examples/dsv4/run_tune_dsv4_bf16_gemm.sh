#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Host wrapper: tune DSV4 BF16 GEMM on local gfx942 GPU (auto cu_num via get_cu_num).
#
# Usage:
#   bash examples/dsv4/run_tune_dsv4_bf16_gemm.sh
#   TUNE_LIBTYPE=asm,triton,skinny,opus TUNE_BATCH=10 bash examples/dsv4/run_tune_dsv4_bf16_gemm.sh
#
# Optional:
#   TUNE_INPUT=path/to/untuned.csv
#   CU_NUM=304  # force override (normally auto-detected)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"
# shellcheck source=examples/dsv4/dsv4_gemm_detect_cu.sh
source "${SCRIPT_DIR}/dsv4_gemm_detect_cu.sh"

IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"
GEMM_TUNE_DIR="${LUMEN_DIR}/examples/dsv4/.gemm_tune"
RUNTIME_CSV="${LUMEN_DIR}/examples/dsv4/configs/dsv4_bf16_tuned_gemm_mi308x.csv"
mkdir -p "${LOG_DIR}" "${GEMM_TUNE_DIR}"

TARGET_CU="$(dsv4_detect_gemm_cu)"
OVERLAY="$(dsv4_gemm_overlay_path "${TARGET_CU}")"
TUNE_INPUT="${TUNE_INPUT:-${GEMM_TUNE_DIR}/dsv4_bf16_moe_bucket_untuned.csv}"
TUNE_INPUT_CONTAINER="$(dsv4_gemm_tune_input_container "${TUNE_INPUT}")"
OVERLAY_CONTAINER="/workspace/Lumen/examples/dsv4/.gemm_tune/$(basename "${OVERLAY}")"
LOGFILE="${LOG_DIR}/dsv4_bf16_gemm_tune_cu${TARGET_CU}_$(date +%Y%m%d_%H%M%S).log"

echo "[run_tune] auto cu_num=${TARGET_CU} (gfx942)"
echo "[run_tune] overlay=${OVERLAY}"
echo "[run_tune] runtime CSV=${RUNTIME_CSV}"
echo "[run_tune] log=${LOGFILE}"

docker run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc=host \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    -e LUMEN_DIR=/workspace/Lumen \
    -e AITER_ROOT=/opt/aiter \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e TUNE_INPUT="${TUNE_INPUT_CONTAINER}" \
    -e TUNE_OUTPUT="${OVERLAY_CONTAINER}" \
    -e TUNE_LIBTYPE="${TUNE_LIBTYPE:-asm,triton,skinny,opus}" \
    -e TUNE_BATCH="${TUNE_BATCH:-10}" \
    "${IMAGE}" \
    bash -c '
        python3 - <<PY
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
cu, gfx = get_cu_num(), get_gfx()
print(f"[preflight] gfx={gfx} cu_num={cu}")
PY
        bash /workspace/Lumen/examples/dsv4/tune_dsv4_bf16_gemm_mi308x.sh
        test -s "${TUNE_OUTPUT}"
    ' \
    2>&1 | tee "${LOGFILE}"

python3 "${LUMEN_DIR}/examples/dsv4/merge_dsv4_bf16_gemm_gfx942.py" \
    --overlay "${OVERLAY}"

echo "[run_tune] merged cu=${TARGET_CU} -> ${RUNTIME_CSV}"
echo "[run_tune] tune log -> ${LOGFILE}"
