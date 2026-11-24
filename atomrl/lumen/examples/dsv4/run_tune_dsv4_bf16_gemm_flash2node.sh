#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Tune 2-node flash DSV4 BF16 GEMM gaps (lm_head, wq_b buckets, 7168-K).
# cu_num auto-detected via get_cu_num() at merge time.
#
# Usage:
#   bash examples/dsv4/run_tune_dsv4_bf16_gemm_flash2node.sh
#   TUNE_LIBTYPE=asm,triton,skinny,opus TUNE_BATCH=5 bash examples/dsv4/run_tune_dsv4_bf16_gemm_flash2node.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"
# shellcheck source=examples/dsv4/dsv4_gemm_detect_cu.sh
source "${SCRIPT_DIR}/dsv4_gemm_detect_cu.sh"

IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"
GEMM_TUNE_DIR="${LUMEN_DIR}/examples/dsv4/.gemm_tune"
UNTUNED="${GEMM_TUNE_DIR}/dsv4_bf16_flash2node_untuned.csv"
OVERLAY="${GEMM_TUNE_DIR}/dsv4_bf16_tuned_flash2node_gfx942.csv"
LOGFILE="${LOG_DIR}/dsv4_bf16_gemm_tune_flash2node_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}" "${GEMM_TUNE_DIR}"

TARGET_CU="$(dsv4_detect_gemm_cu)"
echo "[run_tune_flash2node] auto cu_num=${TARGET_CU}"

python3 "${LUMEN_DIR}/examples/dsv4/generate_dsv4_bf16_moe_bucket_untuned.py" \
    --profile flash2node \
    --min-m 1 \
    --max-m 4096 \
    --include-existing \
    --target-cu "${TARGET_CU}" \
    --output "${UNTUNED}"

_n=$(($(wc -l < "${UNTUNED}") - 1))
if [[ "${_n}" -le 0 ]]; then
    echo "[run_tune_flash2node] nothing to tune — CSV already covers flash2node shapes for cu=${TARGET_CU}"
    exit 0
fi
echo "[run_tune_flash2node] ${_n} untuned shapes -> ${UNTUNED}"
echo "[run_tune_flash2node] log=${LOGFILE}"

docker run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc=host \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    -e LUMEN_DIR=/workspace/Lumen \
    -e AITER_ROOT=/opt/aiter \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e TUNE_INPUT="/workspace/Lumen/examples/dsv4/.gemm_tune/dsv4_bf16_flash2node_untuned.csv" \
    -e TUNE_OUTPUT="/workspace/Lumen/examples/dsv4/.gemm_tune/dsv4_bf16_tuned_flash2node_gfx942.csv" \
    -e TUNE_LIBTYPE="${TUNE_LIBTYPE:-asm,triton,skinny,opus}" \
    -e TUNE_BATCH="${TUNE_BATCH:-8}" \
    "${IMAGE}" \
    bash /workspace/Lumen/examples/dsv4/tune_dsv4_bf16_gemm_mi308x.sh \
    2>&1 | tee "${LOGFILE}"

python3 "${LUMEN_DIR}/examples/dsv4/merge_dsv4_bf16_gemm_gfx942.py" \
    --overlay "${OVERLAY}"

echo "[run_tune_flash2node] merged cu=${TARGET_CU} -> ${LUMEN_DIR}/examples/dsv4/configs/dsv4_bf16_tuned_gemm_mi308x.csv"
echo "[run_tune_flash2node] tune log -> ${LOGFILE}"
