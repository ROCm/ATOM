#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Tune DSV4 BF16 GEMM on MI308X (gfx942). Run inside lumen/dsv4-lumen:mi308x.
#
# Env:
#   TUNE_INPUT    untuned CSV (default: .gemm_tune/dsv4_bf16_moe_bucket_untuned.csv)
#   TUNE_OUTPUT   gfx942 tuned overlay (default: .gemm_tune/dsv4_bf16_tuned_gfx942_mi308x.csv)
#   TUNE_LIBTYPE  comma backends (default: asm,triton,skinny,opus — no torch)
#   TUNE_BATCH    shapes per batch (default: 10)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
AITER_ROOT="${AITER_ROOT:-/opt/aiter}"

GEMM_TUNE_DIR="${LUMEN_DIR}/examples/dsv4/.gemm_tune"
mkdir -p "${GEMM_TUNE_DIR}"

TUNE_INPUT="${TUNE_INPUT:-${GEMM_TUNE_DIR}/dsv4_bf16_moe_bucket_untuned.csv}"
TUNE_OUTPUT="${TUNE_OUTPUT:-${GEMM_TUNE_DIR}/dsv4_bf16_tuned_gfx942_mi308x.csv}"
TUNE_LIBTYPE="${TUNE_LIBTYPE:-asm,triton,skinny,opus}"
TUNE_BATCH="${TUNE_BATCH:-10}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${AITER_ROOT}:${PYTHONPATH:-}"

cd "${AITER_ROOT}"

echo "[tune] input=${TUNE_INPUT}"
echo "[tune] output=${TUNE_OUTPUT}"
echo "[tune] libtype=${TUNE_LIBTYPE} batch=${TUNE_BATCH}"

python3 csrc/gemm_a16w16/gemm_tuner.py \
    --input_file "${TUNE_INPUT}" \
    --tuned_file "${TUNE_OUTPUT}" \
    --batch "${TUNE_BATCH}" \
    --libtype "${TUNE_LIBTYPE}"

echo "[tune] done -> ${TUNE_OUTPUT}"
wc -l "${TUNE_OUTPUT}"
