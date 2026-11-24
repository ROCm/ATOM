#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Extract validation loss from Lumen log and format as MLPerf eval_accuracy
# for direct comparison with MLPerf reference results.
#
# Usage: bash extract_eval_accuracy.sh <lumen_log> [<mlperf_result>]

set -euo pipefail

LUMEN_LOG="${1:-./lumen_launch_full.log}"
MLPERF_RESULT="${2:-}"

echo "============================================================"
echo " Lumen vs MLPerf Reference: eval_accuracy (val_loss)"
echo "============================================================"
echo ""

# Extract Lumen validation loss
echo "--- Lumen Validation Loss (from ${LUMEN_LOG}) ---"
echo "  Step   Samples  val_loss     PPL"
echo "  ----   -------  --------     ---"
grep "validation loss at iteration" "${LUMEN_LOG}" | while IFS= read -r line; do
    iter=$(echo "${line}" | grep -oP 'iteration \K\d+')
    loss=$(echo "${line}" | grep -oP 'lm loss value: \K[0-9.eE+-]+')
    ppl=$(echo "${line}" | grep -oP 'lm loss PPL: \K[0-9.eE+-]+')
    samples=$((iter * 8))
    printf "  %-6s %-8s %-12s %s\n" "${iter}" "${samples}" "${loss}" "${ppl}"
done

# Extract latest training loss at eval checkpoints
echo ""
echo "--- Lumen Training Loss (at eval checkpoints) ---"
echo "  Step   Samples  train_loss   grad_norm"
echo "  ----   -------  ----------   ---------"
for step in 48 96 144 192 240 288 336 384 432 480; do
    line=$(grep "iteration[[:space:]]*${step}/" "${LUMEN_LOG}" | tail -1 2>/dev/null || true)
    if [ -n "${line}" ]; then
        loss=$(echo "${line}" | grep -oP 'lm loss: \K[0-9.eE+-]+')
        gnorm=$(echo "${line}" | grep -oP 'grad norm: \K[0-9.]+')
        samples=$((step * 8))
        printf "  %-6s %-8s %-12s %s\n" "${step}" "${samples}" "${loss}" "${gnorm}"
    fi
done

if [ -n "${MLPERF_RESULT}" ] && [ -f "${MLPERF_RESULT}" ]; then
    echo ""
    echo "--- MLPerf Reference eval_accuracy (from ${MLPERF_RESULT}) ---"
    echo "  Step   Samples  eval_accuracy (val_loss)"
    echo "  ----   -------  -----------------------"
    grep "eval_accuracy" "${MLPERF_RESULT}" | while IFS= read -r line; do
        val=$(echo "${line}" | grep -oP '"value": \K[0-9.]+')
        samples=$(echo "${line}" | grep -oP '"samples_count": \K\d+')
        step=$((samples / 8))
        printf "  %-6s %-8s %s\n" "${step}" "${samples}" "${val}"
    done

    echo ""
    echo "============================================================"
    echo " KEY COMPARISON"
    echo "============================================================"
    echo " MLPerf 'eval_accuracy' IS the validation loss (per-token CE)."
    echo " MLPerf target: val_loss < 0.925"
    echo ""
    echo " Lumen 'lm loss value' in validation IS the same metric."
    echo " If Lumen val_loss >> 1.0, the model is not converging properly."
    echo "============================================================"
fi

echo ""
echo "--- Summary ---"
echo "Lumen converged val_loss:"
grep "validation loss at iteration" "${LUMEN_LOG}" | tail -1 | grep -oP 'lm loss value: \K[0-9.eE+-]+'
echo ""
echo "MLPerf target: val_loss < 0.925"
