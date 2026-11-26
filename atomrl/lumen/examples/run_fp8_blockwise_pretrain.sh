#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Run llama2-7b / llama31-8b / qwen3-8b Megatron pretrain smoke (FP8 blockwise).
# TRAIN_STEPS defaults to 20. The inner scripts stay on delayed scaling; this
# wrapper rewrites those flags to blockwise before launch.
#
# Usage:
#   MODEL=llama2 bash examples/run_fp8_blockwise_pretrain.sh
#   MODEL=all  bash examples/run_fp8_blockwise_pretrain.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IMAGE="${IMAGE:-lumen:dev}"
MODEL="${MODEL:-all}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
MBS="${MBS:-1}"
GBS="${GBS:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/results/fp8_blockwise}"
mkdir -p "${LOG_DIR}"

# Patch FP8_ARGS in official scripts: delayed -> blockwise
fp8_blockwise_sed='s/--linear-fp8-scaling delayed/--linear-fp8-scaling blockwise --linear-fp8-block-size 128/; s/--linear-fp8-amax-history 1024/--linear-fp8-amax-history 4/; s/--lr-warmup-iters 2/--lr-warmup-iters 1/'

run_one() {
    local name="$1" script="$2"
    local log="${LOG_DIR}/${name}_fp8_blockwise_${TRAIN_STEPS}step_$(date +%Y%m%d_%H%M%S).log"
    local tmp_script
    tmp_script="$(mktemp /tmp/lumen_pretrain_XXXXXX.sh)"
    sed "${fp8_blockwise_sed}" "${script}" > "${tmp_script}"
    chmod +x "${tmp_script}"
    echo "=== ${name} → ${log}"
    IMAGE="${IMAGE}" LUMEN_DIR="${LUMEN_DIR}" TRAIN_STEPS="${TRAIN_STEPS}" MBS="${MBS}" GBS="${GBS}" SEQ_LEN="${SEQ_LEN}" PRECISION=fp8 \
        bash "${tmp_script}" 2>&1 | tee "${log}"
    rm -f "${tmp_script}"
    if grep -qE 'ChildFailedError|Traceback' "${log}"; then
        echo "[FAIL] ${name} — see ${log}"
        return 1
    elif grep -qE "iteration[[:space:]]+${TRAIN_STEPS}/" "${log}" || \
         grep -qE '\[DONE\]|\[done\]' "${log}"; then
        echo "[OK] ${name} completed"
    else
        echo "[FAIL] ${name} did not reach ${TRAIN_STEPS} iterations — see ${log}" >&2
        return 1
    fi
}

case "${MODEL}" in
    llama2)  run_one llama2-7b  "${LUMEN_DIR}/examples/llama2/run_pretrain_llama2_7b.sh" ;;
    llama31) run_one llama31-8b "${LUMEN_DIR}/examples/llama31/run_pretrain_llama31_8b.sh" ;;
    qwen3)   run_one qwen3-8b   "${LUMEN_DIR}/examples/qwen3/run_pretrain_qwen3_8b.sh" ;;
    all)
        run_one llama2-7b  "${LUMEN_DIR}/examples/llama2/run_pretrain_llama2_7b.sh"
        run_one llama31-8b "${LUMEN_DIR}/examples/llama31/run_pretrain_llama31_8b.sh"
        run_one qwen3-8b   "${LUMEN_DIR}/examples/qwen3/run_pretrain_qwen3_8b.sh"
        ;;
    *) echo "Unknown MODEL=${MODEL}"; exit 1 ;;
esac

echo "[DONE] logs in ${LOG_DIR}"
