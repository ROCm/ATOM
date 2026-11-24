#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Compare Lumen vs MLPerf reference LLaMA2-70B LoRA finetune results.
#
# Extracts key metrics from training logs: loss, step time, memory, FP8 config.
#
# Usage:
#   bash compare_results.sh

set -euo pipefail

LUMEN_LOG="${LUMEN_LOG:-./lumen_launch_full.log}"
MLPERF_LOG="${MLPERF_LOG:-./mlperf_llama2_mi300x_reference.log}"
GPU_MEM_CSV="${GPU_MEM_CSV:-./gpu_memory_lumen.csv}"

fmt_header() { printf "\n%-40s %-30s %-30s\n" "$1" "LUMEN (Megatron)" "REFERENCE (NeMo+TE)"; printf "%-40s %-30s %-30s\n" "$(printf '%0.s-' {1..38})" "$(printf '%0.s-' {1..28})" "$(printf '%0.s-' {1..28})"; }
fmt_row()    { printf "%-40s %-30s %-30s\n" "$1" "$2" "$3"; }

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  LLaMA2-70B LoRA Finetune — Lumen vs MLPerf Reference (MI300X)       ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

# ---- Extract Lumen metrics ---------------------------------------------------
L_STEPS="N/A"; L_LOSS="N/A"; L_STEPTIME="N/A"; L_MEM="N/A"; L_MAXMEM="N/A"
if [ -f "${LUMEN_LOG}" ]; then
    L_LAST_ITER=$(grep "iteration.*elapsed time" "${LUMEN_LOG}" | tail -1)
    if [ -n "${L_LAST_ITER}" ]; then
        L_STEPS=$(echo "${L_LAST_ITER}" | grep -oP 'iteration\s+\K\d+')
        L_LOSS=$(echo "${L_LAST_ITER}" | grep -oP 'lm loss:\s*\K[0-9.E+-]+')
        L_STEPTIME=$(echo "${L_LAST_ITER}" | grep -oP 'elapsed time per iteration \(ms\):\s*\K[0-9.]+')
        L_MEM_FRAC=$(echo "${L_LAST_ITER}" | grep -oP 'mem usages:\s*\K[0-9.]+')
    fi
    L_RANK0_MEM=$(grep "\[Rank 0\].*memory" "${LUMEN_LOG}" | tail -1)
    if [ -n "${L_RANK0_MEM}" ]; then
        L_ALLOC=$(echo "${L_RANK0_MEM}" | grep -oP 'allocated:\s*\K[0-9.]+' | head -1)
        L_MAXALLOC=$(echo "${L_RANK0_MEM}" | grep -oP 'max allocated:\s*\K[0-9.]+' | head -1)
        L_RESERVED=$(echo "${L_RANK0_MEM}" | grep -oP 'max reserved:\s*\K[0-9.]+' | head -1)
        L_MEM="$(printf '%.0f' "${L_ALLOC}") MB"
        L_MAXMEM="$(printf '%.0f' "${L_MAXALLOC}") / $(printf '%.0f' "${L_RESERVED}") MB"
    fi

    L_FP8=$(grep "FP8 training enabled" "${LUMEN_LOG}" | head -1 || echo "N/A")
    L_TP=$(grep "tensor_model_parallel_size" "${LUMEN_LOG}" | tail -1 | awk '{print $NF}')
    L_SP=$(grep "sequence_parallel " "${LUMEN_LOG}" | tail -1 | awk '{print $NF}')
    L_LOSS_FIRST=$(grep "iteration.*2/" "${LUMEN_LOG}" | head -1 | grep -oP 'lm loss:\s*\K[0-9.E+-]+' || echo "N/A")

    L_AVG_STEPTIME=$(grep "elapsed time per iteration" "${LUMEN_LOG}" | tail -20 | \
        grep -oP 'elapsed time per iteration \(ms\):\s*\K[0-9.]+' | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
fi

# ---- Extract MLPerf reference metrics ----------------------------------------
R_STEPS="N/A"; R_LOSS="N/A"; R_STEPTIME="N/A"; R_MEM="N/A"; R_MAXMEM="N/A"
if [ -f "${MLPERF_LOG}" ]; then
    R_LAST_LINE=$(grep -E "step_time|train_loss|elapsed" "${MLPERF_LOG}" | tail -1 || echo "")
    R_STEPS=$(grep -cP "step_time|train_step" "${MLPERF_LOG}" || echo "N/A")
    R_LOSS=$(grep "train_loss" "${MLPERF_LOG}" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1 || echo "N/A")
    R_STEPTIME=$(grep "step_time" "${MLPERF_LOG}" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1 || echo "N/A")
    R_MEM=$(grep -iE "allocated|memory" "${MLPERF_LOG}" | tail -1 || echo "N/A")
else
    echo ""
    echo "  [!] MLPerf reference log not found at ${MLPERF_LOG}"
    echo "      The reference training may still be building/running."
fi

# ---- Print comparison --------------------------------------------------------
fmt_header "TRAINING PROGRESS"
fmt_row "Iterations completed" "${L_STEPS:-N/A} / 1024" "${R_STEPS:-N/A} / 1024"
fmt_row "Latest loss (lm_loss)" "${L_LOSS:-N/A}" "${R_LOSS:-N/A}"
fmt_row "First loss (iter 2)" "${L_LOSS_FIRST:-N/A}" "${R_LOSS:-N/A}"

fmt_header "PERFORMANCE"
fmt_row "Last step time (ms)" "${L_STEPTIME:-N/A}" "${R_STEPTIME:-N/A}"
fmt_row "Avg step time (last 20, ms)" "${L_AVG_STEPTIME:-N/A}" "${R_STEPTIME:-N/A}"
fmt_row "Samples/sec (approx)" \
    "$(echo "${L_AVG_STEPTIME:-0}" | awk '{if($1>0) printf "%.2f", 8000/$1; else print "N/A"}')" \
    "N/A"

fmt_header "GPU MEMORY (Rank 0)"
fmt_row "Current allocated" "${L_MEM:-N/A}" "${R_MEM:-N/A}"
fmt_row "Peak allocated + reserved" "${L_MAXMEM:-N/A}" "N/A"
fmt_row "Mem utilization fraction" "${L_MEM_FRAC:-N/A}" "N/A"

fmt_header "CONFIGURATION"
fmt_row "Tensor Parallelism" "${L_TP:-N/A}" "1"
fmt_row "Sequence Parallelism" "${L_SP:-N/A}" "False"
fmt_row "Framework" "Megatron-LM-AMD + Lumen" "NeMo + TransformerEngine"

echo ""
echo "FP8 Config (Lumen):"
echo "  ${L_FP8:-N/A}"
echo ""
echo "FP8 Config (Reference):"
echo "  FP8=True, format=e4m3, DPA=0, amax_algo=most_recent, history=4, activation=True"

# ---- GPU Memory CSV summary --------------------------------------------------
if [ -f "${GPU_MEM_CSV}" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo " GPU Memory Monitor Summary (from ${GPU_MEM_CSV})"
    echo "═══════════════════════════════════════════════════════════════════════"
    SAMPLES=$(tail -n +2 "${GPU_MEM_CSV}" | wc -l)
    if [ "${SAMPLES}" -gt 0 ]; then
        echo ""
        echo "  Total samples: ${SAMPLES} across 8 GPUs"
        echo ""
        echo "  Per-GPU peak VRAM usage (MB):"
        for gpu in 0 1 2 3 4 5 6 7; do
            PEAK=$(awk -F, -v g="${gpu}" '$2==g {if($3>max) max=$3} END {printf "%.0f", max}' "${GPU_MEM_CSV}")
            TOTAL=$(awk -F, -v g="${gpu}" '$2==g {print $4; exit}' "${GPU_MEM_CSV}")
            echo "    GPU ${gpu}: ${PEAK} MB / ${TOTAL} MB"
        done
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Logs:"
echo "   Lumen:   ${LUMEN_LOG}"
echo "   MLPerf:  ${MLPERF_LOG}"
echo "   GPU Mem: ${GPU_MEM_CSV}"
echo "═══════════════════════════════════════════════════════════════════════"
