#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Qwen3-8B LoRA SFT with FSDP2 + MXFP4 on gfx950.
#
# Bash entry point for the MXFP4 SFT case. run_qwen3_fsdp.sh rejects
# MODE=mxfp4, and run_qwen3_fsdp_mxfp4_pretrain.sh is the full-parameter run.
#
# Required:
#   MODEL_PATH=/path/to/Qwen3-8B
#   TRAIN_DATA_PATH=/path/to/alpaca.jsonl
#
# Optional: VALID_DATA_PATH, NPROC, MBS, GRAD_ACCUM, SEQ_LEN, TRAIN_STEPS, LR,
# WARMUP_STEPS, MXFP4_COMM (1|0, default 0), RESULTS_DIR, EVAL_INTERVAL,
# EXTRA_TRAIN_ARGS (extra trainer flags).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen3 model directory}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to alpaca-style JSONL}"
VALID_DATA_PATH="${VALID_DATA_PATH:-}"
NPROC="${NPROC:-8}"
MBS="${MBS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
LR="${LR:-4e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MXFP4_COMM="${MXFP4_COMM:-0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
read -r -a EXTRA_TRAIN_ARGS <<< "${EXTRA_TRAIN_ARGS:-}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

EXTRA_ARGS=()
if [[ "${MXFP4_COMM}" != "0" ]]; then
    EXTRA_ARGS+=(--fsdp-mxfp4-comm)
fi
if [[ -n "${VALID_DATA_PATH}" ]]; then
    EXTRA_ARGS+=(--val-data-path "${VALID_DATA_PATH}")
fi

mkdir -p "${RESULTS_DIR}"

# Same fetch as the Megatron cases: examples/qwen3/tokenizer, not the model dir.
source "${LUMEN_ROOT}/examples/scripts/ensure_tokenizer.sh"
example_tokenizer qwen3 "${SCRIPT_DIR}/tokenizer" || exit 1
export PYTHONPATH="${LUMEN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export USE_HIPBLASLT="${USE_HIPBLASLT:-1}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-1}"

torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/train_qwen3_fsdp.py" \
    --model-name-or-path "${MODEL_PATH}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --task sft \
    --mode mxfp4 \
    --fsdp-version 2 \
    --seq-length "${SEQ_LEN}" \
    --micro-batch-size "${MBS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-steps "${TRAIN_STEPS}" \
    --lr "${LR}" --lr-warmup-steps "${WARMUP_STEPS}" \
    --eval-interval "${EVAL_INTERVAL}" --num-workers 0 --seed 1234 \
    --tokenizer-name-or-path "${SCRIPT_DIR}/tokenizer" \
    "${EXTRA_ARGS[@]}" ${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"} \
    2>&1 | tee "${RESULTS_DIR}/qwen3_fsdp_sft_mxfp4.log"
