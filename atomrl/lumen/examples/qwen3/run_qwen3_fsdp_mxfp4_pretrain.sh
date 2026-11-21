#!/bin/bash
# Qwen3-8B full-parameter pretraining with FSDP2 + MXFP4 on gfx950.
#
# Required:
#   MODEL_PATH=/path/to/Qwen3-8B
#   TRAIN_DATA_PATH=/path/to/train.jsonl
#
# Optional: VALID_DATA_PATH, NPROC, MBS, GBS, SEQ_LEN, TRAIN_STEPS, LR,
# WARMUP_STEPS, PRECISION (mxfp4|bf16), MXFP4_COMM (1|0), RESULTS_DIR,
# INIT_FROM_SCRATCH (1|0), TRAIN_SAMPLES (0 = whole corpus),
# SHARDING (full_shard|shard_grad_op), EVAL_INTERVAL,
# EXTRA_TRAIN_ARGS (extra trainer flags).
#
# For step time, pass the three bf16 kernel swaps through EXTRA_TRAIN_ARGS:
#
#   EXTRA_TRAIN_ARGS="--aiter-attn --lumen-norm --fuse-rope"
#
# They replace attention, RMSNorm and RoPE, which stay bf16 at every PRECISION,
# so they are orthogonal to MXFP4 and are left off by default here. Measured
# together on Qwen3-8B, 8x MI350X, GBS 16 (grad accum 1), seq 8192, median of
# steps 21-250: 1310 ms against 1844 ms for MXFP4 alone (1.41x) and 1951 ms for
# BF16 (1.49x). Held-out val loss stayed inside the band that two identical
# MXFP4 baselines span, so this buys step time without a precision trade. Both
# figures are grad accum 1; GBS 128 amortizes the optimizer step over eight
# micro-batches and has not been measured.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen3 model/config directory}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to text or JSONL with a text field}"
VALID_DATA_PATH="${VALID_DATA_PATH:-}"
PRECISION="${PRECISION:-mxfp4}"
NPROC="${NPROC:-8}"
MBS="${MBS:-2}"
GBS="${GBS:-128}"
SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
LR="${LR:-1.0e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
MXFP4_COMM="${MXFP4_COMM:-1}"
INIT_FROM_SCRATCH="${INIT_FROM_SCRATCH:-1}"
SHARDING="${SHARDING:-full_shard}"
EVAL_INTERVAL="${EVAL_INTERVAL:-150}"
# Word-split on purpose: this is how a caller reaches the trainer's optional
# flags (--aiter-attn, --lumen-norm, --fuse-rope, --no-grad-checkpointing)
# without a launcher variable per flag.
read -r -a EXTRA_TRAIN_ARGS <<< "${EXTRA_TRAIN_ARGS:-}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

if [[ "${PRECISION}" != "mxfp4" && "${PRECISION}" != "bf16" ]]; then
    echo "ERROR: PRECISION must be mxfp4 or bf16" >&2
    exit 2
fi
if (( GBS % (NPROC * MBS) != 0 )); then
    echo "ERROR: GBS=${GBS} must be divisible by NPROC*MBS=$((NPROC * MBS))" >&2
    exit 2
fi
GRAD_ACCUM=$((GBS / (NPROC * MBS)))
# Default to the samples this run actually consumes (plus a little slack), so
# startup does not tokenize a corpus far larger than the step count needs.
TRAIN_SAMPLES="${TRAIN_SAMPLES:-$(( GBS * (TRAIN_STEPS + 2) ))}"

EXTRA_ARGS=(--train-samples "${TRAIN_SAMPLES}")
if [[ "${INIT_FROM_SCRATCH}" != "0" ]]; then
    EXTRA_ARGS+=(--init-from-scratch)
fi
if [[ "${PRECISION}" == "mxfp4" ]]; then
    EXTRA_ARGS+=(--mode mxfp4 --first-last-layers-bf16
                 --num-layers-at-start-in-bf16 0
                 --num-layers-at-end-in-bf16 5)
    if [[ "${MXFP4_COMM}" != "0" ]]; then
        EXTRA_ARGS+=(--fsdp-mxfp4-comm)
    fi
else
    EXTRA_ARGS+=(--mode bf16)
fi
if [[ -n "${VALID_DATA_PATH}" ]]; then
    EXTRA_ARGS+=(--val-data-path "${VALID_DATA_PATH}")
fi

mkdir -p "${RESULTS_DIR}"
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
export LUMEN_MXFP4_AUTOTUNE_CACHE="${LUMEN_MXFP4_AUTOTUNE_CACHE:-${RESULTS_DIR}/mxfp4_autotune_qwen3_fsdp.json}"

torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/train_qwen3_fsdp.py" \
    --model-name-or-path "${MODEL_PATH}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --task pretrain --lora-rank 0 \
    --fsdp-version 2 --sharding "${SHARDING}" \
    --seq-length "${SEQ_LEN}" \
    --micro-batch-size "${MBS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-steps "${TRAIN_STEPS}" \
    --lr "${LR}" --lr-warmup-steps "${WARMUP_STEPS}" \
    --weight-decay 0.1 --max-grad-norm 1.0 \
    --eval-interval "${EVAL_INTERVAL}" --num-workers 0 --seed 1234 \
    "${EXTRA_ARGS[@]}" ${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"} \
    2>&1 | tee "${RESULTS_DIR}/qwen3_fsdp_pretrain_${PRECISION}.log"
