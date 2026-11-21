#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI308X 1-node 8-GPU configuration for LLaMA2-70B LoRA SFT via PyTorch FSDP.
#
# BF16 LoRA fine-tuning aligned with the MLPerf v5.1 Llama2-70B reference
# hyperparameters (see examples/llama2/README.md), but using the FSDP backend
# instead of Megatron and BF16 instead of FP8 (FSDP+FP8+LoRA is less validated).
#
# Usage (inside container, via run_fsdp_lora.sh):
#   CONFIG=config_MI308X_fsdp_lora_70b.sh BACKEND=fsdp bash run_finetune.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Capture env overrides BEFORE sourcing the base config (which exports these
# unconditionally and would otherwise clobber values passed via the launcher).
_OV_SAVE_INTERVAL="${SAVE_INTERVAL:-}"
_OV_EVAL_EVERY="${EVAL_EVERY:-}"
_OV_VAL_SAMPLES="${VAL_SAMPLES:-}"
_OV_TRAIN_STEPS="${TRAIN_STEPS:-}"
# Inherit all FSDP launcher defaults (NCCL/ROCm perf env, arg plumbing).
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

# ---- Backend / hardware ------------------------------------------------------
export BACKEND="fsdp"
export NGPU=8
export NNODES=1

# ---- Model (HuggingFace Llama-2-70B, mounted at /model-hf) -------------------
export MODEL="/model-hf"
export TOKENIZER="${SCRIPT_DIR}/tokenizer"

# ---- Data (preprocessed .npy, mounted at /data) -----------------------------
export TRAIN_DATA="/data/train.npy"
export VALID_DATA="/data/validation.npy"
export TRAIN_SAMPLES=10000
export VAL_SAMPLES="${VAL_SAMPLES:-500}"

# ---- Training hyperparameters (MLPerf-aligned, see README) ------------------
export SEQ_LEN=8192
export MBS=1
export GRAD_ACCUM=1          # GBS = MBS(1) x DP(8) x accum(1) = 8  (matches README GBS=8)
export TRAIN_STEPS=1024      # README: cosine decay over full 1024 steps
export LR=4e-4
export MIN_LR=0
export LR_WARMUP_STEPS=0     # README: 0 warmup steps
export WEIGHT_DECAY=1e-4
export MAX_GRAD_NORM=0.3     # README: gradient clip 0.3
export SEED=1234

# ---- LoRA (rank 16, alpha 32, dropout 0.1) ----------------------------------
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.1

# ---- Precision / scaling mode (switchable via MODE) -------------------------
export MODE="${MODE:-bf16}"
export LUMEN_NORM=0          # keep plain HF RMSNorm for stability

# Shared FP8 knobs (consumed only when FP8_TRAINING=1; overridden per-mode).
export FP8_FORMAT="fp8_e4m3"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="max"
export FP8_AMAX_HISTORY=16
export FP8_REDUCE_AMAX=0
export FP8_ACTIVATION=1
export GRAD_QUANT_TYPE=""
export FIRST_LAST_BF16=0

case "${MODE}" in
  bf16)
    export FP8_TRAINING=0
    ;;
  fp8_delayed)
    export FP8_TRAINING=1
    export FP8_SCALING="delayed"
    export FP8_WGRAD=1
    ;;
  fp8_blockwise)
    export FP8_TRAINING=1
    export FP8_SCALING="blockwise"
    export FP8_WGRAD=0          # plain blockwise: BF16 wgrad (scales misalign on w.t())
    ;;
  fp8_blockwise2d)
    export FP8_TRAINING=1
    export FP8_SCALING="blockwise2d"
    export FP8_WGRAD=1          # blockwise2d supports full FP8 DGrad+WGrad (M=8192 %128 ok)
    ;;
  *)
    echo "ERROR: unknown MODE='${MODE}' (use: bf16 | fp8_delayed | fp8_blockwise | fp8_blockwise2d)" >&2
    exit 1
    ;;
esac
echo "[config_MI308X_fsdp_lora_70b] MODE=${MODE} FP8_TRAINING=${FP8_TRAINING} scaling=${FP8_SCALING:-n/a} wgrad=${FP8_WGRAD:-n/a}"

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=0
# MLPerf target: stop as soon as a validation hits val_loss <= this.
export VAL_LOSS_TARGET="${VAL_LOSS_TARGET:-0.925}"
export STEP_TIME_ATOL=0      # no step-time assertion (70B FSDP steps are ~minute-scale)
export TARGET_LOG_PPL=0.0    # unused when VAL_LOSS_TARGET is set

# ---- Logging / checkpoint / eval --------------------------------------------
export LOG_INTERVAL=1
export SAVE_DIR="/results/checkpoints"
export SAVE_INTERVAL="${SAVE_INTERVAL:-256}"     # LoRA adapters are small; set 0 to disable (FSDP save_state_dict collective is broken)
export EVAL_EVERY="${EVAL_EVERY:-384}"   # eval cadence in sequences -> ~48 steps at GBS=8
export EVAL_INTERVAL=0       # let run_finetune.sh derive from EVAL_EVERY
export START_EVAL_AT=0
export NUM_WORKERS=4
export SHARDING="full_shard"

# ---- Misc --------------------------------------------------------------------
export USE_SDMA=0
export USE_CKPT=0
export SAVE_CKPT=0

# Disable online GEMM autotuning: at 70B/8192 the first-step tuning sweep takes
# 20+ min and only buys per-step speed, which is irrelevant for a convergence
# reproduction. Default hipBLASLt heuristics keep step 1 fast.
export PYTORCH_TUNABLEOP_ENABLED=0

export NCCL_DEBUG=WARN

# Re-apply launcher env overrides captured before the base source above, so they
# win over both the base config and this file's defaults.
# Use if/fi (not `test && cmd`) so an empty override does not trip `set -e`.
if [ -n "${_OV_SAVE_INTERVAL}" ]; then export SAVE_INTERVAL="${_OV_SAVE_INTERVAL}"; fi
if [ -n "${_OV_EVAL_EVERY}" ];    then export EVAL_EVERY="${_OV_EVAL_EVERY}"; fi
if [ -n "${_OV_VAL_SAMPLES}" ];   then export VAL_SAMPLES="${_OV_VAL_SAMPLES}"; fi
if [ -n "${_OV_TRAIN_STEPS}" ];   then export TRAIN_STEPS="${_OV_TRAIN_STEPS}"; fi
