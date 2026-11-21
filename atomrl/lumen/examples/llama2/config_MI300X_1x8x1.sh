#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI300X 1-node 8-GPU configuration for LLaMA2-70B LoRA fine-tuning.
#
# Aligned with MI300X_EPYC_9575F_pytorch_llama2_70b MLPerf reference.
#
# Usage:
#   CONFIG=config_MI300X_1x8x1.sh bash run_finetune.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1

# ---- MI300X GPU performance tuning (aligned with MLPerf reference) -----------
export VBOOST_VALUE=1
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export CK_FUSED_ATTN_LOG_CONFIG=0
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

# ---- Training hyperparameters (MLPerf) ---------------------------------------
export BACKEND=megatron
export MODEL_SIZE="llama2-70B"
export TRAIN_STEPS=1024
export MBS=1
export GBS=8
export LR=0.0004
export EVAL_EVERY=384

# ---- Data / tokenizer (override for /data1 mounts) --------------------------
export CKPT_DIR="/data1/lumen/megatron_ckpt"
export TRAIN_DATA="/data1/lumen/data/train.npy"
export VALID_DATA="/data1/lumen/data/validation.npy"
export TOKENIZER="${SCRIPT_DIR}/tokenizer"
export SAVE_DIR="/data1/lumen/results/checkpoints"

# ---- Parallelism (single node, TP=8 for 70B on 8×MI300X) -------------------
export TP=8
export PP=1
export CP=1
export SP=1

# ---- LoRA (all-to-all for multi-GPU efficiency, dim=16 like reference) -------
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_A2A=1

# ---- Activation checkpointing (aligned with MLPerf reference: full/block/21) -
export RECOMPUTE_GRANULARITY="full"
export RECOMPUTE_METHOD="block"
export RECOMPUTE_NUM_LAYERS=21

# ---- FP8 training (aligned exactly with MLPerf MI300X reference) -------------
# Reference: FP8=True, FP8_DPA=0 (no FP8 dot-product attention),
#            FP8_AMAX_ALGO=most_recent, FP8_REDUCE_AMAX=False,
#            FP8_AMAX_HISTORY=4, FP8_ACTIVATION=True
export FP8_TRAINING=1
export FP8_FORMAT="e4m3"
export FP8_SCALING="delayed"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="most_recent"
export FP8_REDUCE_AMAX=0
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=1
export FP8_WGRAD=1

# ---- Lumen attention (MI300X: CK csrc, no FP8 attention = FP8_DPA=0) --------
export LUMEN_ATTN_BACKEND="csrc"
export LUMEN_FP8_QUANT="blockwise"

# ---- Memory logging (for comparison with MLPerf reference) -------------------
export LOG_MEMORY=1

# ---- Warmup ------------------------------------------------------------------
export WARMUP_STEPS=5

# ---- MLPerf submission -------------------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI300X"
