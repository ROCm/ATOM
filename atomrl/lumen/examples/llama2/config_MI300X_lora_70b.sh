#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# TP=1 DP=8 config (v47) — LLaMA2-70B LoRA MLPerf (config_MI300X_lora_70b.sh).
# Used by run_lora_finetune_llama2_70b.sh.
# NousResearch/Llama-2-70b-hf, FP8 HYBRID + LoRA (attention-only), seed=1234.
# Self-contained (does NOT source base config).
#
# v47 results: 4,730 ms/step pre-eval, val_loss 0.9223, 1.19x vs local MLPerf ref (3,967 ms).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Backend -----------------------------------------------------------------
export BACKEND=megatron
export MODEL_SIZE="llama2-70B"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1

# ---- Parallelism: TP=1 DP=8 (matching MLPerf) -------------------------------
export TP=1
export PP=1
export VP=0
export CP=1
export SP=0

# ---- Training hyperparameters (aligned with MLPerf MI300X actual submission) -
export MBS=1
export GBS=8
export SEQ_LEN=8192
export LR=4e-4                  # MLPerf: LR=0.0004
export MIN_LR=0                 # MLPerf: min_lr=0.0
export TRAIN_STEPS=${TRAIN_STEPS:-1024}  # MLPerf: MAX_STEPS=1024
export WARMUP_STEPS=5           # Synthetic warmup with zero loss_mask
export LR_WARMUP_STEPS=0        # MLPerf: warmup_ratio=0.0 → 0 warmup iters
export LOG_INTERVAL=1
export SAVE_INTERVAL=999999
export SEED=1234                # User requested; MLPerf default=1
export WEIGHT_DECAY=1e-4        # MLPerf: weight_decay=0.0001
export GRADIENT_CLIP=0.3        # MLPerf: gradient_clip_val=0.3
export ADAM_BETA1=0.9           # MLPerf: betas=[0.9, 0.999]
export ADAM_BETA2=0.999
export ADAM_EPS=1e-8
export PRECISION="bf16"

# ---- LoRA (matching MLPerf: attention-only, rank=16, alpha=32, dropout=0.1) --
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.1
export LORA_A2A=1
export LORA_TARGET_MODULES="attention"  # MLPerf: target_modules=['attention']

# ---- Activation checkpointing -----------------------------------------------
# MLPerf uses ACL=21 (full/block recompute).
# Key fixes enabling ACL=21 in Lumen:
#   - deterministic=False → CK v3 tiled attention bwd (no 16GB alloc)
#   - FP8 SwiGLU input store (448 MB vs 896 MB per layer)
#   - FP8 GEMM for mixed-dtype backward (no BF16 weight dequant copies)
export RECOMPUTE_GRANULARITY="full"
export RECOMPUTE_METHOD="block"
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-21}

# ---- FP8 training (matching MLPerf) -----------------------------------------
export FP8_TRAINING=1
export FP8_FORMAT="hybrid"      # MLPerf: fp8_hybrid=True → E4M3 fwd, E5M2 bwd
export FP8_SCALING="delayed"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="most_recent"
export FP8_REDUCE_AMAX=0        # MLPerf: reduce_amax=False
export FP8_AMAX_HISTORY=4       # MLPerf: FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=1         # MLPerf: FP8_ACTIVATION=True
export FP8_WGRAD=1
export FIRST_LAST_BF16=0
export GRAD_QUANT_TYPE=""
export FP8_CHECKPOINT=1         # FP8-aware activation checkpointing
export FP8_ACT_STORE=1

# ---- Lumen attention ---------------------------------------------------------
export LUMEN_ATTN_BACKEND="csrc"
export LUMEN_ATTN_KERNEL_BACKEND="triton"
export LUMEN_FP8_QUANT="blockwise"
export LUMEN_RMSNORM=0
export LUMEN_NORM=0
export LUMEN_LINEAR=1

# ---- Evaluation --------------------------------------------------------------
# LUMEN_EVAL_ALIGNED=1: eval every 192 steps (1536 samples), matching MLPerf
#   wall-clock budget (5 evals instead of 21, saves ~15 min).
# Default (0): eval every 48 steps (384 samples) for detailed convergence tracking.
export EVAL_ITERS=22            # Full validation set: 173 samples / GBS=8 ~ 22
export START_EVAL_AT=0
if [ "${LUMEN_EVAL_ALIGNED:-0}" = "1" ]; then
    export EVAL_EVERY=1536      # 1536/GBS=8 = every 192 steps
    export EVAL_INTERVAL=0
else
    export EVAL_EVERY=384       # 384/GBS=8 = every 48 steps
    export EVAL_INTERVAL=0
fi

# ---- Warmup / early stopping ------------------------------------------------
export VAL_LOSS_TARGET=""

# ---- Checkpoint management ---------------------------------------------------
export USE_CKPT=0
export FROM_HF=1
export SAVE_CKPT=0
export CONTINUAL_CKPT="/data1/lumen/results/tp1_fp8/saved_ckpts"
export CKPT_START_STEP=0
export FP8_PARAMS=0
export FP8_PARAM_STORAGE=1      # Store frozen weights in FP8 (Lumen optimization)

# ---- Distributed timeout (kernel tuning can take >10min on first step) ------
export DIST_TIMEOUT_MINUTES=120

# ---- Optimizer ---------------------------------------------------------------
# MLPerf uses fused_adam for single-node (not distributed optimizer).
# With attention-only LoRA, optimizer states are small (~45M params * 12 bytes
# = 540 MB per GPU). Distributed optimizer is not needed.
export USE_DIST_OPTIMIZER=0
export OVERLAP_GRAD_REDUCE=1

# ---- Experiment management ---------------------------------------------------
export TAG=""
export TARGET_LOG_PPL="3.3"
export STEP_TIME_ATOL=18000

# ---- Launcher compatibility --------------------------------------------------
export USE_SDMA=0
export LOG_MEMORY=1

# ---- Paths (TP=1 checkpoint) ------------------------------------------------
export CKPT_DIR="/data1/lumen/megatron_ckpt_nous_tp1"
export TRAIN_DATA="/data1/lumen/data/train.npy"
export VALID_DATA="/data1/lumen/data/validation.npy"
export TOKENIZER="${SCRIPT_DIR}/tokenizer"
export SAVE_DIR="/data1/lumen/results/tp1_fp8"

# ---- MLPerf-aligned flags ----------------------------------------------------
export MAKE_VOCAB_DIVISIBLE_BY=128  # MLPerf: make_vocab_size_divisible_by=128
export DISABLE_RESET_FLAGS=1

# ---- MI300X tuning -----------------------------------------------------------
export VBOOST_VALUE=1
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export CK_FUSED_ATTN_LOG_CONFIG=0
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

# ---- hipBLASLt ---------------------------------------------------------------
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# ---- Misc ROCm perf ---------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="tunableop_results.csv"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TORCHDYNAMO_DISABLE=1

# ---- NCCL / ROCm -----------------------------------------------------------
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export HSA_ENABLE_SDMA=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# ---- MLPerf submission -------------------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI300X"

# ---- MXFP8 attention block sizes (unused but expected by scripts) ------------
export MXFP8_BLOCK_M_FWD=128
export MXFP8_BLOCK_N_FWD=128
export MXFP8_BLOCK_M_DQ_BWD=128
export MXFP8_BLOCK_N_DQ_BWD=128
export MXFP8_BLOCK_M_DKV_BWD=128
export MXFP8_BLOCK_N_DKV_BWD=128
export MXFP8_QUANT_BLOCK_SIZE=128
