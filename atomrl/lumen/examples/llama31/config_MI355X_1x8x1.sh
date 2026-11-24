#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

#
# MI355X 1-node 8-GPU configuration for LLaMA 3.1 pretraining.
#
# Usage:
#   bash run_pretrain.sh                                       # uses this config
#   CONFIG=config_MI300X_1x8x1.sh bash run_pretrain.sh         # uses MI300X config

# ---- Backend selection -------------------------------------------------------
export BACKEND="megatron"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1
export GPUS_PER_NODE=${NGPU}

# ---- Model -------------------------------------------------------------------
export SIZE="8b"

# ---- Training hyperparameters ------------------------------------------------
export MBS=2
export GBS=32
export SEQ_LEN=8192
export MAX_LR="8e-4"
export MIN_LR="8e-5"
export TRAIN_STEPS=1200000
export LR_WARMUP_STEPS=128
export LOG_INTERVAL=1
export SAVE_INTERVAL=0
export EVAL_INTERVAL=0
export SAVE_DIR="/results/checkpoints"
export SEED=1234

# ---- Data / tokenizer --------------------------------------------------------
export TRAIN_DATA="/data/train.npy"
export VALID_DATA="/data/validation.npy"
export TOKENIZER="/model"

# ---- LoRA --------------------------------------------------------------------
export LORA_RANK=0
export LORA_ALPHA=32
export LORA_DROPOUT=0.1

# ---- FP8 training (MLPerf defaults) -----------------------------------------
export FP8_TRAINING=1
export FP8_FORMAT="fp8_e4m3"
export FP8_SCALING="delayed"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="most_recent"
export FP8_REDUCE_AMAX=0
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=1
export FP8_WGRAD=1
export GRAD_QUANT_TYPE=""
export FIRST_LAST_BF16=0
export BF16_LAYERS_START=1
export BF16_LAYERS_END=1

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=0
export VAL_LOSS_TARGET=""

# ---- MXFP8 attention block sizes --------------------------------------------
export MXFP8_BLOCK_M_FWD=128
export MXFP8_BLOCK_N_FWD=128
export MXFP8_BLOCK_M_DQ_BWD=128
export MXFP8_BLOCK_N_DQ_BWD=128
export MXFP8_BLOCK_M_DKV_BWD=128
export MXFP8_BLOCK_N_DKV_BWD=128
export MXFP8_QUANT_BLOCK_SIZE=128

# ---- Checkpoint management (Docker run_and_time.sh compat) -------------------
export USE_CKPT=0
export FROM_HF=1
export SAVE_CKPT=0
export CONTINUAL_CKPT="/results/saved_ckpts"
export CKPT_START_STEP=0
export FP8_PARAMS=1

# ---- MLPerf / experiment management -----------------------------------------
export TAG=""
export TARGET_LOG_PPL="3.3"
export STEP_TIME_ATOL=18000

# ---- Evaluation (in training sequences, Docker convention) -------------------
export EVAL_EVERY=12288
export START_EVAL_AT=0

# ---- Launcher compatibility --------------------------------------------------
export USE_SDMA=0

# ---- Lumen -------------------------------------------------------
# Attention backend: "aiter_csrc", "aiter_triton", "aiter_triton_fp8", "aiter_csrc_fp8", "aiter_asm_fp8"
#   aiter_asm_fp8 uses ASM kernels with fallback: asm -> csrc -> triton
export LUMEN_ATTN_BACKEND="aiter_asm_fp8"
# FP8 attention quantization type (used when LUMEN_ATTN_BACKEND is *_fp8):
#   "blockwise"  — per-block FP8 scaling
#   "dynamic"    — per-tensor dynamic FP8 scaling
#   "delayed"    — delayed FP8 scaling (uses amax history)
#   "per_token"  — per-token FP8 quantization
#   "none"       — no FP8 quantization (fall back to bf16)
#   "mxfp8"      — microscaling FP8 (legacy)
export LUMEN_FP8_QUANT="blockwise"
# Use Lumen Triton-accelerated RMSNorm (0 = native Megatron norm)
export LUMEN_RMSNORM=0
export LUMEN_NORM=0
# Use Lumen parallel linear modules (0 = Megatron native, 1 = Lumen spec provider)
export LUMEN_LINEAR=0
# Use Lumen Triton parallel cross-entropy (0 = Megatron native)
export LUMEN_CROSS_ENTROPY=0

# ---- Megatron backend --------------------------------------------------------
export TP=1
export PP=1
export CP=1
export VP=0
export SP=0
export WEIGHT_DECAY=0.1
export GRADIENT_CLIP=1.0
export EVAL_ITERS=10
export PRECISION="bf16"
export CKPT_DIR="/model"
export LORA_A2A=0

# ---- FSDP backend ------------------------------------------------------------
export MODEL="/model"
export GRAD_ACCUM=8
export MAX_GRAD_NORM=1.0
export NUM_WORKERS=4
export TRAIN_SAMPLES=""
export VAL_SAMPLES=""
export SHARDING="full_shard"

# ---- MI355X platform ---------------------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI355X"

# ---- NCCL / ROCm performance (MI355X tuned) ----------------------------------
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Disable SDMA engines — known to cause RCCL P2P hangs on MI-series GPUs.
export HSA_ENABLE_SDMA=0
# Disable InfiniBand probing — avoids "No device found" errors.
export NCCL_IB_DISABLE=1
# Force RCCL bootstrap sockets onto loopback for single-node runs.
# For multi-node runs override with the inter-node interface: NCCL_SOCKET_IFNAME=ens3
export NCCL_SOCKET_IFNAME=lo
# Set to INFO to debug RCCL errors; WARN for normal runs.
export NCCL_DEBUG=WARN

# ---- hipBLASLt ---------------------------------------------------------------
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# ---- Misc ROCm perf ----------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="tunableop_results.csv"
export OMP_NUM_THREADS=1

# ---- Torch Dynamo / Inductor -------------------------------------------------
# Megatron's _warmup_jit_function calls torch.compile on bias_swiglu, which
# triggers the Inductor backend.  Inductor imports triton_key from Triton at
# graph-cache time; if the installed Triton build doesn't export that symbol
# the warmup crashes — and even when it succeeds, async JIT compilation can
# stall one rank while others proceed to a collective, causing the RCCL
# watchdog to fire ("WorkNCCL ... ran for 600000 ms before timing out").
# Disabling Dynamo skips the torch.compile path entirely.
export TORCHDYNAMO_DISABLE=1
