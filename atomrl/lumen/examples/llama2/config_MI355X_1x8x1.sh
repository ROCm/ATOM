#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

#
# MI355X 1-node 8-GPU configuration for LLaMA2 SFT fine-tuning.
#
# Usage:
#   bash run_finetune.sh                                      # uses this config
#   CONFIG=config_MI300X_1x8x1.sh bash run_finetune.sh        # uses MI300X config
#   TRAIN_STEPS=500 bash run_finetune.sh                      # override by editing this file

# ---- Backend selection -------------------------------------------------------
export BACKEND="fsdp"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1

# ---- Training hyperparameters ------------------------------------------------
export MBS=1
export SEQ_LEN=8192
export LR=4e-4
export MIN_LR=0
export TRAIN_STEPS=800
export LR_WARMUP_STEPS=0
export LOG_INTERVAL=1
export SAVE_INTERVAL=200
export SAVE_DIR="/results/checkpoints"
export SEED=1234

# ---- Data / tokenizer --------------------------------------------------------
export TRAIN_DATA="/data/train.npy"
export VALID_DATA="/data/validation.npy"
_LUMEN_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TOKENIZER="${_LUMEN_CONFIG_DIR}/tokenizer"

# ---- LoRA --------------------------------------------------------------------
export LORA_RANK=0
export LORA_ALPHA=32
export LORA_DROPOUT=0.1

# ---- FP8 training ------------------------------------------------------------
export FP8_TRAINING=1
export FP8_FORMAT="fp8_e4m3"
export FP8_SCALING="delayed"
export FP8_BLOCK_SIZE=128
export FP8_AMAX_ALGO="max"
export FP8_REDUCE_AMAX=0
export FP8_AMAX_HISTORY=16
export FP8_ACTIVATION=1
export FP8_WGRAD=1
export GRAD_QUANT_TYPE=""
export FIRST_LAST_BF16=0
export BF16_LAYERS_START=1
export BF16_LAYERS_END=1

# ---- Warmup / early stopping -------------------------------------------------
export WARMUP_STEPS=0
export VAL_LOSS_TARGET=""

# ---- Checkpoint management ---------------------------------------------------
export USE_CKPT=0
export FROM_HF=1
export SAVE_CKPT=0
export CONTINUAL_CKPT="/results/saved_ckpts"
export CKPT_START_STEP=0
export FP8_PARAMS=0

# ---- Experiment management ---------------------------------------------------
export TAG=""
export TARGET_LOG_PPL="3.3"
export STEP_TIME_ATOL=18000

# ---- Evaluation (in training sequences, shared launcher contract) ------------
export EVAL_EVERY=400
export EVAL_INTERVAL=0
export START_EVAL_AT=0

# ---- Launcher compatibility --------------------------------------------------
export USE_SDMA=0

# ---- Megatron backend --------------------------------------------------------
export MODEL_SIZE="llama2-70B"
export TP=8
export PP=1
export CP=1
export VP=0
export SP=1
export GBS=8
export WEIGHT_DECAY=1e-4
export GRADIENT_CLIP=0.3
export EVAL_ITERS=10
export PRECISION="bf16"
export CKPT_DIR="/model"
export LORA_A2A=0

# Lumen
# LUMEN_ATTN_BACKEND options:
#   aiter_csrc          – CK/C++ flash-attention (fastest on MI300X)
#   aiter_triton        – Triton flash-attention (no attention quantization)
#   aiter_triton_fp8    – Triton FP8-quantized attention
#   aiter_csrc_fp8      – CK/C++ FP8 attention (forward-only, inference)
#   aiter_asm_fp8       – ASM FP8 attention with fallback: asm -> csrc -> triton
#
# LUMEN_FP8_QUANT options (used when backend is *_fp8):
#   blockwise   – per-block FP8 scaling (works on all MI-series)
#   dynamic     – per-tensor dynamic FP8 scaling
#   delayed     – delayed FP8 scaling (uses amax history)
#   per_token   – per-token FP8 quantization
#   none        – no FP8 quantization (fall back to bf16 attention)
#   mxfp8       – microscaling FP8 (legacy, gfx950 / MI355X only)
export LUMEN_ATTN_BACKEND="aiter_asm_fp8"
# LUMEN_ATTN_KERNEL_BACKEND selects the implementation family used by the
# attention wrapper. Keep it aligned with LUMEN_ATTN_BACKEND for this example.
export LUMEN_ATTN_KERNEL_BACKEND="triton"
export LUMEN_FP8_QUANT="blockwise"
export LUMEN_RMSNORM=0
export LUMEN_NORM=0

# MXFP8 attention block sizes
export MXFP8_BLOCK_M_FWD=128
export MXFP8_BLOCK_N_FWD=128
export MXFP8_BLOCK_M_DQ_BWD=128
export MXFP8_BLOCK_N_DQ_BWD=128
export MXFP8_BLOCK_M_DKV_BWD=128
export MXFP8_BLOCK_N_DKV_BWD=128
export MXFP8_QUANT_BLOCK_SIZE=128

# ---- FSDP backend ------------------------------------------------------------
export MODEL="/model-hf"
export GRAD_ACCUM=8
export MAX_GRAD_NORM=1.0
export NUM_WORKERS=4
export TRAIN_SAMPLES=10000
export VAL_SAMPLES=500
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
# Without this, RCCL scans all interfaces and may select a physical NIC
# (eno1, eno0, …) that is unreachable inside Kubernetes / container
# environments, causing "socketPollConnect: No route to host" at init time.
# Actual GPU data transfers use PCIe/xGMI and are unaffected by this setting.
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
# Reduce fragmentation for large-model FSDP — allows PyTorch to expand
# existing memory segments instead of allocating many fixed-size blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Torch Dynamo / Inductor ------------------------------------------------
# Megatron's _warmup_jit_function calls torch.compile on bias_swiglu, which
# triggers the Inductor backend.  Inductor imports triton_key from Triton at
# graph-cache time; if the installed Triton build doesn't export that symbol
# the warmup crashes.  Disabling Dynamo skips the torch.compile path entirely
# — Megatron's own fused kernels are used instead, with no perf impact.
export TORCHDYNAMO_DISABLE=1
