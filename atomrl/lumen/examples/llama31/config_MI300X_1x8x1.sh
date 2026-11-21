#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MI300X 1-node 8-GPU configuration for LLaMA 3.1 pretraining.
#
# Usage:
#   CONFIG=config_MI300X_1x8x1.sh bash run_pretrain.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

# ---- Hardware ----------------------------------------------------------------
export NGPU=8
export NNODES=1
export GPUS_PER_NODE=${NGPU}

# ---- MI300X GPU performance tuning -------------------------------------------
export VBOOST_VALUE=1
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export CK_FUSED_ATTN_LOG_CONFIG=0
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

# ---- Lumen attention ---------------------------------------------------------
# MI300X examples prefer the CK/csrc attention path instead of ASM FP8 kernels.
export LUMEN_ATTN_BACKEND="aiter_csrc"
export LUMEN_FP8_QUANT="blockwise"

# ---- MLPerf submission -------------------------------------------------------
export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI300X"
