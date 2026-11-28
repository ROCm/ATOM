#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Source-build flash-attention's ROCm (CK) backend against the image's torch.
# Runs inside the image build; no GPU required, only the target arch.
#
# The vLLM base carries no flash_attn. Without it Megatron's attention_backend
# "flash" falls back to unfused attention, which routing replay (R3) refuses, and
# the packed log-prob path loses its fused cross-entropy kernel.
set -eux

ARCH=${PYTORCH_ROCM_ARCH:-gfx950}
FLASH_ATTN_REF=${FLASH_ATTN_REF:-v2.8.3}
SRC=${SRC:-/tmp/flash_attn_src}

git clone --branch "$FLASH_ATTN_REF" https://github.com/Dao-AILab/flash-attention.git "$SRC"
git -C "$SRC" submodule update --init csrc/composable_kernel
cd "$SRC"
GPU_ARCHS="$ARCH" PYTORCH_ROCM_ARCH="$ARCH" BUILD_TARGET=rocm FLASH_ATTENTION_FORCE_BUILD=TRUE \
  MAX_JOBS=${MAX_JOBS:-64} python3 -m pip install -v . --no-build-isolation --no-deps
python3 -c "import flash_attn, flash_attn_2_cuda; print('flash_attn', flash_attn.__version__)"
rm -rf "$SRC"
