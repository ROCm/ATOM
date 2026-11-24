# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build a vime + AtomRL plugin image for Qwen3-8B Megatron training.
#
# Usage:
#   docker build \
#     -f atomrl/examples/plugin/vime/Dockerfile \
#     --build-arg VIME_REPO=https://github.com/your-org/vime.git \
#     -t atomrl-vime-plugin .
#
#   docker run --gpus all -v /data:/data \
#     -e DATA_ROOT=/data -e VIME_ROOT=/workspace/vime \
#     atomrl-vime-plugin
#
# Build context must be the ATOM repository root.

ARG BASE_IMAGE=rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

# ─── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake \
        libopenmpi-dev libibverbs-dev rdma-core && \
    rm -rf /var/lib/apt/lists/*

# ─── Megatron-LM (ROCm fork) ────────────────────────────────────────────────
ARG MEGATRON_REPO=https://github.com/ROCm/Megatron-LM.git
ARG MEGATRON_BRANCH=rocm_dev
RUN git clone --depth 1 -b ${MEGATRON_BRANCH} ${MEGATRON_REPO} /workspace/Megatron-LM && \
    cd /workspace/Megatron-LM && pip install -e .

# ─── vime ────────────────────────────────────────────────────────────────────
ARG VIME_REPO=https://github.com/your-org/vime.git
ARG VIME_BRANCH=main
RUN git clone --depth 1 -b ${VIME_BRANCH} ${VIME_REPO} /workspace/vime && \
    cd /workspace/vime && pip install -e .

# ─── Unified ATOM/AtomRL distribution and vime hook dependencies ───────────
COPY . /workspace/ATOM
RUN pip install --no-cache-dir "ray[default]>=2.9" "omegaconf>=2.3" \
        datasets "accelerate>=0.28" safetensors peft \
        "megatron-core>=0.9" "megatron-bridge>=0.1" && \
    LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation --no-deps -e /workspace/ATOM

# ─── AITER (official ROCm source) ───────────────────────────────────────────
ARG AITER_REPOSITORY=https://github.com/ROCm/aiter.git
ARG AITER_SHA=ff745fb09d669f40cf81d15d3e7598b61fe2a61c
RUN git clone "${AITER_REPOSITORY}" /workspace/aiter && \
    cd /workspace/aiter && \
    git checkout --detach "${AITER_SHA}" && \
    git submodule update --init --recursive && \
    PREBUILD_KERNELS=1 pip install -e .

# ─── Verify hooks ───────────────────────────────────────────────────────────
RUN python3 -c "\
import accelerate, datasets, omegaconf, ray, safetensors; \
from atomrl.plugin.vime.hooks import custom_megatron_init, custom_model_provider; \
print(f'vime dependencies: ray={ray.__version__}, accelerate={accelerate.__version__}'); \
print('vime hooks verification: OK')"

ENV DATA_ROOT=/data
ENV VIME_ROOT=/workspace/vime
WORKDIR /workspace/ATOM/atomrl

CMD ["bash", "examples/plugin/vime/run.sh"]
