# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build a verl + AtomRL plugin image for Qwen3-8B GRPO (FSDP2 + ATOM).
#
# Usage:
#   docker build -f atomrl/examples/plugin/verl/Dockerfile -t atomrl-verl-plugin .
#   docker run --gpus all -v /data:/data -e DATA_ROOT=/data atomrl-verl-plugin
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

# ─── verl ────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir verl

# ─── ATOM + AtomRL (ATOM owns the verl.plugins entry point) ─────────────────
COPY . /workspace/ATOM
RUN pip install --no-cache-dir "ray[default]>=2.9" "omegaconf>=2.3" \
        datasets accelerate safetensors peft && \
    LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation --no-deps -e /workspace/ATOM

# ─── AITER (official ROCm source) ───────────────────────────────────────────
ARG AITER_REPOSITORY=https://github.com/ROCm/aiter.git
ARG AITER_SHA=ff745fb09d669f40cf81d15d3e7598b61fe2a61c
RUN git clone "${AITER_REPOSITORY}" /workspace/aiter && \
    cd /workspace/aiter && \
    git checkout --detach "${AITER_SHA}" && \
    git submodule update --init --recursive && \
    PREBUILD_KERNELS=1 pip install -e .

# ─── Verify plugin registration ─────────────────────────────────────────────
RUN python3 -c "\
import verl; \
from verl.workers.rollout.base import _ROLLOUT_REGISTRY; \
assert ('atom', 'async') in _ROLLOUT_REGISTRY, 'ATOM rollout not registered'; \
print('Plugin verification: OK')"

ENV DATA_ROOT=/data
WORKDIR /workspace/ATOM/atomrl

CMD ["bash", "examples/plugin/verl/run.sh"]
