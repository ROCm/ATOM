# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
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

ARG BASE_IMAGE=rocm/atom-dev:vllm-latest
FROM ${BASE_IMAGE}

WORKDIR /workspace

# ─── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake \
        libopenmpi-dev libibverbs-dev rdma-core && \
    rm -rf /var/lib/apt/lists/*

# ─── Megatron-LM (ROCm fork) ────────────────────────────────────────────────
ARG MEGATRON_REPO=https://github.com/ROCm/Megatron-LM.git
ARG MEGATRON_REF=8dd45f5a51378ec1ee7937dee3c20d8626df4763
RUN git clone ${MEGATRON_REPO} /workspace/Megatron-LM && \
    git -C /workspace/Megatron-LM checkout --detach ${MEGATRON_REF} && \
    pip install --no-deps -e /workspace/Megatron-LM

# ─── vime ────────────────────────────────────────────────────────────────────
ARG VIME_REPO
ARG VIME_REF
RUN test -n "${VIME_REPO}" && \
    test -n "${VIME_REF}" && \
    git clone ${VIME_REPO} /workspace/vime && \
    git -C /workspace/vime checkout ${VIME_REF} && \
    pip install --no-deps -e /workspace/vime

# ─── Unified ATOM/AtomRL distribution and vime hook dependencies ───────────
COPY . /workspace/ATOM
RUN pip install --no-cache-dir "ray[default]>=2.9" "omegaconf>=2.3" \
        datasets "accelerate>=0.28" safetensors peft \
        "megatron-core>=0.9" "megatron-bridge>=0.1" && \
    pip install --no-build-isolation --no-deps -e /workspace/ATOM && \
    LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation -e /workspace/ATOM/atomrl

# ─── Verify hooks ───────────────────────────────────────────────────────────
RUN python3 -c "\
import accelerate, datasets, omegaconf, ray, safetensors; \
from atomrl.plugin.vime.hooks import custom_megatron_init, custom_model_provider, custom_update_weight_post_write; \
from atomrl.plugin.vime.lifecycle import setup_atom_manager; \
print(f'vime dependencies: ray={ray.__version__}, accelerate={accelerate.__version__}'); \
print('vime hooks verification: OK')"

ENV DATA_ROOT=/data
ENV VIME_ROOT=/workspace/vime
WORKDIR /workspace/ATOM/atomrl

CMD ["bash", "examples/plugin/vime/run.sh"]
