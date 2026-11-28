# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build context: ATOM repository root.
ARG BASE_IMAGE=rocm/atom-dev:vllm-latest
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake libopenmpi-dev libibverbs-dev \
        rdma-core && \
    rm -rf /var/lib/apt/lists/*

ARG MEGATRON_REPOSITORY=https://github.com/ROCm/Megatron-LM.git
ARG MEGATRON_REF=8dd45f5a51378ec1ee7937dee3c20d8626df4763
RUN git clone "${MEGATRON_REPOSITORY}" /workspace/Megatron-LM && \
    git -C /workspace/Megatron-LM checkout --detach "${MEGATRON_REF}" && \
    git -C /workspace/Megatron-LM submodule update --init --recursive && \
    pip install --no-deps -e /workspace/Megatron-LM

ARG MILES_REPOSITORY
ARG MILES_REF
RUN test -n "${MILES_REPOSITORY}" && \
    test -n "${MILES_REF}" && \
    git clone "${MILES_REPOSITORY}" /workspace/miles && \
    git -C /workspace/miles checkout "${MILES_REF}" && \
    pip install --no-deps -e /workspace/miles

COPY . /workspace/ATOM
RUN pip install --no-build-isolation --no-deps -e /workspace/ATOM && \
    LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation -e /workspace/ATOM/atomrl

RUN python3 -c "\
from atomrl.plugin.miles.hooks import custom_megatron_init, custom_update_weight_post_write; \
from atomrl.plugin.miles.lifecycle import setup_atom_manager; \
print('AtomRL miles plugin import smoke: OK')"

ENV DATA_ROOT=/data \
    MILES_ROOT=/workspace/miles
WORKDIR /workspace/ATOM/atomrl
CMD ["bash", "examples/plugin/miles/run.sh"]
