# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# DSV4 training image: Lumen base + bootstrap artifacts (TileKernels + pip deps).
#
# Build:
#   bash examples/dsv4/build_dsv4_lumen_image.sh
#
# Requires:
#   lumen/tests:latest
#   bootstrap dir from examples/dsv4/prepare_bootstrap.sh

ARG LUMEN_IMAGE=lumen/tests:latest
FROM ${LUMEN_IMAGE}

WORKDIR /workspace/ATOM/atomrl/lumen

# Bootstrap artifacts (ROCm Megatron rocm_dev, TileKernels, mbridge via pip).
# Replace bootstrap entirely (COPY merges; stale miles files remain from old base layers).
RUN rm -rf /opt/dsv4-bootstrap
COPY examples/dsv4/.bootstrap-build/ /opt/dsv4-bootstrap/
RUN test -f /opt/dsv4-bootstrap/.ready

# mooncake + native libs; Megatron fused CE needs libglog on lumen/tests:latest.
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git libgoogle-glog0v5 rsync \
    && rm -rf /var/lib/apt/lists/*

# Pre-stage writable runtime tree (no TE — ABI mismatch with Lumen PyTorch 2.8).
RUN bash -c 'set -euo pipefail; \
    mkdir -p /opt/dsv4-runtime/site-packages; \
    rsync -a \
        --exclude=transformer_engine \
        --exclude=transformer_engine*.dist-info \
        --exclude=transformer_engine*.egg-info \
        /opt/dsv4-bootstrap/site-packages/. /opt/dsv4-runtime/site-packages/; \
    printf "import megatron.core.jit as _jit\n_jit.disable_jit_fuser()\n" \
        > /opt/dsv4-runtime/site-packages/lumen_dsv4_bootstrap.pth'

# Lumen source and pinned official dependencies.
COPY . /workspace/ATOM

ARG AITER_REPOSITORY=https://github.com/ROCm/aiter.git
ARG AITER_SHA=ff745fb09d669f40cf81d15d3e7598b61fe2a61c
RUN rm -rf /opt/aiter && git clone "${AITER_REPOSITORY}" /opt/aiter && \
    cd /opt/aiter && git checkout --detach "${AITER_SHA}" && \
    git submodule update --init --recursive && PREBUILD_KERNELS=1 pip install -e . -q
ARG MORI_REPOSITORY=https://github.com/ROCm/mori.git
ARG MORI_SHA=9c42e3240b33359331b2cf2141b28886055da583
RUN rm -rf /opt/mori && git clone "${MORI_REPOSITORY}" /opt/mori && \
    cd /opt/mori && git fetch origin "${MORI_SHA}" && git checkout --detach FETCH_HEAD && \
    git submodule update --init --recursive && \
    pip install setuptools-scm -q && \
    pip install -e . -q
RUN mkdir -p /opt/dsv4-runtime/mori-libs && \
    cp /opt/mori/python/mori/*.so /opt/dsv4-runtime/mori-libs/
RUN cd /workspace/ATOM && LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation --no-deps -e .
RUN pip install pyarrow -q \
    && (pip install fast-hadamard-transform -q \
        || echo "[dsv4-lumen] skip fast-hadamard-transform (optional on ROCm)")

ENV V4_INDEXER_IMPL=aiter
ENV V4_SPARSE_MLA_BACKEND=triton
ENV BOOTSTRAP_DIR=/opt/dsv4-bootstrap
ENV WRITABLE_ROOT=/opt/dsv4-runtime
ENV LD_LIBRARY_PATH="/opt/dsv4-bootstrap/native-libs:${LD_LIBRARY_PATH:-}"
ENV TORCHDYNAMO_DISABLE=1
ENV PYTHONPATH="/workspace/ATOM/atomrl:/opt/aiter:/opt/mori/python:${PYTHONPATH:-}"
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WORKDIR /workspace/ATOM/atomrl/lumen
