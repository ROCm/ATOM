# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# ═══════════════════════════════════════════════════════════════════════════════
# AtomRL + vLLM + ATOM + AITER + MORI — MI350 (ROCm gfx950)
#
# Self-contained image for Kimi K2.5 Eagle3 SDDD training.
# Dependencies come from the current checkout or pinned official sources:
#   - aiter:  GPU compute kernels (shared by ATOM inference + AtomRL training)
#   - ATOM:   vLLM plugin for MXFP4 online quantization (uses aiter)
#   - Lumen:  Quantized training engine with aiter-patched FSDP2 (uses aiter)
#   - mori:   MoE expert-parallel all-to-all communication
#
# Also includes:
#   - vLLM 0.19.1 + extract_hidden_states patches
#   - Mooncake Transfer Engine (source-built HIP, TCP/RDMA)
#   - AtomRL (training framework, includes MooncakeHiddenStatesConnector)
#
# Build:
#   docker buildx build \
#       -f atomrl/examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/docker/Dockerfile \
#       -t atomrl-vllm-mi350:latest .
#
# Or use the build script:
#   bash examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/docker/build.sh
#
# (primary build context = ATOM repository root)
# ═══════════════════════════════════════════════════════════════════════════════
FROM vllm/vllm-openai-rocm:v0.19.1

WORKDIR /root/

# ---- System dependencies ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git rsync dnsutils libibverbs-dev libnuma-dev \
        cmake build-essential libjsoncpp-dev libgoogle-glog-dev \
        wget golang-go pybind11-dev && \
    rm -rf /var/lib/apt/lists/*

# ---- Mooncake: pip scaffolding first, then source-build HIP backend ----
RUN pip install --no-cache-dir "mooncake-transfer-engine==0.3.10.post1" 2>/dev/null || \
    pip install --no-cache-dir "mooncake-transfer-engine-non-cuda>=0.3.10.post1"

ARG MOONCAKE_COMMIT=f2853a80a494b999d2115449dfd153dd25f5b42d
RUN git clone https://github.com/kvcache-ai/Mooncake.git /tmp/mooncake && \
    cd /tmp/mooncake && \
    git checkout ${MOONCAKE_COMMIT} && \
    git submodule update --init --recursive && \
    bash dependencies.sh -y && \
    pip install pybind11 && \
    export PATH="/usr/local/go/bin:$PATH" && \
    mkdir build && cd build && \
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON && \
    make -j$(nproc) && \
    make install && \
    SITE=$(python3 -c "import mooncake; import os; print(os.path.dirname(mooncake.__file__))") && \
    echo "Replacing pip mooncake .so files in $SITE with source-built versions" && \
    SRC_STORE=$(find /tmp/mooncake/build -name 'store*.so' -path '*/mooncake-integration/*' | head -1) && \
    SRC_ENGINE=$(find /tmp/mooncake/build -name 'engine*.so' -path '*/mooncake-integration/*' | head -1) && \
    if [ -n "$SRC_STORE" ]; then \
        rm -f "$SITE"/store*.so && \
        cp -v "$SRC_STORE" "$SITE/store.so" && \
        echo "Replaced all store .so with source-built HIP version"; \
    fi && \
    if [ -n "$SRC_ENGINE" ]; then \
        rm -f "$SITE"/engine*.so && \
        cp -v "$SRC_ENGINE" "$SITE/engine.so" && \
        echo "Replaced all engine .so with source-built HIP version"; \
    fi && \
    SRC_MASTER=$(find /tmp/mooncake/build -name 'mooncake_master' -type f -executable | head -1) && \
    if [ -n "$SRC_MASTER" ]; then cp -v "$SRC_MASTER" "$SITE/mooncake_master"; fi && \
    rm -rf /tmp/mooncake

# ---- Apply vLLM patches (extract_hidden_states + CacheOnlyAttentionLayer) ----
COPY atomrl/examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/docker/patches/vllm/v0.19.1/*.patch /tmp/patches/
RUN VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
    cd "$VLLM_DIR" && \
    for p in /tmp/patches/*.patch; do patch -p1 < "$p"; done && \
    rm -rf /tmp/patches

# ---- AITER: official ROCm GPU compute kernels ----
# Must be installed FIRST — both ATOM and Lumen depend on it.
ARG AITER_REPOSITORY=https://github.com/ROCm/aiter.git
ARG AITER_SHA=adbb6ac8b84110134e0cf4dc513b250d2cc0a75c
RUN git clone "${AITER_REPOSITORY}" /root/aiter && \
    cd /root/aiter && \
    git checkout --detach "${AITER_SHA}" && \
    git submodule update --init --recursive && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 PYTORCH_ROCM_ARCH=gfx950 pip install -e . && \
    echo "AITER installed: $(python3 -c 'import aiter; print(aiter.__version__)')" || \
    echo "AITER build failed — will install post-build inside container"

# ---- MORI: official pinned MoE communication source ----
ARG MORI_REPOSITORY=https://github.com/ROCm/mori.git
ARG MORI_SHA=9c42e3240b33359331b2cf2141b28886055da583
RUN pip install --no-cache-dir setuptools-scm && \
    git clone "${MORI_REPOSITORY}" /root/mori && \
    cd /root/mori && \
    git fetch origin "${MORI_SHA}" && git checkout --detach FETCH_HEAD && \
    git submodule update --init --recursive && \
    PYTORCH_ROCM_ARCH=gfx950 python setup.py build_ext --inplace

# ---- Unified ATOM/AtomRL checkout (uses aiter) ----
COPY . /root/ATOM

# ---- Case stack + unified ATOM/AtomRL install ----
# Keep the vLLM image's case-owned Transformers resolution, which is within
# ATOM's unified compatibility range; the root install stays dependency-free.
RUN pip install --no-cache-dir omegaconf "ray[default]>=2.9" accelerate wandb \
        pyzmq qwen-vl-utils \
        datasets huggingface_hub safetensors pydantic openai numba && \
    cd /root/ATOM && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ATOM=0.1.0 LUMEN_BUILD_EXTENSIONS=1 \
        pip install --no-build-isolation --no-deps -e . && \
    echo "ATOM/AtomRL installed: $(python3 -c 'import atom, atomrl, ray; print(atom.__version__, atomrl.__version__, ray.__version__)')"

# ---- triton_kernels: routing/swiglu/matmul_ogs for ATOM triton MOE (from ROCm/triton) ----
RUN git clone --depth=1 --branch release/internal/3.5.x \
        https://github.com/ROCm/triton.git /tmp/triton-tk && \
    pip install /tmp/triton-tk/python/triton_kernels && \
    python3 -c "from triton_kernels.routing import routing; print('triton_kernels OK')" && \
    rm -rf /tmp/triton-tk


# ---- Environment ----
ENV PYTORCH_ROCM_ARCH=gfx950
ENV HIP_FORCE_DEV_KERNARG=1
ENV PYTHONPATH="/root/mori/python:${PYTHONPATH:-}"
RUN python3 -c "import mori; print('MORI installed:', mori.__file__)"

ENTRYPOINT []
WORKDIR /root/ATOM/atomrl
