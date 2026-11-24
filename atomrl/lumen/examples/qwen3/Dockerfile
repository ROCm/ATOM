# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Qwen3 MXFP4 training image: Lumen base + baked source + Qwen3 deps.
#
# Build:
#   bash examples/qwen3/build_mxfp4_lumen_image.sh
#
# Requires:
#   lumen/tests:latest   (bash build.sh)

ARG LUMEN_IMAGE=lumen/tests:latest
FROM ${LUMEN_IMAGE}

WORKDIR /workspace/ATOM/atomrl/lumen

# Re-copy Lumen onto this commit; external dependencies stay under /opt.
COPY . /workspace/ATOM
# The tokenizer is not in the repository; TOKENIZER_PATH below points here, and
# the image runs with the Hub offline.
RUN bash -c 'source examples/scripts/ensure_tokenizer.sh && example_tokenizer qwen3 examples/qwen3/tokenizer'

# Refresh AITER from the verified official commit.
ARG AITER_REPOSITORY=https://github.com/ROCm/aiter.git
ARG AITER_SHA=ff745fb09d669f40cf81d15d3e7598b61fe2a61c
RUN rm -rf /opt/aiter && \
    git clone "${AITER_REPOSITORY}" /opt/aiter && \
    cd /opt/aiter && \
    git checkout --detach "${AITER_SHA}" && \
    git submodule update --init --recursive && \
    PREBUILD_KERNELS=1 pip install -e .

# Tokenizer and dataset deps; the base only installs the top-level requirements.
RUN pip install -r /workspace/ATOM/atomrl/lumen/examples/qwen3/requirements.txt -q

# Re-resolve in case the copied tree moved a pin.
RUN cd /workspace/ATOM && LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation --no-deps -e .

# MXFP4 defaults. Only knobs that hold for every launcher and every parallel
# layout are baked; see the block below for the ones that must not be.
ENV LUMEN_FAST_QUANT_DISPATCH=1
# Off: swiglu_fwd/swiglu_bwd are Lumen's own Triton kernels and the pinned
# official AITER does not carry them; with 1 the probe would fall back anyway.
ENV LUMEN_FUSED_SWIGLU=0
ENV LUMEN_FUSED_RESIDUAL_NORM=1
ENV LUMEN_FUSED_RES_BWD=1
ENV LUMEN_SKIP_BACKEND_SYNC=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HIP_FORCE_DEV_KERNARG=1
ENV GPU_MAX_HW_QUEUES=8
ENV TORCHDYNAMO_DISABLE=1
ENV USE_HIPBLASLT=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=1
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ENV OMP_NUM_THREADS=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV LUMEN_ROOT=/workspace/ATOM/atomrl/lumen
ENV TOKENIZER_PATH=/workspace/ATOM/atomrl/lumen/examples/qwen3/tokenizer

# Deliberately not baked:
#
#   CUDA_DEVICE_MAX_CONNECTIONS — the tuned value is 8, which Megatron asserts
#     is exactly 1 once tensor or context parallelism is on.
#   NCCL_IB_DISABLE, NCCL_SOCKET_IFNAME — single-node values.

WORKDIR /workspace/ATOM/atomrl/lumen/examples/qwen3

CMD ["bash", "run_pretrain_qwen3_8b_mxfp4.sh"]
