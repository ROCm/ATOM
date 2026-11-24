# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

#
# Build context must be the ATOM repository root:
#
#   docker build \
#     -f atomrl/examples/GRPO/docker/Dockerfile \
#     --target trainer \
#     -t qwen3-30b-a3b:trainer .
#
#   docker build \
#     -f atomrl/examples/GRPO/docker/Dockerfile \
#     --target rollout \
#     -t qwen3-30b-a3b:rollout .
#
# The two targets intentionally keep different base-image NumPy/Triton stacks.
# Do not collapse them into one image: the rollout Triton build is tied to vLLM.

ARG TRAINER_BASE_IMAGE=rocm/atom-dev:latest
ARG ROLLOUT_BASE_IMAGE=rocm/atom-dev:vllm-latest
ARG DSV4_BASE_IMAGE=vllm/vllm-openai-rocm:v0.25.1
ARG DSV4_BOOTSTRAP_IMAGE=lumen/dsv4-lumen:mi308x

FROM alpine/git:latest AS sources

ARG MEGATRON_REPOSITORY=https://github.com/ROCm/Megatron-LM.git
ARG MEGATRON_REF=8dd45f5a51378ec1ee7937dee3c20d8626df4763
ARG AITER_REPOSITORY=https://github.com/ROCm/aiter.git
ARG AITER_SHA=ff745fb09d669f40cf81d15d3e7598b61fe2a61c
ARG FLASH_ATTN_REPOSITORY=https://github.com/ROCm/flash-attention.git
ARG FLASH_ATTN_REF=main

# Build context is the ATOM repository root.
COPY . /sources/ATOM

RUN set -eux; \
    git clone "${MEGATRON_REPOSITORY}" /sources/Megatron-LM; \
    git -C /sources/Megatron-LM checkout "${MEGATRON_REF}"; \
    git -C /sources/Megatron-LM submodule update --init --recursive; \
    git clone "${AITER_REPOSITORY}" /sources/aiter; \
    git -C /sources/aiter checkout --detach "${AITER_SHA}"; \
    git -C /sources/aiter submodule update --init --recursive; \
    git clone "${FLASH_ATTN_REPOSITORY}" /sources/flash-attention; \
    git -C /sources/flash-attention checkout "${FLASH_ATTN_REF}"; \
    git -C /sources/flash-attention submodule update --init --recursive; \
    printf '%s\n' \
      "AtomRL=$(git -C /sources/ATOM/atomrl rev-parse HEAD 2>/dev/null || echo local-working-tree)" \
      "Lumen=in-tree" \
      "Megatron-LM=$(git -C /sources/Megatron-LM rev-parse HEAD)" \
      "aiter=$(git -C /sources/aiter rev-parse HEAD)" \
      "flash-attention=$(git -C /sources/flash-attention rev-parse HEAD)" \
      > /sources/SOURCE_VERSIONS


FROM ${TRAINER_BASE_IMAGE} AS trainer

ARG IMAGE_ROLE=trainer
ARG EXPECTED_NUMPY=2.4.6
ARG EXPECTED_TRITON=3.7.0+amd.rocm7.2.0.git89002410
ARG EXPECTED_VLLM=absent

COPY --from=sources /sources /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN rm -f /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdgpu.list 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN PIP=/opt/venv/bin/pip && \
    "${PIP}" install --no-cache-dir \
      "ray[default]==2.56.1" \
      "transformers==5.2.0" \
      "datasets==5.0.0" \
      "accelerate==1.14.0" \
      "safetensors==0.8.0" \
      "omegaconf==2.3.1" \
      "tokenizers==0.22.2" \
      "math_verify==0.3.3" \
      "antlr4-python3-runtime==4.9.3" \
      "wandb==0.28.1" \
      "numpy==${EXPECTED_NUMPY}" \
      "pybind11>=3.0"

RUN cd /workspace/aiter && \
    /opt/venv/bin/pip install --no-cache-dir --no-build-isolation --no-deps -e .

RUN cd /workspace/flash-attention && \
    GPU_ARCHS=gfx942 /opt/venv/bin/pip install \
      --no-cache-dir --no-build-isolation --no-deps -e .

# ATOM metadata pins Transformers 5.16.1; this case owns its validated 5.2.0
# resolution above, so the unified package is intentionally installed --no-deps.
RUN LUMEN_BUILD_EXTENSIONS=1 /opt/venv/bin/pip install \
      --no-cache-dir --no-build-isolation --no-deps -e /workspace/ATOM

# Keep the base image's validated megatron-core metadata while making the pinned
# ROCm source tree (megatron.core + megatron.training + RouterReplay) authoritative.
RUN SITE=$(/opt/venv/bin/python -c 'import site; print(site.getsitepackages()[0])') && \
    printf '%s\n' /workspace/Megatron-LM > "${SITE}/megatron-lm-source.pth"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/workspace/Megatron-LM:/workspace/ATOM/atomrl:/workspace/aiter" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HIP_FORCE_DEV_KERNARG=1 \
    HSA_NO_SCRATCH_RECLAIM=0 \
    HSA_DISABLE_FRAGMENT_ALLOCATOR=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    NCCL_TIMEOUT=7200 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_DMABUF_ENABLE=0 \
    RAY_DEDUP_LOGS=0 \
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
    RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    LUMEN_DISABLE_HF_ATTN_PATCH=1 \
    USE_ROCM_AITER_ROPE_BACKEND=0 \
    VLLM_ROCM_USE_AITER=0 \
    VLLM_ROCM_USE_AITER_MHA=0 \
    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 \
    VLLM_ROCM_USE_AITER_LINEAR=0 \
    VLLM_USE_V1=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    ATOM_DISABLE_VLLM_PLUGIN=1

RUN IMAGE_ROLE="${IMAGE_ROLE}" \
    EXPECTED_NUMPY="${EXPECTED_NUMPY}" \
    EXPECTED_TRITON="${EXPECTED_TRITON}" \
    EXPECTED_VLLM="${EXPECTED_VLLM}" \
    /opt/venv/bin/python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import os
import torch

expected = {
    "ray": "2.56.1",
    "transformers": "5.2.0",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
    "omegaconf": "2.3.1",
    "tokenizers": "0.22.2",
    "wandb": "0.28.1",
    "numpy": os.environ["EXPECTED_NUMPY"],
    "triton": os.environ["EXPECTED_TRITON"],
    "flash-attn": "2.8.4",
    "amd-aiter": "0.1.0",
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise RuntimeError(f"{package}: expected {wanted}, got {actual}")

if not torch.__version__.startswith("2.10.0+rocm7.2.4"):
    raise RuntimeError(f"unexpected torch build: {torch.__version__}")
if torch.version.hip != "7.2.53211":
    raise RuntimeError(f"unexpected HIP build: {torch.version.hip}")

from megatron.training import get_args  # noqa: F401
from megatron.core.transformer.moe.router_replay import RouterReplay  # noqa: F401

expected_vllm = os.environ["EXPECTED_VLLM"]
try:
    actual_vllm = version("vllm")
except PackageNotFoundError:
    actual_vllm = "absent"
if actual_vllm != expected_vllm:
    raise RuntimeError(f"vllm: expected {expected_vllm}, got {actual_vllm}")

print(f"{os.environ['IMAGE_ROLE']} dependency verification passed")
PY

LABEL org.opencontainers.image.title="Qwen3-30B-A3B AtomRL trainer" \
      org.opencontainers.image.description="Megatron TP4/EP8 trainer for the two-node GRPO flow"

WORKDIR /workspace/ATOM/atomrl
CMD ["/bin/bash"]


FROM ${ROLLOUT_BASE_IMAGE} AS rollout

ARG IMAGE_ROLE=rollout
ARG EXPECTED_NUMPY=2.1.3
ARG EXPECTED_TRITON=3.7.0+amd.rocm7.2.0.gitd0d77a509
ARG EXPECTED_VLLM=0.22.1.dev0+g0b3ba88f1.d20260629.rocm724

COPY --from=sources /sources /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN rm -f /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdgpu.list 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN PIP=/opt/venv/bin/pip && \
    "${PIP}" install --no-cache-dir \
      "ray[default]==2.56.1" \
      "transformers==5.2.0" \
      "datasets==5.0.0" \
      "accelerate==1.14.0" \
      "safetensors==0.8.0" \
      "omegaconf==2.3.1" \
      "tokenizers==0.22.2" \
      "math_verify==0.3.3" \
      "antlr4-python3-runtime==4.9.3" \
      "wandb==0.28.1" \
      "numpy==${EXPECTED_NUMPY}" \
      "pybind11>=3.0"

RUN cd /workspace/aiter && \
    /opt/venv/bin/pip install --no-cache-dir --no-build-isolation --no-deps -e .

RUN cd /workspace/flash-attention && \
    GPU_ARCHS=gfx942 /opt/venv/bin/pip install \
      --no-cache-dir --no-build-isolation --no-deps -e .

# ATOM metadata pins Transformers 5.16.1; this case owns its validated 5.2.0
# resolution above, so the unified package is intentionally installed --no-deps.
RUN LUMEN_BUILD_EXTENSIONS=1 /opt/venv/bin/pip install \
      --no-cache-dir --no-build-isolation --no-deps -e /workspace/ATOM

RUN SITE=$(/opt/venv/bin/python -c 'import site; print(site.getsitepackages()[0])') && \
    printf '%s\n' /workspace/Megatron-LM > "${SITE}/megatron-lm-source.pth"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/workspace/Megatron-LM:/workspace/ATOM/atomrl:/workspace/aiter" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HIP_FORCE_DEV_KERNARG=1 \
    HSA_NO_SCRATCH_RECLAIM=0 \
    HSA_DISABLE_FRAGMENT_ALLOCATOR=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    NCCL_TIMEOUT=7200 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_DMABUF_ENABLE=0 \
    RAY_DEDUP_LOGS=0 \
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
    RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    LUMEN_DISABLE_HF_ATTN_PATCH=1 \
    USE_ROCM_AITER_ROPE_BACKEND=0 \
    VLLM_ROCM_USE_AITER=0 \
    VLLM_ROCM_USE_AITER_MHA=0 \
    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 \
    VLLM_ROCM_USE_AITER_LINEAR=0 \
    VLLM_USE_V1=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    ATOM_DISABLE_VLLM_PLUGIN=1

RUN IMAGE_ROLE="${IMAGE_ROLE}" \
    EXPECTED_NUMPY="${EXPECTED_NUMPY}" \
    EXPECTED_TRITON="${EXPECTED_TRITON}" \
    EXPECTED_VLLM="${EXPECTED_VLLM}" \
    /opt/venv/bin/python - <<'PY'
from importlib.metadata import version
import os
import torch

expected = {
    "ray": "2.56.1",
    "transformers": "5.2.0",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
    "omegaconf": "2.3.1",
    "tokenizers": "0.22.2",
    "wandb": "0.28.1",
    "numpy": os.environ["EXPECTED_NUMPY"],
    "triton": os.environ["EXPECTED_TRITON"],
    "flash-attn": "2.8.4",
    "amd-aiter": "0.1.0",
    "vllm": os.environ["EXPECTED_VLLM"],
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise RuntimeError(f"{package}: expected {wanted}, got {actual}")

if not torch.__version__.startswith("2.10.0+rocm7.2.4"):
    raise RuntimeError(f"unexpected torch build: {torch.__version__}")
if torch.version.hip != "7.2.53211":
    raise RuntimeError(f"unexpected HIP build: {torch.version.hip}")

from megatron.training import get_args  # noqa: F401
from megatron.core.transformer.moe.router_replay import RouterReplay  # noqa: F401

print(f"{os.environ['IMAGE_ROLE']} dependency verification passed")
PY

LABEL org.opencontainers.image.title="Qwen3-30B-A3B AtomRL rollout" \
      org.opencontainers.image.description="vLLM TP2 rollout for the two-node GRPO flow"

WORKDIR /workspace/ATOM/atomrl
CMD ["/bin/bash"]


# ═══════════════════════════════════════════════════════════════════════════════
# DSV4-Flash targets
#
# Both trainer and rollout share the same base image (vllm/vllm-openai-rocm:v0.25.1)
# to guarantee identical Python and PyTorch versions across the Ray cluster.
#
# The trainer additionally installs Megatron-LM, Lumen (with aiter/mori),
# and flash-attention for the DSV4 MLA/MoE training spec.
#
# Build (context = ATOM repository root):
#   docker build -f atomrl/examples/GRPO/docker/Dockerfile --target dsv4-trainer \
#     -t dsv4-flash:trainer .
#
#   docker build -f atomrl/examples/GRPO/docker/Dockerfile --target dsv4-rollout \
#     -t dsv4-flash:rollout .
# ═══════════════════════════════════════════════════════════════════════════════

FROM ${DSV4_BOOTSTRAP_IMAGE} AS dsv4-bootstrap
# Extract tilelang, Megatron (Miles fork with DSV4 fields), and tile_kernels
# from the existing lumen/dsv4-lumen:mi308x image.

FROM ${DSV4_BASE_IMAGE} AS dsv4-trainer

ARG IMAGE_ROLE=dsv4-trainer

COPY --from=sources /sources /workspace

# Bootstrap artifacts from the Lumen DSV4 image:
# - Megatron-LM (Miles fork with MLA/DSV4 TransformerConfig fields + RouterReplay)
# - tilelang (Sparse MLA + indexer kernels, pre-built for gfx942)
# - tile_kernels (mHC backend)
# - native libs (mooncake)
COPY --from=dsv4-bootstrap /opt/dsv4-bootstrap/Megatron-LM /workspace/Megatron-LM
COPY --from=dsv4-bootstrap /opt/dsv4-runtime/tilelang /opt/tilelang
COPY --from=dsv4-bootstrap /opt/dsv4-bootstrap/site-packages/tile_kernels /opt/dsv4-site/tile_kernels
COPY --from=dsv4-bootstrap /opt/dsv4-bootstrap/native-libs /opt/dsv4-native-libs

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Avoid rdma-core — it conflicts with the pre-installed rccl in the vLLM image.
# libibverbs is already present in the base image. RDMA works via --privileged.
RUN rm -f /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdgpu.list 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ninja-build && \
    rm -rf /var/lib/apt/lists/*

# RL + training dependencies
RUN pip install --no-cache-dir \
      "ray[default]==2.56.1" \
      "transformers>=5.2,<5.17,!=5.6.0" \
      "datasets>=5.0" \
      "accelerate>=1.14" \
      "safetensors>=0.8" \
      "omegaconf==2.3.1" \
      "tokenizers>=0.22" \
      "math_verify>=0.3" \
      "antlr4-python3-runtime==4.9.3" \
      "wandb>=0.28" \
      "einops" \
      "pybind11>=3.0"

# aiter GPU kernels (build for MI308X/MI300X gfx942)
RUN cd /workspace/aiter && \
    GPU_ARCHS=gfx942 pip install --no-cache-dir --no-build-isolation --no-deps -e . || true

# flash-attention (ROCm)
RUN cd /workspace/flash-attention && \
    GPU_ARCHS=gfx942 pip install --no-cache-dir --no-build-isolation --no-deps -e . || true

# Unified ATOM/AtomRL package (including atomrl.lumen). The case-owned
# Transformers resolution intentionally overrides ATOM's 5.16.1 metadata pin.
RUN LUMEN_BUILD_EXTENSIONS=1 pip install --no-cache-dir --no-build-isolation --no-deps \
      -e /workspace/ATOM

# Megatron-LM (Miles fork) via .pth + tilelang + tile_kernels on PYTHONPATH
RUN SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])') && \
    printf '%s\n' /workspace/Megatron-LM > "${SITE}/megatron-lm-source.pth" && \
    ln -sf /opt/dsv4-site/tile_kernels "${SITE}/tile_kernels"

ENV PYTHONPATH="/workspace/Megatron-LM:/workspace/ATOM/atomrl:/workspace/aiter:/opt/tilelang" \
    LD_LIBRARY_PATH="/opt/dsv4-native-libs:${LD_LIBRARY_PATH:-}" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HSA_NO_SCRATCH_RECLAIM=0 \
    HSA_DISABLE_FRAGMENT_ALLOCATOR=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    NCCL_TIMEOUT=7200 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_DMABUF_ENABLE=0 \
    RAY_DEDUP_LOGS=0 \
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
    RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    LUMEN_DISABLE_HF_ATTN_PATCH=1 \
    USE_ROCM_AITER_ROPE_BACKEND=0 \
    TORCHDYNAMO_DISABLE=1 \
    ATOM_DISABLE_VLLM_PLUGIN=1

RUN python3 -c "\
import ray; print(f'ray: {ray.__version__}'); \
import torch; print(f'torch: {torch.__version__}'); \
import atomrl; print('atomrl: ok'); \
from megatron.training import get_args; print('megatron.training: ok'); \
print('${IMAGE_ROLE} verification passed')"

LABEL org.opencontainers.image.title="DSV4-Flash AtomRL trainer" \
      org.opencontainers.image.description="Megatron TP4/PP4/EP4 DSV4 trainer for the three-node GRPO flow"

WORKDIR /workspace/ATOM/atomrl
CMD ["/bin/bash"]


FROM ${DSV4_BASE_IMAGE} AS dsv4-rollout

ARG IMAGE_ROLE=dsv4-rollout

COPY --from=sources /sources/ATOM /workspace/ATOM

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Preserve this case's Transformers resolution instead of ATOM's 5.16.1 pin;
# the root editable install is intentionally --no-deps.
RUN pip install --no-cache-dir \
      "ray[default]==2.56.1" \
      "transformers>=5.2,<5.17,!=5.6.0" \
      "datasets>=5.0" \
      "accelerate>=1.14" \
      "safetensors>=0.8" \
      "omegaconf==2.3.1" \
      "tokenizers>=0.22" \
      "math_verify>=0.3" \
      "antlr4-python3-runtime==4.9.3" \
      "wandb>=0.28" && \
    LUMEN_BUILD_EXTENSIONS=1 pip install --no-cache-dir --no-build-isolation --no-deps \
      -e /workspace/ATOM

ENV PYTHONPATH="/workspace/ATOM/atomrl:${PYTHONPATH:-}" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HIP_FORCE_DEV_KERNARG=1 \
    HSA_NO_SCRATCH_RECLAIM=0 \
    HSA_DISABLE_FRAGMENT_ALLOCATOR=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    NCCL_TIMEOUT=7200 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_DMABUF_ENABLE=0 \
    RAY_DEDUP_LOGS=0 \
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
    RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    LUMEN_DISABLE_HF_ATTN_PATCH=1 \
    USE_ROCM_AITER_ROPE_BACKEND=0 \
    VLLM_ROCM_USE_AITER=1 \
    VLLM_USE_V1=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    ATOM_DISABLE_VLLM_PLUGIN=1

RUN python3 -c "\
import ray; print(f'ray: {ray.__version__}'); \
import vllm; print(f'vllm: {vllm.__version__}'); \
import atomrl; print('atomrl: ok'); \
print('${IMAGE_ROLE} verification passed')"

LABEL org.opencontainers.image.title="DSV4-Flash AtomRL rollout" \
      org.opencontainers.image.description="vLLM TP8 FP8 rollout for the three-node DSV4 GRPO flow"

WORKDIR /workspace/ATOM/atomrl
CMD ["/bin/bash"]
