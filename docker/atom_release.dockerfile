# Keep the release reproducible on the ROCm 7.2.4 / PyTorch 2.10 stack.
# The digest prevents this historical tag from being moved underneath us.
ARG BASE_IMAGE="rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
ARG GPU_ARCH="gfx942;gfx950"
# ROCm 10 flavor: pass --build-arg BASE_IMAGE=rocm10-base to build the whole
# image on the pip-installed ROCm 10 SDK below instead of the rocm/pytorch
# apt image. All ROCm 10 component versions are ARGs so the 10.1 tracking
# line only changes build-args (see ROCM_INDEX_URL / ROCM_SDK_VERSION).
ARG BASE_IMAGE_ROCM10="ubuntu:24.04"

# ====================================================================
# ATOM image: multi-stage parallel build
#
# BuildKit runs independent builder stages in parallel:
#   base ──┬── build_rccl  ──┐
#          └── build_aiter ──┴── atom_image (merge builders + install MORI/ATOM)
#
# Triton is NOT built from source: the ROCm PyTorch base image already ships a
# matching Triton (installed as a torch dependency), and aiter handles its own
# Triton needs at install time. The previous build_triton stage that compiled
# ROCm/triton release/internal/3.5.x has been removed.
# ====================================================================

# --------------------------------------------------------------------
# Stage -1: ROCm 10 base (pip SDK assembly on plain Ubuntu).
#
# Assemble the ROCm 10 stack from AMD's stable wheel channel instead of a
# rocm/pytorch apt image (the TheRock distribution model: ROCm ships as pip
# wheels, installed into site-packages rather than /opt/rocm). Ported from
# sglang's docker/rocm.Dockerfile rocm1000-base stage.
#
# Version policy (ticket: ATOM v0.1.7 for ROCm 10.1, GA 2026-10-05):
#   - Main line: stable channel 10.0.0 (the combination sglang validated).
#   - 10.1 tracking line: nightly channel with 10.1.0a<date> versions
#     (rc.repo.amd.com has no 10.1 artifacts yet). NOT a release artifact;
#     it exists to surface 10.1 breaks early. Once 10.1 RC/GA lands on the
#     stable channel, flip these ARGs and re-run the golden test.
#
# Python 3.12 (Ubuntu 24.04 default): the wheel channel publishes cp312 for
# the whole stack; torch 2.11+rocm10.0.0 is the version sglang validated.
# --------------------------------------------------------------------
FROM ${BASE_IMAGE_ROCM10} AS rocm10-base

# Redeclare the global selector here so --build-arg GPU_ARCH lands in this
# stage too; every device payload below is derived from it.
ARG GPU_ARCH

# ROCM_TRITON_VERSION rather than TRITON_VERSION to leave room for a
# stage-local TRITON_VERSION without a --build-arg landing on both.
ARG ROCM_SDK_VERSION="10.0.0"
ARG ROCM_TORCH_VERSION="2.11.0"
ARG ROCM_TORCHVISION_VERSION="0.26.0"
ARG ROCM_TORCHAUDIO_VERSION="2.11.0"
ARG ROCM_TRITON_VERSION="3.8.0+git4cff872c"
ARG ROCM_INDEX_URL="https://stable.repo.amd.com/rocm/whl-next/"
# 10.1 tracking line overrides (nightly channel, date-stamped versions):
#   --build-arg ROCM_SDK_VERSION=10.1.0a<yyyymmdd>
#   --build-arg ROCM_TORCH_VERSION=2.12.0
#   --build-arg ROCM_TRITON_VERSION=3.8.0+gitc01b6774
#   --build-arg ROCM_INDEX_URL=https://nightly.repo.amd.com/rocm/whl-next/
# (the +rocm10.1.0a<date> local version suffix is derived from
#  ROCM_SDK_VERSION below, so only the base version needs overriding)

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gnupg \
        libstdc++-12-dev \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3.12-venv \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install --no-cache-dir -U pip setuptools setuptools_scm wheel

# Unlike the sglang rocm1000 flavors (one device payload per image), the ATOM
# release image is a single multi-arch image: GPU_ARCH="gfx942;gfx950" gets a
# device payload for every listed arch. Loop over GPU_ARCH_LIST so adding an
# arch stays a GPU_ARCH change, not a new package list.
RUN set -eux; \
    for arch in $(printf '%s' "${GPU_ARCH}" | tr ';' ' '); do \
        python3 -m pip install --no-cache-dir \
            --index-url ${ROCM_INDEX_URL} \
            "rocm-sdk-device-${arch}==${ROCM_SDK_VERSION}" \
            "amd-torch-device-${arch}==${ROCM_TORCH_VERSION}+rocm${ROCM_SDK_VERSION}" \
            "amd-torchvision-device-${arch}==${ROCM_TORCHVISION_VERSION}+rocm${ROCM_SDK_VERSION}"; \
    done; \
    python3 -m pip install --no-cache-dir \
        --index-url ${ROCM_INDEX_URL} \
        "rocm-sdk-core==${ROCM_SDK_VERSION}" \
        "rocm-sdk-libraries==${ROCM_SDK_VERSION}" \
        "rocm-sdk-devel==${ROCM_SDK_VERSION}" \
        "torch==${ROCM_TORCH_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "torchvision==${ROCM_TORCHVISION_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "torchaudio==${ROCM_TORCHAUDIO_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "triton==${ROCM_TRITON_VERSION}.rocm${ROCM_SDK_VERSION}"; \
    for arch in $(printf '%s' "${GPU_ARCH}" | tr ';' ' '); do \
        python3 -m pip show "rocm-sdk-device-${arch}" >/dev/null; \
        python3 -m pip show "amd-torch-device-${arch}" >/dev/null; \
    done

RUN rocm-sdk init && rocm-sdk targets

# rocm-sdk init expands a devel tree that carries its own copy of libamd_smi,
# byte-identical to the one in _rocm_sdk_core that HIP loads through its RPATH.
# Since ROCM_HOME below puts the devel tree on LD_LIBRARY_PATH, the amdsmi
# python package binds that second copy while torch already holds the first,
# and two independent copies in one process each keep their own global state:
# whichever initialises second enumerates no devices. torch asks amdsmi for the
# device count before HIP, so `torch.cuda.device_count()` comes back 0 on a
# machine where hipGetDeviceCount() says 8. Collapse the duplicate so both land
# on the same library. Idempotent when the SDK already ships a symlink here.
RUN set -eux; \
    SP="$VIRTUAL_ENV/lib/python3.12/site-packages"; \
    CORE=$(ls "$SP"/_rocm_sdk_core/lib/libamd_smi.so.* 2>/dev/null | head -1); \
    DEVEL="$SP/_rocm_sdk_devel/lib/libamd_smi.so"; \
    if [ -n "${CORE}" ] && [ -e "${DEVEL}" ] && [ ! -L "${DEVEL}" ]; then \
        ln -sf "${CORE}" "${DEVEL}"; \
        echo "linked ${DEVEL} -> ${CORE}"; \
    fi

ENV ROCM_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_devel
ENV ROCM_PATH=$ROCM_HOME
ENV CPATH=$ROCM_HOME/include
ENV LIBRARY_PATH=$ROCM_HOME/lib
ENV LD_LIBRARY_PATH=$ROCM_HOME/lib
RUN echo 'export PATH=$ROCM_HOME/llvm/bin:$ROCM_HOME/bin:$PATH' >> /etc/bash.bashrc

# The SDK's hsakmtTargets.cmake hardcodes /usr/lib64/libc.so from its own build
# host; Ubuntu keeps libc in /lib/x86_64-linux-gnu, so cmake would otherwise
# fail with "ninja: error: /usr/lib64/libc.so missing and no known rule to make it".
RUN mkdir -p /usr/lib64 && ln -sf /lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so

# ROCm lives in site-packages here, but AITER shells out to
# /opt/rocm/llvm/bin/amdgpu-arch at runtime to pick DEFAULT_GPU_ARCH, RCCL's
# install.sh and Mooncake's cmake both expect /opt/rocm, and the amdsmi pip
# install below refers to /opt/rocm/share/amd_smi.
RUN ln -s ${ROCM_HOME} /opt/rocm

# amdsmi: the pip SDK (unlike the rocm/pytorch apt images) does not preinstall
# the AMD SMI python package; ATOM and the validation steps import it.
RUN cd /opt/rocm/share/amd_smi && python3 -m pip install --no-cache-dir . && \
    python3 -c "import amdsmi; print('amdsmi ok')"

# Keep pip from resolving the ROCm torch stack away to PyPI CUDA builds in any
# later pip install (AITER requirements, MORI, ATOM deps, ...). The local
# +rocm10.x version numbers lose to plain "torch==X" specs from PyPI unless
# constrained. Deliberately shipped in the image (not just build-time) so user
# pip installs stay on the ROCm stack too. Only the torch trio is named.
ENV PIP_CONSTRAINT="/etc/atom/constraints/torch-rocm.txt"
RUN mkdir -p /etc/atom/constraints && \
    python3 -m pip freeze | grep -E '^(torch|torchvision|torchaudio)(==| @ )' \
        > /etc/atom/constraints/torch-rocm.txt && \
    cat /etc/atom/constraints/torch-rocm.txt

# --------------------------------------------------------------------
# Stage 0: Common base (apt + pip foundations, shared by all builders)
# --------------------------------------------------------------------
FROM ${BASE_IMAGE} AS base

ARG GPU_ARCH
ARG BASE_IMAGE
ENV GPU_ARCH_LIST=$GPU_ARCH
ENV PYTORCH_ROCM_ARCH=$GPU_ARCH
# Stamp the chosen base so later stages can branch on "is this the rocm10
# flavor" without guessing from torch/HIP versions (10.0 and 10.1 both
# report torch 2.11+ in some combinations).
ENV ATOM_BASE_IMAGE=${BASE_IMAGE}

# AITER's prebuilt and runtime-JIT modules must use the same pybind ABI.
RUN pip install --upgrade pip "pybind11==3.0.4" && \
    apt-get update && \
    apt --fix-broken install -y && \
    apt-get install -y \
        git cython3 ibverbs-utils openmpi-bin libopenmpi-dev \
        libpci-dev cmake libdw1 locales && \
    rm -rf /var/lib/apt/lists/*

# Newer rocm/pytorch images install ROCm libraries through Python wheels
# instead of /opt/rocm. Register those directories so dpkg-shlibdeps can
# resolve RCCL's dependencies while retaining compatibility with /opt/rocm.
# Covers both the rocm10-base layout (SDK under /opt/venv, /opt/rocm symlink)
# and any future rocm/pytorch image that moves to the same wheel layout.
RUN ROCM_SDK_LIB_DIRS="$(python -c \
        'import glob, os; print("\n".join(sorted({os.path.dirname(p) for p in glob.glob("/opt/venv/lib/python*/site-packages/_rocm_sdk*/lib/*.so*")})))')" && \
    if [ -n "${ROCM_SDK_LIB_DIRS}" ]; then \
        printf '%s\n' "${ROCM_SDK_LIB_DIRS}" \
            > /etc/ld.so.conf.d/rocm-python-sdk.conf; \
        ldconfig; \
    fi

# ROCm 10 torch stack tripwire: fail the build right here if any earlier step
# let a PyPI CUDA torch replace the +rocm10.x stack, or if the venv carries
# NVIDIA runtime packages. Only the rocm10 flavor asserts (the rocm/pytorch
# apt images ship their own, differently-versioned, stacks).
RUN if [ "${ATOM_BASE_IMAGE}" = "rocm10-base" ]; then \
        python -m pip check && \
        python -c "import torch; assert torch.version.hip is not None, torch.__version__; print('rocm10 base torch:', torch.__version__, 'hip:', torch.version.hip)" && \
        if pip list --format=freeze 2>/dev/null | grep -Eq '^nvidia-.*-cu[0-9]+'; then \
            echo "ERROR: NVIDIA CUDA runtime packages leaked into the ROCm 10 image"; \
            exit 1; \
        fi; \
    fi

# --------------------------------------------------------------------
# Stage 1: RCCL — parallel
# --------------------------------------------------------------------
FROM base AS build_rccl
ARG RCCL_REPO="https://github.com/ROCm/rccl.git"
ARG RCCL_BRANCH="29e1567b95e28823b0beb1a988adc587bfab5b4f"

RUN echo "========== [Parallel] Building RCCL ==========" && \
    pip install cmake && \
    git clone "$RCCL_REPO" /app/rccl && \
    cd /app/rccl && \
    git checkout "$RCCL_BRANCH" && \
    ./install.sh -p --amdgpu_targets=$GPU_ARCH_LIST

# --------------------------------------------------------------------
# Stage 2: Aiter — parallel
# --------------------------------------------------------------------
FROM base AS build_aiter
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="HEAD"
ARG PREBUILD_KERNELS=1
ARG MAX_JOBS

RUN pip install --upgrade setuptools_scm
RUN echo "========== [Parallel] Building Aiter ==========" && \
    git clone $AITER_REPO /app/aiter-test && \
    cd /app/aiter-test && \
    git checkout $AITER_COMMIT && \
    git submodule sync && git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    MAX_JOBS=$MAX_JOBS PREBUILD_KERNELS=$PREBUILD_KERNELS \
    GPU_ARCHS=$GPU_ARCH_LIST python3 setup.py develop

# --------------------------------------------------------------------
# Stage 3: Final merge — collect all build artifacts + install MORI/ATOM
# --------------------------------------------------------------------
FROM base AS atom_image
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"

# pip packages (lm-eval is lightweight, install directly)
RUN pip install lm-eval[api]

# MORI: install the prebuilt nightly wheel directly (no source build needed).
# The `amd-mori-nightly` PyPI package provides the `mori` Python module.
# See: https://pypi.org/project/amd-mori-nightly/
RUN echo "========== [ATOM] Installing MORI nightly ==========" && \
    pip install --pre amd-mori-nightly && \
    python -c "import mori; print(f'mori: {mori.__file__}')" && \
    pip show amd-mori-nightly

# ========== Mooncake TransferEngine ==========
# Mooncake and Rust apt operations MUST run before the RCCL dpkg -i --force-all
# step below, because that step overwrites the Ubuntu-repo rccl with a custom
# ROCm build whose version string doesn't match rocm-hip's declared dependency,
# leaving dpkg in a broken state that blocks all subsequent apt-get install calls.
ARG INSTALL_MOONCAKE=1
ARG MOONCAKE_REPO="https://github.com/Jasen2201/Mooncake.git"
ARG MOONCAKE_COMMIT="fix/ionic-mr-and-qp-resource-fixes"
ARG VENV_PYTHON="/opt/venv/bin/python"

# [MC 1/4] Clone
RUN if [ "${INSTALL_MOONCAKE}" = "1" ]; then \
        echo "========== [MC 1/4] Clone Mooncake =========="; \
        git clone ${MOONCAKE_REPO} /app/mooncake && \
        cd /app/mooncake && \
        git checkout "${MOONCAKE_COMMIT}" && \
        git submodule update --init --recursive && \
        echo "Mooncake commit: $(git rev-parse HEAD)"; \
    else \
        echo "========== Skipped Mooncake (INSTALL_MOONCAKE=0) =========="; \
    fi

# [MC 2/4] Install dependencies (system packages + RDMA + Go + submodules)
ENV PATH="/usr/local/go/bin:${PATH}"
RUN if [ "${INSTALL_MOONCAKE}" = "1" ]; then \
        echo "========== [MC 2/4] Install Mooncake dependencies =========="; \
        apt-get update && apt-get install -y --no-install-recommends \
            zip unzip wget gcc make libtool autoconf \
            librdmacm-dev rdmacm-utils infiniband-diags perftest ethtool \
            libibverbs-dev rdma-core \
            openssh-server openmpi-common && \
        cd /app/mooncake && bash dependencies.sh -y && \
        rm -rf /usr/local/go && \
        wget -q https://go.dev/dl/go1.22.2.linux-amd64.tar.gz && \
        tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz && \
        rm go1.22.2.linux-amd64.tar.gz; \
    fi

# [MC 2.5/4] Install AMD Pensando ionic RDMA provider for Mooncake RDMA transport.
# The container's apt rdma-core (v39) predates the ionic provider. The upstream
# rdma-core v61 ionic source only supports kernel ABI 1, but Pensando's ionic NIC
# driver uses kernel ABI 4. Pensando's custom libionic1 deb (based on rdma-core v54
# fork) supports ABI 1-4 and is required for correct RDMA operation.
ARG IONIC_DEB_URL="https://repo.radeon.com/amdainic/pensando/ubuntu/1.117.1-a-63/pool/main/r/rdma-core/libionic1_54.0-149.g3304be71_amd64.deb"
RUN if [ "${INSTALL_MOONCAKE}" = "1" ]; then \
        echo "========== [MC 2.5/4] Install ionic RDMA provider =========="; \
        curl -fSL "${IONIC_DEB_URL}" -o /tmp/libionic1.deb && \
        dpkg -i /tmp/libionic1.deb && \
        echo "driver ionic" > /etc/libibverbs.d/ionic.driver && \
        ldconfig && \
        echo "Installed ionic provider:" && \
        ls -la /usr/lib/x86_64-linux-gnu/libibverbs/libionic* && \
        rm -f /tmp/libionic1.deb; \
    fi

# [MC 3/4] CMake build with HIP support + system install
RUN if [ "${INSTALL_MOONCAKE}" = "1" ]; then \
        echo "========== [MC 3/4] Build and install Mooncake (USE_HIP=ON) =========="; \
        mkdir -p /app/mooncake/build && cd /app/mooncake/build \
        && cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON \
        && make -j$(nproc) && make install \
        && ldconfig \
        && echo "--- Clean up build artifacts ---" \
        && rm -rf /app/mooncake/build /app/mooncake/.git; \
    fi

# ========== Install Rust toolchain ==========
ARG RUST_VERSION="1.94.0"

RUN echo "========== Install Rust toolchain ==========" \
    && apt-get update && apt-get install -y --no-install-recommends curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal \
    && . "$HOME/.cargo/env" \
    && rustc --version && cargo --version

ENV PATH="/root/.cargo/bin:${PATH}"

# RCCL: install .deb from build stage
# WARNING: dpkg -i --force-all overwrites Ubuntu-repo rccl with the ROCm custom
# build, breaking rocm-hip's version dep in dpkg metadata. All apt-get install
# operations (Mooncake, Rust, etc.) MUST be completed before this step.
COPY --from=build_rccl /app/rccl/build/release/*.deb /tmp/rccl/
RUN DEBIAN_FRONTEND=noninteractive dpkg -i --force-all /tmp/rccl/*.deb && \
    rm -rf /tmp/rccl

# Triton ships with the ROCm PyTorch base image (installed as a torch dependency);
# no separate build/copy step is needed here.

# Aiter: copy compiled source tree + re-register editable install
# (pip install -e creates egg-link automatically, no need to COPY them)
COPY --from=build_aiter /app/aiter-test /app/aiter-test
RUN cd /app/aiter-test && pip install -e . --no-build-isolation && \
    pip show amd-aiter

# RTL (rocm-trace-lite): lightweight GPU kernel profiler (~250KB, no build deps)
RUN pip install rocm-trace-lite && \
    rtl --version || true

# ATOM: Python package install (editable) with the atomesh build hook enabled.
# CACHEBUST invalidates only this layer so parallel stages stay cached
ARG CACHEBUST=1
RUN git clone $ATOM_REPO /app/ATOM && \
    cd /app/ATOM && \
    git checkout $ATOM_COMMIT && \
    ATOM_MESH_BUILD=1 python -m pip install -e .
RUN pip show atom || true

RUN pip install --no-cache-dir msgpack msgspec quart

# atomesh: install the binary produced by the ATOM package build hook to /usr/local/bin
RUN echo "========== Install atomesh binary ==========" && \
    cd /app/ATOM/atom/mesh && \
    strip target/release/atomesh && \
    cp target/release/atomesh /usr/local/bin/atomesh && \
    atomesh --version

# ========== LMCache (HIP c_ops) for KV offload ==========
# ATOM's KV offload uses the LMCache connector, which needs LMCache's c_ops
# built for ROCm. NEVER `pip install lmcache` — it pulls CUDA torch and breaks
# the ROCm stack. Build from source pinned to a release tag, against the image's
# torch. KEY: the install must be EDITABLE (`pip install -e .`); at this tag a
# non-editable `pip install .` silently SKIPS the c_ops extension and falls back
# to the slow python backend. Verified end-to-end (tp8 offload store via c_ops)
# on gfx950. NOTE: tp>1 offload also needs aiter's eager-NCCL-init fix
# (device_id= to init_process_group); that belongs in aiter (tracked separately),
# not here — the CI build_aiter stage picks it up once merged.
ARG LMCACHE_TAG=v0.4.5
# PYTORCH_ROCM_ARCH is inherited as ENV from the `base` stage (=${GPU_ARCH});
# hipcc reads it to target both gfx942 and gfx950. Do not re-derive from
# ${GPU_ARCH} here — ARG does not cross FROM so it would be empty in this stage.
# Docker builds do not expose a GPU, so LMCache's torch.cuda.is_available()
# backend predicate is overridden only in the validation process below.
RUN echo "========== [ATOM] LMCache HIP c_ops (${LMCACHE_TAG}, arch=${PYTORCH_ROCM_ARCH}) ==========" && \
    git clone https://github.com/LMCache/LMCache.git /opt/LMCache && \
    cd /opt/LMCache && git checkout ${LMCACHE_TAG} && \
    "${VENV_PYTHON}" -m pip install -r requirements/build.txt && \
    CXX=hipcc BUILD_WITH_HIP=1 \
      "${VENV_PYTHON}" -m pip install -e . --no-build-isolation --no-deps && \
    "${VENV_PYTHON}" -m pip install --no-deps \
        prometheus_client==0.25.0 aiofile==3.11.1 caio==0.9.25 && \
    "${VENV_PYTHON}" -c "import glob, torch; \
c_ops_paths = glob.glob('/opt/LMCache/lmcache/c_ops*.so'); \
assert c_ops_paths, 'LMCache HIP c_ops extension was not built'; \
torch.cuda.is_available = lambda: True; \
import lmcache, lmcache.c_ops; \
from lmcache.v1.cache_engine import LMCacheEngineBuilder; \
from lmcache.v1.memory_management import MemoryFormat; \
from lmcache.v1.lookup_client.factory import LookupClientFactory; \
from lmcache.v1.config import LMCacheEngineConfig; \
from lmcache.v1.metadata import LMCacheMetadata; \
assert 'rocm' in torch.__version__, torch.__version__; \
assert lmcache.c_ops.__file__.endswith('.so'), 'c_ops fell back to python backend!'; \
print('OK: lmcache', lmcache.__version__, 'HIP c_ops; torch', torch.__version__)"

# ========== SemiAnalysis aiperf agentic benchmark tool ==========
# The SemiAnalysis fork, which is what carries the SA agentic datasets
# (semianalysis_cc_traces_weka_062126*).
#
# Pinned to the commit InferenceX's `utils/aiperf` submodule points at, so our
# image ships the aiperf they measure with. Their pointer is a deliberate,
# frequently-moved pin -- four bumps in the first half of August 2026, and a
# same-day revert on 2026-07-28 -- with commit titles that read "pin AIPerf v1
# timing watchdog", "pin additive AIPerf main warmup". Tracking aiperf's master
# instead would take in exactly the upstream changes they evaluate and
# sometimes reject.
#
# It therefore has to be followed by hand. To re-check:
#   git ls-tree main utils/aiperf     # in a clone of SemiAnalysisAI/InferenceX
#
# `SA_AIPERF_REF` accepts any ref; empty means "whatever HEAD points at", which
# is why the checkout below is conditional rather than naming a branch (an
# upstream default-branch rename would otherwise break unrelated builds).
ARG INSTALL_SA_AIPERF=1
ARG SA_AIPERF_REF="754356e9a39acc6cc6afb242d123bb57c3fb6f75"
RUN if [ "${INSTALL_SA_AIPERF}" = "1" ]; then \
        echo "========== [ATOM] Install SemiAnalysis aiperf (ref=${SA_AIPERF_REF:-<default branch>}) =========="; \
        rm -rf /opt/aiperf && \
        git clone https://github.com/SemiAnalysisAI/aiperf.git /opt/aiperf && \
        cd /opt/aiperf && \
        { [ -z "${SA_AIPERF_REF}" ] || git checkout "${SA_AIPERF_REF}"; } && \
        echo "[ATOM] aiperf resolved to $(git rev-parse HEAD)" && \
        sed -i '/^[[:space:]]*"transformers @ git+/d' pyproject.toml && \
        ! grep -q '^[[:space:]]*"transformers @ git+' pyproject.toml && \
        "${VENV_PYTHON}" -m pip install -e . && \
        "${VENV_PYTHON}" -c "import transformers; print(f'transformers.__version__ = {transformers.__version__}')" && \
        "${VENV_PYTHON}" -m pip show aiperf || true && \
        command -v aiperf && aiperf --help >/dev/null; \
    else \
        echo "========== Skipped SemiAnalysis aiperf (INSTALL_SA_AIPERF=0) =========="; \
    fi

CMD ["/bin/bash"]
