# Default base image
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG GPU_ARCH="gfx942;gfx950"
ARG TORCH_ROCM_WHEEL_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torch-2.11.0%2Brocm7.2.2.lw.git4e323059-cp312-cp312-linux_x86_64.whl"
ARG TORCHVISION_ROCM_WHEEL_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torchvision-0.26.0%2Brocm7.2.2.git336d36e8-cp312-cp312-linux_x86_64.whl"
ARG TORCHAUDIO_ROCM_WHEEL_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torchaudio-2.10.0%2Brocm7.2.2.git5047768f-cp312-cp312-linux_x86_64.whl"

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
# Stage 0: Common base (apt + pip foundations, shared by all builders)
# --------------------------------------------------------------------
FROM ${BASE_IMAGE} AS base

ARG GPU_ARCH
ARG TORCH_ROCM_WHEEL_URL
ARG TORCHVISION_ROCM_WHEEL_URL
ARG TORCHAUDIO_ROCM_WHEEL_URL
ENV GPU_ARCH_LIST=$GPU_ARCH
ENV PYTORCH_ROCM_ARCH=$GPU_ARCH

RUN pip install --upgrade pip && \
    pip install --no-deps --force-reinstall \
        "${TORCH_ROCM_WHEEL_URL}" \
        "${TORCHVISION_ROCM_WHEEL_URL}" \
        "${TORCHAUDIO_ROCM_WHEEL_URL}" && \
    python -c "import torch, torchvision, torchaudio; print(f'torch={torch.__version__}'); print(f'hip={torch.version.hip}'); print(f'torchvision={torchvision.__version__}'); print(f'torchaudio={torchaudio.__version__}'); assert torch.version.hip is not None" && \
    apt-get update && \
    apt --fix-broken install -y && \
    apt-get install -y \
        git cython3 ibverbs-utils openmpi-bin libopenmpi-dev \
        libpci-dev cmake libdw1 locales && \
    rm -rf /var/lib/apt/lists/*

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
        cd /app/mooncake && \
        sed -i 's|tar -C /usr/local -xzf|tar --no-same-owner --no-same-permissions -C /usr/local -xzf|' dependencies.sh && \
        bash dependencies.sh -y && \
        rm -rf /usr/local/go && \
        wget -q https://go.dev/dl/go1.22.2.linux-amd64.tar.gz && \
        tar --no-same-owner --no-same-permissions \
            -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz && \
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
RUN echo "========== [ATOM] LMCache HIP c_ops (${LMCACHE_TAG}, arch=${PYTORCH_ROCM_ARCH}) ==========" && \
    git clone https://github.com/LMCache/LMCache.git /opt/LMCache && \
    cd /opt/LMCache && git checkout ${LMCACHE_TAG} && \
    "${VENV_PYTHON}" -m pip install -r requirements/build.txt && \
    CXX=hipcc BUILD_WITH_HIP=1 \
      "${VENV_PYTHON}" -m pip install -e . --no-build-isolation --no-deps && \
    "${VENV_PYTHON}" -m pip install --no-deps \
        prometheus_client==0.25.0 aiofile==3.11.1 caio==0.9.25 && \
    "${VENV_PYTHON}" -c "import torch, lmcache, lmcache.c_ops; \
from lmcache.v1.cache_engine import LMCacheEngineBuilder; \
from lmcache.v1.memory_management import MemoryFormat; \
from lmcache.v1.lookup_client.factory import LookupClientFactory; \
from lmcache.v1.config import LMCacheEngineConfig; \
from lmcache.v1.metadata import LMCacheMetadata; \
assert 'rocm' in torch.__version__, torch.__version__; \
assert lmcache.c_ops.__file__.endswith('.so'), 'c_ops fell back to python backend!'; \
print('OK: lmcache', lmcache.__version__, 'HIP c_ops; torch', torch.__version__)"

# ========== SemiAnalysis aiperf agentic benchmark tool ==========
# Install the SemiAnalysis fork pinned to the commit that supports the SA
# agentic datasets (semianalysis_cc_traces_weka_062126*).
ARG INSTALL_SA_AIPERF=1
ARG SA_AIPERF_COMMIT="0d2aa0572ac685943d38c580675c4a61023581d3"
RUN if [ "${INSTALL_SA_AIPERF}" = "1" ]; then \
        echo "========== [ATOM] Install SemiAnalysis aiperf (${SA_AIPERF_COMMIT}) =========="; \
        rm -rf /opt/aiperf && \
        git clone https://github.com/SemiAnalysisAI/aiperf.git /opt/aiperf && \
        cd /opt/aiperf && \
        git checkout "${SA_AIPERF_COMMIT}" && \
        "${VENV_PYTHON}" -m pip install -e . && \
        "${VENV_PYTHON}" -m pip install --no-cache-dir "transformers==5.2.0" && \
        "${VENV_PYTHON}" -c "import transformers; print(f'transformers.__version__ = {transformers.__version__}')" && \
        "${VENV_PYTHON}" -m pip show aiperf || true && \
        command -v aiperf && aiperf --help >/dev/null; \
    else \
        echo "========== Skipped SemiAnalysis aiperf (INSTALL_SA_AIPERF=0) =========="; \
    fi

CMD ["/bin/bash"]
