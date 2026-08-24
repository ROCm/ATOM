# Keep the release reproducible on the ROCm 7.2.4 / PyTorch 2.10 stack.
# The digest prevents this historical tag from being moved underneath us.
ARG BASE_IMAGE="rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
ARG GPU_ARCH="gfx942;gfx950"

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
ENV GPU_ARCH_LIST=$GPU_ARCH
ENV PYTORCH_ROCM_ARCH=$GPU_ARCH

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
RUN ROCM_SDK_LIB_DIRS="$(python -c \
        'import glob, os; print("\n".join(sorted({os.path.dirname(p) for p in glob.glob("/opt/venv/lib/python*/site-packages/_rocm_sdk*/lib/*.so*")})))')" && \
    if [ -n "${ROCM_SDK_LIB_DIRS}" ]; then \
        printf '%s\n' "${ROCM_SDK_LIB_DIRS}" \
            > /etc/ld.so.conf.d/rocm-python-sdk.conf; \
        ldconfig; \
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
# Upstream Mooncake (includes HIP dma-buf MR via ibv_reg_dmabuf_mr, PR #2225).
# Pin a SHA: floating `main` would silently drop dma-buf / ionic behavior.
# Jasen2201 `fix/ionic-mr-and-qp-resource-fixes` only has CUDA dma-buf; HIP
# still falls back to ibv_reg_mr (EINVAL 22 on ionic GPU VAs).
ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
ARG MOONCAKE_COMMIT="bfca1ce2af8419c50dc8d464820a95d97d43c930"
ARG USE_HIP_DMABUF=ON
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

# [MC 3/4] CMake build with HIP + HIP dma-buf MR (ibv_reg_dmabuf_mr)
# USE_HIP_DMABUF must compile into rdma_transport (rdma_context.cpp). Without
# hsa-runtime64, CMake silently disables dma-buf and GPU MRs stay on ibv_reg_mr.
# ATOM only consumes the TransferEngine (`mooncake.engine`), so Mooncake Store
# and its Rust bindings stay off: they add build surface (cachelib, cargo) that
# nothing here loads. WITH_STORE_RUST=ON is a hard error without WITH_STORE.
# HIP hipify copies tests/*.cpp into the build tree but not headers such as
# rdma_test_peers.h, so BUILD_UNIT_TESTS=ON fails. Upstream ROCm CI also sets
# BUILD_UNIT_TESTS=OFF; examples are unused in this image.
RUN if [ "${INSTALL_MOONCAKE}" = "1" ]; then \
        echo "========== [MC 3/4] Build and install Mooncake (USE_HIP=ON USE_HIP_DMABUF=${USE_HIP_DMABUF}) =========="; \
        HSA_PREFIXS="/opt/rocm"; \
        for cfg in /opt/rocm/lib/cmake/hsa-runtime64/hsa-runtime64Config.cmake \
                   /opt/rocm/lib64/cmake/hsa-runtime64/hsa-runtime64Config.cmake; do \
            if [ -f "${cfg}" ]; then HSA_PREFIXS="$(dirname "$(dirname "$(dirname "${cfg}")")"):${HSA_PREFIXS}"; fi; \
        done; \
        mkdir -p /app/mooncake/build && cd /app/mooncake/build \
        && cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=${USE_HIP_DMABUF} -DUSE_ETCD=ON \
             -DWITH_TE=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
             -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF \
             -DCMAKE_PREFIX_PATH="${HSA_PREFIXS}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" \
             > /tmp/mooncake-cmake.log 2>&1 \
        || { cat /tmp/mooncake-cmake.log; exit 1; } \
        && cat /tmp/mooncake-cmake.log \
        && if [ "${USE_HIP_DMABUF}" = "ON" ]; then \
             grep -q "HIP dmabuf MR registration enabled" /tmp/mooncake-cmake.log \
               || { echo "ERROR: HIP dma-buf was not enabled (hsa-runtime64 missing?)"; \
                    grep -E "HIP dmabuf|hsa-runtime64" /tmp/mooncake-cmake.log || true; \
                    exit 1; }; \
           fi \
        && { make -j$(nproc) > /tmp/mooncake-make.log 2>&1 \
             || { echo "ERROR: Mooncake build failed. Compiler diagnostics:"; \
                  grep -nE "error:|fatal error|undefined reference|Error [0-9]" \
                       /tmp/mooncake-make.log | head -n 80; \
                  echo "--- tail of build log ---"; \
                  tail -n 120 /tmp/mooncake-make.log; \
                  exit 1; }; } \
        && make install \
        && ldconfig \
        && ( grep -a -R -l "ibv_reg_dmabuf_mr" /usr/local/lib /opt/venv 2>/dev/null | head -n 1 \
             || { echo "ERROR: installed Mooncake libs have no ibv_reg_dmabuf_mr"; exit 1; } ) \
        && echo "--- Clean up build artifacts ---" \
        && rm -rf /app/mooncake/build /app/mooncake/.git; \
    fi

# Runtime: do not set MOONCAKE_DISABLE_HIP_DMABUF (any non-0 forces ibv_reg_mr).
ENV MOONCAKE_DISABLE_HIP_DMABUF=0

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
