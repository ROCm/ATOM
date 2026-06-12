# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG OOT_BASE_IMAGE="rocm/atom-dev:latest"
ARG SGLANG_BASE_IMAGE="rocm/atom-dev:latest"
ARG GPU_ARCH="gfx942;gfx950"

# OOT image extends an ATOM base image that already contains atom/aiter/mori.
FROM ${OOT_BASE_IMAGE} AS atom_oot

ARG MAX_JOBS
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG VLLM_REPO="https://github.com/vllm-project/vllm.git"
ARG VLLM_COMMIT="0b3ba88f165976e77ca5e6a7a3f5bba4562b80af"
ARG INSTALL_LM_EVAL=1
ARG INSTALL_FASTSAFETENSORS=1
# Let PR OOT CI verify whether a pulled prebuilt image still matches this vLLM commit
LABEL com.rocm.atom.vllm_commit="${VLLM_COMMIT}"

ENV PATH="/opt/venv/bin:${PATH}"
ENV MAX_JOBS=${MAX_JOBS}
ENV VLLM_TARGET_DEVICE=rocm
ENV CMAKE_MAKE_PROGRAM=/usr/local/bin/ninja

RUN echo "========== [OOT 1/7] Prepare build tools ==========" && \
    apt-get update && \
    apt --fix-broken install -y && \
    apt-get install -y --no-install-recommends ca-certificates jq ninja-build vim && \
    "${VENV_PYTHON}" -m pip install --upgrade cmake && \
    cmake --version && \
    mkdir -p /usr/local/bin && \
    ln -sf "$(command -v ninja)" /usr/local/bin/ninja && \
    /usr/local/bin/ninja --version && \
    rm -rf /var/lib/apt/lists/* && \
    TORCH_LIB=$("${VENV_PYTHON}" -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))") && \
    echo "${TORCH_LIB}" > /etc/ld.so.conf.d/torch.conf && \
    ldconfig

# Keep OOT aligned with the Triton that the ATOM base image ships.
# vLLM/OOT installs can perturb Triton; back up everything and restore after.
RUN echo "========== [OOT 2/7] Verify base packages (atom/aiter/mori) ==========" && \
    "${VENV_PYTHON}" -m pip show atom || true && \
    "${VENV_PYTHON}" -m pip show amd-aiter || true && \
    "${VENV_PYTHON}" -m pip show amd-mori-nightly || true && \
    echo "========== [OOT 2/7] Back up base image triton ==========" && \
    SITE_PACKAGES=$("${VENV_PYTHON}" -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    BASE_TRITON_VERSION="$("${VENV_PYTHON}" -c "import triton; print(triton.__version__)")" && \
    mkdir -p /tmp/triton-base-backup && \
    cp -a "${SITE_PACKAGES}/triton" /tmp/triton-base-backup/ && \
    for f in "${SITE_PACKAGES}"/triton-*.dist-info; do \
      [ -d "$f" ] || continue; \
      cp -a "$f" /tmp/triton-base-backup/; \
    done && \
    echo "Base image triton backed up: import_version=${BASE_TRITON_VERSION}" && \
    ls /tmp/triton-base-backup/

RUN echo "========== [OOT 3/7] Clone vLLM ==========" && \
    rm -rf /app/vllm && \
    git clone "${VLLM_REPO}" /app/vllm && \
    cd /app/vllm && \
    git checkout "${VLLM_COMMIT}" && \
    git submodule update --init --recursive && \
    echo "vLLM commit:" && \
    git rev-parse HEAD

RUN echo "========== [OOT 4/7] Install vLLM ROCm build dependencies ==========" && \
    cd /app/vllm && \
    "${VENV_PYTHON}" -m pip install --upgrade pip && \
    # keeps the base image's ATOM-installed transformers
    sed -i -e '/^transformers[[:space:]]/d' requirements/common.txt && \
    echo "Removed transformers constraint from cloned vLLM requirements/common.txt to preserve base image transformers" && \
    ! grep '^transformers' requirements/common.txt && \
    sed -i -e '/xgrammar/d' -e '/compressed-tensors/d' requirements/common.txt && \
    "${VENV_PYTHON}" -m pip install --no-deps xgrammar==0.1.29 compressed-tensors==0.13.0 loguru && \
    sed -i -e '/peft/d' -e '/tensorizer/d' -e '/runai/d' -e '/timm/d' -e '/tilelang/d' requirements/rocm.txt && \
    "${VENV_PYTHON}" -m pip install --no-deps peft tensorizer==2.10.1 runai-model-streamer[s3,gcs]==0.15.3 timm>=1.0.17 tilelang==0.1.10 && \
    "${VENV_PYTHON}" -m pip install -r requirements/rocm.txt

RUN echo "========== [OOT 5/7] Build and install amd-smi wheel ==========" && \
    cd /opt/rocm/share/amd_smi && \
    pip wheel . --wheel-dir=dist && \
    pip install dist/*.whl

RUN echo "========== [OOT 6/7] Build vLLM wheel ==========" && \
    cd /app/vllm && \
    VLLM_TARGET_DEVICE=rocm "${VENV_PYTHON}" setup.py clean --all && \
    MAX_JOBS="${MAX_JOBS}" VLLM_TARGET_DEVICE=rocm "${VENV_PYTHON}" setup.py bdist_wheel --dist-dir=/tmp/vllm-wheels && \
    ls -lh /tmp/vllm-wheels

RUN echo "========== [OOT 7/7] Install vLLM runtime dependencies ==========" && \
    cd /app/vllm && \
    "${VENV_PYTHON}" -m pip uninstall -y vllm || true && \
    "${VENV_PYTHON}" -m pip install /tmp/vllm-wheels/*.whl && \
    "${VENV_PYTHON}" -m pip install \
      uvloop \
      "botocore>=1.43.7,<1.44.0" \
      "s3transfer>=0.17.0,<0.18.0" && \
    if [ "${INSTALL_LM_EVAL}" = "1" ]; then "${VENV_PYTHON}" -m pip install "lm-eval[api]"; else echo "Skip lm-eval install"; fi && \
    if [ "${INSTALL_FASTSAFETENSORS}" = "1" ]; then "${VENV_PYTHON}" -m pip install "git+https://github.com/foundation-model-stack/fastsafetensors.git"; else echo "Skip fastsafetensors install"; fi && \
    "${VENV_PYTHON}" -c "import boto3, botocore, s3transfer; print(f'boto3: {boto3.__version__}'); print(f'botocore: {botocore.__version__}'); print(f's3transfer: {s3transfer.__version__}')" && \
    "${VENV_PYTHON}" -c "import glob, os, torch; print(f'torch.version.hip: {torch.version.hip}'); print(f'torch.version.cuda: {torch.version.cuda}'); torch_lib_dir=os.path.join(os.path.dirname(torch.__file__), 'lib'); print(f'torch lib dir: {torch_lib_dir}'); print(f'libtorch_hip candidates: {glob.glob(os.path.join(torch_lib_dir, \"libtorch_hip.so*\"))}'); assert torch.version.hip is not None, 'Torch is not ROCm build (torch.version.hip is None).'" && \
    "${VENV_PYTHON}" -m pip show vllm torch triton torchvision torchaudio amdsmi amd-aiter atom amd-mori-nightly || true

RUN echo "========== [VLLM-ATOM] Validate vision/audio wheels ==========" && \
    "${VENV_PYTHON}" -c "import torch, torchvision, torchaudio; from torchvision.transforms import InterpolationMode; from transformers.models.auto.image_processing_auto import get_image_processor_config; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'InterpolationMode: {InterpolationMode.BILINEAR}'); print(f'get_image_processor_config: {get_image_processor_config.__name__}')"

# Restore the exact base-image Triton after all OOT installs finish.
RUN echo "========== [OOT] Restore base image triton ==========" && \
    SITE_PACKAGES=$("${VENV_PYTHON}" -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    "${VENV_PYTHON}" -m pip uninstall -y triton 2>/dev/null || true && \
    rm -rf "${SITE_PACKAGES}/triton" \
           "${SITE_PACKAGES}"/triton-*.dist-info && \
    cp -a /tmp/triton-base-backup/triton "${SITE_PACKAGES}/" && \
    for f in /tmp/triton-base-backup/triton-*.dist-info; do \
      [ -d "$f" ] || continue; \
      cp -a "$f" "${SITE_PACKAGES}/"; \
    done && \
    rm -rf /tmp/triton-base-backup && \
    "${VENV_PYTHON}" -c "import triton; print(f'triton.__version__ = {triton.__version__}')" && \
    "${VENV_PYTHON}" -m pip show triton || true


RUN echo "========== [vLLM-ATOM] Final transformers version ==========" && \
    "${VENV_PYTHON}" -c "import transformers; print(f'transformers.__version__ = {transformers.__version__}')" && \
    "${VENV_PYTHON}" -m pip show transformers || true

CMD ["/bin/bash"]

# SGLang image extends an ATOM base image and layers on an sglang checkout.
FROM ${SGLANG_BASE_IMAGE} AS atom_sglang

ARG GPU_ARCH
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG SGLANG_REPO="https://github.com/sgl-project/sglang.git"
ARG SGLANG_REF="v0.5.12"
ARG SGLANG_TRITON_VERSION="3.6.0"
LABEL com.rocm.atom.sglang_ref="${SGLANG_REF}"

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/app/sglang/python:/app/ATOM:${PYTHONPATH}"

RUN echo "========== [SGLANG-ATOM 0/6] Check Aiter/FlyDSL versions before SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom amd-mori-nightly amd-aiter flydsl || true

RUN echo "========== [SGLANG-ATOM 1/6] Clone SGLang ==========" && \
    rm -rf /app/sglang && \
    git clone "${SGLANG_REPO}" /app/sglang && \
    cd /app/sglang && \
    git checkout "${SGLANG_REF}" && \
    git submodule update --init --recursive && \
    echo "sglang ref:" && \
    git rev-parse HEAD

RUN echo "========== [SGLANG-ATOM 2/6] Build sglang kernel ==========" && \
    "${VENV_PYTHON}" -m pip uninstall -y sgl-kernel sglang-kernel sglang || true && \
    "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel && \
    DETECTED_AMDGPU_TARGET="$("${VENV_PYTHON}" -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] if torch.cuda.is_available() else '')" 2>/dev/null || true)" && \
    if [ -n "${DETECTED_AMDGPU_TARGET}" ]; then \
      FINAL_AMDGPU_TARGET="${DETECTED_AMDGPU_TARGET}"; \
      echo "Detected AMDGPU_TARGET=${FINAL_AMDGPU_TARGET} from build GPU"; \
    else \
      FINAL_AMDGPU_TARGET="$(printf '%s' "${GPU_ARCH}" | awk -F';' '{print $NF}' | xargs)"; \
      test -n "${FINAL_AMDGPU_TARGET}"; \
      echo "GPU not detectable during build; fallback AMDGPU_TARGET=${FINAL_AMDGPU_TARGET} from GPU_ARCH=${GPU_ARCH}"; \
    fi && \
    cd /app/sglang/sgl-kernel && \
    AMDGPU_TARGET="${FINAL_AMDGPU_TARGET}" "${VENV_PYTHON}" setup_rocm.py install && \
    "${VENV_PYTHON}" -m pip show sglang-kernel || true

RUN echo "========== [SGLANG-ATOM 3/6] Install SGLang dependencies ==========" && \
    cd /app/sglang/python && \
    rm -f pyproject.toml && \
    cp pyproject_other.toml pyproject.toml && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir --no-deps -e . && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir tomli && \
    "${VENV_PYTHON}" -c "import tomli; from pathlib import Path; deps = tomli.loads(Path('pyproject_other.toml').read_text())['project']['optional-dependencies']['runtime_common']; blocked_prefixes = ('compressed-tensors', 'outlines==', 'timm==', 'torchao==', 'transformers==', 'xgrammar=='); Path('/tmp/sglang-runtime-common.txt').write_text(''.join(f'{dep}\\n' for dep in deps if dep != 'numpy' and not any(dep.startswith(prefix) for prefix in blocked_prefixes)))" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir \
      -r /tmp/sglang-runtime-common.txt \
      airportsdata \
      cloudpickle==3.1.2 \
      diskcache \
      jsonschema \
      lark \
      loguru \
      nest_asyncio \
      outlines_core==0.1.26 \
      pycountry \
      pybase64 \
      referencing \
      safetensors && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
      compressed-tensors==0.13.0 \
      outlines==0.1.11 \
      petit_kernel==0.0.2 \
      timm==1.0.16 \
      torchao==0.9.0 \
      wave-lang==3.8.2 \
      xgrammar==0.1.27 && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir \
      "cython>=0.29.36,<3.0" \
      "apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@37d0485b2058885bf4e7a486f7d7b2174a8ac1ce" \
      "z3-solver==4.15.4.0" && \
    rm -f /tmp/sglang-runtime-common.txt && \
    "${VENV_PYTHON}" -m pip show sglang torch triton transformers IPython orjson pybase64 petit-kernel wave-lang xgrammar outlines apache-tvm-ffi || true

RUN echo "========== [SGLANG-ATOM 4/6] Validate vision/audio wheels ==========" && \
    "${VENV_PYTHON}" -m sglang.launch_server --help >/dev/null && \
    "${VENV_PYTHON}" -c "import os, torch, torchvision, torchaudio, sglang, triton, transformers; from torchvision.io import decode_jpeg; assert torch.version.hip is not None, 'Torch is not ROCm build (torch.version.hip is None).'; print(f'torch: {torch.__version__}'); print(f'triton: {triton.__version__}'); print(f'transformers: {transformers.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'decode_jpeg: {decode_jpeg.__name__}'); print(f'sglang imported from: {sglang.__file__}'); print(f'PYTHONPATH={os.environ.get(\"PYTHONPATH\", \"\")}')" && \
    echo "Validated sglang launch_server entrypoint"

# Only the derived SGLang image needs the newer Triton compiler. SGLang serving
# for DeepSeek-R1 hit `ConvertTritonAMDGPUToLLVM` failures with the Triton that
# ships in the shared ATOM base image, while Triton 3.6.0 fixes that compiler
# path for this flow. Install Triton after the torchvision/torchaudio wheel pin
# so pip does not downgrade it back to the torch-pinned 3.5.1 dependency. Keep
# this override local to `atom_sglang` so the OOT/atom_release flow preserves
# its prior Triton behavior.
RUN echo "========== [SGLANG-ATOM 5/6] Install validated Triton ==========" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir "triton==${SGLANG_TRITON_VERSION}" && \
    "${VENV_PYTHON}" -m pip show triton

RUN echo "========== [SGLANG-ATOM 5.5/6] Pin smg-grpc-servicer ==========" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir "smg-grpc-servicer==0.5.2" && \
    "${VENV_PYTHON}" -m pip show smg-grpc-proto smg-grpc-servicer

RUN echo "========== [SGLANG-ATOM 6/6] Check Aiter/FlyDSL versions after SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom amd-mori-nightly amd-aiter flydsl || true

CMD ["/bin/bash"]

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

RUN pip install --upgrade pip && \
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

CMD ["/bin/bash"]
