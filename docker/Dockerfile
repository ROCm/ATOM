# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG OOT_BASE_IMAGE="rocm/atom-dev:latest"
ARG GPU_ARCH="gfx942;gfx950"

# OOT image extends an ATOM base image that already contains atom/aiter/mori.
FROM ${OOT_BASE_IMAGE} AS atom_oot

ARG MAX_JOBS
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG VLLM_REPO="https://github.com/vllm-project/vllm.git"
ARG VLLM_COMMIT="b31e9326a7d9394aab8c767f8ebe225c65594b60"
ARG INSTALL_LM_EVAL=1
ARG INSTALL_FASTSAFETENSORS=1

ENV PATH="/opt/venv/bin:${PATH}"
ENV MAX_JOBS=${MAX_JOBS}
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
ENV VLLM_TARGET_DEVICE=rocm
ENV CMAKE_MAKE_PROGRAM=/usr/local/bin/ninja

RUN echo "========== [OOT 1/7] Prepare build tools ==========" && \
    apt-get update && \
    apt --fix-broken install -y && \
    apt-get install -y --no-install-recommends ca-certificates jq ninja-build vim && \
    mkdir -p /usr/local/bin && \
    ln -sf "$(command -v ninja)" /usr/local/bin/ninja && \
    /usr/local/bin/ninja --version && \
    rm -rf /var/lib/apt/lists/*

RUN echo "========== [OOT 2/7] Verify base packages (atom/aiter/mori) ==========" && \
    "${VENV_PYTHON}" -m pip show atom || true && \
    "${VENV_PYTHON}" -m pip show amd-aiter || true && \
    "${VENV_PYTHON}" -m pip show mori || true

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
    sed -i -e '/xgrammar/d' -e '/compressed-tensors/d' requirements/common.txt && \
    "${VENV_PYTHON}" -m pip install --no-deps xgrammar==0.1.29 compressed-tensors==0.13.0 loguru && \
    sed -i -e '/peft/d' -e '/tensorizer/d' -e '/runai/d' -e '/timm/d' requirements/rocm.txt && \
    "${VENV_PYTHON}" -m pip install --no-deps peft tensorizer==2.10.1 runai-model-streamer[s3,gcs]==0.15.3 timm>=1.0.17 && \
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
    "${VENV_PYTHON}" -m pip install uvloop && \
    if [ "${INSTALL_LM_EVAL}" = "1" ]; then "${VENV_PYTHON}" -m pip install "lm-eval[api]"; else echo "Skip lm-eval install"; fi && \
    if [ "${INSTALL_FASTSAFETENSORS}" = "1" ]; then "${VENV_PYTHON}" -m pip install "git+https://github.com/foundation-model-stack/fastsafetensors.git"; else echo "Skip fastsafetensors install"; fi && \
    "${VENV_PYTHON}" -c "import glob, os, torch; print(f'torch.version.hip: {torch.version.hip}'); print(f'torch.version.cuda: {torch.version.cuda}'); torch_lib_dir=os.path.join(os.path.dirname(torch.__file__), 'lib'); print(f'torch lib dir: {torch_lib_dir}'); print(f'libtorch_hip candidates: {glob.glob(os.path.join(torch_lib_dir, \"libtorch_hip.so*\"))}'); assert torch.version.hip is not None, 'Torch is not ROCm build (torch.version.hip is None).'" && \
    "${VENV_PYTHON}" -m pip show vllm torch triton torchvision torchaudio amdsmi amd-aiter atom mori || true

CMD ["/bin/bash"]

FROM ${BASE_IMAGE} AS atom_image

ARG GPU_ARCH
ENV GPU_ARCH_LIST=$GPU_ARCH
ENV PYTORCH_ROCM_ARCH=$GPU_ARCH
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="HEAD"
ARG MORI_COMMIT="b0dce4beebeb1f26c784eee17d5fd9785ee9447f"
ARG PREBUILD_KERNELS=1
ARG MAX_JOBS

RUN pip install --upgrade pip

RUN pip install lm-eval[api]

RUN pip show lm-eval || true

# Install MORI (Modular RDMA Interface)
# https://github.com/ROCm/mori
RUN apt-get update && apt --fix-broken install -y

RUN apt-get update && apt-get install -y \
    git \
    cython3 \
    ibverbs-utils \
    openmpi-bin \
    libopenmpi-dev \
    libpci-dev \
    cmake \
    libdw1 \
    locales

RUN rm -rf /app/mori && \
    git clone https://github.com/ROCm/mori.git /app/mori && \
    cd /app/mori && \
    git checkout $MORI_COMMIT && \
    pip install -r requirements-build.txt && \
    git submodule update --init --recursive && \
    python setup.py install

RUN pip show mori || true


# Update RCCL
ARG RCCL_REPO="https://github.com/ROCm/rccl.git"
ARG RCCL_BRANCH="29e1567b95e28823b0beb1a988adc587bfab5b4f"
ARG RCCL_SRC_DIR="/app/rccl/"
RUN pip install cmake

RUN rm -rf $RCCL_SRC_DIR && \
    git clone "$RCCL_REPO" $RCCL_SRC_DIR \
    && cd $RCCL_SRC_DIR \
    && git checkout "$RCCL_BRANCH" \
    && ./install.sh -p --amdgpu_targets=$GPU_ARCH_LIST

RUN DEBIAN_FRONTEND=noninteractive dpkg -i --force-all $RCCL_SRC_DIR/build/release/*.deb && rm -rf $RCCL_SRC_DIR/build

# Update Triton
RUN pip show triton || true

RUN pip uninstall -y triton

RUN rm -rf /triton-test && \
    git clone --depth=1 --branch release/internal/3.5.x https://github.com/ROCm/triton.git /triton-test && \
    cd /triton-test && \
    pip install -r python/requirements.txt && \
    pip install filecheck && \
    MAX_JOBS=64 pip --retries=10 --default-timeout=60 install .

RUN pip show triton || true

# Install Aiter
RUN mkdir -p /app

RUN pip uninstall -y aiter || true

RUN rm -rf /app/aiter-test && \
    git clone $AITER_REPO /app/aiter-test && \
    cd /app/aiter-test && \
    pip install -r requirements.txt && \
    git checkout $AITER_COMMIT && \
    git submodule sync && git submodule update --init --recursive && \
    MAX_JOBS=$MAX_JOBS PREBUILD_KERNELS=$PREBUILD_KERNELS GPU_ARCHS=$GPU_ARCH_LIST python3 setup.py develop

RUN pip show amd-aiter || true


# Install ATOM
RUN pip uninstall -y atom || true

RUN rm -rf /app/ATOM && \
    git clone $ATOM_REPO /app/ATOM && \
    cd /app/ATOM && \
    git checkout $ATOM_COMMIT && \
    pip install -e .

RUN pip show atom || true

CMD ["/bin/bash"]
