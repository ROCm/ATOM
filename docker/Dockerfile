# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG GPU_ARCH="gfx942;gfx950"

FROM $BASE_IMAGE

ARG GPU_ARCH
ENV GPU_ARCH_LIST=$GPU_ARCH
ENV PYTORCH_ROCM_ARCH=$GPU_ARCH
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="f9b010e4f4f09558f9e74319e77bdc1c70049893"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="HEAD"
ARG PREBUILD_KERNELS=1
ARG MAX_JOBS

RUN pip install --upgrade pip
RUN pip install lm-eval[api]
RUN pip show lm-eval || true

# Install MORI (Modular RDMA Interface)
# https://github.com/ROCm/mori
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

RUN git clone https://github.com/ROCm/mori.git /app/mori && \
    cd /app/mori && \
    pip install -r requirements-build.txt && \
    git submodule update --init --recursive && \
    python setup.py install
RUN pip show mori || true


# Update RCCL
ARG RCCL_REPO="https://github.com/ROCm/rccl.git"
ARG RCCL_BRANCH="29e1567b95e28823b0beb1a988adc587bfab5b4f"
ARG RCCL_SRC_DIR="/app/rccl/"
RUN pip install cmake
RUN git clone "$RCCL_REPO" $RCCL_SRC_DIR \
    && cd $RCCL_SRC_DIR \
    && git checkout "$RCCL_BRANCH" \
    && ./install.sh -p --amdgpu_targets=$GPU_ARCH_LIST
RUN DEBIAN_FRONTEND=noninteractive dpkg -i --force-all $RCCL_SRC_DIR/build/release/*.deb && rm -rf $RCCL_SRC_DIR/build

# Update Triton
RUN pip show triton || true
RUN pip uninstall -y triton
RUN git clone --depth=1 --branch release/internal/3.5.x https://github.com/ROCm/triton.git /triton-test && \
  cd /triton-test && \
  pip install -r python/requirements.txt && \
  pip install filecheck && \
  MAX_JOBS=64 pip --retries=10 --default-timeout=60 install .
RUN pip show triton || true

# Install Aiter
RUN mkdir -p /app
RUN pip uninstall -y aiter || true
RUN git clone $AITER_REPO /app/aiter-test && \
    cd /app/aiter-test && \
    pip install -r requirements.txt && \
    git checkout $AITER_COMMIT && \
    git submodule sync && git submodule update --init --recursive && \
    MAX_JOBS=$MAX_JOBS PREBUILD_KERNELS=$PREBUILD_KERNELS GPU_ARCHS=$GPU_ARCH_LIST python3 setup.py develop
RUN pip show amd-aiter || true


# Install ATOM
RUN pip uninstall -y atom || true
RUN git clone $ATOM_REPO /app/ATOM && \
    cd /app/ATOM && \
    git checkout $ATOM_COMMIT && \
    pip install -e .
RUN pip show atom || true

CMD ["/bin/bash"]
