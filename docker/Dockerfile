# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG GPU_ARCH="gfx942"

FROM $BASE_IMAGE

ARG GPU_ARCH
ENV GPU_ARCH_LIST=$GPU_ARCH
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="b5ade227732c8f7aca9238ff6b448381bdc6508c"
ARG TRITON_REPO="https://github.com/ROCm/triton.git"
ARG TRITON_COMMIT="release/internal/3.5.x"

# Update Triton
RUN pip uninstall -y triton
RUN cd /tmp && \
  git clone --depth=1 --branch ${TRITON_COMMIT} ${TRITON_REPO} triton-test && \
  cd triton-test && \
  pip install -r python/requirements.txt && \
  pip install filecheck && \
  MAX_JOBS=64 pip --retries=10 --default-timeout=60 install . \
  cd triton_kernels && \
  pip install -e . 
RUN pip show triton || true

# Install Aiter
RUN pip uninstall -y aiter || true
RUN cd /tmp && rm -rf aiter aiter-test && \
    git clone --recursive ${AITER_REPO} && \
    cd aiter && git reset --hard ${AITER_COMMIT} && \
    pip uninstall -y amd-aiter && \
    pip install --upgrade pip setuptools wheel && \
    pip install ninja && \
    pip install pybind11==3.0.1 && \
    pip install psutil && \
    PREBUILD_KERNELS=2 GPU_ARCHS="gfx942" python3 setup.py install
RUN pip show amd-aiter || true

# Install ATOM
RUN pip uninstall -y atom || true
RUN git clone $ATOM_REPO /ATOM && \
    cd /ATOM && \
    git checkout $ATOM_COMMIT && \
    pip install -e .
RUN pip show atom || true

CMD ["/bin/bash"]
