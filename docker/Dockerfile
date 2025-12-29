# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG GPU_ARCH="gfx942;gfx950"

FROM $BASE_IMAGE

ARG GPU_ARCH
ENV GPU_ARCH_LIST=$GPU_ARCH
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="HEAD"

# Update Triton
RUN pip show triton || true
RUN pip uninstall -y triton
RUN git clone --depth=1 --branch release/internal/3.5.x https://github.com/triton-lang/triton.git /triton-test && \
  cd /triton-test && \
  pip install -r python/requirements.txt && \
  pip install filecheck && \
  MAX_JOBS=64 pip --retries=10 --default-timeout=60 install .
RUN pip show triton || true

# Install Aiter
RUN pip uninstall -y aiter || true
RUN git clone $AITER_REPO /aiter-test && \
    cd /aiter-test && \
    pip install -r requirements.txt && \
    git checkout $AITER_COMMIT && \
    git submodule sync && git submodule update --init --recursive && \
    PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python3 setup.py develop
RUN pip show amd-aiter || true

# Install ATOM
RUN pip uninstall -y atom || true
RUN git clone $ATOM_REPO /ATOM && \
    cd /ATOM && \
    git checkout $ATOM_COMMIT && \
    pip install -e .
RUN pip show atom || true

CMD ["/bin/bash"]
