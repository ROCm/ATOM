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
ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG TRITON_COMMIT="v3.5.1"

# Update Triton
RUN pip uninstall -y triton || true
RUN git clone $TRITON_REPO --depth=1 --branch $TRITON_COMMIT /triton-test && \
    cd /triton-test && \
    pip install -r python/requirements.txt && \
    pip install filecheck && \
    MAX_JOBS=64 pip --retries=10 --default-timeout=60 install .
RUN pip show triton || true

# Install Aiter
RUN pip uninstall -y aiter || true
RUN git clone $AITER_REPO /aiter-test && \
    cd /aiter-test && \
    git checkout $AITER_COMMIT && \
    git submodule sync && git submodule update --init --recursive && \
    pip install -r requirements.txt && \
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
