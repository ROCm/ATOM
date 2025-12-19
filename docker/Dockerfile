# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"

FROM $BASE_IMAGE

ARG GPU_ARCH=gfx950
ENV GPU_ARCH_LIST=${GPU_ARCH%-*}
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="207594140d0e62ed76110666ee387099c59f3ef4"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="f2d16272cc618c404d83a4355ee17ba81450e9da"

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
