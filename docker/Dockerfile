# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG GPU_ARCH="gfx942"

FROM $BASE_IMAGE

ARG GPU_ARCH
ENV GPU_ARCH_LIST=$GPU_ARCH
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"
#ARG AITER_REPO="https://github.com/ROCm/aiter.git"
#ARG AITER_COMMIT="HEAD"

# Install Aiter
RUN pip uninstall -y aiter || true
#RUN git clone $AITER_REPO /aiter-test && \
#    cd /aiter-test && \
#    pip install -r requirements.txt && \
#    git checkout $AITER_COMMIT && \
#    git submodule sync && git submodule update --init --recursive && \
#    PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python3 setup.py develop

RUN cd /tmp && rm -rf aiter aiter-test && \
    git clone --recursive https://github.com/ROCm/aiter.git && \
    cd aiter && git reset --hard v0.1.9 && \
    pip uninstall -y amd-aiter && \
    pip install --upgrade pip setuptools wheel && \
    pip install ninja && \
    pip install pybind11==3.0.1 && \
    time pip install psutil && \
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
