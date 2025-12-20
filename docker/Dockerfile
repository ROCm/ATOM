# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"

FROM $BASE_IMAGE

ARG GPU_ARCH=gfx950
ENV GPU_ARCH_LIST=${GPU_ARCH%-*}
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="207594140d0e62ed76110666ee387099c59f3ef4"
#ARG AITER_REPO="https://github.com/ROCm/aiter.git"
#ARG AITER_COMMIT="db619aa38ecf6a92812223816782260f756b7367"

# Install Aiter
RUN pip uninstall -y aiter || true
RUN cd /tmp && \
    rm -rf aiter && \
    git clone --recursive -b atom_quick_fix_srok https://github.com/ROCm/aiter.git && \
    cd aiter && \
    pip uninstall -y aiter || true && \
    pip install --upgrade pip setuptools wheel && \
    pip install ninja && \
    pip install pybind11==3.0.1 && \
    pip install psutil && \
    AITER_REBUILD=2 PREBUILD_KERNELS=3 GPU_ARCHS="gfx942;gfx950" python3 setup.py install
RUN pip show amd-aiter || true

# Install ATOM
RUN pip uninstall -y atom || true
RUN git clone $ATOM_REPO /ATOM && \
    cd /ATOM && \
    git checkout $ATOM_COMMIT && \
    pip install -e .
RUN pip show atom || true

CMD ["/bin/bash"]
