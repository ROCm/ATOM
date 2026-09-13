ARG SGLANG_BASE_IMAGE="rocm/atom-dev:latest"
ARG GPU_ARCH="gfx942;gfx950"

# Build TileLang in an isolated stage. The ATOM runtime intentionally carries a
# custom RCCL package, which leaves the rocm-hip meta-package unsatisfied from
# apt's perspective. Repairing apt is safe here and cannot replace runtime RCCL.
FROM ${SGLANG_BASE_IMAGE} AS build_tilelang
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG TILELANG_REPO="https://github.com/tile-ai/tilelang.git"
ARG TILELANG_COMMIT="a55a82302bf7f3c5af635b5c9146f728185cc900"
ARG TILELANG_BUILD_JOBS=32
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get --fix-broken install -y && \
    apt-get install -y --no-install-recommends \
      curl gnupg libgtest-dev libgmock-dev libgflags-dev libsqlite3-dev \
      libtinfo-dev zlib1g-dev libedit-dev libxml2-dev && \
    rm -rf /var/lib/apt/lists/* && \
    cmake -S /usr/src/googletest -B /tmp/build-gtest \
      -DBUILD_GTEST=ON -DBUILD_GMOCK=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/build-gtest -j"${TILELANG_BUILD_JOBS}" && \
    cp -v /tmp/build-gtest/lib/*.a /usr/lib/x86_64-linux-gnu/ && \
    rm -rf /tmp/build-gtest

RUN LLVM_CONFIG="" && \
    for candidate in \
      /opt/rocm/llvm/bin/llvm-config \
      /opt/rocm/llvm-*/bin/llvm-config \
      /opt/rocm-*/llvm*/bin/llvm-config; do \
      if [ -x "${candidate}" ]; then LLVM_CONFIG="${candidate}"; break; fi; \
    done && \
    if [ -z "${LLVM_CONFIG}" ]; then \
      install -d -m 0755 /etc/apt/keyrings && \
      curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg && \
      echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-18 main" \
        > /etc/apt/sources.list.d/llvm.list && \
      apt-get update && \
      apt-get install -y --no-install-recommends llvm-18 && \
      rm -rf /var/lib/apt/lists/*; \
    fi && \
    if [ -z "${LLVM_CONFIG}" ]; then LLVM_CONFIG="$(command -v llvm-config-18)"; fi && \
    test -x "${LLVM_CONFIG}"

RUN \
    "${VENV_PYTHON}" -m pip install --upgrade \
      "setuptools>=77.0.3,<80" wheel "cmake==4.3.4" ninja scikit-build-core \
      "cython>=0.29.36,<3.0" \
      "apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@37d0485b2058885bf4e7a486f7d7b2174a8ac1ce" \
      "z3-solver==4.15.4.0"

RUN LLVM_CONFIG="" && \
    for candidate in \
      /opt/rocm/llvm/bin/llvm-config \
      /opt/rocm/llvm-*/bin/llvm-config \
      /opt/rocm-*/llvm*/bin/llvm-config; do \
      if [ -x "${candidate}" ]; then LLVM_CONFIG="${candidate}"; break; fi; \
    done && \
    if [ -z "${LLVM_CONFIG}" ]; then LLVM_CONFIG="$(command -v llvm-config-18)"; fi && \
    test -x "${LLVM_CONFIG}" && \
    git clone --recursive "${TILELANG_REPO}" /opt/tilelang && \
    cd /opt/tilelang && \
    git checkout --detach "${TILELANG_COMMIT}" && \
    git submodule update --init --recursive && \
    CMAKE_BUILD_PARALLEL_LEVEL="${TILELANG_BUILD_JOBS}" \
    CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm -DLLVM_CONFIG=${LLVM_CONFIG} -DSKBUILD_SABI_VERSION=" \
      "${VENV_PYTHON}" -m pip wheel . --wheel-dir /tmp/tilelang-wheel \
        --no-build-isolation --no-deps

# SGLang image extends an ATOM base image and layers on an sglang checkout.
FROM ${SGLANG_BASE_IMAGE} AS atom_sglang

ARG GPU_ARCH
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG SGLANG_REPO="https://github.com/sgl-project/sglang.git"
ARG SGLANG_REF="v0.5.17"
LABEL com.rocm.atom.sglang_ref="${SGLANG_REF}"

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/app/sglang/python:/app/ATOM:${PYTHONPATH}"

RUN echo "========== [SGLANG-ATOM 0/6] Check Aiter/FlyDSL/Triton versions before SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom amd-mori-nightly amd-aiter flydsl triton || true && \
    echo "========== [SGLANG-ATOM 0/6] Back up base image Triton ==========" && \
    SITE_PACKAGES=$("${VENV_PYTHON}" -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    BASE_TRITON_VERSION="$("${VENV_PYTHON}" -c "import triton; print(triton.__version__)")" && \
    mkdir -p /tmp/triton-base-backup && \
    cp -a "${SITE_PACKAGES}/triton" /tmp/triton-base-backup/ && \
    for f in "${SITE_PACKAGES}"/triton-*.dist-info; do \
      [ -d "$f" ] || continue; \
      cp -a "$f" /tmp/triton-base-backup/; \
    done && \
    echo "Base image Triton backed up: import_version=${BASE_TRITON_VERSION}" && \
    ls /tmp/triton-base-backup/

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
    cd /app/sglang/python/sglang/kernels/aot && \
    AMDGPU_TARGET="${FINAL_AMDGPU_TARGET}" "${VENV_PYTHON}" setup_rocm.py install && \
    "${VENV_PYTHON}" -m pip show sglang-kernel || true

RUN echo "========== [SGLANG-ATOM 3/6] Install SGLang dependencies ==========" && \
    cd /app/sglang/python && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir /opt/rocm/share/amd_smi && \
    rm -f pyproject.toml && \
    cp pyproject_other.toml pyproject.toml && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir --no-deps -e . && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir tomli && \
    "${VENV_PYTHON}" -c "import tomli; from pathlib import Path; deps = tomli.loads(Path('pyproject_other.toml').read_text())['project']['optional-dependencies']['runtime_common']; blocked_prefixes = ('compressed-tensors', 'outlines==', 'timm==', 'torchao==', 'xgrammar=='); Path('/tmp/sglang-runtime-common.txt').write_text(''.join(f'{dep}\\n' for dep in deps if dep != 'numpy' and not any(dep.startswith(prefix) for prefix in blocked_prefixes)))" && \
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
      compressed-tensors==0.15.0 \
      outlines==0.1.11 \
      petit_kernel==0.0.2 \
      timm==1.0.16 \
      torchao==0.9.0 \
      wave-lang==3.8.2 \
      xgrammar==0.2.1 && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir \
      "cython>=0.29.36,<3.0" \
      "apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@37d0485b2058885bf4e7a486f7d7b2174a8ac1ce" \
      "z3-solver==4.15.4.0" && \
    rm -f /tmp/sglang-runtime-common.txt && \
    "${VENV_PYTHON}" -m pip show sglang torch triton transformers IPython orjson pybase64 petit-kernel wave-lang xgrammar outlines apache-tvm-ffi || true

COPY --from=build_tilelang /tmp/tilelang-wheel /tmp/tilelang-wheel
RUN echo "========== [SGLANG-ATOM 3.25/6] Install ROCm TileLang ==========" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
      /tmp/tilelang-wheel/*.whl && \
    rm -rf /tmp/tilelang-wheel && \
    "${VENV_PYTHON}" -c "import tilelang; print(tilelang.__version__)"

ARG FHT_REPO="https://github.com/jeffdaily/fast-hadamard-transform.git"
ARG FHT_BRANCH="rocm"
ARG FHT_COMMIT="46efb7d776d38638fc39f3c803eaee3dd7016bd1"
RUN echo "========== [SGLANG-ATOM 3.5/6] Build ROCm fast-hadamard-transform ==========" && \
    git clone --branch "${FHT_BRANCH}" "${FHT_REPO}" /app/fast-hadamard-transform && \
    cd /app/fast-hadamard-transform && \
    git checkout --detach "${FHT_COMMIT}" && \
    "${VENV_PYTHON}" setup.py install

# Keep SGLang aligned with the Triton that the ATOM base image ships.  SGLang
# runtime installs can perturb Triton; restore the base package before final
# validation so ATOM, AITER, triton_kernels, and SGLang run with one coherent
# runtime stack, matching the vLLM/OOT image policy above.
RUN echo "========== [SGLANG-ATOM 4/6] Restore base image Triton ==========" && \
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
    "${VENV_PYTHON}" -m pip show triton

RUN echo "========== [SGLANG-ATOM 5/6] Validate vision/audio wheels ==========" && \
    "${VENV_PYTHON}" -m sglang.launch_server --help >/dev/null && \
    "${VENV_PYTHON}" -c "import amdsmi, fast_hadamard_transform, os, tilelang, torch, torchvision, torchaudio, sglang, triton, transformers; from torchvision.io import decode_jpeg; assert torch.version.hip is not None, 'Torch is not ROCm build (torch.version.hip is None).'; print(f'torch: {torch.__version__}'); print(f'triton: {triton.__version__}'); print(f'transformers: {transformers.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'decode_jpeg: {decode_jpeg.__name__}'); print(f'sglang imported from: {sglang.__file__}'); print(f'PYTHONPATH={os.environ.get(\"PYTHONPATH\", \"\")}')" && \
    echo "Validated sglang launch_server entrypoint"

RUN echo "========== [SGLANG-ATOM 5.5/6] Pin smg-grpc-servicer ==========" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir "smg-grpc-servicer==0.5.2" && \
    "${VENV_PYTHON}" -m pip show smg-grpc-proto smg-grpc-servicer

RUN echo "========== [SGLANG-ATOM 6/6] Check Aiter/FlyDSL versions after SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom amd-mori-nightly amd-aiter flydsl sglang triton triton-kernels || true

CMD ["/bin/bash"]
