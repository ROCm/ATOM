ARG SGLANG_BASE_IMAGE="rocm/atom-dev:latest"
ARG GPU_ARCH="gfx942;gfx950"

# SGLang image extends an ATOM base image and layers on an sglang checkout.
FROM ${SGLANG_BASE_IMAGE} AS atom_sglang

ARG GPU_ARCH
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG SGLANG_REPO="https://github.com/sgl-project/sglang.git"
ARG SGLANG_REF="v0.5.12"
LABEL com.rocm.atom.sglang_ref="${SGLANG_REF}"

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/app/sglang/python:/app/ATOM:${PYTHONPATH}"

COPY docker/patches/sglang/cohere2-moe-drop-invalid-hf-strict.patch /tmp/sglang-patches/cohere2-moe-drop-invalid-hf-strict.patch

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
    git apply /tmp/sglang-patches/cohere2-moe-drop-invalid-hf-strict.patch && \
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
    "${VENV_PYTHON}" -c "import os, torch, torchvision, torchaudio, sglang, triton, transformers; from torchvision.io import decode_jpeg; assert torch.version.hip is not None, 'Torch is not ROCm build (torch.version.hip is None).'; print(f'torch: {torch.__version__}'); print(f'triton: {triton.__version__}'); print(f'transformers: {transformers.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'decode_jpeg: {decode_jpeg.__name__}'); print(f'sglang imported from: {sglang.__file__}'); print(f'PYTHONPATH={os.environ.get(\"PYTHONPATH\", \"\")}')" && \
    echo "Validated sglang launch_server entrypoint"

RUN echo "========== [SGLANG-ATOM 5.5/6] Pin smg-grpc-servicer ==========" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir "smg-grpc-servicer==0.5.2" && \
    "${VENV_PYTHON}" -m pip show smg-grpc-proto smg-grpc-servicer

RUN echo "========== [SGLANG-ATOM 6/6] Check Aiter/FlyDSL versions after SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom amd-mori-nightly amd-aiter flydsl sglang triton triton-kernels || true

CMD ["/bin/bash"]
