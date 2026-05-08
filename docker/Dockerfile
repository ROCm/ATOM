# Default base images
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG OOT_BASE_IMAGE="rocm/atom-dev:latest"
ARG SGLANG_BASE_IMAGE="rocm/atom-dev:latest"
ARG GPU_ARCH="gfx942;gfx950"

# OOT image extends an ATOM base image that already contains atom/aiter/mori.
FROM ${OOT_BASE_IMAGE} AS atom_oot

ARG MAX_JOBS
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG VLLM_REPO="https://github.com/vllm-project/vllm.git"
ARG VLLM_COMMIT="b31e9326a7d9394aab8c767f8ebe225c65594b60"
ARG INSTALL_LM_EVAL=1
ARG INSTALL_FASTSAFETENSORS=1
# Let PR OOT CI verify whether a pulled prebuilt image still matches this vLLM commit
LABEL com.rocm.atom.vllm_commit="${VLLM_COMMIT}"

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

# Keep OOT aligned with the Triton that the ATOM base image actually imports.
# vLLM/OOT installs can perturb Triton metadata, and some base images may carry
# extra triton-*.dist-info residues. We therefore treat `import triton` in the
# base image as the source of truth and back up only the matching dist-info
RUN echo "========== [OOT 2/7] Verify base packages (atom/aiter/mori) ==========" && \
    "${VENV_PYTHON}" -m pip show atom || true && \
    "${VENV_PYTHON}" -m pip show amd-aiter || true && \
    "${VENV_PYTHON}" -m pip show mori || true && \
    echo "========== [OOT 2/7] Back up base image triton ==========" && \
    BASE_TRITON_VERSION="$("${VENV_PYTHON}" -c "import triton; print(triton.__version__)")" && \
    mkdir -p /tmp/triton-base-backup && \
    cp -a /opt/venv/lib/python3.12/site-packages/triton /tmp/triton-base-backup/ && \
    MATCHED_DIST_INFO="" && \
    MATCH_COUNT=0 && \
    BACKUP_COUNT=0 && \
    for f in /opt/venv/lib/python3.12/site-packages/triton-*.dist-info; do \
      [ -d "$f" ] || continue; \
      DIST_VERSION="$(awk -F': ' '$1 == "Version" { print $2; exit }' "$f/METADATA")"; \
      if [ "${DIST_VERSION}" = "${BASE_TRITON_VERSION}" ]; then \
        cp -a "$f" /tmp/triton-base-backup/; \
        MATCHED_DIST_INFO="$(basename "$f")"; \
        MATCH_COUNT=$((MATCH_COUNT + 1)); \
      fi; \
    done && \
    if [ "${MATCH_COUNT}" -eq 1 ]; then \
      BACKUP_COUNT="${MATCH_COUNT}"; \
    else \
      echo "WARNING: Expected exactly one triton dist-info matching imported version ${BASE_TRITON_VERSION}, found ${MATCH_COUNT}. Falling back to backing up all available triton dist-info entries to preserve the base image state." && \
      rm -rf /tmp/triton-base-backup/triton-*.dist-info && \
      MATCHED_DIST_INFO="" && \
      BACKUP_COUNT=0 && \
      for f in /opt/venv/lib/python3.12/site-packages/triton-*.dist-info; do \
        [ -d "$f" ] || continue; \
        cp -a "$f" /tmp/triton-base-backup/; \
        echo "Backed up fallback triton dist-info: $(basename "$f")"; \
        BACKUP_COUNT=$((BACKUP_COUNT + 1)); \
      done; \
      ls -d /opt/venv/lib/python3.12/site-packages/triton-*.dist-info || true; \
    fi && \
    echo "Base image triton backup summary: version=${BASE_TRITON_VERSION}, matched_dist_info=${MATCHED_DIST_INFO:-<none>}, backed_up_count=${BACKUP_COUNT}"

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
    # keeps the base image's ATOM-installed transformers
    sed -i -e '/^transformers[[:space:]]/d' requirements/common.txt && \
    echo "Removed transformers constraint from cloned vLLM requirements/common.txt to preserve base image transformers" && \
    ! grep '^transformers' requirements/common.txt && \
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

RUN echo "========== [VLLM-ATOM] Validate vision/audio wheels ==========" && \
    "${VENV_PYTHON}" -c "import torch, torchvision, torchaudio; from torchvision.transforms import InterpolationMode; from transformers.models.auto.image_processing_auto import get_image_processor_config; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'InterpolationMode: {InterpolationMode.BILINEAR}'); print(f'get_image_processor_config: {get_image_processor_config.__name__}')"

# Restore that exact base-image Triton after all OOT installs finish. The goal
# is not to choose a new Triton version, but to preserve ATOM base-image runtime
# behavior and fail fast if `pip show triton` would disagree with `import triton`.
RUN echo "========== [OOT] Restore base image triton ==========" && \
    "${VENV_PYTHON}" -m pip uninstall -y triton 2>/dev/null || true && \
    rm -rf /opt/venv/lib/python3.12/site-packages/triton \
           /opt/venv/lib/python3.12/site-packages/triton-*.dist-info && \
    cp -a /tmp/triton-base-backup/triton /opt/venv/lib/python3.12/site-packages/ && \
    RESTORED_DIST_INFO="" && \
    RESTORED_COUNT=0 && \
    for f in /tmp/triton-base-backup/triton-*.dist-info; do \
      [ -d "$f" ] || continue; \
      cp -a "$f" /opt/venv/lib/python3.12/site-packages/; \
      RESTORED_DIST_INFO="$(basename "$f")"; \
      RESTORED_COUNT=$((RESTORED_COUNT + 1)); \
    done && \
    rm -rf /tmp/triton-base-backup && \
    if [ "${RESTORED_COUNT}" -eq 1 ]; then \
      echo "Restored base image triton dist-info: ${RESTORED_DIST_INFO}"; \
    else \
      echo "WARNING: Restored ${RESTORED_COUNT} triton dist-info directories from the base image backup. Keeping build non-fatal to preserve the base image state." && \
      ls -d /opt/venv/lib/python3.12/site-packages/triton-*.dist-info || true; \
    fi && \
    "${VENV_PYTHON}" -m pip show triton > /tmp/triton_pip_show.txt 2>&1 || true && \
    PIP_TRITON_VERSION="$(awk -F': ' '/^Version: / { print $2; exit }' /tmp/triton_pip_show.txt)" && \
    IMPORT_TRITON_VERSION="$("${VENV_PYTHON}" -c "import triton; print(triton.__version__)")" && \
    if [ -z "${PIP_TRITON_VERSION}" ] || [ "${PIP_TRITON_VERSION}" != "${IMPORT_TRITON_VERSION}" ]; then \
      echo "WARNING: pip show triton reports '${PIP_TRITON_VERSION:-<missing>}', but import triton reports '${IMPORT_TRITON_VERSION}'. Keeping build non-fatal so OOT image release is not blocked by pre-existing base image metadata drift." && \
      cat /tmp/triton_pip_show.txt || true && \
      ls -d /opt/venv/lib/python3.12/site-packages/triton-*.dist-info || true; \
    else \
      echo "Restored base image triton: version=${IMPORT_TRITON_VERSION}, dist-info=${RESTORED_DIST_INFO}" && \
      cat /tmp/triton_pip_show.txt; \
    fi && \
    rm -f /tmp/triton_pip_show.txt && \
    "${VENV_PYTHON}" -c "import triton; print(f'triton.__version__ = {triton.__version__}')"

RUN echo "========== [vLLM-ATOM] Final transformers version ==========" && \
    "${VENV_PYTHON}" -c "import transformers; print(f'transformers.__version__ = {transformers.__version__}')" && \
    "${VENV_PYTHON}" -m pip show transformers || true

CMD ["/bin/bash"]

# SGLang image extends an ATOM base image and layers on an sglang checkout.
FROM ${SGLANG_BASE_IMAGE} AS atom_sglang

ARG GPU_ARCH
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG SGLANG_REPO="https://github.com/sgl-project/sglang.git"
ARG SGLANG_REF="v0.5.10"
ARG SGLANG_TRITON_VERSION="3.6.0"
LABEL com.rocm.atom.sglang_ref="${SGLANG_REF}"

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/app/sglang/python:/app/ATOM:${PYTHONPATH}"

RUN echo "========== [SGLANG-ATOM 0/6] Check Aiter/FlyDSL versions before SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom mori amd-aiter flydsl || true

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
    cd /app/sglang/sgl-kernel && \
    AMDGPU_TARGET="${FINAL_AMDGPU_TARGET}" "${VENV_PYTHON}" setup_rocm.py install && \
    "${VENV_PYTHON}" -m pip show sglang-kernel || true

RUN echo "========== [SGLANG-ATOM 3/6] Install SGLang dependencies ==========" && \
    cd /app/sglang/python && \
    rm -f pyproject.toml && \
    cp pyproject_other.toml pyproject.toml && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir --no-deps -e . && \
    "${VENV_PYTHON}" -c "import tomllib; from pathlib import Path; deps = tomllib.loads(Path('pyproject_other.toml').read_text())['project']['optional-dependencies']['runtime_common']; blocked_prefixes = ('compressed-tensors', 'outlines==', 'timm==', 'torchao==', 'transformers==', 'xgrammar=='); Path('/tmp/sglang-runtime-common.txt').write_text(''.join(f'{dep}\\n' for dep in deps if dep != 'numpy' and not any(dep.startswith(prefix) for prefix in blocked_prefixes)))" && \
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

RUN echo "========== [SGLANG-ATOM 4/6] Validate vision/audio wheels ==========" && \
    "${VENV_PYTHON}" -m sglang.launch_server --help >/dev/null && \
    "${VENV_PYTHON}" -c "import os, torch, torchvision, torchaudio, sglang, triton, transformers; from torchvision.io import decode_jpeg; assert torch.version.hip is not None, 'Torch is not ROCm build (torch.version.hip is None).'; print(f'torch: {torch.__version__}'); print(f'triton: {triton.__version__}'); print(f'transformers: {transformers.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'decode_jpeg: {decode_jpeg.__name__}'); print(f'sglang imported from: {sglang.__file__}'); print(f'PYTHONPATH={os.environ.get(\"PYTHONPATH\", \"\")}')" && \
    echo "Validated sglang launch_server entrypoint"

# Only the derived SGLang image needs the newer Triton compiler. SGLang serving
# for DeepSeek-R1 hit `ConvertTritonAMDGPUToLLVM` failures with the Triton that
# ships in the shared ATOM base image, while Triton 3.6.0 fixes that compiler
# path for this flow. Install Triton after the torchvision/torchaudio wheel pin
# so pip does not downgrade it back to the torch-pinned 3.5.1 dependency. Keep
# this override local to `atom_sglang` so the OOT/atom_release flow preserves
# its prior Triton behavior.
RUN echo "========== [SGLANG-ATOM 5/6] Install validated Triton ==========" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir "triton==${SGLANG_TRITON_VERSION}" && \
    "${VENV_PYTHON}" -m pip show triton

RUN echo "========== [SGLANG-ATOM 6/6] Check Aiter/FlyDSL versions after SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom mori amd-aiter flydsl || true

CMD ["/bin/bash"]

# ====================================================================
# ATOM image: multi-stage parallel build
#
# BuildKit runs independent stages in parallel:
#   base ──┬── build_mori   ──┐
#          ├── build_rccl   ──┤
#          ├── build_triton ──┼── atom_image (merge + ATOM install)
#          └── build_aiter  ──┘
# ====================================================================

# --------------------------------------------------------------------
# Stage 0: Common base (apt + pip foundations, shared by all builders)
# --------------------------------------------------------------------
FROM ${BASE_IMAGE} AS base

ARG GPU_ARCH
ENV GPU_ARCH_LIST=$GPU_ARCH
ENV PYTORCH_ROCM_ARCH=$GPU_ARCH

RUN pip install --upgrade pip && \
    apt-get update && \
    apt --fix-broken install -y && \
    apt-get install -y \
        git cython3 ibverbs-utils openmpi-bin libopenmpi-dev \
        libpci-dev cmake libdw1 locales && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------
# Stage 1: MORI — parallel
# --------------------------------------------------------------------
FROM base AS build_mori
ARG MORI_COMMIT="v1.1.0"

RUN echo "========== [Parallel] Building MORI ==========" && \
    git clone https://github.com/ROCm/mori.git /app/mori && \
    cd /app/mori && \
    git checkout $MORI_COMMIT && \
    pip install .

# --------------------------------------------------------------------
# Stage 2: RCCL — parallel
# --------------------------------------------------------------------
FROM base AS build_rccl
ARG RCCL_REPO="https://github.com/ROCm/rccl.git"
ARG RCCL_BRANCH="29e1567b95e28823b0beb1a988adc587bfab5b4f"

RUN echo "========== [Parallel] Building RCCL ==========" && \
    pip install cmake && \
    git clone "$RCCL_REPO" /app/rccl && \
    cd /app/rccl && \
    git checkout "$RCCL_BRANCH" && \
    ./install.sh -p --amdgpu_targets=$GPU_ARCH_LIST

# --------------------------------------------------------------------
# Stage 3: Triton — parallel
# --------------------------------------------------------------------
FROM base AS build_triton

RUN echo "========== [Parallel] Building Triton ==========" && \
    pip uninstall -y triton && \
    git clone --depth=1 --branch release/internal/3.5.x \
        https://github.com/ROCm/triton.git /triton-test && \
    cd /triton-test && \
    pip install -r python/requirements.txt && \
    pip install filecheck && \
    MAX_JOBS=64 pip --retries=10 --default-timeout=60 install . && \
    # Companion `triton_kernels` package (pure-Python). Required by ATOM when
    # `ATOM_USE_TRITON_MOE=1` (e.g. DeepSeek-V4 launch path). Lives under
    # python/triton_kernels in the same ROCm/triton checkout.
    pip install ./python/triton_kernels && \
    python -c "import triton_kernels; print(f'triton_kernels: {triton_kernels.__file__}')"

# --------------------------------------------------------------------
# Stage 4: Aiter — parallel
# --------------------------------------------------------------------
FROM base AS build_aiter
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="HEAD"
ARG PREBUILD_KERNELS=1
ARG MAX_JOBS

RUN pip install --upgrade setuptools_scm
RUN echo "========== [Parallel] Building Aiter ==========" && \
    git clone $AITER_REPO /app/aiter-test && \
    cd /app/aiter-test && \
    pip install -r requirements.txt && \
    git checkout $AITER_COMMIT && \
    git submodule sync && git submodule update --init --recursive && \
    MAX_JOBS=$MAX_JOBS PREBUILD_KERNELS=$PREBUILD_KERNELS \
    GPU_ARCHS=$GPU_ARCH_LIST python3 setup.py develop

# --------------------------------------------------------------------
# Stage 5: Final merge — collect all build artifacts + install ATOM
# --------------------------------------------------------------------
FROM base AS atom_image
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"

# pip packages (lm-eval is lightweight, install directly)
RUN pip install lm-eval[api]

# MORI: copy installed package from build stage
COPY --from=build_mori /opt/venv /opt/venv
COPY --from=build_mori /app/mori /app/mori
RUN pip show mori || true

# RCCL: install .deb from build stage
COPY --from=build_rccl /app/rccl/build/release/*.deb /tmp/rccl/
RUN DEBIAN_FRONTEND=noninteractive dpkg -i --force-all /tmp/rccl/*.deb && \
    rm -rf /tmp/rccl

# Triton: copy package from build stage into current venv
# Use RUN --mount to avoid COPY glob issues, and preserve mori already in venv
# Also copy the companion `triton_kernels` package (pure-Python, built in the
# same stage) — required by ATOM_USE_TRITON_MOE=1 model paths.
RUN --mount=from=build_triton,source=/opt/venv/lib/python3.12/site-packages,target=/triton-pkgs \
    cp -a /triton-pkgs/triton /opt/venv/lib/python3.12/site-packages/ && \
    cp -a /triton-pkgs/triton-*.dist-info /opt/venv/lib/python3.12/site-packages/ && \
    cp -a /triton-pkgs/triton_kernels /opt/venv/lib/python3.12/site-packages/ && \
    cp -a /triton-pkgs/triton_kernels-*.dist-info /opt/venv/lib/python3.12/site-packages/ && \
    pip show triton triton_kernels && \
    python -c "import triton, triton_kernels; print(f'triton {triton.__version__}; triton_kernels {triton_kernels.__file__}')"

# Aiter: copy compiled source tree + re-register editable install
# (pip install -e creates egg-link automatically, no need to COPY them)
COPY --from=build_aiter /app/aiter-test /app/aiter-test
RUN cd /app/aiter-test && pip install -e . --no-build-isolation && \
    pip show amd-aiter

# RTL (rocm-trace-lite): lightweight GPU kernel profiler (~250KB, no build deps)
RUN pip install rocm-trace-lite && \
    rtl --version || true

# ATOM: lightweight install (no compilation needed)
# CACHEBUST invalidates only this layer so parallel stages stay cached
ARG CACHEBUST=1
RUN git clone $ATOM_REPO /app/ATOM && \
    cd /app/ATOM && \
    git checkout $ATOM_COMMIT && \
    pip install -e .
RUN pip show atom || true

CMD ["/bin/bash"]
