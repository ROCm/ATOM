ARG SGLANG_BASE_IMAGE="rocm/atom-dev:latest"
ARG GPU_ARCH="gfx942;gfx950"

# SGLang image extends an ATOM base image and layers on an sglang checkout.
# sglang_base also serves as the read-only source of the base image's Triton
# for the restore step below.
FROM ${SGLANG_BASE_IMAGE} AS sglang_base
FROM sglang_base AS atom_sglang

ARG GPU_ARCH
ARG VENV_PYTHON="/opt/venv/bin/python"
# Build-time only: keep downloaded wheels out of the layers.
ARG PIP_NO_CACHE_DIR=1
ARG SGLANG_REPO="https://github.com/sgl-project/sglang.git"
ARG SGLANG_REF="v0.5.22"
LABEL com.rocm.atom.sglang_ref="${SGLANG_REF}"

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/app/sglang/python:/app/ATOM:${PYTHONPATH}"

RUN echo "========== [SGLANG-ATOM 0/6] Check Aiter/FlyDSL/Triton versions before SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom amd-mori-nightly amd-aiter flydsl triton || true

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
    rm -f pyproject.toml && \
    cp pyproject_other.toml pyproject.toml && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir --no-deps -e . && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir tomli && \
    "${VENV_PYTHON}" -c "import tomli; from pathlib import Path; extras = tomli.loads(Path('pyproject_other.toml').read_text())['project']['optional-dependencies']; blocked = ('compressed-tensors', 'outlines==', 'timm==', 'torchao==', 'xgrammar==', 'transformers'); seen = set(); out = []; walk = lambda names: [None for dep in names for key in [dep.strip()] if key and key not in seen and not seen.add(key) and (walk(extras[key[len('sglang['):-1]]) if key.startswith('sglang[') and key.endswith(']') else out.append(key))]; pkg = lambda dep: dep.split('[')[0].split('=')[0].split('<')[0].split('>')[0].split('!')[0].strip(); walk(extras['runtime_common']); Path('/tmp/sglang-runtime-common.txt').write_text(''.join(f'{dep}\\n' for dep in out if pkg(dep) != 'numpy' and not any(dep.startswith(p) or pkg(dep) == p.split('=', 1)[0] for p in blocked)))" && \
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
    TORCH_HIP="$("${VENV_PYTHON}" -c "import torch; print(torch.version.hip or '')")" && \
    test -n "${TORCH_HIP}" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir --no-deps "transformers==5.16.1" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir \
      "huggingface-hub==1.5.0" \
      "datasets==5.0.0" \
      "tokenizers==0.23.1" \
      "safetensors>=0.8.0" \
      "regex>=2025.10.22" \
      "typer" \
      "packaging>=20.0" \
      "pyyaml>=5.1" \
      "tqdm>=4.60" && \
    "${VENV_PYTHON}" -m pip show sglang torch triton transformers IPython orjson pybase64 petit-kernel wave-lang xgrammar outlines apache-tvm-ffi || true

# Keep SGLang aligned with the Triton that the ATOM base image ships.  SGLang
# runtime installs can perturb Triton; restore the base package before final
# validation so ATOM, AITER, triton_kernels, and SGLang run with one coherent
# runtime stack, matching the vLLM/OOT image policy above.
# The base image's Triton is read through a bind mount of sglang_base instead
# of a /tmp backup: a backup taken in an earlier RUN stays in that layer even
# after it is deleted, and copying back an unchanged tree writes the same
# ~1.6 GB again. Restore only when the installed tree differs from the base.
RUN --mount=type=bind,from=sglang_base,source=/opt/venv,target=/tmp/base-venv \
    echo "========== [SGLANG-ATOM 4/6] Restore base image Triton ==========" && \
    SITE_PACKAGES=$("${VENV_PYTHON}" -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    BASE_SITE_PACKAGES="/tmp/base-venv${SITE_PACKAGES#/opt/venv}" && \
    test -d "${BASE_SITE_PACKAGES}/triton" && \
    BASE_DIST_INFO="$(cd "${BASE_SITE_PACKAGES}" && ls -d triton-*.dist-info 2>/dev/null | sort | tr '\n' ' ')" && \
    CUR_DIST_INFO="$(cd "${SITE_PACKAGES}" && ls -d triton-*.dist-info 2>/dev/null | sort | tr '\n' ' ')" && \
    TRITON_CHANGED=0 && \
    if [ "${BASE_DIST_INFO}" != "${CUR_DIST_INFO}" ]; then TRITON_CHANGED=1; fi && \
    if [ "${TRITON_CHANGED}" = "0" ]; then \
      for p in triton ${BASE_DIST_INFO}; do \
        diff -rq -x __pycache__ "${BASE_SITE_PACKAGES}/${p}" "${SITE_PACKAGES}/${p}" >/dev/null \
          || { TRITON_CHANGED=1; break; }; \
      done; \
    fi && \
    if [ "${TRITON_CHANGED}" = "0" ]; then \
      echo "Triton unchanged since the base image (${BASE_DIST_INFO}); nothing to restore"; \
    else \
      echo "Triton changed (${CUR_DIST_INFO:-none} vs base ${BASE_DIST_INFO}); restoring the base image copy" && \
      { "${VENV_PYTHON}" -m pip uninstall -y triton 2>/dev/null || true; } && \
      rm -rf "${SITE_PACKAGES}/triton" \
             "${SITE_PACKAGES}"/triton-*.dist-info && \
      cp -a "${BASE_SITE_PACKAGES}/triton" "${SITE_PACKAGES}/" && \
      for f in "${BASE_SITE_PACKAGES}"/triton-*.dist-info; do \
        [ -d "$f" ] || continue; \
        cp -a "$f" "${SITE_PACKAGES}/"; \
      done; \
    fi && \
    "${VENV_PYTHON}" -c "import triton; print(f'triton.__version__ = {triton.__version__}')" && \
    "${VENV_PYTHON}" -m pip show triton

RUN echo "========== [SGLANG-ATOM 5/6] Validate vision/audio wheels ==========" && \
    "${VENV_PYTHON}" -m sglang.launch_server --help >/dev/null && \
    "${VENV_PYTHON}" -c "import os, torch, torchvision, torchaudio, sglang, triton, transformers; from torchvision.io import decode_jpeg; assert torch.version.hip is not None, 'Torch is not ROCm build (torch.version.hip is None).'; print(f'torch: {torch.__version__}'); print(f'triton: {triton.__version__}'); print(f'transformers: {transformers.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'decode_jpeg: {decode_jpeg.__name__}'); print(f'sglang imported from: {sglang.__file__}'); print(f'PYTHONPATH={os.environ.get(\"PYTHONPATH\", \"\")}')" && \
    echo "Validated sglang launch_server entrypoint"

RUN echo "========== [SGLANG-ATOM 5.5/6] Pin smg-grpc-servicer ==========" && \
    "${VENV_PYTHON}" -m pip install --no-cache-dir "smg-grpc-servicer==0.5.2" && \
    "${VENV_PYTHON}" -m pip show smg-grpc-proto smg-grpc-servicer && \
    "${VENV_PYTHON}" -c "import importlib.metadata as m; import torch; from transformers import AutoConfig; assert torch.version.hip is not None, 'pip replaced ROCm torch'; assert m.version('transformers') == '5.16.1', m.version('transformers'); assert m.version('tokenizers') == '0.23.1', m.version('tokenizers'); assert m.version('huggingface-hub') == '1.5.0', m.version('huggingface-hub'); assert m.version('datasets') == '5.0.0', m.version('datasets'); cfg = AutoConfig.for_model('qwen4_exp_text'); assert cfg.model_type == 'qwen4_exp_text', cfg.model_type"

RUN echo "========== [SGLANG-ATOM 6/6] Check Aiter/FlyDSL versions after SGLang build ==========" && \
    "${VENV_PYTHON}" -m pip show atom amd-mori-nightly amd-aiter flydsl sglang triton triton-kernels || true

CMD ["/bin/bash"]
