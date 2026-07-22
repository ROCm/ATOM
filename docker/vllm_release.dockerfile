ARG OOT_BASE_IMAGE="rocm/atom-dev:latest"

# OOT image extends an ATOM base image that already contains atom/aiter/mori.
FROM ${OOT_BASE_IMAGE} AS atom_oot

ARG MAX_JOBS
ARG VENV_PYTHON="/opt/venv/bin/python"
ARG VLLM_REPO="https://github.com/vllm-project/vllm.git"
ARG VLLM_COMMIT="v0.25.1"
ARG INSTALL_LM_EVAL=1
ARG INSTALL_FASTSAFETENSORS=1
# Let PR OOT CI verify whether a pulled prebuilt image still matches this vLLM commit
LABEL com.rocm.atom.vllm_commit="${VLLM_COMMIT}"

ENV PATH="/opt/venv/bin:${PATH}"
ENV MAX_JOBS=${MAX_JOBS}
ENV VLLM_TARGET_DEVICE=rocm
ENV CMAKE_MAKE_PROGRAM=/usr/local/bin/ninja

RUN echo "========== [OOT 1/7] Prepare build tools ==========" && \
    apt-get update && \
    apt --fix-broken install -y && \
    apt-get install -y --no-install-recommends ca-certificates jq ninja-build vim && \
    "${VENV_PYTHON}" -m pip install --upgrade cmake && \
    cmake --version && \
    mkdir -p /usr/local/bin && \
    ln -sf "$(command -v ninja)" /usr/local/bin/ninja && \
    /usr/local/bin/ninja --version && \
    rm -rf /var/lib/apt/lists/* && \
    TORCH_LIB=$("${VENV_PYTHON}" -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))") && \
    echo "${TORCH_LIB}" > /etc/ld.so.conf.d/torch.conf && \
    ldconfig

# Keep OOT aligned with the Triton that the ATOM base image ships.
# vLLM/OOT installs can perturb Triton; back up everything and restore after.
RUN echo "========== [OOT 2/7] Verify base packages (atom/aiter/mori) ==========" && \
    "${VENV_PYTHON}" -m pip show atom || true && \
    "${VENV_PYTHON}" -m pip show amd-aiter || true && \
    "${VENV_PYTHON}" -m pip show amd-mori-nightly || true && \
    echo "========== [OOT 2/7] Back up base image triton ==========" && \
    SITE_PACKAGES=$("${VENV_PYTHON}" -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    BASE_TRITON_VERSION="$("${VENV_PYTHON}" -c "import triton; print(triton.__version__)")" && \
    mkdir -p /tmp/triton-base-backup && \
    cp -a "${SITE_PACKAGES}/triton" /tmp/triton-base-backup/ && \
    for f in "${SITE_PACKAGES}"/triton-*.dist-info; do \
      [ -d "$f" ] || continue; \
      cp -a "$f" /tmp/triton-base-backup/; \
    done && \
    echo "Base image triton backed up: import_version=${BASE_TRITON_VERSION}" && \
    ls /tmp/triton-base-backup/

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
    sed -i -e '/xgrammar/d' -e '/compressed-tensors/d' requirements/common.txt && \
    "${VENV_PYTHON}" -m pip install --no-deps "xgrammar>=0.2.1,<1.0.0" "compressed-tensors==0.17.0" loguru && \
    sed -i -e '/peft/d' -e '/tensorizer/d' -e '/runai/d' -e '/timm/d' -e '/tilelang/d' requirements/rocm.txt && \
    "${VENV_PYTHON}" -m pip install --no-deps peft "tensorizer==2.10.1" "runai-model-streamer[s3,gcs,azure]==0.15.7" "timm>=1.0.17" "tilelang==0.1.10" "torch-c-dlpack-ext==0.1.5" "z3-solver==4.15.4.0" && \
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
    if [ "${INSTALL_LM_EVAL}" = "1" ]; then "${VENV_PYTHON}" -m pip install "lm-eval[api]"; else echo "Skip lm-eval install"; fi && \
    if [ "${INSTALL_FASTSAFETENSORS}" = "1" ]; then "${VENV_PYTHON}" -m pip install "git+https://github.com/foundation-model-stack/fastsafetensors.git"; else echo "Skip fastsafetensors install"; fi && \
    "${VENV_PYTHON}" -c "import glob, os, torch; print(f'torch.version.hip: {torch.version.hip}'); print(f'torch.version.cuda: {torch.version.cuda}'); torch_lib_dir=os.path.join(os.path.dirname(torch.__file__), 'lib'); print(f'torch lib dir: {torch_lib_dir}'); print(f'libtorch_hip candidates: {glob.glob(os.path.join(torch_lib_dir, \"libtorch_hip.so*\"))}'); assert torch.version.hip is not None, 'Torch is not ROCm build (torch.version.hip is None).'" && \
    "${VENV_PYTHON}" -m pip show vllm torch triton torchvision torchaudio amdsmi amd-aiter atom amd-mori-nightly || true

RUN echo "========== [VLLM-ATOM] Validate vision/audio wheels ==========" && \
    "${VENV_PYTHON}" -c "import torch, torchvision, torchaudio; from torchvision.transforms import InterpolationMode; from transformers.models.auto.image_processing_auto import get_image_processor_config; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'InterpolationMode: {InterpolationMode.BILINEAR}'); print(f'get_image_processor_config: {get_image_processor_config.__name__}')"

# Restore the exact base-image Triton after all OOT installs finish.
RUN echo "========== [OOT] Restore base image triton ==========" && \
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
    "${VENV_PYTHON}" -m pip show triton || true


RUN echo "========== [vLLM-ATOM] Final transformers version ==========" && \
    "${VENV_PYTHON}" -c "import transformers; print(f'transformers.__version__ = {transformers.__version__}')" && \
    "${VENV_PYTHON}" -m pip show transformers || true

CMD ["/bin/bash"]
