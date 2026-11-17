#!/usr/bin/env bash
set -euo pipefail
runtime=/tmp/atom-pr-review-runtime
mkdir -p "$runtime"/{home,cache,bin}
python3 -m venv --system-site-packages "$runtime/venv"
source "$runtime/venv/bin/activate"
export PATH="$runtime/bin:$PATH"
cat > "$runtime/bin/nvcc" <<'SHIM'
#!/usr/bin/env bash
if [[ "${1:-}" == --version ]]; then exec hipcc --version; fi
echo 'nvcc shim supports only the ROCm version probe' >&2
exit 127
SHIM
chmod +x "$runtime/bin/nvcc"
python -m pip install --upgrade 'pybind11>=3.0.1'
git clone --filter=blob:none --no-checkout https://github.com/ROCm/aiter.git "$AITER_SOURCE_PATH"
git -C "$AITER_SOURCE_PATH" fetch --depth=1 origin "$AITER_GIT_REF"
git -C "$AITER_SOURCE_PATH" checkout --detach FETCH_HEAD
git -C "$AITER_SOURCE_PATH" submodule update --init --recursive
MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 \
  python -m pip install --no-build-isolation -e "$AITER_SOURCE_PATH"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ATOM=0.0.0+prreview python -m pip install -e /workspace
python -m pip install 'lm_eval[api]==0.4.13' pytest
