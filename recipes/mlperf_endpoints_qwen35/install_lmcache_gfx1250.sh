#!/usr/bin/env bash
# Build for the ROCm/PyTorch ABI in gfx1250-atom-20261031-qwen35.
set -euo pipefail
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LMCACHE_COMMIT=f22dec2832124ea4ffeb464553b819a6369d4a30
LMCACHE_SOURCE=${LMCACHE_SOURCE:-/work/LMCache}
LMCACHE_VENV=${LMCACHE_VENV:-/work/lmcache-venv}
WHEEL_DIR=${WHEEL_DIR:-/work/wheels}
if [[ ! -d "$LMCACHE_SOURCE/.git" ]]; then
    git clone https://github.com/LMCache/LMCache.git "$LMCACHE_SOURCE"
    git -C "$LMCACHE_SOURCE" checkout --detach "$LMCACHE_COMMIT"
fi
[[ "$(git -c safe.directory="$LMCACHE_SOURCE" -C "$LMCACHE_SOURCE" rev-parse HEAD)" = "$LMCACHE_COMMIT" ]] || {
    echo "LMCache source must be at $LMCACHE_COMMIT" >&2
    exit 1
}
# The image has pip but no ensurepip; reuse pip through system-site-packages.
python3 -m venv --without-pip --system-site-packages "$LMCACHE_VENV"
PY="$LMCACHE_VENV/bin/python"
"$PY" -m pip install --no-deps -r "$RECIPE_DIR/lmcache-gfx1250-requirements.txt"
export BUILD_WITH_HIP=1 CXX=hipcc PYTORCH_ROCM_ARCH=gfx1250
export MAX_JOBS=${MAX_JOBS:-8}
export SETUPTOOLS_SCM_PRETEND_VERSION=0.5.6.dev139+gf22dec28.gfx1250
mkdir -p "$WHEEL_DIR"
"$PY" -m pip wheel "$LMCACHE_SOURCE" --no-build-isolation --no-deps -w "$WHEEL_DIR"
"$PY" -m pip install --no-deps "$WHEEL_DIR"/lmcache-0.5.6.dev139+gf22dec28.gfx1250-*.whl
LMCACHE_TRACK_USAGE=false "$PY" - <<'PY'
import torch
import lmcache
import lmcache.cuda_ops
import lmcache.lmcache_native
from lmcache.v1.cache_engine import LMCacheEngineBuilder

assert torch.version.hip, "A ROCm PyTorch build is required"
assert lmcache.cuda_ops.__file__.endswith(".so")
assert lmcache.lmcache_native.__file__.endswith(".so")
print("LMCache:", lmcache.__version__, "PyTorch:", torch.__version__)
PY
