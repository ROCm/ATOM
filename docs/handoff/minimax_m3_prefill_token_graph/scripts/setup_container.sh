#!/bin/bash
# In the container: what the ATOM image lacks for the agentic run (MI455 gfx1250 or MI355X gfx950).
#   aiperf 754356e9 (ATOM image pin), datasets 4.8.5 (the trace uses the Json feature),
#   LMCache f22dec28 (ATOM image pin) built for ARCH — only the CONC>=40 points need it (SKIP_LMCACHE=1).
# Usage: [ARCH=gfx950|gfx1250] [SKIP_LMCACHE=1] bash setup_container.sh [LMCache checkout dir, default /opt]
set -euxo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
SRC=${1:-/opt}
PY=python3
ARCH=${ARCH:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')}
echo "ARCH=$ARCH"
$PY -c "import torch; print('torch', torch.__version__)"

# aiperf. Its prometheus_client~=0.23.1 pin wins over ATOM's 0.25, as in the ATOM image.
if [ "$(git -C /opt/aiperf rev-parse HEAD 2>/dev/null)" != 754356e9a39acc6cc6afb242d123bb57c3fb6f75 ]; then
  rm -rf /opt/aiperf
  git clone -q https://github.com/SemiAnalysisAI/aiperf.git /opt/aiperf
  git -C /opt/aiperf checkout -q 754356e9a39acc6cc6afb242d123bb57c3fb6f75
  $PY -m pip install -q -e /opt/aiperf
fi
$PY -m pip install -q datasets==4.8.5

# LMCache. hipcc links against <ROCM_PATH>/lib/libamdhip64.so, which this image ships only as .so.7.
HIPLIB=$($PY -c "import os; print(os.path.realpath('/opt/rocm/lib'))")
[ -e "$HIPLIB/libamdhip64.so" ] || ln -s libamdhip64.so.7 "$HIPLIB/libamdhip64.so"
if [ "${SKIP_LMCACHE:-0}" != 1 ] && ! $PY -c "import lmcache" 2>/dev/null; then
  L=$SRC/lmcache_src
  [ -d "$L" ] || git clone -q https://github.com/LMCache/LMCache.git "$L"
  git -C "$L" checkout -q f22dec2832124ea4ffeb464553b819a6369d4a30
  git config --global --add safe.directory "$L"
  cd "$L"
  $PY -m pip install -q -r requirements/build.txt pybind11
  $PY -m pip install -q prometheus_client aiofile==3.11.1 aiofiles caio==0.9.25 blake3 redis \
    sortedcontainers pyzmq cachetools cryptography numba openai py-cpuinfo nvtx awscrt \
    opentelemetry-api==1.40.0 opentelemetry-sdk==1.40.0 opentelemetry-exporter-otlp==1.40.0 \
    opentelemetry-exporter-prometheus==0.61b0 "protobuf>=6.31.1,<7"
  rm -rf build dist_raw
  find csrc -name '*_hip.*' -delete 2>/dev/null || true
  find csrc -name '*.hip' -delete 2>/dev/null || true
  BUILD_WITH_HIP=1 CXX=hipcc PYTORCH_ROCM_ARCH=$ARCH MAX_JOBS=64 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.5.6.dev139+gf22dec28 $PY setup.py bdist_wheel --dist-dir=dist_raw
  $PY -m pip install -q --no-deps --force-reinstall dist_raw/*.whl
  cd /
fi
# Re-pin after the LMCache deps, which may move it.
$PY -m pip install -q "prometheus_client~=0.23.1"

$PY - <<'EOF'
import os
import aiperf, datasets
print("datasets", datasets.__version__, "| aiperf ok")
if os.environ.get("SKIP_LMCACHE") != "1":
    import torch; torch.cuda.is_available = lambda: True
    import lmcache, lmcache.cuda_ops, lmcache.lmcache_native
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    import atom.kv_transfer.offload.cache_policy  # registers ATOM_SLRU
    print("lmcache", lmcache.__version__, "| ATOM_SLRU ok")
EOF
echo SETUP_DONE
