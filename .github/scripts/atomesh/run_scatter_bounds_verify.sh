#!/usr/bin/env bash
# GPU-node entrypoint for the atomesh-benchmark "scatter_bounds" suite.
# Runs inside a Slurm allocation: start a container, install the fixed aiter
# (git ref and/or vendored patch), prove max_sorted, run scatter GPU tests.
#
# Required env:
#   SCATTER_IMAGE       docker image
#   SCATTER_REPO_ROOT   ATOM checkout visible on this node (NFS path)
#   SCATTER_OUT_DIR     writable log directory on NFS
# Optional:
#   AITER_GIT_REF       ROCm/aiter ref (default: pinned companion commit)
#   SCATTER_INSTALL_MODE  aiter_ref | patch  (default: aiter_ref)
#
# The in-container steps live in a host-written script that is docker-cp'd and
# executed as `bash /tmp/scatter_bounds_inner.sh`. Do not put that body in a
# nested `bash -lc '...'` string: apostrophes in comments already broke CI
# (runs 38532471586 / 38535702939).
set -euo pipefail

: "${SCATTER_IMAGE:?}"
: "${SCATTER_REPO_ROOT:?}"
: "${SCATTER_OUT_DIR:?}"

AITER_GIT_REF="${AITER_GIT_REF:-d755e51200f6346287d79573e46b9cf804b955ca}"
SCATTER_INSTALL_MODE="${SCATTER_INSTALL_MODE:-aiter_ref}"
job_id="${SLURM_JOB_ID:-local}"
container="atomesh-scatter-bounds-${job_id}-$$"
host="$(hostname)"
mkdir -p "${SCATTER_OUT_DIR}"
chmod 0777 "${SCATTER_OUT_DIR}" 2>/dev/null || true
log="${SCATTER_OUT_DIR}/rank.log"
rc_file="${SCATTER_OUT_DIR}/rank-rc"

exec > >(tee -a "${log}") 2>&1
echo "=== scatter_bounds rank start host=${host} job=${job_id} container=${container} ==="
echo "image=${SCATTER_IMAGE}"
echo "repo=${SCATTER_REPO_ROOT}"
echo "aiter_ref=${AITER_GIT_REF} mode=${SCATTER_INSTALL_MODE}"

cleanup() {
  local rc=$?
  # Always publish a rank-rc so the GHA wrapper can fail closed if Slurm
  # accounting is flaky (previous run left no rank-rc and GHA stayed green).
  printf '%s\n' "${rc}" > "${rc_file}.tmp" 2>/dev/null || true
  mv "${rc_file}.tmp" "${rc_file}" 2>/dev/null || true
  if command -v timeout >/dev/null 2>&1; then
    timeout 20 docker kill "${container}" >/dev/null 2>&1 || true
    timeout 20 docker rm -f "${container}" >/dev/null 2>&1 || true
  else
    docker kill "${container}" >/dev/null 2>&1 || true
    docker rm -f "${container}" >/dev/null 2>&1 || true
  fi
  exit "${rc}"
}
trap cleanup EXIT

docker pull "${SCATTER_IMAGE}"

cache="/tmp/atomesh-scatter-${job_id}"
mkdir -p "${cache}/home" "${cache}/aiter/jit"

inner_script="${cache}/scatter_bounds_inner.sh"
# Quoted heredoc: body is copied verbatim into the container; no host expansion.
cat > "${inner_script}" <<'INNER_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "=== env ==="
python3 -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip); print('devices', torch.cuda.device_count())"
python3 -c "import aiter; print('aiter_before', aiter.__file__)"

# Keep the image pybind11 ABI. Upgrading 3.0.4 -> 3.1.0 (run 38523159059)
# rebuilt module_moe_mxfp4_aux with internals v12 while module_aiter_core
# stayed on v11 and every scatter launch failed the ABI check.
if ! python3 -c "import pybind11" 2>/dev/null; then
  python3 -m pip install "pybind11>=3.0.1"
fi
python3 -m pip show pybind11 | grep -E '^(Name|Version):' || true

if [[ "${SCATTER_INSTALL_MODE}" == "aiter_ref" && -n "${AITER_GIT_REF}" ]]; then
  echo "=== install aiter from ref ${AITER_GIT_REF} ==="
  python3 -m pip uninstall -y amd-aiter || true
  rm -rf /app/aiter-test
  git clone --filter=blob:none --no-checkout https://github.com/ROCm/aiter.git /app/aiter-test
  cd /app/aiter-test
  git fetch --filter=blob:none --depth=1 origin "${AITER_GIT_REF}"
  git checkout --detach FETCH_HEAD
  actual_sha="$(git rev-parse HEAD)"
  if [[ "${AITER_GIT_REF}" =~ ^[0-9a-f]{40}$ && "${actual_sha}" != "${AITER_GIT_REF}" ]]; then
    echo "ERROR: aiter checkout does not match ${AITER_GIT_REF}" >&2
    exit 1
  fi
  printf '%s\n' "${actual_sha}" | tee /out/aiter-commit.txt
  git submodule sync
  git submodule update --init --recursive
  MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 python3 setup.py develop
  cd /workspace/ATOM
else
  echo "=== apply vendored scatter_reduce bounds patch ==="
  REPO_ROOT=/workspace/ATOM bash /workspace/ATOM/.github/scripts/apply_aiter_scatter_reduce_bounds_patch.sh
fi

# Drop any JIT blobs so aiter_core and moe_mxfp4_aux rebuild together under
# the same pybind11 (error text from run 38523159059).
for d in "${AITER_JIT_DIR:-}" "${AITER_CACHE_DIR:-}/jit" "${HOME}/.aiter"; do
  [[ -n "${d}" && -d "${d}" ]] || continue
  echo "clearing aiter JIT dir: ${d}"
  rm -rf "${d}"
  mkdir -p "${d}"
done
export AITER_REBUILD=1

python3 - <<'PY'
from pathlib import Path
import aiter
src = Path(aiter.__file__).resolve().parent.parent / (
    "csrc/kernels/mxfp4_moe/moe_aux/moe_scatter_reduce.cuh"
)
text = src.read_text(encoding="utf-8", errors="replace")
assert "sorted_pos >= max_sorted" in text, f"unpatched scatter_reduce at {src}"
print(f"OK patched kernel: {src}")
print(f"aiter={aiter.__file__}")
PY

# Do not pip install -e: /workspace/ATOM is mounted read-only and writing
# atom.egg-info fails (run 38520975962). Prefer source import via PYTHONPATH.
export PYTHONPATH="/workspace/ATOM${PYTHONPATH:+:${PYTHONPATH}}"
python3 -c "import atom, aiter; print('atom', atom.__file__); print('aiter', aiter.__file__)"

echo "=== ATOM MiniMax-M3 scatter_reduce tests ==="
AITER_REBUILD=1 python3 -m pytest -p no:cacheprovider -v -s \
  --junitxml=/out/pytest_atom_scatter.xml \
  /workspace/ATOM/tests/test_minimax_m3_moe_scatter_reduce.py \
  2>&1 | tee /out/pytest_atom_scatter.log

if [[ -f /app/aiter-test/op_tests/test_moe_scatter_reduce.py ]]; then
  T=/app/aiter-test/op_tests/test_moe_scatter_reduce.py
else
  T=/workspace/ATOM/.github/patches/test_moe_scatter_reduce_aiter.py
fi
echo "=== aiter scatter_reduce tests (${T}) ==="
AITER_REBUILD=1 python3 -m pytest -p no:cacheprovider -v -s \
  --junitxml=/out/pytest_aiter_scatter.xml "$T" \
  2>&1 | tee /out/pytest_aiter_scatter.log

python3 /workspace/ATOM/.github/scripts/atomesh/check_scatter_bounds_results.py \
  /out/pytest_atom_scatter.xml /out/pytest_aiter_scatter.xml

echo "=== scatter_bounds verify OK ==="
INNER_EOF
chmod +x "${inner_script}"

# Keep the ATOM checkout :ro — tests import via PYTHONPATH (no editable pip
# write of atom.egg-info into the mount).
docker run -d --name "${container}" \
  --network host --ipc host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_NICE \
  --shm-size=16G \
  -e HOME="${cache}/home" \
  -e XDG_CACHE_HOME="${cache}" \
  -e AITER_CACHE_DIR="${cache}/aiter" \
  -e AITER_JIT_DIR="${cache}/aiter/jit" \
  -e AITER_LOG_LEVEL=WARNING \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e AITER_GIT_REF="${AITER_GIT_REF}" \
  -e SCATTER_INSTALL_MODE="${SCATTER_INSTALL_MODE}" \
  -e PYTHONPATH="/workspace/ATOM${PYTHONPATH:+:${PYTHONPATH}}" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${SCATTER_REPO_ROOT}:/workspace/ATOM:ro" \
  -v "${SCATTER_OUT_DIR}:/out" \
  -v "${cache}:${cache}" \
  -w /workspace/ATOM \
  "${SCATTER_IMAGE}" \
  sleep infinity

docker cp "${inner_script}" "${container}:/tmp/scatter_bounds_inner.sh"
docker exec \
  -e AITER_GIT_REF="${AITER_GIT_REF}" \
  -e SCATTER_INSTALL_MODE="${SCATTER_INSTALL_MODE}" \
  -e PYTHONPATH="/workspace/ATOM" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e AITER_LOG_LEVEL=WARNING \
  -e AMD_SERIALIZE_KERNEL=3 \
  "${container}" bash /tmp/scatter_bounds_inner.sh

echo "=== scatter_bounds rank done ==="
