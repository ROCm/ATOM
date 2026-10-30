#!/usr/bin/env bash
# Source inside the job container before importing AITER or preparing JIT caches.
# The MXFP4 recipe needs unmerged AITER kernels; pin a full commit, never a branch.
create_m31_aiter_venv() {
  local venv="$1"
  python3 -m venv --system-site-packages "${venv}" || return $?
  # --system-site-packages alone loses dependencies when the image itself uses
  # a venv. Include its site directories (and editable installs) explicitly.
  python3 - "${venv}" <<'PYENV' || return $?
from pathlib import Path
import site
import subprocess
import sys

venv = Path(sys.argv[1])
purelib = subprocess.check_output(
    [str(venv / "bin/python"), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
    text=True,
).strip()
paths = site.getsitepackages()
(Path(purelib) / "atomesh_image.pth").write_text(
    "import site; " + "; ".join(f"site.addsitedir({path!r})" for path in paths) + "\n"
)
PYENV

}

setup_m31_aiter() {
  local ref="${ATOMESH_M31_AITER_REF:-}"
  [[ -n "${ref}" ]] || return 0
  if [[ ! "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: ATOMESH_M31_AITER_REF must be a full commit SHA" >&2
    return 2
  fi
  local root source venv actual
  root="$(mktemp -d /tmp/atomesh-m31-aiter.XXXXXXXX)" || return $?
  source="${root}/source"
  venv="${root}/venv"
  git init -q "${source}" || return $?
  git -C "${source}" remote add origin https://github.com/ROCm/aiter.git || return $?
  git -C "${source}" fetch --depth=1 origin "${ref}" || return $?
  git -C "${source}" checkout --detach FETCH_HEAD || return $?
  actual="$(git -C "${source}" rev-parse HEAD)" || return $?
  [[ "${actual}" == "${ref}" ]] || return 2
  git -C "${source}" submodule update --init --recursive --depth=1 || return $?
  create_m31_aiter_venv "${venv}" || return $?
  export PATH="${venv}/bin:${PATH}"
  # Install only into this job's venv. Keep the image's torch/ROCm stack.
  "${venv}/bin/python" -m pip install 'pybind11>=3.0.1' 'flydsl==0.4.0' || return $?
  (
    cd "${source}" || exit 1
    AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 AITER_MAX_JOBS=16 \
      "${venv}/bin/python" -m pip install --no-deps --no-build-isolation -e .
  ) || return $?
  # Prefer this source over the image's editable-install path.
  export PYTHONPATH="${source}${PYTHONPATH:+:${PYTHONPATH}}"
  export AITER_MAX_JOBS=16
  # An image's AOT/JIT modules do not describe this revision. The normal worker
  # cache setup below now sees this source, which compiles its own modules.
  unset ATOMESH_FLYDSL_AOT_DIR
  ATOMESH_M31_AITER_SOURCE="${source}" python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ATOMESH_M31_AITER_SOURCE"]).resolve()
import aiter
from aiter.ops.flydsl.pa_decode_mxfp4 import pa_decode_mxfp4
from aiter.ops.flydsl import (
    minimax_m3_index_score_fp4_flydsl,
    minimax_m3_index_score_prefill_fp4_flydsl,
)
from aiter import fused_qknorm_idxrqknorm

assert Path(aiter.__file__).resolve().is_relative_to(root), aiter.__file__
assert all(callable(f) for f in (
    pa_decode_mxfp4,
    minimax_m3_index_score_fp4_flydsl,
    minimax_m3_index_score_prefill_fp4_flydsl,
    fused_qknorm_idxrqknorm,
))
print(f"[m31-runtime] AITER source={aiter.__file__}", flush=True)
PY
  local rc=$?
  [[ "${rc}" == 0 ]] || return "${rc}"
  echo "[m31-runtime] pinned AITER ${ref}; isolated venv ${venv}"
}
setup_m31_aiter
