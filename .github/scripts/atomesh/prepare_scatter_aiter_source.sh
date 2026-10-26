#!/usr/bin/env bash
# Source this file so both P/D workers inherit the pinned aiter import path.
set -euo pipefail
scatter_ref="${1:?full aiter commit SHA required}"
if [[ ! "${scatter_ref}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: ATOM_AITER_SCATTER_REF must be a full commit SHA" >&2
  return 1
fi
scatter_source="$(mktemp -d /tmp/atomesh-scatter-aiter.XXXXXX)"
git -C "${scatter_source}" init -q
git -C "${scatter_source}" remote add origin https://github.com/ROCm/aiter.git
git -C "${scatter_source}" fetch --depth=1 origin "${scatter_ref}"
git -C "${scatter_source}" checkout --detach FETCH_HEAD
[[ "$(git -C "${scatter_source}" rev-parse HEAD)" == "${scatter_ref}" ]]
git -C "${scatter_source}" submodule update --init --recursive --depth=1
export PYTHONPATH="${scatter_source}${PYTHONPATH:+:${PYTHONPATH}}"
export AITER_REBUILD=1
# A checkout is a supported develop-mode source tree; all dependencies and
# the HIP toolchain come from the image, so no global/non-root pip mutation.
python3 - "${scatter_source}" "${scatter_ref}" <<'PY'
import sys
from pathlib import Path
import aiter
root = Path(sys.argv[1]).resolve()
assert Path(aiter.__file__).resolve().parent == root / "aiter", aiter.__file__
src = root / "csrc/kernels/mxfp4_moe/moe_aux/moe_scatter_reduce.cuh"
assert "sorted_pos >= max_sorted" in src.read_text()
print(f"[scatter-bounds] aiter_sha={sys.argv[2]} source={aiter.__file__}")
PY
