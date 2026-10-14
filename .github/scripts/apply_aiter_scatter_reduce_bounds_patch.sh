#!/usr/bin/env bash
# Apply the moe_scatter_reduce max_sorted bounds fix onto the amd-aiter
# source tree that the current Python process will import, then drop the
# develop JIT module so the next mxfp4_moe_scatter_reduce call rebuilds
# with the new ABI.
#
# Intended for CI / atomesh containers where the image still has the
# unpatched scatter epilog. Idempotent when the patch is already applied.
#
# Usage (inside the container, cwd anywhere):
#   bash /workspace/ATOM/.github/scripts/apply_aiter_scatter_reduce_bounds_patch.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-}"
if [[ -z "${REPO_ROOT}" ]]; then
  if [[ -f /.github/scripts/apply_aiter_scatter_reduce_bounds_patch.sh ]]; then
    REPO_ROOT="/"
  elif [[ -f /workspace/ATOM/.github/scripts/apply_aiter_scatter_reduce_bounds_patch.sh ]]; then
    REPO_ROOT="/workspace/ATOM"
  else
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  fi
fi

PATCH="${REPO_ROOT}/.github/patches/aiter_moe_scatter_reduce_bounds.patch"
if [[ ! -f "${PATCH}" ]]; then
  echo "ERROR: patch not found at ${PATCH}" >&2
  exit 1
fi

AITER_PKG="$(python3 -c 'import aiter, os; print(os.path.dirname(aiter.__file__))')"
if [[ -d "${AITER_PKG}/../csrc/kernels/mxfp4_moe" ]]; then
  SRC_ROOT="$(cd "${AITER_PKG}/.." && pwd)"
elif [[ -d "${AITER_PKG}/csrc/kernels/mxfp4_moe" ]]; then
  SRC_ROOT="${AITER_PKG}"
else
  echo "ERROR: cannot find csrc/kernels/mxfp4_moe next to ${AITER_PKG}" >&2
  exit 1
fi

echo "aiter package: ${AITER_PKG}"
echo "aiter sources: ${SRC_ROOT}"
echo "patch: ${PATCH}"

DRY_OUT="$(mktemp)"
set +e
patch --batch --forward -p1 -d "${SRC_ROOT}" --dry-run < "${PATCH}" >"${DRY_OUT}" 2>&1
dry_rc=$?
set -e
if [[ "${dry_rc}" -eq 0 ]]; then
  patch --batch --forward -p1 -d "${SRC_ROOT}" < "${PATCH}"
elif patch --batch --reverse -p1 -d "${SRC_ROOT}" --dry-run < "${PATCH}" >/dev/null 2>&1; then
  echo "patch already applied"
else
  cat "${DRY_OUT}" >&2
  rm -f "${DRY_OUT}"
  exit 1
fi
rm -f "${DRY_OUT}"

# Prove the kernel now has the upper-bound check before clearing caches.
if ! grep -q 'sorted_pos >= max_sorted' \
  "${SRC_ROOT}/csrc/kernels/mxfp4_moe/moe_aux/moe_scatter_reduce.cuh"; then
  echo "ERROR: moe_scatter_reduce.cuh missing max_sorted skip after patch" >&2
  exit 1
fi

for d in \
  "${AITER_JIT_DIR:-}" \
  "${AITER_CACHE_DIR:-}/jit" \
  "${HOME}/.aiter" \
  "/tmp/aiter" \
  ; do
  [[ -n "${d}" && -d "${d}" ]] || continue
  find "${d}" -iname '*moe_mxfp4_aux*' -print -delete 2>/dev/null || true
done

python3 - <<'PY'
import aiter
from pathlib import Path
src = Path(aiter.__file__).resolve().parent.parent / "csrc/kernels/mxfp4_moe/moe_aux/moe_scatter_reduce.cuh"
text = src.read_text(encoding="utf-8", errors="replace")
assert "max_sorted" in text and "sorted_pos >= max_sorted" in text, src
print(f"verified patched scatter_reduce at {src}")
print(f"amd-aiter import path: {aiter.__file__}")
PY

echo "OK: aiter scatter_reduce bounds patch applied; next import rebuilds module_moe_mxfp4_aux"
