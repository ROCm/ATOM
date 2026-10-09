#!/usr/bin/env bash
set -euo pipefail

readonly ATOM_BASE="a526f0d557eeed08396670249af58bd85fa57338"
readonly AITER_BASE="e6ded2168ef43111c57d591dd310eccdfc27aaf1"
readonly ATOM_PATCH_SHA256="16731df5e737abfd0ddf1323809c7ee9f7dde4dfa64901930a773b32068a534a"
readonly AITER_PATCH_SHA256="07301f9fa69d8915f2c66fb821dbaf3c0a12909b497a95cfd56e93c180bfbeb6"
readonly EXPERIMENTAL_MORI_SHA256="5760a68d04d60b3f3866ab10d41ad22b534904c75f472bb0f1a431380b07b560"
readonly BF16_SHA256="196fe1af733b2e62681c779b280829580de440c4917db3a86924bc7a8261ad9f"
readonly MIN_TRITON_VERSION="3.9"

EXPERIMENTAL_MORI=0
if [[ ${1:-} == "--experimental-mori" ]]; then
  EXPERIMENTAL_MORI=1
  shift
fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 [--experimental-mori] <ATOM checkout> <AITER checkout>" >&2
  exit 2
fi

ATOM_ROOT=$(realpath "$1")
AITER_ROOT=$(realpath "$2")
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ATOM_PATCH="${HERE}/atom.patch"
AITER_PATCH="${HERE}/aiter.patch"
EXPERIMENTAL_MORI_PATCH="${HERE}/experimental_mori_aiter.patch"
BF16_SOURCE="${HERE}/k3_bf16_hot_gfx1250_production_safe.csv"
BF16_DEST="${AITER_ROOT}/aiter/configs/k3_bf16_hot_gfx1250_production_safe.csv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

verify_triton() {
  "${PYTHON_BIN}" - "${MIN_TRITON_VERSION}" <<'PY'
import sys

minimum_text = sys.argv[1]
try:
    from packaging.version import InvalidVersion, Version
except ImportError as exc:
    raise SystemExit(
        "packaging is required for the Triton version preflight; "
        "install/upgrade packaging before applying this recipe"
    ) from exc

try:
    import triton
except ImportError as exc:
    raise SystemExit(
        "Triton is not importable; install Triton >= "
        f"{minimum_text}. Do not apply the retired gather kernel workaround."
    ) from exc

version_text = str(getattr(triton, "__version__", "unknown"))
path = str(getattr(triton, "__file__", "unknown"))
print(f"Triton version: {version_text}")
print(f"Triton path: {path}")
try:
    actual = Version(version_text)
except InvalidVersion as exc:
    raise SystemExit(f"cannot parse Triton version {version_text!r}") from exc
minimum = Version(minimum_text)
if actual < minimum:
    raise SystemExit(
        f"Triton >= {minimum} is required, found {actual}. "
        "Upgrade Triton; do not apply the retired #6120 kernel workaround."
    )
PY
}

verify_repo() {
  local name=$1 root=$2 expected=$3 actual
  git -C "${root}" rev-parse --is-inside-work-tree >/dev/null
  actual=$(git -C "${root}" rev-parse HEAD)
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${name}: expected base ${expected}, found ${actual}" >&2
    return 1
  fi
}

verify_sha256() {
  local expected=$1 file=$2 actual
  actual=$(sha256sum "${file}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "checksum mismatch: ${file}" >&2
    return 1
  fi
}

patch_mode() {
  local root=$1 patch=$2
  if git -C "${root}" apply --check "${patch}" 2>/dev/null; then
    printf '%s\n' pending
  elif git -C "${root}" apply --reverse --check "${patch}" 2>/dev/null; then
    printf '%s\n' applied
  else
    echo "patch is neither cleanly applicable nor already applied: ${patch}" >&2
    return 1
  fi
}

# Complete every preflight before changing either checkout.
verify_triton
verify_repo ATOM "${ATOM_ROOT}" "${ATOM_BASE}"
verify_repo AITER "${AITER_ROOT}" "${AITER_BASE}"
verify_sha256 "${ATOM_PATCH_SHA256}" "${ATOM_PATCH}"
verify_sha256 "${AITER_PATCH_SHA256}" "${AITER_PATCH}"
verify_sha256 "${EXPERIMENTAL_MORI_SHA256}" "${EXPERIMENTAL_MORI_PATCH}"
verify_sha256 "${BF16_SHA256}" "${BF16_SOURCE}"
ATOM_MODE=$(patch_mode "${ATOM_ROOT}" "${ATOM_PATCH}")
if git -C "${AITER_ROOT}" apply --reverse --check \
  "${EXPERIMENTAL_MORI_PATCH}" 2>/dev/null; then
  AITER_MODE=applied
  EXPERIMENTAL_MORI_MODE=applied
else
  AITER_MODE=$(patch_mode "${AITER_ROOT}" "${AITER_PATCH}")
  EXPERIMENTAL_MORI_MODE=absent
fi
if [[ -e "${BF16_DEST}" ]] && ! cmp -s "${BF16_SOURCE}" "${BF16_DEST}"; then
  echo "refusing to replace different config: ${BF16_DEST}" >&2
  exit 1
fi

if [[ "${ATOM_MODE}" == pending ]]; then
  git -C "${ATOM_ROOT}" apply "${ATOM_PATCH}"
fi
if [[ "${AITER_MODE}" == pending ]]; then
  git -C "${AITER_ROOT}" apply "${AITER_PATCH}"
fi
if [[ "${EXPERIMENTAL_MORI}" == 1 && "${EXPERIMENTAL_MORI_MODE}" == absent ]]; then
  EXPERIMENTAL_MORI_MODE=$(patch_mode "${AITER_ROOT}" "${EXPERIMENTAL_MORI_PATCH}")
  if [[ "${EXPERIMENTAL_MORI_MODE}" == pending ]]; then
    git -C "${AITER_ROOT}" apply "${EXPERIMENTAL_MORI_PATCH}"
    EXPERIMENTAL_MORI_MODE=applied
  fi
fi

install -m 0644 "${BF16_SOURCE}" "${BF16_DEST}"

# Reverse checks prove the complete patches, rather than partial hunks, are present.
git -C "${ATOM_ROOT}" apply --reverse --check "${ATOM_PATCH}"
if [[ "${EXPERIMENTAL_MORI_MODE}" == applied ]]; then
  git -C "${AITER_ROOT}" apply --reverse --check "${EXPERIMENTAL_MORI_PATCH}"
else
  git -C "${AITER_ROOT}" apply --reverse --check "${AITER_PATCH}"
fi

printf 'ATOM: %s\nAITER required: %s\nExperimental MegaMoE overlay: %s\nBF16 config: %s\n' \
  "${ATOM_MODE}" "${AITER_MODE}" "${EXPERIMENTAL_MORI_MODE}" "${BF16_DEST}"
