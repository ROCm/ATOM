#!/usr/bin/env bash
set -euo pipefail

readonly ATOM_BASE="a526f0d557eeed08396670249af58bd85fa57338"
readonly AITER_BASE="57b7cf03ad32f72247c151cff2933abd11ff77d1"
readonly ATOM_PATCH_SHA256="2cc961463aca0396b4decca2b1ffb03410d988bc1a21e7dfa2066eae3c03193c"
readonly AITER_PATCH_SHA256="f05053b7d9453d5aa0c3030e12ae870a0ac127d1cb121985962baa278b7d0c88"
readonly EXPERIMENTAL_MORI_SHA256="31d6aa6e615007008692030ac3236055fec99427555a6ebf69e0f53571509d83"
readonly BF16_SHA256="196fe1af733b2e62681c779b280829580de440c4917db3a86924bc7a8261ad9f"

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
