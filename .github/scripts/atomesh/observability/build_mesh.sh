#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:?Mesh source directory is required}"
log_dir="${2:?Build log directory is required}"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/atomesh-ci-mesh.XXXXXX")"
mkdir -p "${log_dir}"

# The checkout is mounted read-only. Cargo may need to create a lockfile.
python3 - "${source_dir}" "${build_dir}/source" <<'PY'
import shutil
import sys

shutil.copytree(sys.argv[1], sys.argv[2], ignore=shutil.ignore_patterns("target", ".git"))
PY

target_dir="${ATOMESH_MESH_TARGET_DIR:-${build_dir}/target}"
lock_args=()
if [[ -f "${build_dir}/source/Cargo.lock" ]]; then
  lock_args+=(--locked)
fi
echo "[metrics] building Mesh from the current checkout" >&2
if ! cargo build --profile ci --bin atomesh \
  "${lock_args[@]}" \
  --manifest-path "${build_dir}/source/Cargo.toml" \
  --target-dir "${target_dir}" > "${log_dir}/mesh-build.log" 2>&1; then
  cat "${log_dir}/mesh-build.log" >&2
  exit 1
fi
cp "${build_dir}/source/Cargo.lock" "${log_dir}/mesh-Cargo.lock"
printf '%s\n' "${target_dir}/ci/atomesh"
