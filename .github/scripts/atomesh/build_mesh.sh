#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:?Mesh source directory is required}"
log_dir="${2:?Build log directory is required}"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/atomesh-ci-mesh.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT
mkdir -p "${log_dir}"
profile="${ATOMESH_MESH_BUILD_PROFILE:-release}"
case "${profile}" in
  release|ci) ;;
  *) echo "Unsupported Mesh build profile: ${profile}" >&2; exit 2 ;;
esac
source_commit="${ATOMESH_MESH_SOURCE_COMMIT:-$(git -C "${source_dir}" rev-parse HEAD)}"

# The checkout is mounted read-only. Cargo may need to create a lockfile.
python3 - "${source_dir}" "${build_dir}/source" <<'PY'
import shutil
import sys

shutil.copytree(sys.argv[1], sys.argv[2], ignore=shutil.ignore_patterns("target", ".git"))
PY

target_dir="${ATOMESH_MESH_TARGET_DIR:-${TMPDIR:-/tmp}/atomesh-mesh-cache-$(id -u)}"
mkdir -p "${target_dir}"
# Protect the cached binary through the artifact copy as well as compilation.
exec 9> "${target_dir}/.build.lock"
flock 9
# Some CI images install Rust below /root, inaccessible to Spur's service UID.
# Keep a pinned fallback toolchain in the writable build cache for those jobs.
if ! cargo --version >/dev/null 2>&1; then
  export CARGO_HOME="${target_dir}/cargo-home"
  export RUSTUP_HOME="${target_dir}/rustup-home"
  export PATH="${CARGO_HOME}/bin:${PATH}"
  export RUSTUP_TOOLCHAIN="${ATOMESH_MESH_RUST_TOOLCHAIN:-1.94.0}"
  if ! cargo --version >/dev/null 2>&1; then
    rust_arch="$(uname -m)"
    case "${rust_arch}" in
      x86_64|aarch64) ;;
      *) echo "Unsupported Rust build architecture: ${rust_arch}" >&2; exit 2 ;;
    esac
    curl --fail --silent --show-error --location \
      "https://static.rust-lang.org/rustup/dist/${rust_arch}-unknown-linux-gnu/rustup-init" \
      -o "${build_dir}/rustup-init"
    chmod +x "${build_dir}/rustup-init"
    "${build_dir}/rustup-init" -y --no-modify-path --profile minimal \
      --default-toolchain "${RUSTUP_TOOLCHAIN}" >&2
  fi
fi
lock_args=()
if [[ -f "${build_dir}/source/Cargo.lock" ]]; then
  lock_args+=(--locked)
fi
echo "[build] Mesh commit=${source_commit} profile=${profile}" >&2
if ! cargo build --profile "${profile}" --bin atomesh \
  "${lock_args[@]}" \
  --manifest-path "${build_dir}/source/Cargo.toml" \
  --target-dir "${target_dir}" > "${log_dir}/mesh-build.log" 2>&1; then
  cat "${log_dir}/mesh-build.log" >&2
  exit 1
fi
cp "${build_dir}/source/Cargo.lock" "${log_dir}/mesh-Cargo.lock"
cp "${target_dir}/${profile}/atomesh" "${log_dir}/atomesh"
python3 - "${log_dir}" "${source_commit}" "${profile}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
metadata = {
    "commit": sys.argv[2],
    "profile": sys.argv[3],
    "source_dirty": os.environ.get("ATOMESH_MESH_SOURCE_DIRTY", "unknown"),
    "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
    "binary_sha256": hashlib.sha256((output / "atomesh").read_bytes()).hexdigest(),
    "lockfile_sha256": hashlib.sha256((output / "mesh-Cargo.lock").read_bytes()).hexdigest(),
}
(output / "mesh-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
printf '%s\n' "${log_dir}/atomesh"
