#!/usr/bin/env bash
# Explicit CI build stage, run on the router's host before the service container.
set -euo pipefail

repo_root="${1:?Repository root is required}"
run_dir="${2:?Run directory is required}"
image="${3:?Build image is required}"
env_file="${4:?Container environment file is required}"
job_id="${5:?Job ID is required}"

if [[ -n "${ATOMESH_MESH_BINARY:-}" ]]; then
  printf '%s\n' "${ATOMESH_MESH_BINARY}"
  exit 0
fi
if [[ "${ATOMESH_BUILD_MESH:-false}" != "true" ]]; then
  printf '%s\n' /app/ATOM/atom/mesh/target/release/atomesh
  exit 0
fi

artifact_dir="${run_dir}/mesh-build"
container_dir="/run_logs/slurm_job-${job_id}/mesh-build"
# The same job's benchmark and eval phases consume the same build artifact.
if [[ ! -x "${artifact_dir}/atomesh" || ! -f "${artifact_dir}/mesh-build.json" ]]; then
  cache_dir="${ATOMESH_MESH_TARGET_DIR:-${TMPDIR:-/tmp}/atomesh-mesh-cache-$(id -u)}"
  mkdir -p "${artifact_dir}" "${cache_dir}"
  source_commit="$(git -C "${repo_root}" rev-parse HEAD)"
  source_dirty=false
  if [[ -n "$(git -C "${repo_root}" status --porcelain -- atom/mesh)" ]]; then
    source_dirty=true
  fi
  echo "[setup] Building Mesh before launching services" >&2
  docker run --rm --user "$(id -u):$(id -g)" \
    --env-file "${env_file}" \
    -e ATOMESH_MESH_SOURCE_COMMIT="${source_commit}" \
    -e ATOMESH_MESH_SOURCE_DIRTY="${source_dirty}" \
    -e ATOMESH_MESH_TARGET_DIR=/mesh-cache \
    -v "${repo_root}:/workspace/ATOM:ro" \
    -v "${artifact_dir}:${container_dir}" \
    -v "${cache_dir}:/mesh-cache" \
    "${image}" \
    bash -lc 'exec bash /workspace/ATOM/.github/scripts/atomesh/build_mesh.sh "$1" "$2"' \
      -- /workspace/ATOM/atom/mesh "${container_dir}" >&2
fi
printf '%s\n' "${container_dir}/atomesh"
