#!/usr/bin/env bash
# Explicit CI staging step, run on each compute node's host before the service
# container.
#
# The image ships AITER's AOT artifacts -- the FlyDSL kernel cache and the
# prebuilt CK modules -- as mode 0600 root:root. Verified on the pinned CI image
# ubuntu24.04_py3.12_pytorch_release_2.10.0_kimi_k3_agentic_0911: 10032 .pkl
# payloads, 8.5G, every one of them 0600 root:root, plus 244 prebuilt modules.
#
# The Spur path reaches the service container through run_container_rank, which
# passes --user "$(id -u):$(id -g)", so the server runs as the submitting Slurm
# uid and can read none of it. An in-container `cp -a` therefore fails on every
# payload, and because seeding must never be fatal the failure is discarded: the
# run starts with an empty kernel cache and JIT-compiles each tuned GEMM inside
# the first prefill forward that selects it.
#
# Only the host can fix this, because only the host can start a container as
# root and read those payloads. Extract them once per node into a readable
# staging tree and hand the path to the service container.
set -euo pipefail

image="${1:?Docker image is required}"
stage_base="${2:-${ATOMESH_AOT_CACHE_STAGE_BASE:-${TMPDIR:-/tmp}}}"

warn() { echo "[stage-aot] $*" >&2; }

# Seeding is an optimization, never a precondition for serving. Every failure
# below warns and exits 0 with no stdout, which the caller reads as "no stage".
give_up() {
  warn "WARNING: $*; tuned GEMMs will be JIT-compiled during serving"
  exit 0
}

command -v docker >/dev/null 2>&1 || give_up "docker is not available on $(hostname)"

# The local image ID is exact and stable, unlike a floating tag, so a node that
# has already staged this image reuses the tree instead of re-extracting 8GB.
image_id="$(docker image inspect -f '{{.Id}}' "${image}" 2>/dev/null || true)"
if [[ -n "${image_id}" ]]; then
  image_key="${image_id#sha256:}"
  image_key="${image_key:0:16}"
else
  # Not pulled yet, or inspect is unavailable: fall back to a slug of the ref so
  # staging still works, just without cross-tag sharing.
  image_key="$(printf '%s' "${image}" | tr -c 'A-Za-z0-9_.-' '-')"
fi

stage_dir="${stage_base}/atomesh-flydsl-stage-$(id -u)/${image_key}"
marker="${stage_dir}/.staged.json"

if [[ -s "${marker}" ]]; then
  staged_kernels="$(sed -n 's/.*"kernels":[[:space:]]*\([0-9]\+\).*/\1/p' "${marker}")"
  if [[ -n "${staged_kernels}" && "${staged_kernels}" -gt 0 ]]; then
    warn "reusing stage ${stage_dir} (${staged_kernels} kernels)"
    printf '%s\n' "${stage_dir}"
    exit 0
  fi
  warn "discarding stage ${stage_dir}: it recorded no kernels"
  rm -f "${marker}" 2>/dev/null || true
fi

mkdir -p "${stage_dir}" || give_up "cannot create ${stage_dir}"

warn "extracting AOT cache from ${image} into ${stage_dir}"
if ! docker run --rm --user 0:0 \
  -e ATOMESH_STAGE_IMAGE="${image}" \
  -e ATOMESH_STAGE_UID="$(id -u)" \
  -e ATOMESH_STAGE_GID="$(id -g)" \
  -v "${stage_dir}:/stage" \
  "${image}" \
  bash -lc '
set -euo pipefail

# find_spec locates the package without importing it, so this costs nothing and
# cannot initialize the GPU. Same technique pd_server_atom.sh uses.
aiter_root="$(python3 -c "import importlib.util, os, sys
spec = importlib.util.find_spec(\"aiter\")
sys.stdout.write(os.path.dirname(spec.origin) if spec and spec.origin else \"\")")"
if [[ -z "${aiter_root}" || ! -d "${aiter_root}/jit/flydsl_cache" ]]; then
  echo "no AOT flydsl cache under ${aiter_root:-<aiter not found>}" >&2
  exit 3
fi

rm -rf /stage/kernels /stage/aiter-jit
mkdir -p /stage/kernels /stage/aiter-jit
cp -a "${aiter_root}/jit/flydsl_cache/." /stage/kernels/
# The prebuilt CK modules are shipped with the same ownership and are hidden the
# same way once AITER_JIT_DIR points at a fresh per-worker root.
if [[ -d "${aiter_root}/jit" ]]; then
  cp -a "${aiter_root}/jit/." /stage/aiter-jit/ 2>/dev/null || true
fi

# Readable by whichever uid the service container ends up as, on either path.
chmod -R a+rX /stage/kernels /stage/aiter-jit

# Hand the tree to the uid the service container will run as, so that uid can
# also prune or re-stage it later. The chmod above is what actually makes the
# payloads readable, so this is best effort and a failure costs nothing.
# It deliberately does NOT enable hardlinking out of the stage: the service
# container sees this tree as a bind mount, and Linux refuses a hardlink across
# mount points even when both sides report the same st_dev. Seeding copies.
if [[ -n "${ATOMESH_STAGE_UID:-}" ]]; then
  chown -R "${ATOMESH_STAGE_UID}:${ATOMESH_STAGE_GID:-${ATOMESH_STAGE_UID}}" \
    /stage/kernels /stage/aiter-jit 2>/dev/null || true
fi

# Count payloads, never files: the world-readable 0-byte .lock siblings would
# otherwise make an empty stage look like a full one.
kernels="$(find /stage/kernels -name "*.pkl" -size +0c | wc -l)"
modules="$(find /stage/aiter-jit -name "module_*.so" | wc -l)"
bytes="$(du -sb /stage/kernels 2>/dev/null | cut -f1)"
printf "{\"image\":\"%s\",\"kernels\":%s,\"modules\":%s,\"bytes\":%s}\n" \
  "${ATOMESH_STAGE_IMAGE}" "${kernels}" "${modules}" "${bytes:-0}" > /stage/.staged.json
chmod a+r /stage/.staged.json
echo "staged ${kernels} kernels, ${modules} prebuilt modules (${bytes:-0} bytes)" >&2
' >&2; then
  give_up "could not extract the AOT cache from ${image}"
fi

if [[ ! -s "${marker}" ]]; then
  give_up "the staging container wrote no manifest under ${stage_dir}"
fi
staged_kernels="$(sed -n 's/.*"kernels":[[:space:]]*\([0-9]\+\).*/\1/p' "${marker}")"
if [[ -z "${staged_kernels}" || "${staged_kernels}" -eq 0 ]]; then
  give_up "the image exposed no readable kernel payloads"
fi

warn "stage ready at ${stage_dir} (${staged_kernels} kernels)"
printf '%s\n' "${stage_dir}"
