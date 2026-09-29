#!/usr/bin/env bash
# Host-side staging for AITER's tuned-config directory, run on each compute node
# before the service container, on BOTH arms of the seeding A/B.
#
# AITER loads its tuned GEMM tables through jit/core.py:mp_lock, which takes a
# file_baton on a lock file next to each table:
#
#   File "/app/aiter-test/aiter/jit/utils/file_baton.py", line 51, in try_acquire
#     self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
#   PermissionError: [Errno 13] Permission denied: '/tmp/aiter_configs/bf16_tuned_gemm.csv.lock'
#
# The path is hardcoded inside the image, and the image can ship that directory
# root-owned -- confirmed on
# ubuntu24.04_py3.12_pytorch_release_2.10.0_kimi_k3_agentic_0911, where it kills
# model load outright (Slurm job 4819: "Engine Core: load model runner failed").
# The nightly tags do not hit it, so this is image-specific, not a regression.
#
# Unlike stage_aot_cache.sh this is NOT gated on ATOMESH_SEED_AOT_CACHE. The
# failure is in the image's config layer and has nothing to do with seeding, so
# the unseeded control arm hits it too; gating would fix one arm and leave the
# comparison with nothing to compare against.
#
# Copy the tables out as root, make the copy writable by the serving uid, and
# let the caller bind-mount it read-write over /tmp/aiter_configs. That keeps
# the image's tuned tables -- wiping them with a bare tmpfs would silently
# retune and confound any benchmark run on top of it.
set -euo pipefail

image="${1:?Docker image is required}"
stage_base="${2:-${ATOMESH_AITER_CONFIG_STAGE_BASE:-${TMPDIR:-/tmp}}}"

# Where the image puts them, and where aiter will look inside the container.
config_path="${ATOMESH_AITER_CONFIG_PATH:-/tmp/aiter_configs}"

warn() { echo "[stage-aiter-configs] $*" >&2; }

# Exit 0 with no stdout on every failure, which the caller reads as "no stage"
# and leaves the container exactly as it is today. This staging is load-bearing
# on the affected images, but refusing to launch here would turn a container
# that might still start into one that certainly does not.
give_up() {
  warn "WARNING: $*; ${config_path} will be used as the image ships it"
  exit 0
}

command -v docker >/dev/null 2>&1 || give_up "docker is not available on $(hostname)"

# Key by image ID so a second job on this node reuses the tree, and so a new
# image never silently inherits the previous one's tables.
image_id="$(docker image inspect -f '{{.Id}}' "${image}" 2>/dev/null || true)"
if [[ -n "${image_id}" ]]; then
  image_key="${image_id#sha256:}"
  image_key="${image_key:0:16}"
else
  image_key="$(printf '%s' "${image}" | tr -c 'A-Za-z0-9_.-' '-')"
fi

stage_dir="${stage_base}/atomesh-aiter-configs-$(id -u)/${image_key}"
marker="${stage_dir}/.staged.json"

# Reuse is by manifest only. Unlike the AOT stage, an empty tree is a perfectly
# good result here -- an image that ships no tuned tables still needs a writable
# directory -- so the marker, not a file count, is what says staging ran.
if [[ -s "${marker}" ]]; then
  warn "reusing stage ${stage_dir}"
  printf '%s\n' "${stage_dir}"
  exit 0
fi

mkdir -p "${stage_dir}" || give_up "cannot create ${stage_dir}"

warn "extracting ${config_path} from ${image} into ${stage_dir}"
if ! docker run --rm --user 0:0 \
  -e ATOMESH_STAGE_IMAGE="${image}" \
  -e ATOMESH_STAGE_PATH="${config_path}" \
  -e ATOMESH_STAGE_UID="$(id -u)" \
  -e ATOMESH_STAGE_GID="$(id -g)" \
  -v "${stage_dir}:/stage" \
  "${image}" \
  bash -lc '
set -euo pipefail

src="${ATOMESH_STAGE_PATH}"
rm -rf /stage/configs
mkdir -p /stage/configs

if [[ -d "${src}" ]]; then
  # Record what the image actually ships, so the next failure of this shape can
  # be read straight out of the job log instead of re-derived.
  echo "image ships ${src}: $(ls -ld -- "${src}")" >&2
  cp -a "${src}/." /stage/configs/ 2>/dev/null || true
else
  echo "image ships no ${src}; staging an empty writable directory" >&2
fi

# Writable, not merely readable: aiter creates <table>.lock siblings here on
# every load, which is the exact call that fails today.
chmod -R a+rwX /stage/configs
if [[ -n "${ATOMESH_STAGE_UID:-}" ]]; then
  chown -R "${ATOMESH_STAGE_UID}:${ATOMESH_STAGE_GID:-${ATOMESH_STAGE_UID}}" \
    /stage/configs 2>/dev/null || true
fi

# Stale locks would deadlock the baton for a full timeout before it gives up.
# Any lock present in an image layer is by definition orphaned.
find /stage/configs -name "*.lock" -delete 2>/dev/null || true

files="$(find /stage/configs -type f | wc -l)"
printf "{\"image\":\"%s\",\"path\":\"%s\",\"files\":%s}\n" \
  "${ATOMESH_STAGE_IMAGE}" "${src}" "${files}" > /stage/.staged.json
chmod a+r /stage/.staged.json
echo "staged ${files} aiter config files from ${src}" >&2
' >&2; then
  give_up "could not extract ${config_path} from ${image}"
fi

if [[ ! -s "${marker}" ]]; then
  give_up "the staging container wrote no manifest under ${stage_dir}"
fi

warn "stage ready at ${stage_dir}"
printf '%s\n' "${stage_dir}"
