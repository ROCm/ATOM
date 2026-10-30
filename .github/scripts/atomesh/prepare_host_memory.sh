#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Release clean host caches before the service allocates pinned Store segments.
set -euo pipefail

# pd_submit.sh prefixes case env.common settings with ATOMESH_ENV_.
enabled="${ATOMESH_ENV_ATOMESH_DROP_HOST_CACHES:-${ATOMESH_DROP_HOST_CACHES:-0}}"
[[ "${enabled}" == "1" ]] || exit 0
image="$1"
container="${2}-host-cache-drop"
cleanup_timeout="${ATOMESH_HOST_CACHE_DROP_TIMEOUT:-180}"
if [[ ! "${cleanup_timeout}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[host-memory][FAIL] ATOMESH_HOST_CACHE_DROP_TIMEOUT must be a positive integer" >&2
  exit 2
fi

cleanup() {
  # Bound removal too: a timed-out sync must not strand a helper container.
  timeout --kill-after=5s 20s docker rm -f "${container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[host-memory] sync and drop_caches=3 before service startup (timeout=${cleanup_timeout}s)"
# The Spur service runs as the Slurm uid. Use a short root helper for this
# host sysctl; no model/log mounts or network access are needed.
if timeout --kill-after=10s "${cleanup_timeout}s" \
  docker run --rm --name "${container}" --user 0 --privileged \
    --network none --entrypoint /bin/bash "${image}" -ec '
      snapshot() {
        echo "[host-memory] $1 (kB)"
        awk "/^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SReclaimable|Shmem):/" /proc/meminfo
        for info in /sys/devices/system/node/node[0-9]*/meminfo; do
          [[ -r "${info}" ]] || continue
          awk "/ (MemTotal|MemFree|FilePages|SReclaimable|Shmem):/" "${info}"
        done
      }
      snapshot before
      sync
      echo 3 > /proc/sys/vm/drop_caches
      snapshot after
    '; then
  echo "[host-memory] cache cleanup completed"
else
  rc=$?
  echo "[host-memory][FAIL] cache cleanup failed (rc=${rc}); refusing to start services" >&2
  exit "${rc}"
fi
