#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Release clean host caches before the service allocates pinned Store segments.
set -euo pipefail

# pd_submit.sh prefixes case env.common settings with ATOMESH_ENV_.
# 1 refuses to start services when the drop fails; best_effort only warns, for
# nodes that deny the privileged helper (the Store then reclaims the cache
# through its own unprivileged page-cache drop and compaction). Under 1 a host
# whose Docker authorization plugin denies --privileged, as it does every job,
# is recorded in ATOMESH_UNFIT_HOSTS_DIR, for pd_submit.sh to resubmit without.
enabled="${ATOMESH_ENV_ATOMESH_DROP_HOST_CACHES:-${ATOMESH_DROP_HOST_CACHES:-0}}"
[[ "${enabled}" == "1" || "${enabled}" == "best_effort" ]] || exit 0
image="$1"
container="${2}-host-cache-drop"
rank="${3:-}"
# Comma-separated node ranks that drop; unset means every rank. Nodes hosting
# no Store need no clean caches, and some deny the privileged helper.
ranks="${ATOMESH_ENV_ATOMESH_DROP_HOST_CACHES_RANKS:-${ATOMESH_DROP_HOST_CACHES_RANKS:-}}"
if [[ -n "${ranks}" && -n "${rank}" && ",${ranks// /}," != *",${rank},"* ]]; then
  echo "[host-memory] rank ${rank} is not in ATOMESH_DROP_HOST_CACHES_RANKS=${ranks}; skipping cache cleanup"
  exit 0
fi
cleanup_timeout="${ATOMESH_ENV_ATOMESH_HOST_CACHE_DROP_TIMEOUT:-${ATOMESH_HOST_CACHE_DROP_TIMEOUT:-180}}"
if [[ ! "${cleanup_timeout}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[host-memory][FAIL] ATOMESH_HOST_CACHE_DROP_TIMEOUT must be a positive integer" >&2
  exit 2
fi

helper_err="$(mktemp)"
cleanup() {
  # Bound removal too: a timed-out sync must not strand a helper container.
  timeout --kill-after=5s 20s docker rm -f "${container}" >/dev/null 2>&1 || true
  rm -f "${helper_err}"
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
    ' 2>"${helper_err}"; then
  cat "${helper_err}" >&2
  echo "[host-memory] cache cleanup completed"
else
  rc=$?
  cat "${helper_err}" >&2
  if [[ "${enabled}" == "best_effort" ]]; then
    echo "[host-memory][WARN] cache cleanup failed (rc=${rc}); starting services anyway (best_effort)" >&2
    exit 0
  fi
  if [[ -n "${ATOMESH_UNFIT_HOSTS_DIR:-}" ]] && grep -q "authorization denied" "${helper_err}"; then
    mkdir -p "${ATOMESH_UNFIT_HOSTS_DIR}" && touch "${ATOMESH_UNFIT_HOSTS_DIR}/$(hostname)"
    echo "[host-memory][FAIL] $(hostname) denies the privileged helper; recorded in ${ATOMESH_UNFIT_HOSTS_DIR}" >&2
  fi
  echo "[host-memory][FAIL] cache cleanup failed (rc=${rc}); refusing to start services" >&2
  exit "${rc}"
fi
