#!/usr/bin/env bash

set -euo pipefail

MODEL_ID="${1:-}"
TARGET_DIR="${2:-}"

if [ -z "${MODEL_ID}" ] || [ -z "${TARGET_DIR}" ]; then
  echo "Usage: $0 <model-id> <target-dir>"
  exit 1
fi

MODEL_DOWNLOAD_TIMEOUT="${MODEL_DOWNLOAD_TIMEOUT:-2h}"
MODEL_LOCK_WAIT_SECONDS="${MODEL_LOCK_WAIT_SECONDS:-1800}"
MODEL_LOCK_POLL_INTERVAL="${MODEL_LOCK_POLL_INTERVAL:-30}"
MODEL_PROGRESS_INTERVAL="${MODEL_PROGRESS_INTERVAL:-60}"
LOCK_ROOT="${MODEL_LOCK_ROOT:-/models/.locks}"
STAGING_ROOT="${MODEL_STAGING_ROOT:-/models/.staging}"
SAFE_KEY="$(printf '%s' "${MODEL_ID}" | sed 's#[^A-Za-z0-9._-]#_#g')"
LOCK_FILE="${LOCK_ROOT}/${SAFE_KEY}.flock"
LOCK_MARKER_FILE="${LOCK_ROOT}/${SAFE_KEY}.lock"
READY_MARKER_FILE="${TARGET_DIR}/.atom_download_complete"
STAGING_DIR="${STAGING_ROOT}/${SAFE_KEY}.$$"

LOCK_ACQUIRED=0
DOWNLOAD_PID=""
PROGRESS_PID=""

log() {
  printf '[model-download] %s\n' "$*"
}

print_dir_snapshot() {
  local dir="$1"
  if [ ! -d "${dir}" ]; then
    log "Directory does not exist: ${dir}"
    return
  fi

  local size
  local file_count
  size="$(du -sh "${dir}" 2>/dev/null | awk '{print $1}')"
  file_count="$(find "${dir}" -type f 2>/dev/null | wc -l | tr -d ' ')"
  log "Directory snapshot for ${dir}: size=${size:-0} files=${file_count:-0}"
  find "${dir}" -maxdepth 2 -type f 2>/dev/null | sort | head -20 | sed 's/^/[model-download]   /' || true
}

has_model_payload() {
  local dir="$1"
  find "${dir}" -type f ! -name 'config.json' ! -name '.atom_download_complete' -print -quit 2>/dev/null | grep -q .
}

is_model_ready() {
  local dir="$1"
  [ -f "${dir}/config.json" ] && ([ -f "${dir}/.atom_download_complete" ] || has_model_payload "${dir}")
}

start_progress_reporter() {
  (
    while true; do
      if [ -d "${STAGING_DIR}" ]; then
        local_size="$(du -sh "${STAGING_DIR}" 2>/dev/null | awk '{print $1}')"
        local_files="$(find "${STAGING_DIR}" -type f 2>/dev/null | wc -l | tr -d ' ')"
        log "Download in progress: staging_dir=${STAGING_DIR} size=${local_size:-0} files=${local_files:-0}"
      else
        log "Download in progress: staging directory not created yet"
      fi
      sleep "${MODEL_PROGRESS_INTERVAL}"
    done
  ) &
  PROGRESS_PID="$!"
}

cleanup() {
  local exit_code=$?

  if [ -n "${PROGRESS_PID}" ] && kill -0 "${PROGRESS_PID}" 2>/dev/null; then
    kill "${PROGRESS_PID}" 2>/dev/null || true
    wait "${PROGRESS_PID}" 2>/dev/null || true
  fi

  if [ -n "${DOWNLOAD_PID}" ] && kill -0 "${DOWNLOAD_PID}" 2>/dev/null; then
    log "Stopping unfinished hf download process ${DOWNLOAD_PID}"
    kill "${DOWNLOAD_PID}" 2>/dev/null || true
    wait "${DOWNLOAD_PID}" 2>/dev/null || true
  fi

  rm -rf "${STAGING_DIR}" 2>/dev/null || true

  if [ "${LOCK_ACQUIRED}" -eq 1 ]; then
    rm -f "${LOCK_MARKER_FILE}" 2>/dev/null || true
  fi

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

mkdir -p "${LOCK_ROOT}" "${STAGING_ROOT}" "$(dirname "${TARGET_DIR}")"

find "${STAGING_ROOT}" -maxdepth 1 -type d -name "${SAFE_KEY}.*" -mmin +1440 -exec rm -rf {} + 2>/dev/null || true

exec 9>"${LOCK_FILE}"

deadline_epoch=$(( $(date +%s) + MODEL_LOCK_WAIT_SECONDS ))
while ! flock -n 9; do
  now_epoch="$(date +%s)"
  remaining=$(( deadline_epoch - now_epoch ))
  if [ "${remaining}" -le 0 ]; then
    log "Timed out waiting ${MODEL_LOCK_WAIT_SECONDS}s for lock ${LOCK_MARKER_FILE}"
    exit 1
  fi

  log "Waiting for lock ${LOCK_MARKER_FILE}. Remaining ${remaining}s"
  if [ -f "${LOCK_MARKER_FILE}" ]; then
    sed 's/^/[model-download] lock-info: /' "${LOCK_MARKER_FILE}" || true
  fi
  sleep "${MODEL_LOCK_POLL_INTERVAL}"
done

LOCK_ACQUIRED=1
cat > "${LOCK_MARKER_FILE}" <<EOF
model_id=${MODEL_ID}
target_dir=${TARGET_DIR}
hostname=$(hostname)
pid=$$
started_at=$(date -u +%FT%TZ)
EOF

log "Acquired download lock for ${MODEL_ID}"

if is_model_ready "${TARGET_DIR}"; then
  log "Model cache hit for ${MODEL_ID}. Skipping download."
  print_dir_snapshot "${TARGET_DIR}"
  exit 0
fi

if [ -d "${TARGET_DIR}" ]; then
  log "Found incomplete model directory. Removing stale target ${TARGET_DIR}"
  print_dir_snapshot "${TARGET_DIR}"
  rm -rf "${TARGET_DIR}"
fi

mkdir -p "${STAGING_DIR}"
log "Downloading ${MODEL_ID} to staging directory ${STAGING_DIR}"
log "Download timeout is ${MODEL_DOWNLOAD_TIMEOUT}"

start_progress_reporter

(
  export HF_HUB_ENABLE_HF_TRANSFER=1
  timeout --signal=TERM --kill-after=5m "${MODEL_DOWNLOAD_TIMEOUT}" \
    hf download "${MODEL_ID}" --local-dir "${STAGING_DIR}"
) &
DOWNLOAD_PID="$!"
wait "${DOWNLOAD_PID}"
DOWNLOAD_PID=""

if [ -n "${PROGRESS_PID}" ] && kill -0 "${PROGRESS_PID}" 2>/dev/null; then
  kill "${PROGRESS_PID}" 2>/dev/null || true
  wait "${PROGRESS_PID}" 2>/dev/null || true
fi
PROGRESS_PID=""

if ! is_model_ready "${STAGING_DIR}"; then
  log "Downloaded content failed readiness check for ${MODEL_ID}"
  print_dir_snapshot "${STAGING_DIR}"
  exit 1
fi

mv "${STAGING_DIR}" "${TARGET_DIR}"
cat > "${READY_MARKER_FILE}" <<EOF
model_id=${MODEL_ID}
downloaded_at=$(date -u +%FT%TZ)
hostname=$(hostname)
EOF

log "Model download completed successfully for ${MODEL_ID}"
print_dir_snapshot "${TARGET_DIR}"
