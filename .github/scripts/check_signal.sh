#!/usr/bin/env bash

set -euo pipefail

TARGET_SHA="${GITHUB_SHA:-${1:-}}"
if [ -z "${TARGET_SHA}" ]; then
  echo "GITHUB_SHA is not set and no SHA argument was provided."
  exit 1
fi

ARTIFACT_NAME="checks-signal-${TARGET_SHA}"
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"

for i in $(seq 1 "${MAX_RETRIES}"); do
  ATTEMPT_DIR="$(mktemp -d)"
  echo "Attempt ${i}: downloading artifact ${ARTIFACT_NAME}..."

  if gh run download \
    --repo "${GITHUB_REPOSITORY}" \
    --name "${ARTIFACT_NAME}" \
    --dir "${ATTEMPT_DIR}"; then
    SIGNAL_FILE="${ATTEMPT_DIR}/checks_signal.txt"
    if [ -f "${SIGNAL_FILE}" ]; then
      echo "Artifact ${ARTIFACT_NAME} downloaded successfully."
      SIGNAL="$(head -n 1 "${SIGNAL_FILE}")"
      if [ "${SIGNAL}" = "success" ]; then
        echo "Pre Checkin passed, continuing workflow."
        rm -rf "${ATTEMPT_DIR}"
        exit 0
      fi

      if [ "${SIGNAL}" = "failure" ]; then
        echo "Pre Checkin failed, skipping workflow. Details:"
        tail -n +2 "${SIGNAL_FILE}" || true
        rm -rf "${ATTEMPT_DIR}"
        exit 78
      fi

      echo "Unknown signal '${SIGNAL}' in ${SIGNAL_FILE}."
      rm -rf "${ATTEMPT_DIR}"
      exit 1
    fi
  fi

  rm -rf "${ATTEMPT_DIR}"
  echo "Artifact not found yet, retrying in ${RETRY_DELAY_SECONDS}s..."
  sleep "${RETRY_DELAY_SECONDS}"
done

echo "Failed to download ${ARTIFACT_NAME} after ${MAX_RETRIES} attempts."
exit 1
