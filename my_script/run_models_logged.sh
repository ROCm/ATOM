#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${ATOM_SERVER_LOG_FILE:-${SCRIPT_DIR}/server.log}"

mkdir -p "$(dirname "${LOG_FILE}")"
: > "${LOG_FILE}"
printf '===== model server start %s =====\n' "$(date -Is)" >> "${LOG_FILE}"

exec "${SCRIPT_DIR}/run_models.sh" "$@" >> "${LOG_FILE}" 2>&1
