#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_dir="${RUN_DIR:?RUN_DIR is required}"
phase="${ATOMESH_EXECUTION_PHASE:-combined}"
mkdir -p "${run_dir}/ipc-probe"
export TORCH_NCCL_BLOCKING_WAIT=0 NCCL_BLOCKING_WAIT=0
python3 -u "${script_dir}/stress.py" --out "${run_dir}/ipc-probe" \
  --variants original_retain nogil_serial_retain nogil_retain
printf '%s:%s\n' "${ATOMESH_RUN_TOKEN:?run token is required}" "${phase}" > "${run_dir}/phase-${phase}.complete.tmp"
mv "${run_dir}/phase-${phase}.complete.tmp" "${run_dir}/phase-${phase}.complete"
