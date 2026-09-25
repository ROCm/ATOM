#!/usr/bin/env bash
set -euo pipefail

log_root="${1:?source log directory required}"
result_dir="${2:?artifact directory required}"
if [[ ! -d "${log_root}" ]]; then
  echo "::warning::Slurm log directory is unavailable: ${log_root}"
  exit 0
fi
mkdir -p "${result_dir}"
tar --exclude='.cache' --exclude='.aiter' -C "${log_root}" -cf - . |
  tar --no-same-owner --no-same-permissions -C "${result_dir}" -xf -
