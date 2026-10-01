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

# Keep correctness evidence explicitly staged outside generic logs/ directories.
# Native build failures must retain preflight/build details, not only their tail
# in container logs. Smoke and bounded probe evidence belong to the same gate.
python3 - "${log_root}" "${result_dir}/validation-evidence" <<'PY'
import shutil
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
for job in source.glob("slurm_job-*"):
    files = list((job / "logs").glob("**/native-*"))
    for directory in ("pd-smoke", "read-replay"):
        files.extend((job / directory).glob("**/*"))
    for path in files:
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
PY
