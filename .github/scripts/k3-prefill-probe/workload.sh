#!/usr/bin/env bash
set -euo pipefail
port="$1"
results="$2/prefill-probe"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${results}"
exec > >(tee "${results}/workload.log") 2>&1
python3 -u "${script_dir}/lease.py" --port "${port}" --results "${results}"
