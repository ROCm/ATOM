#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --worker ]]; then
  hostname
  date -u
  scontrol show job "${SLURM_JOB_ID:-${SPUR_JOB_ID}}"
  python3 - <<'PY'
import json
from pathlib import Path
cards = []
for device in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
    used = device / 'mem_info_vram_used'
    total = device / 'mem_info_vram_total'
    if used.exists() and total.exists():
        cards.append({'device': str(device.resolve()), 'used': int(used.read_text()),
                      'total': int(total.read_text())})
print(json.dumps({'physical_gpu_memory': cards}))
PY
  timeout 10s docker ps --format '{{.Names}} {{.Status}}'
  if [[ -n "${2:-}" ]]; then
    bash "$(dirname "$0")/pd_inspect_processes.sh" "$2"
  fi
  exit
fi

nodes="${1:?node list required}"
inspect_job_id="${2:-}"
[[ -z "${inspect_job_id}" || "${inspect_job_id}" =~ ^[0-9]+$ ]]
IFS=',' read -r -a selected <<< "${nodes}"
(( ${#selected[@]} >= 1 && ${#selected[@]} <= 2 ))
for node in "${selected[@]}"; do
  [[ "${node}" =~ ^pit2-p03-g[0-9]+$ ]]
  [[ ",${ATOMESH_NODE_POOL}," == *",${node},"* ]]
done
name="mooncake-capacity-${GITHUB_RUN_ID}"
rc=0
timeout --kill-after=5s 150s srun \
  --job-name "${name}" --account amd-frameworks --partition amd-spur \
  --qos amd-frameworks-qos --nodes "${#selected[@]}" \
  --ntasks "${#selected[@]}" --ntasks-per-node 1 --cpus-per-task 1 \
  --gpus 0 --time 00:02:00 --nodelist "${nodes}" \
  bash "${GITHUB_WORKSPACE}/.github/scripts/atomesh/pd_inspect_gpu_capacity.sh" \
  --worker "${inspect_job_id}" || rc=$?
# Bound even a diagnostic that stayed queued; only cancel its unique job name.
while IFS='|' read -r job_id job_name; do
  if [[ "${job_id}" =~ ^[0-9]+$ && "${job_name}" == "${name}" ]]; then
    timeout 10s scancel "${job_id}"
  fi
done < <(timeout 15s squeue --noheader --format='%i|%j')
exit "${rc}"
