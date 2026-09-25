#!/usr/bin/env bash
set -euo pipefail

job_id="${1:?existing Slurm job ID required}"
[[ "${job_id}" =~ ^[0-9]+$ ]]
hostname
date -u
ps -eo pid,ppid,etimes,pcpu,stat,wchan:24,comm |
  awk 'NR == 1 || /VLLM|python|ninja|clang|hipcc|cmake/'

while IFS= read -r container; do
  [[ "${container}" == atomesh-* && "${container}" =~ -${job_id}-[0-9]+(-benchmark|-eval)?$ ]] || continue
  printf 'Container: %s\n' "${container}"
  # shellcheck disable=SC2016
  timeout 20s docker exec "${container}" bash -c '
    if ! command -v py-spy >/dev/null; then
      echo "py-spy is not installed in this container"
      exit 0
    fi
    for pid in $(pgrep -f "^VLLM::Worker_TP0" || true); do
      timeout --kill-after=2s 8s py-spy dump --pid "${pid}" --native || true
    done
  ' || true
done < <(docker ps --format '{{.Names}}')
