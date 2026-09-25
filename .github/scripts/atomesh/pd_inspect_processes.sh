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
  timeout --kill-after=5s 90s docker exec --privileged --user 0 "${container}" bash -c '
    spy="$(command -v py-spy || true)"
    if [[ -z "${spy}" ]]; then
      target=/tmp/atomesh-inspection-py-spy
      timeout --kill-after=2s 40s python3 -m pip install --disable-pip-version-check \
        --no-deps --target "${target}" py-spy==0.4.2 || exit 0
      spy="${target}/bin/py-spy"
    fi
    deadline=$((SECONDS + 45))
    pids=$(pgrep -f "^[^ ]*python[^ ]* .*lmcache server" || true)
    pids+=" $(pgrep -f "^VLLM::EngineCore" || true)"
    pids+=" $(pgrep -f "^VLLM::Worker_TP[0-9]+" || true)"
    for pid in ${pids}; do
      remaining=$((deadline - SECONDS))
      if ((remaining <= 0)); then
        printf "Stack sampling budget exhausted before process %s\n" "${pid}"
        break
      fi
      ((remaining <= 8)) || remaining=8
      printf "Process %s\n" "${pid}"
      timeout --kill-after=1s "${remaining}s" "${spy}" dump --pid "${pid}" --native || true
    done
  ' || true
done < <(docker ps --format '{{.Names}}')
