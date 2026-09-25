#!/usr/bin/env bash
set -euo pipefail

job_id="${1:?existing Slurm job ID required}"
[[ "${job_id}" =~ ^[0-9]+$ ]]
hostname
date -u
grep -E '^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree|Mlocked|Unevictable):' /proc/meminfo
timeout 5s journalctl -k --since '30 minutes ago' --no-pager 2>&1 |
  grep -Ei 'oom|out of memory|killed process|amdgpu|gpu reset|permission|no journal' |
  tail -n 80 || true
ps -eo pid,ppid,etimes,pcpu,stat,wchan:24,comm |
  awk 'NR == 1 || /VLLM|python|ninja|clang|hipcc|cmake/'

while IFS= read -r container; do
  [[ "${container}" == atomesh-* && "${container}" =~ -${job_id}-[0-9]+(-benchmark|-eval)?$ ]] || continue
  printf 'Container: %s\n' "${container}"
  # shellcheck disable=SC2016
  timeout --kill-after=5s 90s docker exec --user 0 --privileged "${container}" bash -c '
    id
    ps -eo pid,ppid,etimes,pcpu,stat,wchan:24,comm
    declare -A metric_endpoints=()
    offset="${ATOMESH_SERVICE_PORT_OFFSET:-0}"
    IFS=, read -r -a metric_hosts <<< "${IPADDRS:-127.0.0.1}"
    for port in "${PREFILL_PORT:-}" "${DECODE_PORT:-}"; do
      [[ "${port}" =~ ^[0-9]{1,5}$ && "${offset}" =~ ^[0-9]{1,5}$ ]] || continue
      port=$((10#${port} + 10#${offset}))
      ((port >= 1 && port <= 65535)) || continue
      for metric_host in "${metric_hosts[@]}"; do
        [[ "${metric_host}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || continue
        endpoint="${metric_host}:${port}"
        [[ -z "${metric_endpoints[${endpoint}]:-}" ]] || continue
        metric_endpoints["${endpoint}"]=1
        printf "Metrics endpoint=%s\n" "${endpoint}"
        date -u
        curl -fsS --connect-timeout 2 --max-time 5 "http://${endpoint}/metrics" |
          grep -E "^vllm:(request_success_total|request_generation_tokens_(sum|count|bucket)|request_prompt_tokens_(sum|count)|num_requests_running|num_requests_waiting|num_preemptions_total|kv_cache_usage_perc|generation_tokens_total|prompt_tokens_total)" || true
      done
    done
    spy="$(command -v py-spy || true)"
    if [[ -z "${spy}" ]]; then
      target=/tmp/atomesh-inspection-py-spy
      timeout --kill-after=2s 40s python3 -m pip install --disable-pip-version-check \
        --no-deps --target "${target}" py-spy==0.4.2 || exit 0
      spy="${target}/bin/py-spy"
    fi
    deadline=$((SECONDS + 45))
    pids=$(pgrep -f "^[^ ]*python[^ ]* .*lmcache server" || true)
    pids+=" $(pgrep -x lmcache || true)"
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
      ps -L -p "${pid}" -o pid,tid,stat,wchan:32,comm || true
      timeout --kill-after=1s "${remaining}s" "${spy}" dump --pid "${pid}" --native || true
    done
  ' || true
done < <(docker ps --format '{{.Names}}')
