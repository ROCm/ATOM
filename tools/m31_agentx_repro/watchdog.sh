#!/usr/bin/env bash
# watchdog.sh <local job id> <log dir>: stop a local run that can no longer
# finish. Exits when the run's container is gone. Triggers:
#  - a Mooncake CQ completion stall in the prefill log (>= 3 lines: a stalled
#    NIC context never recovers and wedges the prefill through quarantined slots);
#  - no request finished on decode for STALL_MIN minutes (default 25).
set -uo pipefail
JID="${1:?jid}"; DIR="${2:?log dir}"
STALL_MIN="${STALL_MIN:-25}"
BASE="${BASE:?set BASE}"
log() { echo "[$(date -u +%H:%M:%S)] watchdog ${JID}: $*" | tee -a "${BASE}/queue.log"; }
last=""; since=$(date +%s)
sleep 120
while docker ps --format '{{.Names}}' | grep -qE -- "-${JID}-0$"; do
  pre="${DIR}/slurm_job-${JID}/logs/prefill-rank-0.log"
  cq=$(grep -c "CQ completion timeout" "${pre}" 2>/dev/null); cq=${cq:-0}
  if (( cq >= 3 )); then
    log "CQ completion stall ($(grep -m1 -o 'context=rdma[0-9]*' "${pre}")) -> stopping"
    bash "$(dirname "$(readlink -f "$0")")/stop_run.sh" "${JID}"; exit 1
  fi
  # After profiling, AgentX post-processes for ~20 min with no traffic.
  if grep -aqs "Phase profiling (profiling) complete" "${DIR}"/slurm_job-"${JID}"/benchmark_results/aiperf-*/aiperf.log; then
    sleep 60; continue
  fi
  dec="${DIR}/slurm_job-${JID}/logs/decode-rank-0.log"
  # Requests the decode finished: end-to-end progress in warmup and profiling.
  cur=$(grep -c 'POST /v1/chat/completions HTTP/1.1" 200' "${dec}" 2>/dev/null); cur=${cur:-0}
  # Arm once AgentX sends traffic, so a run where nothing ever finishes stops too.
  if (( cur > 0 )) || grep -aqs "Phase warmup (warmup) started" "${DIR}"/slurm_job-"${JID}"/benchmark_results/aiperf-*/aiperf.log; then
    if [[ "${cur}" != "${last}" ]]; then last="${cur}"; since=$(date +%s)
    elif (( $(date +%s) - since > STALL_MIN * 60 )); then
      log "no decode completions for ${STALL_MIN} min (${cur} so far) -> stopping"
      bash "$(dirname "$(readlink -f "$0")")/stop_run.sh" "${JID}"; exit 1
    fi
  else
    since=$(date +%s)
  fi
  sleep 60
done
