#!/usr/bin/env bash
set -euo pipefail

mkdir -p inspection
controller=http://pit2-vm-amd-large-02:6817
caller="$(id -un)"
cleanup_complete=1
declare -A expected=(
  [4545]=kimi-k3-mxfp4-vllm-dspark-kimi-k3-vllm-dspark4-1p1d-tp8-dcp8-agentic-lmcache-1m-c64-latent-fp4-vllm-36213342229-1
  [4546]=k3-prefill-timeline-probe-36213608583-1
)

for job in 4545 4546; do
  if queue="$(timeout 15s squeue --controller "${controller}" --noheader --format='%i|%j|%u|%T' \
    2> "inspection/obsolete-job-${job}-query-error.txt")"; then
    :
  else
    echo "Cannot query job ${job}; stopping mutation attempts."
    cleanup_complete=0
    break
  fi
  row="$(awk -F'|' -v id="${job}" '$1 == id {print}' <<< "${queue}")"
  if [[ -z "${row}" ]]; then
    echo "Job ${job} absent; no cancellation requested."
    continue
  fi
  IFS='|' read -r found name owner state <<< "${row}"
  [[ "${found}" == "${job}" && "${name}" == "${expected[${job}]}" && "${owner}" == "${caller}" ]] || {
    echo "Job ${job} identity mismatch; no cancellation requested."
    cleanup_complete=0
    break
  }
  printf '%s\n' "${row}" > "inspection/obsolete-job-${job}-before.txt"
  if timeout 15s scancel --controller "${controller}" "${job}" \
    > "inspection/obsolete-job-${job}-cancel.txt" 2>&1; then
    echo 0 > "inspection/obsolete-job-${job}-cancel-rc.txt"
  else
    echo "$?" > "inspection/obsolete-job-${job}-cancel-rc.txt"
  fi
  # Spur 0.11 can return zero even when an individual cancellation failed.
  if timeout 15s scontrol --controller "${controller}" show job "${job}" \
    > "inspection/obsolete-job-${job}-after.txt" 2>&1 &&
    grep -Eq '^[[:space:]]*JobState=CANCELLED([[:space:]]|$)' "inspection/obsolete-job-${job}-after.txt"; then
    echo "Cancellation of ${job} confirmed."
  else
    echo "Cancellation of ${job} is unconfirmed; stopping mutation attempts."
    cleanup_complete=0
    break
  fi
done

for host in pit2-vm-amd-large-02 pit2-vm-amd-large-03 pit2-vm-amd-large-04; do
  timeout 10s sdiag --controller "http://${host}:6817" \
    > "inspection/${host}-sdiag.txt" 2>&1 || true
  timeout --kill-after=2s 20s ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
    -o ConnectTimeout=4 "${host}" \
    'hostname; id; systemctl show spurctld -p ActiveState -p SubState -p ExecMainStartTimestamp -p NRestarts;
     curl -sS --connect-timeout 2 --max-time 3 -w "\nhttp_status=%{http_code}\n" http://127.0.0.1:6822/metrics/scheduler;
     journalctl -u spurctld --since "2026-09-26 03:00:00 UTC" --no-pager -n 2000 2>&1 |
       grep -E "snapshot not found|not the Raft leader|panicked|FatalStorageError|when Read Snapshot|Permission denied|No journal" | tail -40' \
    > "inspection/${host}-host-health.txt" 2>&1 || true
done
echo "${cleanup_complete}" > inspection/obsolete-jobs-cleanup-confirmed.txt
exit "$((1 - cleanup_complete))"
