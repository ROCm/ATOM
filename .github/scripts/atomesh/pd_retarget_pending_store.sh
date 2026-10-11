#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?source run ID required}"
node="${2:?approved target node required}"
output="${3:?evidence directory required}"
[[ "${run_id}" =~ ^[0-9]+$ ]]
[[ "${node}" =~ ^pit2-p03-g[0-9]+$ ]]
[[ ",${ATOMESH_NODE_POOL}," == *",${node},"* ]]
mkdir -p "${output}"
timeout 15s squeue --noheader --format='%i|%T|%j' > "${output}/queue-before.txt"
mapfile -t matches < <(awk -F '|' -v name="mooncake-store-smoke-${run_id}" \
  '$3 == name {print $1 " " $2}' "${output}/queue-before.txt")
if (( ${#matches[@]} != 1 )); then
  echo "Expected one exact Store job; found ${#matches[@]}" >&2
  exit 1
fi
read -r job_id state <<< "${matches[0]}"
[[ "${job_id}" =~ ^[0-9]+$ && "${state}" == PENDING ]]
timeout 15s scontrol show job "${job_id}" > "${output}/job-before.txt"
grep -Eq '(^|[[:space:]])JobState=PENDING([[:space:]]|$)' "${output}/job-before.txt"
grep -Eq '(^|[[:space:]])NumNodes=1([[:space:]]|$)' "${output}/job-before.txt"
grep -Eq '(^|[[:space:]])Account=amd-frameworks([[:space:]]|$)' "${output}/job-before.txt"
grep -Eq '(^|[[:space:]])QOS=amd-frameworks-qos([[:space:]]|$)' "${output}/job-before.txt"
printf '%s\n' "scontrol update JobId=${job_id} ReqNodeList=${node}" > "${output}/command.txt"
timeout 15s scontrol update "JobId=${job_id}" "ReqNodeList=${node}" \
  > "${output}/update.log" 2>&1
timeout 15s scontrol show job "${job_id}" > "${output}/job-after.txt"
grep -Eq "(^|[[:space:]])ReqNodeList=${node}([[:space:]]|$)" "${output}/job-after.txt"
cat "${output}/command.txt" "${output}/job-after.txt"
