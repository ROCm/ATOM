#!/usr/bin/env bash
# stop_run.sh <local job id>: stop one local run on this host (its run_local_m31.sh, container, processes).
jid="${1:?local job id}"
for p in $(pgrep -f "LOCAL_JOB_ID=${jid}|run_local_m31.sh" ); do
  if tr '\0' ' ' < /proc/$p/environ 2>/dev/null | grep -q "LOCAL_JOB_ID=${jid}"; then kill "$p" 2>/dev/null; fi
done
for c in $(docker ps -a --format '{{.Names}}' | grep -E -- "-${jid}-[0-9]+(-benchmark|-eval)?$"); do
  docker stop -t 10 "$c" >/dev/null 2>&1; docker rm -f "$c" >/dev/null 2>&1; echo "removed $c"
done
for i in $(seq 1 30); do n=$(amd-smi monitor -v 2>/dev/null | awk 'NR>1 && $3+0 > 1000' | wc -l); [ "$n" = 0 ] && break; sleep 5; done
echo "job ${jid} on $(hostname): containers_left=$(docker ps --format '{{.Names}}' | grep -cE -- "-${jid}-") busy_gpus=${n}"
