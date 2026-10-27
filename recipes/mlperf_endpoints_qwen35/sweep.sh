#!/bin/bash
# 30-minute MLPerf Endpoints points in a row, one fresh server each (the client's salt is
# deterministic, so a warm prefix cache from the previous point would inflate the next one).
#   R=<work dir with dataset/> bash sweep.sh "1 4 16 64 256 768 1536 2048"
S=$(cd "$(dirname "$0")" && pwd)
export R=${R:-/work/qwen35_ep}; export RUNS=${RUNS:-$R/runs}; mkdir -p $RUNS
[ -n "$1" ] || { echo "usage: sweep.sh \"<concurrencies>\""; exit 2; }
echo "SWEEP_START $(date -u +%FT%TZ) points=[$1]" >> $RUNS/status.txt
for c in $1; do
  bash $S/run_ep.sh $c || echo "POINT_FAILED c=$c $(date -u +%FT%TZ)" >> $RUNS/status.txt
done
echo "SWEEP_DONE $(date -u +%FT%TZ)" >> $RUNS/status.txt
