#!/usr/bin/env bash
# agx_status.sh <run dir>: one compact line per role of a running point (newest engine-log line),
# the aiperf phase / progress, and the driver's last line.
R="${1:?run dir}"
J=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]+"/run.json"))["job_dir"])' "$R")
for L in "$J"/logs/prefill-rank-*.log "$J"/logs/decode-rank-*.log; do
  [[ -f "$L" ]] || continue
  l=$(grep -a "Engine 000" "$L" | tail -1 | sed -E 's/.*\[atom ([0-9:]+)\].*prompt throughput: ([0-9.]+).*generation throughput: ([0-9.]+).*Running: ([0-9]+).*Waiting: ([0-9]+).*usage: ([0-9.na/]+)%?.*hit rate: ([0-9.na/]+).*/\1 prompt=\2 gen=\3 run=\4 wait=\5 kv=\6 hit=\7/')
  m=$(grep -a "MTP Stats Interval" "$L" | tail -1 | sed -E 's/.*toks\/fwd: ([0-9.]+).*/ toks\/fwd=\1/')
  echo "$(basename "$L" .log): ${l:-no engine line yet}${m}"
done
A=$(ls "$J"/benchmark_results/aiperf-*/aiperf.log 2>/dev/null | head -1)
[[ -n "$A" ]] && grep -a -E "Phase (warmup|profiling) (progress|started|sending complete|complete)" "$A" | tail -1 | cut -c1-220
tail -1 "$R/driver.log" 2>/dev/null
