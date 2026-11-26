#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# prep_hour.sh NODE_A NODE_B [OUT]: one round of the hour point's Store host-memory prep (see ../README.md, "The hour
# point"), then probes and the Store plan. Run it only on nodes you hold. On every NUMA node that has a target, all in
# parallel: memory pressure (evict_cache.py: fault and free 4 KiB pages bound to the node until MemFree + Inactive(file)
# nears the floor), then the launcher's compact_huge_pages.py to min(target, MemFree - 48) GiB for at most
# COMPACT_TIMEOUT seconds. Then node_probe.py --drop on both nodes (as the harness probes) and plan_hour.py, which
# writes OUT/store_hour.{sh,json}.
#   FLOOR_A, FLOOR_B  pressure floor in GiB on node A / node B, or - for no pressure (96, 96)
#   TA0 TA1 TB0 TB1   compaction target per NUMA node in GiB, or - to leave that NUMA node alone (1184 1280 1184 1264)
#   COMPACT_TIMEOUT   seconds per compaction (420)
#   EXTRA_RESERVE     plan_hour.py --extra-reserve entries HOST:NUMA:GiB, space-separated (none)
#   PLAN_TOTAL        plan_hour.py --total in GiB (3776)
# The defaults are the hour's round 1. Rounds 1 and 2 ran as two campaign scripts and round 3 by hand, with these steps.
set -uo pipefail
A=${1:?NODE_A (P + D0)} B=${2:?NODE_B (D1)}
OUT=${3:-${AGX_RUNS:-$PWD}/prep-hour-$(date -u +%H%M%S)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=$(cd "$HERE/../harness/bin" && pwd)
S=$(cd "$HERE/../../.." && pwd)/.github/scripts/atomesh
E=$HERE/evict_cache.py
P=${AGX_HOST_PREFIX-pit2-p03-} DROP=${AGX_PROBE_DROP-/share_nfs/models}
CT=${COMPACT_TIMEOUT:-420}
mkdir -p "$OUT"
echo "$(date -u +%T) start A=$A B=$B OUT=$OUT"
bash "$HERE/numa_snapshot.sh" "$A" "$B" > "$OUT/numa-before.txt"; cat "$OUT/numa-before.txt"
declare -A FLOOR=(["$A"]=${FLOOR_A:-96} ["$B"]=${FLOOR_B:-96})
declare -A WANT=(["$A:0"]=${TA0:-1184} ["$A:1"]=${TA1:-1280} ["$B:0"]=${TB0:-1184} ["$B:1"]=${TB1:-1264})
pids=()
for key in "$A:0" "$A:1" "$B:0" "$B:1"; do
  n=${key%%:*} numa=${key##*:} t=${WANT[$key]}
  [[ "$t" == - ]] && continue
  if [[ "${FLOOR[$n]}" == - ]]; then
    pre="echo 'no pressure';"
  else
    pre="EVICT_NODE=$numa timeout 600 python3 $S/numa_exec.py $numa python3 $E pressure --gib 1450 --floor-gib ${FLOOR[$n]};"
  fi
  ssh -o BatchMode=yes "$P$n" "$pre \
    fr=\$(awk '/MemFree/ {printf \"%d\", \$4/1048576}' /sys/devices/system/node/node$numa/meminfo); t=$t; (( t > fr - 48 )) && t=\$(( fr - 48 )); \
    echo \"compaction target \$t GiB (MemFree \$fr)\"; timeout $CT python3 $S/numa_exec.py $numa python3 $S/compact_huge_pages.py \$t; echo \"compaction rc \$?\"" \
    > "$OUT/prep-$n-numa$numa.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]+"${pids[@]}"}"; do wait "$p"; done
for f in "$OUT"/prep-*.log; do
  [[ -f "$f" ]] && echo "  $(basename "$f" .log): $(grep -h 'pressure:\|stop at\|no pressure' "$f" | tail -1 | cut -c1-160) | $(grep -h 'compaction target' "$f") | $(tail -2 "$f" | tr '\n' ' ' | cut -c1-220)"
done
for n in "$A" "$B"; do timeout 300 ssh -o BatchMode=yes "$P$n" "python3 $H/node_probe.py ${DROP:+--drop $DROP}" > "$OUT/probe-$n.json" & done; wait
bash "$HERE/numa_snapshot.sh" "$A" "$B" > "$OUT/numa-after.txt"; cat "$OUT/numa-after.txt"
xr=(); for x in ${EXTRA_RESERVE:-}; do xr+=(--extra-reserve "$x"); done
python3 "$HERE/plan_hour.py" "$OUT/probe-$A.json" "$OUT/probe-$B.json" "$OUT/store_hour" --total "${PLAN_TOTAL:-3776}" \
  --note "probes $(date -u +%H:%M) UTC" "${xr[@]+"${xr[@]}"}"
echo "$(date -u +%T) done OUT=$OUT"
