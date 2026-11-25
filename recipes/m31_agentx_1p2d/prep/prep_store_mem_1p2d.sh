#!/usr/bin/env bash
# prep_store_mem_1p2d.sh <nodeX> <nodeY>: the curve's prep_store_mem.sh (03b) for the 1P2D layout.
# Same steps: drop the model page cache, probe free >= 2 MiB blocks per NUMA, node A (P + D0) = the node with more free
# huge-page blocks, fault + free THP with the launcher's compact_huge_pages.py (numa_exec.py) on every NUMA below its
# target, re-probe, print the predicted store_plan (--topo 1p2d). Targets = the 1p2d plan caps / 0.97 (thp-frac):
# A NUMA0 (P) 1060 -> plan 1024, A NUMA1 (D0) 800 -> 768, B NUMA0 (no role) 1190 -> 1152, B NUMA1 (D1) 800 -> 768.
set -uo pipefail
X=$1 Y=$2 TA0=${3:-1060} TB0=${4:-1190} TD=${5:-800} TB1=${6:-${5:-800}}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=$(cd "$HERE/../harness/bin" && pwd)
S=$(cd "$HERE/../../.." && pwd)/.github/scripts/atomesh
OUT=${PREP_OUT:-${AGX_RUNS:-$PWD}/prep-$(date -u +%H%M%S)}
P=${AGX_HOST_PREFIX-pit2-p03-} DROP=${AGX_PROBE_DROP-/share_nfs/models}
mkdir -p "$OUT"
probe() { ssh -o BatchMode=yes "$P$1" "python3 $H/node_probe.py $2" > "$OUT/$3-$1.json"; }
big() { python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(" ".join(str(round(p["numa"][n]["big_free_gib"])) for n in sorted(p["numa"])))' "$1"; }
probe "$X" "${DROP:+--drop $DROP}" before & probe "$Y" "${DROP:+--drop $DROP}" before & wait
read -r X0 X1 <<< "$(big "$OUT/before-$X.json")"; read -r Y0 Y1 <<< "$(big "$OUT/before-$Y.json")"
echo "before: $X big_free GiB numa0=$X0 numa1=$X1 | $Y numa0=$Y0 numa1=$Y1"
if (( X0 + X1 >= Y0 + Y1 )); then A=$X B=$Y; else A=$Y B=$X; fi
[[ -n "${FORCE_A:-}" ]] && { A=$FORCE_A; [[ $A == "$X" ]] && B=$Y || B=$X; }
echo "NODE_A=$A NODE_B=$B"
declare -A TARGET=(["$A:0"]=$TA0 ["$A:1"]=$TD ["$B:0"]=$TB0 ["$B:1"]=$TB1)
declare -A HAVE=(["$X:0"]=$X0 ["$X:1"]=$X1 ["$Y:0"]=$Y0 ["$Y:1"]=$Y1)
pids=()
for key in "$A:0" "$A:1" "$B:0" "$B:1"; do
  n=${key%%:*} numa=${key##*:} t=${TARGET[$key]} have=${HAVE[$key]}
  if (( have >= t )); then echo "  $n numa$numa: $have >= $t GiB, no compaction"; continue; fi
  echo "  $n numa$numa: $have < $t GiB -> compacting $t GiB"
  ssh -o BatchMode=yes "$P$n" "timeout ${COMPACT_TIMEOUT:-900} python3 $S/numa_exec.py $numa python3 $S/compact_huge_pages.py $t" > "$OUT/compact-$n-numa$numa.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]+"${pids[@]}"}"; do wait "$p"; done
for f in "$OUT"/compact-*.log; do [[ -f "$f" ]] && echo "  $(basename "$f" .log): $(tail -1 "$f")"; done
probe "$X" "" after & probe "$Y" "" after & wait
read -r X0 X1 <<< "$(big "$OUT/after-$X.json")"; read -r Y0 Y1 <<< "$(big "$OUT/after-$Y.json")"
echo "after:  $X big_free GiB numa0=$X0 numa1=$X1 | $Y numa0=$Y0 numa1=$Y1"
python3 $H/store_plan.py --topo ${PLAN_TOPO:-1p2d} --pools per_nic --probe "$A=$OUT/after-$A.json" --probe "$B=$OUT/after-$B.json" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("predicted plan:", d["total_gib"], "GiB | A:", d["owners"], "| B:", d["decode_owners"])'
echo "NODE_A=$A NODE_B=$B OUT=$OUT"
