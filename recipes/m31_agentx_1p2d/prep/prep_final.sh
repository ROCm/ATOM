#!/usr/bin/env bash
# prep_final.sh TOPO [OUT]: Store prep for the final full hour (CONTROL 06:15: largest Store the nodes allow, target >= 4 TB).
# 1. memory pressure on all four NUMA nodes of g27 (node A) and g23 (node B) in parallel (evict_cache.py pressure: fault +
#    free 4 KiB anon pages bound to the node until MemFree + Inactive(file) nears 96 GiB) -> foreign clean page cache out;
# 2. THP compaction (launcher compact_huge_pages.py) toward per-NUMA targets, COMPACT_TIMEOUT s cap (default 420);
# 3. probe both nodes and plan with relaxed reserves (measured role host memory <= ~90 GiB per NUMA: reserve_role 96,
#    reserve_base 48; caps P/D/FREE 1152/1152/1344); writes OUT/store_final.sh (overlay with the explicit owners) and
#    OUT/plan.json. The launcher itself still keeps 128 GiB per NUMA in reserve.
set -uo pipefail
TOPO=${1:?1p2d|1p3d}; OUT=${2:-${AGX_RUNS:-$PWD}/prep-final-$(date -u +%H%M)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=$(cd "$HERE/../harness/bin" && pwd)
S=$(cd "$HERE/../../.." && pwd)/.github/scripts/atomesh
E=$HERE/evict_cache.py
A=${NODE_A:?set NODE_A (node A: P + D0)} B=${NODE_B:?set NODE_B (node B: D1 + Store owners)}
P=${AGX_HOST_PREFIX-pit2-p03-} DROP=${AGX_PROBE_DROP-/share_nfs/models}
mkdir -p "$OUT"
snap() { bash "$HERE/numa_snapshot.sh" $A $B > "$OUT/numa-$1.txt"; cat "$OUT/numa-$1.txt"; }
snap before
pids=()
for n in $A $B; do for numa in 0 1; do
  ssh -o BatchMode=yes "$P$n" "EVICT_NODE=$numa timeout 600 python3 $S/numa_exec.py $numa python3 $E pressure --gib 1450 --floor-gib 96" > "$OUT/pressure-$n-numa$numa.log" 2>&1 &
  pids+=($!)
done; done
for p in "${pids[@]}"; do wait "$p"; done
grep -h "pressure:" "$OUT"/pressure-*.log
snap after-pressure
declare -A TARGET=(["$A:0"]=1190 ["$A:1"]=1190 ["$B:0"]=1390 ["$B:1"]=1390)
pids=()
for key in "$A:0" "$A:1" "$B:0" "$B:1"; do
  n=${key%%:*} numa=${key##*:} t=${TARGET[$key]}
  ssh -o BatchMode=yes "$P$n" "timeout ${COMPACT_TIMEOUT:-420} python3 $S/numa_exec.py $numa python3 $S/compact_huge_pages.py $t" > "$OUT/compact-$n-numa$numa.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for f in "$OUT"/compact-*.log; do echo "  $(basename "$f" .log): $(tail -1 "$f")"; done
for n in $A $B; do ssh -o BatchMode=yes "$P$n" "python3 $H/node_probe.py ${DROP:+--drop $DROP}" > "$OUT/probe-$n.json" & done; wait
python3 $H/store_plan.py --topo "$TOPO" --pools per_nic --probe "$A=$OUT/probe-$A.json" --probe "$B=$OUT/probe-$B.json" \
  --reserve-role 96 --reserve-base 48 --cap-p 1152 --cap-d 1152 --cap-free 1344 > "$OUT/plan.json"
python3 - "$OUT/plan.json" "$OUT/store_final.sh" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
open(sys.argv[2], "w").write(
    "# Sourced last in every rank env: the final hour's explicit Store owners (p2d/scripts/prep_final.sh).\n"
    f"export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS='{d['owners']}'\n"
    f"export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS='{d['decode_owners']}'\n")
print("final plan:", d["total_gib"], "GiB | A:", d["owners"], "| B:", d["decode_owners"])
for node, numas in d["per_node"].items():
    print(" ", node, {k: (v["role"], v["owners_gib"], v["big_free_gib"]) for k, v in numas.items()})
PY
snap final
echo "OUT=$OUT"
