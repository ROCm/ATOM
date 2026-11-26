#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""plan_hour.py PROBE_A PROBE_B OUT_PREFIX [--total 3776] [--note TEXT] [--extra-reserve HOST:NUMA:GiB ...]

The hour point's Store owner planner, as it ran; only the header, this docstring and the overlay's comment line changed.
It plans a 1P2D: node A runs the prefill on NUMA0 (with its four pd-send processes) and decode D0, the Store masters
and the router on NUMA1; node B runs decode D1 on NUMA1 and no role on NUMA0. PROBE_A / PROBE_B are node_probe.py
JSONs taken after the host-memory prep. Pools are per NIC: 4 equal owners per NUMA node, one per rdma0-3, so every
pool gets the sum of the four per-owner sizes.

Per NUMA node, the per-owner ceiling is the smallest of these (GiB), divided by 4 and rounded down to 16:
  THP    0.97 x the free memory in >= 2 MiB blocks (owners must be all huge pages; the harness store_plan.py's rule)
  SPARE  those blocks minus a margin left free after the owners: A NUMA0 (P) 250 + 16 for the four send processes
         (3.5-3.7 GiB each), A NUMA1 (D0) 120, B NUMA0 (no role) 96, B NUMA1 (D1) 160
  MEM    free memory minus a role reserve: P 256 + 16 for the send processes, a decode 192, no role 64
  LAUNCH MemTotal - 128 (the launcher's budget check)
  OWNER  304 per owner
--extra-reserve HOST:NUMA:GiB lowers SPARE and MEM there, for other users' jobs that may grow during the run (the hour
kept 400 GiB on g27 NUMA0). HOST is the probe's host name, with or without the pit2-p03- prefix.
The plan starts from an earlier run's per-owner sizes (U0M1B, "u0m1b" in the printout); a NUMA node over its ceiling
drops to it, and the deficit goes 16 GiB at a time to the NUMA node with the most ceiling headroom (A NUMA0, the
prefill's, last), until the per-pool sum is total / 4 or nobody has headroom (then the plan is the largest
equal-per-pool plan the ceilings allow, and says so). Writes OUT_PREFIX.json and OUT_PREFIX.sh (overlay)."""
import argparse
import json
from pathlib import Path

U0M1B = {("A", "0"): 128, ("A", "1"): 208, ("B", "0"): 272, ("B", "1"): 192}
CHILDREN_GIB = 16  # 4 production send processes: U3b VmRSS 3.5-3.7 GiB each at T0+386 s (U3a's bare 1.5 GB undercounts)
SPARE = {("A", "0"): 250 + CHILDREN_GIB, ("A", "1"): 120, ("B", "0"): 96, ("B", "1"): 160}
RESERVE = {("A", "0"): 256 + CHILDREN_GIB, ("A", "1"): 192, ("B", "0"): 64, ("B", "1"): 192}
ROLE = {("A", "0"): "P+4 send", ("A", "1"): "D0+masters", ("B", "0"): "-", ("B", "1"): "D1"}
STEP, OWNER_MAX, THP_FRAC = 16, 304, 0.97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_a")
    ap.add_argument("probe_b")
    ap.add_argument("out")
    ap.add_argument("--total", type=int, default=3776)
    ap.add_argument("--note", default="")
    ap.add_argument("--extra-reserve", action="append", default=[],
                    help="HOST:NUMA:GiB more room to keep free there (e.g. g27:0:400 for its foreign CPU sims' swings)")
    a = ap.parse_args()
    extra = {}
    for e in a.extra_reserve:
        h, n, g = e.split(":")
        extra[(h.replace("pit2-p03-", ""), n)] = float(g)
    probes = {"A": json.loads(Path(a.probe_a).read_text()), "B": json.loads(Path(a.probe_b).read_text())}
    ceil, lims = {}, {}
    for key in U0M1B:
        node, n = key
        m = probes[node]["numa"][n]
        x = extra.get((probes[node]["host"].replace("pit2-p03-", ""), n), 0.0)
        lim = {"THP": THP_FRAC * m["big_free_gib"], "SPARE": m["big_free_gib"] - SPARE[key] - x,
               "MEM": m["free_gib"] - RESERVE[key] - x, "LAUNCH": m["total_gib"] - 128, "OWNER": 4 * OWNER_MAX}
        bind = min(lim, key=lim.get)
        ceil[key] = max(0, int(lim[bind] / 4 // STEP * STEP))
        lims[key] = (lim, bind)
    per = {k: min(v, ceil[k]) for k, v in U0M1B.items()}
    want = a.total // 4
    while sum(per.values()) < want:
        room = {k: ceil[k] - per[k] for k in per if ceil[k] - per[k] >= STEP}
        if not room:
            break
        # A NUMA0 (the prefill's) takes extra owners last: U0's failed mounts were there
        k = max(room, key=lambda x: (x != ("A", "0"), room[x]))
        per[k] += STEP
    pool = sum(per.values())
    grand = 4 * pool
    out = {"note": a.note, "probes": [a.probe_a, a.probe_b], "per_numa": {}, "total_gib": grand, "per_pool_gib": pool,
           "target_gib": a.total, "reached": grand >= a.total}
    specs = {}
    for node in ("A", "B"):
        host = probes[node]["host"].replace("pit2-p03-", "")
        entries = []
        print(f"{node} = {host}")
        for n in ("0", "1"):
            key = (node, n)
            m = probes[node]["numa"][n]
            lim, bind = lims[key]
            each = per[key]
            entries += [f"{n}:{each}:rdma{k}" for k in range(4)] if each else []
            ls = " ".join(f"{k} {v:.0f}" for k, v in lim.items())
            print(f"  numa{n} {ROLE[key]:10s} free {m['free_gib']:.1f} file {m['file_gib']:.1f} 2MiB {m['big_free_gib']:.1f}"
                  f" | {ls} -> ceiling 4 x {ceil[key]} ({bind}); u0m1b 4 x {U0M1B[key]} -> 4 x {each} = {4 * each};"
                  f" 2 MiB left {m['big_free_gib'] - 4 * each:.0f} (rule >= {SPARE[key]}), free left {m['free_gib'] - 4 * each:.0f}")
            out["per_numa"][f"{host}:{n}"] = {"role": ROLE[key], "owners_gib": 4 * each, "per_owner_gib": each,
                                              "ceiling_per_owner": ceil[key], "bind": bind,
                                              "limits": {k: round(v, 1) for k, v in lim.items()}, **m}
        specs[node] = ";".join(entries)
    out.update({"owners": specs["A"], "decode_owners": specs["B"]})
    print(f"TOTAL {grand} GiB ({pool} per pool){'' if grand >= a.total else f'  BELOW the {a.total} GiB target'}")
    print(f"A: {specs['A']}\nB: {specs['B']}")
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    Path(a.out + ".sh").write_text(
        "# Sourced last in every rank env: Store owners from prep/plan_hour.py (u0m1b start, equal pools"
        f"{'; ' + a.note if a.note else ''}): {grand} GiB, {pool} GiB per pool.\n"
        f"export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS='{specs['A']}'\n"
        f"export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS='{specs['B']}'\n")


if __name__ == "__main__":
    main()
