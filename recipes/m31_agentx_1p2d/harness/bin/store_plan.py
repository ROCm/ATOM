#!/usr/bin/env python3
"""store_plan.py: size the Mooncake Store owners of a run from each node's free DRAM.

Input: --probe <node>=<node_probe.json> per node (node A first), --topo.
Output (JSON on stdout): {"owners": "<numa>:<GiB>:<nic>;...", "decode_owners": ...,
"per_node": {...}, "total_gib": N}.

Layout (per_nic pools on rdma0-3, the PP4 prefill stages' NICs on node A):
every NUMA node of every node gets one owner per NIC rdma0-3 (an owner on NUMA1
then serves over a NUMA0 NIC; it stays bound to NUMA1 memory). Each NUMA node's
total is the smallest of
  - its cap: --cap-p / --cap-d GiB where the prefill / a decode lives on it,
    --cap-free where no role does,
  - what it can give: free + reclaimable file cache x --file-frac, minus
    --reserve-role GiB for a role's host memory (or --reserve-base without one),
  - --thp-frac of its free memory in >= 2 MiB blocks (after the page-cache drop): owners must
    be huge pages, and compaction may not free more,
rounded down to a multiple of 4 x --step GiB, and no owner above 832 GiB (an
ionic NIC registers no larger owner).
"""

import argparse
import json
import math

# per_nic: one owner per stage NIC on every NUMA node (pool k = rdma k, the stage on rdma k reads it).
# shared: one master; owners off the stages' NICs, on rdma4-7 (NUMA0's on rdma4/5, NUMA1's on rdma6/7),
# the measured 1P2D C220 layout.
NICS_BY_POOLS = {"per_nic": {"0": ("rdma0", "rdma1", "rdma2", "rdma3"), "1": ("rdma0", "rdma1", "rdma2", "rdma3")},
                 "shared": {"0": ("rdma4", "rdma5"), "1": ("rdma6", "rdma7")}}
MAX_OWNER_GIB = 832

# Which NUMA nodes host a role (P or D) on node A / node B, per topology.
ROLE_NUMA = {
    "1p1d": [{"0": "P", "1": "D0"}],
    "1p2d": [{"0": "P", "1": "D0"}, {"1": "D1"}],
    "1p3d": [{"0": "P", "1": "D0"}, {"0": "D1", "1": "D2"}],
}


def numa_total(probe_numa, role, args):
    cap = args.cap_p if role == "P" else args.cap_d if role else args.cap_free
    reserve = args.reserve_role if role else args.reserve_base
    give = probe_numa["free_gib"] + args.file_frac * max(
        0.0, probe_numa["file_gib"] - probe_numa.get("shmem_gib", 0.0)
    )
    # Never plan past MemTotal - reserve - 128 (the launcher's budget check).
    hard = probe_numa["total_gib"] - reserve - 128
    # Owners must be all huge pages (4 KiB pages fail NIC registration), and compaction cannot be
    # counted on to free more 2 MiB blocks than the node has: g53 NUMA1 reached 719 of 768 GiB in 15 min.
    thp = args.thp_frac * probe_numa.get("big_free_gib", hard)
    usable = min(cap, give - reserve, hard, thp)
    unit = 4 * args.step  # 4 owners per NUMA node at most
    return max(0, math.floor(usable / unit) * unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", required=True, choices=sorted(ROLE_NUMA))
    ap.add_argument("--probe", action="append", required=True, help="node=probe.json, node A first")
    ap.add_argument("--cap-p", type=float, default=1024, help="GiB of owners on the prefill's NUMA node")
    ap.add_argument("--cap-d", type=float, default=768, help="GiB of owners on a decode's NUMA node")
    ap.add_argument("--cap-free", type=float, default=1152)
    ap.add_argument("--reserve-role", type=float, default=192)
    ap.add_argument("--reserve-base", type=float, default=64)
    ap.add_argument("--file-frac", type=float, default=0.8)
    ap.add_argument("--thp-frac", type=float, default=0.97,
                    help="plan at most this fraction of the free >= 2 MiB blocks a NUMA node has")
    ap.add_argument("--step", type=int, default=16)
    ap.add_argument("--pools", default="per_nic", choices=sorted(NICS_BY_POOLS))
    args = ap.parse_args()

    roles = ROLE_NUMA[args.topo]
    if len(args.probe) != len(roles):
        raise SystemExit(f"{args.topo} needs {len(roles)} node probe(s), got {len(args.probe)}")
    specs, per_node, total = [], {}, 0
    for (item, role_map) in zip(args.probe, roles):
        node, _, path = item.partition("=")
        with open(path) as handle:
            probe = json.load(handle)
        entries, node_plan = [], {}
        for numa in sorted(probe["numa"], key=int):
            role = role_map.get(numa, "")
            nics = NICS_BY_POOLS[args.pools][numa]
            gib = numa_total(probe["numa"][numa], role, args)
            each = min(gib // len(nics), MAX_OWNER_GIB)
            node_plan[numa] = {"role": role or "-", "owners_gib": each * len(nics), "per_owner_gib": each,
                               **probe["numa"][numa]}
            if each > 0:
                entries += [f"{numa}:{each}:{nic}" for nic in nics]
            total += each * len(nics)
        per_node[node] = node_plan
        specs.append(";".join(entries))
    out = {
        "pools": args.pools,
        "owners": specs[0],
        "decode_owners": specs[1] if len(specs) > 1 else "",
        "total_gib": total,
        "per_node": per_node,
        "args": vars(args),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
