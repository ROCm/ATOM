#!/usr/bin/env python3
"""Check that the memory a job pins on each NUMA node fits there.

Processes bound with MPOL_BIND never borrow another node's memory: pins larger
than their node end in an out-of-memory kill once the pages are touched, and
pins larger than the node's free memory make every huge-page fault past it
reclaim page cache through compaction first, which has taken long enough to
time a start out (rocm7: over an hour).

Usage:
  numa_memory_budget.py [--reserve-gib R] [--gpus 0,1,... --per-gpu-gib G]
                        [<node>:<GiB> ...]

``<node>:<GiB>`` pins that much on a node; each GPU of ``--gpus`` (HIP
ordinals) pins ``--per-gpu-gib`` on its own node. Exits 2 when a node's pins
plus ``--reserve-gib`` exceed its MemTotal, and warns when they exceed its
MemFree.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

GIB = 1 << 30


def node_meminfo(sysfs: Path, node: int) -> dict[str, int]:
    """A NUMA node's meminfo, kB fields in bytes."""
    path = sysfs / f"devices/system/node/node{node}/meminfo"
    try:
        text = path.read_text()
    except OSError as exc:
        raise SystemExit(
            f"[numa-budget][FAIL] cannot read NUMA node {node}: {exc}"
        ) from exc
    info = {}
    for line in text.splitlines():
        # "Node 0 MemFree:        838449356 kB"
        fields = line.split()
        if len(fields) >= 4 and fields[2].endswith(":") and fields[3].isdigit():
            scale = 1024 if fields[-1] == "kB" else 1
            info[fields[2][:-1]] = int(fields[3]) * scale
    return info


def kfd_gpu_pci_addresses(sysfs: Path) -> list[str]:
    """PCI addresses of the GPUs in KFD topology order, the order HIP numbers them."""
    nodes = sysfs / "class/kfd/kfd/topology/nodes"
    addresses = []
    for node in sorted(nodes.iterdir(), key=lambda path: int(path.name)):
        properties = {}
        for line in (node / "properties").read_text().splitlines():
            name, _, value = line.partition(" ")
            properties[name] = value.strip()
        if int(properties.get("simd_count", "0")) == 0:  # a CPU node
            continue
        location = int(properties["location_id"])
        domain = int(properties.get("domain", "0"))
        addresses.append(
            f"{domain:04x}:{location >> 8:02x}:{(location >> 3) & 0x1F:02x}."
            f"{location & 0x7:x}"
        )
    return addresses


def gpu_numa_nodes(sysfs: Path, ordinals: list[int]) -> list[int]:
    """The NUMA node of each GPU, by HIP ordinal."""
    try:
        addresses = kfd_gpu_pci_addresses(sysfs)
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(
            f"[numa-budget][FAIL] cannot read the GPUs from the KFD topology: {exc!r}"
        ) from exc
    nodes = []
    for ordinal in ordinals:
        if not 0 <= ordinal < len(addresses):
            raise SystemExit(
                f"[numa-budget][FAIL] GPU {ordinal} is not one of the "
                f"{len(addresses)} GPUs in the KFD topology"
            )
        path = sysfs / "bus/pci/devices" / addresses[ordinal] / "numa_node"
        try:
            node = int(path.read_text())
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"[numa-budget][FAIL] cannot read GPU {ordinal}'s NUMA node: {exc}"
            ) from exc
        # -1: the platform reports no node, i.e. a single one.
        nodes.append(max(node, 0))
    return nodes


def plan_pins(
    pins: list[str], gpus: str, per_gpu_gib: float, sysfs: Path
) -> dict[int, float]:
    """GiB to pin per NUMA node."""
    planned: dict[int, float] = defaultdict(float)
    for pin in pins:
        node, sep, gib = pin.partition(":")
        if not sep or not node.isdigit():
            raise SystemExit(f"[numa-budget][FAIL] '{pin}' is not <node>:<GiB>")
        planned[int(node)] += float(gib)
    ordinals = [int(ordinal) for ordinal in gpus.split(",") if ordinal.strip()]
    if ordinals and per_gpu_gib > 0:
        for node in gpu_numa_nodes(sysfs, ordinals):
            planned[node] += per_gpu_gib
    return dict(planned)


def check_budget(planned: dict[int, float], reserve_gib: float, sysfs: Path) -> int:
    """Print one line per node; 2 when a node cannot hold its pins, else 0."""
    status = 0
    for node in sorted(planned):
        info = node_meminfo(sysfs, node)
        total = info.get("MemTotal", 0) / GIB
        free = info.get("MemFree", 0) / GIB
        cache = info.get("FilePages", 0) / GIB
        line = (
            f"NUMA node {node}: pins {planned[node]:.0f} GiB, keeps "
            f"{reserve_gib:.0f} GiB in reserve, of {total:.0f} GiB; {free:.0f} GiB "
            f"free, {cache:.0f} GiB page cache"
        )
        if planned[node] + reserve_gib > total:
            print(
                f"[numa-budget][FAIL] {line}: the pins do not fit, and a bound "
                "process that overruns its node is killed out of memory",
                file=sys.stderr,
            )
            status = 2
        elif planned[node] > free:
            print(
                f"[numa-budget][WARN] {line}: each huge-page fault past the free "
                "memory reclaims page cache through compaction first, which can "
                "take minutes; drop the page cache of the files behind it",
                file=sys.stderr,
            )
        else:
            print(f"[numa-budget] {line}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pins", nargs="*", metavar="NODE:GIB")
    parser.add_argument(
        "--gpus", default="", help="HIP ordinals; each pins --per-gpu-gib"
    )
    parser.add_argument("--per-gpu-gib", type=float, default=0.0)
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=0.0,
        help="memory each node keeps for everything that is not pinned here",
    )
    parser.add_argument("--sysfs", type=Path, default=Path("/sys"))
    args = parser.parse_args(argv)
    planned = plan_pins(args.pins, args.gpus, args.per_gpu_gib, args.sysfs)
    return check_budget(planned, args.reserve_gib, args.sysfs)


if __name__ == "__main__":
    sys.exit(main())
