#!/usr/bin/env python3
"""ATOM's Store memory recipe: evict the models' page cache, then fault each NUMA node's share of the Store
segments as THP from one bound thread. Mooncake pre-touches from many threads whose racing compactions leave
4 KiB pages, and ionic maps a whole MR at 4 KiB for one of them (job 10077).

Usage: store_memory_prep.py <segment bytes on this node> <model dir> [<model dir> ...]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from drop_page_cache import drop_page_cache

HERE = Path(__file__).resolve().parent


def compact_nodes(segments: int) -> None:
    nodes = [p for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")) if (p / "cpulist").read_text().strip()]
    for node in nodes:
        info = {f[2].rstrip(":"): int(f[3]) << 10 for f in map(str.split, (node / "meminfo").read_text().splitlines())
                if len(f) == 5}
        # A process bound to a node is killed out of memory past what the node can free; the workers hold theirs.
        gib = min(segments / len(nodes), info["MemFree"] + info["Active(file)"] + info["Inactive(file)"] - (64 << 30)) / 2**30
        print(f"[C48] Store memory: compacting {node.name} for {gib:.0f} of {segments / len(nodes) / 2**30:.0f} GiB"
              f" (MemFree {info['MemFree'] >> 30} GiB, file {(info['Active(file)'] + info['Inactive(file)']) >> 30} GiB)",
              flush=True)
        if gib < 1:
            continue
        # 1211 GiB on node1 reached ~830 GiB of THP in 600 s (job 10183) and enough in 1800 s (job 10122).
        try:
            subprocess.run([sys.executable, str(HERE / "numa_exec.py"), node.name.removeprefix("node"), sys.executable,
                            str(HERE / "compact_huge_pages.py"), f"{gib:.0f}"], check=False, timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"[C48][WARN] Store memory: compacting {node.name} did not finish in 1800 s", flush=True)


def main(argv: list[str]) -> None:
    segments, roots = int(argv[0]), argv[1:]
    start = time.monotonic()
    for root in roots:
        drop_page_cache(root)
    print(f"[C48] Store memory: dropped the page cache of {roots} in {time.monotonic() - start:.0f} s", flush=True)
    compact_nodes(segments)


if __name__ == "__main__":
    main(sys.argv[1:])
