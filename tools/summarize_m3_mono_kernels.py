#!/usr/bin/env python3
"""Print duration distributions for MiniMax-M3 mono kernels in a reduced trace."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    args = parser.parse_args()

    with gzip.open(args.trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    durations: dict[str, list[float]] = collections.defaultdict(list)
    for event in events:
        name = event.get("name", "")
        if event.get("cat") == "kernel" and "minimax_m3_mono" in name:
            durations[name].append(float(event["dur"]))
    for name, values in sorted(durations.items()):
        values.sort()
        p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
        print(f"{len(values):6d} {statistics.median(values):9.3f} {p90:9.3f} {name}")


if __name__ == "__main__":
    main()
