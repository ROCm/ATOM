#!/usr/bin/env python3
"""Summarize gfx1250 MiniMax-M3 mono per-wave timeline captures."""

from __future__ import annotations

import argparse
import collections
import glob
import math
import statistics
from pathlib import Path

import torch

POINTS = 32
WAVES = 16
TICK_US = 0.01  # s_memrealtime runs at 100 MHz.


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def critical_interval(layer: torch.Tensor, start: int, end: int) -> int | None:
    """Longest same-wave interval; XCD clocks are not mutually aligned."""
    starts = layer[:, start, :]
    ends = layer[:, end, :]
    valid = (starts != 0) & (ends != 0) & (ends >= starts)
    values = ends[valid] - starts[valid]
    return int(values.max()) if values.numel() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", help="glob for mono_tl_r*_s*_*.pt")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files match {args.pattern!r}")

    stages: dict[str, list[float]] = {
        "K1": [],
        "select/split after K1": [],
        "merge + attention output": [],
        "router": [],
        "MoE + FFN reduce": [],
        "whole sparse layer": [],
    }
    details: dict[str, list[float]] = {
        "K1 0->27": [],
        "K1 27->28": [],
        "K1 28->29": [],
        "K1 29->30": [],
        "K1 30->31": [],
        "MoE route rows 5->2": [],
        "MoE expert table 2->6": [],
        "MoE preload 6->7": [],
        "MoE up/gate 7->8": [],
        "MoE down 8->11": [],
        "MoE FFN tail 11->16": [],
        "router input wait 4->22": [],
        "router norm reduce 22->24": [],
        "router normalize 24->25": [],
        "router quant publish 25->26": [],
        "router gate GEMV 26->23": [],
        "router stage tail 23->5": [],
        "attention select 31->17": [],
        "attention operand load 17->12": [],
        "attention QK 12->13": [],
        "attention softmax 13->14": [],
        "attention PV 14->15": [],
        "attention split publish 15->1": [],
        "attention merge wait 1->3": [],
        "attention O projection 3->4": [],
    }
    detail_points = {
        "K1 0->27": (0, 27),
        "K1 27->28": (27, 28),
        "K1 28->29": (28, 29),
        "K1 29->30": (29, 30),
        "K1 30->31": (30, 31),
        "MoE route rows 5->2": (5, 2),
        "MoE expert table 2->6": (2, 6),
        "MoE preload 6->7": (6, 7),
        "MoE up/gate 7->8": (7, 8),
        "MoE down 8->11": (8, 11),
        "MoE FFN tail 11->16": (11, 16),
        "router input wait 4->22": (4, 22),
        "router norm reduce 22->24": (22, 24),
        "router normalize 24->25": (24, 25),
        "router quant publish 25->26": (25, 26),
        "router gate GEMV 26->23": (26, 23),
        "router stage tail 23->5": (23, 5),
        "attention select 31->17": (31, 17),
        "attention operand load 17->12": (17, 12),
        "attention QK 12->13": (12, 13),
        "attention softmax 13->14": (13, 14),
        "attention PV 14->15": (14, 15),
        "attention split publish 15->1": (15, 1),
        "attention merge wait 1->3": (1, 3),
        "attention O projection 3->4": (3, 4),
    }
    rank_whole: dict[str, list[float]] = {}
    cta_whole: dict[int, list[float]] = collections.defaultdict(list)

    for filename in files:
        tensor = torch.load(filename, map_location="cpu")
        if tensor.ndim != 3 or tensor.shape[1:] != (256, POINTS * WAVES):
            raise ValueError(f"{filename}: unexpected shape {tuple(tensor.shape)}")
        tensor = tensor.view(tensor.shape[0], 256, POINTS, WAVES)
        rank = Path(filename).name.split("_")[2]
        rank_whole.setdefault(rank, [])

        for layer in tensor:
            ticks = {
                "K1": critical_interval(layer, 0, 31),
                "select/split after K1": critical_interval(layer, 31, 1),
                "merge + attention output": critical_interval(layer, 1, 4),
                "router": critical_interval(layer, 4, 5),
                "MoE + FFN reduce": critical_interval(layer, 5, 16),
                "whole sparse layer": critical_interval(layer, 0, 16),
            }
            if any(value is None for value in ticks.values()):
                continue

            for name, value in ticks.items():
                stages[name].append(value * TICK_US)
            rank_whole[rank].append(ticks["whole sparse layer"] * TICK_US)
            starts = layer[:, 0, :]
            ends = layer[:, 16, :]
            for cta in range(layer.shape[0]):
                valid = (
                    (starts[cta] != 0) & (ends[cta] != 0) & (ends[cta] >= starts[cta])
                )
                values = ends[cta][valid] - starts[cta][valid]
                if values.numel():
                    cta_whole[cta].append(float(values.max()) * TICK_US)
            for name, (start, end) in detail_points.items():
                value = critical_interval(layer, start, end)
                if value is not None:
                    details[name].append(value * TICK_US)

    whole_median = statistics.median(stages["whole sparse layer"])
    print(f"files={len(files)} valid_layer_samples={len(stages['whole sparse layer'])}")
    print("stage,median_us,p90_us,share_of_layer")
    for name, values in stages.items():
        median = statistics.median(values)
        print(
            f"{name},{median:.3f},{percentile(values, 0.90):.3f},"
            f"{median / whole_median:.1%}"
        )
    print("rank,median_layer_us,p90_layer_us")
    for rank, values in sorted(rank_whole.items()):
        print(
            f"{rank},{statistics.median(values):.3f}," f"{percentile(values, 0.90):.3f}"
        )
    print("detail,median_us,p90_us")
    for name, values in details.items():
        if values:
            print(
                f"{name},{statistics.median(values):.3f},"
                f"{percentile(values, 0.90):.3f}"
            )
    print("slowest_cta,median_us,p90_us")
    ranked = sorted(
        cta_whole.items(), key=lambda item: statistics.median(item[1]), reverse=True
    )
    for cta, values in ranked[:32]:
        print(
            f"{cta},{statistics.median(values):.3f}," f"{percentile(values, 0.90):.3f}"
        )


if __name__ == "__main__":
    main()
