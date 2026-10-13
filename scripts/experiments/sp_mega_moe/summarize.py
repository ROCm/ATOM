#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate requests and summarize paired serving runs without mixing budgets."""

import argparse
import json
import statistics
from pathlib import Path

from run_ab import validate_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.output / "plan.json").read_text())
    pairs = {}
    for case in plan["cases"]:
        directory = args.output / Path(case["directory"]).name
        command = case["client"]
        output_len = int(command[command.index("--random-output-len") + 1])
        validate_result(directory / "result.json", output_len)
        data = json.loads((directory / "result.json").read_text())
        pair = directory.name.rsplit("-", 1)[0]
        pairs.setdefault((case["batch"], pair), {})[case["mode"]] = data
    metrics = (
        "total_token_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
        "p99_ttft_ms",
        "p99_itl_ms",
    )
    rows = []
    for batch in sorted({key[0] for key in pairs}):
        selected = [v for (b, _), v in pairs.items() if b == batch]
        if any(set(pair) != {"native", "mega"} for pair in selected):
            raise RuntimeError(f"Missing comparison arm for budget {batch}")
        for metric in metrics:
            native = [pair["native"][metric] for pair in selected]
            mega = [pair["mega"][metric] for pair in selected]
            changes = [100 * (m / n - 1) if n else None for n, m in zip(native, mega)]
            rows.append(
                {
                    "batch": batch,
                    "metric": metric,
                    "native": native,
                    "mega": mega,
                    "native_median": statistics.median(native),
                    "mega_median": statistics.median(mega),
                    "paired_change_percent": changes,
                    "paired_change_median_percent": (
                        statistics.median(changes) if None not in changes else None
                    ),
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
