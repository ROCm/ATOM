#!/usr/bin/env python3
"""Regression summary built on top of shared summarize helpers."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import summarize


def build_report(
    current_results: list[dict],
    regression_count: int,
    regressions: list[dict],
) -> dict:
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "regression_count": regression_count,
        "regressions": regressions,
        "all_results": [
            {
                "backend": summarize._backend_name(result),
                "model": summarize._display_model(result),
                "isl": int(result.get("random_input_len", 0)),
                "osl": int(result.get("random_output_len", 0)),
                "conc": int(result.get("max_concurrency", 0)),
                "output_throughput": result.get("output_throughput", 0),
                "total_token_throughput": result.get("total_token_throughput", 0),
                "mean_ttft_ms": result.get("mean_ttft_ms", 0),
                "mean_tpot_ms": result.get("mean_tpot_ms", 0),
            }
            for result in current_results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print only the regression report without the full results table"
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory containing current benchmark result JSON files",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Directory containing baseline benchmark result JSON files",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write structured regression JSON report",
    )
    args = parser.parse_args()

    current_results = summarize.load_results(args.result_dir)
    if not current_results:
        print("> No successful benchmark results found for regression comparison.\n")
        report = build_report([], 0, [])
        Path(args.output_json).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return 0

    regression_count = 0
    regressions: list[dict] = []

    if args.baseline_dir:
        baseline_results = summarize.load_results(args.baseline_dir, recursive=True)
        if baseline_results:
            regression_count, regressions = summarize.print_regression_report(
                current_results, baseline_results
            )
        else:
            print("\n> No baseline results found for regression comparison\n")
    else:
        print("\n> No baseline results found for regression comparison\n")

    report = build_report(current_results, regression_count, regressions)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 2 if regression_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
