#! /usr/bin/env python3
"""Summarize ATOM benchmark results with optional regression detection.

Usage:
    # Basic (existing behavior):
    python summarize.py <result_dir>

    # With regression detection against a previous run:
    python summarize.py <result_dir> --baseline-dir <baseline_dir>
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

THROUGHPUT_REGRESSION_PCT = -5.0  # throughput drop >5% = regression
LATENCY_REGRESSION_PCT = 10.0  # latency increase >10% = regression

TRACKED_METRICS = [
    # (json_key, display_name, higher_is_better)
    ("output_throughput", "Output Tput", True),
    ("total_token_throughput", "Total Tput", True),
    ("mean_ttft_ms", "Mean TTFT", False),
    ("mean_tpot_ms", "Mean TPOT", False),
]


def load_results(result_dir, recursive=False):
    """Load benchmark JSON files from a directory.

    Args:
        result_dir: Path to directory containing JSON result files.
        recursive: If True, search subdirectories (needed for ``gh run download``
                   output which nests each artifact in its own folder).

    Returns:
        Sorted list of benchmark result dicts.
    """
    root = Path(result_dir)
    if not root.exists():
        return []

    glob_fn = root.rglob if recursive else root.glob
    results = []
    for json_path in glob_fn("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if "output_throughput" not in data:
            continue
        if "random_input_len" not in data or "random_output_len" not in data:
            parts = json_path.stem.rsplit("-", 4)
            if len(parts) == 5:
                data.setdefault("random_input_len", int(parts[1]))
                data.setdefault("random_output_len", int(parts[2]))
        results.append(data)

    results.sort(
        key=lambda d: (
            d.get("model_id", "").split("/")[-1],
            int(d.get("random_input_len", 0)),
            int(d.get("random_output_len", 0)),
            int(d.get("max_concurrency", 0)),
        )
    )
    return results


def _config_key(data):
    """Unique identifier for matching a benchmark configuration across runs."""
    return (
        data.get("model_id", "").split("/")[-1],
        int(data.get("random_input_len", 0)),
        int(data.get("random_output_len", 0)),
        int(data.get("max_concurrency", 0)),
    )


def _pct_change(current, baseline):
    if baseline == 0:
        return float("inf") if current != 0 else 0.0
    return ((current - baseline) / abs(baseline)) * 100.0


def _is_regression(pct, higher_is_better):
    if higher_is_better:
        return pct <= THROUGHPUT_REGRESSION_PCT
    else:
        return pct >= LATENCY_REGRESSION_PCT


def _format_delta(value, pct, higher_is_better):
    """Format a metric value with percentage change indicator."""
    sign = "+" if pct >= 0 else ""
    if _is_regression(pct, higher_is_better):
        return f"**{value:.2f}** ({sign}{pct:.1f}%)"
    return f"{value:.2f} ({sign}{pct:.1f}%)"


def print_results_table(results):
    """Print the full benchmark results table (preserves existing output format)."""
    summary_header = """\
| Date | Backend | Model | ISL | OSL | Best of | Number of prompts | Request rate | Burstiness | Max concurrency | Duration | Completed | Total input tokens | Total output tokens | Request throughput | Request goodput | Output throughput | Total token throughput | Mean TTFT (ms) | Median TTFT (ms) | Std TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Std TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Std ITL (ms) | P99 ITL (ms) | Mean E2EL (ms) | Median E2EL (ms) | Std E2EL (ms) | P99 E2EL (ms) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
    """
    print(summary_header)

    for data in results:
        row = [
            datetime.datetime.strptime(data.get("date", ""), "%Y%m%d-%H%M%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "ATOM",
            data.get("model_id", "").split("/")[-1],
            data.get("random_input_len", ""),
            data.get("random_output_len", ""),
            data.get("best_of", ""),
            data.get("num_prompts", ""),
            data.get("request_rate", ""),
            f"{data.get('burstiness', '')}%",
            f"{data.get('max_concurrency', '')}",
            f"{data.get('duration', ''):.2f}",
            data.get("completed", ""),
            f"{data.get('total_input_tokens', ''):.2f}",
            f"{data.get('total_output_tokens', ''):.2f}",
            f"{data.get('request_throughput', ''):.2f}",
            data.get("request_goodput", ""),
            f"{data.get('output_throughput', ''):.2f}",
            f"{data.get('total_token_throughput', ''):.2f}",
            f"{data.get('mean_ttft_ms', ''):.2f}",
            f"{data.get('median_ttft_ms', ''):.2f}",
            f"{data.get('std_ttft_ms', ''):.2f}",
            f"{data.get('p99_ttft_ms', ''):.2f}",
            f"{data.get('mean_tpot_ms', ''):.2f}",
            f"{data.get('median_tpot_ms', ''):.2f}",
            f"{data.get('std_tpot_ms', ''):.2f}",
            f"{data.get('p99_tpot_ms', ''):.2f}",
            f"{data.get('mean_itl_ms', ''):.2f}",
            f"{data.get('median_itl_ms', ''):.2f}",
            f"{data.get('std_itl_ms', ''):.2f}",
            f"{data.get('p99_itl_ms', ''):.2f}",
            f"{data.get('mean_e2el_ms', ''):.2f}",
            f"{data.get('median_e2el_ms', ''):.2f}",
            f"{data.get('std_e2el_ms', ''):.2f}",
            f"{data.get('p99_e2el_ms', ''):.2f}",
        ]
        print("| " + " | ".join(str(x) for x in row) + " |")


def print_regression_report(current_results, baseline_results):
    """Compare current results against baseline and print a regression summary.

    Returns:
        Number of configurations with at least one regressed metric.
    """
    baseline_map = {_config_key(d): d for d in baseline_results}
    if not baseline_map:
        return 0

    print("\n---\n")
    print("## Regression Report\n")
    print(
        f"Compared against previous nightly run "
        f"({len(baseline_map)} baseline configurations).  "
    )
    print(
        f"Thresholds: throughput drop **>{abs(THROUGHPUT_REGRESSION_PCT):.0f}%** "
        f"or latency increase **>{LATENCY_REGRESSION_PCT:.0f}%**\n"
    )

    cols = ["Model", "ISL", "OSL", "Conc"]
    for _, display_name, _ in TRACKED_METRICS:
        cols.append(display_name)
    cols.append("Status")

    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join([":-:"] * len(cols)) + " |")

    regression_count = 0

    for data in current_results:
        key = _config_key(data)
        baseline = baseline_map.get(key)
        model, isl, osl, conc = key
        row = [model, str(isl), str(osl), str(conc)]

        has_regression = False

        for metric_key, _, higher_is_better in TRACKED_METRICS:
            cur_val = data.get(metric_key, 0)
            if baseline is not None:
                base_val = baseline.get(metric_key, 0)
                pct = _pct_change(cur_val, base_val)
                row.append(_format_delta(cur_val, pct, higher_is_better))
                if _is_regression(pct, higher_is_better):
                    has_regression = True
            else:
                row.append(f"{cur_val:.2f}")

        if has_regression:
            row.append("⚠️ **REGRESSION**")
            regression_count += 1
        elif baseline is None:
            row.append("🆕 New")
        else:
            row.append("✅ OK")

        print("| " + " | ".join(row) + " |")

    print()
    if regression_count > 0:
        print(
            f"> ⚠️ **{regression_count} configuration(s) show performance regression**"
        )
    else:
        print("> ✅ **No regressions detected** across all configurations")

    return regression_count


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ATOM benchmark results with optional regression detection"
    )
    parser.add_argument(
        "result_dir",
        help="Directory containing current benchmark result JSON files",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Directory containing baseline result JSON files for comparison",
    )
    args = parser.parse_args()

    current_results = load_results(args.result_dir)
    if not current_results:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(1)

    print_results_table(current_results)

    if args.baseline_dir:
        baseline_results = load_results(args.baseline_dir, recursive=True)
        if baseline_results:
            print_regression_report(current_results, baseline_results)
        else:
            print("\n> No baseline results found for regression comparison\n")


if __name__ == "__main__":
    main()
