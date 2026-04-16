#!/usr/bin/env python3
"""Generate a resilient benchmark summary table.

This script is intentionally tolerant of partial or total benchmark failure:
- missing result JSON => case is marked FAIL
- invalid result JSON => case is marked FAIL
- successful result JSON => key performance metrics are shown

The script always exits with status 0 so the summary job can complete even when
all benchmark cases fail.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

TP_RE = re.compile(r"--tensor-parallel-size\s+(\d+)")


def _format_metric(value: object) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _failed_metric() -> str:
    return "FAIL"


def _expected_tensor_parallel(model: dict) -> int:
    extra_args = str(model.get("extra_args", ""))
    match = TP_RE.search(extra_args)
    if match:
        return int(match.group(1))
    return 1


def _expected_cases(matrix_payload: dict) -> list[dict]:
    cases: list[dict] = []
    for cell in matrix_payload.get("include", []):
        model = cell["model"]
        params = cell["params"]
        ratio_text = str(params["random_range_ratio"])
        result_filename = (
            f'{model["prefix"]}-'
            f'{params["input_length"]}-'
            f'{params["output_length"]}-'
            f'{params["concurrency"]}-'
            f"{ratio_text}.json"
        )
        cases.append(
            {
                "display": model.get("display", model["prefix"]),
                "prefix": model["prefix"],
                "isl": int(params["input_length"]),
                "osl": int(params["output_length"]),
                "concurrency": int(params["concurrency"]),
                "ratio": ratio_text,
                "tensor_parallel_size": _expected_tensor_parallel(model),
                "result_filename": result_filename,
            }
        )
    return cases


def _load_result(result_path: Path) -> tuple[dict | None, str]:
    if not result_path.exists():
        return None, "FAIL"

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "FAIL"

    if "output_throughput" not in payload:
        return None, "FAIL"

    return payload, "PASS"


def _note_for_case(status: str, result_path: Path, payload: dict | None) -> str:
    if status == "PASS":
        return ""
    if not result_path.exists():
        return "Result artifact missing"
    if payload is None:
        return "Invalid or incomplete benchmark JSON"
    return "Benchmark failed"


def _build_rows(result_dir: Path, matrix_payload: dict) -> list[dict]:
    rows: list[dict] = []
    for case in _expected_cases(matrix_payload):
        result_path = result_dir / case["result_filename"]
        payload, status = _load_result(result_path)
        tp_size = case["tensor_parallel_size"]
        if payload is not None:
            try:
                tp_size = int(payload.get("tensor_parallel_size", tp_size))
            except (TypeError, ValueError):
                tp_size = case["tensor_parallel_size"]

        total_tput = None if payload is None else payload.get("total_token_throughput")
        total_tput_per_gpu = None
        if total_tput is not None and tp_size > 0:
            try:
                total_tput_per_gpu = float(total_tput) / tp_size
            except (TypeError, ValueError):
                total_tput_per_gpu = None

        rows.append(
            {
                **case,
                "status": status,
                "note": _note_for_case(status, result_path, payload),
                "mean_ttft_ms": (
                    _failed_metric()
                    if payload is None
                    else _format_metric(payload.get("mean_ttft_ms"))
                ),
                "mean_tpot_ms": (
                    _failed_metric()
                    if payload is None
                    else _format_metric(payload.get("mean_tpot_ms"))
                ),
                "total_token_throughput": (
                    _failed_metric() if payload is None else _format_metric(total_tput)
                ),
                "total_token_throughput_per_gpu": (
                    _failed_metric()
                    if payload is None
                    else _format_metric(total_tput_per_gpu)
                ),
            }
        )
    return rows


def _print_markdown_table(rows: list[dict], run_url: str | None, title: str) -> None:
    total_cases = len(rows)
    passed_cases = sum(1 for row in rows if row["status"] == "PASS")
    failed_cases = total_cases - passed_cases

    print(f"## {title}\n")
    if run_url:
        print(f"Run: {run_url}\n")
    print(
        f"Expected cases: **{total_cases}**  "
        f"Passed: **{passed_cases}**  "
        f"Failed: **{failed_cases}**\n"
    )

    print(
        "| Status | Model | ISL | OSL | Concurrency | Ratio | "
        "Mean TTFT (ms) | Mean TPOT (ms) | Total Tput (tok/s) | "
        "Total Tput / GPU (tok/s) | Notes |"
    )
    print("| :-: | :- | -: | -: | -: | -: | -: | -: | -: | -: | :- |")

    for row in rows:
        status_text = "PASS" if row["status"] == "PASS" else "FAIL"
        print(
            "| {status} | {model} | {isl} | {osl} | {conc} | {ratio} | "
            "{ttft} | {tpot} | {total_tput} | {total_tput_per_gpu} | {note} |".format(
                status=status_text,
                model=row["display"],
                isl=row["isl"],
                osl=row["osl"],
                conc=row["concurrency"],
                ratio=row["ratio"],
                ttft=row["mean_ttft_ms"],
                tpot=row["mean_tpot_ms"],
                total_tput=row["total_token_throughput"],
                total_tput_per_gpu=row["total_token_throughput_per_gpu"],
                note=row["note"] or "-",
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory containing downloaded benchmark JSON files",
    )
    parser.add_argument(
        "--matrix-json",
        required=True,
        help="Benchmark matrix JSON from the workflow",
    )
    parser.add_argument(
        "--run-url",
        default=None,
        help="Optional GitHub Actions run URL for the summary header",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write a structured summary report",
    )
    parser.add_argument(
        "--title",
        default="Benchmark Summary",
        help="Title for the markdown report",
    )
    args = parser.parse_args()

    matrix_payload = json.loads(args.matrix_json)
    rows = _build_rows(Path(args.result_dir), matrix_payload)

    _print_markdown_table(rows, args.run_url, args.title)

    if args.output_json:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_cases": len(rows),
            "passed_cases": sum(1 for row in rows if row["status"] == "PASS"),
            "failed_cases": sum(1 for row in rows if row["status"] == "FAIL"),
            "cases": rows,
        }
        Path(args.output_json).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
