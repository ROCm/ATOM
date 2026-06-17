#!/usr/bin/env python3
"""Run Atomesh mocker benchmark cells and generate an aggregate summary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Atomesh mocker benchmark cells and summarize results."
    )
    parser.add_argument(
        "--cells-json",
        default=os.environ.get("CELLS_JSON", "[]"),
        help="JSON array of benchmark cells. Defaults to CELLS_JSON.",
    )
    parser.add_argument(
        "--result-dir",
        default=os.environ.get("RESULT_DIR", "atomesh-mocker-results"),
        help="Directory where per-cell results and summary are written.",
    )
    parser.add_argument(
        "--benchmark-script",
        default=".github/scripts/atomesh_mocker_benchmark.sh",
        help="Single-cell benchmark script to invoke.",
    )
    return parser.parse_args()


def run_cells(cells: list[dict], result_dir: Path, benchmark_script: str) -> int:
    result_dir.mkdir(parents=True, exist_ok=True)

    for index, cell in enumerate(cells, start=1):
        print(
            f"=== Running benchmark cell {index}/{len(cells)}: {cell['display']} ===",
            flush=True,
        )
        env = os.environ.copy()
        env.update(
            {
                "BENCHMARK_NAME": cell["id"],
                "SCENARIO": cell["scenario"],
                "DURATION": cell["duration"],
                "PREFILL_WORKERS": str(cell["prefill_workers"]),
                "DECODE_WORKERS": str(cell["decode_workers"]),
                "PRODUCER_THREADS": str(cell["producer_threads"]),
                "CONSUMER_THREADS": str(cell["consumer_threads"]),
                "RESULT_DIR": str(result_dir),
            }
        )
        try:
            subprocess.run([benchmark_script], check=True, env=env)
        except subprocess.CalledProcessError as error:
            print(
                f"Benchmark cell {cell['id']} failed with status {error.returncode}",
                file=sys.stderr,
            )
            return error.returncode

    return 0


def collect_rows(result_dir: Path) -> list[tuple]:
    rows = []
    for path in sorted(result_dir.glob("pd-chat-*.json")):
        if path.name.endswith("-benchmark-action.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                payload["prefill_workers"],
                payload["decode_workers"],
                payload["consumer_threads"],
                payload["duration_seconds"],
                payload["completed"],
                payload["failed"],
                payload["request_throughput"],
                payload["avg_latency_ms"],
                payload["p99_latency_ms"],
                payload["p999_latency_ms"],
            )
        )
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return rows


def write_summary(result_dir: Path) -> str:
    rows = collect_rows(result_dir)
    lines = [
        "### Atomesh Mocker Benchmark Summary",
        "",
        "| Topology | Concurrency | Duration (s) | Completed | Failed | Throughput (req/s) | Avg Latency (ms) | P99 (ms) | P999 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (
        prefill,
        decode,
        consumers,
        duration,
        completed,
        failed,
        throughput,
        avg,
        p99,
        p999,
    ) in rows:
        lines.append(
            f"| {prefill}P{decode}D | {consumers} | {duration} | {completed} | {failed} | "
            f"{throughput:.2f} | {avg:.3f} | {p99:.3f} | {p999:.3f} |"
        )

    summary = "\n".join(lines) + "\n"
    (result_dir / "benchmark-summary.md").write_text(summary, encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    cells = json.loads(args.cells_json)
    result_dir = Path(args.result_dir)

    exit_code = run_cells(cells, result_dir, args.benchmark_script)
    summary = write_summary(result_dir)
    print(summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
