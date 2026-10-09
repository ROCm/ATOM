#!/usr/bin/env python3
"""Render AIPerf's own console summary tables as a markdown section.

process_result.py reduces every run to one row of a comparison table, which
drops the metrics AIPerf reports but the dashboard does not track -- prefill vs
decode throughput, tokens in flight, CO-aware latency, the full percentile
spread. Those matter when triaging an agentic run, so the raw tables are
reproduced verbatim next to the summary.

Only agentic runs have anything to show: profile_export_console.txt is written
by AIPerf alone, so its absence is what makes this a no-op for standard
ISL/OSL cases rather than an explicit benchmark-kind check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CONSOLE_FILENAME = "profile_export_console.txt"


def artifact_label(path: Path) -> str:
    """Identify the run a console dump belongs to.

    The artifact directory is named aiperf-<model>-<topology>-c<conc>, which is
    the case identity; the slurm_job-<id> above it disambiguates re-runs of the
    same case merged into one result tree.
    """
    parts = [path.parent.name]
    for parent in path.parents:
        if parent.name.startswith("slurm_job-"):
            parts.insert(0, parent.name)
            break
    return " / ".join(parts)


def render(root: Path) -> str:
    sections = []
    for path in sorted(root.rglob(CONSOLE_FILENAME)):
        try:
            body = path.read_text(encoding="utf-8", errors="replace").strip("\n")
        except OSError as exc:
            print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
            continue
        if not body:
            continue
        # Collapsed by default: each dump is ~80 lines of fixed-width tables and
        # a run can hold several, which would bury the comparison table above it.
        sections.append(
            f"<details><summary>AIPerf raw metrics — {artifact_label(path)}"
            f"</summary>\n\n```text\n{body}\n```\n\n</details>"
        )
    if not sections:
        return ""
    return "\n\n".join(["### AIPerf Raw Metrics (agentic runs)", *sections]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", help="Directory containing benchmark artifacts")
    parser.add_argument(
        "--output", default=None, help="Write markdown here instead of stdout"
    )
    args = parser.parse_args()

    markdown = render(Path(args.result_dir))
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(
            f"AIPerf raw metrics: {'written to ' + args.output if markdown else 'none found'}"
        )
        return
    sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
