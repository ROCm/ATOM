#!/usr/bin/env python3
"""Write the 455-CI report of one session from perf.csv and accuracy.csv.

    python3 make_report.py --date 2026-11-18 [--scenario endpoint|8k1k] [--data-dir recipes/455-CI] [--mock]

A session is one day's run (Saturday: MLPerf Endpoint on the image built that
day; Wednesday: 8K/1K on the previous Saturday's image). Writes one file,
reports/455-CI-report-<YYYYMMDD>.md: the short summary (highlights, tagging
each model's owner from owners.csv, and the summary table), then the details
(environment, every number, the trend charts of make_charts.py and each
failure's core error log from runs/<YYYYMMDD>/<slug>/error.log). Prints the
highlights, a warning for news without an owner, and last the report's path.
The image comes from runs/<image date>/image.json, written when the image was
built (455-CI-guide.md §2.4), else from the CSV rows.
"""

import argparse
import json
import os

from ci455_data import MODELS, acc_cell, perf_cells, read_csv
from ci455_report import ModelResult, highlights, missing_owners, report_md

HERE = os.path.dirname(os.path.abspath(__file__))


def _rows(path):
    return read_csv(path) if os.path.exists(path) else []


def _read(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def environment(data_dir, rows):
    """The session's image: its image.json, filled in from the CSV rows."""
    env = {}
    if rows:
        r = rows[0]
        path = os.path.join(
            data_dir, "runs", r["image_date"].replace("-", ""), "image.json"
        )
        env = json.loads(_read(path)) if os.path.exists(path) else {}
        env.setdefault("image", r.get("image"))
        env.setdefault("atom", {"commit": r.get("atom_commit")})
        env.setdefault("aiter", {"commit": r.get("aiter_commit")})
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", required=True, help="session (test) date, YYYY-MM-DD")
    ap.add_argument(
        "--scenario", choices=["endpoint", "8k1k"], help="default: what ran that day"
    )
    ap.add_argument(
        "--data-dir",
        default=os.path.dirname(HERE),
        help="dir of perf.csv / accuracy.csv",
    )
    ap.add_argument("--mock", action="store_true", help="mark the report as an example")
    args = ap.parse_args(argv)

    day, tag = args.date, args.date.replace("-", "")
    perf = _rows(os.path.join(args.data_dir, "perf.csv"))
    acc = _rows(os.path.join(args.data_dir, "accuracy.csv"))
    today = [r for r in perf + acc if r["test_date"] == day]
    scenarios = (
        [args.scenario]
        if args.scenario
        else [s for s in MODELS if any(r["scenario"] == s for r in today)]
    )
    if not scenarios:
        print(f"no session on {day}: nothing to report")
        return
    run_dir = os.path.join(args.data_dir, "runs", tag)
    results = [
        ModelResult(
            s,
            slug,
            name,
            perf_cells(perf, s, slug, day),
            acc_cell(acc, s, slug, day),
            _read(os.path.join(run_dir, slug, "error.log")),
        )
        for s in scenarios
        for slug, name in MODELS[s]
    ]
    env = environment(args.data_dir, [r for r in today if r["scenario"] in scenarios])
    owners = {
        r["model"]: r.get("owner", "")
        for r in _rows(os.path.join(args.data_dir, "owners.csv"))
    }

    out = os.path.join(args.data_dir, "reports")
    path = os.path.abspath(os.path.join(out, f"455-CI-report-{tag}.md"))
    os.makedirs(out, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            report_md(
                day,
                env,
                results,
                os.path.relpath(os.path.join(run_dir, "charts"), out),
                os.path.relpath(run_dir, out),
                args.mock,
                owners,
            )
        )
    for line in highlights(results, owners):
        print("-", line)
    for name in missing_owners(results, owners):
        print(
            f"WARNING: {name} is in the highlights but has no owner in owners.csv, so nobody is tagged."
        )
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
