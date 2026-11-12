#!/usr/bin/env python3
"""Write the 455-CI report of one session from perf.csv and accuracy.csv.

    python3 make_report.py --date 2026-11-18 [--scenario endpoint|8k1k] [--data-dir recipes/455-CI] [--mock]

A session is one day's run (Saturday: MLPerf Endpoint on the image built that
day; Wednesday: 8K/1K on the previous Saturday's image). Writes
reports/455-CI-report-<YYYYMMDD>.md (highlights and the summary table) and
reports/455-CI-report-<YYYYMMDD>-detail.md (environment, every number, the
trend charts of make_charts.py and each failure's core error log from
runs/<YYYYMMDD>/<slug>/error.log), and prints the highlights. The image comes
from runs/<image date>/image.json, written when the image was built
(455-CI-guide.md §2.4), else from the CSV rows.
"""

import argparse
import json
import os

from ci455_data import MODELS, acc_cell, perf_cells, read_csv
from ci455_report import ModelResult, detail_md, highlights, summary_md

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

    out = os.path.join(args.data_dir, "reports")
    summary, detail = f"455-CI-report-{tag}.md", f"455-CI-report-{tag}-detail.md"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, summary), "w", encoding="utf-8") as f:
        f.write(summary_md(day, env, results, detail, args.mock))
    with open(os.path.join(out, detail), "w", encoding="utf-8") as f:
        f.write(
            detail_md(
                day,
                env,
                results,
                summary,
                os.path.relpath(os.path.join(run_dir, "charts"), out),
                os.path.relpath(run_dir, out),
                args.mock,
            )
        )
    print(f"wrote {os.path.join(out, summary)}\nwrote {os.path.join(out, detail)}\n")
    for line in highlights(results):
        print("-", line)


if __name__ == "__main__":
    main()
