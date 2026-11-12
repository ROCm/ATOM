#!/usr/bin/env python3
"""Write the 455-CI report of one image from perf.csv and accuracy.csv.

    python3 make_report.py --image-date 2026-11-14 [--data-dir recipes/455-CI]

Writes reports/455-CI-report-<YYYYMMDD>.md (highlights and the summary table,
for the boss) and reports/455-CI-report-<YYYYMMDD>-detail.md (environment,
every number, the trend charts of make_charts.py and each failure's core error
log from runs/<YYYYMMDD>/<slug>/error.log), and prints the highlights. The
environment comes from runs/<YYYYMMDD>/image.json (455-CI-guide.md §2.4),
else from the CSV rows. A scenario without results on the image is reported
as pending.
"""

import argparse
import json
import os

from ci455_data import LABEL, MODELS, acc_cell, perf_cells, read_csv
from ci455_report import ModelResult, detail_md, highlights, summary_md

HERE = os.path.dirname(os.path.abspath(__file__))


def _rows(path):
    return read_csv(path) if os.path.exists(path) else []


def _read(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def environment(run_dir, rows):
    """image.json, filled in from the image's CSV rows where it is missing."""
    path = os.path.join(run_dir, "image.json")
    env = json.loads(_read(path)) if os.path.exists(path) else {}
    if rows:
        r = rows[0]
        env.setdefault("image", r.get("image"))
        env.setdefault("base_image", r.get("base_image"))
        env.setdefault("atom", {"commit": r.get("atom_commit")})
        env.setdefault("aiter", {"commit": r.get("aiter_commit")})
    env["test_dates"] = {}
    for s in ("endpoint", "8k1k"):
        dates = sorted(
            r["test_date"] for r in rows if r["scenario"] == s and r.get("test_date")
        )
        if dates:
            env["test_dates"][s] = dates[0]
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image-date", required=True, help="image date, YYYY-MM-DD")
    ap.add_argument(
        "--data-dir",
        default=os.path.dirname(HERE),
        help="dir of perf.csv / accuracy.csv",
    )
    ap.add_argument("--mock", action="store_true", help="mark the report as mock data")
    args = ap.parse_args(argv)

    day, tag = args.image_date, args.image_date.replace("-", "")
    perf = _rows(os.path.join(args.data_dir, "perf.csv"))
    acc = _rows(os.path.join(args.data_dir, "accuracy.csv"))
    run_dir = os.path.join(args.data_dir, "runs", tag)
    out = os.path.join(args.data_dir, "reports")
    on_image = [r for r in perf + acc if r["image_date"] == day]

    results, pending = [], []
    for scenario in ("endpoint", "8k1k"):
        if not any(r["scenario"] == scenario for r in on_image):
            pending.append(LABEL[scenario])
            continue
        for slug, name in MODELS[scenario]:
            results.append(
                ModelResult(
                    scenario,
                    slug,
                    name,
                    perf_cells(perf, scenario, slug, day),
                    acc_cell(acc, scenario, slug, day),
                    _read(os.path.join(run_dir, slug, "error.log")),
                )
            )

    env = environment(run_dir, on_image)
    summary, detail = f"455-CI-report-{tag}.md", f"455-CI-report-{tag}-detail.md"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, summary), "w", encoding="utf-8") as f:
        f.write(summary_md(day, env, results, pending, detail, args.mock))
    with open(os.path.join(out, detail), "w", encoding="utf-8") as f:
        f.write(
            detail_md(
                day,
                env,
                results,
                pending,
                summary,
                os.path.relpath(os.path.join(run_dir, "charts"), out),
                os.path.relpath(run_dir, out),
                args.mock,
            )
        )
    print(f"wrote {os.path.join(out, summary)}\nwrote {os.path.join(out, detail)}\n")
    for line in highlights(results, pending):
        print("-", line)


if __name__ == "__main__":
    main()
