#!/usr/bin/env python3
"""Write the 455-CI result of one session from perf.csv and accuracy.csv.

    python3 make_report.py --date 2026-11-18 [--scenario endpoint|8k1k] [--data-dir recipes/455-CI] [--out FILE] [--mock]

A session is one day's run (Saturday: MLPerf Endpoint, Wednesday: 8K/1K), each on
the image built for it that day from ATOM main and aiter main. Writes one file,
recipes/result/<YYYYMMDD>-result.md (the result dir next to the data dir, or
--out), titled "<date> MI455 Workload Weekly Status Report": the lowlights and
highlights (each left out when empty), the results table, the environment, the findings from
runs/<YYYYMMDD>/findings.md, then the details (environment, every number, the
trend charts of make_charts.py and each failure's core error log from
runs/<YYYYMMDD>/<slug>/error.log). Each problem
in the lowlights gets its next step from followups.csv (455-CI-guide.md §3.7);
for one without, the unmerged PRs pending_fixes.csv lists for the model are
suggested as a ready followups.csv row.
GSM8K is also held to the known-good value in accuracy_baseline.csv (model,
baseline, source). Prints the lowlights and highlights, a warning for a problem without a follow-up, and last the
report's path. The result tags no one: the owners are added where it is posted.
The image comes from runs/<image date>/image.json, written when the image was
built (455-CI-guide.md §2.4), else from the CSV rows.
"""

import argparse
import csv
import io
import json
import os

from ci455_data import FOLLOWUP_COLUMNS, MODELS, acc_cell, perf_cells, read_csv
from ci455_report import (
    ModelResult,
    highlights_md,
    lowlights_md,
    missing_followups,
    pr_label,
    report_md,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _rows(path):
    return read_csv(path) if os.path.exists(path) else []


def _read(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


CMD_ORDER = ("server", "gsm8k", "bench")


def commands(model_dir):
    """(file name, text) of a model's recorded commands, *_cmd*.sh in its run
    dir: the servers first, then GSM8K, then the bench (guide §3.1)."""
    if not os.path.isdir(model_dir):
        return []
    names = [n for n in os.listdir(model_dir) if n.endswith(".sh") and "_cmd" in n]
    rank = {
        n: next((i for i, k in enumerate(CMD_ORDER) if n.startswith(k)), len(CMD_ORDER))
        for n in names
    }
    return [
        (n, _read(os.path.join(model_dir, n)))
        for n in sorted(names, key=lambda n: (rank[n], n))
    ]


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
        for repo in ("atom", "aiter"):
            if r.get(f"{repo}_branch"):
                env[repo].setdefault("branch", r[f"{repo}_branch"])
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", required=True, help="session (test) date, YYYY-MM-DD")
    ap.add_argument(
        "--scenario",
        choices=["endpoint", "8k1k"],
        help="only for a one-scenario preview; leave it out for the result, which holds every scenario run that day",
    )
    ap.add_argument(
        "--data-dir",
        default=os.path.dirname(HERE),
        help="dir of perf.csv / accuracy.csv",
    )
    ap.add_argument(
        "--out",
        help="result file; default ../result/<YYYYMMDD>-result.md of the data dir",
    )
    ap.add_argument("--mock", action="store_true", help="mark the result as an example")
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
    baselines = {}
    for r in _rows(os.path.join(args.data_dir, "accuracy_baseline.csv")):
        try:
            baselines[r["model"]] = float(r["baseline"])
        except (KeyError, TypeError, ValueError):
            pass
    results = [
        ModelResult(
            s,
            slug,
            name,
            perf_cells(perf, s, slug, day),
            acc_cell(acc, s, slug, day),
            _read(os.path.join(run_dir, slug, "error.log")),
            commands(os.path.join(run_dir, slug)),
            baseline=baselines.get(slug),
        )
        for s in scenarios
        for slug, name in MODELS[s]
    ]
    env = environment(args.data_dir, [r for r in today if r["scenario"] in scenarios])

    path = os.path.abspath(
        args.out or os.path.join(args.data_dir, os.pardir, "result", f"{tag}-result.md")
    )
    out = os.path.dirname(path)  # links in the result are relative to it
    followups = {}
    for row in _rows(os.path.join(args.data_dir, "followups.csv")):
        if row.get("test_date") != day:
            continue
        link = (row.get("link") or "").strip()
        if row.get("action") == "repro" and link and "://" not in link:
            # A repro's path is relative to the data dir, not to the result.
            link = os.path.relpath(os.path.join(args.data_dir, link), out)
            row = {**row, "link": link}
        followups[row["model"]] = row  # the same session rewritten: the last row wins
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
                followups,
                _read(os.path.join(run_dir, "findings.md")),
            )
        )
    for title, table in (
        ("Lowlights", lowlights_md(results, followups)),
        ("Highlights", highlights_md(results)),
    ):
        if table:
            print(f"{title}:", *table, sep="\n")
    pending = {
        r["model"]: (r.get("prs") or "").split()
        for r in _rows(os.path.join(args.data_dir, "pending_fixes.csv"))
    }
    slug = {r.name: r.slug for r in results}
    for name in missing_followups(results, followups):
        print(
            f"WARNING: {name} is a problem in the lowlights but has no follow-up in followups.csv: find an open PR, fix it, or write a repro (455-CI-guide.md §3.7)."
        )
        prs = pending.get(slug[name])
        if prs:
            row = io.StringIO()
            fields = {
                "test_date": day,
                "model": slug[name],
                "action": "open_pr",
                "link": " ".join(prs),
            }
            csv.writer(row, lineterminator="").writerow(
                [fields.get(c, "") for c in FOLLOWUP_COLUMNS]
            )
            print(
                f"  Known unmerged fix in pending_fixes.csv: {' + '.join(map(pr_label, prs))}. If it is still open and fixes this, add to followups.csv:\n  {row.getvalue()}"
            )
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
