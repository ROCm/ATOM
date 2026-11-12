#!/usr/bin/env python3
"""Draw the 455-CI charts of one session from perf.csv and accuracy.csv.

    python3 make_charts.py --date 2026-11-18 [--scenario endpoint|8k1k]
        [--data-dir recipes/455-CI] [--out recipes/455-CI/runs/20261118/charts] [--mock]

A session is one day's run (Saturday: MLPerf Endpoint, Wednesday: 8K/1K).
Writes, for the scenario run that day, perf-<scenario>.png (tok/s per GPU by
session) and accuracy-<scenario>.png (GSM8K by session): the charts the
session's detailed report links to. An optional accuracy_baseline.csv in the
data dir (columns model, baseline) adds the dashboard's baseline line. --mock
marks every chart as an example.
"""

import argparse
import os

from ci455_charts import render_accuracy, render_perf_trends
from ci455_data import (
    CONCURRENCIES,
    LABEL,
    MODELS,
    accuracy_history,
    perf_trend,
    read_csv,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _rows(path):
    return read_csv(path) if os.path.exists(path) else []


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
    ap.add_argument("--out", help="default: <data-dir>/runs/<YYYYMMDD>/charts")
    ap.add_argument("--mock", action="store_true", help="mark the charts as an example")
    args = ap.parse_args(argv)

    day = args.date
    out = args.out or os.path.join(
        args.data_dir, "runs", day.replace("-", ""), "charts"
    )
    perf = _rows(os.path.join(args.data_dir, "perf.csv"))
    acc = _rows(os.path.join(args.data_dir, "accuracy.csv"))
    baselines = {
        r["model"]: float(r["baseline"])
        for r in _rows(os.path.join(args.data_dir, "accuracy_baseline.csv"))
    }
    if args.scenario:
        scenarios = [args.scenario]
    else:
        scenarios = [
            s
            for s in MODELS
            if any(r["scenario"] == s and r["test_date"] == day for r in perf + acc)
        ]
    if not scenarios:
        print(f"no session on {day}: nothing to draw")

    for scenario in scenarios:
        panels = [
            (f"{name} · c{conc}", perf_trend(perf, scenario, slug, conc, day))
            for slug, name in MODELS[scenario]
            for conc in CONCURRENCIES
        ]
        if any(trend for _, trend in panels):
            os.makedirs(out, exist_ok=True)
            render_perf_trends(
                panels,
                os.path.join(out, f"perf-{scenario}.png"),
                LABEL[scenario],
                args.mock,
            )
        else:
            print(f"skip perf-{scenario}.png: no {scenario} results up to {day}")
        histories = [
            (name, accuracy_history(acc, scenario, slug, day))
            for slug, name in MODELS[scenario]
        ]
        if any(hist for _, hist in histories):
            os.makedirs(out, exist_ok=True)
            render_accuracy(
                histories,
                os.path.join(out, f"accuracy-{scenario}.png"),
                LABEL[scenario],
                baselines={
                    name: baselines[slug]
                    for slug, name in MODELS[scenario]
                    if slug in baselines
                },
                mock=args.mock,
            )
        else:
            print(f"skip accuracy-{scenario}.png: no {scenario} accuracy up to {day}")


if __name__ == "__main__":
    main()
