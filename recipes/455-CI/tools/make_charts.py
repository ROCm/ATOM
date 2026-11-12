#!/usr/bin/env python3
"""Draw the 455-CI charts of one image from perf.csv and accuracy.csv.

    python3 make_charts.py --image-date 2026-11-14 [--scenario endpoint|8k1k|all]
        [--data-dir recipes/455-CI] [--out recipes/455-CI/runs/20261114/charts] [--mock]

Writes, for every model of the scenario with a successful point up to the
image, perf-<slug>.png, and per scenario accuracy-<scenario>.png: the files
the image's report links to. --image-date is the image's (Saturday) date, the
same for its weekend and midweek runs. An optional accuracy_baseline.csv in
the data dir (columns model, baseline) adds the dashboard's baseline line.
--mock marks every chart as mock data, for layout previews.
"""

import argparse
import os

from ci455_charts import failure_note, render_8k1k, render_accuracy, render_endpoint
from ci455_data import accuracy_history, failed_concurrencies, perf_curves, read_csv

HERE = os.path.dirname(os.path.abspath(__file__))

# The test matrix of 455-CI-guide.md §1: (slug, chart name) per scenario.
MODELS = {
    "endpoint": [
        ("deepseek-r1", "DeepSeek R1"),
        ("deepseek-v4.1-flash", "DeepSeek-V4.1-Flash"),
        ("kimi-k3", "Kimi-K3 (2.8T)"),
        ("qwen-35b-a3b", "Qwen-35B-A3B"),
    ],
    "8k1k": [
        ("deepseek-v4-pro", "DeepSeek-V4-Pro"),
        ("glm-5.2", "GLM-5.2"),
        ("minimax-m3", "MiniMax-M3 (428B)"),
        ("qwen-3.8-flash", "Qwen-3.8 Flash"),
    ],
}
SCENARIO_LABEL = {"endpoint": "MLPerf Endpoint", "8k1k": "8K/1K"}


def _rows(path):
    return read_csv(path) if os.path.exists(path) else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image-date", required=True, help="image date, YYYY-MM-DD")
    ap.add_argument("--scenario", choices=["endpoint", "8k1k", "all"], default="all")
    ap.add_argument(
        "--data-dir",
        default=os.path.dirname(HERE),
        help="dir of perf.csv / accuracy.csv",
    )
    ap.add_argument("--out", help="default: <data-dir>/runs/<YYYYMMDD>/charts")
    ap.add_argument("--mock", action="store_true", help="mark the charts as mock data")
    args = ap.parse_args(argv)

    day = args.image_date
    out = args.out or os.path.join(
        args.data_dir, "runs", day.replace("-", ""), "charts"
    )
    perf = _rows(os.path.join(args.data_dir, "perf.csv"))
    acc = _rows(os.path.join(args.data_dir, "accuracy.csv"))
    baselines = {
        r["model"]: float(r["baseline"])
        for r in _rows(os.path.join(args.data_dir, "accuracy_baseline.csv"))
    }
    scenarios = ["endpoint", "8k1k"] if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        histories = []
        for slug, name in MODELS[scenario]:
            curves = perf_curves(perf, scenario, slug, day)
            if curves:
                ran = {
                    int(r["concurrency"])
                    for r in perf
                    if (r["scenario"], r["model"], r["image_date"])
                    == (scenario, slug, day)
                }
                note = failure_note(
                    day, failed_concurrencies(perf, scenario, slug, day), ran
                )
                render = render_endpoint if scenario == "endpoint" else render_8k1k
                os.makedirs(out, exist_ok=True)
                render(
                    curves,
                    os.path.join(out, f"perf-{slug}.png"),
                    name,
                    note=note,
                    mock=args.mock,
                )
            else:
                print(
                    f"skip perf-{slug}.png: no successful {scenario} point up to {day}"
                )
            histories.append((name, accuracy_history(acc, scenario, slug, day), slug))
        if any(hist for _, hist, _ in histories):
            os.makedirs(out, exist_ok=True)
            render_accuracy(
                [(name, hist) for name, hist, _ in histories],
                os.path.join(out, f"accuracy-{scenario}.png"),
                SCENARIO_LABEL[scenario],
                baselines={
                    name: baselines[slug]
                    for name, _, slug in histories
                    if slug in baselines
                },
                mock=args.mock,
            )
        else:
            print(f"skip accuracy-{scenario}.png: no {scenario} accuracy up to {day}")


if __name__ == "__main__":
    main()
