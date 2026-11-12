"""End-to-end test of make_charts.py: the files the report links to."""

import csv

from ci455_data import ACCURACY_COLUMNS, PERF_COLUMNS
from make_charts import main


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)


def perf(image_date, conc, model="kimi-k3", status="ok"):
    return {
        "image_date": image_date,
        "scenario": "endpoint",
        "model": model,
        "gpus": "8",
        "concurrency": str(conc),
        "status": status,
        "tok_s_per_gpu": "100" if status == "ok" else "",
        "interactivity": "50" if conc == 64 else "150",
    }


def test_writes_one_perf_chart_per_model_with_data_and_one_accuracy_chart_per_scenario(
    tmp_path,
):
    write_csv(
        tmp_path / "perf.csv",
        PERF_COLUMNS,
        [
            perf("2026-11-07", 4),
            perf("2026-11-07", 64),
            perf("2026-11-14", 4),
            perf("2026-11-14", 64, status="fail"),
            perf(
                "2026-11-14", 4, model="deepseek-r1", status="fail"
            ),  # no successful point: no chart
            perf(
                "2026-11-21", 4, model="qwen-35b-a3b"
            ),  # after the report image: ignored
        ],
    )
    write_csv(
        tmp_path / "accuracy.csv",
        ACCURACY_COLUMNS,
        [
            {
                "image_date": "2026-11-14",
                "scenario": "endpoint",
                "model": "kimi-k3",
                "status": "ok",
                "gsm8k": "0.95",
                "gsm8k_strict": "0.94",
            },
        ],
    )
    main(["--image-date", "2026-11-14", "--data-dir", str(tmp_path)])
    out = tmp_path / "runs" / "20261114" / "charts"
    assert sorted(p.name for p in out.iterdir()) == [
        "accuracy-endpoint.png",
        "perf-kimi-k3.png",
    ]


def test_writes_nothing_for_a_scenario_without_data(tmp_path):
    write_csv(tmp_path / "perf.csv", PERF_COLUMNS, [perf("2026-11-14", 4)])
    main(
        [
            "--image-date",
            "2026-11-14",
            "--scenario",
            "8k1k",
            "--data-dir",
            str(tmp_path),
        ]
    )
    out = tmp_path / "runs" / "20261114" / "charts"
    assert not out.exists() or list(out.iterdir()) == []
