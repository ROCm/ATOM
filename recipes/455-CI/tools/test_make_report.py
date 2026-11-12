"""End-to-end tests of make_report.py: one report per session."""

import csv
import json
import re

from ci455_data import ACCURACY_COLUMNS, PERF_COLUMNS
from make_report import main

CJK = "[\u4e00-\u9fff]"


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)


def perf(
    date,
    conc,
    tput,
    status="ok",
    error="",
    scenario="endpoint",
    model="kimi-k3",
    image_date=None,
    tpot="",
):
    return {
        "image_date": image_date or date,
        "test_date": date,
        "scenario": scenario,
        "model": model,
        "gpus": "8",
        "concurrency": str(conc),
        "status": status,
        "tok_s_per_gpu": tput,
        "interactivity": "50" if tput and scenario == "endpoint" else "",
        "tpot_mean_ms": tpot,
        "error": error,
    }


def acc(date, score, scenario="endpoint", model="kimi-k3", image_date=None):
    return {
        "image_date": image_date or date,
        "test_date": date,
        "scenario": scenario,
        "model": model,
        "status": "ok",
        "gsm8k": score,
        "gsm8k_strict": score,
    }


def image_json(run, digest):
    run.mkdir(parents=True, exist_ok=True)
    (run / "image.json").write_text(
        json.dumps(
            {
                "image": "rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114",
                "digest": digest,
                "base_image": "rocm/fw-bringup-mlperf:gfx1250-atom-20261113",
                "atom": {"commit": "a" * 40, "date": "2026-11-14", "title": "t"},
                "aiter": {"commit": "b" * 40, "date": "2026-11-14", "title": "u"},
                "dependency_changes": ["flydsl 0.4.1 -> 0.5.0"],
            }
        )
    )


def test_a_saturday_report_has_highlights_a_dated_table_and_the_core_error_log(
    tmp_path,
):
    write_csv(
        tmp_path / "perf.csv",
        PERF_COLUMNS,
        [
            perf("2026-11-07", 4, "100"),
            perf("2026-11-07", 64, "1000"),
            perf("2026-11-14", 4, "104"),
            perf(
                "2026-11-14",
                64,
                "",
                status="fail",
                error="RuntimeError: HIP error: out of memory",
            ),
        ],
    )
    write_csv(
        tmp_path / "accuracy.csv",
        ACCURACY_COLUMNS,
        [acc("2026-11-07", "0.95"), acc("2026-11-14", "0.95")],
    )
    run = tmp_path / "runs" / "20261114"
    image_json(run, "sha256:abc")
    (run / "kimi-k3").mkdir()
    (run / "kimi-k3" / "error.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: HIP error: out of memory\n"
    )

    main(["--date", "2026-11-14", "--data-dir", str(tmp_path)])

    summary = (tmp_path / "reports" / "455-CI-report-20261114.md").read_text()
    detail = (tmp_path / "reports" / "455-CI-report-20261114-detail.md").read_text()
    assert summary.startswith("# 455-CI report · 2026-11-14 · MLPerf Endpoint")
    assert (
        "## Highlights" in summary
        and "❌" in summary
        and "455-CI-report-20261114-detail.md" in summary
    )
    assert (
        "| Kimi-K3 (2.8T) | MLPerf Endpoint | 1,000 (2026-11-07) | N/A (2026-11-14) | N/A |"
        in summary
    )
    assert "base" not in summary.split("## Highlights")[0].lower()
    assert "| Test date | 2026-11-14 |" in detail and "sha256:abc" in detail
    assert "Base image" not in detail and "Dependency" not in detail
    assert (
        "Traceback" in detail and "../runs/20261114/charts/perf-endpoint.png" in detail
    )
    assert not re.search(CJK, summary + detail)


def test_a_wednesday_report_covers_its_own_session_on_the_saturday_image(tmp_path):
    g = {"scenario": "8k1k", "model": "glm-5.2"}
    write_csv(
        tmp_path / "perf.csv",
        PERF_COLUMNS,
        [
            perf(
                "2026-11-14", 64, "1000"
            ),  # Saturday's endpoint session: not in this report
            perf("2026-11-11", 64, "12000", tpot="30", image_date="2026-11-07", **g),
            perf("2026-11-18", 64, "12300", tpot="30", image_date="2026-11-14", **g),
        ],
    )
    image_json(tmp_path / "runs" / "20261114", "sha256:saturday")

    main(["--date", "2026-11-18", "--data-dir", str(tmp_path)])

    summary = (tmp_path / "reports" / "455-CI-report-20261118.md").read_text()
    detail = (tmp_path / "reports" / "455-CI-report-20261118-detail.md").read_text()
    assert summary.startswith("# 455-CI report · 2026-11-18 · 8K/1K")
    assert "| GLM-5.2 | 8K/1K | 12,000 (2026-11-11) | 12,300 (2026-11-18) |" in summary
    assert "MLPerf Endpoint" not in summary
    assert "sha256:saturday" in detail and "| Test date | 2026-11-18 |" in detail
    assert "../runs/20261118/charts/perf-8k1k.png" in detail


def test_a_mock_report_says_example_at_the_top(tmp_path):
    write_csv(tmp_path / "perf.csv", PERF_COLUMNS, [perf("2026-11-14", 4, "100")])
    main(["--date", "2026-11-14", "--data-dir", str(tmp_path), "--mock"])
    for name in ("455-CI-report-20261114.md", "455-CI-report-20261114-detail.md"):
        head = (tmp_path / "reports" / name).read_text().splitlines()[:4]
        assert any("EXAMPLE" in line for line in head)
