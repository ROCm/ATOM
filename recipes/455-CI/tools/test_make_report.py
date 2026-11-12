"""End-to-end test of make_report.py: the summary and the detailed report."""

import csv
import json

from ci455_data import ACCURACY_COLUMNS, PERF_COLUMNS
from make_report import main


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)


def perf(image_date, conc, tput, status="ok", error=""):
    return {
        "image_date": image_date,
        "test_date": image_date,
        "scenario": "endpoint",
        "model": "kimi-k3",
        "gpus": "8",
        "concurrency": str(conc),
        "status": status,
        "tok_s_per_gpu": tput,
        "interactivity": "50" if tput else "",
        "error": error,
    }


def acc(image_date, score):
    return {
        "image_date": image_date,
        "test_date": image_date,
        "scenario": "endpoint",
        "model": "kimi-k3",
        "status": "ok",
        "gsm8k": score,
        "gsm8k_strict": score,
    }


def test_writes_a_summary_with_highlights_and_a_detail_with_the_core_error_log(
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
    (run / "kimi-k3").mkdir(parents=True)
    (run / "kimi-k3" / "error.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: HIP error: out of memory\n"
    )
    (run / "image.json").write_text(
        json.dumps(
            {
                "image": "rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114",
                "digest": "sha256:abc",
                "base_image": "rocm/fw-bringup-mlperf:gfx1250-atom-20261113",
                "atom": {"commit": "a" * 40, "date": "2026-11-14", "title": "t"},
                "aiter": {"commit": "b" * 40, "date": "2026-11-14", "title": "u"},
                "dependency_changes": [],
            }
        )
    )

    main(["--image-date", "2026-11-14", "--data-dir", str(tmp_path)])

    summary = (tmp_path / "reports" / "455-CI-report-20261114.md").read_text()
    detail = (tmp_path / "reports" / "455-CI-report-20261114-detail.md").read_text()
    assert "Highlights" in summary and "❌" in summary
    assert "455-CI-report-20261114-detail.md" in summary
    assert "| 模型 | 场景 | 上次性能" in summary
    assert (
        "| Kimi-K3 (2.8T) | MLPerf Endpoint | 1,000 | N/A | N/A |" in summary
    )  # c64 failed
    assert "Traceback" in detail and "out of memory" in detail  # the core error log
    assert "../runs/20261114/charts/perf-endpoint.png" in detail
    assert "sha256:abc" in detail


def test_a_mock_report_says_so_at_the_top(tmp_path):
    write_csv(tmp_path / "perf.csv", PERF_COLUMNS, [perf("2026-11-14", 4, "100")])
    main(["--image-date", "2026-11-14", "--data-dir", str(tmp_path), "--mock"])
    for name in ("455-CI-report-20261114.md", "455-CI-report-20261114-detail.md"):
        head = (tmp_path / "reports" / name).read_text().splitlines()[:4]
        assert any("EXAMPLE" in line for line in head)
