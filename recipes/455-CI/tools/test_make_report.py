"""End-to-end tests of make_report.py: one report per session."""

import csv
import json
import re

from ci455_data import (
    ACCURACY_COLUMNS,
    FOLLOWUP_COLUMNS,
    PENDING_COLUMNS,
    PERF_COLUMNS,
)
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
        "parallel": "tp8dcp8",
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


def test_a_saturday_report_is_one_file_with_the_summary_then_the_details(
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

    (tmp_path / "owners.csv").write_text("model,owner\nkimi-k3,@k3_owner\n")

    main(["--date", "2026-11-14", "--data-dir", str(tmp_path)])

    reports = tmp_path / "reports"
    assert [p.name for p in reports.iterdir()] == ["455-CI-report-20261114.md"]
    report = (reports / "455-CI-report-20261114.md").read_text()
    summary, details = report.split("\n## Details\n")
    assert summary.startswith("# 455-CI report · 2026-11-14 · MLPerf Endpoint")
    assert "## Summary" in summary and "### Highlights" in summary and "❌" in summary
    assert "| Kimi-K3 (2.8T) | MLPerf Endpoint | 1,000 | N/A | N/A |" in summary
    assert "base" not in summary.split("### Highlights")[0].lower()
    assert "| Test date | 2026-11-14 |" in details and "sha256:abc" in details
    assert "Base image" not in details and "Dependency" not in details
    assert (
        "Traceback" in details
        and "../runs/20261114/charts/perf-endpoint.png" in details
    )
    assert "| Kimi-K3 (2.8T) | tp8dcp8 | 4 |" in details
    assert "@k3_owner" in summary and "please" not in summary
    assert not re.search(CJK, report)


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

    reports = tmp_path / "reports"
    assert [p.name for p in reports.iterdir()] == ["455-CI-report-20261118.md"]
    report = (reports / "455-CI-report-20261118.md").read_text()
    assert report.startswith("# 455-CI report · 2026-11-18 · 8K/1K")
    assert "| GLM-5.2 | 8K/1K | 12,000 | 12,300 | +2.5% |" in report
    assert "MLPerf Endpoint" not in report
    assert (
        "| Model | Scenario | 2026-11-11 (tok/s/GPU) | 2026-11-18 (tok/s/GPU) | Delta | GSM8K delta |"
        in report
    )
    assert "sha256:saturday" in report and "| Test date | 2026-11-18 |" in report
    assert "../runs/20261118/charts/perf-8k1k.png" in report


def test_a_mock_report_says_example_at_the_top(tmp_path):
    write_csv(tmp_path / "perf.csv", PERF_COLUMNS, [perf("2026-11-14", 4, "100")])
    main(["--date", "2026-11-14", "--data-dir", str(tmp_path), "--mock"])
    report = tmp_path / "reports" / "455-CI-report-20261114.md"
    assert any("EXAMPLE" in line for line in report.read_text().splitlines()[:4])


def test_the_console_ends_with_the_report_path(tmp_path, capsys):
    write_csv(tmp_path / "perf.csv", PERF_COLUMNS, [perf("2026-11-14", 64, "1000")])
    main(["--date", "2026-11-14", "--data-dir", str(tmp_path)])
    last = capsys.readouterr().out.rstrip().splitlines()[-1]
    assert last == f"Report: {tmp_path / 'reports' / '455-CI-report-20261114.md'}"


def test_news_about_a_model_without_an_owner_is_flagged_on_the_console(
    tmp_path, capsys
):
    write_csv(
        tmp_path / "perf.csv",
        PERF_COLUMNS,
        [
            perf("2026-11-07", 64, "1000"),
            perf("2026-11-14", 64, "", status="fail", error="RuntimeError: x"),
        ],
    )
    main(["--date", "2026-11-14", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "no owner" in out and "Kimi-K3" in out


def test_a_model_left_untested_by_a_machine_failure_is_not_news(tmp_path):
    # Kimi-K3 failed on 11-07; on 11-14 its node went down before it ran, so it
    # has no rows (455-CI-guide.md §3.6): no failure, no recovery, only dashes.
    write_csv(
        tmp_path / "perf.csv",
        PERF_COLUMNS,
        [
            perf("2026-10-31", 64, "1000"),
            perf("2026-11-07", 64, "", status="fail", error="RuntimeError: x"),
            perf("2026-11-14", 64, "1500", model="deepseek-r1"),
        ],
    )
    main(["--date", "2026-11-14", "--data-dir", str(tmp_path)])
    report = (tmp_path / "reports" / "455-CI-report-20261114.md").read_text()
    assert "Kimi-K3" not in report.split("### Results")[0]
    assert "| Kimi-K3 (2.8T) | MLPerf Endpoint | 1,000 | — | — | — |" in report


def test_follow_ups_reach_the_highlights_and_a_problem_without_one_is_flagged(
    tmp_path, capsys
):
    r1 = {"model": "deepseek-r1"}
    write_csv(
        tmp_path / "perf.csv",
        PERF_COLUMNS,
        [
            perf("2026-11-07", 64, "1000"),
            perf("2026-11-14", 64, "", status="fail", error="RuntimeError: x"),
            perf("2026-11-07", 64, "1000", **r1),
            perf("2026-11-14", 64, "800", **r1),
        ],
    )
    write_csv(
        tmp_path / "followups.csv",
        FOLLOWUP_COLUMNS,
        [
            {
                "test_date": "2026-11-14",
                "model": "kimi-k3",
                "action": "repro",
                "link": "runs/20261114/kimi-k3/repro.md",
            }
        ],
    )
    main(["--date", "2026-11-14", "--data-dir", str(tmp_path)])
    report = (tmp_path / "reports" / "455-CI-report-20261114.md").read_text()
    assert (
        "No fix yet; [repro](../runs/20261114/kimi-k3/repro.md). @owner_tbd" in report
    )
    out = capsys.readouterr().out
    assert "DeepSeek R1" in out and "no follow-up" in out
    assert out.rstrip().splitlines()[-1].startswith("Report: ")


QWEN38_PRS = (
    "https://github.com/ROCm/ATOM/pull/3225 https://github.com/ROCm/aiter/pull/7847"
)


def qwen38_failure(tmp_path):
    q = {"scenario": "8k1k", "model": "qwen-3.8-flash"}
    write_csv(
        tmp_path / "perf.csv",
        PERF_COLUMNS,
        [
            perf("2026-11-11", 64, "26000", tpot="16", **q),
            perf("2026-11-18", 64, "", status="fail", error="RuntimeError: x", **q),
        ],
    )
    write_csv(
        tmp_path / "pending_fixes.csv",
        PENDING_COLUMNS,
        [
            {
                "model": "qwen-3.8-flash",
                "prs": QWEN38_PRS,
                "note": "gfx1250 fixes",
                "since": "2026-11-13",
            }
        ],
    )


def test_a_failure_with_a_known_unmerged_fix_gets_a_ready_follow_up_row(
    tmp_path, capsys
):
    qwen38_failure(tmp_path)
    main(["--date", "2026-11-18", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "ROCm/ATOM#3225 + ROCm/aiter#7847" in out
    assert f"2026-11-18,qwen-3.8-flash,open_pr,{QWEN38_PRS},," in out


def test_the_known_fix_is_not_suggested_once_a_follow_up_is_recorded(tmp_path, capsys):
    qwen38_failure(tmp_path)
    write_csv(
        tmp_path / "followups.csv",
        FOLLOWUP_COLUMNS,
        [
            {
                "test_date": "2026-11-18",
                "model": "qwen-3.8-flash",
                "action": "open_pr",
                "link": QWEN38_PRS,
            }
        ],
    )
    main(["--date", "2026-11-18", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "pending_fixes.csv" not in out and "no follow-up" not in out
    report = (tmp_path / "reports" / "455-CI-report-20261118.md").read_text()
    assert "+ [ROCm/aiter#7847](https://github.com/ROCm/aiter/pull/7847)." in report
