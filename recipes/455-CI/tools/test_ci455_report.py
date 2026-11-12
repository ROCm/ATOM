"""Tests for the 455-CI highlights and summary table (ci455_report.py)."""

from ci455_data import AccCell, PerfCell
from ci455_report import ModelResult, highlights, summary_row


def pc(conc, value, prev, status="ok", best=None, prev_failed=False, error=""):
    return PerfCell(
        concurrency=conc,
        status=status,
        value=value,
        second=None,
        prev=prev,
        prev_second=None,
        prev_date="2026-11-07" if prev is not None else None,
        best=prev if best is None else best,
        prev_failed=prev_failed,
        error=error,
        gpus=4,
    )


def ac(value, prev, status="ok", prev_failed=False, error=""):
    return AccCell(
        status=status,
        value=value,
        strict=None,
        prev=prev,
        prev_date="2026-11-07" if prev is not None else None,
        prev_failed=prev_failed,
        error=error,
    )


def model(name, c4, c64, accuracy, scenario="endpoint"):
    return ModelResult(
        scenario=scenario, slug=name.lower(), name=name, perf=[c4, c64], acc=accuracy
    )


FLAT = {
    "c4": pc(4, 100.0, 100.0),
    "c64": pc(64, 1000.0, 1000.0),
    "accuracy": ac(0.95, 0.95),
}


def about(lines, name):
    return [h for h in lines if name in h]


def test_a_highlight_leads_with_the_short_model_name_and_leaves_out_the_scenario():
    hl = highlights(
        [
            model(
                "Kimi-K3 (2.8T)",
                pc(4, 100.0, 100.0),
                pc(64, 1040.0, 1000.0),
                ac(0.95, 0.95),
                scenario="8k1k",
            )
        ]
    )
    (line,) = about(hl, "Kimi-K3")
    assert line.startswith("🟢 Kimi-K3：")
    assert "(2.8T)" not in line and "8K/1K" not in line and "MLPerf" not in line


def test_a_failure_leads_with_the_failed_point_and_the_error():
    hl = highlights(
        [
            model("Alpha", pc(4, 100.0, 100.0), pc(64, 970.0, 1000.0), ac(0.95, 0.95)),
            model(
                "Bravo",
                pc(4, 50.0, 50.0),
                pc(
                    64,
                    None,
                    900.0,
                    status="fail",
                    error="RuntimeError: HIP error: out of memory",
                ),
                ac(0.9, 0.9),
            ),
        ]
    )
    assert (
        hl[0].startswith("❌ Bravo：") and "c64" in hl[0] and "out of memory" in hl[0]
    )
    assert len(about(hl, "Bravo")) == 1


def test_a_throughput_drop_of_3_percent_is_a_regression_and_10_percent_is_severe():
    hl = highlights(
        [
            model("Alpha", pc(4, 100.0, 100.0), pc(64, 970.0, 1000.0), ac(0.95, 0.95)),
            model("Bravo", pc(4, 100.0, 100.0), pc(64, 890.0, 1000.0), ac(0.95, 0.95)),
            model(
                "Charlie", pc(4, 100.0, 100.0), pc(64, 980.0, 1000.0), ac(0.95, 0.95)
            ),
        ]
    )
    (alpha,), (bravo,) = about(hl, "Alpha"), about(hl, "Bravo")
    assert alpha.startswith("⚠️ Alpha：") and "吞吐回退 3.0%" in alpha
    assert bravo.startswith("🔴 Bravo：") and "吞吐回退 11.0%" in bravo
    assert about(hl, "Charlie") == []  # −2%: counted in the closing "flat" line


def test_a_change_only_at_low_concurrency_says_so():
    hl = highlights(
        [model("Alpha", pc(4, 95.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("⚠️ Alpha：") and "低并发" in alpha and "5.0%" in alpha


def test_a_gain_of_3_percent_is_progress():
    hl = highlights(
        [model("Alpha", pc(4, 100.0, 100.0), pc(64, 1040.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("🟢 Alpha：") and "吞吐提升 4.0%" in alpha


def test_a_slow_slide_below_the_best_is_flagged_though_each_step_is_small():
    hl = highlights(
        [
            model(
                "Alpha",
                pc(4, 100.0, 100.0),
                pc(64, 980.0, 990.0, best=1040.0),
                ac(0.95, 0.95),
            )
        ]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("⚠️ Alpha：") and "历史最佳" in alpha and "5.8%" in alpha


def test_an_accuracy_drop_over_004_is_flagged_and_over_006_is_severe():
    hl = highlights(
        [
            model("Alpha", FLAT["c4"], FLAT["c64"], ac(0.905, 0.95)),
            model("Bravo", FLAT["c4"], FLAT["c64"], ac(0.88, 0.95)),
            model(
                "Charlie", FLAT["c4"], FLAT["c64"], ac(0.91, 0.95)
            ),  # exactly −0.04: not flagged
        ]
    )
    (alpha,), (bravo,) = about(hl, "Alpha"), about(hl, "Bravo")
    assert (
        alpha.startswith("⚠️ Alpha：")
        and "GSM8K 掉了 4.5 个点" in alpha
        and "90.5%" in alpha
    )
    assert bravo.startswith("🔴 Bravo：") and "7.0 个点" in bravo
    assert about(hl, "Charlie") == []


def test_a_model_that_failed_last_time_and_runs_now_is_a_recovery():
    hl = highlights(
        [
            model(
                "Alpha",
                pc(4, 100.0, 99.0, prev_failed=True),
                pc(64, 1000.0, 990.0),
                ac(0.95, 0.95),
            )
        ]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("✅ Alpha：")


def test_models_with_no_history_are_listed_as_new():
    hl = highlights(
        [model("Alpha", pc(4, 100.0, None), pc(64, 1000.0, None), ac(0.95, None))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("🆕")


def test_unchanged_models_are_summed_up_in_one_closing_line():
    hl = highlights([model(n, **FLAT) for n in ("Alpha", "Bravo", "Charlie")])
    assert len(hl) == 1 and "3 个模型" in hl[0] and "持平" in hl[0]


def test_a_scenario_not_run_yet_is_reported_as_pending():
    hl = highlights([model("Alpha", **FLAT)], pending=["8K/1K"])
    assert any("8K/1K" in h and "待测" in h for h in hl)


def test_summary_row_shows_the_c64_throughput_before_and_now_and_the_accuracy_change():
    row = summary_row(
        model("Alpha", pc(4, 50.0, 60.0), pc(64, 1040.0, 1000.0), ac(0.939, 0.941))
    )
    assert row == [
        "Alpha",
        "MLPerf Endpoint",
        "1,000",
        "1,040",
        "🟢 +4.0%",
        "−0.2 个点（当前 93.9%）",
    ]


def test_summary_row_of_a_failed_run_shows_na():
    row = summary_row(
        model(
            "Bravo",
            pc(4, 50.0, 50.0),
            pc(64, None, 900.0, status="fail", error="x"),
            ac(None, 0.9, status="fail", error="x"),
        )
    )
    assert row == ["Bravo", "MLPerf Endpoint", "900", "N/A", "N/A", "N/A"]


def test_summary_row_of_a_first_run_has_no_change():
    row = summary_row(
        model("Alpha", pc(4, 50.0, None), pc(64, 1000.0, None), ac(0.95, None))
    )
    assert row == [
        "Alpha",
        "MLPerf Endpoint",
        "—",
        "1,000",
        "首次",
        "首次（当前 95.0%）",
    ]


def test_the_low_concurrency_is_mentioned_when_it_is_worse_than_c64():
    hl = highlights(
        [model("Alpha", pc(4, 88.0, 100.0), pc(64, 960.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert (
        alpha.startswith("🔴 Alpha：")
        and "吞吐回退 4.0%" in alpha
        and "低并发" in alpha
        and "12.0%" in alpha
    )


def test_the_low_concurrency_is_left_out_when_c64_already_says_it():
    hl = highlights(
        [model("Alpha", pc(4, 96.0, 100.0), pc(64, 950.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert "吞吐回退 5.0%" in alpha and "低并发" not in alpha
