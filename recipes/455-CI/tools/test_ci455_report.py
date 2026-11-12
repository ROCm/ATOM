"""Tests for the 455-CI highlights (ci455_report.py)."""

from ci455_data import AccCell, PerfCell
from ci455_report import ModelResult, highlights


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


def test_a_failure_leads_with_the_model_the_failed_point_and_the_error():
    hl = highlights(
        [
            model("Alpha", pc(4, 97.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95)),
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
        hl[0].startswith("❌")
        and "Bravo" in hl[0]
        and "c64" in hl[0]
        and "out of memory" in hl[0]
    )
    assert len(about(hl, "Bravo")) == 1


def test_a_throughput_drop_of_3_percent_is_a_regression_and_10_percent_is_severe():
    hl = highlights(
        [
            model("Alpha", pc(4, 97.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95)),
            model("Bravo", pc(4, 89.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95)),
            model(
                "Charlie", pc(4, 98.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95)
            ),
        ]
    )
    (alpha,), (bravo,) = about(hl, "Alpha"), about(hl, "Bravo")
    assert alpha.startswith("⚠️") and "c4" in alpha and "−3.0%" in alpha
    assert bravo.startswith("🔴") and "−11.0%" in bravo
    assert about(hl, "Charlie") == []  # −2%: counted in the closing "flat" line


def test_a_gain_of_3_percent_is_progress():
    hl = highlights(
        [model("Alpha", pc(4, 100.0, 100.0), pc(64, 1040.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("🟢") and "c64" in alpha and "+4.0%" in alpha


def test_a_slow_slide_below_the_best_is_flagged_though_each_step_is_small():
    hl = highlights(
        [
            model(
                "Alpha",
                pc(4, 98.0, 99.0, best=104.0),
                pc(64, 1000.0, 1000.0),
                ac(0.95, 0.95),
            )
        ]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("⚠️") and "历史最佳" in alpha and "−5.8%" in alpha


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
    assert alpha.startswith("⚠️") and "GSM8K" in alpha and "−4.5" in alpha
    assert bravo.startswith("🔴") and "−7.0" in bravo
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
    assert alpha.startswith("✅")


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
