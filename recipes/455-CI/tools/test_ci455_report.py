"""Tests for the 455-CI highlights and summary table (ci455_report.py)."""

from ci455_data import AccCell, PerfCell
from ci455_report import ModelResult, detail_md, highlights, missing_owners, summary_row


def pc(
    conc,
    value,
    prev,
    status="ok",
    best=None,
    prev_failed=False,
    error="",
    prev_date="2026-11-07",
):
    return PerfCell(
        concurrency=conc,
        status=status,
        value=value,
        second=None,
        prev=prev,
        prev_second=None,
        prev_date=prev_date if prev is not None else None,
        best=prev if best is None else best,
        prev_failed=prev_failed,
        error=error,
        gpus=4,
        parallel="tp4",
    )


def ac(value, prev, status="ok", prev_failed=False, error="", prev_date="2026-11-07"):
    return AccCell(
        status=status,
        value=value,
        strict=None,
        prev=prev,
        prev_date=prev_date if prev is not None else None,
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


NONE = ["No failures or major changes."]


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
    assert line.startswith("🟢 Kimi-K3: ")
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
        hl[0].startswith("❌ Bravo: ")
        and "c64 failed" in hl[0]
        and "out of memory" in hl[0]
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
    assert alpha.startswith("⚠️ Alpha: ") and "throughput down 3.0%" in alpha
    assert bravo.startswith("🔴 Bravo: ") and "throughput down 11.0%" in bravo
    assert about(hl, "Charlie") == []  # −2%: not news


def test_a_change_only_at_low_concurrency_says_so():
    hl = highlights(
        [model("Alpha", pc(4, 95.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("⚠️ Alpha: ") and "c4 throughput down 5.0%" in alpha


def test_the_low_concurrency_is_mentioned_when_it_is_worse_than_c64():
    hl = highlights(
        [model("Alpha", pc(4, 88.0, 100.0), pc(64, 960.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert (
        alpha.startswith("🔴 Alpha: ")
        and "throughput down 4.0%" in alpha
        and "c4 down 12.0%" in alpha
    )


def test_the_low_concurrency_is_left_out_when_c64_already_says_it():
    hl = highlights(
        [model("Alpha", pc(4, 96.0, 100.0), pc(64, 950.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert "throughput down 5.0%" in alpha and "c4" not in alpha


def test_a_gain_of_3_percent_is_progress():
    hl = highlights(
        [model("Alpha", pc(4, 100.0, 100.0), pc(64, 1040.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha.startswith("🟢 Alpha: ") and "throughput up 4.0%" in alpha


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
    assert alpha.startswith("⚠️ Alpha: ") and "5.8% below its best" in alpha


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
    assert alpha.startswith("⚠️ Alpha: ") and "GSM8K down 4.5 points to 90.5%" in alpha
    assert bravo.startswith("🔴 Bravo: ") and "down 7.0 points" in bravo
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
    assert alpha.startswith("✅ Alpha: ")


def test_flat_models_are_not_mentioned():
    assert highlights([model(n, **FLAT) for n in ("Alpha", "Bravo", "Charlie")]) == NONE


def test_a_first_result_is_not_news():
    assert (
        highlights(
            [model("Alpha", pc(4, 100.0, None), pc(64, 1000.0, None), ac(0.95, None))]
        )
        == NONE
    )


def test_summary_row_shows_dated_c64_throughput_before_and_now_and_the_accuracy_change():
    row = summary_row(
        model("Alpha", pc(4, 50.0, 60.0), pc(64, 1040.0, 1000.0), ac(0.939, 0.941)),
        "2026-11-14",
    )
    assert row == [
        "Alpha",
        "MLPerf Endpoint",
        "1,000 (2026-11-07)",
        "1,040 (2026-11-14)",
        "🟢 +4.0%",
        "−0.2 pts (now 93.9%)",
    ]


def test_summary_row_of_a_failed_run_shows_na():
    row = summary_row(
        model(
            "Bravo",
            pc(4, 50.0, 50.0),
            pc(64, None, 900.0, status="fail", error="x"),
            ac(None, 0.9, status="fail", error="x"),
        ),
        "2026-11-14",
    )
    assert row == [
        "Bravo",
        "MLPerf Endpoint",
        "900 (2026-11-07)",
        "N/A (2026-11-14)",
        "N/A",
        "N/A",
    ]


def test_summary_row_of_a_first_run_has_no_change():
    row = summary_row(
        model("Alpha", pc(4, 50.0, None), pc(64, 1000.0, None), ac(0.95, None)),
        "2026-11-14",
    )
    assert row == [
        "Alpha",
        "MLPerf Endpoint",
        "—",
        "1,000 (2026-11-14)",
        "first",
        "first (now 95.0%)",
    ]


def test_detail_tables_compare_with_the_previous_session_by_date_and_name_an_older_one():
    a = model("Alpha", pc(4, 100.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95))
    b = model(
        "Bravo",
        pc(4, 100.0, 99.0, prev_date="2026-10-31"),
        pc(64, 1000.0, 990.0, prev_date="2026-10-31"),
        ac(0.95, 0.95, prev_date="2026-10-31"),
    )
    text = detail_md(
        "2026-11-14", {}, [a, b], "r.md", "../runs/20261114/charts", "../runs/20261114"
    )
    assert "| vs 2026-11-07 |" in text and "(vs 2026-10-31)" in text
    assert "previous" not in text.lower() and "vs best" not in text.lower()


def test_detail_environment_lists_the_image_the_commits_and_one_test_date():
    env = {
        "image": "img:tag",
        "digest": "sha256:abc",
        "base_image": "base:tag",
        "dependency_changes": ["x 1 -> 2"],
        "atom": {"commit": "a" * 40},
        "aiter": {"commit": "b" * 40},
    }
    text = detail_md("2026-11-18", env, [model("Alpha", **FLAT)], "r.md", "c", "r")
    assert "| Test date | 2026-11-18 |" in text and "sha256:abc" in text
    assert (
        "Base image" not in text and "base:tag" not in text and "Dependency" not in text
    )


def test_detail_performance_rows_show_the_parallelism_not_the_gpu_count():
    text = detail_md("2026-11-14", {}, [model("Alpha", **FLAT)], "r.md", "c", "r")
    assert "| Parallelism |" in text and "| GPUs |" not in text
    assert "| Alpha | tp4 | 4 |" in text and "| Alpha | tp4 | 64 |" in text


FAILING = {
    "c4": pc(4, 50.0, 50.0),
    "c64": pc(64, None, 900.0, status="fail", error="RuntimeError: x"),
    "accuracy": ac(0.9, 0.9),
}
GAINING = {
    "c4": pc(4, 100.0, 100.0),
    "c64": pc(64, 1040.0, 1000.0),
    "accuracy": ac(0.95, 0.95),
}


def test_a_problem_tags_the_owner_and_asks_them_to_look():
    (line,) = highlights([model("Bravo", **FAILING)], owners={"bravo": "@bravo_owner"})
    assert line.startswith("❌ Bravo: ") and line.endswith(
        "@bravo_owner please take a look."
    )


def test_good_news_tags_the_owner_without_asking():
    (line,) = highlights(
        [model("Alpha", **GAINING)], owners={"alpha": "@alpha_owner @alpha_backup"}
    )
    assert line.endswith("@alpha_owner @alpha_backup") and "please" not in line


def test_a_model_without_an_owner_gets_no_mention():
    (line,) = highlights([model("Alpha", **GAINING)], owners={"bravo": "@bravo_owner"})
    assert "@" not in line


def test_missing_owners_names_the_models_with_news_and_no_owner():
    results = [
        model("Alpha", **GAINING),
        model("Bravo", **FAILING),
        model("Charlie", **FLAT),
    ]
    assert missing_owners(results, {"alpha": "@alpha_owner"}) == ["Bravo"]
