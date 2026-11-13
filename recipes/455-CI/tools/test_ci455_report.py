"""Tests for the 455-CI report: highlights, summary table, details (ci455_report.py)."""

from ci455_data import AccCell, PerfCell
from ci455_report import (
    ModelResult,
    highlights,
    missing_followups,
    missing_owners,
    report_md,
    summary_row,
)


def pc(
    conc,
    value,
    prev,
    status="ok",
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


def test_detail_tables_compare_with_the_previous_session_by_date_and_name_an_older_one():
    a = model("Alpha", pc(4, 100.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95))
    b = model(
        "Bravo",
        pc(4, 100.0, 99.0, prev_date="2026-10-31"),
        pc(64, 1000.0, 990.0, prev_date="2026-10-31"),
        ac(0.95, 0.95, prev_date="2026-10-31"),
    )
    text = report_md(
        "2026-11-14", {}, [a, b], "../runs/20261114/charts", "../runs/20261114"
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
    text = report_md("2026-11-18", env, [model("Alpha", **FLAT)], "c", "r")
    assert "| Test date | 2026-11-18 |" in text and "sha256:abc" in text
    assert (
        "Base image" not in text and "base:tag" not in text and "Dependency" not in text
    )


def test_detail_performance_rows_show_the_parallelism_not_the_gpu_count():
    text = report_md("2026-11-14", {}, [model("Alpha", **FLAT)], "c", "r")
    assert "| Parallelism |" in text and "| GPUs |" not in text
    assert "| Alpha | tp4 | 4 |" in text and "| Alpha | tp4 | 64 |" in text


def test_one_report_holds_the_summary_and_then_the_details():
    text = report_md(
        "2026-11-14", {}, [model("Alpha", **FLAT), model("Bravo", **FAILING)], "c", "r"
    )
    order = [
        "\n## Summary\n",
        "\n### Highlights\n",
        "\n| Model | Scenario |",
        "\n## Details\n",
        "\n### Environment\n",
        "\n### Performance · MLPerf Endpoint\n",
        "\n### Failures\n",
        "\n#### Bravo (MLPerf Endpoint): c64\n",
    ]
    at = [text.index(heading) for heading in order]
    assert at == sorted(at)
    assert "-detail.md" not in text and "Detailed report" not in text


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


def test_summary_row_shows_c64_throughput_before_and_now_with_the_delta():
    row = summary_row(
        model("Alpha", pc(4, 50.0, 60.0), pc(64, 1040.0, 1000.0), ac(0.939, 0.941)),
        "2026-11-14",
        "2026-11-07",
    )
    assert row == [
        "Alpha",
        "MLPerf Endpoint",
        "1,000",
        "1,040",
        "🟢 +4.0%",
        "−0.2 pts (now 93.9%)",
    ]


def test_summary_row_dates_a_previous_value_older_than_the_column():
    row = summary_row(
        model(
            "Alpha",
            pc(4, 50.0, 60.0),
            pc(64, 340.0, 336.0, prev_date="2026-10-31"),
            ac(0.961, 0.962, prev_date="2026-10-31"),
        ),
        "2026-11-14",
        "2026-11-07",
    )
    assert row[2:] == [
        "336 (2026-10-31)",
        "340",
        "+1.2%",
        "−0.1 pts (now 96.1%) (vs 2026-10-31)",
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
        "2026-11-07",
    )
    assert row == ["Bravo", "MLPerf Endpoint", "900", "N/A", "N/A", "N/A"]


def test_summary_row_of_a_first_run_has_no_delta():
    row = summary_row(
        model("Alpha", pc(4, 50.0, None), pc(64, 1000.0, None), ac(0.95, None)),
        "2026-11-14",
        None,
    )
    assert row == [
        "Alpha",
        "MLPerf Endpoint",
        "—",
        "1,000",
        "first",
        "first (now 95.0%)",
    ]


def test_a_problem_just_tags_the_owner():
    (line,) = highlights([model("Bravo", **FAILING)], owners={"bravo": "@bravo_owner"})
    assert (
        line.startswith("❌ Bravo: ")
        and line.endswith(" @bravo_owner")
        and "please" not in line
    )


def test_only_the_previous_session_counts_not_an_older_best():
    from ci455_data import acc_cell, perf_cells

    def row(date, tput):
        return {
            "image_date": date,
            "test_date": date,
            "scenario": "endpoint",
            "model": "alpha",
            "gpus": "4",
            "concurrency": "64",
            "status": "ok",
            "tok_s_per_gpu": tput,
            "interactivity": "50",
        }

    # about 1% down each session: 5.8% below the 10-17 best, 1.0% below the last session
    rows = [
        row("2026-10-17", "1040"),
        row("2026-10-24", "1010"),
        row("2026-10-31", "1000"),
        row("2026-11-07", "990"),
        row("2026-11-14", "980"),
    ]
    result = ModelResult(
        "endpoint",
        "alpha",
        "Alpha",
        perf_cells(rows, "endpoint", "alpha", "2026-11-14", (64,)),
        acc_cell([], "endpoint", "alpha", "2026-11-14"),
    )
    assert highlights([result]) == NONE


def test_a_model_that_did_not_run_this_session_has_not_recovered():
    # e.g. its machine went down before it ran (455-CI-guide.md §3.6): no rows
    hl = highlights(
        [
            model(
                "Alpha",
                pc(4, None, 100.0, status="missing", prev_failed=True),
                pc(64, None, 1000.0, status="missing", prev_failed=True),
                ac(None, 0.95, status="missing", prev_failed=True),
            )
        ]
    )
    assert hl == NONE


def test_summary_row_of_a_model_that_did_not_run_shows_dashes_not_na():
    row = summary_row(
        model(
            "Alpha",
            pc(4, None, 100.0, status="missing"),
            pc(64, None, 1000.0, status="missing"),
            ac(None, 0.95, status="missing"),
        ),
        "2026-11-14",
        "2026-11-07",
    )
    assert row == ["Alpha", "MLPerf Endpoint", "1,000", "—", "—", "—"]


PR_OPEN = "https://github.com/ROCm/aiter/pull/1234"
PR_OURS = "https://github.com/ROCm/ATOM/pull/2601"


def follow(action, link, owner="", note=""):
    return {"action": action, "link": link, "owner": owner, "note": note}


def test_a_failure_carries_the_open_pr_that_fixes_it_before_the_owner():
    (line,) = highlights(
        [model("Bravo", **FAILING)],
        owners={"bravo": "@bravo_owner"},
        followups={"bravo": follow("open_pr", PR_OPEN)},
    )
    assert line.startswith("❌ Bravo: c64 failed")
    assert line.endswith(f"Fix in progress: [ROCm/aiter#1234]({PR_OPEN}). @bravo_owner")


def test_a_regression_we_fixed_links_our_verified_pr():
    (line,) = highlights(
        [model("Alpha", pc(4, 100.0, 100.0), pc(64, 950.0, 1000.0), ac(0.95, 0.95))],
        followups={"alpha": follow("our_pr", PR_OURS)},
    )
    assert line.startswith("⚠️ Alpha: throughput down 5.0%.")
    assert line.endswith(f"Fixed by us in [ROCm/ATOM#2601]({PR_OURS}), verified.")


def test_a_problem_with_only_a_repro_gets_the_placeholder_owner():
    (line,) = highlights(
        [model("Bravo", **FAILING)],
        owners={"bravo": "@bravo_owner"},
        followups={"bravo": follow("repro", "../runs/20261114/bravo/repro.md")},
    )
    assert line.endswith(
        "No fix yet; [repro](../runs/20261114/bravo/repro.md). @owner_tbd"
    )


def test_the_follow_up_owner_replaces_the_model_owner():
    (line,) = highlights(
        [model("Bravo", **FAILING)],
        owners={"bravo": "@bravo_owner"},
        followups={"bravo": follow("repro", "r.md", owner="@fixer_name")},
    )
    assert line.endswith(" @fixer_name") and "@bravo_owner" not in line


def test_good_news_takes_no_follow_up():
    (line,) = highlights(
        [model("Alpha", **GAINING)], followups={"alpha": follow("open_pr", PR_OPEN)}
    )
    assert "Fix" not in line and "repro" not in line


def test_missing_followups_names_problems_without_a_usable_follow_up():
    results = [
        model("Alpha", **GAINING),
        model("Bravo", **FAILING),
        model("Charlie", **FAILING),
        model("Delta", **FAILING),
        model("Echo", **FAILING),
    ]
    followups = {
        "charlie": follow("repro", "r.md"),
        "delta": follow("open_pr", ""),
        "echo": follow("asked_around", PR_OPEN),
    }
    assert missing_followups(results, followups) == ["Bravo", "Delta", "Echo"]


def test_a_placeholder_owner_counts_as_an_owner():
    results = [model("Bravo", **FAILING)]
    assert missing_owners(results, {}, {"bravo": follow("repro", "r.md")}) == []
    assert missing_owners(results, {}, {}) == ["Bravo"]


def test_the_details_list_each_problem_with_its_next_step_owner_and_note():
    text = report_md(
        "2026-11-14",
        {},
        [
            model("Alpha", **FLAT),
            model("Bravo", **FAILING),
            model("Charlie", **FAILING),
        ],
        "c",
        "r",
        followups={"bravo": follow("our_pr", PR_OURS, note="c64 905 tok/s/GPU on 455")},
    )
    details = text.split("\n## Details\n")[1]
    table = details.split("\n### Follow-ups\n")[1].split("\n### ")[0]
    assert (
        f"| Bravo | Fixed by us in [ROCm/ATOM#2601]({PR_OURS}), verified. | — | c64 905 tok/s/GPU on 455 |"
        in table
    )
    assert "| Charlie | Missing: needs a follow-up | — | — |" in table
    assert "Alpha" not in table


def test_a_fix_that_takes_two_prs_links_both():
    links = (
        "https://github.com/ROCm/ATOM/pull/3225 https://github.com/ROCm/aiter/pull/7847"
    )
    (line,) = highlights(
        [model("Bravo", **FAILING)], followups={"bravo": follow("open_pr", links)}
    )
    want = "Fix in progress: [ROCm/ATOM#3225](https://github.com/ROCm/ATOM/pull/3225) + [ROCm/aiter#7847](https://github.com/ROCm/aiter/pull/7847)."
    assert line.endswith(want)
