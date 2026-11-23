"""Tests for the 455-CI report: lowlights, summary table, details (ci455_report.py)."""

from ci455_data import AccCell, PerfCell
from ci455_report import (
    ModelResult,
    highlights,
    lowlights,
    lowlights_md,
    missing_followups,
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


def about(rows, name):
    """The lowlight rows of one model: [flag and model, what happened, cause, fix, owner]."""
    return [row for row in rows if name in row[0]]


def test_a_lowlight_leads_with_the_short_model_name_and_leaves_out_the_scenario():
    hl = lowlights(
        [
            model(
                "Kimi-K3 (2.8T)",
                pc(4, 100.0, 100.0),
                pc(64, 960.0, 1000.0),
                ac(0.95, 0.95),
                scenario="8k1k",
            )
        ]
    )
    (row,) = about(hl, "Kimi-K3")
    assert row[0] == "⚠️ Kimi-K3"
    assert not any(
        word in cell for cell in row for word in ("(2.8T)", "8K/1K", "MLPerf")
    )


def test_a_failure_leads_with_the_failed_point_and_the_error():
    hl = lowlights(
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
    assert hl[0][:2] == ["❌ Bravo", "c64 failed"] and "out of memory" in hl[0][2]
    assert len(about(hl, "Bravo")) == 1


def test_a_throughput_drop_of_3_percent_is_a_regression_and_10_percent_is_severe():
    hl = lowlights(
        [
            model("Alpha", pc(4, 100.0, 100.0), pc(64, 970.0, 1000.0), ac(0.95, 0.95)),
            model("Bravo", pc(4, 100.0, 100.0), pc(64, 890.0, 1000.0), ac(0.95, 0.95)),
            model(
                "Charlie", pc(4, 100.0, 100.0), pc(64, 980.0, 1000.0), ac(0.95, 0.95)
            ),
        ]
    )
    (alpha,), (bravo,) = about(hl, "Alpha"), about(hl, "Bravo")
    assert alpha[:2] == ["⚠️ Alpha", "throughput down 3.0%"]
    assert bravo[:2] == ["🔴 Bravo", "throughput down 11.0%"]
    assert about(hl, "Charlie") == []  # −2%: not news


def test_a_change_only_at_low_concurrency_says_so():
    hl = lowlights(
        [model("Alpha", pc(4, 95.0, 100.0), pc(64, 1000.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha[:2] == ["⚠️ Alpha", "c4 throughput down 5.0%"]


def test_the_low_concurrency_is_mentioned_when_it_is_worse_than_c64():
    hl = lowlights(
        [model("Alpha", pc(4, 88.0, 100.0), pc(64, 960.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha[:2] == ["🔴 Alpha", "throughput down 4.0%, c4 down 12.0%"]


def test_the_low_concurrency_is_left_out_when_c64_already_says_it():
    hl = lowlights(
        [model("Alpha", pc(4, 96.0, 100.0), pc(64, 950.0, 1000.0), ac(0.95, 0.95))]
    )
    (alpha,) = about(hl, "Alpha")
    assert alpha[1] == "throughput down 5.0%"


def test_a_gain_is_not_a_lowlight():
    hl = lowlights(
        [model("Alpha", pc(4, 100.0, 100.0), pc(64, 1040.0, 1000.0), ac(0.95, 0.95))]
    )
    assert hl == []


def test_an_accuracy_drop_over_004_is_flagged_and_over_006_is_severe():
    hl = lowlights(
        [
            model("Alpha", FLAT["c4"], FLAT["c64"], ac(0.905, 0.95)),
            model("Bravo", FLAT["c4"], FLAT["c64"], ac(0.88, 0.95)),
            model(
                "Charlie", FLAT["c4"], FLAT["c64"], ac(0.91, 0.95)
            ),  # exactly −0.04: not flagged
        ]
    )
    (alpha,), (bravo,) = about(hl, "Alpha"), about(hl, "Bravo")
    assert alpha[:2] == ["⚠️ Alpha", "GSM8K down 4.5 points to 90.5%"]
    assert bravo[0] == "🔴 Bravo" and "down 7.0 points" in bravo[1]
    assert about(hl, "Charlie") == []


def test_a_recovery_is_not_a_lowlight():
    hl = lowlights(
        [
            model(
                "Alpha",
                pc(4, 100.0, 99.0, prev_failed=True),
                pc(64, 1000.0, 990.0),
                ac(0.95, 0.95),
            )
        ]
    )
    assert hl == []


def test_flat_models_are_not_mentioned():
    assert lowlights([model(n, **FLAT) for n in ("Alpha", "Bravo", "Charlie")]) == []


def test_without_problems_or_good_news_neither_lights_section_is_written():
    flat = [model("Alpha", **FLAT)]
    assert lowlights_md(flat) == []
    text = report_md("2026-11-14", {}, flat, "c", "r")
    assert "## Lowlights" not in text and "## Highlights" not in text
    assert "No failures or regressions" not in text
    assert text.index("\n## Results\n") < 60  # straight after the title


def test_a_first_result_is_not_news():
    assert (
        lowlights(
            [model("Alpha", pc(4, 100.0, None), pc(64, 1000.0, None), ac(0.95, None))]
        )
        == []
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


def test_one_report_holds_the_lights_results_environment_findings_and_details():
    text = report_md(
        "2026-11-14",
        {},
        [model("Alpha", **GAINING), model("Bravo", **FAILING)],
        "c",
        "r",
    )
    assert "## Summary" not in text
    order = [
        "# 2026-11-14 MI455 Workload Weekly Status Report\n",
        "\n## Lowlights\n",
        "\n| Model | What happened | Cause | Fix | Owner |",
        "\n## Highlights\n",
        "\n| Model | What happened |\n",
        "\n## Results\n",
        "\n| Model | Scenario |",
        "\n## Environment\n",
        "\n## Findings\n",
        "\n## Details\n",
        "\n### Environment details\n",
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


def test_the_lowlights_tag_no_one_and_leave_the_owner_blank():
    # The owner is filled in by hand, never by the generator.
    rows = lowlights(
        [model("Alpha", **GAINING), model("Bravo", **FAILING)],
        followups={"bravo": {"action": "repro", "link": "r.md"}},
    )
    assert len(rows) == 1 and rows[0][4] == ""
    assert not any("@" in cell for row in rows for cell in row)


def test_the_owner_column_stays_blank_whatever_the_follow_up_says():
    (row,) = lowlights(
        [model("Bravo", **FAILING)],
        followups={"bravo": {"action": "repro", "link": "r.md", "owner": "Alice"}},
    )
    assert row[4] == ""


def test_gains_and_recoveries_are_highlights_without_an_owner_column():
    rows = highlights(
        [
            model("Alpha", **GAINING),
            model(
                "Bravo",
                pc(4, 100.0, 99.0, prev_failed=True),
                pc(64, 1000.0, 990.0),
                ac(0.95, 0.95),
            ),
            model("Charlie", **FAILING),
        ]
    )
    assert rows == [
        ["🟢 Alpha", "throughput up 4.0%"],
        ["✅ Bravo", "recovered from the previous failure"],
    ]


def test_without_good_news_there_is_no_highlights_section():
    text = report_md("2026-11-14", {}, [model("Bravo", **FAILING)], "c", "r")
    assert "\n## Lowlights\n" in text and "## Highlights" not in text


def test_the_lowlights_are_a_table_of_problems():
    text = report_md(
        "2026-11-14",
        {},
        [model("Alpha", **GAINING), model("Bravo", **FAILING)],
        "c",
        "r",
        followups={"bravo": follow("open_pr", PR_OPEN)},
    )
    table = text.split("\n## Lowlights\n\n")[1].split("\n\n")[0].splitlines()
    assert table == [
        "| Model | What happened | Cause | Fix | Owner |",
        "|---|---|---|---|---|",
        f"| ❌ Bravo | c64 failed | `RuntimeError: x` | Needs [ROCm/aiter#1234]({PR_OPEN}) (open) |  |",
    ]


def test_a_pipe_in_a_cause_does_not_break_the_table():
    (line,) = lowlights_md(
        [model("Bravo", **FAILING)],
        followups={"bravo": {"problem": "a|b layout"}},
    )[2:]
    assert line == "| ❌ Bravo | c64 failed | a\\|b layout | — |  |"


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
        "—",
        "—",
        "−0.2 pts",
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
        "—",
        "—",
        "−0.1 pts",
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
    assert row == ["Bravo", "MLPerf Endpoint", "900", "N/A", "N/A", "—", "—", "N/A"]


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
        "—",
        "—",
        "first",
    ]


def kimi(c64):
    """Kimi-K3 with its MLPerf endpoint target, 250 tok/s/GPU at c64."""
    r = model("Kimi-K3 (2.8T)", pc(4, 14.29, None), c64, ac(0.963, None))
    r.targets = {64: 250.0}
    return r


def test_summary_row_shows_the_c64_throughput_as_a_share_of_its_target():
    row = summary_row(kimi(pc(64, 39.9, None)), "2026-11-21", None)
    assert row[3:6] == ["39.9", "first", "16.0% of 250"]


def test_a_failed_point_has_no_share_of_its_target():
    row = summary_row(
        kimi(pc(64, None, None, status="fail", error="x")), "2026-11-21", None
    )
    assert row[3:6] == ["N/A", "N/A", "N/A"]


def test_summary_row_links_the_unmerged_optimization_prs_by_repo_and_number():
    r = kimi(pc(64, 39.9, None))
    r.optimization_prs = [
        "https://github.com/ROCm/ATOM/pull/2942",
        "https://github.com/ROCm/aiter/pull/7876",
    ]
    row = summary_row(r, "2026-11-21", None)
    assert row[5:] == [
        "16.0% of 250",
        (
            "[ATOM-2942](https://github.com/ROCm/ATOM/pull/2942), "
            "[aiter-7876](https://github.com/ROCm/aiter/pull/7876)"
        ),
        "first",
    ]


def test_the_results_and_details_put_each_concurrency_against_its_own_target():
    text = report_md("2026-11-21", {}, [kimi(pc(64, 39.9, None))], "c", "r")
    assert (
        "| Model | Scenario | — (tok/s/GPU) | 2026-11-21 (tok/s/GPU) | Delta | % of target | Unmerged optimization PRs | GSM8K delta |"
        in text
    )
    assert "| tok/s/GPU | vs — | % of target |" in text
    assert "| Kimi-K3 (2.8T) | tp4 | 4 | 14.3 | first | — |" in text
    assert "| Kimi-K3 (2.8T) | tp4 | 64 | 39.9 | first | 16.0% of 250 |" in text


def test_a_problem_without_a_follow_up_is_just_the_problem():
    (row,) = lowlights([model("Bravo", **FAILING)])
    assert row == ["❌ Bravo", "c64 failed", "`RuntimeError: x`", "—", ""]


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
    assert lowlights([result]) == []


def test_a_model_that_did_not_run_this_session_has_not_recovered():
    # e.g. its machine went down before it ran (455-CI-guide.md §3.6): no rows
    hl = lowlights(
        [
            model(
                "Alpha",
                pc(4, None, 100.0, status="missing", prev_failed=True),
                pc(64, None, 1000.0, status="missing", prev_failed=True),
                ac(None, 0.95, status="missing", prev_failed=True),
            )
        ]
    )
    assert hl == []


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
    assert row == ["Alpha", "MLPerf Endpoint", "1,000", "—", "—", "—", "—", "—"]


PR_OPEN = "https://github.com/ROCm/aiter/pull/1234"
PR_OURS = "https://github.com/ROCm/ATOM/pull/2601"


def follow(action, link, note=""):
    return {"action": action, "link": link, "note": note}


def test_a_failure_carries_the_open_pr_that_fixes_it():
    (row,) = lowlights(
        [model("Bravo", **FAILING)],
        followups={"bravo": follow("open_pr", PR_OPEN)},
    )
    assert row[:2] == ["❌ Bravo", "c64 failed"]
    assert row[3] == f"Needs [ROCm/aiter#1234]({PR_OPEN}) (open)"


def test_a_regression_we_fixed_links_our_verified_pr():
    (row,) = lowlights(
        [model("Alpha", pc(4, 100.0, 100.0), pc(64, 950.0, 1000.0), ac(0.95, 0.95))],
        followups={"alpha": follow("our_pr", PR_OURS)},
    )
    assert row == [
        "⚠️ Alpha",
        "throughput down 5.0%",
        "—",
        f"Passes with [ROCm/ATOM#2601]({PR_OURS}) (verified)",
        "",
    ]


def test_a_regression_takes_its_cause_from_the_follow_up_problem():
    (row,) = lowlights(
        [model("Alpha", pc(4, 100.0, 100.0), pc(64, 950.0, 1000.0), ac(0.95, 0.95))],
        followups={
            "alpha": {**follow("open_pr", PR_OPEN), "problem": "MoE sorting default"}
        },
    )
    assert row[2] == "MoE sorting default"


def test_a_problem_with_only_a_repro_links_it():
    (row,) = lowlights(
        [model("Bravo", **FAILING)],
        followups={"bravo": follow("repro", "../runs/20261114/bravo/repro.md")},
    )
    assert row[3] == "No fix yet; [reproducer](../runs/20261114/bravo/repro.md)"


def test_good_news_is_not_a_lowlight_even_with_a_follow_up():
    rows = lowlights(
        [model("Alpha", **GAINING)],
        followups={"alpha": {**follow("open_pr", PR_OPEN), "problem": "x"}},
    )
    assert rows == []


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


def test_the_details_list_each_problem_with_its_next_step_and_note():
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
        f"| Bravo | — | Passes with [ROCm/ATOM#2601]({PR_OURS}) (verified) | c64 905 tok/s/GPU on 455 |"
        in table
    )
    assert "| Charlie | — | Missing: needs a follow-up | — |" in table
    assert "Alpha" not in table


def test_a_fix_that_takes_two_prs_links_both():
    links = (
        "https://github.com/ROCm/ATOM/pull/3225 https://github.com/ROCm/aiter/pull/7847"
    )
    (row,) = lowlights(
        [model("Bravo", **FAILING)], followups={"bravo": follow("open_pr", links)}
    )
    want = "Needs [ROCm/ATOM#3225](https://github.com/ROCm/ATOM/pull/3225) + [ROCm/aiter#7847](https://github.com/ROCm/aiter/pull/7847) (open)"
    assert row[3] == want


def test_the_findings_text_goes_into_its_own_section():
    text = report_md(
        "2026-11-14", {}, [model("Alpha", **FLAT)], "c", "r", findings="- Note one.\n"
    )
    assert "\n## Findings\n\n- Note one.\n\n## Details\n" in text
    assert text.startswith("# 2026-11-14 MI455 Workload Weekly Status Report\n")


def test_a_failure_says_why_with_the_follow_up_problem_instead_of_the_error():
    (row,) = lowlights(
        [model("Bravo", **FAILING)],
        followups={
            "bravo": {
                "action": "open_pr",
                "link": PR_OPEN,
                "problem": "no cross-node TP on main",
            }
        },
    )
    assert row == [
        "❌ Bravo",
        "c64 failed",
        "no cross-node TP on main",
        f"Needs [ROCm/aiter#1234]({PR_OPEN}) (open)",
        "",
    ]


def test_a_fix_merged_after_the_image_was_built_says_main_has_it():
    (row,) = lowlights(
        [model("Bravo", **FAILING)], followups={"bravo": follow("merged_pr", PR_OURS)}
    )
    assert row[3] == (
        f"Fixed on main by [ROCm/ATOM#2601]({PR_OURS}) (merged after the image was built)"
    )
    assert (
        missing_followups(
            [model("Bravo", **FAILING)], {"bravo": follow("merged_pr", PR_OURS)}
        )
        == []
    )


def test_prs_of_one_repo_name_the_repo_once():
    a1, a2 = (
        "https://github.com/ROCm/ATOM/pull/2647",
        "https://github.com/ROCm/ATOM/pull/2760",
    )
    b1 = "https://github.com/ROCm/aiter/pull/7913"
    (row,) = lowlights(
        [model("Bravo", **FAILING)],
        followups={"bravo": follow("our_pr", f"{a1} {a2} {b1}")},
    )
    assert row[3] == (
        f"Passes with [ROCm/ATOM#2647]({a1}) + [#2760]({a2}) + [ROCm/aiter#7913]({b1}) (verified)"
    )


ENV = {
    "image": "rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114",
    "atom": {
        "commit": "9fec99786b088168aec7729858c1176e99d818e6",
        "date": "2026-11-13",
        "branch": "main",
    },
    "aiter": {
        "commit": "b6f1ff6b7f3754aca8cc8c403b5fcd7a258eba47",
        "date": "2026-11-13",
    },
}


def test_the_environment_sits_after_the_results_and_before_the_findings():
    text = report_md("2026-11-14", ENV, [model("Alpha", **FLAT)], "c", "r")
    assert "- Image:" not in text.split("\n## Results\n")[0]
    head = text.split("\n## Environment\n")[1].split("\n## Findings\n")[0]
    assert "- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114`" in head
    assert (
        "- ATOM: `main` @ `9fec99786b088168aec7729858c1176e99d818e6` (2026-11-13)"
        in head
    )
    assert (
        "- aiter: `main` @ `b6f1ff6b7f3754aca8cc8c403b5fcd7a258eba47` (2026-11-13)"
        in head
    )


def test_a_failure_shows_how_to_reproduce_it_and_the_shared_client_once():
    bench = "SERVED_NAME='/models/x'; SERVER='127.0.0.1:8000'\n# point_yaml <name>\npoint_yaml() {\n  echo yaml\n}\n"
    broken = []
    for name in ("Bravo", "Charlie"):
        r = model(name, **FAILING)
        r.commands = [
            (
                "server_cmd.sh",
                "python3 -m atom.entrypoints.openai_server --model /models/x\n",
            ),
            ("bench_cmd.sh", bench),
        ]
        broken.append(r)
    text = report_md("2026-11-14", ENV, broken, "c", "r")
    failures = text.split("\n### Failures\n")[1]
    assert failures.count("Reproduce:") == 2
    assert "  rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114" in failures
    assert "python3 -m atom.entrypoints.openai_server --model /models/x" in failures
    assert (
        failures.count("point_yaml() {") == 1
    )  # the shared client, shown once at the end
    assert failures.index("#### Shared MLPerf Endpoint client") > failures.rindex(
        "Reproduce:"
    )


def m3(acc, prev=None, baseline=0.9469):
    r = model("MiniMax-M3 (428B)", pc(4, 2260.0, None), pc(64, 12389.0, None), acc)
    r.baseline = baseline
    return r


def test_gsm8k_well_below_the_known_good_value_is_a_lowlight_without_history():
    (row,) = lowlights([m3(ac(0.8726, None))])
    assert row[:2] == [
        "🔴 MiniMax-M3",
        "GSM8K 87.3%, 7.4 points below the known-good 94.7%",
    ]


def test_gsm8k_within_4_points_of_the_known_good_value_is_fine():
    assert lowlights([m3(ac(0.8726, None), baseline=0.91)]) == []


def test_a_gsm8k_drop_already_flagged_against_the_last_session_is_not_repeated():
    rows = lowlights([m3(ac(0.88, 0.95), baseline=0.95)])
    assert [r[1] for r in rows] == ["GSM8K down 7.0 points to 88.0%"]


def acc_only(name, acc, scenario="endpoint"):
    """A Wednesday model: GSM8K ran, performance did not (no perf rows)."""
    return model(
        name,
        pc(4, None, 100.0, status="missing"),
        pc(64, None, 1000.0, status="missing"),
        acc,
        scenario=scenario,
    )


def test_a_gsm8k_only_run_that_fails_says_gsm8k_failed():
    (row,) = lowlights(
        [acc_only("Bravo", ac(None, 0.95, status="fail", error="RuntimeError: y"))]
    )
    assert row[:2] == ["❌ Bravo", "GSM8K failed"]


def test_an_accuracy_only_session_reports_gsm8k_and_no_performance():
    text = report_md(
        "2026-11-18",
        {},
        [
            acc_only("Alpha", ac(0.952, 0.95, prev_date="2026-11-14")),
            acc_only("Bravo", ac(0.88, 0.95, prev_date="2026-11-14"), scenario="8k1k"),
        ],
        "c",
        "r",
    )
    results = text.split("\n## Results\n\n")[1].split("\n\n")[0].splitlines()
    assert results == [
        "| Model | 2026-11-14 (GSM8K) | 2026-11-18 (GSM8K) | Delta |",
        "|---|---|---|---|",
        "| Alpha | 95.0% | 95.2% | +0.2 pts |",
        "| Bravo | 95.0% | 88.0% | 🔴 −7.0 pts |",
    ]
    assert "| 🔴 Bravo | GSM8K down 7.0 points to 88.0% |" in text
    assert "### Performance" not in text and "tok/s/GPU" not in text
    assert "perf-" not in text and "c/accuracy-8k1k.png" in text
