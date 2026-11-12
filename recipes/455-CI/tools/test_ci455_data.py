"""Tests for picking and comparing the 455-CI results (ci455_data.py).

A session is one day's run of one scenario (Saturday: MLPerf Endpoint,
Wednesday: 8K/1K on the previous Saturday's image), keyed by its test date."""

from ci455_data import acc_cell, accuracy_history, perf_cells, perf_trend


def perf_row(
    date,
    conc,
    status="ok",
    tput="100",
    inter="50",
    itl="",
    tpot="",
    scenario="endpoint",
    model="kimi-k3",
    gpus="8",
    error="",
    image_date=None,
):
    return {
        "image_date": image_date or date,
        "test_date": date,
        "scenario": scenario,
        "model": model,
        "gpus": gpus,
        "concurrency": str(conc),
        "status": status,
        "tok_s_per_gpu": tput,
        "interactivity": inter,
        "itl_p50_ms": itl,
        "tpot_mean_ms": tpot,
        "error": error,
    }


def acc_row(
    date,
    status="ok",
    flexible="0.95",
    strict="0.94",
    scenario="endpoint",
    model="kimi-k3",
    error="",
    image_date=None,
):
    return {
        "image_date": image_date or date,
        "test_date": date,
        "scenario": scenario,
        "model": model,
        "status": status,
        "gsm8k": flexible,
        "gsm8k_strict": strict,
        "error": error,
    }


def test_perf_cells_compare_with_the_newest_earlier_success_not_the_failure_after_it():
    rows = [
        perf_row("2026-10-24", 64, tput="420", inter="39"),
        perf_row("2026-10-31", 64, tput="400", inter="40"),
        perf_row("2026-11-07", 64, status="fail", tput="", inter="", error="HIP error"),
        perf_row("2026-11-14", 64, tput="380", inter="42"),
    ]
    _, c64 = perf_cells(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert (c64.status, c64.value, c64.second) == ("ok", 380.0, 42.0)
    assert (c64.prev, c64.prev_second, c64.prev_date) == (400.0, 40.0, "2026-10-31")
    assert (c64.best, c64.prev_failed, c64.gpus) == (420.0, True, 8)
    assert (round(c64.delta, 4), round(c64.delta_best, 4)) == (-0.05, -0.0952)


def test_a_wednesday_session_on_a_saturday_image_is_keyed_by_its_test_date():
    rows = [
        perf_row(
            "2026-11-11",
            4,
            scenario="8k1k",
            model="glm-5.2",
            tput="2400",
            tpot="12",
            image_date="2026-11-07",
        ),
        perf_row(
            "2026-11-18",
            4,
            scenario="8k1k",
            model="glm-5.2",
            tput="2500",
            tpot="12",
            image_date="2026-11-14",
        ),
    ]
    c4, _ = perf_cells(rows, "8k1k", "glm-5.2", "2026-11-18")
    assert (c4.value, c4.prev, c4.prev_date) == (2500.0, 2400.0, "2026-11-11")


def test_a_concurrency_without_a_row_in_the_session_is_missing():
    c4, _ = perf_cells(
        [perf_row("2026-11-14", 64)], "endpoint", "kimi-k3", "2026-11-14"
    )
    assert (c4.status, c4.value, c4.delta) == ("missing", None, None)


def test_a_row_marked_ok_but_without_a_number_is_missing_not_failed():
    c4, _ = perf_cells(
        [perf_row("2026-11-14", 4, tput="")], "endpoint", "kimi-k3", "2026-11-14"
    )
    assert (c4.status, c4.value) == ("missing", None)


def test_a_failed_point_keeps_its_error_and_has_no_value():
    rows = [
        perf_row("2026-11-07", 4, tput="60"),
        perf_row("2026-11-14", 4, status="fail", tput="", error="RuntimeError: boom"),
    ]
    c4, _ = perf_cells(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert (c4.status, c4.value, c4.prev, c4.error) == (
        "fail",
        None,
        60.0,
        "RuntimeError: boom",
    )


def test_the_last_row_of_a_rerun_wins():
    rows = [
        perf_row("2026-11-14", 4, status="fail", tput=""),
        perf_row("2026-11-14", 4, tput="70"),
    ]
    c4, _ = perf_cells(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert (c4.status, c4.value) == ("ok", 70.0)


def test_8k1k_cells_carry_the_itl_p50_as_their_second_metric():
    rows = [
        perf_row(
            "2026-11-18",
            4,
            scenario="8k1k",
            model="glm-5.2",
            gpus="4",
            tput="2500",
            inter="",
            itl="12.1",
            tpot="12.5",
        )
    ]
    c4, _ = perf_cells(rows, "8k1k", "glm-5.2", "2026-11-18")
    assert (c4.value, c4.second, c4.gpus) == (2500.0, 12.1, 4)


def test_perf_cells_ignore_other_models_and_later_sessions():
    rows = [
        perf_row("2026-11-14", 4, model="deepseek-r1", tput="1"),
        perf_row("2026-11-21", 4, tput="2"),
        perf_row("2026-11-14", 4, tput="3"),
    ]
    c4, _ = perf_cells(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert (c4.value, c4.prev) == (3.0, None)


def test_perf_trend_lists_each_session_the_model_ran_with_none_for_a_failure():
    rows = [
        perf_row("2026-10-31", 64, tput="400"),
        perf_row("2026-11-07", 4, tput="60"),
        perf_row("2026-11-07", 64, status="fail", tput=""),
        perf_row("2026-11-14", 64, tput="410"),
        perf_row("2026-11-21", 64, tput="999"),
    ]
    assert perf_trend(rows, "endpoint", "kimi-k3", 64, "2026-11-14") == [
        ("2026-10-31", 400.0),
        ("2026-11-07", None),
        ("2026-11-14", 410.0),
    ]


def test_perf_trend_keeps_the_twelve_newest_sessions():
    days = [f"2026-08-{d:02d}" for d in range(1, 15)]
    rows = [perf_row(d, 4) for d in days]
    assert [
        d for d, _ in perf_trend(rows, "endpoint", "kimi-k3", 4, "2026-08-14")
    ] == days[2:]


def test_acc_cell_compares_with_the_newest_earlier_success():
    rows = [
        acc_row("2026-10-31", flexible="0.95"),
        acc_row("2026-11-07", status="fail", flexible="", strict="", error="HIP error"),
        acc_row("2026-11-14", flexible="0.90", strict="0.89"),
    ]
    cell = acc_cell(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert (
        cell.status,
        cell.value,
        cell.strict,
        cell.prev,
        cell.prev_date,
        cell.prev_failed,
    ) == ("ok", 0.90, 0.89, 0.95, "2026-10-31", True)
    assert round(cell.delta, 4) == -0.05


def test_acc_cell_of_a_model_not_run_in_the_session_is_missing():
    cell = acc_cell([acc_row("2026-11-07")], "endpoint", "kimi-k3", "2026-11-14")
    assert (cell.status, cell.value, cell.prev, cell.delta) == (
        "missing",
        None,
        0.95,
        None,
    )


def test_accuracy_history_keeps_a_failed_session_as_a_gap():
    rows = [
        acc_row("2026-10-31"),
        acc_row("2026-11-07", status="fail", flexible="", strict=""),
        acc_row("2026-11-14"),
    ]
    hist = accuracy_history(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert [(h.date, h.flexible, h.strict) for h in hist] == [
        ("2026-10-31", 0.95, 0.94),
        ("2026-11-07", None, None),
        ("2026-11-14", 0.95, 0.94),
    ]


def test_accuracy_history_keeps_the_twelve_newest_sessions_and_the_last_rerun():
    days = [f"2026-08-{d:02d}" for d in range(1, 15)]
    rows = [acc_row(d, flexible="0.90") for d in days] + [
        acc_row("2026-08-14", flexible="0.93")
    ]
    hist = accuracy_history(rows, "endpoint", "kimi-k3", "2026-08-14")
    assert [h.date for h in hist] == days[2:]
    assert hist[-1].flexible == 0.93
