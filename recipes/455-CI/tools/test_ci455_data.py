"""Tests for picking the rows the 455-CI charts draw (ci455_data.py)."""

from ci455_data import accuracy_history, curve_styles, failed_concurrencies, perf_curves

WEEKS = [
    "2026-10-10",
    "2026-10-17",
    "2026-10-24",
    "2026-10-31",
    "2026-11-07",
    "2026-11-14",
]


def perf_row(
    image_date,
    conc,
    status="ok",
    tput="100",
    inter="50",
    tpot="",
    scenario="endpoint",
    model="kimi-k3",
    gpus="8",
):
    return {
        "image_date": image_date,
        "scenario": scenario,
        "model": model,
        "gpus": gpus,
        "concurrency": str(conc),
        "status": status,
        "tok_s_per_gpu": tput,
        "interactivity": inter,
        "tpot_mean_ms": tpot,
    }


def acc_row(
    image_date,
    status="ok",
    flexible="0.95",
    strict="0.94",
    scenario="endpoint",
    model="kimi-k3",
):
    return {
        "image_date": image_date,
        "scenario": scenario,
        "model": model,
        "status": status,
        "gsm8k": flexible,
        "gsm8k_strict": strict,
    }


def test_perf_curves_keep_the_four_newest_images_up_to_the_report_image():
    rows = [perf_row(d, c) for d in WEEKS + ["2026-11-21"] for c in (4, 64)]
    curves = perf_curves(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert [c.image_date for c in curves] == [
        "2026-10-24",
        "2026-10-31",
        "2026-11-07",
        "2026-11-14",
    ]


def test_a_rerun_replaces_the_earlier_row_of_the_same_image_and_concurrency():
    rows = [
        perf_row("2026-11-14", 64, tput="100"),
        perf_row("2026-11-14", 64, tput="120"),
    ]
    (curve,) = perf_curves(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert [(p.c, p.y) for p in curve.points] == [(64, 120.0)]


def test_failed_points_are_not_drawn_and_an_image_without_points_has_no_curve():
    rows = [
        perf_row("2026-10-31", 4, status="fail", tput="", inter=""),
        perf_row("2026-10-31", 64, status="fail", tput="", inter=""),
        perf_row("2026-11-07", 4, tput="90", inter="150"),
        perf_row("2026-11-07", 64, status="fail", tput="", inter=""),
    ]
    curves = perf_curves(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert [(c.image_date, [p.c for p in c.points]) for c in curves] == [
        ("2026-11-07", [4])
    ]


def test_a_row_marked_ok_but_missing_its_numbers_is_not_drawn():
    rows = [
        perf_row("2026-11-14", 4, tput=""),
        perf_row("2026-11-14", 64, tput="900", inter="40"),
    ]
    (curve,) = perf_curves(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert [p.c for p in curve.points] == [64]


def test_endpoint_x_is_the_interactivity_column():
    rows = [perf_row("2026-11-14", 4, tput="150", inter="180.5")]
    (curve,) = perf_curves(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert (curve.points[0].x, curve.points[0].y, curve.gpus) == (180.5, 150.0, 8)


def test_8k1k_x_is_1000_over_the_mean_tpot():
    rows = [
        perf_row(
            "2026-11-18",
            4,
            scenario="8k1k",
            model="glm-5.2",
            gpus="4",
            tput="2500",
            inter="",
            tpot="12.5",
        )
    ]
    (curve,) = perf_curves(rows, "8k1k", "glm-5.2", "2026-11-18")
    assert (curve.points[0].x, curve.points[0].y) == (80.0, 2500.0)


def test_points_of_a_curve_are_sorted_by_x_so_the_line_does_not_zigzag():
    rows = [
        perf_row("2026-11-14", 4, inter="180"),
        perf_row("2026-11-14", 64, inter="45"),
    ]
    (curve,) = perf_curves(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert [p.c for p in curve.points] == [64, 4]


def test_only_the_requested_model_and_scenario_are_picked():
    rows = [
        perf_row("2026-11-14", 4, model="deepseek-r1", tput="1"),
        perf_row("2026-11-14", 4, scenario="8k1k", tpot="10", tput="2"),
        perf_row("2026-11-14", 4, tput="3"),
    ]
    (curve,) = perf_curves(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert [p.y for p in curve.points] == [3.0]


def test_failed_concurrencies_of_the_report_image_follow_its_last_rows():
    rows = [
        perf_row("2026-11-07", 4, status="fail"),  # another image: not reported
        perf_row("2026-11-14", 4, status="fail"),
        perf_row("2026-11-14", 4),  # rerun fixed c4
        perf_row("2026-11-14", 64, status="fail"),
    ]
    assert failed_concurrencies(rows, "endpoint", "kimi-k3", "2026-11-14") == [64]


def test_accuracy_history_keeps_a_failed_image_as_a_gap():
    rows = [
        acc_row("2026-10-31"),
        acc_row("2026-11-07", status="fail", flexible="", strict=""),
        acc_row("2026-11-14"),
    ]
    hist = accuracy_history(rows, "endpoint", "kimi-k3", "2026-11-14")
    assert [(h.image_date, h.flexible, h.strict) for h in hist] == [
        ("2026-10-31", 0.95, 0.94),
        ("2026-11-07", None, None),
        ("2026-11-14", 0.95, 0.94),
    ]


def test_accuracy_history_keeps_the_twelve_newest_images_and_the_last_rerun():
    days = [f"2026-08-{d:02d}" for d in range(1, 15)]  # 14 images
    rows = [acc_row(d, flexible="0.90") for d in days] + [
        acc_row("2026-08-14", flexible="0.93")
    ]
    hist = accuracy_history(rows, "endpoint", "kimi-k3", "2026-08-14")
    assert [h.image_date for h in hist] == days[2:]
    assert hist[-1].flexible == 0.93


def _luminance(hex_color):
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_the_newest_curve_is_solid_and_the_same_deepest_colour_however_many_curves():
    one, four = curve_styles(1), curve_styles(4)
    assert one[-1] == four[-1]
    assert [line for _, line in four] == ["dashed", "dashed", "dashed", "solid"]


def test_older_curves_are_lighter():
    lum = [_luminance(color) for color, _ in curve_styles(4)]
    assert lum == sorted(lum, reverse=True) and len(set(lum)) == 4
