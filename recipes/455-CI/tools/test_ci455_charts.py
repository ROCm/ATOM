"""Tests for the 455-CI chart builders and renderers (ci455_charts.py)."""

import struct

from ci455_charts import (
    axis_range,
    endpoint_series,
    failure_note,
    render_8k1k,
    render_accuracy,
    render_endpoint,
)
from ci455_data import AccuracyPoint, Curve, Point

CURVES = [
    Curve("2026-11-07", 8, [Point(64, 45.0, 900.0), Point(4, 160.0, 120.0)]),
    Curve("2026-11-14", 8, [Point(64, 48.0, 950.0), Point(4, 170.0, 125.0)]),
]


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def test_failure_note_says_the_whole_run_failed_when_every_concurrency_did():
    note = failure_note("2026-11-14", [4, 64], [4, 64])
    assert "11-14" in note and "run failed" in note


def test_failure_note_names_only_the_concurrency_that_failed():
    note = failure_note("2026-11-14", [64], [4, 64])
    assert "c64" in note and "c4 " not in note and "run failed" not in note


def test_no_failure_note_when_nothing_failed():
    assert failure_note("2026-11-14", [], [4, 64]) is None


def test_axis_range_covers_the_data_with_headroom_on_whole_ticks():
    for values, from_zero in (
        ([45.0, 170.0], False),
        ([120.0, 950.0], True),
        ([7.5, 7.5], False),
    ):
        lo, hi, step = axis_range(values, from_zero=from_zero)
        assert isinstance(step, int) and step > 0
        assert lo % step == 0 and hi % step == 0
        assert lo <= min(values) and hi > max(values)
        assert 2 <= (hi - lo) // step <= 8
        if from_zero:
            assert lo == 0


def test_endpoint_series_label_each_curve_with_its_image_date_and_gpu_count():
    specs = endpoint_series(CURVES)
    assert [
        (day in s["legend"], "8 GPUs" in s["legend"])
        for day, s in zip(["11-07", "11-14"], specs, strict=True)
    ] == [(True, True)] * 2
    assert specs[-1]["line"] == "solid" and specs[0]["line"] == "dashed"
    assert all(s["stub"] is False for s in specs)
    one_card = endpoint_series([Curve("2026-11-14", 1, [Point(4, 100.0, 50.0)])])
    assert "1 GPU" in one_card[0]["legend"] and "1 GPUs" not in one_card[0]["legend"]


def test_endpoint_series_points_carry_per_gpu_throughput():
    (spec,) = endpoint_series([CURVES[1]])
    assert [(p["c"], p["x"], p["y_gpu"]) for p in spec["points"]] == [
        (64, 48.0, 950.0),
        (4, 170.0, 125.0),
    ]


def test_render_endpoint_writes_a_2640px_wide_png(tmp_path):
    out = tmp_path / "perf-kimi-k3.png"
    render_endpoint(
        CURVES,
        str(out),
        "Kimi-K3 (2.8T)",
        note="11-14 image: c64 failed (N/A), see report.",
    )
    assert png_size(out)[0] == 2640


def test_render_8k1k_writes_a_2640px_wide_png(tmp_path):
    out = tmp_path / "perf-glm-5.2.png"
    curves = [
        Curve("2026-11-07", 4, [Point(64, 30.0, 11000.0), Point(4, 95.0, 2600.0)]),
        Curve("2026-11-14", 4, [Point(64, 32.0, 11800.0), Point(4, 99.0, 2750.0)]),
    ]
    render_8k1k(curves, str(out), "GLM-5.2")
    assert png_size(out)[0] == 2640


def test_render_accuracy_writes_a_2640px_wide_png(tmp_path):
    out = tmp_path / "accuracy-endpoint.png"
    hist = [
        AccuracyPoint("2026-10-31", 0.951, 0.943),
        AccuracyPoint("2026-11-07", None, None),
        AccuracyPoint("2026-11-14", 0.957, 0.949),
    ]
    render_accuracy(
        [("Kimi-K3 (2.8T)", hist), ("DeepSeek R1", [])],
        str(out),
        "MLPerf Endpoint",
        baselines={"Kimi-K3 (2.8T)": 0.95},
    )
    assert png_size(out)[0] == 2640
