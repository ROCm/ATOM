"""Tests for the 455-CI chart renderers (ci455_charts.py)."""

import struct

from ci455_charts import axis_range, render_accuracy, render_perf_trends
from ci455_data import AccuracyPoint


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


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


def test_render_perf_trends_writes_a_2640px_wide_png(tmp_path):
    out = tmp_path / "perf-endpoint.png"
    trend = [("2026-10-31", 400.0), ("2026-11-07", None), ("2026-11-14", 410.0)]
    render_perf_trends(
        [
            ("Kimi-K3 (2.8T) · c4", [("2026-11-14", 60.0)]),
            ("Kimi-K3 (2.8T) · c64", trend),
            ("DeepSeek R1 · c4", []),
            ("DeepSeek R1 · c64", [("2026-11-14", None)]),
        ],
        str(out),
        "MLPerf Endpoint",
    )
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
