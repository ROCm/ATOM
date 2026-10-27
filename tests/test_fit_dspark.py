import itertools

import numpy as np
import pytest

from atom.benchmarks.fit_dspark import cumulative_ece, fit_sps, fit_sts


def test_sps_uses_latency_interpolation_and_conservative_envelope():
    table = fit_sps([[1, 1.0], [6, 2.0], [6, 3.0], [12, 2.0]], 12)
    assert table[1] == 1000
    assert table[6] == pytest.approx(1000 / 3)
    assert table[12] == pytest.approx(1000 / 3)
    assert table[3] == pytest.approx(1000 / 1.8)
    assert all(a >= b for a, b in itertools.pairwise(table))


def test_sps_refuses_uncovered_runtime():
    with pytest.raises(ValueError):
        fit_sps([[6, 1.0]], 12)


def test_fit_reduces_overconfidence_on_request_heldout_blocks():
    rng = np.random.default_rng(7)
    records = []
    for request in range(100):
        for _ in range(20):
            hits = rng.uniform(size=5) < 0.65
            accepted = next((i for i, v in enumerate(hits) if not v), 5)
            records.append(
                {
                    "request": f"{request:08x}",
                    "confidence": [0.95] * 5,
                    "accepted": accepted,
                }
            )
    temps, stats = fit_sts(records)
    assert stats["training_blocks"] == 1600
    assert stats["held_out_blocks"] == 400
    assert stats["held_out_fitted_ece"] < stats["held_out_raw_ece"]
    assert all(np.isfinite(temps))
    assert cumulative_ece([[0.5] * 5], [0], [1] * 5) > 0


def test_missing_heldout_and_invalid_records_rejected():
    with pytest.raises(ValueError):
        fit_sts(
            [{"request": "00000001", "confidence": [0.8] * 5, "accepted": 3}] * 1000
        )
    with pytest.raises(ValueError):
        fit_sts(
            [{"request": "00000001", "confidence": [float("nan")] * 5, "accepted": 3}]
        )
