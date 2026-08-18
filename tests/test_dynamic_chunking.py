# SPDX-License-Identifier: MIT

import pytest

from atom.model_engine.dynamic_chunking import ChunkSizePredictor


def test_fit_recovers_quadratic_latency_curve():
    lengths = [64 * i for i in range(1, 17)]
    latencies = [2.5e-5 * n * n + 0.015 * n + 3.0 for n in lengths]

    predictor = ChunkSizePredictor.fit(lengths, latencies)

    assert predictor.quadratic_coeff == pytest.approx(2.5e-5)
    assert predictor.linear_coeff == pytest.approx(0.015)
    assert predictor.constant_coeff == pytest.approx(3.0)


def test_prediction_equalizes_quadratic_increment_and_aligns_down():
    predictor = ChunkSizePredictor(1.0, 0.0, 0.0)

    assert (
        predictor.predict(
            history_len=1024,
            base_chunk_size=1024,
            smooth_factor=1.0,
            alignment=64,
            max_chunk_size=1024,
        )
        == 384
    )


def test_prediction_enforces_quarter_base_floor():
    predictor = ChunkSizePredictor(1.0, 0.0, 0.0)

    assert (
        predictor.predict(
            history_len=16384,
            base_chunk_size=1024,
            smooth_factor=1.0,
            alignment=64,
            max_chunk_size=1024,
        )
        == 256
    )


def test_zero_smoothing_matches_fixed_chunking():
    predictor = ChunkSizePredictor(1.0, 0.0, 0.0)

    assert (
        predictor.predict(
            history_len=8192,
            base_chunk_size=1024,
            smooth_factor=0.0,
            alignment=64,
            max_chunk_size=1024,
        )
        == 1024
    )


@pytest.mark.parametrize("smooth_factor", [-0.1, 1.1])
def test_invalid_smoothing_factor_is_rejected(smooth_factor):
    predictor = ChunkSizePredictor(1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        predictor.predict(
            history_len=0,
            base_chunk_size=1024,
            smooth_factor=smooth_factor,
            alignment=64,
            max_chunk_size=1024,
        )
