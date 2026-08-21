# SPDX-License-Identifier: MIT

import pytest

from atom.model_engine.dynamic_chunking import (
    MAX_PREFIX_OVERHEAD_FRACTION,
    ChunkSizePredictor,
    has_sole_prefill,
    parse_chunking_calibration,
)

A, B, C, GAMMA = 2.5e-5, 0.015, 3.0, 0.002


def _samples(grid):
    prefix_lens = [prefix for prefix, _ in grid]
    chunk_sizes = [chunk for _, chunk in grid]
    latencies = [
        C + GAMMA * prefix + B * chunk + A * (2 * prefix * chunk + chunk * chunk)
        for prefix, chunk in grid
    ]
    return prefix_lens, chunk_sizes, latencies


def test_fit_recovers_latency_coefficients():
    grid = [(0, 64 * i) for i in range(1, 17)]
    grid += [(prefix, chunk) for prefix in (512, 4096) for chunk in (256, 512, 1024)]

    predictor = ChunkSizePredictor.fit(*_samples(grid))

    assert predictor.quadratic_coeff == pytest.approx(A)
    assert predictor.linear_coeff == pytest.approx(B)
    assert predictor.constant_coeff == pytest.approx(C)
    assert predictor.prefix_coeff == pytest.approx(GAMMA)


def test_fit_without_prefix_samples_is_rank_deficient():
    # A prefix-free sweep cannot tell the prefix term from the constant one.
    with pytest.raises(ValueError, match="rank deficient"):
        ChunkSizePredictor.fit(*_samples([(0, 64 * i) for i in range(1, 17)]))


def test_fit_rejects_non_positive_linear_coefficient():
    # A window where fixed per-forward overhead dominates fits with a negative
    # marginal token cost. Clamping that to 0 would leave the solver running on
    # geometry alone, so the fit is refused and chunking stays fixed.
    grid = [(0, 256 * i) for i in range(1, 17)]
    grid += [(prefix, chunk) for prefix in (512, 4096) for chunk in (1024, 2048)]
    latencies = [
        1000.0 + GAMMA * prefix - 0.01 * chunk + A * (2 * prefix * chunk + chunk**2)
        for prefix, chunk in grid
    ]

    with pytest.raises(ValueError, match="positive linear latency"):
        ChunkSizePredictor.fit(
            [prefix for prefix, _ in grid],
            [chunk for _, chunk in grid],
            latencies,
        )


def test_prediction_equalizes_quadratic_increment_and_aligns_down():
    predictor = ChunkSizePredictor(1.0, 0.0, 0.0)

    assert (
        predictor.predict(
            history_len=1024,
            base_chunk_size=1024,
            smooth_factor=1.0,
            alignment=64,
            max_chunk_size=1024,
            min_chunk_size=64,
        )
        == 384
    )


def test_min_chunk_size_floors_the_prediction():
    # Equal latency asks for 32 tokens after this prefix. Every chunk re-pays the
    # per-chunk costs, so the configured floor is what stops the split there.
    predictor = ChunkSizePredictor(1.0, 0.0, 0.0)

    assert (
        predictor.predict(
            history_len=16384,
            base_chunk_size=1024,
            smooth_factor=1.0,
            alignment=64,
            max_chunk_size=1024,
            min_chunk_size=256,
        )
        == 256
    )


def test_prefix_rebuild_is_charged_against_the_equal_latency_budget():
    # gamma * L is a floor the chunk pays before its own tokens are attended to,
    # so it comes out of the budget the chunk has to match the first one.
    with_prefix_cost = ChunkSizePredictor(1e-5, 0.01, 0.0, 0.01)
    without_prefix_cost = ChunkSizePredictor(1e-5, 0.01, 0.0)

    assert with_prefix_cost.equal_latency_chunk(
        16384, 4096
    ) < without_prefix_cost.equal_latency_chunk(16384, 4096)


def test_prefix_rebuild_cost_floors_the_chunk():
    # Same curve twice, once with a per-chunk prefix rebuild. The rebuild is
    # pure overhead, so the chunk must not shrink to where it dominates.
    history_len = 16384
    kwargs = dict(
        history_len=history_len,
        base_chunk_size=4096,
        smooth_factor=1.0,
        alignment=64,
        max_chunk_size=4096,
        min_chunk_size=64,
    )
    without_prefix_cost = ChunkSizePredictor(1e-5, 0.01, 0.0)
    with_prefix_cost = ChunkSizePredictor(1e-5, 0.01, 0.0, 0.01)

    small = without_prefix_cost.predict(**kwargs)
    floored = with_prefix_cost.predict(**kwargs)

    assert floored > small
    # The floor is solved before the chunk is aligned down onto the block grid,
    # so the budget holds within one alignment unit of the returned chunk.
    overhead = with_prefix_cost.prefix_coeff * history_len
    work = with_prefix_cost.chunk_latency(history_len, floored + 64)
    assert overhead / (overhead + work) <= MAX_PREFIX_OVERHEAD_FRACTION


def test_unreachable_equal_latency_keeps_the_fixed_chunk():
    # The prefix rebuild alone costs more than the whole first chunk: no chunk
    # size matches it, and splitting further only pays the rebuild more often.
    predictor = ChunkSizePredictor(1e-5, 0.01, 0.0, 10.0)

    assert (
        predictor.predict(
            history_len=131072,
            base_chunk_size=4096,
            smooth_factor=1.0,
            alignment=64,
            max_chunk_size=4096,
            min_chunk_size=64,
        )
        is None
    )


def test_chunk_never_exceeds_the_batch_budget():
    # A base size decoupled from --max-num-batched-tokens still cannot schedule
    # more tokens than the batch has room for.
    predictor = ChunkSizePredictor(1e-5, 0.01, 0.0)

    assert (
        predictor.predict(
            history_len=0,
            base_chunk_size=32768,
            smooth_factor=1.0,
            alignment=64,
            max_chunk_size=8192,
            min_chunk_size=4096,
        )
        == 8192
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
            min_chunk_size=64,
        )
        == 1024
    )


def test_flat_model_is_not_worth_acting_on():
    # Cost linear in the chunk size and independent of the prefix: equalizing
    # latency returns the initial chunk everywhere, so there is nothing to gain.
    flat = ChunkSizePredictor(1e-12, 6e-3, 0.0, 1e-9)
    growing = ChunkSizePredictor(3e-7, 6e-3, 0.0, 1e-3)

    assert not flat.predicts_useful_shrink(base_chunk_size=32768, history_len=131072)
    assert growing.predicts_useful_shrink(base_chunk_size=32768, history_len=131072)


def test_from_coefficients_accepts_legacy_triples():
    predictor = ChunkSizePredictor.from_coefficients((1.0, 0.5, 2.0))

    assert predictor.prefix_coeff == 0.0


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
            min_chunk_size=64,
        )


def test_sole_prefill_requires_no_other_prefill_work():
    assert has_sole_prefill(1)
    assert has_sole_prefill(0)
    assert not has_sole_prefill(2)


def test_sole_prefill_ignores_a_momentary_lull():
    # One request finished just before the next was admitted. Sizing chunks for
    # an empty pipeline in that gap commits them to run alongside the arrivals.
    assert not has_sole_prefill(1, [1, 5, 3, 1])
    assert has_sole_prefill(1, [1, 1, 0, 1])


def test_calibration_string_parses_into_coefficients():
    assert parse_chunking_calibration("3.25e-7,6.03e-3,12.0,5.39e-4") == (
        pytest.approx(3.25e-7),
        pytest.approx(6.03e-3),
        pytest.approx(12.0),
        pytest.approx(5.39e-4),
    )


@pytest.mark.parametrize(
    "text,match",
    [
        ("not,a,number,here", "comma-separated floats"),
        ("1e-7,6e-3", "three or four coefficients"),
        ("-1e-7,6e-3,1.0,1e-4", "quadratic coefficient must be positive"),
        ("1e-7,6e-3,1.0,-1e-4", "prefix coefficient must be non-negative"),
    ],
)
def test_calibration_string_is_validated(text, match):
    with pytest.raises(ValueError, match=match):
        parse_chunking_calibration(text)
