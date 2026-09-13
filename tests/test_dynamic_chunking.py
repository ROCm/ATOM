# SPDX-License-Identifier: MIT

import pytest

from atom.model_engine.dynamic_chunking import (
    CALIBRATION_SWEEP_RATIO,
    MAX_CALIBRATION_FIT_FAILURES,
    MAX_PREFIX_OVERHEAD_FRACTION,
    ChunkLatencyCalibrator,
    ChunkSizePredictor,
    fit_chunk_overhead,
    has_sole_prefill,
)

A, B, C, GAMMA = 2.5e-5, 0.015, 3.0, 0.002

TRUTH = ChunkSizePredictor(A, B, C, GAMMA)


def _dummy_sweep(chunks):
    """Startup profiling samples: attention is bypassed, so only `c + b*x`."""
    return list(chunks), [C + B * chunk for chunk in chunks]


def _serve(calibrator, prompt_len, chunk_size, latency=None):
    """Feed one request's chunks, as fixed-size chunked prefill produces them."""
    latency = latency or TRUTH.predicted_latency
    prefix = 0
    while prefix < prompt_len:
        chunk = min(chunk_size, prompt_len - prefix)
        calibrator.add(prefix, chunk, latency(prefix, chunk))
        prefix += chunk


def _sweep(calibrator, prompt_len=8192, base=2048, latency=None):
    """The scheduler's calibration sweep: one prompt per base chunk size."""
    for chunk_size in (base, base // CALIBRATION_SWEEP_RATIO):
        _serve(calibrator, prompt_len, chunk_size, latency=latency)


def test_chunk_overhead_fit_recovers_the_attention_free_terms():
    linear, constant = fit_chunk_overhead(*_dummy_sweep(range(64, 2048, 64)))

    assert linear == pytest.approx(B)
    assert constant == pytest.approx(C)


def test_chunk_overhead_fit_rejects_non_positive_linear_coefficient():
    # A window where fixed per-forward overhead dominates fits with a negative
    # marginal token cost. Calibration would then read the whole chunk cost as
    # attention, so the baseline is refused and chunking stays fixed.
    chunks = list(range(256, 4352, 256))
    latencies = [1000.0 - 0.01 * chunk for chunk in chunks]

    with pytest.raises(ValueError, match="positive linear latency"):
        fit_chunk_overhead(chunks, latencies)


def test_calibration_recovers_the_model_from_the_sweep():
    # Only `b` is carried over from startup profiling; the sweep's two chunk sizes
    # are what let the other three terms be told apart.
    calibrator = ChunkLatencyCalibrator(B, C)
    _sweep(calibrator)

    predictor = calibrator.fit()

    assert predictor.quadratic_coeff == pytest.approx(A)
    assert predictor.prefix_coeff == pytest.approx(GAMMA)
    assert predictor.constant_coeff == pytest.approx(C)
    assert predictor.linear_coeff == pytest.approx(B)


def test_calibration_will_not_fit_a_single_chunk_size():
    # However many prefills arrive, one chunk size makes the area column affine in
    # the prefix, so the terms are not identifiable at all. Fitting anyway is what
    # left `gamma` around 60% high and the feature no better than a tuned fixed
    # chunk, so the sweep's second size is a precondition, not an improvement.
    calibrator = ChunkLatencyCalibrator(B, C)
    for _ in range(6):
        _serve(calibrator, prompt_len=8192, chunk_size=2048)

    assert calibrator.num_chunk_sizes == 1
    assert calibrator.maybe_fit() is None


def test_calibration_corrects_a_biased_startup_constant():
    # A dummy forward reaches `b` without ever setting attention up, so its
    # constant is not the one a real chunk pays. Pinned, that error lands on the
    # prefix rebuild - the term the feature turns on - so the sweep refits it.
    calibrator = ChunkLatencyCalibrator(B, C * 3.0)
    _sweep(calibrator)

    predictor = calibrator.fit()

    assert predictor.constant_coeff == pytest.approx(C)
    assert predictor.prefix_coeff == pytest.approx(GAMMA)


def test_calibration_survives_a_biased_linear_baseline():
    # Startup profiling measures `b` on a different code path than serving, so it
    # comes in a few percent off, and that one is not refitted. The error is
    # constant in the prefix, so it lands in the area term and leaves the solved
    # chunks close to the truth.
    calibrator = ChunkLatencyCalibrator(B * 1.07, C)
    _sweep(calibrator, prompt_len=131072, base=32768)

    predictor = calibrator.fit()
    kwargs = {
        "base_chunk_size": 32768,
        "smooth_factor": 1.0,
        "alignment": 64,
        "max_chunk_size": 32768,
        "min_chunk_size": 4096,
    }
    for history_len in (32768, 65536, 98304):
        assert predictor.predict(history_len=history_len, **kwargs) == pytest.approx(
            TRUTH.predict(history_len=history_len, **kwargs), rel=0.05
        )


def test_calibration_waits_for_a_spread_of_prefixes():
    # Every sample at the same prefix leaves the area term and the prefix
    # rebuild indistinguishable, so no fit is attempted.
    calibrator = ChunkLatencyCalibrator(B, C)
    for chunk in range(64, 1024, 64):
        calibrator.add(4096, chunk, TRUTH.predicted_latency(4096, chunk))

    assert calibrator.maybe_fit() is None


def test_calibration_rejects_an_ill_conditioned_chunk_grid():
    # Merely seeing two distinct chunk sizes is insufficient when they are
    # effectively identical: timing noise would then dominate their difference.
    calibrator = ChunkLatencyCalibrator(B, C)
    for chunk in (2048, 2032):
        for prefix in (0, 2048, 4096, 6144):
            calibrator.add(prefix, chunk, TRUTH.predicted_latency(prefix, chunk))

    with pytest.raises(ValueError, match="design condition"):
        calibrator.fit()


def test_calibration_rejects_an_uncertain_latency_prediction():
    # This noise is small enough to pass the broad residual sanity check, but it
    # leaves the fitted latency uncertain by more than the scheduler may consume.
    calibrator = ChunkLatencyCalibrator(B, C)

    def noisy_latency(prefix, chunk):
        latency = TRUTH.predicted_latency(prefix, chunk)
        direction = 1.0 if (prefix // chunk) % 3 == 0 else -1.0
        return latency * (1.0 + 0.06 * direction)

    _sweep(calibrator, latency=noisy_latency)

    with pytest.raises(ValueError, match="prediction uncertainty"):
        calibrator.maybe_fit()

    # Repeated measurements refine the per-shape medians and permit one retry;
    # no new scheduler shapes are required to recover from transient noise.
    _sweep(calibrator)
    assert calibrator.maybe_fit() is not None


def test_calibration_converges_on_a_uniform_length_workload():
    # The shape a benchmark and a fixed-length production workload both produce:
    # every prompt is the same length, so the run offers only `prompt / chunk`
    # shapes per base size however many requests arrive. The sweep is what makes
    # that enough, and it has to be, because nothing else will widen it.
    calibrator = ChunkLatencyCalibrator(B, C)
    for _ in range(3):
        _sweep(calibrator, prompt_len=131072, base=32768)

    assert calibrator.num_shapes == 4 + 4 * CALIBRATION_SWEEP_RATIO
    assert calibrator.maybe_fit() is not None


def test_calibration_retries_on_fresh_samples_but_not_on_every_poll():
    # A fit that raises should not be retried until there is a sample it has not
    # already seen, or a busy server re-solves the same unusable data every poll.
    calibrator = ChunkLatencyCalibrator(B, C)
    _sweep(
        calibrator,
        latency=lambda prefix, chunk: TRUTH.predicted_latency(prefix, chunk)
        * (3.0 if prefix % 4096 else 0.4),
    )

    with pytest.raises(ValueError, match="not described by the chunk latency model"):
        calibrator.maybe_fit()
    assert calibrator.maybe_fit() is None


def test_calibration_gives_up_after_repeated_rejections():
    # Noise a workload never shakes off would otherwise retry for the life of the
    # process, keeping every other request at the sweep size. Fresh samples buy
    # retries, but only a bounded number of them.
    calibrator = ChunkLatencyCalibrator(B, C)

    def unusable(prefix, chunk):
        return TRUTH.predicted_latency(prefix, chunk) * (3.0 if prefix % 4096 else 0.4)

    for _ in range(MAX_CALIBRATION_FIT_FAILURES):
        assert not calibrator.gave_up
        _sweep(calibrator, latency=unusable)
        with pytest.raises(ValueError):
            calibrator.maybe_fit()

    assert calibrator.gave_up
    assert calibrator.num_failed_fits == MAX_CALIBRATION_FIT_FAILURES


def test_calibration_rejects_a_cost_that_falls_with_attention_area():
    calibrator = ChunkLatencyCalibrator(B, C)
    _sweep(
        calibrator,
        latency=lambda prefix, chunk: (
            C + B * chunk - (A / 1000.0) * (2 * prefix * chunk + chunk * chunk)
        ),
    )

    with pytest.raises(ValueError, match="no attention growth"):
        calibrator.fit()


def test_calibration_rejects_no_attention_growth():
    # A cost that does not rise with attention area leaves an equal-latency solver
    # nothing to equalize. An exact zero can fit as a tiny value of either sign
    # across NumPy versions, so the scale-aware growth guard rejects both.
    calibrator = ChunkLatencyCalibrator(B, C)
    _sweep(calibrator, latency=lambda prefix, chunk: C + B * chunk)

    with pytest.raises(ValueError, match="no attention growth"):
        calibrator.fit()


def test_calibration_drops_a_negative_prefix_term():
    # Shape noise can push the prefix rebuild negative, which is not physical.
    # The area term is refitted without it rather than left paired with a clamp.
    calibrator = ChunkLatencyCalibrator(B, C)
    _sweep(
        calibrator,
        latency=lambda prefix, chunk: (
            ChunkSizePredictor(A, B, C, 0.0).predicted_latency(prefix, chunk)
            - 0.001 * prefix
        ),
    )

    predictor = calibrator.fit()

    assert predictor.prefix_coeff == 0.0
    assert predictor.quadratic_coeff > 0.0


def test_calibration_keeps_the_startup_constant_over_a_negative_fit():
    # A fitted constant below zero would mean a forward that costs less than
    # nothing to launch, and it drags the terms fitted beside it.
    calibrator = ChunkLatencyCalibrator(B, C)
    _sweep(
        calibrator,
        latency=lambda prefix, chunk: (
            TRUTH.predicted_latency(prefix, chunk) - 4.0 * C
        ),
    )

    assert calibrator.fit().constant_coeff == C


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
    kwargs = {
        "history_len": history_len,
        "base_chunk_size": 4096,
        "smooth_factor": 1.0,
        "alignment": 64,
        "max_chunk_size": 4096,
        "min_chunk_size": 64,
    }
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


@pytest.mark.parametrize(
    "coefficients,match",
    [
        ((1e-7, 6e-3), "three or four coefficients"),
        ((-1e-7, 6e-3, 1.0, 1e-4), "quadratic coefficient must be positive"),
        ((1e-7, 6e-3, 1.0, -1e-4), "prefix coefficient must be non-negative"),
    ],
)
def test_coefficients_are_validated(coefficients, match):
    with pytest.raises(ValueError, match=match):
        ChunkSizePredictor.from_coefficients(coefficients)
