# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from atom.model_engine.dynamic_chunking import (
    CALIBRATION_SWEEP_RATIO,
    MAX_CALIBRATION_FIT_FAILURES,
    MAX_PREFIX_OVERHEAD_FRACTION,
    ChunkLatencyCalibrator,
    ChunkSizePredictor,
    fit_chunk_overhead,
    has_sole_prefill,
)
from atom.model_engine.dynamic_chunking_worker import (
    MAX_CALIBRATION_POLLS_WITHOUT_SAMPLES,
    MAX_CALIBRATION_REQUESTS_WITHOUT_DIVERSITY,
    ChunkTimingSample,
    DynamicChunkingWorker,
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


def test_an_unaligned_min_chunk_size_still_floors_the_prediction():
    # The floor is aligned up, not down: a min chunk off the block grid must not
    # let the chunk drop a whole alignment unit below it.
    predictor = ChunkSizePredictor(1.0, 0.0, 0.0)

    assert (
        predictor.predict(
            history_len=131072,
            base_chunk_size=8192,
            smooth_factor=1.0,
            alignment=128,
            max_chunk_size=8192,
            min_chunk_size=4000,
        )
        == 4096
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
    assert has_sole_prefill([1])
    assert has_sole_prefill([0])
    assert not has_sole_prefill([2])
    # No census yet is no evidence of a sole prefill.
    assert not has_sole_prefill([])


def test_sole_prefill_ignores_a_momentary_lull():
    # One request finished just before the next was admitted. Sizing chunks for
    # an empty pipeline in that gap commits them to run alongside the arrivals.
    assert not has_sole_prefill([1, 2, 2, 1])
    assert has_sole_prefill([1, 1, 0, 1])


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


class _FakeEvent:
    """A CUDA timing event on a manual clock, already complete when recorded."""

    clock = 0.0

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self):
        self.t = _FakeEvent.clock

    def synchronize(self):
        pass

    def query(self):
        return True

    def elapsed_time(self, end):
        return end.t - self.t


@pytest.fixture
def fake_events(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    _FakeEvent.clock = 0.0


def _runner(rank=0, pp_rank=0, enabled=True, warmup_max_tokens=1024):
    return SimpleNamespace(
        rank=rank,
        label=f"pp{pp_rank}",
        block_size=64,
        hf_text_config=SimpleNamespace(vocab_size=1024),
        config=SimpleNamespace(
            enable_dynamic_chunking=enabled,
            pipeline_parallel_size=2,
            max_model_len=131072,
            parallel_config=SimpleNamespace(pipeline_parallel_rank=pp_rank),
        ),
        warmup_max_tokens=lambda: warmup_max_tokens,
        flush_pp_send=MagicMock(),
    )


def _bracketed_forward(worker, cost_ms):
    """Mimic `ModelRunner.run_model`: only the model call sits in the bracket."""

    def forward(batch):
        _FakeEvent.clock += 100.0  # input staging and the PP receive
        sample = worker.start_sample(batch)
        _FakeEvent.clock += cost_ms(batch.total_tokens_num)
        worker.finish_sample(sample)
        _FakeEvent.clock += 100.0  # the PP send

    return forward


def _profiling_worker(**runner_kwargs):
    runner = _runner(**runner_kwargs)
    worker = DynamicChunkingWorker(runner)
    runner.forward = _bracketed_forward(worker, lambda chunk: C + B * chunk)
    return worker


def test_startup_sweep_times_the_window_runtime_calibration_times(fake_events):
    # The constant fitted at startup is reused at runtime, so both must measure
    # the same window: the model call, without staging or PP transfers.
    worker = _profiling_worker()

    latencies = worker._sweep([256, 128, 64], vocab_size=1024)

    assert latencies == pytest.approx([C + B * 256, C + B * 128, C + B * 64])
    worker._runner.flush_pp_send.assert_called()
    assert worker._sweep_samples is None


def test_startup_profiling_fits_the_overhead_within_the_warmup_bound(fake_events):
    worker = _profiling_worker(warmup_max_tokens=1000)
    grids = []
    sweep = worker._sweep
    worker._sweep = lambda grid, vocab: grids.append(grid) or sweep(grid, vocab)

    result = worker.profile()

    assert result["linear_coeff"] == pytest.approx(B)
    assert result["constant_coeff"] == pytest.approx(C)
    # Aligned down from what warmup sized activations for, never above it.
    assert max(grids[0]) == 960
    assert worker._calibrator is not None


@pytest.mark.parametrize(
    "rank,expected",
    [(0, {"error": "profiling needs hf_text_config.vocab_size; got None"}), (1, None)],
)
def test_a_sweep_that_cannot_run_answers_from_rank_zero_before_any_forward(
    fake_events, rank, expected
):
    # `call_func(wait_out=True)` waits on rank 0 with no timeout, and every
    # stage has to skip the sweep before a forward crosses stages.
    worker = _profiling_worker(rank=rank)
    worker._runner.hf_text_config = SimpleNamespace(vocab_size=None)
    worker._runner.forward = MagicMock()

    assert worker.profile() == expected
    worker._runner.forward.assert_not_called()
    assert worker._calibrator is None


def test_a_failing_sweep_forward_takes_the_worker_down(fake_events):
    # Answering it as an error would let this stage start serving while its
    # neighbours stay blocked in the PP send/recv. Escaping fails startup, as
    # a failed warmup forward does.
    worker = _profiling_worker()
    worker._runner.forward = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        worker.profile()
    assert worker._calibrator is None


def test_profiling_answers_when_the_feature_is_off():
    assert "error" in DynamicChunkingWorker(_runner(enabled=False)).profile()


def test_only_the_head_stage_calibrates_from_real_prefills(fake_events):
    worker = _profiling_worker(pp_rank=1)

    assert "linear_coeff" in worker.profile()
    assert worker._calibrator is None


def test_a_calibration_error_gives_up_instead_of_raising():
    worker = DynamicChunkingWorker(_runner())
    worker._calibrator = ChunkLatencyCalibrator(B, C)
    worker._take_fit = MagicMock(side_effect=RuntimeError("boom"))

    assert worker.take_fit() == {"coefficients": None, "gave_up": True}
    assert worker._calibrator is None


def test_worker_gives_up_when_calibration_never_sees_two_chunk_sizes():
    worker = DynamicChunkingWorker(_runner())
    worker._calibrator = ChunkLatencyCalibrator(B, C)
    worker._requests_seen = MAX_CALIBRATION_REQUESTS_WITHOUT_DIVERSITY

    fit = worker.take_fit()

    assert fit["gave_up"] is True
    assert fit["coefficients"] is None
    assert worker._calibrator is None


def test_worker_gives_up_when_polls_see_no_samples():
    worker = DynamicChunkingWorker(_runner())
    worker._calibrator = ChunkLatencyCalibrator(B, C)

    for _ in range(MAX_CALIBRATION_POLLS_WITHOUT_SAMPLES - 1):
        assert worker.take_fit() == {"coefficients": None}
    fit = worker.take_fit()

    assert fit["gave_up"] is True
    assert worker._calibrator is None


def test_a_cache_hit_follow_up_counts_as_a_new_request(fake_events):
    # A multi-turn follow-up can start on a cache hit deeper than where the
    # previous turn's last chunk began; only the request id tells them apart.
    worker = DynamicChunkingWorker(_runner())
    worker._calibrator = ChunkLatencyCalibrator(B, C)
    for req_id, prefix in ((1, 0), (1, 2048), (2, 4096)):
        start, end = _FakeEvent(), _FakeEvent()
        start.record()
        _FakeEvent.clock += TRUTH.predicted_latency(prefix, 2048)
        end.record()
        worker._pending.append(ChunkTimingSample(start, end, req_id, prefix, 2048))

    worker._drain()

    assert worker._requests_seen == 2
    # The first request's timings carry kernel warmup and are discarded.
    assert worker._calibrator.num_shapes == 1


def test_startup_profiling_batch_carries_no_block_table():
    # As in warmup: a dummy forward writes no KV, and fabricated block ids would
    # have to fit every stage's own pool.
    batch = DynamicChunkingWorker(_runner())._dummy_batch(256, vocab_size=1024)

    assert batch.block_tables == []


def test_startup_profiling_batch_covers_the_chunk_it_times():
    """One prefill sequence, no cached prefix: the shape the sweep fits on."""
    chunk_size = 192
    batch = DynamicChunkingWorker(_runner())._dummy_batch(chunk_size, vocab_size=1024)

    assert batch.is_dummy_run
    assert batch.total_seqs_num == batch.total_seqs_num_prefill == 1
    assert batch.total_tokens_num == batch.total_tokens_num_prefill == chunk_size
    assert list(batch.num_scheduled_tokens) == [chunk_size]
    assert list(batch.num_cached_tokens) == [0]


def test_a_served_prefill_is_not_timed_without_a_calibrator(fake_events):
    worker = DynamicChunkingWorker(_runner())
    batch = worker._dummy_batch(64, vocab_size=1024)
    batch.is_dummy_run = False

    assert worker.start_sample(batch) is None
