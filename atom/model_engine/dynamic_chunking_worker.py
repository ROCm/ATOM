# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker half of dynamic chunking: startup sweep, prefill timing, fit hand-off.

``ModelRunner`` owns one ``DynamicChunkingWorker``, forwards its two engine RPCs
and brackets its model call with ``start_sample`` / ``finish_sample``. Both the
startup sweep and serving are timed through that one bracket, so the overhead
fitted at startup and the samples calibrated at runtime measure the same window.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from atom.model_engine.dynamic_chunking import (
    MAX_CALIBRATION_FIT_FAILURES,
    MIN_CALIBRATION_CHUNK_SIZES,
    MIN_PROFILE_SAMPLES,
    ChunkLatencyCalibrator,
    chunk_alignment,
    fit_chunk_overhead,
    profile_chunk_grid,
)
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType

if TYPE_CHECKING:
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_engine.scheduler import ScheduledBatch

logger = logging.getLogger("atom")

# Maximum asynchronous timing samples in flight.
MAX_PENDING_CHUNK_SAMPLES = 8

# Discard one request's kernel and allocator warmup timings.
DISCARD_FIRST_CHUNK_REQUESTS = 1

# When calibration stops waiting. Sole-prefill requests sampled without ever
# seeing two chunk sizes, polls that see no sample at all, or simply this many
# requests without the spread of shapes a fit needs: each means the sweep will
# not converge, and leaving it up sizes every later sole prefill for a fit that
# will not arrive.
MAX_CALIBRATION_REQUESTS_WITHOUT_DIVERSITY = 8
MAX_CALIBRATION_POLLS_WITHOUT_SAMPLES = 16
MAX_CALIBRATION_REQUESTS = 32


class ChunkTimingSample(NamedTuple):
    """One prefill forward being timed by a pair of CUDA events."""

    start: torch.cuda.Event
    end: torch.cuda.Event
    req_id: int
    prefix_len: int
    chunk_size: int


class DynamicChunkingWorker:
    """Worker-side driver for the chunk latency model.

    Holds everything a ``ModelRunner`` does for dynamic chunking: the startup
    dummy sweep that fits the attention-free terms, the CUDA events that time
    real prefills, and the fit the engine polls for. The runner owns one of
    these unconditionally and it stays inert unless the feature is on.

    Both RPCs answer from TP rank 0 on every path that returns:
    ``call_func(wait_out=True)`` blocks on rank 0's reply with no timeout, so
    returning without a reply would hang the engine. Failures that leave every
    stage in step are answered as errors and degrade to fixed chunking; an
    exception that escapes takes the worker down, which fails startup or the
    engine instead of hanging it.
    """

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        self._calibrator: ChunkLatencyCalibrator | None = None
        self._pending: deque[ChunkTimingSample] = deque()
        self._free_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        # Collects the startup sweep's samples; None outside the sweep.
        self._sweep_samples: list[ChunkTimingSample] | None = None
        self._samples_seen = 0
        self._last_req_id: int | None = None
        self._requests_seen = 0
        self._calibration_polls = 0
        self._rng = np.random.default_rng(0)

    def _reply(self, result: dict) -> dict | None:
        return result if self._runner.rank == 0 else None

    def profile(self) -> dict | None:
        """Fit attention-free chunk overhead from startup dummy forwards.

        All workers run the sweep in lockstep; only the PP head's TP rank 0 keeps
        the result and calibrates attention terms from real prefills. A plan
        that cannot run, or a fit that fails, reports ``{"error": ...}`` so the
        engine serves fixed chunking.
        """
        runner = self._runner
        config = runner.config
        if not config.enable_dynamic_chunking or config.pipeline_parallel_size <= 1:
            return self._reply({"error": "dynamic chunking is off on this worker"})
        plan = self._sweep_plan()
        if isinstance(plan, str):
            result = {"error": plan}
        else:
            chunk_grid, vocab_size = plan
            logger.info(
                "%s: profiling dynamic chunking chunk overhead at %d sizes "
                "(max chunk=%d)",
                runner.label,
                len(chunk_grid),
                max(chunk_grid),
            )
            # Not caught: a forward that fails on one stage leaves its
            # neighbours blocked in the PP send/recv for good. Taking the
            # worker down fails startup instead, as a failed warmup forward
            # does.
            latencies_ms = self._sweep(chunk_grid, vocab_size)
            result = self._fit_overhead(chunk_grid, latencies_ms)
        if "error" in result:
            logger.warning(
                "%s: dynamic chunking profiling: %s", runner.label, result["error"]
            )
        return self._reply(result)

    def _sweep_plan(self) -> tuple[list[int], int] | str:
        """Return the sweep's chunk grid and vocab size, or why it cannot run.

        Built from config alone, so every stage reaches the same answer before
        any forward crosses stages.
        """
        runner = self._runner
        config = runner.config
        alignment = chunk_alignment(runner.block_size)
        # Stage-invariant, which the sweep's send/recv chain needs: a grid that
        # is shorter on one stage than its neighbour deadlocks it. And bounded
        # by what warmup sized activations for, because the KV pool already
        # holds the rest of HBM.
        max_chunk = runner.warmup_max_tokens()
        if config.max_model_len:
            max_chunk = min(max_chunk, config.max_model_len)
        max_chunk -= max_chunk % alignment
        chunk_grid = profile_chunk_grid(alignment, max_chunk)
        vocab_size = getattr(runner.hf_text_config, "vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            return f"profiling needs hf_text_config.vocab_size; got {vocab_size!r}"
        if len(chunk_grid) < MIN_PROFILE_SAMPLES:
            # Too little room to profile is a reason to serve with fixed
            # chunking, not to fail startup.
            return (
                f"needs at least {MIN_PROFILE_SAMPLES} aligned profiling "
                f"lengths, got {len(chunk_grid)} from max_chunk={max_chunk}, "
                f"alignment={alignment}"
            )
        return chunk_grid, vocab_size

    def _fit_overhead(self, chunk_grid: list[int], latencies_ms: list[float]) -> dict:
        """Fit b and c once every stage has finished the sweep."""
        runner = self._runner
        try:
            linear, constant = fit_chunk_overhead(chunk_grid, latencies_ms)
        except ValueError as exc:
            return {"error": f"chunk overhead fit failed: {exc}"}

        # Only the scheduling PP head collects runtime samples.
        samples_here = runner.config.parallel_config.pipeline_parallel_rank == 0
        logger.info(
            "%s: dynamic chunking chunk overhead b=%.3e c=%.3e; attention terms %s",
            runner.label,
            linear,
            constant,
            (
                "will be calibrated from this stage's real prefills"
                if samples_here
                else "come from the head stage"
            ),
        )
        if samples_here and runner.rank == 0:
            self._calibrator = ChunkLatencyCalibrator(linear, constant)
        return {"linear_coeff": linear, "constant_coeff": constant}

    def _sweep(self, chunk_grid: list[int], vocab_size: int) -> list[float]:
        """Time one dummy prefill per grid size, best of two, warmup excluded."""
        # Absorbs first-shape compilation and lazy communicator costs.
        self._timed_forward(chunk_grid[0], vocab_size)
        latencies_ms = [
            min(self._timed_forward(chunk_size, vocab_size) for _ in range(2))
            for chunk_size in chunk_grid
        ]
        self._runner.flush_pp_send()
        return latencies_ms

    def _timed_forward(self, chunk_size: int, vocab_size: int) -> float:
        """Run one dummy prefill and return the model call's GPU time in ms.

        The window is the ``start_sample`` / ``finish_sample`` bracket inside the
        runner, as for a served prefill: batch construction, input staging, the
        PP receive and the PP send all stay outside it.
        """
        runner = self._runner
        batch = self._dummy_batch(chunk_size, vocab_size)
        # Drain the previous send first, so the stage behind this one is not
        # still holding a send this forward has to commit.
        runner.flush_pp_send()
        self._sweep_samples = []
        try:
            runner.forward(batch)
            samples = self._sweep_samples
        finally:
            self._sweep_samples = None
        if len(samples) != 1:
            raise RuntimeError(
                f"startup sweep forward produced {len(samples)} timing samples"
            )
        sample = samples[0]
        sample.end.synchronize()
        self._free_events.append((sample.start, sample.end))
        return sample.start.elapsed_time(sample.end)

    def _dummy_batch(self, chunk_size: int, vocab_size: int) -> ScheduledBatch:
        # Imported here because `scheduler` imports `dynamic_chunking` at load.
        from atom.model_engine.scheduler import ScheduledBatch

        # Random tokens avoid an unrealistic single-expert MoE profile.
        tokens = self._rng.integers(
            0, vocab_size, size=chunk_size, dtype=np.int64
        ).tolist()
        seq = Sequence(tokens, block_size=self._runner.block_size, id=-2)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.PREFILL
        # No block table, as in `warmup_model`: a dummy forward never writes KV,
        # and fabricated block ids would have to fit every stage's own pool.
        return ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=[chunk_size],
            total_tokens_num=chunk_size,
            total_tokens_num_prefill=chunk_size,
            total_seqs_num=1,
            total_seqs_num_prefill=1,
            is_dummy_run=True,
            num_cached_tokens=[0],
            is_final_chunk=[False],
        )

    def take_fit(self) -> dict | None:
        """Return newly calibrated coefficients, or ``None`` if none are ready.

        TP rank 0 always returns a dict so the polling RPC receives a response.
        """
        runner = self._runner
        if runner.rank != 0:
            return None
        try:
            return self._take_fit()
        except Exception:
            logger.exception("%s: dynamic chunking calibration failed", runner.label)
            return self._give_up("calibration raised")

    def _take_fit(self) -> dict:
        calibrator = self._calibrator
        if calibrator is None:
            return {"coefficients": None}
        self._drain()
        self._calibration_polls += 1
        stalled = self._stall_reason(calibrator)
        if stalled is not None:
            return self._give_up(stalled)
        try:
            predictor = calibrator.maybe_fit()
        except ValueError as exc:
            logger.warning(
                "%s: dynamic chunking calibration: %s", self._runner.label, exc
            )
            if not calibrator.gave_up:
                return {"coefficients": None}
            return self._give_up(
                f"{MAX_CALIBRATION_FIT_FAILURES} rejected fits over "
                f"{calibrator.num_shapes} shapes"
            )
        if predictor is None:
            logger.info(
                "%s: dynamic chunking still calibrating: %d timed prefills over "
                "%d requests, %d shapes, %d distinct prefixes, %d chunk sizes",
                self._runner.label,
                self._samples_seen,
                self._requests_seen,
                calibrator.num_shapes,
                calibrator.num_prefixes,
                calibrator.num_chunk_sizes,
            )
            return {"coefficients": None}
        logger.info(
            "%s: dynamic chunking calibrated from %d real prefill shapes over "
            "%d chunk sizes: a=%.3e b=%.3e c=%.3e gamma=%.3e",
            self._runner.label,
            calibrator.num_shapes,
            calibrator.num_chunk_sizes,
            predictor.quadratic_coeff,
            predictor.linear_coeff,
            predictor.constant_coeff,
            predictor.prefix_coeff,
        )
        # Freeze the accepted fit and stop timing.
        self._calibrator = None
        self._pending.clear()
        return {
            "coefficients": (
                predictor.quadratic_coeff,
                predictor.linear_coeff,
                predictor.constant_coeff,
                predictor.prefix_coeff,
            ),
            "num_shapes": calibrator.num_shapes,
        }

    def _give_up(self, reason: str) -> dict:
        """Stop timing prefills and tell the engine to end the sweep."""
        logger.warning(
            "%s: giving up on dynamic chunking calibration (%s); chunking stays fixed",
            self._runner.label,
            reason,
        )
        self._calibrator = None
        self._pending.clear()
        return {"coefficients": None, "gave_up": True}

    def _stall_reason(self, calibrator: ChunkLatencyCalibrator) -> str | None:
        """Why more polls cannot produce a fit, or None while one still can."""
        requests = self._requests_seen
        if (
            requests >= MAX_CALIBRATION_REQUESTS_WITHOUT_DIVERSITY
            and calibrator.num_chunk_sizes < MIN_CALIBRATION_CHUNK_SIZES
        ):
            return f"{requests} sampled requests saw one chunk size"
        if (
            requests == 0
            and self._calibration_polls >= MAX_CALIBRATION_POLLS_WITHOUT_SAMPLES
        ):
            return f"no sole prefill sampled in {self._calibration_polls} polls"
        if requests >= MAX_CALIBRATION_REQUESTS:
            return (
                f"{requests} sampled requests left {calibrator.num_shapes} shapes "
                f"over {calibrator.num_prefixes} prefixes"
            )
        return None

    def start_sample(self, batch: ScheduledBatch | None) -> ChunkTimingSample | None:
        """Start timing ``batch`` if it is a single-request prefill worth a sample.

        Records the start event, so the caller must pair a non-``None`` return
        with ``finish_sample`` around the model call and nothing else. Dummy
        forwards are timed only by the startup sweep.
        """
        if batch is None:
            return None
        if batch.is_dummy_run:
            if self._sweep_samples is None:
                return None
        elif (
            self._calibrator is None or len(self._pending) >= MAX_PENDING_CHUNK_SAMPLES
        ):
            return None
        if batch.total_seqs_num != 1 or batch.total_seqs_num_prefill != 1:
            return None
        if self._free_events:
            start, end = self._free_events.pop()
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        start.record()
        return ChunkTimingSample(
            start,
            end,
            int(batch.req_ids[0]),
            int(batch.num_cached_tokens[0]),
            int(batch.num_scheduled_tokens[0]),
        )

    def finish_sample(self, sample: ChunkTimingSample | None) -> None:
        if sample is None:
            return
        sample.end.record()
        if self._sweep_samples is not None:
            self._sweep_samples.append(sample)
            return
        self._pending.append(sample)
        self._drain()

    def _drain(self) -> None:
        """Pass completed event timings to the calibrator without synchronizing."""
        if self._calibrator is None:
            return
        pending = self._pending
        while pending and pending[0].end.query():
            sample = pending.popleft()
            self._samples_seen += 1
            # By request id: a prefix cannot tell requests apart, since a
            # multi-turn follow-up can start on a cache hit deeper than where
            # the previous turn's last chunk began.
            if sample.req_id != self._last_req_id:
                self._requests_seen += 1
                self._last_req_id = sample.req_id
            if self._requests_seen > DISCARD_FIRST_CHUNK_REQUESTS:
                self._calibrator.add(
                    sample.prefix_len,
                    sample.chunk_size,
                    sample.start.elapsed_time(sample.end),
                )
            self._free_events.append((sample.start, sample.end))
