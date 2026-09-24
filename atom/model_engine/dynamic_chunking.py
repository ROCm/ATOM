# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dynamic chunked-prefill latency calibration and prediction.

A chunk of ``x`` tokens after a cached prefix of ``L`` tokens is modeled as

    t(L, x) = c + gamma * L + b * x + a * (2 * L * x + x**2)

Startup dummy forwards fit ``b``; a two-size runtime prefill sweep fits
``a``, ``gamma`` and ``c``. The accepted model then selects equal-latency chunks.

``DynamicChunkingWorker`` at the bottom of this module is the worker half of the
feature: it drives the startup sweep, times real prefills and hands the fitted
model back, so ``ModelRunner`` only forwards its two RPCs and brackets the model
call.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    SequenceType,
    new_block_table,
)

if TYPE_CHECKING:
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_engine.scheduler import ScheduledBatch

logger = logging.getLogger("atom")

# Maximum share of chunk latency spent rebuilding the cached prefix.
MAX_PREFIX_OVERHEAD_FRACTION = 0.2

MIN_PROFILE_SAMPLES = 8

# Span of the startup profiling sweep: it runs from the token budget down to
# this fraction of it.
PROFILE_SWEEP_RATIO = 8

# Sizes visited by the startup sweep before alignment collapses duplicates.
PROFILE_SWEEP_POINTS = 24

# Maximum asynchronous timing samples in flight.
MAX_PENDING_CHUNK_SAMPLES = 8

# Discard one request's kernel and allocator warmup timings.
DISCARD_FIRST_CHUNK_REQUESTS = 1

# Minimum sample diversity required for an identifiable runtime fit.
MIN_CALIBRATION_PREFIXES = 3
MIN_CALIBRATION_CHUNK_SIZES = 2
MIN_CALIBRATION_SHAPES = MIN_CALIBRATION_PREFIXES * MIN_CALIBRATION_CHUNK_SIZES

# Separation between the two calibration chunk sizes.
CALIBRATION_SWEEP_RATIO = 4

MAX_CALIBRATION_SHAPES = 512
MAX_CALIBRATION_TIMINGS_PER_SHAPE = 4

# Fit-quality gates.
MAX_CALIBRATION_RESIDUAL_FRACTION = 0.25
MAX_CALIBRATION_DESIGN_CONDITION = 100.0
MAX_CALIBRATION_PREDICTION_STDERR_FRACTION = 0.05

# Rejected fits tolerated before calibration gives up. Timing noise on a busy
# server can keep every fit outside the gates, and retrying for the life of the
# process would leave the scheduler's sweep sizing chunks for a model that is
# never going to arrive.
MAX_CALIBRATION_FIT_FAILURES = 8

# Sole-prefill requests sampled without ever seeing two chunk sizes, or polls
# that see no sample at all. Either means the sweep cannot converge, and
# leaving it up sizes every later sole prefill for a fit that will not arrive.
MAX_CALIBRATION_REQUESTS_WITHOUT_DIVERSITY = 8
MAX_CALIBRATION_POLLS_WITHOUT_SAMPLES = 16

# Engine steps between checks for a newly calibrated chunk latency model. The
# workers do the timing and the fitting; this only paces the RPC that collects
# the result, which is why it can be this frequent without costing anything.
DYNAMIC_CHUNKING_POLL_STEPS = 32

# Ignore models that shrink the reference chunk by less than this fraction.
MIN_USEFUL_SHRINK_FRACTION = 0.05


def has_sole_prefill(sources: int, recent_sources: Iterable[int] = ()) -> bool:
    """Whether one request has been the pipeline's only prefill work recently.

    ``sources`` counts every request with prefill left to do, including the one
    being chunked, so 1 is the sole-prefill case.
    """
    return max((sources, *recent_sources)) <= 1


def _attention_area(prefix_len: Any, chunk_size: Any) -> Any:
    """Attention work of a chunk: its own tokens plus the prefix it re-reads.

    Scalars or arrays; the fit and the model it produces both measure a chunk
    through this one expression.
    """
    return 2.0 * prefix_len * chunk_size + chunk_size * chunk_size


def _design_matrix(
    chunks: np.ndarray,
    prefixes: np.ndarray,
    *,
    with_prefix: bool,
    with_constant: bool,
) -> np.ndarray:
    """Latency-model columns, in coefficient order ``(a, gamma, c)``.

    Built here rather than at each call site because the fit and the gates that
    judge its uncertainty have to describe the same model: a column present in
    one and not the other silently changes the degrees of freedom.
    """
    columns = [_attention_area(prefixes, chunks)]
    if with_prefix:
        columns.append(prefixes)
    if with_constant:
        columns.append(np.ones_like(chunks))
    return np.column_stack(columns)


def _scale_columns(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each column to unit maximum, with the scales applied.

    The columns span many orders of magnitude - attention area is O(1e10) next to
    a constant column of ones - and an unscaled solve reports rank deficiency on
    data that is perfectly well conditioned once normalized.
    """
    scales = np.max(np.abs(design), axis=0)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("Dynamic chunking latency samples have a degenerate column")
    return design / scales, scales


def _scaled_lstsq(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares on scaled columns, returning unscaled coefficients."""
    scaled, scales = _scale_columns(design)
    try:
        solution, _, rank, _ = np.linalg.lstsq(scaled, target, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Failed to fit dynamic chunking latency model") from exc
    if rank < design.shape[1]:
        raise ValueError("Dynamic chunking latency samples are rank deficient")
    return solution / scales


def fit_chunk_overhead(
    chunk_sizes: list[int], latencies_ms: list[float]
) -> tuple[float, float]:
    """Fit ``(b, c)`` of ``t(x) = c + b * x`` from a dummy chunk sweep.

    Dummy forwards bypass attention, so this is the whole of what they measure
    and the two attention coefficients are left to runtime calibration.
    """
    if len(chunk_sizes) != len(latencies_ms):
        raise ValueError("chunk_sizes and latencies_ms must have equal length")
    if len(chunk_sizes) < MIN_PROFILE_SAMPLES:
        raise ValueError(
            f"Dynamic chunking needs at least {MIN_PROFILE_SAMPLES} latency "
            "profiling samples"
        )

    chunks = np.asarray(chunk_sizes, dtype=np.float64)
    latencies = np.asarray(latencies_ms, dtype=np.float64)
    for name, values in (("chunk_sizes", chunks), ("latencies_ms", latencies)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Dynamic chunking {name} must be finite")
    if np.unique(chunks).size < 2:
        raise ValueError(
            "Dynamic chunking needs at least 2 distinct profiling chunk sizes"
        )

    design = np.column_stack((chunks, np.ones_like(chunks)))
    linear, constant = (float(value) for value in _scaled_lstsq(design, latencies))
    if linear <= 0:
        # A non-positive slope means the sweep never left the plateau where
        # per-forward overhead dominates. Calibration would then read the whole
        # chunk cost as attention, so refuse the baseline instead.
        raise ValueError(
            "Dynamic chunking requires a positive linear latency coefficient, got "
            f"b={linear:.3e}; the profiling window is dominated by per-forward "
            "overhead"
        )
    return linear, max(constant, 0.0)


class _LatencyTerms(NamedTuple):
    """Fitted ``(a, gamma, c)`` and which columns produced them."""

    quadratic: float
    prefix: float
    constant: float
    with_prefix: bool
    free_constant: bool


@dataclass
class ChunkLatencyCalibrator:
    """Fit the attention terms of the chunk latency model from real prefills.

    ``b`` is taken from startup profiling and held fixed; what is fitted here is
    the part dummy forwards cannot see:

        measured - b * x = a * (2 * L * x + x**2) + gamma * L + c

    Three unknowns and linear in all of them. A request being chunked walks ``L``
    from 0 to its prompt length, and the scheduler's calibration sweep serves the
    first few requests at two widely separated chunk sizes, which is what pulls
    the area term, the per-chunk prefix rebuild and the per-forward overhead
    apart. One sample at ``L = 0`` anchors ``a``.

    Timings are collected per ``(chunk, prefix)`` shape and reduced by median, so
    a straggler forward moves the fit far less than it moves any single sample.
    """

    linear_coeff: float
    constant_coeff: float
    _timings: dict[tuple[int, int], deque[float]] = field(default_factory=dict)
    _since_fit: int = 0
    _failed_fits: int = 0

    def add(self, prefix_len: int, chunk_size: int, elapsed_ms: float) -> None:
        """Record one real prefill forward."""
        if chunk_size <= 0 or prefix_len < 0 or not math.isfinite(elapsed_ms):
            return
        if elapsed_ms <= 0.0:
            return
        shape = (int(chunk_size), int(prefix_len))
        timings = self._timings.get(shape)
        if timings is None:
            if len(self._timings) >= MAX_CALIBRATION_SHAPES:
                return
            timings = deque(maxlen=MAX_CALIBRATION_TIMINGS_PER_SHAPE)
            self._timings[shape] = timings
        timings.append(float(elapsed_ms))
        # A fresh shape improves conditioning; a repeat refines its median after
        # an uncertainty rejection. Polling still cannot refit unchanged data.
        self._since_fit += 1

    @property
    def num_shapes(self) -> int:
        return len(self._timings)

    @property
    def num_prefixes(self) -> int:
        return len({prefix for _, prefix in self._timings})

    @property
    def num_chunk_sizes(self) -> int:
        return len({chunk for chunk, _ in self._timings})

    @property
    def num_failed_fits(self) -> int:
        return self._failed_fits

    @property
    def gave_up(self) -> bool:
        """Whether this workload has rejected enough fits to stop trying."""
        return self._failed_fits >= MAX_CALIBRATION_FIT_FAILURES

    def _is_due(self) -> bool:
        if self.num_shapes < MIN_CALIBRATION_SHAPES:
            return False
        if self.num_prefixes < MIN_CALIBRATION_PREFIXES:
            return False
        if self.num_chunk_sizes < MIN_CALIBRATION_CHUNK_SIZES:
            return False
        # A fit is attempted once per timing the last attempt did not see, so a
        # failure costs one retry per new measurement rather than one per poll.
        return self._since_fit > 0

    def maybe_fit(self) -> ChunkSizePredictor | None:
        """Fit if the samples can support one, else ``None``.

        Raises ``ValueError`` when the samples are present but unusable, so the
        caller can log why calibration is not converging. Rejections are counted
        against ``MAX_CALIBRATION_FIT_FAILURES``, after which ``gave_up`` tells
        the caller to stop sampling and leave chunking fixed.
        """
        if not self._is_due():
            return None
        self._since_fit = 0
        try:
            return self.fit()
        except ValueError:
            self._failed_fits += 1
            raise

    def _fit_terms(
        self, chunks: np.ndarray, prefixes: np.ndarray, latencies: np.ndarray
    ) -> _LatencyTerms:
        """Solve for ``(a, gamma, c)``, keeping only ``b`` from startup profiling.

        A dummy forward does the same per-token arithmetic serving does, so ``b``
        is the term it measures honestly. It never sets attention up, though, so
        its constant is not the one a real chunk pays, and pinning it makes the
        prefix rebuild absorb the difference - the term the whole feature turns
        on. ``MIN_CALIBRATION_CHUNK_SIZES`` is what makes fitting it instead
        possible, so this runs on sweep samples by construction.
        """
        overhead = latencies - self.linear_coeff * chunks

        def solve(free_constant: bool, with_prefix: bool) -> _LatencyTerms:
            design = _design_matrix(
                chunks, prefixes, with_prefix=with_prefix, with_constant=free_constant
            )
            target = overhead if free_constant else overhead - self.constant_coeff
            values = [float(value) for value in _scaled_lstsq(design, target)]
            return _LatencyTerms(
                quadratic=values[0],
                prefix=values[1] if with_prefix else 0.0,
                constant=values[-1] if free_constant else self.constant_coeff,
                with_prefix=with_prefix,
                free_constant=free_constant,
            )

        terms = solve(True, True)
        if terms.constant < 0.0:
            # A forward that costs less than nothing to launch is not physical,
            # and a negative constant drags the terms fitted beside it.
            terms = solve(False, True)
        if terms.prefix < 0.0:
            # Refit rather than clamp: dropping the prefix column leaves the
            # whole prefix cost in the area term, where a clamp would have left
            # `a` carrying a negative partner's bias instead.
            terms = solve(terms.free_constant, False)
        return terms

    def _validate_fit_quality(
        self, *, design: np.ndarray, error: np.ndarray, mean_latency: float
    ) -> None:
        """Reject identifiable but noise-sensitive fits.

        ``design`` must be the matrix the accepted fit was solved on, so that the
        uncertainty below is the uncertainty of the model being installed.
        """
        scaled, _ = _scale_columns(design)

        condition = float(np.linalg.cond(scaled))
        if not math.isfinite(condition) or condition > MAX_CALIBRATION_DESIGN_CONDITION:
            raise ValueError(
                "Dynamic chunking calibration design condition is "
                f"{condition:.1f}, above the "
                f"{MAX_CALIBRATION_DESIGN_CONDITION:.0f} bound: sampled chunk "
                "sizes do not separate the latency terms"
            )

        degrees_of_freedom = design.shape[0] - design.shape[1]
        if degrees_of_freedom <= 0:
            raise ValueError(
                "Dynamic chunking calibration has too few samples to estimate "
                "fit uncertainty"
            )
        residual_variance = float(error @ error) / degrees_of_freedom
        covariance = residual_variance * np.linalg.inv(scaled.T @ scaled)
        prediction_variance = np.einsum("ij,jk,ik->i", scaled, covariance, scaled)
        max_stderr = float(np.sqrt(np.maximum(prediction_variance, 0.0)).max())
        uncertainty = max_stderr / mean_latency
        if uncertainty > MAX_CALIBRATION_PREDICTION_STDERR_FRACTION:
            raise ValueError(
                "Dynamic chunking calibration prediction uncertainty is "
                f"{uncertainty:.1%}, above the "
                f"{MAX_CALIBRATION_PREDICTION_STDERR_FRACTION:.0%} bound: "
                "more stable timing samples are required"
            )

    def fit(self) -> ChunkSizePredictor:
        shapes = sorted(self._timings)
        chunks = np.asarray([chunk for chunk, _ in shapes], dtype=np.float64)
        prefixes = np.asarray([prefix for _, prefix in shapes], dtype=np.float64)
        latencies = np.asarray(
            [float(np.median(self._timings[shape])) for shape in shapes],
            dtype=np.float64,
        )

        terms = self._fit_terms(chunks, prefixes, latencies)
        attention_span = terms.quadratic * float(
            np.ptp(_attention_area(prefixes, chunks))
        )
        roundoff = (
            64.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(latencies))), 1.0)
        )
        if attention_span <= roundoff:
            raise ValueError(
                "Dynamic chunking calibration measured no attention growth "
                f"(a={terms.quadratic:.3e}): chunk cost does not rise with "
                "attention area, so equal-latency chunking has nothing to equalize"
            )

        predictor = ChunkSizePredictor(
            terms.quadratic, self.linear_coeff, terms.constant, terms.prefix
        )
        modeled = np.asarray(
            [
                predictor.predicted_latency(int(prefix_len), int(chunk))
                for chunk, prefix_len in shapes
            ],
            dtype=np.float64,
        )
        error = modeled - latencies
        rms = float(np.sqrt(np.mean(np.square(error))))
        mean = float(np.mean(latencies))
        if rms > MAX_CALIBRATION_RESIDUAL_FRACTION * mean:
            raise ValueError(
                f"Dynamic chunking calibration residual is {rms:.1f}ms against a "
                f"{mean:.1f}ms mean, above the "
                f"{MAX_CALIBRATION_RESIDUAL_FRACTION:.0%} bound: the samples are "
                "not described by the chunk latency model"
            )
        self._validate_fit_quality(
            design=_design_matrix(
                chunks,
                prefixes,
                with_prefix=terms.with_prefix,
                with_constant=terms.free_constant,
            ),
            error=error,
            mean_latency=mean,
        )
        return predictor


@dataclass(frozen=True)
class ChunkSizePredictor:
    """Predict equal-latency prefill chunks with a prefix-rebuild floor."""

    quadratic_coeff: float
    linear_coeff: float
    constant_coeff: float
    prefix_coeff: float = 0.0

    @classmethod
    def from_coefficients(
        cls, coefficients: tuple[float, ...] | list[float]
    ) -> ChunkSizePredictor:
        if len(coefficients) not in (3, 4):
            raise ValueError(
                "Dynamic chunking requires three or four coefficients "
                "(a, b, c[, gamma])"
            )
        predictor = cls(*(float(value) for value in coefficients))
        if predictor.quadratic_coeff <= 0:
            raise ValueError("Dynamic chunking quadratic coefficient must be positive")
        if predictor.linear_coeff < 0:
            raise ValueError("Dynamic chunking linear coefficient must be non-negative")
        if predictor.prefix_coeff < 0:
            raise ValueError("Dynamic chunking prefix coefficient must be non-negative")
        return predictor

    def target_latency(self, base_chunk_size: int) -> float:
        """Runtime of the initial chunk, with prefix and constant terms removed."""
        return (
            self.quadratic_coeff * base_chunk_size * base_chunk_size
            + self.linear_coeff * base_chunk_size
        )

    def chunk_latency(self, history_len: int, chunk_size: int) -> float:
        """Modeled attention-area runtime of ``chunk_size`` tokens after a prefix."""
        return (
            self.quadratic_coeff * _attention_area(history_len, chunk_size)
            + self.linear_coeff * chunk_size
        )

    def predicts_useful_shrink(self, *, base_chunk_size: int, history_len: int) -> bool:
        """Whether the model sees enough prefix growth to be worth acting on.

        Callers use this to keep fixed-size chunking when profiling produced a
        model that is technically valid but flat in the prefix - see
        ``MIN_USEFUL_SHRINK_FRACTION``.
        """
        raw = self.equal_latency_chunk(history_len, base_chunk_size)
        if not math.isfinite(raw) or raw <= 0:
            return False
        return raw <= base_chunk_size * (1.0 - MIN_USEFUL_SHRINK_FRACTION)

    def predicted_latency(self, history_len: int, chunk_size: int) -> float:
        """Full modeled runtime of one chunk, overheads included."""
        return (
            self.constant_coeff
            + self.prefix_coeff * history_len
            + self.chunk_latency(history_len, chunk_size)
        )

    def _solve_chunk(self, history_len: int, target: float) -> float:
        """Smallest ``x`` with ``chunk_latency(history_len, x) == target``."""
        a = self.quadratic_coeff
        b = 2.0 * a * history_len + self.linear_coeff
        return (-b + math.sqrt(b * b + 4.0 * a * target)) / (2.0 * a)

    def equal_latency_chunk(self, history_len: int, base_chunk_size: int) -> float:
        """Chunk after ``history_len`` tokens that costs as much as the first one.

        The budget is the initial chunk's runtime *minus* the prefix rebuild this
        chunk owes, because ``gamma * L`` is a floor the chunk pays before any of
        its own tokens are attended to. Equalizing only the attention-area terms
        instead - as a model without ``gamma`` has to - leaves every chunk paying
        that floor on top of an already-equal budget, so the later chunks come
        out both too large to be equal-latency and too numerous.

        Returns ``nan`` when the floor alone exceeds the budget: no chunk size
        matches the first chunk's runtime. ``predict`` turns that into ``None``,
        and the scheduler keeps the fixed chunk instead of shrinking further —
        a smaller chunk would pay this floor again on the next forward.
        """
        target = self.target_latency(base_chunk_size) - self.prefix_coeff * history_len
        if target <= 0.0:
            return math.nan
        return self._solve_chunk(history_len, target)

    def prefix_bounded_chunk(self, history_len: int) -> float:
        """Smallest chunk whose prefix rebuild stays inside its overhead budget."""
        if self.prefix_coeff <= 0.0 or history_len <= 0:
            return 0.0
        fraction = MAX_PREFIX_OVERHEAD_FRACTION
        overhead = self.prefix_coeff * history_len
        return self._solve_chunk(history_len, overhead * (1.0 - fraction) / fraction)

    def predict(
        self,
        *,
        history_len: int,
        base_chunk_size: int,
        smooth_factor: float,
        alignment: int,
        max_chunk_size: int,
        min_chunk_size: int,
    ) -> int | None:
        """Solve for the equal-latency chunk and apply serving constraints."""
        if history_len < 0:
            raise ValueError("history_len must be non-negative")
        if base_chunk_size <= 0 or alignment <= 0 or max_chunk_size <= 0:
            raise ValueError("Chunk sizes and alignment must be positive")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if not 0.0 <= smooth_factor <= 1.0:
            raise ValueError("smooth_factor must be in [0, 1]")

        raw = self.equal_latency_chunk(history_len, base_chunk_size)
        if not math.isfinite(raw) or raw <= 0:
            return None

        smoothed = base_chunk_size + smooth_factor * (raw - base_chunk_size)
        lower_bound = max(
            alignment,
            min_chunk_size,
            int(self.prefix_bounded_chunk(history_len)),
        )
        constrained = min(
            max(int(smoothed), lower_bound),
            base_chunk_size,
            max_chunk_size,
        )
        aligned = constrained - constrained % alignment
        return aligned if aligned >= alignment else None


def profile_chunk_grid(alignment: int, max_chunk: int) -> list[int]:
    """Return aligned startup profiling sizes spanning the sweep ratio."""

    def align(value: int) -> int:
        return max(alignment, int(value) // alignment * alignment)

    chunk_floor = max(alignment, max_chunk // PROFILE_SWEEP_RATIO)
    return list(
        dict.fromkeys(
            align(chunk)
            for chunk in np.linspace(
                max_chunk, chunk_floor, num=PROFILE_SWEEP_POINTS, dtype=np.int64
            )
        )
    )


class ChunkTimingSample(NamedTuple):
    """One prefill forward being timed by a pair of CUDA events."""

    start: torch.cuda.Event
    end: torch.cuda.Event
    prefix_len: int
    chunk_size: int


class DynamicChunkingWorker:
    """Worker-side driver for the chunk latency model.

    Holds everything a ``ModelRunner`` does for dynamic chunking: the startup
    dummy sweep that fits the attention-free terms, the CUDA events that time
    real prefills, and the fit the engine polls for. The runner owns one of
    these unconditionally and it stays inert unless the feature is on.
    """

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        self._calibrator: ChunkLatencyCalibrator | None = None
        self._pending: deque[ChunkTimingSample] = deque()
        self._free_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._samples_seen = 0
        self._last_prefix = -1
        self._requests_seen = 0
        self._calibration_polls = 0
        self._rng = np.random.default_rng(0)

    def profile(self) -> dict | None:
        """Fit attention-free chunk overhead from startup dummy forwards.

        All workers run the sweep in lockstep; only the PP head's TP rank 0 keeps
        the result and calibrates attention terms from real prefills.
        """
        runner = self._runner
        config = runner.config
        if not config.enable_dynamic_chunking or config.pipeline_parallel_size <= 1:
            return None

        alignment = max(runner.block_size, 64)
        # Stage-invariant. `num_kvcache_blocks` differs per PP stage (layer
        # count and free memory), and a grid that is shorter on one stage than
        # its neighbour deadlocks the sweep's send/recv before the ready signal.
        max_chunk = config.max_num_batched_tokens
        if config.max_model_len:
            max_chunk = min(max_chunk, config.max_model_len)
        max_chunk -= max_chunk % alignment
        chunk_grid = profile_chunk_grid(alignment, max_chunk)
        vocab_size = getattr(
            getattr(runner, "hf_text_config", None), "vocab_size", None
        )
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            # Multimodal configs nest the text vocab under `text_config`.
            # `hf_config.vocab_size` is missing there, and raising would kill
            # the worker the way a bad block table used to.
            reason = (
                "Dynamic chunking profiling needs hf_text_config.vocab_size; "
                f"got {vocab_size!r}"
            )
            logger.warning("%s: %s", runner.label, reason)
            return {"error": reason} if runner.rank == 0 else None
        self._profile_vocab_size = vocab_size

        if len(chunk_grid) < MIN_PROFILE_SAMPLES:
            # Report it like a failed fit rather than raising: too little room to
            # profile is a reason to serve with fixed chunking, not to fail startup.
            reason = (
                f"Dynamic chunking needs at least {MIN_PROFILE_SAMPLES} aligned "
                f"profiling lengths, got {len(chunk_grid)} from "
                f"max_chunk={max_chunk}, alignment={alignment}"
            )
            logger.warning("%s: %s", runner.label, reason)
            return {"error": reason} if runner.rank == 0 else None

        logger.info(
            "%s: profiling dynamic chunking chunk overhead at %d sizes (max chunk=%d)",
            runner.label,
            len(chunk_grid),
            max_chunk,
        )
        latencies_ms = self._sweep(chunk_grid)

        try:
            linear, constant = fit_chunk_overhead(chunk_grid, latencies_ms)
        except ValueError as exc:
            logger.warning(
                "%s: dynamic chunking chunk overhead fit failed: %s", runner.label, exc
            )
            return {"error": str(exc)} if runner.rank == 0 else None

        # Only the scheduling PP head collects runtime samples.
        pp_rank = config.parallel_config.pipeline_parallel_rank
        samples_here = pp_rank == 0
        logger.info(
            "%s: dynamic chunking chunk overhead b=%.3e c=%.3e on PP stage %d; "
            "attention terms %s",
            runner.label,
            linear,
            constant,
            pp_rank,
            (
                "will be calibrated from this stage's real prefills"
                if samples_here
                else "come from the head stage"
            ),
        )
        if runner.rank != 0:
            # Other TP ranks ran duplicate shapes.
            return None
        if samples_here:
            self._calibrator = ChunkLatencyCalibrator(linear, constant)
        return {"linear_coeff": linear, "constant_coeff": constant}

    def _sweep(self, chunk_grid: list[int]) -> list[float]:
        """Time one dummy prefill per grid size, warmup excluded."""
        runner = self._runner

        def run(chunk_size: int) -> None:
            # Time `forward` only. `flush_pp_send` waits until the next stage
            # posts its recv, so folding it into the window puts downstream
            # back-pressure into `b`. Runtime samples stop at `finish_sample`,
            # before `commit_pp_send_work`, and the two windows have to match.
            runner.forward(self._dummy_batch(chunk_size))

        # Absorbs first-shape compilation and lazy communicator costs.
        run(chunk_grid[0])
        torch.cuda.synchronize(runner.device)

        latencies_ms: list[float] = []
        for chunk_size in chunk_grid:
            # Best of two limits straggler bias.
            best_ms = math.inf
            for _ in range(2):
                # Drain the previous send outside the window, so this forward's
                # async send is not waiting on the stage behind it.
                runner.flush_pp_send()
                torch.cuda.synchronize(runner.device)
                start = time.perf_counter()
                run(chunk_size)
                torch.cuda.synchronize(runner.device)
                best_ms = min(best_ms, (time.perf_counter() - start) * 1000.0)
            latencies_ms.append(best_ms)
        runner.flush_pp_send()
        return latencies_ms

    def _dummy_batch(self, chunk_size: int) -> ScheduledBatch:
        # Imported here because `scheduler` imports this module at load time.
        from atom.model_engine.scheduler import ScheduledBatch

        runner = self._runner
        block_size = runner.block_size
        # Random tokens avoid an unrealistic single-expert MoE profile.
        vocab_size = getattr(self, "_profile_vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            vocab_size = getattr(
                getattr(runner, "hf_text_config", None), "vocab_size", None
            )
        tokens = self._rng.integers(
            0, int(vocab_size), size=chunk_size, dtype=np.int64
        ).tolist()
        seq = Sequence(tokens, block_size=block_size, id=-2)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.PREFILL
        # `new_block_table`, not a list: every forward marshals block tables
        # into the int32 `block_tables` buffer through `pack_rows`, which needs
        # a row that exports a buffer and raises TypeError on a list.
        seq.block_table = new_block_table(range(math.ceil(chunk_size / block_size)))
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
        calibrator = self._calibrator
        if calibrator is None:
            return {"coefficients": None}
        self._drain()
        self._calibration_polls += 1
        if self._calibration_stalled(calibrator):
            # Not a rejected fit: the sweep never produced two chunk sizes, or
            # no sole prefill was ever sampled. Either way more polls will not
            # grow a design matrix, so stop timing and leave chunking fixed.
            logger.warning(
                "%s: giving up on dynamic chunking calibration with %d sampled "
                "requests, %d chunk sizes, after %d polls; chunking stays fixed",
                runner.label,
                self._requests_seen,
                calibrator.num_chunk_sizes,
                self._calibration_polls,
            )
            self._calibrator = None
            return {"coefficients": None, "gave_up": True}
        try:
            predictor = calibrator.maybe_fit()
        except ValueError as exc:
            logger.warning("%s: dynamic chunking calibration: %s", runner.label, exc)
            if not calibrator.gave_up:
                return {"coefficients": None}
            logger.warning(
                "%s: giving up on dynamic chunking calibration after %d rejected "
                "fits over %d shapes; chunking stays fixed",
                runner.label,
                MAX_CALIBRATION_FIT_FAILURES,
                calibrator.num_shapes,
            )
            # Stop timing prefills: more of the same samples cannot pass the gates.
            self._calibrator = None
            return {"coefficients": None, "gave_up": True}
        if predictor is None:
            logger.info(
                "%s: dynamic chunking still calibrating: %d timed prefills, "
                "%d shapes, %d distinct prefixes, %d chunk sizes",
                runner.label,
                self._samples_seen,
                calibrator.num_shapes,
                calibrator.num_prefixes,
                calibrator.num_chunk_sizes,
            )
            return {"coefficients": None}
        logger.info(
            "%s: dynamic chunking calibrated from %d real prefill shapes over "
            "%d chunk sizes: a=%.3e b=%.3e c=%.3e gamma=%.3e",
            runner.label,
            calibrator.num_shapes,
            calibrator.num_chunk_sizes,
            predictor.quadratic_coeff,
            predictor.linear_coeff,
            predictor.constant_coeff,
            predictor.prefix_coeff,
        )
        # Freeze the accepted fit and stop timing.
        self._calibrator = None
        return {
            "coefficients": (
                predictor.quadratic_coeff,
                predictor.linear_coeff,
                predictor.constant_coeff,
                predictor.prefix_coeff,
            ),
            "num_shapes": calibrator.num_shapes,
        }

    def _calibration_stalled(self, calibrator: ChunkLatencyCalibrator) -> bool:
        if (
            self._requests_seen >= MAX_CALIBRATION_REQUESTS_WITHOUT_DIVERSITY
            and calibrator.num_chunk_sizes < MIN_CALIBRATION_CHUNK_SIZES
        ):
            return True
        return (
            self._requests_seen == 0
            and self._calibration_polls >= MAX_CALIBRATION_POLLS_WITHOUT_SAMPLES
        )

    def start_sample(self, batch: ScheduledBatch | None) -> ChunkTimingSample | None:
        """Start timing ``batch`` if it is a single-request prefill worth a sample.

        Records the start event, so the caller must pair a non-``None`` return
        with ``finish_sample`` around the model call and nothing else.
        """
        if self._calibrator is None or batch is None or batch.is_dummy_run:
            return None
        if batch.total_seqs_num != 1 or batch.total_seqs_num_prefill != 1:
            return None
        if len(self._pending) >= MAX_PENDING_CHUNK_SAMPLES:
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
            int(batch.num_cached_tokens[0]),
            int(batch.num_scheduled_tokens[0]),
        )

    def finish_sample(self, sample: ChunkTimingSample | None) -> None:
        if sample is None:
            return
        sample.end.record()
        self._pending.append(sample)
        self._drain()

    def _drain(self) -> None:
        """Pass completed event timings to the calibrator without synchronizing."""
        if self._calibrator is None:
            return
        pending = self._pending
        while pending and pending[0].end.query():
            sample = pending.popleft()
            prefix = sample.prefix_len
            self._samples_seen += 1
            # A non-growing prefix starts a new request, including cache hits.
            if self._last_prefix < 0 or prefix <= self._last_prefix:
                self._requests_seen += 1
            self._last_prefix = prefix
            if self._requests_seen > DISCARD_FIRST_CHUNK_REQUESTS:
                self._calibrator.add(
                    prefix, sample.chunk_size, sample.start.elapsed_time(sample.end)
                )
            self._free_events.append((sample.start, sample.end))
