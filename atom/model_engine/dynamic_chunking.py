# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Latency model used by dynamic chunked prefill.

A prefill chunk of ``x`` fresh tokens appended after ``L`` already cached
tokens is modeled as

    t(L, x) = c + gamma * L + b * x + a * (2 * L * x + x**2)

The ``a``/``b``/``c`` part is the increment of the cumulative runtime curve
``f(L) = a L^2 + b L + c`` used by SGLang's dynamic chunking, i.e. it only
depends on how much *attention area* the chunk covers.

``gamma * L`` is the extra per-chunk term this implementation adds. Backends
that keep a compressed KV cache (MLA/DeepSeek-style latent KV) rebuild the
whole cached prefix on every chunk, so the cost of a chunk grows with the
prefix even when the chunk itself stays small. That work is paid once per
chunk, so it is invariant to ``x`` and invisible to a model fitted only on
prefix-free prefills - which is what makes an equal-latency solver shrink
chunks far past the point where the extra chunks pay for themselves.

``gamma`` is therefore what decides whether the feature helps, and it is also
the coefficient startup profiling gets most wrong: dummy batches carry no real
cached prefix, so a fit taken from them underestimates ``gamma`` by more than an
order of magnitude on MLA models and predicts chunk sizes that do not shrink.
Coefficients fitted from real requests are supplied through
``--dynamic-chunking-calibration``; see
``docs/dynamic_chunked_pipeline_parallelism.md`` for how to collect them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# A chunk whose prefix rebuild costs more than this fraction of its total
# modeled runtime is mostly redoing old work, so chunk sizes are floored
# before that point even if the equal-latency objective asks for less.
MAX_PREFIX_OVERHEAD_FRACTION = 0.2

MIN_PROFILE_SAMPLES = 8

# Equal-latency chunking only pays off when a chunk gets measurably more
# expensive as the prefix in front of it grows. A model that predicts the same
# chunk at a long prefix as at none has not measured that growth, and acting on
# it can only add chunks: same forwards, more prefix rebuilt. Require the
# predicted chunk at the profiled prefix to be at least this much smaller than
# the initial chunk before dynamic chunking is allowed to run at all.
MIN_USEFUL_SHRINK_FRACTION = 0.05

# Concurrent prefills at which the equal-latency solver stops being useful.
#
# Rebalancing one request's chunks changes how well the pipeline is occupied
# only while that request is its only source of prefill microbatches. A second
# prefilling request interleaves its own chunks into the stages, so the fill and
# drain cost is already amortized and shrinking this request's chunks only adds
# chunks - each one re-paying `gamma * L`.
SOLE_PREFILL_THRESHOLD = 2

# Trailing schedule ticks `has_sole_prefill` takes the peak supply over.
#
# The condition has to hold for the whole of a request's prefill, not just the
# instant a chunk is sized: an instantaneous reading drops to one prefill for a
# step or two whenever a request finishes just before the next is admitted, and
# a chunk sequence committed in that dip then executes alongside everything that
# arrives behind it. One tick is one forward, so this window spans a good part of
# a long prompt's prefill.
GATE_SUPPLY_WINDOW = 16


def has_sole_prefill(sources: int, recent_sources: Iterable[int] = ()) -> bool:
    """Whether one request has been the pipeline's only prefill work recently.

    ``sources`` counts every request with prefill left to do, including the one
    being chunked, so 1 is the sole-prefill case.
    """
    return max((sources, *recent_sources)) < SOLE_PREFILL_THRESHOLD


def parse_chunking_calibration(text: str) -> tuple[float, ...]:
    """Parse ``"a,b,c,gamma"`` (or ``"a,b,c"``) into latency model coefficients."""
    try:
        coefficients = tuple(float(value) for value in text.split(","))
    except ValueError as exc:
        raise ValueError(
            f"Dynamic chunking calibration must be comma-separated floats, got {text!r}"
        ) from exc
    # Reuse the predictor's own validation so a bad fit fails at startup.
    ChunkSizePredictor.from_coefficients(coefficients)
    return coefficients


@dataclass(frozen=True)
class ChunkSizePredictor:
    """Predict equal-latency prefill chunks with a prefix-rebuild floor."""

    quadratic_coeff: float
    linear_coeff: float
    constant_coeff: float
    prefix_coeff: float = 0.0

    @classmethod
    def fit(
        cls,
        prefix_lens: list[int],
        chunk_sizes: list[int],
        latencies_ms: list[float],
    ) -> "ChunkSizePredictor":
        """Fit ``t(L, x)`` from startup profiling samples.

        Samples with ``prefix_len > 0`` are what separates ``gamma`` from the
        attention-area terms; a prefix-free sweep alone leaves ``gamma``
        unidentifiable and pushes the prefix cost into ``b`` and ``c``.
        """
        if not len(prefix_lens) == len(chunk_sizes) == len(latencies_ms):
            raise ValueError(
                "prefix_lens, chunk_sizes and latencies_ms must have equal length"
            )
        if len(chunk_sizes) < MIN_PROFILE_SAMPLES:
            raise ValueError(
                f"Dynamic chunking needs at least {MIN_PROFILE_SAMPLES} latency "
                "profiling samples"
            )

        prefixes = np.asarray(prefix_lens, dtype=np.float64)
        chunks = np.asarray(chunk_sizes, dtype=np.float64)
        latencies = np.asarray(latencies_ms, dtype=np.float64)
        for name, values in (
            ("prefix_lens", prefixes),
            ("chunk_sizes", chunks),
            ("latencies_ms", latencies),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Dynamic chunking {name} must be finite")
        if np.unique(chunks).size < 3:
            raise ValueError(
                "Dynamic chunking needs at least 3 distinct profiling chunk sizes"
            )

        area = 2.0 * prefixes * chunks + chunks * chunks
        design = np.column_stack((area, chunks, np.ones_like(chunks), prefixes))
        try:
            coeffs, _, rank, _ = np.linalg.lstsq(design, latencies, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Failed to fit dynamic chunking latency model") from exc
        if rank < design.shape[1]:
            raise ValueError("Dynamic chunking latency samples are rank deficient")

        quadratic, linear, constant, prefix = (float(value) for value in coeffs)
        if quadratic <= 0:
            raise ValueError(
                "Dynamic chunking requires a positive quadratic latency "
                f"coefficient, got a={quadratic:.3e}"
            )
        if linear <= 0:
            # Clamping this to 0 (as a prefix-free fit has to) makes the solver
            # ignore the hardware calibration entirely: the predicted chunk
            # collapses to sqrt(L^2 + C^2) - L, a pure geometry term. Refusing
            # the fit keeps fixed-size chunking instead of that fiction.
            raise ValueError(
                "Dynamic chunking requires a positive linear latency "
                f"coefficient, got b={linear:.3e}; the profiling window is "
                "dominated by per-forward overhead"
            )
        # Shape noise can make the prefix term slightly negative. Negative
        # prefix cost is not physical, and 0 simply disables the floor.
        return cls(quadratic, linear, constant, max(prefix, 0.0))

    @classmethod
    def from_coefficients(
        cls, coefficients: tuple[float, ...] | list[float]
    ) -> "ChunkSizePredictor":
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
            self.quadratic_coeff
            * (2.0 * history_len * chunk_size + chunk_size * chunk_size)
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
        matches the first chunk's runtime, and the caller should stop shrinking.
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
