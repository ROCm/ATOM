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
"""

from __future__ import annotations

import math
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
        raw = self._solve_chunk(history_len, self.target_latency(base_chunk_size))
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
    ) -> int | None:
        """Solve ``f(L+x)-f(L)=f(base)-f(0)`` and apply serving constraints."""
        if history_len < 0:
            raise ValueError("history_len must be non-negative")
        if base_chunk_size <= 0 or alignment <= 0 or max_chunk_size <= 0:
            raise ValueError("Chunk sizes and alignment must be positive")
        if not 0.0 <= smooth_factor <= 1.0:
            raise ValueError("smooth_factor must be in [0, 1]")

        raw = self._solve_chunk(history_len, self.target_latency(base_chunk_size))
        if not math.isfinite(raw) or raw <= 0:
            return None

        smoothed = base_chunk_size + smooth_factor * (raw - base_chunk_size)
        lower_bound = max(
            base_chunk_size // 4,
            int(self.prefix_bounded_chunk(history_len)),
        )
        constrained = min(
            max(int(smoothed), lower_bound),
            base_chunk_size,
            max_chunk_size,
        )
        aligned = constrained - constrained % alignment
        return aligned if aligned >= alignment else None
