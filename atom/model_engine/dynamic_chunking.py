# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Quadratic latency model used by dynamic chunked prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkSizePredictor:
    """Predict equal-latency prefill chunks from ``f(L) = aL² + bL + c``."""

    quadratic_coeff: float
    linear_coeff: float
    constant_coeff: float

    @classmethod
    def fit(
        cls, sequence_lengths: list[int], latencies_ms: list[float]
    ) -> "ChunkSizePredictor":
        """Fit a cumulative-runtime curve from startup profiling samples."""
        if len(sequence_lengths) != len(latencies_ms):
            raise ValueError("sequence_lengths and latencies_ms must have equal length")
        if len(sequence_lengths) < 8:
            raise ValueError(
                "Dynamic chunking needs at least 8 latency profiling samples"
            )

        lengths = np.asarray(sequence_lengths, dtype=np.float64)
        latencies = np.asarray(latencies_ms, dtype=np.float64)
        if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(latencies)):
            raise ValueError("Dynamic chunking profiling samples must be finite")
        if np.unique(lengths).size < 8:
            raise ValueError(
                "Dynamic chunking needs at least 8 distinct profiling lengths"
            )

        design = np.column_stack((lengths * lengths, lengths, np.ones_like(lengths)))
        try:
            coeffs, _, rank, _ = np.linalg.lstsq(design, latencies, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Failed to fit dynamic chunking latency model") from exc
        if rank < 3:
            raise ValueError("Dynamic chunking latency samples are rank deficient")

        quadratic, linear, constant = (float(value) for value in coeffs)
        if quadratic <= 0:
            raise ValueError(
                "Dynamic chunking requires a positive quadratic latency coefficient"
            )
        # Kernel shape noise can make the fitted linear term slightly negative.
        # A negative marginal runtime is not physical, so follow SGLang and clamp it.
        linear = max(linear, 0.0)
        return cls(quadratic, linear, constant)

    @classmethod
    def from_coefficients(
        cls, coefficients: tuple[float, float, float] | list[float]
    ) -> "ChunkSizePredictor":
        if len(coefficients) != 3:
            raise ValueError("Dynamic chunking requires exactly three coefficients")
        predictor = cls(*(float(value) for value in coefficients))
        if predictor.quadratic_coeff <= 0:
            raise ValueError("Dynamic chunking quadratic coefficient must be positive")
        if predictor.linear_coeff < 0:
            raise ValueError("Dynamic chunking linear coefficient must be non-negative")
        return predictor

    def target_latency(self, base_chunk_size: int) -> float:
        """Runtime of the initial chunk, with the constant overhead cancelled."""
        return (
            self.quadratic_coeff * base_chunk_size * base_chunk_size
            + self.linear_coeff * base_chunk_size
        )

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

        a = self.quadratic_coeff
        b = 2.0 * a * history_len + self.linear_coeff
        target = self.target_latency(base_chunk_size)
        discriminant = b * b + 4.0 * a * target
        raw = (-b + math.sqrt(discriminant)) / (2.0 * a)
        if not math.isfinite(raw) or raw <= 0:
            return None

        smoothed = base_chunk_size + smooth_factor * (raw - base_chunk_size)
        constrained = min(
            max(int(smoothed), base_chunk_size // 4),
            base_chunk_size,
            max_chunk_size,
        )
        aligned = constrained - constrained % alignment
        return aligned if aligned >= alignment else None
