"""Performance metrics aggregation and analysis."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from atom.autotuner.types import BenchmarkResult


@dataclass
class AggregatedMetrics:
    """Statistical summary of multiple benchmark runs."""
    count: int
    throughput_per_gpu_mean: float
    throughput_per_gpu_std: float
    throughput_per_user_mean: float
    throughput_per_user_std: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_mean_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float


class MetricsAggregator:
    """Aggregate and analyze benchmark results."""

    @staticmethod
    def aggregate(results: Sequence[BenchmarkResult]) -> AggregatedMetrics:
        if not results:
            return AggregatedMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        tpg = [r.throughput_per_gpu for r in results]
        tpu = [r.throughput_per_user for r in results]
        ttfts = sorted(r.ttft_ms for r in results)
        tpots = sorted(r.tpot_ms for r in results)

        return AggregatedMetrics(
            count=len(results),
            throughput_per_gpu_mean=statistics.mean(tpg),
            throughput_per_gpu_std=statistics.stdev(tpg) if len(tpg) > 1 else 0,
            throughput_per_user_mean=statistics.mean(tpu),
            throughput_per_user_std=statistics.stdev(tpu) if len(tpu) > 1 else 0,
            ttft_mean_ms=statistics.mean(ttfts),
            ttft_p50_ms=_percentile(ttfts, 50),
            ttft_p99_ms=_percentile(ttfts, 99),
            tpot_mean_ms=statistics.mean(tpots),
            tpot_p50_ms=_percentile(tpots, 50),
            tpot_p99_ms=_percentile(tpots, 99),
        )

    @staticmethod
    def compare(baseline: BenchmarkResult, candidate: BenchmarkResult) -> dict:
        """Compare two results and return improvement percentages."""
        def pct(new: float, old: float) -> float:
            if old == 0:
                return 0
            return (new - old) / abs(old) * 100

        return {
            "throughput_per_gpu_pct": pct(
                candidate.throughput_per_gpu, baseline.throughput_per_gpu
            ),
            "throughput_per_user_pct": pct(
                candidate.throughput_per_user, baseline.throughput_per_user
            ),
            "ttft_pct": pct(baseline.ttft_ms, candidate.ttft_ms),  # inverted: lower is better
            "tpot_pct": pct(baseline.tpot_ms, candidate.tpot_ms),
        }


def _percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (pct / 100) * (len(sorted_data) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_data[lo]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac
