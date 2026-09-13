"""Histogram primitives shared by metrics owners."""

import math
import threading
from bisect import bisect_left
from itertools import accumulate

from prometheus_client import Histogram

LATENCY_BUCKETS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    30,
    60,
    120,
    300,
    600,
)


class CumulativeHistogram:
    def __init__(self, bounds):
        self.bounds = (*bounds, math.inf)
        self.counts = [0] * len(self.bounds)
        self.total = 0.0

    def observe(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        self.counts[bisect_left(self.bounds, value)] += 1
        self.total += value

    def snapshot(self) -> dict:
        return {
            "buckets": list(zip(self.bounds, accumulate(self.counts))),
            "sum": self.total,
        }


class WeightedHistogram(Histogram):
    """Standard Prometheus histogram with an atomic weighted observation.

    prometheus_client has no public weighted observe API. Keep its private
    bucket access here; reuse its validation, labels, registration and export.
    The lock makes updates and scrapes consistent within this process.
    """

    def _metric_init(self):
        super()._metric_init()
        self._observation_lock = threading.Lock()

    def observe_weighted(self, total: float, weight: int) -> None:
        """Record weight equal samples whose sum is total, without a token loop."""
        self._raise_if_not_observable()
        if weight <= 0:
            return
        index = bisect_left(self._upper_bounds, total / weight)
        with self._observation_lock:
            self._sum.inc(total)
            self._buckets[index].inc(weight)

    def observe(self, amount: float, exemplar=None) -> None:
        self._raise_if_not_observable()
        with self._observation_lock:
            super().observe(amount, exemplar)

    def _child_samples(self):
        with self._observation_lock:
            return super()._child_samples()


def prometheus_buckets(snapshot):
    """Translate cumulative snapshot bounds to Prometheus bucket labels."""
    return [
        ("+Inf" if bound == math.inf else str(bound), count)
        for bound, count in snapshot["buckets"]
    ]
