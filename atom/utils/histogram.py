"""Histogram primitives shared by metrics owners."""

import threading
from bisect import bisect_left

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
