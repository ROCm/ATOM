# SPDX-License-Identifier: MIT
"""Bounded affine correction of a measured DSpark cost curve.

`latency(rows, batch) = baseline(rows) + fixed(batch) + slope * rows + post(batch)`.

The slope is a marginal per-row cost, so it is fitted only from contrasts that
share everything but the row count: the same batch size AND the same request
set (context generation), whose graph sizes differ because the verify cap q
changed. Comparing different batches or contexts would read their different
fixed overheads (more requests, longer KV) as per-row cost. Each (batch,
context, graph size) contributes one median residual; the slope is the median
of the pairwise slopes inside each (batch, context), Theil-Sen style, so a
constant offset cannot tilt it. A batch with its own contrasts gets its own
slope; others share the slope pooled from every batch's within-context pairs,
and with no contrasts at all the slope is zero (the measured curve's own
marginal cost stands). No request identities, tokens or confidence values are
retained by the estimator.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import defaultdict, deque

# Distinct graph sizes the contrasts of a fit must span: two sizes always admit
# a line, so a level shift between two sizes would read as slope.
_MIN_GRAPH_SIZES = 3


def _within_context_slopes(samples):
    """Pairwise slopes between graph sizes of the same context.

    ``samples``: ``(context, rows, residual)`` of ONE batch size. Returns the
    slopes, the distinct sizes they span, and the samples behind them.
    """
    by_size = defaultdict(lambda: defaultdict(list))
    for context, rows, residual in samples:
        by_size[context][rows].append(residual)
    slopes, sizes, support = [], set(), 0
    for per_rows in by_size.values():
        if len(per_rows) < 2:
            continue  # one size: no contrast, only a level
        points = sorted((r, statistics.median(v)) for r, v in per_rows.items())
        slopes.extend(
            (b[1] - a[1]) / (b[0] - a[0]) for a, b in itertools.combinations(points, 2)
        )
        sizes.update(per_rows)
        support += sum(len(v) for v in per_rows.values())
    return slopes, sizes, support


class OnlineSPS:
    def __init__(self, baseline_sps, max_batch, *, window=5000, min_samples=10):
        if window < min_samples or min_samples < 1 or max_batch < 1:
            raise ValueError("Invalid online SPS window/batch limits")
        values = [float(v) for v in baseline_sps]
        if not values or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(
                "Online SPS requires finite positive baseline measurements"
            )
        if any(b > a for a, b in itertools.pairwise(values)):
            raise ValueError("Online SPS baseline must be non-increasing")
        self.baseline_ms = [1000.0 / v for v in values]
        self.max_batch = max_batch
        self.window = window
        self.min_samples = min_samples
        self.samples = deque()
        self.slope_by_batch = {}
        self.fixed_by_batch = {}
        self.support_by_batch = {}
        self.post_by_batch = {}
        self.slope_global = 0.0
        self.global_support = 0
        self.fixed_global = 0.0
        self.post_global = 0.0
        self.version = 0
        self.accepted = 0
        self.rejected = 0
        self.current_step = 0
        self.last_sample_step = 0
        self._dirty = False

    def observe(
        self, batch, graph_rows, forward_ms, post_ms, *, step=None, context=None
    ):
        step = self.last_sample_step + 1 if step is None else step
        if (
            type(step) is not int
            or step <= self.last_sample_step
            or step <= self.current_step - self.window
            or type(batch) is not int
            or not 1 <= batch <= self.max_batch
            or type(graph_rows) is not int
            or not batch <= graph_rows < len(self.baseline_ms)
            or not math.isfinite(forward_ms)
            or forward_ms <= 0
            or not math.isfinite(post_ms)
            or post_ms < 0
            or (context is not None and (type(context) is not int or context < 0))
        ):
            self.rejected += 1
            return False
        residual = forward_ms - self.baseline_ms[graph_rows]
        self.last_sample_step = step
        self.current_step = max(self.current_step, step)
        self._expire()
        self.samples.append((step, batch, graph_rows, residual, post_ms, context))
        self.accepted += 1
        self._dirty = True
        return True

    def _expire(self):
        while self.samples and self.samples[0][0] <= self.current_step - self.window:
            self.samples.popleft()
            self._dirty = True

    @staticmethod
    def _recent_median(values):
        return max(0.0, statistics.median(values[-9:])) if len(values) >= 5 else None

    def publish(self, *, step=None):
        if step is not None:
            if type(step) is not int or step < self.current_step:
                raise ValueError("Publication step must be monotone")
            self.current_step = step
        self._expire()
        if not self._dirty:
            return False

        def fit(slopes, sizes, support):
            if len(sizes) < _MIN_GRAPH_SIZES or support < self.min_samples:
                return None
            return max(0.0, statistics.median(slopes))

        per_batch = defaultdict(list)
        for _, batch, rows, residual, _, context in self.samples:
            per_batch[batch].append((context, rows, residual))
        self.slope_by_batch, self.support_by_batch = {}, {}
        pooled, pooled_sizes, pooled_support = [], set(), 0
        for batch, samples in per_batch.items():
            slopes, sizes, support = _within_context_slopes(samples)
            self.support_by_batch[batch] = support
            slope = fit(slopes, sizes, support)
            if slope is not None:
                self.slope_by_batch[batch] = slope
            pooled += slopes
            pooled_sizes |= sizes
            pooled_support += support
        self.global_support = pooled_support
        self.slope_global = fit(pooled, pooled_sizes, pooled_support) or 0.0

        # The fixed component is what is left after the slope: per batch from
        # its recent samples, globally as the fallback.
        fixed = defaultdict(list)
        post = defaultdict(list)
        fixed_all, post_all = [], []
        for _, batch, rows, residual, post_ms, _ in self.samples:
            offset = residual - rows * self.slope_ms_for(batch)
            fixed[batch].append(offset)
            post[batch].append(post_ms)
            fixed_all.append(offset)
            post_all.append(post_ms)
        self.fixed_global = self._recent_median(fixed_all) or 0.0
        self.post_global = self._recent_median(post_all) or 0.0
        self.fixed_by_batch = {
            b: value
            for b, v in fixed.items()
            if (value := self._recent_median(v)) is not None
        }
        self.post_by_batch = {
            b: value
            for b, v in post.items()
            if (value := self._recent_median(v)) is not None
        }
        self.version += 1
        self._dirty = False
        return True

    def slope_ms_for(self, batch):
        return self.slope_by_batch.get(batch, self.slope_global)

    @property
    def slope_ms(self):
        """The global slope; a batch with its own spread may differ."""
        return self.slope_global

    def fixed_ms(self, batch):
        return self.fixed_by_batch.get(batch, self.fixed_global)

    def post_ms(self, batch):
        return self.post_by_batch.get(batch, self.post_global)

    def estimate_ms(self, graph_rows, batch):
        if not 0 <= graph_rows < len(self.baseline_ms):
            raise ValueError("Online SPS cannot extrapolate outside profile coverage")
        return (
            self.baseline_ms[graph_rows]
            + self.fixed_ms(batch)
            + self.slope_ms_for(batch) * graph_rows
            + self.post_ms(batch)
        )

    def snapshot(self):
        return {
            "version": self.version,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "window_samples": len(self.samples),
            "window_steps": self.window,
            "current_step": self.current_step,
            "oldest_sample_step": self.samples[0][0] if self.samples else None,
            "post_window_samples": 9,
            "slope_ms_per_row": self.slope_ms,
            "slope_global_support": self.global_support,
            "slope_ms_per_row_by_batch": dict(self.slope_by_batch),
            "slope_support_by_batch": dict(self.support_by_batch),
            "fixed_ms_by_batch": dict(self.fixed_by_batch),
            "fixed_ms_global": self.fixed_global,
            "post_ms_by_batch": dict(self.post_by_batch),
            "post_ms_global": self.post_global,
        }


def common_rank_samples(rank_rows):
    """Intersect completed TP samples and take conservative phase maxima.

    Rows contain (step, batch, rows, target_ms, post_ms[, context_generation]).
    A context mismatch also rejects the sample. The returned five-column API
    is unchanged; callers retain their already matched local generation.
    """
    if not rank_rows:
        return []
    maps = [{int(row[0]): row for row in rows if row[0] > 0} for rows in rank_rows]
    shared = set(maps[0]).intersection(*(set(m) for m in maps[1:]))
    result = []
    for step in sorted(shared):
        rows = [m[step] for m in maps]
        if any(
            tuple(r[1:3]) != tuple(rows[0][1:3]) or tuple(r[5:]) != tuple(rows[0][5:])
            for r in rows
        ):
            continue
        if any(
            not all(math.isfinite(v) for v in r) or r[3] <= 0 or r[4] < 0 for r in rows
        ):
            continue
        result.append(
            (
                step,
                int(rows[0][1]),
                int(rows[0][2]),
                max(r[3] for r in rows),
                max(r[4] for r in rows),
            )
        )
    return result
