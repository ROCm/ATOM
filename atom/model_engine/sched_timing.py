"""Per-step host/GPU timing for the serving loop (``ATOM_SCHED_TIMING=1``).

Off by default: every hook is a no-op returning immediately. When on, the
engine loop and rank 0's forward record named spans (seconds) and the batch
shape of each step, and log a percentile summary every
``ATOM_SCHED_TIMING_INTERVAL`` seconds (default 30).

GPU idle between steps is read from CUDA events recorded at the first and
last enqueue of each forward: ``gap = start[n] - end[n-1]`` is time the
device had no work from this runner (it waited for the host), ``busy =
end[n] - start[n]``.
"""

from __future__ import annotations

import collections
import logging
import os
import time

logger = logging.getLogger("atom")

ENABLED = os.getenv("ATOM_SCHED_TIMING", "0") == "1"
INTERVAL_S = float(os.getenv("ATOM_SCHED_TIMING_INTERVAL", "30"))

# Host time blocked on device events during the current forward.
SYNC_WAIT = [0.0]


def timed_sync(event) -> None:
    if not ENABLED:
        event.synchronize()
        return
    t = time.perf_counter()
    event.synchronize()
    SYNC_WAIT[0] += time.perf_counter() - t


def prefill_kind(tokens: int) -> str:
    """'p<=N' with N the power-of-two KiB bucket of a prefill step's tokens."""
    k = 1
    while k * 1024 < tokens:
        k *= 2
    return f"p{k}k"


def _pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


class StepTimer:
    def __init__(self, label: str):
        self.label = label
        self.spans: dict[str, list[float]] = collections.defaultdict(list)
        self.shapes: collections.Counter = collections.Counter()
        self.steps = 0
        self.t_window = time.perf_counter()
        self._mark = None

    def add(self, name: str, dt: float) -> None:
        self.spans[name].append(dt)

    def shape(self, key) -> None:
        self.shapes[key] += 1

    def step_done(self) -> None:
        self.steps += 1
        now = time.perf_counter()
        if now - self.t_window >= INTERVAL_S:
            self.flush(now)

    def flush(self, now: float) -> None:
        wall = now - self.t_window
        parts = [f"[sched_timing {self.label}] {self.steps} steps in {wall:.1f}s"]
        for name, vals in sorted(self.spans.items()):
            tot = sum(vals)
            parts.append(
                f"  {name:14s} n={len(vals):6d} mean={1e3 * tot / len(vals):7.3f}ms "
                f"p50={1e3 * _pct(vals, 50):7.3f} p90={1e3 * _pct(vals, 90):7.3f} "
                f"p99={1e3 * _pct(vals, 99):7.3f} sum%={100 * tot / wall:5.1f}"
            )
        if self.shapes:
            top = self.shapes.most_common(24)
            parts.append("  shapes " + " ".join(f"{k}:{v}" for k, v in top))
        logger.info("\n".join(parts))
        self.spans.clear()
        self.shapes.clear()
        self.steps = 0
        self.t_window = now


class GpuGapTimer:
    """Device idle between consecutive forwards, from timing events."""

    def __init__(self, timer: StepTimer):
        import torch

        self._torch = torch
        self.timer = timer
        self.pending: collections.deque = collections.deque()
        self.prev_end = None
        self.cur_start = None

    def start(self) -> None:
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        self.cur_start = ev

    def end(self, kind: str) -> None:
        if self.cur_start is None:
            return
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        self.pending.append((self.prev_end, self.cur_start, ev, kind))
        self.prev_end = ev
        self.cur_start = None
        self._drain()

    def _drain(self) -> None:
        while self.pending and self.pending[0][2].query():
            prev_end, start, end, kind = self.pending.popleft()
            busy = start.elapsed_time(end) / 1e3
            self.timer.add(f"gpu_busy_{kind}", busy)
            if prev_end is not None:
                gap = prev_end.elapsed_time(start) / 1e3
                # A gap longer than a second is the engine idling between
                # requests, not step overhead.
                if gap < 1.0:
                    self.timer.add(f"gpu_gap_{kind}", gap)
