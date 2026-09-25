"""Periodic per-step forward wall-time summary (ATOM_STEP_TIMING_LOG_S).

Diagnostics only. Times the synchronous ``forward`` call of each engine step
on the host, so it includes every wait a step pays -- under DP-attention that
is the lockstep MoE collectives waiting for the slowest rank -- without the
device-side instrumentation a torch profile adds.
"""

import logging
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger("atom")


class StepTimingLog:
    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.next_log = time.monotonic() + interval_s
        self.samples: dict[str, list[float]] = defaultdict(list)

    def record(self, batch, seconds: float, label: str) -> None:
        if batch.total_seqs_num_prefill > 0:
            kind = "prefill" if batch.total_seqs_num_decode == 0 else "mixed"
        else:
            kind = "decode"
        self.samples[kind].append(seconds * 1000)
        now = time.monotonic()
        if now < self.next_log:
            return
        self.next_log = now + self.interval_s
        total = sum(sum(v) for v in self.samples.values())
        parts = []
        for kind, ms in sorted(self.samples.items()):
            arr = np.asarray(ms)
            parts.append(
                f"{kind}: n={arr.size} sum={arr.sum() / 1000:.1f}s "
                f"({100 * arr.sum() / total:.0f}%) mean={arr.mean():.1f} "
                f"p50={np.percentile(arr, 50):.1f} p90={np.percentile(arr, 90):.1f} "
                f"max={arr.max():.1f}ms"
            )
        logger.info(f"[step-timing] {label} " + " | ".join(parts))
        self.samples.clear()
