# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which steps a model's decode round served (``model_hooks.DecodeRound``).

The round's forward runs inside a captured graph, so its Python never sees a
replay: the one per-step record is the builder's routing decision
(``DecodeRound.service``). ``RoundStats`` counts those decisions by step kind
and request count, and logs the tally once a minute on TP rank 0.
"""

from __future__ import annotations

import logging
import time

from atom.utils.forward_context import RoundService

logger = logging.getLogger("atom")

INTERVAL_S = 60.0
# Requests counted one by one up to here; more fall into one bucket.
MAX_BUCKET = 16


class RoundStats:
    def __init__(self, interval_s: float = INTERVAL_S, clock=time.monotonic):
        self._interval = interval_s
        self._clock = clock
        self._next = clock() + interval_s
        self.window: dict[tuple[str, int], list[int]] = {}
        self.total: dict[tuple[str, int], list[int]] = {}

    def record(self, *, is_prefill: bool, requests: int, service: RoundService):
        kind = "prefill" if is_prefill else "decode"
        key = (kind, min(int(requests), MAX_BUCKET + 1))
        served = int(service != RoundService.NONE)
        for table in (self.window, self.total):
            cell = table.setdefault(key, [0, 0])
            cell[0] += served
            cell[1] += 1
        now = self._clock()
        if now >= self._next:
            self._next = now + self._interval
            if _is_rank0():
                logger.info(
                    "decode round (last %.0fs): %s",
                    self._interval,
                    self.summary(self.window),
                )
                logger.info("decode round (total): %s", self.summary(self.total))
            self.window = {}

    @staticmethod
    def summary(table) -> str:
        parts = []
        for kind in ("decode", "prefill"):
            cells = sorted((r, c) for (k, r), c in table.items() if k == kind)
            if not cells:
                continue
            served = sum(c[0] for _, c in cells)
            steps = sum(c[1] for _, c in cells)
            by = ", ".join(
                f"{f'{r}+' if r > MAX_BUCKET else r}: {c[0]}/{c[1]}" for r, c in cells
            )
            parts.append(f"{kind} served {served}/{steps} by requests {{{by}}}")
        return "; ".join(parts) or "no steps"


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:  # noqa: BLE001
        return True
