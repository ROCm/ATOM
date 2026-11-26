# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prefill duty cycle while decodes run (``ATOM_SCHED_PREFILL_DUTY``).

Prefill and decode never share a step, so a running decode makes no progress
during a prefill step. Admitting prefill whenever there is some lets a burst of
long prefills stall every decode back to back, and the requests that overlap
the burst carry it in their inter-token latency. This keeps the share of
recent step time spent in prefill at or below ``target`` while any decode is
running: past it, the tick schedules decode only, until the share falls back
or a hold has lasted ``max_hold_s`` (so prefill never starves).

Step time is the wall time between two ``schedule()`` calls. Under deferred
output the forward RPC of step n returns once step n-1 has finished on the
device, so that interval is charged to the step scheduled two calls earlier.
Shares decay with time constant ``tau_s``.
"""

from __future__ import annotations

import math


class PrefillDuty:
    def __init__(
        self, target: float, tau_s: float, max_hold_s: float, max_wait_s: float = 0.0
    ):
        self.target = target
        self.tau_s = tau_s
        self.max_hold_s = max_hold_s
        # Never hold while the oldest prefill has waited this long (0: off), so a
        # load above `target` raises TTFT by a bounded amount instead of piling up.
        self.max_wait_s = max_wait_s
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self._last_t: float | None = None
        # kinds of the steps scheduled by the previous two calls, oldest first
        self._kinds: list[str | None] = [None, None]
        self._hold_since: float | None = None

    def observe(self, now: float, kind: str | None) -> None:
        """After a schedule() call that produced a step of ``kind``
        ('p' prefill, 'd' decode, None nothing)."""
        charged = self._kinds[0]
        if self._last_t is not None and charged is not None:
            dt = now - self._last_t
            decay = math.exp(-dt / self.tau_s)
            self.prefill_s *= decay
            self.decode_s *= decay
            if charged == "p":
                self.prefill_s += dt
            else:
                self.decode_s += dt
        self._kinds = [self._kinds[1], kind]
        self._last_t = now

    def share(self) -> float:
        total = self.prefill_s + self.decode_s
        return self.prefill_s / total if total > 0 else 0.0

    def hold(self, now: float, decoding: bool, oldest_wait_s=None) -> bool:
        """Whether this tick should skip prefill. ``oldest_wait_s``: a callable
        giving how long the oldest pending prefill has waited, asked only when
        a hold is otherwise due."""
        if not decoding or self.share() <= self.target:
            self._hold_since = None
            return False
        if (
            self.max_wait_s > 0
            and oldest_wait_s is not None
            and oldest_wait_s() >= self.max_wait_s
        ):
            self._hold_since = None
            return False
        if self.max_hold_s <= 0:
            self._hold_since = None
            return False
        if self._hold_since is None:
            self._hold_since = now
        elif now - self._hold_since >= self.max_hold_s:
            self._hold_since = None
            return False
        return True
