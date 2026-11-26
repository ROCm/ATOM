# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Finish-aware prefill hold (``ATOM_SCHED_FINISH_HOLD_LEFT``, opt-in).

Prefill and decode never share a step, so every running decode stalls for the
whole prefill step. The scored metric (per-request inter-token latency,
``(latency - TTFT) / (output_len - 1)``) charges that stall per output token,
so it hurts a short request far more than a long one. When a short running
request is a few steps from its ``max_tokens``, holding the prefill for those
steps lets it finish unstalled, and the prefill then stalls one request fewer.

A tick holds prefill while some running decode with ``max_tokens <=
max_tokens_cap`` has at most ``tokens_left`` tokens to go to its
``max_tokens``. In tokens, not steps: tokens per step depend on acceptance.
A hold never lasts longer than ``max_hold_s`` (then one tick is released),
so prefill cannot starve. EOS can end a request sooner; ``max_tokens`` is
only an upper bound, so the policy never holds for a request that cannot end.
"""

from __future__ import annotations


class FinishHold:
    def __init__(self, tokens_left: int, max_hold_s: float, max_tokens_cap: int):
        self.tokens_left = tokens_left
        self.max_hold_s = max_hold_s
        self.max_tokens_cap = max_tokens_cap
        self._since: float | None = None
        self.holds = 0
        self.releases = 0

    def due(self, running) -> bool:
        """Whether some running decode is a short request near its end."""
        reach = self.tokens_left
        for seq in running:
            if seq.is_partial_prefill:
                continue
            if seq.max_tokens > self.max_tokens_cap:
                continue
            if seq.max_tokens - seq.num_completion_tokens <= reach:
                return True
        return False

    def hold(self, now: float, due: bool) -> bool:
        """Whether this tick should skip prefill. ``due``: there is prefill
        work, decodes run and :meth:`due` holds; False resets the hold."""
        if not due:
            self._since = None
            return False
        if self.max_hold_s <= 0:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        elif now - self._since >= self.max_hold_s:
            self._since = None
            self.releases += 1
            return False
        self.holds += 1
        return True
