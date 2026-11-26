# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ITL-risk prefill hold (``ATOM_SCHED_ITL_RISK_MS``, default off).

Prefill and decode never share a step, so a prefill step stalls every running
decode for its whole duration. The scored interactivity is a percentile over
requests of ``(latency - TTFT) / (output_len - 1)``: a stall of S seconds adds
``S / (output_len - 1)`` to one request's value. Only requests near the
percentile matter: one that ends far below the target ``theta`` absorbs the
stall, and one that is above it anyway is lost either way. So hold a prefill
while some running decode would be pushed from at or below ``theta`` to above
it by this step's stall, and release it otherwise -- instead of holding every
prefill for a fixed number of steps or a duty cycle.

Per running decode (its own ``max_tokens`` is its output length):
    base = (elapsed since its first token + steps_left * decode_step) / (max_tokens - 1)
    at risk  <=>  base <= theta < base + stall / (max_tokens - 1)
``decode_step`` and the tokens a decode gains per step are measured here
(exponential averages over consecutive decode ticks); the stall is
``pf_fixed + pf_per_ktok * tokens`` for the prefill tokens pending this tick.
A hold lasts at most ``max_hold_s`` (then one tick is released), so prefill
cannot starve. Holding changes only which steps run, never their numerics.
"""

from __future__ import annotations

import math

# exponential averaging weight per decode-tick sample
_EMA = 0.1


class ItlRiskHold:
    def __init__(
        self,
        theta_ms: float,
        max_hold_s: float,
        pf_fixed_ms: float,
        pf_ms_per_ktok: float,
    ):
        self.theta_s = theta_ms / 1000.0
        self.max_hold_s = max_hold_s
        self.pf_fixed_s = pf_fixed_ms / 1000.0
        self.pf_s_per_tok = pf_ms_per_ktok / 1e6
        self.decode_step_s: float | None = None
        self.tokens_per_step: float | None = None
        self._kinds: list[str | None] = [None, None]
        self._last_t: float | None = None
        self._last_tokens: dict[int, int] = {}
        self._since: float | None = None
        self.holds = 0
        self.releases = 0

    def observe(self, now: float, kind: str | None, running) -> None:
        """After a schedule() call that produced a step of ``kind`` ('p', 'd' or
        None). Learns the decode step time and the tokens a decode gains per
        step from runs of consecutive decode ticks (prefill ticks and their
        deferred outputs are skipped)."""
        clean = kind == "d" and self._kinds[0] == "d" and self._kinds[1] == "d"
        if clean and self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:
                self.decode_step_s = (
                    dt
                    if self.decode_step_s is None
                    else self.decode_step_s + _EMA * (dt - self.decode_step_s)
                )
        tokens = {}
        if kind == "d":
            for seq in running:
                if not seq.is_partial_prefill:
                    tokens[seq.id] = seq.num_completion_tokens
            if clean and self._last_tokens:
                gains = [
                    n - self._last_tokens[i]
                    for i, n in tokens.items()
                    if i in self._last_tokens
                ]
                if gains:
                    g = sum(gains) / len(gains)
                    if g > 0:
                        self.tokens_per_step = (
                            g
                            if self.tokens_per_step is None
                            else self.tokens_per_step
                            + _EMA * (g - self.tokens_per_step)
                        )
        self._last_tokens = tokens
        self._kinds = [self._kinds[1], kind]
        self._last_t = now

    def stall_s(self, prefill_tokens: int) -> float:
        return self.pf_fixed_s + self.pf_s_per_tok * prefill_tokens

    def due(self, wall_now: float, running, prefill_tokens: int) -> bool:
        """Whether this tick's prefill (``prefill_tokens`` pending) would push
        some running decode across ``theta``. False until both rates are
        measured."""
        step = self.decode_step_s
        per_step = self.tokens_per_step
        if step is None or not per_step:
            return False
        stall = self.stall_s(prefill_tokens)
        theta = self.theta_s
        for seq in running:
            if seq.is_partial_prefill or seq.first_token_time <= 0.0:
                continue
            out_len = seq.max_tokens
            if out_len is None or out_len < 2:
                continue
            left = max(0, out_len - seq.num_completion_tokens)
            base = (
                max(0.0, wall_now - seq.first_token_time)
                + math.ceil(left / per_step) * step
            ) / (out_len - 1)
            if base <= theta < base + stall / (out_len - 1):
                return True
        return False

    def hold(self, now: float, due: bool) -> bool:
        """Whether this tick should skip prefill; bounded like FinishHold."""
        if not due:
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
