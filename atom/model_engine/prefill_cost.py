# SPDX-License-Identifier: MIT
"""Shared prefill work units for routing and chunk scheduling.

This is the scheduler's existing heuristic, not a fitted latency model for
sparse attention. Work is additive across chunks, including their prefixes.
"""


def prefill_cost(prefix: int, tokens: int, token_weight: float) -> float:
    return tokens * (token_weight + prefix + tokens / 2)
