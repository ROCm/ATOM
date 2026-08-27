# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reproducible sampling: turning a per-request seed into a per-token seed.

Kept separate from :mod:`scheduler` and :mod:`model_runner`, and deliberately
free of engine, GPU and torch imports, so it stays testable on its own. Keep it
that way when extending it.
"""

from collections.abc import Iterable
from typing import Any

#: A batch's seeded rows: (row indices, derived per-row seeds).
SeedRows = tuple[list[int], list[int]]

_U64 = 0xFFFFFFFFFFFFFFFF


def mix_seed(seed: int, position: int) -> int:
    """Derive the RNG seed for one (sequence, token position) from a request seed.

    The splitmix64 finalizer, chosen because ``seed + position`` both correlates
    neighbouring seeds and collides outright -- seed 42 at position 1 would equal
    seed 43 at position 0.

    Being pure is load-bearing: a sequence replays the same draws on any rank and
    after any preemption, since nothing here depends on call order.
    """
    x = ((seed & _U64) * 0x9E3779B97F4A7C15 + position) & _U64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _U64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _U64
    x ^= x >> 31
    # torch.Generator.manual_seed takes a signed 64-bit value.
    return x & 0x7FFFFFFFFFFFFFFF


def build_seed_rows(seqs: Iterable[Any]) -> SeedRows | None:
    """Per-row ``(row, derived seed)``, or ``None`` when no sequence has a seed.

    ``None`` is what keeps the sampler on its shared-noise path.
    """
    rows: list[int] = []
    values: list[int] = []
    for row, seq in enumerate(seqs):
        seed = getattr(seq, "seed", None)
        if seed is None:
            continue
        rows.append(row)
        values.append(mix_seed(seed, seq.num_completion_tokens))
    if not rows:
        return None
    return rows, values
