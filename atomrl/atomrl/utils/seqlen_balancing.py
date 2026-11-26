# Copyright 2025 The AtomRL Authors.
# Derived from verl (verl-project/verl):
#   verl/utils/seqlen_balancing.py L27-254
#     (calculate_workload, karmarkar_karp, get_seqlen_balanced_partitions,
#      log_seqlen_unbalance)
#
# Licensed under the Apache License, Version 2.0.
"""Sequence-length balanced partitioning via Karmarkar-Karp differencing."""

from __future__ import annotations

import heapq
import logging
from typing import Any

from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workload estimator  (verl/utils/seqlen_balancing.py L27-46)
# ---------------------------------------------------------------------------


def calculate_workload(seqlen_list: Tensor) -> Tensor:
    """Approximate transformer FLOPs: ``24576 * seqlen + seqlen^2`` (7B calibration)."""
    return 24576 * seqlen_list + seqlen_list**2


# ---------------------------------------------------------------------------
# Karmarkar-Karp  (verl/utils/seqlen_balancing.py L49-172)
# ---------------------------------------------------------------------------


def karmarkar_karp(
    seqlen_list: list[int], k_partitions: int, equal_size: bool
) -> list[list[int]]:
    """Partition items into *k* balanced groups via Largest Differencing Method."""

    class _Set:
        __slots__ = ("items", "sum")

        def __init__(self) -> None:
            self.sum = 0
            self.items: list[tuple[int, int]] = []

        def add(self, idx: int, val: int) -> None:
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other: _Set) -> None:
            self.items.extend(other.items)
            self.sum += other.sum

        def __lt__(self, other: _Set) -> bool:
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class _State:
        __slots__ = ("k", "sets")

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            self.sets = [_Set() for _ in range(k)]
            for i, (idx, v) in enumerate(items):
                self.sets[i].add(idx, v)
            self.sets.sort(reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def merge(self, other: _State) -> None:
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets.sort(reverse=True)

        def partitions(self) -> list[list[int]]:
            return [[idx for idx, _ in s.items] for s in self.sets]

        def __lt__(self, other: _State) -> bool:
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

    sorted_sl = sorted([(v, i) for i, v in enumerate(seqlen_list)])
    pq: list[_State] = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0
        for off in range(0, len(sorted_sl), k_partitions):
            items = [
                (sorted_sl[off + j][1], sorted_sl[off + j][0])
                for j in range(k_partitions)
            ]
            heapq.heappush(pq, _State(items, k_partitions))
    else:
        for v, idx in sorted_sl:
            heapq.heappush(pq, _State([(idx, v)], k_partitions))

    while len(pq) > 1:
        s0 = heapq.heappop(pq)
        s1 = heapq.heappop(pq)
        s0.merge(s1)
        heapq.heappush(pq, s0)

    return pq[0].partitions()


# ---------------------------------------------------------------------------
# Public API  (verl/utils/seqlen_balancing.py L213-254)
# ---------------------------------------------------------------------------


def get_seqlen_balanced_partitions(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool = False,
) -> list[list[int]]:
    """Return *k* balanced index partitions using Karmarkar-Karp.

    Each partition list is sorted; asserts coverage and non-emptiness.
    """
    assert (
        len(seqlen_list) >= k_partitions
    ), f"items ({len(seqlen_list)}) < k_partitions ({k_partitions})"
    parts = karmarkar_karp(seqlen_list, k_partitions, equal_size)
    seen: set[int] = set()
    out: list[list[int]] = []
    for p in parts:
        assert len(p) > 0, "Empty partition"
        seen.update(p)
        out.append(sorted(p))
    assert seen == set(range(len(seqlen_list)))
    return out


# ---------------------------------------------------------------------------
# Imbalance logging  (verl/utils/seqlen_balancing.py L257-302)
# ---------------------------------------------------------------------------


def log_seqlen_unbalance(
    seqlen_list: list[int],
    partitions: list[list[int]],
    prefix: str = "seqlen",
) -> dict[str, Any]:
    """Compute before/after imbalance metrics for logging."""
    k = len(partitions)
    bs = len(seqlen_list) // k

    min_s = max_s = total = 0
    for off in range(0, len(seqlen_list), bs):
        cur = sum(seqlen_list[off : off + bs])
        if off == 0:
            min_s = max_s = cur
        else:
            min_s = min(min_s, cur)
            max_s = max(max_s, cur)
        total += cur

    bal = [sum(seqlen_list[i] for i in p) for p in partitions]
    return {
        f"{prefix}/min": min_s,
        f"{prefix}/max": max_s,
        f"{prefix}/minmax_diff": max_s - min_s,
        f"{prefix}/balanced_min": min(bal),
        f"{prefix}/balanced_max": max(bal),
        f"{prefix}/mean": total / len(partitions),
    }
