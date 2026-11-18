# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""LDS selection segments sized for a standalone indexer."""

from dataclasses import dataclass

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.mono.plan.arith import sel


def row_segment(bound_max):
    # Leave room for a short speculative suffix instead of adding a tiny leaf
    # at the common 8K/80K boundaries. Widths remain whole per-thread quads.
    return 18432 if bound_max > 16384 else 10240


def segment_for(segment, kind):
    return min(segment, 8192) if kind == "blk" else segment


@dataclass(frozen=True)
class SelectionPlan:
    SEG: int

    @property
    def CHUNKS_PER_SEG(self):
        return self.SEG // ip.CHUNK

    def fan(self, k):
        return self.SEG // k

    def segments(self, bound):
        return ip.cdiv(ip.imax(bound, 0), self.SEG)

    def seg_len(self, bound, j):
        return ip.imax(ip.imin(bound - j * self.SEG, self.SEG), 0)

    def seg_chunks(self, bound, j):
        return sel(ip.scored(bound), ip.cdiv(self.seg_len(bound, j), ip.CHUNK), 0)

    def block_chunks(self, bound, j):
        rows = self.SEG * ip.BLOCK_ROWS
        count = ip.cdiv(ip.imax(ip.imin(bound, (j + 1) * rows) - j * rows, 0), ip.CHUNK)
        return j * (rows // ip.CHUNK), sel(ip.scored(bound), count, 0)

    def lists(self, bound, level, k):
        n = self.segments(bound)
        for _ in range(level):
            n = ip.cdiv(n, self.fan(k))
        return n

    def list_inputs(self, bound, level, j, k):
        if level == 0:
            return self.seg_len(bound, j)
        below = self.lists(bound, level - 1, k)
        first = j * self.fan(k)
        count = ip.imin(below - first, self.fan(k))
        last = first + count - 1
        return ip.imax(count - 1, 0) * k + ip.winners(
            self.list_inputs(bound, level - 1, last, k), k
        )

    def list_count(self, bound, level, k):
        if level == 0:
            return ip.imax(self.lists(bound, 0, k), 1)
        return sel(self.lists(bound, level - 1, k) > 1, self.lists(bound, level, k), 0)

    def is_final(self, bound, level, k):
        return self.lists(bound, level, k) <= 1

    def top_level(self, kind, bound_max):
        n = self.segments(ip.kind_bound(kind, bound_max))
        level = 0
        while n > 1:
            n = ip.cdiv(n, self.fan(ip.KINDS[kind]))
            level += 1
        return level

    def list_slots(self, kind, level, bound_max):
        return self.lists(ip.kind_bound(kind, bound_max), level, ip.KINDS[kind])


def list_slots(seg, *args):
    return SelectionPlan(seg).list_slots(*args)


def list_count(seg, *args):
    return SelectionPlan(seg).list_count(*args)


def list_inputs(seg, *args):
    return SelectionPlan(seg).list_inputs(*args)


def is_final(seg, *args):
    return SelectionPlan(seg).is_final(*args)


def top_level(seg, *args):
    return SelectionPlan(seg).top_level(*args)


def seg_chunks(seg, *args):
    return SelectionPlan(seg).seg_chunks(*args)


def block_chunks(seg, *args):
    return SelectionPlan(seg).block_chunks(*args)


def fan(seg, *args):
    return SelectionPlan(seg).fan(*args)
