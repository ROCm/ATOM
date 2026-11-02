# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A step's routing as per-expert token bitmaps in LDS, so every CTA derives
the same pick order whatever order its threads' writes land in: each pick
ORs its token's bit into its expert's bitmap (``mark_token``); an expert's
pick count is a popcount, and a pick's rank among its expert's picks the
bitmap's tokens below its own (``tokens_below``)."""

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from atom.mono.device.ops import popcount


def mark_token(mask, words: int, e, t) -> None:
    """Token t into expert e's bitmap (``words`` i32 an expert at LDS ``mask``)."""
    fx.llvm.atomic_or(
        mask + (e * words + t // 32), fx.Int32(1) << (t % 32), syncscope="workgroup"
    )


def tokens_below(mask, words: int, e, t):
    """The tokens below t in expert e's bitmap: token t's rank among e's picks."""
    rank = fx.Int32(0)
    for w in range_constexpr(words):
        lo = t - 32 * w
        keep = (lo >= 32).select(
            fx.Int32(-1),
            (lo > 0).select((fx.Int32(1) << fx.max(lo, 0)) - 1, fx.Int32(0)),
        )
        rank = rank + popcount(fx.ptr_load(mask + (e * words + w)) & keep)
    return rank
