# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compact one-copy peer regions for deferred TP hand-offs."""

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.shapes import ALIGN, HIDDEN
from atom.mono.plan.layout import pair_layout

SLOTS = 2
PART_COLS = 32
PARTS = HIDDEN // PART_COLS


def peer_layout(tokens: int, tp: int) -> dict[str, tuple[int, int]]:
    """Return the symmetric peer regions for one rank.

    A source writes one copy into its own symmetric buffer. Consumers select
    the source through its peer base, so a region has no redundant source-rank
    dimension. Each BF16 pair occupies a tagged 8-byte mailbox pair.
    """

    if tokens < 1:
        raise ValueError(f"Qwen mono peer layout needs tokens >= 1, got {tokens}")
    Dims(tp)
    row_pairs = tokens * HIDDEN // 2
    slot_pairs = SLOTS * row_pairs
    return pair_layout(
        (("attention", slot_pairs), ("moe", slot_pairs)),
        align=256,
    )


def peer_offsets(tokens: int, tp: int) -> dict[str, int]:
    return {name: offset for name, (offset, _) in peer_layout(tokens, tp).items()}


def peer_bytes(tokens: int, tp: int) -> int:
    layout = peer_layout(tokens, tp)
    end = max(offset + size for offset, size in layout.values())
    return (end + ALIGN - 1) // ALIGN * ALIGN
