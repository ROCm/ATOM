# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Exact decode-frontier contracts shared by ordinary and MTP3 serving."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuationRow:
    """One exact live row in a TP-agreed decode frontier."""

    slot: int
    slot_generation: int
    position: int
    blocks: tuple[int, ...]


@dataclass(frozen=True)
class BatchContinuation:
    """Exact-width graph continuation for one stable decode membership."""

    owner_generation: int
    graph_key: tuple[int, int]
    graph_generation: int
    rows: tuple[ContinuationRow, ...]

    @property
    def width(self) -> int:
        return len(self.rows)


def continues_batch(
    previous: BatchContinuation | None,
    graph_key: tuple[int, int],
    current: tuple[tuple[int, int, tuple[int, ...]], ...] | None,
) -> bool:
    """Whether every request continues inside the same captured q window.

    Ordinary decode has ``q=1`` and therefore still requires an exact +1
    position advance.  An MTP verify window may commit any prefix of its q
    rows, so the next window may start 1..q positions after the previous one.
    The block frontier is recorded through the end of each window; q is far
    smaller than one cache block, so a stable request can append at most one
    block between adjacent windows.
    """
    if (
        previous is None
        or current is None
        or graph_key != previous.graph_key
        or len(current) != previous.width
    ):
        return False
    max_advance = graph_key[1]
    for old, (position, slot, blocks) in zip(previous.rows, current):
        if (
            slot != old.slot
            or not 1 <= position - old.position <= max_advance
            or len(blocks) not in (len(old.blocks), len(old.blocks) + 1)
            or blocks[: len(old.blocks)] != old.blocks
        ):
            return False
    return True
