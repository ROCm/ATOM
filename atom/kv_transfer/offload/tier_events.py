# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The in-process LMCache CPU tier in the engine's KV events (`medium="CPU"`).

A router that follows the KV events indexes these chunks beside the HBM blocks,
one event block per chunk. A chunk is Stored once every rank stored it. LMCache
reports no evictions, so a chunk is Removed when a lookup asked after it was
announced stops short of it, or when more chunks are announced than the cap.

A chunk is named by the HBM prefix hash of its last hash block, which covers its
whole prefix, so the scheduler hashes nothing for it.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict

logger = logging.getLogger("atom")

# About 134M tokens of 256-token chunks, far more than a CPU tier holds.
MAX_ANNOUNCED_CHUNKS = 1 << 19


class TierEvents:
    """The Stored and Removed events of one CPU tier's chunks."""

    def __init__(
        self, block_manager, chunk_tokens: int, max_chunks: int = MAX_ANNOUNCED_CHUNKS
    ) -> None:
        self._blocks = block_manager
        self._chunk = int(chunk_tokens)
        self._per_chunk = self._chunk // int(block_manager.hash_block_size)
        self._max_chunks = int(max_chunks)
        # Chunk hash -> when it was announced, the oldest first.
        self._announced: OrderedDict[int, float] = OrderedDict()

    @classmethod
    def for_tier(cls, block_manager, chunk_tokens: int) -> TierEvents | None:
        """The events of a tier of `chunk_tokens` chunks; None unless
        `block_manager` records KV events and its hash blocks tile a chunk."""
        if block_manager is None or not getattr(
            block_manager, "kv_events_enabled", False
        ):
            return None
        hash_block = int(block_manager.hash_block_size)
        if int(chunk_tokens) % hash_block:
            logger.warning(
                "Not publishing the CPU tier's KV events: its %d-token chunk is "
                "not a whole number of %d-token hash blocks",
                chunk_tokens,
                hash_block,
            )
            return None
        return cls(block_manager, chunk_tokens)

    def _hashes(self, seq, chunks: int) -> list[int]:
        """The hashes of `seq`'s first `chunks` chunks."""
        per = self._per_chunk
        chain = self._blocks.prefix_hash_chain(seq, [], chunks * per)
        return chain[per - 1 :: per]

    def stored(self, seq, start: int, end: int) -> None:
        """Every rank stored `seq`'s tokens `[start, end)`: announce the whole
        chunks among them."""
        chunk = self._chunk
        first, last = -(-int(start) // chunk), int(end) // chunk
        if last <= first:
            return
        hashes = self._hashes(seq, last)
        now = time.monotonic()
        for h in hashes[first:]:
            self._announced[h] = now
            self._announced.move_to_end(h)
        self._blocks.record_offload_store(
            hashes[first:],
            list(seq.token_ids[first * chunk : last * chunk]),
            hashes[first - 1] if first else None,
            chunk,
            token_offset=first * chunk,
            cache_seed=getattr(seq, "cache_seed", -1),
        )
        over = len(self._announced) - self._max_chunks
        if over > 0:
            self._removed([self._announced.popitem(last=False)[0] for _ in range(over)])

    def looked_up(self, seq, extent: int, hit: int, asked: float) -> None:
        """A lookup of `seq`'s first `extent` tokens, asked at `asked`
        (`time.monotonic`), found the tier holds `hit` of them: the chunks it
        stops short of are gone, if they were announced before it was asked."""
        chunk = self._chunk
        held, chunks = int(hit) // chunk, int(extent) // chunk
        if held >= chunks or not self._announced:
            return
        announced = self._announced
        self._removed(
            [
                h
                for h in self._hashes(seq, chunks)[held:]
                if announced.get(h, math.inf) < asked
            ]
        )

    def _removed(self, hashes: list[int]) -> None:
        for h in hashes:
            self._announced.pop(h, None)
        self._blocks.record_offload_removed(hashes)
