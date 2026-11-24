# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The scheduler's own record of what the CPU tier holds, for ranking only.

``ATOM_PREFILL_ORDER=cached_first`` ranks waiting requests by the prompt tokens
left to compute, so it needs each one's tier hit before admission. The tier's
answer comes from a lookup, which pins the hit chunks on every worker and
refreshes them in the tier's eviction order; admission asks it once, for the
requests it reaches. A request further back would rank as if the tier held
nothing, wait behind colder ones, and could lose its tier copy meanwhile.

:class:`TierIndex` answers without asking: the chunk hashes the tier is known
to hold, from what this scheduler stored there (a lazy demotion's report) and
what lookups found, least recently touched first. It is capped at the chunks
the tier's budget holds (`lmcache_tier_horizon_tokens`), dropping the oldest,
and a lookup that stops short of an indexed chunk drops that chunk. LMCache's
own evictions are not reported, so it can still claim a chunk the tier lost:
the request ranks early, admission's lookup finds less and it computes the
rest, as it would have in arrival order. Nothing here touches the tier.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable


class TierIndex:
    """Chunk hashes the tier is known to hold, oldest first, at most `capacity`."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"tier index capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self._chunks: OrderedDict[int, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_hash: int) -> bool:
        return chunk_hash in self._chunks

    def touch(self, hashes: Iterable[int]) -> None:
        """The tier stored or served these chunks: newest now."""
        chunks = self._chunks
        for chunk_hash in hashes:
            chunks[chunk_hash] = None
            chunks.move_to_end(chunk_hash)
        while len(chunks) > self.capacity:
            chunks.popitem(last=False)

    def observe_lookup(self, hashes: list[int], run: int) -> None:
        """A lookup of `hashes` found their first `run`: those are held, the
        next is not."""
        run = max(0, min(int(run), len(hashes)))
        self.touch(hashes[:run])
        if run < len(hashes):
            self._chunks.pop(hashes[run], None)
