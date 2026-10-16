# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The boundary blocks a PAGE transfer carries alongside its KV.

Under the ``mp`` backend a recurrent snapshot is not a leg of its own: it is
registered as an engine group beside the PAGE planes, and every transfer names
one block per group. The scheduler asks for those blocks through
``install_recurrent_state_hook``; this mixin is the answer, and it is the same
answer for every backend whose state is a vLLM-hashed, block-addressed
boundary -- Kimi-K3's KDA groups and DeepSeek-V4.1's state-checkpoint groups
alike. The two differ only in which group ids hold the boundary and on what
grid a boundary may sit, which is why those are attributes rather than code.

The host planner must provide:

``self._pool``
    vLLM's GPU block pool, or None before :meth:`bind_gpu_block_pool`.
``self._counters``
    The planner's tally dict, pre-seeded with the ``ride_*`` keys below.
``self.hash_block_size``
    The unit ``request.block_hashes`` is indexed in.
``self.ride_boundary_grid``
    The grid a boundary may sit on. A multiple of ``hash_block_size``; equal
    to it when the two do not differ.
``self.ride_group_ids``
    The kv cache groups holding the boundary, in the order the transfer names
    them -- which is the order the worker registered them in, and therefore
    the key space.
``self.boundary_hash(block_hashes, boundary_tokens)``
    The prefix-hash key for a boundary.
"""

from __future__ import annotations

from typing import Any

#: The counters this mixin keeps. Seeded by the host so a planner's stats line
#: has every key whether or not a ride was ever taken -- a key that appears
#: only once it is non-zero reads as "not instrumented" rather than "zero".
RIDE_COUNTERS = (
    "ride_recorded",
    "ride_served",
    "ride_missing",
    # Split by the gate that refused, because one number for three reasons
    # could not say whether a configuration stores nothing or looks nothing up.
    "ride_missing_boundary_unaligned",
    "ride_missing_boundary_past_hashes",
    "ride_missing_no_group_cached",
    "ride_missing_no_pool",
    "ride_pinned",
    "ride_unpinned",
)

#: vLLM's placeholder block. ``vllm.v1.attention.backends.utils.NULL_BLOCK_ID``
#: is the authority; it is mirrored rather than imported so this module stays
#: importable in a unit test that has no vLLM. A block id at or below it is
#: not a real block.
NULL_BLOCK_ID = 0


class PageRideMixin:
    """Resolve, pin and retire the state blocks a PAGE transfer rides with."""

    def _init_page_ride(self) -> None:
        """Set up the ride state. Called from the host's ``__init__``."""
        self._pool = None
        # req_id -> {boundary: blocks}. A load's destination, recorded when it
        # is resolved; see `_record_ride_boundary` for why it is not looked up.
        self._ride_boundaries: dict[str, dict[int, tuple[int, ...]]] = {}
        # operation -> the blocks pinned for it, given back in
        # `retire_ride_pins` once the scheduler no longer lists it as live.
        self._ride_pins: dict[Any, tuple[int, ...]] = {}
        self._last_ride_gate: str | None = None
        for key in RIDE_COUNTERS:
            self._counters.setdefault(key, 0)

    def bind_gpu_block_pool(self, pool) -> None:
        self._pool = pool

    def _record_ride_boundary(
        self, req_id: str, boundary_tokens: int, block_ids: tuple[int, ...]
    ) -> None:
        """Name the destination blocks a PAGE load ending there will fill.

        Recorded rather than looked up, because a load's destination is what
        vLLM just allocated for the external hit -- the hash-keyed pool lookup
        would answer about the committed state, which for a hit is precisely
        what is not there yet.
        """
        if not block_ids:
            return
        self._ride_boundaries.setdefault(str(req_id), {})[int(boundary_tokens)] = tuple(
            block_ids
        )
        self._counters["ride_recorded"] += 1

    def forget_ride(self, req_id: str) -> None:
        """Drop a finished request's recorded destinations."""
        self._ride_boundaries.pop(str(req_id), None)

    def source_blocks_for(self, request, boundary_tokens: int) -> tuple[int, ...]:
        """This boundary's committed state block in each group, or ``()``.

        Keyed by the same ``BlockHash`` vLLM uses to serve a local hit, never
        by indexing a block table: a block that was superseded, freed, nulled
        or relocated is no longer registered under that hash, so a stale row
        cannot be mistaken for a live boundary. All groups or none -- a state
        stored for some of them is an image that cannot be restored.
        """
        pool = self._pool
        self._last_ride_gate = "no_pool"
        if pool is None:
            return ()
        block_hashes = getattr(request, "block_hashes", None) or ()
        boundary_tokens = int(boundary_tokens)
        # The grid, not the hash unit: with a draft group the hash unit is the
        # smallest group's and a boundary legal on it need not be one the
        # state can sit on. The grid is a multiple of the hash unit, so the
        # index below is still exact.
        if boundary_tokens <= 0 or boundary_tokens % self.ride_boundary_grid:
            self._last_ride_gate = "boundary_unaligned"
            return ()
        index = boundary_tokens // self.hash_block_size - 1
        if index < 0 or index >= len(block_hashes):
            self._last_ride_gate = "boundary_past_hashes"
            return ()
        blocks = pool.get_cached_block(block_hashes[index], list(self.ride_group_ids))
        if not blocks:
            self._last_ride_gate = "no_group_cached"
            return ()
        block_ids = tuple(int(block.block_id) for block in blocks)
        if any(block_id <= NULL_BLOCK_ID for block_id in block_ids):
            self._last_ride_gate = "no_group_cached"
            return ()
        return block_ids

    def take_ride_state(self, request, req_id, end: int, operation=None):
        """The state blocks a transfer ending at *end* should carry.

        Two sources, in this order. A load has already had its destination
        resolved and its blocks allocated by vLLM, so the recorded tuple is
        the only correct answer. A save has not: its source is whatever vLLM
        has committed and still holds under this boundary's hash.

        Only the save source is pinned, and only when the caller names the
        operation that will report it back. A load's destination needs no pin:
        those blocks are the parked request's own allocation, and vLLM holds
        them for as long as it holds the request in WAITING_FOR_REMOTE_KVS,
        which is strictly longer than the H2D.

        ``None`` means this boundary has no whole state. The caller must then
        not build the transfer at all -- a PAGE-only object on a model with
        recurrent groups registered is refused by the transport, which is the
        point: it would restore a prefix whose state is someone else's.
        """
        from atom.kv_transfer.offload.metadata import RecurrentStateTransfer

        end = int(end)
        by_boundary = self._ride_boundaries.get(str(req_id))
        recorded = None if by_boundary is None else by_boundary.get(end)
        block_ids = recorded or (self.source_blocks_for(request, end) or None)
        if not block_ids:
            self._counters["ride_missing"] += 1
            key = f"ride_missing_{getattr(self, '_last_ride_gate', None)}"
            if key in self._counters:
                self._counters[key] += 1
            return None
        self._counters["ride_served"] += 1
        pool = self._pool
        pinnable = (
            not recorded
            and pool is not None
            and operation is not None
            and operation not in self._ride_pins
        )
        if pinnable:
            # The copy runs asynchronously on the worker, and a block vLLM
            # reuses in the meantime would put another prefix's state under
            # this key. Keyed by the operation so `retire_ride_pins` can give
            # the share back on every exit the save has, terminal or not.
            pool.touch([pool.blocks[block_id] for block_id in block_ids])
            self._ride_pins[operation] = tuple(block_ids)
            self._counters["ride_pinned"] += 1
        return RecurrentStateTransfer(boundary_tokens=end, block_ids=tuple(block_ids))

    def retire_ride_pins(self, live_operations) -> None:
        """Give back the shares of every ride whose save the scheduler forgot.

        Reconciliation rather than a release call per exit: a save operation
        leaves the scheduler through a terminal store report, an abandonment, a
        stale-lease reclaim or a request teardown, and a pin that only one of
        those paths released would leak on the others with nothing logged.
        The scheduler's own live set is the single source of truth.
        """
        pool = self._pool
        if not self._ride_pins:
            return
        live = set(live_operations)
        for operation in [op for op in self._ride_pins if op not in live]:
            block_ids = self._ride_pins.pop(operation)
            self._counters["ride_unpinned"] += 1
            if pool is not None:
                pool.free_blocks([pool.blocks[block_id] for block_id in block_ids])

    def has_live_rides(self) -> bool:
        return bool(self._ride_pins)
