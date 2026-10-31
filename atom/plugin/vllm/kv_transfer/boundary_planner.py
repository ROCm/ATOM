# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The rules every recurrent-state planner on the plugin path shares.

A hybrid model's restored KV prefix is correct only if the recurrent state at
the same token boundary is restored with it. Two planners keep that pair
together -- Kimi-K3's KDA leg (`kda_state`) and DeepSeek-V4.1's checkpoint ride
(`v41_state`) -- and they hold it with the same rules: the hit cap is armed per
lookup and declines a hit armed for another request, and a load's destination
is one block per state group at the boundary's row, every group or none. Those
rules are written once, here, so the two legs cannot drift apart on them.

A subclass supplies what genuinely differs: where its boundaries may sit, how a
hit is shortened to one, and which way a cap fired outside a lookup fails.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("atom")

#: vLLM's placeholder block. ``vllm.v1.attention.backends.utils.NULL_BLOCK_ID``
#: is the authority; it is mirrored rather than imported so the planners stay
#: importable in a unit test that has no vLLM. A block id at or below it is not
#: a real block.
NULL_BLOCK_ID = 0


class BoundaryPlanner(ABC):
    """Scheduler half of a recurrent-state leg: caps hits, names load blocks.

    The geometry is given to ``__init__`` by the subclass, after its own
    validation:

    ``boundary_grid``
        Tokens per state block. A boundary sits only on a multiple of it, and
        the state a request resumes from at ``n`` tokens is row
        ``(n - 1) // boundary_grid`` of each state group.
    ``boundary_group_ids``
        The kv cache groups one boundary's state spans, in the order a
        transfer names them.
    """

    #: Prefix of this leg's log lines.
    leg_name = "state offload"

    def __init__(self, *, boundary_grid: int, boundary_group_ids) -> None:
        self.boundary_grid = int(boundary_grid)
        self.boundary_group_ids = tuple(int(g) for g in boundary_group_ids)
        # vLLM's GPU block pool, None until `bind_gpu_block_pool`.
        self._pool = None
        # Set by the connector immediately before it delegates a lookup, and
        # read by `cap_hit` -- ATOM's scheduler hands the hook a SeqView, which
        # carries no block hashes of its own.
        self._lookup_ctx: tuple[str, Any] | None = None

    # -- block pool ------------------------------------------------------
    def bind_gpu_block_pool(self, pool) -> None:
        self._pool = pool

    # -- lookup ----------------------------------------------------------
    def begin_lookup(self, request) -> None:
        self._lookup_ctx = (str(request.request_id), request)

    def end_lookup(self) -> None:
        self._lookup_ctx = None

    def cap_hit(self, seq, hit: int) -> int:
        """Shorten *hit* to a boundary whose state can be restored with it.

        The hook ATOM's scheduler calls on every tier hit
        (`install_hit_cap_hook`). The checks every leg needs run here, in this
        order: a hit of nothing is returned as it is; a hit outside a lookup
        is the leg's call (`_cap_unarmed`); a hit for any request but the one
        the lookup was armed for is declined; the rest is shortened by
        `_cap_armed`. A shorter hit costs a recompute, a longer one restores a
        prefix on somebody else's state.
        """
        hit = int(hit)
        if hit <= 0:
            return hit
        if self._lookup_ctx is None:
            return self._cap_unarmed(seq, hit)
        req_id, request = self._lookup_ctx
        if str(seq.id) != req_id:
            # The hook is armed per lookup; a mismatch means the scheduler
            # called it for a different sequence than the one the connector is
            # in. Refusing the hit is the safe direction.
            logger.warning(
                "%s: hit cap armed for %s but called for %s; "
                "declining the external hit",
                self.leg_name,
                req_id,
                seq.id,
            )
            return 0
        return self._cap_armed(request, hit)

    @abstractmethod
    def _cap_unarmed(self, seq, hit: int) -> int:
        """The cap for a positive *hit* reported outside any lookup.

        Which way this fails is the leg's decision, so there is no default.
        """

    @abstractmethod
    def _cap_armed(self, request, hit: int) -> int:
        """Shorten a positive *hit* for the armed *request* to a boundary."""

    # -- load ------------------------------------------------------------
    def _boundary_blocks(
        self, group_blocks, num_total_computed: int
    ) -> tuple[int, ...]:
        """This boundary's destination block in each state group, or ``()``.

        All or nothing: a state scattered into some of its groups leaves the
        rest holding the previous occupant's state, which is exactly the half
        restore a planner exists to prevent. One unresolved group therefore
        fails the whole load.
        """
        # The row the resuming forward reads its initial state from,
        # ``(num_computed_tokens - 1) // block_size`` (`preprocess_mamba`).
        # Equal to ``num_computed // block_size - 1`` only when num_computed
        # is a whole number of blocks, so any other end has no row.
        num_total_computed = int(num_total_computed)
        if not group_blocks or num_total_computed % self.boundary_grid:
            return ()
        row = (num_total_computed - 1) // self.boundary_grid
        if row < 0:
            return ()
        block_ids: list[int] = []
        for group_id in self.boundary_group_ids:
            if group_id >= len(group_blocks):
                return ()
            blocks = group_blocks[group_id]
            if row >= len(blocks):
                return ()
            block_id = int(blocks[row])
            if block_id <= NULL_BLOCK_ID:
                return ()
            block_ids.append(block_id)
        return tuple(block_ids)
