# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's per-request CSA2 state, carried inside the PAGE object.

V4.1 buys its cache in two currencies. PAGE bytes scale with history and are
what the byte codec moves. STATE bytes do not: every in-flight request owns one
fixed-size entry holding a window ring per layer, the compressor's incomplete
group per owner, and the Engram cursor. A restored PAGE prefix whose STATE was
not restored with it is not a degraded answer -- `PagedAttentionCache` refuses
it outright ("replay from a recoverable boundary"), because the cursor it finds
is not the position the scheduler claims. There is no partial reconstruction:
the window ring holds raw per-token rows the compressed pages do not contain.

So the two travel together or not at all. State checkpoints copy the entry into
PAGE units at chunk boundaries, in blocks of single-layer checkpoint groups that
vLLM hashes like any other group; the block in checkpoint group ``j`` is unit
``j`` of the native checkpoint image. A transfer names those units as a
native-state image (`v41_mp_scheduler`), so the checkpoint commits and restores
as part of the PAGE object it continues. This module is the scheduler half: it
decides which boundary a transfer may carry and caps hits to one. The units
themselves are resolved by `PageRideMixin`.
"""

from __future__ import annotations

import logging
from typing import Any

from .page_ride import NULL_BLOCK_ID, PageRideMixin

logger = logging.getLogger("atom")


class V41BoundaryPlanner(PageRideMixin):
    """Scheduler half: picks the boundary a transfer carries, and caps hits to it.

    Everything this decides is about keeping one invariant: a request that vLLM
    is told has `n` computed tokens must have a STATE checkpoint whose cursor
    reads exactly `n`. A hit that does not end on a checkpoint boundary is
    shortened to one (`cap_hit`), and a load is only admitted with the
    checkpoint blocks it restores named (`resolve_load`).
    """

    def __init__(
        self,
        *,
        hash_block_size: int,
        boundary_grid: int | None = None,
        chunk_size: int,
        state_checkpoints_can_serve: bool = False,
        checkpoint_group_ids: tuple[int, ...] = (),
        prefill_tail: int = 1,
    ) -> None:
        # `hash_block_size` is the unit `request.block_hashes` is in;
        # `boundary_grid` is where a STATE boundary may sit (V4.1's PAGE
        # block). They differ once a second kv cache group makes the hash unit
        # the smallest group's -- with DSpark, 32 against the proxy's 256.
        self.hash_block_size = int(hash_block_size)
        self.boundary_grid = int(boundary_grid or hash_block_size)
        if self.boundary_grid % self.hash_block_size:
            raise ValueError(
                f"DeepSeek-V4.1 state boundary grid {self.boundary_grid} is "
                f"not a multiple of the hash block size "
                f"{self.hash_block_size}; a boundary on that grid could not "
                "be hashed"
            )
        self.chunk_size = int(chunk_size)
        # Tokens a hit leaves to compute (`PagedStateCheckpointAdapter.
        # prefill_tail`). Two under speculative decoding: a hit leaving one is
        # admitted as a decode row padded to `1 + num_spec_tokens` with
        # placeholder drafts (`pad_spec_decode`), whose only guard is against
        # padding *after* a prefill was scheduled -- and V4.1 refuses a step
        # that verifies and prefills. Two or more is a plain prefill, which the
        # scheduling patch keeps out of verification steps.
        self.prefill_tail = int(prefill_tail)
        # The groups holding the checkpoint, in image unit order -- that order
        # is the key space, so it is taken from the caller rather than rebuilt
        # here.
        self.ride_group_ids = tuple(int(g) for g in checkpoint_group_ids)
        # A V4.1 boundary may sit only on a whole PAGE edge, which with a
        # DSpark draft group is coarser than the engine-wide hash unit.
        self.ride_boundary_grid = self.boundary_grid
        if not self.ride_group_ids:
            raise ValueError(
                "DeepSeek-V4.1 state offload: the checkpoint cannot ride the PAGE "
                "object with no checkpoint group to name it; a transfer would "
                "carry KV whose state is someone else's"
            )
        # Decided by the caller, which holds the VllmConfig: the scheduler
        # calls `cap_local_hit` outside any `set_current_vllm_config` scope.
        self._state_checkpoints_can_serve = bool(state_checkpoints_can_serve)
        self._lookup_ctx: tuple[str, Any] | None = None
        self._init_page_ride()
        # One warning per planner, not per hit: an unarmed cap fires once
        # per request on whatever path is missing its arming.
        self._warned_unarmed = False

    # -- lookup ----------------------------------------------------------
    def begin_lookup(self, request) -> None:
        self._lookup_ctx = (str(request.request_id), request)

    def end_lookup(self) -> None:
        self._lookup_ctx = None

    def cap_hit(self, seq, hit: int) -> int:
        """Shorten *hit* to the last PAGE edge, where a checkpoint can sit.

        No index to probe: the KV and the checkpoint that continues it are one
        object, so whatever the dense lookup reported present is present with
        its checkpoint. What still has to hold is the *shape* -- a hit must end
        on a whole PAGE edge for there to be a stored checkpoint at all -- and
        that is what this truncation is.

        Returning a shorter hit costs a recompute. Returning a longer one
        hands the forward a cursor that disagrees with the scheduler, which
        `_report_stale_state` turns into a dead engine -- so when the hit
        cannot be tested the answer is 0, never *hit*.
        """
        hit = int(hit)
        if hit <= 0:
            return hit
        if self._lookup_ctx is None:
            # Fail closed: an external hit that never reached a boundary test
            # would restore PAGE to a position whose STATE was never written.
            if not self._warned_unarmed:
                self._warned_unarmed = True
                logger.warning(
                    "V4.1 state offload: the hit cap fired outside a lookup "
                    "(seq=%s, hit=%d); declining the external hit",
                    getattr(seq, "id", None),
                    hit,
                )
            return 0
        req_id, request = self._lookup_ctx
        if str(seq.id) != req_id:
            logger.warning(
                "V4.1 state offload: hit cap armed for %s but called for %s; "
                "declining the external hit",
                req_id,
                seq.id,
            )
            return 0
        boundary = (hit // self.boundary_grid) * self.boundary_grid
        total = int(getattr(request, "num_tokens", 0) or 0)
        if total and total - boundary < self.prefill_tail:
            # A one-token tail would be admitted as a padded verify row; one
            # grid back keeps it a prefill, at a PAGE of recompute.
            boundary -= self.boundary_grid
        if boundary <= 0:
            return 0
        return min(hit, boundary)

    def cap_local_hit(self, request, hit: int) -> int:
        """vLLM's *own* prefix-cache hit, served only if a checkpoint backs it.

        vLLM has already reduced the hit to a boundary the checkpoint groups
        hold a checkpoint for: its coordinator takes the minimum across groups,
        and a checkpoint group offers only boundaries it has a cached block
        for. So with checkpoint groups the hit stands as it is (capping it
        again would land on a position nothing wrote). Without them nothing
        could restore the state, and the hit is declined.
        """
        hit = int(hit)
        if hit <= 0:
            return hit
        if not self._state_checkpoints_can_serve:
            return 0
        total = int(getattr(request, "num_tokens", 0) or 0)
        if total and total - hit < self.prefill_tail:
            # `cap_hit`'s rule: a one-token tail is admitted as a padded verify
            # row. The scheduler patch has already re-asked vLLM for a hit that
            # leaves two (`apply_vllm_v41_local_hit_cap_patch`), so this is the
            # guard for a vLLM that cannot be asked: declined rather than moved
            # a PAGE back, where no checkpoint need exist.
            return 0
        return hit

    # -- load ------------------------------------------------------------
    def _boundary_blocks(
        self, group_blocks, num_total_computed: int
    ) -> tuple[int, ...]:
        """This boundary's destination block in each checkpoint group, or ``()``.

        All or nothing: a checkpoint scattered into some of its groups leaves the
        rest holding the previous occupant's state, which is the half restore
        this module exists to prevent. One unresolved group fails the load.
        """
        if not group_blocks or num_total_computed % self.boundary_grid:
            return ()
        row = (int(num_total_computed) - 1) // self.boundary_grid
        if row < 0:
            return ()
        block_ids: list[int] = []
        for group_id in self.ride_group_ids:
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

    def resolve_load(self, request, num_total_computed: int, group_blocks=()) -> bool:
        """Name the checkpoint blocks a restored PAGE prefix fills.

        The PAGE load restores these same blocks as part of its own object, so
        there is no job to queue -- only the destination to name, which is
        what the transfer spec carries. False means the destination could not
        be resolved; the caller must then not let the load run, because the
        admission has already happened and PAGE alone would leave the request
        on someone else's state.
        """
        # Every admission reallocates, so what an earlier one recorded names
        # blocks vLLM has since freed (a failed or withdrawn load sends the
        # request back to WAITING). Dropped before the early returns: a
        # failure here must not leave the old destination for
        # `take_ride_units` to find.
        self.forget_ride(str(request.request_id))
        boundary = int(num_total_computed)
        if boundary <= 0 or boundary % self.hash_block_size:
            return False
        block_ids = self._boundary_blocks(group_blocks, boundary)
        if not block_ids:
            return False
        self._record_ride_boundary(str(request.request_id), boundary, block_ids)
        return True

    def forget_request(self, req_id: str) -> None:
        """Drop a finished or preempted request's recorded destinations.

        A preempted request reallocates, so its old blocks name someone else's
        bytes.
        """
        self.forget_ride(req_id)
