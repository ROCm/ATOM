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
vLLM hashes like any other group. The LMCache MP worker registers those groups
as engine groups beside the PAGE planes, so a transfer names one block in each
and the checkpoint commits and restores as part of the PAGE object it
continues. This module is the scheduler half: it decides which boundary a
transfer may carry and caps hits to one. The blocks themselves are resolved by
`PageRideMixin`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from .page_ride import NULL_BLOCK_ID, PageRideMixin

logger = logging.getLogger("atom")


#: Whether a checkpoint may back a LOCAL prefix hit. Separate from whether
#: checkpoints are wired, so the two halves can be measured apart: storing and
#: restoring checkpoints is testable without opening the gate that lets vLLM
#: serve a hit which -- if no checkpoint happens to sit at that exact position --
#: is a dead engine rather than a slow one.
_CHECKPOINT_LOCAL_HITS_ENABLED = os.environ.get("ATOM_V41_CHECKPOINT_LOCAL_HITS") in (
    "1",
    "true",
    "True",
)


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
        keep_a_prefill_tail: bool = False,
    ) -> None:
        self.hash_block_size = int(hash_block_size)
        # Two different things that used to be one.
        #
        # `hash_block_size` is the unit vLLM's `request.block_hashes` are in;
        # indexing that list with anything else reads the wrong entry.
        # `boundary_grid` is the grid a STATE boundary may sit on, which for
        # V4.1 is its PAGE block: state is only restorable at a whole PAGE
        # edge. They coincide while there is one kv cache group and diverge
        # the moment there are two, because the hash unit is engine-wide and
        # becomes the smallest group's -- with DSpark, 32 against the proxy's
        # 256. Measured: `cap_hit` then accepted a boundary at 96 tokens,
        # which is a legal 32-grid point, is not a PAGE edge, and floors to
        # zero chunks at the caller. The hit it capped was 1280.
        self.boundary_grid = int(boundary_grid or hash_block_size)
        if self.boundary_grid % self.hash_block_size:
            raise ValueError(
                f"DeepSeek-V4.1 state boundary grid {self.boundary_grid} is "
                f"not a multiple of the hash block size "
                f"{self.hash_block_size}; a boundary on that grid could not "
                "be hashed"
            )
        self.chunk_size = int(chunk_size)
        # With speculative decoding, a hit must not leave exactly one token to
        # compute. vLLM admits such a request as a decode row and pads it to
        # `1 + num_spec_tokens` with placeholder drafts (`pad_spec_decode`);
        # its only guard is against padding *after* a prefill was scheduled,
        # not against a prefill being admitted after it -- and V4.1 refuses a
        # step that verifies and prefills. A tail of two or more is a plain
        # prefill, which the scheduling patch keeps out of verification steps.
        self.keep_a_prefill_tail = bool(keep_a_prefill_tail)
        # The groups holding the checkpoint, in the order the worker registered
        # them -- that order is the key space, so it is taken from the caller
        # rather than rebuilt here.
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
        # Decided by the caller, which holds the VllmConfig. Asking vLLM for
        # its *current* config from here returns None -- the scheduler calls
        # `cap_local_hit` outside any `set_current_vllm_config` scope -- so a
        # check made here was false on every request and this gate never
        # opened (measured: local_cap_checkpoint=0 with ATOM_V41_CHECKPOINT_LOCAL_HITS=1).
        self._state_checkpoints_can_serve = bool(state_checkpoints_can_serve)
        self._lookup_ctx: tuple[str, Any] | None = None
        self._counters: dict[str, int] = {
            "cap_kept": 0,
            "cap_declined": 0,
            "cap_unarmed": 0,
            # The same pair, for hits vLLM served from its own pool without
            # asking the connector. Kept apart because they answer different
            # questions: the connector pair measures the tier, this pair
            # measures what prefix caching costs on a model whose state the
            # block pool carries only through checkpoints.
            "local_cap_declined": 0,
            #: Local hits passed through whole because a checkpoint
            #: can restore the state at the hit. Zero while checkpoints are off.
            "local_cap_checkpoint": 0,
            # Ride hits pulled one grid back so a speculative run never
            # resumes with a single token left; see `keep_a_prefill_tail`.
            "cap_prefill_tail": 0,
        }
        # Seeds the `ride_*` counters too; see `page_ride.RIDE_COUNTERS`.
        self._init_page_ride()
        # One warning per planner, not per hit: an unarmed cap fires once
        # per request on whatever path is missing its arming.
        self._warned_unarmed = False
        self._last_stats_log = 0.0

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
            # Fail closed. An unarmed hook used to return the hit untouched,
            # so an external hit that never reached a boundary test was served
            # whole: PAGE restored to a position whose STATE was never
            # written, and `_report_stale_state` then killing the engine
            # ("needs state at 6912, found 3").
            self._counters["cap_unarmed"] += 1
            if not self._warned_unarmed:
                self._warned_unarmed = True
                logger.warning(
                    "V4.1 state offload: the hit cap fired outside a lookup "
                    "(seq=%s, hit=%d) and declined it. Every external hit has "
                    "to pass a boundary test, so a path that caps without "
                    "arming is a path that would restore PAGE bytes with no "
                    "STATE behind them.",
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
        if self.keep_a_prefill_tail and total and total - boundary < 2:
            # Measured: a 1025-token prompt hit at 1024 came back padded to
            # a `(1024, 6, 5)` verify row, beside a resumed `(768, 200, 0)`
            # prefill, and the batch was refused. One grid back costs a
            # PAGE of recompute for prompts that end one token past a
            # boundary.
            boundary -= self.boundary_grid
            self._counters["cap_prefill_tail"] += 1
        if boundary <= 0:
            self._counters["cap_declined"] += 1
            return 0
        self._counters["cap_kept"] += 1
        return min(hit, boundary)

    def counters(self) -> dict:
        return dict(self._counters)

    def cap_local_hit(self, request, hit: int) -> int:
        """Pass vLLM's *own* prefix-cache hit only if a checkpoint backs it.

        The connector-side `cap_hit` only sees hits vLLM asked it about. With
        prefix caching on, vLLM's block pool answers first and keeps whatever
        it can serve locally -- a path the connector is never consulted on. A
        request admitted that way arrives with its PAGE prefix restored and a
        state slot that was never written, which `_report_stale_state` turns
        into a dead engine. So a local hit is served only when the checkpoint
        group already reduced it to a boundary it holds a checkpoint for, and
        declined otherwise -- which sends the whole prefix through the tier,
        with its state.
        """
        hit = int(hit)
        self._counters["local_cap_calls"] = self._counters.get("local_cap_calls", 0) + 1
        if hit <= 0:
            return hit
        if self._state_checkpoints_can_serve and _CHECKPOINT_LOCAL_HITS_ENABLED:
            # Left alone. vLLM has already reduced this hit to a boundary
            # the state group can serve: its coordinator takes the minimum
            # across groups, and the checkpoint group only offers boundaries it
            # has a cached checkpoint block for (measured: the group
            # answered 1280 for a 1345-token prompt whose checkpoint is at
            # 1280).
            #
            # Capping it here manufactured the very mismatch this branch
            # was written to avoid. `get_mamba_prefill_checkpoint_position`
            # is a rule about a raw prefix length; applied to an
            # already-snapped hit it returns one block lower, which is a
            # position nothing wrote -- `needs state at 1024, found 3`.
            self._counters["local_cap_checkpoint"] += 1
            return hit
        # Declining is the only position a request may legally sit at besides
        # a stored boundary, and the tier then serves the whole prefix with
        # its checkpoint.
        self._counters["local_cap_declined"] += 1
        return 0

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
        boundary = int(num_total_computed)
        if boundary <= 0 or boundary % self.hash_block_size:
            return False
        block_ids = self._boundary_blocks(group_blocks, boundary)
        if not block_ids:
            self._counters["ride_missing"] += 1
            return False
        self._record_ride_boundary(str(request.request_id), boundary, block_ids)
        return True

    def forget_request(self, req_id: str) -> None:
        """Drop a finished or preempted request's recorded destinations.

        A preempted request reallocates, so its old blocks name someone else's
        bytes.
        """
        self.forget_ride(req_id)

    # -- observability ---------------------------------------------------
    def stats(self) -> dict[str, int]:
        return dict(self._counters)

    #: Seconds between stat lines.
    _STATS_INTERVAL = 60.0

    def log_stats(
        self, *, interval_s: float | None = None, force: bool = False
    ) -> None:
        """Say what this leg did, including when it did nothing.

        Unconditional, for the reason K3's counters exist: a run that produced
        exactly zero joint hits otherwise produces exactly zero lines saying
        so, and a by-construction null reads as a quiet success.
        """
        now = time.monotonic()
        if interval_s is None:
            interval_s = self._STATS_INTERVAL
        if not force and now - self._last_stats_log < interval_s:
            return
        self._last_stats_log = now
        logger.info(
            "ATOM LMCache offload: V4.1 state leg %s",
            ", ".join(f"{k}={v}" for k, v in sorted(self.stats().items())),
        )
