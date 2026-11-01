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

from .boundary_planner import BoundaryPlanner
from .page_ride import PageRideMixin

logger = logging.getLogger("atom")


def _state_checkpoints_can_serve(vllm_config) -> bool:
    """Whether a V4.1 state checkpoint may back a LOCAL prefix hit.

    Takes the config: the scheduler calls `cap_local_hit` outside any
    `set_current_vllm_config` scope. Where the checkpoint module cannot be
    imported, local hits are refused.
    """
    try:
        from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
            register_v41_state_checkpoint_adapter,
        )
        from atom.plugin.vllm.paged_state_checkpoint import state_checkpoints_on

        return bool(
            vllm_config is not None
            and state_checkpoints_on(
                register_v41_state_checkpoint_adapter(), vllm_config
            )
        )
    except ImportError:
        # Absent mechanism means no checkpoints; any other error is a real
        # fault and must not quietly turn every V4.1 HBM hit off.
        return False


def _refuse_ride_without_dense_checkpoints(chunk: int) -> None:
    """A save needs a state checkpoint at the boundary it ends on.

    One save spans many chunks and carries one checkpoint image, the one
    at its end; its earlier chunks name the null block, and the MP server
    stores no image for them. The server serves a hit length only if every
    image group holds the length's last chunk (with
    `--separate-object-groups`, which keeps the image groups apart from
    PAGE), so a hit never ends inside a save: the hit grain is the set of
    saved forward ends, not the chunk. `cap_hit` relies on that.

    What has to hold on this side is that a save's end has a checkpoint
    at all. A prefill forward ends on a rung of the checkpoint interval
    (`Placement.cut`) and a save ends on a chunk boundary; with the two
    equal they coincide. With V4.1's default interval of 8192 against a
    chunk of 256, most saves would end where no forward did, find no
    checkpoint, and be dropped -- correct, but a tier that fills only at
    interval multiples and says nothing about it. Refused instead, naming
    the setting that fixes it.
    """
    from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
        V41CheckpointPlacement,
    )

    placement = V41CheckpointPlacement.from_env()
    interval = int(placement.interval)
    chunk = int(chunk)
    if interval == chunk:
        return
    raise ValueError(
        "ATOM offload connector: DeepSeek-V4.1 state cannot ride the PAGE "
        f"object with a checkpoint interval of {interval} tokens against "
        f"an LMCache chunk of {chunk}. A save carries the checkpoint at the "
        "boundary it ends on and a prefill forward ends on a rung of the "
        "interval, so with the two unequal most saves find no checkpoint "
        "and are dropped. Bring the two together from either side -- "
        f"ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS={chunk}, or an LMCache "
        f"chunk of {interval} on the server and the client, which is the "
        "cheaper one: the interval decides how often a checkpoint is "
        "written, while the chunk only decides how coarse a tier hit is, "
        "and a V4.1 hit is already quantised to a checkpoint."
    )


class V41BoundaryPlanner(BoundaryPlanner, PageRideMixin):
    """Scheduler half: picks the boundary a transfer carries, and caps hits to it.

    Everything this decides is about keeping one invariant: a request that vLLM
    is told has `n` computed tokens must have a STATE checkpoint whose cursor
    reads exactly `n`. A hit that does not end on a checkpoint boundary is
    shortened to one (`cap_hit`), and a load is only admitted with the
    checkpoint blocks it restores named (`resolve_load`).
    """

    leg_name = "V4.1 state offload"

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
        # The groups holding the checkpoint are in image unit order -- that
        # order is the key space, so it is taken from the caller rather than
        # rebuilt here.
        super().__init__(
            boundary_grid=int(boundary_grid or hash_block_size),
            boundary_group_ids=checkpoint_group_ids,
        )
        self.hash_block_size = int(hash_block_size)
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
        # `PageRideMixin`'s names for the same geometry. A V4.1 boundary may
        # sit only on a whole PAGE edge, which with a DSpark draft group is
        # coarser than the engine-wide hash unit.
        self.ride_group_ids = self.boundary_group_ids
        self.ride_boundary_grid = self.boundary_grid
        if not self.boundary_group_ids:
            raise ValueError(
                "DeepSeek-V4.1 state offload: the checkpoint cannot ride the PAGE "
                "object with no checkpoint group to name it; a transfer would "
                "carry KV whose state is someone else's"
            )
        # Decided by the caller, which holds the VllmConfig: the scheduler
        # calls `cap_local_hit` outside any `set_current_vllm_config` scope.
        self._state_checkpoints_can_serve = bool(state_checkpoints_can_serve)
        self._init_page_ride()
        # One warning per planner, not per hit: an unarmed cap fires once
        # per request on whatever path is missing its arming.
        self._warned_unarmed = False

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config,
        kv_cache_config,
        *,
        scheduler,
        page_block_size: int,
        checkpoint_group_ids: tuple[int, ...],
        request_of,
    ) -> V41BoundaryPlanner:
        """Stand up the scheduler half of V4.1's STATE leg, bound to *scheduler*.

        *page_block_size* is the proxy group's block size (0 when unknown),
        and ``request_of(seq)`` the vLLM request a scheduler ``SeqView``
        stands for, or None.
        """
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        _, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
        # The grid a STATE boundary may sit on is V4.1's PAGE block, which
        # is the attention group's. `hash_block_size` is engine-wide and
        # becomes the smallest group's, so with a draft group the two part
        # company and a boundary chosen on the hash grid is not a PAGE edge.
        grid = int(page_block_size or 0) or int(hash_block_size)
        # When checkpoints are registered the two mechanisms have to agree on one
        # grid: the planner decides where a boundary may sit, and the checkpoint
        # group decides which column vLLM hashes. A boundary the planner
        # accepts on a different grid is a store nothing can look up, and the
        # symptom is a silent zero hit rate rather than an error.
        from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
            register_v41_state_checkpoint_adapter,
        )

        adapter = register_v41_state_checkpoint_adapter()
        checkpoint_block = int(adapter.block_size)
        if checkpoint_block and checkpoint_block != grid:
            raise ValueError(
                f"ATOM LMCache offload: V4.1 boundary grid {grid} does not match the "
                f"state checkpoint block {checkpoint_block}. A boundary claimed on one grid is "
                "hashed on the other, so nothing stored would ever be found."
            )
        chunk = int(scheduler.chunk_size)
        _refuse_ride_without_dense_checkpoints(chunk)
        planner = cls(
            hash_block_size=int(hash_block_size),
            boundary_grid=grid,
            chunk_size=chunk,
            state_checkpoints_can_serve=_state_checkpoints_can_serve(vllm_config),
            checkpoint_group_ids=checkpoint_group_ids,
            prefill_tail=adapter.prefill_tail(vllm_config),
        )
        scheduler.bind_planner(planner, request_of)
        logger.info(
            "ATOM LMCache offload: V4.1 state leg on (hash_block=%d, chunk=%d, "
            "checkpoint_groups=%s)",
            int(hash_block_size),
            chunk,
            ",".join(str(g) for g in checkpoint_group_ids) or "-",
        )
        return planner

    # -- lookup ----------------------------------------------------------
    def _cap_unarmed(self, seq, hit: int) -> int:
        """Decline a hit reported outside any lookup.

        Returning a shorter hit costs a recompute. Returning a longer one
        hands the forward a cursor that disagrees with the scheduler, which
        `_report_stale_state` turns into a dead engine -- so when the hit
        cannot be tested the answer is 0, never *hit*.
        """
        # Fail closed: an external hit that never reached a boundary test
        # would restore PAGE to a position whose STATE was never written.
        if not self._warned_unarmed:
            self._warned_unarmed = True
            logger.warning(
                "%s: the hit cap fired outside a lookup (seq=%s, hit=%d); "
                "declining the external hit",
                self.leg_name,
                getattr(seq, "id", None),
                hit,
            )
        return 0

    def _cap_armed(self, request, hit: int) -> int:
        """Shorten *hit* to the last PAGE edge, where a checkpoint can sit.

        No index to probe: the KV and the checkpoint that continues it are one
        object, so whatever the dense lookup reported present is present with
        its checkpoint. What still has to hold is the *shape* -- a hit must end
        on a whole PAGE edge for there to be a stored checkpoint at all -- and
        that is what this truncation is.
        """
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
    def admit_load(
        self,
        request,
        *,
        group_blocks,
        num_total_computed: int,
        num_external_tokens: int,
        attention_group_id: int,
        attention_block_size: int,
    ) -> None:
        """Name the checkpoint blocks an admitted PAGE load restores.

        The answer of `resolve_load` is not passed on: when it cannot name
        them, the scheduler finds no destination in `take_ride_units` and
        declines the load, and the connector withdraws the promise
        (`_settle_promised_loads`) -- the request recomputes rather than
        resuming on a state nobody restored.
        """
        self.resolve_load(request, num_total_computed, group_blocks=group_blocks)

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

    def has_pending_work(self) -> bool:
        """A ride pin is given back by a step's reconciliation
        (`V41StateMPScheduler.build_connector_meta`), so one still held needs
        the engine to keep stepping."""
        return self.has_ride_pins()

    def has_pending_frees(self, horizon_s: float) -> bool:
        """Only a recent pin that is a block's last holder is blocks about to
        come back; retiring any other drops a share and frees nothing. The pin
        is retired in the step's metadata build, after vLLM asks this."""
        return self.has_ride_pins(max_age_s=horizon_s, sole_holder=True)
