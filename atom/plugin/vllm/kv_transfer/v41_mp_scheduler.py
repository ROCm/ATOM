# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1's LMCache MP scheduler: every transfer carries its checkpoint.

V4.1's state checkpoints are ``k`` single-layer vLLM KV cache groups (the
checkpoint groups) that overlay the proxy pool from byte 0 at the same block
stride, so the block a checkpoint group holds at a boundary *is* one PAGE unit
of a native `StateCopies` image, and ascending group id is image unit order. A
transfer therefore names its checkpoint the way ATOM's native-state transport
does (`NativeStateTransfer.unit_ids`), and the worker registers and moves it
through `native_state_layout` -- one mechanism for both engines.

What differs from native is who owns the units. vLLM does: a save's source is
the checkpoint vLLM has committed under the boundary's hash, pinned in its pool
while the save may read it (`PageRideMixin`); a load's destination is what vLLM
allocated for the external hit, and the plugin restores the state from those
units itself (`_v41_checkpoint_restore`). So no transfer names an Active SLOT.
"""

from __future__ import annotations

import logging
from typing import Any

from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LoadSpec, NativeStateTransfer
from atom.kv_transfer.offload.mp.scheduler import LMCacheMPConnectorScheduler

logger = logging.getLogger("atom")


class V41StateMPScheduler(LMCacheMPConnectorScheduler):
    """LMCache MP scheduler whose transfers each carry a V4.1 checkpoint image."""

    def __init__(self, config: Any, *, checkpoint_spec: Any) -> None:
        # Images on: the namespace names the answer, the workers reach the same
        # one from `state_replay_tokens` (0 on the plugin: there is no bounded
        # replay to stand in for an image), and a scheduler that disagreed
        # would look up a namespace nobody stores into.
        super().__init__(config, checkpoint_spec=checkpoint_spec, images=True)
        self._planner: Any = None
        self._request_of: Any = None
        # sid -> (seq, frontier): a finished request's save frontier, fixed
        # the first time it is asked for; see `_save_frontier`.
        self._finished_frontiers: dict[str, tuple[Any, int]] = {}

    def bind_planner(self, planner: Any, request_of: Any) -> None:
        """Take the planner that resolves checkpoint units (`V41BoundaryPlanner`)
        and `request_of(seq)`, the vLLM request a SeqView stands for or None.
        Bound before any request is scheduled."""
        self._planner = planner
        self._request_of = request_of

    # -- lookup ----------------------------------------------------------

    def _lookup_token_ids(self, seq: Any) -> list[int]:
        return list(seq.token_ids[: self._lookup_len(seq)])

    def _lookup_len(self, seq: Any) -> int:
        # Truncate the query, not its answer, to the last chunk edge that
        # leaves the prefill tail: the server prefetches the image at the
        # lookup's endpoint, so a hit shortened afterwards would load an image
        # it never prefetched.
        tail = int(self._planner.prefill_tail)
        return self._chunk_floor(max(0, int(seq.num_prompt_tokens) - tail))

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        # Not the MP base's: it extends a whole-prompt hit to the prompt's end,
        # past the checkpoint the hit was capped to, and a PAGE transfer ending
        # there would have no state to continue from.
        return ChunkedOffloadSchedulerBase.get_num_new_matched_tokens(self, seq)

    # -- load ------------------------------------------------------------

    def _load_units(self, seq: Any, end: int) -> tuple[int, ...] | None:
        """The checkpoint units a load ending at *end* fills, or None."""
        request = self._request_of(seq)
        if request is None:
            return None
        return self._planner.take_ride_units(request, str(seq.id), int(end))

    def _decide_load_after_alloc(self, seq: Any, ls: LoadSpec):
        decision = super()._decide_load_after_alloc(seq, ls)
        should_load, _, hbm, lmc, need, chunk = decision
        # `can_load` marks a load vLLM has allocated for, which is also when
        # its destination units are recorded; asked at lookup, before the
        # allocation, there is nothing to check yet.
        if not should_load or not ls.can_load:
            return decision
        end = lmc if ls.transfer_end_tokens is None else int(ls.transfer_end_tokens)
        if self._load_units(seq, end) is None:
            # No destination for the state this KV needs, so no correct load
            # to emit. Declined before the operation is registered, so the
            # request recomputes: a performance loss where the alternative is
            # a wrong answer.
            return False, "no_recurrent_state", hbm, lmc, need, chunk
        return decision

    def _build_load_request(
        self,
        seq: Any,
        load_spec: LoadSpec,
        load_operation: Any,
        transfer_end: int,
    ):
        request = super()._build_load_request(
            seq, load_spec, load_operation, transfer_end
        )
        units = self._load_units(seq, transfer_end)
        if units is None:
            raise RuntimeError(
                f"V4.1 offload load for {seq.id} lost its checkpoint destination "
                "between its admission check and its dispatch"
            )
        request.native_state = NativeStateTransfer(
            units,
            int(transfer_end),
            self._planner.boundary_hash(self._request_of(seq), transfer_end),
        )
        return request

    # -- save ------------------------------------------------------------

    def _save_frontier(self, seq: Any) -> int:
        """A finished request saves to its last boundary that has a checkpoint.

        A save carries one checkpoint, at its end. A running request whose end
        has none yet waits for one; a finished request commits no further
        checkpoint, so it would wait forever, and dropping the save would also
        drop every earlier boundary whose checkpoint still exists. Fixed the
        first time it is asked, at teardown, so the blocks leased for the save
        and the blocks the save reads stay the same range.
        """
        aligned = super()._save_frontier(seq)
        if not getattr(seq, "_offload_finished", False):
            return aligned
        sid = str(seq.id)
        fixed = self._finished_frontiers.get(sid)
        if fixed is not None and fixed[0] is seq:
            return fixed[1]
        frontier = self._last_checkpointed_boundary(seq, aligned)
        self._finished_frontiers[sid] = (seq, frontier)
        return frontier

    def _last_checkpointed_boundary(self, seq: Any, aligned: int) -> int:
        """The highest chunk edge in (saved, aligned] whose checkpoint vLLM
        still holds in every group, or *aligned* when none does (the save is
        then declined and retired, `_unbuilt_save_is_final`)."""
        entry = self._save_tracker.get(str(seq.id))
        saved = int(entry[1]) if entry is not None and entry[0] is seq else 0
        request = self._request_of(seq)
        if request is not None:
            for boundary in range(int(aligned), saved, -int(self.chunk_size)):
                if self._planner.source_blocks_for(request, boundary):
                    return boundary
        return int(aligned)

    def _build_save_request(
        self, seq, saved, aligned, operation, block_ids, is_last_prefill
    ):
        # The end the worker derives from `token_ids`; equal to `aligned`,
        # which is a chunk edge, and computed rather than assumed so the two
        # cannot drift into a boundary mismatch the worker refuses.
        end = self._chunk_floor(int(aligned))
        request = self._request_of(seq)
        units = (
            None
            if request is None
            else self._planner.take_ride_units(
                request, str(seq.id), end, operation=operation
            )
        )
        if units is None:
            # The whole save waits, not just its image: a PAGE-only object
            # would restore a prefix whose state is someone else's.
            logger.debug(
                "ATOM offload save not built: seq=%s end=%d has no checkpoint",
                seq.id,
                end,
            )
            return None
        request_meta = super()._build_save_request(
            seq, saved, end, operation, block_ids, is_last_prefill
        )
        request_meta.native_state = NativeStateTransfer(
            units, end, self._planner.boundary_hash(request, end)
        )
        return request_meta

    def _unbuilt_save_is_final(self) -> bool:
        # A save is declined only for want of a checkpoint, and a finished
        # request commits no further one.
        return True

    def build_connector_meta(self):
        metadata = super().build_connector_meta()
        # After this step's saves are built, so a pin taken for one of them
        # sees its own operation live. A save leaves the live set on every
        # exit it has -- terminal report, abandonment, teardown -- so this is
        # the one place a pin is given back.
        self._planner.retire_ride_pins(self._live_transfers())
        for sid in [
            sid
            for sid, (seq, _) in self._finished_frontiers.items()
            if (entry := self._save_tracker.get(sid)) is None or entry[0] is not seq
        ]:
            del self._finished_frontiers[sid]
        return metadata


__all__ = ["V41StateMPScheduler"]
