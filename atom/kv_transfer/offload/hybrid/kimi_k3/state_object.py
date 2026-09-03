# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""One state checkpoint, one opaque object, keyed by ATOM's own hash.

`ChunkedTokenDatabase` is bypassed: `StateSlotPool._resumable_from` looks up one
integer, and state's bytes cannot be sliced by token anyway (there is no "first
three chunks hit", so chunking would produce N keys useful only together). It
also sidesteps LMCache's chunk-alignment loss.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("atom")


class StateByteCodec:
    """Pack/unpack one entry's state and move it through `storage_manager`.

    An opaque flat uint8 blob: the x-packed / strided / multi-plane state
    layouts cannot be expressed in LMCache's token-major model at all.

    The two directions read different things by design: a store gathers the
    checkpoint's PAGE units (`page_unit_views`, where #2045 keeps the image), a
    load scatters into the Active Slot the resuming forward reads
    (`state_entry_views`). The blob is the same ordered byte stream either way.
    """

    def __init__(
        self,
        backend,
        staged,
        entry_bytes: int,
        *,
        model_name: str,
        world_size: int,
        worker_id: int,
        layout_id: str,
    ) -> None:
        self._backend = backend
        self._staged = staged
        self.entry_bytes = int(entry_bytes)
        if self.entry_bytes <= 0:
            raise ValueError("state entry bytes must be > 0")
        if not isinstance(layout_id, str) or not layout_id:
            raise ValueError("a state entry key needs a non-empty layout id")
        self._model_name = model_name
        self._world_size = int(world_size)
        self._worker_id = int(worker_id)
        self._layout_id = layout_id
        self._misfit_reads = 0
        self._storage = None
        # Never hard-code a size: V4 keeps six compressor fields across
        # n_csa/n_hca layers plus an optional window; GDN keeps
        # 2 * num_gdn_attn_state * (1 + num_spec) slots. Measured at 53.6 MiB
        # per entry on the real model -- MB-scale, not KB.
        logger.info(
            "state offload: entry_bytes=%d (%.2f MiB) per request",
            self.entry_bytes,
            self.entry_bytes / (1 << 20),
        )

    def bind_storage_manager(self, storage_manager) -> None:
        self._storage = storage_manager

    def key(self, h: int):
        """ATOM's hash, bound to the geometry the bytes were written under.

        *Build safety.* One prefix hash maps to a different image under a
        different `num_spec`, TP size, or conv/ssm dtype. `entry_bytes` catches a
        size mismatch; the same size in a different order reads back silently
        wrong state. `layout_id` names all of it and is enforced HBM-side -- this
        is the CPU side of the same check.

        *Namespace separation.* `CacheEngineKey` has no field saying what an
        entry IS, and KV and state keys now share one pool. Folding `layout_id`
        in makes the two key spaces disjoint by construction.

        `xxh64`, not `hash((h, layout_id))`: Python salts `hash` of a str per
        process, so a restart would silently orphan every prior entry.
        """
        import xxhash
        from lmcache.utils import CacheEngineKey

        digest = xxhash.xxh64()
        # Unsigned, and it must be: an ATOM block hash spans the full
        # 0..2**64-1. With `signed=True` every hash above 2**63-1 raised
        # `OverflowError: int too big to convert` -- about half of them, a
        # measured 27 of 46 stores.
        digest.update(int(h).to_bytes(8, "little", signed=False))
        digest.update(self._layout_id.encode())
        return CacheEngineKey(
            self._model_name,
            self._world_size,
            self._worker_id,
            digest.intdigest(),
            torch.uint8,
        )

    def put(self, h: int, unit_ids, on_source_released=None) -> bool:
        """Store one checkpoint image. False when nothing was stored.

        Reads PAGE units where `get` writes an Active Slot; safe because the
        copy plan intersects two *ordered byte streams*, so a blob gathered in
        unit order is byte-identical to one gathered in slot order.

        A refusal is not an error -- `_allocate` returns None under CPU pressure,
        and a whole image is refused sooner than a KV chunk, so the state leg
        feels a full pool first.

        `on_source_released` fires when `pack` returns -- after the gather and
        D2H are synchronized, i.e. once the GPU has stopped reading the units.
        Separate from the return value on purpose: the units are the KV pool's,
        and holding them across `batched_put` would keep an image out of the
        pool for a CPU operation that cannot touch them.
        """
        if self._storage is None:
            return False
        # Computed before the allocation: `key` can raise (see the `signed`
        # note), and a raise from inside the `batched_put` argument list would
        # strand the MemoryObj the same way a throwing `pack` used to.
        key = self.key(h)
        obj = self._allocate(self.entry_bytes)
        if obj is None:
            return False
        # `batched_put` discharges the reference it is handed, but only in its
        # terminal tail loop (LMCache `storage_manager.py`, one
        # `ref_count_down()` per object), which runs *after* every step that can
        # raise -- the scheduler-role guard, `get_allocator_backend`,
        # `allocate_and_copy_objects`, `batched_submit_put_task`. So the down is
        # `batched_put`'s last action: on a successful return the reference is
        # already discharged, and there is no window where the object is adopted
        # yet a later step raises.
        #
        # That means on ANY exception below -- from `pack`, from
        # `on_source_released`, or from a pre-adoption raise inside
        # `batched_put` -- the reference was never downed and we still own it, so
        # we down it exactly once, unconditionally.
        #
        # Do not reintroduce an "already handed off" flag set before the call to
        # skip that down: there is no adopted-then-raised window for it to
        # protect, and it would suppress the down on a pre-adoption raise,
        # stranding the allocation at ref_count=1 (LMCache reports "garbage
        # collected with ref_count=1, pin_count=0" much later) and shrinking the
        # CPU pool one entry per failure. `get` guards its own reference with
        # `finally` for the same reason.
        try:
            self._staged.pack(self._backend.page_unit_views(unit_ids), obj)
            # Source first: `pack` has synchronized the stream that reads the
            # units, so nothing on the device touches them from here.
            if on_source_released is not None:
                on_source_released()
            self._storage.batched_put([key], [obj])
        except Exception:
            obj.ref_count_down()
            raise
        return True

    def get(self, h: int, slot: int) -> bool:
        """Load one image back into Active Slot `slot`. False on a miss.

        The reference must be discharged here: `get_blocking` does a
        `ref_count_up()` for the caller, and without the matching down LRU drops
        the block from the index but never returns it to the pinned allocator --
        an entry-sized leak per hit. `finally`, so a throwing unpack does not
        leak either. (`put` differs: `batched_put` discharges its own reference.)
        """
        if self._storage is None:
            return False
        obj = self._storage.get(self.key(h))
        if obj is None:
            return False
        size = self._object_bytes(obj)
        if size is not None and size != self.entry_bytes:
            # Unreachable while `layout_id` is in the key, which is the point: a
            # wrong-size hit means two things collided in the shared pool, and
            # unpacking would write another entry's bytes over live state.
            # Degrade to a miss (caller disowns and recomputes) and count it --
            # a nonzero count means the key is no longer doing its job.
            self._misfit_reads += 1
            logger.warning(
                "state offload: hash %d came back %d bytes, expected %d; "
                "treating as a miss (misfit_reads=%d)",
                h,
                size,
                self.entry_bytes,
                self._misfit_reads,
            )
            obj.ref_count_down()
            return False
        try:
            self._staged.unpack(obj, self._backend.state_entry_views(slot))
        finally:
            obj.ref_count_down()
        return True

    @staticmethod
    def _object_bytes(obj) -> int | None:
        """Bytes in a `MemoryObj`, or None when it will not say.

        `get_size()` is the documented accessor; the tensor is the fallback for
        buffer-backed objects with no size of their own. None means "cannot
        measure", never "size 0". Neither accessor is wrapped -- an allocator
        that raises when asked its own object's size is broken in a way this
        method must not paper over into a silent miss.
        """
        get_size = getattr(obj, "get_size", None)
        if callable(get_size):
            size = get_size()
            if size is not None:
                return int(size)
        tensor = getattr(obj, "tensor", None)
        if tensor is not None:
            return int(tensor.numel()) * tensor.element_size()
        return None

    def _allocate(self, nbytes: int) -> Any:
        from lmcache.v1.memory_management import MemoryFormat

        # `fmt` is inert (shape/dtype force a flat blob); passed only because
        # `MixedMemoryAllocator.allocate` rejects anything outside its tensor
        # formats. `busy_loop=False` because this is a *store*: LMCache warns
        # busy_loop "should only be used for retrieve" (concurrent stores
        # deadlock), yet defaults it True -- under which a full pool spins
        # forever instead of returning the None this caller handles.
        return self._storage.allocate(
            torch.Size([nbytes]),
            torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
            busy_loop=False,
        )
