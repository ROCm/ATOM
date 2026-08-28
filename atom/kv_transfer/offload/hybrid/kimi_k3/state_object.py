# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""One state checkpoint, one opaque object, keyed by ATOM's own hash.

`ChunkedTokenDatabase` is bypassed. `StateSlotPool._resumable_from` looks up
one integer in HBM and in this tier, and a chunker-derived key would not be the
same thing; and state's bytes cannot be sliced by token anyway -- there is no
"first three chunks hit", so chunking would produce N keys useful only
together. It also sidesteps LMCache's chunk-alignment loss.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("atom")


class StateByteCodec:
    """Pack/unpack one entry's state and move it through `storage_manager`.

    The object is an opaque flat uint8 blob: the x-packed / strided /
    multi-plane state layouts cannot be expressed in LMCache's token-major
    model at all.

    The two directions read different things, and that is not an oversight:
    a store gathers the checkpoint's PAGE units (`page_unit_views`) because
    #2045 is where the image lives, while a load scatters into the Active Slot
    the resuming forward will read (`state_entry_views`). The blob is the same
    ordered byte stream either way.
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
        self.puts_refused = 0
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
        different `num_spec`, TP size, or conv/ssm dtype. A size mismatch is
        caught by `entry_bytes`; the same size with a different order reads back
        silently wrong state, and a request resumed onto it produces wrong
        output with nothing raised. `layout_id` already names all of it and is
        enforced HBM-side -- this is the CPU side of the same check.

        *Namespace separation.* `CacheEngineKey` has no field saying what an
        entry IS, KV and state keys come from different hash functions into one
        integer space, and both now share one pool. Folding `layout_id` in makes
        the two spaces disjoint by construction.

        `xxh64`, not `hash((h, layout_id))`: Python salts `hash` of a str per
        process, so a restart would silently orphan every entry the previous run
        wrote.
        """
        import xxhash
        from lmcache.utils import CacheEngineKey

        digest = xxhash.xxh64()
        # Unsigned, and it must be: an ATOM block hash is `xxh64().intdigest()`,
        # which spans the full 0..2**64-1, and `BlockManager.compute_hash`
        # chains it with `prefix.to_bytes(8, "little")` -- unsigned too. With
        # `signed=True` every hash above 2**63-1 raised `OverflowError: int too
        # big to convert` from inside the `batched_put` argument list, which is
        # about half of them: a measured 27 of 46 stores.
        digest.update(int(h).to_bytes(8, "little", signed=False))
        digest.update(self._layout_id.encode())
        return CacheEngineKey(
            self._model_name,
            self._world_size,
            self._worker_id,
            digest.intdigest(),
            torch.uint8,
        )

    def put(self, h: int, unit_ids) -> bool:
        """Store one checkpoint image. False when nothing was stored.

        Reads the checkpoint's PAGE units, while `get` writes an Active Slot.
        The asymmetry is deliberate and safe: the copy plan intersects two
        *ordered byte streams*, so a blob gathered in unit order is
        byte-identical to one gathered in slot order.

        A refusal is not an error -- `_allocate` returns None under CPU memory
        pressure, and a whole image is refused sooner than a KV chunk, so the
        state leg feels a full pool first. `puts_refused` makes that visible
        instead of a silent hit-rate drift.
        """
        if self._storage is None:
            return False
        # Before the allocation, not inside the `batched_put` argument list.
        # `key` is pure and can raise (it did: see the `signed` note there), and
        # an argument-position raise happens *after* the allocation and *before*
        # `batched_put` can take ownership -- stranding the MemoryObj exactly
        # the way a throwing `pack` used to. Computing it first removes the
        # window rather than guarding it.
        key = self.key(h)
        obj = self._allocate(self.entry_bytes)
        if obj is None:
            self.puts_refused += 1
            return False
        # `batched_put` discharges the reference it is handed, so the success
        # path owes nothing -- but only if it is reached. A throwing `pack`
        # skips it and strands the allocation at ref_count=1, which LMCache
        # reports much later as "garbage collected with ref_count=1,
        # pin_count=0" and which shrinks the CPU pool by one entry per failure
        # until the tier can no longer allocate at all. `get` already guards
        # its own reference with `finally` for the same reason.
        try:
            self._staged.pack(self._backend.page_unit_views(unit_ids), obj)
            self._storage.batched_put([key], [obj])
        except Exception:
            obj.ref_count_down()
            raise
        return True

    def get(self, h: int, slot: int) -> bool:
        """Load one image back into Active Slot `slot`. False on a miss.

        The reference must be discharged here: `get_blocking` does a
        `ref_count_up()` on the caller's behalf, and without the matching down
        LRU drops the block from the index but never returns it to the pinned
        allocator -- an entry-sized leak per hit. `finally`, so a throwing
        unpack does not leak either. `put` is deliberately not symmetric:
        `batched_put` discharges its own reference.
        """
        if self._storage is None:
            return False
        obj = self._storage.get(self.key(h))
        if obj is None:
            return False
        size = self._object_bytes(obj)
        if size is not None and size != self.entry_bytes:
            # Unreachable while `layout_id` is in the key, which is the point:
            # a hit of the wrong size means two things collided in the shared
            # pool, and unpacking it would write another entry's bytes over a
            # request's live state. Degrade to a miss -- the caller disowns the
            # boundary and recomputes -- and count it, because a nonzero count
            # means the key is no longer doing its job.
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
        the buffer-backed objects that have no size of their own (their
        `.tensor` is None). None means "cannot measure", never "size 0" --
        refusing a load we cannot measure would turn an unknown into a
        guaranteed miss.

        Neither accessor is wrapped: an allocator that raises when asked its own
        object's size is broken in a way this method must not paper over, and
        swallowing it here would turn every read into a silent miss.
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

    def contains(self, h: int) -> bool:
        """Ask storage, never a local set -- a set goes stale the moment
        LMCache's LRU evicts. `contains` answers with a location name or None,
        so the truthiness test is the membership test."""
        if self._storage is None:
            return False
        return bool(self._storage.contains(self.key(h)))

    def _allocate(self, nbytes: int) -> Any:
        from lmcache.v1.memory_management import MemoryFormat

        # `fmt` is inert: the shape/dtype above already force a flat opaque
        # blob. It is passed only because `MixedMemoryAllocator.allocate`
        # rejects anything outside its tensor formats -- same reason the KV path
        # sets `self._engine.fmt`.
        #
        # `busy_loop=False` because this is a *store*. LMCache's own
        # `LocalCPUBackend.allocate` says busy_loop "should only be used for
        # retrieve", since "many stores happen concurrently (if they busy_loop,
        # deadlock happens)" -- and its default is True, under which a full pool
        # spins forever instead of returning the None this caller handles.
        return self._storage.allocate(
            torch.Size([nbytes]),
            torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
            busy_loop=False,
        )
