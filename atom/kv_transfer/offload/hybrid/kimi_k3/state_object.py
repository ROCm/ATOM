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

        **Two jobs, and the second one is why this is not the bare hash.**

        *Build safety.* The same prefix hash maps to a completely different
        image under a different `num_spec`, TP size, or conv/ssm dtype. A size
        mismatch would be caught by `entry_bytes`; the same size with a
        different order or meaning reads back silently wrong state, and a
        request resumed onto wrong state produces wrong output with nothing
        raised anywhere. #2045 already encodes all of it in `layout_id`
        (layers / conv shape+dtype / ssm shape+dtype / order / tp / spec /
        carry) and enforces it HBM-side in `_validate_paged_state_op`; this is
        the CPU side of the same check.

        *Namespace separation.* `CacheEngineKey` has no field saying what an
        entry IS -- KV chunk keys come from `ChunkedTokenDatabase`'s chunk hash
        and state keys from ATOM's chained block hash, both plain integers in
        one space. Since Phase 3a they also share one pool. They were separated
        only accidentally, by this side hard-coding `torch.uint8` while KV
        carries the KV dtype. Folding `layout_id` in makes them disjoint by
        construction: no KV key can ever carry a K3 layout id.

        Not a plain `hash((h, layout_id))`: Python's `hash` of a str is salted
        per process (`PYTHONHASHSEED`), so a restart would silently orphan every
        entry written by the previous run. `xxh64` is what the block hashes
        themselves use.
        """
        import xxhash
        from lmcache.utils import CacheEngineKey

        digest = xxhash.xxh64()
        digest.update(int(h).to_bytes(8, "little", signed=True))
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

        The source is the checkpoint's PAGE units, not an Active Slot: under
        #2045 that is where the image lives, and `page_unit_views` names those
        bytes as the contiguous tensors the packer takes. The two directions
        are deliberately asymmetric -- a load writes the slot (`get`) -- and
        they compose because #2045's copy plan intersects two *ordered byte
        streams*, so a blob gathered in unit order is byte-identical to one
        gathered in slot order.

        A refusal is not an error. `_allocate` returns None under CPU memory
        pressure with nothing evictable, and a 53.6 MiB request is refused
        sooner than a KV chunk's 884 KB, so the state leg is the first to feel
        a full pool. Best-effort by design: the cost is one later miss, and
        `puts_refused` is what makes it visible instead of a silent hit-rate
        drift.
        """
        if self._storage is None:
            return False
        obj = self._allocate(self.entry_bytes)
        if obj is None:
            self.puts_refused += 1
            return False
        self._staged.pack(self._backend.page_unit_views(unit_ids), obj)
        self._storage.batched_put([self.key(h)], [obj])
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
        the buffer-backed objects that have no size of their own. None means
        "cannot check", never "size 0" -- refusing a load we cannot measure
        would turn an unknown into a guaranteed miss.
        """
        get_size = getattr(obj, "get_size", None)
        if callable(get_size):
            try:
                return int(get_size())
            except Exception:
                pass
        tensor = getattr(obj, "tensor", None)
        if tensor is not None:
            try:
                return int(tensor.numel()) * tensor.element_size()
            except Exception:
                pass
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

        # `MixedMemoryAllocator.allocate` accepts BINARY_BUFFER (-> a
        # BytesBufferMemoryObj, whose `.tensor` is None and would fail
        # `StagedTransfer.memory_tensor`) or one of the tensor formats
        # {KV_2LTD, KV_2TD, KV_T2D, KV_MLA_FMT, EC_TD}; anything else, BINARY
        # included, hits its `raise ValueError`. The shape/dtype above already
        # force a flat opaque blob regardless of `fmt`, so the value is inert
        # apart from passing that check -- same reasoning as the KV path's
        # `self._engine.fmt = MemoryFormat.KV_2LTD` in connector.py.
        # `busy_loop=False` because this is a *store*. LMCache's own
        # `LocalCPUBackend.allocate` docstring is explicit that busy_loop "should
        # only be used for retrieve", since "many stores happen concurrently (if
        # they busy_loop, deadlock happens)". Its default is True, and under that
        # default a pool with no eviction candidate spins `while True` on 0.1s
        # sleeps with no attempt bound -- so it would never return the None this
        # method's caller is written to handle, and would instead hang the state
        # tier's save worker for as long as CPU memory stays full. LMCache's own
        # store paths pass False here for the same reason (`cache_engine.py:666`,
        # `local_disk_backend.py:468`, `storage_manager.py:96`).
        return self._storage.allocate(
            torch.Size([nbytes]),
            torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
            busy_loop=False,
        )
