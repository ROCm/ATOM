# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""One state checkpoint, one opaque object, keyed by ATOM's own hash.

`ChunkedTokenDatabase` is deliberately bypassed. Two reasons, in order:

1. The two branches of the membership test must query the same integer.
   `StateGroupPool._resumable_from(h)` checks HBM and this tier; a
   chunker-derived key would not be the same thing being looked up, and the
   whole feature stops being a one-line widening.
2. State has a token range -- `[0, pos)`, which is exactly why it can share
   KV's key -- but its bytes cannot be sliced by token. There is no such thing
   as "the first three chunks hit". Chunking would produce N keys useful only
   all together.

It also sidesteps LMCache's chunk-alignment loss: keys exist only at chunk
boundaries (`token_database.py:387-391`).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("atom")


class StateByteCodec:
    """Pack/unpack one entry's state and move it through `storage_manager`.

    The object is an opaque flat uint8 blob -- `torch.Size([nbytes])` of
    `torch.uint8` -- allocated under a `MemoryFormat` the allocator accepts,
    because the x-packed / strided / multi-plane state layouts cannot be
    expressed in LMCache's token-major model at all.

    `entry_index` is not always a pool group, which is why it is not called
    one. On the spill/put path it is a **staging-ring slot**: the eviction hook
    copied the group's bytes there so `pop()` could hand the original group out
    immediately. On the load/get path it is a real pool group. Both are indices
    into the same `state_entry_views` space, and the backend resolves either.
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
    ) -> None:
        self._backend = backend
        self._staged = staged
        self.entry_bytes = int(entry_bytes)
        if self.entry_bytes <= 0:
            raise ValueError("state entry bytes must be > 0")
        self._model_name = model_name
        self._world_size = int(world_size)
        self._worker_id = int(worker_id)
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
        """The key carries ATOM's hash unmodified.

        `StateGroupPool._resumable_from(h)` looks up the same integer in HBM
        and in this tier, so hashing, salting, or stringifying it here would
        make the two branches ask different questions. `dtype` is part of the
        key's identity and must match the object we allocate.
        """
        from lmcache.utils import CacheEngineKey

        return CacheEngineKey(
            self._model_name,
            self._world_size,
            self._worker_id,
            int(h),
            torch.uint8,
        )

    def put(self, h: int, entry_index: int) -> bool:
        """Spill one entry. Returns False when nothing was stored.

        A refusal is not an error: `allocate` returns None under memory
        pressure, and spilling is best-effort by design -- the pool already
        counted the eviction either way, so the cost is one later prefix miss.
        """
        if self._storage is None:
            return False
        obj = self._allocate(self.entry_bytes)
        if obj is None:
            return False
        self._staged.pack(self._backend.state_entry_views(entry_index), obj)
        self._storage.batched_put([self.key(h)], [obj])
        return True

    def get(self, h: int, entry_index: int) -> bool:
        """Load one entry back into `entry_index`. False on a miss.

        The reference must be discharged here. `LocalCPUBackend.get_blocking`
        does a `ref_count_up()` on the caller's behalf, and LMCache's own
        callers pair it with a `ref_count_down()`. Without it the count never
        reaches zero, so LRU eviction drops the block from the index but never
        returns it to the pinned allocator -- an entry-sized leak per hit,
        ending with `allocate` returning None forever. `finally`, so the
        unpack throwing does not turn into a leak too.

        `put` is deliberately *not* symmetric: `StorageManager.batched_put`
        discharges the allocate reference itself, so a `ref_count_down` there
        would be a double free.
        """
        if self._storage is None:
            return False
        obj = self._storage.get(self.key(h))
        if obj is None:
            return False
        try:
            self._staged.unpack(obj, self._backend.state_entry_views(entry_index))
        finally:
            obj.ref_count_down()
        return True

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
        return self._storage.allocate(
            torch.Size([nbytes]), torch.uint8, fmt=MemoryFormat.KV_2LTD
        )
