# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 on the Mooncake Store: dense KV chunks plus KDA checkpoint images,
one image object per (PP stage, TP rank), moved GPUDirect from an HBM lane."""

from __future__ import annotations

import hashlib
import logging
import queue
from collections.abc import Callable

import torch

from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
    KimiK3OffloadConnector,
    KimiK3StateScheduling,
)
from atom.kv_transfer.offload.hybrid.kimi_k3.staging import StagedTransfer
from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import TouchResult
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store import pool as pool_mod
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    MooncakeStoreOffloadScheduler,
)
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)

logger = logging.getLogger("atom")

#: One lane per tier thread: `StateOffloadTier` runs one store and one load.
STATE_LANES = 2
#: Seconds a state transfer waits for a free lane before it fails.
LANE_WAIT_S = 30.0
_HASH_MASK = (1 << 64) - 1


class _DeviceBlob:
    """A device tensor where `StagedTransfer` expects a MemoryObj."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


def state_key_prefixes(
    namespace: str, rank: int, world: int, layout_id: str
) -> tuple[str, str]:
    """`(key prefix, group prefix)` of a worker's checkpoint images: keys per
    worker, groups per hash, so a checkpoint's parts are evicted together."""
    layout = hashlib.sha256(layout_id.encode()).hexdigest()[:16]
    return (
        f"{namespace}/w{int(rank)}of{int(world)}/kda-{layout}",
        f"{namespace}/kda-{layout}",
    )


class StoreStateCodec:
    """`StateByteCodec`'s contract over the Store, which evicts on its own; a
    lane that RDMA work may still reach is retired, as KV slots are quarantined."""

    _max_entries = None

    def __init__(
        self,
        backend,
        staged: StagedTransfer,
        entry_bytes: int,
        *,
        client,
        key_prefix: str,
        group_prefix: str,
        device,
        lanes: int = STATE_LANES,
        on_unsettled_put: Callable[[str], None] | None = None,
        load_refusal: Callable[[], str | None] | None = None,
    ) -> None:
        self._backend = backend
        self._staged = staged
        self.entry_bytes = int(entry_bytes)
        if self.entry_bytes <= 0:
            raise ValueError("state entry bytes must be > 0")
        self._client = client
        self._key_prefix = key_prefix
        self._group_prefix = group_prefix
        self._on_unsettled_put = on_unsettled_put
        self._load_refusal = load_refusal
        self._lanes = torch.empty(
            (int(lanes), self.entry_bytes), dtype=torch.uint8, device=device
        )
        client.register(self._lanes.data_ptr(), self._lanes.numel())
        self._free: queue.SimpleQueue[int] = queue.SimpleQueue()
        for lane in range(int(lanes)):
            self._free.put(lane)
        self._retired = 0
        self._closed = False

    def key(self, h: int) -> str:
        return f"{self._key_prefix}/{int(h) & _HASH_MASK:016x}"

    def _group(self, h: int) -> str:
        return f"{self._group_prefix}/{int(h) & _HASH_MASK:016x}"

    def _take_lane(self) -> int:
        try:
            return self._free.get(timeout=LANE_WAIT_S)
        except queue.Empty:
            raise RuntimeError(
                f"no state lane came free in {LANE_WAIT_S:.0f}s "
                f"({self._retired} retired)"
            ) from None

    def _retire(self, lane: int, why: str) -> None:
        self._retired += 1
        logger.error(
            "Mooncake Store state tier: lane %d retired for good: %s", lane, why
        )

    def _settle(self, lane: int, code: int, seconds: float) -> bool:
        """Free `lane` if no NIC access can still reach it, else retire it."""
        if store_client.buffer_settled(code, seconds):
            self._free.put(lane)
            return True
        self._retire(
            lane,
            f"a transfer gave up after {seconds:.0f}s "
            f"({store_client.describe(code)}) with RDMA work possibly posted",
        )
        return False

    def _distrust(self, why: str) -> None:
        if self._on_unsettled_put is not None:
            self._on_unsettled_put(why)

    def _refused(self) -> bool:
        return self._load_refusal is not None and self._load_refusal() is not None

    def put(self, h: int, unit_ids, on_source_released=None) -> bool:
        key = self.key(h)
        if self._client.exists([key])[0] == 1:
            if on_source_released is not None:
                on_source_released()
            return True
        lane = self._take_lane()
        blob = _DeviceBlob(self._lanes[lane])
        try:
            segments = getattr(self._backend, "page_unit_segments", None)
            if callable(segments):
                ptrs, sizes = segments(unit_ids)
                self._staged.pack_segments(ptrs, sizes, blob)
            else:
                self._staged.pack(self._backend.page_unit_views(unit_ids), blob)
        except BaseException:
            # A failed pack drained the device: nothing reads or writes the lane.
            self._free.put(lane)
            raise
        # `pack` synchronized its stream: the units are no longer read.
        if on_source_released is not None:
            on_source_released()
        clock = store_client.CallClock()
        try:
            [code] = self._client.put(
                [key],
                [self._lanes[lane].data_ptr()],
                [self.entry_bytes],
                group_ids=[self._group(h)],
            )
        except store_client.CallRefused:
            self._free.put(lane)
            raise
        except BaseException as exc:
            self._retire(lane, "a put raised")
            self._distrust(f"a state image put raised {exc!r}")
            raise
        seconds = clock.seconds()
        if not self._settle(lane, code, seconds):
            self._distrust(f"a state image put gave up after {seconds:.0f}s")
        return code == 0

    def get(self, h: int, slot) -> bool:
        if self._refused():
            return False
        lane = self._take_lane()
        clock = store_client.CallClock()
        try:
            [code] = self._client.get(
                [self.key(h)], [self._lanes[lane].data_ptr()], [self.entry_bytes]
            )
        except store_client.CallRefused:
            self._free.put(lane)
            raise
        except BaseException:
            # A get's late RDMA writes only this lane, never the owners'.
            self._retire(lane, "a get raised")
            raise
        if code != self.entry_bytes:
            self._settle(lane, code, clock.seconds())
            return False
        try:
            self._staged.unpack(
                _DeviceBlob(self._lanes[lane]), self._backend.state_entry_views(slot)
            )
        finally:
            # `unpack` synchronized, or drained the device before raising.
            self._free.put(lane)
        # A taint mark placed while the get ran: what it read may be garbage.
        return not self._refused()

    def touch(self, h: int) -> TouchResult:
        # `exists` grants a short read lease but does not reorder eviction.
        found = self._client.exists([self.key(h)])[0]
        return TouchResult.PRESENT if found == 1 else TouchResult.GONE

    def close(self) -> None:
        """Unregister the lanes, unless a retired one may still be reached;
        after the tier's threads are joined, before the client closes."""
        if self._closed:
            return
        self._closed = True
        if self._retired:
            pool_mod._UNSETTLED_ALLOCATIONS.append(self._lanes)
            logger.warning(
                "Mooncake Store state tier: %d retired lane(s); its %d bytes stay "
                "registered and allocated for the life of the process",
                self._retired,
                self._lanes.numel(),
            )
        else:
            self._client.unregister(self._lanes.data_ptr())


class MooncakeStoreKimiK3Connector(
    KimiK3OffloadConnector, MooncakeStoreOffloadConnector
):
    """K3's state tier over the Store's dense KV worker."""

    def _make_state_codec(self, backend, spec, entry_bytes, layout_id, rank, world):
        device = self._codec.device
        key_prefix, group_prefix = state_key_prefixes(
            self._namespace, rank, world, layout_id
        )
        # Never allocates: the lanes are on its device, so it packs in place.
        staged = StagedTransfer(device, staging_buffer_bytes=entry_bytes)
        return StoreStateCodec(
            backend,
            staged,
            entry_bytes,
            client=self._client,
            key_prefix=key_prefix,
            group_prefix=group_prefix,
            device=device,
            on_unsettled_put=self._distrust_store,
            load_refusal=lambda: self._load_refusal(self._client),
        )


class MooncakeStoreKimiK3Scheduler(
    KimiK3StateScheduling, MooncakeStoreOffloadScheduler
):
    """K3's state tier over the Store's chunked KV scheduling."""

    # `multi` routes the state face to the sub that says it hosts the tier.
    has_state_tier = True
    # Neither has run under a joint KV-and-state load.
    supports_early_load = False
    _supports_early_block_release = False
