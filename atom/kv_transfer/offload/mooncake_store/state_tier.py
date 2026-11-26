# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A per-request state tier on the Mooncake Store: KV chunks plus checkpoint
images of the state (Kimi-K3's KDA state, or a draft window kept as state
beside M3's PAGEs), one image object per (PP stage, TP rank) and prefix hash,
moved GPUDirect from an HBM lane.

The tier's worker and scheduler halves are the ones Kimi-K3 runs over LMCache
(`hybrid.kimi_k3`), mixed in front of the Store's; only the codec is the
Store's own (`StoreStateCodec`)."""

from __future__ import annotations

import hashlib
import logging
import queue
from collections.abc import Callable

import torch

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
    KimiK3OffloadConnector,
    KimiK3StateScheduling,
)
from atom.kv_transfer.offload.hybrid.kimi_k3.staging import StagedTransfer
from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import TouchResult
from atom.kv_transfer.offload.kv_store import pool as pool_mod
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    MooncakeStoreOffloadScheduler,
)
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)
from atom.model_engine.prefix_hash import chained_block_hashes

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
        f"{namespace}/w{int(rank)}of{int(world)}/state-{layout}",
        f"{namespace}/state-{layout}",
    )


def _image_key(key_prefix: str, h: int) -> str:
    return f"{key_prefix}/{int(h) & _HASH_MASK:016x}"


class WindowImageProbe:
    """The image keys a KV lookup also asks for a rebuildable state.

    A boundary less than `rebuild_tokens` from the end resumes only from an
    image (`PagedStateCheckpointCoordinator.rebuilds`). For each one up to the
    furthest a hit may reach, the lookup asks every holder's key; an image is
    found when all holders have their part."""

    def __init__(
        self,
        namespace: str,
        world: int,
        holders,
        layout_id: str,
        hash_block_size: int,
        rebuild_tokens: int,
    ) -> None:
        self._prefixes = [
            state_key_prefixes(namespace, rank, world, layout_id)[0] for rank in holders
        ]
        self._hash_block_size = int(hash_block_size)
        self._rebuild_tokens = int(rebuild_tokens)

    def keys(self, seq) -> tuple[list[int], list[str]]:
        """The candidate boundaries' prefix hashes, and their keys hash by hash."""
        hbs = self._hash_block_size
        first = max(1, (seq.num_tokens - self._rebuild_tokens) // hbs + 1)
        # `BlockManager._joint_kv_boundary`'s cap, at most the prompt.
        last = min(seq.num_prompt_tokens // hbs, -(-seq.num_tokens // hbs) - 1)
        if last < first:
            return [], []
        # The engine's chain (`BlockManager._chain_to`), off its caches: this
        # runs on the lookup thread.
        chain = chained_block_hashes(seq.token_ids, hbs, 0, last, seq.cache_seed)
        hashes = chain[first - 1 :]
        return hashes, [_image_key(p, h) for h in hashes for p in self._prefixes]

    def present(self, hashes: list[int], codes: list) -> list[int]:
        """The hashes every holder's key is present for."""
        n = len(self._prefixes)
        return [
            h
            for i, h in enumerate(hashes)
            if all(code == 1 for code in codes[i * n : (i + 1) * n])
        ]


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
        return _image_key(self._key_prefix, h)

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


class NullStateCodec:
    """A PP stage holding none of the state (a draft window's other stages):
    every image it is asked for is whole at no bytes, so the stages' joint
    verdict is the stages' that hold some."""

    _max_entries = None
    entry_bytes = 0

    def put(self, h: int, unit_ids, on_source_released=None) -> bool:
        if on_source_released is not None:
            on_source_released()
        return True

    def get(self, h: int, slot) -> bool:
        return True

    def touch(self, h: int) -> TouchResult:
        return TouchResult.PRESENT


class MooncakeStoreStateConnector(
    KimiK3OffloadConnector, MooncakeStoreOffloadConnector
):
    """The state tier over the Store's KV worker."""

    def _make_state_codec(self, backend, spec, entry_bytes, layout_id, rank, world):
        if not entry_bytes:
            return NullStateCodec()
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


class MooncakeStoreStateScheduler(KimiK3StateScheduling, MooncakeStoreOffloadScheduler):
    """The state tier over the Store's chunked KV scheduling."""

    # `multi` routes the state face to the sub that says it hosts the tier.
    has_state_tier = True
    # Neither has run under a joint KV-and-state load.
    supports_early_load = False
    _supports_early_block_release = False

    def bind_block_manager(self, block_manager) -> None:
        """A rebuildable state's images are also found by key, in the KV
        lookup's RPC; K3's only through the engine index."""
        super().bind_block_manager(block_manager)
        coordinator = block_manager.paged_state_checkpoints
        index = block_manager.state_offload
        if coordinator is None or not coordinator.rebuild_tokens:
            return
        if index is None or not index.can_load:
            return
        world = offcfg.lmcache_replica_world_size(self._config)
        stages = int(getattr(self._config, "pipeline_parallel_size", 1) or 1)
        self._lookup_client.backend.image_probe = WindowImageProbe(
            self._namespace,
            world,
            # A draft window is held by the last stage only, beside the draft.
            range(world - world // stages, world),
            coordinator.store.spec.layout_id,
            coordinator.hash_block_size,
            coordinator.rebuild_tokens,
        )

    def get_num_new_matched_tokens(self, seq):
        matched = super().get_num_new_matched_tokens(seq)
        # The lookup's found images, indexed before admission weighs them.
        for h in vars(seq).pop("_store_state_images", ()):
            self._block_manager.state_offload.note_stored(h)
        return matched
