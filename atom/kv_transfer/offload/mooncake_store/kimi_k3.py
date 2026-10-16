# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 on the Mooncake Store: dense MLA KV plus the KDA state images.

The KV leg is the Store's dense offload (:mod:`.worker`, :mod:`.scheduler`):
one object per (rank, chunk), under DCP a chunk of whole virtual blocks with
each rank's share of it. The state leg is K3's state tier
(:mod:`atom.kv_transfer.offload.hybrid.kimi_k3`) with its LMCache codec
swapped for :class:`StoreStateCodec`: one Store object per (rank, checkpoint
image), packed straight into a registered HBM slot and put from there.

The engine's state index stays the one source of what is resumable, as with
LMCache. It mirrors an LRU of ``OFFLOAD_STATE_CPU_SIZE`` worth of images, so
the images are hard-pinned -- the master's eviction, which makes room for KV
chunks, never takes one behind the index's back -- and the codec removes the
oldest itself once it holds that many, as LMCache's state codec does.

Early block release stays off, as on K3's LMCache path: a save reports once
it is over, and the chunks' source-safe groups are not published.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from typing import Any

from atom.kv_transfer.offload._offload_common import StateOffloadFace
from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
    STATE_INDEX_CHANNEL,
    STATE_LOAD_DISPOSITION_CHANNEL,
    STATE_SOURCE_CHANNEL,
    KimiK3OffloadConnector,
    KimiK3StateSchedulerMixin,
)
from atom.kv_transfer.offload.hybrid.kimi_k3.staging import StagedTransfer
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store.config import (
    parse_mooncake_store_config,
)
from atom.kv_transfer.offload.mooncake_store.pool import (
    SLOT_ALIGN,
    SlotPoolExhausted,
    TransferSlotPool,
    pool_allocation_bytes,
)
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    MooncakeStoreOffloadScheduler,
)
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)
from atom.model_engine.state_offload import state_tier_cpu_bytes

logger = logging.getLogger("atom")

# A remove the Store refused (a read lease, a failed RPC) is retried before
# each later store; past this many seconds it is forced. No get can still be
# reading the image by then: Mooncake gives up on a transfer after 60 s.
_FORCE_REMOVE_AFTER_S = 2 * store_client.BATCH_WAIT_S
# Seconds between two `[OFFLOAD-STATE-STORE-STATS]` lines.
_STATS_LOG_INTERVAL_S = 60.0
# Removal results after which the image is gone.
_REMOVED = (0, store_client.OBJECT_NOT_FOUND)


def state_key_prefix(namespace: str, rank: int, world: int) -> str:
    """The part of a state key every image of one rank shares.

    Apart from the chunk keys' ``{namespace}/w{rank}of{world}/``, so a state
    image and a KV chunk never share a name.
    """
    return f"{namespace}/state/w{int(rank)}of{int(world)}/"


class StoreStateCodec:
    """Move one rank's KDA checkpoint images through the Store.

    The duck type :class:`StateOffloadTier` drives (``entry_bytes``, ``put``,
    ``get``), like LMCache's ``StateByteCodec``: a store gathers the
    checkpoint's PAGE units, a load scatters into the Active Slot the resuming
    forward reads, and the image is the same ordered byte stream both ways.
    """

    def __init__(
        self,
        backend: Any,
        staged: StagedTransfer,
        entry_bytes: int,
        *,
        client: store_client.MooncakeStoreClient,
        pool: TransferSlotPool,
        namespace: str,
        rank: int,
        world: int,
        layout_id: str,
        max_cache_bytes: int,
        load_refusal: Any,
        distrust: Any,
        settle: Any,
    ) -> None:
        self._backend = backend
        self._staged = staged
        self.entry_bytes = int(entry_bytes)
        if self.entry_bytes <= 0:
            raise ValueError("state entry bytes must be > 0")
        if not isinstance(layout_id, str) or not layout_id:
            raise ValueError("a state entry key needs a non-empty layout id")
        self._client = client
        self._pool = pool
        self._prefix = state_key_prefix(namespace, rank, world)
        self._layout_id = layout_id.encode()
        self._rank = int(rank)
        self._max_entries = int(max_cache_bytes) // self.entry_bytes
        if self._max_entries < 1:
            raise ValueError("the state budget must fit at least one checkpoint")
        # Why the worker loads nothing from the Store now, or None; the KV
        # leg's taint checks (`_load_refusal`).
        self._load_refusal = load_refusal
        # Marks the Store tainted after a put that may still write.
        self._distrust = distrust
        # Releases or quarantines slots by a call's result (`_settle`).
        self._settle = settle
        # The images this rank holds, oldest first, at most `_max_entries`.
        self._cache_keys: OrderedDict[str, None] = OrderedDict()
        self._cache_lock = threading.Lock()
        # Stores run one at a time: each may evict before it puts.
        self._store_lock = threading.Lock()
        # Evicted images the Store has not removed yet, and since when.
        self._pending_removals: dict[str, float] = {}
        # Per prefix hash, the deadlines of its queued stores, oldest first;
        # see `note_store_deadline`.
        self._deadlines: dict[int, deque[float]] = {}
        self._deadline_lock = threading.Lock()
        self._stats: dict[str, int] = dict.fromkeys(
            (
                "stored",
                "held",
                "store_failed",
                "past_deadline",
                "loaded",
                "load_failed",
                "removed",
                "forced",
            ),
            0,
        )
        self._next_stats_at = time.monotonic() + _STATS_LOG_INTERVAL_S

    # -- keys ----------------------------------------------------------------
    def key(self, h: int) -> str:
        """ATOM's prefix hash, bound to the geometry the image was written under.

        The layout id names the state geometry (num_spec, TP, dtypes): one
        prefix hash maps to a different image under a different one. Unsigned:
        an ATOM block hash spans the full 0..2**64-1.
        """
        digest = hashlib.blake2b(
            int(h).to_bytes(8, "little", signed=False) + self._layout_id,
            digest_size=16,
            person=b"atom-k3-state-v1",
        )
        return self._prefix + digest.hexdigest()

    def note_store_deadline(self, h: int, deadline: float) -> None:
        """Queue the time a store of ``h`` must start its GPU read by.

        The engine reclaims a store's pinned PAGE units once it has not
        reported for its save-abandon window, and hands them to other
        requests; a store still queued then would pack their bytes under this
        prefix -- an image the Store keeps for the rest of the run, since a
        put of a key it holds moves nothing. The store tier runs its stores
        in submission order, so each one takes the oldest deadline of its hash.
        """
        with self._deadline_lock:
            self._deadlines.setdefault(int(h), deque()).append(float(deadline))

    def _take_deadline(self, h: int) -> float | None:
        with self._deadline_lock:
            queued = self._deadlines.get(int(h))
            if not queued:
                return None
            deadline = queued.popleft()
            if not queued:
                del self._deadlines[int(h)]
            return deadline

    # -- store -----------------------------------------------------------------
    def put(self, h: int, unit_ids: Any, on_source_released: Any = None) -> bool:
        """Store one checkpoint image; False when the Store does not hold it.

        ``on_source_released`` fires once the GPU has stopped reading the
        units: after the pack, before the put.
        """
        key = self.key(h)
        deadline = self._take_deadline(h)
        with self._store_lock:
            self._retry_removals()
            with self._cache_lock:
                held = key in self._cache_keys
                if held:
                    self._cache_keys.move_to_end(key)
            if not held and self._pending_removals.pop(key, None) is not None:
                # Evicted, but its removal has not gone through: still there.
                held = True
                self._remember(key)
            if held:
                # Hard-pinned, so the Store holds it: a second put would move
                # nothing anyway.
                self._count("held")
                if on_source_released is not None:
                    on_source_released()
                return True
            self._evict_down_to(self._max_entries - 1)
            stored = self._put(key, unit_ids, on_source_released, deadline)
            if stored:
                self._remember(key)
            self._count("stored" if stored else "store_failed")
            return stored

    def _put(
        self,
        key: str,
        unit_ids: Any,
        on_source_released: Any,
        deadline: float | None,
    ) -> bool:
        pool, client = self._pool, self._client
        try:
            [slot] = pool.acquire("save", 1)
        except SlotPoolExhausted:
            logger.warning(
                "kimi_k3 Store offload: rank=%d has no usable state save slot; "
                "the image is not stored",
                self._rank,
            )
            return False
        # Checked after the wait for the slot, right before the GPU read.
        if deadline is not None and time.monotonic() >= deadline:
            pool.release([slot])
            self._count("past_deadline")
            logger.warning(
                "kimi_k3 Store offload: rank=%d state store queued past its "
                "source read deadline; the engine may have reclaimed its units, "
                "so the image is not stored",
                self._rank,
            )
            return False
        try:
            self._staged.pack(self._backend.page_unit_views(unit_ids), slot)
        except BaseException:
            # `pack` drained the device before raising.
            pool.release([slot])
            raise
        # The pack synchronized its stream: nothing on the device reads the
        # units from here, so the engine may have them back before the put.
        try:
            if on_source_released is not None:
                on_source_released()
        except BaseException:
            pool.release([slot])
            raise
        clock = store_client.CallClock()
        try:
            codes = client.put([key], [slot.ptr], [self.entry_bytes], hard_pin=True)
        except store_client.CallRefused:
            pool.release([slot])
            return False
        except Exception as exc:  # noqa: BLE001  # the image is not stored
            pool.quarantine([slot], reason="a state put raised")
            self._distrust(f"a state image put raised {exc!r}")
            return False
        call_seconds = clock.seconds()
        if self._settle(pool, [slot], codes, call_seconds):
            self._distrust(f"a state image put gave up after {call_seconds:.0f}s")
        if codes[0] != 0:
            logger.warning(
                "kimi_k3 Store offload: rank=%d state image put failed (%s)",
                self._rank,
                store_client.describe(codes[0]),
            )
            return False
        return True

    def _remember(self, key: str) -> None:
        with self._cache_lock:
            self._cache_keys.pop(key, None)
            self._cache_keys[key] = None

    def _evict_down_to(self, entries: int) -> None:
        """Remove the oldest images until at most ``entries`` remain."""
        while True:
            with self._cache_lock:
                if len(self._cache_keys) <= entries:
                    return
                old = next(iter(self._cache_keys))
                del self._cache_keys[old]
            self._remove(old, since=time.monotonic(), force=False)

    def _retry_removals(self) -> None:
        now = time.monotonic()
        for key, since in list(self._pending_removals.items()):
            self._remove(key, since=since, force=now - since >= _FORCE_REMOVE_AFTER_S)

    def _remove(self, key: str, *, since: float, force: bool) -> None:
        """Remove an evicted image; one the Store refuses is retried later.

        Not forced at first: a read lease (a lookup's, a get's) means a load
        may be reading it.
        """
        try:
            rc = self._client.remove(key, force=force)
        except Exception as exc:  # noqa: BLE001  # retried before the next store
            rc = None
            detail = repr(exc)
        else:
            detail = store_client.describe(rc)
        if rc in _REMOVED:
            self._pending_removals.pop(key, None)
            self._count("forced" if force else "removed")
            return
        first = key not in self._pending_removals
        self._pending_removals[key] = since
        if first or force:
            logger.warning(
                "kimi_k3 Store offload: rank=%d could not remove an evicted state "
                "image (%s%s); retried before the next store",
                self._rank,
                detail,
                ", forced" if force else "",
            )

    # -- load ------------------------------------------------------------------
    def get(self, h: int, slot: Any) -> bool:
        """Load one image into the Active Slot ``slot``; False when it cannot."""
        loaded = self._get(h, slot)
        self._count("loaded" if loaded else "load_failed")
        self._maybe_log_stats()
        return loaded

    def _get(self, h: int, slot: Any) -> bool:
        pool, client = self._pool, self._client
        # Before the get and after it, as the KV leg does: another client may
        # mark the Store meanwhile.
        if self._load_refusal() is not None:
            return False
        key = self.key(h)
        try:
            [buffer] = pool.acquire("load", 1)
        except SlotPoolExhausted:
            return False
        clock = store_client.CallClock()
        try:
            codes = client.get([key], [buffer.ptr], [self.entry_bytes])
        except store_client.CallRefused:
            pool.release([buffer])
            return False
        except Exception:  # a failed load recomputes
            pool.quarantine([buffer], reason="a state get raised")
            logger.warning(
                "kimi_k3 Store offload: rank=%d state image get raised",
                self._rank,
                exc_info=True,
            )
            return False
        if codes[0] != self.entry_bytes:
            # An object of another size is another geometry: never unpacked.
            self._settle(pool, [buffer], codes, clock.seconds())
            if codes[0] != store_client.OBJECT_NOT_FOUND:
                logger.warning(
                    "kimi_k3 Store offload: rank=%d state image get returned %s "
                    "for %d bytes",
                    self._rank,
                    (
                        store_client.describe(codes[0])
                        if codes[0] < 0
                        else f"{codes[0]} bytes"
                    ),
                    self.entry_bytes,
                )
            return False
        try:
            self._staged.unpack(buffer, self._backend.state_entry_views(slot))
        except BaseException:
            # `unpack` drained the device before raising.
            pool.release([buffer])
            raise
        pool.release([buffer])
        if self._load_refusal() is not None:
            # The failed load recomputes over whatever was unpacked.
            return False
        with self._cache_lock:
            if key in self._cache_keys:
                self._cache_keys.move_to_end(key)
        return True

    # -- lifecycle / observability ----------------------------------------------
    def close(self) -> None:
        self._pool.close()

    def _count(self, name: str) -> None:
        with self._cache_lock:
            self._stats[name] += 1

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        with self._cache_lock:
            if now < self._next_stats_at:
                return
            self._next_stats_at = now + _STATS_LOG_INTERVAL_S
            stats = dict(self._stats)
            entries = len(self._cache_keys)
        logger.info(
            "[OFFLOAD-STATE-STORE-STATS] rank=%d entries=%d/%d pending_removals=%d "
            "%s quarantined_slots=save:%d,load:%d",
            self._rank,
            entries,
            self._max_entries,
            len(self._pending_removals),
            " ".join(f"{name}={count}" for name, count in stats.items()),
            self._pool.quarantined("save"),
            self._pool.quarantined("load"),
        )


class KimiK3MooncakeStoreConnector(
    KimiK3OffloadConnector, MooncakeStoreOffloadConnector
):
    """Worker side: the Store's KV leg plus K3's state tier on the same Store.

    The MRO runs K3's hooks around the Store worker's: K3's
    ``register_kv_caches`` registers the Store's KV first, then builds the tier;
    its ``start_load_kv`` arms the joint loads, then hands the KV leg to the
    Store's, which stamps each save's dispatch.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._state_codec: StoreStateCodec | None = None

    @classmethod
    def kv_budget_reserve_bytes(cls, config: Any) -> int:
        """The KV leg's pool and staging, plus the state images' pool."""
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        cfg = parse_mooncake_store_config(kvc)
        half = cfg.state_pool_bytes // 2
        return super().kv_budget_reserve_bytes(config) + pool_allocation_bytes(
            half, cfg.state_pool_bytes - half
        )

    def close(self) -> None:
        """Drain the state tier, close its pool, then the Store worker."""
        tier = getattr(self, "_state_tier", None)
        if tier is not None:
            tier.drain()
            tier.shutdown()
        codec, self._state_codec = self._state_codec, None
        if codec is not None:
            codec.close()
        # Past K3's own close, which would drain the tier a second time.
        MooncakeStoreOffloadConnector.close(self)

    def _make_state_codec(self, backend, entry_bytes, layout_id, rank, world):
        """A codec over this worker's Store client, with a pool of its own.

        An image is far larger than a KV chunk and moves as one object, from
        one buffer: its own region of image-sized slots, so the KV regions
        never have to free a run of slots for it.
        """
        pool, client, _ = self._registered()
        cfg = self._store_cfg
        save_bytes = cfg.state_pool_bytes // 2
        load_bytes = cfg.state_pool_bytes - save_bytes
        slot_bytes = -(-entry_bytes // SLOT_ALIGN) * SLOT_ALIGN
        if slot_bytes > save_bytes:
            raise ValueError(
                f"mooncake_store.state_pool_mib={cfg.state_pool_mib} leaves "
                f"{save_bytes} bytes per direction, less than one {slot_bytes}-byte "
                f"slot for a {entry_bytes}-byte KDA state image; raise it to at "
                f"least {-(-2 * slot_bytes // (1 << 20))}"
            )
        state_pool = TransferSlotPool(
            device=pool.device,
            chunk_bytes=entry_bytes,
            save_bytes=save_bytes,
            load_bytes=load_bytes,
            client=client,
        )
        codec = StoreStateCodec(
            backend,
            # Packs into and unpacks from the registered slot itself, on the
            # device: the staging buffer is never allocated.
            StagedTransfer(pool.device, staging_buffer_bytes=entry_bytes),
            entry_bytes,
            client=client,
            pool=state_pool,
            namespace=self._namespace,
            rank=rank,
            world=world,
            layout_id=layout_id,
            max_cache_bytes=state_tier_cpu_bytes(),
            load_refusal=lambda: self._load_refusal(client),
            distrust=self._distrust_store,
            settle=self._settle,
        )
        self._state_codec = codec
        logger.info(
            "kimi_k3 Store offload: state images rank=%d entry_bytes=%d "
            "max_entries=%d save_slots=%d load_slots=%d",
            rank,
            entry_bytes,
            codec._max_entries,
            state_pool.capacity("save"),
            state_pool.capacity("load"),
        )
        return codec

    def _start_state_stores(self, metadata) -> None:
        """Give each store its source read deadline, then hand it to the tier."""
        stores = getattr(metadata, "state_stores", None)
        codec = self._state_codec
        if stores and codec is not None:
            deadline = time.monotonic() + self._source_read_window_s
            for op, _units in stores:
                codec.note_store_deadline(op.prefix_hash, deadline)
        super()._start_state_stores(metadata)

    def _source_group_safe(self, identity) -> None:
        """Published only under early release, which K3 keeps off."""
        if self._early_release:
            super()._source_group_safe(identity)

    def _save_abandon_timeout_s(self) -> float:
        return self._store_cfg.save_abandon_timeout_s


class KimiK3MooncakeStoreScheduler(
    KimiK3StateSchedulerMixin, MooncakeStoreOffloadScheduler, StateOffloadFace
):
    """Scheduler side: Store lookups for the KV leg, K3's state queues and
    joint boundary on top. Hosts the state tier, so a ``multi`` composite
    routes the state face here (``has_state_tier``)."""

    has_state_tier = True

    def _may_emit_save(self) -> bool:
        """K3's stall latch, then the Store's retry pause and pending cap."""
        return not self._save_stalled and MooncakeStoreOffloadScheduler._may_emit_save(
            self
        )

    def connector_completion(self, completion):
        if completion.channel in (
            STATE_SOURCE_CHANNEL,
            STATE_INDEX_CHANNEL,
            STATE_LOAD_DISPOSITION_CHANNEL,
        ):
            return super().connector_completion(completion)
        return MooncakeStoreOffloadScheduler.connector_completion(self, completion)


__all__ = [
    "KimiK3MooncakeStoreConnector",
    "KimiK3MooncakeStoreScheduler",
    "StoreStateCodec",
    "state_key_prefix",
]
