# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""One state checkpoint, one opaque object, keyed by ATOM's own hash.

`ChunkedTokenDatabase` is bypassed: `StateSlotPool._resumable_from` looks up one
integer, and state's bytes cannot be sliced by token anyway (there is no "first
three chunks hit", so chunking would produce N keys useful only together). It
also sidesteps LMCache's chunk-alignment loss.
"""

from __future__ import annotations

import enum
import logging
import threading
from collections import OrderedDict
from typing import Any

import torch

logger = logging.getLogger("atom")


class TouchResult(enum.IntEnum):
    """What a touch found. Truthy when the image can still be read."""

    GONE = 0
    #: Outside LocalCPU's hot cache but held by another backend (`get` reads
    #: it and promotes it back): present, its eviction order not refreshed.
    PRESENT = 1
    REFRESHED = 2


def _local_cpu_lru(storage_manager):
    """LMCache's LocalCPU backend, checked once for what a touch needs, or
    None when a touch cannot refresh anything.

    `cpu_lock`, `hot_cache` and `cache_policy.update_on_hit` are LMCache
    internals (written against 0.5.6), not API. Without one of them, touches
    only confirm presence, like a storage manager with no LocalCPU backend:
    loads still work, only the refresh is lost. Not silent: it warns here and
    every touch is counted in `state_offload_touches_unrefreshed`.
    """
    backend = getattr(storage_manager, "local_cpu_backend", None)
    if backend is None:
        missing = ["local_cpu_backend"]
    else:
        missing = [
            f"local_cpu_backend.{name}"
            for name in ("cpu_lock", "hot_cache", "cache_policy")
            if getattr(backend, name, None) is None
        ]
        if not missing and not callable(
            getattr(backend.cache_policy, "update_on_hit", None)
        ):
            missing.append("local_cpu_backend.cache_policy.update_on_hit")
    if not missing:
        return backend
    logger.warning(
        "state offload: LMCache's storage manager has no %s, so state touches "
        "only confirm presence and refresh no eviction order (every touch is "
        "counted in state_offload_touches_unrefreshed)",
        ", ".join(missing),
    )
    return None


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
        max_cache_bytes: int | None = None,
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
        self._unrefreshed_touches = 0
        self._storage = None
        self._local_cpu = None
        self._max_entries = (
            None
            if max_cache_bytes is None
            else int(max_cache_bytes) // self.entry_bytes
        )
        if self._max_entries is not None and self._max_entries < 1:
            raise ValueError("state CPU budget must fit at least one checkpoint")
        # The cap's LRU, in the same order as LMCache's: refreshed by `get`,
        # `touch` and a re-store, so the cap retires images nobody resumes
        # from rather than the oldest stores.
        self._cache_keys: OrderedDict[Any, None] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._store_lock = threading.Lock()
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
        self._local_cpu = _local_cpu_lru(storage_manager)
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
        # Cap state independently of KV while retaining the shared allocator;
        # within the cap, images and KV chunks compete in LMCache's one LRU.
        # Only stores serialize; a load holds its MemoryObj reference while
        # unpacking, so removing its cache entry cannot free live bytes.
        if self._storage is None:
            return False
        key = self.key(h)
        if self._touch_key(key) is TouchResult.REFRESHED:
            # Same prefix hash and layout, same bytes, still in LocalCPU's hot
            # cache: storing again would only cost an image-sized D2H that
            # LocalCPU then refuses as a duplicate key. The touch counts as the
            # use it is. Only there: an image another backend holds (disk) is
            # stored again, which puts it back into the CPU tier.
            if on_source_released is not None:
                on_source_released()
            return True
        if self._max_entries is None:
            return self._put(h, unit_ids, on_source_released)
        with self._store_lock:
            with self._cache_lock:
                if key not in self._cache_keys:
                    while len(self._cache_keys) >= self._max_entries:
                        old = next(iter(self._cache_keys))
                        self._storage.remove(old)
                        del self._cache_keys[old]
            ok = self._put(h, unit_ids, on_source_released)
            if ok:
                with self._cache_lock:
                    self._cache_keys.pop(key, None)
                    self._cache_keys[key] = None
            return ok

    def touch(self, h: int) -> TouchResult:
        """Refresh `h`'s image in LMCache's LRU; falsy (`GONE`) when it is gone.

        The state-image twin of what a KV lookup does to its chunks
        (`LocalCPUBackend.batched_async_contains(pin=True)` then
        `touch_cache()` -> `cache_policy.update_on_hit`). State hits are
        answered engine-side by `StateOffloadIndex`, so without this LMCache
        never learns an image was used, and a prefix hit on every turn keeps
        its KV at the young end and its image at the cold end, where pressure
        evicts it first.

        `update_on_hit` under the backend's `cpu_lock`, not `contains(pin=True)`
        + `touch_cache()`: a pin needs a matching unpin, and `keys_in_request`
        is the list the KV lookup is filling at the same time.
        """
        if self._storage is None:
            return TouchResult.GONE
        result = self._touch_key(self.key(h))
        if result is TouchResult.PRESENT:
            # Still reported resident: `get` can read it now, and the
            # confirmation that opens only lets later loads skip the hold,
            # where a miss degrades to a recompute.
            self._unrefreshed_touches += 1
            if self._unrefreshed_touches == 1:
                logger.warning(
                    "state offload: a touched state image is outside "
                    "LocalCPU's hot cache, so the touch only confirmed it is "
                    "present and did not refresh its eviction order (counted "
                    "as state_offload_touches_unrefreshed)"
                )
        return result

    def _touch_key(self, key) -> TouchResult:
        result = self._refresh(key)
        with self._cache_lock:
            if result:
                if key in self._cache_keys:
                    self._cache_keys.move_to_end(key)
            else:
                # Evicted by LMCache: it no longer counts against the cap.
                self._cache_keys.pop(key, None)
        return result

    def _refresh(self, key) -> TouchResult:
        backend = self._local_cpu
        if backend is not None:
            with backend.cpu_lock:
                if key in backend.hot_cache:
                    backend.cache_policy.update_on_hit(key, backend.hot_cache)
                    return TouchResult.REFRESHED
        # Not in the hot cache (or there is none): another backend may still
        # hold it, and `get` reads every backend. Presence only.
        if self._storage.contains(key):
            return TouchResult.PRESENT
        return TouchResult.GONE

    def _put(self, h: int, unit_ids, on_source_released=None) -> bool:
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
        # we down it exactly once. The earlier `handed_off = True` set *before*
        # the call did the opposite: it suppressed the down on a pre-adoption
        # `batched_put` raise, stranding the allocation at ref_count=1 (LMCache
        # reports "garbage collected with ref_count=1, pin_count=0" much later)
        # and shrinking the CPU pool one entry per failure. `get` guards its own
        # reference with `finally` for the same reason.
        try:
            segments = getattr(self._backend, "page_unit_segments", None)
            if callable(segments):
                ptrs, sizes = segments(unit_ids)
                self._staged.pack_segments(ptrs, sizes, obj)
            else:
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

    def get(self, h: int, slot) -> bool:
        """Load one image back into `slot`. False on a miss.

        `slot` is whatever the bound backend addresses an entry by, and is
        passed through untouched: ATOM's native path an Active Slot index, the
        vLLM plugin path one block id per mamba KV cache group.

        The reference must be discharged here: `get_blocking` does a
        `ref_count_up()` for the caller, and without the matching down LRU drops
        the block from the index but never returns it to the pinned allocator --
        an entry-sized leak per hit. `finally`, so a throwing unpack does not
        leak either. (`put` differs: `batched_put` discharges its own reference.)
        """
        if self._storage is None:
            return False
        key = self.key(h)
        obj = self._storage.get(key)
        if obj is None:
            with self._cache_lock:
                self._cache_keys.pop(key, None)
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
        # `get_blocking` does not refresh LMCache's LRU; a resume is a use.
        self._touch_key(key)
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
