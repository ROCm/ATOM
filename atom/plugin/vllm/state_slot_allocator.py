"""Stable per-request state-slot assignment for ATOM's vLLM proxy caches.

A model whose attention state is per request rather than per KV block --
DeepSeek-V4's SWA ring and compressor state, DeepSeek-V4.1's window ring,
compressor rings and Engram cursor -- cannot take its slot from vLLM's block
table, because vLLM owns blocks and knows nothing about the slot. It needs a
stable mapping from request to slot that survives chunked prefill and every
decode step, and that recycles a slot only once its request is gone.

That mapping is the same for every such model, so it lives here rather than in
any one bridge. It was extracted from the V4 bridge, which still imports both
names from here under its old private spellings -- so a reader who greps for
``_V4StateSlotAllocator`` lands on one implementation, not two that drift.
"""

from __future__ import annotations

import numpy as np


class StateSlotAllocator:
    """Stable per-request state-slot allocator over ``[0, num_slots)``.

    Keyed by each request's id (``req_id``), the canonical, host-resident
    request identity from vLLM's ``InputBatch``. This hands back the same state
    slot for every chunked-prefill step and every decode step of a request, so
    its SWA ring and compressor state accumulate in one place -- matching native
    ATOM's per-request cache slots.

    Keying on ``req_id`` (rather than the first KV block id, which lived on the
    GPU block table) removes the per-step D2H copy + host<->device sync that the
    block-id key required, and is immune to vLLM recycling a finished request's
    blocks to a new request within the same step.

    A slot is reported as freshly allocated (caller resets it) when it is newly
    bound to an unseen ``req_id``, or when a known ``req_id`` reappears with
    ``num_computed == 0`` -- vLLM recomputes preempted requests from scratch
    under the same id, so the slot's accumulated state must be cleared on resume.
    ``last_fresh_rows`` lists the batch rows bound to a new slot by the last
    ``assign``; a fresh row with ``num_computed > 0`` is a prefix hit whose slot
    the caller has to restore before the forward.

    With ``evict_lru`` (the default) slots are reclaimed lazily on exhaustion by
    evicting the least-recently-seen slot whose ``req_id`` is absent from the
    current step. Without it slots come back only through ``release`` -- the
    caller feeds it vLLM's finished / preempted / resumed request ids -- and an
    exhausted free list is an error rather than a guess about which request is
    gone.
    """

    def __init__(self, num_slots: int, *, evict_lru: bool = True):
        self.num_slots = max(1, int(num_slots))
        self.evict_lru = bool(evict_lru)
        self._key_to_slot: dict[object, int] = {}
        self._slot_to_key: list[object] = [None] * self.num_slots
        self._free: list[int] = list(range(self.num_slots - 1, -1, -1))
        # Slots bound outside a batch, by `reserve`, and not yet claimed by
        # one. They are live without being active: a parked request's restore
        # has already written bytes into them. See `reserve`.
        self._reserved: set = set()
        self._last_active: set = set()
        self._last_seen: list[int] = [-1] * self.num_slots
        self._step = 0
        self.last_fresh_rows: list[int] = []

    def release(self, req_keys) -> list[int]:
        """Return the slots of requests vLLM has finished, preempted or resumed.

        A reservation is dropped with the key. Reservations are not otherwise
        reclaimed until the slot is the least-recently-seen eviction victim,
        so a request cancelled between its tier load and its first forward
        would hold one indefinitely. `_last_seen` is reset with it, or the
        freed slot keeps a recency that makes it the last victim chosen.
        """
        freed = []
        for k in req_keys:
            self._reserved.discard(k)
            slot = self._key_to_slot.pop(k, None)
            if slot is None:
                continue
            self._slot_to_key[slot] = None
            self._last_seen[slot] = -1
            if slot not in self._free:
                self._free.append(slot)
            freed.append(slot)
        return freed

    def assign(self, req_keys, num_computed):
        """Return ``(slots: np.int32[num_reqs], reset_slots: set[int])``.

        ``req_keys`` is a per-request sequence of stable, hashable keys (the
        ``req_id`` strings), aligned with the batch rows.
        """
        # Pull num_computed to a Python list in one C call (per-element
        # numpy-scalar -> int was the dominant cost of this per-decode-step
        # loop). req_keys is already a host-side list[str]. Local-bind the
        # dict/list fields too -- attribute lookups inside the bs-length loop
        # add up at large batch (profiled #1 build cost).
        keys = list(req_keys)
        nc = (
            num_computed.tolist()
            if hasattr(num_computed, "tolist")
            else list(num_computed)
        )
        n = len(keys)
        active = set(keys)
        # Reserved-but-unclaimed slots are live too: one holds bytes a parked
        # request's restore already wrote, and handing it to a batch member
        # puts two requests on one ring. They are not in `active` because
        # their request has no batch row yet -- which is why reservations
        # exist at all.
        live = active | self._reserved
        if len(live) > self.num_slots:
            raise RuntimeError(
                f"state slot capacity exceeded: {len(active)} active requests + "
                f"{len(self._reserved - active)} unclaimed restores > "
                f"{self.num_slots} slots"
            )
        # Stepped only once the batch is known to fit: an overfull batch must
        # leave the allocator exactly as it found it, or the caller that
        # catches the refusal sees a clock that moved and an eviction order
        # that shifted with it.
        self._step += 1
        key_to_slot = self._key_to_slot
        slot_to_key = self._slot_to_key
        last_seen = self._last_seen
        step = self._step
        slots = [0] * n
        reset: set[int] = set()
        fresh: list[int] = []
        for i in range(n):
            k = keys[i]
            slot = key_to_slot.get(k)
            if slot is None:
                slot = self._acquire(live)
                key_to_slot[k] = slot
                slot_to_key[slot] = k
                reset.add(slot)
                fresh.append(i)
            elif nc[i] == 0:
                # Known request recomputed from scratch (preemption resume).
                reset.add(slot)
            slots[i] = slot
            last_seen[slot] = step
        self.last_fresh_rows = fresh
        # The batch now owns these: a reservation has been claimed, and the
        # set a future reservation must not evict is this batch, not the
        # accumulated history.
        self._reserved -= active
        self._last_active = active
        return np.asarray(slots, dtype=np.int32), reset

    def slot_for(self, key):
        """This key's slot, or None if it has none.

        A reader that needs the slot outside a forward -- the offload leg
        snapshotting a request's state -- must not create one by asking.
        """
        return self._key_to_slot.get(key)

    def reserve(self, key):
        """Bind *key* to a slot now, outside any batch. Idempotent.

        A connector that restores per-request state has to write it before the
        request is ever scheduled, and ``assign`` only runs from inside a
        forward. Reserving is also what keeps the following ``assign`` from
        reporting the slot as freshly allocated: that report means "reset me",
        and resetting would zero the bytes the restore just wrote.

        The reservation counts as a sighting, so a slot reserved this step is
        not the eviction victim chosen by the next one.

        Returns None when every slot belongs to the current batch or to
        another reservation. That is a real condition, not a corner case:
        reservations are for requests that are NOT in the batch, so
        `batch + reservations` can exceed the pool even though vLLM caps
        concurrency at its size. The caller must then decline the restore and
        let the request recompute -- taking a slot anyway puts two requests on
        one ring, which is how this was found (two requests straddling one
        cursor by a token each).
        """
        self._step += 1
        slot = self._key_to_slot.get(key)
        if slot is None:
            # Live = the last batch's keys plus reservations not yet claimed.
            # A finished request's stale entry is deliberately NOT live: it is
            # exactly what this reservation should be recycling.
            slot = self._acquire_unused(self._last_active | self._reserved)
            if slot is None:
                return None
            self._key_to_slot[key] = slot
            self._slot_to_key[slot] = key
        self._reserved.add(key)
        self._last_seen[slot] = self._step
        return slot

    def release_unclaimed(self, key) -> bool:
        """Release *key* only if it is a reservation no batch ever claimed.

        That is the whole leak: `reserve` binds a slot for a request parked on
        a restore, and only the `assign` that claims it clears the
        reservation, so a request aborted before it ever reaches a forward
        keeps its slot for good.

        A request that *did* run is deliberately left alone. Its slot is
        recycled by `_acquire`'s eviction exactly as before, and releasing it
        eagerly instead measured -21% throughput: the slot churn tripled the
        stores (16.5 to 59.5 per window) and cost 20% on ITL, because each
        store is a 5 MiB device-to-host copy contending with decode. The leak
        is real and the fix for it has to be this narrow.
        """
        if key not in self._reserved:
            return False
        # `release` takes the batch form -- a bare string would be iterated
        # character by character and release nothing.
        self.release([key])
        return True

    def _acquire_unused(self, live: set):
        """A slot no live tenant holds, or None without changing ownership."""
        if self._free:
            return self._free.pop()
        victim, victim_seen = None, None
        for s in range(self.num_slots):
            if self._slot_to_key[s] in live:
                continue
            if victim_seen is None or self._last_seen[s] < victim_seen:
                victim, victim_seen = s, self._last_seen[s]
        if victim is None:
            return None
        old = self._slot_to_key[victim]
        if old is not None:
            self._key_to_slot.pop(old, None)
        self._slot_to_key[victim] = None
        return victim

    def lookup(self, req_keys) -> np.ndarray:
        """The slots ``req_keys`` are bound to, without binding or resetting any."""
        key_to_slot = self._key_to_slot
        missing = [k for k in req_keys if k not in key_to_slot]
        if missing:
            raise RuntimeError(
                f"requests {sorted(map(str, missing))[:4]} have no state slot; the "
                "target binds one before anything else may read it"
            )
        return np.asarray([key_to_slot[k] for k in req_keys], dtype=np.int32)

    def _acquire(self, active: set) -> int:
        if self._free:
            return self._free.pop()
        if not self.evict_lru:
            raise RuntimeError(
                f"all {self.num_slots} state slots are bound to live requests "
                f"({sorted(map(str, self._key_to_slot))[:4]}...); a slot comes back "
                "only when vLLM reports its request finished or preempted"
            )
        victim = -1
        victim_seen = None
        for s in range(self.num_slots):
            if self._slot_to_key[s] in active:
                continue
            if victim_seen is None or self._last_seen[s] < victim_seen:
                victim = s
                victim_seen = self._last_seen[s]
        # A negative victim means every slot belongs to a request active this
        # step, which needs concurrency above num_slots and vLLM forbids it.
        # Slot 0 rather than a crash.
        victim = max(victim, 0)
        old = self._slot_to_key[victim]
        if old is not None:
            self._key_to_slot.pop(old, None)
        self._slot_to_key[victim] = None
        return victim


class SharedSlotLookup:
    """The target's slot allocator as a model sharing its slots sees it.

    The MTP draft forwards the requests the target forwarded earlier in the same
    step, so every request already has its slot, and binding, resetting and
    releasing stay with the target. ``assign`` only looks the slots up.
    """

    def __init__(self, owner: StateSlotAllocator):
        self.owner = owner
        self.last_fresh_rows: list[int] = []

    def assign(self, req_keys, num_computed):
        return self.owner.lookup(list(req_keys)), set()
