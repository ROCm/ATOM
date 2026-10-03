"""Stable per-request state-slot assignment for ATOM's vLLM proxy caches.

A model whose attention state is per request rather than per KV block --
DeepSeek-V4's SWA ring and compressor state, DeepSeek-V4.1's window ring,
compressor rings and Engram cursor -- cannot take its slot from vLLM's block
table, because vLLM owns blocks and knows nothing about the slot. It needs a
stable mapping from request to slot that survives chunked prefill and every
decode step, and that recycles a slot only once its request is gone.

That mapping is the same for every such model, so it lives here rather than in
any one bridge.
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

    Slots are reclaimed lazily on exhaustion by evicting the least-recently-seen
    slot whose ``req_id`` is absent from the current step (its request finished
    or was preempted). vLLM caps concurrency at ``num_slots`` (max_num_seqs), so
    a request that is live this step never has its slot evicted.
    """

    def __init__(self, num_slots: int):
        self.num_slots = max(1, int(num_slots))
        self._key_to_slot: dict[object, int] = {}
        self._slot_to_key: list[object] = [None] * self.num_slots
        self._free: list[int] = list(range(self.num_slots - 1, -1, -1))
        self._last_seen: list[int] = [-1] * self.num_slots
        self._step = 0
        # Keys bound by `reserve` that no batch has claimed yet, and the keys
        # the last batch held. Together they are what a reservation may not
        # evict. `_key_to_slot` is NOT that set: a finished request's entry
        # stays in it until its slot is recycled, so treating every bound key
        # as live makes the pool look full after `num_slots` requests and
        # sends `_acquire` to its slot-0 fallback -- handing a live request's
        # slot to a second tenant.
        self._reserved: set = set()
        self._last_active: set = set()

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

    def release(self, key) -> None:
        """Give a reserved slot back, for a request that will never arrive.

        Reservations are not otherwise reclaimed until the slot is the
        least-recently-seen eviction victim, so a request that is cancelled
        between its load and its first forward would hold one indefinitely.
        """
        self._reserved.discard(key)
        slot = self._key_to_slot.pop(key, None)
        if slot is None:
            return
        self._slot_to_key[slot] = None
        self._last_seen[slot] = -1
        if slot not in self._free:
            self._free.append(slot)

    def assign(self, req_keys, num_computed):
        """Return ``(slots: np.int32[num_reqs], reset_slots: set[int])``.

        ``req_keys`` is a per-request sequence of stable, hashable keys (the
        ``req_id`` strings), aligned with the batch rows.
        """
        self._step += 1
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
        key_to_slot = self._key_to_slot
        slot_to_key = self._slot_to_key
        last_seen = self._last_seen
        step = self._step
        slots = [0] * n
        reset: set[int] = set()
        for i in range(n):
            k = keys[i]
            slot = key_to_slot.get(k)
            if slot is None:
                # Reserved-but-unclaimed slots are live too: one holds bytes a
                # parked request's restore already wrote, and handing it to a
                # batch member puts two requests on one ring. They are not in
                # `active` because their request has no batch row yet -- which
                # is the whole reason reservations exist.
                slot = self._acquire(active | self._reserved)
                key_to_slot[k] = slot
                slot_to_key[slot] = k
                reset.add(slot)
            elif nc[i] == 0:
                # Known request recomputed from scratch (preemption resume).
                reset.add(slot)
            slots[i] = slot
            last_seen[slot] = step
        # The batch now owns these: a reservation has been claimed, and the
        # set a future reservation must not evict is this batch, not the
        # accumulated history.
        self._reserved -= active
        self._last_active = active
        return np.asarray(slots, dtype=np.int32), reset

    def _acquire_unused(self, live: set):
        """A slot no live tenant holds, or None. Never a fallback victim.

        `_acquire`'s slot-0 fallback is correct for `assign`, where vLLM has
        already guaranteed the batch fits. A reservation has no such
        guarantee, so here "no room" has to be sayable.
        """
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

    def _acquire(self, active: set) -> int:
        if self._free:
            return self._free.pop()
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
