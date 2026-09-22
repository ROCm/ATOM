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
                slot = self._acquire(active)
                key_to_slot[k] = slot
                slot_to_key[slot] = k
                reset.add(slot)
            elif nc[i] == 0:
                # Known request recomputed from scratch (preemption resume).
                reset.add(slot)
            slots[i] = slot
            last_seen[slot] = step
        return np.asarray(slots, dtype=np.int32), reset

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
