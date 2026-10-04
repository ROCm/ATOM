# SPDX-License-Identifier: MIT
"""Reserving a state slot ahead of the batch that will use it.

A connector restoring per-request state writes it before the request is ever
scheduled, so the slot has to be bound outside a forward. The contract that
matters is the second half: the `assign` that follows must *not* report the
slot as freshly allocated. That report means "reset me", and the reset would
zero the bytes the restore just wrote -- leaving a request running on an empty
window ring with a KV prefix that claims otherwise. Nothing downstream checks
it, so it is pinned here.
"""

from atom.plugin.vllm.state_slot_allocator import StateSlotAllocator


def test_a_reserved_slot_is_not_reported_as_reset():
    allocator = StateSlotAllocator(4)
    slot = allocator.reserve("r0")
    slots, reset = allocator.assign(["r0"], [8192])
    assert int(slots[0]) == slot
    assert slot not in reset


def test_reserving_twice_is_the_same_slot():
    allocator = StateSlotAllocator(4)
    assert allocator.reserve("r0") == allocator.reserve("r0")


def test_an_unreserved_request_is_still_reported_as_reset():
    """The guard is narrow: a request nobody restored still needs its reset."""
    allocator = StateSlotAllocator(4)
    slots, reset = allocator.assign(["r0"], [0])
    assert int(slots[0]) in reset


def test_a_resumed_request_resets_even_if_it_was_reserved():
    """`num_computed == 0` is preemption, and its accumulated state is stale.

    This is the case the reservation must not mask: the request resumes under
    the same id with nothing restored, so the slot does have to be cleared.
    """
    allocator = StateSlotAllocator(4)
    slot = allocator.reserve("r0")
    _, reset = allocator.assign(["r0"], [0])
    assert slot in reset


def test_a_reservation_never_evicts_a_request_in_the_current_batch():
    """Two tenants in one slot is the failure this prevents.

    Observed on hardware as `_report_stale_state` refusing two requests whose
    positions straddled one cursor by a token each -- they were sharing a slot.
    """
    allocator = StateSlotAllocator(2)
    allocator.assign(["live0", "live1"], [100, 100])
    live = {k: allocator.slot_for(k) for k in ("live0", "live1")}
    # No room: both slots belong to the batch. Refusing is the only safe
    # answer -- taking one anyway puts two requests on one ring.
    assert allocator.reserve("newcomer") is None
    assert allocator.slot_for("live0") == live["live0"]
    assert allocator.slot_for("live1") == live["live1"]


def test_a_reservation_recycles_a_finished_request_s_slot():
    """The pool must not look full just because it remembers old requests.

    A finished request's entry survives in the key table until its slot is
    recycled. Counting those as live is what made `_acquire` find no victim
    after `num_slots` requests and fall back to slot 0 -- handing a live
    request's slot to a second tenant.
    """
    allocator = StateSlotAllocator(2)
    allocator.assign(["done0", "done1"], [100, 100])
    allocator.assign(["live"], [100])  # done0/done1 are gone from the batch
    live_slot = allocator.slot_for("live")
    # One finished request's slot is recyclable; the other slot is the batch's.
    first = allocator.reserve("new0")
    assert first is not None and first != live_slot
    assert allocator.reserve("new1") is None, "a reservation evicted the batch"


def test_reservations_do_not_evict_each_other():
    allocator = StateSlotAllocator(4)
    slots = [allocator.reserve(f"r{i}") for i in range(4)]
    assert len(set(slots)) == 4
    assert allocator.reserve("one-too-many") is None


def test_releasing_returns_the_slot_to_the_pool():
    allocator = StateSlotAllocator(1)
    first = allocator.reserve("r0")
    allocator.release("r0")
    assert allocator.reserve("r1") == first


def test_releasing_an_unknown_key_is_a_no_op():
    allocator = StateSlotAllocator(2)
    allocator.release("never-seen")
    assert allocator.reserve("r0") == allocator.reserve("r0")


def test_a_released_slot_is_reset_for_its_next_tenant():
    """Release must not leave the next request reading the old one's state."""
    allocator = StateSlotAllocator(1)
    allocator.reserve("r0")
    allocator.release("r0")
    _, reset = allocator.assign(["r1"], [4096])
    assert reset == {0}


def test_assign_does_not_evict_a_reserved_slot():
    """A parked request's restored bytes must survive the next batch.

    The reservation is not in `active` -- that is the point of it, the request
    has no batch row yet -- so `assign` has to be told about it separately.
    Without this the restore lands, the next batch takes the slot, and the two
    requests run on one ring.
    """
    allocator = StateSlotAllocator(2)
    parked = allocator.reserve("parked")
    allocator.assign(["batch0"], [100])
    assert allocator.slot_for("parked") == parked
    assert allocator.slot_for("batch0") != parked


def test_a_claimed_reservation_stops_protecting_its_slot():
    """Otherwise every reservation shrinks the pool for good."""
    allocator = StateSlotAllocator(2)
    allocator.reserve("r0")
    allocator.assign(["r0"], [4096])          # claimed: no longer a reservation
    allocator.assign(["other"], [100])        # r0 gone from the batch
    assert allocator.reserve("r1") is not None


def test_release_unclaimed_frees_a_reservation_nobody_took():
    """The leak: parked for a restore, then aborted before any forward."""
    allocator = StateSlotAllocator(1)
    first = allocator.reserve("parked")
    assert allocator.release_unclaimed("parked") is True
    assert allocator.reserve("next") == first


def test_release_unclaimed_leaves_a_request_that_actually_ran():
    """Its slot is the allocator's to recycle, and taking it back costs.

    Releasing these eagerly as well measured -21% throughput: the churn
    tripled the 5 MiB state stores and cost 20% on ITL. The leak is only ever
    the reservation nobody claimed.
    """
    allocator = StateSlotAllocator(2)
    allocator.reserve("r0")
    allocator.assign(["r0"], [4096])          # claimed -> no longer reserved
    assert allocator.release_unclaimed("r0") is False
    assert allocator.slot_for("r0") is not None


def test_release_unclaimed_is_a_no_op_for_an_unknown_key():
    assert StateSlotAllocator(2).release_unclaimed("never-seen") is False
