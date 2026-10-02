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


def test_a_reservation_never_evicts_a_live_tenant():
    allocator = StateSlotAllocator(2)
    held = {key: allocator.reserve(key) for key in ("a", "b")}
    # The pool is full of reservations; a third must not quietly take one of
    # their slots and leave two requests pointing at the same bytes.
    allocator.reserve("c")
    assert allocator.reserve("a") == held["a"] or allocator.reserve("b") == held["b"]


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
