# SPDX-License-Identifier: MIT
"""`DraftQSlots`: which persistent q rows a request's drafts were drawn into.

The safety property is that a verify never reads another request's q: a request
either finds the rows its own last propose wrote, or the -1 sentinel, which the
accept kernel answers by rejecting and drawing from the target alone. Pure
python, so this runs wherever the unit suite does.
"""

import pytest

from atom.spec_decode.draft_q_slots import DraftQSlots

T = 4


def bases(slots, req_ids, *, claim):
    out = [None] * len(req_ids)
    (slots.claim_bases if claim else slots.verify_bases)(req_ids, out)
    return out


def test_a_request_never_drafted_for_reads_the_sentinel():
    slots = DraftQSlots(4, T)
    assert bases(slots, ["a"], claim=False) == [-1]


def test_a_verify_reads_the_rows_its_propose_was_given():
    slots = DraftQSlots(4, T)
    drafted = bases(slots, ["a", "b", "c"], claim=True)
    # The scheduler reorders between the propose and the verify.
    assert bases(slots, ["c", "a", "b"], claim=False) == [
        drafted[2],
        drafted[0],
        drafted[1],
    ]
    assert all(base % T == 0 for base in drafted)


def test_a_returning_request_keeps_its_slot():
    slots = DraftQSlots(4, T)
    first = bases(slots, ["a", "b"], claim=True)
    assert bases(slots, ["b", "a"], claim=True) == [first[1], first[0]]
    assert slots.evictions == 0


def test_steps_for_other_requests_leave_a_waiting_requests_slot_alone():
    """A prefill-only step in between proposes for other ids; the decode
    request's verify must still find its own rows."""
    slots = DraftQSlots(8, T)
    (decoding,) = bases(slots, ["d"], claim=True)
    for step in range(3):
        bases(slots, [f"p{step}a", f"p{step}b"], claim=True)
    assert bases(slots, ["d"], claim=False) == [decoding]


def test_the_stalest_request_expires_first_and_then_reads_the_sentinel():
    slots = DraftQSlots(2, T)
    a, b = bases(slots, ["a", "b"], claim=True)
    bases(slots, ["b"], claim=True)  # b is now fresher than a
    (c,) = bases(slots, ["c"], claim=True)
    assert c == a, "the free list was empty, so c takes the stalest slot"
    assert slots.evictions == 1
    # a's next verify must not read c's rows.
    assert bases(slots, ["a", "b", "c"], claim=False) == [-1, b, c]


def test_slots_claimed_in_one_batch_are_distinct_under_pressure():
    max_bs = 6
    slots = DraftQSlots(2 * max_bs, T)
    for step in range(40):
        # Rotating ids, so most claims expire someone.
        req_ids = [f"r{(step * 5 + i) % 31}" for i in range(max_bs)]
        drafted = bases(slots, req_ids, claim=True)
        assert len(set(drafted)) == max_bs
        assert all(0 <= base < 2 * max_bs * T for base in drafted)
    assert slots.evictions > 0


def test_a_claim_never_expires_a_member_of_its_own_batch():
    """LRU [b, x], free list empty, batch [a, b]: a takes x's slot and b keeps
    its own -- one expiry, not a chain where a evicts b and b then evicts x."""
    slots = DraftQSlots(2, T)
    b, x = bases(slots, ["b", "x"], claim=True)
    assert bases(slots, ["a", "b"], claim=True) == [x, b]
    assert slots.evictions == 1
    assert bases(slots, ["x"], claim=False) == [-1]


def test_a_verify_computed_before_the_claims_of_its_own_batch_is_unaffected():
    """The order `stage_draft_sampling_host` uses: verify bases first, then
    claims. The new request takes the slot of one outside the batch; the
    batch's own requests keep the rows their drafts came from."""
    slots = DraftQSlots(3, T)
    a, b, x = bases(slots, ["a", "b", "x"], claim=True)
    assert bases(slots, ["c", "a", "b"], claim=False) == [-1, a, b]
    assert bases(slots, ["c", "a", "b"], claim=True) == [x, a, b]
    assert slots.evictions == 1


def test_more_live_requests_than_slots_never_alias():
    slots = DraftQSlots(3, T)
    owner = {}
    for step in range(50):
        req_ids = [f"r{(step + i) % 7}" for i in range(3)]
        for req_id, base in zip(req_ids, bases(slots, req_ids, claim=True)):
            for other, other_base in list(owner.items()):
                if other_base == base:
                    del owner[other]
            owner[req_id] = base
        for req_id in [f"r{i}" for i in range(7)]:
            (base,) = bases(slots, [req_id], claim=False)
            assert base == owner.get(req_id, -1)


def test_skipped_requests_claim_nothing():
    slots = DraftQSlots(4, T)
    out = [None] * 3
    slots.claim_bases(["a", "g", "b"], out, [False, True, False])
    assert out[1] == -1 and out[0] >= 0 and out[2] >= 0 and out[0] != out[2]
    assert "g" not in slots and len(slots) == 2


def test_sizes_must_be_positive():
    with pytest.raises(ValueError):
        DraftQSlots(0, T)
    with pytest.raises(ValueError):
        DraftQSlots(4, 0)
