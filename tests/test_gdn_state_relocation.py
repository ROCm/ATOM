# SPDX-License-Identifier: MIT
# Tests for relocating a GDN state slot's bytes.
#
# GDN checkpoints by forking, so this path is not about checkpoints: moving the
# state pool's boundary has to be able to shift a slot out of the way, and that
# is a byte move whatever mechanism the class uses to checkpoint.
#
# The unit that moves is one slot -- one complete recurrent state, across every
# layer. A request under speculative decoding holds `1 + num_spec` of them, but
# they are allocated one at a time and need not be adjacent, so relocating such
# a request is several pairs rather than one wider pair. That is the whole point
# of the per-slot allocation: a checkpoint holds a committed state and has no
# speculation to roll back, so it costs one slot rather than a full group.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions.gdn_attn import GDNStateMixin

LAYERS = 3
SLOTS = 12
SHAPE_K = (2, 5)
SHAPE_V = (2, 3, 4)


CAP = 6
REC_K = (2, 5)
REC_V = (2, 3)


def build(num_spec: int, replayssm: bool = False):
    """Caches whose every (layer, slot) plane carries a distinct value."""
    k = torch.zeros((LAYERS, SLOTS) + SHAPE_K)
    v = torch.zeros((LAYERS, SLOTS) + SHAPE_V)
    for layer in range(LAYERS):
        for slot in range(SLOTS):
            k[layer, slot] = layer * 100 + slot
            v[layer, slot] = -(layer * 100 + slot)
    runner = SimpleNamespace(mamba_k_cache=k, mamba_v_cache=v)
    if replayssm:
        # Distinct per (layer, slot) here too, and a cursor that is distinct
        # per slot -- a relocation that drops either is then visible rather
        # than accidentally right.
        bufs = {}
        for name, shape in (
            ("replayssm_buf_k", (CAP,) + REC_K),
            ("replayssm_buf_u", (CAP,) + REC_V),
            ("replayssm_buf_g", (CAP,) + REC_K),
        ):
            t = torch.zeros((LAYERS, SLOTS) + shape)
            for layer in range(LAYERS):
                for slot in range(SLOTS):
                    t[layer, slot] = layer * 100 + slot + 0.5
            bufs[name] = t
        setattr_all(runner, bufs)
        runner.replayssm_write_pos = torch.arange(1, SLOTS + 1, dtype=torch.int32)
    stub = SimpleNamespace(
        num_spec=num_spec,
        replayssm=replayssm,
        model_runner=runner,
    )
    return stub, k, v


def setattr_all(obj, mapping):
    for name, value in mapping.items():
        setattr(obj, name, value)


@pytest.mark.parametrize("num_spec", [0, 2])
def test_relocation_moves_every_layer_of_the_slot(num_spec):
    """And moves exactly one slot, whatever `num_spec` says.

    Parametrized over `num_spec` precisely because the answer must not depend
    on it: the slot is the unit, so a wider request is more pairs, not a wider
    pair. Under the old group-width relocation this moved `1 + num_spec` slots
    and the two parametrizations disagreed.
    """
    stub, k, v = build(num_spec)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    assert torch.equal(k[:, 3], before_k[:, 1])
    assert torch.equal(v[:, 3], before_v[:, 1])
    # The source is untouched: relocation duplicates, the caller retires the
    # old index afterwards.
    assert torch.equal(k[:, 1], before_k[:, 1])


def test_relocation_leaves_every_other_slot_alone():
    stub, k, v = build(num_spec=2)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    for slot in range(SLOTS):
        if slot == 3:
            continue
        assert torch.equal(k[:, slot], before_k[:, slot])
        assert torch.equal(v[:, slot], before_v[:, slot])


def test_several_pairs_in_one_call():
    stub, k, _ = build(num_spec=1)
    before_k = k.clone()

    GDNStateMixin.relocate_state_slots(stub, [(0, 2), (1, 3)])

    for src, dst in ((0, 2), (1, 3)):
        assert torch.equal(k[:, dst], before_k[:, src])


def test_a_whole_request_is_relocated_one_slot_at_a_time():
    """A speculating request's slots are not a span and need not be adjacent.

    Written as its own case because the old contract was the opposite one, and
    getting it wrong is silent: a caller that still passes a base index and
    expects `1 + num_spec` slots to follow would move two slots it does not own
    and leave the request's real ones behind.
    """
    stub, k, _ = build(num_spec=2)
    before_k = k.clone()
    # Scattered on purpose -- this is what `pop_many` may return.
    request_slots = [7, 2, 9]
    targets = [1, 4, 6]

    GDNStateMixin.relocate_state_slots(stub, list(zip(request_slots, targets)))

    for src, dst in zip(request_slots, targets):
        assert torch.equal(k[:, dst], before_k[:, src])


def test_no_pairs_is_a_no_op():
    stub, k, v = build(num_spec=2)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.relocate_state_slots(stub, [])

    assert torch.equal(k, before_k)
    assert torch.equal(v, before_v)


def test_replayssm_records_and_cursor_travel_with_the_slot():
    """Under ReplaySSM the records ARE the state, not a cache in front of it.

    The checkpoint only describes the sequence up to its last flush; the
    records carry everything since. Relocate one without the other and the
    request resumes against the destination slot's previous tenant -- and
    because the cursor decides how many records get folded, a stale cursor
    corrupts the rebuild even when the records themselves moved.
    """
    stub, _, _ = build(num_spec=7, replayssm=True)
    runner = stub.model_runner
    names = (
        "replayssm_buf_k",
        "replayssm_buf_u",
        "replayssm_buf_g",
        "replayssm_write_pos",
    )
    before = {n: getattr(runner, n).clone() for n in names}

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    for name in names[:3]:
        moved = getattr(runner, name)
        assert torch.equal(moved[:, 3], before[name][:, 1]), f"{name} left behind"
        assert torch.equal(
            moved[:, 2], before[name][:, 2]
        ), f"{name} clobbered a neighbour"
    cursor = runner.replayssm_write_pos
    assert cursor[3] == before["replayssm_write_pos"][1], "cursor left behind"
    assert cursor[2] == before["replayssm_write_pos"][2], "cursor clobbered a neighbour"


def test_baseline_relocation_does_not_look_for_replay_buffers():
    """`replayssm` off means the runner has no record buffers at all; touching
    them would be an AttributeError, not a silent miss."""
    stub, k, _ = build(num_spec=2)
    assert not hasattr(stub.model_runner, "replayssm_buf_k")
    before_k = k.clone()

    GDNStateMixin.relocate_state_slots(stub, [(0, 2)])

    assert torch.equal(k[:, 2], before_k[:, 0])
