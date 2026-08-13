# SPDX-License-Identifier: MIT
# Tests for relocating a GDN state slot's bytes.
#
# GDN checkpoints by forking, so this path is not about checkpoints: moving the
# state pool's boundary has to be able to shift a slot out of the way, and that
# is a byte move whatever mechanism the class uses to checkpoint.
#
# The unit that moves is ONE slot. A request under speculative decoding holds
# `1 + num_spec` of them, but they are allocated one at a time and need not be
# adjacent, so relocating a whole request is several of these calls and the
# caller names each slot. `num_spec` therefore does not appear in this path at
# all -- which is the property most of these cases are pinning down.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions.gdn_attn import GDNStateMixin

LAYERS = 3
SLOTS = 12
SHAPE_K = (2, 5)
SHAPE_V = (2, 3, 4)


def build(num_spec: int = 0, replayssm: bool = False):
    """Caches whose every (layer, slot) plane carries a distinct value."""
    k = torch.zeros((LAYERS, SLOTS) + SHAPE_K)
    v = torch.zeros((LAYERS, SLOTS) + SHAPE_V)
    for layer in range(LAYERS):
        for slot in range(SLOTS):
            k[layer, slot] = layer * 100 + slot
            v[layer, slot] = -(layer * 100 + slot)
    runner = SimpleNamespace(mamba_k_cache=k, mamba_v_cache=v)
    if replayssm:
        # The record buffers share the (layer, slot) axes, so they carry the
        # same distinct values; the cursor is per-slot and one-dimensional.
        for name, shape in (
            ("replayssm_buf_k", SHAPE_K),
            ("replayssm_buf_u", SHAPE_V),
            ("replayssm_buf_g", (2,)),
        ):
            buf = torch.zeros((LAYERS, SLOTS) + shape)
            for layer in range(LAYERS):
                for slot in range(SLOTS):
                    buf[layer, slot] = layer * 100 + slot
            setattr(runner, name, buf)
        runner.replayssm_write_pos = torch.arange(SLOTS, dtype=torch.int32)
    stub = SimpleNamespace(
        num_spec=num_spec,
        replayssm=replayssm,
        model_runner=runner,
    )
    return stub, k, v


@pytest.mark.parametrize("num_spec", [0, 2])
def test_relocation_moves_every_layer_of_the_slot(num_spec):
    """And moves exactly the one slot, whatever `num_spec` says.

    Parametrized over `num_spec` precisely because the answer must not depend
    on it: the slot is the unit, and a wider request is more calls, not a wider
    call.
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
    """Including a scattered set, which is what a whole request looks like now.

    (0, 2) and (1, 7) are one request's two slots going to two destinations
    that are neither adjacent to each other nor at a fixed offset from the
    sources. Nothing in this path may assume otherwise.
    """
    stub, k, _ = build(num_spec=1)
    before_k = k.clone()

    GDNStateMixin.relocate_state_slots(stub, [(0, 2), (1, 7)])

    assert torch.equal(k[:, 2], before_k[:, 0])
    assert torch.equal(k[:, 7], before_k[:, 1])


def test_no_pairs_is_a_no_op():
    stub, k, v = build(num_spec=2)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.relocate_state_slots(stub, [])

    assert torch.equal(k, before_k)
    assert torch.equal(v, before_v)


# --- ReplaySSM -------------------------------------------------------------
# Under ReplaySSM a slot's state is not the two caches alone: the (k, u, g)
# records written since the checkpoint, and the cursor saying how many there
# are, are as much a part of it. A relocation that moved only the caches would
# leave the destination reading the previous tenant's decode history -- which
# is silent, because every tensor involved is validly shaped and populated.


def test_relocation_moves_the_record_buffers_too():
    stub, _, _ = build(num_spec=2, replayssm=True)
    runner = stub.model_runner
    before = {
        name: getattr(runner, name).clone()
        for name in ("replayssm_buf_k", "replayssm_buf_u", "replayssm_buf_g")
    }

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    for name, was in before.items():
        now = getattr(runner, name)
        assert torch.equal(now[:, 3], was[:, 1])
        # Every other slot untouched, same as the caches.
        for slot in range(SLOTS):
            if slot != 3:
                assert torch.equal(now[:, slot], was[:, slot])


def test_relocation_moves_the_write_cursor():
    """The cursor is what makes the records mean anything.

    It is indexed by slot rather than by (layer, slot) -- one cursor serves
    every linear-attention layer -- so it moves by a different call than the
    buffers and is worth its own case.
    """
    stub, _, _ = build(num_spec=2, replayssm=True)
    write_pos = stub.model_runner.replayssm_write_pos
    before = write_pos.clone()

    GDNStateMixin.relocate_state_slots(stub, [(1, 3), (0, 7)])

    assert write_pos[3] == before[1]
    assert write_pos[7] == before[0]
    for slot in range(SLOTS):
        if slot not in (3, 7):
            assert write_pos[slot] == before[slot]


def test_relocation_ignores_replay_buffers_when_disabled():
    """A stub with no ReplaySSM attributes at all must still relocate.

    ReplaySSM is off by default, so the baseline path may not so much as look
    at `replayssm_buf_*` -- reading them would turn the default configuration
    into an AttributeError.
    """
    stub, k, _ = build(num_spec=2)
    assert not hasattr(stub.model_runner, "replayssm_buf_k")
    before_k = k.clone()

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    assert torch.equal(k[:, 3], before_k[:, 1])
