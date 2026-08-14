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


def build(num_spec: int = 0):
    """Caches whose every (layer, slot) plane carries a distinct value."""
    k = torch.zeros((LAYERS, SLOTS) + SHAPE_K)
    v = torch.zeros((LAYERS, SLOTS) + SHAPE_V)
    for layer in range(LAYERS):
        for slot in range(SLOTS):
            k[layer, slot] = layer * 100 + slot
            v[layer, slot] = -(layer * 100 + slot)
    stub = SimpleNamespace(
        num_spec=num_spec,
        model_runner=SimpleNamespace(mamba_k_cache=k, mamba_v_cache=v),
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
