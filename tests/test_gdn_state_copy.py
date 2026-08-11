# SPDX-License-Identifier: MIT
# Tests for relocating a GDN state group's bytes.
#
# GDN checkpoints by forking, so this path is not about checkpoints: moving the
# state pool's boundary has to be able to shift a group out of the way, and that
# is a byte move whatever mechanism the class uses to checkpoint. The unit that
# moves is the whole group -- `1 + num_spec` consecutive slots, because the
# extra ones hold the per-draft states a rejected speculation rolls back to.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions.gdn_attn import GDNStateMixin

LAYERS = 3
GROUPS = 4
SHAPE_K = (2, 5)
SHAPE_V = (2, 3, 4)


def build(num_spec: int):
    """Caches whose every (layer, slot) plane carries a distinct value."""
    span = 1 + num_spec
    slots = GROUPS * span
    k = torch.zeros((LAYERS, slots) + SHAPE_K)
    v = torch.zeros((LAYERS, slots) + SHAPE_V)
    for layer in range(LAYERS):
        for slot in range(slots):
            k[layer, slot] = layer * 100 + slot
            v[layer, slot] = -(layer * 100 + slot)
    stub = SimpleNamespace(
        num_spec=num_spec,
        model_runner=SimpleNamespace(mamba_k_cache=k, mamba_v_cache=v),
    )
    return stub, k, v, span


@pytest.mark.parametrize("num_spec", [0, 2])
def test_copy_moves_every_layer_and_every_slot_of_the_group(num_spec):
    stub, k, v, span = build(num_spec)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.copy_state_entries(stub, [(1, 3)])

    src, dst = 1 * span, 3 * span
    assert torch.equal(k[:, dst : dst + span], before_k[:, src : src + span])
    assert torch.equal(v[:, dst : dst + span], before_v[:, src : src + span])
    # The source is untouched: relocation duplicates, the caller retires the
    # old index afterwards.
    assert torch.equal(k[:, src : src + span], before_k[:, src : src + span])


def test_copy_leaves_neighbouring_groups_alone():
    stub, k, v, span = build(num_spec=2)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.copy_state_entries(stub, [(1, 3)])

    for group in (0, 2):
        lo = group * span
        assert torch.equal(k[:, lo : lo + span], before_k[:, lo : lo + span])
        assert torch.equal(v[:, lo : lo + span], before_v[:, lo : lo + span])


def test_several_pairs_in_one_call():
    stub, k, _, span = build(num_spec=1)
    before_k = k.clone()

    GDNStateMixin.copy_state_entries(stub, [(0, 2), (1, 3)])

    for src, dst in ((0, 2), (1, 3)):
        lo_s, lo_d = src * span, dst * span
        assert torch.equal(k[:, lo_d : lo_d + span], before_k[:, lo_s : lo_s + span])


def test_no_pairs_is_a_no_op():
    stub, k, v, _ = build(num_spec=2)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.copy_state_entries(stub, [])

    assert torch.equal(k, before_k)
    assert torch.equal(v, before_v)


def test_a_group_is_not_one_slot_when_speculating():
    """The span is what distinguishes this from copying a single slot.

    Written as its own case because getting it wrong is silent: with `num_spec`
    slots left behind, a relocated request keeps drafting against another
    group's rollback states.
    """
    stub, k, _, span = build(num_spec=2)
    assert span == 3
    before_k = k.clone()

    GDNStateMixin.copy_state_entries(stub, [(0, 2)])

    for offset in range(span):
        assert torch.equal(k[:, 2 * span + offset], before_k[:, offset])
