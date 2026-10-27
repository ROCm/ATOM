# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which PP stage owns which aux tap, and what each stage hooks.

Not CPU-only: `pp_aux_relay` imports `spec_decode.drafter`, which imports the
model loader and so the real aiter. Stubbing that chain would mean stubbing
the loader's whole quant surface, so this file skips on the non-GPU runner.
"""

import types
from unittest import mock

import pytest
import torch
from torch import nn

pytest.importorskip(
    "atom.spec_decode.pp_aux_relay",
    reason="requires AITER to import the drafter chain",
    exc_type=ImportError,
)

from atom.models.utils import PPMissingLayer, get_pp_indices
from atom.spec_decode.drafter import AuxCaptureSpec
from atom.spec_decode.pp_aux_relay import PPAuxRelay


class _FakeTarget(nn.Module):
    """A PP slice of a target: layers outside the stage are `PPMissingLayer`."""

    def __init__(self, n_layers, resident_range, with_embed):
        super().__init__()
        lo, hi = resident_range
        self.layers = nn.ModuleList(
            nn.Identity() if lo <= i < hi else PPMissingLayer() for i in range(n_layers)
        )
        if with_embed:
            self.embed_tokens = nn.Identity()


def _relay(layer_ids, n_layers, resident_range, *, first, last, hidden=8):
    target = _FakeTarget(n_layers, resident_range, with_embed=first)
    spec = AuxCaptureSpec(
        layer_ids=tuple(layer_ids), hidden_size=hidden, extract=lambda out, mod: out
    )
    group = types.SimpleNamespace(is_first_rank=first, is_last_rank=last)
    with mock.patch("atom.spec_decode.pp_aux_relay.get_pp_group", return_value=group):
        relay = PPAuxRelay(spec, target, 4, torch.device("cpu"), torch.float32)
        relay.arm(target, lambda: False)
    return relay, target


def test_relay_embedding_tap_hooks_the_embedding_not_the_last_layer():
    # Regression: `layers[-1]` is the LAST decoder layer, a PPMissingLayer on
    # the first stage, so the hook silently never fired and the relayed buffer
    # stayed zero.
    relay, target = _relay([-1, 2], 8, (0, 4), first=True, last=False)
    assert relay.own_ids == (-1, 2)
    assert target.embed_tokens._forward_hooks, "embedding tap was not hooked"
    assert not target.layers[-1]._forward_hooks
    assert not target.layers[-1]._forward_pre_hooks


def test_relay_embedding_tap_captures_through_the_hook():
    relay, target = _relay([-1, 2], 8, (0, 4), first=True, last=False)
    rows = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)
    target.embed_tokens(rows)
    torch.testing.assert_close(relay.buffers[-1][:2], rows)


def test_relay_embedding_tap_ignores_capture_input():
    # -1 taps the embedding OUTPUT even when the spec captures layer inputs.
    _, target = _relay([-1, 2], 8, (0, 4), first=True, last=False)
    assert not target.embed_tokens._forward_pre_hooks


def test_relay_embedding_tap_is_incoming_on_a_later_stage():
    relay, target = _relay([-1, 2, 6], 8, (4, 8), first=False, last=False)
    assert relay.own_ids == (6,)
    assert relay.incoming_ids == (-1, 2)
    assert relay.outgoing_ids == (-1, 2, 6)
    assert not hasattr(target, "embed_tokens")


@pytest.mark.parametrize("pp_size", [2, 4, 8])
def test_relay_tap_split_partitions_the_layer_ids(pp_size):
    layer_ids = (-1, 2, 23, 47, 71, 89)
    n_layers, seen = 93, []
    for rank in range(pp_size):
        lo, hi = get_pp_indices(n_layers, rank, pp_size)
        relay, _ = _relay(
            layer_ids,
            n_layers,
            (lo, hi),
            first=rank == 0,
            last=rank == pp_size - 1,
        )
        seen.extend(relay.own_ids)
        # What a stage forwards is exactly what it was handed plus its own.
        expected = () if rank == pp_size - 1 else relay.incoming_ids + relay.own_ids
        assert relay.outgoing_ids == expected
    assert tuple(seen) == layer_ids, "every tap is owned by exactly one stage"
