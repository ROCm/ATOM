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
        relay = PPAuxRelay(spec, target)
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


def _run_target(relay, target, n, hidden):
    """A target forward on this stage: each own tap's module outputs `lid`."""
    for lid in relay.own_ids:
        module = target.embed_tokens if lid == -1 else target.layers[lid]
        module(torch.full((n, hidden), float(lid)))


def test_relay_embedding_tap_captures_through_the_hook():
    relay, target = _relay([-1, 2], 8, (0, 4), first=True, last=False)
    rows = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)
    target.embed_tokens(rows)
    target.layers[2](torch.zeros(2, 8))
    torch.testing.assert_close(relay.pack(None, 2)[:, :8], rows)


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


@pytest.mark.parametrize("pp_size", [2, 4, 8])
@pytest.mark.parametrize(
    "layer_ids", [(-1, 2, 23, 47, 71, 89), (47, -1, 89, 2), (-1, 2, 89)]
)
def test_relay_round_trip_lands_each_tap_in_its_drafter_buffer(pp_size, layer_ids):
    # Unsorted taps, and stages that own none, must still hand the last stage
    # each tap's rows in the drafter's `layer_ids` slot.
    n_layers, hidden, n = 93, 8, 3
    payload = None
    for rank in range(pp_size):
        lo, hi = get_pp_indices(n_layers, rank, pp_size)
        last = rank == pp_size - 1
        relay, target = _relay(
            layer_ids, n_layers, (lo, hi), first=rank == 0, last=last, hidden=hidden
        )
        if last:
            aux = [torch.zeros(4, hidden) for _ in layer_ids]
            relay.absorb(payload, aux)
            break
        _run_target(relay, target, n, hidden)
        payload = relay.pack(payload, n)
    for lid, buf in zip(layer_ids, aux):
        if lid in relay.incoming_ids:
            torch.testing.assert_close(buf[:n], torch.full((n, hidden), float(lid)))
    assert set(relay.incoming_ids) == {lid for lid in layer_ids if lid < lo}


def test_relay_rejects_a_payload_of_the_wrong_width():
    relay, _ = _relay([-1, 2, 6], 8, (4, 8), first=False, last=True)
    with pytest.raises(RuntimeError, match="expected 16 aux columns"):
        relay.absorb(torch.zeros(3, 8), [torch.zeros(4, 8) for _ in range(3)])


def test_relay_send_is_sized_to_the_forward_rows():
    # Nothing resident: each forward's first own tap allocates its own rows.
    relay, target = _relay([-1, 2, 6], 8, (4, 8), first=False, last=False)
    assert relay._send is None
    for n in (3, 5, 2):
        _run_target(relay, target, n, 8)
        sent = relay.pack(torch.ones(n, 16), n)
        assert sent.shape == (n, 24)
        torch.testing.assert_close(sent[:, :16], torch.ones(n, 16))
        torch.testing.assert_close(sent[:, 16:], torch.full((n, 8), 6.0))
        assert relay._send is None, "pack must hand the buffer over"
