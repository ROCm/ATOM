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
        # Like the real models: off the first stage the embedding is missing.
        self.embed_tokens = nn.Identity() if with_embed else PPMissingLayer()


_CTX = types.SimpleNamespace(is_draft=False, ubatch_token_offset=0)


@pytest.fixture(autouse=True)
def _forward_context():
    _CTX.is_draft, _CTX.ubatch_token_offset = False, 0
    fc = types.SimpleNamespace(context=_CTX)
    with mock.patch("atom.spec_decode.drafter.get_forward_context", return_value=fc):
        yield _CTX


def _relay(layer_ids, n_layers, resident_range, *, first, last, hidden=8):
    target = _FakeTarget(n_layers, resident_range, with_embed=first)
    spec = AuxCaptureSpec(
        layer_ids=tuple(layer_ids), hidden_size=hidden, extract=lambda out, mod: out
    )
    group = types.SimpleNamespace(is_first_rank=first, is_last_rank=last)
    with mock.patch("atom.spec_decode.pp_aux_relay.get_pp_group", return_value=group):
        relay = PPAuxRelay(spec, target)
        relay.arm(target)
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
    assert not target.embed_tokens._forward_hooks


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
@pytest.mark.parametrize("layer_ids", [(-1, 2, 23, 47, 71, 89), (-1, 2, 89)])
def test_relay_round_trip_hands_the_drafter_each_tap_in_order(pp_size, layer_ids):
    # Stages that own no tap must still hand the last stage each tap's rows,
    # in `layer_ids` order.
    n_layers, hidden, n = 93, 8, 3
    payload, drafter = None, mock.Mock()
    for rank in range(pp_size):
        lo, hi = get_pp_indices(n_layers, rank, pp_size)
        last = rank == pp_size - 1
        relay, target = _relay(
            layer_ids, n_layers, (lo, hi), first=rank == 0, last=last, hidden=hidden
        )
        if last:
            relay.absorb(payload, drafter)
            break
        _run_target(relay, target, n, hidden)
        payload = relay.pack(payload, n)
    assert set(relay.incoming_ids) == {lid for lid in layer_ids if lid < lo}
    if not relay.incoming_ids:
        drafter.take_relayed_aux.assert_not_called()
        return
    (block,) = drafter.take_relayed_aux.call_args.args
    for col, lid in enumerate(relay.incoming_ids):
        torch.testing.assert_close(
            block[:, col * hidden : (col + 1) * hidden],
            torch.full((n, hidden), float(lid)),
        )


def test_aux_for_serves_the_relayed_block_uncopied_before_own_taps():
    from atom.spec_decode.drafter import Drafter

    hidden, n = 8, 3
    block = torch.randn(4, 2 * hidden)
    own = torch.randn(4, hidden)
    drafter = types.SimpleNamespace(
        _captures_aux=True,
        _aux_buffers=[torch.zeros(4, hidden), torch.zeros(4, hidden), own],
        _relayed_aux=block,
    )
    with mock.patch("atom.spec_decode.drafter.get_pcp_world_size", return_value=1):
        aux = Drafter.aux_for(drafter, torch.zeros(n, hidden))
    assert len(aux) == 2
    assert aux[0].data_ptr() == block.data_ptr() and aux[0].shape == (n, 2 * hidden)
    torch.testing.assert_close(aux[1], own[:n])


def test_last_stage_arms_only_its_own_taps():
    from atom.spec_decode.drafter import Drafter

    hidden = 8
    target = _FakeTarget(8, (4, 8), with_embed=False)
    spec = AuxCaptureSpec(
        layer_ids=(-1, 2, 5), hidden_size=hidden, extract=lambda out, mod: out
    )
    drafter = types.SimpleNamespace(
        _aux_capture_spec=lambda: spec,
        max_num_tokens=4,
        device="cpu",
        dtype=torch.float32,
    )
    drafter._make_aux_hook = lambda i, e: Drafter._make_aux_hook(drafter, i, e)
    Drafter.arm_aux_capture(drafter, target)
    assert [b.shape[0] for b in drafter._aux_buffers] == [0, 0, 4]
    assert not target.embed_tokens._forward_hooks
    assert not target.layers[2]._forward_hooks
    assert target.layers[5]._forward_hooks


@pytest.mark.parametrize("layer_ids", [(47, -1, 89, 2), (-1, 2, 2), (-1, 2, 93)])
def test_relay_rejects_the_ids_the_drafter_rejects(layer_ids):
    with pytest.raises(ValueError, match="strictly increasing|out of range"):
        _relay(layer_ids, 93, (0, 47), first=True, last=False)


def test_relay_skips_draft_forwards(_forward_context):
    relay, target = _relay([-1, 2], 8, (0, 4), first=True, last=False)
    _forward_context.is_draft = True
    _run_target(relay, target, 3, 8)
    assert relay._send is None


def test_relay_rejects_a_payload_of_the_wrong_width():
    relay, _ = _relay([-1, 2, 6], 8, (4, 8), first=False, last=True)
    with pytest.raises(RuntimeError, match="expected 16 aux columns"):
        relay.absorb(torch.zeros(3, 8), mock.Mock())


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


@pytest.mark.parametrize("world_size", [1, 2])
def test_build_takes_the_drafter_class_spec_and_arms_only_under_pp(world_size):
    target = _FakeTarget(8, (0, 4), with_embed=True)
    spec = AuxCaptureSpec(layer_ids=(-1, 2), hidden_size=8, extract=lambda o, m: o)
    # Every stage, drafter or not, takes the spec from the drafter class.
    drafter_cls = types.SimpleNamespace(aux_capture_spec=lambda config: spec)
    config = types.SimpleNamespace(speculative_config=object())
    group = types.SimpleNamespace(
        world_size=world_size, is_first_rank=True, is_last_rank=False
    )
    with (
        mock.patch("atom.spec_decode.pp_aux_relay.get_pp_group", return_value=group),
        mock.patch(
            "atom.spec_decode.pp_aux_relay.drafter_class", return_value=drafter_cls
        ),
    ):
        relay = PPAuxRelay.build(config, target)
    if world_size == 1:
        assert relay is None
        return
    assert relay.spec is spec and relay.own_ids == (-1, 2)
    assert target.embed_tokens._forward_hooks, "build must arm the relay"


def test_relay_hook_compiles_without_an_in_place_update():
    # Regression: `_filled.add` inside the compiled backbone made Dynamo replay
    # the set via `update`, which trips the cudagraph bytecode hook's depyf
    # import on images without depyf (PP2 server warmup crashed).
    from torch._dynamo.convert_frame import register_bytecode_hook

    names = []
    handle = register_bytecode_hook(lambda old, new: names.extend(new.co_names))
    try:
        relay, target = _relay([-1, 2], 8, (0, 4), first=True, last=False)
        fwd = torch.compile(
            lambda x: target.layers[2](target.embed_tokens(x) + 1), backend="eager"
        )
        for _ in range(2):
            fwd(torch.randn(3, 8))
            assert relay.pack(None, 3).shape == (3, 16)
    finally:
        handle.remove()
        torch._dynamo.reset()
    assert "update" not in names
