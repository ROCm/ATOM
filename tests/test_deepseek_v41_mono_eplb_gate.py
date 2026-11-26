# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""V4.1 mono under EPLB: the router's gate in slot order (``mono.eplb_gate``)."""

from types import SimpleNamespace

import pytest
import torch
from conftest import gpu_stack_import

from atom.models.deepseek_v41.config import AttentionMode
from atom.models.deepseek_v41.mono.weights import LayerWeights

EXPERTS, HIDDEN, TOPK, EP = 16, 8, 3, 4
PLACED = 2  # layers 0..1 placed, layer 2 past ``eplb_layer_end``


def _layer(layer_id, gen):
    return LayerWeights(
        layer_id,
        AttentionMode.FULL,
        {
            "ffn.gate.weight": torch.randn(EXPERTS, HIDDEN, generator=gen),
            "ffn.gate.e_score_correction_bias": torch.randn(EXPERTS, generator=gen),
            "ffn.experts.w13_weight": torch.zeros(EXPERTS // EP, 1),
        },
    )


def _placement(gen):
    return torch.stack(
        [torch.randperm(EXPERTS, generator=gen) for _ in range(PLACED)]
    ).to(torch.int32)


@pytest.fixture
def gate(monkeypatch):
    mod = gpu_stack_import("atom.models.deepseek_v41.mono.eplb_gate")
    gen = torch.Generator().manual_seed(0)
    meta = SimpleNamespace(num_layers=PLACED, physical_to_logical_map=_placement(gen))
    listeners = []
    monkeypatch.setattr(mod, "get_live_expert_location_metadata", lambda: meta)
    monkeypatch.setattr(mod, "add_eplb_commit_listener", listeners.append)
    layers = [_layer(i, gen) for i in range(PLACED + 1)]
    return SimpleNamespace(
        mod=mod, meta=meta, listeners=listeners, layers=layers, gen=gen
    )


def _slots(layer, x):
    """The router's picks over the gate's rows (the kernel's: biased top-k)."""
    scores = x @ layer["ffn.gate.weight"].T + layer["ffn.gate.e_score_correction_bias"]
    return scores.topk(TOPK, dim=-1).indices


def _assert_routes_to_placement(original, bound, p2l, x):
    logical = _slots(original, x)
    slots = _slots(bound, x)
    # slot s holds expert p2l[s]: the bound gate's slots hold the original picks
    assert torch.equal(p2l[slots].long(), logical)


def test_placed_layers_route_to_the_slots_holding_their_experts(gate):
    bound = gate.mod.bind_eplb_gates(gate.layers)
    x = torch.randn(32, HIDDEN, generator=gate.gen)
    for layer_id in range(PLACED):
        _assert_routes_to_placement(
            gate.layers[layer_id],
            bound[layer_id],
            gate.meta.physical_to_logical_map[layer_id],
            x,
        )
        # the original model's gate is left as it was (the prefill path's)
        assert bound[layer_id]["ffn.gate.weight"] is not (
            gate.layers[layer_id]["ffn.gate.weight"]
        )
    # past the placed layers the gate is the model's own
    assert bound[PLACED] is gate.layers[PLACED]
    # the experts themselves are read in place
    for old, new in zip(gate.layers, bound):
        assert new["ffn.experts.w13_weight"] is old["ffn.experts.w13_weight"]


def test_a_commit_reorders_its_layer_in_place(gate):
    bound = gate.mod.bind_eplb_gates(gate.layers)
    (listener,) = gate.listeners
    held = {i: bound[i]["ffn.gate.weight"].data_ptr() for i in range(PLACED)}
    before = bound[0]["ffn.gate.weight"].clone()
    # EPLB commits a new placement of layer 1 alone (``ExpertLocationMetadata.update``)
    p2l = gate.meta.physical_to_logical_map
    p2l[1].copy_(torch.randperm(EXPERTS, generator=gate.gen).to(torch.int32))
    x = torch.randn(32, HIDDEN, generator=gate.gen)
    with pytest.raises(AssertionError):
        _assert_routes_to_placement(gate.layers[1], bound[1], p2l[1], x)
    listener(1)
    _assert_routes_to_placement(gate.layers[1], bound[1], p2l[1], x)
    # the kernels hold the copies' addresses: a commit writes them in place
    assert all(bound[i]["ffn.gate.weight"].data_ptr() == held[i] for i in held)
    assert torch.equal(bound[0]["ffn.gate.weight"], before)


def test_without_eplb_the_layers_are_unchanged(monkeypatch):
    mod = gpu_stack_import("atom.models.deepseek_v41.mono.eplb_gate")
    monkeypatch.setattr(mod, "get_live_expert_location_metadata", lambda: None)
    layers = [_layer(0, torch.Generator().manual_seed(1))]
    assert mod.bind_eplb_gates(layers) is layers
