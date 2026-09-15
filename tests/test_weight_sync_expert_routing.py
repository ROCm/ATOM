# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How an expert weight reaches the fused buffers, and what happens if it
doesn't reach them the same way twice.

Two properties, both about the relayout. `weight_loader` writes plain
row-major bytes over buffers the kernel reads through aiter's per-expert
permutation, so every route into those buffers owes the same three things:
refuse the combinations the path does not implement, write, and register the
slices it wrote so the layout is re-established once at the end.

A route that skips the registration is silent -- the sync still reports
`updated=N` -- and so is a sync that registers a slice, shuffles it, and then
leaves the registration behind for the next sync to shuffle again. Per
`_finalize_expert_weight_sync`'s own docstring: shuffling an already-shuffled
slice does not undo the first shuffle, it produces a third layout.
"""

import pytest
import torch
from torch import nn

from atom.rollout.weight_updater import WeightUpdaterMixin

needs_aiter = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="atom.model_ops.utils imports aiter, which resolves the chip "
    "architecture through rocminfo",
)

HIDDEN = 8
INTERMEDIATE = 4
EXPERTS = 2


def _updater(model):
    class _Updater(WeightUpdaterMixin):
        device = torch.device("cpu")
        label = "test"
        rank = 0
        world_size = 1

        def clear_kv_cache(self):
            pass

    updater = _Updater()
    updater.model = model
    return updater


def _moe_model(dtype=torch.bfloat16, **module_attrs):
    """A layer holding the fused expert buffers under ATOM's own names."""
    experts = nn.Module()
    experts.w13_weight = nn.Parameter(
        torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=dtype),
        requires_grad=False,
    )
    experts.w2_weight = nn.Parameter(
        torch.zeros(EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype),
        requires_grad=False,
    )
    experts.weight_loader = lambda *a, **k: None
    experts.expert_map = None
    experts.num_redundant_experts = 0
    for name, value in module_attrs.items():
        setattr(experts, name, value)
    mlp = nn.Module()
    mlp.experts = experts
    model = nn.Module()
    model.mlp = mlp
    return model, experts


# ── the route a trainer takes when it mirrors ATOM's own state dict ────────


def test_an_atom_named_expert_buffer_is_not_a_plain_parameter(monkeypatch):
    """`w13_weight` is a real parameter of the FusedMoE, so it resolves in
    `_get_param_to_module_mapping` and never reaches `_apply_unmatched_weight`.

    Down the plain dispatch it is a row-major `copy_` into a buffer the kernel
    reads through the expert permutation, with nothing registered for relayout
    and `updated` counting it as a success.
    """
    model, experts = _moe_model()
    updater = _updater(model)
    routed = []
    monkeypatch.setattr(
        type(updater),
        "_apply_named_expert_buffer",
        lambda self, *a: routed.append(a[0]),
        raising=True,
    )

    assert "mlp.experts.w13_weight" in updater._get_param_to_module_mapping()
    assert (
        updater.update_weights(
            [("mlp.experts.w13_weight", torch.ones_like(experts.w13_weight))]
        )
        == 1
    )

    assert routed == ["mlp.experts.w13_weight"]


@pytest.mark.parametrize("param_name", ["w13_weight", "w2_weight"])
def test_a_named_expert_buffer_registers_every_slice(param_name):
    """One tensor covers every expert, so every slice of the buffer is new and
    all of them need the layout re-established."""
    model, experts = _moe_model()
    updater = _updater(model)
    param = getattr(experts, param_name)
    incoming = torch.full_like(param, 3.0)

    updater._apply_named_expert_buffer(
        f"mlp.experts.{param_name}", param_name, experts, param, incoming
    )

    assert param.eq(3.0).all(), "the write did not land"
    pending = updater._pending_expert_relayout
    assert list(pending) == [(experts, param_name)]
    arrived = pending[(experts, param_name)]
    assert sorted(arrived) == list(range(EXPERTS))
    expected = {"w1", "w3"} if param_name == "w13_weight" else {"w2"}
    assert all(shards == expected for shards in arrived.values())


def test_a_partial_named_expert_buffer_is_refused():
    """Re-establishing the layout works on whole expert slices, so a write
    that covers part of one cannot be relaid out."""
    model, experts = _moe_model()
    updater = _updater(model)

    with pytest.raises(NotImplementedError, match="fused expert buffer"):
        updater._apply_named_expert_buffer(
            "mlp.experts.w13_weight",
            "w13_weight",
            experts,
            experts.w13_weight,
            torch.zeros(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16),
        )

    assert not updater._pending_expert_relayout


def test_a_named_expert_buffer_is_refused_on_a_quantized_moe():
    """The plain dispatch would have written it and reported updated=1.
    `_check_expert_sync_supported` is the check this route was missing."""
    model, experts = _moe_model(dtype=torch.float8_e4m3fnuz)
    updater = _updater(model)

    with pytest.raises(NotImplementedError, match="quantized storage format"):
        updater._apply_named_expert_buffer(
            "mlp.experts.w13_weight",
            "w13_weight",
            experts,
            experts.w13_weight,
            torch.zeros_like(experts.w13_weight),
        )


def test_a_named_expert_buffer_is_refused_under_expert_parallelism():
    model, experts = _moe_model(expert_map=torch.zeros(EXPERTS, dtype=torch.int32))
    updater = _updater(model)

    with pytest.raises(NotImplementedError, match="expert-parallel"):
        updater._apply_named_expert_buffer(
            "mlp.experts.w13_weight",
            "w13_weight",
            experts,
            experts.w13_weight,
            torch.zeros_like(experts.w13_weight),
        )


# ── a sync that fails part way through ────────────────────────────────────


def test_finalize_is_a_no_op_with_nothing_pending():
    """And does not import aiter to find that out, which is why the rest of
    this file's expert-routing tests run on a CPU box."""
    model, _ = _moe_model()
    updater = _updater(model)

    updater._finalize_expert_weight_sync()

    assert not updater._pending_expert_relayout


def test_the_updater_needs_no_expert_mapping_for_this_route():
    """`get_expert_mapping` is how a *checkpoint*-named expert is resolved; a
    buffer sent under ATOM's own name never consults it."""
    model, experts = _moe_model()
    assert not hasattr(model, "get_expert_mapping")
    updater = _updater(model)

    updater._apply_named_expert_buffer(
        "mlp.experts.w2_weight",
        "w2_weight",
        experts,
        experts.w2_weight,
        torch.full_like(experts.w2_weight, 5.0),
    )

    assert experts.w2_weight.eq(5.0).all()
