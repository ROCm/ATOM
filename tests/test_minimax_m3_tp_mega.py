# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3 under TP MegaMoE: a MoE layer's output is all-reduced exactly
once a step, whichever path the step takes.

The kernel all-reduces the steps it takes; every other step leaves the output
TP-partial for the next fused all-reduce + norm. The MoE block, the next
layer's input norm, the final norm and the DSpark tap each decide from the
step's row count (`MiniMaxM3MoE.tp_mega_serves`), so they must agree for
every row count, served or not.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("aiter", reason="the MiniMax-M3 model module imports aiter")

import torch

from atom.models.minimax_m3 import model as m3

SERVED_MAX = 64


@pytest.fixture
def stack(monkeypatch):
    """A TP MegaMoE block and the norm after it, with a backend serving up to
    SERVED_MAX rows, and a log of every all-reduce of the block's output."""
    context = {}
    monkeypatch.setattr(
        m3,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context)
        ),
    )
    reductions = []

    def kernel(experts, hidden, logits):
        reductions.append("kernel")
        return hidden * 3

    backend = SimpleNamespace(runs=lambda rows: 0 < rows <= SERVED_MAX, forward=kernel)
    moe = object.__new__(m3.MiniMaxM3MoE)
    vars(moe).update(
        tp_mega=True,
        layer_name="layers.0.block_sparse_moe",
        _forward=lambda hidden: hidden * 2,
        gate=lambda hidden: hidden[:, :2],
    )
    object.__setattr__(moe, "experts", SimpleNamespace(sp_mega_moe=backend))
    context[moe.layer_name] = moe

    def fused(hidden, residual, norm, reduced=False):
        if not reduced:
            reductions.append("norm")
        return hidden + residual, hidden + residual

    monkeypatch.setattr(m3, "fused_allreduce_gemma_rms_norm", fused)

    def all_reduce(hidden):
        reductions.append("tap")
        return hidden

    monkeypatch.setattr(m3, "tensor_model_parallel_all_reduce", all_reduce)
    norm = m3._TPMegaInputNorm(Mock(), moe.layer_name)
    return SimpleNamespace(moe=moe, norm=norm, reductions=reductions)


@pytest.mark.parametrize("rows", [1, SERVED_MAX, SERVED_MAX + 1, 4096])
def test_each_step_reduces_the_moe_output_exactly_once(stack, rows):
    hidden = torch.ones(rows, 4)
    out = stack.moe.tp_mega_forward(hidden)
    served = rows <= SERVED_MAX
    torch.testing.assert_close(out, hidden * (3 if served else 2))
    stack.norm.add_norm(out, torch.zeros_like(out))
    assert stack.reductions == ["kernel" if served else "norm"]


@pytest.mark.parametrize("rows", [1, SERVED_MAX + 1])
def test_the_dspark_tap_reduces_only_an_unreduced_output(stack, rows):
    stack.moe.tp_mega_layer_output(torch.ones(rows, 4), torch.ones(rows, 4))
    assert stack.reductions == ([] if rows <= SERVED_MAX else ["tap"])


def test_before_the_backend_exists_every_step_is_the_standard_path(stack):
    stack.moe.experts.sp_mega_moe = None
    out = stack.moe.tp_mega_forward(torch.ones(8, 4))
    stack.norm.add_norm(out, torch.zeros_like(out))
    assert stack.reductions == ["norm"]


def test_sibling_layer_prefix():
    assert m3._sibling_layer_prefix("model.layers.7", 6) == "model.layers.6"
    with pytest.raises(ValueError):
        m3._sibling_layer_prefix("model.layers", 6)
