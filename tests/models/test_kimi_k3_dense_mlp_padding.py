# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 dense-MLP K padding: the load-time layout, GPU-free.

The dense MLP's per-rank intermediate is 33792 / TP8 = 4224, which the
preshuffled MXFP4 GEMMs cannot handle (no K-tail masking). ``KimiMLP`` pads it
to 4352. These tests pin the part that runs before any kernel: each checkpoint
tensor is re-laid so the stock tensor-parallel loader -- which simply chunks the
full tensor by TP -- puts every rank's real slice at the front of its padded
slice, with zero weights and finite scales behind it.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.models.kimi_k3.padding import (
    pad_param_loads,
    pad_tp_chunks,
    padded_local_size,
)

TP = 8
REAL, PADDED = 33792 // TP, 4352  # 4224 -> 4352 per rank


@pytest.mark.parametrize(
    "real,expected",
    [(4224, 4352), (4352, 4352), (8448, 8448), (16896, 16896), (768, 768)],
)
def test_padded_local_size(real, expected):
    """Only Kimi-K3's TP8 dense MLP moves; TP4/TP2 and the shared experts don't."""
    assert padded_local_size(real, 256) == expected


def _rank_slices(t, dim):
    return t.chunk(TP, dim=dim)


# (dim, per-rank units of the real tensor, per-rank units after padding)
#   gate_up weight rows (dim 0): K-independent, one row per output channel
#   down_proj fp4x2 bytes (dim 1): half the elements
#   down_proj e8m0 scale groups (dim 1): one per 32 elements
LAYOUTS = {
    "gate_up rows": (0, REAL, PADDED),
    "down fp4x2 bytes": (1, REAL // 2, PADDED // 2),
    "down e8m0 groups": (1, REAL // 32, PADDED // 32),
}


@pytest.mark.parametrize("layout", LAYOUTS, ids=list(LAYOUTS))
def test_stock_chunking_lands_each_rank_at_the_front(layout):
    dim, real_units, padded_units = LAYOUTS[layout]
    other = 3
    shape = [other, other]
    shape[dim] = real_units * TP
    t = torch.arange(torch.Size(shape).numel(), dtype=torch.int32).reshape(shape)

    padded = pad_tp_chunks(t, dim, TP, REAL, PADDED, fill=-1)

    assert padded.shape[dim] == padded_units * TP
    for rank, (got, want) in enumerate(
        zip(_rank_slices(padded, dim), _rank_slices(t, dim), strict=True)
    ):
        assert got.shape[dim] == padded_units, rank
        assert torch.equal(got.narrow(dim, 0, real_units), want), rank
        tail = got.narrow(dim, real_units, padded_units - real_units)
        assert (tail == -1).all(), rank


def test_a_size_that_does_not_split_by_tp_is_refused():
    with pytest.raises(AssertionError, match="cannot pad"):
        pad_tp_chunks(torch.zeros(10, 4), 0, TP, REAL, PADDED, fill=0)


def _linear_double(scale_dtype):
    calls = {}

    def recorder(name):
        def loader(param, loaded, *args):
            calls[name] = (loaded, args)

        return loader

    weight = nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
    weight.weight_loader = recorder("weight")
    scale = nn.Parameter(torch.empty(0, dtype=scale_dtype), requires_grad=False)
    scale.weight_loader = recorder("weight_scale")
    return SimpleNamespace(weight=weight, weight_scale=scale), calls


@pytest.mark.parametrize(
    "scale_dtype,scale_fill", [(torch.uint8, 127), (torch.float32, 1.0)]
)
def test_loaders_pad_with_zero_weights_and_finite_scales(scale_dtype, scale_fill):
    """Zero weights make the tail contribute nothing whatever its scale; the
    scale only has to be finite (e8m0 127 == 2**0)."""
    linear, calls = _linear_double(scale_dtype)
    pad_param_loads(linear, 1, TP, REAL, PADDED)

    w = torch.full((2, REAL // 2 * TP), 7, dtype=torch.uint8)
    s = torch.full((2, REAL // 32 * TP), 9, dtype=scale_dtype)
    linear.weight.weight_loader(linear.weight, w)
    linear.weight_scale.weight_loader(linear.weight_scale, s)

    w_tail = calls["weight"][0].chunk(TP, dim=1)[0][:, REAL // 2 :]
    s_tail = calls["weight_scale"][0].chunk(TP, dim=1)[0][:, REAL // 32 :]
    assert (w_tail == 0).all()
    assert (s_tail == scale_fill).all()


def test_loaders_pass_the_shard_id_through():
    """MergedColumnParallelLinear's loader takes the gate/up shard id."""
    linear, calls = _linear_double(torch.uint8)
    pad_param_loads(linear, 0, TP, REAL, PADDED)
    linear.weight.weight_loader(
        linear.weight, torch.zeros(REAL * TP, 4, dtype=torch.uint8), 1
    )
    assert calls["weight"][1] == (1,)


def test_a_missing_scale_param_is_skipped():
    """An unquantized (bf16) layer has no weight_scale to wrap."""
    linear, _ = _linear_double(torch.uint8)
    linear.weight_scale = None
    pad_param_loads(linear, 0, TP, REAL, PADDED)
