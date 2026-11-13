# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`append_linear_rows`: grow one linear's output rows with other linears'
rows (and per-row scales), so a single GEMM produces all of them.

Driven with stand-in modules: the helper only reads ``quant_type``, ``bias``,
``weight``, ``weight_scale``, ``output_size`` and ``output_partition_sizes``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")  # atom.model_ops.linear imports aiter at module scope

from aiter import QuantType

from atom.model_ops.linear import append_linear_rows

K = 16


def _linear(rows, quant_type, seed, bias=False):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, K, generator=g)
    weight_scale = None
    if quant_type == QuantType.per_Token:
        w = w.to(torch.float8_e4m3fn)
        weight_scale = nn.Parameter(
            torch.rand(rows, 1, generator=g) + 0.5, requires_grad=False
        )
        weight_scale.weight_loader_process = "scale loader"
    weight = nn.Parameter(w, requires_grad=False)
    weight.weight_loader_process = "weight loader"
    return SimpleNamespace(
        weight=weight,
        weight_scale=weight_scale,
        bias=nn.Parameter(torch.zeros(rows)) if bias else None,
        quant_type=quant_type,
        output_size=rows,
        output_partition_sizes=[rows],
    )


def test_per_token_rows_and_scales_concatenate_in_order():
    """A per-output-channel scale is indexed by the axis being concatenated, so
    each row keeps its own scale: the grown scale is the concatenation."""
    dst = _linear(8, QuantType.per_Token, 0)
    tails = (_linear(2, QuantType.per_Token, 1), _linear(4, QuantType.per_Token, 2))
    mods = (dst, *tails)
    want_w = torch.cat([m.weight.data.view(torch.uint8) for m in mods])
    want_s = torch.cat([m.weight_scale.data for m in mods])

    append_linear_rows(dst, tails)

    assert torch.equal(dst.weight.data.view(torch.uint8), want_w)
    assert torch.equal(dst.weight_scale.data, want_s)
    # the tails are released, scales included; the modules stay as empty shells
    for tail in tails:
        assert tail.weight.numel() == 0 and tail.weight_scale.numel() == 0


def test_bf16_rows_concatenate_and_no_scale_appears():
    dst = _linear(8, QuantType.No, 0)
    tails = (_linear(4, QuantType.No, 1),)
    want = torch.cat([dst.weight.data, tails[0].weight.data])
    append_linear_rows(dst, tails)
    assert torch.equal(dst.weight.data, want)
    assert dst.weight_scale is None


@pytest.mark.parametrize("quant_type", [QuantType.No, QuantType.per_Token])
def test_sizes_grow_by_the_appended_rows(quant_type):
    dst = _linear(8, quant_type, 0)
    dst.output_partition_sizes = [4, 4]
    append_linear_rows(dst, (_linear(2, quant_type, 1), _linear(4, quant_type, 2)))
    assert dst.output_size == 14
    assert dst.output_partition_sizes == [4, 4, 2, 4]


@pytest.mark.parametrize("quant_type", [QuantType.No, QuantType.per_Token])
def test_the_grown_parameters_keep_their_loader_attributes(quant_type):
    """A later weight update (e.g. an RLHF rollout sync) goes through
    ``param.weight_loader_process``; the grown parameters must still carry it."""
    dst = _linear(8, quant_type, 0)
    weight, weight_scale = dst.weight, dst.weight_scale
    append_linear_rows(dst, (_linear(4, quant_type, 1),))
    assert dst.weight is weight
    assert dst.weight.weight_loader_process == "weight loader"
    if quant_type == QuantType.per_Token:
        assert dst.weight_scale is weight_scale
        assert dst.weight_scale.weight_loader_process == "scale loader"


@pytest.mark.parametrize("which", ["dst", "tail"])
def test_bias_is_refused(which):
    dst = _linear(8, QuantType.No, 0, bias=which == "dst")
    tail = _linear(4, QuantType.No, 1, bias=which == "tail")
    with pytest.raises(NotImplementedError, match="bias"):
        append_linear_rows(dst, (tail,))


def test_a_mixed_scheme_is_refused():
    """An unquantized tail beside an FP8 destination: concatenating them would
    mix dtypes."""
    dst = _linear(8, QuantType.per_Token, 0)
    with pytest.raises(NotImplementedError, match="one scheme"):
        append_linear_rows(dst, (_linear(4, QuantType.No, 1),))


def test_an_unsupported_scheme_is_refused():
    dst = _linear(8, QuantType.per_1x128, 0)
    with pytest.raises(NotImplementedError, match="per_Token"):
        append_linear_rows(dst, (_linear(4, QuantType.per_1x128, 1),))


def test_a_missing_per_token_scale_is_reported_by_the_guard():
    """The guard, not an AttributeError on `.data`, reports a missing scale."""
    dst = _linear(8, QuantType.per_Token, 0)
    tail = _linear(4, QuantType.per_Token, 1)
    tail.weight_scale = None
    with pytest.raises(ValueError, match="one row per output channel"):
        append_linear_rows(dst, (tail,))


class _ForeignQuantType:
    """A QuantType member as a differently identified aiter import yields it in
    a plugin process: same value, but unequal (and unhashable-equal) to this
    import's members -- and to each other."""

    def __init__(self, member):
        self.value = member.value
        self.name = member.name

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


def test_quant_types_from_another_aiter_import_are_matched_by_value():
    mods = (_linear(8, QuantType.per_Token, 0), _linear(4, QuantType.per_Token, 1))
    want_s = torch.cat([m.weight_scale.data for m in mods])
    for m in mods:
        m.quant_type = _ForeignQuantType(QuantType.per_Token)  # one each
    append_linear_rows(mods[0], mods[1:])
    assert torch.equal(mods[0].weight_scale.data, want_s)
