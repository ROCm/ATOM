# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`strided_per_token_quant` must be opaque to torch.compile, not traced.

It moved onto the traced path when the KDA splitting-op boundary shrank to
conv1d-onward: `KimiKDAAttention._kda_prologue` now runs inside the compiled
graph, and it calls this quant. The body is not traceable -- it branches on
`t == 0` (a data-dependent guard on the dynamic token dim), asserts with
f-strings over `x.shape`, and launches on a symbolic grid `(t,)` -- so it is
registered as a custom op and Dynamo sees one node instead.

These tests are CPU-only: fake tensors never allocate, so the CUDA impl is
never reached. They pin the fake's contract (which is all the compiler sees)
and the op's opacity.
"""

from __future__ import annotations

import operator

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

import atom.model_ops.kimi_k3.quant  # noqa: F401  -- registers the op

FP8 = torch.float8_e4m3fn
OP = torch.ops.aiter.kimi_k3_strided_per_token_quant


def _fake_quant(shape, *, slice_cols=None):
    """Run the op under FakeTensorMode, optionally on a column slice."""
    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        if slice_cols is not None:
            lo, hi = slice_cols
            x = x[..., lo:hi]
        return OP(x, FP8), x


def test_fake_returns_the_documented_shapes_and_dtypes():
    """`(quantized [T, D], scale [T, 1] float32)` -- the layout a per-token a8w8
    GEMM takes as `x_scale=`. A wrong fake here miscompiles silently."""
    (q, s), _ = _fake_quant((17, 128))

    assert q.shape == (17, 128)
    assert q.dtype == FP8
    assert s.shape == (17, 1)
    assert s.dtype == torch.float32
    assert q.device == s.device


def test_fake_result_is_contiguous_even_for_a_strided_input():
    """The kernel reads at the source row stride but writes a fresh contiguous
    result -- that is the whole point (the consuming GEMM needs it)."""
    (q, _), x = _fake_quant((17, 1024), slice_cols=(512, 640))

    assert x.shape == (17, 128)
    assert x.stride() == (1024, 1), "the test lost the strided-view setup"
    assert q.is_contiguous()


def test_the_body_is_never_traced():
    """One opaque node. If the body were inlined, the `t == 0` branch and the
    f-string asserts would raise on a symbolic token count."""

    def f(x):
        return OP(x, FP8)

    gm = make_fx(f, tracing_mode="fake")(
        torch.empty((17, 128), dtype=torch.bfloat16, device="cuda")
    )
    # Everything but the two `getitem`s that unpack the returned pair.
    targets = [
        n.target
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is not operator.getitem
    ]

    assert targets == [OP.default], f"expected only the op, traced into {targets}"


def test_dynamic_token_count_does_not_specialize():
    """The token dim is dynamic in the KDA prologue (padded cudagraph buckets
    for decode, real length for prefill). The fake must carry the symbol
    through rather than baking a concrete T."""

    def f(x):
        return OP(x, FP8)

    gm = make_fx(f, tracing_mode="symbolic")(
        torch.empty((17, 128), dtype=torch.bfloat16, device="cuda")
    )
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]

    assert OP.default in targets


def test_the_quant_op_is_not_a_splitting_op():
    """It must stay *inside* the compiled piece. Marking it splitting would put
    it back in the eager submodule and undo the whole boundary shrink."""
    assert not getattr(OP, "spliting_op", False)


@pytest.mark.parametrize("cols", [1, 63, 128, 129])
def test_fake_handles_non_power_of_two_feature_widths(cols):
    """head_dim is not required to be a power of two; the kernel pads BLOCK, but
    the returned shape is exactly D."""
    (q, s), _ = _fake_quant((5, cols))

    assert q.shape == (5, cols)
    assert s.shape == (5, 1)
