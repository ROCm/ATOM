# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SwiGLU-OAI must retain eager FP32 arithmetic under torch.compile."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="SwiGLU-OAI requires a ROCm GPU",
)


def _reference(x, alpha, beta, limit, out_dtype):
    gate, up = x.float().chunk(2, dim=-1)
    if limit is not None:
        gate = torch.minimum(gate, gate.new_tensor(limit))
        up = torch.clamp(up, min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(out_dtype)


def _assert_reference(actual, x, alpha, beta, limit):
    # Triton's approximate sigmoid may differ from torch at rounding boundaries.
    rtol = 1e-2 if actual.dtype == torch.bfloat16 else 2e-3
    atol = 1e-2 if actual.dtype == torch.bfloat16 else 2e-3
    if actual.dtype == torch.float32:
        rtol, atol = 2e-5, 2e-5
    torch.testing.assert_close(
        actual, _reference(x, alpha, beta, limit, actual.dtype), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("limit", [None, 7.0])
def test_compiled_swiglu_preserves_eager_fp32(dtype, limit):
    # Import only after the ROCm skip gate; CPU CI does not install Triton.
    from atom.model_ops.swiglu_oai import swiglu_oai_split

    # Runtime Python float arguments are FP64 in Inductor's Triton signature
    # on the affected stack, unlike eager Triton. Keep them as runtime scalars,
    # rather than replacing them with constexprs or pre-rounded constants.
    def activate(x, alpha, beta, cap):
        return swiglu_oai_split(x, alpha, beta, cap)

    torch._dynamo.reset()
    compiled = torch.compile(activate, fullgraph=True, dynamic=True)
    for rows in (12, 257, 2517):
        torch.manual_seed(17)
        x = torch.randn(rows, 6144, device="cuda", dtype=dtype)
        eager = activate(x, 1.702, 1.0, limit)
        actual = compiled(x, 1.702, 1.0, limit)
        _assert_reference(actual, x, 1.702, 1.0, limit)
        # This is the regression gate: exact dtype-rounded eager semantics,
        # not just loose closeness to a high-precision mathematical reference.
        assert torch.equal(actual, eager)
        x.neg_()
        changed = compiled(x, 1.702, 0.3, limit)
        assert torch.equal(changed, activate(x, 1.702, 0.3, limit))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "row_strided", "column_strided"])
@pytest.mark.parametrize("out_dtype", [None, torch.float32])
def test_compiled_swiglu_tail_layout_and_output_dtype(dtype, layout, out_dtype):
    from atom.model_ops.swiglu_oai import swiglu_oai_split

    def activate(x, alpha, beta, cap):
        return swiglu_oai_split(x, alpha, beta, cap, out_dtype=out_dtype)

    torch._dynamo.reset()
    compiled = torch.compile(activate, fullgraph=True, dynamic=True)
    torch.manual_seed(31)
    # 136 intermediate elements exercise the masked tile; a leading batch
    # dimension checks reshape restoration instead of only 2-D inputs.
    shape = (2, 7, 272)
    if layout == "row_strided":
        x = torch.randn(2, 7, 288, device="cuda", dtype=dtype)[..., :272]
    elif layout == "column_strided":
        x = torch.randn(2, 7, 544, device="cuda", dtype=dtype)[..., ::2]
    else:
        x = torch.randn(shape, device="cuda", dtype=dtype)
    x *= 8
    actual = compiled(x, 1.702, 0.3, 7.0)
    eager = activate(x, 1.702, 0.3, 7.0)
    assert actual.shape == (2, 7, 136)
    assert actual.dtype == (out_dtype or dtype)
    _assert_reference(actual, x, 1.702, 0.3, 7.0)
    assert torch.equal(actual, eager)


def _hip_per_token_fp8(x):
    # What a per-token FP8 down_proj runs on the bf16 activation it is given.
    from aiter import QuantType, dtypes
    from aiter.ops.quant import get_hip_quant

    return get_hip_quant(QuantType.per_Token)(x, quant_dtype=dtypes.fp8)


def _assert_bitwise_quant(actual, expected):
    (q, s), (q_ref, s_ref) = actual, expected
    assert q.dtype == q_ref.dtype and s.dtype == s_ref.dtype == torch.float32
    assert s.shape == s_ref.shape == (*q.shape[:-1], 1)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s.view(torch.int32), s_ref.view(torch.int32))


def _edge_rows(gate_up):
    # Rows 0 to 7.
    n = gate_up.shape[-1] // 2
    gate, up = gate_up[:, :n], gate_up[:, n:]
    gate[0], up[0] = 0.0, 0.0  # zero row (a pad row): scale 0, 0 * inf
    gate[1] = -60.0  # sigmoid underflows: -0 everywhere
    gate[2], up[2] = 1e4, -1e4  # clamped by the limit
    gate[3, ::7] = float("nan")
    gate[4, ::5] = float("-inf")  # -inf * sigmoid(-inf) = NaN
    up[5, ::3] = float("inf")
    gate[6] = -51.5  # amax ~1e-36: the scale is subnormal, flushed to 0
    gate[7] = -45.0  # tiny activations, normal scale


@pytest.mark.parametrize("rows", [4, 16, 64, 2517])
@pytest.mark.parametrize("splits", [0, 4])
def test_swiglu_quant_is_swiglu_then_hip_per_token_quant(rows, splits):
    from aiter import dtypes

    from atom.model_ops.swiglu_oai import swiglu_oai_split, swiglu_oai_split_quant

    torch.manual_seed(rows + splits)
    # MiniMax-M3 TP4 dense MLP: 2 x 3072 per rank.
    gate_up = torch.randn(rows, 6144, device="cuda", dtype=torch.float32) * 3
    if rows >= 8:
        _edge_rows(gate_up)
    gate_up = gate_up.to(torch.bfloat16)
    partials = None
    if splits:
        partials = torch.randn(splits, rows, 6144, device="cuda") * 2
        partials[:, 0] = 0.0  # still a zero row after the reduction
    gate_up_ref = gate_up.clone()
    expected = _hip_per_token_fp8(
        swiglu_oai_split(gate_up_ref, 1.702, 1.0, 7.0, gate_up_partials=partials)
    )
    actual = swiglu_oai_split_quant(
        gate_up, 1.702, 1.0, 7.0, dtypes.fp8, gate_up_partials=partials
    )
    _assert_bitwise_quant(actual, expected)
    # Reduced partials are written back as without the quant.
    assert torch.equal(gate_up.view(torch.int16), gate_up_ref.view(torch.int16))


def test_swiglu_quant_without_limit_and_compiled():
    from atom.model_ops.swiglu_oai import (
        swiglu_oai_split,
        swiglu_oai_split_quant,
        swiglu_oai_split_reducing_quant,
    )

    torch.manual_seed(5)
    x = torch.randn(33, 6144, device="cuda", dtype=torch.float32) * 3
    _edge_rows(x)
    x[8] *= 1e15  # inf activations without the limit
    x = x.to(torch.bfloat16)
    _assert_bitwise_quant(
        swiglu_oai_split_quant(x, 1.702, 1.0, None),
        _hip_per_token_fp8(swiglu_oai_split(x, 1.702, 1.0, None)),
    )

    def activate(x, alpha, beta, cap):
        return swiglu_oai_split_quant(x, alpha, beta, cap)

    def activate_reducing(x, alpha, beta, cap):
        return swiglu_oai_split_reducing_quant(x, alpha, beta, cap)

    torch._dynamo.reset()
    compiled = torch.compile(activate, fullgraph=True, dynamic=True)
    compiled_reducing = torch.compile(activate_reducing, fullgraph=True, dynamic=True)
    for rows in (12, 257):
        torch.manual_seed(rows)
        x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16) * 3
        expected = _hip_per_token_fp8(swiglu_oai_split(x, 1.702, 1.0, 7.0))
        _assert_bitwise_quant(compiled(x, 1.702, 1.0, 7.0), expected)
        _assert_bitwise_quant(compiled_reducing(x, 1.702, 1.0, 7.0), expected)
