# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SwiGLU-OAI must retain eager FP32 arithmetic under torch.compile."""

import pytest
import torch

from atom.model_ops.swiglu_oai import swiglu_oai_split

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
