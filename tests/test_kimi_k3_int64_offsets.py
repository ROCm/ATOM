# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's per-token kernels address rows past 2**31 elements.

At TP1 (each stage of a PP8 x TP1 prefill) a row of a K3 activation is wide: the
layer-1 dense FFN's gate_up output holds 2 x 33792 elements, KDA's fused
in_proj ~49K, AttnRes's block_residual blocks x 7168. A prefill batch of a few
tens of thousands of tokens then puts `row * stride` past 2**31, which these
kernels formed in 32 bits: a 32K-token batch faulted in _situ_and_mul_kernel.

Each test fills only the last rows of a tensor whose last rows start past 2**31
elements, and compares them with the same op run on those rows alone.
"""

import pytest
import torch
from conftest import gpu_stack_import

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a GPU"
)

TAIL = 4


def _rows_past(width: int) -> int:
    """Rows of `width` elements whose last TAIL rows start past 2**31 elements."""
    return 2**31 // width + TAIL + 1


def _tail(rows: int, *shape: int) -> torch.Tensor:
    t = torch.zeros(rows, *shape, dtype=torch.bfloat16, device="cuda")
    t[-TAIL:] = torch.randn(TAIL, *shape, dtype=torch.bfloat16, device="cuda")
    return t


def test_situ_and_mul():
    act = gpu_stack_import("atom.model_ops.kimi_k3.activations")
    d = 33792
    x = _tail(_rows_past(2 * d), 2 * d)
    assert (x.shape[0] - TAIL) * x.stride(0) >= 2**31

    y = act.situ_and_mul(x, beta=25.0, linear_beta=None)

    want = act.situ_and_mul(x[-TAIL:].clone(), beta=25.0, linear_beta=None)
    assert torch.equal(y[-TAIL:], want)


def test_strided_per_token_quant():
    quant = gpu_stack_import("atom.model_ops.kimi_k3.quant")
    from aiter import dtypes

    n, d = 49408, 128  # a head-wide column slice of a fused in_proj output
    fused = _tail(_rows_past(n), n)
    x = fused[:, :d]
    assert (x.shape[0] - TAIL) * x.stride(0) >= 2**31

    out, scale = quant.strided_per_token_quant(x, dtypes.fp8)

    want, want_scale = quant.strided_per_token_quant(x[-TAIL:].contiguous(), dtypes.fp8)
    assert torch.equal(out[-TAIL:].view(torch.uint8), want.view(torch.uint8))
    assert torch.equal(scale[-TAIL:], want_scale)


def test_apply_attn_res():
    attn_res = gpu_stack_import("atom.model_ops.kimi_k3.attention_residual")
    blocks, h = 8, 7168
    t = _rows_past(blocks * h)
    block_residual = _tail(t, blocks, h)
    prefix_sum = _tail(t, h)
    score_weight = torch.randn(h, dtype=torch.bfloat16, device="cuda")
    assert (t - TAIL) * block_residual.stride(0) >= 2**31

    y, _ = attn_res.apply_attn_res(prefix_sum, block_residual, score_weight, 1e-6)

    want, _ = attn_res.apply_attn_res(
        prefix_sum[-TAIL:].clone(), block_residual[-TAIL:].clone(), score_weight, 1e-6
    )
    torch.testing.assert_close(y[-TAIL:], want, rtol=1e-2, atol=1e-2)
