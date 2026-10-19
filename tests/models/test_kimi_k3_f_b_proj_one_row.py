# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Kimi-K3 ``f_b_proj`` input at one token.

``KimiDeltaAttention._forward_impl`` slices ``f_a`` out of the fused in_proj
output ``[q | k | v | g | b | f_a]``. When f_b_proj is unquantized (mono keeps it
out of online quant) that slice goes to a bf16 GEMM. ``.contiguous()`` copies it
at T >= 2 but returns it as is at T == 1, because torch ignores the stride of a
size-1 dim, so a one-token prefill chunk handed the GEMM the fused row stride
(6284 elements at TP8) and a 12312 B offset. AITER's gfx950 table routes bf16
(M=1, N=1536, K=128) to FlyDSL, whose DMA path needs a 16 B aligned pointer and
leading stride, and raised. The model now copies into a fresh buffer;
``fresh_copy`` below is that expression.
"""

import pytest
import torch
import torch.nn.functional as F

# TP8 Kimi-K3 KDA layer: local_proj_size, num_local_heads, head_dim.
LP, NLH, HD = 12 * 128, 12, 128
FUSED = 4 * LP + NLH + HD
F_A = 4 * LP + NLH
BF16 = torch.bfloat16


def f_a_view(tokens, device="cpu"):
    fused_in = torch.randn(tokens, FUSED, device=device, dtype=BF16)
    return fused_in[..., F_A : F_A + HD]


def fresh_copy(view):
    return torch.empty(view.shape, dtype=view.dtype, device=view.device).copy_(view)


def test_contiguous_keeps_the_fused_row_stride_at_one_token():
    assert f_a_view(1).contiguous().stride() == (FUSED, 1)
    assert f_a_view(2).contiguous().stride() == (HD, 1)


@pytest.mark.parametrize("tokens", [1, 2, 8])
def test_fresh_copy_has_a_unit_row_stride(tokens):
    assert fresh_copy(f_a_view(tokens)).stride() == (HD, 1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required",
)
@pytest.mark.parametrize("tokens", [1, 2, 8])
def test_bf16_f_b_proj_gemm(tokens):
    from aiter.tuned_gemm import tgemm

    torch.manual_seed(0)
    x = fresh_copy(f_a_view(tokens, "cuda"))
    w = torch.randn(LP, HD, device="cuda", dtype=BF16) / HD**0.5
    out = tgemm.mm(x, w, None, otype=BF16)
    ref = F.linear(x.float(), w.float()).to(BF16)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
