# SPDX-License-Identifier: MIT
"""MiniMax-M3 SP prefill glue kernels: the all-to-all unpack layout copy and the
MoE chain's final reduce with the routed scale and the shared expert's output."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("world,rows,d", [(4, 64, 2048), (4, 96, 2048), (2, 33, 4096)])
def test_a2a_unpack_matches_transpose(world, rows, d):
    sp_layout = pytest.importorskip("atom.models.minimax_m3.sp_layout")
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randint(
        0, 255, (world * rows, d), generator=g, device="cuda", dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    s = torch.randint(
        100, 140, (world * rows, d // 32), generator=g, device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    want_q = q.view(world, rows, d).transpose(0, 1).reshape(rows, world * d)
    want_s = s.view(world, rows, d // 32).transpose(0, 1).reshape(rows, -1)
    got_q, got_s = sp_layout.a2a_unpack(q, s, world, rows)
    assert torch.equal(got_q.view(torch.uint8), want_q.view(torch.uint8))
    assert torch.equal(got_s.view(torch.uint8), want_s.view(torch.uint8))


@pytest.mark.parametrize(
    "tokens,add_rows,row0",
    [(256, 0, 0), (256, 256, 0), (256, 64, 0), (256, 64, 96), (256, 64, 192)],
)
@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("fused", [False, True])
def test_reduce_scale_add(tokens, add_rows, row0, scale, fused):
    if not fused and (add_rows or scale != 1.0):
        pytest.skip("the plain reduce has no scale or add")
    pytest.importorskip("flydsl")
    from atom.model_ops.minimax_m3.flydsl_moe.flydsl_utils import run_compiled
    from atom.model_ops.minimax_m3.flydsl_moe.moe_flydsl_common.launch import (
        _get_reduce_bf16,
    )
    from atom.model_ops.minimax_m3.flydsl_moe.moe_mxfp4.prefill import _f32_bits

    h, k = 6144, 4
    g = torch.Generator(device="cuda").manual_seed(1)
    partial = (torch.randn(tokens * k, h, generator=g, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    add = torch.randn(max(add_rows, 1), h, generator=g, device="cuda").to(
        torch.bfloat16
    )
    out = torch.empty(tokens, h, dtype=torch.bfloat16, device="cuda")
    run_compiled(
        _get_reduce_bf16(h, k, fused),
        partial.view(-1),
        out.view(-1),
        add.view(-1),
        tokens,
        row0,
        add_rows,
        _f32_bits(scale),
        torch.cuda.current_stream(),
    )
    ref = partial.view(tokens, k, h).float().sum(1) * scale
    ref[row0 : row0 + add_rows] += add[:add_rows].float()
    # one bf16 rounding of the fp32 result
    assert torch.equal(out, ref.to(torch.bfloat16))
