# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 DSpark draft's fused MXFP8 norm (`draft_fused`)
against the unfused pair it replaces: GemmaRMSNorm then `quantize_fp8`."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _dequant(q, s):
    return q.float() * s.float().repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("rows", [1, 7, 56, 448])
@pytest.mark.parametrize("with_res", [False, True])
def test_gemma_rmsnorm_mxfp8(rows, with_res):
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.layernorm import GemmaRMSNorm
    from atom.models.minimax_m3.draft_fused import gemma_rmsnorm_mxfp8

    torch.manual_seed(rows)
    K = 6144
    norm = GemmaRMSNorm(K, eps=1e-6).cuda()
    norm.weight.data = (torch.randn(K, device="cuda") * 0.3).to(torch.bfloat16)
    x = (torch.randn(rows, K, device="cuda") * 3).to(torch.bfloat16)
    res = (
        (torch.randn(rows, K, device="cuda") * 3).to(torch.bfloat16)
        if with_res
        else None
    )
    if with_res:
        ref_y, ref_res = norm(x.clone(), res.clone())
    else:
        ref_y, ref_res = norm(x.clone()), None
    ref_q, ref_s = quantize_fp8(ref_y)
    q, s, res_out = gemma_rmsnorm_mxfp8(x, norm.weight, 1e-6, res)
    if with_res:
        torch.testing.assert_close(res_out, ref_res, atol=0, rtol=0)
    # scales: identical except where the norm's last bf16 bit moved a group amax
    scale_match = (s.view(torch.uint8) == ref_s.view(torch.uint8)).float().mean().item()
    assert scale_match > 0.999, scale_match
    a, b = _dequant(q, s), _dequant(ref_q, ref_s)
    # a flipped E4M3 code is one step (up to 2**-3 of the value) away
    torch.testing.assert_close(a, b, atol=0.07, rtol=0.15)
    exact = (q.view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    assert exact > 0.995, exact
