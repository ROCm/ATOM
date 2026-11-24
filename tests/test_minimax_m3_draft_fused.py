# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 DSpark draft's fused MXFP8 producers (`draft_fused`)
against the unfused pair they replace: GemmaRMSNorm then `quantize_fp8`, and
`swiglu_oai_split` then `quantize_fp8`."""

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
    res = (torch.randn(rows, K, device="cuda") * 3).to(torch.bfloat16) if with_res else None
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
    torch.testing.assert_close(a, b, atol=0.07, rtol=0.07)
    exact = (q.view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    assert exact > 0.995, exact


@pytest.mark.parametrize("rows", [1, 7, 448])
def test_swiglu_oai_mxfp8(rows):
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.swiglu_oai import swiglu_oai_split
    from atom.models.minimax_m3.draft_fused import swiglu_oai_mxfp8

    torch.manual_seed(rows)
    inter = 3072
    gu = (torch.randn(rows, 2 * inter, device="cuda") * 4).to(torch.bfloat16)
    ref_q, ref_s = quantize_fp8(swiglu_oai_split(gu, 1.702, 1.0, 7.0))
    q, s = swiglu_oai_mxfp8(gu, 1.702, 1.0, 7.0)
    assert torch.equal(s.view(torch.uint8), ref_s.view(torch.uint8))
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


@pytest.mark.parametrize("bs", [1, 3, 64])
@pytest.mark.parametrize("shard", [0, 2])
def test_block_embedding(bs, shard):
    """`MiniMaxM3DSpark.embed_block` (one rank, tp 1 so no all-reduce) equals
    the masked lookup of ``[anchor, MASK, ...]`` on that rank's vocab shard."""
    from types import SimpleNamespace

    from atom.models.minimax_m3.dspark import MiniMaxM3DSpark

    torch.manual_seed(bs + shard)
    T, V, D, tp = 7, 4096, 6144, 4
    n = V // tp
    v0 = shard * n
    weight = torch.randn(n, D, device="cuda").to(torch.bfloat16)
    noise = 2 * n + 5  # in shard 2 only
    anchors = torch.randint(0, V, (bs,), device="cuda", dtype=torch.int32)
    anchors[0] = v0 + 3
    model = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=weight, vocab_start_idx=v0, tp_size=1),
        noise_token_id=noise,
    )
    out = MiniMaxM3DSpark.embed_block(model, anchors, T)
    ids = torch.full((bs, T), noise, device="cuda", dtype=torch.int32)
    ids[:, 0] = anchors
    ids = ids.view(-1)
    hit = (ids >= v0) & (ids < v0 + n)
    ref = torch.where(hit[:, None], weight[(ids - v0).clamp(0, n - 1).long()], 0)
    assert torch.equal(out, ref)
