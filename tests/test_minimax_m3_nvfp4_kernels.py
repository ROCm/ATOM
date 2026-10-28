# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 FP4 index cache and NVFP4 sparse KV kernels, against torch.

gfx950 only: the e2m1 decode is a gfx950 instruction and the attention / fused
top-k kernels are gfx950 Gluon. Elsewhere every test skips.

- the store kernels against a torch quantizer (exact), K/V against the index path;
- the per-row (request, causal length) maps;
- the FP4 scorers against the BF16 scorers fed the dequantized cache (e2m1 x E4M3
  is exact in bf16, so the selected blocks must be identical);
- the NVFP4 attention against softmax attention over the dequantized cache;
- the fused top-k against the FP4 scorer + top-k.
"""

import pytest
import torch

pytest.importorskip("triton", reason="the kernels are triton / gluon")


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", "").startswith("gfx950")


pytestmark = pytest.mark.skipif(not _gfx950(), reason="needs a gfx950 GPU")

DEV, D, PAGE = "cuda", 128, 128
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 planes ``[..., D/2]`` / ``[..., D/16]`` -> fp32 ``[..., D]``."""
    lut = torch.tensor(E2M1 + tuple(-x for x in E2M1), device=payload.device)
    p = payload.to(torch.int64)
    vals = torch.stack([lut[p & 15], lut[p >> 4]], dim=-1).flatten(-2)
    s = scale.view(torch.float8_e4m3fn).to(torch.float32)
    return (vals.unflatten(-1, (-1, 16)) * s[..., None]).flatten(-2)


def quantize_ref(x: torch.Tensor) -> torch.Tensor:
    """What the store must produce, dequantized: per 16 elements
    scale = e4m3(amax / 6), codes rounded half-to-even, saturating at 6."""
    x = x.float().unflatten(-1, (-1, 16))
    s = (x.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn).float()
    y = (x * torch.where(s > 0, 1.0 / s, 0.0)[..., None]).clamp(-6, 6)
    a = y.abs()
    mag = sum((a > t).long() for t in (0.25, 1.25, 2.5, 5.0)) + sum(
        (a >= t).long() for t in (0.75, 1.75, 3.5)
    )
    grid = torch.tensor(E2M1, device=x.device)
    return (grid[mag] * y.sign() * s[..., None]).flatten(-2)


def fp4_cache(blocks: int, seed: int = 0):
    from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store

    g = torch.Generator(device=DEV).manual_seed(seed)
    keys = torch.randn(blocks * PAGE, D, device=DEV, dtype=torch.bfloat16, generator=g)
    payload = torch.zeros(blocks, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    scale = torch.zeros(blocks, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    index_fp4_store(keys, torch.arange(blocks * PAGE, device=DEV), payload, scale)
    return payload, scale


def test_index_store_matches_torch_and_skips_padding():
    from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store

    torch.manual_seed(0)
    blocks, tokens = 64, 3000
    keys = torch.randn(tokens, D, device=DEV, dtype=torch.bfloat16) * 3
    slots = torch.randperm(blocks * PAGE, device=DEV)[:tokens].to(torch.int64)
    slots[5] = -1  # a graph pad row: must not be written
    payload = torch.zeros(blocks, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    scale = torch.zeros(blocks, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    index_fp4_store(keys, slots, payload, scale)
    got = dequant(payload, scale).view(blocks * PAGE, D)
    ok = slots >= 0
    assert torch.equal(got[slots[ok]], quantize_ref(keys)[ok])
    written = torch.zeros(blocks * PAGE, dtype=torch.bool, device=DEV)
    written[slots[ok]] = True
    assert int(payload.view(blocks * PAGE, -1)[~written].sum()) == 0


@pytest.mark.parametrize("heads", [1, 2])
def test_kv_store_equals_per_head_index_store(heads):
    from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store, kv_nvfp4_store

    torch.manual_seed(1)
    blocks, tokens = 16, 700
    stage = torch.randn(6, 2, PAGE, heads, D, device=DEV, dtype=torch.bfloat16)
    slots = torch.randperm(blocks * PAGE, device=DEV)[:tokens].to(torch.int64)
    slots[3] = -1
    kp = torch.zeros(blocks, heads, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    ks = torch.zeros(blocks, heads, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    vp, vs = torch.zeros_like(kp), torch.zeros_like(ks)
    kv_nvfp4_store(stage, slots, kp, ks, vp, vs)
    for which, (p, s) in enumerate(((kp, ks), (vp, vs))):
        src = stage[:, which].reshape(-1, heads, D)[:tokens]
        for h in range(heads):
            rp = torch.zeros(blocks, PAGE, D // 2, dtype=torch.uint8, device=DEV)
            rs = torch.zeros(blocks, PAGE, D // 16, dtype=torch.uint8, device=DEV)
            index_fp4_store(src[:, h].contiguous(), slots, rp, rs)
            assert torch.equal(p[:, h], rp) and torch.equal(s[:, h], rs)


@pytest.mark.parametrize("bs, q", [(1, 8), (6, 8), (64, 8), (5, 4)])
def test_decode_rows(bs, q):
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import decode_rows

    seq = torch.randint(100, 300000, (bs,), device=DEV, dtype=torch.int32)
    req, seq_len = decode_rows(seq, bs * q, q)
    tok = torch.arange(bs * q, device=DEV)
    want_req = tok // q
    assert torch.equal(req.long(), want_req)
    assert torch.equal(seq_len.long(), (seq[want_req] - q + tok % q + 1).long())


def test_decode_rows_padding():
    """3 requests of 8 rows padded to 32: the live rows keep their mapping (not
    32 // 3 = 10 rows a request), the pad rows read request 0 at length 0."""
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import decode_rows

    seq = torch.tensor([500, 900, 1300], device=DEV, dtype=torch.int32)
    req, seq_len = decode_rows(seq, 32, 8)
    tok = torch.arange(24, device=DEV)
    assert torch.equal(req[:24].long(), tok // 8)
    assert torch.equal(seq_len[:24].long(), (seq[tok // 8] - 8 + tok % 8 + 1).long())
    assert req[24:].tolist() == [0] * 8 and seq_len[24:].tolist() == [0] * 8


def test_prefill_rows():
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import prefill_rows

    qlens, prefix = [3, 1, 5], [10, 0, 7]
    cu = torch.tensor([0, 3, 4, 9], device=DEV, dtype=torch.int32)
    req, seq_len = prefill_rows(cu, torch.tensor(prefix, device=DEV), 9)
    want_req = [b for b, n in enumerate(qlens) for _ in range(n)]
    want_len = [prefix[b] + i + 1 for b, n in enumerate(qlens) for i in range(n)]
    assert req.tolist() == want_req and seq_len.tolist() == want_len


def test_prefill_rows_padding():
    """A prefill padded to a token-graph bucket: rows past cu_seqlens_q[-1]
    read request 0 at length 0 instead of indexing past the requests."""
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import prefill_rows

    cu = torch.tensor([0, 3, 4], device=DEV, dtype=torch.int32)
    req, seq_len = prefill_rows(cu, torch.tensor([10, 0], device=DEV), 8)
    assert req.tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert seq_len.tolist() == [11, 12, 13, 1, 0, 0, 0, 0]


def _same_sets(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.sort(-1).values, b.sort(-1).values)


def test_fp4_scorers_select_like_bf16():
    from atom.model_ops.minimax_m3.index_topk import (
        minimax_m3_index_topk,
        minimax_m3_index_topk_decode,
    )

    torch.manual_seed(2)
    blocks = 64
    payload, scale = fp4_cache(blocks)
    bf16 = dequant(payload, scale).to(torch.bfloat16)
    common = {"topk": 16, "init_blocks": 0, "local_blocks": 1, "num_kv_heads": 1}

    # decode: 3 requests x 8 verify rows
    seq = torch.tensor([5000, 1300, 7900], device=DEV, dtype=torch.int32)
    bt = (torch.arange(3 * 64, device=DEV, dtype=torch.int32) % blocks).view(3, 64)
    q = torch.randn(24, 1, D, device=DEV, dtype=torch.bfloat16)
    kw = dict(max_seq_len=int(seq.max()), sm_scale=D**-0.5, max_query_len=8, **common)
    ref = minimax_m3_index_topk_decode(q, bf16, bt, seq, **kw)[0]
    got = minimax_m3_index_topk_decode(q, payload, bt, seq, index_kv_scale=scale, **kw)[
        0
    ]
    assert _same_sets(ref, got)

    # prefill: two chunks behind a prefix
    qlens = torch.tensor([700, 300], device=DEV, dtype=torch.int32)
    seq2 = torch.tensor([4000, 2300], device=DEV, dtype=torch.int32)
    cu = torch.zeros(3, device=DEV, dtype=torch.int32)
    cu[1:] = torch.cumsum(qlens, 0)
    bt2 = torch.arange(64, device=DEV, dtype=torch.int32).view(2, 32)
    q2 = torch.randn(1000, 1, D, device=DEV, dtype=torch.bfloat16)
    kw2 = dict(max_query_len=700, max_seq_len=4000, sm_scale=D**-0.5, **common)
    ref = minimax_m3_index_topk(q2, bf16, bt2, cu, seq2, seq2 - qlens, **kw2)[0]
    got = minimax_m3_index_topk(
        q2, payload, bt2, cu, seq2, seq2 - qlens, index_kv_scale=scale, **kw2
    )[0]
    assert _same_sets(ref, got)


def test_nvfp4_attention_matches_dequantized_reference():
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import (
        check_nvfp4_sparse_attention,
        decode_rows,
        nvfp4_sparse_attention,
    )

    torch.manual_seed(3)
    blocks, topk, nreq, q_rows, maxb = 512, 16, 6, 8, 200
    kp, ks = fp4_cache(blocks, seed=4)
    vp, vs = fp4_cache(blocks, seed=5)
    kd, vd = dequant(kp, ks).view(-1, D), dequant(vp, vs).view(-1, D)
    shape4 = lambda t: t.view(blocks, 1, PAGE, -1)
    seq = torch.randint(
        topk * PAGE, maxb * PAGE, (nreq,), device=DEV, dtype=torch.int32
    )
    bt = torch.stack([torch.randperm(blocks, device=DEV)[:maxb] for _ in range(nreq)])
    bt = bt.to(torch.int32)
    rows = nreq * q_rows + q_rows  # one padded request: length 0, no blocks
    req, row_seq = decode_rows(seq, rows, q_rows)
    topk_idx = torch.full((1, rows, topk), -1, device=DEV, dtype=torch.int32)
    for r in range(nreq * q_rows):
        n = int((row_seq[r] + PAGE - 1) // PAGE)
        sel = torch.randperm(n - 1, device=DEV)[: topk - 1]
        sel = torch.cat([sel, torch.tensor([n - 1], device=DEV)])  # the local block
        topk_idx[0, r, : sel.numel()] = sel.to(torch.int32)
    # A pad row's top-k may be left uninitialized (FP8 index, unfused selector):
    # a dirty list must still select nothing at length 0.
    topk_idx[0, nreq * q_rows :, :3] = torch.tensor([0, -1, 0], dtype=torch.int32)
    q = torch.randn(rows, 16, D, device=DEV, dtype=torch.bfloat16)
    args = (q, shape4(kp), shape4(ks), shape4(vp), shape4(vs), topk_idx)
    check_nvfp4_sparse_attention(*args)
    out = nvfp4_sparse_attention(*args, bt, req, row_seq, D**-0.5)
    # Few rows split the top-k into chunks; the empty pad rows merge to 0, not NaN.
    assert torch.equal(out[nreq * q_rows :], torch.zeros_like(out[nreq * q_rows :]))
    for r in range(nreq * q_rows):
        blks = topk_idx[0, r][topk_idx[0, r] >= 0].long()
        toks = bt[req[r]].long()[blks][:, None] * PAGE + torch.arange(PAGE, device=DEV)
        logical = blks[:, None] * PAGE + torch.arange(PAGE, device=DEV)
        toks = toks.flatten()[logical.flatten() < row_seq[r]]
        p = torch.softmax((q[r].float() @ kd[toks].T) * D**-0.5, -1)
        ref = p @ vd[toks]
        err = (out[r].float() - ref).abs().max() / ref.abs().max()
        assert err < 0.02, (r, float(err))


def test_attention_check_refuses_eight_q_heads():
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import (
        check_nvfp4_sparse_attention,
    )

    kp, ks = fp4_cache(4)
    shape4 = lambda t: t.view(4, 1, PAGE, -1)
    q = torch.randn(8, 8, D, device=DEV, dtype=torch.bfloat16)  # TP8: 8 q heads
    idx = torch.zeros(1, 8, 16, device=DEV, dtype=torch.int32)
    with pytest.raises(ValueError, match="16 q heads per kv head"):
        check_nvfp4_sparse_attention(
            q, shape4(kp), shape4(ks), shape4(kp), shape4(ks), idx
        )


@pytest.mark.parametrize("bs, ctx", [(3, 20000), (16, 186000)])
def test_fused_verify_topk_matches_scorer(bs, ctx):
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
    from atom.model_ops.minimax_m3.nvfp4_kernels.nvfp4 import Nvfp4Cache
    from atom.model_ops.minimax_m3.nvfp4_kernels.verify_topk_gfx950 import (
        can_fuse_verify_topk,
        fused_verify_topk_idx,
    )
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import decode_rows

    torch.manual_seed(6)
    blocks, q_rows, topk = 2048, 8, 16
    payload, scale = fp4_cache(blocks, seed=7)
    cache = Nvfp4Cache(payload=payload, scales=scale)
    maxb = (ctx + PAGE - 1) // PAGE
    # Distinct physical blocks per request: a repeat would tie two blocks' scores
    # exactly, and the two top-k paths may break that tie differently.
    bt = torch.stack([torch.randperm(blocks, device=DEV)[:maxb] for _ in range(bs)])
    bt = bt.to(torch.int32)
    seq = torch.randint(ctx // 2, ctx + 1, (bs,), device=DEV, dtype=torch.int32)
    q = torch.randn(bs * q_rows, 1, D, device=DEV, dtype=torch.bfloat16)
    assert can_fuse_verify_topk(q, cache, q_rows, topk, 1, PAGE, PAGE)
    ref = minimax_m3_index_topk_decode(
        q, payload, bt, seq, max_seq_len=ctx, topk=topk, init_blocks=0,
        local_blocks=1, num_kv_heads=1, sm_scale=D**-0.5, max_query_len=q_rows,
        index_kv_scale=scale,
    )[0]  # fmt: skip
    req, row_seq = decode_rows(seq, bs * q_rows, q_rows)
    got = fused_verify_topk_idx(
        q, cache, bt, row_seq, ctx, req, q_rows, PAGE, topk, 0, 1, D**-0.5, 1.0
    )
    assert _same_sets(ref, got[0])  # [heads, rows, topk] -> the one index head
    # The fused kernel's bounds: below 3 requests the separate kernels win, and
    # a padded row count that is not whole draft groups goes unfused.
    assert not can_fuse_verify_topk(q[: 2 * q_rows], cache, q_rows, topk, 1, PAGE, PAGE)
    padded = torch.cat([q, q[:2]])  # 2 rows past the last whole group
    assert not can_fuse_verify_topk(padded, cache, q_rows, topk, 1, PAGE, PAGE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quant_stage_follows_the_model_dtype(dtype):
    """The fused writer stores the staged rows in qkv's dtype under "auto", so
    the staging buffer must be the model's dtype (fp16 models too)."""
    from types import SimpleNamespace

    from atom.model_ops.attentions.aiter_attention import AiterAttentionMetadataBuilder

    builder = AiterAttentionMetadataBuilder.__new__(AiterAttentionMetadataBuilder)
    builder.block_size = PAGE
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(
            max_num_batched_tokens=300,
            max_num_seqs=64,
            speculative_config=SimpleNamespace(num_speculative_tokens=7),
            torch_dtype=dtype,
        )
    )
    stage, slots = builder._quant_stage("index", (PAGE, D))
    # 64 seqs x 8 verify rows = 512 > 300 tokens: the decode graph sizes it.
    assert stage.dtype == dtype and stage.shape == (4, PAGE, D)
    assert torch.equal(slots, torch.arange(512, device=slots.device))
