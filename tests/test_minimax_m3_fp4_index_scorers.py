# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 FP4 index scorers on gfx1250: the scaled-WMMA read.

On gfx1250 the FP4 index scorers -- the tiled decode scorer, the indexer-CP
context scorer and the prefill scorer -- hand the packed rows to the scaled WMMA
(``nvfp4_scaled_dot``: e2m1 codes with their E4M3 per-16 scales, the query in E4M3,
two terms in decode and CP). Here:

- the scaled-WMMA read: one-hot queries return the dequantized key elements
  exactly, random queries the float64 product of the E4M3 / two-term query within
  the WMMA's summation error (SCALED_DOT_REL);
- the decode scorer's block scores against a float64 reference on the dequantized
  cache (two-term query);
- the full decode selection against the BF16 scorer fed the dequantized cache
  (near ties at the top-k edge may swap).

CP-vs-TP exactness is in ``tests/model_ops/test_indexer_cp_parity.py``.
"""

import pytest
import torch

triton = pytest.importorskip("triton", reason="the kernels are triton")
import triton.language as tl

nvfp4 = pytest.importorskip("atom.model_ops.minimax_m3.nvfp4_kernels.nvfp4")


def _gfx() -> str:
    if not torch.cuda.is_available():
        return ""
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0]


pytestmark = pytest.mark.skipif(_gfx() != "gfx1250", reason="needs a gfx1250 GPU")

DEV, D, PAGE = "cuda", 128, 128
LOG2E = 1.4426950409
# The scaled WMMA (v_wmma_scale16_f32_16x16x128_f8f6f4) does not sum its 128
# products exactly in fp32: on MI455X a random dot lands ~2^-12 of the sum of the
# products' magnitudes from float64. Its results are bounded by this fraction of
# that sum.
SCALED_DOT_REL = 2.0**-9
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant(payload, scale):
    lut = torch.tensor(E2M1 + tuple(-x for x in E2M1), device=payload.device)
    p = payload.to(torch.int64)
    vals = torch.stack([lut[p & 15], lut[p >> 4]], dim=-1).flatten(-2)
    s = scale.view(torch.float8_e4m3fn).to(torch.float32)
    return (vals.unflatten(-1, (-1, 16)) * s[..., None]).flatten(-2)


def fp4_cache(pages, seed):
    from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store

    g = torch.Generator(device=DEV).manual_seed(seed)
    keys = (torch.randn(pages * PAGE, D, device=DEV, generator=g) * 3).to(
        torch.bfloat16
    )
    payload = torch.zeros(pages, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    scale = torch.zeros(pages, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    index_fp4_store(keys, torch.arange(pages * PAGE, device=DEV), payload, scale)
    return payload, scale


def _decode_scores(kernel, q, payload, scale, bt, seq, max_q, chunks, warps):
    batch = seq.numel()
    max_block = triton.cdiv(int(seq.max()), PAGE)
    out = torch.full((1, batch * max_q, max_block), float("nan"), device=DEV)
    kernel[(chunks, batch)](
        q, payload, scale, out, bt, seq, D, D**-0.5, triton.cdiv(max_block, chunks),
        *q.stride(), *payload.stride(), scale.stride(0), scale.stride(1),
        *out.stride(), bt.stride(0), NUM_IDX_HEADS=1, MAX_Q=max_q,
        BLOCK_SIZE_K=PAGE, IS_FP4=True, num_warps=warps, num_stages=1,
    )  # fmt: skip
    return out


def split_query(q):
    """``nvfp4_split_query`` in torch: hi = e4m3(q), lo = e4m3((q - hi) * 16) / 16."""
    q32 = q.float().clamp(-448, 448)
    hi = q32.to(torch.float8_e4m3fn).float()
    lo = ((q32 - hi) * 16).to(torch.float8_e4m3fn).float() / 16
    return hi.double() + lo.double()


@pytest.mark.parametrize("max_q", [1, 4, 8])
@pytest.mark.parametrize("warps", [2, 4])
def test_decode_scorer_matches_reference(max_q, warps):
    from atom.model_ops.minimax_m3 import index_topk as m

    torch.manual_seed(max_q)
    pages, batch = 64, 4
    payload, scale = fp4_cache(pages, 2)
    seq = torch.tensor([20 * PAGE + 9, max_q, 7 * PAGE, 32 * PAGE], dtype=torch.int32)
    bt = torch.stack([torch.randperm(pages)[:32] for _ in range(batch)]).to(torch.int32)
    q = torch.randn(batch * max_q, 1, D, dtype=torch.bfloat16)
    seq, bt, q = seq.to(DEV), bt.to(DEV), q.to(DEV)
    # The production gfx1250 read: packed rows to the scaled WMMA, two-term query.
    got = _decode_scores(m._decode_index_score_tiled_kernel, q, payload, scale, bt,
                         seq, max_q, 3, warps)  # fmt: skip
    k = dequant(payload, scale).double()
    q2 = split_query(q[:, 0])
    for bi in range(batch):
        for t in range(max_q):
            r = bi * max_q + t
            cut = int(seq[bi]) - max_q + t + 1
            for blk in range(triton.cdiv(int(seq[bi]), PAGE)):
                masked = blk * PAGE + torch.arange(PAGE, device=DEV) >= cut
                kb = k[int(bt[bi, blk])]
                s = (kb @ q2[r]) * D**-0.5 * LOG2E
                s[masked] = float("-inf")
                want = s.max().float()
                g = got[0, r, blk]
                if torch.isinf(want):
                    assert g == want
                else:
                    mag = ((kb.abs() @ q2[r].abs()) * D**-0.5 * LOG2E).max()
                    assert abs(g - want) <= SCALED_DOT_REL * mag + 1e-6, (r, blk)
            # Blocks past the request's last are never written.
            past = triton.cdiv(int(seq[bi]), PAGE)
            assert torch.isnan(got[0, bi * max_q : (bi + 1) * max_q, past:]).all()


@pytest.mark.parametrize(
    "max_q, batch, ctx", [(1, 64, 65536), (8, 4, 300000), (4, 16, 8192)]
)
def test_decode_selection_matches_bf16_on_the_dequantized_cache(max_q, batch, ctx):
    """The production entry point (gfx1250 FP4: the scaled-WMMA read, the
    query in two E4M3 terms ~ bf16) against the BF16 scorer fed the dequantized keys
    (e2m1 x E4M3 is exact in bf16)."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    torch.manual_seed(5)
    blocks = triton.cdiv(ctx, PAGE)
    pages = min(batch * blocks, 4096)
    payload, scale = fp4_cache(pages, 3)
    bf16 = dequant(payload, scale).to(torch.bfloat16)
    seq = torch.randint(ctx // 2, ctx + 1, (batch,), device=DEV, dtype=torch.int32)
    bt = torch.randint(0, pages, (batch, blocks), device=DEV, dtype=torch.int32)
    q = torch.randn(batch * max_q, 1, D, device=DEV, dtype=torch.bfloat16)
    kw = {
        "max_seq_len": ctx,
        "topk": 16,
        "init_blocks": 0,
        "local_blocks": 1,
        "num_kv_heads": 1,
        "sm_scale": D**-0.5,
        "max_query_len": max_q,
    }
    ref = minimax_m3_index_topk_decode(q, bf16, bt, seq, **kw)
    got = minimax_m3_index_topk_decode(q, payload, bt, seq, index_kv_scale=scale, **kw)
    # The scaled WMMA's inexact sum (SCALED_DOT_REL) may swap a near tie at the
    # edge of the top-16: nearly every row is the same set, and no row loses more
    # than one block.
    a, b = ref.sort(-1).values, got.sort(-1).values
    same = (a == b).all(-1).float().mean().item()
    worst = max(
        16 - len(set(x.tolist()) & set(y.tolist()))
        for x, y in zip(a.flatten(0, 1).cpu(), b.flatten(0, 1).cpu())
    )
    assert same >= 0.99 and worst <= 1, (same, worst)


@triton.jit
def _scaled_dot_kernel(q_ptr, p_ptr, s_ptr, o_ptr, TWO: tl.constexpr):
    n = tl.arange(0, 16)
    d = tl.arange(0, 128)
    pos = tl.arange(0, 128)
    q = tl.load(q_ptr + n[:, None] * 128 + d[None, :])
    payload, scale = nvfp4.nvfp4_load_packed(p_ptr, s_ptr, 0, 0, pos, 128)
    if TWO:
        hi, lo, lo_scale = nvfp4.nvfp4_split_query(q)
        acc = nvfp4.nvfp4_scaled_dot(hi, lo, lo_scale, payload, scale)
    else:
        acc = nvfp4.nvfp4_scaled_dot(
            nvfp4.nvfp4_query_e4m3(q), None, None, payload, scale
        )
    tl.store(o_ptr + n[:, None] * 128 + pos[None, :], acc)


def test_scaled_dot_reads_the_dequantized_rows():
    """v_wmma_scale16_f32_16x16x128_f8f6f4 through tl.dot_scaled with E4M3 per-16
    scales: one-hot queries pick single dequantized key elements exactly (nibble
    order, scale groups, subnormal to large scale bytes); random queries, E4M3 and
    two-term (its E8M0 2^-4 scale on the query operand), within SCALED_DOT_REL of
    float64."""
    g = torch.Generator().manual_seed(2)
    payload = torch.randint(0, 256, (128, 64), generator=g, dtype=torch.uint8)
    classes = torch.tensor([0x01, 0x07, 0x20, 0x38, 0x3C, 0x55, 0x70, 0x7E])
    scale = classes[torch.randint(0, 8, (128, 8), generator=g)].to(torch.uint8)
    k = dequant(payload, scale).double()
    payload, scale = payload.to(DEV), scale.to(DEV)
    out = torch.empty(16, 128, device=DEV)
    for base in range(0, 128, 16):
        q = torch.zeros(16, 128, dtype=torch.bfloat16)
        q[torch.arange(16), base + torch.arange(16)] = 1.0
        _scaled_dot_kernel[(1,)](q.to(DEV), payload, scale, out, TWO=False)
        assert torch.equal(out.cpu(), k.cpu()[:, base : base + 16].T.float()), base
    for two in (False, True):
        q = (torch.randn(16, 128, generator=g) * 2).to(torch.bfloat16)
        _scaled_dot_kernel[(1,)](q.to(DEV), payload, scale, out, TWO=two)
        qq = split_query(q) if two else q.float().to(torch.float8_e4m3fn).double()
        want = (qq @ k.cpu().T).float()
        mag = (qq.abs() @ k.cpu().abs().T).float()
        err = ((out.cpu() - want).abs() / mag.clamp(min=1e-30)).max()
        assert err <= SCALED_DOT_REL, (two, float(err))
