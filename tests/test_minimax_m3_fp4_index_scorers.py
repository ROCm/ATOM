# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 FP4 index scorers reading whole rows (gfx1250).

On gfx1250 the FP4 index scorers -- the transposed decode scorer, the indexer-CP
context scorer and the prefill scorer -- hand the packed rows to the scaled WMMA
(``nvfp4_scaled_dot``: e2m1 codes with their E4M3 per-16 scales, the query in E4M3,
two terms in decode), the tiled decode scorer too. The fp16 read
(``nvfp4_load_rows``: integer e2m1 decode, the block scale by packed multiply) is
left to the transposed scorer with FP4_SCALED_DOT off (a benchmark reference).
Here:

- the fp16 read's integer decode, every byte with scale bytes of every class;
- ``nvfp4_load_rows`` against the dequantized planes: exact;
- the scaled-WMMA read (``nvfp4_scaled_dot``, the production decode / CP / prefill
  read): one-hot queries return the dequantized key elements exactly, random
  queries the float64 product of the E4M3 / two-term query within the WMMA's
  summation error (SCALED_DOT_REL);
- the decode scorers' block scores against a float64 reference on the dequantized
  cache (the scaled read, transposed and tiled: two-term query; the fp16 read,
  transposed with FP4_SCALED_DOT off: fp16 query);
- the full decode selection against the BF16 scorers fed the dequantized cache
  (near ties at the top-k edge may swap).

The CPU twin of these checks is ``TRITON_INTERPRET=1 GPU_ARCHS=gfx1250`` with the
recipe's ``tools/v4/cpu_checks_index_fp4.py``; CP-vs-TP exactness is in
``tests/model_ops/test_indexer_cp_parity.py``.
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
# products' magnitudes from float64 (probe_fp4_gfx1250.py section 3 measures it).
# Its results are bounded by this fraction of that sum.
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


@triton.jit
def _words_kernel(p_ptr, s_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    w0, w1, w2, w3 = nvfp4._fp4x8_scaled_f16_words(
        tl.load(p_ptr + i), tl.load(s_ptr + i), tl.full((N,), 0x7F7F7F7F, tl.int32)
    )
    tl.store(o_ptr + i * 4, w0)
    tl.store(o_ptr + i * 4 + 1, w1)
    tl.store(o_ptr + i * 4 + 2, w2)
    tl.store(o_ptr + i * 4 + 3, w3)


def test_fp4x8_words_decode_every_byte():
    scale_bytes = torch.tensor(
        [0x01, 0x07, 0x08, 0x30, 0x38, 0x3F, 0x55, 0x7E], dtype=torch.uint8
    )
    codes = torch.arange(256, dtype=torch.int64)
    dw = codes | (codes.roll(1) << 8) | (codes.roll(2) << 16) | (codes.roll(3) << 24)
    dw = dw.repeat(len(scale_bytes))
    s16 = scale_bytes.repeat_interleave(256).view(torch.float8_e4m3fn).to(torch.float16)
    sb = s16.view(torch.int16).to(torch.int64) & 0xFFFF
    n = dw.numel()
    out = torch.zeros(n * 4, dtype=torch.int32, device=DEV)
    _words_kernel[(1,)](
        dw.to(torch.int32).to(DEV), (sb | (sb << 16)).to(torch.int32).to(DEV), out, N=n
    )
    lut = torch.tensor(E2M1 + tuple(-x for x in E2M1))
    want = torch.empty(n, 4, 2)
    for j in range(4):
        b = (dw >> (8 * j)) & 0xFF
        want[:, j, 0] = lut[b & 15] * s16.float()
        want[:, j, 1] = lut[b >> 4] * s16.float()
    got = out.cpu().view(torch.int16).view(n, 4, 2).view(torch.float16).float()
    assert torch.equal(got, want)


@triton.jit
def _rows_kernel(p_ptr, s_ptr, o_ptr, page, sp, ss, D_: tl.constexpr):
    rows = tl.arange(0, 128)
    k = nvfp4.nvfp4_load_rows(p_ptr, s_ptr, page * sp, page * ss, rows, D_)
    tl.store(o_ptr + rows[:, None] * D_ + tl.arange(0, D_)[None, :], k)


def test_load_rows_is_the_dequantized_planes():
    payload, scale = fp4_cache(5, 1)
    for page in range(5):
        out = torch.empty(PAGE, D, dtype=torch.float16, device=DEV)
        _rows_kernel[(1,)](
            payload, scale, out, page, payload.stride(0), scale.stride(0), D_=D
        )
        assert torch.equal(out.float(), dequant(payload[page], scale[page]))


def _decode_scores(
    kernel, transposed, q, payload, scale, bt, seq, max_q, chunks, warps, **kw
):
    batch = seq.numel()
    max_block = triton.cdiv(int(seq.max()), PAGE)
    out = torch.full((1, batch * max_q, max_block), float("nan"), device=DEV)
    grid = (batch, chunks) if transposed else (chunks, batch)
    kernel[grid](
        q, payload, scale, out, bt, seq, D, D**-0.5, triton.cdiv(max_block, chunks),
        *q.stride(), *payload.stride(), scale.stride(0), scale.stride(1),
        *out.stride(), bt.stride(0), NUM_IDX_HEADS=1, MAX_Q=max_q,
        BLOCK_SIZE_K=PAGE, IS_FP4=True, num_warps=warps, num_stages=1, **kw,
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
def test_decode_scorers_match_reference_and_each_other(max_q, warps):
    from atom.model_ops.minimax_m3 import index_topk as m

    torch.manual_seed(max_q)
    pages, batch = 64, 4
    payload, scale = fp4_cache(pages, 2)
    seq = torch.tensor([20 * PAGE + 9, max_q, 7 * PAGE, 32 * PAGE], dtype=torch.int32)
    bt = torch.stack([torch.randperm(pages)[:32] for _ in range(batch)]).to(torch.int32)
    q = torch.randn(batch * max_q, 1, D, dtype=torch.bfloat16)
    seq, bt, q = seq.to(DEV), bt.to(DEV), q.to(DEV)
    a = _decode_scores(m._decode_index_score_transposed_kernel, True, q, payload, scale,
                       bt, seq, max_q, 3, warps)  # fmt: skip
    b = _decode_scores(m._decode_index_score_tiled_kernel, False, q, payload, scale, bt,
                       seq, max_q, 3, warps)  # fmt: skip
    # The production gfx1250 read: packed rows to the scaled WMMA, two-term query.
    c = _decode_scores(m._decode_index_score_transposed_kernel, True, q, payload, scale,
                       bt, seq, max_q, 3, warps, FP4_SCALED_DOT=True)  # fmt: skip
    k = dequant(payload, scale).double()
    q16 = q[:, 0].to(torch.float16).double()
    q2 = split_query(q[:, 0])
    for bi in range(batch):
        for t in range(max_q):
            r = bi * max_q + t
            cut = int(seq[bi]) - max_q + t + 1
            for blk in range(triton.cdiv(int(seq[bi]), PAGE)):
                masked = blk * PAGE + torch.arange(PAGE, device=DEV) >= cut
                kb = k[int(bt[bi, blk])]
                for got, qq, scaled in (
                    (a, q16, False),
                    (b, q2, True),
                    (c, q2, True),
                ):
                    s = (kb @ qq[r]) * D**-0.5 * LOG2E
                    s[masked] = float("-inf")
                    want = s.max().float()
                    g = got[0, r, blk]
                    if torch.isinf(want):
                        assert g == want
                    elif scaled:
                        mag = ((kb.abs() @ qq[r].abs()) * D**-0.5 * LOG2E).max()
                        assert abs(g - want) <= SCALED_DOT_REL * mag + 1e-6, (r, blk)
                    else:
                        assert torch.isclose(g, want, rtol=1e-5, atol=1e-5), (r, blk)
    assert torch.equal(torch.isnan(c), torch.isnan(a))
    assert torch.equal(torch.isnan(a), torch.isnan(b))


@pytest.mark.parametrize(
    "max_q, batch, ctx", [(1, 64, 65536), (8, 4, 300000), (4, 16, 8192)]
)
def test_decode_selection_matches_bf16_on_the_dequantized_cache(max_q, batch, ctx):
    """The production entry point (gfx1250 FP4: transposed, scaled-WMMA read, the
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
