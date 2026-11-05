# SPDX-License-Identifier: MIT
"""`gather_mxfp4_prefix`: the dense cached-prefix gather of the MXFP4 KV pool.

The oracle is `formats.dequantize_mxfp4` over `page_views` of the pool, indexed by
the same metadata `cp_mha_gather_cache` takes (one kv length a sequence, cached +
new, `seq_starts` zero). The CPU tests pin the page layout and the e8m0 edge
semantics the kernel is written to; the GPU tests need a CUDA/HIP device and triton.
"""

import pytest
import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, page_views
from atom.mono.weights.formats import dequantize_mxfp4, e8m0_to_float

SCALE_OFF = 8192  # the scales follow the 128 x 64 B data


def make_pool(pages, hkv, seed, device, scale_lo=100, scale_hi=140, extra_heads=0):
    """Random uint8 pool [pages, hkv, 8704], valid scale bytes in [lo, hi); with
    `extra_heads` it is a head-slice of a wider pool, so its page stride is larger."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = hkv + extra_heads
    p = torch.randint(0, 256, (pages, n, PAGE_BYTES), generator=g, dtype=torch.uint8)
    sc = torch.randint(scale_lo, scale_hi, (pages, n, 512), generator=g)
    p[:, :, SCALE_OFF:] = sc.to(torch.uint8)
    return p.to(device)[:, :hkv]


def make_meta(seq_lens, pages, seed, device):
    """cu_seqlens_kv, batch_id_per_k_token, seq_starts, block_tables with scattered pages."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lens = torch.tensor(seq_lens)
    cu = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
    maxblk = max(-(-n // 128) for n in seq_lens) + 1
    ids = torch.randperm(pages, generator=g)
    bt = torch.zeros(len(seq_lens), maxblk, dtype=torch.long)
    used = 0
    for b, n in enumerate(seq_lens):
        nb = -(-n // 128)
        bt[b, :nb] = ids[used : used + nb]
        used += nb
    assert used <= pages
    bid = torch.repeat_interleave(torch.arange(len(seq_lens)), lens)
    return (
        cu.to(torch.int32).to(device),
        bid.to(torch.int32).to(device),
        torch.zeros(len(seq_lens), dtype=torch.int32, device=device),
        bt.to(torch.int32).to(device),
    )


def reference(pool, cu, bid, starts, bt):
    """[T, hkv, 128] bf16 by dequantizing the whole pool then indexing by position."""
    dq = dequantize_mxfp4(*page_views(pool))  # [pages, hkv, 128, 128] fp32
    bid, cu, starts = bid.long(), cu.long(), starts.long()
    pos = torch.arange(bid.numel(), device=pool.device) - cu[bid] + starts[bid]
    page = bt.long()[bid, pos // 128]
    return dq[page, :, pos % 128].to(torch.bfloat16)


# ---- CPU: the layout and the scale semantics the kernel is written to ----------------


def test_page_layout_low_nibble_even_scale_per_32():
    page = torch.zeros(1, 1, PAGE_BYTES, dtype=torch.uint8)
    page[0, 0, 0] = 0x21  # token 0: element 0 is code 1 (0.5), element 1 code 2 (1.0)
    page[0, 0, 1] = 0xF7  # elements 2, 3: code 7 (6.0), code 15 (-6.0)
    page[0, 0, SCALE_OFF : SCALE_OFF + 4] = torch.tensor([128, 127, 126, 125])
    out = dequantize_mxfp4(*page_views(page))[0, 0, 0]
    assert out[:4].tolist() == [1.0, 2.0, 12.0, -12.0]  # group 0 scale 2^1
    assert out[32:64].abs().sum() == 0  # data zero in group 1 whatever the scale


def test_e8m0_valid_rows_decode_like_aiter_and_edges_differ_from_the_helper():
    """aiter's device decode is `byte << 23` bitcast; the torch helper differs at the
    two bytes pa_decode declares unsupported in valid rows (0 and 255)."""
    b = torch.arange(256, dtype=torch.uint8)
    raw = (b.to(torch.int32) << 23).view(torch.float32)
    helper = e8m0_to_float(b)
    assert torch.equal(raw[1:255], helper[1:255])
    assert raw[0] == 0 and helper[0] > 0
    assert torch.isinf(raw[255]) and torch.isnan(helper[255])


def test_e2m1_formula_the_kernel_computes_is_the_table():
    """The kernel's closed form (sign, (1 + m/2) * 2^(e-1), subnormal m/2) over every
    byte, at scale 2^0, equals the helper's table -- low nibble first."""
    page = torch.zeros(1, 1, PAGE_BYTES, dtype=torch.uint8)
    page[0, 0, :SCALE_OFF] = torch.arange(SCALE_OFF) % 256  # every byte value
    page[0, 0, SCALE_OFF:] = 127
    table = dequantize_mxfp4(*page_views(page))[0, 0]  # [128, 128]
    byte = page[0, 0, :SCALE_OFF].view(128, 64).to(torch.int32)
    code = torch.stack([byte & 0xF, byte >> 4], -1).reshape(128, 128)
    e, m = (code & 7) >> 1, (code & 1).float()
    val = torch.where(e == 0, m * 0.5, (1 + 0.5 * m) * 2.0 ** (e - 1).clamp(min=0))
    val = torch.where(code >> 3 == 1, -val, val)
    assert torch.equal(val, table)


# ---- GPU: the kernel against the oracle ----------------------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@gpu
@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("extra_heads", [0, 1])
@pytest.mark.parametrize("seq_lens", [[300], [128, 1, 129, 257], [5, 640, 17]], ids=str)
def test_gather_matches_dequantized_pool(hkv, extra_heads, seq_lens):
    pytest.importorskip("triton")
    from atom.model_ops.minimax_m3.gather_mxfp4 import gather_mxfp4_prefix

    pages = 24
    k = make_pool(pages, hkv, 1, "cuda", extra_heads=extra_heads)
    v = make_pool(pages, hkv, 2, "cuda", extra_heads=extra_heads)
    assert extra_heads == 0 or k.stride(0) != hkv * PAGE_BYTES
    cu, bid, starts, bt = make_meta(seq_lens, pages, 3, "cuda")
    kk, vv = gather_mxfp4_prefix(
        key_cache=k,
        value_cache=v,
        block_tables=bt,
        cu_seqlens_kv=cu,
        batch_id_per_k_token=bid,
        seq_starts=starts,
        total_tokens=sum(seq_lens),
    )
    assert kk.dtype == vv.dtype == torch.bfloat16
    assert kk.shape == (sum(seq_lens), hkv, 128)
    torch.testing.assert_close(kk, reference(k, cu, bid, starts, bt), rtol=0, atol=0)
    torch.testing.assert_close(vv, reference(v, cu, bid, starts, bt), rtol=0, atol=0)


@gpu
def test_scale_edges_follow_the_aiter_decode():
    """Bytes 1 and 254 (the valid extremes) equal the oracle; 0 and 255 are not valid
    rows and are pinned to aiter's raw decode, not the torch helper."""
    pytest.importorskip("triton")
    from atom.model_ops.minimax_m3.gather_mxfp4 import gather_mxfp4_prefix

    k = make_pool(1, 1, 5, "cuda")
    v = make_pool(1, 1, 6, "cuda")
    cu, bid, starts, bt = make_meta([4], 1, 0, "cuda")
    edges = torch.tensor([1, 254, 0, 255], dtype=torch.uint8, device="cuda")
    for pool in (k, v):  # token t uses scale byte edges[t] in all four groups
        pool[0, 0, SCALE_OFF:].view(128, 4)[:4] = edges[:, None]
    kk, vv = gather_mxfp4_prefix(k, v, bt, cu, bid, starts, 4)
    for out, pool in ((kk, k), (vv, v)):
        data, _ = page_views(pool)
        codes = dequantize_mxfp4(data, torch.full_like(page_views(pool)[1], 127))[
            0, 0, :4
        ]  # unscaled e2m1 values
        raw = (edges.to(torch.int32) << 23).view(torch.float32)
        expect = (codes * raw[:, None]).to(torch.bfloat16)
        torch.testing.assert_close(out[:, 0], expect, rtol=0, atol=0, equal_nan=True)
        valid = dequantize_mxfp4(*page_views(pool))[0, 0, :2].to(torch.bfloat16)
        torch.testing.assert_close(out[:2, 0], valid, rtol=0, atol=0)


@gpu
def test_every_e2m1_code_and_both_nibbles():
    pytest.importorskip("triton")
    from atom.model_ops.minimax_m3.gather_mxfp4 import gather_mxfp4_prefix

    k = make_pool(1, 1, 1, "cuda")
    k[0, 0, :SCALE_OFF] = (torch.arange(SCALE_OFF, device="cuda") % 256).to(torch.uint8)
    cu, bid, starts, bt = make_meta([128], 1, 0, "cuda")
    kk, _ = gather_mxfp4_prefix(k, k, bt, cu, bid, starts, 128)
    torch.testing.assert_close(kk, reference(k, cu, bid, starts, bt), rtol=0, atol=0)


@gpu
def test_invalid_page_ids_read_nothing():
    """A page id outside the pool, or a position past the block table, gathers zeros
    and does not fault; valid tokens are unaffected."""
    pytest.importorskip("triton")
    from atom.model_ops.minimax_m3.gather_mxfp4 import gather_mxfp4_prefix

    k = make_pool(4, 1, 1, "cuda")
    cu, bid, starts, bt = make_meta([128, 300], 4, 0, "cuda")  # 1 + 3 blocks
    good = reference(k, cu, bid, starts, bt)
    bad = bt[:, :3].clone()
    bad[1, 1] = 4  # == num_pages
    bad[1, 2] = -1
    out = gather_mxfp4_prefix(k, k, bad, cu, bid, starts, 428)[0]
    torch.testing.assert_close(out[:256], good[:256], rtol=0, atol=0)
    assert not out[256:].any()
    narrow = gather_mxfp4_prefix(k, k, bt[:, :2].contiguous(), cu, bid, starts, 428)[0]
    torch.testing.assert_close(narrow[:384], good[:384], rtol=0, atol=0)
    assert not narrow[384:].any()  # seq 1 positions 256+ lie past a 2-wide table


@gpu
def test_negative_position_reads_nothing():
    """seq_starts = -1 puts token 0 at position -1: zeros, no fault; the rest shift."""
    pytest.importorskip("triton")
    from atom.model_ops.minimax_m3.gather_mxfp4 import gather_mxfp4_prefix

    k = make_pool(4, 1, 1, "cuda")
    cu, bid, starts, bt = make_meta([130], 4, 0, "cuda")
    good = reference(k, cu, bid, starts, bt)  # positions 0..129
    out = gather_mxfp4_prefix(k, k, bt, cu, bid, starts - 1, 130)[0]
    assert not out[0].any()
    torch.testing.assert_close(out[1:], good[:129], rtol=0, atol=0)


@gpu
def test_empty_and_bad_inputs():
    pytest.importorskip("triton")
    from atom.model_ops.minimax_m3.gather_mxfp4 import gather_mxfp4_prefix

    k = make_pool(2, 1, 1, "cuda")
    cu, bid, starts, bt = make_meta([3], 2, 0, "cuda")
    kk, vv = gather_mxfp4_prefix(k, k, bt, cu, bid[:0], starts, 0)
    assert kk.shape == vv.shape == (0, 1, 128)
    with pytest.raises(ValueError, match="uint8"):
        gather_mxfp4_prefix(k.view(torch.int8), k, bt, cu, bid, starts, 3)
    with pytest.raises(ValueError, match="uint8"):
        gather_mxfp4_prefix(k, k[..., :8192], bt, cu, bid, starts, 3)
    with pytest.raises(ValueError, match="int32"):
        gather_mxfp4_prefix(k, k, bt.long(), cu, bid, starts, 3)
    with pytest.raises(ValueError, match="int32"):
        gather_mxfp4_prefix(k, k, bt, cu, bid, starts.cpu(), 3)
    with pytest.raises(ValueError, match="contiguous"):
        gather_mxfp4_prefix(
            k, k, torch.zeros_like(bt).repeat(1, 2)[:, ::2], cu, bid, starts, 3
        )
    with pytest.raises(ValueError, match="covers"):
        gather_mxfp4_prefix(k, k, bt, cu, bid, starts, 4)
    with pytest.raises(ValueError, match="bfloat16"):
        gather_mxfp4_prefix(k, k, bt, cu, bid, starts, 3, torch.float16)
