# SPDX-License-Identifier: MIT
"""The norm-free MXFP4 K/V writer (``store_kv_mxfp4``) EAGLE3's draft layer uses.

CPU part (aiter + triton importable, no GPU): the argument contract raises before any
launch, and the private aiter quant op the kernel is built on still has the
signature it calls. GPU part (gfx950, SKIPPED without one; NOT RUN until a GPU is
authorized): the written pages are, bit for bit, ``per_1x32_f4_quant`` of the rows,
with the zero-group scale rule the aiter fused writer pins (byte 1, never 0); a
slot < 0 leaves its bytes at the pad pattern; K and V may be strided views of one
pool and of a packed qkv; the write is graph-capturable and replays with new slots;
and the pages read back through the prefix gather and ``pa_decode`` (bf16 q, q_len 1
and 4, MHA and GQA kv-head counts, group 1 included).
"""

import inspect

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, PAGE_TOKENS
from atom.model_ops.minimax_m3.store_mxfp4 import store_kv_mxfp4

D = 128
PAD = 0xA5  # fills the pages first: a byte the writer must not touch stays this


def rows(tokens, heads, dtype=torch.bfloat16):
    return torch.randn(tokens, heads, D).to(dtype)


def pool(pages, heads):
    return torch.full((pages, heads, PAGE_BYTES), PAD, dtype=torch.uint8)


# --- CPU: contract ------------------------------------------------------------


def test_the_private_quant_op_keeps_the_signature_the_kernel_calls():
    from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

    params = inspect.signature(_mxfp4_quant_op.fn).parameters
    assert list(params)[:4] == [
        "x",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_M",
        "MXFP4_QUANT_BLOCK_SIZE",
    ]
    assert "SCALING_MODE" in params and "USE_ASM" in params


@pytest.mark.parametrize(
    "mutate, why",
    [
        (lambda a: a.update(k=a["k"].float()), "bf16"),
        (lambda a: a.update(k=a["k"][..., ::2]), "unit last stride"),
        (lambda a: a.update(v=rows(3, 2)), "must match"),
        (lambda a: a.update(k_pages=pool(2, 3)), "uint8 \\[pages, 2"),
        (lambda a: a.update(v_pages=pool(5, 2)), "same page count"),
        (lambda a: a.update(slots=a["slots"].float()), "int32/int64"),
        (lambda a: a.update(slots=a["slots"][:2]), "slot_mapping"),
    ],
)
def test_a_bad_argument_raises_before_any_launch(mutate, why):
    args = {
        "k": rows(4, 2),
        "v": rows(4, 2),
        "k_pages": pool(2, 2),
        "v_pages": pool(2, 2),
        "slots": torch.arange(4),
    }
    mutate(args)
    with pytest.raises(ValueError, match=why):
        store_kv_mxfp4(
            args["k"], args["v"], args["k_pages"], args["v_pages"], args["slots"]
        )


def test_no_tokens_is_a_no_op():
    k = rows(0, 2)
    p = pool(2, 2)
    store_kv_mxfp4(k, k.clone(), p, p.clone(), torch.zeros(0, dtype=torch.int64))
    assert (p == PAD).all()


# --- GPU (gfx950): NOT RUN until authorized -------------------------------------


def _gfx950():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


gpu = pytest.mark.skipif(not _gfx950(), reason="needs a gfx950 GPU")


def expected(k, v, slots, num_pages, heads):
    """The (K, V) pool the writer must leave: ``per_1x32_f4_quant`` of the rows at
    their slots (an all-zero group gets byte 1, the fused writer's rule), the rest
    at PAD."""
    from aiter.ops.quant import per_1x32_f4_quant

    ok = (slots >= 0).nonzero().flatten()
    out = []
    for src in (k, v):
        want = pool(num_pages, heads).cuda()
        flat = src[ok].reshape(-1, D)
        q, s = per_1x32_f4_quant(flat)
        data = q.view(torch.uint8).reshape(-1, heads, 64)
        scale = s.view(torch.uint8).reshape(-1, 4)[: flat.shape[0]]
        zero = flat.float().reshape(-1, 4, 32).abs().amax(-1) == 0
        scale = torch.where(zero & (scale == 0), torch.ones_like(scale), scale)
        scale = scale.reshape(-1, heads, 4)
        strides = (heads * PAGE_BYTES, PAGE_BYTES)
        d = torch.as_strided(want, (num_pages, heads, 128, 64), (*strides, 64, 1))
        sc = torch.as_strided(
            want, (num_pages, heads, 128, 4), (*strides, 4, 1), 128 * 64
        )
        at = slots[ok]
        d[at // 128, :, at % 128] = data
        sc[at // 128, :, at % 128] = scale
        out.append(want)
    return out


@gpu
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_pages_are_per_1x32_f4_quant_of_the_rows_and_skipped_slots_are_untouched(heads):
    tokens, num_pages = 70, 3
    gen = torch.Generator().manual_seed(heads)
    # k, v as views of a packed qkv: row stride is the whole row, not heads * 128
    qkv = torch.randn(tokens, 3 * heads, D, generator=gen).to(torch.bfloat16).cuda()
    k, v = qkv[:, heads : 2 * heads], qkv[:, 2 * heads :]
    k[3] = 0  # an all-zero row: scale byte 1, not 0
    k[4] = 3e4  # large magnitudes
    v[5] = 1e-30  # near-zero, still a valid scale
    perm = torch.randperm(num_pages * PAGE_TOKENS, generator=gen)
    slots = perm[(perm != 127) & (perm != 128)][:tokens].cuda()  # distinct slots
    slots[1::5] = -1
    slots[0], slots[2] = 127, 128  # the page boundary
    # K and V are views of one pool: neither page stride is the tight one
    both = torch.full((num_pages, 2, heads, PAGE_BYTES), PAD, dtype=torch.uint8).cuda()
    store_kv_mxfp4(k, v, both[:, 0], both[:, 1], slots)
    want_k, want_v = expected(k, v, slots, num_pages, heads)
    assert torch.equal(both[:, 0], want_k)
    assert torch.equal(both[:, 1], want_v)
    scales = both[:, 0].reshape(num_pages, heads, PAGE_BYTES)[:, :, 128 * 64 :]
    written = scales.reshape(-1, 4)[scales.reshape(-1, 4)[:, 0] != PAD]
    assert ((written >= 1) & (written <= 254)).all()


@gpu
def test_the_write_is_capturable_and_replays_with_new_slots():
    heads, tokens, num_pages = 2, 8, 2
    k, v = rows(tokens, heads).cuda(), rows(tokens, heads).cuda()
    pages = pool(num_pages, heads).cuda(), pool(num_pages, heads).cuda()
    slots = torch.arange(tokens, device="cuda")
    store_kv_mxfp4(k, v, *pages, slots)  # compile before capture
    for p in pages:
        p.fill_(PAD)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        store_kv_mxfp4(k, v, *pages, slots)
    new = torch.tensor([200, -1, 3, -1, 130, 7, -1, 129], device="cuda")
    slots.copy_(new)
    graph.replay()
    want_k, want_v = expected(k, v, new, num_pages, heads)
    assert torch.equal(pages[0], want_k) and torch.equal(pages[1], want_v)


@gpu
def test_the_pages_read_back_through_the_prefix_gather():
    from aiter.utility import fp4_utils

    from atom.model_ops.minimax_m3.gather_mxfp4 import gather_mxfp4_prefix

    heads, tokens = 2, 130
    k, v = rows(tokens, heads).cuda(), rows(tokens, heads).cuda()
    kp, vp = pool(2, heads).cuda(), pool(2, heads).cuda()
    slots = torch.arange(tokens, device="cuda")
    store_kv_mxfp4(k, v, kp, vp, slots)
    i32 = {"dtype": torch.int32, "device": "cuda"}
    out_k, out_v = gather_mxfp4_prefix(
        kp,
        vp,
        torch.tensor([[0, 1]], **i32),
        torch.tensor([0, tokens], **i32),
        torch.zeros(tokens, **i32),
        torch.zeros(1, **i32),
        tokens,
    )
    want_k, want_v = expected(k, v, slots, 2, heads)
    for got, pages in ((out_k, want_k), (out_v, want_v)):
        data = pages[:, :, : 128 * 64].reshape(2, heads, 128, 64)
        scale = pages[:, :, 128 * 64 :].reshape(2, heads, 128, 4)
        deq = fp4_utils.mxfp4_to_f32(data.reshape(-1, 64).clone()) * (
            fp4_utils.e8m0_to_f32(scale.reshape(-1, 4)).repeat_interleave(32, -1)
        )
        deq = deq.reshape(2, heads, 128, D).permute(0, 2, 1, 3).reshape(-1, heads, D)
        torch.testing.assert_close(got.float(), deq[:tokens], atol=0, rtol=0)


@gpu
@pytest.mark.parametrize("heads, kv_heads", [(4, 4), (4, 2), (16, 1)])
@pytest.mark.parametrize("qlen", [1, 4])
def test_decode_over_the_written_pages_matches_the_dequantized_reference(
    heads, kv_heads, qlen
):
    """bf16 q (the kernel scales it to e4m3 per head row) at group 1 (the MHA draft),
    4 and 16 (the GQA draft at TP4), q_len 1 and 4 (MTP verify width)."""
    from aiter.utility import fp4_utils

    import atom.model_ops.attention_mha as am

    ctx = torch.tensor([130, 257, 9], dtype=torch.int32)
    seqs, blocks = len(ctx), 3
    pages_needed = seqs * blocks
    gen = torch.Generator().manual_seed(heads * 10 + qlen)
    keys = torch.randn(seqs, blocks * 128, kv_heads, D, generator=gen)
    vals = torch.randn(seqs, blocks * 128, kv_heads, D, generator=gen)
    keys, vals = keys.bfloat16().cuda(), vals.bfloat16().cuda()
    kp, vp = pool(pages_needed, kv_heads).cuda(), pool(pages_needed, kv_heads).cuda()
    bt = torch.arange(pages_needed, dtype=torch.int32).view(seqs, blocks).cuda()
    slots = (bt.long() * 128).repeat_interleave(128, dim=1) + torch.arange(
        128, device="cuda"
    ).repeat(blocks)
    # only the first ctx[i] tokens of a request are real: the rest stay unwritten
    live = torch.arange(blocks * 128, device="cuda")[None] < ctx.cuda()[:, None]
    slots = torch.where(live, slots, torch.full_like(slots, -1))
    store_kv_mxfp4(
        keys.reshape(-1, kv_heads, D),
        vals.reshape(-1, kv_heads, D),
        kp,
        vp,
        slots.reshape(-1),
    )
    # a peaky softmax, so a wrong token or page changes the output by far more than
    # the e4m3 rounding of q does (tolerance to be calibrated on the first GPU run)
    q = (2 * torch.randn(seqs * qlen, heads, D, generator=gen)).bfloat16().cuda()
    scale = D**-0.5
    rows_ctx, rows_bt = am.dense_decode_rows(ctx.cuda(), bt, qlen)
    got = am.dense_pa_mxfp4(q, kp, vp, rows_ctx, rows_bt, scale, plan_step_owner=None)

    def dequant(pages):
        data = pages[:, :, : 128 * 64].reshape(-1, 64).clone()
        sc = pages[:, :, 128 * 64 :].reshape(-1, 4)
        f = fp4_utils.mxfp4_to_f32(data) * fp4_utils.e8m0_to_f32(sc).repeat_interleave(
            32, -1
        )
        return f.reshape(pages.shape[0], kv_heads, 128, D)

    k_deq, v_deq = dequant(kp), dequant(vp)
    group = heads // kv_heads
    for row in range(seqs * qlen):
        b = row // qlen
        n = int(rows_ctx[row])
        k_row = k_deq[bt[b].long()].permute(0, 2, 1, 3).reshape(-1, kv_heads, D)[:n]
        v_row = v_deq[bt[b].long()].permute(0, 2, 1, 3).reshape(-1, kv_heads, D)[:n]
        for h in range(heads):
            kv = h // group
            p = torch.softmax(q[row, h].float() @ k_row[:, kv].T * scale, -1)
            torch.testing.assert_close(
                got[row, h].float(), p @ v_row[:, kv], atol=0.15, rtol=0.1
            )
