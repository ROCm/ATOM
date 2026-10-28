# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono over an MXFP4 index cache (gfx950): the K1 index-K write against aiter's
fused writer, byte for byte, and the scores K1 emits against aiter's
``score_flydsl_fp4``. The same pages are written by Mono steps and by the
original path's steps, so the bytes are a contract, not a tolerance."""

import math

import pytest
import torch

aiter = pytest.importorskip("aiter")
flyc = pytest.importorskip("flydsl.compiler")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter, "fused_qknorm_idxrqknorm"),
    reason="needs gfx950 and aiter.fused_qknorm_idxrqknorm",
)

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import Int32, Int64, T

from atom.model_ops.minimax_m3.kv_mxfp4 import (
    PAGE_BYTES,
    dequantize_pages,
)
from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
)
from atom.models.minimax_m3.mono.kernels.common import (
    ld_bf16x4,
    mxfp4_row_store,
)
from atom.models.minimax_m3.mono.kernels.index_score import (
    build_index_score_kernel,
    index_scale_log2e,
)
from atom.mono.device.ops import rsrc

PAGES = 6
SM_SCALE = HEAD_DIM**-0.5


def _row_store_launcher(n_tokens: int):
    """A launch of ``mxfp4_row_store`` the way K1's index_k head task makes it: a
    token a half wave, lane l holding elements 4 l .. 4 l + 3."""
    blocks = (n_tokens + 1) // 2

    @flyc.kernel(name="mono_fp4_row_store_test", known_block_size=[64, 1, 1])
    def kernel(rows: Int64, slots: Int64, cache: Int64, n: Int32):
        tid = fx.thread_idx.x
        tk = fx.block_idx.x * 2 + tid // 32
        l32 = tid % 32
        if tk < n:
            slot = fx.Int32(
                bo.buffer_load(rsrc(slots), 2 * tk, vec_width=1, dtype=T.i32)
            )
            vals = ld_bf16x4(rsrc(rows), tk * HEAD_DIM + l32 * 4)
            if slot >= 0:
                mxfp4_row_store(rsrc(cache), slot, vals, l32)

    @flyc.jit
    def launch(
        rows: Int64,
        slots: Int64,
        cache: Int64,
        n: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(rows, slots, cache, n).launch(grid=(blocks,), block=(64,), stream=stream)

    return launch


def _cos_sin(device):
    pos = torch.arange(4096, device=device, dtype=torch.float32)[:, None]
    inv = 1e7 ** (-torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
    ang = pos * inv[None]
    return torch.cat((ang.cos(), ang.sin()), dim=-1).to(torch.bfloat16)


def _aiter_index_rows(qkv, position, slots, mxfp4: bool):
    """The index key rows aiter's fused writer leaves: bf16 rows [PAGES, 128, D]
    (``mxfp4`` False) or the 8704 B pages it quantizes them into."""
    dev = qkv.device
    nq, nkv, niq = 16, 1, 1
    g = torch.Generator(device=dev).manual_seed(0)

    def w():
        return (torch.rand(HEAD_DIM, device=dev, generator=g) + 0.5).to(torch.bfloat16)

    qn, kn, iqn, ikn = w(), w(), w(), w()
    t = qkv.shape[0]
    kc = torch.zeros(PAGES, 128, nkv, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    qo = torch.empty(t, nq * HEAD_DIM, device=dev, dtype=torch.bfloat16)
    iqo = torch.empty(t, niq * HEAD_DIM, device=dev, dtype=torch.bfloat16)
    if mxfp4:
        ic = torch.zeros(PAGES, PAGE_BYTES, device=dev, dtype=torch.uint8)
    else:
        ic = torch.zeros(PAGES, 128, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    cs = _cos_sin(dev)
    aiter.fused_qknorm_idxrqknorm(
        qkv, qn, kn, cs, position, nq, nkv, 64, 1e-6, iqn, ikn, niq, slots, kc, vc,
        ic, 128, qo, iqo, slots, kv_cache_dtype="auto",
        index_cache_dtype="mxfp4" if mxfp4 else "auto", asm_layout=False,
    )  # fmt: skip
    return ic


@pytest.mark.parametrize("scale", [1.0, 1e-3, 40.0])
def test_the_index_k_row_store_is_aiters_writer_byte_for_byte(scale):
    t = 61
    g = torch.Generator(device="cuda").manual_seed(3)
    width = (16 + 2 + 1 + 1) * HEAD_DIM
    qkv = (
        torch.randn(t, width, device="cuda", generator=g, dtype=torch.float32) * scale
    ).to(torch.bfloat16)
    # a few rows with a whole 32-group of zeros, and one of tiny magnitude: the
    # scale floor
    qkv[3, 19 * HEAD_DIM : 19 * HEAD_DIM + 64] = 0
    position = torch.arange(t, device="cuda", dtype=torch.int64) + 5
    slots = torch.randperm(PAGES * 128, device="cuda", generator=g)[:t].to(torch.int64)
    slots[7] = -1
    want_rows = _aiter_index_rows(qkv, position, slots, mxfp4=False)
    want_pages = _aiter_index_rows(qkv, position, slots, mxfp4=True)

    valid = slots >= 0
    rows = torch.zeros(t, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    rows[valid] = want_rows.view(-1, HEAD_DIM)[slots[valid]]
    pages = torch.zeros(PAGES, PAGE_BYTES, device="cuda", dtype=torch.uint8)
    _row_store_launcher(t)(rows.data_ptr(), slots.data_ptr(), pages.data_ptr(), t)
    torch.cuda.synchronize()
    assert torch.equal(pages, want_pages), (
        "bytes differ at "
        f"{(pages != want_pages).nonzero()[:8].tolist()} of {int((pages != want_pages).sum())}"
    )


def _fp4_pages(blocks, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = torch.randint(
        0, 256, (blocks, PAGE_BYTES), device="cuda", dtype=torch.uint8, generator=g
    )
    # e8m0 codes around 1: a sane scale, never 0 / 255
    pages[:, 8192:] = torch.randint(
        118,
        128,
        (blocks, PAGE_BYTES - 8192),
        device="cuda",
        dtype=torch.uint8,
        generator=g,
    )
    return pages


@pytest.mark.parametrize(
    "ctx, requests, q_len",
    [
        (100 * 128 + 37, 1, 8),
        (600 * 128 + 5, 1, 8),
        (300 * 128 + 70, 2, 8),
        (200 * 128 + 3, 1, 1),
    ],
)
def test_mono_block_scores_match_the_flydsl_fp4_scorer(ctx, requests, q_len):
    from atom.model_ops.minimax_m3.index_mxfp4 import decode_score_mxfp4

    tokens = requests * q_len
    n_blk = (ctx + 127) // 128
    total_blocks = requests * n_blk + 1
    pages = _fp4_pages(total_blocks, 11)
    g = torch.Generator(device="cuda").manual_seed(5)
    iq = torch.randn(tokens, 1, HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
    block_table = (
        torch.randperm(total_blocks, device="cuda", generator=g)[: requests * n_blk]
        .view(requests, n_blk)
        .to(torch.int32)
        .contiguous()
    )
    seq = torch.full((requests,), ctx, device="cuda", dtype=torch.int32)
    back = torch.arange(q_len - 1, -1, -1, device="cuda", dtype=torch.int32)
    seq_tok = (seq[:, None] - back).reshape(-1).contiguous()
    bt_tok = block_table.repeat_interleave(q_len, dim=0).contiguous()

    want = decode_score_mxfp4(
        iq, pages, block_table, seq, n_blk, q_len, 1, SM_SCALE
    )  # [1, tokens, n_blk]

    launch = build_index_score_kernel(SM_SCALE, 0, 1, tokens, mxfp4=True)
    iscore = torch.zeros(
        2 * MAX_INDEX_BLOCKS * MAX_TOKENS, device="cuda", dtype=torch.int32
    )
    layer = 3
    launch(
        iq.data_ptr(), pages.data_ptr(), bt_tok.data_ptr(), seq_tok.data_ptr(),
        n_blk, q_len, iscore.data_ptr(), layer,
    )  # fmt: skip
    torch.cuda.synchronize()
    got = (
        iscore.view(-1, 2)[:, 0].view(torch.float32).view(MAX_TOKENS, MAX_INDEX_BLOCKS)
    )
    tags = iscore.view(-1, 2)[:, 1].view(MAX_TOKENS, MAX_INDEX_BLOCKS)
    # blocks past the TOPK_BLOCKS short-context cut and outside the pinned last
    # block are scored: block 0 .. n_blk - 2 (init 0, local 1)
    if n_blk <= 16:
        pytest.skip("a context of <= 16 blocks is not scored")
    scored = slice(0, n_blk - 1)
    assert (tags[:tokens, scored] == layer + 1).all()
    ref = want[0, :, scored]
    mine = got[:tokens, scored]
    assert torch.isfinite(mine).all()
    # the scale: both fold fp32(sm_scale * log2 e)
    assert math.isclose(
        index_scale_log2e(SM_SCALE), SM_SCALE * 1.4426950408889634, rel_tol=1e-6
    )
    torch.testing.assert_close(mine, ref, rtol=2e-5, atol=1e-5)
    # the selection a score feeds: the same top-16 blocks per token
    k = min(16, n_blk - 1)
    assert torch.equal(
        mine.topk(k, dim=-1).indices.sort(-1).values,
        ref.topk(k, dim=-1).indices.sort(-1).values,
    )


def test_the_page_dequantization_oracle_agrees_with_the_cache_layout():
    """The test's own pages decode through ``dequantize_pages`` (the layout both
    sides read), so a wrong layout constant would show as non-finite values."""
    pages = _fp4_pages(4, 1)
    assert torch.isfinite(dequantize_pages(pages)).all()
