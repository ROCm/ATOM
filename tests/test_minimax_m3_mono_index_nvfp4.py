# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono over an NVFP4 index cache (gfx950, single-buffer 9216 B pages): the block
scores of ``build_index_score_kernel(nvfp4=True)`` against aiter's
``score_flydsl_nvfp4`` (``k_round="e4m3"`` with an e4m3 q, the served combination: K =
e4m3_rn_satfinite(e2m1 x scale), q e4m3 as vLLM's quantizer makes it) on the same pages, bit for bit, and
the top-16 selection they feed.

The reference trees are not part of this repo: set ``ATOM_NVFP4_AITER_KERNELS`` (the
aiter ``aiter/ops/flydsl/kernels`` directory of the nvfp4 index scorer) and
``ATOM_NVFP4_VLLM_REF`` (vLLM's ``tests/kernels/attention``, for its quantizer);
the test skips without them."""

import os
import sys

import pytest
import torch

aiter = pytest.importorskip("aiter")
pytest.importorskip("flydsl")
KERNELS = os.environ.get("ATOM_NVFP4_AITER_KERNELS")
VLLM_REF = os.environ.get("ATOM_NVFP4_VLLM_REF")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not KERNELS
    or not VLLM_REF,
    reason="needs gfx950 and the nvfp4 reference trees (see the module doc)",
)

from atom.models.minimax_m3.mono.config import HEAD_DIM, MAX_INDEX_BLOCKS, MAX_TOKENS
from atom.models.minimax_m3.mono.kernels.index_score import (
    NVFP4_PAGE_BYTES,
    build_index_score_kernel,
)

SM_SCALE = HEAD_DIM**-0.5


def _reference_modules():
    import aiter.ops.flydsl.kernels as pkg

    pkg.__path__.insert(0, KERNELS)
    from aiter.ops.flydsl.kernels import minimax_m3_index_score as scorer

    sys.path.insert(0, VLLM_REF)
    import minimax_m3_nvfp4_ref as ref

    return scorer, ref


def _pages(ref, blocks, seed):
    """[blocks, 9216] single-buffer pages of random keys, quantized as vLLM's
    writer does (``quantize_nvfp4``, no global scale)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = torch.randn(blocks * 128, HEAD_DIM, device="cuda", generator=g)
    codes, scales = ref.quantize_nvfp4(keys.to(torch.bfloat16).float(), 1.0)
    codes = codes.to(torch.uint8).view(blocks, 128, HEAD_DIM)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    pages = torch.empty(blocks, NVFP4_PAGE_BYTES, device="cuda", dtype=torch.uint8)
    pages[:, : 128 * 64] = data.reshape(blocks, -1)
    pages[:, 128 * 64 :] = scales.reshape(blocks, -1)
    return pages


@pytest.mark.parametrize(
    "ctx, requests, q_len",
    [
        (100 * 128 + 37, 1, 8),
        (600 * 128 + 5, 1, 8),
        (300 * 128 + 70, 2, 8),
        (200 * 128 + 3, 1, 1),
        (40 * 128 + 128, 3, 1),
    ],
)
def test_mono_block_scores_equal_the_aiter_nvfp4_scorer_bit_for_bit(
    ctx, requests, q_len
):
    scorer, ref = _reference_modules()
    tokens = requests * q_len
    n_blk = (ctx + 127) // 128
    total_blocks = requests * n_blk + 1
    pages = _pages(ref, total_blocks, 11)
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

    kd = torch.as_strided(pages, (total_blocks, 128, 64), (NVFP4_PAGE_BYTES, 64, 1))
    ks = torch.as_strided(
        pages, (total_blocks, 128, 8), (NVFP4_PAGE_BYTES, 8, 1), 128 * 64
    )
    iq8 = iq.float().clamp(-448, 448).to(torch.float8_e4m3fn)  # vLLM's index q
    want = scorer.score_flydsl_nvfp4(
        iq8, kd, ks, block_table, seq, q_len, 1, SM_SCALE, n_blk, k_round="e4m3"
    )  # [1, tokens, n_blk]

    launch = build_index_score_kernel(SM_SCALE, 0, 1, tokens, nvfp4=True)
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
    if n_blk <= 16:
        pytest.skip("a context of <= 16 blocks is not scored")
    scored = slice(0, n_blk - 1)  # init 0, local 1: the last block is pinned
    assert (tags[:tokens, scored] == layer + 1).all()
    mine, theirs = got[:tokens, scored], want[0, :, scored]
    assert torch.isfinite(mine).all()
    assert torch.equal(mine, theirs), (mine - theirs).abs().max().item()
    k = min(16, n_blk - 1)
    assert torch.equal(
        mine.topk(k, dim=-1).indices.sort(-1).values,
        theirs.topk(k, dim=-1).indices.sort(-1).values,
    )


def _row_store_launcher(n_tokens: int):
    """A launch of ``nvfp4_row_store`` the way K1's index_k head task makes it: a
    token a half wave, lane l holding elements 4 l .. 4 l + 3."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int32, Int64, T

    from atom.models.minimax_m3.mono.kernels.common import ld_bf16x4, nvfp4_row_store
    from atom.mono.device.ops import rsrc

    blocks = (n_tokens + 1) // 2

    @flyc.kernel(name="mono_nvfp4_row_store_test", known_block_size=[64, 1, 1])
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
                nvfp4_row_store(rsrc(cache), slot, vals, l32)

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


@pytest.mark.parametrize("scale", [1.0, 1e-3, 40.0, 600.0])
def test_the_index_k_row_store_is_vllms_writer_byte_for_byte(scale):
    _, ref = _reference_modules()
    t, blocks = 61, 6
    g = torch.Generator(device="cuda").manual_seed(3)
    rows = (torch.randn(t, HEAD_DIM, device="cuda", generator=g) * scale).to(
        torch.bfloat16
    )
    rows[3, :64] = 0  # whole groups of zeros: a zero scale
    rows[5, 16:32] = 1e-7  # tiny: a subnormal-range scale
    slots = torch.randperm(blocks * 128, device="cuda", generator=g)[:t].to(torch.int64)
    slots[7] = -1
    want = torch.zeros(blocks, 128, 72, device="cuda", dtype=torch.uint8)
    ref.insert_nvfp4_index(want, rows, slots)
    got = torch.zeros(blocks, NVFP4_PAGE_BYTES, device="cuda", dtype=torch.uint8)
    _row_store_launcher(t)(rows.data_ptr(), slots.data_ptr(), got.data_ptr(), t)
    torch.cuda.synchronize()
    assert int((want != 0).sum()) > 1000  # not a vacuous comparison
    bad = (got != want.view(blocks, -1)).nonzero()
    assert bad.numel() == 0, f"{bad.shape[0]} bytes differ, first {bad[:6].tolist()}"


def _kv_store_launcher(n_tokens: int):
    """K1's K / V head tasks' stores: a token a half wave, lane l holding elements
    4 l .. 4 l + 3 of the K row and of the V row."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int32, Int64, T

    from atom.models.minimax_m3.mono.kernels.common import (
        ld_bf16x4,
        nvfp4_k_store,
        nvfp4_v_store,
    )
    from atom.mono.device.ops import rsrc

    blocks = (n_tokens + 1) // 2

    @flyc.kernel(name="mono_nvfp4_kv_store_test", known_block_size=[64, 1, 1])
    def kernel(
        krows: Int64, vrows: Int64, slots: Int64, cache: Int64, gs: Int64, n: Int32
    ):
        tid = fx.thread_idx.x
        tk = fx.block_idx.x * 2 + tid // 32
        l32 = tid % 32
        if tk < n:
            slot = fx.Int32(
                bo.buffer_load(rsrc(slots), 2 * tk, vec_width=1, dtype=T.i32)
            )
            kv = ld_bf16x4(rsrc(krows), tk * HEAD_DIM + l32 * 4)
            vv = ld_bf16x4(rsrc(vrows), tk * HEAD_DIM + l32 * 4)
            kg = fx.Float32(bo.buffer_load(rsrc(gs), 0, vec_width=1, dtype=T.f32))
            vg = fx.Float32(bo.buffer_load(rsrc(gs), 1, vec_width=1, dtype=T.f32))
            if slot >= 0:
                nvfp4_k_store(rsrc(cache), slot, kv, l32, kg)
                nvfp4_v_store(rsrc(cache), cache, slot, vv, l32, vg)

    @flyc.jit
    def launch(
        krows: Int64,
        vrows: Int64,
        slots: Int64,
        cache: Int64,
        gs: Int64,
        n: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(krows, vrows, slots, cache, gs, n).launch(
            grid=(blocks,), block=(64,), stream=stream
        )

    return launch


@pytest.mark.parametrize("k_scale, v_scale", [(1.0, 1.0), (0.37, 2.5), (8.0, 0.05)])
def test_the_kv_row_stores_are_vllms_writer_byte_for_byte(k_scale, v_scale):
    _, ref = _reference_modules()
    t, blocks = 90, 3
    g = torch.Generator(device="cuda").manual_seed(4)
    k = torch.randn(t, 1, HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
    v = (torch.randn(t, 1, HEAD_DIM, device="cuda", generator=g) * 3).to(torch.bfloat16)
    k[3, 0, :32] = 0
    v[5, 0, 16:32] = 1e-7
    # consecutive slots (a verify's rows share octets) and scattered ones
    slots = torch.cat(
        [
            torch.arange(130, 130 + 40, device="cuda"),
            torch.randperm(128, device="cuda", generator=g)[: t - 41],
            torch.tensor([-1], device="cuda"),
        ]
    ).to(torch.int64)
    assert slots.numel() == t
    want = torch.zeros(blocks, 2, 128, 72, device="cuda", dtype=torch.uint8)
    ref.insert_nvfp4_kv(k, v, want, slots, k_scale, v_scale)
    got = torch.zeros(blocks, 2 * 9216, device="cuda", dtype=torch.uint8)
    gs = torch.tensor([k_scale, v_scale], device="cuda", dtype=torch.float32)
    _kv_store_launcher(t)(
        k.data_ptr(), v.data_ptr(), slots.data_ptr(), got.data_ptr(), gs.data_ptr(), t
    )
    torch.cuda.synchronize()
    assert int((want != 0).sum()) > 1000
    bad = (got != want.view(blocks, -1)).nonzero()
    assert bad.numel() == 0, f"{bad.shape[0]} bytes differ, first {bad[:6].tolist()}"
