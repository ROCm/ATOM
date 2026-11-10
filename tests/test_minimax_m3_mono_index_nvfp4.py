# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono over an NVFP4 index cache (gfx950, single-buffer 9216 B pages): the block
scores of ``build_index_score_kernel(nvfp4=True)`` against aiter's
``score_flydsl_nvfp4`` (``k_round="exact"``) on the same pages, bit for bit, and
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
    want = scorer.score_flydsl_nvfp4(
        iq, kd, ks, block_table, seq, q_len, 1, SM_SCALE, n_blk, k_round="exact"
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
