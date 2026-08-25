# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Spec-decode (q>1) ASM/gluon vs Triton decode parity for MiniMax-M3.

The real tp=4 + ASM_PA=1 config runs ``minimax_m3_sparse_attn_decode_asm`` (gluon
split-KV over the page-16 SHUFFLE cache). Earlier parity tests verified the Triton
decode kernel for q>1; this checks the ASM/gluon path consumes the per-token
``sparse_bt`` / ``sparse_ctx`` (one row per query token) correctly for q>1, by A/B
against the (verified) Triton decode kernel on the SAME KV data + same topk.

Spec layout: each request has ``max_q`` query tokens at the end of its sequence,
laid out request-major (row pid_t = req*max_q + tok).
"""

from __future__ import annotations

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()


def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.minimax_m3.index_topk import (  # noqa: E402
    minimax_m3_index_topk_decode,
)
from atom.model_ops.minimax_m3.sparse_attn import (  # noqa: E402
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
    minimax_m3_sparse_attn_decode_asm,
)

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA/ROCm")

HEAD_DIM = 128
BLOCK = SPARSE_BLOCK_SIZE
ASM_PAGE = 16
PPB = BLOCK // ASM_PAGE  # 8
NKV = 1
O_ATOL, O_RTOL = 3e-2, 5e-2


def _shuffle_k_p16(k_pages):
    nb, nkv, ps, hd = k_pages.shape
    x = 16 // k_pages.element_size()
    return k_pages.view(nb, nkv, ps, hd // x, x).permute(0, 1, 3, 2, 4).contiguous()


def _shuffle_v_p16(v_pages):
    nb, nkv, ps, hd = v_pages.shape
    x = 16 // v_pages.element_size()
    return v_pages.view(nb, nkv, ps // x, x, hd).permute(0, 1, 2, 4, 3).contiguous()


@pytest.mark.parametrize(
    "seq_lens,max_q,num_heads",
    [
        ([300, 512], 4, 16),
        ([900], 4, 16),
        ([256, 384, 130], 2, 16),
    ],
)
def test_spec_decode_asm_matches_triton(seq_lens, max_q, num_heads):
    torch.manual_seed(0)
    dev = "cuda"
    topk = 16
    batch = len(seq_lens)
    total_q = batch * max_q
    sm = HEAD_DIM**-0.5
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)

    max_seq = max(seq_lens)
    max_block = (max_seq + BLOCK - 1) // BLOCK
    num_logical = batch * max_block + 4

    # plain page-128 cache + logical block table
    kv_plain = torch.zeros(
        num_logical, 2, BLOCK, NKV, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    block_table = (
        torch.randperm(num_logical, device=dev)[: batch * max_block]
        .view(batch, max_block)
        .to(torch.int32)
    )
    for b in range(batch):
        nb = (seq_lens[b] + BLOCK - 1) // BLOCK
        for j in range(nb):
            page = int(block_table[b, j])
            valid = min(BLOCK, seq_lens[b] - j * BLOCK)
            kv_plain[page, 0, :valid, 0] = torch.randn(
                valid, HEAD_DIM, dtype=torch.bfloat16, device=dev
            )
            kv_plain[page, 1, :valid, 0] = torch.randn(
                valid, HEAD_DIM, dtype=torch.bfloat16, device=dev
            )

    # page-16 SHUFFLE cache from the SAME data (logical L -> physical L*8+j)
    num_p16 = num_logical * PPB
    k_pages = torch.zeros(
        num_p16, NKV, ASM_PAGE, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    v_pages = torch.zeros(
        num_p16, NKV, ASM_PAGE, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    k_src = kv_plain[:, 0].permute(0, 2, 1, 3)  # [num_logical, nkv, 128, hd]
    v_src = kv_plain[:, 1].permute(0, 2, 1, 3)
    k_pages.copy_(
        k_src.reshape(num_logical, NKV, PPB, ASM_PAGE, HEAD_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_p16, NKV, ASM_PAGE, HEAD_DIM)
    )
    v_pages.copy_(
        v_src.reshape(num_logical, NKV, PPB, ASM_PAGE, HEAD_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_p16, NKV, ASM_PAGE, HEAD_DIM)
    )
    k_shuf = _shuffle_k_p16(k_pages)
    v_shuf = _shuffle_v_p16(v_pages)

    q = torch.randn(total_q, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    index_q = torch.randn(total_q, NKV, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    index_cache = (
        kv_plain[:, 0, :, 0].reshape(num_logical, BLOCK, HEAD_DIM).contiguous()
    )

    # Per-token topk + fused sparse_bt/ctx (the inputs the gluon path consumes).
    tk, sbt, sctx = minimax_m3_index_topk_decode(
        index_q,
        index_cache,
        block_table,
        seq_lens_t,
        max_seq,
        topk,
        0,
        1,
        NKV,
        sm,
        emit_sparse_block_table=True,
        max_query_len=max_q,
    )

    # Reference: Triton split-K decode (verified for q>1) over the plain cache.
    out_triton = torch.empty(total_q, num_heads, HEAD_DIM, dtype=q.dtype, device=dev)
    minimax_m3_sparse_attn_decode(
        q,
        kv_plain,
        tk,
        block_table,
        seq_lens_t,
        NKV,
        sm,
        out_triton,
        max_query_len=max_q,
    )

    # Candidate: ASM/gluon decode over the SHUFFLE cache, fed the per-token sbt/ctx.
    out_asm = torch.empty(total_q, num_heads, HEAD_DIM, dtype=q.dtype, device=dev)
    minimax_m3_sparse_attn_decode_asm(
        q,
        k_shuf,
        v_shuf,
        tk,
        block_table,
        seq_lens_t,
        NKV,
        sm,
        out_asm,
        k_scale=None,
        v_scale=None,
        sparse_bt=sbt,
        sparse_ctx=sctx,
    )

    for b in range(batch):
        for tok in range(max_q):
            r = b * max_q + tok
            torch.testing.assert_close(
                out_asm[r],
                out_triton[r],
                atol=O_ATOL,
                rtol=O_RTOL,
                msg=f"ASM vs Triton mismatch req={b} tok={tok} "
                f"(causal_len={seq_lens[b]-max_q+tok+1})",
            )
