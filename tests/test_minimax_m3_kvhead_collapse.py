# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""kv-head collapse: ASM/gluon sparse decode for num_kv_heads > 1 == Triton GQA.

The ASM/gluon sparse paged-attention path folds each (token, kv_head) pair into
its own row so the kernel runs with num_kv_heads_view == 1. The topk emit encodes
the kv-head into the sparse_bt page ids (page = phys16*Hkv + kvh) matching the
collapsed cache view [num_phys16*Hkv, 1, ...]. This test verifies the collapsed
ASM path matches the Triton split-K decode kernel (which handles GQA Hkv>1
natively) on the SAME KV data + selection, per (token, head), for Hkv=4.
"""

from __future__ import annotations

import pytest
import torch


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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/ROCm"
)

HEAD_DIM = 128
BLOCK = SPARSE_BLOCK_SIZE
ASM_PAGE = 16
PPB = BLOCK // ASM_PAGE  # 8


def _shuffle_k_p16(k_pages):  # [num_p16, Hkv, 16, hd] -> [num_p16, Hkv, hd//x, 16, x]
    nb, nkv, ps, hd = k_pages.shape
    x = 16 // k_pages.element_size()
    return k_pages.view(nb, nkv, ps, hd // x, x).permute(0, 1, 3, 2, 4).contiguous()


def _shuffle_v_p16(v_pages):  # [num_p16, Hkv, 16, hd] -> [num_p16, Hkv, 16//x, hd, x]
    nb, nkv, ps, hd = v_pages.shape
    x = 16 // v_pages.element_size()
    return v_pages.view(nb, nkv, ps // x, x, hd).permute(0, 1, 2, 4, 3).contiguous()


@pytest.mark.parametrize(
    "seq_lens,max_q,Hkv,Hq",
    [
        ([300, 512], 1, 4, 32),  # plain decode, GQA group 8
        ([900], 1, 4, 32),
        ([256, 384, 130], 2, 4, 32),  # spec decode q>1
        ([512], 1, 2, 16),  # Hkv=2
    ],
)
def test_kvhead_collapse_asm_matches_triton(seq_lens, max_q, Hkv, Hq):
    torch.manual_seed(0)
    dev = "cuda"
    g = Hq // Hkv
    topk = 8
    batch = len(seq_lens)
    T = batch * max_q
    sm = HEAD_DIM**-0.5
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)

    max_seq = max(seq_lens)
    max_block = (max_seq + BLOCK - 1) // BLOCK
    num_logical = batch * max_block + 4

    # plain page-128 cache [num_logical, 2, 128, Hkv, hd] + logical block table
    kv_plain = torch.zeros(
        num_logical, 2, BLOCK, Hkv, HEAD_DIM, dtype=torch.bfloat16, device=dev
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
            kv_plain[page, 0, :valid] = torch.randn(
                valid, Hkv, HEAD_DIM, dtype=torch.bfloat16, device=dev
            )
            kv_plain[page, 1, :valid] = torch.randn(
                valid, Hkv, HEAD_DIM, dtype=torch.bfloat16, device=dev
            )

    # page-16 SHUFFLE cache from the SAME data: logical L -> phys16 L*8+j, per kv head
    num_p16 = num_logical * PPB
    k_pages = torch.zeros(
        num_p16, Hkv, ASM_PAGE, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    v_pages = torch.zeros(
        num_p16, Hkv, ASM_PAGE, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    # kv_plain[:,0] is [num_logical, 128, Hkv, hd] -> [num_logical, Hkv, 8, 16, hd]
    k_src = kv_plain[:, 0].permute(0, 2, 1, 3)  # [num_logical, Hkv, 128, hd]
    v_src = kv_plain[:, 1].permute(0, 2, 1, 3)
    k_pages.copy_(
        k_src.reshape(num_logical, Hkv, PPB, ASM_PAGE, HEAD_DIM)
        .permute(0, 2, 1, 3, 4)  # [num_logical, 8, Hkv, 16, hd]
        .reshape(num_p16, Hkv, ASM_PAGE, HEAD_DIM)
    )
    v_pages.copy_(
        v_src.reshape(num_logical, Hkv, PPB, ASM_PAGE, HEAD_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_p16, Hkv, ASM_PAGE, HEAD_DIM)
    )
    k_shuf = _shuffle_k_p16(k_pages)
    v_shuf = _shuffle_v_p16(v_pages)

    q = torch.randn(T, Hq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    index_q = torch.randn(T, Hkv, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    # index cache = K side flattened to [num_logical, 128, hd] using kv head 0's K
    # (the indexer has its own single-head index cache in the model; here any
    # consistent K suffices since both paths share the SAME topk_idx).
    index_cache = (
        kv_plain[:, 0, :, 0].reshape(num_logical, BLOCK, HEAD_DIM).contiguous()
    )

    # Per-(kv-head) topk selection -> fused kv-head-encoded sparse_bt/ctx.
    tk, sbt, sctx = minimax_m3_index_topk_decode(
        index_q,
        index_cache,
        block_table,
        seq_lens_t,
        max_seq,
        topk,
        0,
        1,
        Hkv,
        sm,
        emit_sparse_block_table=True,
        max_query_len=max_q,
    )

    # Reference: Triton split-K decode (native GQA, plain page-128 cache).
    out_ref = torch.empty(T, Hq, HEAD_DIM, dtype=q.dtype, device=dev)
    minimax_m3_sparse_attn_decode(
        q,
        kv_plain,
        tk,
        block_table,
        seq_lens_t,
        Hkv,
        sm,
        out_ref,
        max_query_len=max_q,
    )

    # Candidate: collapsed ASM/gluon decode over the SHUFFLE cache (Hkv>1).
    out_asm = torch.empty(T, Hq, HEAD_DIM, dtype=q.dtype, device=dev)
    minimax_m3_sparse_attn_decode_asm(
        q,
        k_shuf,
        v_shuf,
        tk,
        block_table,
        seq_lens_t,
        Hkv,
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
            cos = torch.nn.functional.cosine_similarity(
                out_asm[r].flatten().double(), out_ref[r].flatten().double(), dim=0
            ).item()
            assert (
                cos > 0.999
            ), f"Hkv={Hkv} req={b} tok={tok} collapsed-ASM vs Triton cos={cos:.5f}"
