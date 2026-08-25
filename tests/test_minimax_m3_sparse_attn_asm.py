# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A/B correctness test for the MiniMax-M3 ASM paged-attention decode path.

Compares ``minimax_m3_sparse_attn_decode_asm`` (AITER ASM paged-attention over a
SHUFFLE-layout KV cache) against the reference Triton split-K decode kernel
``minimax_m3_sparse_attn_decode`` (over the plain
``[num_blocks, 2, 128, num_kv_heads, head_dim]`` layout).

Both consume the SAME selected blocks (``topk_idx``) and the SAME underlying K/V
data, written into their respective cache layouts. The two kernels use different
reduction orders and softmax bases, so bit-exactness is not expected; agreement
should be within bf16 noise.

The ASM path requires per-rank ``num_kv_heads == 1`` and ``head_dim == 128``, so
the test fixes those. ``page_size == sparse_block_size == 128``.

Run (GPU required):
    source /opt/venv/bin/activate
    python -m pytest tests/test_minimax_m3_sparse_attn_asm.py -v
"""

from __future__ import annotations

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()
if _HAS_CUDA:
    try:
        import aiter  # noqa: F401

        _HAS_AITER = True
    except ImportError:
        _HAS_AITER = False
else:
    _HAS_AITER = False


# The repo conftest stubs atom.config for CPU scheduler tests; evict so the real
# kernels import (mirrors tests/test_chunk_gated_delta_rule_fused.py).
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.minimax_m3.sparse_attn import (  # noqa: E402
    PAGES_PER_SPARSE_BLOCK,
    SPARSE_BLOCK_SIZE,
    minimax_m3_build_sparse_block_table,
    minimax_m3_sparse_attn_decode,
    minimax_m3_sparse_attn_decode_asm,
)

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="No GPU available"),
    pytest.mark.skipif(not _HAS_AITER, reason="aiter not importable"),
]

# bf16 decode output: the ASM kernel and the Triton split-K kernel differ in
# reduction order and softmax base (base-2 vs the ASM kernel's internal).
# Agreement should be within bf16 noise for unit-magnitude outputs.
O_ATOL, O_RTOL = 3e-2, 5e-2

HEAD_DIM = 128
NUM_KV_HEADS = 1
BLOCK = SPARSE_BLOCK_SIZE  # 128


ASM_PAGE = 16
PAGES_PER_BLOCK = BLOCK // ASM_PAGE  # 8


def _shuffle_k_p16(k_pages: torch.Tensor) -> torch.Tensor:
    # k_pages: [num_p16, num_kv_heads, 16, head_dim]
    # -> SHUFFLE K [num_p16, num_kv_heads, head_dim//x, 16, x]
    nb, nkv, ps, hd = k_pages.shape
    x = 16 // k_pages.element_size()
    return k_pages.view(nb, nkv, ps, hd // x, x).permute(0, 1, 3, 2, 4).contiguous()


def _shuffle_v_p16(v_pages: torch.Tensor) -> torch.Tensor:
    # v_pages: [num_p16, num_kv_heads, 16, head_dim]
    # -> SHUFFLE V [num_p16, num_kv_heads, 16//x, head_dim, x]
    nb, nkv, ps, hd = v_pages.shape
    x = 16 // v_pages.element_size()
    return v_pages.view(nb, nkv, ps // x, x, hd).permute(0, 1, 2, 4, 3).contiguous()


def _build_inputs(batch, num_heads, seq_lens, topk, device, dtype=torch.bfloat16):
    """Construct a shared paged KV cache + per-request topk selection.

    The reference (Triton split-K) reads a PAGE-128 plain cache. The ASM path
    reads a PAGE-16 SHUFFLE cache. Both are filled from the SAME random K/V data,
    with the invariant that logical 128-block ``L`` occupies physical 16-pages
    ``L*8 .. L*8+7`` (matching block_convert: physical = logical*ratio + j).

    Returns (kv_plain[p128], k_shuf16, v_shuf16, q, topk_idx, block_table[logical]).
    """
    torch.manual_seed(0)
    max_seq = int(max(seq_lens))
    max_blocks = (max_seq + BLOCK - 1) // BLOCK
    # logical 128-pages; one private logical range per request
    num_logical = batch * max_blocks + 4

    # plain page-128 cache for the reference: [num_logical, 2, 128, num_kv_heads, hd]
    kv_plain = torch.zeros(
        num_logical, 2, BLOCK, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device
    )
    # logical block_table (128-granularity), distinct logical pages per request
    perm = torch.randperm(num_logical, device=device)[: batch * max_blocks]
    block_table = perm.view(batch, max_blocks).to(torch.int32)

    for b in range(batch):
        sl = int(seq_lens[b])
        nb = (sl + BLOCK - 1) // BLOCK
        for j in range(nb):
            page = int(block_table[b, j])
            valid = min(BLOCK, sl - j * BLOCK)
            kv_plain[page, 0, :valid, 0] = torch.randn(
                valid, HEAD_DIM, dtype=dtype, device=device
            )
            kv_plain[page, 1, :valid, 0] = torch.randn(
                valid, HEAD_DIM, dtype=dtype, device=device
            )

    # page-16 SHUFFLE cache from the SAME data. Physical 16-page P holds the
    # tokens of logical block P//8 at intra-block offset (P%8)*16 .. +16.
    num_p16 = num_logical * PAGES_PER_BLOCK
    k_pages = torch.zeros(
        num_p16, NUM_KV_HEADS, ASM_PAGE, HEAD_DIM, dtype=dtype, device=device
    )
    v_pages = torch.zeros(
        num_p16, NUM_KV_HEADS, ASM_PAGE, HEAD_DIM, dtype=dtype, device=device
    )
    # kv_plain[:,0] is [num_logical, 128, nkv, hd]; reshape 128 -> (8, 16)
    k_src = kv_plain[:, 0].permute(0, 2, 1, 3)  # [num_logical, nkv, 128, hd]
    v_src = kv_plain[:, 1].permute(0, 2, 1, 3)
    k_pages.copy_(
        k_src.reshape(num_logical, NUM_KV_HEADS, PAGES_PER_BLOCK, ASM_PAGE, HEAD_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_p16, NUM_KV_HEADS, ASM_PAGE, HEAD_DIM)
    )
    v_pages.copy_(
        v_src.reshape(num_logical, NUM_KV_HEADS, PAGES_PER_BLOCK, ASM_PAGE, HEAD_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_p16, NUM_KV_HEADS, ASM_PAGE, HEAD_DIM)
    )
    k_cache_shuf = _shuffle_k_p16(k_pages)
    v_cache_shuf = _shuffle_v_p16(v_pages)

    q = torch.randn(batch, num_heads, HEAD_DIM, dtype=dtype, device=device)

    # topk selection per request: choose a random subset of the request's valid
    # blocks, ALWAYS including the tail block (the one containing seq_len-1),
    # right-padded with -1. Shape [num_kv_heads(==1), batch, topk].
    topk_idx = torch.full(
        (NUM_KV_HEADS, batch, topk), -1, dtype=torch.int32, device=device
    )
    for b in range(batch):
        sl = int(seq_lens[b])
        nb = (sl + BLOCK - 1) // BLOCK
        tail = nb - 1
        pool = list(range(nb))
        k = min(topk, nb)
        # ensure tail included
        sel = set([tail])
        i = 0
        while len(sel) < k and i < len(pool):
            sel.add(pool[i])
            i += 1
        sel = sorted(sel)
        topk_idx[0, b, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=device)

    return kv_plain, k_cache_shuf, v_cache_shuf, q, topk_idx, block_table


# Realistic per-rank head counts with num_kv_heads == 1: gqa ratio 8 (tp8) or
# 16 (tp4). ASM pa_fwd_asm only ships bf16 kernels for Gqa in {8, 16}.
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [200],  # single req, 2 blocks (tail partial)
        [128],  # exactly one full block (tail full)
        [300, 130, 512],  # varlen batch, mixed tails
    ],
)
def test_asm_matches_triton_decode(num_heads, seq_lens):
    device = "cuda"
    batch = len(seq_lens)
    topk = 16
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    sm_scale = HEAD_DIM**-0.5

    kv_plain, k_shuf, v_shuf, q, topk_idx, block_table = _build_inputs(
        batch, num_heads, seq_lens_t, topk, device
    )

    # reference: Triton split-K decode over the plain cache
    ref_out = torch.empty(batch, num_heads, HEAD_DIM, dtype=q.dtype, device=device)
    minimax_m3_sparse_attn_decode(
        q,
        kv_plain,
        topk_idx,
        block_table,
        seq_lens_t,
        NUM_KV_HEADS,
        sm_scale,
        ref_out,
    )

    # candidate: ASM paged-attention over the page-16 SHUFFLE cache
    asm_out = torch.empty(batch, num_heads, HEAD_DIM, dtype=q.dtype, device=device)
    minimax_m3_sparse_attn_decode_asm(
        q,
        k_shuf,
        v_shuf,
        topk_idx,
        block_table,
        seq_lens_t,
        NUM_KV_HEADS,
        sm_scale,
        asm_out,
        k_scale=None,
        v_scale=None,
    )

    torch.testing.assert_close(asm_out, ref_out, atol=O_ATOL, rtol=O_RTOL)


def test_build_sparse_block_table_packs_tail_last():
    """Compaction to 16-pages: full blocks first, tail block last; ctx exact.

    Each selected logical 128-block expands to 8 physical 16-pages
    (logical_id*8 + j). Full blocks packed first (selection order), tail last.
    """
    device = "cuda"
    seq_len = 300  # logical blocks 0,1 full; block 2 is tail with 300-256=44 tokens
    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=device)
    # logical 128-page ids 10, 11, 12 for logical blocks 0, 1, 2
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32, device=device)
    # select blocks 2 (tail), 0, 1 in scrambled order
    topk = 4
    topk_idx = torch.full((1, 1, topk), -1, dtype=torch.int32, device=device)
    topk_idx[0, 0, 0] = 2  # tail first in selection order
    topk_idx[0, 0, 1] = 0
    topk_idx[0, 0, 2] = 1

    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table(
        topk_idx, block_table, seq_lens_t
    )

    ppb = PAGES_PER_SPARSE_BLOCK  # 8
    # slot 0 = logical block 0 (page 10) -> physical 80..87
    assert sparse_bt[0, 0:ppb].tolist() == [10 * ppb + j for j in range(ppb)]
    # slot 1 = logical block 1 (page 11) -> physical 88..95
    assert sparse_bt[0, ppb : 2 * ppb].tolist() == [11 * ppb + j for j in range(ppb)]
    # slot 2 (tail, packed last) = logical block 2 (page 12) -> physical 96..103
    assert sparse_bt[0, 2 * ppb : 3 * ppb].tolist() == [
        12 * ppb + j for j in range(ppb)
    ]
    # context_lens = 2 full blocks * 128 + tail remainder (300 - 256 = 44)
    assert sparse_ctx[0].item() == 2 * BLOCK + (seq_len - 2 * BLOCK)
