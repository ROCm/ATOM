# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A/B correctness test for the MiniMax-M3 page-16 SHUFFLE PREFILL attention.

Compares ``minimax_m3_sparse_attn_shuffle`` (block-sparse GQA prefill over a
page-16 SHUFFLE KV cache, split K/V tensors) against the reference plain-layout
kernel ``minimax_m3_sparse_attn`` (over the combined
``[num_blocks, 2, 128, num_kv_heads, head_dim]`` cache).

Both consume the SAME selected blocks (``topk_idx``), the SAME q, and the SAME
underlying K/V data written into their respective cache layouts, with the
invariant that logical 128-block ``L`` occupies physical 16-pages
``L*8 .. L*8+7`` (physical = logical*PAGES_PER_SPARSE_BLOCK + j).

The math is identical; only the K/V load addressing differs, so agreement should
be within bf16 noise.

Run (GPU required):
    source /opt/venv/bin/activate
    python -m pytest tests/test_minimax_m3_sparse_prefill_shuffle.py -v
"""

from __future__ import annotations

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()


# The repo conftest stubs atom.config for CPU scheduler tests; evict so the real
# kernels import (mirrors tests/test_minimax_m3_sparse_attn_asm.py).
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.minimax_m3.sparse_attn import (  # noqa: E402
    PAGES_PER_SPARSE_BLOCK,
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_shuffle,
)

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="No GPU available"),
]

# bf16 prefill output: the two kernels are math-identical and differ only in K/V
# load addressing, so agreement should be within bf16 noise.
O_ATOL, O_RTOL = 3e-2, 5e-2

HEAD_DIM = 128
BLOCK = SPARSE_BLOCK_SIZE  # 128
ASM_PAGE = 16
PAGES_PER_BLOCK = PAGES_PER_SPARSE_BLOCK  # 8


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


def _build_inputs(
    batch,
    num_heads,
    num_kv_heads,
    seq_lens,
    prefix_lens,
    topk,
    device,
    dtype=torch.bfloat16,
):
    """Construct a shared paged KV cache + per-(kv-head, query-token) selection.

    The reference reads a PAGE-128 plain combined cache. The shuffle path reads a
    PAGE-16 SHUFFLE cache (split K/V). Both are filled from the SAME random K/V
    data, with the invariant logical 128-block ``L`` -> physical 16-pages
    ``L*8 .. L*8+7``.

    ``seq_lens`` is the full KV length per request; ``prefix_lens`` is the number
    of already-cached tokens (chunked prefill). The number of NEW query tokens
    per request is ``seq_lens - prefix_lens``.

    Returns (kv_plain[p128], k_shuf16, v_shuf16, q, topk_idx,
             block_table[logical], cu_seqlens_q).
    """
    torch.manual_seed(0)
    max_seq = int(max(seq_lens))
    max_blocks = (max_seq + BLOCK - 1) // BLOCK
    # logical 128-pages; one private logical range per request, plus slack.
    num_logical = batch * max_blocks + 4

    # plain page-128 combined cache: [num_logical, 2, 128, num_kv_heads, hd]
    kv_plain = torch.zeros(
        num_logical, 2, BLOCK, num_kv_heads, HEAD_DIM, dtype=dtype, device=device
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
            kv_plain[page, 0, :valid] = torch.randn(
                valid, num_kv_heads, HEAD_DIM, dtype=dtype, device=device
            )
            kv_plain[page, 1, :valid] = torch.randn(
                valid, num_kv_heads, HEAD_DIM, dtype=dtype, device=device
            )

    # page-16 SHUFFLE cache from the SAME data. Physical 16-page P holds the
    # tokens of logical block P//8 at intra-block offset (P%8)*16 .. +16.
    num_p16 = num_logical * PAGES_PER_BLOCK
    # kv_plain[:,0] is [num_logical, 128, nkv, hd]; reshape 128 -> (8, 16).
    k_src = kv_plain[:, 0].permute(0, 2, 1, 3)  # [num_logical, nkv, 128, hd]
    v_src = kv_plain[:, 1].permute(0, 2, 1, 3)
    k_pages = (
        k_src.reshape(num_logical, num_kv_heads, PAGES_PER_BLOCK, ASM_PAGE, HEAD_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_p16, num_kv_heads, ASM_PAGE, HEAD_DIM)
        .contiguous()
    )
    v_pages = (
        v_src.reshape(num_logical, num_kv_heads, PAGES_PER_BLOCK, ASM_PAGE, HEAD_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_p16, num_kv_heads, ASM_PAGE, HEAD_DIM)
        .contiguous()
    )
    k_cache_shuf = _shuffle_k_p16(k_pages)
    v_cache_shuf = _shuffle_v_p16(v_pages)

    # Query: only the NEW tokens (seq_len - prefix_len) per request, varlen.
    q_lens = [int(seq_lens[b]) - int(prefix_lens[b]) for b in range(batch)]
    total_q = sum(q_lens)
    q = torch.randn(total_q, num_heads, HEAD_DIM, dtype=dtype, device=device)
    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.tensor(q_lens, dtype=torch.int32, device=device).cumsum(0)

    # topk selection per (kv-head, query-token): choose a subset of the valid
    # blocks for the request, right-padded with -1. The kernel applies the
    # causal mask, so selecting all blocks is fine. Shape [num_kv_heads, total_q,
    # topk].
    topk_idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=device
    )
    g = torch.Generator(device="cpu").manual_seed(1234)
    for b in range(batch):
        sl = int(seq_lens[b])
        nb = (sl + BLOCK - 1) // BLOCK
        qs = int(cu_seqlens_q[b])
        qe = int(cu_seqlens_q[b + 1])
        for h in range(num_kv_heads):
            for n in range(qs, qe):
                k = min(topk, nb)
                # random subset of the nb valid blocks for this request
                sel = torch.randperm(nb, generator=g)[:k].sort().values
                topk_idx[h, n, : len(sel)] = sel.to(torch.int32).to(device)

    return kv_plain, k_cache_shuf, v_cache_shuf, q, topk_idx, block_table, cu_seqlens_q


# gqa 16 (num_kv_heads=1) and gqa 8 (num_kv_heads=2), num_heads=16, head_dim=128.
@pytest.mark.parametrize("num_heads,num_kv_heads", [(16, 1), (16, 2)])
@pytest.mark.parametrize(
    "seq_lens,prefix_lens",
    [
        # pure prefill: prefix_lens all 0, varlen batch with mixed tails
        ([200, 130, 300], [0, 0, 0]),
        # single request, pure prefill
        ([300], [0]),
        # chunked prefill: nonzero prefix_lens (already-cached prefix tokens)
        ([200, 130, 300], [64, 128, 200]),
    ],
)
def test_shuffle_prefill_matches_plain(num_heads, num_kv_heads, seq_lens, prefix_lens):
    device = "cuda"
    batch = len(seq_lens)
    topk = 8
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    sm_scale = HEAD_DIM**-0.5

    (
        kv_plain,
        k_shuf,
        v_shuf,
        q,
        topk_idx,
        block_table,
        cu_seqlens_q,
    ) = _build_inputs(
        batch, num_heads, num_kv_heads, seq_lens_t, prefix_lens_t, topk, device
    )
    total_q = q.shape[0]
    max_query_len = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())

    # reference: plain page-128 combined cache
    ref_out = torch.empty(total_q, num_heads, HEAD_DIM, dtype=q.dtype, device=device)
    minimax_m3_sparse_attn(
        q,
        kv_plain,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens_t,
        prefix_lens_t,
        max_query_len,
        num_kv_heads,
        sm_scale,
        ref_out,
    )

    # candidate: page-16 SHUFFLE cache (split K/V)
    shuf_out = torch.empty(total_q, num_heads, HEAD_DIM, dtype=q.dtype, device=device)
    minimax_m3_sparse_attn_shuffle(
        q,
        k_shuf,
        v_shuf,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens_t,
        prefix_lens_t,
        max_query_len,
        num_kv_heads,
        sm_scale,
        shuf_out,
    )

    torch.testing.assert_close(shuf_out, ref_out, atol=O_ATOL, rtol=O_RTOL)
