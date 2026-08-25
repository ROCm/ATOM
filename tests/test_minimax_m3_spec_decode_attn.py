# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Spec-decode (q>1) ATTENTION OUTPUT parity for MiniMax-M3 sparse.

The topk parity test covers block selection; this covers the full attention
numerics. Spec-verify routes q = num_spec+1 tokens per request through the DECODE
path. We compare, for the SAME query vectors and KV cache:

  decode-path output  (minimax_m3_index_topk_decode + decode attn, max_query_len=Q)
  vs
  prefill-path output (minimax_m3_index_topk + prefill attn, the proven reference)

laid out request-major so row pid_t = req*Q + tok. They must agree within bf16
noise for every query token. A divergence at a specific `tok` localizes a
per-token-causal bug in the decode path.
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
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
)
from atom.model_ops.minimax_m3.sparse_attn import (  # noqa: E402
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA/ROCm")

HEAD_DIM = 128
BLOCK = SPARSE_BLOCK_SIZE
O_ATOL, O_RTOL = 3e-2, 5e-2


@pytest.mark.parametrize(
    "seq_lens,max_q",
    [
        ([300, 512], 4),
        ([900], 4),
        ([256, 384, 130], 2),
    ],
)
def test_spec_decode_attn_matches_prefill(seq_lens, max_q):
    torch.manual_seed(0)
    dev = "cuda"
    num_heads, num_kv_heads = 16, 1
    topk = 16
    batch = len(seq_lens)
    total_q = batch * max_q
    sm = HEAD_DIM**-0.5

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    prefix_lens = [s - max_q for s in seq_lens]
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=dev)
    cu = torch.arange(0, (batch + 1) * max_q, max_q, dtype=torch.int32, device=dev)
    max_seq = max(seq_lens)
    max_block = (max_seq + BLOCK - 1) // BLOCK
    num_logical = batch * max_block + 4

    # Shared plain page-128 KV cache [num_logical, 2, 128, nkv, hd].
    kv_plain = torch.zeros(
        num_logical, 2, BLOCK, num_kv_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev
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

    # Shared queries: one row per query token (request-major), plus the index
    # queries that drive block selection.
    q = torch.randn(total_q, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    index_q = torch.randn(
        total_q, num_kv_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    # index cache = the K side of the plain cache, flattened to [num_logical*128, hd]
    index_cache = (
        kv_plain[:, 0, :, 0].reshape(num_logical * BLOCK, HEAD_DIM).contiguous()
    )

    # ---- DECODE path ----
    tk_d = minimax_m3_index_topk_decode(
        index_q,
        index_cache.view(num_logical, BLOCK, HEAD_DIM),
        block_table,
        seq_lens_t,
        max_seq,
        topk,
        0,
        1,
        num_kv_heads,
        sm,
        emit_sparse_block_table=False,
        max_query_len=max_q,
    )
    out_d = torch.empty(total_q, num_heads, HEAD_DIM, dtype=q.dtype, device=dev)
    minimax_m3_sparse_attn_decode(
        q,
        kv_plain,
        tk_d,
        block_table,
        seq_lens_t,
        num_kv_heads,
        sm,
        out_d,
        max_query_len=max_q,
    )

    # ---- PREFILL path (reference) ----
    tk_p = minimax_m3_index_topk(
        index_q,
        index_cache.view(num_logical, BLOCK, HEAD_DIM),
        block_table,
        cu,
        seq_lens_t,
        prefix_lens_t,
        max_q,
        max_seq,
        topk,
        0,
        1,
        num_kv_heads,
        sm,
        emit_sparse_block_table=False,
    )
    out_p = torch.empty(total_q, num_heads, HEAD_DIM, dtype=q.dtype, device=dev)
    minimax_m3_sparse_attn(
        q,
        kv_plain,
        tk_p,
        block_table,
        cu,
        seq_lens_t,
        prefix_lens_t,
        max_q,
        num_kv_heads,
        sm,
        out_p,
    )

    # Per-token report so a divergence localizes which `tok` is wrong.
    for b in range(batch):
        for tok in range(max_q):
            r = b * max_q + tok
            torch.testing.assert_close(
                out_d[r],
                out_p[r],
                atol=O_ATOL,
                rtol=O_RTOL,
                msg=f"mismatch req={b} tok={tok} (causal_len={seq_lens[b]-max_q+tok+1})",
            )
