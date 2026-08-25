# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Spec-decode (q>1) DECODE path == PREFILL path parity for MiniMax-M3 sparse.

Spec-verify (EAGLE, q = num_spec+1) is now routed through the DECODE sparse
indexer instead of the prefill one. The decode kernels gained a per-query-token
dimension with per-token causal length `causal_len = seq_len - max_q + tok + 1`,
which is the SAME math the prefill kernels already implement. This test feeds
identical spec-shaped inputs to both paths and asserts the selected top-k blocks
and the fused sparse block-table / context-lengths are byte-identical.

Spec shape: every request has exactly `max_q` query tokens, positioned at the end
of its sequence (prefix_len = seq_len - max_q). That is precisely how
prepare_decode lays out a spec-verify batch.
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
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/ROCm"
)

HEAD_DIM = 128


@pytest.mark.parametrize(
    "seq_lens,max_q",
    [
        ([300, 128, 900, 257], 4),  # num_spec=3 -> max_q=4
        ([129], 4),
        ([256, 384], 2),
        ([512, 130, 1000], 4),
        ([128], 1),  # degenerate: max_q==1 must equal plain decode
    ],
)
def test_spec_decode_topk_matches_prefill(seq_lens, max_q):
    torch.manual_seed(0)
    dev = "cuda"
    nidx = 1
    topk = 16
    batch = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    # Spec layout: max_q query tokens per request, prefix = seq_len - max_q.
    prefix_lens = [s - max_q for s in seq_lens]
    assert all(p >= 0 for p in prefix_lens), "seq_len must be >= max_q"
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=dev)

    total_q = batch * max_q
    # cu_seqlens_q for the PREFILL reference: uniform max_q tokens per request.
    cu = torch.arange(0, (batch + 1) * max_q, max_q, dtype=torch.int32, device=dev)

    max_seq = max(seq_lens)
    max_block = (max_seq + 127) // 128
    nblk_total = batch * max_block + 4
    sm = HEAD_DIM**-0.5

    # Shared inputs. idx_q has one row per query token (total_q rows), laid out
    # request-major: [req0 tok0..tok{max_q-1}, req1 tok0.., ...] — the same order
    # both paths consume (decode: pid_t//max_q==req; prefill: cu_seqlens_q segments).
    idx_q = torch.randn(total_q, nidx, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    idxc = torch.randn(nblk_total, 128, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    bt = (
        torch.randperm(nblk_total, device=dev)[: batch * max_block]
        .view(batch, max_block)
        .to(torch.int32)
    )

    # DECODE path (the new q>1 route): seq_lens is per-request total length.
    tk_d, sbt_d, sctx_d = minimax_m3_index_topk_decode(
        idx_q,
        idxc,
        bt,
        seq_lens_t,
        max_seq,
        topk,
        0,  # init_blocks
        1,  # local_blocks
        nidx,
        sm,
        emit_sparse_block_table=True,
        max_query_len=max_q,
    )

    # PREFILL path (the proven per-token causal reference).
    tk_p, sbt_p, sctx_p = minimax_m3_index_topk(
        idx_q,
        idxc,
        bt,
        cu,
        seq_lens_t,
        prefix_lens_t,
        max_q,  # max_query_len
        max_seq,
        topk,
        0,
        1,
        nidx,
        sm,
        emit_sparse_block_table=True,
    )

    assert torch.equal(tk_d, tk_p), "decode topk_idx != prefill topk_idx"
    assert torch.equal(sbt_d, sbt_p), "decode sparse_bt != prefill sparse_bt"
    assert torch.equal(sctx_d, sctx_p), "decode sparse_ctx != prefill sparse_ctx"
