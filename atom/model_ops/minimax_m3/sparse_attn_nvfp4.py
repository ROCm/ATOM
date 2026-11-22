# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Row mapping for block-sparse attention over the NVFP4 KV cache
(``--kv-cache-dtype nvfp4``).

The attention itself is aiter's ``sparse_pa_nvfp4_{decode,prefill}``
(``aiter.ops.flydsl``), called straight from the attention impl. They take what the
selector emits as it is: ``topk_idx [Hkv, rows, 16]`` of logical block ids (-1 padded,
any order, the tail block always among them) and the request-level block table of
128-token pages. There is no page-16 table to build (``emit_sparse_block_table=False``)
and no length array per row; q is bf16 ``[rows, Hq, 128]`` and the output bf16 as well.
The layer's fp32 global K / V scale tensors ride along (K folds into the softmax scale,
V into the output, inside the kernel).

Decode rows are dense: ``query_len`` rows per request, row ``i`` the query token
``i % query_len`` of request ``i // query_len``, causal against ``seq_lens``. Prefill
rows are ragged: the kernel takes each row's request and key count (position + 1), which
:func:`prefill_row_positions` builds once per step from the chunk's ``cu_seqlens_q`` and
``prefix_lens``.
"""

import torch


def prefill_row_positions(
    cu_seqlens_q: torch.Tensor, prefix_lens: torch.Tensor, total_q: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(request id, causal key count) of each of ``total_q`` prefill rows, int32.

    Row ``r`` of request ``b`` (``cu[b] <= r < cu[b+1]``) sits at position
    ``prefix_lens[b] + r - cu[b]`` and so attends ``position + 1`` keys. No host
    sync: a searchsorted over the chunk's cumulative lengths.
    """
    rows = torch.arange(total_q, dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    last = prefix_lens.shape[0] - 1
    req = torch.searchsorted(cu_seqlens_q[1:], rows, right=True).clamp_(max=last)
    kv_lens = prefix_lens[req] + (rows - cu_seqlens_q[req]) + 1
    return req.to(torch.int32), kv_lens.to(torch.int32)
