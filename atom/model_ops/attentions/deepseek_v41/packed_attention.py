# SPDX-License-Identifier: MIT
"""Adapt V4.1 packed rows to the unchanged V4 BF16 attention interface.

Only the CSR-selected history is decoded. Scratch is bounded by sparse top-k
plus the window, per decode query or per 32-query prefill tile; its size does
not grow with the full KV history. V4 owns attention arithmetic and dispatch,
including OPUS prefill rounding. This module owns cache-format adaptation.
"""

import torch

from atom.model_ops.v4_kernels import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_prefill,
)

from .packed_rows import gather_prefix_rows


def packed_prefill(
    q,
    pool,
    indices,
    indptr,
    kv,
    extend,
    extend_indptr,
    sink,
    scale,
    *,
    out=None,
    query_tile=32,
):
    if pool.dtype != torch.uint8 or indices.dtype != torch.int64:
        raise ValueError(
            "Packed prefill requires byte storage and tagged int64 addresses"
        )
    tokens, _, dim = q.shape
    if out is None:
        out = torch.empty_like(q)
    if tokens == 0:
        return out
    rows_per_query = indices.numel() // tokens
    if rows_per_query * tokens != indices.numel():
        raise ValueError("Packed CSR capacity must be declared per query")
    capacity = min(query_tile, tokens) * rows_per_query
    scratch = torch.empty((capacity, dim), device=q.device, dtype=q.dtype)
    local_indices = torch.arange(capacity, device=q.device, dtype=torch.int32)
    for begin in range(0, tokens, query_tile):
        end = min(begin + query_tile, tokens)
        gather_prefix_rows(pool, indices, indptr, scratch, begin, end)
        local_ptr = indptr[begin : end + 1] - indptr[begin]
        result = sparse_attn_v4_paged_prefill(
            q[begin:end],
            scratch,
            local_indices,
            local_ptr,
            kv,
            extend,
            extend_indptr[begin : end + 1],
            sink,
            scale,
            out=out[begin:end],
        )
        if result.data_ptr() != out[begin:end].data_ptr():
            out[begin:end].copy_(result)
    return out


def packed_decode(q, pool, indices, indptr, sink, scale):
    if pool.dtype != torch.uint8 or indices.dtype != torch.int64:
        raise ValueError(
            "Packed decode requires byte storage and tagged int64 addresses"
        )
    # Keep the original batch size so V4 chooses the same split-K geometry.
    scratch = torch.empty(
        (indices.numel(), q.shape[-1]), device=q.device, dtype=q.dtype
    )
    local_indices = torch.arange(indices.numel(), device=q.device, dtype=torch.int32)
    gather_prefix_rows(pool, indices, indptr, scratch, 0, q.shape[0])
    return sparse_attn_v4_paged_decode(q, scratch, local_indices, indptr, sink, scale)
