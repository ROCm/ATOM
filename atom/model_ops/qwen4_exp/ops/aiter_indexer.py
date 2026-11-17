# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER selection over ATOM's prepared Q and existing compressed-key pool."""

import functools
import inspect
import logging

import torch

from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger(__name__)


@functools.cache
def _selector():
    try:
        from aiter.ops.qsa import qsa_select_paged_tokens
    except ImportError:
        return None
    # Earlier AITER releases have the scorer-only implementation of this API.
    params = inspect.signature(qsa_select_paged_tokens).parameters
    if "query_lens" not in params or "prefix_lens" not in params:
        return None
    return qsa_select_paged_tokens


@torch.compiler.assume_constant_result
def _available():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx942" and _selector() is not None


@functools.cache
def _log_active():
    logger.info("QSA indexer selection uses AITER; ATOM owns prep and cache updates")


def _select(
    q: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    positions: torch.Tensor,
    context_lens: torch.Tensor,
    max_seq_len: int | None,
    logits_workspace_bytes: int,
    query_lens: list[int] | None,
    prefix_lens: list[int] | None,
) -> torch.Tensor:
    op = _selector()
    if op is None:
        raise RuntimeError("The installed AITER lacks the QSA indexer selector")
    return op(
        q,
        compressed,
        page_table,
        token_to_request,
        positions,
        context_lens,
        token_topk=2048,
        compress_ratio=4,
        max_seq_len=max_seq_len,
        logits_workspace_bytes=logits_workspace_bytes,
        allow_flydsl=True,
        query_lens=query_lens,
        prefix_lens=prefix_lens,
    )


def _fake(q, *args, **kwargs):
    return torch.empty((q.shape[0], 2051), dtype=torch.int32, device=q.device)


direct_register_custom_op(
    op_name="atom_qsa_indexer_select",
    op_func=_select,
    mutates_args=[],
    fake_impl=_fake,
)


def try_select(
    q,
    compressed,
    page_table,
    token_to_request,
    positions,
    context_lens,
    token_topk,
    compress_ratio,
    columns,
    output_width,
    logits_workspace_bytes,
    max_seq_len,
    compressed_k_scale,
    *,
    query_lens=None,
    prefix_lens=None,
):
    """Return AITER token selections, or None before any unsupported call."""
    if (
        not q.is_cuda
        or q.dtype != torch.bfloat16
        or q.ndim != 3
        or q.shape[1:] != (4, 128)
        or not q.is_contiguous()
        or q.shape[0] == 0
        or token_topk != 2048
        or compress_ratio != 4
        or output_width != 2051
        or compressed_k_scale is not None
        or compressed.dtype != torch.bfloat16
        or compressed.ndim != 4
        or compressed.shape[1:] != (16, 1, 128)
        or not compressed.is_contiguous()
        or compressed.numel() * 2 >= 2**31
        or page_table.dtype != torch.int32
        or page_table.stride(1) != 1
        or not 512 <= columns <= 262144
        or any(
            t.dtype not in (torch.int32, torch.int64) or not t.is_contiguous()
            for t in (token_to_request, positions, context_lens)
        )
        or not _available()
    ):
        return None
    if query_lens is None or prefix_lens is None:
        query_lens = prefix_lens = None
        if (
            q.shape[0] > 256
            or (q.shape[0] * columns + 512) * 4 > logits_workspace_bytes
        ):
            return None
    else:
        if sum(query_lens) != q.shape[0]:
            return None
        query_lens, prefix_lens = list(query_lens), list(prefix_lens)
    # Keep pointer extraction and host planning outside Dynamo. The schema is
    # read-only, and its fake implementation supplies the fixed selection width.
    result = torch.ops.aiter.atom_qsa_indexer_select(
        q,
        compressed,
        page_table,
        token_to_request,
        positions,
        context_lens,
        max_seq_len,
        logits_workspace_bytes,
        query_lens,
        prefix_lens,
    )
    if not torch.compiler.is_compiling():
        _log_active()
    return result


def packed_lengths(metadata, rows=None):
    """Host query/prefix lengths, clipped to the caller's existing SP row shard."""
    queries = getattr(metadata, "indexer_query_lens", None)
    prefixes = getattr(metadata, "indexer_prefix_lens", None)
    if queries is None or prefixes is None:
        host = getattr(metadata, "host_lens", None)
        if host is None:
            return None, None
        queries, sequences = host
        prefixes = tuple(s - q for s, q in zip(sequences, queries))
    if rows is None:
        return queries, prefixes
    if sum(queries) != rows.num_tokens:
        return None, None
    first, end = rows.start, min(rows.start + rows.chunk, rows.num_tokens)
    local_queries, local_prefixes = [], []
    offset = 0
    for query, prefix in zip(queries, prefixes):
        low = max(first, offset)
        high = min(end, offset + query)
        local_queries.append(max(0, high - low))
        local_prefixes.append(prefix + min(query, max(0, first - offset)))
        offset += query
    return tuple(local_queries), tuple(local_prefixes)
