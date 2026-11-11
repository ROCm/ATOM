# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA selection with an injected scorer.

Two copies of the stock ``atom/model_ops/qwen4_exp/ops/qsa.py`` selection
entries (split by #2847), statement for statement:

* ``qsa_select_paged_groups_with_scorer`` -- ``qsa_select_paged_groups`` (score
  + top-k, the row-sharded entry) with its one call to ``qsa_paged_mqa_logits``
  replaced by ``scorer``: same ``_selection_widths`` bucketing, same row
  chunking under ``logits_workspace_bytes``, same per-chunk ``row_starts``
  staging, same stable top-k.
* ``qsa_select_paged_tokens_with_scorer`` -- ``qsa_select_paged_tokens`` (the
  whole-batch entry: groups, then ``qsa_expand_block_indices``) calling the
  copy above with the same ``scorer``.

``qsa.py`` itself is not modified, and
``tests/model_ops/test_qsa_index_scorer.py`` AST-compares both bodies with the
stock ones, so an upstream change to the loop fails CPU CI until these copies
are re-synced.

Imported only when the switch is on (it pulls in qsa.py: Triton + aiter).
"""

import torch
from aiter.ops.topk import top_k_per_row_prefill

from atom.model_ops.qwen4_exp.ops.qsa import (
    DEFAULT_LOGITS_WORKSPACE_BYTES,
    _selection_widths,
    qsa_expand_block_indices,
)


def qsa_select_paged_groups_with_scorer(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_topk: int,
    compress_ratio: int = 4,
    logits_workspace_bytes: int = DEFAULT_LOGITS_WORKSPACE_BYTES,
    max_seq_len: int | None = None,
    allow_flydsl: bool = False,
    compressed_k_scale: torch.Tensor | None = None,
    *,
    scorer,
) -> tuple[torch.Tensor, int]:
    """Stock ``qsa_select_paged_groups`` scoring through ``scorer``.

    ``scorer`` takes ``qsa_paged_mqa_logits``'s 12 arguments and returns its
    ``(logits, visible_groups)`` -- aiter's neutral op
    ``aiter.ops.qsa.qsa_paged_mqa_logits`` in production. Whatever it raises
    propagates: there is no second (ATOM) scorer behind it.
    """
    columns, block_topk, output_width = _selection_widths(
        compressed_k_cache, page_table, token_topk, compress_ratio, max_seq_len
    )
    rows = q.shape[0]
    groups = torch.empty((rows, block_topk), dtype=torch.int32, device=q.device)
    # Chunk over query rows so the FP32 logits buffer stays within budget.
    rows_per_chunk = max(1, logits_workspace_bytes // max(columns * 4, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        row_starts = torch.empty(
            row_end - row_start, dtype=torch.int32, device=q.device
        )
        logits, visible_groups = scorer(
            q[row_slice],
            compressed_k_cache,
            page_table,
            token_to_request[row_slice],
            query_positions[row_slice],
            context_lens,
            compress_ratio,
            max_columns=columns,
            row_starts=row_starts,
            allow_flydsl=allow_flydsl,
            compressed_k_scale=compressed_k_scale,
        )
        top_k_per_row_prefill(
            logits,
            row_starts,
            visible_groups,
            groups[row_slice],
            None,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
            # Equal index scores must choose the same keys and ordering in
            # single-token decode and multi-token verification.
            stable=True,
        )
    return groups, output_width


def qsa_select_paged_tokens_with_scorer(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_topk: int,
    compress_ratio: int = 4,
    out: torch.Tensor | None = None,
    logits_workspace_bytes: int = DEFAULT_LOGITS_WORKSPACE_BYTES,
    max_seq_len: int | None = None,
    allow_flydsl: bool = False,
    compressed_k_scale: torch.Tensor | None = None,
    *,
    scorer,
) -> torch.Tensor:
    """Stock ``qsa_select_paged_tokens`` selecting through
    ``qsa_select_paged_groups_with_scorer`` (same ``scorer``)."""
    rows = q.shape[0]
    _, _, output_width = _selection_widths(
        compressed_k_cache, page_table, token_topk, compress_ratio, max_seq_len
    )
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape[0] != rows or out.shape[1] < output_width:
        raise ValueError("out has an invalid shape")
    else:
        out = out[:, :output_width]
    if rows == 0:
        return out
    groups, _ = qsa_select_paged_groups_with_scorer(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        token_topk,
        compress_ratio,
        logits_workspace_bytes=logits_workspace_bytes,
        max_seq_len=max_seq_len,
        allow_flydsl=allow_flydsl,
        compressed_k_scale=compressed_k_scale,
        scorer=scorer,
    )
    return qsa_expand_block_indices(
        groups,
        query_positions,
        context_lens,
        token_to_request,
        compress_ratio,
        token_topk,
        out,
        output_width=output_width,
    )
