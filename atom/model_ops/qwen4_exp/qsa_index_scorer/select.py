# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``qsa_select_paged_tokens`` with an injected scorer.

The body below is the stock ``atom/model_ops/qwen4_exp/ops/qsa.py::
qsa_select_paged_tokens`` statement for statement, with the one call to
``qsa_paged_mqa_logits`` replaced by ``scorer``: same max_columns bucketing
(power-of-two ``reachable`` groups), same row chunking under
``logits_workspace_bytes``, same per-chunk ``row_starts`` staging, same top-k
and expansion. ``qsa.py`` itself stays byte-identical (frozen golden
reference, sha256 aaa78f43... CRLF / c2e46e50... LF), and
``tests/model_ops/test_qsa_index_scorer.py`` AST-compares the two bodies, so
an upstream change to the loop fails CPU CI until this copy is re-synced.

Imported only when the switch is on (it pulls in qsa.py: Triton + aiter).
"""

import torch
from aiter.ops.topk import top_k_per_row_prefill

from atom.model_ops.qwen4_exp.ops.qsa import (
    DEFAULT_LOGITS_WORKSPACE_BYTES,
    qsa_expand_block_indices,
)


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
    """Stock ``qsa_select_paged_tokens`` scoring through ``scorer``.

    ``scorer`` takes ``qsa_paged_mqa_logits``'s 12 arguments and returns its
    ``(logits, visible_groups)`` -- aiter's neutral op
    ``aiter.ops.qsa.qsa_paged_mqa_logits`` in production. Whatever it raises
    propagates: there is no second (ATOM) scorer behind it.
    """
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    rows = q.shape[0]

    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if columns < 1:
        raise ValueError("compressed paged-cache capacity must be positive")
    block_topk = min(token_topk // compress_ratio, columns)
    if max_seq_len is not None:
        reachable = (int(max_seq_len) + compress_ratio - 1) // compress_ratio
        # Round the bucket up to a power of two. `BLOCK_TOPK`, `OUTPUT_WIDTH`
        # and `TOPK` are Triton constexprs, so an exact width would recompile
        # the expand and GQA kernels every few decoded tokens -- which costs
        # far more than the work it saves. Rounding up is always safe: a
        # top-k wider than the visible groups just returns -1 for the extras.
        bucket = 1 << (max(reachable, 1) - 1).bit_length()
        columns = max(1, min(columns, bucket))
        block_topk = max(1, min(block_topk, bucket))
    output_width = block_topk * compress_ratio + compress_ratio - 1

    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape[0] != rows or out.shape[1] < output_width:
        raise ValueError("out has an invalid shape")
    else:
        out = out[:, :output_width]
    if rows == 0:
        return out

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
        selected_groups = torch.empty(
            (row_end - row_start, block_topk), dtype=torch.int32, device=q.device
        )
        top_k_per_row_prefill(
            logits,
            row_starts,
            visible_groups,
            selected_groups,
            None,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
            # Equal index scores must choose the same keys and ordering in
            # single-token decode and multi-token verification.
            stable=True,
        )
        qsa_expand_block_indices(
            selected_groups,
            query_positions[row_slice],
            context_lens,
            token_to_request[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
            output_width=output_width,
        )
    return out
