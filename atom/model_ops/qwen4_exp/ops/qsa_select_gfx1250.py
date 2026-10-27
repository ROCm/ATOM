# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 (MI450 / MI455) QSA index scorer for decode verification.

Replaces `_qsa_paged_mqa_logits_kernel` for a four-token MTP verification
batch. The existing exact, stable radix top-k consumes its FP32 output.

* The scorer packs the query rows of one request with their index heads into
  a single 16-row WMMA operand (4 MTP rows x 4 heads), so each compressed-key
  tile is read once per request instead of once per row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_index_scores_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    row_starts_ptr,
    logits_ptr,
    stride_q_token,
    stride_q_head,
    stride_cache_page,
    stride_cache_token,
    stride_table_request,
    stride_logits_token,
    num_tokens,
    num_columns,
    num_cache_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROWS: tl.constexpr,
    BN: tl.constexpr,
):
    """Score `ROWS` consecutive query rows against compressed key tiles.

    Query vector m of the WMMA operand is (row m // NUM_HEADS, head
    m % NUM_HEADS). Rows of one request are contiguous, so a program usually
    walks one request; otherwise it walks each request its rows touch and
    keeps the rows that belong to it. Columns past a row's causal horizon are
    left unwritten: the selector never reads them.
    """
    M: tl.constexpr = ROWS * NUM_HEADS
    pid_m = tl.program_id(0)
    rows = pid_m * ROWS + tl.arange(0, ROWS)
    rvalid = rows < num_tokens
    request = tl.load(token_to_request_ptr + rows, mask=rvalid, other=-1)
    request_ok = rvalid & (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    position = tl.load(query_positions_ptr + rows, mask=rvalid, other=-1)
    context = tl.load(context_lens_ptr + safe_request, mask=request_ok, other=0)
    visible = tl.maximum(
        0, tl.minimum((position + 1) // COMPRESS_RATIO, context // COMPRESS_RATIO)
    )
    visible = tl.where(request_ok, visible, 0)
    if tl.program_id(1) == 0:
        tl.store(visible_groups_ptr + rows, visible, mask=rvalid)
        tl.store(row_starts_ptr + rows, 0, mask=rvalid)
    num_tiles = tl.cdiv(tl.minimum(tl.max(visible, axis=0), num_columns), BN)
    if tl.program_id(1) >= num_tiles:
        return

    m = tl.arange(0, M)
    q_row = pid_m * ROWS + m // NUM_HEADS
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + q_row[:, None].to(tl.int64) * stride_q_token
        + (m % NUM_HEADS)[:, None] * stride_q_head
        + dims[None, :],
        mask=(q_row < num_tokens)[:, None],
        other=0.0,
    )
    active = visible > 0
    req_lo = tl.min(tl.where(active, safe_request, num_requests), axis=0)
    req_hi = tl.max(tl.where(active, safe_request, -1), axis=0)

    for tile in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        columns = tile * BN + tl.arange(0, BN)
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        page_ok = logical_page < PAGE_TABLE_WIDTH
        safe_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
        score = tl.zeros((ROWS, BN), dtype=tl.float32)
        key_valid = tl.zeros((ROWS, BN), dtype=tl.int1)
        for req in range(req_lo, req_hi + 1):
            physical = tl.load(
                page_table_ptr + req.to(tl.int64) * stride_table_request + safe_page,
                mask=page_ok,
                other=-1,
            )
            key_ok = page_ok & (physical >= 0) & (physical < num_cache_pages)
            keys_t = tl.load(
                k_cache_ptr
                + tl.maximum(physical, 0).to(tl.int64)[None, :] * stride_cache_page
                + page_offset[None, :] * stride_cache_token
                + dims[:, None],
                mask=key_ok[None, :],
                other=0.0,
            )
            s = tl.maximum(tl.dot(q, keys_t), 0.0)
            s = tl.sum(tl.reshape(s, (ROWS, NUM_HEADS, BN)), axis=1)
            mine = (safe_request == req)[:, None]
            score = tl.where(mine, s, score)
            key_valid = tl.where(mine, key_ok[None, :], key_valid)
        score = score / score_divisor
        in_row = columns[None, :] < visible[:, None]
        tl.store(
            logits_ptr
            + rows[:, None].to(tl.int64) * stride_logits_token
            + columns[None, :],
            tl.where(key_valid, score, -float("inf")),
            mask=rvalid[:, None] & in_row,
        )


def qsa_index_scores(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int,
    columns: int,
    score_divisor: float,
    page_table_width: int,
    num_cu: int,
    logits: torch.Tensor | None = None,
    visible_groups: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    bn: int = 64,
    num_warps: int = 2,
    ctas_per_cu: int = 8,
) -> torch.Tensor:
    """FP32 index logits `[tokens, columns]`, written only below each row's
    visible count."""
    tokens, heads, head_dim = q.shape
    rows_per_program = 16 // heads
    if logits is None:
        logits = torch.empty((tokens, columns), dtype=torch.float32, device=q.device)
    if visible_groups is None:
        visible_groups = torch.empty(tokens, dtype=torch.int32, device=q.device)
    if row_starts is None:
        row_starts = torch.empty(tokens, dtype=torch.int32, device=q.device)
    if tokens == 0:
        return logits
    row_programs = triton.cdiv(tokens, rows_per_program)
    column_programs = max(
        1,
        min(triton.cdiv(columns, bn), (ctas_per_cu * num_cu) // row_programs),
    )
    _qsa_index_scores_kernel[(row_programs, column_programs)](
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        visible_groups,
        row_starts,
        logits,
        q.stride(0),
        q.stride(1),
        compressed_k_cache.stride(0),
        compressed_k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        tokens,
        columns,
        compressed_k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=compressed_k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table_width,
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        COMPRESS_RATIO=compress_ratio,
        ROWS=rows_per_program,
        BN=bn,
        num_warps=num_warps,
    )
    return logits


def qsa_index_scores_supports(q, compressed_k_cache, page_table) -> bool:
    _, heads, head_dim = q.shape
    return (
        q.dtype == torch.bfloat16
        and compressed_k_cache.dtype == torch.bfloat16
        and 16 % heads == 0
        and head_dim in (64, 128, 256)
        and q.stride(2) == 1
        and compressed_k_cache.stride(3) == 1
        and page_table.stride(1) == 1
    )
