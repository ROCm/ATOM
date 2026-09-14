# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA operators for cache preparation, index selection and sparse paged GQA.

Cache writes skip negative slots. Only complete groups are pooled and scored;
selection expands those groups and appends each query's incomplete causal tail.
mRoPE preserves intermediate dtype roundings and the group's first-token position.

Source provenance:
- Scoring and sparse attention: ROCm/aiter PR #4882 ("[Triton/Gluon] [QSA]
  Add paged sparse attention kernels"), branch
  `haic0:qsa-paged-sparse-attention`,
  `aiter/ops/triton/_triton_kernels/attention/qsa_*.py`; MIT licensed,
  Copyright (C) 2026 Advanced Micro Devices, Inc.
  The portable Triton implementation is vendored because that PR is absent
  from the dependency build used for this adaptation. Its gfx950 Gluon
  specializations target head_dim 128 / GQA group 5 / 8 index heads, rather
  than this checkpoint's 256 / 12 / 4.
- Cache writes and pooling: `_store_qsa_rows_kernel` and
  `_compress_qsa_groups_kernel` from the reference
  `qwen3_8_flash_next/nvidia/ops/qsa.py`.
- Index expansion: adapted from the GLM pooled-index algorithm, kept local
  without changing the other model's implementation.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.topk import top_k_per_row_prefill


@triton.jit
def _qsa_compressed_slots_kernel(
    slots_ptr,
    positions_ptr,
    out_ptr,
    num_tokens,
    stride_slots,
    stride_positions,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = tokens < num_tokens
    slot = tl.load(slots_ptr + tokens * stride_slots, mask, -1).to(tl.int64)
    position = tl.load(positions_ptr + tokens * stride_positions, mask, -1)
    closes_group = (position >= 0) & ((position + 1) % RATIO == 0) & (slot >= 0)
    tl.store(out_ptr + tokens, tl.where(closes_group, slot // RATIO, -1), mask)


def qsa_compressed_slots(
    slot_mapping: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
    out: torch.Tensor,
) -> None:
    """Map complete groups into the existing persistent compressed-slot buffer.

    QSA shares the scheduler's physical pages and block_size is divisible by
    compress_ratio. Therefore floor(physical_slot / ratio) is exactly
    physical_page * (block_size / ratio) + compressed_offset; no second block
    table lookup, cast, or temporary token-sized tensor is needed.
    """
    tokens = slot_mapping.numel()
    if (
        compress_ratio <= 0
        or logical_positions.numel() != tokens
        or out.numel() != tokens
    ):
        raise ValueError(
            "QSA compressed slots require equal token lengths and a positive ratio"
        )
    if out.ndim != 1 or out.stride(0) != 1:
        raise ValueError("QSA compressed slots output must be contiguous")
    if tokens:
        _qsa_compressed_slots_kernel[(triton.cdiv(tokens, 256),)](
            slot_mapping,
            logical_positions,
            out,
            tokens,
            slot_mapping.stride(0),
            logical_positions.stride(0),
            RATIO=compress_ratio,
            BLOCK=256,
            num_warps=4,
        )


@triton.jit
def _qsa_mrope_kernel(
    q_ptr,
    k_ptr,
    out_q_ptr,
    out_k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    pos_axis_stride,
    pos_token_stride,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // (Q_HEADS + K_HEADS)
    head = row % (Q_HEADS + K_HEADS)
    dims = tl.arange(0, BLOCK_D)
    if head < Q_HEADS:
        src = q_ptr + token * q_token_stride + head * q_head_stride
        dst = out_q_ptr + (token * Q_HEADS + head) * HEAD_DIM
    else:
        src = k_ptr + token * k_token_stride + (head - Q_HEADS) * k_head_stride
        dst = out_k_ptr + (token * K_HEADS + head - Q_HEADS) * HEAD_DIM
    x = tl.load(src + dims, dims < HEAD_DIM, other=0)
    half = ROTARY_DIM // 2
    freq = dims % half
    if INTERLEAVED:
        axis = tl.where((freq % 3 == 1) & (freq < 3 * SECTION_H), 1, 0)
        axis = tl.where((freq % 3 == 2) & (freq < 3 * SECTION_W), 2, axis)
    else:
        axis = tl.where(
            freq < SECTION_T, 0, tl.where(freq < SECTION_T + SECTION_H, 1, 2)
        )
    position = tl.load(pos_ptr + axis * pos_axis_stride + token * pos_token_stride)
    # MRotaryEmbedding/HF cast frequencies and EACH product to the activation
    # dtype. Keeping those roundings is important near cancellation in BF16.
    cos = tl.load(cos_ptr + position * half + freq).to(x.dtype).to(tl.float32)
    sin = tl.load(sin_ptr + position * half + freq).to(x.dtype).to(tl.float32)
    paired = tl.where(dims < half, dims + half, dims - half)
    y = tl.load(src + paired, dims < ROTARY_DIM, other=0).to(tl.float32)
    y = tl.where(dims < half, -y, y)
    xc = (x.to(tl.float32) * cos).to(x.dtype).to(tl.float32)
    ys = (y * sin).to(x.dtype).to(tl.float32)
    result = tl.where(dims < ROTARY_DIM, (xc + ys).to(x.dtype), x)
    tl.store(dst + dims, result, dims < HEAD_DIM)


def qsa_apply_mrope(
    rotary_emb,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Partial NeoX mRoPE for [tokens, heads, dim], without dummy indexer K.

    Reuses `rotary_emb`'s cached frequencies and preserves intermediate dtype
    roundings. Strided token/head and position inputs need no contiguous copy.
    Normalization and KV cache writes are handled separately.
    """
    tokens, heads, dim = q.shape
    if q.stride(-1) != 1 or not rotary_emb.is_neox_style:
        raise ValueError(
            "QSA mRoPE requires contiguous head dimensions and NeoX layout"
        )
    if k is not None and (
        k.shape[0] != tokens or k.shape[2] != dim or k.stride(-1) != 1
    ):
        raise ValueError("QSA mRoPE Q/K shapes or strides disagree")
    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = (
        torch.empty(k.shape, dtype=k.dtype, device=k.device) if k is not None else None
    )
    if tokens == 0:
        return out_q, out_k
    rotary_dim = rotary_emb.rotary_dim
    sections = getattr(rotary_emb, "mrope_section", None) or [rotary_dim // 2, 0, 0]
    if positions.ndim not in (1, 2) or (
        positions.ndim == 2 and positions.shape[0] not in (1, 3)
    ):
        raise ValueError("QSA positions must be [tokens], [1, tokens] or [3, tokens]")
    if positions.shape[-1] < tokens:
        raise ValueError("QSA positions are shorter than Q/K")
    axis_stride = (
        positions.stride(0) if positions.ndim == 2 and positions.shape[0] == 3 else 0
    )
    kv_heads = k.shape[1] if k is not None else 0
    _qsa_mrope_kernel[(tokens * (heads + kv_heads),)](
        q,
        k if k is not None else q,
        out_q,
        out_k if k is not None else out_q,
        rotary_emb.cos_cache,
        rotary_emb.sin_cache,
        positions,
        q.stride(0),
        q.stride(1),
        k.stride(0) if k is not None else 0,
        k.stride(1) if k is not None else 0,
        axis_stride,
        positions.stride(-1),
        Q_HEADS=heads,
        K_HEADS=kv_heads,
        HEAD_DIM=dim,
        ROTARY_DIM=rotary_dim,
        SECTION_T=sections[0],
        SECTION_H=sections[1],
        SECTION_W=sections[2],
        INTERLEAVED=getattr(rotary_emb, "mrope_interleaved", False),
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out_q, out_k


@triton.jit
def _qsa_store_rows_kernel(
    cache_ptr,
    slots_ptr,
    values_ptr,
    stride_cache_slot,
    stride_cache_head,
    stride_cache_dim,
    stride_value_row,
    stride_value_head,
    stride_value_dim,
    num_rows,
    num_slots,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)

    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_slots)
    if not valid:
        return

    values = tl.load(
        values_ptr
        + row * stride_value_row
        + head * stride_value_head
        + dims * stride_value_dim,
        mask=dims < HEAD_DIM,
        other=0,
    )
    tl.store(
        cache_ptr
        # A physical slot times the per-slot stride overflows int32 well
        # inside a normal pool size, so widen before multiplying.
        + slot.to(tl.int64) * stride_cache_slot
        + head * stride_cache_head
        + dims * stride_cache_dim,
        values,
        mask=dims < HEAD_DIM,
    )


def qsa_store_rows(
    cache: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Scatter `values [rows, heads, dim]` into `cache` at `slots`, skipping -1.

    `cache` is a paged tensor `[pages, page_size, heads, dim]`; the slot index
    addresses it flattened over its first two axes, exactly as ATOM's
    `slot_mapping` already does for the main KV pool.
    """
    if cache.ndim != 4:
        raise ValueError("cache must be [pages, page_size, heads, dim]")
    if values.ndim != 3:
        raise ValueError("values must be [rows, heads, dim]")
    if values.shape[1] != cache.shape[2] or values.shape[2] != cache.shape[3]:
        raise ValueError("values and cache disagree on heads/dim")
    rows = values.shape[0]
    if rows == 0:
        return
    if slots.shape[0] < rows:
        raise ValueError("slot mapping is shorter than the value rows")

    flat = cache.view(cache.shape[0] * cache.shape[1], cache.shape[2], cache.shape[3])
    head_dim = values.shape[2]
    _qsa_store_rows_kernel[(rows, values.shape[1])](
        flat,
        slots,
        values,
        flat.stride(0),
        flat.stride(1),
        flat.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        rows,
        flat.shape[0],
        NUM_HEADS=values.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


@triton.jit
def _qsa_compress_groups_kernel(
    raw_cache_ptr,
    position_cache_ptr,
    page_table_ptr,
    token_to_request_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_page,
    stride_raw_token,
    stride_raw_dim,
    stride_position_page,
    stride_position_token,
    stride_position_axis,
    stride_table_request,
    stride_table_page,
    stride_pooled_row,
    stride_pooled_dim,
    stride_first_row,
    stride_first_axis,
    num_rows,
    num_pages,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_POSITIONS: tl.constexpr,
) -> None:
    """Mean-pool the group that ends at each token, reading the paged raw cache."""
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)

    request = tl.load(token_to_request_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    # A row is only pooled when it closes a complete group -- which the
    # compressed slot mapping already encodes by leaving every other row at -1.
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if valid_row:
        for offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - offset)
            logical_page = position // PAGE_SIZE
            page_offset = position % PAGE_SIZE
            valid = logical_page < PAGE_TABLE_WIDTH
            physical_page = tl.load(
                page_table_ptr
                + safe_request * stride_table_request
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_pages)
            accumulator += tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_page
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    axes = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_POSITIONS:
        # mRoPE: the group's first token carries three independent axes that
        # cannot be recovered from the current token, so read them back.
        logical_page = first_position // PAGE_SIZE
        page_offset = first_position % PAGE_SIZE
        valid = valid_row & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_pages)
        values = tl.load(
            position_cache_ptr
            + tl.maximum(physical_page, 0).to(tl.int64) * stride_position_page
            + page_offset * stride_position_token
            + axes * stride_position_axis,
            mask=valid & (axes < 3),
            other=0,
        )
        tl.store(
            first_positions_ptr + row * stride_first_row + axes * stride_first_axis,
            values,
            mask=(row < num_rows) & (axes < 3),
        )
    else:
        first_position = tl.where(valid_row, first_position, 0)
        tl.store(
            first_positions_ptr + row * stride_first_row + axes * stride_first_axis,
            first_position,
            mask=(row < num_rows) & (axes < 3),
        )


def qsa_compress_groups(
    raw_key_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    position_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the group closed by each token; return `(pooled, first_positions)`.

    `pooled` is `[rows, 1, head_dim]` in the raw cache's dtype and is only
    meaningful where `compressed_slots >= 0`; the caller stores it back under
    the same mapping, so junk rows are never written. `first_positions` is
    `[rows, 3]` int64 -- three identical linear positions for a text model, the
    cached mRoPE axes when `position_cache` is supplied.
    """
    if raw_key_cache.ndim != 4 or raw_key_cache.shape[2] != 1:
        raise ValueError("raw_key_cache must be [pages, page_size, 1, head_dim]")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    rows = int(token_to_request.shape[0])
    head_dim = raw_key_cache.shape[3]
    device = raw_key_cache.device
    # Both outputs are fully written, including the invalid/padded rows.
    pooled = torch.empty((rows, 1, head_dim), dtype=raw_key_cache.dtype, device=device)
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=device)
    if rows == 0:
        return pooled, first_positions

    load_positions = position_cache is not None
    if load_positions:
        if position_cache.ndim != 4 or position_cache.shape[3] < 3:
            raise ValueError("position_cache must be [pages, page_size, 1, >=3]")
        position_strides = (
            position_cache.stride(0),
            position_cache.stride(1),
            position_cache.stride(3),
        )
    else:
        position_cache = raw_key_cache
        position_strides = (0, 0, 0)

    _qsa_compress_groups_kernel[(rows,)](
        raw_key_cache,
        position_cache,
        page_table,
        token_to_request,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_key_cache.stride(0),
        raw_key_cache.stride(1),
        raw_key_cache.stride(3),
        *position_strides,
        page_table.stride(0),
        page_table.stride(1),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        raw_key_cache.shape[0],
        page_table.shape[0],
        PAGE_SIZE=raw_key_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        LOAD_POSITIONS=load_positions,
        num_warps=4,
    )
    return pooled, first_positions


@triton.jit
def _qsa_paged_mqa_logits_kernel(
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
    stride_q_dim,
    stride_cache_page,
    stride_cache_token,
    stride_cache_dim,
    stride_table_request,
    stride_table_page,
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
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Score BF16 compressed QSA keys directly from a paged cache."""
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)

    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + token)
    context_len = tl.load(
        context_lens_ptr + safe_request,
        mask=request_valid,
        other=0,
    )
    visible_groups = tl.maximum(
        0,
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            context_len // COMPRESS_RATIO,
        ),
    )
    if tl.program_id(1) == 0:
        tl.store(visible_groups_ptr + token, visible_groups)
        if row_starts_ptr is not None:
            tl.store(row_starts_ptr + token, 0)

    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (token < num_tokens)
        & (columns < num_columns)
        & (columns < visible_groups)
        & request_valid
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_request
        + safe_logical_page * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)

    score = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + token * stride_q_token + head * stride_q_head + dims * stride_q_dim,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_page
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        score += tl.maximum(tl.sum(keys * query[None, :], axis=1), 0.0)

    score /= score_divisor
    tl.store(
        logits_ptr + token * stride_logits_token + columns,
        tl.where(valid, score, -float("inf")),
        mask=(token < num_tokens) & (columns < num_columns),
    )


@triton.jit
def _expand_selected_groups(
    Groups,
    Context,
    Positions,
    Requests,
    Out,
    num_requests,
    n_groups,
    out_cols,
    group_s0,
    group_s1,
    out_s0,
    out_s1,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(Requests + row)
    valid_req = (req >= 0) & (req < num_requests)
    context = tl.load(Context + req, valid_req, 0)
    pos = tl.load(Positions + row)
    length = tl.where(valid_req & (pos >= 0) & (pos < context), pos + 1, 0)
    full_groups = length // RATIO
    tail_start = full_groups * RATIO
    # Append the incomplete group immediately after the selected groups, not
    # at the fixed budget: short requests must retain their newest tokens.
    history_end = tl.minimum(full_groups, n_groups) * RATIO
    is_history = cols < history_end
    group = tl.load(
        Groups + row * group_s0 + (cols // RATIO) * group_s1,
        (cols < out_cols) & is_history,
        -1,
    )
    history = tl.where(
        (group >= 0) & (group < full_groups), group * RATIO + cols % RATIO, -1
    )
    tail_offset = cols - history_end
    tail = tl.where(
        (tail_offset >= 0) & (tail_offset < length - tail_start),
        tail_start + tail_offset,
        -1,
    )
    tl.store(
        Out + row * out_s0 + cols * out_s1,
        tl.where(is_history, history, tail).to(tl.int32),
        cols < out_cols,
    )


# Cap on the FP32 logits buffer; scoring is chunked over query rows to respect
# it, because `columns` grows with the paged-cache capacity.
DEFAULT_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024

_SCORING_BLOCK_N = 32


def _check_vector(name: str, tensor: torch.Tensor, length: int | None = None) -> None:
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional int32/int64 tensor")
    if length is not None and tensor.shape[0] != length:
        raise ValueError(f"{name} must contain {length} entries")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def qsa_paged_mqa_logits(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = 4,
    score_divisor: float | None = None,
    max_columns: int | None = None,
    row_starts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed key groups.

    `q` is `[tokens, index_heads, head_dim]` and `compressed_k_cache` is
    `[pages, page_size, 1, head_dim]`. Returns FP32 logits `[tokens, columns]`
    and the per-row count of causally visible groups.

    If supplied, `row_starts` is filled with zeros so top-k selection starts
    at column zero in each row, avoiding a separate initialization kernel.

    `max_columns` caps how many compressed groups are scored. Column `c`
    addresses group `c` through the page table, so dropping the tail simply
    stops at a group no row can see -- the scores of the ones that remain are
    unchanged. Without it the width is the whole block table, and a 1k-token
    request on a 16k-token engine pays for 16k tokens of scoring and top-k on
    every one of the 12 QSA layers.
    """
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("q must have shape [tokens, heads, head_dim]")
    if (
        compressed_k_cache.ndim != 4
        or compressed_k_cache.shape[2] != 1
        or compressed_k_cache.shape[3] != q.shape[2]
    ):
        raise ValueError(
            "compressed_k_cache must have shape [pages, page_size, 1, head_dim]"
        )
    if q.dtype != compressed_k_cache.dtype:
        raise ValueError("q and compressed_k_cache must have the same dtype")
    if page_table.ndim != 2 or page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("page_table must be a two-dimensional integer tensor")
    _check_vector("token_to_request", token_to_request, q.shape[0])
    _check_vector("query_positions", query_positions, q.shape[0])
    _check_vector("context_lens", context_lens, page_table.shape[0])
    if row_starts is not None:
        _check_vector("row_starts", row_starts, q.shape[0])

    divisor = q.shape[2] ** 0.5 if score_divisor is None else score_divisor
    if divisor <= 0:
        raise ValueError("score_divisor must be positive")
    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if max_columns is not None:
        columns = max(1, min(columns, int(max_columns)))
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    # The scorer writes every row's count; no separate zero-fill launch.
    visible_groups = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if q.shape[0] == 0 or columns == 0:
        visible_groups.zero_()
        if row_starts is not None:
            row_starts.zero_()
        return logits, visible_groups

    _qsa_paged_mqa_logits_kernel[(q.shape[0], triton.cdiv(columns, _SCORING_BLOCK_N))](
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
        q.stride(2),
        compressed_k_cache.stride(0),
        compressed_k_cache.stride(1),
        compressed_k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        compressed_k_cache.shape[0],
        page_table.shape[0],
        float(divisor),
        PAGE_SIZE=compressed_k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        COMPRESS_RATIO=compress_ratio,
        BLOCK_N=_SCORING_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        num_warps=4,
    )
    return logits, visible_groups


def qsa_expand_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
    output_width: int | None = None,
) -> torch.Tensor:
    """Expand selected groups to token ids and append the causal tail.

    `output_width` may be narrower than the full `token_topk + ratio - 1` when
    the caller knows the context cannot fill it: column `c` of the result
    depends only on `c` and the row's own counts, so truncating drops columns
    that would have been `-1` padding anyway.
    """
    if compress_ratio <= 0 or token_topk <= 0:
        raise ValueError("compress_ratio and token_topk must be positive")
    if block_indices.ndim != 2 or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be a two-dimensional int32 tensor")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    _check_vector("query_positions", query_positions, block_indices.shape[0])
    _check_vector("token_to_request", token_to_request, block_indices.shape[0])
    _check_vector("context_lens", context_lens)

    block_topk = block_indices.shape[1]
    if block_topk > token_topk // compress_ratio:
        raise ValueError(
            f"block_indices must have at most {token_topk // compress_ratio} columns"
        )
    if output_width is None:
        output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.dtype != torch.int32 or out.shape != (
        block_indices.shape[0],
        output_width,
    ):
        raise ValueError("out has an invalid shape")
    if block_indices.shape[0] == 0:
        return out

    _expand_selected_groups[(block_indices.shape[0], triton.cdiv(output_width, 128))](
        block_indices,
        context_lens,
        query_positions,
        token_to_request,
        out,
        context_lens.numel(),
        block_topk,
        output_width,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        RATIO=compress_ratio,
        BLOCK=128,
    )
    return out


def qsa_select_paged_tokens(
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
) -> torch.Tensor:
    """Score, select and expand in one call.

    Returns `[tokens, width]`, where `width` is `budget + ratio - 1` unless
    `max_seq_len` -- the longest sequence in the batch -- shows the context
    cannot fill it, in which case both the scored width and the emitted width
    shrink to what the context can actually reach. Everything dropped would
    have been `-1` padding, so the result is identical column for column, but
    a 1k-token request on a 16k-token engine stops paying 16k-token scoring
    and top-k costs on each of the 12 QSA layers.
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
        logits, visible_groups = qsa_paged_mqa_logits(
            q[row_slice],
            compressed_k_cache,
            page_table,
            token_to_request[row_slice],
            query_positions[row_slice],
            context_lens,
            compress_ratio,
            max_columns=columns,
            row_starts=row_starts,
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


@triton.jit
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    widths_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_token,
    stride_indices_column,
    stride_table_request,
    stride_table_page,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    num_tokens,
    num_cache_pages,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KV_SPLITS: tl.constexpr = 1,
) -> None:
    """Apply GQA over arbitrary logical tokens in separate paged BF16 K/V."""
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    part = tl.program_id(2)
    if KV_SPLITS > 1:  # noqa: SIM102 -- compile-time guard for widths_ptr=None
        if kv_head == 0 and part == 0:
            # Reduction needs each query's selection width. Write it alongside
            # the partial results to avoid a separate initialization kernel.
            tl.store(widths_ptr + token, TOPK)
    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_q_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + token * stride_q_token
        + (first_q_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    running_max = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    partition_size = tl.cdiv(TOPK, KV_SPLITS * BLOCK_N) * BLOCK_N
    for start in tl.range(
        part * partition_size, tl.minimum((part + 1) * partition_size, TOPK), BLOCK_N
    ):
        columns = start + column_offsets
        logical_token = tl.load(
            logical_indices_ptr
            + token * stride_indices_token
            + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        safe_logical_token = tl.maximum(logical_token, 0)
        logical_page = safe_logical_token // PAGE_SIZE
        page_offset = safe_logical_token % PAGE_SIZE
        valid = (
            (token < num_tokens)
            & request_valid
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)

        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[None, :] * stride_k_page
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_physical_page[:, None] * stride_v_page
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )

        scores = tl.where(valid[None, :], tl.dot(query, keys), -1.0e20)
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.math.exp2(running_max - next_max)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_max[:, None]),
            0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = next_max

    if KV_SPLITS > 1:
        # Reduction consumes unnormalized [token, head, split, dim] partials
        # with each split's running maximum and sum for base-2 softmax.
        partial_offset = (
            token * NUM_KV_HEADS * GROUP_SIZE + first_q_head + head_offsets
        ) * KV_SPLITS + part
        tl.store(
            partial_max_ptr + partial_offset, running_max, head_offsets < GROUP_SIZE
        )
        tl.store(
            partial_sum_ptr + partial_offset, running_sum, head_offsets < GROUP_SIZE
        )
        output_ptr += part * HEAD_DIM
        output = accumulator
    else:
        output = tl.where(
            running_sum[:, None] > 0,
            accumulator / tl.maximum(running_sum[:, None], 1.0e-20),
            0.0,
        )
    tl.store(
        output_ptr
        + token * stride_output_token
        + (first_q_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(token < num_tokens)
        & (head_offsets[:, None] < GROUP_SIZE)
        & (dim_offsets[None, :] < HEAD_DIM),
    )


def qsa_sparse_paged_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    softmax_scale: float | None = None,
    kv_splits: int | None = None,
) -> torch.Tensor:
    """Grouped-query attention restricted to `logical_indices` (-1 = padding)."""
    if q.ndim != 3:
        raise ValueError("q must be [tokens, query_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V caches must have matching [pages, page, heads, dim]")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache and v_cache must have the same dtype")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("query heads must form equal groups over KV heads")
    if (
        logical_indices.ndim != 2
        or logical_indices.shape[0] != q.shape[0]
        or logical_indices.dtype != torch.int32
    ):
        raise ValueError("logical_indices must be int32 [tokens, selection_width]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be a two-dimensional integer tensor")
    if (
        token_to_request.shape != (q.shape[0],)
        or token_to_request.dtype not in (torch.int32, torch.int64)
        or not token_to_request.is_contiguous()
    ):
        raise ValueError("token_to_request must be contiguous int32/int64 [tokens]")

    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    # Split-K reduction requires contiguous output head dimensions.
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if q.shape[0] == 0:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = max(16, triton.next_power_of_2(group_size))
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    block_n = 32
    if kv_splits is None:
        # Small decode batches otherwise expose as little as one CTA/layer.
        kv_splits = (
            min(
                16,
                triton.next_power_of_2(triton.cdiv(512, q.shape[0] * k_cache.shape[2])),
            )
            if logical_indices.shape[1] >= 512
            else 1
        )
    if kv_splits < 1 or kv_splits & (kv_splits - 1):
        raise ValueError("kv_splits must be a positive power of two")
    if logical_indices.shape[1] == 0:
        # Empty selections produce zero attention output; skip split reduction.
        return out.zero_()
    target = out
    partial_max = partial_sum = out
    widths = None
    if kv_splits > 1:
        shape = (q.shape[0], q.shape[1], kv_splits)
        partial_max = torch.empty(shape, device=q.device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)
        target = torch.empty((*shape, q.shape[2]), device=q.device, dtype=torch.float32)
        widths = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], k_cache.shape[2], kv_splits)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        target,
        partial_max,
        partial_sum,
        widths,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        target.stride(0),
        target.stride(1),
        target.stride(-1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        KV_SPLITS=kv_splits,
        num_warps=4,
        num_stages=2,
    )
    if kv_splits > 1:
        from aiter.ops.triton._triton_kernels.attention.mla import (
            _mla_decode_fwd_reduce_kernel,
        )

        _mla_decode_fwd_reduce_kernel[(q.shape[0], q.shape[1])](
            out,
            target,
            partial_max,
            partial_sum,
            widths,
            None,
            q.shape[0],
            q.shape[1],
            out.stride(0),
            out.stride(1),
            1,
            1,
            q.shape[0],
            TILE_SIZE=block_n,
            KV_LORA_RANK=q.shape[2],
            query_start_len_ptr=None,
            BLOCK_Q=1,
            NUM_SEGMENTS_PER_SEQ=kv_splits,
            ALL_DECODE=True,
        )
    return out
