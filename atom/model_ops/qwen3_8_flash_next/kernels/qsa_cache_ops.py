# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cache-side Triton kernels for Qwen3.8-Flash-Next QSA.

Two operations the scoring/selection kernels in this directory assume have
already happened:

  * `qsa_store_rows` scatters one row per token into a paged cache using a
    slot mapping, skipping `-1` slots. Used for the main K/V, the raw index
    key, and the packed mRoPE positions.
  * `qsa_compress_groups` mean-pools each *complete* group of
    `compress_ratio` raw index keys, reading them back out of the paged raw
    cache so a group that straddles a chunk boundary still sees all of its
    members. It also reports the position of the group's FIRST token, which
    is where the pooled key must later be rotated.

Both take slot mappings that are `-1` for rows they must not touch: padded
CUDA-graph rows, and (for the compressed cache) every token that does not
close a group. Nothing here reads a length off the device, so the whole path
stays synchronization-free.

Ported from the reference implementation's `_store_qsa_rows_kernel` and
`_compress_qsa_groups_kernel` (`qwen3_8_flash_next/nvidia/ops/qsa.py`).
"""

import torch
import triton
import triton.language as tl


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
    pooled = torch.zeros(
        (rows, 1, head_dim), dtype=raw_key_cache.dtype, device=device
    )
    first_positions = torch.zeros((rows, 4), dtype=torch.int64, device=device)
    if rows == 0:
        return pooled, first_positions[:, :3]

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
    return pooled, first_positions[:, :3]
