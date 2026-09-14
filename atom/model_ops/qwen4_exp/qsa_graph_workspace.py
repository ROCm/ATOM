# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grow-once workspaces for QSA ops under CUDA-graph decode.

SGLang decode CUDA graphs capture Triton kernel pointer operands. Any
``torch.empty`` / ``torch.zeros`` allocated *inside* the captured forward
becomes a dangling alias after a long eager prefill reuses that caching-
allocator block — which is exactly the long-context first-decode HSA fault
mode (short greedy still works because nothing large reallocates the slab).

These buffers are reserved to a high-water mark before capture and only return
views; they are never freed or replaced after the first reservation.
"""

from __future__ import annotations

import threading

import torch

_lock = threading.Lock()
_enabled = False
_device: torch.device | None = None
_max_tokens = 0
_max_columns = 0
_max_block_topk = 0
_head_dim = 0
_num_q_heads = 0

_logits: torch.Tensor | None = None
_visible_groups: torch.Tensor | None = None
_row_starts: torch.Tensor | None = None
_selected_groups: torch.Tensor | None = None
_pooled: torch.Tensor | None = None
_first_positions: torch.Tensor | None = None
_attn_out: torch.Tensor | None = None


def enable() -> None:
    global _enabled
    _enabled = True


def disable() -> None:
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


def max_tokens() -> int:
    return _max_tokens


def fits(
    rows: int,
    *,
    columns: int | None = None,
    block_topk: int | None = None,
    head_dim: int | None = None,
    num_q_heads: int | None = None,
) -> bool:
    """True when graph workspaces are active and large enough for this call.

    After ``reserve()`` the buffers stay allocated for decode CUDA-graph
    capture/replay (bs<=max_tokens). Eager prefill can still run with far
    more tokens — those paths must fall back to ephemeral allocs, otherwise
    views silently truncate (``[:rows]`` with rows>_max_tokens) and QSA
    shape checks fail mid-extend.
    """
    if not _enabled or _logits is None:
        return False
    if int(rows) > _max_tokens:
        return False
    if columns is not None and int(columns) > _max_columns:
        return False
    if block_topk is not None and int(block_topk) > _max_block_topk:
        return False
    if head_dim is not None and int(head_dim) > _head_dim:
        return False
    if num_q_heads is not None and int(num_q_heads) > _num_q_heads:
        return False
    return True


def reserve(
    *,
    max_tokens: int,
    max_columns: int,
    max_block_topk: int,
    head_dim: int,
    num_q_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Reserve (or grow once) all decode workspaces before CUDA-graph capture."""
    global _device, _max_tokens, _max_columns, _max_block_topk, _head_dim, _num_q_heads
    global _logits, _visible_groups, _row_starts, _selected_groups, _pooled, _first_positions, _attn_out

    max_tokens = max(int(max_tokens), 1)
    max_columns = max(int(max_columns), 1)
    max_block_topk = max(int(max_block_topk), 1)
    head_dim = max(int(head_dim), 1)
    num_q_heads = max(int(num_q_heads), 1)

    with _lock:
        need = (
            _logits is None
            or _device != device
            or _max_tokens < max_tokens
            or _max_columns < max_columns
            or _max_block_topk < max_block_topk
            or _head_dim < head_dim
            or _num_q_heads < num_q_heads
        )
        if not need:
            enable()
            return
        if _logits is not None:
            # Growing after capture leaves older graphs pointing at freed
            # storage — callers must reserve the high-water mark up front.
            import logging

            logging.getLogger(__name__).warning(
                "QSA graph workspace grew after init "
                "(tokens=%s cols=%s); existing CUDA graphs may be stale",
                max(_max_tokens, max_tokens),
                max(_max_columns, max_columns),
            )
        _max_tokens = max(_max_tokens, max_tokens)
        _max_columns = max(_max_columns, max_columns)
        _max_block_topk = max(_max_block_topk, max_block_topk)
        _head_dim = max(_head_dim, head_dim)
        _num_q_heads = max(_num_q_heads, num_q_heads)
        _device = device
        _logits = torch.empty(
            (_max_tokens, _max_columns), dtype=torch.float32, device=device
        )
        _visible_groups = torch.zeros(
            (_max_tokens,), dtype=torch.int32, device=device
        )
        _row_starts = torch.zeros((_max_tokens,), dtype=torch.int32, device=device)
        _selected_groups = torch.empty(
            (_max_tokens, _max_block_topk), dtype=torch.int32, device=device
        )
        _pooled = torch.zeros(
            (_max_tokens, 1, _head_dim), dtype=dtype, device=device
        )
        _first_positions = torch.zeros(
            (_max_tokens, 4), dtype=torch.int64, device=device
        )
        _attn_out = torch.empty(
            (_max_tokens, _num_q_heads, _head_dim), dtype=dtype, device=device
        )
        enable()


def logits(rows: int, columns: int) -> torch.Tensor:
    assert _logits is not None
    return _logits[:rows, :columns]


def visible_groups(rows: int) -> torch.Tensor:
    assert _visible_groups is not None
    return _visible_groups[:rows]


def row_starts(rows: int) -> torch.Tensor:
    assert _row_starts is not None
    return _row_starts[:rows]


def selected_groups(rows: int, block_topk: int) -> torch.Tensor:
    assert _selected_groups is not None
    return _selected_groups[:rows, :block_topk]


def pooled(rows: int, head_dim: int, dtype: torch.dtype) -> torch.Tensor:
    assert _pooled is not None
    if _pooled.dtype != dtype:
        # dtype mismatch is rare; fall back to a fresh buffer rather than
        # silently casting a graph-captured allocation.
        return torch.zeros((rows, 1, head_dim), dtype=dtype, device=_pooled.device)
    return _pooled[:rows, :, :head_dim]


def first_positions(rows: int) -> torch.Tensor:
    assert _first_positions is not None
    return _first_positions[:rows]


def attn_out(
    rows: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    assert _attn_out is not None
    if _attn_out.dtype != dtype:
        return torch.empty(
            (rows, num_heads, head_dim), dtype=dtype, device=_attn_out.device
        )
    return _attn_out[:rows, :num_heads, :head_dim]
