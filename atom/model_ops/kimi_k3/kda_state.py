# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused KDA initial-state gathering for Kimi-K3."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _gather_kda_state_kernel(
        src_ptr,  # ssm_state viewed as [num_lines, S]
        idx_ptr,  # state_indices [num_seqs]
        mask_ptr,  # has_initial_state [num_seqs] int8 (unused when HAS_MASK=False)
        dst_ptr,  # initial viewed as [num_seqs, S]
        S,
        stride_src,
        stride_dst,
        HAS_MASK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        seq = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        col_mask = offs < S
        if HAS_MASK:
            keep = tl.load(mask_ptr + seq) != 0
            load_mask = col_mask & keep
        else:
            load_mask = col_mask
        # int64 offsets: line * S can exceed int32 for large state caches.
        line = tl.load(idx_ptr + seq).to(tl.int64)
        vals = tl.load(src_ptr + line * stride_src + offs, mask=load_mask, other=0.0)
        tl.store(dst_ptr + seq.to(tl.int64) * stride_dst + offs, vals, mask=col_mask)

    @triton.jit
    def _gather_kda_rows_kernel(
        src_ptr,  # ssm_state [num_lines, H, V, K], each [H, V, K] plane dense
        idx_ptr,  # state_indices [num_rows]
        mask_ptr,  # has_initial_state [num_rows] bytes (unused when HAS_MASK=False)
        rows_ptr,  # [num_seqs]: the row each sequence starts from
        dst_ptr,  # [num_seqs, H, V, K], or [num_seqs, H, K, V] when K_FIRST
        stride_src,
        H,
        V: tl.constexpr,
        K: tl.constexpr,
        BV: tl.constexpr,
        HAS_MASK: tl.constexpr,
        K_FIRST: tl.constexpr,
    ):
        seq = tl.program_id(0).to(tl.int64)
        plane = tl.program_id(1).to(tl.int64) * V * K
        v = tl.program_id(2) * BV + tl.arange(0, BV)
        k = tl.arange(0, K)
        r = tl.load(rows_ptr + seq)
        line = tl.load(idx_ptr + r).to(tl.int64)
        src = src_ptr + line * stride_src + plane + v[:, None] * K + k[None, :]
        if HAS_MASK:
            vals = tl.zeros([BV, K], dtype=dst_ptr.dtype.element_ty)
            if tl.load(mask_ptr + r) != 0:
                vals = tl.load(src)
        else:
            vals = tl.load(src)
        dst = dst_ptr + seq * H * V * K + plane
        if K_FIRST:
            tl.store(dst + k[None, :] * V + v[:, None], vals)
        else:
            tl.store(dst + v[:, None] * K + k[None, :], vals)


def _mask_bytes(has_initial_state: torch.Tensor) -> torch.Tensor:
    """``has_initial_state`` as the kernels read it: one byte per sequence,
    non-zero to keep. A dense bool is that already."""
    if has_initial_state.dtype == torch.bool and has_initial_state.is_contiguous():
        return has_initial_state.view(torch.uint8)
    return has_initial_state.to(torch.int8)


def gather_kda_initial_state(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather ``ssm_state[state_indices]`` into a packed ``[num_seqs, ...]``
    initial-state tensor, zeroing sequences whose ``has_initial_state`` is
    False -- in a single kernel.

    Fuses the ``ssm_state[idx].contiguous()`` gather with the
    ``initial[~has_initial_state] = 0`` masking so fresh sequences are written
    as zeros in the same pass instead of a gather followed by a separate
    zero-write pass. Falls back to the torch path when triton is unavailable or
    there are no sequences.
    """
    num_seqs = int(state_indices.shape[0])
    if not _HAS_TRITON or num_seqs == 0:
        initial = ssm_state[state_indices].contiguous()
        if has_initial_state is not None:
            initial[~has_initial_state] = 0
        return initial
    tail = ssm_state.shape[1:]
    src = ssm_state.reshape(ssm_state.shape[0], -1)
    S = src.shape[1]
    initial = torch.empty(
        (num_seqs, *tail), dtype=ssm_state.dtype, device=ssm_state.device
    )
    dst = initial.reshape(num_seqs, -1)
    has_mask = has_initial_state is not None
    mask_arg = _mask_bytes(has_initial_state) if has_mask else src
    BLOCK = 1024
    grid = (num_seqs, triton.cdiv(S, BLOCK))
    _gather_kda_state_kernel[grid](
        src,
        state_indices,
        mask_arg,
        dst,
        S,
        src.stride(0),
        dst.stride(0),
        HAS_MASK=has_mask,
        BLOCK=BLOCK,
    )
    return initial


# V rows per program of `_gather_kda_rows_kernel`.
_ROWS_BV = 32


def gather_kda_initial_rows(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    rows: torch.Tensor,
    k_first: bool = False,
) -> torch.Tensor:
    """``gather_kda_initial_state(ssm_state, state_indices,
    has_initial_state).index_select(0, rows)`` in one kernel: sequence ``n``
    starts from row ``rows[n]``'s state, zero where that row has none.

    With ``k_first`` each ``[V, K]`` plane is stored transposed and returned as
    a V-first view, the layout `flash_kda_fwd` normalizes a V-first initial
    state to, so its ``contiguous()`` copy becomes a no-op.
    """
    _, _, V, K = ssm_state.shape
    if (
        not _HAS_TRITON
        or not ssm_state.is_cuda
        or rows.shape[0] == 0
        or not ssm_state[0].is_contiguous()
        or V % _ROWS_BV
        or K & (K - 1)
    ):
        initial = ssm_state[state_indices[rows]]
        if has_initial_state is not None:
            initial[~has_initial_state[rows]] = 0
        if k_first:
            initial = initial.transpose(-1, -2).contiguous().transpose(-1, -2)
        return initial
    return _gather_rows(ssm_state, state_indices, has_initial_state, rows, k_first)


def _gather_rows(ssm_state, state_indices, has_initial_state, rows, k_first):
    num_seqs = int(rows.shape[0])
    _, H, V, K = ssm_state.shape
    BV = _ROWS_BV
    shape = (num_seqs, H, K, V) if k_first else (num_seqs, H, V, K)
    initial = torch.empty(shape, dtype=ssm_state.dtype, device=ssm_state.device)
    has_mask = has_initial_state is not None
    _gather_kda_rows_kernel[(num_seqs, H, V // BV)](
        ssm_state,
        state_indices,
        _mask_bytes(has_initial_state) if has_mask else state_indices,
        rows,
        initial,
        ssm_state.stride(0),
        H,
        V=V,
        K=K,
        BV=BV,
        HAS_MASK=has_mask,
        K_FIRST=k_first,
    )
    return initial.transpose(-1, -2) if k_first else initial
