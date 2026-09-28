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
    mask_arg = has_initial_state.to(torch.int8) if has_mask else src
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
