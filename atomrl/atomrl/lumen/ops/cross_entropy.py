# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
###############################################################################

"""Vocab-parallel cross-entropy loss — thin wrapper over AITER kernels.

The actual Triton kernel implementations live in
``aiter.ops.triton._triton_kernels.cross_entropy`` and the Python
forward/backward orchestration lives in
``aiter.ops.triton.cross_entropy``.

When ``use_sdma=True`` the TP all-gather is routed through mori SDMA
instead of ``torch.distributed.all_gather_into_tensor``.

This module re-exports a single ``parallel_cross_entropy`` autograd
function for vocab-parallel cross-entropy computation.
"""

from functools import reduce
from operator import mul

import torch
import torch.distributed as dist
import triton
from aiter.ops.triton._triton_kernels.cross_entropy import (
    _ce_fused_loss_grad_kernel,
    _ce_local_softmax_stats_kernel,
)
from aiter.ops.triton.cross_entropy import (
    cross_entropy_backward,
)
from aiter.ops.triton.cross_entropy import cross_entropy_forward as _aiter_ce_forward
from aiter.ops.triton.cross_entropy import (
    cross_entropy_forward_chunked as _aiter_ce_forward_chunked,
)

__all__ = ["parallel_cross_entropy"]

# AITER's forward validates every target with a host sync (``valid.all()``),
# which stalls the stream each step and cannot run inside CUDA-graph capture.
# Lumen never validated targets, so keep that: callers feed labels straight
# from the tokenizer, and an out-of-range label is a data bug rather than
# something to pay a sync per step to catch.
_VALIDATE_TARGETS = False

_WAVES_PER_EU = 2
_NUM_STAGES = 2


def _block_size_and_warps(element_size: int, n_cols: int):
    """Kernel tile width and warp count for a vocab shard of ``n_cols``.

    The tile is capped at 64 KiB of logits per row-block, and a warp covers 64
    lanes; this is the sizing ``aiter.ops.triton.cross_entropy`` launches its
    own kernels with, so the SDMA path below runs the same tiling.
    """
    block_size = min(65536 // element_size, triton.next_power_of_2(n_cols))
    num_warps = min(16, max(1, block_size // 64))
    return block_size, num_warps


def _n_valid(target: torch.Tensor, reduce_loss: bool, ignore_idx: int) -> torch.Tensor:
    """Row count that ``reduce_loss`` divides the loss and gradient by.

    AITER's mean reduction is over the *non-ignored* rows (matching
    ``F.cross_entropy``), kept on-device as a 0-d tensor so there is no sync.
    ``clamp_min(1)`` keeps an all-ignored batch finite.
    """
    if reduce_loss:
        return (target != ignore_idx).sum().clamp_min(1)
    return torch.ones((), dtype=torch.int64, device=target.device)


def _cross_entropy_forward_sdma(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: dist.ProcessGroup | None,
    ignore_idx: int,
    chunk_rows: int = 0,
):
    """SDMA-accelerated cross-entropy forward.

    Identical to the AITER ``cross_entropy_forward`` except the TP
    all-gather uses mori SDMA via :class:`SdmaTpComm`.

    When ``chunk_rows > 0`` the row dimension is split into chunks so that
    each chunk's allgather transfers only ``chunk_rows * 3`` floats instead
    of ``n_rows * 3``, reducing peak activation memory.
    """
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm

    B, SQ, V = _input.shape
    n_rows = B * SQ
    assert reduce(mul, list(target.size())) == n_rows
    BLOCK_SIZE, num_warps = _block_size_and_warps(_input.element_size(), V)

    # Full contiguity, not just stride(-1) == 1: both kernels walk B*SQ rows
    # with one row stride, so a tensor sliced along SQ would read the wrong rows.
    if not _input.is_contiguous():
        _input = _input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    rank = 0 if dist_group is None else dist.get_rank(dist_group)
    world_size = 1 if dist_group is None else dist.get_world_size(dist_group)

    # Same guard as aiter.ops.triton.cross_entropy: the smoothing term only sums
    # this rank's vocab shard, so it is wrong under tensor parallelism.
    assert not (label_smoothing > 0 and world_size > 1), (
        "label_smoothing > 0 is not supported with world_size > 1: the "
        "smoothing term only covers the local vocab shard"
    )

    n_valid = _n_valid(target, reduce_loss, ignore_idx)

    if chunk_rows > 0:
        return _cross_entropy_forward_sdma_chunked(
            _input,
            target,
            label_smoothing,
            reduce_loss,
            dist_group,
            ignore_idx,
            chunk_rows,
            B,
            SQ,
            V,
            n_rows,
            BLOCK_SIZE,
            num_warps,
            rank,
            world_size,
            n_valid,
        )

    # zeros, not empty: ignored rows never store a loss.
    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
    m_d_Xy = torch.empty(n_rows * 3, dtype=torch.float32, device=_input.device)

    _ce_local_softmax_stats_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        target,
        target.stride(-1),
        m_d_Xy,
        m_d_Xy.stride(-1),
        rank,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        waves_per_eu=_WAVES_PER_EU,
        num_stages=_NUM_STAGES,
    )

    if world_size > 1:
        comm = SdmaTpComm.get(dist_group)
        ag_result = comm._ag(m_d_Xy)  # (npes, n_rows * 3)
        gathered = ag_result.reshape(-1)  # (world_size * n_rows * 3,)
    else:
        gathered = m_d_Xy

    _ce_fused_loss_grad_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        target,
        target.stride(-1),
        loss_1d,
        loss_1d.stride(-1),
        gathered,
        gathered.stride(-1),
        rank,
        world_size,
        ignore_idx,
        V,
        n_rows,
        n_valid,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        waves_per_eu=_WAVES_PER_EU,
        num_stages=_NUM_STAGES,
    )

    loss = loss_1d.reshape(B, SQ) if not reduce_loss else (loss_1d.sum() / n_valid)
    return loss, _input


def _cross_entropy_forward_sdma_chunked(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: dist.ProcessGroup | None,
    ignore_idx: int,
    chunk_rows: int,
    B: int,
    SQ: int,
    V: int,
    n_rows: int,
    BLOCK_SIZE: int,
    num_warps: int,
    rank: int,
    world_size: int,
    n_valid: torch.Tensor,
):
    """SDMA chunked cross-entropy: row-wise chunks to cap peak buffer size.

    Each chunk allgathers only ``chunk_rows * 3`` floats via SDMA instead of
    the full ``n_rows * 3``.  The gradient is written in-place into ``_input``
    chunk by chunk.

    ``_TpSdmaAllgather._ensure`` grows the handle buffer on demand and reuses
    it across chunks — so the first chunk (or the largest, when the last chunk
    is smaller) pays the realloc cost; subsequent same-size chunks are free.
    """
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm

    input_2d = _input.reshape(n_rows, V)  # view
    target_1d = target.reshape(n_rows)  # view
    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)

    # Pre-allocate at full chunk size; last chunk may be smaller but the
    # SDMA handle will reuse capacity (no realloc for smaller slices).
    m_d_Xy = torch.empty(chunk_rows * 3, dtype=torch.float32, device=_input.device)

    comm = SdmaTpComm.get(dist_group) if world_size > 1 else None

    row = 0
    while row < n_rows:
        rows_this = min(chunk_rows, n_rows - row)
        chunk_x = input_2d[row : row + rows_this]  # view [rows_this, V]
        chunk_y = target_1d[row : row + rows_this]  # view [rows_this]
        chunk_loss = loss_1d[row : row + rows_this]  # view [rows_this]
        m_d_Xy_chunk = m_d_Xy[: rows_this * 3]

        _ce_local_softmax_stats_kernel[(rows_this,)](
            chunk_x,
            chunk_x.stride(0),
            chunk_y,
            chunk_y.stride(0),
            m_d_Xy_chunk,
            1,
            rank,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            waves_per_eu=_WAVES_PER_EU,
            num_stages=_NUM_STAGES,
        )

        if world_size > 1:
            # SDMA allgather: transfers rows_this*3 floats.
            # _ag._ensure grows on first call; subsequent same-size chunks reuse.
            ag_result = comm._ag(m_d_Xy_chunk)  # (npes, rows_this*3)
            gathered = ag_result.reshape(-1)  # (world_size*rows_this*3,)
        else:
            gathered = m_d_Xy_chunk

        # ``n_valid`` is the whole batch's denominator, shared by every chunk, so
        # the reduced gradient matches the non-chunked path.
        _ce_fused_loss_grad_kernel[(rows_this,)](
            chunk_x,
            chunk_x.stride(0),
            chunk_y,
            chunk_y.stride(0),
            chunk_loss,
            chunk_loss.stride(0),
            gathered,
            1,
            rank,
            world_size,
            ignore_idx,
            V,
            rows_this,
            n_valid,
            reduce_loss=reduce_loss,
            label_smoothing=label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            waves_per_eu=_WAVES_PER_EU,
            num_stages=_NUM_STAGES,
        )

        row += rows_this

    loss = loss_1d.reshape(B, SQ) if not reduce_loss else (loss_1d.sum() / n_valid)
    return loss, _input


class _ParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
        is_cg_capturable=False,
        use_sdma=False,
        chunk_rows=0,
    ):
        # AITER Triton kernels only check stride(-1) == 1 before using
        # stride(-2) as the row stride.  A transposed-but-last-dim-contiguous
        # tensor (e.g. from .transpose(0,1)) can have a huge stride(-2) that
        # causes out-of-bounds GPU memory reads.  Ensure full contiguity here.
        if not _input.is_contiguous():
            _input = _input.contiguous()

        if use_sdma and dist_process_group is not None:
            loss, grad_input = _cross_entropy_forward_sdma(
                _input,
                target,
                label_smoothing,
                reduce_loss,
                dist_process_group,
                ignore_idx,
                chunk_rows,
            )
        elif chunk_rows > 0:
            loss, grad_input = _aiter_ce_forward_chunked(
                _input,
                target,
                label_smoothing,
                reduce_loss,
                dist_process_group,
                ignore_idx,
                chunk_rows,
                validate_targets=_VALIDATE_TARGETS,
            )
        else:
            loss, grad_input = _aiter_ce_forward(
                _input,
                target,
                label_smoothing,
                reduce_loss,
                dist_process_group,
                ignore_idx,
                validate_targets=_VALIDATE_TARGETS,
            )
        ctx.save_for_backward(grad_input.detach())
        ctx.is_cg_capturable = is_cg_capturable
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input,) = ctx.saved_tensors
        grad_input = cross_entropy_backward(
            grad_input, grad_output, ctx.is_cg_capturable
        )
        # extra None for chunk_rows
        return grad_input, None, None, None, None, None, None, None, None


parallel_cross_entropy = _ParallelCrossEntropy.apply
