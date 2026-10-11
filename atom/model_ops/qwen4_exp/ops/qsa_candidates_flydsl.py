# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL packing around QSA's index-only context-parallel exchange.

The score kernel and the two stable top-k calls stay unchanged. These kernels
only move their candidates into and out of the TP all-gather: a packed score
bit pattern and global group id per slot, then rank-major to row-major scores
and ids for the global selector. Each output slot is written on every replay.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64, Stream, T

from atom.mono.device.ops import rsrc

THREADS = 256
_CURRENT_STREAM = Stream(None)


@functools.cache
def _build(rows: int, topk: int, world: int, columns: int):
    if min(rows, topk, world, columns) < 1:
        raise ValueError("QSA candidate geometry must be positive")
    local_slots = rows * topk
    merged_width = world * topk
    merged_slots = rows * merged_width

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def pack_kernel(
        logits: Int64,
        chosen: Int64,
        visible: Int64,
        payload: Int64,
        row_stride: Int32,
        first_group: Int32,
    ):
        index = gpu.block_idx.x * THREADS + gpu.thread_idx.x
        if index < local_slots:
            row, slot = index // topk, index % topk
            selected = fx.Int32(
                bo.buffer_load(
                    rsrc(chosen), row * topk + slot, vec_width=1, dtype=T.i32
                )
            )
            width = fx.Int32(
                bo.buffer_load(rsrc(visible), row, vec_width=1, dtype=T.i32)
            )
            valid = (selected >= 0) & (selected < width)
            column = selected
            if column < 0:
                column = fx.Int32(0)
            if column >= columns:
                column = fx.Int32(columns - 1)
            score = fx.Float32(
                bo.buffer_load(
                    rsrc(logits),
                    row * row_stride + column,
                    vec_width=1,
                    dtype=T.f32,
                )
            )
            group = selected + first_group
            if not valid:
                score = fx.Float32(-float("inf"))
                group = fx.Int32(-1)
            bo.buffer_store(
                fx.Vector.from_elements([score.bitcast(fx.Int32), group], fx.Int32),
                rsrc(payload),
                index * 2,
            )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def unpack_kernel(
        gathered: Int64,
        scores: Int64,
        group_ids: Int64,
        lengths: Int64,
    ):
        index = gpu.block_idx.x * THREADS + gpu.thread_idx.x
        if index < merged_slots:
            row, candidate = index // merged_width, index % merged_width
            rank, slot = candidate // topk, candidate % topk
            source = ((rank * rows + row) * topk + slot) * 2
            pair = fx.Vector(
                bo.buffer_load(rsrc(gathered), source, vec_width=2, dtype=T.i32)
            )
            bo.buffer_store(pair[0].bitcast(fx.Float32), rsrc(scores), index)
            bo.buffer_store(pair[1], rsrc(group_ids), index)
            if candidate == 0:
                bo.buffer_store(fx.Int32(merged_width), rsrc(lengths), row)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def gather_kernel(group_ids: Int64, offsets: Int64, selected: Int64):
        index = gpu.block_idx.x * THREADS + gpu.thread_idx.x
        if index < local_slots:
            row = index // topk
            offset = fx.Int32(
                bo.buffer_load(rsrc(offsets), index, vec_width=1, dtype=T.i32)
            )
            group = fx.Int32(
                bo.buffer_load(
                    rsrc(group_ids),
                    row * merged_width + offset,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            bo.buffer_store(group, rsrc(selected), index)

    @flyc.jit
    def launch_pack(
        logits: Int64,
        chosen: Int64,
        visible: Int64,
        payload: Int64,
        row_stride: Int32,
        first_group: Int32,
        stream: Stream = _CURRENT_STREAM,
    ):
        pack_kernel(logits, chosen, visible, payload, row_stride, first_group).launch(
            grid=((local_slots + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_unpack(
        gathered: Int64,
        scores: Int64,
        group_ids: Int64,
        lengths: Int64,
        stream: Stream = _CURRENT_STREAM,
    ):
        unpack_kernel(gathered, scores, group_ids, lengths).launch(
            grid=((merged_slots + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_gather(
        group_ids: Int64,
        offsets: Int64,
        selected: Int64,
        stream: Stream = _CURRENT_STREAM,
    ):
        gather_kernel(group_ids, offsets, selected).launch(
            grid=((local_slots + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    for name, launch in (
        ("pack", launch_pack),
        ("unpack", launch_unpack),
        ("gather", launch_gather),
    ):
        launch.func.__name__ = (
            f"qwen38_qsa_candidate_{name}_r{rows}_k{topk}_w{world}_c{columns}"
        )
    return launch_pack, launch_unpack, launch_gather


def pack_candidates(
    logits: torch.Tensor,
    selected: torch.Tensor,
    visible: torch.Tensor,
    first_group: int,
    world: int,
) -> torch.Tensor:
    """Build [rows, topk, 2] score-bit/id candidates in one launch."""
    rows, topk = selected.shape
    columns = logits.shape[1]
    payload = torch.empty((rows, topk, 2), dtype=torch.int32, device=logits.device)
    pack, _, _ = _build(rows, topk, world, columns)
    pack(
        logits.data_ptr(),
        selected.data_ptr(),
        visible.data_ptr(),
        payload.data_ptr(),
        logits.stride(0),
        first_group,
        stream=torch.cuda.current_stream(logits.device),
    )
    return payload


def unpack_candidates(gathered: torch.Tensor):
    """Rank-major exchange to row-major contiguous scores, ids and lengths."""
    world, rows, topk, pair = gathered.shape
    if pair != 2:
        raise ValueError("QSA candidate payload must have score/id pairs")
    scores = torch.empty(
        (rows, world * topk), dtype=torch.float32, device=gathered.device
    )
    group_ids = torch.empty(
        (rows, world * topk), dtype=torch.int32, device=gathered.device
    )
    lengths = torch.empty((rows,), dtype=torch.int32, device=gathered.device)
    _, unpack, _ = _build(rows, topk, world, world * topk)
    unpack(
        gathered.data_ptr(),
        scores.data_ptr(),
        group_ids.data_ptr(),
        lengths.data_ptr(),
        stream=torch.cuda.current_stream(gathered.device),
    )
    return scores, group_ids, lengths


def gather_candidates(group_ids: torch.Tensor, offsets: torch.Tensor):
    """Map global top-k offsets to their group ids in one launch."""
    rows, topk = offsets.shape
    world = group_ids.shape[1] // topk
    selected = torch.empty_like(offsets)
    _, _, gather = _build(rows, topk, world, group_ids.shape[1])
    gather(
        group_ids.data_ptr(),
        offsets.data_ptr(),
        selected.data_ptr(),
        stream=torch.cuda.current_stream(offsets.device),
    )
    return selected
