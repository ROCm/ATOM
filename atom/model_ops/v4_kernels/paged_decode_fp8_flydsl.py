# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL candidates for V4 native-FP8 decode.

Decode and sparse prefill share the same per-query attention equation.  This
first no-split candidate reuses the true FlyDSL H=128 sparse-prefill kernel by
placing the decode KV list in its prefix segment and passing an empty extend
segment.  It intentionally does not allocate temporary tensors in the timed
path: callers provide the zero-length index view and the zero indptr buffer.

The split1 adapter is useful for CSA B13+ exploration.  The HCA path can finish
with either the existing Triton reducer or a pure FlyDSL wave-group reducer.
The production adapter builds a fixed-size split task map on the GPU, so both
the launch sequence and tensor addresses are safe for CUDA Graph replay.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
import triton.language as tl
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_buf_tensor

from atom.model_ops.v4_kernels.paged_decode import LOG2E, _paged_decode_reduce_kernel
from atom.model_ops.v4_kernels.paged_prefill_fp8_flydsl import (
    V4_DIM_QK,
    sparse_attn_v4_paged_prefill_fp8_flydsl,
)


@triton.jit
def _build_graphsafe_split_tasks_kernel(
    kv_indptr_ptr,
    split_task_query_ptr,
    split_task_start_ptr,
    split_task_len_ptr,
    split_task_row_ptr,
    split_counts_ptr,
    NUM_TASKS: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_TILES: tl.constexpr,
    SPLIT_TILES_SHORT: tl.constexpr,
    SPLIT_SHORT_MAX_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build one fixed task slot per ``(query, split)`` entirely on device.

    Inactive split slots carry ``length=0``.  The FlyDSL stage therefore uses
    the same maximum grid during graph capture and replay while the reducer
    consumes only ``split_counts[query]`` live rows.
    """
    task = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    task_mask = task < NUM_TASKS
    query = task // MAX_SPLITS
    split = task % MAX_SPLITS

    kv_start = tl.load(kv_indptr_ptr + query, mask=task_mask, other=0)
    kv_end = tl.load(kv_indptr_ptr + query + 1, mask=task_mask, other=0)
    kv_len = kv_end - kv_start
    num_tiles = tl.cdiv(kv_len, BLOCK_K)
    segment_tiles = tl.full(task.shape, SPLIT_TILES, tl.int32)
    if SPLIT_TILES_SHORT > 0:
        segment_tiles = tl.where(
            num_tiles <= SPLIT_SHORT_MAX_TILES,
            SPLIT_TILES_SHORT,
            segment_tiles,
        )
    active_splits = tl.minimum(tl.cdiv(num_tiles, segment_tiles), MAX_SPLITS)
    active = task_mask & (split < active_splits)
    segment_rows = segment_tiles * BLOCK_K
    local_start = split * segment_rows
    local_end = tl.where(
        split == MAX_SPLITS - 1,
        kv_len,
        tl.minimum(local_start + segment_rows, kv_len),
    )

    tl.store(split_task_query_ptr + task, query, mask=task_mask)
    tl.store(
        split_task_start_ptr + task,
        tl.where(active, kv_start + local_start, kv_start),
        mask=task_mask,
    )
    tl.store(
        split_task_len_ptr + task,
        tl.where(active, tl.maximum(local_end - local_start, 0), 0),
        mask=task_mask,
    )
    tl.store(split_task_row_ptr + task, task, mask=task_mask)
    tl.store(split_counts_ptr + query, active_splits, mask=task_mask & (split == 0))


@lru_cache(maxsize=32)
def _build_split_reduce_launcher(
    max_splits: int,
    head_group: int = 1,
    waves_per_eu: int = 0,
):
    """Build the fixed-reference H128/D512 FlyDSL split reducer."""

    if head_group not in (1, 2, 4, 8):
        raise ValueError("head_group must be 1, 2, 4, or 8")
    threads_per_head = 64
    block_size = threads_per_head * head_group
    head_groups = 128 // head_group

    @flyc.kernel(
        name=(
            f"v4_fp8_flydsl_decode_reduce_s{max_splits}_hg{head_group}"
            + (f"_weu{waves_per_eu}" if waves_per_eu else "")
        ),
        known_block_size=[block_size, 1, 1],
    )
    def kernel(
        partial_l: fx.Tensor,
        partial_acc: fx.Tensor,
        attn_sink: fx.Tensor,
        split_counts: fx.Tensor,
        out: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        work = fx.Int32(fx.block_idx.x)
        query = work // fx.Int32(head_groups)
        head_base = (work % fx.Int32(head_groups)) * fx.Int32(head_group)
        head_offset = tid // fx.Int32(threads_per_head)
        head_lane = tid % fx.Int32(threads_per_head)
        head = head_base + head_offset
        l_f32 = ptr_buf_tensor(partial_l, fx.Float32)
        acc_f16 = ptr_buf_tensor(partial_acc, fx.Float16)
        sink_f32 = ptr_buf_tensor(attn_sink, fx.Float32)
        counts_i32 = ptr_buf_tensor(split_counts, fx.Int32)
        out_bf16 = ptr_buf_tensor(out, fx.BFloat16)
        num_splits = fx.Int32(counts_i32[query])

        scale = fx.Float32(0.0)
        if head_lane == fx.Int32(0):
            l_total = fx.Float32(0.0)
            split = fx.Int32(0)
            while split < num_splits:
                meta_offset = (query * fx.Int32(max_splits) + split) * fx.Int32(
                    128
                ) + head
                l_total = l_total + fx.Float32(l_f32[meta_offset])
                split = split + fx.Int32(1)
            sink_weight = fx.rocdl.exp2(
                T.f32,
                (fx.Float32(sink_f32[head]) * fx.Float32(LOG2E)).ir_value(),
            )
            scale = fx.Float32(1.0) / (l_total + sink_weight)
        scale = fx.Int32(
            fx.rocdl.ds_bpermute(
                T.i32,
                fx.Int32(0).ir_value(),
                scale.bitcast(fx.Int32).ir_value(),
            )
        ).bitcast(fx.Float32)
        d_steps = V4_DIM_QK // threads_per_head
        accumulators = [fx.Float32(0.0) for _ in fx.range_constexpr(d_steps)]
        split = fx.Int32(0)
        while split < num_splits:
            row = query * fx.Int32(max_splits) + split
            base = (row * fx.Int32(128) + head) * fx.Int32(V4_DIM_QK)
            for d_step in fx.range_constexpr(d_steps):
                d = head_lane + fx.Int32(d_step * threads_per_head)
                accumulators[d_step] = accumulators[d_step] + fx.Float32(
                    acc_f16[base + d]
                )
            split = split + fx.Int32(1)
        out_base = (query * fx.Int32(128) + head) * fx.Int32(V4_DIM_QK)
        for d_step in fx.range_constexpr(d_steps):
            d = head_lane + fx.Int32(d_step * threads_per_head)
            out_bf16[out_base + d] = (accumulators[d_step] * scale).to(fx.BFloat16)

    kernel_value_attrs = {
        "rocdl.flat_work_group_size": f"{block_size},{block_size}",
    }
    if waves_per_eu > 0:
        kernel_value_attrs["rocdl.waves_per_eu"] = int(waves_per_eu)

    @flyc.jit
    def launch(
        partial_l: fx.Tensor,
        partial_acc: fx.Tensor,
        attn_sink: fx.Tensor,
        split_counts: fx.Tensor,
        out: fx.Tensor,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(partial_l, partial_acc, attn_sink, split_counts, out).launch(
            grid=(grid_x, 1, 1),
            block=(block_size, 1, 1),
            stream=stream,
            value_attrs=kernel_value_attrs,
        )

    return launch


def sparse_attn_v4_paged_decode_fp8_flydsl_split1(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    kv_packed: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    empty_kv_indices: torch.Tensor,
    empty_kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    *,
    out: torch.Tensor | None = None,
    assume_full_tiles: bool = False,
    waves_per_eu: int = 0,
) -> torch.Tensor:
    """Run the benchmark-only H=128, no-split FlyDSL decode candidate.

    ``empty_kv_indices`` must contain zero elements and
    ``empty_kv_indptr`` must be an all-zero int32 tensor with ``T + 1``
    elements.  ``assume_full_tiles`` additionally promises that every decode
    KV length is divisible by 32.
    """
    tokens, heads, _ = q_packed.shape
    if heads != 128:
        raise ValueError("the initial FlyDSL decode candidate requires H=128")
    if kv_indptr.numel() != tokens + 1:
        raise ValueError("kv_indptr must contain T+1 elements")
    if empty_kv_indices.numel() != 0:
        raise ValueError("empty_kv_indices must contain zero elements")
    if empty_kv_indptr.dtype != torch.int32 or empty_kv_indptr.numel() != tokens + 1:
        raise ValueError("empty_kv_indptr must be int32 with T+1 elements")

    return sparse_attn_v4_paged_prefill_fp8_flydsl(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        kv_indices,
        kv_indptr,
        kv_packed,
        kv_rope,
        empty_kv_indices,
        empty_kv_indptr,
        attn_sink,
        softmax_scale,
        out=out,
        waves_per_eu=waves_per_eu,
        pipeline_two=True,
        wave_padded_k=True,
        lds_padding=4,
        fixed_softmax_ref=True,
        transpose_v=True,
        reuse_q_across_n=True,
        no_sentinel=True,
        full_tile_fastpath=True,
        cache_all_q=True,
        assume_full_tiles=assume_full_tiles,
        permute_k_scales=True,
        pairwise_pv=True,
        prefix_only=True,
    )


def sparse_attn_v4_paged_decode_fp8_flydsl_splitk(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    kv_packed: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    empty_kv_indices: torch.Tensor,
    empty_kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    split_task_query: torch.Tensor,
    split_task_start: torch.Tensor,
    split_task_len: torch.Tensor,
    split_task_row: torch.Tensor,
    split_counts: torch.Tensor,
    *,
    max_splits: int,
    split_tiles: int,
    split_tiles_short: int = 0,
    split_short_max_tiles: int = 0,
    block_k: int = 32,
    reduce_d_chunk: int = 512,
    reduce_num_warps: int = 1,
    reduce_head_group: int = 1,
    sequential_reduce: bool = True,
    use_flydsl_reduce: bool = False,
    partial_m: torch.Tensor | None = None,
    partial_l: torch.Tensor | None = None,
    partial_acc: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    waves_per_eu: int = 0,
) -> torch.Tensor:
    """Run the FlyDSL split-K decode kernel over a caller-provided task map.

    The task arrays may be either a host-planned compact list of active
    ``(query, split-start, split-length, partial-row)`` entries.  This first
    structural prototype can also consume the fixed-size map emitted by
    :func:`sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe`. Stage 2 can use
    either the proven Triton reducer or the H128 pure FlyDSL reducer.
    """
    tokens, heads, _ = q_packed.shape
    if heads != 128:
        raise ValueError("the FlyDSL split-K candidate requires H=128")
    if max_splits <= 1:
        raise ValueError("max_splits must be greater than one")
    if block_k != 32:
        raise ValueError("the current FlyDSL stage1 uses a fixed K32 tile")
    if reduce_d_chunk not in (64, 128, 256, 512):
        raise ValueError("reduce_d_chunk must be 64, 128, 256, or 512")
    if reduce_head_group not in (1, 2, 4, 8):
        raise ValueError("reduce_head_group must be 1, 2, 4, or 8")
    task_count = split_task_query.numel()
    if not (
        split_task_start.numel()
        == split_task_len.numel()
        == split_task_row.numel()
        == task_count
    ):
        raise ValueError("split task tensors must have matching lengths")
    if (
        split_counts.dtype != torch.int32
        or split_counts.shape != (tokens,)
        or not split_counts.is_contiguous()
    ):
        raise ValueError("split_counts must be contiguous int32 shape [T]")

    partial_shape = (tokens, max_splits, heads)
    if partial_m is None:
        partial_m = torch.empty(
            partial_shape, dtype=torch.float32, device=q_packed.device
        )
    if partial_l is None:
        partial_l = torch.empty_like(partial_m)
    if partial_acc is None:
        partial_acc = torch.empty(
            (*partial_shape, V4_DIM_QK),
            dtype=torch.float16,
            device=q_packed.device,
        )
    if out is None:
        out = torch.empty(
            (tokens, heads, V4_DIM_QK),
            dtype=torch.bfloat16,
            device=q_packed.device,
        )

    sparse_attn_v4_paged_prefill_fp8_flydsl(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        kv_indices,
        kv_indptr,
        kv_packed,
        kv_rope,
        empty_kv_indices,
        empty_kv_indptr,
        attn_sink,
        softmax_scale,
        out=partial_acc,
        waves_per_eu=waves_per_eu,
        pipeline_two=True,
        wave_padded_k=True,
        lds_padding=4,
        fixed_softmax_ref=True,
        transpose_v=True,
        reuse_q_across_n=True,
        no_sentinel=True,
        full_tile_fastpath=True,
        cache_all_q=True,
        permute_k_scales=True,
        pairwise_pv=True,
        prefix_only=True,
        split_partial=True,
        split_task_query=split_task_query,
        split_task_start=split_task_start,
        split_task_len=split_task_len,
        split_task_row=split_task_row,
        partial_m=partial_m,
        partial_l=partial_l,
    )

    if use_flydsl_reduce:
        stream = torch.cuda.current_stream(q_packed.device)
        with torch.cuda.device(q_packed.device.index):
            with CompilationContext.compile_hints(
                {"fast_fp_math": True, "unsafe_fp_math": True}
            ):
                _run_compiled(
                    _build_split_reduce_launcher(
                        max_splits,
                        reduce_head_group,
                    ),
                    partial_l,
                    partial_acc,
                    attn_sink,
                    split_counts,
                    out,
                    int(tokens * heads // reduce_head_group),
                    stream,
                )
        return out

    reduce_grid = (tokens, heads, triton.cdiv(V4_DIM_QK, reduce_d_chunk))
    _paged_decode_reduce_kernel[reduce_grid](
        partial_m,
        partial_l,
        partial_acc,
        attn_sink,
        kv_indptr,
        out,
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_l.stride(0),
        partial_l.stride(1),
        partial_l.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        LOG2E,
        heads,
        V4_DIM_QK,
        max_splits,
        BLOCK_D=V4_DIM_QK,
        D_CHUNK=reduce_d_chunk,
        BLOCK_K=block_k,
        SPLIT_TILES=split_tiles,
        SPLIT_TILES_SHORT=split_tiles_short,
        SPLIT_SHORT_MAX_TILES=split_short_max_tiles,
        SPLIT_TILES_MID=0,
        SPLIT_MID_MAX_TILES=0,
        PACKED_QUERY_GROUP=0,
        PACKED_UNIFORM_SPLITS=0,
        SEQUENTIAL_FALLBACK=sequential_reduce,
        num_warps=reduce_num_warps,
    )
    return out


def sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    kv_packed: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    empty_kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    *,
    max_splits: int = 12,
    block_k: int = 32,
    split_tiles: int = 16,
    split_tiles_short: int = 9,
    split_short_max_tiles: int = 64,
    reduce_head_group: int = 4,
    waves_per_eu: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the production-shaped H128 FlyDSL split-K decode path.

    A small Triton planner writes all ``T * max_splits`` task slots on the GPU.
    The stage-1 grid is therefore fixed by captured tensor shapes, and inactive
    slots use a zero-length segment.  This avoids the host ``.tolist()`` and
    active-sized launch used by the tuning harness while preserving its split
    policy and pure-FlyDSL reducer.
    """
    tokens, heads, _ = q_packed.shape
    if heads != 128:
        raise ValueError("the graph-safe FlyDSL decode path requires H=128")
    if kv_indptr.dtype != torch.int32 or kv_indptr.numel() != tokens + 1:
        raise ValueError("kv_indptr must be int32 with T+1 elements")
    if empty_kv_indptr.dtype != torch.int32 or empty_kv_indptr.numel() != tokens + 1:
        raise ValueError("empty_kv_indptr must be int32 with T+1 elements")
    if max_splits <= 1:
        raise ValueError("max_splits must be greater than one")
    if block_k != 32:
        raise ValueError("the production FlyDSL stage1 uses a fixed K32 tile")

    num_tasks = tokens * max_splits
    task_shape = (num_tasks,)
    split_task_query = torch.empty(
        task_shape, dtype=torch.int32, device=q_packed.device
    )
    split_task_start = torch.empty_like(split_task_query)
    split_task_len = torch.empty_like(split_task_query)
    split_task_row = torch.empty_like(split_task_query)
    split_counts = torch.empty((tokens,), dtype=torch.int32, device=q_packed.device)

    planner_block = 256
    _build_graphsafe_split_tasks_kernel[(triton.cdiv(num_tasks, planner_block),)](
        kv_indptr,
        split_task_query,
        split_task_start,
        split_task_len,
        split_task_row,
        split_counts,
        NUM_TASKS=num_tasks,
        MAX_SPLITS=max_splits,
        BLOCK_K=block_k,
        SPLIT_TILES=split_tiles,
        SPLIT_TILES_SHORT=split_tiles_short,
        SPLIT_SHORT_MAX_TILES=split_short_max_tiles,
        BLOCK=planner_block,
        num_warps=4,
    )

    return sparse_attn_v4_paged_decode_fp8_flydsl_splitk(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        kv_indices,
        kv_indptr,
        kv_indices[:0],
        empty_kv_indptr,
        attn_sink,
        softmax_scale,
        split_task_query,
        split_task_start,
        split_task_len,
        split_task_row,
        split_counts,
        max_splits=max_splits,
        split_tiles=split_tiles,
        split_tiles_short=split_tiles_short,
        split_short_max_tiles=split_short_max_tiles,
        block_k=block_k,
        reduce_head_group=reduce_head_group,
        use_flydsl_reduce=True,
        out=out,
        waves_per_eu=waves_per_eu,
    )


__all__ = [
    "sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe",
    "sparse_attn_v4_paged_decode_fp8_flydsl_split1",
    "sparse_attn_v4_paged_decode_fp8_flydsl_splitk",
]
