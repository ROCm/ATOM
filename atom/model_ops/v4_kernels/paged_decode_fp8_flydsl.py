# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph-safe FlyDSL kernels for qualified V4 native-FP8 decode shapes."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_buf_tensor
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import T

from atom.model_ops.v4_kernels.paged_decode import LOG2E
from atom.model_ops.v4_kernels.paged_prefill_fp8_flydsl import (
    V4_DIM_QK,
    sparse_attn_v4_paged_prefill_fp8_flydsl,
)


@lru_cache(maxsize=32)
def _build_split_reduce_launcher(
    max_splits: int,
    head_group: int = 1,
    waves_per_eu: int = 0,
    implicit_split_counts: bool = False,
    block_k: int = 32,
    split_tiles: int = 0,
    split_tiles_short: int = 0,
    split_short_max_tiles: int = 0,
    split_tiles_mid: int = 0,
    split_mid_max_tiles: int = 0,
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
            + (
                f"_implicit_bk{block_k}_long{split_tiles}"
                f"_short{split_tiles_short}_shortmax{split_short_max_tiles}"
                f"_mid{split_tiles_mid}_midmax{split_mid_max_tiles}"
                if implicit_split_counts
                else ""
            )
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
        out_bf16 = ptr_buf_tensor(out, fx.BFloat16)
        counts_i32 = ptr_buf_tensor(split_counts, fx.Int32)
        if fx.const_expr(implicit_split_counts):
            kv_len = fx.Int32(counts_i32[query + 1]) - fx.Int32(counts_i32[query])
            num_tiles = (kv_len + fx.Int32(block_k - 1)) // fx.Int32(block_k)
            segment_tiles = fx.Int32(split_tiles)
            if fx.const_expr(split_tiles_mid > 0):  # noqa: SIM102
                if num_tiles <= fx.Int32(split_mid_max_tiles):
                    segment_tiles = fx.Int32(split_tiles_mid)
            if fx.const_expr(split_tiles_short > 0):  # noqa: SIM102
                if num_tiles <= fx.Int32(split_short_max_tiles):
                    segment_tiles = fx.Int32(split_tiles_short)
            num_splits = (num_tiles + segment_tiles - fx.Int32(1)) // segment_tiles
            if num_splits > fx.Int32(max_splits):  # noqa: PLR1730
                num_splits = fx.Int32(max_splits)
        else:
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


def _sparse_attn_v4_paged_decode_fp8_flydsl_splitk(
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
    max_splits: int,
    split_tiles: int,
    split_tiles_short: int = 0,
    split_short_max_tiles: int = 0,
    split_tiles_mid: int = 0,
    split_mid_max_tiles: int = 0,
    block_k: int = 32,
    reduce_head_group: int = 1,
    partial_m: torch.Tensor | None = None,
    partial_l: torch.Tensor | None = None,
    partial_acc: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    waves_per_eu: int = 0,
) -> torch.Tensor:
    """Run the fixed-grid FlyDSL stage-1 and reducer for native-FP8 decode."""
    tokens, heads, _ = q_packed.shape
    if heads != 128:
        raise ValueError("the FlyDSL split-K candidate requires H=128")
    if max_splits <= 1:
        raise ValueError("max_splits must be greater than one")
    if block_k != 32:
        raise ValueError("the current FlyDSL stage1 uses a fixed K32 tile")
    if reduce_head_group not in (1, 2, 4, 8):
        raise ValueError("reduce_head_group must be 1, 2, 4, or 8")

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
        kv_indices[:0],
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
        split_task_query=kv_indptr,
        split_task_start=kv_indptr,
        split_task_len=kv_indptr,
        split_task_row=kv_indptr,
        partial_m=partial_m,
        partial_l=partial_l,
        implicit_split_tasks=True,
        implicit_max_splits=max_splits,
        implicit_block_k=block_k,
        implicit_split_tiles=split_tiles,
        implicit_split_tiles_short=split_tiles_short,
        implicit_split_short_max_tiles=split_short_max_tiles,
        implicit_split_tiles_mid=split_tiles_mid,
        implicit_split_mid_max_tiles=split_mid_max_tiles,
    )

    stream = torch.cuda.current_stream(q_packed.device)
    with (
        torch.cuda.device(q_packed.device.index),
        CompilationContext.compile_hints(
            {"fast_fp_math": True, "unsafe_fp_math": True}
        ),
    ):
        _run_compiled(
            _build_split_reduce_launcher(
                max_splits,
                reduce_head_group,
                waves_per_eu,
                True,
                block_k,
                split_tiles,
                split_tiles_short,
                split_short_max_tiles,
                split_tiles_mid,
                split_mid_max_tiles,
            ),
            partial_l,
            partial_acc,
            attn_sink,
            kv_indptr,
            out,
            int(tokens * heads // reduce_head_group),
            stream,
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
    max_splits: int = 13,
    block_k: int = 32,
    split_tiles: int = 14,
    split_tiles_short: int = 11,
    split_short_max_tiles: int = 64,
    split_tiles_mid: int = 0,
    split_mid_max_tiles: int = 0,
    reduce_head_group: int = 8,
    waves_per_eu: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the production-shaped H128 FlyDSL split-K decode path.

    Stage 1 derives a split-major task map directly from ``kv_indptr`` on the
    GPU. The fixed ``T * max_splits`` grid and temporary tensor shapes depend
    only on the captured query shape, while replay can consume new lengths.
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

    return _sparse_attn_v4_paged_decode_fp8_flydsl_splitk(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        kv_indices,
        kv_indptr,
        empty_kv_indptr,
        attn_sink,
        softmax_scale,
        max_splits=max_splits,
        split_tiles=split_tiles,
        split_tiles_short=split_tiles_short,
        split_short_max_tiles=split_short_max_tiles,
        split_tiles_mid=split_tiles_mid,
        split_mid_max_tiles=split_mid_max_tiles,
        block_k=block_k,
        reduce_head_group=reduce_head_group,
        out=out,
        waves_per_eu=waves_per_eu,
    )


def sparse_attn_v4_paged_decode_fp8_flydsl_auto(
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
    query_group: int,
    kv_kind: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the measured gfx950 H128/q7 graph-safe FlyDSL schedule."""

    tokens, heads, _ = q_packed.shape
    if heads != 128 or query_group != 7 or tokens % query_group:
        raise ValueError("the FlyDSL auto decode path requires H128/q7")

    if kv_kind == "hca":
        config = (13, 14, 10, 50, 11, 64)
        waves_per_eu = 1
    elif kv_kind == "csa":
        # Keep the captured grid independent of the live K length. vLLM passes
        # a compact index view, while SGLang graphs retain a worst-case backing
        # buffer, so ``kv_indices.numel()`` is not a portable K signal. Stage 1
        # instead reads each token's live length from ``kv_indptr`` and selects
        # the short (K384) segment size on device. The batch-visible split cap
        # keeps launch overhead low at B13+ while preserving K1152 throughput.
        requests = tokens // query_group
        if requests <= 6:
            config = (6, 6, 3, 20, 0, 0)
            waves_per_eu = 0
        elif requests <= 8:
            config = (6, 6, 4, 12, 0, 0)
            waves_per_eu = 1
        elif requests <= 12:
            config = (3, 8, 6, 12, 0, 0)
            waves_per_eu = 1
        elif requests <= 16:
            config = (2, 9, 6, 12, 0, 0)
            waves_per_eu = 1
        elif requests <= 18:
            config = (2, 10, 6, 12, 0, 0)
            waves_per_eu = 1
        else:
            config = (2, 9, 6, 12, 0, 0)
            waves_per_eu = 1
    else:
        raise ValueError(f"unsupported FlyDSL decode KV kind: {kv_kind}")

    (
        max_splits,
        split_tiles,
        split_tiles_short,
        split_short_max_tiles,
        split_tiles_mid,
        split_mid_max_tiles,
    ) = config
    return sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        kv_indices,
        kv_indptr,
        empty_kv_indptr,
        attn_sink,
        softmax_scale,
        max_splits=max_splits,
        split_tiles=split_tiles,
        split_tiles_short=split_tiles_short,
        split_short_max_tiles=split_short_max_tiles,
        split_tiles_mid=split_tiles_mid,
        split_mid_max_tiles=split_mid_max_tiles,
        reduce_head_group=8,
        waves_per_eu=waves_per_eu,
        out=out,
    )


__all__ = [
    "sparse_attn_v4_paged_decode_fp8_flydsl_auto",
    "sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe",
]
