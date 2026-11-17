# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 AttnRes mix: a CTA a token, seven columns a thread.

``_attn_res_fused_kernel``'s mix -- the token's ``count`` block rows and its
prefix (plus the folded addends) as candidates, each scored by the RMS-normed
dot with ``score_weight``, softmaxed and summed, then RMS-normed by
``out_weight`` -- with every candidate's seven columns loaded by a thread at
once rather than two candidates a step. The candidates' squares and scores
reduce together, over a wave and then across the 32 waves through LDS, so the
token takes two reduction rounds and one more for the output norm. The
reductions run in another order than the Triton kernel's and the softmax is
not online; the arithmetic is otherwise the same.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages.k2_allreduce_gfx1250 import (
    _seven,
    _store_seven,
)
from atom.mono.device.ops import butterfly, exp, hw_rsq, rsrc, traced, xred
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = 1024
WAVES = THREADS // GFX1250.wave_size
ELEMENTS = 7
assert THREADS * ELEMENTS == HIDDEN
# the per-wave sums of every candidate's two reductions sit a wave a row
MAX_CANDIDATES = WAVES // 2

ATTN_RES_ABI = KernelAbi(
    (
        "blocks",
        "prefix",
        "add",
        "add2",
        "score_weight",
        "out_weight",
        "out",
        "prefix_out",
        "token_stride",
        "tokens",
    )
)


@dataclass(frozen=True)
class AttnResKernel:
    name: str
    launcher: object
    abi: KernelAbi = ATTN_RES_ABI


def _sum(a, b):
    return a + b


@traced
def _wave_sums(values):
    """Each of ``values`` summed over the wave, the reductions side by side."""
    for offset in GFX1250.reduction_offsets:
        values = [xred(v, offset, _sum, GFX1250) for v in values]
    return values


def _seven_f32(pointer, element):
    """FP32 ``element .. element + 7`` of ``pointer``."""
    four = fx.Vector(bo.buffer_load(pointer, element, vec_width=4, dtype=T.f32))
    two = fx.Vector(bo.buffer_load(pointer, element + 4, vec_width=2, dtype=T.f32))
    one = fx.Float32(bo.buffer_load(pointer, element + 6, vec_width=1, dtype=T.f32))
    return [fx.Float32(four[j]) for j in range(4)] + [
        fx.Float32(two[0]),
        fx.Float32(two[1]),
        one,
    ]


def build_attn_res(
    count: int, addends: int, out_norm: bool, eps: float, out_eps: float
) -> AttnResKernel:
    """The mix of ``count`` block rows and the prefix with ``addends`` (0 to
    2) folded in, written back as ``prefix_out`` when there are any;
    ``out_weight`` is read only with ``out_norm``. ``blocks`` rows are HIDDEN
    apart within a token and ``token_stride`` apart across tokens; the other
    tensors are ``[tokens, HIDDEN]`` BF16 but ``score_weight``, ``[HIDDEN]``
    FP32."""
    candidates = count + 1
    if not 1 <= count < MAX_CANDIDATES:
        raise ValueError(f"the AttnRes mix holds 1 to {MAX_CANDIDATES - 1} blocks")
    if addends not in (0, 1, 2):
        raise ValueError("the AttnRes mix folds up to two addends")
    reductions = 2 * candidates
    name = f"k3_gfx1250_attn_res_b{count}_a{addends}{'_n' if out_norm else ''}"

    @fx.struct
    class Lds:
        partial: fx.Array[fx.Float32, WAVES * reductions, 16]
        total: fx.Array[fx.Float32, reductions, 16]
        squares: fx.Array[fx.Float32, WAVES, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        blocks: Int64,
        prefix: Int64,
        add: Int64,
        add2: Int64,
        score_weight: Int64,
        out_weight: Int64,
        out: Int64,
        prefix_out: Int64,
        token_stride: Int32,
        tokens: Int32,
    ):
        _ = tokens
        tid = fx.Int32(fx.thread_idx.x)
        token = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        lds = fx.SharedAllocator().allocate(Lds).peek()
        column = tid * ELEMENTS
        element = token * HIDDEN + column
        rows = [
            _seven(rsrc(blocks), token * token_stride + b * HIDDEN + column)
            for b in range_constexpr(count)
        ]
        last = _seven(rsrc(prefix), element)
        for addend in [add, add2][:addends]:
            last = [a + b for a, b in zip(last, _seven(rsrc(addend), element))]
        if const_expr(addends > 0):
            _store_seven(rsrc(prefix_out), element, last)
        rows.append(last)
        weight = _seven_f32(rsrc(score_weight), column)
        partials = []
        for row in rows:
            square = row[0] * row[0]
            dot = row[0] * weight[0]
            for j in range_constexpr(1, ELEMENTS):
                square = square + row[j] * row[j]
                dot = dot + row[j] * weight[j]
            partials += [square, dot]
        partials = _wave_sums(partials)
        if lane == 0:
            for k in range_constexpr(reductions):
                fx.ptr_store(partials[k], lds.partial.ptr + wave * reductions + k)
        gpu.barrier()
        if wave < reductions:
            value = butterfly(
                fx.ptr_load(lds.partial.ptr + lane * reductions + wave),
                GFX1250.reduction_offsets,
                architecture=GFX1250,
            )
            if lane == 0:
                fx.ptr_store(value, lds.total.ptr + wave)
        gpu.barrier()
        totals = [fx.ptr_load(lds.total.ptr + k) for k in range(reductions)]
        scores = [
            totals[2 * c + 1]
            * hw_rsq(totals[2 * c] * fx.Float32(1.0 / HIDDEN) + fx.Float32(eps))
            for c in range_constexpr(candidates)
        ]
        top = scores[0]
        for score in scores[1:]:
            top = fx.max(top, score)
        odds = [exp(score - top) for score in scores]
        denominator = odds[0]
        for p in odds[1:]:
            denominator = denominator + p
        mixed = []
        for j in range_constexpr(ELEMENTS):
            total = odds[0] * rows[0][j]
            for c in range_constexpr(1, candidates):
                total = total + odds[c] * rows[c][j]
            mixed.append(total / denominator)
        if const_expr(out_norm):
            square = mixed[0] * mixed[0]
            for value in mixed[1:]:
                square = square + value * value
            square = butterfly(square, GFX1250.reduction_offsets, architecture=GFX1250)
            if lane == 0:
                fx.ptr_store(square, lds.squares.ptr + wave)
            gpu.barrier()
            norm = fx.ptr_load(lds.squares.ptr)
            for w in range_constexpr(1, WAVES):
                norm = norm + fx.ptr_load(lds.squares.ptr + w)
            inverse = hw_rsq(norm * fx.Float32(1.0 / HIDDEN) + fx.Float32(out_eps))
            gains = _seven(rsrc(out_weight), column)
            mixed = [m * inverse * g for m, g in zip(mixed, gains)]
        _store_seven(rsrc(out), element, mixed)

    @flyc.jit
    def launch(
        blocks: Int64,
        prefix: Int64,
        add: Int64,
        add2: Int64,
        score_weight: Int64,
        out_weight: Int64,
        out: Int64,
        prefix_out: Int64,
        token_stride: Int32,
        tokens: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            blocks,
            prefix,
            add,
            add2,
            score_weight,
            out_weight,
            out,
            prefix_out,
            token_stride,
            tokens,
        ).launch(grid=(fx.Int64(tokens), 1, 1), block=(THREADS, 1, 1), stream=stream)

    ATTN_RES_ABI.check(kernel, launch)
    return AttnResKernel(name, launch)
