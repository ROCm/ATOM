# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 AttnRes mix, column-sliced: a cluster a token group, a CTA a slice.

``attn_res_gfx1250``'s mix -- the token's ``count`` block rows and its prefix
(plus the folded addends) as candidates, each scored by the RMS-normed dot
with ``score_weight``, softmaxed and summed, then RMS-normed by ``out_weight``
-- with a token's columns spread over the ``CLUSTER`` CTAs of one hardware
cluster instead of one CTA, as gfx950's seam slices them over its attention
CTAs. Each CTA reduces its slice's per-candidate sums of squares and score
dots and stores them to ``scratch`` at cluster scope; past a cluster barrier
every CTA of the cluster sums all slices' in slice order, so each scores and
softmaxes the token alike and mixes its own columns. The output norm's sum of
squares takes one more such exchange. No flag, tag or fence: each CTA drains
its cluster-scope stores before the cluster barrier and reads at cluster scope
after it, and a launch's exchanges end with it. A cluster shares one shader
engine, so cluster scope is gfx1250's SCOPE_SE (LLVM's ``cluster`` syncscope):
the exchange stays in that engine's L2 rather than reaching device scope.

The clusters split the tokens round-robin, a token a ``TOKEN_THREADS``-thread
group of the CTA, ``THREADS // TOKEN_THREADS`` tokens a pass. The reductions
run in another order than ``attn_res_gfx1250``'s; the arithmetic is otherwise
the same.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster as hw_cluster
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.kernels.stages.attn_res_gfx1250 import _wave_sums
from atom.models.kimi_k3.mono.kernels.stages.attn_res_sliced_layout import (
    ELEMENTS,
    HIDDEN,
    THREADS,
    WAVES,
    SlicedGeometry,
    check,
    pairs,
    sums,
)
from atom.mono.device.ops import bf_hi, bf_lo, exp, hw_rsq, rsrc
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

# buffer cache policy scope field SCOPE_SE
CM_CLUSTER = 8

ATTN_RES_SLICED_ABI = KernelAbi(
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
        "scratch",
    )
)


@dataclass(frozen=True)
class AttnResSlicedKernel:
    name: str
    launcher: object
    geometry: SlicedGeometry
    abi: KernelAbi = ATTN_RES_SLICED_ABI


def _eight(pointer, element):
    """BF16 ``element .. element + 8`` of ``pointer`` as fp32: one 16 B load."""
    words = fx.Vector(bo.buffer_load(pointer, element // 2, vec_width=4, dtype=T.i32))
    values = []
    for w in range(4):
        word = fx.Int32(words[w])
        values += [bf_lo(word), bf_hi(word)]
    return values


def _eight_f32(pointer, element):
    values = []
    for half in range(2):
        four = fx.Vector(
            bo.buffer_load(pointer, element + 4 * half, vec_width=4, dtype=T.f32)
        )
        values += [fx.Float32(four[j]) for j in range(4)]
    return values


def _store_eight(pointer, element, values):
    words = (
        fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    )
    bo.buffer_store(words, pointer, element // 2)


def _slices_sum(pointer, word, slices):
    """``slices`` consecutive cluster-scope FP32 at ``word``, summed in order."""
    values = []
    for quad in range_constexpr(slices // 4):
        four = fx.Vector(
            bo.buffer_load(
                pointer,
                word + 4 * quad,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_CLUSTER,
            )
        )
        values += [fx.Float32(four[j]) for j in range(4)]
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def _cluster_exchange():
    """Every wave's cluster-scope stores drained, then the cluster barrier."""
    rocdl.s_wait_storecnt(0)
    hw_cluster.cluster_barrier()


def build_attn_res_sliced(
    count: int,
    addends: int,
    out_norm: bool,
    eps: float,
    out_eps: float,
    cluster: int = 16,
    gram: bool = False,
) -> AttnResSlicedKernel:
    """``attn_res_gfx1250.build_attn_res``'s mix, sliced over ``cluster``-CTA
    clusters; ``scratch`` holds ``scratch_bytes(count, tokens, cluster, gram)``.

    ``gram``: the output norm's sum of squares as the softmax weights'
    quadratic form in the candidates' Gram matrix, whose slice sums travel
    with the scores', so one cluster exchange a pass instead of two; the
    reassociation is not only a reduction order. Its FP32 error scales with
    the mix's cancellation R = (sum_c p_c |v_c|)^2 / |sum_c p_c v_c|^2: the
    rstd's relative error is about 2e-9 R, the direct sum's about 1e-7 at any
    R, so the two agree to R ~ 50 and Gram flips measurably more BF16 outputs
    past R ~ 100."""
    check(count, addends, out_norm, gram)
    candidates = count + 1
    geometry = SlicedGeometry(cluster)
    lanes = geometry.lanes
    group = geometry.token_threads
    group_waves = geometry.token_waves
    passing = geometry.tokens_a_pass
    gram = gram and out_norm
    record = geometry.record_words(candidates, gram)
    reductions = sums(candidates, gram)
    crossed = pairs(candidates)
    crosses = len(crossed) if gram else 0
    msq_at = reductions * cluster
    # the gate's work items a pass: a candidate's score, or a pair's product
    items = candidates + crosses
    name = (
        f"k3_gfx1250_attn_res_sliced_c{cluster}_b{count}_a{addends}"
        f"{'_n' if out_norm else ''}{'_g' if gram else ''}"
    )

    @fx.struct
    class Lds:
        partial: fx.Array[fx.Float32, WAVES * reductions, 16]
        scores: fx.Array[fx.Float32, passing * candidates, 16]
        diagonal: fx.Array[fx.Float32, passing * candidates, 16]
        cross: fx.Array[fx.Float32, passing * max(crosses, 1), 16]
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
        scratch: Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        rank = fx.Int32(fx.block_idx.x) % cluster
        clusters = fx.min(tokens, fx.Int32(geometry.max_clusters))
        home = fx.Int32(fx.block_idx.x) // cluster
        lds = fx.SharedAllocator().allocate(Lds).peek()
        local = tid // group
        slot = tid % group
        busy = slot < lanes
        column = rank * geometry.columns + fx.min(slot, lanes - 1) * ELEMENTS
        weight = _eight_f32(rsrc(score_weight), column)
        if const_expr(out_norm):
            gains = _eight(rsrc(out_weight), column)
        mine = (tokens - home + clusters - 1) // clusters
        passes = (mine + passing - 1) // passing
        for step in range(passes):
            token = home + (step * passing + local) * clusters
            live = token < tokens
            token = fx.min(token, tokens - 1)
            element = token * HIDDEN + column
            rows = [
                _eight(rsrc(blocks), token * token_stride + b * HIDDEN + column)
                for b in range_constexpr(count)
            ]
            last = _eight(rsrc(prefix), element)
            for addend in [add, add2][:addends]:
                last = [a + b for a, b in zip(last, _eight(rsrc(addend), element))]
            if const_expr(addends > 0):  # noqa: SIM102
                if live & busy:
                    _store_eight(rsrc(prefix_out), element, last)
            rows.append(last)
            partials = []
            for row in rows:
                square = row[0] * row[0]
                dot = row[0] * weight[0]
                for j in range_constexpr(1, ELEMENTS):
                    square = square + row[j] * row[j]
                    dot = dot + row[j] * weight[j]
                partials += [
                    busy.select(square, fx.Float32(0.0)),
                    busy.select(dot, fx.Float32(0.0)),
                ]
            if const_expr(gram):
                for c, d in crossed:
                    cross = rows[c][0] * rows[d][0]
                    for j in range_constexpr(1, ELEMENTS):
                        cross = cross + rows[c][j] * rows[d][j]
                    partials.append(busy.select(cross, fx.Float32(0.0)))
            partials = _wave_sums(partials)
            if lane == 0:
                for k in range_constexpr(reductions):
                    fx.ptr_store(partials[k], lds.partial.ptr + wave * reductions + k)
            gpu.barrier()
            # the slice's sums: a thread a (token, sum), the token's waves in order
            for first in range_constexpr(0, passing * reductions, THREADS):
                index = tid + first
                if index < passing * reductions:
                    owner = index // reductions
                    k = index % reductions
                    total = fx.ptr_load(
                        lds.partial.ptr + owner * group_waves * reductions + k
                    )
                    for w in range_constexpr(1, group_waves):
                        total = total + fx.ptr_load(
                            lds.partial.ptr + (owner * group_waves + w) * reductions + k
                        )
                    theirs = home + (step * passing + owner) * clusters
                    if theirs < tokens:
                        bo.buffer_store(
                            total,
                            rsrc(scratch),
                            theirs * record + k * cluster + rank,
                            cache_modifier=CM_CLUSTER,
                        )
            _cluster_exchange()
            # every slice's sums, in slice order: a thread a (token, item)
            for first in range_constexpr(0, passing * items, THREADS):
                index = tid + first
                if index < passing * items:
                    owner = index // items
                    item = index % items
                    theirs = fx.min(
                        home + (step * passing + owner) * clusters, tokens - 1
                    )
                    if item < candidates:
                        at = theirs * record + 2 * item * cluster
                        square = _slices_sum(rsrc(scratch), at, cluster)
                        dot = _slices_sum(rsrc(scratch), at + cluster, cluster)
                        score = dot * hw_rsq(
                            square * fx.Float32(1.0 / HIDDEN) + fx.Float32(eps)
                        )
                        fx.ptr_store(score, lds.scores.ptr + owner * candidates + item)
                        fx.ptr_store(
                            square, lds.diagonal.ptr + owner * candidates + item
                        )
                    else:
                        at = theirs * record + (item + candidates) * cluster
                        fx.ptr_store(
                            _slices_sum(rsrc(scratch), at, cluster),
                            lds.cross.ptr + owner * max(crosses, 1) + item - candidates,
                        )
            gpu.barrier()
            scores = [
                fx.ptr_load(lds.scores.ptr + local * candidates + c)
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
            if const_expr(gram):
                # sum_j mixed_j^2 = odds^T G odds / denominator^2
                norm = fx.Float32(0.0)
                for c in range_constexpr(candidates):
                    row = odds[c] * fx.ptr_load(
                        lds.diagonal.ptr + local * candidates + c
                    )
                    for d in range_constexpr(c + 1, candidates):
                        pair = crossed.index((c, d))
                        row = row + fx.Float32(2.0) * odds[d] * fx.ptr_load(
                            lds.cross.ptr + local * crosses + pair
                        )
                    norm = norm + odds[c] * row
                # a near-cancelling mix can round the form below zero
                norm = fx.max(norm, fx.Float32(0.0)) / (denominator * denominator)
                inverse = hw_rsq(norm * fx.Float32(1.0 / HIDDEN) + fx.Float32(out_eps))
                mixed = [m * inverse * g for m, g in zip(mixed, gains)]
            elif const_expr(out_norm):
                square = mixed[0] * mixed[0]
                for value in mixed[1:]:
                    square = square + value * value
                (square,) = _wave_sums([busy.select(square, fx.Float32(0.0))])
                if lane == 0:
                    fx.ptr_store(square, lds.squares.ptr + wave)
                gpu.barrier()
                if tid < passing:
                    total = fx.ptr_load(lds.squares.ptr + tid * group_waves)
                    for w in range_constexpr(1, group_waves):
                        total = total + fx.ptr_load(
                            lds.squares.ptr + tid * group_waves + w
                        )
                    theirs = home + (step * passing + tid) * clusters
                    if theirs < tokens:
                        bo.buffer_store(
                            total,
                            rsrc(scratch),
                            theirs * record + msq_at + rank,
                            cache_modifier=CM_CLUSTER,
                        )
                _cluster_exchange()
                norm = _slices_sum(rsrc(scratch), token * record + msq_at, cluster)
                inverse = hw_rsq(norm * fx.Float32(1.0 / HIDDEN) + fx.Float32(out_eps))
                mixed = [m * inverse * g for m, g in zip(mixed, gains)]
            if live & busy:
                _store_eight(rsrc(out), element, mixed)

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
        scratch: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        clusters = fx.min(fx.Int64(tokens), fx.Int64(geometry.max_clusters))
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
            scratch,
            value_attrs={"rocdl.cluster_dims": f"{cluster},1,1"},
        ).launch(
            grid=(clusters * cluster, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            cluster=(cluster, 1, 1),
        )

    ATTN_RES_SLICED_ABI.check(kernel, launch)
    return AttnResSlicedKernel(name, launch, geometry)
