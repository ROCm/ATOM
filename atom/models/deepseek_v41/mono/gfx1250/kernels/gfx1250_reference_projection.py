# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wave32 matrix fragments for the original c1 N16/K-split execution model.

Each of eight logical wave64 groups retains its K128 steps w, w+8, ... .
The 512-thread binding uses one physical wave32 per group for WMMA and its
other half for the input copy and joins. The 256-thread binding maps each group
to one wave32 and covers both virtual-thread copy rounds. Weights stay in
registers through the tagged input wait; task geometry stays unchanged.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import (
    HIDDEN, QKV_ROWS, X8_WORDS, lds_row, poll_copy,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_k1 import stage_slice_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_norm import stage_norm_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_plan import (
    BLOCKS, THREADS, WAVES, C1ReferencePlan,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import CM_NT_GFX1250, bf16_round, kernel_symbol, launcher, rsrc, traced, uniform
from atom.mono.device.sync import sreg


@traced
def _weight_step(c, rw, rs, inner, task, st, cache_modifier=CM_NT_GFX1250):
    """One WMMA fragment from the mother's (16,16) FP8 weight shuffle.

    The four K32 blocks are 512 bytes each. WMMA lane l holds 16 bytes of
    row l%16 at K offset (l//16)*16 from each block. All four group32 scale
    codes come from the same N32 group, in one unshuffled dword.
    """
    lane = c["lane"]
    lane16, half = lane % 16, (lane % 32) // 16
    if const_expr(inner % 128 != 0):
        # Preserve the mother's K32 tail: clamp each physical load, then
        # zero absent K groups at unit scale before WMMA consumes them.
        groups = inner // 32
        chunks, scale = [], fx.Int32(0)
        for j in range_constexpr(4):
            logical = st * 4 + j
            group = fx.min(logical, groups - 1)
            at = task * (inner * 4) + group * 128 + half * 64 + lane16 * 4
            value = fx.Vector(bo.buffer_load(rw, at, vec_width=4, dtype=T.i32,
                                             cache_modifier=cache_modifier))
            chunks.append(fx.Vector.from_elements([
                (logical < groups).select(value[q], fx.Int32(0)) for q in range(4)
            ], fx.Int32))
            si = (task // 2) * groups + group
            word = fx.Int32(bo.buffer_load(rs, si // 4, vec_width=1, dtype=T.i32))
            code = (word >> (8 * (si % 4))) & 255
            scale = scale | ((logical < groups).select(code, fx.Int32(127)) << (8 * j))
        return fx.Vector.from_elements([chunks[j][q] for j in range(4) for q in range(4)], fx.Int32), scale
    at = (task * inner * 16 + st * 2048 + half * 256 + lane16 * 16) // 4
    v0 = fx.Vector(bo.buffer_load(rw, at, vec_width=4, dtype=T.i32, cache_modifier=cache_modifier))
    v1 = fx.Vector(bo.buffer_load(rw, at + 128, vec_width=4, dtype=T.i32, cache_modifier=cache_modifier))
    v2 = fx.Vector(bo.buffer_load(rw, at + 256, vec_width=4, dtype=T.i32, cache_modifier=cache_modifier))
    v3 = fx.Vector(bo.buffer_load(rw, at + 384, vec_width=4, dtype=T.i32, cache_modifier=cache_modifier))
    scale = fx.Int32(bo.buffer_load(rs, (task // 2) * (inner // 128) + st, vec_width=1, dtype=T.i32))
    return v0.shuffle(v1, list(range(8))).shuffle(
        v2.shuffle(v3, list(range(8))), list(range(16))
    ), scale


@traced
def reference_fp8_loads(c, weight, scale, inner, task, cache_modifier=CM_NT_GFX1250):
    """All matrix K-step loads issue before the caller's mailbox polling.

    As in the mother, a logical step past the row loads its last legal step;
    its matrix operands are subsequently zeroed at unit scale.
    """
    assert inner >= 32 and inner % 32 == 0
    rw, rs = rsrc(weight), rsrc(scale)
    steps = (inner + 127) // 128
    per = (steps + WAVES - 1) // WAVES
    weights = fx.Vector.filled(per * 16, 0, fx.Int32)
    scales = fx.Vector.filled(per, 0x7F7F7F7F, fx.Int32)
    # One wave-uniform branch around the complete prefetch. A branch around
    # each K step would force separate load drains before successive steps.
    if uniform(c["lane"]) < 32:
        loaded = [_weight_step(c, rw, rs, inner, task, fx.min(c["wave"] + WAVES * i, steps - 1), cache_modifier)
                  for i in range(per)]
        weights = fx.Vector.from_elements([loaded[i][0][j] for i in range(per) for j in range(16)], fx.Int32)
        scales = fx.Vector.from_elements([loaded[i][1] for i in range(per)], fx.Int32)
    fx.rocdl.sched_barrier(0)
    return [(fx.Vector.from_elements([weights[i * 16 + j] for j in range(16)], fx.Int32), scales[i]) for i in range(per)]


@traced
def reference_fp8_matrices_c1(c, inner, ops, mother_red=False, join=True):
    """Original eight K partials; private input reused by every N16 matrix."""
    lane, wave = c["lane"], c["wave"]
    if lane < 32:
        half = lane // 16
        atom = fx.make_mma_atom(fx.rocdl.WMMAScale(
            16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32
        ))
        acc = fx.make_rmem_tensor(8, fx.Float32)
        acc.store(fx.full(8, 0.0, fx.Float32))
        steps = (inner + 127) // 128
        for i in range_constexpr((steps + WAVES - 1) // WAVES):
            logical_step = wave + WAVES * i
            st = fx.min(logical_step, steps - 1)
            if const_expr(inner % 128 != 0):
                chunks = []
                for j in range_constexpr(4):
                    logical = st * 4 + j
                    group = fx.min(logical, inner // 32 - 1)
                    value = fx.Vector(fx.ptr_load(c["xl"] + group * 8 + half * 4,
                        result_type=fx.Vector.make_type(4, fx.Int32)))
                    chunks.append(fx.Vector.from_elements([
                        (logical < inner // 32).select(value[q], fx.Int32(0)) for q in range(4)
                    ], fx.Int32))
            else:
                chunks = [fx.Vector(fx.ptr_load(
                    c["xl"] + st * 32 + half * 4 + j * 8,
                    result_type=fx.Vector.make_type(4, fx.Int32)
                )) for j in range(4)]
            xv = chunks[0].shuffle(chunks[1], list(range(8))).shuffle(
                chunks[2].shuffle(chunks[3], list(range(8))), list(range(16))
            )
            if const_expr(inner % 128 != 0):
                codes = [(st * 4 + j < inner // 32).select(
                    fx.ptr_load(c["xsl"] + fx.min(st * 4 + j, inner // 32 - 1)), fx.Int32(127))
                    for j in range(4)]
            else:
                codes = [fx.ptr_load(c["xsl"] + st * 4 + j) for j in range(4)]
            xs = codes[0] | (codes[1] << 8) | (codes[2] << 16) | (codes[3] << 24)
            weights, sa = ops[i]
            if const_expr(steps % WAVES != 0):
                live = logical_step < steps
                weights = fx.Vector.from_elements([
                    live.select(weights[j], fx.Int32(0)) for j in range(16)
                ], fx.Int32)
                xv = fx.Vector.from_elements([
                    live.select(xv[j], fx.Int32(0)) for j in range(16)
                ], fx.Int32)
                sa = live.select(sa, fx.Int32(0x7F7F7F7F))
                xs = live.select(xs, fx.Int32(0x7F7F7F7F))
            a, b = fx.make_rmem_tensor(16, fx.Int32), fx.make_rmem_tensor(16, fx.Int32)
            a.store(weights); b.store(xv)
            fx.gemm(atom, acc, a, b, acc, scale_a=sa, scale_b=xs)
        # WMMA C: lane%16 is token, lane//16 selects eight output rows.
        # c1 keeps row0 only; no duplicate upper-half contribution.
        if lane % 16 == 0:
            values = acc.load()
            for j in range_constexpr(8):
                if const_expr(mother_red):
                    row = half * 8 + j
                    at = (wave * 64 + 16 * (row // 4)) * 4 + row % 4
                    fx.ptr_store(values[j], c["red"] + at)
                else:
                    fx.ptr_store(values[j], c["red"] + wave * 16 + half * 8 + j)
    if const_expr(join):
        gpu.barrier()


@traced
def stage_wqkv_c1(c, task, ops):
    reference_fp8_matrices_c1(c, HIDDEN, ops)
    if c["tid"] < 16:
        total = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            total = total + fx.ptr_load(c["red"] + w * 16 + c["tid"])
        c["put"](c["qkv"], task * 16 + c["tid"], bf16_round(total))
    gpu.barrier()


@traced
def reference_load_x8_c1(c, inner=HIDDEN, words_region="x8", scales_region="x8s"):
    """Original complete tagged copy, folded over physical CTA threads."""
    words, groups = inner // 4, inner // 32
    poll_copy(c, c["tid"], c.get("threads", THREADS), [
        (c[words_region], lambda u: u, words, c["xl"], words),
        (c[scales_region], lambda u: u, groups, c["xsl"], groups),
    ])
    gpu.barrier()


def projection_symbol(front, fold, feeds_index, threads=THREADS):
    return kernel_symbol("gfx1250_reference_k1_projection", front=int(front),
                         fold=int(fold), ix=int(feeds_index), b=BLOCKS, t=threads)


@lru_cache(None)
def build_projection_c1(front=False, fold=True, feeds_index=False, threads=THREADS):
    """Independent WQKV or the original slice→norm→WQKV source order."""
    if threads not in (256, 512):
        raise ValueError("reference c1 projection requires 256 or 512 physical threads")
    plan = C1ReferencePlan(feeds_index=feeds_index)
    layout = plan.k1_scratch_layout()
    name = projection_symbol(front, fold, feeds_index, threads)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 512, 16]

    @fx.struct
    class SliceLds:
        rl: fx.Array[fx.Float32, 128, 16]
        fl: fx.Array[fx.Float32, 24 * 128, 16]

    @fx.struct
    class WqkvLds:
        xl: fx.Array[fx.Int32, lds_row(X8_WORDS), 16]
        xsl: fx.Array[fx.Int32, lds_row(HIDDEN // 32), 16]

    @fx.union
    class StageLds:
        slice: SliceLds
        wqkv: WqkvLds

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def projection(res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64,
                   pre_in: Int64, hc_fn: Int64, res_out: Int64, scratch: Int64,
                   diag: Int64, layer: Int64, attn_w: Int64, normed: Int64,
                   weight: Int64, scale: Int64):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator(); lds = alloc.allocate(Smem).peek()
        stages = alloc.allocate(StageLds)
        sl, wl = stages.slice.peek(), stages.wqkv.peek()
        pointer = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        failure = fx.inttoptr(pointer, diag + fx.Int64(BLOCKS * 8 * 4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(pointer, layer)), diag,
            {"lin": 1, "pmix": 2, "x8": 3, "x8s": 4, "nb": 5, "qkv": 6}, failure)
        args = dict(res_in=res_in, pend=pend, post_in=post_in, comb_in=comb_in,
                    pre_in=pre_in, hc_fn=hc_fn, res_out=res_out, attn_w=attn_w, normed=normed)
        c = dict(S=1, tid=tid, bid=bid, lane=tid % 64, wave=tid // 64, threads=threads,
                 rl=sl.rl.ptr, fl=sl.fl.ptr, red=lds.red.ptr, xl=wl.xl.ptr, xsl=wl.xsl.ptr,
                 args=args, fold=fold, aux=False, index=feeds_index,
                 put=mb.put, put_words=mb.put_words, put_bf=mb.put_bf, poll=mb.poll)
        for region in ("lin", "pmix", "x8", "x8s", "nb", "qkv"):
            c[region] = sreg(scratch, layout[region][0], region)
        if const_expr(front):
            for task in range(plan.stages["k1.slice"].first(bid), 160, BLOCKS):
                stage_slice_c1(c, task)
            for task in range(plan.stages["k1.norm"].first(bid), 1, BLOCKS):
                stage_norm_c1(c)
        task = plan.stages["k1.wqkv_a"].first(bid)
        if task < QKV_ROWS // 16:
            matrix = dict(c, lane=tid % (threads // WAVES), wave=tid // (threads // WAVES))
            ops = reference_fp8_loads(matrix, weight, scale, HIDDEN, task)
            reference_load_x8_c1(matrix)
            stage_wqkv_c1(matrix, task, ops)

    @launcher(name)
    def launch(res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64,
               pre_in: Int64, hc_fn: Int64, res_out: Int64, scratch: Int64,
               diag: Int64, layer: Int64, attn_w: Int64, normed: Int64,
               weight: Int64, scale: Int64, stream: fx.Stream):
        projection(res_in, pend, post_in, comb_in, pre_in, hc_fn, res_out,
                   scratch, diag, layer, attn_w, normed, weight, scale).launch(
                       grid=(BLOCKS,), block=(threads,), stream=stream)
    return launch
