# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One multirow token tile, preserving the reference N16 tasks and K streams.

This module is being validated outside the product checkout. It does not
install a multirow runner or change the existing c1 specialization.
"""
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import lds_row, poll_copy
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_projection import reference_fp8_loads
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import bf16_round, kernel_symbol, launcher, row_sum, traced
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import BLOCKS, first_task
from atom.mono.plan.layout import pair_layout

THREADS, WAVES = 256, 8


@traced
def reference_fp8_matrices_rows(c, inner, ops, join=True):
    """Original eight K partials; private input reused by every N16 matrix."""
    lane, wave = c["lane"], c["wave"]
    # Input rows are physically present only for S tokens. The other WMMA
    # columns load the final legal row, then consume zero at unit scale.
    token = fx.min(lane % 16, c["S"] - 1)
    xl = c["xl"] + token * lds_row(inner // 4)
    xsl = c["xsl"] + token * lds_row(inner // 32)
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
                    value = fx.Vector(fx.ptr_load(xl + group * 8 + half * 4,
                        result_type=fx.Vector.make_type(4, fx.Int32)))
                    chunks.append(fx.Vector.from_elements([
                        (logical < inner // 32).select(value[q], fx.Int32(0)) for q in range(4)
                    ], fx.Int32))
            else:
                chunks = [fx.Vector(fx.ptr_load(
                    xl + st * 32 + half * 4 + j * 8,
                    result_type=fx.Vector.make_type(4, fx.Int32)
                )) for j in range(4)]
            xv = chunks[0].shuffle(chunks[1], list(range(8))).shuffle(
                chunks[2].shuffle(chunks[3], list(range(8))), list(range(16))
            )
            if const_expr(inner % 128 != 0):
                codes = [(st * 4 + j < inner // 32).select(
                    fx.ptr_load(xsl + fx.min(st * 4 + j, inner // 32 - 1)), fx.Int32(127))
                    for j in range(4)]
            else:
                codes = [fx.ptr_load(xsl + st * 4 + j) for j in range(4)]
            xs = codes[0] | (codes[1] << 8) | (codes[2] << 16) | (codes[3] << 24)
            token_live = lane % 16 < c["S"]
            xv = fx.Vector.from_elements([
                token_live.select(xv[j], fx.Int32(0)) for j in range(16)
            ], fx.Int32)
            xs = token_live.select(xs, fx.Int32(0x7F7F7F7F))
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
        # Each live WMMA token column is restored to the original MFMA
        # [wave64lane][4] red slot before the original wave-ordered sum.
        if lane % 16 < c["S"]:
            values = acc.load()
            for j in range_constexpr(8):
                row = half * 8 + j
                at = (wave * 64 + 16 * (row // 4) + lane % 16) * 4 + row % 4
                fx.ptr_store(values[j], c["red"] + at)
    if const_expr(join):
        gpu.barrier()

@traced
def reference_load_x8_rows(c, inner, words_region="x8", scales_region="x8s"):
    """One complete S-row input tile reused by all of this CTA's N tasks."""
    words, groups = inner // 4, inner // 32
    poll_copy(c, c["tid"], THREADS, [
        (c[words_region], lambda u:u, c["S"]*words, c["xl"], words),
        (c[scales_region], lambda u:u, c["S"]*groups, c["xsl"], groups),
    ])
    gpu.barrier()


@traced
def stage_projection_rows(c, task, ops, inner, output_rows):
    reference_fp8_matrices_rows(c, inner, ops)
    if c["tid"] < 16*c["S"]:
        token, row = c["tid"]//16, c["tid"]%16
        total = row_sum(c["red"], row, token)
        c["put"](c["qkv"], token*output_rows+task*16+row, bf16_round(total))
    gpu.barrier()


def projection_rows_layout(rows, inner=5120, output_rows=1792):
    if rows not in (2,3,4) or inner < 32 or inner%32 or output_rows%16:
        raise ValueError("One 2/3/4-row tile, K32 and N16 geometry required")
    return pair_layout({"x8":rows*inner//4,"x8s":rows*inner//32,
                        "qkv":rows*output_rows}.items())


def projection_rows_symbol(rows, inner=5120, output_rows=1792):
    return kernel_symbol("gfx1250_reference_projection_rows",s=rows,k=inner,n=output_rows,b=BLOCKS,t=THREADS)


@lru_cache(None)
def build_projection_rows(rows, inner=5120, output_rows=1792):
    """Isolated matrix stage; the full K1 stages are connected separately."""
    layout=projection_rows_layout(rows,inner,output_rows)
    name=projection_rows_symbol(rows,inner,output_rows)
    @fx.struct
    class Smem:
        red:fx.Array[fx.Float32,WAVES*2*64*4,16]
        xl:fx.Array[fx.Int32,rows*lds_row(inner//4),16]
        xsl:fx.Array[fx.Int32,rows*lds_row(inner//32),16]
    @flyc.kernel(name=name,known_block_size=[THREADS,1,1])
    def kernel(weight:Int64,scale:Int64,scratch:Int64,diag:Int64,layer:Int64):
        tid,bid=fx.thread_idx.x,fx.block_idx.x
        allocator=fx.SharedAllocator()
        smem=allocator.allocate(Smem).peek()
        iptr=fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4)
        failure=fx.inttoptr(iptr,diag+fx.Int64(BLOCKS*8*4))
        mb=reference_mailbox(fx.ptr_load(fx.inttoptr(iptr,layer)),diag,
                             {k:i+1 for i,k in enumerate(layout)},failure)
        c=dict(S=rows,tid=tid,bid=bid,lane=tid%32,wave=tid//32,
               red=smem.red.ptr,xl=smem.xl.ptr,xsl=smem.xsl.ptr,
               put=mb.put,put_words=mb.put_words,poll=mb.poll)
        for key in layout:
            c[key]=sreg(scratch,layout[key][0],key)
        # WQKV's original base and 112 N16 tasks; no row multiplication.
        for task in range(first_task(bid,176),output_rows//16,BLOCKS):
            ops=reference_fp8_loads(c,weight,scale,inner,task)
            reference_load_x8_rows(c,inner)
            stage_projection_rows(c,task,ops,inner,output_rows)
    @launcher(name)
    def launch(weight:Int64,scale:Int64,scratch:Int64,diag:Int64,layer:Int64,stream:fx.Stream):
        kernel(weight,scale,scratch,diag,layer).launch(grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
