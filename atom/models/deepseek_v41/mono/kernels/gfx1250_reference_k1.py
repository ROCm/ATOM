# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stage translations for the gfx950 execution model on gfx1250.

The c1 slice keeps the original task, input, weight and tagged-output helpers.
Its FP32 mix dot uses eight logical K slices; scalar FP32 instructions replace
gfx950's unsupported FP32 MFMA. This is a correctness translation, without a
new task schedule or performance tuning. Norm and later stages are not yet
part of this launcher.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64

from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    MIX, KT, PART, HIDDEN, COLS, SLICES, K_MIX, ld_f32,
    batched_rounds, bf16_round, first_rounds, slice_issued, slice_loads, slice_row, slice_input,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_plan import (
    BLOCKS, THREADS, WAVES, C1ReferencePlan,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import kernel_symbol, launcher, traced
from atom.mono.device.sync import sreg


@traced
def slice_prefetch_c1(c, task):
    """Mother residual/gate and mix-weight loads before the TP partial wait."""
    threads = c.get("threads", THREADS)
    rows = first_rounds(c["tid"], KT, lambda e: slice_loads(c, task*COLS, e))
    fns = []
    for i in range_constexpr(MIX*KT//threads):
        e = c["tid"]+threads*i
        fns.append(ld_f32(c["args"]["hc_fn"],
            (e//KT)*K_MIX+((e%KT)//COLS)*HIDDEN+task*COLS+e%COLS))
    return fns, rows


@traced
def stage_slice_c1(c, task, pre=None):
    """Original slice load/collapse, then eight logical FP32 mix K slices."""
    tid = c["tid"]
    threads = c.get("threads", THREADS)
    lane, wave = tid % (threads // WAVES), tid // (threads // WAVES)
    c0 = task * COLS
    if const_expr(threads == THREADS):
        fns, rows = slice_issued(c, task, pre)
    else:
        # Two original virtual-thread rounds per physical thread. Every
        # weight still issues before the residual loads, at the same task.
        if const_expr(pre is not None):
            fns, rows = pre
        else:
            fns = []
            for i in range_constexpr(MIX * KT // threads):
                e = tid + threads * i
                fns.append(ld_f32(c["args"]["hc_fn"],
                    (e // KT) * K_MIX + ((e % KT) // COLS) * HIDDEN + c0 + e % COLS))
            rows = None
    batched_rounds(
        tid, KT, lambda e: slice_loads(c, c0, e),
        lambda e, got: slice_row(c, c0, e, got), first=rows,
    )
    for i in range_constexpr(MIX * KT // threads):
        hi = bf16_round(fns[i])
        fx.ptr_store(hi + bf16_round(fns[i] - hi), c["fl"] + tid + threads * i)
    gpu.barrier()
    slice_input(c, task, tid)
    # The original 64-lane sum starts by adding lane xor32. These two
    # physical wave32 halves exchange through private LDS, then reduce in
    # the original xor16,8,4,2,1 order. All CTA waves join the exchange.
    if tid < 64:
        # The sum of squares retains the original 64 logical lanes even
        # when the eight matrix groups map to eight physical wave32s.
        x0 = fx.ptr_load(c["rl"] + 2 * tid)
        x1 = fx.ptr_load(c["rl"] + 2 * tid + 1)
        fx.ptr_store(x0 * x0 + x1 * x1, c["red"] + tid)
    gpu.barrier()
    if tid < 64:
        sq = fx.ptr_load(c["red"] + tid) + fx.ptr_load(c["red"] + (tid ^ 32))
        for off in (16, 8, 4, 2, 1):
            sq = sq + sq.shuffle_xor(off, 32)
        if tid == 0:
            c["put"](c["pmix"], task * PART + MIX, sq)
    gpu.barrier()
    # Same logical eight K16 slices as the mother's FP32 MFMA stage. The
    # low physical half of a logical wave computes rows0..23; its other
    # half reaches each join. No global intermediates or extra owner.
    if (lane < 32) & (lane < MIX):
        acc = fx.Float32(0.0)
        for k in range_constexpr(KT // WAVES):
            col = wave * (KT // WAVES) + k
            w = fx.ptr_load(c["fl"] + lane * KT + col)
            x = fx.ptr_load(c["rl"] + col)
            acc = fx.Float32(fmath.fma(w, x, acc))
        fx.ptr_store(acc, c["red"] + wave * 32 + lane)
    gpu.barrier()
    if tid < MIX:
        total = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            total = total + fx.ptr_load(c["red"] + w * 32 + tid)
        c["put"](c["pmix"], task * PART + tid, total)
    gpu.barrier()


def slice_symbol(fold):
    return kernel_symbol("gfx1250_reference_k1_slice", fold=int(fold), b=BLOCKS,t=THREADS)


@lru_cache(None)
def build_slice_c1(fold=True):
    """Pointer-only c1 stage binding; caller supplies original scratch/diag."""
    plan = C1ReferencePlan()
    layout = plan.k1_scratch_layout()
    name = slice_symbol(fold)

    @fx.struct
    class Smem:
        rl: fx.Array[fx.Float32, KT, 16]
        fl: fx.Array[fx.Float32, MIX * KT, 16]
        red: fx.Array[fx.Float32, WAVES * 32, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def slice_kernel(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, res_out: Int64, scratch: Int64, diag: Int64, layer: Int64,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        failure = fx.inttoptr(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
            diag + fx.Int64(BLOCKS * 8 * 4),
        )
        # Layer is a pointer in this binding, so changing it reuses a graph.
        layerp = fx.inttoptr(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4), layer,
        )
        mb = reference_mailbox(fx.ptr_load(layerp), diag, {"lin": 1, "pmix": 2}, failure)
        args = dict(res_in=res_in, pend=pend, post_in=post_in, comb_in=comb_in,
            pre_in=pre_in, hc_fn=hc_fn, res_out=res_out)
        c = dict(S=1, tid=tid, lane=tid % 64, wave=tid // 64,
            rl=lds.rl.ptr, fl=lds.fl.ptr, red=lds.red.ptr, args=args,
            fold=fold, aux=False, put=mb.put, put_bf=mb.put_bf,
            lin=sreg(scratch,layout["lin"][0],"lin"),
            pmix=sreg(scratch,layout["pmix"][0],"pmix"))
        for task in range(plan.stages["k1.slice"].first(bid), SLICES, BLOCKS):
            stage_slice_c1(c, task)

    @launcher(name)
    def launch(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, res_out: Int64, scratch: Int64, diag: Int64, layer: Int64,
        stream: fx.Stream,
    ):
        slice_kernel(res_in,pend,post_in,comb_in,pre_in,hc_fn,res_out,scratch,diag,layer).launch(
            grid=(BLOCKS,),block=(THREADS,),stream=stream,
        )
    return launch
