# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 gate structure with virtual-thread rounds on physical wave32.

The arithmetic below is the frozen b4388fdb8 stage_gate. Only the 500 partial
owners are folded over physical threads; the eight-slice and twenty-group
orders, wave-local gate/Sinkhorn operations and output owners are preserved.
"""
from functools import lru_cache
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64
from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    HC, HIDDEN, QKV_ROWS, X8_WORDS, SLICES, GATE_BLOCK, PART, COMB0, MIX,
    K_MIX, EPS, HC_EPS, POST_MULT, LOG2E, SINKHORN, lds_row, ld_f32,
    fma, sigmoid, butterfly, hw_rsq, hw_exp2, hw_rcp, _row_sum, _col_sum,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_k1 import stage_slice_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_norm import stage_norm_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_plan import BLOCKS, THREADS, WAVES, C1ReferencePlan
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_projection import reference_fp8_loads, stage_wqkv_c1, reference_load_x8_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import kernel_symbol, launcher, rsrc, traced
from atom.mono.device.sync import sreg

@traced
def stage_gate_c1(c, t):
    """Token t's 160 slices, in slice order, -> post / comb / next pre."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red = c["red"]
    a = c["args"]
    groups = SLICES // GATE_BLOCK  # 20
    threads = c.get("threads", THREADS)
    for base_thread in range_constexpr(0, groups * PART, threads):
        idx = tid + base_thread
        if idx < groups * PART:
            g = idx // PART
            j = idx % PART
            vals = c["poll"](
                [
                    (c["pmix"], ((g * GATE_BLOCK + k) * s + t) * PART + j, 1)
                    for k in range(GATE_BLOCK)
                ]
            )
            acc = vals[0][0].bitcast(fx.Float32)
            for k in range_constexpr(1, GATE_BLOCK):
                acc = acc + vals[k][0].bitcast(fx.Float32)
            fx.ptr_store(acc, red + (g * PART + j))
    gpu.barrier()
    if wave == 0:
        # lane -> its mix: pre / post on lanes 0 .. 7, comb from lane COMB0
        j = (lane < COMB0).select(
            lane % (2 * HC), fx.min(lane - COMB0 + 2 * HC, MIX - 1)
        )
        mix = fx.ptr_load(red + j)
        sq = fx.ptr_load(red + MIX)
        for g in range_constexpr(1, groups):
            mix = mix + fx.ptr_load(red + (g * PART + j))
            sq = sq + fx.ptr_load(red + (g * PART + MIX))
        sc = [ld_f32(a["hc_scale"], i) for i in range(3)]
        base = ld_f32(a["hc_base"], j)
        # aiter's mhc_fused_post_pre_delayed_rmsnorm reduce: one rstd =
        # rsq(fma(sum, 1/K, eps)), then each gate fma(mix rstd, scale, base)
        v = mix * hw_rsq(fma(sq, fx.Float32(1.0 / K_MIX), fx.Float32(EPS)))
        if lane < HC:
            pre = sigmoid(fma(v, sc[0], base))
            bo.buffer_store(pre + fx.Float32(HC_EPS), rsrc(a["pre_out"]), t * HC + lane)
        if (lane >= HC) & (lane < 2 * HC):
            post = sigmoid(fma(v, sc[1], base))
            bo.buffer_store(
                post * fx.Float32(POST_MULT), rsrc(a["post_out"]), t * HC + lane - HC
            )
        cv = fma(v, sc[2], base)
        # Triton's exp: exp2 of (x - max) log2(e)
        m = butterfly(cv, (1, 2), fx.max)
        cv = hw_exp2((cv - m) * fx.Float32(LOG2E))
        # fast_dividef is x rcp(y); its first "+ eps" contracts into the fma
        cv = fma(cv, hw_rcp(_row_sum(cv)), fx.Float32(HC_EPS))
        cv = cv * hw_rcp(_col_sum(cv) + fx.Float32(HC_EPS))
        for _ in range_constexpr(SINKHORN - 1):
            cv = cv * hw_rcp(_row_sum(cv) + fx.Float32(HC_EPS))
            cv = cv * hw_rcp(_col_sum(cv) + fx.Float32(HC_EPS))
        if (lane >= COMB0) & (lane < COMB0 + HC * HC):
            bo.buffer_store(cv, rsrc(a["comb_out"]), t * HC * HC + lane - COMB0)
    gpu.barrier()

def gate_symbol(fold, feeds_index, threads=256):
    return kernel_symbol("gfx1250_reference_k1_gate", fold=int(fold), ix=int(feeds_index), b=BLOCKS, t=threads)


@lru_cache(None)
def build_gate_c1(fold=True, feeds_index=False, threads=256):
    """Mother slice→norm→WQKV→gate with original CTA160 gate owner."""
    if threads not in (256, 512):
        raise ValueError("reference c1 gate requires 256 or 512 physical threads")
    plan = C1ReferencePlan(feeds_index=feeds_index)
    layout = plan.k1_scratch_layout()
    name = gate_symbol(fold, feeds_index, threads)

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
                   weight: Int64, scale: Int64, hc_scale: Int64, hc_base: Int64,
                   pre_out: Int64, post_out: Int64, comb_out: Int64):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator(); lds = alloc.allocate(Smem).peek()
        stages = alloc.allocate(StageLds)
        sl, wl = stages.slice.peek(), stages.wqkv.peek()
        pointer = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        failure = fx.inttoptr(pointer, diag + fx.Int64(BLOCKS * 8 * 4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(pointer, layer)), diag,
            {"lin": 1, "pmix": 2, "x8": 3, "x8s": 4, "nb": 5, "qkv": 6}, failure)
        args = dict(res_in=res_in, pend=pend, post_in=post_in, comb_in=comb_in,
                    pre_in=pre_in, hc_fn=hc_fn, res_out=res_out, attn_w=attn_w, normed=normed,
                    hc_scale=hc_scale, hc_base=hc_base, pre_out=pre_out, post_out=post_out, comb_out=comb_out)
        c = dict(S=1, tid=tid, bid=bid, lane=tid % 64, wave=tid // 64, threads=threads,
                 rl=sl.rl.ptr, fl=sl.fl.ptr, red=lds.red.ptr, xl=wl.xl.ptr, xsl=wl.xsl.ptr,
                 args=args, fold=fold, aux=False, index=feeds_index,
                 put=mb.put, put_words=mb.put_words, put_bf=mb.put_bf, poll=mb.poll)
        for region in ("lin", "pmix", "x8", "x8s", "nb", "qkv"):
            c[region] = sreg(scratch, layout[region][0], region)
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
        for task in range(plan.stages["k1.gate"].first(bid), 1, BLOCKS):
            stage_gate_c1(dict(c, lane=tid % 32, wave=tid // 32), task)

    @launcher(name)
    def launch(res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64,
               pre_in: Int64, hc_fn: Int64, res_out: Int64, scratch: Int64,
               diag: Int64, layer: Int64, attn_w: Int64, normed: Int64,
               weight: Int64, scale: Int64, hc_scale: Int64, hc_base: Int64,
               pre_out: Int64, post_out: Int64, comb_out: Int64, stream: fx.Stream):
        projection(res_in, pend, post_in, comb_in, pre_in, hc_fn, res_out,
                   scratch, diag, layer, attn_w, normed, weight, scale, hc_scale, hc_base, pre_out, post_out, comb_out).launch(
                       grid=(BLOCKS,), block=(threads,), stream=stream)
    return launch
