# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""c1 index-query stages with the mother's wv_splitk and quantization order.

Two physical wave32s represent each 64-lane logical group. The c1 writer
remains logical wave0; the other groups only mirror its token and do not
publish. Cross-half exchanges use private LDS and full CTA joins.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int64, T

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.index_query import (
    AMAX_SEED, IDX_WEIGHTS_SCALE, stage_iquant_fp4,
)
from atom.mono.device.mx import FP8_MAX, clamp_fp8
from atom.mono.device.ops import (
    CM_NT_GFX1250, bf2_f32, bf16_round, bf_hi, bf_lo, butterfly, fp8_pack4,
    lane_gather, row_shr, rsrc, traced,
)
from atom.mono.plan.execution import WAVES
from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    HIDDEN, QKV_ROWS, X8_WORDS, QX8_WORDS, Q_RANK, lds_row, poll_copy,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_k1 import stage_slice_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_norm import stage_norm_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_gate import stage_gate_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_qkv import stage_qkv_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_projection import (
    reference_fp8_loads, reference_load_x8_c1, stage_wqkv_c1,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_wqb import run_wqb_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_plan import BLOCKS, C1ReferencePlan
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import kernel_symbol, launcher
from atom.mono.device.sync import sreg


@traced
def wv_reduce_c1(v, lane, tid, red):
    """Original row_shr8/4/2, wave_shr1, bcast15/31, lane63 sequence."""
    for k in (8, 4, 2):
        v = row_shr(v, lane, k) + v
    base = tid - lane
    fx.ptr_store(v, red + tid)
    gpu.barrier()
    moved = fx.ptr_load(red + base + fx.max(lane - 1, 0))
    v = fx.Float32((lane >= 1).select(moved + v, v))
    gpu.barrier()
    row = lane // 16
    # Active bcast15 consumers are rows1/3, within each physical wave32.
    b15 = lane_gather(v.bitcast(fx.Int32), fx.max(row * 16 - 1, 0)).bitcast(fx.Float32)
    v = fx.Float32((row % 2 == 1).select(b15 + v, v))
    fx.ptr_store(v, red + tid)
    gpu.barrier()
    b31 = fx.ptr_load(red + base + 31)
    v = fx.Float32((row >= 2).select(b31 + v, v))
    gpu.barrier()
    fx.ptr_store(v, red + tid)
    gpu.barrier()
    result = fx.ptr_load(red + base + 63)
    gpu.barrier()
    return result


@traced
def stage_iquant_c1(c, task):
    """Mother FP8 quantization: logical wave0 owns 128 dims, two per lane."""
    tid, lane, red = c["tid"], c["lane"], c["red"]
    a = c["args"]
    x0, x1, w = fx.Float32(0), fx.Float32(0), fx.Float32(0)
    amax = fx.Float32(AMAX_SEED)
    if tid < 64:
        got = c["poll"]([(c["iq"], task * (ip.DIM // 2) + lane, 1),
                         (c["iw"], task, 1)])
        x0, x1 = bf2_f32(got[0][0])
        w = got[1][0].bitcast(fx.Float32)
        amax = butterfly(fx.max(fx.max(abs(x0), abs(x1)), fx.Float32(AMAX_SEED)),
                         (1, 2, 4, 8, 16), fx.max)
        fx.ptr_store(amax, red + tid)
    gpu.barrier()
    if tid < 64:
        amax = fx.max(amax, fx.ptr_load(red + (tid ^ 32)))
        scale = amax * fx.Float32(1.0 / FP8_MAX)
        inv = fx.Float32(1.0) / scale
        q = fp8_pack4(clamp_fp8(x0 * inv), clamp_fp8(x1 * inv),
                      fx.Float32(0), fx.Float32(0))
        bo.buffer_store(fx.Int16(q & 0xFFFF), rsrc(a["iq_out"]),
                        task * (ip.DIM // 2) + lane)
        if lane == 0:
            bo.buffer_store(scale, rsrc(a["iqs_out"]), task)
            bo.buffer_store((w * scale) * fx.Float32(IDX_WEIGHTS_SCALE),
                            rsrc(a["iw_out"]), task)
    gpu.barrier()


@traced
def stage_iw_c1(c, task, hidden, nbl, load_group):
    """Row ``task`` of weights_proj; wave w its tokens w, w + WAVES, ..., a group
    of WAVES tokens' normed rows at a time in LDS ``nbl`` (bf16 pairs,
    row-packed; ``load_group(first token, tokens)`` puts them there) -> IW. The
    weights load once; each (row, token) sum keeps wv_splitk's per-lane order,
    so spreading the tokens over waves changes nothing but the latency."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    a = c["args"]
    m = task
    ks = [
        k1 + k2 * 512
        for k1 in range(0, hidden, 2048)
        for k2 in range(4)
        if k1 + k2 * 512 < hidden
    ]
    wvs = []
    for k0 in ks:
        kk = fx.min(k0 + lane * 8, hidden - 8)
        wvs.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["iw_w"]),
                    (m * hidden + kk) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT_GFX1250,
                )
            )
        )
    for g in range_constexpr(0, s, WAVES):
        n = min(WAVES, s - g)
        load_group(g, n)
        tl = fx.min(wave, n - 1)
        t = tl if g == 0 else tl + g
        total = fx.Float32(0.0)
        for i in range_constexpr(len(ks)):
            k = ks[i] + lane * 8
            live = k < hidden
            kk = fx.min(k, hidden - 8)
            xv = fx.Vector(
                fx.ptr_load(
                    nbl + ((tl * hidden + kk) // 2),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            wv = wvs[i]
            ax = [bf_lo(xv[j]) for j in range(4)]
            ay = [bf_hi(xv[j]) for j in range(4)]
            bx = [bf_lo(wv[j]) for j in range(4)]
            by = [bf_hi(wv[j]) for j in range(4)]
            accx = ax[0] * bx[0]
            accy = ay[0] * by[0]
            for j in range_constexpr(1, 4):
                accx = fx.Float32(fmath.fma(ax[j], bx[j], accx))
                accy = fx.Float32(fmath.fma(ay[j], by[j], accy))
            total = fx.Float32(live.select(total + (accx + accy), total))
        w = wv_reduce_c1(total, lane, c["tid"], c["red"])
        if (lane == 0) & (wave < n):
            if const_expr(c["tok_par"]):
                # token parallel: to every rank, at the token's row of the step
                for p in range_constexpr(c["d"].tp):
                    dst = c["iw_dsts"][p]
                    c["put"](dst, (c["t0"] + t) * ip.HEADS + m, bf16_round(w))
            else:
                c["put"](c["iw"], t * ip.HEADS + m, bf16_round(w))
        gpu.barrier()


def reference_k1_symbol(fold, feeds_index, index_fp4=False, projection_diagnostic=False, tp=2):
    return kernel_symbol("gfx1250_reference_k1", fold=int(fold), ix=int(feeds_index),
                         q4=int(index_fp4), pd=int(projection_diagnostic), b=BLOCKS, t=256, **({"tp": tp} if tp != 2 else {}))


@lru_cache(None)
def build_reference_k1_c1(fold=True, feeds_index=True, index_fp4=False,
                          projection_diagnostic=False, tp=2):
    """Complete original TP2/TP4 c1 K1, including separate IW and iquant phases."""
    plan = C1ReferencePlan(tp=tp, feeds_index=feeds_index)
    layout = plan.k1_scratch_layout()
    name = reference_k1_symbol(fold, feeds_index, index_fp4, projection_diagnostic, tp)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 512, 16]
        qxl: fx.Array[fx.Int32, lds_row(QX8_WORDS), 16]
        qxsl: fx.Array[fx.Int32, lds_row(Q_RANK // 32), 16]

    @fx.struct
    class SliceLds:
        rl: fx.Array[fx.Float32, 128, 16]
        fl: fx.Array[fx.Float32, 24 * 128, 16]

    @fx.struct
    class WqkvLds:
        xl: fx.Array[fx.Int32, lds_row(X8_WORDS), 16]
        xsl: fx.Array[fx.Int32, lds_row(HIDDEN // 32), 16]

    @fx.struct
    class IndexLds:
        nbl: fx.Array[fx.Int32, lds_row(HIDDEN // 2), 16]

    @fx.union
    class StageLds:
        slice: SliceLds
        wqkv: WqkvLds
        index: IndexLds

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64,
               pre_in: Int64, hc_fn: Int64, res_out: Int64, scratch: Int64,
               diag: Int64, layer: Int64, attn_w: Int64, normed: Int64,
               weight: Int64, scale: Int64, hc_scale: Int64, hc_base: Int64,
               pre_out: Int64, post_out: Int64, comb_out: Int64, qn_w: Int64,
               kvn_w: Int64, cos: Int64, sin: Int64, pos: Int64,
               ring_rows: Int64, ring: Int64, ring_off: Int64, qr_out: Int64,
               qrs_out: Int64, wqb: Int64, wqb_s: Int64, q_out: Int64,
               iq_w: Int64, iq_ws: Int64, projection_values: Int64, iw_w: Int64,
               iq_out: Int64, iqs_out: Int64, iw_out: Int64):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        stages = alloc.allocate(StageLds)
        sl, wl, il = stages.slice.peek(), stages.wqkv.peek(), stages.index.peek()
        pointer = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        failure = fx.inttoptr(pointer, diag + fx.Int64(BLOCKS * 8 * 4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(pointer, layer)), diag,
            {"lin": 1, "pmix": 2, "x8": 3, "x8s": 4, "nb": 5, "qkv": 6,
             "qx8": 7, "qx8s": 8, "iq": 9, "iw": 10}, failure)
        args = dict(res_in=res_in, pend=pend, post_in=post_in, comb_in=comb_in,
                    pre_in=pre_in, hc_fn=hc_fn, res_out=res_out, attn_w=attn_w,
                    normed=normed, hc_scale=hc_scale, hc_base=hc_base,
                    pre_out=pre_out, post_out=post_out, comb_out=comb_out,
                    qn_w=qn_w, kvn_w=kvn_w, cos=cos, sin=sin, pos=pos,
                    ring_rows=ring_rows, ring=ring,
                    ring_off=fx.ptr_load(fx.inttoptr(fx.PointerType.get(
                        fx.Int64.ir_type, fx.AddressSpace.Global, 8), ring_off)),
                    qr_out=qr_out, qrs_out=qrs_out, q_out=q_out,
                    projection_values=projection_values, iw_w=iw_w,
                    iq_out=iq_out, iqs_out=iqs_out, iw_out=iw_out)
        c = dict(S=1, tid=tid, bid=bid, lane=tid % 64, wave=tid // 64, threads=256,
                 rl=sl.rl.ptr, fl=sl.fl.ptr, red=lds.red.ptr,
                 xl=wl.xl.ptr, xsl=wl.xsl.ptr, qxl=lds.qxl.ptr, qxsl=lds.qxsl.ptr,
                 args=args, fold=fold, aux=False, index=feeds_index, tok_par=False,
                 projection_diagnostic=projection_diagnostic, d=plan.dims,
                 put=mb.put, put_words=mb.put_words, put_bf=mb.put_bf, poll=mb.poll)
        for region in layout:
            c[region] = sreg(scratch, layout[region][0], region)
        for task in range(plan.stages["k1.slice"].first(bid), 160, BLOCKS):
            stage_slice_c1(c, task)
        for task in range(plan.stages["k1.norm"].first(bid), 1, BLOCKS):
            stage_norm_c1(c)
        task = plan.stages["k1.wqkv_a"].first(bid)
        if task < QKV_ROWS // 16:
            matrix = dict(c, lane=tid % 32, wave=tid // 32)
            ops = reference_fp8_loads(matrix, weight, scale, HIDDEN, task)
            reference_load_x8_c1(matrix)
            stage_wqkv_c1(matrix, task, ops)
        for task in range(plan.stages["k1.gate"].first(bid), 1, BLOCKS):
            stage_gate_c1(dict(c, lane=tid % 32, wave=tid // 32), task)
        for task in range(plan.stages["k1.qkv"].first(bid), 1, BLOCKS):
            stage_qkv_c1(c, task)
        run_wqb_c1(c, plan, wqb, wqb_s, iq_w, iq_ws)
        if const_expr(feeds_index):
            def load_group(g, n):
                poll_copy(c, tid, 256,
                          [(c["nb"], lambda u: g * HIDDEN // 2 + u,
                            n * HIDDEN // 2, il.nbl.ptr, None)])
                gpu.barrier()
            for task in range(plan.stages["k1.index_w"].first(bid), ip.HEADS, BLOCKS):
                stage_iw_c1(c, task, HIDDEN, il.nbl.ptr, load_group)
            for task in range(plan.stages["k1.index_quant"].first(bid), ip.HEADS, BLOCKS):
                if const_expr(index_fp4):
                    stage_iquant_fp4(c, task)
                else:
                    stage_iquant_c1(c, task)

    @launcher(name)
    def launch(res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64,
               pre_in: Int64, hc_fn: Int64, res_out: Int64, scratch: Int64,
               diag: Int64, layer: Int64, attn_w: Int64, normed: Int64,
               weight: Int64, scale: Int64, hc_scale: Int64, hc_base: Int64,
               pre_out: Int64, post_out: Int64, comb_out: Int64, qn_w: Int64,
               kvn_w: Int64, cos: Int64, sin: Int64, pos: Int64,
               ring_rows: Int64, ring: Int64, ring_off: Int64, qr_out: Int64,
               qrs_out: Int64, wqb: Int64, wqb_s: Int64, q_out: Int64,
               iq_w: Int64, iq_ws: Int64, projection_values: Int64, iw_w: Int64,
               iq_out: Int64, iqs_out: Int64, iw_out: Int64, stream: fx.Stream):
        kernel(res_in, pend, post_in, comb_in, pre_in, hc_fn, res_out, scratch,
               diag, layer, attn_w, normed, weight, scale, hc_scale, hc_base,
               pre_out, post_out, comb_out, qn_w, kvn_w, cos, sin, pos, ring_rows,
               ring, ring_off, qr_out, qrs_out, wqb, wqb_s, q_out, iq_w, iq_ws,
               projection_values, iw_w, iq_out, iqs_out, iw_out).launch(
                   grid=(BLOCKS,), block=(256,), stream=stream)
    return launch
