# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 four-task WQB and interleaved IQ on eight physical waves.

The first WQB pair and IQ load before the single private q-input copy; the
second WQB pair loads afterwards. This binding retains original c1 ownership,
input lifetime, two-pair order and RoPE/tagged IQ epilogues.
"""
from atom.models.deepseek_v41.mono import index_plan as ip
from functools import lru_cache
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import (
    HIDDEN,QKV_ROWS,X8_WORDS,QX8_WORDS,Q_RANK,KV_DIM,ROPE,HALF,EPS,
    lds_row,ld_bf,ld_i32,butterfly,hw_rsq,div_rn,bf16_round,
    code_round,pow2,fp8_pack4,code_ceil,FP8_MAX,fp8_value,clamp_fp8,
    wqb_rope,iq_rope,HEAD_DIM,IDX_ROPE,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_k1 import stage_slice_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_norm import stage_norm_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_gate import stage_gate_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_qkv import stage_qkv_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_plan import BLOCKS,THREADS,WAVES,C1ReferencePlan
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_projection import reference_fp8_loads,reference_fp8_matrices_c1,stage_wqkv_c1,reference_load_x8_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import kernel_symbol,launcher,rsrc,traced
from atom.mono.device.sync import sreg

@traced
def stage_qprojection_c1(c,task,ops,rope,index=False):
    reference_fp8_matrices_c1(c,Q_RANK,ops)
    tid=c["tid"]
    projection_diagnostic = c["projection_diagnostic"]
    projection_values = rsrc(c["args"]["projection_values"])
    if tid<8:
        row0=task*16+2*tid
        vs=[]
        for j in range_constexpr(2):
            total=fx.Float32(0)
            for w in range_constexpr(WAVES):
                total=total+fx.ptr_load(c["red"]+w*16+2*tid+j)
            vs.append(bf16_round(total))
            if const_expr(projection_diagnostic):
                bo.buffer_store(vs[j], projection_values, (16384 if index else 0)+row0+j)
        dim=ip.DIM if index else HEAD_DIM
        rotated=IDX_ROPE if index else ROPE
        live=row0%dim>=dim-rotated
        cs,sn=rope
        e0=fx.Float32(live.select(vs[0]*cs-vs[1]*sn,vs[0]))
        e1=fx.Float32(live.select(vs[0]*sn+vs[1]*cs,vs[1]))
        if const_expr(index):
            c["put_bf"](c["iq"],row0,[e0,e1])
        else:
            bo.buffer_store(e0.to(fx.BFloat16),rsrc(c["args"]["q_out"]),row0)
            bo.buffer_store(e1.to(fx.BFloat16),rsrc(c["args"]["q_out"]),row0+1)
    gpu.barrier()


@traced
def run_wqb_c1(c, plan, wqb, wqb_s, iq_w, iq_ws):
    """Mother's four WQB tasks and optional IQ share one private input copy."""
    tid, bid = c["tid"], c["bid"]
    matrix = dict(c, lane=tid % 32, wave=tid // 32,
                  xl=c["qxl"], xsl=c["qxsl"])
    per = plan.dims.wqb_tasks // BLOCKS
    assert per in (2, 4)
    first = plan.stages["k1.wq_b"].first(bid)
    first_ops = [reference_fp8_loads(matrix, wqb, wqb_s, Q_RANK, first+i*BLOCKS)
                 for i in range(2)]
    if const_expr(c["index"]):
        iq_task = plan.stages["k1.iq"].first(bid)
        iq_ops = reference_fp8_loads(matrix, iq_w, iq_ws, Q_RANK, iq_task)
    ropes = [wqb_rope(c, first+i*BLOCKS, 0, 1) for i in range(per)]
    if const_expr(c["index"]):
        iq_ropes = iq_rope(c, iq_task, 0, 1)
    reference_load_x8_c1(matrix, Q_RANK, "qx8", "qx8s")
    for i in range_constexpr(2):
        stage_qprojection_c1(matrix, first+i*BLOCKS, first_ops[i], ropes[i])
    if const_expr(per > 2):
        next_ops = [reference_fp8_loads(matrix, wqb, wqb_s, Q_RANK, first+(2+i)*BLOCKS)
                    for i in range(2)]
        for i in range_constexpr(2):
            stage_qprojection_c1(matrix, first+(2+i)*BLOCKS, next_ops[i], ropes[2+i])
    if const_expr(c["index"]):
        stage_qprojection_c1(matrix, iq_task, iq_ops, iq_ropes, index=True)


def wqb_symbol(fold,feeds_index,threads=256,projection_diagnostic=False):
    return kernel_symbol("gfx1250_reference_k1_wqb",fold=int(fold),ix=int(feeds_index),b=BLOCKS,t=threads,pd=int(projection_diagnostic))

@lru_cache(None)
def build_wqb_c1(fold=True, feeds_index=False, threads=256, projection_diagnostic=False):
    """Mother K1 source order through WQB/IQ, eight physical waves."""
    if threads != 256:
        raise ValueError("reference c1 QKV requires 256 physical threads")
    plan = C1ReferencePlan(feeds_index=feeds_index)
    layout = plan.k1_scratch_layout()
    name = wqb_symbol(fold, feeds_index, threads, projection_diagnostic)

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

    @fx.union
    class StageLds:
        slice: SliceLds
        wqkv: WqkvLds

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def projection(res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64,
                   pre_in: Int64, hc_fn: Int64, res_out: Int64, scratch: Int64,
                   diag: Int64, layer: Int64, attn_w: Int64, normed: Int64,
                   weight: Int64, scale: Int64, hc_scale: Int64, hc_base: Int64,
                   pre_out: Int64, post_out: Int64, comb_out: Int64, qn_w: Int64, kvn_w: Int64, cos: Int64, sin: Int64, pos: Int64,
                   ring_rows: Int64, ring: Int64, ring_off: Int64, qr_out: Int64, qrs_out: Int64, wqb: Int64, wqb_s: Int64, q_out: Int64, iq_w: Int64, iq_ws: Int64, projection_values: Int64):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator(); lds = alloc.allocate(Smem).peek()
        stages = alloc.allocate(StageLds)
        sl, wl = stages.slice.peek(), stages.wqkv.peek()
        pointer = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        failure = fx.inttoptr(pointer, diag + fx.Int64(BLOCKS * 8 * 4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(pointer, layer)), diag,
            {"lin": 1, "pmix": 2, "x8": 3, "x8s": 4, "nb": 5, "qkv": 6, "qx8":7,"qx8s":8,"iq":9}, failure)
        args = dict(res_in=res_in, pend=pend, post_in=post_in, comb_in=comb_in,
                    pre_in=pre_in, hc_fn=hc_fn, res_out=res_out, attn_w=attn_w, normed=normed,
                    hc_scale=hc_scale, hc_base=hc_base, pre_out=pre_out, post_out=post_out, comb_out=comb_out,
                    qn_w=qn_w,kvn_w=kvn_w,cos=cos,sin=sin,pos=pos,ring_rows=ring_rows,ring=ring,
                    ring_off=fx.ptr_load(fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type,fx.AddressSpace.Global,8),ring_off)),
                    qr_out=qr_out,qrs_out=qrs_out,q_out=q_out,projection_values=projection_values)
        c = dict(S=1, tid=tid, bid=bid, lane=tid % 64, wave=tid // 64, threads=threads,
                 rl=sl.rl.ptr, fl=sl.fl.ptr, red=lds.red.ptr, xl=wl.xl.ptr, xsl=wl.xsl.ptr,
                 args=args, fold=fold, aux=False, index=feeds_index, tok_par=False,projection_diagnostic=projection_diagnostic,d=plan.dims,qxl=lds.qxl.ptr,qxsl=lds.qxsl.ptr,
                 put=mb.put, put_words=mb.put_words, put_bf=mb.put_bf, poll=mb.poll)
        for region in ("lin", "pmix", "x8", "x8s", "nb", "qkv", "qx8", "qx8s", "iq"):
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
        for task in range(plan.stages["k1.qkv"].first(bid), 1, BLOCKS):
            stage_qkv_c1(c, task)
        run_wqb_c1(c, plan, wqb, wqb_s, iq_w, iq_ws)

    @launcher(name)
    def launch(res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64,
               pre_in: Int64, hc_fn: Int64, res_out: Int64, scratch: Int64,
               diag: Int64, layer: Int64, attn_w: Int64, normed: Int64,
               weight: Int64, scale: Int64, hc_scale: Int64, hc_base: Int64,
               pre_out: Int64, post_out: Int64, comb_out: Int64, qn_w: Int64, kvn_w: Int64, cos: Int64, sin: Int64, pos: Int64,
                   ring_rows: Int64, ring: Int64, ring_off: Int64, qr_out: Int64, qrs_out: Int64, wqb: Int64, wqb_s: Int64, q_out: Int64, iq_w: Int64, iq_ws: Int64, projection_values: Int64, stream: fx.Stream):
        projection(res_in, pend, post_in, comb_in, pre_in, hc_fn, res_out,
                   scratch, diag, layer, attn_w, normed, weight, scale, hc_scale, hc_base, pre_out, post_out, comb_out,
                   qn_w,kvn_w,cos,sin,pos,ring_rows,ring,ring_off,qr_out,qrs_out,wqb,wqb_s,q_out,iq_w,iq_ws,projection_values).launch(
                       grid=(BLOCKS,), block=(threads,), stream=stream)
    return launch
