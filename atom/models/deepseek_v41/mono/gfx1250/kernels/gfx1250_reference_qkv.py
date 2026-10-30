# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Original c1 QKV wait, dual norm, quantization and ring publication.

The xor32 reduction is bridged in private LDS; the remaining frozen mother
stage_qkv arithmetic, weight/table-before-wait order and outputs are retained.
"""
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
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_k1 import stage_slice_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_norm import stage_norm_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_gate import stage_gate_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_plan import BLOCKS,THREADS,WAVES,C1ReferencePlan
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_projection import reference_fp8_loads,stage_wqkv_c1,reference_load_x8_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import kernel_symbol,launcher,rsrc,traced
from atom.mono.device.sync import sreg

@traced
def stage_qkv_c1(c, t):
    """Token t: q_norm -> MXFP8 (the Triton dual norm, 4 warps x 8 columns) and
    kv_norm -> GPT-J RoPE of the tail -> bf16 -> group-32 FP8 QAT -> the ring.

    Threads 256.. mirror thread 255's loads (so every wave reaches the barrier)
    and store nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    qc = [fx.min(tt * 8 + j, Q_RANK - 1) for j in range(8)]
    # every weight and table value the token's rows do not decide, loaded
    # before their wait
    qnw = [ld_bf(a["qn_w"], qc[j]) for j in range(8)]
    kvnw = [ld_bf(a["kvn_w"], tt * 2 + j) for j in range(2)]
    pos = ld_i32(a["pos"], 2 * t)
    jr = tt - (KV_DIM - ROPE) // 2
    live = jr >= 0
    jj = fx.max(jr, 0)
    cs = live.select(ld_bf(a["cos"], pos * HALF + jj), fx.Float32(1.0))
    sn = live.select(ld_bf(a["sin"], pos * HALF + jj), fx.Float32(0.0))
    rel = ld_i32(a["ring_rows"], t)
    got = c["poll"](
        [(c["qkv"], t * QKV_ROWS + qc[j], 1) for j in range(8)]
        + [(c["qkv"], t * QKV_ROWS + Q_RANK + tt * 2 + j, 1) for j in range(2)]
    )
    qv, kvv = got[:8], got[8:]
    xq = [
        ((tt * 8 + j) < Q_RANK).select(qv[j][0].bitcast(fx.Float32), fx.Float32(0.0))
        for j in range(8)
    ]
    xk = [kvv[j][0].bitcast(fx.Float32) for j in range(2)]
    ssq = xq[0] * xq[0]
    for j in range_constexpr(1, 8):
        ssq = ssq + xq[j] * xq[j]
    # Retain the mother's xor8,4,2,1,16,32 order across two wave32s.
    for off in (8, 4, 2, 1, 16):
        ssq = ssq + ssq.shuffle_xor(off, 32)
    ssk = xk[0] * xk[0] + xk[1] * xk[1]
    for off in (8, 4, 2, 1, 16):
        ssk = ssk + ssk.shuffle_xor(off, 32)
    fx.ptr_store(ssq, red + tid)
    fx.ptr_store(ssk, red + 256 + tid)
    gpu.barrier()
    ssq = ssq + fx.ptr_load(red + (tid ^ 32))
    ssk = ssk + fx.ptr_load(red + 256 + (tid ^ 32))
    gpu.barrier()
    if (lane == 63) & mine:
        fx.ptr_store(ssq, red + wave)
        fx.ptr_store(ssk, red + 4 + wave)
    gpu.barrier()
    tq = butterfly(fx.ptr_load(red + lane % 4), (2, 1))
    tk = butterfly(fx.ptr_load(red + 4 + lane % 4), (2, 1))
    nq = hw_rsq(
        div_rn(tq, fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK))
        + fx.Float32(EPS)
    )
    nk = hw_rsq(
        div_rn(tk, fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM))
        + fx.Float32(EPS)
    )
    # q: MXFP8 by _mxfp8_quant_op, a group of 32 = 4 threads
    ys = [(xq[j] * nq) * qnw[j] for j in range(8)]
    amax = fx.Float32(0.0)
    for y in ys:
        amax = fx.max(amax, abs(y))
    code = code_round(butterfly(amax, (1, 2), fx.max))
    mul = pow2(254 - code)
    if mine & (tt * 8 < Q_RANK):
        w0 = fp8_pack4(*[ys[i] * mul for i in range(4)])
        w1 = fp8_pack4(*[ys[4 + i] * mul for i in range(4)])
        if const_expr(c["tok_par"]):
            # token parallel: to every rank, at the token's row of the step
            tg = c["t0"] + t
            for p in range_constexpr(c["d"].tp):
                qw, qs = c["q_dsts"][p]
                c["put_words"](qw, tg * QX8_WORDS + 2 * tt, [w0, w1])
                if tt % 4 == 0:
                    c["put"](qs, tg * (Q_RANK // 32) + tt // 4, code)
        else:
            c["put_words"](c["qx8"], t * QX8_WORDS + 2 * tt, [w0, w1])
            if tt % 4 == 0:
                c["put"](c["qx8s"], t * (Q_RANK // 32) + tt // 4, code)
        if const_expr(c["index"]):
            # the (qr, qr_scale) pair the original indexer reads
            bo.buffer_store(
                fx.Vector.from_elements([w0, w1], fx.Int32),
                rsrc(a["qr_out"]),
                t * QX8_WORDS + 2 * tt,
            )
            if tt % 4 == 0:
                bo.buffer_store(
                    fx.Int8(code), rsrc(a["qrs_out"]), t * (Q_RANK // 32) + tt // 4
                )
    # kv: this thread's pair (2 tt, 2 tt + 1), RoPE on the last 64 columns
    kn = [bf16_round((xk[j] * nk) * kvnw[j]) for j in range(2)]
    e0 = bf16_round(kn[0] * cs - kn[1] * sn)
    e1 = bf16_round(kn[0] * sn + kn[1] * cs)
    amax = butterfly(fx.max(abs(e0), abs(e1)), (1, 2, 4, 8), fx.max)
    step = pow2(code_ceil(fx.max(amax, fx.Float32(1e-4)) * fx.Float32(1.0 / FP8_MAX)))
    qat = [fp8_value(clamp_fp8(v / step)) * step for v in (e0, e1)]
    # the ring row of a [rows, 512] bf16 pool: the step's row of this token (int32,
    # -1: none) past the layer's ring start
    row = rel + a["ring_off"]
    if mine & (rel >= 0):
        dst = rsrc(a["ring"] + fx.Int64(row) * (KV_DIM * 2))
        for j in range_constexpr(2):
            bo.buffer_store(qat[j].to(fx.BFloat16), dst, tt * 2 + j)
    if const_expr(c["tok_par"]):  # noqa: SIM102
        # token parallel: the row to every other rank (``ring_row_from``)
        if mine:
            tg = c["t0"] + t
            for p in range_constexpr(c["d"].tp):
                if c["rank"] != p:
                    c["put_bf"](c["kv_dsts"][p], tg * KV_DIM + 2 * tt, qat)
    gpu.barrier()

def qkv_symbol(fold,feeds_index,threads=256):
    return kernel_symbol("gfx1250_reference_k1_qkv",fold=int(fold),ix=int(feeds_index),b=BLOCKS,t=threads)


@lru_cache(None)
def build_qkv_c1(fold=True, feeds_index=False, threads=256):
    """Mother K1 source order through CTA32 q/kv, eight physical waves."""
    if threads != 256:
        raise ValueError("reference c1 QKV requires 256 physical threads")
    plan = C1ReferencePlan(feeds_index=feeds_index)
    layout = plan.k1_scratch_layout()
    name = qkv_symbol(fold, feeds_index, threads)

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
                   pre_out: Int64, post_out: Int64, comb_out: Int64, qn_w: Int64, kvn_w: Int64, cos: Int64, sin: Int64, pos: Int64,
                   ring_rows: Int64, ring: Int64, ring_off: Int64, qr_out: Int64, qrs_out: Int64):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator(); lds = alloc.allocate(Smem).peek()
        stages = alloc.allocate(StageLds)
        sl, wl = stages.slice.peek(), stages.wqkv.peek()
        pointer = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        failure = fx.inttoptr(pointer, diag + fx.Int64(BLOCKS * 8 * 4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(pointer, layer)), diag,
            {"lin": 1, "pmix": 2, "x8": 3, "x8s": 4, "nb": 5, "qkv": 6, "qx8":7,"qx8s":8}, failure)
        args = dict(res_in=res_in, pend=pend, post_in=post_in, comb_in=comb_in,
                    pre_in=pre_in, hc_fn=hc_fn, res_out=res_out, attn_w=attn_w, normed=normed,
                    hc_scale=hc_scale, hc_base=hc_base, pre_out=pre_out, post_out=post_out, comb_out=comb_out,
                    qn_w=qn_w,kvn_w=kvn_w,cos=cos,sin=sin,pos=pos,ring_rows=ring_rows,ring=ring,
                    ring_off=fx.ptr_load(fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type,fx.AddressSpace.Global,8),ring_off)),
                    qr_out=qr_out,qrs_out=qrs_out)
        c = dict(S=1, tid=tid, bid=bid, lane=tid % 64, wave=tid // 64, threads=threads,
                 rl=sl.rl.ptr, fl=sl.fl.ptr, red=lds.red.ptr, xl=wl.xl.ptr, xsl=wl.xsl.ptr,
                 args=args, fold=fold, aux=False, index=feeds_index, tok_par=False,
                 put=mb.put, put_words=mb.put_words, put_bf=mb.put_bf, poll=mb.poll)
        for region in ("lin", "pmix", "x8", "x8s", "nb", "qkv", "qx8", "qx8s"):
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

    @launcher(name)
    def launch(res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64,
               pre_in: Int64, hc_fn: Int64, res_out: Int64, scratch: Int64,
               diag: Int64, layer: Int64, attn_w: Int64, normed: Int64,
               weight: Int64, scale: Int64, hc_scale: Int64, hc_base: Int64,
               pre_out: Int64, post_out: Int64, comb_out: Int64, qn_w: Int64, kvn_w: Int64, cos: Int64, sin: Int64, pos: Int64,
                   ring_rows: Int64, ring: Int64, ring_off: Int64, qr_out: Int64, qrs_out: Int64, stream: fx.Stream):
        projection(res_in, pend, post_in, comb_in, pre_in, hc_fn, res_out,
                   scratch, diag, layer, attn_w, normed, weight, scale, hc_scale, hc_base, pre_out, post_out, comb_out,
                   qn_w,kvn_w,cos,sin,pos,ring_rows,ring,ring_off,qr_out,qrs_out).launch(
                       grid=(BLOCKS,), block=(threads,), stream=stream)
    return launch
