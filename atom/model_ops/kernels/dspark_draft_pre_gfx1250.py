# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSpark draft layer's attention head as one launch (gfx1250, MiniMax-M3.1 TP4, decode rows <= 8): the fused
qkv MXFP8 GEMV (16 q heads, 1 k, 1 v a rank), then per row and head the q / k Gemma RMSNorm, the neox RoPE of the
first 64 lanes, and the per-token fp8 K / V cache writes (page-16 SHUFFLE layout). Production: the qkv GEMM
(gemm_a8w8, split-K 4 in a cluster) and _fused_qkv_norm_rope_cache (Triton), two launches.

Workgroups (128 threads, clusters of 8 = one head): head h x K quarter s x n-block half; a wave one 16-column
n-block and the quarter's 12 K steps (production's WMMA chain of the split, all weight loads in flight at entry).
The cluster exchanges its FP32 partials; workgroup c of the cluster then takes row c: the split-order sum and
bf16 rounding (production's qkv output), and the head epilogue with a thread a lane of the head.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_load,
    ptr_buf_tensor,
)

from .dspark_draft_mlp_gfx1250 import (
    A_ROW,
    BLOCK,
    CL,
    HIDDEN,
    KS,
    MAXR,
    SPS,
    WAVE,
    WMMA_K,
    _bf_lo,
    _cat16,
    _cluster_sync,
    _f2bf2,
    _fabs,
    _fmax,
    _lds_barrier,
    _now32,
    _rmem,
    _rsq,
)

# weights stream non-temporally (TH NT, as production's decode GEMMs, wcpol 1): checklist item 7
WCPOL = 1
K1 = HIDDEN
NQ = 16  # q heads a rank
NH = NQ + 2  # + k + v
HD = 128
GRID = NH * CL  # a cluster a head: 4 K quarters x 2 n-block halves
PART_BYTES = NH * KS * 2 * 4 * 256 * 4
FP8_MAX = 448.0
LDS_BYTES = MAXR * A_ROW + 4096


@flyc.jit
def launch_draft_pre(
    arg_x: fx.Pointer,
    arg_xs: fx.Pointer,
    arg_w: fx.Pointer,
    arg_ws: fx.Pointer,
    arg_qw: fx.Pointer,
    arg_kw: fx.Pointer,
    arg_cos: fx.Pointer,
    arg_sin: fx.Pointer,
    arg_pos: fx.Pointer,
    arg_slot: fx.Pointer,
    arg_kc: fx.Pointer,
    arg_vc: fx.Pointer,
    arg_ks: fx.Pointer,
    arg_vs: fx.Pointer,
    arg_q: fx.Pointer,
    arg_part: fx.Pointer,
    arg_tl: fx.Pointer,
    i32_m: fx.Int32,
    i32_lda: fx.Int32,
    i32_cs_stride: fx.Int32,
    f32_eps: fx.Float32,
    stream: fx.Stream,
    cs_bf16: Constexpr[int] = 0,
    tl: Constexpr[int] = 0,
    pos_i64: Constexpr[int] = 1,
    slot_i64: Constexpr[int] = 1,
):
    """``arg_x`` e4m3 [m, 6144] (row stride ``i32_lda`` bytes) + ``arg_xs`` e8m0 [m, 192]; ``arg_w``/``arg_ws``
    the fused qkv weight [2304, 6144] (preshuffled e4m3 + shuffled e8m0); ``arg_qw``/``arg_kw`` bf16 [128] (Gemma);
    ``arg_cos``/``arg_sin`` [max_pos, 32] (row stride ``i32_cs_stride`` elements; bf16 with cs_bf16, else fp32);
    ``arg_pos``/``arg_slot`` int64 (or int32: pos_i64 / slot_i64 0) [m]; the page-16 SHUFFLE fp8 caches ``arg_kc`` [P16, 1, 8, 16, 16] and ``arg_vc``
    [P16, 1, 1, 128, 16] with fp32 per-slot scales ``arg_ks``/``arg_vs``; out ``arg_q`` bf16 [m, 16, 128];
    ``arg_part`` PART_BYTES of scratch."""
    kernel_name = (f"dspark_draft_pre_gfx1250{'_csbf16' if cs_bf16 else ''}{'_tl' if tl else ''}"
                   f"{'' if pos_i64 else '_p32'}{'' if slot_i64 else '_s32'}")

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK, 1, 1])
    def kernel_draft_pre(
        arg_x: fx.Pointer,
        arg_xs: fx.Pointer,
        arg_w: fx.Pointer,
        arg_ws: fx.Pointer,
        arg_qw: fx.Pointer,
        arg_kw: fx.Pointer,
        arg_cos: fx.Pointer,
        arg_sin: fx.Pointer,
        arg_pos: fx.Pointer,
        arg_slot: fx.Pointer,
        arg_kc: fx.Pointer,
        arg_vc: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_vs: fx.Pointer,
        arg_q: fx.Pointer,
        arg_part: fx.Pointer,
        arg_tl: fx.Pointer,
        i32_m: fx.Int32,
        i32_lda: fx.Int32,
        i32_cs_stride: fx.Int32,
        f32_eps: fx.Float32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        tlp = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4), arg_tl
        )

        def stamp(slot):
            if const_expr(tl):
                if tid == 0:
                    fx.ptr_store(_now32(), tlp + (bid * 4 + slot))

        stamp(0)
        arena = fx.SharedAllocator(static=False)
        arena.allocate(LDS_BYTES)
        base_ptr = arena.base_ptr
        lds_idx = fx.Int64(fx.ptrtoint(base_ptr))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        lds_f32 = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4),
            base_ptr,
        )
        atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(16, 16, WMMA_K, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0, opsel_b=0)
        )
        h = bid // CL
        cr = bid % CL
        s = cr // 2
        hb = cr % 2
        # ------------------------------------------------------------- the GEMV: an n-block x a K quarter a wave
        nblk = h * 8 + hb * 4 + wave
        ks0 = s * SPS
        b_buf = ptr_buf_tensor(arg_w, fx.Int32, unit_elems=4)
        sb_buf = ptr_buf_tensor(arg_ws, fx.Int32)
        b_base = nblk * K1 + lane
        sb_col = nblk * 16 + lane16
        sb_base = (sb_col // 32) * (K1 // 4) + sb_col % 32
        bfr = [
            [buf_copy_load(b_buf, b_base + (ks0 + i) * 128 + jj * 32, fx.Int32, 4, cache_modifier=WCPOL) for jj in range_constexpr(4)]
            for i in range_constexpr(SPS)
        ]
        sb = [buf_copy_load(sb_buf, sb_base + (ks0 + i) * 32, fx.Int32, cache_modifier=WCPOL) for i in range_constexpr(SPS)]
        a_buf = ptr_buf_tensor(arg_x, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda)
        sa_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_xs)), num_records_bytes=i32_m * (K1 // 32))
        lda16 = i32_lda // 16
        QU = SPS * WMMA_K // 16
        xa = []
        for t in range_constexpr(MAXR * QU // BLOCK):
            q = tid + t * BLOCK
            r = q // QU
            c = q % QU
            xa.append(buf_copy_load(a_buf, r * lda16 + s * QU + c, fx.Int32, 4))
        sa = [fx.Int32(buffer_load(sa_rs, lane16 * (K1 // 128) + ks0 + i, vec_width=1, dtype=T.i32))
              for i in range_constexpr(SPS)]
        for t in range_constexpr(MAXR * QU // BLOCK):
            q = tid + t * BLOCK
            r = q // QU
            c = q % QU
            lds_store_b128(lds_idx, r * A_ROW + c * 16, xa[t])
        _lds_barrier()
        acc = fx.make_rmem_tensor(8, fx.Float32)
        acc.store(fx.full(8, 0.0, fx.Float32))
        for i in range_constexpr(SPS):
            base = lane16 * A_ROW + i * WMMA_K + kgrp * 16
            afr = [Vec(lds_load_b128(lds_idx, base + 32 * jj)) for jj in range_constexpr(4)]
            fx.gemm(atom, acc, _rmem(_cat16(bfr[i])), _rmem(_cat16(afr)), acc, scale_a=sb[i], scale_b=sa[i])
        stamp(1)
        part_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_part)), num_records_bytes=PART_BYTES)
        v = Vec(acc.load())
        poff = ((((h * KS + s) * 2 + hb) * 4 + wave) * 256 + lane16 * 16 + kgrp * 8) * 4
        buffer_store(v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32).ir_value(), part_rs, poff, offset_is_bytes=True)
        buffer_store(v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32).ir_value(), part_rs, poff + 16, offset_is_bytes=True)
        rocdl.s_wait_storecnt(0)
        _cluster_sync()
        # ------------------------------------------------------------- row cr of head h: a thread a lane d
        r = cr
        d = tid
        live = r < i32_m
        rr = live.select(r, fx.Int32(0))
        hb_d = d // 64
        w_d = (d % 64) // 16
        col = d % 16
        tot = None
        for sp in range_constexpr(KS):
            o_ = (((h * KS + sp) * 2 + hb_d) * 4 + w_d) * 256 + rr * 16 + col
            pv = fx.Float32(buffer_load(part_rs, o_, vec_width=1, dtype=T.f32))
            tot = pv if tot is None else tot + pv
        x = _bf_lo(_f2bf2(tot, tot))  # production's bf16 qkv output

        def cta_sum(val, slot0, fn):
            """fn-reduction of one fp32 a thread over the CTA (wave tree, then 4 wave partials via LDS)."""
            for d_ in range_constexpr(5):
                val = fn(val, val.shuffle_xor(fx.Int32(1 << d_), fx.Int32(WAVE)))
            if lane == 0:
                fx.ptr_store(val, lds_f32 + (slot0 + wave))
            _lds_barrier()
            p = [fx.Float32(lds_f32[slot0 + i]) for i in range_constexpr(4)]
            return fn(fn(p[0], p[1]), fn(p[2], p[3]))

        def load_index(p, i64):
            # positions / slots: int64 or int32 (the low word of a non-negative or -1 int64 is enough)
            rs_ = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(p)), num_records_bytes=i32_m * (8 if i64 else 4))
            return fx.Int32(buffer_load(rs_, rr * (2 if i64 else 1), vec_width=1, dtype=T.i32))

        pos = load_index(arg_pos, pos_i64)
        slot = load_index(arg_slot, slot_i64)
        is_v = h == NQ + 1
        is_k = h == NQ
        # Gemma RMSNorm of q / k heads (the v head passes through)
        sq = cta_sum(x * x, 0, lambda a, b: a + b)
        var = sq * fx.Float32(1.0 / HD)
        rs = _rsq(var + f32_eps)
        nw_rs = create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(arg_kw)), num_records_bytes=HD * 2)
        qw_rs = create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(arg_qw)), num_records_bytes=HD * 2)
        wq = fx.Int32(buffer_load(qw_rs, d // 2, vec_width=1, dtype=T.i32))
        wk = fx.Int32(buffer_load(nw_rs, d // 2, vec_width=1, dtype=T.i32))
        ww = is_k.select(wk, wq)
        wv = (d % 2 == 1).select(ww & fx.Int32(-65536), ww << fx.Int32(16)).bitcast(fx.Float32)
        xn = x * rs * (fx.Float32(1.0) + wv)
        # neox RoPE of lanes < 64: partner d ^ 32 through LDS
        fx.ptr_store(xn, lds_f32 + (8 + d))
        _lds_barrier()
        partner = fx.Float32(lds_f32[8 + (d ^ 32)])
        rot = (d < 32).select(fx.Float32(0.0) - partner, partner)
        ci = rr * 0 + pos * i32_cs_stride + d % 32
        if const_expr(cs_bf16):
            cs_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_cos)), num_records_bytes=0x7FFFFFF0)
            sn_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_sin)), num_records_bytes=0x7FFFFFF0)
            cw_ = fx.Int32(buffer_load(cs_rs, ci // 2, vec_width=1, dtype=T.i32))
            sw_ = fx.Int32(buffer_load(sn_rs, ci // 2, vec_width=1, dtype=T.i32))
            cv = (ci % 2 == 1).select(cw_ & fx.Int32(-65536), cw_ << fx.Int32(16)).bitcast(fx.Float32)
            sv = (ci % 2 == 1).select(sw_ & fx.Int32(-65536), sw_ << fx.Int32(16)).bitcast(fx.Float32)
        else:
            cs_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_cos)), num_records_bytes=0x7FFFFFF0)
            sn_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_sin)), num_records_bytes=0x7FFFFFF0)
            cv = fx.Float32(buffer_load(cs_rs, ci, vec_width=1, dtype=T.f32))
            sv = fx.Float32(buffer_load(sn_rs, ci, vec_width=1, dtype=T.f32))
        roped = (d < 64).select(xn * cv + rot * sv, xn)
        y = is_v.select(x, roped)
        if h < NQ:
            q_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_q)), num_records_bytes=i32_m * NQ * HD * 2)
            yb = _f2bf2(y, y) & fx.Int32(0xFFFF)
            if live:
                buffer_store(yb.to(fx.Int16).ir_value(), q_rs, ((rr * NQ + h) * HD + d) * 2, offset_is_bytes=True)
        else:
            # per-token fp8: scale = amax / 448 (1 when 0), values / scale (true divisions)
            amax = cta_sum(_fabs(y), 4, _fmax)
            scale = amax / fx.Float32(FP8_MAX)
            scale = (scale == fx.Float32(0.0)).select(fx.Float32(1.0), scale)
            qv = y / scale
            pk = fx.rocdl.cvt_pk_fp8_f32(T.i32, qv.ir_value(), qv.ir_value(), fx.Int32(0).ir_value(), 0)
            byte = (fx.Int32(pk) & fx.Int32(255)).to(fx.Int8)
            page = slot // 16
            sib = slot % 16
            ok = live & (slot >= 0)
            if is_k:
                kc_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_kc)), num_records_bytes=0x7FFFFFF0)
                ks_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_ks)), num_records_bytes=0x7FFFFFF0)
                if ok:
                    buffer_store(byte.ir_value(), kc_rs, page * 2048 + (d // 16) * 256 + sib * 16 + d % 16,
                                 offset_is_bytes=True)
                    if d == 0:
                        buffer_store(scale.ir_value(), ks_rs, slot * 4, offset_is_bytes=True)
            else:
                vc_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_vc)), num_records_bytes=0x7FFFFFF0)
                vs_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_vs)), num_records_bytes=0x7FFFFFF0)
                if ok:
                    buffer_store(byte.ir_value(), vc_rs, page * 2048 + d * 16 + sib, offset_is_bytes=True)
                    if d == 0:
                        buffer_store(scale.ir_value(), vs_rs, slot * 4, offset_is_bytes=True)
        if const_expr(tl):
            rocdl.s_wait_storecnt(0)
            _lds_barrier()
        stamp(2)

    kernel_draft_pre(
        arg_x, arg_xs, arg_w, arg_ws, arg_qw, arg_kw, arg_cos, arg_sin, arg_pos, arg_slot, arg_kc, arg_vc, arg_ks,
        arg_vs, arg_q, arg_part, arg_tl, i32_m, i32_lda, i32_cs_stride, f32_eps,
        value_attrs={"rocdl.cluster_dims": f"{CL},1,1"},
    ).launch(grid=(GRID, 1, 1), block=(BLOCK, 1, 1), stream=stream, cluster=(CL, 1, 1))


launch_draft_pre.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
