# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1 NVFP4 sparse prefill attention for one TP4 rank (FlyDSL, gfx950).

Drop-in for aiter's ``sparse_pa_nvfp4_prefill``: same operands, same NVFP4 page layout
(``atom.model_ops.minimax_m3.nvfp4_pages``), bf16 output. A row (query token) attends the
keys of its 16 selected 128-token blocks up to its position, 16 query heads per KV head.

Work split as the aiter / amd-kernels prefill kernels: a workgroup owns ``tq`` (4 or 8)
consecutive rows of one KV head and walks the union of the pages they selected (first
occurrence in row-major (row, slot) order); wave w owns keys 32 w .. 32 w + 31 of every page;
S^T = K Q^T and O^T += V^T P^T on v_mfma_f32_16x16x32_f16 (lane (n, g): head n, keys 8 g ..);
the four waves' partials are combined through LDS at the end. Numerics: K / V = e2m1 x e4m3
scale exact in f16, Q bf16 -> f16, fp32 scores / softmax / accumulation, f16 P, bf16 output;
with ``tq`` 4 and ``lazy`` off bitwise amd-kernels' fp16 four-row kernel (aiter's at the same
tile), with ``lazy`` its lazy-rescale rule.

What differs from those kernels:
  * pages reach LDS by async DMA ``depth`` pages ahead of the one being computed (a ring of
    ``depth + 1`` slots) instead of one page ahead through registers;
  * ``pipe``: the page body is software-pipelined over the rows (the S^T MFMAs of the next
    hitting row are issued before the softmax and P V of the current one);
  * ``tq`` 8 (with one wave a SIMD, ``wpe`` 1): each dequantized page serves up to 8 rows.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

BLK = 128  # tokens per page == sparse block
D = 128
HG = 16  # query heads per KV head (MFMA N)
NW = 4  # waves per workgroup
KPW = BLK // NW  # keys per wave and page
TOPK = 16
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
DATA_B = BLK * D // 2  # e2m1 bytes per (page, side, head)
SCALE_B = BLK * D // 16  # e4m3 scale bytes per (page, side, head)
SLOT_B = 2 * (DATA_B + SCALE_B)  # one KV head's K and V of one page, in LDS
Q_ROW = D * 2 // 16 + 1  # 16 B LDS tiles per (row, head) of Q^T, padded by one
LAZY_LOG2 = 8.0
M_FLOOR = -8.0e37
LOG2E = 1.4426950409


# ---------------------------------------------------------------- small helpers
def _gptr(addr, f32=False):
    ty = T.f32 if f32 else T.i32
    return fx.inttoptr(
        fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=4),
        fx.Int64(addr),
    )


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _permlane_swap(op, a, b):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(a), fx.as_ir_value(b), False, False)
    return (
        fx.Int32(_llvm.extractvalue(T.i32, res, [0])),
        fx.Int32(_llvm.extractvalue(T.i32, res, [1])),
    )


def _maxf(a, b):
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def _xlane4(x, op):
    """``op`` over lanes L, L^16, L^32, L^48, broadcast to all four."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    m = op(_as_f32(a), _as_f32(b))
    a, b = _permlane_swap(_rocdl.permlane16_swap, _bits(m), _bits(m))
    return op(_as_f32(a), _as_f32(b))


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def _pk_f16(a, b):
    return (
        Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
        .to(fx.Float16)
        .bitcast(fx.Int32)[0]
    )


def _pk_bf16(a, b):
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b)))
    )


def _fp4x2_f16(word, sel, scale):
    """The two e2m1 codes of byte ``sel`` of ``word`` as two f16, times ``scale``: exact only for
    a power-of-two scale (the hardware applies its exponent), so 1.0 here."""
    return Vec(
        _rocdl_ops.cvt_scalef32_pk_f16_fp4(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(scale)),
            sel,
        )
    )


def _e4m3x2_f16(word, hi):
    return Vec(
        _rocdl_ops.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def _v_e4m3x2_f16(word, hi):
    """``_e4m3x2_f16`` of V scales with NaN mapped to -448 (finite scales are >= 0): a key past a
    row's limit has P = 0, and 0 * NaN would be NaN in the MFMA."""
    return Vec(
        _arith.MaxNumFOp(
            fx.as_ir_value(_e4m3x2_f16(word, hi)),
            fx.as_ir_value(Vec.filled(2, -448.0, fx.Float16)),
        ).result
    )


def _q_f16(v):
    return Vec(v).bitcast(fx.BFloat16).to(fx.Float32).to(fx.Float16).bitcast(fx.Int32)


def _mfma(a4, b4, acc):
    return Vec(
        fx.rocdl.mfma_f32_16x16x32_f16(
            Vec.make_type(4, fx.Float32),
            [Vec(a4).bitcast(fx.Float16), Vec(b4).bitcast(fx.Float16), acc, 0, 0, 0],
        )
    )


# ---------------------------------------------------------------- the kernel
def compile_sparse_pa_prefill(
    depth: int = 2, lazy: bool = False, pipe: bool = True, wpe: int = 2, tq: int = 4,
    pk: bool = False, lmfma: bool = False,
):
    """``depth`` pages in flight ahead of the computed one; ``lazy`` the lazy rescale; ``pipe``
    the row pipeline; ``wpe`` waves per SIMD (1: 512 registers); ``tq`` rows per workgroup;
    ``pk`` packed fp32 math for P and its sum (another sum order); ``lmfma`` the row sums by one
    extra MFMA against a ones tile (sums of the f16 P; lazy only)."""
    assert tq in (4, 8) and wpe in (1, 2) and (tq == 4 or wpe == 1)
    assert lazy or not lmfma
    TQ = tq
    NSET = TQ // 4  # 64-lane (row, slot) sets
    RING = depth + 1
    name = (
        f"m3_spa_prefill_t{TQ}_d{depth}{'_lazy' if lazy else ''}"
        f"{'' if pipe else '_nopipe'}_w{wpe}{'_pk' if pk else ''}{'_lm' if lmfma else ''}"
    )
    # LDS (bytes): the page ring [0, RING_B), Q^T [RING_B, RING_B + Q_B) during the page walk;
    # after it, the combine tile [0, COMB_B) and the (max, sum) table [COMB_B, COMB_B + ML_B).
    RING_B = RING * SLOT_B
    Q_B = TQ * HG * Q_ROW * 16
    COMB_B = NW * TQ * (NDT // 2) * 64 * 16
    ML_B = NW * TQ * 2 * 64 * 4
    LDS_B = max(RING_B + Q_B, COMB_B + ML_B)
    assert LDS_B * wpe <= 160 * 1024, LDS_B  # wpe workgroups per CU
    Q_T0 = RING_B // 16  # first 16 B tile of Q^T
    ML_W = COMB_B // 4  # first dword of the (max, sum) table

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Int32, LDS_B // 4, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [T, H, 128] bf16
        arg_cache: fx.Int64,  # uint8 pages, 2 * Hkv * 9216 B of payload each
        arg_topk: fx.Int64,  # [Hkv, >= T, 16] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_req: fx.Int64,  # [T] int32
        arg_kv: fx.Int64,  # [T] int32 causal key count (position + 1)
        arg_out: fx.Int64,  # [T, H, 128] bf16
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_topk_rs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        sp = smem.buf.ptr
        tx = fx.thread_idx.x
        bidx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = fx.Int32(tx % 64)
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = _gptr(arg_topk)
        bt = _gptr(arg_bt)
        req_p = _gptr(arg_req)
        kv_p = _gptr(arg_kv)
        c = f32_c * fx.Float32(_gptr(arg_ks, True)[0])
        v_scale = fx.Float32(_gptr(arg_vs, True)[0])
        neg_inf = fx.Float32(float("-inf"))
        lds16 = fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, sp), fx.make_layout(LDS_B // 4, 1)),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        def ld16(tile):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(lds_atom, fx.slice(lds16, (None, tile)), r)
            return Vec(r.load())

        def st16(tile, v):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(v)
            fx.copy(lds_atom, r, fx.slice(lds16, (None, tile)))

        # ---- prologue loads, the independent ones first
        row_kv, row_rq = [], []
        for t in range_constexpr(TQ):
            tok = bidx * TQ + t
            tc = (tok < i32_T).select(tok, fx.Int32(0))
            kv = fx.Int32(kv_p[tc])
            row_kv.append((tok < i32_T).select(kv, fx.Int32(0)))
            row_rq.append(fx.Int32(req_p[tc]))
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2)
        )
        tq_, hq, qq = fx.Int32(tx // 64), fx.Int32((tx // 4) % HG), fx.Int32(tx % 4)
        q4 = []
        for s in range_constexpr(NSET):
            tok_q = bidx * TQ + 4 * s + tq_
            tok_q = (tok_q < i32_T).select(tok_q, fx.Int32(0))
            q_src = (tok_q * i32_H + kvh * HG + hq) * (D // 2) + qq * 16
            q4.append(
                [
                    buffer_ops.buffer_load(qr, q_src + 4 * i, vec_width=4, dtype=fx.Int32)
                    for i in range_constexpr(4)
                ]
            )
        # lane (row 4 s + lane // 16, slot lane % 16) of set s: the page it selected, -1 none
        lane_page = []
        for s in range_constexpr(NSET):
            lane_tok = bidx * TQ + 4 * s + lane // TOPK
            lane_in = lane_tok < i32_T
            lane_tc = lane_in.select(lane_tok, fx.Int32(0))
            blk = fx.Int32(topk[kvh * i32_topk_hs + lane_tc * i32_topk_rs + lane % TOPK])
            kv_l = fx.Int32(kv_p[lane_tc])
            lreq = fx.Int32(req_p[lane_tc])
            own_l = (kv_l - 1) // BLK
            lane_ok = lane_in & (kv_l > 0) & (blk >= 0) & (blk <= own_l)
            lp = fx.Int32(bt[lreq * i32_bt_s + lane_ok.select(blk, fx.Int32(0))])
            lane_page.append(lane_ok.select(lp, fx.Int32(-1)))

        # per row: its own block's page (-2: no row) and its causal limit on this wave's keys
        diag, lim = [], []
        for t in range_constexpr(TQ):
            kv = fx.Int32(fx.rocdl.readfirstlane(T.i32, row_kv[t]))
            rq = fx.Int32(fx.rocdl.readfirstlane(T.i32, row_rq[t]))
            pos = kv - 1
            pz = (pos > 0).select(pos, fx.Int32(0))
            dg = fx.Int32(bt[rq * i32_bt_s + pz // BLK])
            dg = (kv > 0).select(dg, fx.Int32(-2))
            diag.append(fx.Int32(fx.rocdl.readfirstlane(T.i32, dg)))
            lim.append(pos % BLK - w * KPW)

        # ---- Q^T of the rows -> LDS as f16
        for s in range_constexpr(NSET):
            for i in range_constexpr(4):
                st16(Q_T0 + ((4 * s + tq_) * HG + hq) * Q_ROW + qq * 4 + i, _q_f16(q4[s][i]))

        # ---- the union: first occurrence of each page, row-major over every (row, slot) lane
        umask = []
        for s in range_constexpr(NSET):
            n_dup = fx.Int32(0)
            for s2 in range_constexpr(s):
                for j in range_constexpr(64):
                    kj = fx.Int32(fx.rocdl.readlane(T.i32, lane_page[s2], j))
                    n_dup = n_dup + (lane_page[s] == kj).select(fx.Int32(1), fx.Int32(0))
            for j in range_constexpr(63):
                kj = fx.Int32(fx.rocdl.readlane(T.i32, lane_page[s], j))
                n_dup = n_dup + ((lane_page[s] == kj) & (lane > j)).select(
                    fx.Int32(1), fx.Int32(0)
                )
            umask.append(fx.Int64(fx.rocdl.ballot(T.i64, (lane_page[s] >= 0) & (n_dup == 0))))

        def first_page1(masks):
            """(page of the lowest lane of the mask, the mask without it)."""
            m0 = masks[0]
            idx = (m0 != fx.Int64(0)).select(fx.Int32(fx.math.cttz(m0).to(fx.Int32)), fx.Int32(0))
            pg = fx.Int32(fx.rocdl.readlane(T.i32, lane_page[0], idx))
            return (pg >= 0).select(pg, fx.Int32(0)), [m0 & (m0 - fx.Int64(1))]

        def first_page2(masks):
            """(page of the lowest lane of the first non-empty mask, the masks without it)."""
            m0, m1 = masks
            use0 = m0 != fx.Int64(0)
            mm = use0.select(m0, m1)
            idx = (mm != fx.Int64(0)).select(fx.Int32(fx.math.cttz(mm).to(fx.Int32)), fx.Int32(0))
            p0 = fx.Int32(fx.rocdl.readlane(T.i32, lane_page[0], idx))
            p1 = fx.Int32(fx.rocdl.readlane(T.i32, lane_page[1], idx))
            pg = use0.select(p0, p1)
            m0n = use0.select(m0 & (m0 - fx.Int64(1)), m0)
            m1n = use0.select(m1, m1 & (m1 - fx.Int64(1)))
            return (pg >= 0).select(pg, fx.Int32(0)), [m0n, m1n]

        first_page = first_page1 if NSET == 1 else first_page2

        def any_left(masks):
            m = masks[0]
            for s in range_constexpr(1, NSET):
                m = m | masks[s]
            return m

        # ---- async page DMA: this KV head's K data | K scales | V data | V scales -> a slot
        side_b = i32_hkv * (DATA_B + SCALE_B)
        reg_off = [
            kvh * DATA_B,
            i32_hkv * DATA_B + kvh * SCALE_B,
            side_b + kvh * DATA_B,
            side_b + i32_hkv * DATA_B + kvh * SCALE_B,
        ]
        slot_off = [0, DATA_B, DATA_B + SCALE_B, 2 * DATA_B + SCALE_B]
        lane16 = lane * fx.Int32(16)
        lane4 = lane * fx.Int32(4)
        aux0 = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0)
        u8p = fx.PointerType.get(
            fx.Uint8.ir_type, address_space=fx.AddressSpace.Global, alignment=16
        )

        def lds_byte_ptr(byte_off):
            return fx.to_llvm_ptr(
                fx.add_offset(fx.recast_iter(fx.Uint8, sp), fx.make_int_tuple(byte_off))
            )

        def dma_page(page, slot, valid):
            addr = fx.Int64(arg_cache) + fx.Int64(page) * fx.Int64(i32_page_b)
            nbytes = valid.select(i32_page_b, fx.Int32(0))
            rsrc = fx.rocdl.get_buffer_rsrc(
                fx.rocdl.make_buffer_ptr(fx.inttoptr(u8p, addr), num_records_bytes=nbytes)
            )
            base = slot * fx.Int32(SLOT_B)
            # per-lane offset in the VGPR operand, the wave-uniform rest in soffset
            for r_ in (0, 2):  # 8192 B: 2 passes x 4 waves x 64 lanes x 16 B
                for h in range_constexpr(2):
                    off = (fx.Int32(h * 4) + w) * fx.Int32(1024)
                    fx.rocdl.raw_ptr_buffer_load_async_lds(
                        rsrc,
                        lds_byte_ptr(base + fx.Int32(slot_off[r_]) + off),
                        fx.Int32(16).ir_value(),
                        lane16.ir_value(),
                        fx.Int32(fx.rocdl.readfirstlane(T.i32, reg_off[r_] + off)).ir_value(),
                        fx.Int32(0).ir_value(),
                        aux=aux0,
                    )
            for r_ in (1, 3):  # 1024 B: 4 waves x 64 lanes x 4 B
                off = w * fx.Int32(256)
                fx.rocdl.raw_ptr_buffer_load_async_lds(
                    rsrc,
                    lds_byte_ptr(base + fx.Int32(slot_off[r_]) + off),
                    fx.Int32(4).ir_value(),
                    lane4.ir_value(),
                    fx.Int32(fx.rocdl.readfirstlane(T.i32, reg_off[r_] + off)).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=aux0,
                )
            fx.rocdl.asyncmark()

        # this lane's 19 dwords of a slot (aiter's page_loader map, slot-relative dwords)
        k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
        k_ld = [(r * (D // 2) + g * 16) // 4 for r in k_rows]
        ks_ld = (DATA_B + ((w * 4 + g) * 16 + n) * 4) // 4
        v_ld = (DATA_B + SCALE_B + ((w * 4 + g) * BLK + 8 * n) * 4) // 4
        vs_ld = (2 * DATA_B + SCALE_B + (n // 2) * BLK + w * KPW + 8 * g) // 4

        def read_slot(slot):
            base = slot * fx.Int32(SLOT_B // 4)
            raw = []
            for st in range_constexpr(2):
                v = ld16((base + k_ld[st]) // 4)
                raw += [v[i] for i in range_constexpr(4)]
            raw.append(fx.Int32(sp[base + ks_ld]))
            for h in range_constexpr(2):
                v = ld16((base + v_ld) // 4 + h)
                raw += [v[i] for i in range_constexpr(4)]
            raw += [fx.Int32(sp[base + vs_ld]), fx.Int32(sp[base + vs_ld + 1])]
            return raw

        def dequant_k(raw):
            kf = []
            for st in range_constexpr(2):
                sc = _e4m3x2_f16(raw[8], st == 1)  # groups 2 g, 2 g + 1 of key row st
                row = []
                for ks in range_constexpr(NKS):
                    sv = Vec.from_elements([sc[ks // 2], sc[ks // 2]], fx.Float16)
                    row.append(
                        Vec.from_elements(
                            [
                                (_fp4x2_f16(raw[st * 4 + ks], i, 1.0) * sv).bitcast(fx.Int32)[0]
                                for i in range_constexpr(4)
                            ],
                            fx.Int32,
                        )
                    )
                kf.append(row)
            return kf

        def dequant_v(raw):
            vs = []
            for h in range_constexpr(2):
                vs += [_v_e4m3x2_f16(raw[17 + h], False), _v_e4m3x2_f16(raw[17 + h], True)]
            return [
                Vec.from_elements(
                    [
                        (_fp4x2_f16(raw[9 + dt], i, 1.0) * vs[i]).bitcast(fx.Int32)[0]
                        for i in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                for dt in range_constexpr(NDT)
            ]

        zero4 = Vec.filled(4, 0.0, fx.Float32)

        def qk(t, kf):
            qf = [ld16(Q_T0 + (t * HG + n) * Q_ROW + g * 4 + ks) for ks in range_constexpr(NKS)]
            a = []
            for st in range_constexpr(2):
                acc = zero4
                for ks in range_constexpr(NKS):
                    acc = _mfma(kf[st][ks], qf[ks], acc)
                a += [acc[e] for e in range_constexpr(4)]
            return a

        # ---- prime the ring: pages 0 .. depth - 1
        rem_dma = list(umask)
        for d in range_constexpr(depth):
            pg, nxt = first_page(rem_dma)
            dma_page(pg, fx.Int32(d), any_left(rem_dma) != fx.Int64(0))
            rem_dma = nxt

        m_init = fx.Float32(M_FLOOR) if lazy else neg_inf
        # lmfma: A operand of the row-sum MFMA, row 0 all ones (lanes n == 0), the rest zero
        one2 = fx.Int32(0x3C003C00)
        ones_a = Vec.from_elements(
            [(n == fx.Int32(0)).select(one2, fx.Int32(0)) for _ in range_constexpr(4)], fx.Int32
        )
        init = list(umask) + list(rem_dma) + [fx.Int32(0)]
        init += [m_init for _ in range_constexpr(TQ)]
        init += [zero4 if lmfma else fx.Float32(0.0) for _ in range_constexpr(TQ)]
        init += [zero4 for _ in range_constexpr(TQ * NDT)]
        i_u = 2 * NSET
        i_m = i_u + 1
        i_l = i_m + TQ
        n_fix = i_l + TQ
        results = init
        rem_test = any_left(umask)
        while rem_test != fx.Int64(0):
            st_ = results
            rem = [fx.Int64(st_[s]) for s in range_constexpr(NSET)]
            rdma = [fx.Int64(st_[NSET + s]) for s in range_constexpr(NSET)]
            u = fx.Int32(st_[i_u])
            m = [fx.Float32(st_[i_m + t]) for t in range_constexpr(TQ)]
            l_run = [
                Vec(st_[i_l + t]) if lmfma else fx.Float32(st_[i_l + t])
                for t in range_constexpr(TQ)
            ]
            o = [
                [Vec(st_[n_fix + t * NDT + dt]) for dt in range_constexpr(NDT)]
                for t in range_constexpr(TQ)
            ]
            page, rem = first_page(rem)
            # page u has landed (depth - 1 younger groups may still be in flight); the barrier
            # also orders Q^T staging before the first read and frees the slot of page u - 1
            fx.rocdl.wait_asyncmark(depth - 1)
            fx.gpu.barrier()
            pg_d, rdma_n = first_page(rdma)
            dma_page(pg_d, (u + fx.Int32(depth)) % fx.Int32(RING), any_left(rdma) != fx.Int64(0))
            rdma = rdma_n
            raw = read_slot(u % fx.Int32(RING))
            hits = [fx.Int64(fx.rocdl.ballot(T.i64, lane_page[s] == page)) for s in range_constexpr(NSET)]
            hit = [
                ((hits[t // 4] >> fx.Int64(TOPK * (t % 4))) & fx.Int64(0xFFFF)) != fx.Int64(0)
                for t in range_constexpr(TQ)
            ]
            own = [hit[t] & (page == diag[t]) for t in range_constexpr(TQ)]
            kf = dequant_k(raw)
            vf = dequant_v(raw)

            # software pipeline over the rows: S of row t + 1 issues before row t's softmax / P V
            s_cur = [fx.Float32(0.0) for _ in range_constexpr(8)]
            if const_expr(pipe):
                if hit[0]:
                    s_cur = qk(0, kf)
            for t in range_constexpr(TQ):
                s_nxt = [fx.Float32(0.0) for _ in range_constexpr(8)]
                if const_expr(pipe and t + 1 < TQ):
                    if hit[t + 1]:
                        s_nxt = qk(t + 1, kf)
                m_t, l_t, o_t = m[t], l_run[t], o[t]
                if hit[t]:
                    a = s_cur
                    if const_expr(not pipe):
                        a = qk(t, kf)
                    if own[t]:
                        # key 8 g + i of this wave is visible iff i <= lim - 8 g
                        lim_g = lim[t] - g * fx.Int32(8)
                        a = [
                            (fx.Int32(i) <= lim_g).select(a[i], neg_inf)
                            for i in range_constexpr(8)
                        ]
                    mx = a[0]
                    for i in range_constexpr(1, 8):
                        mx = _maxf(mx, a[i])
                    if const_expr(lazy):
                        m_new = m_t
                        if fx.Int64(
                            fx.rocdl.ballot(T.i64, mx * c > m_t + fx.Float32(LAZY_LOG2))
                        ) != fx.Int64(0):
                            mc = _xlane4(mx, _maxf) * c
                            m_new = (mc > m_t + fx.Float32(LAZY_LOG2)).select(mc, m_t)
                            alpha = _exp2(m_t - m_new)
                            av = Vec.filled(4, alpha, fx.Float32)
                            o_t = [x * av for x in o_t]
                            if const_expr(lmfma):
                                l_t = l_t * av
                            else:
                                l_t = l_t * alpha
                        if const_expr(pk):
                            cv = Vec.filled(2, c, fx.Float32)
                            mv = Vec.filled(2, -m_new, fx.Float32)
                            pv = [
                                fx.math.fma(Vec.from_elements([a[2 * j], a[2 * j + 1]], fx.Float32), cv, mv)
                                for j in range_constexpr(4)
                            ]
                            p = [_exp2(pv[j // 2][j % 2]) for j in range_constexpr(8)]
                        else:
                            p = [_exp2(fx.math.fma(x, c, -m_new)) for x in a]
                        if const_expr(not lmfma):
                            if const_expr(pk):
                                s2 = [Vec.from_elements([p[2 * j], p[2 * j + 1]], fx.Float32)
                                      for j in range_constexpr(4)]
                                s2 = (s2[0] + s2[1]) + (s2[2] + s2[3])
                                ps = s2[0] + s2[1]
                            else:
                                ps = p[0]
                                for i in range_constexpr(1, 8):
                                    ps = ps + p[i]
                            l_t = l_t + ps
                    else:
                        m_new = _maxf(m_t, _xlane4(mx, _maxf) * c)
                        m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
                        alpha = _exp2(m_t - m_use)
                        p = [_exp2(fx.math.fma(x, c, -m_use)) for x in a]
                        ps = p[0]
                        for i in range_constexpr(1, 8):
                            ps = ps + p[i]
                        l_t = l_t * alpha + ps
                        if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                            av = Vec.filled(4, alpha, fx.Float32)
                            o_t = [x * av for x in o_t]
                    pf = Vec.from_elements(
                        [_pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)], fx.Int32
                    )
                    o_t = [_mfma(vf[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)]
                    if const_expr(lmfma):
                        l_t = _mfma(ones_a, pf, l_t)  # row 0 of the tile: the head's sum of P
                    m_t = m_new
                m[t], l_run[t], o[t] = m_t, l_t, o_t
                s_cur = s_nxt
            state = rem + rdma + [u + fx.Int32(1)] + m + l_run + [x for ot in o for x in ot]
            # (with lmfma each l_run entry is the row-sum tile; flattened by as_ir_value below)
            results = [fx.as_ir_value(x) for x in state]
            rem_test = any_left(rem)

        # ---- combine the four waves' partials (amd-kernels' dimension-split combine); wave w
        # finalizes rows w + 4 s
        fx.rocdl.wait_asyncmark(0)
        fx.gpu.barrier()  # every wave is past its last Q^T / slot read
        for t in range_constexpr(TQ):
            sp[ML_W + ((w * TQ + t) * 2) * 64 + lane] = _bits(results[i_m + t])
            if const_expr(lmfma):
                # the tile's row 0 (lanes g == 0, element 0) holds the wave's sum over its keys
                l_w = (g == fx.Int32(0)).select(Vec(results[i_l + t])[0], fx.Float32(0.0))
            else:
                l_w = results[i_l + t]
            sp[ML_W + ((w * TQ + t) * 2 + 1) * 64 + lane] = _bits(l_w)
        fx.gpu.barrier()
        fin = []
        for s in range_constexpr(NSET):
            t = w + fx.Int32(4 * s)
            ms = [_as_f32(sp[ML_W + ((ww * TQ + t) * 2) * 64 + lane]) for ww in range_constexpr(NW)]
            big = ms[0]
            for ww in range_constexpr(1, NW):
                big = _maxf(big, ms[ww])
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_tot = l_tot + _as_f32(sp[ML_W + ((ww * TQ + t) * 2 + 1) * 64 + lane]) * wt[ww]
            l_tot = _xlane4(l_tot, lambda a, b: a + b)
            inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
            fin.append((t, wt, inv))
        acc = [[zero4 for _ in range_constexpr(NDT)] for _ in range_constexpr(NSET)]
        for rnd in range_constexpr(2):
            for tt in range_constexpr(TQ):
                for dd in range_constexpr(NDT // 2):
                    dt = rnd * (NDT // 2) + dd
                    st16(
                        ((w * TQ + tt) * (NDT // 2) + dd) * 64 + lane,
                        Vec(results[n_fix + tt * NDT + dt]).bitcast(fx.Int32),
                    )
            fx.gpu.barrier()
            for s in range_constexpr(NSET):
                t, wt, _ = fin[s]
                for ww in range_constexpr(NW):
                    wv = Vec.filled(4, wt[ww], fx.Float32)
                    for dd in range_constexpr(NDT // 2):
                        dt = rnd * (NDT // 2) + dd
                        v = ld16(((ww * TQ + t) * (NDT // 2) + dd) * 64 + lane)
                        acc[s][dt] = acc[s][dt] + v.bitcast(fx.Float32) * wv
            if const_expr(rnd == 0):
                fx.gpu.barrier()
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2)
        )
        for s in range_constexpr(NSET):
            t, _, inv = fin[s]
            tok = bidx * TQ + t
            o_row = (tok * i32_H + kvh * HG + n) * (D // 2) + g * 16
            for e in range_constexpr(4):
                val = Vec.from_elements(
                    [
                        _pk_bf16(acc[s][2 * j][e] * inv, acc[s][2 * j + 1][e] * inv)
                        for j in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                buffer_ops.buffer_store(val, outr, o_row + e * 4, mask=tok < i32_T)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_req: fx.Int64,
        arg_kv: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_topk_rs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q, arg_cache, arg_topk, arg_bt, arg_req, arg_kv, arg_out, arg_ks, arg_vs,
            i32_T, i32_H, i32_hkv, i32_topk_hs, i32_topk_rs, i32_bt_s, i32_page_b, f32_c,
            value_attrs={"rocdl.waves_per_eu": wpe, "rocdl.flat_work_group_size": "256,256"},
        ).launch(
            grid=((i32_T + TQ - 1) // TQ, i32_hkv, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch


@functools.cache
def _kernel(depth, lazy, pipe, wpe, tq, device_index, pk=False, lmfma=False):
    return compile_sparse_pa_prefill(depth, lazy, pipe, wpe, tq, pk, lmfma)


@functools.cache
def _unit(device_index):
    return torch.ones(1, dtype=torch.float32, device=f"cuda:{device_index}")


def _run(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def sparse_pa_prefill(
    q: torch.Tensor,  # [rows, Hq, 128] bf16
    kv_cache: torch.Tensor,  # uint8 pages [pages, >= 2 * Hkv * 9216]
    topk_idx: torch.Tensor,  # [Hkv, >= rows, 16] int32
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    row_req_id: torch.Tensor,  # [rows] int32
    row_kv_lens: torch.Tensor,  # [rows] int32, position + 1
    sm_scale: float,
    out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    *,
    depth: int = 2,
    lazy: bool = False,
    pipe: bool = True,
    wpe: int = 2,
    tq: int = 4,
    pk: bool = False,
    lmfma: bool = False,
) -> torch.Tensor:
    """``aiter.ops.flydsl.sparse_pa_nvfp4_prefill`` with the same contract (gfx950, 16 query
    heads per KV head, distinct block ids per row, distinct pages per request)."""
    import numpy as np

    rows, num_heads, head_dim = q.shape
    hkv = topk_idx.shape[0]
    assert head_dim == D and num_heads == hkv * HG, (q.shape, topk_idx.shape)
    assert q.dtype == torch.bfloat16 and q.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(1) == 1
    assert kv_cache.stride(0) % 16 == 0 and kv_cache.data_ptr() % 16 == 0
    assert topk_idx.dtype == torch.int32 and topk_idx.stride(2) == 1
    assert topk_idx.shape[2] == TOPK
    if out is None:
        out = torch.empty_like(q)
    if rows == 0:
        return out
    dev = q.device.index
    ks = _unit(dev) if k_scale is None else k_scale
    vs = _unit(dev) if v_scale is None else v_scale
    c = float(np.float32(sm_scale) * np.float32(LOG2E))
    with torch.cuda.device(q.device):
        _run(
            _kernel(depth, lazy, pipe, wpe, tq, dev, pk, lmfma),
            q.data_ptr(), kv_cache.data_ptr(), topk_idx.data_ptr(), block_table.data_ptr(),
            row_req_id.data_ptr(), row_kv_lens.data_ptr(), out.data_ptr(), ks.data_ptr(),
            vs.data_ptr(), rows, num_heads, hkv, topk_idx.stride(0), topk_idx.stride(1),
            block_table.stride(0), kv_cache.stride(0), c,
            torch._C._cuda_getCurrentRawStream(q.device.index),
        )
    return out
