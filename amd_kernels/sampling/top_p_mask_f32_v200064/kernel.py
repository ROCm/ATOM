# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""vLLM's split top-p pipeline (stats, five pivot rounds, mask) in one launch, bit for bit (``reference.py``).

One 512-thread workgroup per (row, slice), as Triton's programs, two per CU (<= 128 VGPRs); a row's S workgroups run
on one XCD (workgroup w on XCD w % 8) and meet at row barriers (a monotonic agent-scope counter; the last workgroup out
resets it), exchanging their partials through the workspace (plain stores and loads that meet in the XCD's L2). The
slice stays in registers: x in the 4-column load layout (thread t holds columns 2048 q + 4 t + j), the probabilities
in the round layout (2048 q + t + 512 r) after one LDS transpose. Each ``tl.sum`` is the reference's butterfly
through DPP / permlane swaps and the waves' LDS slots; the two fused products of Triton's build are explicit fmas and
its unfused ones go through an opaque move (``_launder``). A round skips the pivots no lane of a wave is above.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .reference import F as NF
from .reference import ROUNDS, V

F, I, I64 = fx.Float32, fx.Int32, fx.Int64
NT = 512
INF = float("inf")
ROW_WORDS = 8192  # workspace words per row: stats [S][4] at 0, partials [ROUNDS][S][F][4] at 256
PARTS_AT = 256
CNT_WORDS = 32  # per row: barrier count, done count (one 128-byte line)
MAX_ROWS = 64
DBG_WORDS = 64  # timing stamps after the counters
SC1 = 0  # the partials meet in the XCD's L2: plain stores (L1 is write-through), loads of lines not read before


def _fbits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


LOG2E = _fbits(0x3FB8AA3B)
LN2_HI, LN2_LO = _fbits(0x3F317217), _fbits(0x3377D1CF)
EPS = _fbits(0x3089705F)  # fp32 1e-9
C_UNIFORM = _fbits(0x3F800008)  # fp32 1 + 1e-6
C_COLLAPSE = _fbits(0x3F800001)  # fp32 1 + 1e-7
TINY = _fbits(0x0DA24260)  # fp32 1e-30


def _bi(x):
    return x.bitcast(I)


def _bf(x):
    return x.bitcast(F)


def _dpp(src, ctrl, old=None, bank_mask=0xF, row_mask=0xF, bound=True):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    return I(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, row_mask, bank_mask, bound))


def _swap(op, v):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False)
    return I(_llvm.extractvalue(T.i32, res, [0])), I(_llvm.extractvalue(T.i32, res, [1]))


def _xor_lane(v, m, lane):
    """i32 value of lane ``lane ^ m`` (m a power of two < 64)."""
    if m == 1:
        return _dpp(v, 0xB1)  # quad_perm [1, 0, 3, 2]
    if m == 2:
        return _dpp(v, 0x4E)  # quad_perm [2, 3, 0, 1]
    if m == 4:  # row_shr:4 into banks 1, 3, then row_shl:4 into banks 0, 2
        return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return _dpp(v, 0x128)  # row_ror:8
    a, b = _swap(_rocdl_ops.permlane16_swap if m == 16 else _rocdl_ops.permlane32_swap, v)
    return ((lane & m) != 0).select(a, b)


def _xsum(v, offsets, lane):
    """fp32 xor butterfly ``v = v + v[lane ^ o]`` in order; every lane ends with the sum."""
    for o in offsets:
        v = v + _bf(_xor_lane(_bi(v), o, lane))
    return v


def _xred(v, op, lane, width=64):
    """Exact (order-free) wave reduction (min / max / integer-valued sums) within aligned groups of ``width`` lanes."""
    o = 1
    while o < width:
        v = op(v, _bf(_xor_lane(_bi(v), o, lane)) if v.dtype == F else _xor_lane(v, o, lane))
        o *= 2
    return v


def _fmin(a, b):
    """minnum (v_min_f32), as Triton's tl.minimum / tl.min."""
    return F(_arith.minnumf(fx.as_ir_value(F(a)), fx.as_ir_value(F(b))))


def _fmax(a, b):
    """maxnum (v_max_f32), as Triton's tl.maximum / tl.max."""
    return F(_arith.maxnumf(fx.as_ir_value(F(a)), fx.as_ir_value(F(b))))


def _launder(v):
    """``v`` through an identity DPP move: the compiler cannot fuse the product behind it into a later add."""
    return _bf(_dpp(_bi(v), 0xE4))


def _uni(v):
    """Wave-uniform copy (readfirstlane); also an opaque value for fma fusion."""
    if v.dtype == F:
        return _bf(I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(_bi(v)))))
    return I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(v)))


def _lane_val(v, k):
    """``v`` of lane k (uniform)."""
    if v.dtype == F:
        return _bf(I(fx.rocdl.readlane(T.i32, fx.as_ir_value(_bi(v)), fx.as_ir_value(I(k)))))
    return I(fx.rocdl.readlane(T.i32, fx.as_ir_value(v), fx.as_ir_value(I(k))))


def _exp(d):
    """Triton's fp32 ``tl.exp`` (llvm.exp2.f32 of d * log2e) on gfx950."""
    t = d * F(LOG2E)
    small = t < F(-126.0)
    r = F(_rocdl_ops.exp2(T.f32, fx.as_ir_value(fx.math.fma(d, F(LOG2E), small.select(F(64.0), F(0.0))))))
    return small.select(r * F(_fbits(0x1F800000)), r)  # 2^-64


def _log(x):
    """Triton's fp32 ``tl.log`` for normal x > 0 (llvm.log.f32 expansion with fp contraction)."""
    y = fx.math.log2(x, fastmath=fx.FastMathFlags.afn)
    r0 = _launder(y * F(LN2_HI))
    f0 = fx.math.fma(y, F(LN2_HI), r0 * F(-1.0))
    f1 = fx.math.fma(y, F(LN2_LO), f0)
    return fx.math.fma(y, F(LN2_HI), f1)


def _gptr_i32(addr):
    return fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4), addr)


def _atomic(addr, op, value):
    return I(_llvm.AtomicRMWOp(op, _gptr_i32(addr).llvm_ptr, I(value).ir_value(), _llvm.AtomicOrdering.monotonic,
                               syncscope="agent", alignment=4).result)


def _scan63(v, op):
    """Triton's wave reduction (row_shr 8, 4, 2, 1, row_bcast 15, row_bcast 31; the value of lane 63), uniform."""
    for ctrl in (0x118, 0x114, 0x112, 0x111):
        v = op(v, _bf(_dpp(_bi(v), ctrl)))
    v = op(v, _bf(_dpp(_bi(v), 0x142, old=_bi(v), row_mask=0xA)))
    v = op(v, _bf(_dpp(_bi(v), 0x143)))
    return _lane_val(v, 63)


def _wsum(v):
    return _scan63(v, lambda a, b: a + b)


def _wmin(v):
    return _scan63(v, _fmin)


def _wmax(v):
    return _scan63(v, _fmax)


def _popc(pred):
    """Lanes of the wave where ``pred`` holds (uniform)."""
    return I(fx.math.ctpop(I64(fx.rocdl.ballot(T.i64, pred))).to(I))


def _wtree(w):
    """The 8 waves' values summed as Triton's cross-wave butterfly (4, 2, 1)."""
    return ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]))


def _ladder(lo, hi):
    """The 8 pivots ``lo * exp(log(hi / lo) * (f + 1) / 9)`` (uniform scalars)."""
    r = _log(hi / lo)
    return [lo * _exp((r * F(float(f + 1))) / F(9.0)) for f in range(NF)]


def _now():
    return I64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


# state words in LDS (per Z group g at g * NSTATE): the search state _topp_sb_combine carries, then the barrier count
FIELDS = ("lo", "hi", "best", "best_mass", "best_minl", "best_nmin", "best_fin", "done", "nomask", "pivot", "dup",
          "numdup", "numkeep", "fin")
NSTATE = len(FIELDS)


@launcher
def compile_top_p_mask(S: int, timing: bool = False):
    """``timing``: workgroup 0 stores s_memrealtime stamps (100 MHz) at the phase ends (debug)."""
    assert S in (16, 32)
    L = V // S
    NQ = -(-L // 2048)
    NCH = -(-L // 8192)
    NW = NQ * NF * 8  # per-wave slots of a round: [q][f][wave]
    name = f"top_p_mask_f32_v200064_s{S}" + ("_timing" if timing else "")
    COMB = (4, 2, 1, 8) if S == 16 else (4, 2, 1, 16, 8)
    PPW = 64 // S  # combine: pivots per wave pass (lane = f_sub * S + s)

    @fx.struct
    class Shared:
        tr: fx.Array[fx.Int32, NQ * 2048]
        red: fx.Array[fx.Int32, 8 * 4]
        wm: fx.Array[fx.Int32, NW]
        wn: fx.Array[fx.Int32, NW]
        wc: fx.Array[fx.Int32, NW]
        cm: fx.Array[fx.Int32, NQ * NF]
        cn: fx.Array[fx.Int32, NQ * NF]
        rk: fx.Array[fx.Int32, NQ * 8]
        st: fx.Array[fx.Int32, 2 * NSTATE + 1]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # fp32 [n, >= V] logits, masked in place
        i32_sx: fx.Int32,  # row stride (elements)
        arg_p: fx.Int64,  # fp32 [n] top-p
        arg_ws: fx.Int64,  # workspace (host._workspace), zeroed once
        i32_n: fx.Int32,
    ):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        wg = fx.block_idx.x
        row = wg % 8 + (wg // 8 // S) * 8
        sl = wg // 8 % S
        row_ok = row < i32_n
        row_c = row_ok.select(row, I(0))
        pres = buffer_ops.create_buffer_resource_from_addr(arg_p + I64(row_c) * 4, num_records_bytes=4)
        p = _uni(row_ok.select(I(buffer_ops.buffer_load(pres, 0, vec_width=1, dtype=I)).bitcast(F), F(1.0)))
        wsr = arg_ws + I64(row_c) * (ROW_WORDS * 4)
        cnt_addr = arg_ws + MAX_ROWS * ROW_WORDS * 4 + I64(row_c) * (CNT_WORDS * 4)
        ws = buffer_ops.create_buffer_resource_from_addr(wsr, num_records_bytes=ROW_WORDS * 4)
        is_t0 = tx == 0
        upper = (lane & 8) != 0  # Z group of this lane in the 4-column and reduction layouts
        g5 = (lane & 32) != 0  # Z group of the column this thread holds in the round layout (column bit 5)
        dbg = buffer_ops.create_buffer_resource_from_addr(
            arg_ws + MAX_ROWS * (ROW_WORDS + CNT_WORDS) * 4,
            num_records_bytes=(is_t0 & (wg == 0)).select(I(DBG_WORDS * 4), I(0)))

        def stamp(k):
            if const_expr(timing):
                t_ = _now()
                buffer_ops.buffer_store(Vec.from_elements([t_.to(I), (t_ >> I64(32)).to(I)], I), dbg, 2 * k)

        def read_state():
            out = {}
            for gi, g in ((0, "a"), (1, "b")):
                out[g] = {k: _bf(I(smem.st.ptr[gi * NSTATE + i])) for i, k in enumerate(FIELDS)}
            return out, I(smem.st.ptr[2 * NSTATE])

        if p < F(1.0):
            stamp(0)
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x + (I64(row_c) * I64(i32_sx) + I64(sl * L)) * 4, num_records_bytes=L * 4)
            x = []
            for q in range_constexpr(NQ):
                v = Vec(buffer_ops.buffer_load(xr, q * 2048 + tx * 4, vec_width=4, dtype=I))
                for j in range_constexpr(4):
                    x.append(((tx * 4 + (q * 2048 + j)) < L).select(I(v[j]).bitcast(F), F(-INF)))

            # ---- 1. slice stats over 8192-column chunks (the 4-column load layout is Triton's stats layout)
            m = F(-INF)
            es = F(0.0)
            mn = F(INF)
            cnt = I(0)
            for k in range_constexpr(NCH):
                regs = [(rp, j) for rp in range_constexpr(4) for j in range_constexpr(4) if 4 * k + rp < NQ]
                cm = F(-INF)
                for rp, j in regs:
                    cm = _fmax(cm, x[(4 * k + rp) * 4 + j])
                cm = _wmax(cm)
                if lane == 0:
                    smem.red.ptr[wave] = _bi(cm)
                fx.gpu.barrier()
                for w in range_constexpr(8):
                    cm = _fmax(cm, _bf(I(smem.red.ptr[w])))
                nm = _uni(_fmax(m, cm))
                rescale = (m == F(-INF)).select(F(0.0), _exp(m - nm))
                e = {}
                for rp, j in regs:
                    xv = x[(4 * k + rp) * 4 + j]
                    e[(rp, j)] = (xv > F(-INF)).select(_exp(xv - nm), F(0.0))

                def ev(rp, j):
                    return e.get((rp, j), F(0.0))

                def half(j0):
                    return (ev(0, j0) + ev(0, j0 + 2) + (ev(1, j0) + ev(1, j0 + 2))) + (
                        ev(2, j0) + ev(2, j0 + 2) + (ev(3, j0) + ev(3, j0 + 2)))

                part = _wsum(half(0) + half(1))
                tm = F(INF)
                tc = I(0)
                for rp, j in regs:
                    xv = x[(4 * k + rp) * 4 + j]
                    tm = _fmin(tm, (xv > F(-INF)).select(xv, F(INF)))
                    tc = tc + _popc(xv > F(-INF))
                tm = _wmin(tm)
                if lane == 0:
                    smem.red.ptr[8 + wave] = _bi(part)
                    smem.red.ptr[16 + wave] = _bi(tm)
                    smem.red.ptr[24 + wave] = tc
                fx.gpu.barrier()
                csum = _wtree([_bf(I(smem.red.ptr[8 + w])) for w in range_constexpr(8)])
                for w in range_constexpr(8):
                    tm = _fmin(tm, _bf(I(smem.red.ptr[16 + w])))
                tcs = I(0)
                for w in range_constexpr(8):
                    tcs = tcs + I(smem.red.ptr[24 + w])
                es = _uni(es * rescale) + _uni(csum)
                m = nm
                mn = _fmin(mn, _uni(tm))
                cnt = cnt + _uni(tcs)
            if is_t0:
                buffer_ops.buffer_store(
                    Vec.from_elements([_bi(m), _bi(es), _bi(mn), _bi(cnt.to(F))], I), ws, sl * 4,
                    cache_modifier=SC1)
            stamp(1)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if is_t0:
                _atomic(cnt_addr, _llvm.AtomicBinOp.add, 1)
                seen = _atomic(cnt_addr, _llvm.AtomicBinOp.add, 0)
                while seen < S:
                    seen = _atomic(cnt_addr, _llvm.AtomicBinOp.add, 0)
            fx.gpu.barrier()
            stamp(2)

            # ---- 2. row stats: lane l holds slice l % S (every wave)
            sv = Vec(buffer_ops.buffer_load(ws, (lane % S) * 4, vec_width=4, dtype=I, cache_modifier=SC1))
            m_s, es_s, mn_s, c_s = (I(sv[0]).bitcast(F), I(sv[1]).bitcast(F), I(sv[2]).bitcast(F),
                                    I(sv[3]).bitcast(F))
            M = _uni(_xred(m_s, _fmax, lane))
            w_s = (m_s == F(-INF)).select(F(0.0), _exp(m_s - M))
            prod = es_s * w_s
            if const_expr(S == 32):
                zl = _xsum(_launder(prod), (16, 8, 4, 2, 1), lane)
                z_a = _lane_val(zl, 0)
                z_b = z_a
            else:
                zl = fx.math.fma(es_s, w_s, _bf(_xor_lane(_bi(prod), 8, lane)))
                zl = _xsum(zl, (4, 2, 1), lane)
                z_a = _lane_val(zl, 0)
                z_b = _lane_val(zl, 8)
            mnr = _uni(_xred(mn_s, _fmin, lane))
            cr = _uni(_xred((lane < S).select(c_s, F(0.0)), lambda a, b: a + b, lane))
            hi_a, hi_b = F(1.0) / z_a, F(1.0) / z_b
            emn = _exp(mnr - M)
            lo_a = _fmax(emn / z_a, F(TINY))
            lo_b = _fmax(emn / z_b, F(TINY))
            run_a = ~((z_a <= F(0.0)) | (cr < F(1.0))) & ~(hi_a <= lo_a * F(C_UNIFORM))
            run_b = ~((z_b <= F(0.0)) | (cr < F(1.0))) & ~(hi_b <= lo_b * F(C_UNIFORM))
            z_t = upper.select(z_b, z_a)

            # ---- 3. probabilities (4-column layout, this lane's Z) -> round layout through LDS; the initial state
            for q in range_constexpr(NQ):
                for j in range_constexpr(4):
                    smem.tr.ptr[q * 2048 + tx * 4 + j] = _bi(_exp(x[q * 4 + j] - M) / z_t)
            if is_t0:
                for gi, lo0, hi0 in ((0, lo_a, hi_a), (1, lo_b, hi_b)):
                    init = dict(lo=lo0, hi=hi0, best=F(-1.0), best_mass=F(0.0), best_minl=F(1.0), best_nmin=F(0.0),
                                best_fin=F(-1.0), done=F(0.0), nomask=F(0.0), pivot=F(0.0), dup=F(1.0),
                                numdup=F(1.0), numkeep=F(1.0), fin=F(-1.0))
                    for i, k in enumerate(FIELDS):
                        smem.st.ptr[gi * NSTATE + i] = _bi(init[k])
                smem.st.ptr[2 * NSTATE] = I(1)
            fx.gpu.barrier()
            pr = [I(smem.tr.ptr[q * 2048 + tx + 512 * r]).bitcast(F) for q in range_constexpr(NQ)
                  for r in range_constexpr(4)]
            stamp(3)

            # ---- 4. rounds (state per Z group in LDS)
            for jr in range_constexpr(ROUNDS):
                stv, nbar = read_state()
                live_a = run_a & (stv["a"]["done"] == F(0.0))
                live_b = run_b & (stv["b"]["done"] == F(0.0))
                if live_a | live_b:
                    pivs = {g: _ladder(stv[g]["lo"], stv[g]["hi"]) for g in ("a", "b")}
                    pv = [upper.select(pivs["b"][f], pivs["a"][f]) for f in range_constexpr(NF)]
                    for i in range_constexpr(-(-NW // NT)):
                        if tx + i * NT < NW:
                            smem.wm.ptr[tx + i * NT] = I(0)
                            smem.wn.ptr[tx + i * NT] = _bi(F(INF))
                            smem.wc.ptr[tx + i * NT] = I(0)
                    fx.gpu.barrier()
                    # per (chunk, pivot): this wave's mass and minimum above the pivot, skipped when no lane has any
                    for q in range_constexpr(NQ):
                        prq = [pr[q * 4 + r] for r in range_constexpr(4)]
                        pmax = _fmax(_fmax(prq[0], prq[1]), _fmax(prq[2], prq[3]))
                        for f in range_constexpr(NF):
                            if _popc(pmax > pv[f]) > 0:
                                ab = [prq[r] > pv[f] for r in range_constexpr(4)]
                                lf = [ab[r].select(prq[r], F(0.0)) for r in range_constexpr(4)]
                                msum = _wsum((lf[0] + lf[2]) + (lf[1] + lf[3]))
                                mf = [ab[r].select(prq[r], F(INF)) for r in range_constexpr(4)]
                                mmin = _wmin(_fmin(_fmin(mf[0], mf[1]), _fmin(mf[2], mf[3])))
                                if lane == 0:
                                    smem.wm.ptr[(q * NF + f) * 8 + wave] = _bi(msum)
                                    smem.wn.ptr[(q * NF + f) * 8 + wave] = _bi(mmin)
                    fx.gpu.barrier()
                    if tx < NQ * NF:
                        smem.cm.ptr[tx] = _bi(_wtree([_bf(I(smem.wm.ptr[tx * 8 + w])) for w in range_constexpr(8)]))
                        tmn = F(INF)
                        for w in range_constexpr(8):
                            tmn = _fmin(tmn, _bf(I(smem.wn.ptr[tx * 8 + w])))
                        smem.cn.ptr[tx] = _bi(tmn)
                    fx.gpu.barrier()
                    # per (chunk, pivot): the count within 1e-9 of the chunk's minimum
                    for q in range_constexpr(NQ):
                        prq = [pr[q * 4 + r] for r in range_constexpr(4)]
                        for f in range_constexpr(NF):
                            tmn = _bf(I(smem.cn.ptr[q * NF + f]))
                            if tmn < F(INF):
                                c = I(0)
                                for r in range_constexpr(4):
                                    c = c + _popc((prq[r] > pv[f]) & (fx.math.absf(prq[r] - tmn) < F(EPS)))
                                if lane == 0:
                                    smem.wc.ptr[(q * NF + f) * 8 + wave] = c
                    fx.gpu.barrier()
                    if tx < NF:
                        mass = F(0.0)
                        minl = F(1.0)
                        nmin = F(0.0)
                        for q in range_constexpr(NQ):
                            tc = I(0)
                            for w in range_constexpr(8):
                                tc = tc + I(smem.wc.ptr[(q * NF + tx) * 8 + w])
                            tcf = tc.to(F)
                            tmn = _bf(I(smem.cn.ptr[q * NF + tx]))
                            mass = mass + _bf(I(smem.cm.ptr[q * NF + tx]))
                            is_new = tmn < minl
                            is_same = fx.math.absf(tmn - minl) < F(EPS)
                            nmin = is_new.select(tcf, nmin + is_same.select(tcf, F(0.0)))
                            minl = _fmin(minl, tmn)
                        buffer_ops.buffer_store(
                            Vec.from_elements([_bi(mass), _bi(minl), _bi(nmin), I(0)], I), ws,
                            PARTS_AT + ((jr * S + sl) * NF + tx) * 4, cache_modifier=SC1)
                    stamp(4 + 2 * jr)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    if is_t0:
                        _atomic(cnt_addr, _llvm.AtomicBinOp.add, 1)
                        seen = _atomic(cnt_addr, _llvm.AtomicBinOp.add, 0)
                        while seen < (nbar + 1) * S:
                            seen = _atomic(cnt_addr, _llvm.AtomicBinOp.add, 0)
                    fx.gpu.barrier()
                    stamp(5 + 2 * jr)
                    # combine (wave 0; lane = f_sub * S + s) -> the new state in LDS
                    if wave == 0:
                        cmass, cgmin, cnmin = [], [], []
                        for ps in range_constexpr(NF // PPW):
                            pvv = Vec(buffer_ops.buffer_load(
                                ws, PARTS_AT + ((jr * S + lane % S) * NF + lane // S + ps * PPW) * 4, vec_width=4,
                                dtype=I, cache_modifier=SC1))
                            ms_, mls, nms = I(pvv[0]).bitcast(F), I(pvv[1]).bitcast(F), I(pvv[2]).bitcast(F)
                            msum = _xsum(ms_, COMB, lane)
                            gmn = _xred(mls, _fmin, lane, S)
                            nsum = _xred((fx.math.absf(mls - gmn) < F(EPS)).select(nms, F(0.0)), lambda a, b: a + b,
                                         lane, S)
                            for fs in range_constexpr(PPW):
                                cmass.append(_lane_val(msum, fs * S))
                                cgmin.append(_lane_val(gmn, fs * S))
                                cnmin.append(_lane_val(nsum, fs * S))
                        ge = [cmass[f] >= p for f in range_constexpr(NF)]
                        okf = [ge[f] & (fx.math.fma(cgmin[f] * F(-1.0), cnmin[f], cmass[f]) < p)
                               for f in range_constexpr(NF)]
                        ok_idx = I(-1)
                        ge_idx = I(-1)
                        for f in range_constexpr(NF):
                            ok_idx = okf[f].select(I(f), ok_idx)
                            ge_idx = ge[f].select(I(f), ge_idx)

                        def pick(vals, idx, default):
                            out = default
                            for f in range_constexpr(NF):
                                out = (idx == f).select(vals[f], out)
                            return out

                        hit_any = ok_idx >= 0
                        new = {}
                        for g, lv, zz in (("a", live_a, z_a), ("b", live_b, z_b)):
                            s_ = dict(stv[g])
                            pg = pivs[g]
                            bp = pick(pg, ge_idx, F(0.0))
                            upd = lv & (ge_idx >= 0) & (bp > s_["best"])
                            s_["best"] = upd.select(bp, s_["best"])
                            s_["best_mass"] = upd.select(pick(cmass, ge_idx, F(0.0)), s_["best_mass"])
                            s_["best_minl"] = upd.select(pick(cgmin, ge_idx, F(0.0)), s_["best_minl"])
                            s_["best_nmin"] = upd.select(pick(cnmin, ge_idx, F(0.0)), s_["best_nmin"])
                            s_["best_fin"] = upd.select((ge_idx + jr * NF).to(F), s_["best_fin"])
                            hit = lv & hit_any
                            nlo = s_["lo"]
                            nhi = s_["hi"]
                            for f in range_constexpr(NF):
                                nlo = _fmax(nlo, ge[f].select(pg[f], F(-INF)))
                                nhi = _fmin(nhi, ge[f].select(F(INF), pg[f]))
                            if const_expr(jr == ROUNDS - 1):
                                stop = lv & ~hit_any
                            else:
                                stop = lv & ~hit_any & ~(nhi > nlo * F(C_COLLAPSE))
                            fall = stop & (s_["best"] > F(0.0))
                            pivot = hit.select(pick(pg, ok_idx, F(0.0)), fall.select(s_["best"], F(0.0)))
                            sel_mass = hit.select(pick(cmass, ok_idx, F(0.0)), fall.select(s_["best_mass"], F(0.0)))
                            dup = hit.select(pick(cgmin, ok_idx, F(1.0)), fall.select(s_["best_minl"], F(1.0)))
                            numdup = hit.select(pick(cnmin, ok_idx, F(1.0)), fall.select(s_["best_nmin"], F(1.0)))
                            chosen = hit | fall
                            nomask = chosen.select((pivot * zz >= F(1.0)).select(F(1.0), F(0.0)),
                                                   (stop & ~fall).select(F(1.0), F(0.0)))
                            qv = _fmax(sel_mass - p, F(0.0)) / dup
                            numkeep = chosen.select(_fmin(_fmax(numdup - qv.to(fx.Uint32).to(F), F(1.0)), numdup),
                                                    F(1.0))
                            adv = lv & ~hit
                            s_["lo"] = adv.select(nlo, s_["lo"])
                            s_["hi"] = adv.select(nhi, s_["hi"])
                            s_["fin"] = hit.select((ok_idx + jr * NF).to(F), s_["fin"])
                            s_["done"] = lv.select((hit | stop).select(F(1.0), F(0.0)), s_["done"])
                            for k_, v_ in (("nomask", nomask), ("pivot", pivot), ("dup", dup), ("numdup", numdup),
                                           ("numkeep", numkeep)):
                                s_[k_] = lv.select(v_, s_[k_])
                            new[g] = s_
                        if lane == 0:
                            for gi, g in ((0, "a"), (1, "b")):
                                for i, k in enumerate(FIELDS):
                                    smem.st.ptr[gi * NSTATE + i] = _bi(new[g][k])
                            smem.st.ptr[2 * NSTATE] = nbar + 1
                    fx.gpu.barrier()
            stamp(14)

            # ---- 5. mask
            stv, _ = read_state()
            masked_a = run_a & (stv["a"]["nomask"] == F(0.0))
            masked_b = run_b & (stv["b"]["nomask"] == F(0.0))
            ties_a = masked_a & (stv["a"]["numkeep"] < stv["a"]["numdup"])
            ties_b = masked_b & (stv["b"]["numkeep"] < stv["b"]["numdup"])
            # pivot below the max probability and no boundary duplicates: keep = prob > pivot (the maximum's
            # probability 1 / Z is above the pivot), on the round layout's probabilities; only -inf is stored
            simple_a = masked_a & ~ties_a & (stv["a"]["pivot"] < hi_a)
            simple_b = masked_b & ~ties_b & (stv["b"]["pivot"] < hi_b)
            if simple_a | simple_b:
                simple_t = g5.select(simple_b, simple_a)
                piv5 = g5.select(stv["b"]["pivot"], stv["a"]["pivot"])
                for q in range_constexpr(NQ):
                    for r in range_constexpr(4):
                        col = tx + (q * 2048 + 512 * r)
                        drop = simple_t & ~(pr[q * 4 + r] > piv5)
                        buffer_ops.buffer_store(_bi(F(-INF)), xr, drop.select(col, I(L)))
            # the rest (ties, or a pivot at the max probability): Triton's mask on the 4-column layout
            cplx_a = masked_a & ~simple_a
            cplx_b = masked_b & ~simple_b
            if cplx_a | cplx_b:
                cplx_t = upper.select(cplx_b, cplx_a)
                pivot_t = upper.select(stv["b"]["pivot"], stv["a"]["pivot"])
                dup_t = upper.select(stv["b"]["dup"], stv["a"]["dup"])
                numkeep_t = upper.select(stv["b"]["numkeep"], stv["a"]["numkeep"])
                ties_t = upper.select(ties_b, ties_a)
                fin_t = upper.select(stv["b"]["fin"], stv["a"]["fin"])
                fin_t = (fin_t < F(0.0)).select(upper.select(stv["b"]["best_fin"], stv["a"]["best_fin"]), fin_t)
                fr = (fin_t / F(8.0)).to(I)
                fsl = (fin_t - fr.to(F) * F(8.0)).to(I)
                prec = F(0.0)
                for s2 in range_constexpr(S):
                    pvv = Vec(buffer_ops.buffer_load(ws, PARTS_AT + ((fr * S + s2) * NF + fsl) * 4, vec_width=4,
                                                     dtype=I, cache_modifier=SC1))
                    dc = (fx.math.absf(I(pvv[1]).bitcast(F) - dup_t) < F(EPS)).select(I(pvv[2]).bitcast(F), F(0.0))
                    prec = prec + (sl > s2).select(dc, F(0.0))
                xc = []
                for q in range_constexpr(NQ):
                    v = Vec(buffer_ops.buffer_load(xr, q * 2048 + tx * 4, vec_width=4, dtype=I))
                    for j in range_constexpr(4):
                        xc.append(((tx * 4 + (q * 2048 + j)) < L).select(I(v[j]).bitcast(F), F(-INF)))
                prob = [_exp(xc[i] - M) / z_t for i in range_constexpr(NQ * 4)]
                dm = [ties_t & (prob[i] == dup_t) for i in range_constexpr(NQ * 4)]
                for q in range_constexpr(NQ):
                    cq = I(0)
                    for j in range_constexpr(4):
                        cq = cq + _popc(dm[q * 4 + j])
                    if lane == 0:
                        smem.rk.ptr[q * 8 + wave] = cq
                fx.gpu.barrier()
                chunk_base = I(0)
                for q in range_constexpr(NQ):
                    before = I(0)
                    tot = I(0)
                    for w in range_constexpr(8):
                        cw = I(smem.rk.ptr[q * 8 + w])
                        before = before + (wave > w).select(cw, I(0))
                        tot = tot + cw
                    lanes_before = I(0)
                    for j in range_constexpr(4):
                        bal = I64(fx.rocdl.ballot(T.i64, dm[q * 4 + j]))
                        mb = I(_llvm.call_intrinsic(T.i32, "llvm.amdgcn.mbcnt.lo",
                                                    [bal.to(I).ir_value(), I(0).ir_value()], [], []))
                        mb = I(_llvm.call_intrinsic(T.i32, "llvm.amdgcn.mbcnt.hi",
                                                    [(bal >> I64(32)).to(I).ir_value(), mb.ir_value()], [], []))
                        lanes_before = lanes_before + mb
                    own = I(0)
                    outv = []
                    for j in range_constexpr(4):
                        k_i = q * 4 + j
                        own = own + dm[k_i].select(I(1), I(0))
                        rank = (chunk_base + before + lanes_before + own).to(F) + prec
                        kd = ((prob[k_i] > pivot_t) | (xc[k_i] == M)) & (~dm[k_i] | (rank <= numkeep_t))
                        outv.append(kd.select(_bi(xc[k_i]), _bi(F(-INF))))
                    chunk_base = chunk_base + tot
                    buffer_ops.buffer_store(Vec.from_elements(outv, I), xr,
                                            cplx_t.select(q * 2048 + tx * 4, I(L)))
            stamp(15)

            # ---- the last workgroup of the row out resets the counters (every workgroup passed its last poll)
            if is_t0:
                t = _atomic(cnt_addr + 4, _llvm.AtomicBinOp.add, 1)
                if t == S - 1:
                    _atomic(cnt_addr, _llvm.AtomicBinOp.xchg, 0)
                    _atomic(cnt_addr + 4, _llvm.AtomicBinOp.xchg, 0)

    @flyc.jit
    def launch_top_p_mask(arg_x: fx.Int64, i32_sx: fx.Int32, arg_p: fx.Int64, arg_ws: fx.Int64, i32_n: fx.Int32,
                          i32_grid: fx.Int32, stream: fx.Stream):
        # two workgroups per CU (<= 128 VGPRs): a row's slices and the next rows' run together
        kernel(arg_x, i32_sx, arg_p, arg_ws, i32_n,
               value_attrs={"rocdl.waves_per_eu": 4, "rocdl.flat_work_group_size": f"{NT},{NT}"}).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch_top_p_mask
