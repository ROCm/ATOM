# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 all-reduce + residual add + Gemma RMSNorm, bf16 out, two-stage on registered graph inputs, whose all-gather moves
the reduced rows as a lossless 12-bit code.

``ar_add_rmsnorm_pull`` with block hand-offs and scoped FTZ (same protocol, register buffers, call counts, sum order and
norm map); only the reduced row a reducer publishes and the pullers read changes format. Lane t holds columns 8t .. 8t
+ 7 of the row (bf16 u, exponent field e); a group of 32 columns (4 lanes) shares E = max e. A value is (sign, d =
min(E - e, 15), 7 mantissa bits): d <= 14 reproduces e exactly (zeros and denormals of a group with E <= 14 too); d = 15
is a signed zero (mantissa field 0) or an exception, whose low 7 bits index the wave's exception list (raw bf16). A wave
with more than 6 exceptions or a lane with more than one is raw: its 1 KB goes to RED slot ``j + OVF_SLOT`` and its
count word says so. Row layout in RED slot j (bytes): plane A [0, 6144) = sign << 7 | mantissa (or index), lane t at
8t; plane B [6144, 9216) = the 8 nibbles d of a lane (value v at bit 4 (v // 2) + 16 (v % 2)), lane t at 6144 + 4t;
headers [9216, 9600) = per wave 32 B: 16 group bytes E, a count word (exceptions | SPECF if a zero or an exception, or
RAWF), 6 exception bf16. 9600 B = 0.781 of the bf16 row.

The puller's wave 0 loads the row's 384 header bytes next to its payload loads and shares them through LDS (one
barrier; per-wave header loads are slower: the links are request-bound). It decodes two values per dword (bf16
pairs); only a wave whose count word has SPECF runs the per-value selects. Every decoded value is the reducer's bf16 bit
for bit, so out / res_out equal ``ar_add_rmsnorm_pull``'s bytes.

Reuse of RED slot j + OVF_SLOT: a reducer writes it only after this call's 3 pullers of the row arrived, i.e. after
their previous calls (and reads) ended, as for every RED slot. Rows <= 2 * OVF_SLOT * WORLD (q <= OVF_SLOT).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm.add_rmsnorm import _block_sum
from ..add_rmsnorm.common import (
    PRELOAD_HINTS,
    H,
    bf16x8,
    pk_bf16,
    rsqrt_rn,
    seq_sum_sq,
    wave_sum_lane63,
)
from .common import (
    SYS, WORLD, _enter_fp32_ftz,
    _restore_fp32_denorm_mode, _sum4_hardware_ftz, load_ptrs, sel4, unrotate,
)
from .ar_add_rmsnorm_register import (
    MAX_ROWS, RED_OFF, register_arrive,
    register_wait_pullers, register_publish, register_wait_reduced,
)

NW = H // 8 // 64  # 12 waves of 8 columns per lane
NT = 64 * NW
ROWB = H * 2
HDR = 9216  # header bytes of a coded row
HDW = HDR // 4
NEXC = 6  # exceptions per wave
RAWF = 0x100  # count word: the wave is raw
SPECF = 0x200  # count word: the wave holds a zero or an exception
SIGN2 = -2147450880  # 0x80008000
NIBPOS = [4 * (v // 2) + 16 * (v % 2) for v in range(8)]
OVF_SLOT = (MAX_ROWS // WORLD) // 2  # raw waves of slot j go to RED slot j + OVF_SLOT
MAX_LOSSLESS_ROWS = OVF_SLOT * WORLD


def _dpp_i(v, ctrl):
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, _llvm.mlir_undef(T.i32), fx.as_ir_value(fx.Int32(v)), ctrl, 0xF,
                                          0xF, True))


def _imax(a, b):
    return (a > b).select(a, b)


def compile_ar_add_rmsnorm_pull_lossless():
    name = "allreduce_pull_lossless12_ranks4_gemma_add_rmsnorm_h6144_block_ftz"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]
        hdr: fx.Array[fx.Int32, 96]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,  # rows per chunk (ceil(rows / 4)); grid = 4 q
        i32_me: fx.Int32,
        i32_sx: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # int64[4]: the ranks' inputs [rows, H] bf16
        arg_b0: fx.Int64,  # the ranks' register buffers (flags + reduced rows)
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,  # residual [rows, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_out: fx.Int64,  # [rows, H] bf16
        arg_res: fx.Int64,  # residual out [rows, H] bf16
        arg_cnt: fx.Int64,  # int32[MAX_ROWS] call counts
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, b = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        me, q = i32_me, i32_q
        reducer = b < q
        puller = b >= q
        k = b - q
        one, zero = fx.Int32(1), fx.Int32(0)
        i = (k >= q).select(one, zero) + (k >= 2 * q).select(one, zero)
        c = reducer.select(me, (me + 1 + i) % WORLD)
        j = reducer.select(b, k - i * q)
        rw = c * q + j
        if rw < i32_rows:
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            hdr = smem.hdr.ptr
            bs = (arg_b0, arg_b1, arg_b2, arg_b3)
            mine = sel4(me, *bs)
            cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=MAX_ROWS * 4)
            e = fx.Int32(buffer_ops.buffer_load(cr, rw, vec_width=1, dtype=fx.Int32, is_scalar=True)) + 1
            off = tx * 4
            rowoff = fx.Int64(rw) * ROWB

            def row(base, cm, m):
                rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
                return Vec(buffer_ops.buffer_load(rs, off, vec_width=4, dtype=fx.Int32, mask=m, cache_modifier=cm))

            xs = load_ptrs(arg_x)
            local = row(sel4(me, *xs) + rowoff, 0, reducer)
            register_arrive(*bs, c, me, rw, fx.Int32(0), e, puller & (wave == 0))
            register_wait_pullers(mine, me, rw, fx.Int32(0), e, reducer & (wave == 0))
            fx.gpu.barrier()
            rot = wave % (WORLD - 1)
            peer = [row(sel4((me + 1 + (t + rot) % (WORLD - 1)) % WORLD, *xs) + rowoff, SYS, reducer)
                    for t in range_constexpr(WORLD - 1)]
            rr = buffer_ops.create_buffer_resource_from_addr(arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2),
                                                            num_records_bytes=ROWB)
            rvraw = Vec(buffer_ops.buffer_load(rr, off, vec_width=4, dtype=fx.Int32))
            wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=ROWB)
            wv = bf16x8(Vec(buffer_ops.buffer_load(wr, off, vec_width=4, dtype=fx.Int32)))
            packs, mode = _enter_fp32_ftz([local] + unrotate(rot, peer))
            reduced, rvpack = _restore_fp32_denorm_mode([_sum4_hardware_ftz(packs), rvraw], mode)
            # the code of the reduced row (the reducer's lanes; pullers compute it on zeros and store nothing)
            rd = [fx.Int32(reduced[t]) for t in range_constexpr(4)]
            us = []
            for t in range_constexpr(4):
                us.append(rd[t] & 0xFFFF)
                us.append((rd[t] >> 16) & 0xFFFF)
            ex = [(u >> 7) & 0xFF for u in us]
            gmax = ex[0]
            for t in range_constexpr(1, 8):
                gmax = _imax(gmax, ex[t])
            gmax = _imax(gmax, _dpp_i(gmax, 0xB1))  # quad_perm [1, 0, 3, 2]
            gmax = _imax(gmax, _dpp_i(gmax, 0x4E))  # quad_perm [2, 3, 0, 1]
            dd = [gmax - ex[t] for t in range_constexpr(8)]
            big = [dd[t] > 14 for t in range_constexpr(8)]
            exc = [big[t] & ((us[t] & 0x7FFF) != 0) for t in range_constexpr(8)]
            nib = fx.Int32(0)
            nbig = fx.Int32(0)
            nexc = fx.Int32(0)
            ev = fx.Int32(0)
            for t in range_constexpr(8):
                nib = nib | (big[t].select(fx.Int32(15), dd[t]) << NIBPOS[t])
                nbig = nbig + big[t].select(one, zero)
                nexc = nexc + exc[t].select(one, zero)
                ev = ev | exc[t].select(us[t], zero)
            has = nexc > 0
            vsp = fx.Int64(fx.rocdl.ballot(T.i64, nbig > 0))
            v1 = fx.Int64(fx.rocdl.ballot(T.i64, has))
            av = [((us[t] >> 8) & 0x80) | (us[t] & 0x7F) for t in range_constexpr(8)]
            a0 = av[0] | (av[1] << 8) | (av[2] << 16) | (av[3] << 24)
            a1 = av[4] | (av[5] << 8) | (av[6] << 16) | (av[7] << 24)
            cnt = zero
            rawi = zero
            idx = one
            if v1 != fx.Int64(0):
                # an exception in this wave (wave-uniform, rare): list indices, or a raw wave
                v2 = fx.Int64(fx.rocdl.ballot(T.i64, nexc > 1))
                cnt = fx.Int64(fx.math.ctpop(v1)).to(fx.Int32)
                rawi = ((v2 != fx.Int64(0)) | (cnt > NEXC)).select(one, zero)
                below = v1 & ((fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1))
                idx = fx.Int64(fx.math.ctpop(below)).to(fx.Int32) + 1
                ax = [exc[t].select(((us[t] >> 8) & 0x80) | idx, av[t]) for t in range_constexpr(8)]
                a0 = ax[0] | (ax[1] << 8) | (ax[2] << 16) | (ax[3] << 24)
                a1 = ax[4] | (ax[5] << 8) | (ax[6] << 16) | (ax[7] << 24)
            rawb = rawi != 0
            cwv = rawb.select(fx.Int32(RAWF), cnt | (vsp != fx.Int64(0)).select(fx.Int32(SPECF), zero))
            slot = buffer_ops.create_buffer_resource_from_addr(mine + RED_OFF + fx.Int64(j) * ROWB,
                                                              num_records_bytes=ROWB)
            ovf = buffer_ops.create_buffer_resource_from_addr(mine + RED_OFF + fx.Int64(j + OVF_SLOT) * ROWB,
                                                             num_records_bytes=ROWB)
            buffer_ops.buffer_store(reduced, ovf, off, mask=reducer & rawb, cache_modifier=SYS)
            buffer_ops.buffer_store(Vec.from_elements([a0, a1], fx.Int32), slot, tx * 2, mask=reducer,
                                    cache_modifier=SYS)
            buffer_ops.buffer_store(nib, slot, H // 4 + tx, mask=reducer, cache_modifier=SYS)
            buffer_ops.buffer_store(gmax.to(fx.Int8), slot, HDR + 32 * wave + lane // 4,
                                    mask=reducer & (lane % 4 == 0), cache_modifier=SYS, offset_is_bytes=True)
            buffer_ops.buffer_store(cwv, slot, HDW + 8 * wave + 4, mask=reducer & (lane == 0), cache_modifier=SYS)
            buffer_ops.buffer_store(ev.to(fx.Int16), slot, HDR + 32 * wave + 20 + 2 * (idx - 1),
                                    mask=reducer & has & (rawi == 0), cache_modifier=SYS, offset_is_bytes=True)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            register_publish(*bs, me, rw, fx.Int32(0), e, reducer & (wave == 0))
            # the reducer never waits on vmcnt after publishing (see ``ar_add_rmsnorm_pull``)
            x0 = rd[0] << 16
            x1 = rd[0] & -65536
            x2 = rd[1] << 16
            x3 = rd[1] & -65536
            x4 = rd[2] << 16
            x5 = rd[2] & -65536
            x6 = rd[3] << 16
            x7 = rd[3] & -65536
            if puller:
                register_wait_reduced(mine, rw, fx.Int32(0), e)
                fx.gpu.barrier()
                src = sel4(c, *bs) + RED_OFF
                rs = buffer_ops.create_buffer_resource_from_addr(src + fx.Int64(j) * ROWB, num_records_bytes=ROWB)
                hv = Vec(buffer_ops.buffer_load(rs, HDW + 4 * lane, vec_width=4, dtype=fx.Int32,
                                                mask=(wave == 0) & (lane < 24), cache_modifier=SYS))
                pa = Vec(buffer_ops.buffer_load(rs, tx * 2, vec_width=2, dtype=fx.Int32, cache_modifier=SYS))
                pb = fx.Int32(buffer_ops.buffer_load(rs, H // 4 + tx, vec_width=1, dtype=fx.Int32,
                                                     cache_modifier=SYS))
                if (wave == 0) & (lane < 24):
                    for t in range_constexpr(4):
                        hdr[4 * lane + t] = fx.Int32(hv[t])
                fx.gpu.barrier()
                hw = [fx.Int32(hdr[8 * wave + t]) for t in range_constexpr(8)]
                q4 = lane // 16
                hs = (q4 == 0).select(hw[0], (q4 == 1).select(hw[1], (q4 == 2).select(hw[2], hw[3])))
                gE = (hs >> (((lane // 4) % 4) * 8)) & 0xFF
                cw = fx.Int32(fx.rocdl.readfirstlane(T.i32, hw[4]))
                pw = [fx.Int32(pa[0]), fx.Int32(pa[1])]
                e16 = gE | (gE << 16)
                dec = []
                for p in range_constexpr(4):
                    w = pw[p // 2]
                    sh = 16 * (p % 2)
                    a16 = ((w >> sh) & 0xFF) | (((w >> (sh + 8)) & 0xFF) << 16)
                    d16 = (pb >> (4 * p)) & 0x000F000F
                    bf = ((a16 << 8) & SIGN2) | (((e16 - d16) << 7) | (a16 & 0x007F007F))
                    dec.append(bf << 16)
                    dec.append(bf & -65536)
                x0, x1, x2, x3, x4, x5, x6, x7 = dec
                if (cw & SPECF) != 0:
                    # a zero or an exception in this wave (wave-uniform): the per-value selects
                    fixed = []
                    xs8 = [x0, x1, x2, x3, x4, x5, x6, x7]
                    for t in range_constexpr(8):
                        a = (pw[t // 4] >> (8 * (t % 4))) & 0xFF
                        d = (pb >> NIBPOS[t]) & 15
                        sg = (a & 0x80) << 24
                        isx = (d == 15) & ((a & 0x7F) != 0)
                        sl = isx.select((a & 0x7F) - 1, zero)
                        h = sl // 2
                        dw = (h == 0).select(hw[5], (h == 1).select(hw[6], hw[7]))
                        xv = ((dw >> ((sl % 2) * 16)) & 0xFFFF) << 16
                        fixed.append(isx.select(xv, (d == 15).select(sg, xs8[t])))
                    x0, x1, x2, x3, x4, x5, x6, x7 = fixed
                if (cw & RAWF) != 0:
                    ovp = buffer_ops.create_buffer_resource_from_addr(src + fx.Int64(j + OVF_SLOT) * ROWB,
                                                                     num_records_bytes=ROWB)
                    rv_ = Vec(buffer_ops.buffer_load(ovp, off, vec_width=4, dtype=fx.Int32, cache_modifier=SYS))
                    r0 = fx.Int32(rv_[0])
                    r1 = fx.Int32(rv_[1])
                    r2 = fx.Int32(rv_[2])
                    r3 = fx.Int32(rv_[3])
                    x0 = r0 << 16
                    x1 = r0 & -65536
                    x2 = r1 << 16
                    x3 = r1 & -65536
                    x4 = r2 << 16
                    x5 = r2 & -65536
                    x6 = r3 << 16
                    x7 = r3 & -65536
            x = [v.bitcast(fx.Float32) for v in (x0, x1, x2, x3, x4, x5, x6, x7)]
            rv = bf16x8(rvpack)
            s = [x[t] + rv[t] for t in range_constexpr(8)]
            res = [pk_bf16(s[2 * t], s[2 * t + 1]) for t in range_constexpr(4)]
            acc = wave_sum_lane63(seq_sum_sq(s))
            if lane == 63:
                part[wave] = acc
            resr = buffer_ops.create_buffer_resource_from_addr(arg_res + rowoff, num_records_bytes=ROWB)
            buffer_ops.buffer_store(Vec.from_elements(res, fx.Int32), resr, off)
            fx.gpu.barrier()
            rstd = rsqrt_rn(_block_sum([part[t] for t in range_constexpr(NW)]), i32_eps.bitcast(fx.Float32))
            y = [(s[t] * rstd) * (wv[t] + fx.Float32(1.0)) for t in range_constexpr(8)]
            out = [pk_bf16(y[2 * t], y[2 * t + 1]) for t in range_constexpr(4)]
            outr = buffer_ops.create_buffer_resource_from_addr(arg_out + rowoff, num_records_bytes=ROWB)
            buffer_ops.buffer_store(Vec.from_elements(out, fx.Int32), outr, off)
            buffer_ops.buffer_store(e, cr, rw, mask=tx == 0)

    @flyc.jit
    def launch_ar_add_rmsnorm_pull_lossless(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,
        i32_me: fx.Int32,
        i32_sx: fx.Int32,
        i32_sr: fx.Int32,
        arg_x: fx.Int64,
        arg_b0: fx.Int64,
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_out: fx.Int64,
        arg_res: fx.Int64,
        arg_cnt: fx.Int64,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows, i32_q, i32_me, i32_sx, i32_sr, arg_x, arg_b0, arg_b1, arg_b2, arg_b3,
            arg_r, arg_w, arg_out, arg_res, arg_cnt, i32_eps,
        ).launch(grid=(fx.Int64(i32_q * WORLD), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_ar_add_rmsnorm_pull_lossless.kernel_name = name
    launch_ar_add_rmsnorm_pull_lossless.compile_hints = PRELOAD_HINTS
    return launch_ar_add_rmsnorm_pull_lossless
