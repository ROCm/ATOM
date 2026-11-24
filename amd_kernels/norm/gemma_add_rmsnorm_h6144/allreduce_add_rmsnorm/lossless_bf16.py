# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Exact BF16 transport: 4-bit exponent deltas, sign and all 7 mantissa bits.

A virtual wave owns 512 contiguous values. Four lanes share the maximum BF16
exponent of 32 values. Deltas 0..14 reconstruct the original exponent. Delta 15
means signed zero or an index into six raw BF16 exceptions. A lane with multiple
exceptions, or a wave with more than six, stores its entire 1 KB in a raw slot.
All 16-bit patterns, including NaN payloads and signed zeros, round-trip exactly.

A row's first 6144 B are 8 sign/mantissa bytes per lane. Each virtual wave then
owns 320 B: 256 B packed delta nibbles, 32 B initialized header, 32 B padding.
The header has 16 exponent bytes, flags/count (4 B), six exception values (12 B).
The reader fetches the nibbles and initialized header in one coalesced load,
then decodes BF16 pairs. The 12 virtual waves transfer 9984 B (0.8125x BF16).
Raw slots are caller-owned and follow the same drained-store/flag protocol.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from amd_kernels.common import buffer_ops
from .common import SYS
from ..add_rmsnorm.common import H
ROWB = H * 2
NIBPOS = [4 * (v // 2) + 16 * (v % 2) for v in range(8)]
RAWF, SPECF, NEXC = 0x100, 0x200, 6
SIGN2 = -2147450880  # 0x80008000


def _dpp_i(v, ctrl):
    # FlyDSL has no typed i32 quad-permutation helper (only the raw rocdl.update_dpp).
    return fx.Int32(_rocdl_ops.update_dpp(
        T.i32, _llvm.mlir_undef(T.i32), fx.as_ir_value(fx.Int32(v)), ctrl, 0xF, 0xF, True
    ))


def _imax(a, b):
    return (a > b).select(a, b)


@flyc.jit
def enc(value, lane):
    """Code one lane's eight BF16 values; return the wave header and payload."""
    one, zero = fx.Int32(1), fx.Int32(0)
    rd = [fx.Int32(value[t]) for t in range_constexpr(4)]
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
    return a0, a1, nib, gmax, cwv, ev, idx, has, rawi


def _dec_fast(pw, pb, hw, lane):
    """4 packed bf16 pairs of a lane from its planes and its wave's header words (no zero / exception)."""
    q4 = lane // 16
    hs = (q4 == 0).select(hw[0], (q4 == 1).select(hw[1], (q4 == 2).select(hw[2], hw[3])))
    gE = (hs >> (((lane // 4) % 4) * 8)) & 0xFF
    e16 = gE | (gE << 16)
    out = []
    for p in range_constexpr(4):
        w = pw[p // 2]
        sh = 16 * (p % 2)
        a16 = ((w >> sh) & 0xFF) | (((w >> (sh + 8)) & 0xFF) << 16)
        d16 = (pb >> (4 * p)) & 0x000F000F
        out.append(((a16 << 8) & SIGN2) | (((e16 - d16) << 7) | (a16 & 0x007F007F)))
    return out


def _dec_fix(pairs, pw, pb, hw):
    """The pairs with zeros and exceptions applied (the per-value selects)."""
    zero = fx.Int32(0)
    e0, e1, e2 = hw[5], hw[6], hw[7]
    v16 = []
    for t in range_constexpr(8):
        a = (pw[t // 4] >> (8 * (t % 4))) & 0xFF
        d = (pb >> NIBPOS[t]) & 15
        isx = (d == 15) & ((a & 0x7F) != 0)
        sl = isx.select((a & 0x7F) - 1, zero)
        h = sl // 2
        dw = (h == 0).select(e0, (h == 1).select(e1, e2))
        xv = (dw >> ((sl % 2) * 16)) & 0xFFFF
        normal = (pairs[t // 2] >> (16 * (t % 2))) & 0xFFFF
        v16.append(isx.select(xv, (d == 15).select((a & 0x80) << 8, normal)))
    return [v16[2 * p] | (v16[2 * p + 1] << 16) for p in range_constexpr(4)]



@flyc.jit
def enc_store(value, base, raw, tid, mask):
    """Write one virtual wave's coded row or exact raw fallback, without publishing."""
    lane, wave = tid % 64, tid // 64
    a0, a1, nib, gmax, cwv, ev, idx, has, rawi = enc(value, lane)
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
    rr = buffer_ops.create_buffer_resource_from_addr(raw, num_records_bytes=ROWB)
    buffer_ops.buffer_store(value, rr, tid * 4, mask=mask & (rawi != 0), cache_modifier=SYS)
    buffer_ops.buffer_store(
        Vec.from_elements([a0, a1], fx.Int32), rs, tid * 2, mask=mask, cache_modifier=SYS
    )
    buffer_ops.buffer_store(nib, rs, H // 4 + wave * 80 + lane, mask=mask, cache_modifier=SYS)
    hd = H + wave * 320 + 256
    buffer_ops.buffer_store(
        gmax.to(fx.Int8), rs, hd + lane // 4, mask=mask & (lane % 4 == 0),
        cache_modifier=SYS, offset_is_bytes=True,
    )
    buffer_ops.buffer_store(cwv, rs, hd // 4 + 4, mask=mask & (lane == 0), cache_modifier=SYS)
    buffer_ops.buffer_store(
        ev.to(fx.Int16), rs, hd + 20 + 2 * (idx - 1),
        mask=mask & has & (rawi == 0), cache_modifier=SYS, offset_is_bytes=True,
    )


@flyc.jit
def fetch(base, tid, mask):
    """Issue both independent payload loads; decode only after all fetches are issued."""
    lane, wave = tid % 64, tid // 64
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
    bl = Vec(buffer_ops.buffer_load(
        rs, H // 4 + wave * 80 + lane * 2, vec_width=2, dtype=fx.Int32,
        mask=mask & (lane < 36), cache_modifier=SYS,
    ))
    pa = Vec(buffer_ops.buffer_load(
        rs, tid * 2, vec_width=2, dtype=fx.Int32, mask=mask, cache_modifier=SYS,
    ))
    return fx.Int32(pa[0]), fx.Int32(pa[1]), fx.Int32(bl[0]), fx.Int32(bl[1])


@flyc.jit
def finish(data, raw, tid, mask):
    """Decode four packed BF16 pairs; raw waves use the caller's separate slot."""
    lane = tid % 64
    pa, bl = data[:2], data[2:]
    b0 = fx.Int32(fx.rocdl.ds_bpermute(T.i32, (lane // 2) * 4, bl[0]))
    b1 = fx.Int32(fx.rocdl.ds_bpermute(T.i32, (lane // 2) * 4, bl[1]))
    pb = ((lane % 2) == 0).select(b0, b1)
    hw = [
        fx.Int32(fx.rocdl.readlane(T.i32, bl[t % 2], fx.Int32(32 + t // 2)))
        for t in range_constexpr(8)
    ]
    pw = [fx.Int32(pa[0]), fx.Int32(pa[1])]
    cw = fx.Int32(fx.rocdl.readfirstlane(T.i32, hw[4]))
    d0, d1, d2, d3 = _dec_fast(pw, pb, hw, lane)
    if (cw & SPECF) != 0:
        d0, d1, d2, d3 = _dec_fix([d0, d1, d2, d3], pw, pb, hw)
    if (cw & RAWF) != 0:
        rr = buffer_ops.create_buffer_resource_from_addr(raw, num_records_bytes=ROWB)
        r = Vec(buffer_ops.buffer_load(
            rr, tid * 4, vec_width=4, dtype=fx.Int32, mask=mask, cache_modifier=SYS,
        ))
        d0, d1, d2, d3 = fx.Int32(r[0]), fx.Int32(r[1]), fx.Int32(r[2]), fx.Int32(r[3])
    return Vec.from_elements([d0, d1, d2, d3], fx.Int32)
