# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 (MI455X) counterparts of ``atom.mono.device.ops``: the wave32 lane
exchanges, reductions and scans, the WMMA operands and accumulator readout, the
gfx12 cache-policy bits and store completion.

The execution model is the gfx950 one in CTAs and threads (BLOCKS CTAs, one a
CU, THREADS threads each) but in waves of 32: WAVES = 16. The gfx950 modules are
untouched; a gfx1250 kernel imports the arch-independent helpers from ``ops``
and these.

WMMA layouts (checked against torch on gfx1250, 8-bit operands): lane l holds
row (A) / column (B) l % 16, byte slot s at k = 32 (s // 16) + 16 (l // 16) +
s % 16; bf16 16x16x32: element slot s at k = 16 (s // 8) + 8 (l // 16) + s % 8.
The f32 accumulator: lane l holds column l % 16, rows 8 (l // 16) + i, i < 8.
"""

from __future__ import annotations

from flydsl._mlir import ir
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.mono.device.ops import CM_NT_GFX1250, CM_SYS_GFX1250, traced

BLOCKS = 256
THREADS = 512
WAVE = 32
WAVES = THREADS // WAVE
LDS_BYTES = 320 * 1024

# gfx12 CPol: TH in bits [0:2], SCOPE in bits [3:4]. SCOPE_DEV = 2 << 3 is past
# the per-XCD L2s; SCOPE_SYS = 3 << 3 reaches peers over XGMI.
CM_DEV = 16
CM_SYS = CM_SYS_GFX1250
CM_NT = CM_NT_GFX1250

_ROW_XMASK = 0x160
_SEL_LO, _SEL_HI = 0x76543210, 0xFEDCBA98 - (1 << 32)


def _bits(v):
    return v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)


def _typed_like(v, bits):
    return bits.bitcast(fx.Float32) if isinstance(v, fx.Float32) else type(v)(bits)


def xshfl(v, off):
    """Value of lane ``lane ^ off``: in-row offsets by one DPP row_xmask, 16 by
    v_permlanex16 (identity selects: lane i of a row reads lane i of the other)."""
    assert off in (1, 2, 4, 8, 16), off
    x = _bits(v)
    if off == 16:
        y = fx.Int32(
            rocdl.permlanex16(
                T.i32, as_ir_value(x), as_ir_value(x), as_ir_value(fx.Int32(_SEL_LO)),
                as_ir_value(fx.Int32(_SEL_HI)), True, True,
            )
        )  # fmt: skip
    else:
        y = fx.Int32(update_dpp_i32(x, x, _ROW_XMASK | off, 0xF, 0xF, False))
    return _typed_like(v, y)


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """op over lanes ``lane ^ off`` for each offset, in the given order (the
    order is the numerical contract of a sum)."""
    for off in offsets:
        v = op(v, xshfl(v, off))
    return v


def wave_sum(v):
    return butterfly(v, (16, 8, 4, 2, 1))


def wave_max(v):
    return butterfly(v, (16, 8, 4, 2, 1), fx.max)


def wave_incl_scan(v, lane):
    """Inclusive prefix sum of an Int32 over the wave, in lane order."""
    blk = v
    for off in (1, 2, 4, 8, 16):
        o = xshfl(blk, off)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v


@traced
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of v over the CTA in thread order, CTA total); ``buf``
    needs WAVES words."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == WAVE - 1:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot


@traced
def block_max(v, lane, wave, red):
    """Max of v over the CTA's WAVES waves; ``red`` needs WAVES words."""
    wv = wave_max(v)
    gpu.barrier()
    if lane == 0:
        fx.ptr_store(wv, red + wave)
    gpu.barrier()
    t = fx.ptr_load(red + 0)
    for w in range_constexpr(1, WAVES):
        t = fx.max(t, fx.ptr_load(red + w))
    gpu.barrier()
    return t


@traced
def group_floor(keys32, k, red, lane, wave):
    """``device.select.group_floor`` on wave32 (groups are lanes of one
    ``tid % k``, as there). ``red``: WAVES * k + 1 words of LDS."""
    assert k in (1, 2, 4, 8, 16, 32), k
    top = keys32[0]
    for v in keys32[1:]:
        top = fx.max(top, v)
    top = butterfly(top, tuple(o for o in (16, 8, 4, 2, 1) if o >= k), fx.max)
    if lane < k:
        fx.ptr_store(top, red + (wave * k + lane))
    gpu.barrier()
    if (wave == 0) & (lane < k):
        g = fx.ptr_load(red + lane)
        for w in range_constexpr(1, WAVES):
            g = fx.max(g, fx.ptr_load(red + (w * k + lane)))
        g = butterfly(g, tuple(o for o in (16, 8, 4, 2, 1) if o < k), fx.min)
        if lane == 0:
            fx.ptr_store(g, red + WAVES * k)
    gpu.barrier()
    return fx.ptr_load(red + WAVES * k)


def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane."""
    return fx.Int32(rocdl.ballot(T.i32, as_ir_value(pred)))


def lanes_below(bal, lane):
    """How many of the lanes in ballot ``bal`` are below ``lane``."""
    return fx.Int32(fx.ctpop(bal & ((fx.Int32(1) << lane) - 1)))


def _llvm_ptr(addr, space):
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=space, alignment=4)
    ptr = fx.to_llvm_ptr(fx.inttoptr(pt, addr))
    return ptr._value if hasattr(ptr, "_value") else ptr


def lds_addr(ptr):
    """The LDS byte address of an LDS array pointer."""
    return fx.Int32(fx.ptrtoint(ptr))


def async_to_lds(gaddr, laddr, b128=True, cm=0):
    """16 B (``b128``) or 4 B global -> LDS a lane, counted by ASYNCcnt: the copy
    is an op with side effects, so it stays where it is issued (``wait_async``).
    ``cm``: its cache policy (CM_NT streaming, CM_DEV another XCD's writes)."""
    g = _llvm_ptr(fx.Int64(gaddr), fx.AddressSpace.Global)
    lp = _llvm_ptr(fx.Int32(laddr), fx.AddressSpace.Shared)
    aux = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cm) if cm else None
    if b128:
        rocdl.global_load_async_to_lds_b128(g, lp, 0, aux=aux)
    else:
        rocdl.global_load_async_to_lds_b32(g, lp, 0, aux=aux)


def wait_async(n):
    """Until at most ``n`` of this wave's async copies are in flight (in order)."""
    rocdl.s_wait_asynccnt(n)


def lds_reads_done():
    """Until this wave's LDS reads are done: an async copy into LDS a read of the
    wave may still be reading (its write lands whenever the copy returns)."""
    rocdl.s_wait_dscnt(0)


def release_stores():
    """Until this wave's stores have completed at their scope: a flag published
    after it is seen after the data (a release fence orders, it does not wait)."""
    rocdl.s_wait_storecnt(0)


@traced
def publish(put, a, i, v, who):
    """``sync.publish`` on gfx12, where stores drain by storecnt: this CTA's plain
    stores readable by every CTA, then (thread ``who``) the pair (``a``, i) := v."""
    release_stores()
    gpu.barrier()
    if who:
        put(a, i, v)


# --- WMMA ---------------------------------------------------------------------


def wmma_fp8_k64(a, b, c):
    """16x16x64 E4M3 x E4M3 -> f32: ``a`` / ``b`` 8 i32 (32 fp8) a lane, ``c`` 8 f32."""
    return fx.Vector(
        rocdl.wmma_f32_16x16x64_fp8_fp8(
            fx.Vector.make_type(8, fx.Float32), as_ir_value(a), as_ir_value(b), as_ir_value(c)
        ).result
    )


def wmma_fp8_k128(a, b, c):
    """16x16x128 E4M3 x E4M3 -> f32: ``a`` / ``b`` 16 i32 (64 fp8) a lane."""
    return fx.Vector(
        rocdl.wmma_f32_16x16x128_fp8_fp8(fx.Vector.make_type(8, fx.Float32), a, b, c)
    )


def wmma_bf16_k32(a, b, c):
    """16x16x32 bf16 -> f32: ``a`` / ``b`` 16 bf16 a lane."""
    return fx.Vector(
        rocdl.wmma_f32_16x16x32_bf16(
            fx.Vector.make_type(8, fx.Float32), as_ir_value(a), as_ir_value(b), as_ir_value(c)
        ).result
    )


def wmma_a8w4(a, b, c, sa, sb):
    """Scaled 16x16x128 E2M1 x E4M3 -> f32: ``a`` 8 i32 (64 fp4, low nibble
    first: k = 64 (n // 32) + 32 (l // 16) + n % 32), ``b`` 16 i32 of fp8, ``sa`` /
    ``sb`` the i32 of A's row / B's column four E8M0 codes (k-blocks 0 .. 3)."""
    return fx.Vector(
        rocdl.wmma_scale_f32_16x16x128_f8f6f4(
            fx.Vector.make_type(8, fx.Float32), a, b, c, sa, sb, fmtA=4, fmtB=0
        )
    )


def shuffled_rows(r_w, rg, k_bytes, kc, lane, chunk_bytes, cm=0):
    """A lane's WMMA A operand of a ``shuffle_weight(layout=(16, 16))`` weight:
    row group rg, k-chunk kc of ``chunk_bytes`` (64: an FP8 k64 / FP4 k128 chunk,
    two 16 B pieces; 128: an FP8 k128 chunk, four), ``k_bytes`` a row."""
    pieces = chunk_bytes // 32
    parts = []
    for c in range_constexpr(pieces):
        w4 = fx.Vector(
            bo.buffer_load(
                r_w,
                ((rg * (k_bytes // 32) + kc * pieces + c) * 512 + lane * 16) // 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        parts += [w4[e] for e in range(4)]
    return fx.Vector.from_elements(parts, fx.Int32)


def n32k4_word(r_s, row, step, k):
    """The i32 of row ``row``'s four E8M0 codes of 128-k step ``step`` in a
    ``shuffle_scale_n32k4`` scale (rows of every expert in order, ``k`` columns)."""
    return fx.Int32(
        bo.buffer_load(
            r_s, ((row // 32) * k + step * 128 + (row % 32) * 4) // 4, vec_width=1, dtype=T.i32
        )
    )


def acc_slot(r, t):
    """(lane, slot) of C[r][t] in a 16x16 f32 accumulator."""
    return t + 16 * (r // 8), r % 8


def acc_row_sum(red, r, t, waves=None):
    """C[r][t] summed over the waves' accumulators (wave w's at ``red`` +
    (w * WAVE + lane) * 8), in wave order from 0 (``waves``: which, in order)."""
    lane, slot = acc_slot(r, t)
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + ((w * WAVE + lane) * 8 + slot))
    return tot


def fp8x8_bf16(words, scale=0x7F):
    """Eight E4M3 of two dwords -> 8 bf16, times 2^(scale - 127) (E8M0 ``scale``,
    unit by default): one v_cvt_scale_pk8_bf16_fp8."""
    return fx.Vector(
        rocdl.cvt_scale_pk8_bf16_fp8(
            fx.Vector.make_type(8, fx.BFloat16),
            as_ir_value(fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32)),
            as_ir_value(fx.Int32(scale)),
            0,
        )
    )
