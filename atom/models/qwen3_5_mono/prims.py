# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device primitives of the Qwen3.5 gfx1250 mono kernels (wave32, one CTA a CU).

Memory-ordering contract
------------------------
* Within a CTA: LDS + ``gpu.barrier``.
* Between the CTAs of a rank: a value one CTA writes and another reads after a grid
  barrier is stored and loaded at agent scope (``st_dev`` / ``ld_dev*``), past the
  per-CU caches, so the barrier only orders (workgroup-scope fences); see
  ``grid_barrier``.
* Between ranks: LL pairs. Every 32-bit value travels as one 64-bit store
  ``{value, tag}`` at system scope into the peer's buffer, and the reader polls the
  pair until the tag matches: no flag, fence or completion wait.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, ptrtoint, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.rocdl.enum import SyncScope as RocdlScope
from flydsl.expr.typing import T

from atom.models.qwen3_5_mono.config import WAVE

AGENT = RocdlScope.Agent
WORKGROUP = RocdlScope.Workgroup
CM_DEV = 2 << 3  # gfx12 CPol SCOPE_DEV
CM_SYS = 3 << 3  # gfx12 CPol SCOPE_SYS


# ---------------------------------------------------------------- addressing


def uniform(v):
    """``v`` (i32) made wave-uniform (v_readfirstlane: an SGPR)."""
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def opaque(v):
    """``v`` (a wave-uniform i32) through an empty inline asm: the compiler cannot
    see it equals ``v``, so loads addressed from it are not merged with (or hoisted
    together with) the same loads in another branch."""
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr.typing import as_ir_value

    op = _llvm.InlineAsmOp(
        res=T.i32, operands_=[as_ir_value(fx.Int32(v))], asm_string="",
        constraints="=s,s", has_side_effects=False, is_align_stack=False,
    )  # fmt: skip
    return fx.Int32(op.res)


def compiler_fence():
    """An empty inline asm with side effects and a memory clobber: no instruction, but
    the compiler cannot move loads across it (a poll loop without one gets its load
    hoisted out, and spins forever)."""
    from flydsl._mlir.dialects import llvm as _llvm

    _llvm.InlineAsmOp(
        res=None, operands_=[], asm_string="", constraints="~{memory}",
        has_side_effects=True, is_align_stack=False,
    )  # fmt: skip


def load_ptr64(res, i):
    """Entry ``i`` of an int64 table behind ``res``, wave-uniform. A pointer loaded
    plainly lands in VGPRs, and a buffer resource built from it makes the compiler
    wrap every access in a waterfall loop (and keep the descriptor in VGPRs)."""
    pw = fx.Vector(bo.buffer_load(res, fx.Int32(i * 2), vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


def load_table(res, n_ptrs):
    """An int64 table's first ``n_ptrs`` entries via scalar (SMEM) loads, all issued before any is used: straight into SGPRs, one latency.
    (A vector load + readfirstlane a pointer costs a memory round trip before the
    pointer can address anything.)"""
    words = []
    # 4 dwords (2 entries) an s_buffer_load: LLVM narrows a load whose tail is unused,
    # and only x2 / x4 / x8 / x16 select (an 8-dword load narrowed to 6 does not)
    for c in range((2 * n_ptrs + 3) // 4):
        v = fx.Vector(bo.buffer_load(res, fx.Int32(4 * c), vec_width=4, dtype=T.i32, is_scalar=True))
        words += [v[i] for i in range(4)]
    return [(fx.Int64(fx.Int32(words[2 * i + 1])) << 32) | fx.Int64(fx.Uint32(fx.Int32(words[2 * i]))) for i in range(n_ptrs)]


def prefetch_line(addr_i64):
    """global_prefetch_b8: bring the line at ``addr_i64`` into the cache hierarchy."""
    ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global, 1)
    rocdl.global_prefetch(fx.to_llvm_ptr(fx.inttoptr(ty, fx.Int64(addr_i64))))


def rsrc_at(addr_i64, live=None):
    """A buffer resource over the address space from raw address ``addr_i64``. With
    ``live`` (wave-uniform), an empty one where it is false: its loads return 0
    without a memory access (an idle wave's clamped preload would otherwise read the
    same row as every other idle wave's, and serialize them on its L2 lines)."""
    if live is None:
        return bo.create_buffer_resource_from_addr(fx.Int64(addr_i64))
    n = live.select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    return bo.create_buffer_resource_from_addr(fx.Int64(addr_i64), num_records_bytes=n.ir_value())


def ld_dw(res, dw_off, n: int = 4, cm: int = 0):
    """``n`` (1, 2 or 4) i32 dwords at dword offset ``dw_off``, as Uint32s."""
    v = bo.buffer_load(res, fx.Int32(dw_off), vec_width=n, dtype=T.i32, cache_modifier=cm)
    if n == 1:
        return [fx.Uint32(v)]
    v = fx.Vector(v)
    return [fx.Uint32(v[i]) for i in range(n)]


def sld(res, dw_off):
    """One dword at a wave-uniform dword offset via a scalar (SMEM) load: tracked by
    kmcnt, so it does not retire behind the wave's outstanding vector loads (a
    metadata chain such as context length -> block-table entry -> K / V address
    would otherwise wait on every weight load issued before it)."""
    return fx.Uint32(bo.buffer_load(res, fx.Int32(dw_off), vec_width=1, dtype=T.i32, is_scalar=True))


def ld_f32(res, off, n: int = 4, cm: int = 0):
    """``n`` f32 at f32 offset ``off``."""
    v = bo.buffer_load(res, fx.Int32(off), vec_width=n, dtype=T.f32, cache_modifier=cm)
    if n == 1:
        return [fx.Float32(v)]
    v = fx.Vector(v)
    return [fx.Float32(v[i]) for i in range(n)]


def st_f32(res, off, vals, cm: int = 0):
    """Store 1, 2 or 4 f32 at f32 offset ``off``."""
    if len(vals) == 1:
        bo.buffer_store(fx.Float32(vals[0]).ir_value(), res, fx.Int32(off), cache_modifier=cm)
    else:
        v = fx.Vector.from_elements([fx.Float32(x) for x in vals], fx.Float32)
        bo.buffer_store(v.ir_value(), res, fx.Int32(off), cache_modifier=cm)


def st_dw(res, dw_off, words, cm: int = 0):
    """Store 1, 2 or 4 i32 words at dword offset ``dw_off``."""
    if len(words) == 1:
        bo.buffer_store(fx.Int32(words[0]).ir_value(), res, fx.Int32(dw_off), cache_modifier=cm)
    else:
        v = fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32)
        bo.buffer_store(v.ir_value(), res, fx.Int32(dw_off), cache_modifier=cm)


# ---------------------------------------------------------------- values


def bf16_pair(dw):
    """The two bf16 halves of a dword as f32 (low element first)."""
    dw = fx.Uint32(dw)
    return (dw << 16).bitcast(fx.Float32), ((dw >> 16) << 16).bitcast(fx.Float32)


def bf16_unpack(dws):
    out = []
    for dw in dws:
        out += list(bf16_pair(dw))
    return out


def bf16_bits(x):
    return fx.Uint32(fx.Float32(x).to(fx.BFloat16).bitcast(fx.Uint16))


def bf16_pack(lo, hi):
    return bf16_bits(lo) | (bf16_bits(hi) << 16)


def bf16r(x):
    """``x`` rounded to bf16, as f32."""
    return (bf16_bits(x) << 16).bitcast(fx.Float32)


def bf16_at(dw, odd):
    """Element ``odd`` (0 / 1, dynamic) of a bf16 pair dword, as f32."""
    dw = fx.Uint32(dw)
    return ((dw >> (odd * 16)) << 16).bitcast(fx.Float32)


def fp4x8(dw, scale):
    """v_cvt_scale_pk8_f32_fp4: 8 E2M1 codes (low nibble first) times 2^(this lane's
    E8M0 code ``scale`` - 127); bits above byte 0 of ``scale`` are ignored. Every
    lane of the wave must call it: the instruction reads its scale operand in the
    wave32 MX layout (lanes 16-31 use byte 1 of lane - 16's operand), so each lane
    passes its own code in byte 0 and its partner's (lane ^ 16) in byte 1."""
    from flydsl.expr.rocdl import cvt_scale_pk8_f32_fp4

    own = fx.Uint32(scale) & 0xFF
    op = own | (own.shuffle_xor(16, WAVE) << 8)
    v = fx.Vector(
        cvt_scale_pk8_f32_fp4(T.vec(8, T.f32), fx.Int32(dw).ir_value(), fx.Int32(op).ir_value(), 0)
    )
    return [v[n] for n in range(8)]


def dot(a, b, acc=None):
    """sum a[n] b[n] (+ acc) as a chain of fmas. Plain mul / add chains get
    reassociated into trees (fast math), which keeps every product of a long dot
    live at once: 256 values a lane in a T = 4 row dot, and spills."""
    start = 0
    if acc is None:
        acc = a[0] * b[0]
        start = 1
    for n in range(start, len(a)):
        acc = fmath.fma(a[n], b[n], acc)
    return acc


def wave_sum(v):
    for off in (16, 8, 4, 2, 1):
        v = v + v.shuffle_xor(off, WAVE)
    return v


def wave_max(v):
    for off in (16, 8, 4, 2, 1):
        v = fx.max(v, v.shuffle_xor(off, WAVE))
    return v


def lane_val(v, lane_idx: int):
    """f32 ``v`` of lane ``lane_idx`` (v_readlane: wave-uniform)."""
    return fx.Int32(rocdl.readlane(T.i32, fx.Float32(v).bitcast(fx.Int32).ir_value(), lane_idx)).bitcast(fx.Float32)


def col8_sum(v, lane):
    """Wave sums of 8 per-lane values v[0..7] at once (a butterfly that halves the
    values a lane carries each stage: 9 shuffles, not 8 x 5): lane l ends with the sum
    of column (l >> 2) & 7. ``lane_val(result, 4 c)`` reads column c."""
    w = []
    b = (lane >> 4) & 1
    for i in range(4):
        keep = (b == 1).select(v[i + 4], v[i])
        send = (b == 1).select(v[i], v[i + 4])
        w.append(keep + send.shuffle_xor(16, WAVE))
    x = []
    b = (lane >> 3) & 1
    for i in range(2):
        keep = (b == 1).select(w[i + 2], w[i])
        send = (b == 1).select(w[i], w[i + 2])
        x.append(keep + send.shuffle_xor(8, WAVE))
    b = (lane >> 2) & 1
    keep = (b == 1).select(x[1], x[0])
    send = (b == 1).select(x[0], x[1])
    y = keep + send.shuffle_xor(4, WAVE)
    y = y + y.shuffle_xor(2, WAVE)
    y = y + y.shuffle_xor(1, WAVE)
    return y


def silu(x):
    return x / (fx.Float32(1.0) + fmath.exp(-x))


def sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(-x))


def softplus(x):
    """log(1 + exp(x)), x itself past 20 (as torch / the GDN gating kernel)."""
    return (x > 20.0).select(x, fmath.log(fx.Float32(1.0) + fmath.exp(x)))


# ---------------------------------------------------------------- LDS


def lds_addr(ptr):
    """An LDS pointer's 32-bit byte address."""
    return fx.Int32(ptrtoint(ptr))


def _lds_ptr(base, f32_index, align):
    ty = fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ty, fx.Int32(base) + fx.Int32(f32_index) * 4)


def lds_ld4(base, i):
    """4 f32 at LDS f32 index ``i`` (a multiple of 4): one 16-byte load."""
    v = fx.Vector(fx.ptr_load(_lds_ptr(base, i, 16), result_type=fx.Vector.make_type(4, fx.Float32)))
    return [v[n] for n in range(4)]


def lds_ld8(base, i):
    return lds_ld4(base, i) + lds_ld4(base, i + 4)


def lds_ld4w(base, i):
    """4 raw dwords at LDS dword index ``i`` (a multiple of 4): one 16-byte load."""
    v = fx.Vector(fx.ptr_load(_lds_ptr(base, i, 16), result_type=fx.Vector.make_type(4, fx.Int32)))
    return [v[n] for n in range(4)]


def lds_ld8w(base, i):
    """8 raw dwords at LDS dword index ``i`` (a multiple of 4): two 16-byte loads."""
    out = []
    for h in range(2):
        v = fx.Vector(fx.ptr_load(_lds_ptr(base, i + 4 * h, 16), result_type=fx.Vector.make_type(4, fx.Int32)))
        out += [v[n] for n in range(4)]
    return out


def wmma_bf16(a, b, c):
    """v_wmma_f32_16x16x32_bf16 (wave32): ``a`` / ``b`` this lane's 8 dwords (16 bf16) of
    A [16 x 32] / B [32 x 16], ``c`` its 8 f32 of C; returns D's 8 f32. Lane l: A row /
    B column l % 16, k = 16 (l // 16) + 0..15 (any K order shared by A and B works);
    C / D column l % 16, rows 8 (l // 16) + 0..7 (probe_wmma_bf16.py)."""
    from aiter.ops.flydsl.kernels.tensor_shim import _to_raw

    av = fx.Vector.from_elements([fx.Int32(w) for w in a], fx.Int32).bitcast(fx.BFloat16)
    bv = fx.Vector.from_elements([fx.Int32(w) for w in b], fx.Int32).bitcast(fx.BFloat16)
    cv = fx.Vector.from_elements([fx.Float32(v) for v in c], fx.Float32)
    d = fx.Vector(rocdl.wmma_f32_16x16x32_bf16(fx.Vector.make_type(8, fx.Float32), _to_raw(av), _to_raw(bv), _to_raw(cv),
                                               reuseA=False, reuseB=False).result)
    return [d[n] for n in range(8)]


def wmma_fp4_fp8(a, b, sa, sb, c):
    """v_wmma_scale_f32_16x16x128_f8f6f4, A FP4 (e2m1) 16 x 128, B FP8 (e4m3) 128 x 16,
    e8m0 scales a 32-K block (probe_wmma_fp4.py, discover_fp4.py). Lane l, h = l // 16:
    ``a`` 8 dwords, row l % 16: K block h (16 B, natural order) then block h + 2;
    ``b`` 16 dwords, column l % 16: for block j = 0..3, k = 32 j + 16 h + 0..15;
    ``sa`` / ``sb`` the row's / column's 4 block scales (byte j = block j); ``c`` / the
    result: column l % 16, rows 8 h + 0..7."""
    from aiter.ops.flydsl.kernels.tensor_shim import _to_raw
    from flydsl._mlir import ir

    av = fx.Vector.from_elements([fx.Int32(w) for w in a], fx.Int32)
    bv = fx.Vector.from_elements([fx.Int32(w) for w in b], fx.Int32)
    cv = fx.Vector.from_elements([fx.Float32(v) for v in c], fx.Float32)
    d = fx.Vector(rocdl.wmma_scale_f32_16x16x128_f8f6f4(
        fx.Vector.make_type(8, fx.Float32), _to_raw(av), _to_raw(bv), _to_raw(cv), _to_raw(fx.Int32(sa)), _to_raw(fx.Int32(sb)),
        fmtA=ir.Attribute.parse("#rocdl<matrix_format fp4_e2m1>"), fmtB=ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>"),
        scaleAType=0, scaleBType=0))
    return [d[n] for n in range(8)]


def mx8_quant(vals, lane):
    """MXFP8 (e4m3, an e8m0 scale a 32 values) of a wave's 128 values, 4 a lane (lane l:
    values 4 l .. 4 l + 3, block l // 8): (this lane's 4 fp8 as a dword, the 4 block
    scales as a dword, byte j = block j). Scale 2^(floor(log2 amax) - 7): the block's
    largest value lands in [128, 256), inside e4m3's range."""
    am = fx.max(fx.max(abs_f(vals[0]), abs_f(vals[1])), fx.max(abs_f(vals[2]), abs_f(vals[3])))
    for o in (1, 2, 4):
        am = fx.max(am, am.shuffle_xor(o, 32))
    e = fx.min(fx.max((am.bitcast(fx.Int32) >> 23) & 0xFF, fx.Int32(7)), fx.Int32(254))
    mult = ((261 - e) << 23).bitcast(fx.Float32)  # 2^(134 - e)
    q = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(vals[0] * mult).ir_value(), fx.Float32(vals[1] * mult).ir_value(), fx.Int32(0).ir_value(), False))
    q = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(vals[2] * mult).ir_value(), fx.Float32(vals[3] * mult).ir_value(), q.ir_value(), True))
    sb = e - 7  # e8m0 of 2^(e - 134)
    jb = lane // 8
    sdw = sb << (8 * jb)
    for d in (8, 16, 24):
        sdw = sdw | (sb.shuffle_xor(d, 32) << (8 * (jb ^ (d // 8))))
    return q, sdw


def abs_f(x):
    return (fx.Float32(x).bitcast(fx.Int32) & 0x7FFFFFFF).bitcast(fx.Float32)


def lds_st4(base, i, vals):
    fx.ptr_store(fx.Vector.from_elements([fx.Float32(v) for v in vals], fx.Float32), _lds_ptr(base, i, 16))


def lds_ld(base, i):
    return fx.Float32(fx.ptr_load(_lds_ptr(base, i, 4)))


def lds_st(base, i, v):
    fx.ptr_store(fx.Float32(v), _lds_ptr(base, i, 4))


def lds_ldi(base, i):
    return fx.Float32(fx.ptr_load(_lds_ptr(base, i, 4))).bitcast(fx.Int32)


def lds_sti(base, i, v):
    fx.ptr_store(fx.Int32(v).bitcast(fx.Float32), _lds_ptr(base, i, 4))


# ---------------------------------------------------------------- sync


def lds_atomic_or(base, i, v):
    """Atomic OR of i32 ``v`` into LDS dword ``i`` (order-independent)."""
    ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
    p = fx.inttoptr(ty, fx.Int32(base) + fx.Int32(i) * 4)
    fx.atomic_or(p, fx.Int32(v), syncscope=WORKGROUP)


def popc(x):
    """Population count of i32 ``x``."""
    return fx.Int32(fmath.ctpop(fx.Int32(x)))


def lds_atomic_add(base, i, v):
    """Atomic add of i32 ``v`` to LDS dword ``i``: the previous value."""
    ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
    p = fx.inttoptr(ty, fx.Int32(base) + fx.Int32(i) * 4)
    return fx.Int32(fx.atomic_add(p, fx.Int32(v), syncscope=WORKGROUP))


def lds_barrier():
    """Workgroup barrier ordering LDS only: wait for this wave's LDS ops, then the
    split barrier. ``gpu.barrier`` carries a memory fence, which makes LLVM wait for
    every outstanding global load first (s_wait_loadcnt 0): it would drain every
    early-issued weight / state load at the first barrier."""
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def counter_base(counter, arrivals_per_launch: int):
    """This launch's base on a monotonic grid counter (thread-uniform)."""
    v = fx.Int32(fx.generic_load(counter, memory_order=fx.AtomicOrdering.Monotonic, syncscope=AGENT))
    return (v // arrivals_per_launch) * arrivals_per_launch


@ASTRewriter.transform
def grid_barrier(counter, target):
    """Wait until ``target`` CTAs have arrived on ``counter`` in total."""
    rocdl.s_wait_storecnt(0)  # this wave's agent-scope stores have landed
    gpu.barrier()
    if fx.thread_idx.x == 0:
        fx.atomic_add(counter, fx.Int32(1), syncscope=AGENT)
        v = fx.Int32(fx.generic_load(counter, memory_order=fx.AtomicOrdering.Monotonic, syncscope=AGENT))
        while v < target:
            rocdl.s_sleep(1)
            v = fx.Int32(fx.generic_load(counter, memory_order=fx.AtomicOrdering.Monotonic, syncscope=AGENT))
        fx.memory_fence(syncscope=WORKGROUP, ordering=fx.AtomicOrdering.Acquire)
    gpu.barrier()


@ASTRewriter.transform
def ll_poll(res, pair_off, tag):
    """Poll the LL pair at pair index ``pair_off`` of ``res`` until it carries ``tag``;
    the value word, as f32."""
    got = fx.Vector(bo.buffer_load(res, fx.Int32(pair_off * 2), vec_width=2, dtype=T.i32, cache_modifier=CM_SYS))
    while fx.Int32(got[1]) != tag:
        rocdl.s_sleep(1)
        got = fx.Vector(bo.buffer_load(res, fx.Int32(pair_off * 2), vec_width=2, dtype=T.i32, cache_modifier=CM_SYS))
    return fx.Int32(got[0]).bitcast(fx.Float32)


def ll_put(res, pair_off, value, tag):
    """Store the LL pair {value, tag} at pair index ``pair_off`` of ``res`` (system scope)."""
    st_dw(res, pair_off * 2, [fx.Float32(value).bitcast(fx.Int32), tag], cm=CM_SYS)


# ---------------------------------------------------------------- tagged mailbox (one rank)
# A hand-off value is an LL pair {value, tag} in rank-local scratch, stored and
# loaded at agent scope; a consumer polls exactly the pairs it reads. Offsets are
# in pairs; a 16-byte line holds two pairs (even pair index).


@ASTRewriter.transform
def spin(load_all, pending):
    """``load_all()`` until ``pending(v)`` clears; every poll reissues every load, so a
    batch of lines costs one memory latency, not one a line. No s_sleep between
    polls: on gfx1250 ``s_sleep 1`` parks the wave ~1.9 us (probe_pingpong.py: GPU to
    GPU one-way 2.13 us with it, 0.22 us without), which a missed first poll pays.
    The fence keeps the loads in the loop (``s_sleep`` did, as a side effect).

    Bounded: after SPIN_LIMIT polls (seconds) the wave traps. A hand-off that never
    comes (a peer rank gone, a launch-order mismatch) then faults the process instead
    of leaving a persistent kernel spinning on the GPU, which the driver cannot tear
    down."""
    v = load_all()
    n = fx.Int32(0)
    while pending(v) & (n < SPIN_LIMIT):
        compiler_fence()
        v = load_all()
        n = n + 1
    if n >= SPIN_LIMIT:
        _trap()
    return v


SPIN_LIMIT = 1 << 24


def _trap():
    from flydsl._mlir.dialects import llvm as _llvm

    _llvm.call_intrinsic(None, "llvm.trap", [], [], [])


def put2(res, pair, v0, v1, tag):
    """Pairs ``pair`` (even) and ``pair + 1``: one 16-byte store."""
    st_dw(res, pair * 2, [fx.Float32(v0).bitcast(fx.Int32), tag, fx.Float32(v1).bitcast(fx.Int32), tag], cm=CM_DEV)


def put1(res, pair, v, tag):
    st_dw(res, pair * 2, [fx.Float32(v).bitcast(fx.Int32), tag], cm=CM_DEV)


def poll_words_sys(res, pairs, tag):
    """The 2 value words of each 16-byte line at even pair indices ``pairs`` of a
    peer buffer (system scope), once every tag is ``tag``; raw i32 words."""

    def load_all():
        els = []
        for p in pairs:
            v = fx.Vector(bo.buffer_load(res, fx.Int32(p * 2), vec_width=4, dtype=T.i32, cache_modifier=CM_SYS))
            els += [v[i] for i in range(4)]
        return fx.Vector.from_elements(els, fx.Int32)

    def pending(v):
        bad = fx.Int32(v[1]) != tag
        for i in range(1, 2 * len(pairs)):
            bad = bad | (fx.Int32(v[2 * i + 1]) != tag)
        return bad

    v = spin(load_all, pending)
    return [fx.Int32(v[2 * i]) for i in range(2 * len(pairs))]


def _poll(res, pairs, tag, width):
    """Values of the lines at ``pairs`` (``width`` pairs a line), once every tag is
    ``tag``."""

    def load_all():
        els = []
        for p in pairs:
            v = fx.Vector(bo.buffer_load(res, fx.Int32(p * 2), vec_width=2 * width, dtype=T.i32, cache_modifier=CM_DEV))
            els += [v[i] for i in range(2 * width)]
        return fx.Vector.from_elements(els, fx.Int32)

    def pending(v):
        bad = fx.Int32(v[1]) != tag
        for i in range(1, len(pairs) * width):
            bad = bad | (fx.Int32(v[2 * i + 1]) != tag)
        return bad

    v = spin(load_all, pending)
    return [fx.Int32(v[2 * i]).bitcast(fx.Float32) for i in range(len(pairs) * width)]


def poll2(res, pairs, tag):
    """The 2 values of each 16-byte line at even pair indices ``pairs`` (in order)."""
    return _poll(res, pairs, tag, 2)


def poll1(res, pairs, tag):
    """The value of each pair at ``pairs``."""
    return _poll(res, pairs, tag, 1)
