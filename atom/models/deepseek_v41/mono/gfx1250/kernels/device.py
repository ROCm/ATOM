# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""Low-level helpers of the step kernel: raw 64-bit global addresses and LDS byte offsets, scoped loads / stores /
atomics, gfx1250 intrinsics (WMMA, FP8 conversions, clocks), and the bounded polls (wait_count, handoff) with their
timeout counter.

All functions are traced inside a @flyc.kernel body.
"""
from __future__ import annotations

import re as _re
import threading
from flydsl.expr import gpu, range_constexpr

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.compiler.ast_rewriter import ASTRewriter as _ASTR
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T, as_ir_value

# ---------------------------------------------------------------- types
def i32():
    return ir.IntegerType.get_signless(32)


def i64():
    return ir.IntegerType.get_signless(64)


def vty(n, elem):
    return ir.VectorType.get([n], elem)


def gptr_ty():
    return ir.Type.parse("!llvm.ptr<1>")


def lptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _v(x):
    return as_ir_value(x) if not isinstance(x, ir.Value) else x


# ---------------------------------------------------------------- addresses
def gptr(addr):
    """Global pointer from a 64-bit byte address (fx.Int64)."""
    return llvm.IntToPtrOp(gptr_ty(), _v(fx.Int64(addr))).result


def lds_base():
    """Byte offset of the kernel's dynamic LDS (fx.Int32)."""
    p = fx.get_dyn_shared()
    return fx.Int32(fx.Int64(fx.ptrtoint(p)))


def lptr(off):
    """LDS pointer from a byte offset (fx.Int32)."""
    return llvm.IntToPtrOp(lptr_ty(), _v(fx.Int32(off))).result


# ---------------------------------------------------------------- loads / stores
_ORD = {None: None, "relaxed": llvm.AtomicOrdering.monotonic}


def _wrap(res, kind, n):
    if n == 1:
        return kind(res)
    return fx.Vector(res, (n,), kind)


def gload(addr, kind=fx.Int32, n=1, nt=False, scope=None, align=None):
    """Load n elements of `kind` at global byte address `addr`. scope='agent' -> relaxed device-scope atomic load."""
    ty = kind.ir_type if n == 1 else vty(n, kind.ir_type)
    kw = dict(alignment=align or (kind.width // 8) * n)
    if nt:
        kw["nontemporal"] = True
    if scope == "system":
        kw["ordering"] = llvm.AtomicOrdering.monotonic
    elif scope:
        kw["ordering"] = llvm.AtomicOrdering.monotonic
        kw["syncscope"] = scope
    return _wrap(llvm.LoadOp(ty, gptr(addr), **kw).result, kind, n)


_EB = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}


def tbytes(t):
    """Size in bytes of a scalar or vector<N x elem> MLIR type."""
    s = str(t)
    m = _re.fullmatch(r"vector<(\d+)x(\w+)>", s)
    return int(m[1]) * _EB[m[2]] if m else _EB[s]


def gstore(addr, val, nt=False, scope=None, align=None):
    v = _v(val)
    kw = {"alignment": align or tbytes(v.type)}
    if nt:
        kw["nontemporal"] = True
    if scope == "system":   # (v3) relaxed system-scope atomic store (SCOPE_SYS: past every non-coherent cache level)
        kw["ordering"] = llvm.AtomicOrdering.monotonic
    elif scope:
        kw["ordering"] = llvm.AtomicOrdering.monotonic
        kw["syncscope"] = scope
    llvm.StoreOp(v, gptr(addr), **kw)


def lload(off, kind=fx.Int32, n=1, align=None):
    ty = kind.ir_type if n == 1 else vty(n, kind.ir_type)
    return _wrap(llvm.LoadOp(ty, lptr(off), alignment=align or (kind.width // 8) * n).result, kind, n)


def lstore(off, val, align=None):
    v = _v(val)
    llvm.StoreOp(v, lptr(off), alignment=align or tbytes(v.type))


# ---------------------------------------------------------------- atomics / fences / sync
def atomic_add(addr, val, scope="agent"):
    return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, gptr(addr), _v(fx.Int32(val)),
                                     llvm.AtomicOrdering.monotonic, syncscope=scope, alignment=4).result)


def atomic_fadd(addr, val, scope="agent"):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, gptr(addr), _v(fx.Float32(val)), llvm.AtomicOrdering.monotonic,
                     syncscope=scope, alignment=4)


def latomic_add_f32(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, lptr(off), _v(fx.Float32(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=4)


def latomic_max_u32(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.umax, lptr(off), _v(fx.Int32(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=4)


def fence_release(scope):
    llvm.fence(llvm.AtomicOrdering.release, syncscope=scope)


def fence_acquire(scope):
    llvm.fence(llvm.AtomicOrdering.acquire, syncscope=scope)


def intrin(res_ty, name, args):
    return llvm.call_intrinsic(res_ty, name, [_v(a) for a in args], [], [])


def s_sleep(n=1):
    rocdl.s_sleep(n)


def wait_storecnt0():
    llvm.inline_asm(None, [], "s_wait_storecnt 0x0", "", has_side_effects=True)


def cluster_barrier():
    """__builtin_amdgcn_s_cluster_barrier()."""
    intrin(None, "llvm.amdgcn.s.cluster.barrier", [])


def clock():
    """__builtin_readsteadycounter() (100 MHz)."""
    return fx.Int64(intrin(i64(), "llvm.readsteadycounter", []))


def prefetch(addr):
    """__builtin_amdgcn_global_prefetch(p, 0)."""
    intrin(None, "llvm.amdgcn.global.prefetch", [gptr(addr), fx.Int32(0)])


def getpc():
    return fx.Int64(llvm.inline_asm(i64(), [], "s_getpc_b64 $0", "=s", has_side_effects=True))


@_ASTR.transform
def code_prefetch(t, b, endp, kb):
    """Warm L2 with this kernel's code from here up to the end PC its previous run recorded at endp (code_end; 0 on the
    first run: nothing), at most `kb` KB: one global_prefetch of a 128-byte line a thread, CTA-major. A launch's first
    pass over its layer loop otherwise fetches the code from HBM (the step's weight stream evicts it); an unbounded
    prefetch past the code object faults."""
    pc = getpc()
    end = gload(endp, fx.Int64)
    nl = fx.min(fx.max((end - pc) // 128, fx.Int64(0)), fx.Int64(kb * 8))
    i = fx.Int64(b) * 256 + fx.Int64(t)
    if i < nl:
        prefetch(pc + i * 128)


@_ASTR.transform
def code_end(t, b, endp):
    """Record this kernel's PC near its end for the next run's code_prefetch."""
    if (b == 0) & (t == 0):
        gstore(endp, getpc())


def async_to_lds_b128(gaddr, loff, cpol=0):
    """__builtin_amdgcn_global_load_async_to_lds_b128(g, l, 0, cpol)."""
    intrin(None, "llvm.amdgcn.global.load.async.to.lds.b128", [gptr(gaddr), lptr(loff), fx.Int32(0), fx.Int32(cpol)])


def wait_asynccnt0():
    rocdl.s_wait_asynccnt(0)


def opaque_v(x):
    """asm volatile("" : "+v"(x)) on an i32."""
    return fx.Int32(llvm.inline_asm(i32(), [_v(fx.Int32(x))], "", "=v,0", has_side_effects=True))


def opaque_s(x):
    """asm volatile("" : "+s"(x)) on an i32."""
    return fx.Int32(llvm.inline_asm(i32(), [_v(fx.Int32(x))], "", "=s,0", has_side_effects=True))


# ---------------------------------------------------------------- matrix
def wmma_mx(a, sa, b, sb, c):
    """__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0, a, 0, b, 0, c, 0, 0, sa, 0, 0, sb, false, false):
    a, b vector<16xi32> (fp8 e4m3), c vector<8xf32>, sa/sb i32 E8M0 x4."""
    r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(vty(8, T.f32), _v(a), _v(b), _v(c), _v(fx.Int32(sa)), _v(fx.Int32(sb)),
                                              fmtA=0, fmtB=0)
    return fx.Vector(r, (8,), fx.Float32)


def wmma_bf16(a, b, c):
    """__builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, 0, c, false, false): a, b vector<16xbf16>."""
    r = rocdl.wmma_f32_16x16x32_bf16_(vty(8, T.f32), _v(a), _v(b), _v(c))
    return fx.Vector(r, (8,), fx.Float32)


def ds_load_tr16(loff):
    """__builtin_amdgcn_ds_load_tr16_b128_v8bf16(lds ptr) -> vector<8xbf16>."""
    r = intrin(vty(8, T.bf16), "llvm.amdgcn.ds.load.tr16.b128.v8bf16", [lptr(loff)])
    return fx.Vector(r, (8,), fx.BFloat16)


def cvt_pk_fp8(a, b=0.0):
    """__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false) & 0xff -> e4m3 byte in an i32."""
    r = rocdl.cvt_pk_fp8_f32(i32(), _v(fx.Float32(a)), _v(fx.Float32(b)), _v(fx.Int32(0)), False)
    return fx.Int32(r)


def stx_dev(addr, val):
    """Exchange store at device scope (scope:SCOPE_DEV): relaxed agent-scope atomic stores of <= 8 bytes each, so a
    fence-free handoff (every wave s_wait_storecnt 0 + relaxed counter) publishes them (HDEV)."""
    nb = tbytes(_v(val).type)
    if nb == 16:
        v = val.bitcast(fx.Int64)
        gstore(addr, v[0], scope="agent", align=8)
        gstore(addr + fx.Int64(8), v[1], scope="agent", align=8)
    elif nb == 8 and isinstance(val, fx.Vector):
        gstore(addr, val.bitcast(fx.Int64)[0], scope="agent", align=8)
    elif nb == 2 and not isinstance(val, fx.Vector):
        gstore(addr, val.bitcast(fx.Int16), scope="agent", align=2)
    else:
        gstore(addr, val, scope="agent", align=nb)


SLOAD_INV = __import__("os").environ.get("SLOAD_INV", "0") == "1"


def sload64(addr):
    """i64 load through the constant address space (addrspace 4) at a wave-uniform address -> s_load_b64 (scalar cache;
    for read-only tables such as pointer tables)."""
    p = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<4>"), _v(fx.Int64(addr))).result
    return fx.Int64(llvm.LoadOp(fx.Int64.ir_type, p, alignment=8, invariant=SLOAD_INV).result)


def llvm_wait_loads():
    """s_wait_loadcnt 0 (diagnostic)."""
    llvm.inline_asm(None, [], "s_wait_loadcnt 0x0", "", has_side_effects=True)


def cvt_pk_fp8_w(old, a, b, hi):
    """__builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, hi): e4m3(a), e4m3(b) into the low (hi=False) / high 16 bits of old."""
    r = rocdl.cvt_pk_fp8_f32(i32(), _v(fx.Float32(a)), _v(fx.Float32(b)), _v(fx.Int32(old)), hi)
    return fx.Int32(r)


def vbitcast_i32x4(v):
    """8 x bf16 (16 bytes) -> 4 x i32 vector (llvm.bitcast)."""
    return _wrap(llvm.BitcastOp(vty(4, i32()), _v(v)).result, fx.Int32, 4)


def latomic_add_i32(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.add, lptr(off), _v(fx.Int32(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=4)


def lane_gather(v, src_lane):
    """Lane src_lane's value of v (ds_bpermute)."""
    return fx.Int32(rocdl.ds_bpermute(T.i32, _v(fx.Int32(src_lane) * 4), _v(fx.Int32(v))))


def fresh(v):
    """v (per-lane Int32) through an opaque move: computed at its use, not hoisted / CSE'd across stages."""
    return fx.Int32(llvm.inline_asm(i32(), [_v(fx.Int32(v))], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True))


def fp8x2_to_f32(word, hi):
    """e4m3 bytes 0,1 (hi=False) or 2,3 (hi=True) of an i32 -> two exact f32 (integer decode: the packed cvt's
    encoding is rejected on gfx1250)."""
    w = fx.Int32(word)
    out = []
    for k in (2, 3) if hi else (0, 1):
        b = (w >> (8 * k)) & 255
        e = (b >> 3) & 15
        m = b & 7
        sg = (b & 128) << 24
        normal = (sg | ((e + 120) << 23) | (m << 20)).bitcast(fx.Float32)
        sub = (m.to(fx.Float32) * fx.Float32(2.0 ** -9))
        sub = ((b & 128) != 0).select(fx.Float32(0.0) - sub, sub)
        out.append((e == 0).select(sub, normal))
    return out[0], out[1]


def unlikely(b):
    """llvm.expect(b, false): block placement moves the guarded code out of the hot path (I-cache)."""
    i1 = ir.IntegerType.get_signless(1)
    f = arith.ConstantOp(i1, ir.IntegerAttr.get(i1, 0)).result
    return fx.Boolean(llvm.call_intrinsic(i1, "llvm.expect.i1", [_v(b), f], [], []))


class TraceSlot:
    """One trace-time value private to the tracing thread (used as ``slot[0]``).

    The step kernel sets such slots while it traces (the poll-timeout counter pointer, a stamp hook) and clears them
    when its body ends. FlyDSL >= 0.5 traces every kernel in its own MLIRContext, and a launcher may trace several
    ranks' kernels in parallel threads; a module-level ``[None]`` list then hands one thread's IR value to another
    thread's trace ("operand type from a different MLIRContext"). A thread-local slot keeps each trace's values in
    its own context and behaves the same when tracing is serial.
    """

    def __init__(self):
        self._local = threading.local()

    def __getitem__(self, i):
        assert i == 0
        return getattr(self._local, "value", None)

    def __setitem__(self, i, value):
        assert i == 0
        self._local.value = value


# Poll timeouts made visible. Every bounded poll (cross-rank tag waits, device handoffs) gives up after 2 s and the
# kernel goes on with whatever it read; a timeout therefore means wrong results. TOFLAG[0] (set by the step kernel to its
# 'cyc' scratch) counts them: int32 [0] += 1, int32 [1] = the last site id. The runner reads it after a step.
TOFLAG = TraceSlot()
POLL_TICKS = 2000000000   # (v2 study) 20 s: the first launch's rank skew (module load) must not time out a poll (a timeout desyncs the TP receive halves for every later step)


DIAG = TraceSlot()   # (v3 diag) the stamp array: a timed-out poll's code into its CTA's slot 1


@_ASTR.transform
def note_timeout(t0, site):
    if fx.const_expr(TOFLAG[0] is not None):
        if clock() - t0 >= fx.Int64(POLL_TICKS):
            atomic_add(TOFLAG[0], 1)
            gstore(TOFLAG[0] + fx.Int64(4), fx.Int32(site))
            if fx.const_expr(DIAG[0] is not None):
                gstore(DIAG[0] + fx.Int64(opaque_s(fx.block_idx.x)) * fx.Int64(256) + fx.Int64(8), fx.Int64(site) | fx.Int64(1 << 40))



@_ASTR.transform
def wait_count(addr, n, site):
    """Spin until the agent-scope int32 counter at addr reaches n (bounded: POLL_TICKS, a timeout counted at site).
    addr: an address, or a function giving it (evaluated in the loop, as the polls were written before this helper)."""
    t0 = clock()
    while (gload(addr() if callable(addr) else addr, fx.Int32, scope="agent") < n) & (clock() - t0 < fx.Int64(POLL_TICKS)):
        s_sleep(1)
    note_timeout(t0, site)


@_ASTR.transform
def handoff(cnt, t, hdev, n=256, site=None):
    """Device-wide handoff of n CTAs on counter cnt (thread t). hdev: the stages' exchange stores are device-scope ->
    every wave waits its stores, no fences; else release / acquire fences around the count."""
    if fx.const_expr(hdev):
        wait_storecnt0()
        gpu.barrier()
        if t == 0:
            atomic_add(cnt, 1)
            wait_count(cnt, n, 4 if site is None else site)
        gpu.barrier()
    else:
        gpu.barrier()
        if t == 0:
            fence_release("agent")
            atomic_add(cnt, 1)
            wait_count(cnt, n, 5 if site is None else site)
            fence_acquire("agent")
        gpu.barrier()


# ==============================================================================================================
I32, I64 = fx.Int32, fx.Int64
ASTRewriter = _ASTR
# (merged from ll.py)
# LL (data + tag) hand-off primitives for the v2 layer (TileRT-style point-to-point dependencies, no counters).
# 
# An LL word is 8 bytes: a 4-byte payload and a 4-byte tag written by ONE device-scope 8-byte store, so a reader that sees
# the tag also sees the payload (no s_wait_storecnt / counter / fence). Two formats, as TileRT's tail kernel:
#   - 32-bit tag (bf16 x2 / f32 / i32 payloads): tag = the site's 32-bit epoch;
#   - FP8 packs: tag = epoch24 | (e8m0 scale << 24), so the block scale arrives with its data.
# Epoch of a site: the step epoch (`tpep`, +1 a step), the layer and a site id -> unique over a step's sites and changes every
# step, so a stale word from an earlier step never matches and the buffers are never zeroed (a 2^32 / 2^24 wrap is far beyond
# any run: 1024 layer-site slots a step).

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


LL_SCOPE = "system"   # (v3) the LL words' scope: agent-scope writers were not enough across XCDs (rare 20 s stalls; Codex saw the same on gfx950)
SITES = 16            # site ids a layer
LAYER_SLOTS = 64      # layers (target + draft) a step


def epoch32(tep, layer, site):
    """32-bit tag of (step epoch, layer, site)."""
    return (tep * (LAYER_SLOTS * SITES) + layer * SITES + site) | I32(-2147483648)   # bit 31 set: never equals a zeroed word


def epoch24(tep, layer, site):
    """24-bit epoch of an FP8 pack (low 24 bits of the tag)."""
    return (tep * (LAYER_SLOTS * SITES) + layer * SITES + site + 1) & I32(0xFFFFFF)


def _w64(lo, hi):
    return (I64(hi) << I64(32)) | (I64(lo) & I64(0xFFFFFFFF))


def put(addr, payload, tag):
    """One LL word: payload (i32 bits) + tag, a single relaxed device-scope 8-byte store."""
    gstore(addr, _w64(payload, tag), scope=LL_SCOPE, align=8)


def put2(addr, p0, p1, tag):
    """Two LL words (16 B): each half self-tagged, so the halves may land in any order."""
    put(addr, p0, tag)
    put(addr + I64(8), p1, tag)


def put_fp8(addr, word, scale_code, ep24):
    """An FP8 pack word (4 fp8 values) with its block's e8m0 scale in the tag's high byte."""
    put(addr, word, ep24 | (I32(scale_code) << 24))


def peek(addr):
    """(payload, tag) of one LL word, one relaxed device-scope 8-byte load."""
    w = gload(addr, I64, scope=LL_SCOPE, align=8)
    return I32(w), I32(w >> I64(32))


@ASTRewriter.transform
def get(addr, tag, site):
    """Spin until the LL word at addr carries tag (bounded: POLL_TICKS, a timeout counted at site); returns the payload."""
    t0 = clock()
    p, g = peek(addr)
    while (g != tag) & (clock() - t0 < I64(POLL_TICKS)):
        s_sleep(1)
        p, g = peek(addr)
    note_timeout(t0, I32(site * 65536) + (tag & I32(0x3FF)))   # (diag) site << 16 | the tag's layer / site bits
    return p


@ASTRewriter.transform
def get_fp8(addr, ep24, site):
    """Spin until the FP8 pack at addr carries epoch ep24 (low 24 tag bits); returns (word, e8m0 scale code)."""
    t0 = clock()
    p, g = peek(addr)
    while ((g & I32(0xFFFFFF)) != ep24) & (clock() - t0 < I64(POLL_TICKS)):
        s_sleep(1)
        p, g = peek(addr)
    note_timeout(t0, site)
    return p, (g >> 24) & I32(0xFF)


@ASTRewriter.transform
def wait_flags(base, n, tag, w, l, site):
    """(v3) Wave 0 waits for the n LL flags base + 8 i (lane l: flags l, l + 32, ..), every lane's peeks issued together
    and re-polled together (one round trip a poll, not one a flag); the caller's barrier follows."""
    if w == 0:
        K = (n + 31) // 32
        ad = [base + I64(fx.min(l + 32 * k, I32(n - 1))) * I64(8) for k in range(K)]
        t0 = clock()
        pg = [peek(ad[k]) for k in range(K)]
        bad = I32(0)
        for k in range_constexpr(K):
            bad = bad | (pg[k][1] != tag).select(I32(1), I32(0))
        while (bad != 0) & (clock() - t0 < I64(POLL_TICKS)):
            s_sleep(1)
            pg = [peek(ad[k]) for k in range(K)]
            bad = I32(0)
            for k in range_constexpr(K):
                bad = bad | (pg[k][1] != tag).select(I32(1), I32(0))
        note_timeout(t0, I32(site * 65536) + (tag & I32(0x3FF)))


@ASTRewriter.transform
def wait_flags2(base_a, n_a, na_max, base_b, n_b, tag, w, l, site):
    """(v3) wave 0 waits for flags base_a + 8 i (i < n_a, runtime, <= na_max) and base_b + 8 j (j < n_b), all peeks
    issued together (as wait_flags); the caller's barrier follows."""
    if w == 0:
        KA = (na_max + 31) // 32
        KB = (n_b + 31) // 32
        ia = [l + 32 * k for k in range(KA)]
        ad = [base_a + I64(fx.min(ia[k], fx.max(n_a - 1, I32(0)))) * I64(8) for k in range(KA)]
        ok = [ia[k] < n_a for k in range(KA)]
        ad = ad + [base_b + I64(fx.min(l + 32 * k, I32(n_b - 1))) * I64(8) for k in range(KB)]
        ok = ok + [l >= 0 for k in range(KB)]
        t0 = clock()
        pg = [peek(a) for a in ad]
        bad = I32(0)
        for k in range_constexpr(KA + KB):
            bad = bad | (ok[k] & (pg[k][1] != tag)).select(I32(1), I32(0))
        while (bad != 0) & (clock() - t0 < I64(POLL_TICKS)):
            s_sleep(1)
            pg = [peek(a) for a in ad]
            bad = I32(0)
            for k in range_constexpr(KA + KB):
                bad = bad | (ok[k] & (pg[k][1] != tag)).select(I32(1), I32(0))
        note_timeout(t0, I32(site * 65536) + (tag & I32(0x3FF)))
