# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::indexer::fused_kernel<1, 0, KR, ROWS>`` (KR 4027 / 2747, ROWS 6;
# KR 499 / 243, ROWS 1) and ``fused_engram_kernel<1, 0, KR, 6, 6>``.
"""DeepSeek-V4.1 decode indexer on gfx950, a FlyDSL port of TileRT's fused indexer (score + top-k in one
persistent launch). Kernarg ABI unchanged: ScoreArgs 0x00 (112 B), SelectArgs 0x70 (152 B), PrepArgs
0x108 (136 B) (+ engram GemvParams 0x190, 80 B), read as i64 words; grid 256 (engram 356) x 1024, LDS
104096 B (engram 120480 B).

PARTIAL PORT (runs/20261115-flydsl-tilert/m3/indexer_notes.md has the role map, the float details and
the 8-rank replay results). Ported and replayed byte-exact: the k GEMV and CTA71 prep roles, the
regular scorer, CTA71's new-key rescoring, the small-n / candidate-block scorer, the engram GEMV CTAs,
the epoch slots, and the small-path selector (CTA 248: the whole <1, 0, 499 / 243, 1> REINDEX launch is
byte-exact, its selection rule measured on the replays, not read from TileRT's radix code). NOT ported:
the big-n selector (s2 > 239: sample threshold, per-16K-chunk filter lists, merge, radix select, the
mode-1 candidate-block table), the big candidate-block scorer (cand and n > 16384), the small-n new-key
score LL (slots + 0x100700) and the split-by-rank cross-GPU path: the <.., 4027 / 2747, 6> and engram
launches (n > 16384) produce no top-k yet, so they must not replace TileRT's kernel.

Workspace (ws = ScoreArgs+0x20; M = ScoreArgs+0x30, the key capacity):
- slots = ws + C(M) (``slots_offset``): dword s2 = CTA s2's epoch (this launch's tag = slot + 1,
  stored back at the end); + 1536: the 4096 threshold samples LL; + 0x100600 CTA71's flag; + 0x100700
  the new keys' score LL; + 0x100900 the k LL [feature 128][row 8] {f32, epoch}.
- ws + 0x6a00 + 4 k: key k's f32 score; ws + ((max(M, 0x4000) 32 + 0x6ae0) & ~0xff) + 4 b: the
  8-key block b's max (mode 1, n > 16384).
Roles: k GEMV (s2 % 8 == 7, s2 < 64), CTA71 (the new keys: RMSNorm, RoPE, MXFP4 -> index_k cache),
scorers (16 keys a wave and tile: two FP4 MFMAs, sum_h w_h ReLU(q_h . k)), selectors (s2 > 239 for big n,
not ported; 248 for small n), engram GEMV (engram variant, s2 >= 256).
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.hi.kernels.gfx950.common import (
    _call,
    bf_rne_hi,
    bperm,
    f32c,
)
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.engram_fused import (
    _gemv as _eng_gemv,
)
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.layer_tail import (
    _ld,
    _lds_ld,
    _lds_st,
    _rsq_scaled,
)
from atom.mono.device.mx import FP4, mfma_scaled
from atom.mono.device.ops import (
    CM_DEV,
    CM_SYS,
    kernel_symbol,
    launcher,
    permlane_swap,
    rsrc,
    spin_pause,
    traced,
    wave_incl_scan,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

# variant: KR (TileRT's third template argument, a per-rank constant with no device-side use found in
# the ISA), ROWS, engram. Selected by TK_IX for the build helper ("4027", "2747", "499", "243",
# "e4027", "e2747"), or build(key).
VARIANTS = {
    "4027": (4027, 6, False),
    "2747": (2747, 6, False),
    "499": (499, 1, False),
    "243": (243, 1, False),
    "e4027": (4027, 6, True),
    "e2747": (2747, 6, True),
}
_KEY = os.environ.get("TK_IX", "4027")
KARG_WORDS = 60 if VARIANTS[_KEY][2] else 50
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
THREADS = 1024
LDS_BYTES = 104096
LDS_BYTES_ENG = 120480
ENG_LDS = (
    0x18E20  # the engram GEMV's [half 2][wave 8][lane 64] f32x4 partials, 8 KiB a half
)
VERSION = "v13"

# byte offsets from ``slots`` (ws + C(M))
FLAG_OFF = 0x100600  # CTA71's "new keys written" flag (dword = epoch)
NEWSC_OFF = 0x100700  # small-n path: the new keys' scores, LL {f32, epoch} a row
KLL_OFF = (
    0x100900  # the k GEMV -> CTA71 LL: [feature 128][row 8] {f32 (bf16 << 16), epoch}
)
SCORE_OFF = 0x6A00  # ws + this: the f32 score of key k at + 4 k
SAMPLE_OFF = 1536  # slots + this: the 4096 threshold samples, LL {f32 or -inf, epoch}
# LDS (scorer): the row's q (32 heads x 128 fp4, 64 B a head), its e8m0 (4 a head), head weights (f32)
Q_LDS, QS_LDS, QW_LDS = 16384, 18432, 18560
# LDS (selector, small path)
SEL_HIST = 0  # 256 radix bins
SEL_WSUM = 1024  # [wave 16] per-wave totals of the radix scans (1024, 1088, 1152)
SEL_PICK = 1280  # (bin, rest) of a radix pass
SEL_OUT = 0x18E20  # the 512 output indices (TileRT's LDS slot)
TOPK = 512
CNT_SMALL = (
    0x102914  # slots + this: the hit counter TileRT's small-path selector adds 1 to
)
# LDS (CTA71)
K_LDS = 0  # [row 6][feature 128] f32
KQ_LDS = 3072  # [row][64 B] the row's packed fp4 (16 dwords)
KE_LDS = 4160  # [row][4 dwords] the 4 blocks' e8m0 exponents


def _ld_sc1(addr, word, n=1):
    return _ld(addr, word, n, CM_DEV)


def _st(addr, word, v):
    bo.buffer_store(Int32(v), rsrc(addr), word, cache_modifier=CM_DEV)


def _st2(addr, word, a, b):
    bo.buffer_store(fx.Vector.from_elements([Int32(a), Int32(b)], Int32), rsrc(addr), word,
                    cache_modifier=CM_DEV)  # fmt: skip


def slots_offset(m):
    """C(M): the epoch-slot area's byte offset in the workspace (TileRT's layout formula, i64)."""
    big = m > Int64(0x4000)
    mc = big.select(m, Int64(0x4000))
    nb = (((m + 7) // 8) + 3) & Int64(
        -4
    )  # signed division (rounds toward 0) as the ISA's ashr fixup
    nch = (mc + 0x3FFF) >> 14
    nbc = (m > Int64(0)).select((nb + 0x3FFF) // 16384, Int64(1))
    a = (mc * 32 + 0x6AE0) & Int64(-256)
    b = (nb * 32 + nch * 32768 + a + nbc * 131072 + 0x80) & Int64(-256)
    mx = (nch > nbc).select(nch, nbc)
    return ((nbc + nch) * 262144 + mx * 0x140 + b + 0x1000C0) & Int64(-256)


@traced
def _wait_tag(addr, word, epoch):
    """One {payload, tag} LL entry polled until its tag is the epoch (sc1 reload, s_sleep 1)."""
    v = _ld_sc1(addr, word, 2)
    while v[1] != epoch:
        spin_pause()
        rocdl.s_sleep(1)
        v = _ld_sc1(addr, word, 2)
    return v[0]


@traced
def k_gemv(tid, s2, slots, latent, i_wk, epoch, rows):
    """CTA s2 % 8 == 7, s2 < 64: index-key feature o = 16 (s2 / 8) + wave for each new token."""
    lane, wv = tid % 64, tid // 64
    o = (s2 >> 3) * 16 + wv
    w = _ld(i_wk, (o * 1024 + lane * 16) // 4, 4)
    for r in range_constexpr(rows):
        x = _ld(latent, (r * 1024 + lane * 16) // 4, 4)
        acc = F32(0.0)
        for j in range_constexpr(4):
            xl, xh = (x[j] << 16).bitcast(F32), (x[j] & Int32(-65536)).bitcast(F32)
            wl, wh = (w[j] << 16).bitcast(F32), (w[j] & Int32(-65536)).bitcast(F32)
            acc = fmath.fma(xl, wl, acc)
            acc = fmath.fma(xh, wh, acc)
        for k in (32, 16, 8, 4, 2, 1):
            acc = acc + bperm(acc, lane ^ k)
        if lane == 0:
            _st2(
                slots,
                (KLL_OFF + o * 64 + r * 8) // 4,
                bf_rne_hi(acc) & Int32(-65536),
                epoch,
            )


def _fp4_code(v):
    """TileRT's e2m1 encode of a pre-scaled, [-6, 6]-clamped value: threshold counts + sign << 3."""
    a = abs(v)
    n = Int32(0)
    for th, ge in ((0x3E800000, False), (0x3F400000, True), (0x3FA00000, False), (0x3FE00000, True),
                   (0x40200000, False), (0x40600000, True), (0x40A00000, False)):  # fmt: skip
        c = (a >= f32c(th)) if ge else (a > f32c(th))
        n = n + c.select(Int32(1), Int32(0))
    return n | ((v.bitcast(Int32) >> 28) & 8)


@traced
def scorer_small(tid, lds, a, eslots, epoch, t0, kc, cand, n62, excl):
    """TileRT's small-n / candidate path (CTAs s2 % 8 == 0, waves 0..7, tiles t0 + 248 j, j < 5): slot i
    < kc (16 a tile) scores key i, or with ``cand`` key 8 blocks[i / 8] + i % 8 of the candidate-block
    table, clamped below n62; lanes 0..15 store it (not slot ``excl``): kc <= 4096 as the LL
    {score, epoch} at slots + 1536 + 8 i, else the score at ws + 0x6a00 + 4 i (both plain stores).
    """
    lane = tid & 63
    col = lane & 15
    kq = lane >> 4
    hb = Q_LDS + col * 64 + kq * 16
    qa0 = _lds_ld(lds, hb, 4)
    qa1 = _lds_ld(lds, hb + 1024, 4)
    sa0 = (_lds_ld(lds, QS_LDS + col * 4) >> (kq * 8)) & 0xFF
    sa1 = (_lds_ld(lds, QS_LDS + 64 + col * 4) >> (kq * 8)) & 0xFF
    w0 = _lds_ld(lds, QW_LDS + (tid & 48), 4)
    w1 = _lds_ld(lds, QW_LDS + 64 + (tid & 48), 4)
    ntile = (kc + 15) >> 4
    for j in range_constexpr(5):
        t = t0 + j * 0xF8
        if t < ntile:
            i = t * 16 + col
            ci = fx.min(kc - 1, i)
            key = ci
            if cand:
                key = (_ld(a[3], 2 + (ci >> 3)) << 3) | (ci & 7)
            key = fx.min(n62 - 1, key)
            kv = _ld(a[0], key * 16 + kq * 4, 4)
            sc = _ld(a[1], key)
            v = _score16(qa0, qa1, sa0, sa1, w0, w1, kv, (sc >> (kq * 8)) & 0xFF)
            if (kq == 0) & (i < kc) & (i != excl):
                if kc < 0x1001:
                    bo.buffer_store(fx.Vector.from_elements([v.bitcast(Int32), Int32(epoch)], Int32),
                                    rsrc(eslots), (SAMPLE_OFF + i * 8) // 4)  # fmt: skip
                else:
                    bo.buffer_store(v.bitcast(Int32), rsrc(a[4]), SCORE_OFF // 4 + i)


def _sortable(bits):
    """TileRT's order-preserving f32 -> u32 key: negative -> ~bits, else bits | sign."""
    return (bits < 0).select(~bits, bits | Int32(-2147483648))


@traced
def _block_excl(lds, base, v, lane, wv):
    """(exclusive prefix of v in thread order, CTA total), 16 waves, LDS words base.. (16)."""
    inc = wave_incl_scan(v, lane)
    if lane == 63:
        _lds_st(lds, base + wv * 4, [inc])
    gpu.barrier()
    off = Int32(0)
    tot = Int32(0)
    for w in range_constexpr(16):
        c = _lds_ld(lds, base + w * 4)
        off = off + (w < wv).select(c, Int32(0))
        tot = tot + c
    gpu.barrier()
    return off + inc - v, tot


@traced
def select_small(tid, lds, a, slots, epoch, kc, cand, n62, nwait, pub8):
    """The small / candidate path's selector (CTA 248; semantics measured, not TileRT's radix code): after
    CTAs 0..nwait-1 published their epoch slot, the top 512 of the kc slot scores (ws + 0x6a00; slots
    whose key, cand-mapped, is >= n62 left out) by the sortable key, ties by slot, in slot order ->
    LDS, -1 padded; tid < 512 writes topk[tid] and {idx, tag} to the LL targets (SelectArgs' 4, with
    ``pub8`` also PrepArgs' 4: KR 4027 / 499; 2747 / 243 publish 4), tid 0 the flag and the hit counter.
    """
    lane, wv = tid & 63, tid >> 6
    if (
        tid < 31
    ):  # TileRT: the small-path scorer CTAs 8 j (j < 31) published their slot (nwait unused)
        _wait_flag(slots, tid * 8, epoch)
    gpu.barrier()
    keys = []
    ids = []
    sv = [_ld_sc1(a[4], SCORE_OFF // 4 + tid * 16 + 4 * q, 4) for q in range(4)]
    llsrc = (
        kc < 0x1001
    )  # the scorer left kc <= 4096 slot scores as the LL at slots + 1536 (unverified path)
    for j in range_constexpr(16):
        i = tid * 16 + j
        sc = sv[j // 4][j % 4]
        if llsrc:
            sc = _ld_sc1(slots, (SAMPLE_OFF // 4) + 2 * fx.min(i, Int32(4095)))
        uk = i
        if cand:
            uk = (_ld(a[3], 2 + (fx.min(i, kc - 1) >> 3)) << 3) | (i & 7)
        ok = (i < kc) & (uk < n62)
        keys.append(ok.select(_sortable(sc), Int32(0)))
        ids.append(uk)
    tot = _top512(tid, lds, keys, ids, lane, wv)
    _publish(tid, lds, a, slots, epoch, tot, pub8, CNT_SMALL)


# ---- large-n selector (wide layout: one CTA a 16 K-key chunk, CTA 248 - nch + c handles chunk c) ----
LIST_OFF = 0x2806A00  # ws + this + 0x8000 c: chunk c's list {u32 key, i32 index} (M = 0x100000; TODO formula)
CNT_OFF = -0x2000  # slots + this + 8 c: chunk c's count LL {count, epoch}
HIST_LDS = 17152  # 4096 u32 threshold buckets
MM_LDS = 0  # [wave 16] (min, max) u32, 16 B a wave; scan scratch at + 256 / 512 / 768, offsets at + 1024
TB_LDS = 0x14300  # the threshold bucket
CNT_SEL = 0x102908  # slots + this: the hit counter the large-n merge adds 1 to (block level + 0x20)
CAND_LDS = 8960  # the block-level merge's 2048 output block indices


def _u32_lt(a, b):
    return (a ^ Int32(-2147483648)) < (b ^ Int32(-2147483648))


def _u32_ge(a, b):
    return _u32_lt(b, a) | (a == b)


def _umin(a, b):
    return _u32_lt(a, b).select(a, b)


def _umax(a, b):
    return _u32_lt(a, b).select(b, a)


def _lshr(x, s):
    """x >> s, logical (u32)."""
    return Int32((Int64(x) & Int64(0xFFFFFFFF)) >> Int64(s))


def _clz(x):
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.typing import as_ir_value

    return Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.ctlz.i32",
            [as_ir_value(Int32(x)), as_ir_value(fx.Boolean(False))],
            [],
            [],
        )
    )


def ws_layout(m):
    """TileRT's workspace offsets for key capacity M (i64; SelectArgs+0x98 / ScoreArgs+0x30):
    (a: 8-key block maxima, LB: chunk lists, LB2: block-chunk lists, Qm, C: slots). M = 1 M: 0x2006a00,
    0x2806a00, 0x3806a00, 0x3a07a00, 0x3a0ba00. The count LLs are at Qm + 128 mx (token chunks) and
    Qm + 192 mx (block chunks), mx = max(nch, nbc)."""
    mc = (m > Int64(0x4000)).select(m, Int64(0x4000))
    nb = (((m + 7) // 8) + 3) & Int64(-4)
    nch = (mc + 0x3FFF) >> 14
    nbc = (m > Int64(0)).select((nb + 0x3FFF) // 16384, Int64(1))
    mx = (nch > nbc).select(nch, nbc)
    a = (mc * 32 + 0x6AE0) & Int64(-256)
    b = (nb * 32 + a + 0x80 + nch * 32768 + nbc * 131072) & Int64(-256)
    lb = b + 0x100000
    lb2 = lb + (nch << 18)
    q = lb2 + (nbc << 18) + (mx << 6) + 0xFF
    return a, lb, lb2, q & Int64(-256), (q + (mx << 8)) & Int64(-256), mx


def _isqrt_f32(r):
    """trunc of TileRT's correctly-rounded f32 sqrt of the i32 r (v_sqrt_f32 + LLVM's fma fixup)."""
    x = r.to(F32)
    scl = x < f32c(0x0F800000)
    x = scl.select(x * f32c(0x4F800000), x)
    sq = F32(_call("llvm.amdgcn.sqrt.f32", T.f32, x))
    sm = (sq.bitcast(Int32) - 1).bitcast(F32)
    sp = (sq.bitcast(Int32) + 1).bitcast(F32)
    rm = fmath.fma(-sm, sq, x)
    rp = fmath.fma(-sp, sq, x)
    s1 = (rm <= F32(0.0)).select(sm, sq)
    s1 = (rp > F32(0.0)).select(sp, s1)
    s1 = scl.select(s1 * f32c(0x37800000), s1)
    s1 = ((x == F32(0.0)) | (x == f32c(0x7F800000))).select(x, s1)
    return s1.to(Int32)


@traced
def sel_samples(tid, eslots, epoch, nsamp):
    """This thread's 4 sample keys (samples 4 tid + j < nsamp at slots + 1536, each re-read until its tag
    is the epoch, no sleep; 0 = no sample; sortable(-inf) = 0x007fffff counts)."""
    keys = []
    for q in range_constexpr(4):
        i = tid * 4 + q
        kq = Int32(0)
        if i < nsamp:
            kq = _sortable(_wait_tag(eslots, (SAMPLE_OFF + i * 8) // 4, epoch))
        keys.append(kq)
    return keys


@traced
def sel_threshold(tid, lds, keys, n_smp, nsamp, kk, block=False):
    """The threshold key T (TileRT L21000..L21460): R = (N + nsamp kk - 1) / N; tgt = R + 2 isqrt(R) + 12;
    over the non-zero sample keys: min / max, shift = d > 0xfff ? 20 - clz(d) : 0 (d = max - min), 4096
    buckets of (key - min) >> shift, b = the highest bucket whose count from the top reaches tgt (0 when
    none), T = min + (b << shift); no sample at all: T = 0xffffffff."""
    lane, wv = tid & 63, tid >> 6
    mn = Int32(-1)
    mx = Int32(0)
    for q in range_constexpr(4):
        nz = keys[q] != 0
        mn = nz.select(_umin(mn, keys[q]), mn)
        mx = _umax(mx, keys[q])
    for o in (1, 2, 4, 8, 16, 32):
        mn = _umin(mn, Int32(bperm(mn.bitcast(F32), lane ^ o).bitcast(Int32)))
        mx = _umax(mx, Int32(bperm(mx.bitcast(F32), lane ^ o).bitcast(Int32)))
    if lane == 0:
        _lds_st(lds, MM_LDS + wv * 16, [mn, mx])
    _lds_st(lds, HIST_LDS + tid * 16, [Int32(0), Int32(0), Int32(0), Int32(0)])
    if tid == 0:
        _lds_st(lds, TB_LDS, [Int32(0)])
    gpu.barrier()
    for w in range_constexpr(16):
        mm = _lds_ld(lds, MM_LDS + w * 16, 2)
        mn = _umin(mn, mm[0])
        mx = _umax(mx, mm[1])
    none = _u32_lt(mx, mn)
    d = mx - mn
    sh = _u32_lt(Int32(0xFFF), d).select(Int32(20) - _clz(d), Int32(0))
    for q in range_constexpr(4):
        if keys[q] != 0:
            fx.llvm.atomic_add(
                lds + (HIST_LDS // 4 + _lshr(keys[q] - mn, sh)),
                Int32(1),
                syncscope="workgroup",
            )
    gpu.barrier()
    h = _lds_ld(lds, HIST_LDS + tid * 16, 4)
    hs = h[0] + h[1] + h[2] + h[3]
    pre, tot = _block_excl(lds, MM_LDS + 256, hs, lane, wv)
    above = tot - pre - hs  # samples in the buckets above this thread's 4
    num = Int64(n_smp) + Int64(nsamp) * Int64(kk) - 1  # block level: nsamp = 4096
    r = Int32(num // Int64(fx.max(n_smp, Int32(1))))
    tg = r + _isqrt_f32(r) * 5 + 16 if block else r + (_isqrt_f32(r) << 1) + 12
    c3 = above + h[3]
    c2 = c3 + h[2]
    c1 = c2 + h[1]
    c0 = c1 + h[0]
    a3 = _u32_lt(above, tg) & _u32_ge(c3, tg)
    a2 = _u32_lt(c3, tg) & _u32_ge(c2, tg)
    a1 = _u32_lt(c2, tg) & _u32_ge(c1, tg)
    a0 = _u32_lt(c1, tg) & _u32_ge(c0, tg)
    if a0 | a1 | a2 | a3:
        _lds_st(
            lds,
            TB_LDS,
            [
                tid * 4
                + a0.select(
                    Int32(0), a1.select(Int32(1), a2.select(Int32(2), Int32(3)))
                )
            ],
        )
    gpu.barrier()
    bb = _lds_ld(lds, TB_LDS)
    gpu.barrier()
    return none.select(Int32(-1), mn + (bb << sh))


@traced
def sel_chunk(tid, lds, ws, eslots, epoch, ck, n_smp, thr, sv, lo_n, hi_n, key_lo, lbase, cbase, cnt_ok,
              block=False, src=0, s94=0):  # fmt: skip
    """One chunk (TileRT L21639..L24953) of items 16384 ck + 16 tid + j. Token level: sv = the preloaded
    scores; a chunk overlapping the new keys [lo_n, hi_n) waits for CTA71's flag (tid 0, sleep 1) and the
    threads whose 16 items overlap re-read theirs. Block level (``block``): no preload; the flag wait
    (same window rule) comes before the load of the block maxima at ws + src; block s94 - key_lo is forced
    to 0xff800000. Item j passes when its key (sortable, 0 past n_smp) >= thr (u32); at most 4096
    entries {key, key_lo + item} (sc1) in thread / j order at ws + lbase + 0x8000 ck; the count LL
    {cnt, tag} (sc1) at ws + cbase + 8 ck after vmcnt(0) + barrier."""
    lane, wv = tid & 63, tid >> 6
    base = ck * 16384
    ln = fx.min(n_smp - base, Int32(0x4000))
    ovl = (hi_n > lo_n) & (lo_n < base + ln) & (hi_n > base)
    if ovl:
        if tid == 0:
            _wait_flag(eslots, FLAG_OFF // 4, epoch)
        gpu.barrier()
        if not block:  # noqa: SIM102 (kept as the validated build)
            if (lo_n < base + tid * 16 + 16) & (hi_n > (base | (tid * 16))):
                sv = [
                    _ld_sc1(ws, SCORE_OFF // 4 + base + tid * 16 + 4 * q, 4)
                    for q in range(4)
                ]
    if block:
        sv = [_ld_sc1(ws + src, base + tid * 16 + 4 * q, 4) for q in range(4)]
    keys = []
    sel = []
    cnt = Int32(0)
    for j in range_constexpr(16):
        kj = _sortable(sv[j // 4][j % 4])
        if block:
            kj = (base + tid * 16 + j + key_lo == s94).select(Int32(-8388608), kj)
        kj = (tid * 16 + j < ln).select(kj, Int32(0))
        sj = _u32_ge(kj, thr)
        keys.append(kj)
        sel.append(sj)
        cnt = cnt + sj.select(Int32(1), Int32(0))
    pos, tot = _block_excl(lds, MM_LDS + 512, cnt, lane, wv)
    lb = ws + lbase + Int64(ck) * 0x8000
    if tot <= 0x1000:
        for j in range_constexpr(16):
            if sel[j]:
                bo.buffer_store(fx.Vector.from_elements([keys[j], key_lo + base + tid * 16 + j], Int32), rsrc(lb),
                                pos * 2, cache_modifier=CM_DEV)  # fmt: skip
            pos = pos + sel[j].select(Int32(1), Int32(0))
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        _st2(ws + cbase, ck * 2, cnt_ok.select(tot, Int32(0)), epoch)
    return tot


@traced
def _top512(tid, lds, keys, ids, lane, wv, k=TOPK, out_lds=SEL_OUT):
    """The top k of the CTA's u32 keys (16 a thread, 0 = dead) by a 4 x 8-bit radix select, ties at the
    k-th key taken in thread / slot order (TileRT's rule); their ids in that order -> LDS out_lds. -> the count.
    """
    pre = Int32(0)
    msk = Int32(0)
    rest = Int32(k)
    for h in range_constexpr((k + 1023) // 1024):
        if tid + h * 1024 < k:
            _lds_st(lds, out_lds + (tid + h * 1024) * 4, [Int32(-1)])
    for p in range_constexpr(4):
        sh = 24 - 8 * p
        if tid < 256:
            _lds_st(lds, SEL_HIST + tid * 4, [Int32(0)])
        gpu.barrier()
        for j in range_constexpr(16):
            kv = keys[j]
            if (kv != 0) & ((kv & msk) == pre):
                fx.llvm.atomic_add(
                    lds + (SEL_HIST // 4 + ((kv >> sh) & 255)),
                    Int32(1),
                    syncscope="workgroup",
                )
        gpu.barrier()
        # the bin holding the rest-th largest: bins from the top in thread order (thread t: bin 255 - t), an
        # exclusive scan of their counts; the one thread with excl < rest <= excl + count names it
        hc = Int32(0)
        if tid < 256:
            hc = _lds_ld(lds, SEL_HIST + (255 - tid) * 4)
        if tid == 0:
            _lds_st(lds, SEL_PICK, [Int32(-1), rest])
        hx, _ = _block_excl(lds, SEL_WSUM + 128, hc, lane, wv)
        if (tid < 256) & (hx < rest) & (hx + hc >= rest):
            _lds_st(lds, SEL_PICK, [255 - tid, rest - hx])
        gpu.barrier()
        pk = _lds_ld(lds, SEL_PICK, 2)
        gpu.barrier()
        bsel = fx.max(pk[0], Int32(0))
        rest = (pk[0] < 0).select(rest, pk[1])
        pre = pre | (bsel << sh)
        msk = msk | Int32((255 << sh) - (1 << 32 if sh == 24 else 0))
        # fewer than 512 live keys: every live key is selected (the threshold stays 0 = dead)
        pre = (pk[0] < 0).select(Int32(0), pre)
    thr = pre
    neq = Int32(0)
    for j in range_constexpr(16):
        neq = neq + ((keys[j] == thr) & (thr != 0)).select(Int32(1), Int32(0))
    eqb, _ = _block_excl(lds, SEL_WSUM, neq, lane, wv)
    ngt = Int32(0)
    picks = []
    r = eqb
    for j in range_constexpr(16):
        kv = keys[j]
        gt = (kv != 0) & u32_gt(kv, thr)
        eq = (kv == thr) & (thr != 0)
        pick = gt | (eq & (r < rest))
        r = r + eq.select(Int32(1), Int32(0))
        picks.append(pick)
        ngt = ngt + pick.select(Int32(1), Int32(0))
    pos, tot = _block_excl(lds, SEL_WSUM + 64, ngt, lane, wv)
    for j in range_constexpr(16):
        if picks[j]:  # noqa: SIM102
            if pos < k:
                _lds_st(lds, out_lds + pos * 4, [ids[j]])
        pos = pos + picks[j].select(Int32(1), Int32(0))
    gpu.barrier()
    return tot


@traced
def sel_fullscan(tid, lds, src, n_items, kk, forced, out_lds):
    """The merge's fallback (TileRT's slow path when a chunk overflowed, total < kk or total > 8192; semantic
    port, the slow path's workspace spills are not reproduced): the exact top kk of all n_items items
    (key = sortable(src[i]) (sc1), item ``forced`` -> 0xff800000), ties to the smaller index, written in
    index order to out_lds. Thread t scans items [t per, (t + 1) per). -> the count."""
    lane, wv = tid & 63, tid >> 6
    per = (n_items + 1023) >> 10
    i0 = fx.min(tid * per, n_items)
    i1 = fx.min(i0 + per, n_items)
    pre = Int32(0)
    msk = Int32(0)
    rest = kk
    for h in range_constexpr(2):
        if tid + h * 1024 < kk:
            _lds_st(lds, out_lds + (tid + h * 1024) * 4, [Int32(-1)])
    for p in range_constexpr(4):
        sh = 24 - 8 * p
        if tid < 256:
            _lds_st(lds, SEL_HIST + tid * 4, [Int32(0)])
        gpu.barrier()
        i = i0
        while i < i1:
            kv = (i == forced).select(Int32(-8388608), _sortable(_ld_sc1(src, i)))
            if (kv != 0) & ((kv & msk) == pre):
                fx.llvm.atomic_add(
                    lds + (SEL_HIST // 4 + ((kv >> sh) & 255)),
                    Int32(1),
                    syncscope="workgroup",
                )
            i = i + 1
        gpu.barrier()
        hc = Int32(0)
        if tid < 256:
            hc = _lds_ld(lds, SEL_HIST + (255 - tid) * 4)
        if tid == 0:
            _lds_st(lds, SEL_PICK, [Int32(-1), rest])
        hx, _ = _block_excl(lds, SEL_WSUM + 128, hc, lane, wv)
        if (tid < 256) & (hx < rest) & (hx + hc >= rest):
            _lds_st(lds, SEL_PICK, [255 - tid, rest - hx])
        gpu.barrier()
        pk = _lds_ld(lds, SEL_PICK, 2)
        gpu.barrier()
        bsel = fx.max(pk[0], Int32(0))
        rest = (pk[0] < 0).select(rest, pk[1])
        pre = pre | (bsel << sh)
        msk = msk | Int32((255 << sh) - (1 << 32 if sh == 24 else 0))
        pre = (pk[0] < 0).select(Int32(0), pre)
    thr = pre
    neq = Int32(0)
    ngt = Int32(0)
    i = i0
    while i < i1:
        kv = (i == forced).select(Int32(-8388608), _sortable(_ld_sc1(src, i)))
        neq = neq + ((kv == thr) & (thr != 0)).select(Int32(1), Int32(0))
        ngt = ngt + ((kv != 0) & u32_gt(kv, thr)).select(Int32(1), Int32(0))
        i = i + 1
    eqb, _eqt = _block_excl(lds, SEL_WSUM, neq, lane, wv)
    take = fx.min(
        fx.max(rest - eqb, Int32(0)), neq
    )  # this thread's equal keys taken, in index order
    pos, tot = _block_excl(lds, SEL_WSUM + 64, ngt + take, lane, wv)
    r = eqb
    i = i0
    while i < i1:
        kv = (i == forced).select(Int32(-8388608), _sortable(_ld_sc1(src, i)))
        gt = (kv != 0) & u32_gt(kv, thr)
        eq = (kv == thr) & (thr != 0)
        pick = gt | (eq & (r < rest))
        if pick & (pos < kk):
            _lds_st(lds, out_lds + pos * 4, [i])
        pos = pos + pick.select(Int32(1), Int32(0))
        r = r + eq.select(Int32(1), Int32(0))
        i = i + 1
    gpu.barrier()
    return tot


@traced
def _publish(tid, lds, a, slots, epoch, tot, pub8, cnt_off):
    """LDS SEL_OUT[0..tot) (-1 past it) -> topk row (plain) and {idx, tag} to the LL targets (sc0 sc1);
    tid 0: the flag (sc1) and + 1 to the hit counter at slots + cnt_off."""
    if tid == 0:
        fx.llvm.atomic_add(_gword(slots + Int64(cnt_off)), Int32(1), syncscope="agent")
    if tid < TOPK:
        v = (tid < tot).select(_lds_ld(lds, SEL_OUT + tid * 4), Int32(-1))
        bo.buffer_store(v, rsrc(a[17]), tid)
        for t in (26, 27, 28, 29, 33, 34, 38, 39) if pub8 else (26, 27, 28, 29):
            bo.buffer_store(fx.Vector.from_elements([v, Int32(epoch)], Int32), rsrc(a[t]), tid * 2,
                            cache_modifier=CM_SYS)  # fmt: skip
    if tid == 0:
        _st(a[30], 0, epoch)


@traced
def sel_merge(
    tid,
    lds,
    a,
    ws,
    eslots,
    epoch,
    nch,
    pub8,
    lbase,
    cbase,
    hit_off,
    k,
    src,
    n_items,
    forced,
    id_off,
):
    """The merge (TileRT L24961..L58451; c = 0 token level k = 512, c = s83 block level k = 2048): wave 0
    polls the nch count LLs at ws + cbase. Normal path (no count > 4096, min(n_items, k) <= total <= 8192):
    the candidates gathered in chunk / list order (position p), the top k by key with ties to the smaller
    p, in p order (= ascending index), hit counter + 1 at slots + hit_off. Otherwise TileRT's slow path:
    here the exact top k of all n_items items rescanned from src (``sel_fullscan``; TileRT's own radix and
    its workspace spills are not reproduced), counter + 1 at slots + hit_off + 4. Token level: LDS
    0x18e20 -> topk / LL targets / flag; block level: temp:cand = {2, cnt, blocks.., -1 past cnt}.
    """
    lane, wv = tid & 63, tid >> 6
    cv = Int32(0)
    if tid < nch:
        cv = _wait_tag(ws + cbase, tid * 2, epoch)
    off, tot = _block_excl(lds, MM_LDS + 768, cv, lane, wv)
    _, novf = _block_excl(
        lds, MM_LDS + 832, (cv > 0x1000).select(Int32(1), Int32(0)), lane, wv
    )
    if tid < nch:
        _lds_st(lds, MM_LDS + 1024 + tid * 4, [off])
    gpu.barrier()
    slow = (novf > 0) | (tot < fx.min(n_items, Int32(k))) | (tot > 0x2000)
    out_lds = SEL_OUT if k == TOPK else CAND_LDS
    n_out = Int32(0)
    if slow:
        n_out = sel_fullscan(tid, lds, src, n_items, Int32(k), forced, out_lds)
        if id_off != 0:
            for h in range_constexpr((k + 1023) // 1024):
                if tid + h * 1024 < fx.min(n_out, Int32(k)):
                    _lds_st(lds, out_lds + (tid + h * 1024) * 4,
                            [_lds_ld(lds, out_lds + (tid + h * 1024) * 4) + id_off])  # fmt: skip
            gpu.barrier()
    else:
        keys = []
        ids = []
        for j in range_constexpr(16):
            e = tid * 16 + j
            live = e < fx.min(tot, Int32(16384))
            c = Int32(0)
            while (c + 1 < nch) & (_lds_ld(lds, MM_LDS + 1024 + (c + 1) * 4) <= e):
                c = c + 1
            kk = Int32(0)
            ii = Int32(-1)
            if live:
                ent = _ld_sc1(
                    ws + lbase + Int64(c) * 0x8000,
                    2 * (e - _lds_ld(lds, MM_LDS + 1024 + c * 4)),
                    2,
                )
                kk = ent[0]
                ii = ent[1]
            keys.append(live.select(kk, Int32(0)))
            ids.append(ii)
        n_out = _top512(tid, lds, keys, ids, lane, wv, k, out_lds)
    hit = slow.select(Int32(hit_off + 4), Int32(hit_off))
    if k == TOPK:
        _publish(tid, lds, a, eslots, epoch, n_out, pub8, hit)
    else:
        if tid == 0:
            fx.llvm.atomic_add(
                _gword(eslots + hit.to(Int64)), Int32(1), syncscope="agent"
            )
        for h in range_constexpr(2):
            i = tid + h * 1024
            bo.buffer_store(
                (i < n_out).select(_lds_ld(lds, CAND_LDS + i * 4), Int32(-1)),
                rsrc(a[16]),
                2 + i,
            )
        if tid == 0:
            bo.buffer_store(
                fx.Vector.from_elements([Int32(2), n_out], Int32), rsrc(a[16]), 0
            )


@traced
def sel_token_level(
    tid, lds, a, eslots, epoch, c, pos0, n93, k_lo, k_hi, nloc, lo, hi, pub8
):
    """Token-level selector CTA c (TileRT L19822..L24953, dec_mode0_chunk): samples, slot poll, chunk c's
    scores preloaded, the threshold, chunk c's list + count; c = 0 then merges (top 512 -> topk / LL).
    """
    nmin = fx.min((pos0 < 0).select(Int32(0), n93), k_hi)
    n_s = (nmin > k_lo).select(nmin - k_lo, Int32(0))  # s84 (mode 0 / 1)
    nck = (n_s < 1).select(Int32(1), _lshr(n_s + 0x3FFF, Int32(14)))
    div = (n_s < 1).select(Int32(1), (n_s + 0xFFF) >> 12)
    nsamp = (n_s + div - 1) // div
    nwait = fx.min((((nloc + 15) >> 4) + 15) >> 4, Int32(256))
    tkeys = [Int32(0), Int32(0), Int32(0), Int32(0)]
    if (n_s >= 1) & (nck >= 2):
        tkeys = sel_samples(tid, eslots, epoch, nsamp)
    if tid < nwait:
        _wait_flag(eslots, tid, epoch)
    gpu.barrier()
    tpre = [fx.Vector.filled(4, 0, Int32) for _ in range(4)]
    if (c < nck) & (nck >= 2):
        tpre = [
            _ld_sc1(a[4], SCORE_OFF // 4 + c * 16384 + tid * 16 + 4 * q, 4)
            for q in range(4)
        ]
    tthr = sel_threshold(tid, lds, tkeys, n_s, nsamp, fx.min(n_s, Int32(TOPK)))
    lo_n = (hi <= lo).select(Int32(0), lo - k_lo)
    hi_n = (hi <= lo).select(Int32(0), hi - k_lo)
    _wa, wlb, _wlb2, wq, _wc, wmx = ws_layout(a[6])
    cb = wq + wmx * 128  # token count LLs (ws + C - 0x2000 for M = 1 M)
    if (nck >= 2) & (c < nck):
        sel_chunk(tid, lds, a[4], eslots, epoch, c, n_s, tthr, tpre, lo_n, hi_n, k_lo, wlb, cb,
                  (n_s > 0) & (nck > 1))  # fmt: skip
        if c == 0:
            sel_merge(tid, lds, a, a[4], eslots, epoch, nck, pub8, wlb, cb, 0x102908, TOPK,
                      a[4] + SCORE_OFF + Int64(k_lo) * 4, n_s, Int32(-1), k_lo)  # fmt: skip


@traced
def sel_block_level(
    tid, lds, a, eslots, epoch, bc, pos0, n93, k_lo, k_hi, nloc, lo, hi, pub8
):
    """Block-level selector (mode 1) CTA s83 + bc (TileRT L20292..L20566 sampling, L21857.. chunks;
    dec_mode1 3.2..3.6): chunk bc (stride 8) of the 8-key block maxima, its own 1024 x 4 samples of them,
    the block-target threshold; bc = 0 merges the top 2048 blocks into temp:cand."""
    s93v = (pos0 < 0).select(Int32(0), n93)
    n_b = fx.min(_lshr(s93v + 7, Int32(3)), (k_hi + 7) // 8) - k_lo // 8
    s6b = k_lo // 8
    nckb = (n_b < 1).select(Int32(1), _lshr(n_b + 0x3FFF, Int32(14)))
    s94 = _lshr(s93v - 1, Int32(3))
    nwait = fx.min((((nloc + 15) >> 4) + 15) >> 4, Int32(256))
    if tid < nwait:
        _wait_flag(eslots, tid, epoch)
    gpu.barrier()
    wa, _wlb, wlb2, wq, _wc, wmx = ws_layout(a[6])
    if (bc < nckb) & (nckb >= 2):
        p4 = Int32((Int64(n_b) * Int64(tid)) >> 10) & -4
        xs = _ld_sc1(a[4] + wa, p4, 4)
        bkeys = []
        for q in range_constexpr(4):
            kq = (p4 + q + s6b == s94).select(Int32(-8388608), _sortable(xs[q]))
            bkeys.append((p4 + q < n_b).select(kq, Int32(0)))
        bthr = sel_threshold(
            tid, lds, bkeys, n_b, Int32(4096), fx.min(n_b, Int32(2048)), True
        )
        lo_b = (hi <= lo).select(Int32(0), (lo >> 3) - s6b)
        hi_b = (hi <= lo).select(Int32(0), (hi >> 3) - s6b)
        cbb = wq + wmx * 192  # block count LLs (ws + C - 0x1000 for M = 1 M)
        bpre = [fx.Vector.filled(4, 0, Int32) for _ in range(4)]
        sel_chunk(tid, lds, a[4], eslots, epoch, bc, n_b, bthr, bpre, lo_b, hi_b, s6b, wlb2, cbb,
                  (n_b > 0) & (nckb > 1), True, wa, s94)  # fmt: skip
        if bc == 0:
            sel_merge(tid, lds, a, a[4], eslots, epoch, nckb, pub8, wlb2, cbb, 0x102928, 2048,
                      a[4] + wa + Int64(s6b) * 4, n_b, s94 - s6b, s6b)  # fmt: skip


def _gword(addr):
    from atom.mono.device.queue import word

    return word(addr)


def u32_gt(a, b):
    return (a ^ Int32(-2147483648)) > (b ^ Int32(-2147483648))


@traced
def _rescore(tid, li_p, vpos_p, cache0, cache1, rows, ws, m_cap, qrow, k_hi):
    lane = tid & 63
    col, kq = lane & 15, lane >> 4
    l0, v0 = _ld(li_p, 0), _ld(vpos_p, 0)
    mx = (v0 > -1).select(fx.max(l0, Int32(-1)), Int32(-1))
    mn = ((v0 | l0) > -1).select(fx.min(l0, Int32(0x40000000)), Int32(0x40000000))
    for r in range_constexpr(1, rows):
        lr, vr = _ld(li_p, r), _ld(vpos_p, r)
        c = (lr | vr) > -1
        mx = c.select(fx.max(lr, mx), mx)
        mn = c.select(fx.min(lr, mn), mn)
    key0 = (mn & -8) + col
    key = fx.min(k_hi - 1, key0)
    kv = _ld_sc1(cache0, key * 16 + kq * 4, 4)
    sb = (_ld_sc1(cache1, key) >> (kq * 8)) & 0xFF
    qb = qrow * 2048 + col * 64 + kq * 16
    qa0 = _ld(ws, qb >> 2, 4)
    qa1 = _ld(ws, (qb + 1024) >> 2, 4)
    sw = (0x4000 + qrow * 128 + col * 4) >> 2
    sa0 = (_ld(ws, sw) >> (kq * 8)) & 0xFF
    sa1 = (_ld(ws, sw + 16) >> (kq * 8)) & 0xFF
    w0 = _ld(ws, (0x4400 + qrow * 128 + (lane & 48)) >> 2, 4)
    w1 = _ld(ws, (0x4440 + qrow * 128 + (lane & 48)) >> 2, 4)
    v = _score16(qa0, qa1, sa0, sa1, w0, w1, kv, sb)
    m = _max8(v, lane)
    if (kq == 0) & (fx.min(k_hi, (mx & -8) + 8) > key0):
        _st(ws, SCORE_OFF // 4 + key0, v.bitcast(Int32))
        if (tid & 7) == 0:
            _st(ws + _bmax_off(m_cap), key0 >> 3, m.bitcast(Int32))


@traced
def k_prep(tid, slots, lds, li_p, vpos_p, knorm, ropec, rope_n, rope_mul, cache0, cache1, cap, epoch,
           rows, rescore, ws, m_cap, qrow, k_hi):  # fmt: skip
    """CTA71: the new tokens' index keys from the k LL -> normed, roped, MXFP4 -> index_k cache; with
    ``rescore`` (TileRT's copy of this role run after the scorers: new compressed keys this step) wave 0
    then scores the 16 keys from the smallest latent index's 8-block (ws's score / block-max arrays).
    """
    if tid < 768:
        e = tid // 6
        r = tid - e * 6
        v = _wait_tag(slots, (KLL_OFF + e * 64 + r * 8) // 4, epoch)
        _lds_st(lds, K_LDS + r * 512 + e * 4, [v])
    if tid < 256:
        _lds_st(lds, KQ_LDS + tid * 4, [Int32(0)])
    gpu.barrier()
    row = (tid >> 5) & 7
    q = tid >> 8
    lane = tid & 63
    l32 = tid & 31
    li = Int32(-1)
    pos = Int32(-1)
    word = Int32(0)
    if row < 6:
        pos = _ld(vpos_p, row)
        li = _ld(li_p, row)
        g = _ld(knorm, q * 32 + l32).bitcast(F32)
        rp = Int32(0)
        if li > -1:
            rp = rope_mul * li
        rp = (rp > -1).select(fx.min(rope_n - 1, rp), Int32(0))
        pair = (tid > 0x1FF).select(q * 16 - 32, Int32(0)) | (l32 >> 1)
        cs = _ld(ropec, rp * 64 + pair).bitcast(F32)
        sn = _ld(ropec, rp * 64 + pair + 32).bitcast(F32)
        xb = K_LDS + row * 512 + l32 * 4
        x0 = _lds_ld(lds, xb).bitcast(F32)
        x32 = _lds_ld(lds, xb + 128).bitcast(F32)
        x64 = _lds_ld(lds, xb + 256).bitcast(F32)
        x96 = _lds_ld(lds, xb + 384).bitcast(F32)
        s0 = fmath.fma(x0, x0, bperm(x0 * x0, lane ^ 16))
        s64 = fmath.fma(x64, x64, bperm(x64 * x64, lane ^ 16))
        s32 = fmath.fma(x32, x32, bperm(x32 * x32, lane ^ 16))
        s96 = fmath.fma(x96, x96, bperm(x96 * x96, lane ^ 16))
        for k in (8, 4, 2, 1):
            s0 = s0 + bperm(s0, lane ^ k)
            s64 = s64 + bperm(s64, lane ^ k)
            s32 = s32 + bperm(s32, lane ^ k)
            s96 = s96 + bperm(s96, lane ^ k)
        tot = (s0 + s32) + (s64 + s96)
        rs = _rsq_scaled(fmath.fma(tot, f32c(0x3C000000), f32c(0x1E3CE508)))
        x = _lds_ld(lds, xb + q * 128).bitcast(F32)
        y = (bf_rne_hi(g * (x * rs)) & Int32(-65536)).bitcast(F32)
        if tid > 0x1FF:
            o = bperm(y, lane ^ 1)
            so = sn * o
            odd = (tid & 1) == 1
            y2 = fmath.fma(cs, y, odd.select(so, -so))
            y = (bf_rne_hi(y2) & Int32(-65536)).bitcast(F32)
        m = abs(y)
        for k in (16, 8, 4, 2, 1):
            m = fx.max(m, bperm(m, lane ^ k))
        m = fx.max(m, f32c(0x01C00000))
        d = (m * f32c(0x3E2AAAAB)).bitcast(Int32)
        up = d + 0x7FFFFF
        inv = (d < 0x7E800001).select(Int32(0x7F000000) - (up & Int32(-8388608)),
                                                          Int32(0x400000))  # fmt: skip
        v = fx.min(fx.max(y * inv.bitcast(F32), f32c(0xC0C00000)), f32c(0x40C00000))
        word = _fp4_code(v) << ((tid & 7) * 4)
        for k in (1, 2, 4):
            word = word | Int32(bperm(word.bitcast(F32), lane ^ k).bitcast(Int32))
        if l32 == 0:
            _lds_st(lds, KE_LDS + row * 16 + q * 4, [(up >> 23) & 0x1FF])
    gpu.barrier()
    if (row < 6) & (pos > -1) & (li > -1) & (li < cap):
        if (tid & 7) == 0:
            _st(cache0, li * 16 + q * 4 + (l32 >> 3), word)
        if (q == 0) & (l32 == 0):
            ex = _lds_ld(lds, KE_LDS + row * 16, 4)
            _st(cache1, li, ex[0] | (ex[1] << 8) | (ex[2] << 16) | (ex[3] << 24))
    gpu.barrier()
    if rescore & (k_hi > 0) & (tid < 64):
        _rescore(tid, li_p, vpos_p, cache0, cache1, rows, ws, m_cap, qrow, k_hi)
    gpu.barrier()
    if tid == 0:
        _st(slots, FLAG_OFF // 4, epoch)


@traced
def _wait_flag(addr, word, epoch):
    """A dword flag polled until it is the epoch (sc1 reload, s_sleep 1)."""
    v = _ld_sc1(addr, word)
    while v != epoch:
        spin_pause()
        rocdl.s_sleep(1)
        v = _ld_sc1(addr, word)


def _relu_bits(x):
    """TileRT's ReLU: v_max_i32 0 on the f32 bits (negative -> +0)."""
    return fx.max(x.bitcast(Int32), Int32(0)).bitcast(F32)


def _score16(qa0, qa1, sa0, sa1, w0, w1, kv, sb):
    """A lane's key column score: two FP4 MFMAs (heads 0-15 / 16-31, the lane's 4 heads each), the
    ReLU-weighted fma chain in head order, then the 4 lane groups summed (permlane16 / 32 swaps).
    """
    z = fx.Vector.filled(4, 0.0, F32)
    d0 = mfma_scaled(qa0, kv, z, sa0, sb, FP4, FP4)
    d1 = mfma_scaled(qa1, kv, z, sa1, sb, FP4, FP4)
    v = fmath.fma(_relu_bits(d0[0]), w0[0].bitcast(F32), F32(0.0))
    for j in range_constexpr(1, 4):
        v = fmath.fma(_relu_bits(d0[j]), w0[j].bitcast(F32), v)
    for j in range_constexpr(4):
        v = fmath.fma(_relu_bits(d1[j]), w1[j].bitcast(F32), v)
    p = permlane_swap(16, v.bitcast(Int32), v.bitcast(Int32))
    v = p[0].bitcast(F32) + p[1].bitcast(F32)
    p = permlane_swap(32, v.bitcast(Int32), v.bitcast(Int32))
    return p[0].bitcast(F32) + p[1].bitcast(F32)


def _max8(v, lane):
    """The max over 8 consecutive lanes (xor 1, 2, 4 bpermutes)."""
    for k in (1, 2, 4):
        v = fx.max(v, bperm(v, lane ^ k))
    return v


@traced
def scorer(tid, s2, lds, a, ws, eslots, epoch, k_lo, k_hi, lo, hi, s81, t0, stride, ntile, n_smp, smp_div,
           bmax):  # fmt: skip
    """Score tiles t0, t0 + stride, .. < ntile (16 keys a wave and tile, keys k_lo + 16 t + lane % 16):
    score = sum_h w_h ReLU(q_h . k) by two FP4 MFMAs (heads 0-15 / 16-31); lanes 0..15 store it (not for
    the new keys [lo, hi), scored as key lo - 1), every smp_div-th key also as a sample LL, and with
    ``bmax`` the 8-key block max."""
    lane = tid & 63
    col = lane & 15
    kq = lane >> 4
    hb = Q_LDS + col * 64 + kq * 16
    qa0 = _lds_ld(lds, hb, 4)
    qa1 = _lds_ld(lds, hb + 1024, 4)
    sa0 = (_lds_ld(lds, QS_LDS + col * 4) >> (kq * 8)) & 0xFF
    sa1 = (_lds_ld(lds, QS_LDS + 64 + col * 4) >> (kq * 8)) & 0xFF
    w0 = _lds_ld(lds, QW_LDS + (tid & 48), 4)
    w1 = _lds_ld(lds, QW_LDS + 64 + (tid & 48), 4)
    inv = F32(1.0) / smp_div.to(F32)
    t = t0
    while t < ntile:
        kb = k_lo + t * 16
        key = fx.min(kb + col, k_hi - 1)
        key = ((hi > lo) & (key >= lo) & (key < hi)).select(lo - 1, key)
        kv = fx.Vector.filled(4, 0, Int32)
        sc = Int32(0)
        if (hi <= lo) & (s81 > -1) & (s81 <= kb + 15):
            if lane == 0:
                _wait_flag(eslots, FLAG_OFF // 4, epoch)
            kv = _ld_sc1(a[0], key * 16 + kq * 4, 4)
            sc = _ld_sc1(a[1], key)
        else:
            kv = _ld(a[0], key * 16 + kq * 4, 4)
            sc = _ld(a[1], key)
        v = _score16(qa0, qa1, sa0, sa1, w0, w1, kv, (sc >> (kq * 8)) & 0xFF)
        rel = t * 16 + col
        ak = k_lo + rel
        oldk = (ak < lo) | (ak >= hi)
        st = (kq == 0) & (ak < k_hi)
        if st & oldk:
            _st(ws, SCORE_OFF // 4 + rel, v.bitcast(Int32))
        i = F32(_call_rint(rel.to(F32) * inv)).to(Int32)
        if st & (rel < n_smp) & (smp_div * i == rel):
            _st2(
                eslots,
                (SAMPLE_OFF + i * 8) // 4,
                oldk.select(v, f32c(0xFF800000)).bitcast(Int32),
                epoch,
            )
        if bmax:
            m = _max8(v, lane)
            if st & oldk & ((tid & 7) == 0):
                _st(a[4] + _bmax_off(a[6]), rel >> 3, m.bitcast(Int32))
        t = t + stride


def _call_rint(x):
    from flydsl._mlir.dialects import llvm

    return llvm.call_intrinsic(T.f32, "llvm.rint.f32", [F32(x).ir_value()], [], [])


def _bmax_off(m):
    """The 8-key block maxima's byte offset in the workspace: (max(M, 0x4000) 32 + 0x6ae0) & ~0xff."""
    mc = (m > Int64(0x4000)).select(m, Int64(0x4000))
    return (mc * 32 + 0x6AE0) & Int64(-256)


@traced
def _entry_eng(tid, s2, lds, a, rows, pub8):
    """fused_engram_kernel: CTAs 0..255 the indexer, 256.. engram_fused's wkv GEMV (two of its 512-thread
    CTAs a CTA: logical CTA 2 (s2 - 256) + tid / 512; GemvParams at 0x190, LDS partials at 0x18e20).
    """
    if s2 < 256:
        _entry(tid, s2, lds, a, rows, pub8)
    else:
        half = tid >> 9
        _eng_gemv(tid & 511, (s2 - 256) * 2 + half, lds + (ENG_LDS // 4 + half * 2048), a[50], a[51],
                  Int32(a[52]), a[53], a[54], Int32(a[55]), a[56], Int32(a[57]), Int32(a[57] >> 32),
                  Int32(a[58]))  # fmt: skip


@traced
def _entry(tid, s2, lds, a, rows, pub8):
    """a: the kernarg i64 words. Roles by s2 (TileRT's dispatch); ported: the prep roles and the regular
    scorer (not yet: the candidate-block scorer, the selector, the split / cross-rank path).
    """
    wv = tid >> 6
    eslots = a[4] + slots_offset(a[6])  # ScoreArgs+0x20 (ws) + C(ScoreArgs+0x30)
    slots = a[46] + slots_offset(
        a[47]
    )  # PrepArgs+0x68 (ws) + C(PrepArgs+0x70): the prep's areas
    epoch = _ld(eslots, s2) + 1
    ratio, mode = Int32(a[7]), Int32(a[7] >> 32)  # ScoreArgs+0x38 / +0x3c
    nsplit, rank, thr = (
        Int32(a[8]),
        Int32(a[8] >> 32),
        Int32(a[9]),
    )  # +0x40 / +0x44 / +0x48
    pos0 = _ld(a[2], 0)  # ScoreArgs+0x10: the row's position
    shift = ratio >> 1
    n93 = (pos0 + 1) >> shift
    n = (pos0 < 0).select(Int32(0), fx.max(n93, Int32(0)))
    split = (nsplit > 1) & (n > thr) & (mode != 2)
    nosplit = (nsplit <= 1) | (n <= thr) | (mode == 2)
    small = nosplit & ((n < 0x4001) | (mode == 2))
    k_lo = Int32(0)
    k_hi = n
    if split:
        ch = (((nsplit + n - 1) // nsplit) + 63) & -64
        k_lo = fx.min(ch * rank, n)
        k_hi = fx.min(k_lo + ch, n)
    li_p = a[10]  # ScoreArgs+0x50: latent_idx (null: no prep)
    vpos_p = a[37]  # PrepArgs+0x20
    # the new compressed keys' 8-aligned block range [lo, hi) (TileRT computes it unless latent_idx is
    # null, split or small)
    lo = Int32(0)
    hi = Int32(0)
    if (li_p != Int64(0)) & nosplit & (n >= 0x4001) & (mode != 2):
        mn = Int32(0x40000000)
        mx = Int32(-1)
        for r in range_constexpr(rows):
            p = _ld(vpos_p, r)
            ok = (p > -1) & (((p + 1) & (ratio - 1)) == 0)
            k1 = (p + 1) >> shift
            mn = (ok & (k1 <= mn) & (k1 < 0x40000001)).select(k1 - 1, mn)
            mx = ok.select(fx.max(k1 - 1, mx), mx)
        lo = (mx < 0).select(Int32(0), mn & -8)
        hi = (mx < 0).select(Int32(0), (mx & 0x7FFFFFF8) + 8)
    # the first latent-indexed new key (TileRT's s81: rows with position and latent index, -1 none)
    s81 = Int32(-1)
    if li_p != Int64(0):
        for r in range_constexpr(rows):
            p = _ld(vpos_p, r)
            lr = _ld(li_p, r)
            if r == 0:
                s81 = (p > -1).select(fx.max(lr, Int32(-1)), Int32(-1))
            else:
                s81 = ((p | lr) > -1).select((s81 < 0).select(lr, fx.min(lr, s81)), s81)
    prep = (s81 > -1) & ((s2 & 7) == 7) & (s2 < 72)
    if prep & (hi <= lo):
        if s2 < 64:
            k_gemv(tid, s2, slots, a[35], a[41], epoch, rows)
        else:
            k_prep(tid, slots, lds, a[36], vpos_p, a[42], a[43], Int32(a[48]), Int32(a[48] >> 32),
                   a[44], a[45], Int32(a[49]), epoch, rows, hi > lo, a[46], a[47], Int32(a[11]),
                   k_hi)  # fmt: skip
    if n > 0x200:
        c0 = Int32(0)
        c1 = Int32(0)
        if mode == 2:
            c0 = _ld(
                a[3], 0
            )  # ScoreArgs+0x18: the candidate-block table {tag, count, blocks..}
            c1 = _ld(a[3], 1)
        cand = (mode == 2) & (pos0 > -1) & (n93 > 0) & (c0 == 2)
        act = (
            s2 < 0
        )  # small: CTAs s2 % 8 == 0 (< 248), waves 0..7; else 16 waves of the first ncta CTAs
        t0 = s2 * 16 + wv
        ntile = (k_hi - k_lo + 15) >> 4
        if small:
            act = ((s2 & 7) == 0) & (s2 < 0xF8)
            t0 = (tid < 512).select(s2 | wv, Int32(0x40000000))
        else:  # (the big candidate-block path, cand and not small, is not ported: no CTA scores)
            nocand = (c0 != 2) | (mode != 2) | (pos0 < 0) | (n93 <= 0)
            act = (s2 < fx.min((ntile + 15) >> 4, Int32(256))) & nocand
        if act:
            if tid < 144:
                row = Int32(a[11])  # ScoreArgs+0x58: the q row
                src = (tid < 128).select(row * 2048 + tid * 16, (tid < 136).select(
                    0x3800 + row * 128 + tid * 16, 0x3B80 + row * 128 + tid * 16))  # fmt: skip
                qv = _ld(a[4], src >> 2, 4)
                _lds_st(lds, Q_LDS + tid * 16, [qv[0], qv[1], qv[2], qv[3]])
            gpu.barrier()
            if small:
                n62 = (pos0 < 0).select(Int32(0), n93)
                kc = cand.select(c1 * 8, n62)
                s33 = pos0
                if li_p != Int64(0):
                    s33 = _ld(li_p, 0)
                s63 = ((li_p != Int64(0)) & (pos0 > -1)).select(
                    fx.max(s33, Int32(-1)), Int32(-1)
                )
                excl = ((s63 < 0) | cand | (pos0 < 0)).select(Int32(-2), s33)
                scorer_small(tid, lds, a, eslots, epoch, t0, kc, cand, n62, excl)
            else:
                nmin = fx.min((pos0 < 0).select(Int32(0), n93), k_hi)
                n_smp = ((mode < 2) & (mode >= 0) & (nmin > k_lo)).select(
                    nmin - k_lo, Int32(0)
                )
                smp_div = (n_smp > 0).select((n_smp + 0xFFF) >> 12, Int32(1))
                scorer(tid, s2, lds, a, a[4], eslots, epoch, k_lo, k_hi, lo, hi, s81, t0, Int32(0x1000),
                       ntile, n_smp, smp_div, (mode == 1) & (n > 0x4000))  # fmt: skip
    gpu.barrier()
    if tid == 0:
        _st(eslots, s2, epoch)
    if small & (n > 0x200) & (s2 == 0xF8):
        c0 = Int32(0)
        c1 = Int32(0)
        if mode == 2:
            c0 = _ld(a[3], 0)
            c1 = _ld(a[3], 1)
        cand = (mode == 2) & (pos0 > -1) & (n93 > 0) & (c0 == 2)
        n62 = (pos0 < 0).select(Int32(0), n93)
        kc = cand.select(c1 * 8, n62)
        tl = cand.select((c1 * 8 + 8) >> 4, (k_hi - k_lo + 15) >> 4)
        select_small(
            tid,
            lds,
            a,
            eslots,
            epoch,
            kc,
            cand,
            n62,
            fx.min((tl + 15) >> 4, Int32(256)),
            pub8,
        )
    # the large-n selector, TileRT's wide layout (L3636..L3732): s83 = min(ceil(n_local / 16384), 64)
    # selector CTAs s2 >= 248 - s83, c = s2 - (248 - s83); c < s83 the token chunks (c = 0 merges), mode 1:
    # c - s83 < 8 the block-max chunks (c = s83 merges into temp:cand)
    nloc = k_hi - k_lo
    if nosplit & (mode != 2) & (n >= 0x4001) & (nloc > 0x20000):
        s83 = fx.min(_lshr(nloc + 0x3FFF, Int32(14)), Int32(64))
        c = s2 - (248 - s83)
        if (c >= 0) & (c < s83):
            sel_token_level(
                tid, lds, a, eslots, epoch, c, pos0, n93, k_lo, k_hi, nloc, lo, hi, pub8
            )
        if (mode == 1) & (c >= s83) & (c < s83 + 8):
            sel_block_level(
                tid,
                lds,
                a,
                eslots,
                epoch,
                c - s83,
                pos0,
                n93,
                k_lo,
                k_hi,
                nloc,
                lo,
                hi,
                pub8,
            )
    if prep & (hi > lo):
        if s2 < 64:
            k_gemv(tid, s2, slots, a[35], a[41], epoch, rows)
        else:
            k_prep(tid, slots, lds, a[36], vpos_p, a[42], a[43], Int32(a[48]), Int32(a[48] >> 32),
                   a[44], a[45], Int32(a[49]), epoch, rows, hi > lo, a[46], a[47], Int32(a[11]),
                   k_hi)  # fmt: skip


def build(key=None):
    """(kernel symbol, launcher) of the port; the launcher takes the variant's kernarg words (50, the
    engram variant 60)."""
    key = _KEY if key is None else key
    kr, rows, eng = VARIANTS[key]
    pub8 = kr in (
        4027,
        499,
    )  # TileRT's KR: 4027 / 499 publish the top-k to 8 LL targets, 2747 / 243 to 4
    name = kernel_symbol(
        "tk_indexer_eng" if eng else "tk_indexer", kr=kr, rows=rows, v=VERSION
    )

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, (LDS_BYTES_ENG if eng else LDS_BYTES) // 4, 16]

    if eng:

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def indexer_eng(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
        a56: Int64, a57: Int64, a58: Int64, a59: Int64,
        ):  # fmt: skip
            lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
            _entry_eng(fx.thread_idx.x, fx.block_idx.x, lds, [a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59], rows, pub8)  # fmt: skip

        @launcher(name)
        def launch_eng(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
        a56: Int64, a57: Int64, a58: Int64, a59: Int64,
            stream: fx.Stream = _CURRENT_STREAM,
        ):  # fmt: skip
            indexer_eng(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59).launch(grid=(356,), block=(THREADS,), stream=stream)  # fmt: skip

        return name, launch_eng

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def indexer(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        a48: Int64, a49: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _entry(fx.thread_idx.x, fx.block_idx.x, lds, [a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49], rows, pub8)  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        a48: Int64, a49: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        indexer(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49).launch(grid=(256,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
