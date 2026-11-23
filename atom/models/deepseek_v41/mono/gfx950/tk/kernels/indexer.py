# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::indexer::fused_kernel<1, 0, KR, ROWS>`` (KR 4027 / 2747, ROWS 6;
# KR 499 / 243, ROWS 1) and ``fused_engram_kernel<1, 0, KR, 6, 6>``.
# ruff: noqa: SIM102  (traced code: nested ifs are nested scf.if, not one fused condition)
"""DeepSeek-V4.1 decode indexer on gfx950, a FlyDSL port of TileRT's fused indexer (score + top-k in one
persistent launch). Kernarg ABI unchanged: ScoreArgs 0x00 (112 B), SelectArgs 0x70 (152 B), PrepArgs
0x108 (136 B) (+ engram GemvParams 0x190, 80 B), read as i64 words; grid 256 (engram 356) x 1024, LDS
104096 B (engram 120480 B).

Ported from the ISA (runs/20261115-flydsl-tilert/m3/indexer_notes.md: role map, float details, the
8-rank replay results): the k GEMV and CTA71 prep roles (incl. the small-n new-key score LL), the regular
and the small-n / candidate-block scorers, the engram GEMV CTAs, the epoch slots, and every selector
route TileRT takes without a split, by n = the compressed key count:
- n <= 0x200: no scoring; CTA 240 publishes [0 .. n-1, -1 ..], mode 1: CTA 241 the trivial cand table.
- 0x200 < n <= 0x4000 (or mode 2): the small-n selector, CTA 248 (exact top 512; scores from the slot
  LLs for n <= 0x1000, else from the score array; a row's latent slot takes CTA71's new-key score).
- 0x4000 < n <= 0x20000, the narrow layout: CTAs 240 + 2 j filter 16 K-key token chunk j (target
  R + 2 isqrt(R) + 4), CTA 240 merges and publishes; mode 1: CTA 241 the exact top 2048 of the 8-key
  block maxima in one chunk -> temp:cand.
- n > 0x20000, the wide layout: CTAs 248 - s83 + c, s83 = ceil(n / 16 K) token chunks (target + 12),
  c = 0 merges; mode 1: 8 more CTAs for the block-max chunks, their merge -> temp:cand.
Not ported: the split-by-rank cross-GPU path (nsplit > 1 and n > the split threshold, a kernarg the
graph sets to 2^30, so never taken) and with it the big candidate-block scorer (cand and not small).
The merges' slow path (a chunk list overflow or too few / too many candidates) gives TileRT's result
by a full rescan; its workspace spills are not reproduced. Selections with fewer than k non-zero keys
(scores with bits 0xffffffff) were not decoded.

Work layout (free where TileRT's outputs do not depend on it): wave 0 alone computes the dispatch state and
hands it to the other waves through LDS; in the small path every CTA scores (TileRT: CTAs 8 j, all on XCD 0 with
the selector CTA 248, so their plain stores stayed in one L2: here the stores are sc1); the selections are a
3-pass 11 / 11 / 10-bit radix with block-wide bin scans, the merge's candidates E a thread (not 16).

Workspace (ws = ScoreArgs+0x20; M = ScoreArgs+0x30, the key capacity):
- slots = ws + C(M) (``slots_offset``): dword s2 = CTA s2's epoch (this launch's tag = slot + 1,
  stored back at the end); + 1536: the 4096 threshold samples LL; + 0x100600 CTA71's flag; + 0x100700
  the new keys' score LL; + 0x100900 the k LL [feature 128][row 8] {f32, epoch}.
- ws + 0x6a00 + 4 k: key k's f32 score; ws + ((max(M, 0x4000) 32 + 0x6ae0) & ~0xff) + 4 b: the
  8-key block b's max (mode 1, n > 16384).
Roles: k GEMV (s2 % 8 == 7, s2 < 64), CTA71 (the new keys: RMSNorm, RoPE, MXFP4 -> index_k cache),
scorers (16 keys a wave and tile: two FP4 MFMAs, sum_h w_h ReLU(q_h . k)), selectors (s2 > 239; 248 for
small n), engram GEMV (engram variant, s2 >= 256).
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import (
    _call,
    bf_rne_hi,
    bperm,
    f32c,
    spin_pause,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.engram_fused import (
    _gemv as _eng_gemv,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.layer_tail import (
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
    memrealtime,
    permlane_swap,
    readlane,
    rsrc,
    traced,
    uniform,
    wave_incl_scan,
    xred,
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
VERSION = "v40"

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
DREC = (
    SEL_OUT + 2048
)  # wave 0's dispatch record for the other waves (<= 32 words, up to LDS_BYTES)
TOPK = 512
CNT_SMALL = (
    0x102914  # slots + this: the hit counter TileRT's small-path selector adds 1 to
)
# LDS (CTA71)
K_LDS = 0  # [row 6][feature 128] f32
KQ_LDS = 3072  # [row][64 B] the row's packed fp4 (16 dwords)
KE_LDS = 4160  # [row][4 dwords] the 4 blocks' e8m0 exponents


# debug builds (TK_STAMP=1): per-stage s_memrealtime stamps, tid 0 of CTA s2 at stage k -> u64 a[23][32 s2 + k]
# (a[23] = SelectArgs+72, TileRT's own profile pointer, null in the graph); production builds emit nothing
_STAMP = os.environ.get("TK_STAMP", "0") == "1"
_SPC = {}  # set by _entry while tracing: the stamp pointer, s2, tid


@traced
def _stk(k):
    if const_expr(_STAMP):
        sp, s2, tid = _SPC["sp"], _SPC["s2"], _SPC["tid"]
        if (tid == 0) & (sp != Int64(0)) & (s2 < 256):
            t = memrealtime()
            bo.buffer_store(fx.Vector.from_elements([Int32(t), Int32(t >> 32)], Int32), rsrc(sp),
                            (s2 * 32 + k) * 2)  # fmt: skip


def _lds_scalar(addr, word, n=1):
    """A wave-uniform load through the scalar cache (s_buffer_load; n = 1 or 4 dwords): every wave of every CTA
    reads the same few words at entry, and as vector loads they all queued on the same L2 lines.
    """
    v = bo.buffer_load(rsrc(addr), word, vec_width=n, is_scalar=True)
    return fx.Vector(v) if n > 1 else Int32(v)


def _ld_live(addr, nbytes, word, n, live, cm=CM_DEV):
    """A per-lane load that a dead lane does not issue to memory: its offset is pushed past the buffer's
    nbytes bound (the hardware returns 0). Dead lanes used to load one clamped address all together, which
    queued thousands of requests on a single L2 line."""
    v = bo.buffer_load(rsrc(addr, nbytes), live.select(word, Int32(0x1FFFFFF0)), vec_width=n, dtype=T.i32,
                       cache_modifier=cm)  # fmt: skip
    return fx.Vector(v) if n > 1 else Int32(v)


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
    _stk(4)
    lane, wv = tid % 64, tid // 64
    o = (s2 >> 3) * 16 + wv
    w = _ld(i_wk, (o * 1024 + lane * 16) // 4, 4)
    xs = [
        _ld(latent, (r * 1024 + lane * 16) // 4, 4) for r in range(rows)
    ]  # all rows' loads before any store
    for r in range_constexpr(rows):
        x = xs[r]
        acc = F32(0.0)
        for j in range_constexpr(4):
            xl, xh = (x[j] << 16).bitcast(F32), (x[j] & Int32(-65536)).bitcast(F32)
            wl, wh = (w[j] << 16).bitcast(F32), (w[j] & Int32(-65536)).bitcast(F32)
            acc = fmath.fma(xl, wl, acc)
            acc = fmath.fma(xh, wh, acc)
        for k in (32, 16, 8, 4, 2, 1):
            acc = xred(
                acc, k, lambda u, w: u + w
            )  # (DPP / permlane, not LDS; the same sums)
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
def scorer_small(tid, lds, a, eslots, epoch, t0, kc, cand, n62, excls):
    """TileRT's small-n / candidate path (TileRT: CTAs s2 % 8 == 0, waves 0..7, tiles t0 + 248 j, j < 5;
    here every CTA's 16 waves, tiles 16 s2 + w and + 4096): slot i < kc (16 a tile) scores key i, or with
    ``cand`` key 8 blocks[i / 8] + i % 8 of the candidate-block table, clamped below n62; lanes 0..15 store it
    (not slot ``excl``): kc <= 4096 as the LL {score, epoch} at slots + 1536 + 8 i, else the score at
    ws + 0x6a00 + 4 i (sc1 stores: the scorers now sit on every XCD).
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
    if tid < 1024:
        # every CTA's 16 waves score (TileRT: CTAs 8 j, 8 waves, tiles t0 + 248 j, j < 5): wave t0 = 16 s2 + w
        # takes tiles t0 and t0 + 4096 (kc <= 16384: one tile at most), so a wave's latency is one tile's, not
        # five; the tiles' loads issued together (a tile past ntile does not load and stores nothing): the cand
        # blocks, then the keys and scales, then the MFMAs
        cbp = (a[3] != Int64(0)).select(a[3], a[2])
        ts = [t0 + j * 4096 for j in range(2)]
        cis = [fx.max(fx.min(kc - 1, t * 16 + col), Int32(0)) for t in ts]
        tl = [t < ntile for t in ts]
        cbs = [
            _ld_live(cbp, 0x40000000, 2 + (ci >> 3), 1, lv, 0)
            for ci, lv in zip(cis, tl)
        ]
        kys = [
            fx.max(fx.min(n62 - 1, cand.select((cb << 3) | (ci & 7), ci)), Int32(0))
            for cb, ci in zip(cbs, cis)
        ]
        kvs = [
            _ld_live(a[0], 0x40000000, key * 16 + kq * 4, 4, lv, 0)
            for key, lv in zip(kys, tl)
        ]
        scs = [_ld_live(a[1], 0x40000000, key, 1, lv, 0) for key, lv in zip(kys, tl)]
        for j in range_constexpr(2):
            t = ts[j]
            if t < ntile:
                i = t * 16 + col
                v = _score16(
                    qa0, qa1, sa0, sa1, w0, w1, kvs[j], (scs[j] >> (kq * 8)) & 0xFF
                )
                keep = (kq == 0) & (i < kc)
                for ex in excls:
                    keep = keep & (i != ex)
                if keep:
                    # (sc1: written through for the selector CTA 248 on XCD 0 - TileRT's scorers were CTAs 8 j,
                    # all on XCD 0 with it, so its plain stores never left that L2)
                    if kc < 0x1001:
                        bo.buffer_store(fx.Vector.from_elements([v.bitcast(Int32), Int32(epoch)], Int32),
                                        rsrc(eslots), (SAMPLE_OFF + i * 8) // 4, cache_modifier=CM_DEV)  # fmt: skip
                    else:
                        bo.buffer_store(
                            v.bitcast(Int32),
                            rsrc(a[4]),
                            SCORE_OFF // 4 + i,
                            cache_modifier=CM_DEV,
                        )


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
def _cand_trivial(tid, a, s93):
    """The mode-1 launch's candidate table for n <= 0x4000 (TileRT L58597.. / L107416..): blocks
    t < ceil(s93 / 8) as themselves, -1 after (2048 entries), then tid 0 the header {s93 > 0, count};
    plain stores, the header after tid 0's own entries."""
    nb = (s93 + 7) // 8
    for h in range_constexpr(2):
        i = tid + h * 1024
        bo.buffer_store((i < nb).select(i, Int32(-1)), rsrc(a[16]), 2 + i)
    if tid == 0:
        bo.buffer_store(
            fx.Vector.from_elements([(s93 > 0).select(Int32(1), Int32(0)), nb], Int32),
            rsrc(a[16]),
            0,
        )


@traced
def _poll_ll(addr, word, epoch, sysc):
    """An {payload, tag} LL entry re-read until its tag is the epoch (busy, no sleep; sc0 sc1 with sysc)."""
    v = _ld(addr, word, 2, CM_SYS if sysc else CM_DEV)
    while v[1] != epoch:
        spin_pause()  # keeps the reload in the loop (without it LLVM hoists the load: an endless spin)
        v = _ld(addr, word, 2, CM_SYS if sysc else CM_DEV)
    return v[0]


@traced
def select_small(tid, lds, a, slots, epoch, kc, cand, n62, s93, pub8, rows, mode):
    """The small path's selector, CTA 248 (TileRT 4027 L97651.., ix499 L40748..; dec_small / dec_499b):
    slot i < kc's score: kc <= 0x1000 the scorer's LL at slots + 1536 + 8 i (no slot poll: each entry
    re-read until its tag is the epoch), else the score array ws + 0x6a00 after every CTA published its
    slot (all score here; TileRT: CTAs 8 j, j < 31); a slot equal to a row's latent index (latent_idx non-null, no cand,
    vpos_r >= 0; the last such row) takes CTA71's new-key score LL at slots + 0x100700 + 8 r instead.
    Live: i < kc and its key (cand-mapped) < n62. The top 512 by the sortable key, ties to the smaller
    slot, slot order, -1 padded; hit counter + 1 at slots + 0x102914; mode 1: the trivial candidate
    table; then the publish."""
    _stk(24)
    lane, wv = tid & 63, tid >> 6
    nocand = cand.select(Int32(0), Int32(1)) != 0
    sex = []
    for r in range_constexpr(rows):
        e = Int32(-2)
        if a[24] != Int64(0):
            vr = _ld(a[37], r)
            lr = _ld(a[24], r)
            e = ((vr >= 0) & nocand).select(lr, Int32(-2))
        sex.append(e)
    keys = []
    ids = []
    # the cand blocks of this thread's 16 slots (two entries: slots 16 tid .. +7 and +8 .. +15, clamped to the
    # last live block), loaded once up front; unused without cand
    cbp = (a[3] != Int64(0)).select(a[3], a[2])
    cbl = fx.max((kc - 1) >> 3, Int32(0))
    cbv = [_ld(cbp, 2 + fx.min(tid * 2 + h, cbl)) for h in range(2)]
    llsrc = kc < 0x1001
    sv = [fx.Vector.filled(4, 0, Int32) for _ in range(4)]
    if kc > 0x1000:
        if tid < 256:  # every CTA scores (scorer_small): all 256 slots
            _wait_flag(slots, tid, epoch)
        gpu.barrier()
        sv = [_ld_sc1(a[4], SCORE_OFF // 4 + tid * 16 + 4 * q, 4) for q in range(4)]
    _stk(27)
    for j in range_constexpr(16):
        i = tid * 16 + j
        sc = sv[j // 4][j % 4]
        rsel = Int32(
            -1
        )  # the last row whose latent slot this is: its score comes from CTA71's LL
        for r in range_constexpr(rows):
            rsel = (sex[r] == i).select(Int32(r), rsel)
        if (
            llsrc & (i < kc) & (rsel < 0)
        ):  # (the scorer skips latent slots: never poll their LL)
            sc = _poll_ll(slots, (SAMPLE_OFF // 4) + 2 * i, epoch, False)
        if (i < kc) & (rsel >= 0):
            sc = _poll_ll(slots, (NEWSC_OFF // 4) + 2 * rsel, epoch, True)
        uk = i
        if cand:
            uk = (cbv[j >> 3] << 3) | (i & 7)
        ok = (i < kc) & (uk < n62)
        keys.append(ok.select(_sortable(sc), Int32(0)))
        ids.append(uk)
    _stk(25)
    tot = _top512(tid, lds, keys, ids, lane, wv)
    _stk(26)
    if tid == 0:
        fx.llvm.atomic_add(
            _gword(slots + Int64(CNT_SMALL)), Int32(1), syncscope="agent"
        )
    if mode == 1:
        _cand_trivial(tid, a, s93)
    _publish(tid, lds, a, slots, epoch, tot, pub8, None)


@traced
def select_trivial(tid, a, slots, epoch, s93, pub8, c, mode):
    """n <= 0x200 (TileRT 4027 L58579..L58674, ix499 L2126..): no scoring; selector c = 0 publishes
    [0, 1, .., s93 - 1, -1 ..] (topk row, LL targets, flag; no hit counter); mode 1: c = 1 writes the
    trivial candidate table."""
    if c == 0:
        if tid < TOPK:
            v = (tid < s93).select(tid, Int32(-1))
            bo.buffer_store(v, rsrc(a[17]), tid)
            for t in (26, 27, 28, 29, 33, 34, 38, 39) if pub8 else (26, 27, 28, 29):
                bo.buffer_store(fx.Vector.from_elements([v, Int32(epoch)], Int32), rsrc(a[t]), tid * 2,
                                cache_modifier=CM_SYS)  # fmt: skip
        if tid == 0:
            _st(a[30], 0, epoch)
    if (mode == 1) & (c == 1):
        _cand_trivial(tid, a, s93)


# ---- large-n selector (wide layout: one CTA a 16 K-key chunk, CTA 248 - nch + c handles chunk c) ----
LIST_OFF = 0x2806A00  # ws + this + 0x8000 c: chunk c's list {u32 key, i32 index} (M = 0x100000; TODO formula)
CNT_OFF = -0x2000  # slots + this + 8 c: chunk c's count LL {count, epoch}
HIST_LDS = 17152  # 4096 u32 threshold buckets
MM_LDS = 0  # [wave 16] (min, max) u32, 16 B a wave; scan scratch at + 256 / 512 / 768, offsets at + 1024
TB_LDS = 0x18E10  # the threshold bucket (one word below SEL_OUT)
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
def _refine_sample_bucket(tid, lds, keys, old_min, old_shift, old_bucket, residual):
    """Only the winning coarse bucket; its residual rank retains the original margin."""
    lane, wv = tid & 63, tid >> 6
    members = [(key != 0) & (_lshr(key - old_min, old_shift) == old_bucket) for key in keys]
    mn, mx = Int32(-1), Int32(0)
    for q in range_constexpr(4):
        mn = members[q].select(_umin(mn, keys[q]), mn)
        mx = members[q].select(_umax(mx, keys[q]), mx)
    for o in (1, 2, 4, 8, 16, 32):
        mn = xred(mn, o, _umin)
        mx = xred(mx, o, _umax)
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
    d = mx - mn
    sh = _u32_lt(Int32(0xFFF), d).select(Int32(20) - _clz(d), Int32(0))
    for q in range_constexpr(4):
        if members[q]:
            fx.llvm.atomic_add(lds + (HIST_LDS // 4 + _lshr(keys[q] - mn, sh)),
                               Int32(1), syncscope="workgroup")
    gpu.barrier()
    h = _lds_ld(lds, HIST_LDS + tid * 16, 4)
    hs = h[0] + h[1] + h[2] + h[3]
    pre, total = _block_excl(lds, MM_LDS + 256, hs, lane, wv)
    above = total - pre - hs
    c3 = above + h[3]
    c2 = c3 + h[2]
    c1 = c2 + h[1]
    c0 = c1 + h[0]
    a3 = _u32_lt(above, residual) & _u32_ge(c3, residual)
    a2 = _u32_lt(c3, residual) & _u32_ge(c2, residual)
    a1 = _u32_lt(c2, residual) & _u32_ge(c1, residual)
    a0 = _u32_lt(c1, residual) & _u32_ge(c0, residual)
    if a0 | a1 | a2 | a3:
        bucket = tid * 4 + a0.select(Int32(0), a1.select(Int32(1), a2.select(Int32(2), Int32(3))))
        _lds_st(lds, TB_LDS, [bucket])
    gpu.barrier()
    bucket = _lds_ld(lds, TB_LDS)
    gpu.barrier()
    return mn + (bucket << sh)


@traced
def sel_threshold(tid, lds, keys, n_smp, nsamp, kk, block=False, tadd=12):
    """The threshold key T (TileRT L21000..L21460): R = (N + nsamp kk - 1) / N; tgt = R + 2 isqrt(R) + tadd
    (12 in the wide layout, 4 in the narrow one, N60408);
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
        mn = xred(mn, o, _umin)
        mx = xred(mx, o, _umax)
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
    tg = r + _isqrt_f32(r) * 5 + 16 if block else r + (_isqrt_f32(r) << 1) + tadd
    c3 = above + h[3]
    c2 = c3 + h[2]
    c1 = c2 + h[1]
    c0 = c1 + h[0]
    a3 = _u32_lt(above, tg) & _u32_ge(c3, tg)
    a2 = _u32_lt(c3, tg) & _u32_ge(c2, tg)
    a1 = _u32_lt(c2, tg) & _u32_ge(c1, tg)
    a0 = _u32_lt(c1, tg) & _u32_ge(c0, tg)
    if a0 | a1 | a2 | a3:
        selected_above = a0.select(c1, a1.select(c2, a2.select(c3, above)))
        _lds_st(lds, MM_LDS + 1280, [tg - selected_above])
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
    coarse_cut = none.select(Int32(-1), mn + (bb << sh))
    fine_cut = coarse_cut
    # Refine only the winning sample bucket while retaining its residual rank.
    # The original no-hit/underflow guards and exact merge fallback stay intact.
    if (~none) & (sh > 0) & _u32_ge(tot, tg):
        residual = _lds_ld(lds, MM_LDS + 1280)
        fine_cut = _refine_sample_bucket(tid, lds, keys, mn, sh, bb, residual)
    return fine_cut


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
        if not block:
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


def _cttz64(x):
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.typing import as_ir_value

    v = llvm.call_intrinsic(T.i64, "llvm.cttz.i64", [as_ir_value(Int64(x)), as_ir_value(fx.Boolean(True))], [], [])  # fmt: skip
    return Int32(Int64(v))


# _top512's three radix histograms (digits: key bits 31..21, 20..10, 9..0). The first pass's keys crowd into a
# few bins, so it has 4 copies (copy = lane & 3, 2052 words apart: one bin's copies sit in 4 different banks),
# summed by the scan: 4 x 2052 + 2048 + 1024 words
RH_LDS = 34816
RH_C0 = 2052
RH_OFF = (0, 4 * RH_C0, 4 * RH_C0 + 2048)
RH_WORDS = 4 * RH_C0 + 2048 + 1024
RH_NB = (2048, 2048, 1024)
RH_SH = (21, 10, 0)
RH_MSK = (0, -2097152, -1024)  # the bits above the pass's digit


@traced
def _radd(lds, hbl, kv, msk, pre, sh, nb):
    """_top512: a key's +1 to its digit's bin when it matches the prefix."""
    if (kv != 0) & ((kv & msk) == pre):
        fx.llvm.atomic_add(
            lds + (hbl + (_lshr(kv, sh) & (nb - 1))), Int32(1), syncscope="workgroup"
        )


def _rge(kv, thr):
    """_top512: a key's packed (greater << 16, equal) count against the threshold."""
    return (((kv == thr) & (thr != 0)).select(Int32(1), Int32(0))
            + ((kv != 0) & u32_gt(kv, thr)).select(Int32(65536), Int32(0)))  # fmt: skip


@traced
def _rpick(lds, kv, iv, thr, rest, pos, r, k, out_lds):
    """_top512: a key taken (greater, or one of the first rest equal ones) -> its id at out_lds[pos]."""
    gt = (kv != 0) & u32_gt(kv, thr)
    eq = (kv == thr) & (thr != 0)
    pick = gt | (eq & (r < rest))
    if pick & (pos < k):
        _lds_st(lds, out_lds + pos * 4, [iv])
    return pos + pick.select(Int32(1), Int32(0)), r + eq.select(Int32(1), Int32(0))


@traced
def _top512(tid, lds, keys, ids, lane, wv, k=TOPK, out_lds=SEL_OUT, ne=None):
    """The top k of the CTA's u32 keys (16 a thread, 0 = dead), ties at the k-th key taken in thread / slot order
    (TileRT's rule); their ids in that order -> LDS out_lds. -> the count. A radix select over 11 + 11 + 10 bit
    digits: a pass counts the keys that match the prefix so far into its own histogram (all three cleared up
    front), then the whole CTA scans the bins from the top (thread t: bins nb - 1 - 2 t, nb - 2 - 2 t; one block
    scan) and the thread holding the rest-th largest names its bin. Finally one packed scan of (greater, equal)
    counts gives each thread's output position: excl(greater) + min(rest, excl(equal)). ``ne`` (uniform):
    only slots j < ne can be live (the merge's E a thread); the others are skipped by a scalar branch.
    """
    pre = Int32(0)
    rest = Int32(k)
    for h in range_constexpr((k + 1023) // 1024):
        if tid + h * 1024 < k:
            _lds_st(lds, out_lds + (tid + h * 1024) * 4, [Int32(-1)])
    for q in range_constexpr(2):
        _lds_st(
            lds,
            RH_LDS + (q * 4096 + tid * 4) * 4,
            [Int32(0), Int32(0), Int32(0), Int32(0)],
        )
    if tid < (RH_WORDS - 8192) // 4:
        _lds_st(
            lds, RH_LDS + (8192 + tid * 4) * 4, [Int32(0), Int32(0), Int32(0), Int32(0)]
        )
    gpu.barrier()
    _stk(16)
    for p in range_constexpr(3):
        hb = RH_LDS // 4 + RH_OFF[p]
        nb = RH_NB[p]
        hbl = hb + ((lane & 3) * RH_C0 if p == 0 else 0)
        for j in range_constexpr(16):
            if const_expr((ne is None) or (j == 0)):
                _radd(lds, hbl, keys[j], RH_MSK[p], pre, RH_SH[p], nb)
            else:
                if Int32(j) < ne:
                    _radd(lds, hbl, keys[j], RH_MSK[p], pre, RH_SH[p], nb)
        gpu.barrier()
        if const_expr(p == 0):
            _stk(17)
        # thread t: bins b1 = nb - 1 - 2 t, b0 = b1 - 1 (2048 bins), or b1 = 1023 - t alone (1024 bins)
        nper = nb // 1024
        b1 = nb - 1 - nper * tid
        h1 = Int32(0)
        h0 = Int32(0)
        for c in range_constexpr(4 if p == 0 else 1):
            if const_expr(nper == 2):
                v2 = _lds_ld(lds, (hb + c * RH_C0 + b1 - 1) * 4, 2)
                h0 = h0 + v2[0]
                h1 = h1 + v2[1]
            else:
                h1 = h1 + _lds_ld(lds, (hb + c * RH_C0 + b1) * 4)
        ex, tot = _block_excl(lds, SEL_WSUM + 128, h1 + h0, lane, wv)
        if (ex < rest) & (ex + h1 >= rest):
            _lds_st(lds, SEL_PICK + p * 8, [b1, rest - ex])
        if const_expr(nper == 2):
            if (ex + h1 < rest) & (ex + h1 + h0 >= rest):
                _lds_st(lds, SEL_PICK + p * 8, [b1 - 1, rest - ex - h1])
        gpu.barrier()
        pk = _lds_ld(lds, SEL_PICK + p * 8, 2)
        found = tot >= rest
        rest = found.select(pk[1], rest)
        pre = found.select(pre | (pk[0] << RH_SH[p]), Int32(0))
        _stk(20 + p)
    thr = pre
    nge = Int32(0)
    for j in range_constexpr(16):
        if const_expr((ne is None) or (j == 0)):
            nge = nge + _rge(keys[j], thr)
        else:
            if Int32(j) < ne:
                nge = nge + _rge(keys[j], thr)
    pex, ptot = _block_excl(lds, SEL_WSUM + 64, nge, lane, wv)
    eqb = pex & 0xFFFF
    pos = _lshr(pex, Int32(16)) + fx.min(rest, eqb)
    r = eqb
    for j in range_constexpr(16):
        if const_expr((ne is None) or (j == 0)):
            pos, r = _rpick(lds, keys[j], ids[j], thr, rest, pos, r, k, out_lds)
        else:
            if Int32(j) < ne:
                pos, r = _rpick(lds, keys[j], ids[j], thr, rest, pos, r, k, out_lds)
    gpu.barrier()
    return _lshr(ptot, Int32(16)) + fx.min(rest, ptot & 0xFFFF)


@traced
def _fs_block(src, n_items, forced, i):
    """sel_fullscan: items i .. i + 15 (16-aligned) as sortable keys (0 past n_items; ``forced`` -> 0xff800000),
    four 16-byte loads issued together (sc1; past the end out of the buffer bound, not loaded).
    """
    ks = []
    for q in range_constexpr(4):
        v4 = _ld_live(src, 0x40000000, i + 4 * q, 4, i + 4 * q < n_items)
        for u in range_constexpr(4):
            ii = i + 4 * q + u
            kv = (ii == forced).select(Int32(-8388608), _sortable(v4[u]))
            ks.append((ii < n_items).select(kv, Int32(0)))
    return ks


@traced
def sel_fullscan(tid, lds, src, n_items, kk, forced, out_lds):
    """The merge's fallback (a chunk overflowed, total < kk or total exceeds the exact merge capacity; semantic
    port, the slow path's workspace spills are not reproduced): the exact top kk of all n_items items
    (key = sortable(src[i]) (sc1), item ``forced`` -> 0xff800000), ties to the smaller index, written in
    index order to out_lds. Thread t scans items [t per, (t + 1) per), per a multiple of 16, in blocks of 16
    (four 16-byte loads in flight; the old one-item loop waited on every load). Each radix pass counts the
    keys equal to the thread's pivot digit (its first key's) in a register, one LDS atomic for them, atomics
    only for the others (the path runs on heavily tied keys, where one bin takes nearly all). In-step this
    runs about once a step on some ranks and sits on the step's critical path: TileRT ~400 us there, the old
    loop ~600 us, this one ~180 us (n0, forced, standalone). -> the count."""
    lane, wv = tid & 63, tid >> 6
    per = (((n_items + 1023) >> 10) + 15) & -16
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
        piv = Int32(-1)
        pc = Int32(0)
        i = i0
        while i < i1:
            ks = _fs_block(src, n_items, forced, i)
            if i == i0:
                piv = (ks[0] >> sh) & 255
            for u in range_constexpr(16):
                kv = ks[u]
                d = (kv >> sh) & 255
                part = (kv != 0) & ((kv & msk) == pre) & (i + u < i1)
                pc = pc + (part & (d == piv)).select(Int32(1), Int32(0))
                if part & (d != piv):
                    fx.llvm.atomic_add(
                        lds + (SEL_HIST // 4 + d), Int32(1), syncscope="workgroup"
                    )
            i = i + 16
        if pc > 0:
            fx.llvm.atomic_add(lds + (SEL_HIST // 4 + piv), pc, syncscope="workgroup")
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
        ks = _fs_block(src, n_items, forced, i)
        for u in range_constexpr(16):
            kv = ks[u]
            inr = i + u < i1
            neq = neq + ((kv == thr) & (thr != 0) & inr).select(Int32(1), Int32(0))
            ngt = ngt + ((kv != 0) & u32_gt(kv, thr) & inr).select(Int32(1), Int32(0))
        i = i + 16
    eqb, _eqt = _block_excl(lds, SEL_WSUM, neq, lane, wv)
    take = fx.min(
        fx.max(rest - eqb, Int32(0)), neq
    )  # this thread's equal keys taken, in index order
    pos, tot = _block_excl(lds, SEL_WSUM + 64, ngt + take, lane, wv)
    r = eqb
    i = i0
    while i < i1:
        ks = _fs_block(src, n_items, forced, i)
        for u in range_constexpr(16):
            kv = ks[u]
            inr = i + u < i1
            gt = (kv != 0) & u32_gt(kv, thr) & inr
            eq = (kv == thr) & (thr != 0) & inr
            pick = gt | (eq & (r < rest))
            if pick & (pos < kk):
                _lds_st(lds, out_lds + pos * 4, [i + u])
            pos = pos + pick.select(Int32(1), Int32(0))
            r = r + eq.select(Int32(1), Int32(0))
        i = i + 16
    gpu.barrier()
    return tot


@traced
def _publish(tid, lds, a, slots, epoch, tot, pub8, cnt_off):
    """LDS SEL_OUT[0..tot) (-1 past it) -> topk row (plain) and {idx, tag} to the LL targets (sc0 sc1);
    tid 0: the flag (sc1) and + 1 to the hit counter at slots + cnt_off."""
    if cnt_off is not None:
        if tid == 0:
            fx.llvm.atomic_add(
                _gword(slots + Int64(cnt_off)), Int32(1), syncscope="agent"
            )
    if tid < TOPK:
        v = (tid < tot).select(_lds_ld(lds, SEL_OUT + tid * 4), Int32(-1))
        bo.buffer_store(v, rsrc(a[17]), tid)
        for t in (26, 27, 28, 29, 33, 34, 38, 39) if pub8 else (26, 27, 28, 29):
            bo.buffer_store(fx.Vector.from_elements([v, Int32(epoch)], Int32), rsrc(a[t]), tid * 2,
                            cache_modifier=CM_SYS)  # fmt: skip
    if tid == 0:
        _st(a[30], 0, epoch)


@traced
def _mgather(lds, ws, lbase, tid, lg, j, lim, nch):
    """The merge's candidate at position e = (tid << lg) + j: its chunk (the largest c < nch with offset
    P[c] <= e, a branchless binary search over the LDS offsets, nch <= 64) and its list entry {key, index}
    (0 / -1 past lim)."""
    e = (tid << lg) + j
    live = e < lim
    c = Int32(0)
    for stp in (32, 16, 8, 4, 2, 1):
        cn = c + stp
        c = (
            (cn < nch) & (_lds_ld(lds, MM_LDS + 1024 + fx.min(cn, Int32(63)) * 4) <= e)
        ).select(cn, c)
    w = c * 0x2000 + 2 * (e - _lds_ld(lds, MM_LDS + 1024 + c * 4))
    ent = _ld_live(ws + lbase, 64 * 0x8000, w, 2, live)
    return live.select(ent[0], Int32(0)), live.select(ent[1], Int32(-1))


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
    polls the nch count LLs at ws + cbase. Normal path (no count > 4096, min(n_items, k) <= total <= capacity,
    8192 for both token top512 and block top2048):
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
    _stk(12)
    # Fine sample buckets reduce over-wide candidate sets before the exact
    # merge. Keep the tested 8K capacity and original overflow/underflow guards.
    capacity = 8192
    slow = (novf > 0) | (tot < fx.min(n_items, Int32(k))) | (tot > capacity)
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
        # the candidates gathered from the chunk lists (_mgather), E a thread
        keys = []
        ids = []
        lim = fx.min(tot, Int32(16384))
        # E = 1, 2, 4, 8 or 16 entries a thread (the least power of two with 1024 E >= lim), positions
        # e = E tid + j: thread / slot order stays position order, and the work of the selection is spread
        # over all 16 waves (with 16 a thread, ~2000 candidates sat in waves 0..1 and serialized there)
        lg = ((lim > 1024).select(Int32(1), Int32(0)) + (lim > 2048).select(Int32(1), Int32(0))
              + (lim > 4096).select(Int32(1), Int32(0)) + (lim > 8192).select(Int32(1), Int32(0)))  # fmt: skip
        ne = Int32(1) << lg
        for j in range_constexpr(16):
            kk = Int32(0)
            ii = Int32(-1)
            if const_expr(j == 0):
                kk, ii = _mgather(lds, ws, lbase, tid, lg, j, lim, nch)
            else:
                if Int32(j) < ne:  # (uniform: slots past E skipped by a scalar branch)
                    kk, ii = _mgather(lds, ws, lbase, tid, lg, j, lim, nch)
            keys.append(kk)
            ids.append(ii)
        _stk(14)
        n_out = _top512(tid, lds, keys, ids, lane, wv, k, out_lds, ne)
    _stk(13)
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
    tid, lds, a, eslots, epoch, c, pos0, n93, k_lo, k_hi, nloc, lo, hi, pub8, tadd=12
):
    """Token-level selector of chunk c (TileRT L19822..L24953, dec_mode0_chunk; the narrow layout's twin
    N58776..N74273 with target + 4, dec_narrow 3.2..3.6): samples, slot poll, chunk c's scores preloaded,
    the threshold, chunk c's list + count; chunk 0's CTA then merges (top 512 -> topk / LL).
    """
    _stk(8)
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
    _stk(9)
    tpre = [fx.Vector.filled(4, 0, Int32) for _ in range(4)]
    if (c < nck) & (nck >= 2):
        tpre = [
            _ld_sc1(a[4], SCORE_OFF // 4 + c * 16384 + tid * 16 + 4 * q, 4)
            for q in range(4)
        ]
    tthr = sel_threshold(
        tid, lds, tkeys, n_s, nsamp, fx.min(n_s, Int32(TOPK)), False, tadd
    )
    _stk(10)
    lo_n = (hi <= lo).select(Int32(0), lo - k_lo)
    hi_n = (hi <= lo).select(Int32(0), hi - k_lo)
    _wa, wlb, _wlb2, wq, _wc, wmx = ws_layout(a[6])
    cb = wq + wmx * 128  # token count LLs (ws + C - 0x2000 for M = 1 M)
    if (nck >= 2) & (c < nck):
        sel_chunk(tid, lds, a[4], eslots, epoch, c, n_s, tthr, tpre, lo_n, hi_n, k_lo, wlb, cb,
                  (n_s > 0) & (nck > 1))  # fmt: skip
        _stk(11)
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
    _stk(16)
    if tid < nwait:
        _wait_flag(eslots, tid, epoch)
    gpu.barrier()
    _stk(17)
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
        _stk(18)
        lo_b = (hi <= lo).select(Int32(0), (lo >> 3) - s6b)
        hi_b = (hi <= lo).select(Int32(0), (hi >> 3) - s6b)
        cbb = wq + wmx * 192  # block count LLs (ws + C - 0x1000 for M = 1 M)
        bpre = [fx.Vector.filled(4, 0, Int32) for _ in range(4)]
        sel_chunk(tid, lds, a[4], eslots, epoch, bc, n_b, bthr, bpre, lo_b, hi_b, s6b, wlb2, cbb,
                  (n_b > 0) & (nckb > 1), True, wa, s94)  # fmt: skip
        _stk(19)
        if bc == 0:
            sel_merge(tid, lds, a, a[4], eslots, epoch, nckb, pub8, wlb2, cbb, 0x102928, 2048,
                      a[4] + wa + Int64(s6b) * 4, n_b, s94 - s6b, s6b)  # fmt: skip


@traced
def sel_block_single(tid, lds, a, eslots, epoch, pos0, n93, k_lo, k_hi, nloc, lo, hi):
    """Narrow layout, mode 1, CTA c = 1 (TileRT N74276..N89247, dec_narrow 3.7): N = ceil(n / 8) <= 16384
    block maxima in one chunk. Slot poll of the scorer CTAs; tid 0 waits for CTA71's flag when the new
    keys' block window [lo / 8, hi / 8) meets [0, N); item i = the f32 block max at ws + a + 4 i (sc1),
    block (s93 - 1) >> 3 forced to 0xff800000, key 0 past N; the exact top 2048 by sortable key, ties to
    the smaller block, ascending; hit counter + 1 at slots + 0x102920; temp:cand = {2, cnt, blocks..,
    -1 past cnt} (plain stores, tid 0's header after its own entries). TileRT's sampled sub-paths only
    pick the code route, the result is the same."""
    lane, wv = tid & 63, tid >> 6
    s93v = (pos0 < 0).select(Int32(0), n93)
    s6b = k_lo // 8
    m_b = fx.min(_lshr(s93v + 7, Int32(3)), (k_hi + 7) // 8)
    n_b = (m_b > s6b).select(m_b - s6b, Int32(0))
    s94 = _lshr(s93v - 1, Int32(3))
    nwait = fx.min((((nloc + 15) >> 4) + 15) >> 4, Int32(256))
    if tid < nwait:
        _wait_flag(eslots, tid, epoch)
    gpu.barrier()
    lo_b = (hi <= lo).select(Int32(0), (lo >> 3) - s6b)
    hi_b = (hi <= lo).select(Int32(0), (hi >> 3) - s6b)
    if fx.max(lo_b, Int32(0)) < fx.min(hi_b, n_b):
        if tid == 0:
            _wait_flag(eslots, FLAG_OFF // 4, epoch)
        gpu.barrier()
    wa, _wlb, _wlb2, _wq, _wc, _wmx = ws_layout(a[6])
    bv = [_ld_sc1(a[4] + wa + Int64(s6b) * 4, tid * 16 + 4 * q, 4) for q in range(4)]
    bkeys = []
    bids = []
    for j in range_constexpr(16):
        bi = tid * 16 + j
        bk = (bi + s6b == s94).select(Int32(-8388608), _sortable(bv[j // 4][j % 4]))
        bkeys.append((bi < n_b).select(bk, Int32(0)))
        bids.append(bi + s6b)
    nbo = _top512(tid, lds, bkeys, bids, lane, wv, 2048, CAND_LDS)
    if tid == 0:
        fx.llvm.atomic_add(
            _gword(eslots + Int64(0x102920)), Int32(1), syncscope="agent"
        )
    for h in range_constexpr(2):
        i = tid + h * 1024
        bo.buffer_store(
            (i < nbo).select(_lds_ld(lds, CAND_LDS + i * 4), Int32(-1)),
            rsrc(a[16]),
            2 + i,
        )
    if tid == 0:
        bo.buffer_store(fx.Vector.from_elements([Int32(2), nbo], Int32), rsrc(a[16]), 0)


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
def _newkey_scores(tid, lds, slots, ws, qrow, epoch):
    """TileRT L885..L946: the 6 new keys (LDS: packed fp4 rows at KQ_LDS, e8m0 at KE_LDS; rows 6..15 zero,
    scale 0x7f) scored like the scorer (q / scales / weights straight from ws) -> {score, tag} sc0 sc1 at
    slots + 0x100700 + 8 r, every row."""
    lane = tid & 63
    col, kq = lane & 15, lane >> 4
    qb = qrow * 2048 + col * 64 + kq * 16
    qa0 = _ld(ws, qb >> 2, 4)
    qa1 = _ld(ws, (qb + 1024) >> 2, 4)
    sw = (0x4000 + qrow * 128 + col * 4) >> 2
    sa0 = (_ld(ws, sw) >> (kq * 8)) & 0xFF
    sa1 = (_ld(ws, sw + 16) >> (kq * 8)) & 0xFF
    w0 = _ld(ws, (0x4400 + qrow * 128 + kq * 16) >> 2, 4)
    w1 = _ld(ws, (0x4440 + qrow * 128 + kq * 16) >> 2, 4)
    kv = _lds_ld(lds, KQ_LDS + col * 64 + kq * 16, 4)
    sb = (col < 6).select(_lds_ld(lds, KE_LDS + col * 16 + kq * 4), Int32(0x7F))
    v = _score16(qa0, qa1, sa0, sa1, w0, w1, kv, sb)
    if (kq == 0) & (col < 6):
        bo.buffer_store(fx.Vector.from_elements([v.bitcast(Int32), Int32(epoch)], Int32), rsrc(slots),
                        (NEWSC_OFF + col * 8) // 4, cache_modifier=CM_SYS)  # fmt: skip


@traced
def k_prep(tid, slots, lds, li_p, vpos_p, knorm, ropec, rope_n, rope_mul, cache0, cache1, cap, epoch,
           rows, rescore, ws, m_cap, qrow, k_hi, small):  # fmt: skip
    """CTA71: the new tokens' index keys from the k LL -> normed, roped, MXFP4 -> index_k cache; in small
    mode (n <= 0x4000 or mode 2; TileRT L885..L946) wave 0 then scores the 6 new keys from its LDS copy into
    the LL at slots + 0x100700 + 8 r (sc0 sc1, every row) for the small selector; with
    ``rescore`` (TileRT's copy of this role run after the scorers: new compressed keys this step) wave 0
    then scores the 16 keys from the smallest latent index's 8-block (ws's score / block-max arrays).
    """
    _stk(4)
    if tid < 768:
        e = tid // 6
        r = tid - e * 6
        v = _wait_tag(slots, (KLL_OFF + e * 64 + r * 8) // 4, epoch)
        _lds_st(lds, K_LDS + r * 512 + e * 4, [v])
    if tid < 256:
        _lds_st(lds, KQ_LDS + tid * 4, [Int32(0)])
    gpu.barrier()
    _stk(5)
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
        if (tid & 7) == 0:
            _lds_st(lds, KQ_LDS + row * 64 + q * 16 + (l32 >> 3) * 4, [word])
    gpu.barrier()
    if (row < 6) & (pos > -1) & (li > -1) & (li < cap):
        if (tid & 7) == 0:
            _st(cache0, li * 16 + q * 4 + (l32 >> 3), word)
        if (q == 0) & (l32 == 0):
            ex = _lds_ld(lds, KE_LDS + row * 16, 4)
            _st(cache1, li, ex[0] | (ex[1] << 8) | (ex[2] << 16) | (ex[3] << 24))
    gpu.barrier()
    _stk(6)
    if rescore & (k_hi > 0) & (tid < 64):
        _rescore(tid, li_p, vpos_p, cache0, cache1, rows, ws, m_cap, qrow, k_hi)
    if small & (tid < 64):
        _newkey_scores(tid, lds, slots, ws, qrow, epoch)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    _stk(7)
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
    """The max over 8 consecutive lanes (xor 1, 2, 4; DPP)."""
    for k in (1, 2, 4):
        v = xred(v, k, fx.max)
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
def _entry(tid, s2, lds, a, rows, pub8, mode2_reader=False):
    """a: the kernarg i64 words. Roles by s2 (TileRT's dispatch): prep, scorers, selectors (module
    docstring); the split / cross-rank path is not ported.
    """
    if const_expr(mode2_reader):
        # The host checks the immutable graph arguments before choosing this
        # program. Constants erase general dispatch, prep and large selectors.
        a = list(a)
        a[7] = (Int64(2) << 32) | (a[7] & Int64(0xFFFFFFFF))
        a[10] = Int64(0)
        a[24] = Int64(0)
    _SPC.update(sp=a[23], s2=s2, tid=tid)
    _stk(0)
    if const_expr(_STAMP):
        if (tid == 960) & (a[23] != Int64(0)) & (s2 < 256):
            t15 = memrealtime()
            bo.buffer_store(fx.Vector.from_elements([Int32(t15), Int32(t15 >> 32)], Int32), rsrc(a[23]),
                            (s2 * 32 + 15) * 2)  # fmt: skip
    wv = tid >> 6
    ratio, mode = Int32(a[7]), Int32(a[7] >> 32)  # ScoreArgs+0x38 / +0x3c
    nsplit, rank, thr = (
        Int32(a[8]),
        Int32(a[8] >> 32),
        Int32(a[9]),
    )  # +0x40 / +0x44 / +0x48
    shift = ratio >> 1
    li_p = a[10]  # ScoreArgs+0x50: latent_idx (null: no prep)
    vpos_p = a[37]  # PrepArgs+0x20
    # the dispatch state, computed by wave 0 alone and handed to the other waves through LDS: it is ~500
    # scalar instructions, and 16 waves running them shared the CU's one scalar unit (the last wave reached
    # the first barrier ~3 us after wave 0)
    if wv == 0:
        if const_expr(mode2_reader):
            eslots_ = a[4] + slots_offset(a[6])
            epoch_word = _ld(eslots_, s2)
            position = _ld(a[2], 0)
            cbh = (a[3] != Int64(0)).select(a[3], a[2])
            header = _ld(cbh, 0, 2)
            rocdl.sched_barrier(0)
            epoch_ = uniform(epoch_word) + 1
            pos0_ = uniform(position)
            n93_ = (pos0_ + 1) >> shift
            n_ = (pos0_ < 0).select(Int32(0), fx.max(n93_, Int32(0)))
            c0_, c1_ = uniform(header[0]), uniform(header[1])
            if tid == 0:
                w = [Int32(eslots_), Int32(eslots_ >> 32), Int32(eslots_), Int32(eslots_ >> 32),
                     epoch_, pos0_, n93_, n_, Int32(0), n_, Int32(0), Int32(0), Int32(-1),
                     c0_, c1_, Int32(0), Int32(-1), Int32(-1), Int32(0), Int32(0)]  # fmt: skip
                for q in range_constexpr(5):
                    _lds_st(lds, DREC + 16 * q, w[4 * q : 4 * q + 4])
        else:
            eslots_ = a[4] + slots_offset(a[6])  # ScoreArgs+0x20 (ws) + C(ScoreArgs+0x30)
            slots_ = a[46] + slots_offset(
                a[47]
            )  # PrepArgs+0x68 (ws) + C(PrepArgs+0x70): the prep's areas
            # every other input the dispatch reads: five vector loads issued back to back (lane r: row r's word),
            # one round trip, then read back by readlane. As scalar loads (s_buffer_load; the kernel is SGPR-starved)
            # the compiler issued them in three rounds, each waiting on the last (epoch's after pos0's after the
            # rows'). A null pointer is swapped for a valid one and its values are never used
            lane = tid & 63
            vpb = (a[37] != Int64(0)).select(a[37], a[2])  # PrepArgs+0x20: vpos
            lib = (a[10] != Int64(0)).select(a[10], a[2])  # ScoreArgs+0x50: latent_idx
            cbh = (a[3] != Int64(0)).select(
                a[3], a[2]
            )  # ScoreArgs+0x18: the cand table's header
            rw = fx.min(lane, Int32(rows - 1))
            ev = _ld(eslots_, s2)
            pv = _ld(a[2], 0)  # ScoreArgs+0x10: the row's position
            vv = _ld(vpb, rw)
            lv = _ld(lib, rw)
            cv = _ld(cbh, fx.min(lane, Int32(1)))
            rocdl.sched_barrier(
                0
            )  # all five issued before the first readlane (SGPR pressure interleaved them)
            epoch_ = uniform(ev) + 1
            pos0_ = uniform(pv)
            vps_ = [readlane(vv, r) for r in range(rows)]
            lis_ = [readlane(lv, r) for r in range(rows)]
            c0_ = readlane(cv, 0)
            c1_ = readlane(cv, 1)
            n93_ = (pos0_ + 1) >> shift
            n_ = (pos0_ < 0).select(Int32(0), fx.max(n93_, Int32(0)))
            split_ = (nsplit > 1) & (n_ > thr) & (mode != 2)
            nosplit_ = (nsplit <= 1) | (n_ <= thr) | (mode == 2)
            ch = (((nsplit + n_ - 1) // fx.max(nsplit, Int32(1))) + 63) & -64
            k_lo_ = split_.select(fx.min(ch * rank, n_), Int32(0))
            k_hi_ = split_.select(fx.min(k_lo_ + ch, n_), n_)
            # the new compressed keys' 8-aligned block range [lo, hi) (TileRT computes it unless latent_idx is
            # null, split or small)
            mn = Int32(0x40000000)
            mx = Int32(-1)
            for r in range_constexpr(rows):
                p = vps_[r]
                ok = (p > -1) & (((p + 1) & (ratio - 1)) == 0)
                k1 = (p + 1) >> shift
                mn = (ok & (k1 <= mn) & (k1 < 0x40000001)).select(k1 - 1, mn)
                mx = ok.select(fx.max(k1 - 1, mx), mx)
            lohi = (li_p != Int64(0)) & nosplit_ & (n_ >= 0x4001) & (mode != 2) & (mx >= 0)
            lo_ = lohi.select(mn & -8, Int32(0))
            hi_ = lohi.select((mx & 0x7FFFFFF8) + 8, Int32(0))
            # the first latent-indexed new key (TileRT's s81: rows with position and latent index, -1 none)
            s81_ = Int32(-1)
            for r in range_constexpr(rows):
                p = vps_[r]
                lr = lis_[r]
                if const_expr(r == 0):
                    s81_ = (p > -1).select(fx.max(lr, Int32(-1)), Int32(-1))
                else:
                    s81_ = ((p | lr) > -1).select(
                        (s81_ < 0).select(lr, fx.min(lr, s81_)), s81_
                    )
            s81_ = (li_p != Int64(0)).select(s81_, Int32(-1))
            if tid == 0:
                w = [Int32(eslots_), Int32(eslots_ >> 32), Int32(slots_), Int32(slots_ >> 32), epoch_, pos0_, n93_, n_,
                     k_lo_, k_hi_, lo_, hi_, s81_, c0_, c1_, Int32(0)] + vps_ + lis_  # fmt: skip
                w = w + [Int32(0)] * (-len(w) % 4)
                for q in range_constexpr(len(w) // 4):
                    _lds_st(lds, DREC + 16 * q, w[4 * q : 4 * q + 4])
    if (tid >= 64) & (
        tid < 208
    ):  # waves 1..3 meanwhile put q in LDS (fp4 rows, e8m0 scales, head weights)
        qi = tid - 64
        qrow = Int32(a[11])  # ScoreArgs+0x58: the q row
        qsrc = (qi < 128).select(qrow * 2048 + qi * 16, (qi < 136).select(
            0x3800 + qrow * 128 + qi * 16, 0x3B80 + qrow * 128 + qi * 16))  # fmt: skip
        qv = _ld(a[4], qsrc >> 2, 4)
        _lds_st(lds, Q_LDS + qi * 16, [qv[0], qv[1], qv[2], qv[3]])
    gpu.barrier()
    rec = []
    for q in range_constexpr((16 + 2 * rows + 3) // 4):
        v4 = _lds_ld(lds, DREC + 16 * q, 4)
        rec += [uniform(v4[0]), uniform(v4[1]), uniform(v4[2]), uniform(v4[3])]
    eslots = (Int64(rec[1]) << 32) | (Int64(rec[0]) & Int64(0xFFFFFFFF))
    slots = (Int64(rec[3]) << 32) | (Int64(rec[2]) & Int64(0xFFFFFFFF))
    if const_expr(mode2_reader):
        epoch, pos0, n93, n = rec[4:8]
        k_lo, k_hi, lo, hi, s81 = Int32(0), n, Int32(0), Int32(0), Int32(-1)
    else:
        epoch, pos0, n93, n, k_lo, k_hi, lo, hi, s81 = rec[4:13]
    c01 = rec[13:15]
    vps = rec[16 : 16 + rows]
    lis = rec[16 + rows : 16 + 2 * rows]
    _stk(28)
    nosplit = (nsplit <= 1) | (n <= thr) | (mode == 2)
    small = nosplit & ((n < 0x4001) | (mode == 2))
    prep = (s81 > -1) & ((s2 & 7) == 7) & (s2 < 72)
    _stk(30)
    if prep & (hi <= lo):
        if s2 < 64:
            k_gemv(tid, s2, slots, a[35], a[41], epoch, rows)
        else:
            k_prep(tid, slots, lds, a[36], vpos_p, a[42], a[43], Int32(a[48]), Int32(a[48] >> 32),
                   a[44], a[45], Int32(a[49]), epoch, rows, hi > lo, a[46], a[47], Int32(a[11]),
                   k_hi, small)  # fmt: skip
    if n > 0x200:
        c0 = Int32(0)
        c1 = Int32(0)
        if (
            mode == 2
        ):  # ScoreArgs+0x18: the candidate-block table {tag, count, blocks..}
            c0 = c01[0]
            c1 = c01[1]
        cand = (mode == 2) & (pos0 > -1) & (n93 > 0) & (c0 == 2)
        act = (
            s2 < 0
        )  # small: every CTA (scorer_small); else 16 waves of the first ncta CTAs
        t0 = s2 * 16 + wv
        ntile = (k_hi - k_lo + 15) >> 4
        if small:
            act = s2 >= 0  # (every CTA; see scorer_small)
        else:  # (the big candidate-block path, cand and not small, is not ported: no CTA scores)
            nocand = (c0 != 2) | (mode != 2) | (pos0 < 0) | (n93 <= 0)
            act = (s2 < fx.min((ntile + 15) >> 4, Int32(256))) & nocand
        if act:
            gpu.barrier()  # q is in LDS (stored at entry)
            _stk(1)
            if small:
                n62 = (pos0 < 0).select(Int32(0), n93)
                kc = cand.select(c1 * 8, n62)
                s33 = pos0
                if li_p != Int64(0):
                    s33 = lis[0]
                s63 = ((li_p != Int64(0)) & (pos0 > -1)).select(
                    fx.max(s33, Int32(-1)), Int32(-1)
                )
                excls = [((s63 < 0) | cand | (pos0 < 0)).select(Int32(-2), s33)]
                if (
                    rows > 1
                ):  # TileRT 4027 L2041..L2075: excl_r = (cand | s81 < 0 | vpos_r < 0) ? -2 : li_r
                    excls = []
                    for r in range_constexpr(rows):
                        e = Int32(-2)
                        if li_p != Int64(0):
                            e = (cand | (s81 < 0) | (vps[r] < 0)).select(
                                Int32(-2), lis[r]
                            )
                        excls.append(e)
                scorer_small(tid, lds, a, eslots, epoch, t0, kc, cand, n62, excls)
            else:
                nmin = fx.min((pos0 < 0).select(Int32(0), n93), k_hi)
                n_smp = ((mode < 2) & (mode >= 0) & (nmin > k_lo)).select(
                    nmin - k_lo, Int32(0)
                )
                smp_div = (n_smp > 0).select((n_smp + 0xFFF) >> 12, Int32(1))
                scorer(tid, s2, lds, a, a[4], eslots, epoch, k_lo, k_hi, lo, hi, s81, t0, Int32(0x1000),
                       ntile, n_smp, smp_div, (mode == 1) & (n > 0x4000))  # fmt: skip
            _stk(2)
    # every wave's score / block-max stores complete before tid 0 publishes the slot (TileRT L2490..L2517:
    # vmcnt(0), barrier, slot store; without the wait a selector could read a block max not yet written)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        _st(eslots, s2, epoch)
    _stk(3)
    if small & (n > 0x200) & (s2 == 0xF8):
        c0 = Int32(0)
        c1 = Int32(0)
        if mode == 2:
            c0 = c01[0]
            c1 = c01[1]
        cand = (mode == 2) & (pos0 > -1) & (n93 > 0) & (c0 == 2)
        n62 = (pos0 < 0).select(Int32(0), n93)
        kc = cand.select(c1 * 8, n62)
        select_small(tid, lds, a, eslots, epoch, kc, cand, n62, n62, pub8, rows, mode)
    if small & (n <= 0x200) & (s2 > 239):
        select_trivial(
            tid,
            a,
            eslots,
            epoch,
            (pos0 < 0).select(Int32(0), n93),
            pub8,
            s2 - 240,
            mode,
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
    # the narrow layout (n_local <= 0x20000, dec_narrow 3): selector CTAs 240..255, c = s2 - 240; even c
    # filters token chunk c / 2 (c = 0 merges and publishes), mode 1: c = 1 the single-chunk block level
    if nosplit & (mode != 2) & (n >= 0x4001) & (nloc <= 0x20000) & (s2 > 239):
        cn = s2 - 240
        nckn = _lshr(nloc + 0x3FFF, Int32(14))
        if ((cn & 1) == 0) & ((cn >> 1) < nckn):
            sel_token_level(tid, lds, a, eslots, epoch, cn >> 1, pos0, n93, k_lo, k_hi, nloc, lo, hi, pub8,
                            4)  # fmt: skip
        if (mode == 1) & (cn == 1):
            sel_block_single(
                tid, lds, a, eslots, epoch, pos0, n93, k_lo, k_hi, nloc, lo, hi
            )
    if prep & (hi > lo):
        if s2 < 64:
            k_gemv(tid, s2, slots, a[35], a[41], epoch, rows)
        else:
            k_prep(tid, slots, lds, a[36], vpos_p, a[42], a[43], Int32(a[48]), Int32(a[48] >> 32),
                   a[44], a[45], Int32(a[49]), epoch, rows, hi > lo, a[46], a[47], Int32(a[11]),
                   k_hi, small)  # fmt: skip
    _stk(31)


def build(key=None):
    return _build(key, mode2_reader=False)


def build_mode2(key=None):
    """One-row mode-2 reader; the host retains the general kernel for other arguments."""
    return _build(key, mode2_reader=True)


def _build(key, mode2_reader):
    """(kernel symbol, launcher) of the port; the launcher takes the variant's kernarg words (50, the
    engram variant 60)."""
    key = _KEY if key is None else key
    kr, rows, eng = VARIANTS[key]
    if mode2_reader and (rows != 1 or eng):
        raise ValueError("mode2 reader requires a one-row non-engram variant")
    pub8 = kr in (
        4027,
        499,
    )  # TileRT's KR: 4027 / 499 publish the top-k to 8 LL targets, 2747 / 243 to 4
    name = kernel_symbol(
        "tk_indexer_eng" if eng else "tk_indexer", kr=kr, rows=rows, v=VERSION + ("_mode2" if mode2_reader else "")
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
        _entry(fx.thread_idx.x, fx.block_idx.x, lds, [a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49], rows, pub8, mode2_reader)  # fmt: skip

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
