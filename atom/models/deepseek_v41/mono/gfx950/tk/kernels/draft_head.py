# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::dspark_head::dspark_head_sample_kernel<5, false, 8>``.
"""The DSpark draft head with its sequential Markov sampler on gfx950 (TP8) as a FlyDSL port of TileRT's
``dspark_head_sample`` kernel: kernarg ABI unchanged (328 B read by offset as 41 i64 words), grid 256 x 512
(``runs/20261115-flydsl-tilert/m3/dspark_head_notes.md``), diffed bit-exact by ``m2/x_replay8.py``.

1. U = head_w . h for the rank's 16160 vocab rows (1010 tiles of 16, CTA s2 the tiles s2, s2 + 256, ..)
   and the five draft rows, bf16 MFMA, kept in LDS.
2. For k = 0 .. 4: logit_k = U_k + W1[x_{k-1}] . W2^T (Markov rank 256), x_{-1} = blk_tok[0]; draft_z
   gets the logits; score = logit / T + Gumbel noise from a murmur3 hash of (seed, k, vpos[k], vocab id);
   the arg max over the 8 ranks' 129280 entries (CTA -> eight 32-CTA groups -> all ranks through peer LL
   slots) is x_k -> vtok[k]; X[k] = embed[x_{k-1}] in its four hc copies (X[5] = embed[x_4] at the end).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import _call, bperm, f32c
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.layer_tail import (
    _ld,
    _ldm,
    _lds_ld,
    _lds_st,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    kernel_symbol,
    launcher,
    mfma_bf16,
    readlane,
    rsrc,
    spin_pause,
    traced,
    uniform,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 41
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
GRID, THREADS = 256, 512
ROWS, VOCAB, VOCAB_RANK, TILES, RANK_DIM = 5, 129280, 16160, 1010, 256
VERSION = "v1"

# LDS (bytes)
X_LDS, X_ROW = 0, 0x2810  # the five draft rows of h (bf16, 16 B row padding)
RED_LDS = 51280  # GEMV: waves 1..7's partials, 1 KB a wave
U_LDS = 0xE450  # U[k][local row 0..63] (f32)
KEY_LDS = 59728  # the waves' arg max keys {lo, hi}
TOK_LDS = 59792  # the round's token
LDS_BYTES = 59800

ROW_SHR, ROW_BCAST15, ROW_BCAST31 = 0x110, 0x142, 0x143
C1, C2 = 0xCC9E2D51 - (1 << 32), 0x1B873593  # murmur3
N1 = 0xE6546B64 - (1 << 32)
F1, F2 = 0x85EBCA6B - (1 << 32), 0xC2B2AE35 - (1 << 32)
NEG_MAX = 0xFF7FFFFF  # -FLT_MAX: an empty row's score
PEER_REGION, PEER_K, PEER_PAR = (
    1024,
    8192,
    0x10000,
)  # the peers' {score, tag} {id, tag} slots
OWN_PEER_OFF = 0x8000  # a single rank's own slots: ws + this


def _lsr(x, n):
    return (x >> n) & ((1 << (32 - n)) - 1)


def _rotl(x, r):
    return (x << r) | _lsr(x, 32 - r)


def _kmix(d):
    """murmur3's block mix: rotl15(d c1) c2."""
    return _rotl(d * Int32(C1), 15) * Int32(C2)


def _hstep(h, k):
    return _rotl(h ^ k, 13) * 5 + Int32(N1)


def _log(x):
    return F32(_call("llvm.log.f32", T.f32, x))


def _st_i(addr, word, v, cm=0):
    bo.buffer_store(Int32(v), rsrc(addr), word, cache_modifier=cm)


def _u_gt(a, b):
    return (a ^ Int32(-(1 << 31))) > (b ^ Int32(-(1 << 31)))


def _k_gt(ah, al, bh, bl):
    """(ah, al) > (bh, bl) as one unsigned 64-bit key (hi word first)."""
    return _u_gt(ah, bh) | ((ah == bh) & _u_gt(al, bl))


def _dpp0(v, ctrl):
    return Int32(update_dpp_i32(Int32(0), Int32(v), ctrl, 0xF, 0xF, True))


def _wave_kmax(hi, lo):
    """The wave's max key: DPP row_shr 1/2/4/8, row_bcast 15/31 (absent sources read 0), lane 63's."""
    for c in (
        ROW_SHR + 1,
        ROW_SHR + 2,
        ROW_SHR + 4,
        ROW_SHR + 8,
        ROW_BCAST15,
        ROW_BCAST31,
    ):
        oh, ol = _dpp0(hi, c), _dpp0(lo, c)
        g = _k_gt(oh, ol, hi, lo)
        hi, lo = g.select(oh, hi), g.select(ol, lo)
    return uniform(readlane(hi, 63)), uniform(readlane(lo, 63))


def _key_hi(sbits):
    """A score's sortable high key word (-0 as +0): negative ~bits, else bits | sign."""
    s = (sbits == Int32(-(1 << 31))).select(Int32(0), sbits)
    return (s < 0).select(~s, s | Int32(-(1 << 31)))


def _unkey(hi):
    return (hi < 0).select(hi & 0x7FFFFFFF, ~hi)


def _wfetch(tid, s2, j, cnt, head_w):
    """Tile j's 20 weight loads for this wave (nt), out of bounds (no traffic) past the CTA's tiles."""
    lane, w = tid % 64, tid // 64
    tile = s2 + 256 * j
    ok = j < cnt
    return [_ldm(head_w, TILES * 160 * 1024, (((tile * 160 + w * 20 + ks) << 10) + lane * 16) >> 2, ok, 4, CM_NT)
            for ks in range(20)]  # fmt: skip


@traced
def _gemv(tid, s2, cnt, head_w, lds, pre):
    """U[k][16j + r] for the CTA's tiles j < cnt: wave w the K range 640w..+640 in 20 chained 16x16x32
    bf16 MFMAs (A: head_w's packed 1 KB tiles, nt; B: the draft rows from LDS, rows h > 4 read row 0),
    waves summed 0 + 1 + .. + 7 -> U_LDS. Tile j + 1's weights are in flight while tile j computes
    (``pre``: tile 0's, issued first thing in the kernel)."""
    lane, w = tid % 64, tid // 64
    h, q = lane % 16, lane >> 4
    xo = X_LDS + (h > 4).select(Int32(0), h) * X_ROW + w * 0x500 + q * 16
    cur = pre
    for j in range_constexpr(4):
        nxt = _wfetch(tid, s2, j + 1, cnt, head_w) if j < 3 else None
        if j < cnt:
            acc = fx.Vector.filled(4, 0.0, F32)
            for ks in range_constexpr(20):
                b = _lds_ld(lds, xo + ks * 64, 4)
                acc = mfma_bf16(
                    cur[ks].bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), acc
                )
            if w != 0:
                for i in range_constexpr(4):
                    _lds_st(
                        lds,
                        RED_LDS + (w - 1) * 1024 + i * 256 + lane * 4,
                        [acc[i].bitcast(Int32)],
                    )
            gpu.barrier()
            if (w == 0) & (h < 5):
                c = []
                for i in range_constexpr(4):
                    s = acc[i]
                    for ww in range_constexpr(1, 8):
                        s = s + _lds_ld(
                            lds, RED_LDS + (ww - 1) * 1024 + i * 256 + lane * 4
                        ).bitcast(F32)
                    c.append(s.bitcast(Int32))
                _lds_st(lds, U_LDS + h * 256 + j * 64 + q * 16, c)
            gpu.barrier()
        cur = nxt


@traced
def _gumbel_score(logit, temp, h3, kid):
    """logit / T - log(-L1), L1 = log1p(-u) (a 7-term series below u = 0.25, else log(max(1 - u, 2^-24))),
    u in (0, 1) from murmur3(h3; kid) finalized (len 16): hi16 2^-16 + (lo16 + 0.5) 2^-32.
    """
    hh = _hstep(h3, kid)
    hh = hh ^ 16
    hh = hh ^ _lsr(hh, 16)
    hh = hh * Int32(F1)
    hh = hh ^ _lsr(hh, 13)
    hh = hh * Int32(F2)
    hi = _lsr(hh, 16)
    lo = (hh ^ hi) & 0xFFFF
    u = fmath.fma(
        hi.to(F32), f32c(0x37800000), (lo.to(F32) + F32(0.5)) * f32c(0x2F800000)
    )
    a = fx.max(F32(1.0) - u, f32c(0x33800000))
    p = fmath.fma(u, F32(0.125), f32c(0x3E124925))
    p = fmath.fma(u, p, f32c(0x3E2AAAAB))
    p = fmath.fma(u, p, f32c(0x3E4CCCCD))
    p = fmath.fma(u, p, F32(0.25))
    p = fmath.fma(u, p, f32c(0x3EAAAAAB))
    p = fmath.fma(u, p, F32(0.5))
    p = fmath.fma(u, p, F32(1.0))
    l1 = (u < F32(0.25)).select(p * (-u), _log(a))
    return logit / temp - _log(-l1)


@traced
def _x_copy(tid, s2, k, tok, embed, xout):
    """X[k] = embed[tok] (0 for an id past the vocabulary) in its four hc copies: wave 1, lanes < 3,
    16 B chunk s2 + 256 lane of the 640."""
    lane, w = tid % 64, tid // 64
    c = s2 + 256 * lane
    if (w == 1) & (lane < 3) & (c < 640):
        ok = _u_gt(Int32(VOCAB), tok)
        v = fx.Vector.filled(4, 0, Int32)
        if ok:
            v = _ld(embed, (tok * 10240 + c * 16) >> 2, 4)
        for hc in range_constexpr(4):
            bo.buffer_store(v, rsrc(xout), (k * 0xA000 + hc * 0x2800 + c * 16) >> 2)


@traced
def _entry(tid, s2, lds, xn, head_w, w1, w2, blk_tok, nk, ws, tab, me, npes, ep, vtok, dz, embed, xout,
           samp, vpos):  # fmt: skip
    lane, w = tid % 64, tid // 64
    r, seg = tid >> 3, tid & 7
    cnt = (1009 - s2) // 256 + 1
    pre = _wfetch(
        tid, s2, 0, cnt, head_w
    )  # tile 0's weights first, as TileRT's prologue
    valid = (r >> 4) < cnt
    row_l = (s2 + 256 * (r >> 4)) * 16 + (r & 15)
    vid = valid.select(me * VOCAB_RANK + row_l, Int32(0x7FFFFFFF))
    kid = _kmix(vid)
    klo = ~vid
    tag = (ep << 1) | 1
    # the peers' slot bases, read once (TileRT's prologue; per round they cost a dependent load each)
    pv = [_ldm(tab, 64, 2 * pp, npes > 1, 2) for pp in range(8)]
    peers = [(Int64(uniform(v[1])) << 32) | Int64(fx.Uint32(uniform(v[0]))) for v in pv]
    ownv = _ldm(tab, 64, 2 * fx.max(me, Int32(0)), npes > 1, 2)
    own = (Int64(uniform(ownv[1])) << 32) | Int64(fx.Uint32(uniform(ownv[0])))
    par = (ep & 1) * PEER_PAR
    # the draft rows into LDS (i = tid + 512 j: row i / 640, 16 B chunk i % 640)
    for jj in range_constexpr(7):
        i = tid + 512 * jj
        if i < ROWS * 640:
            v = _ld(xn, i * 4, 4)
            _lds_st(
                lds,
                X_LDS + (i // 640) * X_ROW + (i % 640) * 16,
                [v[d] for d in range(4)],
            )
    # W2's 32 elements of this thread's row segment, the anchor's W1 segment
    w2v = _ld(w2, (valid.select(row_l, Int32(0)) * 512 + seg * 64) >> 2, 4)
    w2s = [w2v]
    for o in range_constexpr(1, 4):
        w2s.append(
            _ld(w2, (valid.select(row_l, Int32(0)) * 512 + seg * 64 + o * 16) >> 2, 4)
        )
    w2f = []
    for o in range_constexpr(4):
        for d in range_constexpr(4):
            w2f += [
                (w2s[o][d] << 16).bitcast(F32),
                (w2s[o][d] & Int32(-65536)).bitcast(F32),
            ]
    temp = _ld(samp, 2).bitcast(F32)
    h1 = _hstep(Int32(0), _kmix(_ld(samp, 0)))
    s1 = _ld(samp, 1)
    # the five positions up front: read in the round they are a dependent load each (TileRT issues them with W1's)
    vp = [_ld(vpos, kk) for kk in range(ROWS)]
    gpu.barrier()
    _gemv(tid, s2, cnt, head_w, lds, pre)
    tok = uniform(_ld(blk_tok, 0))
    for k in range_constexpr(ROWS):
        if k < nk:
            xr = _u_gt(Int32(VOCAB), tok).select(tok, Int32(0))
            w1f = []
            for o in range_constexpr(4):
                v = _ld(w1, (xr * 512 + seg * 64 + o * 16) >> 2, 4)
                for d in range_constexpr(4):
                    w1f += [
                        (v[d] << 16).bitcast(F32),
                        (v[d] & Int32(-65536)).bitcast(F32),
                    ]
            bias = fmath.fma(w2f[0], w1f[0], F32(0.0))
            for e in range_constexpr(1, 32):
                bias = fmath.fma(w2f[e], w1f[e], bias)
            for off in (1, 2, 4):
                bias = bias + bperm(bias, lane ^ off)
            logit = bias + _lds_ld(lds, U_LDS + k * 256 + r * 4).bitcast(F32)
            if valid & (seg == 0) & (dz != 0):
                bo.buffer_store(logit, rsrc(dz), k * VOCAB_RANK + row_l)
            score = f32c(NEG_MAX)
            if valid:
                h2 = _hstep(h1, _kmix(Int32(k) ^ s1))
                # TileRT's position block: vpos[k] c1 + 0x40000000 before the rotation (s_add 2.0 in its ISA)
                kv = vp[k] * Int32(C1) + Int32(0x40000000)
                h3 = _hstep(h2, _rotl(kv, 15) * Int32(C2))
                score = _gumbel_score(logit, temp, h3, kid)
            kh, kl = _wave_kmax(_key_hi(score.bitcast(Int32)), klo)
            if lane == 0:
                _lds_st(lds, KEY_LDS + w * 8, [kl, kh])
            _x_copy(tid, s2, k, tok, embed, xout)
            gpu.barrier()
            if tid == 0:
                ch, cl = _lds_ld(lds, KEY_LDS + 4), _lds_ld(lds, KEY_LDS)
                for ww in range_constexpr(1, 8):
                    oh, ol = _lds_ld(lds, KEY_LDS + ww * 8 + 4), _lds_ld(
                        lds, KEY_LDS + ww * 8
                    )
                    g = _k_gt(oh, ol, ch, cl)
                    ch, cl = g.select(oh, ch), g.select(ol, cl)
                at = (k * 4096 + (s2 % 8) * 512 + (s2 // 8) * 16) >> 2
                bo.buffer_store(
                    fx.Vector.from_elements([_unkey(ch), tag], Int32),
                    rsrc(ws),
                    at,
                    cache_modifier=CM_DEV,
                )
                bo.buffer_store(
                    fx.Vector.from_elements([~cl, tag], Int32),
                    rsrc(ws),
                    at + 2,
                    cache_modifier=CM_DEV,
                )
            if (s2 < 8) & (w == 0):
                ph = _key_hi(Int32(NEG_MAX))
                pl = Int32(-(1 << 31))
                if lane < 32:
                    lw = (k * 4096 + s2 * 512 + lane * 16) >> 2
                    a0 = _ld(ws, lw, 2, CM_DEV)
                    a1 = _ld(ws, lw + 2, 2, CM_DEV)
                    while ~((a0[1] == tag) & (a1[1] == tag)):
                        spin_pause()
                        a0 = _ld(ws, lw, 2, CM_DEV)
                        a1 = _ld(ws, lw + 2, 2, CM_DEV)
                    ph = _key_hi(a0[0])
                    pl = ~a1[0]
                gh, gl = _wave_kmax(ph, pl)
                for p in range_constexpr(8):
                    if (npes > p) & ((lane >> 3) == p):
                        off = (
                            par
                            + k * PEER_K
                            + (lane & 7) * PEER_REGION
                            + me * 128
                            + s2 * 16
                        )
                        if npes > 1:
                            base = peers[p]
                            bo.buffer_store(fx.Vector.from_elements([_unkey(gh), tag, ~gl, tag], Int32), rsrc(base),
                                            off >> 2, cache_modifier=CM_SYS)  # fmt: skip
                        else:
                            bo.buffer_store(fx.Vector.from_elements([_unkey(gh), tag, ~gl, tag], Int32), rsrc(ws),
                                            (OWN_PEER_OFF + off) >> 2, cache_modifier=CM_SYS)  # fmt: skip
            if w == 0:
                fh = _key_hi(Int32(NEG_MAX))
                fl = Int32(-(1 << 31))
                if lane < npes * 8:
                    ro = par + k * PEER_K + (s2 % 8) * PEER_REGION + lane * 16
                    b4 = fx.Vector.filled(4, 0, Int32)
                    if npes > 1:
                        b4 = _ld(own, ro >> 2, 4, CM_SYS)
                        while ~((b4[1] == tag) & (b4[3] == tag)):
                            spin_pause()
                            b4 = _ld(own, ro >> 2, 4, CM_SYS)
                    else:
                        b4 = _ld(ws, (OWN_PEER_OFF + ro) >> 2, 4, CM_SYS)
                        while ~((b4[1] == tag) & (b4[3] == tag)):
                            spin_pause()
                            b4 = _ld(ws, (OWN_PEER_OFF + ro) >> 2, 4, CM_SYS)
                    fh = _key_hi(b4[0])
                    fl = ~b4[2]
                _, tl = _wave_kmax(fh, fl)
                if lane == 0:
                    _lds_st(lds, TOK_LDS, [~tl])
                    if s2 == 0:
                        _st_i(vtok, k, ~tl)
            gpu.barrier()
            tok = uniform(_lds_ld(lds, TOK_LDS))
    _x_copy(tid, s2, nk, tok, embed, xout)


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 41 kernarg words."""
    name = kernel_symbol("tk_dspark_head", rows=ROWS, v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_BYTES // 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dspark_head(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _entry(fx.thread_idx.x, fx.block_idx.x, lds, a00, a01, a02, a03, a04, Int32(a05), a06, a07, Int32(a08),
               Int32(a08 >> 32), Int32(a09), a10, a11, a12, a13, a39, a40)  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        dspark_head(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
            a32, a33, a34, a35, a36, a37, a38, a39, a40,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
