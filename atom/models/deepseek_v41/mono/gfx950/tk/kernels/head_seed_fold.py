# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::head_seed_fold::head_seed_fold_kernel<6, false, 8, false>``.
"""The target head, its sampler / acceptance fold and the next round's DSpark seed on gfx950 (TP8), a FlyDSL
port of TileRT's ``head_seed_fold`` kernel: kernarg ABI unchanged (792 B read as 99 i64 words), grid 487 x
512 (``runs/20261115-flydsl-tilert/m3/head_seed_fold_notes.md``), diffed by ``m2/x_replay8.py``.

Kernarg (byte offsets): 0x000 Xn, 0x008 head_w, 0x010 logits; 0x0e8.. ``main_proj::SeedArgs`` (SA = 0xe8:
SA + 0x00 cap, + 0x10 main_w, + 0x18 main_w_s, + 0x20 main_norm, + 0x28 ws_main_proj, + 0x30
tab_main_proj, + 0x40 epoch, + 0x50 / 0x68 / 0x80 the three MTP layers' window kv_w / kv_s / kv_norm, +
0x98 / 0xb0 win.{40,41,42} kv / scales, + 0xc8 rope0, + 0xd0 vpos, + 0xd8 embed, + 0xe0 blk_tok, + 0xe8 X,
+ 0xf4 the placeholder token, + 0xf8 an optional {value, tag} list to wait for, + 0x100 its length | tag
<< 32); 0x1f8.. ``sampler::Args``; 0x2b8.. ``mtp_ctl::AcceptArgs``.

Roles by CTA (all ported; the whole kernel diffed bit-exact on 8 ranks at the 400K state both with
simulated acceptance 3.51 (TILERT_SIMULATE_ACC_LEN, the residual branch taken) and with real acceptance
(kernarg 0x314 = 0)):
- 0..255: head GEMV (logits, / T into LDS) and the per-CTA sampler partials (``_head``, ``_stats``); CTA 255
  folds them: the 17 entries a rank to every rank's sym:sample, the 8-rank merge, draft log ratios, Philox
  coins, the accept count, the bonus token or the residual mask (``_fold_*``); every CTA's residual pass
  (``_residual``) and CTA 255's residual merge across CTAs and ranks, tok_out / mtp_ctl write-back and X row
  0 (``_fold_residual``, ``_fold_out``).
- 256..455: main_proj of the six captured rows (``_main_proj``); 256..447 then the 3 MTP layers' window-kv
  GEMV (``_window_kv``); 456..461 main_norm (``_main_norm``); 462..479 window kv RMSNorm / RoPE / E4M3 into
  win.40..42 (``_window_norm``); 480..486 X rows 1..7 = embed[placeholder token] (``_embed_rows``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import (
    _call,
    bf_rne_hi,
    bperm,
    f32c,
    log1p_ocml,
    spin_pause,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.draft_head import (
    F1,
    F2,
    _gumbel_score,
    _hstep,
    _k_gt,
    _key_hi,
    _kmix,
    _lsr,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.engram_fused import (
    _fp8x8_bf16_scaled,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.layer_tail import (
    OOB,
    QX1,
    QX2,
    _dppf,
    _fp8x2,
    _ld,
    _ldm,
    _lds_ld,
    _lds_st,
    _rsq_scaled,
    _scan_sum,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    hw_exp2,
    kernel_symbol,
    launcher,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 99
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
GRID, THREADS = 487, 512
VERSION = "v12"
VOCAB = 0x1F900
HIDDEN_B = 0x2800  # one bf16 hidden row (5120)
X_ROW_B = 0xA000  # X: four mHC copies a row
SA = 0xE8 // 8  # SeedArgs (word index)
EMBED_CTA0 = 480
WAIT_LAST = 484
NB = 1 << 30
assert OOB * 4 > NB
LDS_BYTES = 75872

# head GEMV (CTAs 0..255: the tiles s2, s2 + 256, .. < 1010 of 16 vocab rows, the rank's 16160)
HEAD_CTAS, TILES, VOCAB_RANK = 256, 1010, 16160
XN_ROW = 0x2810  # LDS: the six final-normed rows (bf16, 16 B padding)
HRED_LDS = (
    0x60 + 236 * 256
)  # waves 1..7's accumulators: component i at + 256 i + 1024 w + 4 lane
LG_LDS = 0x11B00  # the CTA's logits / T, [row h][64] f32 (tile j at 16 j)

# sampler statistics (CTAs 0..255) -> ws_sample
STAT_OFF = 0x1600  # {value, tag} at + 4096 item + 2048 half + 8 s2 (items 0..5 rows, 6..10 drafts, 11 key)
VER_OFF = 0x1300  # {p logit / T, tag, q logit, tag} of draft j at + 16 j (from the CTA owning the token)
ITEM_LDS = 0x12100  # CTA 255: the rank's merged item k (M, s) / key (lo, hi) at + 8 k
ENTRIES = 17  # per rank and step parity in every rank's sym:sample: items 0..11 + 5 draft (p, q) slots
RX_LDS = 0x12160  # CTA 255: the 8 ranks' entries, rank r slot k at + 136 r + 8 k
RES_OFF, MASK_OFF = (
    0x1240,
    0x1200,
)  # ws_sample: residual rows' {LSE_p, LSE_q, e_prev, acc_prev} (32 B a row); mask
RROW_LDS, MASK_LDS = (
    0x125A0,
    0x12690,
)  # residual rows' {LSE_p, LSE_q, e_prev, acc_prev} (16 B a row); mask
RSUM_LDS, RKEY_LDS = (
    0x12600,
    0x12630,
)  # per residual row: the CTA's u64 fixed-point sum / max Gumbel key
RPART_OFF = 0xD600  # ws_sample: per-CTA residual partials, {half, tag} at + 2048 (4 row + 2 kind + half) + 8 s2
RX2_OFF = 0xC6000  # sym:sample: the ranks' residual row totals, {sum lo, sum hi, key lo, key hi} x {v, tag}
RSCR_LDS = 0x12700  # CTA 255: the 4 waves' residual partials of a row group (scratch)
VPOS_LDS = (
    0x12660  # not used by the port (kernarg reads); TileRT's LDS copy of vpos / vtok
)

# main_proj (CTAs 256..455 = 40 slices of 16 outputs x 5 sub-slices of K 3072; K = 3 x 5120 captured)
MP_CTA0, MP_SUBS, MP_K = 256, 5, 3072
ACT_ROW = (
    0x1810  # LDS: the six captured rows' fake-quantized K slice (bf16, 16 B padding)
)
RED_LDS = (
    0x60 + 0x9000
)  # the 8 waves' accumulators: component i at + 256 i + 1024 w + 4 lane
OWN_LDS = 45152  # the sub-slice's 96 f32 sums, [seed row n][16 outputs]
PART_LDS = 45536  # sub-slices 1..4's partials, 384 B each
OUT_LDS = 47072  # the slice's 96 bf16
SLICE_UNITS, SUB_UNITS = 0x78, 24  # 2 KiB weight units (16 outputs x 128 K, fp8)
MP_PAR = 0x28000  # tab_main_proj: the odd-epoch half (8 ranks x 40 slices x 512 B)

# main_norm (CTAs 456..461, one seed row each)
NORM_CTA0 = 456
NSUM_LDS = 10240  # the 8 waves' sums of squares (the row's bf16 at LDS 0..10240)
Q8_LDS, QS_LDS = 10272, 15392  # the row's E4M3 bytes, then its 160 ue8m0 bytes
SEED_Q8, SEED_ROW_B = 0x28000, 0x1500  # ws_main_proj: the six quantized seed rows
ROWS_DONE = 0x32800  # ws_main_proj: a row's {tag} once its quantized seed is out
EPS = 0x1E3CE508  # 1e-20

# window kv (CTAs 256..447 after main_proj: 3 MTP layers x 32 tiles of 16 rows x 2 row halves, K 5120)
WKV_CTAS = 192
SEED_LDS_S = 61440  # the six seed rows' ue8m0 scales as f32 [n][160] (their bf16 values at LDS 0, 0x2800 a row)
WRED_LDS = 64256  # waves 1..7's accumulators: component i at + 256 i + 1024 w + 4 lane
WIN_CTA0 = 462  # window kv norm / RoPE / E4M3 (3 layers x 6 rows)
WIN_SLOTS = (
    256  # the window caches' ring (slot = position & 255): 512 E4M3 + 16 ue8m0 a slot
)
WKV_OUT = 0x32C00  # ws_main_proj: the window kv rows, {2 bf16, tag} x 2 at L 0x4000 + n 0x800 + tile 64 + q 16


def _w(a, byte):
    """The kernarg word at a byte offset (8-aligned)."""
    assert byte % 8 == 0
    return a[byte // 8]


@traced
def _wait_list(lane, ptr, nw):
    """Wave 0: every {value, tag} entry i < n of the list at ``ptr`` carries ``tag`` (two entries a lane)."""
    n, tag = Int32(nw), Int32(nw >> 32)
    ok0, ok1 = lane < n, (lane + 64) < n
    t0 = Int32(_ldm(ptr, NB, lane * 2 + 1, ok0, 1, CM_DEV))
    t1 = Int32(_ldm(ptr, NB, (lane + 64) * 2 + 1, ok1, 1, CM_DEV))
    done = ((~ok0) | (t0 == tag)) & ((~ok1) | (t1 == tag))
    while ~done:
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
        rocdl.s_sleep(1)
        n0 = ok0 & (t0 != tag)
        n1 = ok1 & (t1 != tag)
        t0 = n0.select(Int32(_ldm(ptr, NB, lane * 2 + 1, n0, 1, CM_DEV)), t0)
        t1 = n1.select(Int32(_ldm(ptr, NB, (lane + 64) * 2 + 1, n1, 1, CM_DEV)), t1)
        done = ((~ok0) | (t0 == tag)) & ((~ok1) | (t1 == tag))


@traced
def _embed_rows(tid, cta, a):
    """CTAs 480..486: X[cta - 479] = embed[placeholder token] in all four mHC copies."""
    ptr, nw = _w(a, 0xE8 + 0xF8), _w(a, 0xE8 + 0x100)
    if (ptr != 0) & (cta <= WAIT_LAST):
        if tid < 64:
            _wait_list(tid, ptr, nw)
        gpu.barrier()
    tok = Int32(_w(a, 0xE8 + 0xF0) >> 32)
    live = (tok >= 0) & (tok < VOCAB)
    tok = live.select(tok, Int32(0))
    src = _w(a, 0xE8 + 0xD8) + tok.to(Int64) * HIDDEN_B
    dst = _w(a, 0xE8 + 0xE8) + (cta - (EMBED_CTA0 - 1)).to(Int64) * X_ROW_B
    zero = fx.Vector.from_elements([Int32(0)] * 4, Int32)
    for k in range_constexpr(2):
        idx = tid + 512 * k
        if idx < HIDDEN_B // 16:
            v = live.select(
                fx.Vector(bo.buffer_load(rsrc(src), idx * 4, vec_width=4, dtype=T.i32)),
                zero,
            )
            for h in range_constexpr(4):
                bo.buffer_store(v, rsrc(dst), (h * HIDDEN_B) // 4 + idx * 4)


def _bf_pack(lo, hi):
    """Two f32 -> bf16 RNE (no NaN fixup) packed lo | hi << 16."""
    return (bf_rne_hi(hi) & Int32(-65536)) | ((bf_rne_hi(lo) >> 16) & 0xFFFF)


def _fq_chunk(x4, wsb):
    """A lane's 8 bf16 (4 dwords) of a 32-wide group (4 lanes): E4M3 fake-quant with the group's pow2 scale
    2^e (e = ceil log2(max(amax, 1e-4) / 448)), back to bf16 x 2^(e + wsb - 127), RNE.
    """
    xs = []
    for d in x4:
        xs += [(d << 16).bitcast(F32), (d & Int32(-65536)).bitcast(F32)]
    m = abs(xs[0])
    for k in range_constexpr(1, 8):
        m = fx.max(m, abs(xs[k]))
    m = fx.max(m, _dppf(m, QX1))
    m = fx.max(fx.max(m, _dppf(m, QX2)), f32c(0x38D1B717))
    e = (((m * f32c(0x3B124925)).bitcast(Int32) + 0x7FFFFF) >> 23) - 127
    inv = (Int32(0x3F800000) - (e << 23)).bitcast(F32)
    sc = ((e + wsb) << 23).bitcast(F32)
    out = []
    for p in range_constexpr(4):
        q = _fp8x2(xs[2 * p] * inv, xs[2 * p + 1] * inv)
        b = rocdl.cvt_scalef32_pk_bf16_fp8(
            T.bf16x2, as_ir_value(q), as_ir_value(F32(1.0)), False
        )
        bw = fx.Vector(b).bitcast(Int32)[0]
        out.append(
            _bf_pack(
                (bw << 16).bitcast(F32) * sc, (bw & Int32(-65536)).bitcast(F32) * sc
            )
        )
    return out


@traced
def _main_proj(tid, cta, lds, a):
    """CTAs 256..455: seed[n][16 s + m] = sum_K main_w[16 s + m][K] fq(cap)[n][K] for the six captured rows n,
    slice s = c / 5, K sub-slice c % 5; sub-slices 1..4 hand their f32 partials to sub-slice 0 through
    ws_main_proj, which sums them in order, rounds to bf16 and pushes the slice to every rank's
    tab_main_proj area ({value, tag} entries, tag = 2 epoch + 1)."""
    lane, w = tid % 64, tid // 64
    c = cta - MP_CTA0
    s, sub = c // MP_SUBS, c % MP_SUBS
    cap, strides = _w(a, 0xE8), _w(a, 0xE8 + 0x08)
    main_w, main_w_s = _w(a, 0xE8 + 0x10), _w(a, 0xE8 + 0x18)
    ws, tab, rk = _w(a, 0xE8 + 0x28), _w(a, 0xE8 + 0x30), _w(a, 0xE8 + 0x38)
    ep = Int32(_w(a, 0xE8 + 0x40))
    tag = (ep << 1) | 1
    hstr, rstr = Int32(strides), Int32(strides >> 32)
    rank, npes = Int32(rk), Int32(rk >> 32)
    ubase = s * SLICE_UNITS + sub * SUB_UNITS
    wl = []
    for j in range_constexpr(3):
        for h in range_constexpr(2):
            word = (((ubase + 3 * w + j) << 11) + h * 1024 + lane * 16) >> 2
            wl.append(
                fx.Vector(
                    bo.buffer_load(
                        rsrc(main_w),
                        word,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
            )
    for k in range_constexpr(5):
        idx = tid + 512 * k
        ok = idx < 6 * 384
        row = idx // 384
        col = idx - row * 384
        kk = sub * MP_K + col * 8
        hc = kk // 5120
        off = hc * hstr + row * rstr + (kk - hc * 5120)
        x4 = _ldm(cap, NB, off >> 1, ok, 4)
        sb = _ldm(main_w_s, NB, (ubase * 4 + col // 4) >> 2, ok, 1)
        wsb = (Int32(sb) >> (((col // 4) & 3) * 8)) & 0xFF
        vals = _fq_chunk([Int32(x4[i]) for i in range(4)], wsb)
        if ok:
            _lds_st(lds, row * ACT_ROW + col * 16, vals)
    gpu.barrier()
    n, q = lane % 16, lane // 16
    brow = (n < 6).select(n, Int32(0)) * ACT_ROW + w * 0x300 + q * 16
    acc = fx.Vector.filled(4, 0.0, F32)
    for jj in range_constexpr(12):
        wv = wl[jj // 2]
        av = _fp8x8_bf16_scaled(wv[2 * (jj % 2)], wv[2 * (jj % 2) + 1], F32(1.0))
        bv = _lds_ld(lds, brow + 64 * jj, 4).bitcast(fx.BFloat16)
        acc = mfma_bf16(av, bv, acc)
    for i in range_constexpr(4):
        _lds_st(lds, RED_LDS + 256 * i + 1024 * w + 4 * lane, [acc[i].bitcast(Int32)])
    gpu.barrier()
    if tid < 96:
        rb = RED_LDS + (tid & 3) * 256 + 64 * ((tid >> 2) & 3) + 4 * (tid >> 4)
        f = F32(0.0) + _lds_ld(lds, rb).bitcast(F32)
        for k in range_constexpr(1, 8):
            f = f + _lds_ld(lds, rb + 1024 * k).bitcast(F32)
        _lds_st(lds, OWN_LDS + 4 * tid, [f.bitcast(Int32)])
    gpu.barrier()
    pv = [_ld(tab, 2 * p, 2) for p in range(8)]
    par = ((ep & 1) != 0).select(Int32(MP_PAR), Int32(0))
    peers = [
        rsrc((Int64(uniform(pv[p][1])) << 32) | Int64(fx.Uint32(uniform(pv[p][0]))))
        for p in range(8)
    ]
    if sub != 0:
        if tid < 48:
            v = _lds_ld(lds, OWN_LDS + 8 * tid, 2)
            data = fx.Vector.from_elements([Int32(v[0]), tag, Int32(v[1]), tag], Int32)
            bo.buffer_store(
                data,
                rsrc(ws),
                ((s << 12) + ((sub - 1) * 64 + tid) * 16) >> 2,
                cache_modifier=CM_DEV,
            )
    else:
        if tid < 192:
            j = tid // 48
            r = tid - j * 48
            word = ((s << 12) + (j << 10) + r * 16) >> 2
            v = _ld(ws, word, 4, CM_DEV)
            got = (Int32(v[1]) == tag) & (Int32(v[3]) == tag)
            while ~got:
                spin_pause()
                v = _ld(ws, word, 4, CM_DEV)
                got = (Int32(v[1]) == tag) & (Int32(v[3]) == tag)
            _lds_st(lds, PART_LDS + j * 384 + r * 8, [v[0], v[2]])
        gpu.barrier()
        if tid < 48:
            sums = []
            for e in range_constexpr(2):
                t = 2 * tid + e
                f = _lds_ld(lds, OWN_LDS + 4 * t).bitcast(F32)
                for j in range_constexpr(4):
                    f = f + _lds_ld(lds, PART_LDS + 384 * j + 4 * t).bitcast(F32)
                sums.append(f)
            _lds_st(lds, OUT_LDS + 4 * tid, [_bf_pack(sums[0], sums[1])])
        if npes >= 2:
            gpu.barrier()
            if tid < npes * 24:
                p = tid // 24
                r = tid - p * 24
                d = _lds_ld(lds, OUT_LDS + 8 * r, 2)
                data = fx.Vector.from_elements(
                    [Int32(d[0]), tag, Int32(d[1]), tag], Int32
                )
                word = (par + ((rank * 40 + s) << 9) + r * 16) >> 2
                for pp in range_constexpr(8):
                    if p == pp:
                        bo.buffer_store(data, peers[pp], word, cache_modifier=CM_SYS)


def _pair16(v, lane):
    """Even lanes: their u16 | the next lane's << 16 (two adjacent bf16 of a row owned by lanes t, t + 1)."""
    nb = bperm(Int32(v).bitcast(F32), lane + 1).bitcast(Int32)
    return (Int32(v) & 0xFFFF) | (nb << 16)


def _layer_word(a, byte, lay):
    """The kernarg word of MTP layer ``lay`` (0..2, dynamic) in a 3-word array at ``byte``."""
    b = byte // 8
    return (lay == 0).select(a[b], (lay == 1).select(a[b + 1], a[b + 2]))


def _e8m0_f32(b):
    """ue8m0 byte -> 2^(b - 127) (b = 0: 2^-127, a denormal)."""
    return (b == 0).select(Int32(0x400000), b << 23).bitcast(F32)


@traced
def _window_kv(tid, cta, lds, a):
    """CTAs 256..447 after main_proj: window_kv[L][n][16 t + r] = sum_kb (kv_w[L] tile t . seed[n])_kb x
    2^w_kb 2^s_kb over 160 K blocks of 32 (one bf16 MFMA a block, scaled into an fma chain), rows of the tile
    split between the CTA pair (cta & 1 = row bit 3), published as bf16 {value, tag} entries.
    """
    lane, w = tid % 64, tid // 64
    c = cta - MP_CTA0
    lay, tile, half = c >> 6, (cta >> 1) & 31, cta & 1
    ws = _w(a, 0xE8 + 0x28)
    ep = Int32(_w(a, 0xE8 + 0x40))
    tag = (ep << 1) | 1
    kv_w = _layer_word(a, 0xE8 + 0x50, lay)
    kv_s = _layer_word(a, 0xE8 + 0x68, lay)
    if tid < 6:
        f = _ld(ws, (ROWS_DONE >> 2) + tid, 1, CM_DEV)
        while f != tag:
            spin_pause()
            rocdl.s_sleep(1)
            f = _ld(ws, (ROWS_DONE >> 2) + tid, 1, CM_DEV)
    gpu.barrier()
    for k in range_constexpr(4):
        idx = tid + 512 * k
        ok = idx < 6 * 330
        row = idx // 330
        cc = idx - row * 330
        v = _ldm(ws, NB, (SEED_Q8 + row * SEED_ROW_B + cc * 16) >> 2, ok, 4, CM_DEV)
        vv = [Int32(v[i]) for i in range(4)]
        if ok:
            if cc >= 320:
                for d in range_constexpr(4):
                    _lds_st(
                        lds,
                        SEED_LDS_S + row * 640 + (cc - 320) * 64 + d * 16,
                        [
                            _e8m0_f32((vv[d] >> (8 * e)) & 0xFF).bitcast(Int32)
                            for e in range(4)
                        ],
                    )
            else:
                bf = _fp8x8_bf16_scaled(vv[0], vv[1], F32(1.0)).bitcast(Int32)
                bf2 = _fp8x8_bf16_scaled(vv[2], vv[3], F32(1.0)).bitcast(Int32)
                _lds_st(lds, row * 0x2800 + cc * 32, [bf[i] for i in range(4)])
                _lds_st(lds, row * 0x2800 + cc * 32 + 16, [bf2[i] for i in range(4)])
    mine = ((lane >> 3) & 1) == half
    wl = [
        _ldm(
            kv_w,
            NB,
            (((tile * 0x50 + 10 * w + u) << 10) + lane * 16) >> 2,
            mine,
            4,
            CM_NT,
        )
        for u in range(10)
    ]
    sb = _ld(kv_s, ((tile >> 1) * 160 + 20 * w) >> 2, 4)
    sb4 = _ld(kv_s, (((tile >> 1) * 160 + 20 * w) >> 2) + 4, 1)
    sbw = [Int32(sb[0]), Int32(sb[1]), Int32(sb[2]), Int32(sb[3]), Int32(sb4)]
    gpu.barrier()
    n, q = lane % 16, lane // 16
    nr = (n < 6).select(n, Int32(0))
    acc = [F32(0.0)] * 4
    for j in range_constexpr(20):
        u, h = j // 2, j % 2
        av = _fp8x8_bf16_scaled(Int32(wl[u][2 * h]), Int32(wl[u][2 * h + 1]), F32(1.0))
        bv = _lds_ld(
            lds, nr * 0x2800 + q * 16 + (10 * w + u) * 128 + h * 64, 4
        ).bitcast(fx.BFloat16)
        pv = mfma_bf16(av, bv, fx.Vector.filled(4, 0.0, F32))
        wsc = _e8m0_f32((sbw[j // 4] >> (8 * (j % 4))) & 0xFF)
        sc = wsc * _lds_ld(lds, SEED_LDS_S + nr * 640 + (20 * w + j) * 4).bitcast(F32)
        acc = [fmath.fma(pv[i], sc, acc[i]) for i in range(4)]
    if w != 0:
        for i in range_constexpr(4):
            _lds_st(
                lds, WRED_LDS + 256 * i + 1024 * w + 4 * lane, [acc[i].bitcast(Int32)]
            )
    gpu.barrier()
    if (w == 0) & ((lane >> 5) == half) & (n < 6):
        f = list(acc)
        for k in range_constexpr(1, 8):
            f = [
                f[i]
                + _lds_ld(lds, WRED_LDS + 256 * i + 1024 * k + 4 * lane).bitcast(F32)
                for i in range(4)
            ]
        data = fx.Vector.from_elements(
            [_bf_pack(f[0], f[1]), tag, _bf_pack(f[2], f[3]), tag], Int32
        )
        off = WKV_OUT + (lay << 14) + (n << 11) + (tile << 6) + q * 16
        bo.buffer_store(data, rsrc(ws), off >> 2, cache_modifier=CM_DEV)


def _fp8x2s(a, b, scale):
    """Two f32 -> E4M3 pair (low 16 bits) by v_cvt_scalef32_pk_fp8_f32 with the scale operand."""
    o = fx.Vector.from_elements([Int32(0)], Int32).bitcast(fx.Int16)
    r = rocdl.cvt_scalef32_pk_fp8_f32(T.vec(2, T.i16), o.ir_value(), F32(a).ir_value(), F32(b).ir_value(),
                                      F32(scale).ir_value(), False)  # fmt: skip
    return fx.Vector(r).bitcast(Int32)[0] & 0xFFFF


@traced
def _q32_store(y, lane, d, slot, win_kv, win_s):
    """E4M3 of 32-lane groups (lane = channel d): ue8m0 = ceil(log2(max(amax, 1e-4) / 448)) + 127, pairs
    stored by even lanes at win_kv + 512 slot + d, the byte by each group's first lane at win_s + 16 slot +
    d / 32."""
    m = abs(y)
    for k in (16, 8, 4, 2, 1):
        m = fx.max(m, bperm(m, lane ^ k))
    m = fx.max(m, f32c(0x38D1B717))
    eb = (m * f32c(0x3B124925)).bitcast(Int32) + 0x7FFFFF
    e = (eb >> 23) & 0xFF
    sc = (e == 0).select(Int32(0x400000), eb & Int32(-8388608)).bitcast(F32)
    nb = bperm(y, lane ^ 1)
    if (d & 1) == 0:
        bo.buffer_store(
            _fp8x2s(y, nb, sc).to(fx.Int16),
            rsrc(win_kv),
            slot * 512 + d,
            offset_is_bytes=True,
        )
    if (d & 31) == 0:
        bo.buffer_store(
            e.to(fx.Int8), rsrc(win_s), slot * 16 + (d >> 5), offset_is_bytes=True
        )


@traced
def _window_norm(tid, cta, lds, a):
    """CTAs 462..479 (layer L = c / 6, row n = c % 6, skipped when vpos[n] < 0): the window kv row's 512 bf16
    from the CTA pairs' entries, RMSNorm (fma(sum / 512, 1e-20), 1 / sqrt), x kv_norm[L] (bf16), GPT-J RoPE
    on channels 448..511 at vpos[n], E4M3 + ue8m0 into win.{40,41,42}[L] slot vpos[n] & 255.
    """
    lane, w = tid % 64, tid // 64
    c = cta - WIN_CTA0
    lay = c // 6
    n = c - lay * 6
    ws = _w(a, 0xE8 + 0x28)
    ep = Int32(_w(a, 0xE8 + 0x40))
    tag = (ep << 1) | 1
    pos = _ld(_w(a, 0xE8 + 0xD0), n)
    if pos >= 0:
        if tid < 128:
            word = (WKV_OUT + (lay << 14) + (n << 11) + tid * 16) >> 2
            v = _ld(ws, word, 4, CM_DEV)
            while (Int32(v[1]) != tag) | (Int32(v[3]) != tag):
                spin_pause()
                v = _ld(ws, word, 4, CM_DEV)
            _lds_st(lds, 8 * tid, [v[0], v[2]])
        gpu.barrier()
        t = tid & 255
        gam = _layer_word(a, 0xE8 + 0x80, lay)
        g0 = _ld(gam, t).bitcast(F32)
        g1 = _ld(gam, t + 256).bitcast(F32)
        pair = (t >= 192).select((t - 192) >> 1, Int32(0))
        rope = _w(a, 0xE8 + 0xC8) + pos.to(Int64) * 256
        cs = _ld(rope, pair).bitcast(F32)
        sn = _ld(rope, pair + 32).bitcast(F32)
        d0 = _lds_ld(lds, (2 * t) & ~3)
        d1 = _lds_ld(lds, (512 + 2 * t) & ~3)
        sh = (t & 1) * 16
        x0 = ((d0 >> sh) << 16).bitcast(F32)
        x1 = ((d1 >> sh) << 16).bitcast(F32)
        v = fmath.fma(x0, x0, x1 * x1)
        for k in (32, 16, 8, 4, 2, 1):
            v = v + bperm(v, lane ^ k)
        if (lane == 0) & (tid < 256):
            _lds_st(lds, 1024 + 4 * w, [v.bitcast(Int32)])
        gpu.barrier()
        w4 = _lds_ld(lds, 1024, 4)
        tot = (
            ((F32(0.0) + w4[0].bitcast(F32)) + w4[1].bitcast(F32)) + w4[2].bitcast(F32)
        ) + w4[3].bitcast(F32)
        ms = fmath.fma(tot, f32c(0x3B000000), f32c(EPS))
        r = F32(1.0) / fmath.sqrt(ms)
        y0 = (bf_rne_hi(g0 * (r * x0)) & Int32(-65536)).bitcast(F32)
        y1 = (bf_rne_hi(g1 * (r * x1)) & Int32(-65536)).bitcast(F32)
        oth = bperm(y1, lane ^ 1)
        even = (t & 1) == 0
        ye = even.select(y1, oth)
        yo = even.select(oth, y1)
        rot = even.select(cs * ye - sn * yo, sn * ye + cs * yo)
        y1 = (t >= 192).select((bf_rne_hi(rot) & Int32(-65536)).bitcast(F32), y1)
        slot = pos & (WIN_SLOTS - 1)
        win_kv = _layer_word(a, 0xE8 + 0x98, lay)
        win_s = _layer_word(a, 0xE8 + 0xB0, lay)
        if tid < 256:
            _q32_store(y0, lane, tid, slot, win_kv, win_s)
            _q32_store(y1, lane, tid + 256, slot, win_kv, win_s)


@traced
def _main_norm(tid, cta, lds, a):
    """CTAs 456..461: seed row n = cta - 456 gathered from this rank's tab_main_proj area (8 ranks x 40 slices
    x 4 {value, tag} entries), y = bf16(gamma (rsq(sum x^2 / 5120 + 1e-20) x)) (optionally stored at SA +
    0x48), then 1x32 E4M3 + ue8m0 into ws_main_proj + 0x28000 + 0x1500 n and the row's flag {tag}.
    """
    lane, w = tid % 64, tid // 64
    n = cta - NORM_CTA0
    gamma, ws, tab = _w(a, 0xE8 + 0x20), _w(a, 0xE8 + 0x28), _w(a, 0xE8 + 0x30)
    rk, out = _w(a, 0xE8 + 0x38), _w(a, 0xE8 + 0x48)
    ep = Int32(_w(a, 0xE8 + 0x40))
    tag = (ep << 1) | 1
    rank, npes = Int32(rk), Int32(rk >> 32)
    par = ((ep & 1) != 0).select(Int32(MP_PAR), Int32(0))
    g = [_ld(gamma, tid + 512 * k).bitcast(F32) for k in range(10)]
    bv = _ld(tab, rank * 2, 2)
    base = (Int64(uniform(bv[1])) << 32) | Int64(fx.Uint32(uniform(bv[0])))
    nent = npes * 160
    oks, words, ldsb = [], [], []
    for j in range_constexpr(3):
        e = tid + 512 * j
        ok = e < nent
        pr = e // 160
        x = e - pr * 160
        oks.append(ok)
        words.append((par + ((pr * 40 + (x >> 2)) << 9) + n * 64 + (x & 3) * 16) >> 2)
        ldsb.append(pr * 1280 + (x >> 2) * 32 + (x & 3) * 8)
    vs = [_ldm(base, NB, words[j], oks[j], 4, CM_SYS) for j in range(3)]
    got = [
        (~oks[j]) | ((Int32(vs[j][1]) == tag) & (Int32(vs[j][3]) == tag))
        for j in range(3)
    ]
    while ~(got[0] & got[1] & got[2]):
        spin_pause()
        nv, ng = [], []
        for j in range_constexpr(3):
            need = ~got[j]
            v = _ldm(base, NB, words[j], need, 4, CM_SYS)
            nv.append(
                fx.Vector.from_elements(
                    [need.select(Int32(v[i]), Int32(vs[j][i])) for i in range(4)], Int32
                )
            )
            ng.append((~oks[j]) | ((Int32(nv[j][1]) == tag) & (Int32(nv[j][3]) == tag)))
        vs, got = nv, ng
    for j in range_constexpr(3):
        if oks[j]:
            _lds_st(lds, ldsb[j], [vs[j][0], vs[j][2]])
    gpu.barrier()
    xs = []
    for k in range_constexpr(10):
        b = 2 * tid + 1024 * k
        d = _lds_ld(lds, b & ~3)
        xs.append(((d >> ((b & 2) * 8)) << 16).bitcast(F32))
    ss = fmath.fma(xs[0], xs[0], F32(0.0))
    for k in range_constexpr(1, 10):
        ss = fmath.fma(xs[k], xs[k], ss)
    ss = _scan_sum(ss)
    if lane == 63:
        _lds_st(lds, NSUM_LDS + 4 * w, [ss.bitcast(Int32)])
    gpu.barrier()
    tot = F32(0.0)
    for k in range_constexpr(8):
        tot = tot + _lds_ld(lds, NSUM_LDS + 4 * k).bitcast(F32)
    r = _rsq_scaled(tot / F32(5120.0) + f32c(EPS))
    ys = [(bf_rne_hi(g[k] * (r * xs[k])) >> 16) & 0xFFFF for k in range(10)]
    pairs = [_pair16(ys[k], lane) for k in range(10)]
    gpu.barrier()
    if (tid & 1) == 0:
        for k in range_constexpr(10):
            _lds_st(lds, 2 * tid + 1024 * k, [pairs[k]])
            if out != 0:
                bo.buffer_store(
                    pairs[k],
                    rsrc(out),
                    (n * 5120 + tid + 512 * k) * 2,
                    offset_is_bytes=True,
                )
    gpu.barrier()
    for h in range_constexpr(2):
        i = tid + 512 * h
        ok = i < 640
        x4 = _lds_ld(lds, ok.select(i, Int32(0)) * 16, 4)
        ys8 = []
        for d in range_constexpr(4):
            ys8 += [
                (Int32(x4[d]) << 16).bitcast(F32),
                (Int32(x4[d]) & Int32(-65536)).bitcast(F32),
            ]
        m = abs(ys8[0])
        for k in range_constexpr(1, 8):
            m = fx.max(m, abs(ys8[k]))
        m = fx.max(m, _dppf(m, QX1))
        m = fx.max(fx.max(m, _dppf(m, QX2)), f32c(0x38D1B717))
        eb = (m * f32c(0x3B124925)).bitcast(Int32) + 0x7FFFFF
        inv = (Int32(0x7F000000) - (eb & Int32(-8388608))).bitcast(F32)
        q = [_fp8x2(ys8[2 * c] * inv, ys8[2 * c + 1] * inv) for c in range(4)]
        sb = (eb >> 23) & 0xFF
        sw = sb
        for c in range_constexpr(1, 4):
            sw = sw | (
                (bperm(sb.bitcast(F32), lane + 4 * c).bitcast(Int32) & 0xFF) << (8 * c)
            )
        if ok:
            _lds_st(lds, Q8_LDS + 8 * i, [q[0] | (q[1] << 16), q[2] | (q[3] << 16)])
            if (i & 15) == 0:
                _lds_st(lds, QS_LDS + (i >> 2), [sw])
    gpu.barrier()
    if tid < (5120 + 160) // 16:
        v = _lds_ld(lds, Q8_LDS + 16 * tid, 4)
        bo.buffer_store(
            v,
            rsrc(ws),
            (SEED_Q8 + n * SEED_ROW_B + 16 * tid) >> 2,
            cache_modifier=CM_DEV,
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        bo.buffer_store(tag, rsrc(ws), (ROWS_DONE >> 2) + n, cache_modifier=CM_DEV)


def _hfetch(tid, s2, j, cnt, head_w):
    """Tile j's 20 weight loads for this wave (nt), none past the CTA's tiles."""
    lane, w = tid % 64, tid // 64
    tile = s2 + 256 * j
    ok = j < cnt
    return [_ldm(head_w, TILES * 160 * 1024, (((tile * 160 + w * 20 + ks) << 10) + lane * 16) >> 2, ok, 4, CM_NT)
            for ks in range(20)]  # fmt: skip


@traced
def _head(tid, s2, lds, a):
    """CTAs 0..255: logits[h][16 tile + r] = head_w[tile] . Xn[h] for the six rows (wave w the K range 640 w,
    20 chained bf16 MFMAs; waves summed w0 + w1 + .. + w7), stored raw; / T (IEEE, when T != 1) into LDS for
    the sampler statistics."""
    lane, w = tid % 64, tid // 64
    xn, head_w, logits = a[0], a[1], a[2]
    temp = _ld(_w(a, 0x230), 2).bitcast(F32)
    cnt = (1009 - s2) // 256 + 1
    pre = _hfetch(tid, s2, 0, cnt, head_w)
    for jj in range_constexpr(8):
        i = tid + 512 * jj
        if i < 6 * 640:
            v = _ld(xn, i * 4, 4)
            _lds_st(lds, (i // 640) * XN_ROW + (i % 640) * 16, [v[d] for d in range(4)])
    gpu.barrier()
    h, q = lane % 16, lane >> 4
    xo = (h > 5).select(Int32(0), h) * XN_ROW + w * 0x500 + q * 16
    cur = pre
    for j in range_constexpr(4):
        nxt = _hfetch(tid, s2, j + 1, cnt, head_w) if j < 3 else None
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
                        HRED_LDS + 256 * i + 1024 * w + 4 * lane,
                        [acc[i].bitcast(Int32)],
                    )
            gpu.barrier()
            if (w == 0) & (h < 6):
                c = []
                for i in range_constexpr(4):
                    sm = acc[i]
                    for ww in range_constexpr(1, 8):
                        sm = sm + _lds_ld(
                            lds, HRED_LDS + 256 * i + 1024 * ww + 4 * lane
                        ).bitcast(F32)
                    c.append(sm)
                tile = s2 + 256 * j
                bo.buffer_store(
                    fx.Vector.from_elements([x.bitcast(Int32) for x in c], Int32),
                    rsrc(logits),
                    h * VOCAB_RANK + tile * 16 + q * 4,
                )
                one = temp == F32(1.0)
                _lds_st(
                    lds,
                    LG_LDS + h * 256 + j * 64 + q * 16,
                    [one.select(x, x / temp).bitcast(Int32) for x in c],
                )
            gpu.barrier()
        cur = nxt


def _exp_isa(x):
    """OCML expf as hipcc emits it here (exp_ocml's arithmetic; the clamps as unordered compares: NaN
    stays NaN)."""
    t = x * f32c(0x3FB8AA3B)
    err = fmath.fma(x, f32c(0x3FB8AA3B), -t)
    n = F32(_call("llvm.rint.f32", T.f32, t))
    err = fmath.fma(x, f32c(0x32A5705F), err)
    f = (t - n) + err
    e = F32(_call("llvm.ldexp.f32.i32", T.f32, hw_exp2(f), n.to(Int32)))
    e = (~(x < f32c(0xC2CE8ED0))).select(e, F32(0.0))
    return (~(x > f32c(0x42B17218))).select(e, f32c(0x7F800000))


def _sm_merge(m, s, lane, k, fused):
    """Online-softmax (max, sum) merged with lane ^ k's: M = max; M = -inf -> (-inf, 0); else
    e_b s_b + e_a s_a (``fused``: fma(e_b, s_b, e_a s_a)), b the partner."""
    mb = bperm(m, lane ^ k)
    sb = bperm(s, lane ^ k)
    mm = fx.max(m, mb)
    live = mm != f32c(0xFF800000)
    ea = _exp_isa(m - mm)
    eb = _exp_isa(mb - mm)
    sn = fmath.fma(eb, sb, ea * s) if fused else (eb * sb) + (ea * s)
    return live.select(mm, f32c(0xFF800000)), live.select(sn, F32(0.0))


def _fmix(x):
    x = x ^ _lsr(x, 16)
    x = x * Int32(F1)
    x = x ^ _lsr(x, 13)
    x = x * Int32(F2)
    return x ^ _lsr(x, 16)


@traced
def _stats(tid, s2, lds, a, cnt):
    """CTAs 0..255 after the GEMV: per item (wave w: w, and w + 8 for w <= 3) the CTA's partial over its
    cnt x 16 vocab entries: rows 0..5 online softmax of logits / T, items 6..10 of the draft rows' q logits
    draft_z / T, item 11 the Gumbel-max key of row s33 (the bonus row); lanes 0 / 1 publish {lo / hi, tag}
    to ws_sample; the CTA owning a live draft token publishes its p / q logits."""
    lane, w = tid % 64, tid // 64
    draft_z, vtok, vpos = _w(a, 0x208), _w(a, 0x210), _w(a, 0x218)
    samp, ws = _w(a, 0x230), _w(a, 0x250)
    rank = Int32(_w(a, 0x240))
    ep = Int32(_w(a, 0x248))
    temp = _ld(samp, 2).bitcast(F32)
    one = temp == F32(1.0)
    nv = cnt * 16
    vp = [uniform(_ld(vpos, j)) for j in range(6)]
    nlive = Int32(0)
    run = vp[1] >= 0
    for j in range_constexpr(1, 6):
        if j > 1:
            run = run & (vp[j] >= 0)
        nlive = nlive + run.select(Int32(1), Int32(0))
    tag = _fmix(ep * Int32(0x3C6EF362) + _ld(samp, 6) * Int32(0xC2B2AE35 - (1 << 32))
                + vp[0] * Int32(0x85EBCA6B - (1 << 32)) + Int32(0x9E3779B1 - (1 << 32))) | 1  # fmt: skip
    h1 = _hstep(Int32(0), _kmix(_ld(samp, 0)))
    h2 = _hstep(h1, _kmix(_ld(samp, 1)))
    # the draft rows' q logits of the CTA's entries: LDS [row 0..4][64] (0 past the CTA's tiles)
    if tid < 320:
        okz = lane < nv
        zi = (s2 + 256 * ((lane >> 4) & 3)) * 16 + (lane & 15)
        z = _ldm(draft_z, NB, w * VOCAB_RANK + zi, okz, 1)
        _lds_st(lds, 4 * tid, [okz.select(Int32(z), Int32(0))])
    gpu.barrier()
    has = lane < nv
    for ps in range_constexpr(2):
        k = w + 8 * ps
        if (ps == 0) | (w <= 3):
            lo, hi = Int32(0), Int32(0)
            if k != 11:
                xr = _lds_ld(
                    lds, LG_LDS + fx.min(k, Int32(5)) * 256 + 4 * lane
                ).bitcast(F32)
                xd = _lds_ld(lds, fx.max(k - 6, Int32(0)) * 256 + 4 * lane).bitcast(F32)
                xd = one.select(xd, xd / temp)
                x = (k <= 5).select(xr, xd)
                m = f32c(0xFF800000)
                sm = F32(0.0)
                gt = x > m
                s_a = sm + _exp_isa(x - m)
                s_b = fmath.fma(sm, _exp_isa(m - x), F32(1.0))
                m = (has & gt).select(x, m)
                sm = has.select(gt.select(s_b, s_a), sm)
                for kk in (1, 2, 4, 8, 16):
                    m, sm = _sm_merge(m, sm, lane, kk, False)
                m, sm = _sm_merge(m, sm, lane, 32, True)
                lo, hi = m.bitcast(Int32), sm.bitcast(Int32)
            else:
                xr = _lds_ld(lds, LG_LDS + nlive * 256 + 4 * lane).bitcast(F32)
                vid = rank * VOCAB_RANK + (s2 + 256 * (lane >> 4)) * 16 + (lane & 15)
                s30 = _hstep(h2, _kmix(_sel6(vp, nlive)))
                sc = _gumbel_score(xr, F32(1.0), s30, _kmix(vid))
                kh = has.select(_key_hi(sc.bitcast(Int32)), Int32(0))
                kl = has.select(~vid, Int32(0))
                for kk in (1, 2, 4, 8, 16, 32):
                    oh, ol = bperm(kh.bitcast(F32), lane ^ kk).bitcast(Int32), bperm(
                        kl.bitcast(F32), lane ^ kk
                    ).bitcast(Int32)
                    g = _k_gt(oh, ol, kh, kl)
                    kh, kl = g.select(oh, kh), g.select(ol, kl)
                lo, hi = kl, kh
            if lane < 2:
                v = (lane == 0).select(lo, hi)
                bo.buffer_store(fx.Vector.from_elements([v, tag], Int32), rsrc(ws),
                                (STAT_OFF + 4096 * k + 2048 * lane + 8 * s2) >> 2, cache_modifier=CM_DEV)  # fmt: skip
    if (tid < 5) & (tid < nlive):
        d = _ld(vtok, 1 + tid)
        off = d - rank * VOCAB_RANK
        if _u_lt(off, Int32(VOCAB_RANK)):
            tl = off >> 4
            dt = tl - s2
            if (tl >= s2) & ((dt & 0xFF) == 0):
                idx = (dt >> 4) | (d & 15)
                pl = _lds_ld(lds, LG_LDS + tid * 256 + 4 * idx)
                ql = _lds_ld(lds, tid * 256 + 4 * idx)
                bo.buffer_store(fx.Vector.from_elements([pl, tag, ql, tag], Int32), rsrc(ws),
                                (VER_OFF + 16 * tid) >> 2, cache_modifier=CM_DEV)  # fmt: skip
    return tag, nlive, vp


def _sel6(vp, i):
    """vp[i] for a dynamic i in 0..5."""
    r = vp[5]
    for j in (4, 3, 2, 1, 0):
        r = (i == j).select(vp[j], r)
    return r


def _u_lt(a, b):
    return (a ^ Int32(-(1 << 31))) < (b ^ Int32(-(1 << 31)))


@traced
def _fold_gather(tid, lds, a, tag):
    """CTA 255: item k = w (and w + 8 for w <= 3) merged over the 256 CTAs: lane l takes CTAs l, l + 64,
    l + 128, l + 192 in that order, then lanes xor 1..16 (mul, add) and xor 32 (fma) -> LDS ITEM_LDS + 8 k.
    """
    lane, w = tid % 64, tid // 64
    ws = _w(a, 0x250)
    for ps in range_constexpr(2):
        k = w + 8 * ps
        if (ps == 0) | (w <= 3):
            base = STAT_OFF + 4096 * k + 8 * lane
            v = [
                _ld(ws, (base + 512 * c + 2048 * hh) >> 2, 2, CM_DEV)
                for c in range(4)
                for hh in range(2)
            ]
            got = Int32(v[0][1]) == tag
            for i in range_constexpr(1, 8):
                got = got & (Int32(v[i][1]) == tag)
            while ~got:
                spin_pause()
                rocdl.s_sleep(1)
                v = [
                    _ld(ws, (base + 512 * c + 2048 * hh) >> 2, 2, CM_DEV)
                    for c in range(4)
                    for hh in range(2)
                ]
                got = Int32(v[0][1]) == tag
                for i in range_constexpr(1, 8):
                    got = got & (Int32(v[i][1]) == tag)
            lo, hi = [Int32(v[2 * c][0]) for c in range(4)], [
                Int32(v[2 * c + 1][0]) for c in range(4)
            ]
            outl, outh = Int32(0), Int32(0)
            if k != 11:
                m, sm = lo[0].bitcast(F32), hi[0].bitcast(F32)
                for c in range_constexpr(1, 4):
                    mb, sb = lo[c].bitcast(F32), hi[c].bitcast(F32)
                    mm = fx.max(m, mb)
                    live = mm != f32c(0xFF800000)
                    sn = (_exp_isa(mb - mm) * sb) + (_exp_isa(m - mm) * sm)
                    m, sm = live.select(mm, f32c(0xFF800000)), live.select(sn, F32(0.0))
                for kk in (1, 2, 4, 8, 16):
                    m, sm = _sm_merge(m, sm, lane, kk, False)
                m, sm = _sm_merge(m, sm, lane, 32, True)
                outl, outh = m.bitcast(Int32), sm.bitcast(Int32)
            else:
                kh, kl = hi[0], lo[0]
                for c in range_constexpr(1, 4):
                    g = _k_gt(hi[c], lo[c], kh, kl)
                    kh, kl = g.select(hi[c], kh), g.select(lo[c], kl)
                for kk in (1, 2, 4, 8, 16, 32):
                    oh = bperm(kh.bitcast(F32), lane ^ kk).bitcast(Int32)
                    ol = bperm(kl.bitcast(F32), lane ^ kk).bitcast(Int32)
                    g = _k_gt(oh, ol, kh, kl)
                    kh, kl = g.select(oh, kh), g.select(ol, kl)
                outl, outh = kl, kh
            if lane == 0:
                _lds_st(lds, ITEM_LDS + 8 * k, [outl, outh])


@traced
def _fold_publish(tid, lds, a, tag, nlive):
    """CTA 255 (wave 0 publishes; the descriptors are built by every wave, outside divergent code): the rank's 17 entries (items 0..11; slot 12 + j: draft j's (p / T, q) logits when this
    rank owns its token) to every rank's sym:sample at ((parity 8 + rank) 17 + slot) 16, {lo, tag}, {hi,
    tag}."""
    lane = tid % 64
    ws, vtok, tab = _w(a, 0x250), _w(a, 0x210), _w(a, 0x238)
    rk = _w(a, 0x240)
    rank, npes = Int32(rk), Int32(rk >> 32)
    par = Int32(_w(a, 0x248)) & 1
    pv = [_ldm(tab, 64, 2 * p, npes > p, 2) for p in range(8)]
    peers = [
        rsrc((Int64(uniform(v[1])) << 32) | Int64(fx.Uint32(uniform(v[0])))) for v in pv
    ]
    jl = fx.min(lane, Int32(4))
    d = _ld(vtok, 1 + jl)
    w0 = tid < 64
    own = w0 & (lane < nlive) & (d // VOCAB_RANK == rank)
    pq0, pq1 = Int32(0), Int32(0)
    if own:
        e = _ld(ws, (VER_OFF + 16 * jl) >> 2, 4, CM_DEV)
        while (Int32(e[1]) != tag) | (Int32(e[3]) != tag):
            spin_pause()
            e = _ld(ws, (VER_OFF + 16 * jl) >> 2, 4, CM_DEV)
        pq0, pq1 = Int32(e[0]), Int32(e[2])
    pjs = [bperm(pq0.bitcast(F32), j).bitcast(Int32) for j in range(5)]
    qjs = [bperm(pq1.bitcast(F32), j).bitcast(Int32) for j in range(5)]
    owns = [
        bperm(own.select(Int32(1), Int32(0)).bitcast(F32), j).bitcast(Int32) != 0
        for j in range(5)
    ]
    if w0 & (lane < ENTRIES):
        slot = lane
        it = _lds_ld(lds, ITEM_LDS + 8 * fx.min(slot, Int32(11)), 2)
        v0, v1, ok = Int32(it[0]), Int32(it[1]), Int32(1) == 1
        for j in range_constexpr(5):
            isj = slot == 12 + j
            v0 = isj.select(pjs[j], v0)
            v1 = isj.select(qjs[j], v1)
            ok = ok & ((~isj) | owns[j])
        if ok:
            data = fx.Vector.from_elements([v0, tag, v1, tag], Int32)
            word = (((par * 8 + rank) * ENTRIES + slot) * 16) >> 2
            for pp in range_constexpr(
                8
            ):  # one uniform descriptor a store (no per-lane descriptor select)
                if pp < npes:
                    bo.buffer_store(data, peers[pp], word, cache_modifier=CM_SYS)


def _logf(x):
    return F32(_call("llvm.log.f32", T.f32, x))


def _philox_u(vp, samp0, samp1, j):
    """Philox4x32-10 of counter (vp, vp >> 31, 0, 0), key (samp0, samp1 ^ j) -> u in [2^-31, 1)."""
    x0, x1, x2, x3 = vp, vp >> 31, Int32(0), Int32(0)
    k0, k1 = samp0, samp1 ^ j
    m0, m1 = Int32(0xD2511F53 - (1 << 32)), Int32(0xCD9E8D57 - (1 << 32))
    for _ in range_constexpr(10):
        h1 = _umulhi(x2, m1)
        h0 = _umulhi(x0, m0)
        x0, x1, x2, x3 = h1 ^ x1 ^ k0, x2 * m1, h0 ^ x3 ^ k1, x0 * m0
        k0 = k0 + Int32(0x9E3779B9 - (1 << 32))
        k1 = k1 + Int32(0xBB67AE85 - (1 << 32))
    v = x0 ^ (x0 >> 31)
    return fx.max(v.to(F32) * f32c(0x2FFFFFFF), f32c(0x2FFFFFFF))


def _umulhi(a, b):
    return Int32((Int64(fx.Uint32(a)) * Int64(fx.Uint32(b))) >> 32)


def _splitmix_r(counter):
    """TileRT's simulated-acceptance draw: SplitMix64's mix of x = (1 << 32) | counter, top 24 bits x 2^-24."""
    x = (Int64(1) << 32) | Int64(fx.Uint32(counter))
    z = (x ^ _lsr64(x, 30)) * Int64(0xBF58476D1CE4E5B9 - (1 << 64))
    z = (z ^ _lsr64(z, 27)) * Int64(0x94D049BB133111EB - (1 << 64))
    return Int32(_lsr64(z, 40)).to(F32) * f32c(0x33800000)


def _lsr64(x, n):
    return (x >> n) & Int64((1 << (64 - n)) - 1)


@traced
def _fold_accept(tid, lds, a, tag, nlive, vp):
    """CTA 255: the 8 ranks' entries -> per item LSE; draft j's lp = log p - log q, acc_j = min(0, acc_{j-1}
    + lp_j), Philox coins u_j; the accept count (simulated when kernarg 0x314 > 0); either the bonus token
    (the ranks' max Gumbel key) or the residual mask with its rows' {LSE_p, LSE_q, e_prev, acc_prev}
    published to ws_sample (+0x1240 + 32 j, the mask | 0x10000 at +0x1200). Returns (token or -1, mask).
    """
    lane = tid % 64
    w0 = tid < 64
    ws, vtok, tab, samp = _w(a, 0x250), _w(a, 0x210), _w(a, 0x238), _w(a, 0x230)
    rk = _w(a, 0x240)
    rank, npes = Int32(rk), Int32(rk >> 32)
    par = Int32(_w(a, 0x248)) & 1
    temp = _ld(samp, 2).bitcast(F32)
    one = temp == F32(1.0)
    ov = _ld(tab, rank * 2, 2)
    own = (Int64(uniform(ov[1])) << 32) | Int64(fx.Uint32(uniform(ov[0])))
    # the ranks' entries (slot >= 12: draft slot - 12, only from the rank owning its token)
    for ps in range_constexpr(3):
        x = lane + 64 * ps
        r = x // ENTRIES
        slot = x - r * ENTRIES
        jd = fx.min(fx.max(slot - 12, Int32(0)), Int32(4))
        dd = _ld(vtok, 1 + jd)
        need = (
            w0
            & (x < npes * ENTRIES)
            & ((slot < 12) | ((slot - 12 < nlive) & (dd // VOCAB_RANK == r)))
        )
        word = (((par * 8 + r) * ENTRIES + slot) * 16) >> 2
        e = _ldm(own, NB, word, need, 4, CM_SYS)
        while need & ((Int32(e[1]) != tag) | (Int32(e[3]) != tag)):
            spin_pause()
            e = _ldm(own, NB, word, need, 4, CM_SYS)
        if need:
            _lds_st(lds, RX_LDS + 136 * r + 8 * slot, [e[0], e[2]])
    gpu.barrier()
    tok, mask, accepted = Int32(-1), Int32(0), Int32(-1)
    if w0:
        jj = (lane < 11).select(lane, Int32(0))
        it = _lds_ld(lds, RX_LDS + 8 * jj, 2)
        m, sm = Int32(it[0]).bitcast(F32), Int32(it[1]).bitcast(F32)
        for r in range_constexpr(1, 8):
            if r < npes:
                ir = _lds_ld(lds, RX_LDS + 136 * r + 8 * jj, 2)
                mr, sr = Int32(ir[0]).bitcast(F32), Int32(ir[1]).bitcast(F32)
                mm = fx.max(m, mr)
                live = mm != f32c(0xFF800000)
                sn = (_exp_isa(mr - mm) * sr) + (_exp_isa(m - mm) * sm)
                m, sm = live.select(mm, f32c(0xFF800000)), live.select(sn, F32(0.0))
        lse = m + _logf(sm)
        j5 = (lane < 5).select(lane, Int32(0))
        lseq = bperm(lse, j5 + 6)
        lp = F32(0.0)
        if lane < nlive:
            d = _ld(vtok, 1 + j5)
            o = d // VOCAB_RANK
            pq = _lds_ld(lds, RX_LDS + 96 + 136 * o + 8 * j5, 2)
            pl, ql = Int32(pq[0]).bitcast(F32), Int32(pq[1]).bitcast(F32)
            lq = one.select(ql, ql / temp) - lseq
            lp = (pl - lse) - lq
        acc = F32(0.0)
        if vp[1] >= 0:
            run = F32(0.0)
            for i in range_constexpr(5):
                if i < nlive:
                    run = fx.min(F32(0.0), run + bperm(lp, i))
                    acc = (lane == i).select(run, acc)
        u = F32(2.0)
        if lane < nlive:
            u = _philox_u(_sel6(vp, j5), _ld(samp, 0), _ld(samp, 1), lane)
        if lane < 5:
            _lds_st(lds, 0x12694 + 4 * lane, [acc.bitcast(Int32)])
            _lds_st(lds, 0x126AC + 4 * lane, [u.bitcast(Int32)])
        ea = _exp_isa(acc)
        last = fx.max(nlive - 1, Int32(0))
        u_last, e_last = bperm(u, last), bperm(ea, last)
        acc_len = Int32(_w(a, 0x310) >> 32).bitcast(F32)
        if f32c(0) < acc_len:
            ctl = _w(a, 0x2B8)
            cnt = vp[0]
            if ctl != 0:
                cnt = _ld(ctl, 0) + 1
            lv = fx.min(fx.max(acc_len, F32(1.0)), F32(6.0)) - F32(1.0)
            n = fmath.floor(lv).to(Int32)
            frac = lv - n.to(F32)
            if f32c(0) < frac:
                n = n + (_splitmix_r(cnt) < frac).select(Int32(1), Int32(0))
            accepted = fx.min(fx.min(n, nlive), Int32(5))
        more = _u_lt(accepted, nlive)
        rej_last = (vp[1] >= 0) & ~(u_last <= e_last)
        if ~(rej_last | more):
            kh, kl = Int32(0), Int32(0)
            for r in range_constexpr(8):
                if r < npes:
                    kv = _lds_ld(lds, RX_LDS + 136 * r + 88, 2)
                    g = _k_gt(Int32(kv[1]), Int32(kv[0]), kh, kl)
                    kh, kl = g.select(Int32(kv[1]), kh), g.select(Int32(kv[0]), kl)
            tok = (npes >= 1).select(~kl, Int32(-1))
        else:
            ok = ((lane + 1) < nlive) & (
                u <= fmath.fma(ea, f32c(0x3F800347), f32c(0x358637BD))
            )
            bits = Int32(0)
            for i in range_constexpr(5):
                bits = bits | (
                    (
                        bperm(ok.select(Int32(1), Int32(0)).bitcast(F32), i).bitcast(
                            Int32
                        )
                        & 1
                    )
                    << i
                )
            base = more.select((Int32(1) << accepted) | 1, Int32(1))
            mask = (bits << 1) | base
            e_prev = bperm(ea, fx.max(lane - 1, Int32(0)))
            a_prev = bperm(acc, fx.max(lane - 1, Int32(0)))
            if (lane < 6) & (((mask >> lane) & 1) == 1):
                v = [lse, (lane < 5).select(lseq, F32(0.0)), (lane == 0).select(F32(1.0), e_prev),
                     (lane == 0).select(F32(0.0), a_prev)]  # fmt: skip
                data = fx.Vector.from_elements(
                    [v[0].bitcast(Int32), tag, v[1].bitcast(Int32), tag], Int32
                )
                data2 = fx.Vector.from_elements(
                    [v[2].bitcast(Int32), tag, v[3].bitcast(Int32), tag], Int32
                )
                bo.buffer_store(
                    data, rsrc(ws), (RES_OFF + 32 * lane) >> 2, cache_modifier=CM_DEV
                )
                bo.buffer_store(
                    data2,
                    rsrc(ws),
                    (RES_OFF + 32 * lane + 16) >> 2,
                    cache_modifier=CM_DEV,
                )
                _lds_st(lds, RROW_LDS + 16 * lane, [x.bitcast(Int32) for x in v])
        if lane == 0:
            bo.buffer_store(fx.Vector.from_elements([mask | 0x10000, tag], Int32), rsrc(ws), MASK_OFF >> 2,
                            cache_modifier=CM_DEV)  # fmt: skip
            _lds_st(lds, MASK_LDS, [mask])
    gpu.barrier()
    return tok, mask, accepted


def _fixed60(r):
    """round-half-up(r 2^60) as u64 (r >= 0 finite f32), as TileRT's trunc(fma(f64(r), 2^60, 0.5))."""
    b = r.bitcast(Int32)
    e = (b >> 23) & 0xFF
    m = Int64((b & 0x7FFFFF) | (e != 0).select(Int32(0x800000), Int32(0)))
    sh = (
        e - 90 + (e == 0).select(Int32(1), Int32(0))
    )  # value = m 2^(sh); denormals: exponent 1 - 150
    up = m << fx.min(fx.max(sh, Int32(0)), Int32(40)).to(Int64)
    rs = fx.min(fx.max(-sh, Int32(1)), Int32(63))
    dn = (m + (Int64(1) << (rs - 1).to(Int64))) >> rs.to(Int64)
    return (sh >= 0).select(up, (sh > -63).select(dn, Int64(0)))


def _bperm64(v, src):
    lo = bperm(Int32(v).bitcast(F32), src).bitcast(Int32)
    hi = bperm(Int32(v >> 32).bitcast(F32), src).bitcast(Int32)
    return (Int64(hi) << 32) | Int64(fx.Uint32(lo))


@traced
def _residual_rows(tid, s2, lds, a):
    """CTAs 0..254 wave 0: wait for CTA 255's mask (+0x1200) and the flagged rows' entries (+0x1240) ->
    LDS (CTA 255 staged its own)."""
    lane = tid % 64
    ws = _w(a, 0x250)
    tag = _stat_tag(a)
    if tid < 64:
        mv = _ld(ws, MASK_OFF >> 2, 2, CM_DEV)
        while Int32(mv[1]) != tag:
            spin_pause()
            rocdl.s_sleep(1)
            mv = _ld(ws, MASK_OFF >> 2, 2, CM_DEV)
        mask = Int32(mv[0]) & 0xFFFF
        if lane == 0:
            _lds_st(lds, MASK_LDS, [mask])
        j, i = (lane - 1) >> 2, (lane - 1) & 3
        if (lane >= 1) & (lane <= 24):  # noqa: SIM102 (lane 0's j = -1: no shift by it)
            if ((mask >> j) & 1) == 1:
                ev = _ld(ws, (RES_OFF + 32 * j + 8 * i) >> 2, 2, CM_DEV)
                while Int32(ev[1]) != tag:
                    spin_pause()
                    rocdl.s_sleep(1)
                    ev = _ld(ws, (RES_OFF + 32 * j + 8 * i) >> 2, 2, CM_DEV)
                _lds_st(lds, RROW_LDS + 16 * j + 4 * i, [ev[0]])


def _stat_tag(a):
    samp = _w(a, 0x230)
    ep = Int32(_w(a, 0x248))
    v0 = _ld(_w(a, 0x218), 0)
    return _fmix(ep * Int32(0x3C6EF362) + _ld(samp, 6) * Int32(0xC2B2AE35 - (1 << 32))
                 + v0 * Int32(0x85EBCA6B - (1 << 32)) + Int32(0x9E3779B1 - (1 << 32))) | 1  # fmt: skip


@traced
def _residual(tid, s2, lds, a, tag, vp, cnt):
    """CTAs 0..255 when the mask is set: row w (wave w, mask bit w) over the CTA's entries: log p, log q,
    the residual max(0, p e_prev - q) as 2^60 fixed point summed (u64, wrapping; rows >= 1), log residual
    (acc_prev + log p) + log1p(-q / (p e_prev)) and its Gumbel key (seed of vpos[w]); per row the sum and the
    max key -> ws_sample + 0xd600 ({half, tag})."""
    lane, w = tid % 64, tid // 64
    ws, samp = _w(a, 0x250), _w(a, 0x230)
    rank = Int32(_w(a, 0x240))
    temp = _ld(samp, 2).bitcast(F32)
    one = temp == F32(1.0)
    mask = uniform(_lds_ld(lds, MASK_LDS))
    if mask != 0:
        if tid < 6:
            _lds_st(lds, RSUM_LDS + 8 * tid, [Int32(0), Int32(0)])
            _lds_st(lds, RKEY_LDS + 8 * tid, [Int32(0), Int32(0)])
        gpu.barrier()
        if (w < 6) & (((mask >> w) & 1) == 1):
            has = lane < cnt * 16
            rr = _lds_ld(lds, RROW_LDS + 16 * w, 4)
            lse_p, lse_q = Int32(rr[0]).bitcast(F32), Int32(rr[1]).bitcast(F32)
            e_prev, a_prev = Int32(rr[2]).bitcast(F32), Int32(rr[3]).bitcast(F32)
            dz = _lds_ld(lds, 4 * tid).bitcast(F32)
            lg = _lds_ld(lds, LG_LDS + 256 * w + 4 * lane).bitcast(F32)
            lp = lg - lse_p
            lq = one.select(dz, dz / temp) - lse_q
            sm = Int64(0)
            if w != 0:
                r = fx.max(F32(0.0), fmath.fma(_exp_isa(lp), e_prev, -_exp_isa(lq)))
                sm = has.select(_fixed60(r), Int64(0))
            x = lp + a_prev
            t = _exp_isa(lq - x)
            lr = (t < F32(1.0)).select(x + log1p_ocml(-t), f32c(0xFF800000))
            h1 = _hstep(Int32(0), _kmix(_ld(samp, 0)))
            h2 = _hstep(h1, _kmix(_ld(samp, 1)))
            seed = _hstep(h2, _kmix(_sel6(vp, w)))
            vid = rank * VOCAB_RANK + (s2 + 256 * (lane >> 4)) * 16 + (lane & 15)
            sc = _gumbel_score(lr, F32(1.0), seed, _kmix(vid))
            ok = has & (lr != f32c(0xFF800000))
            kh = ok.select(_key_hi(sc.bitcast(Int32)), Int32(0))
            kl = ok.select(~vid, Int32(0))
            for kk in (1, 2, 4, 8, 16, 32):
                sm = sm + _bperm64(sm, lane ^ kk)
                oh = bperm(kh.bitcast(F32), lane ^ kk).bitcast(Int32)
                ol = bperm(kl.bitcast(F32), lane ^ kk).bitcast(Int32)
                g = _k_gt(oh, ol, kh, kl)
                kh, kl = g.select(oh, kh), g.select(ol, kl)
            if lane == 0:
                _lds_st(lds, RSUM_LDS + 8 * w, [Int32(sm), Int32(sm >> 32)])
                _lds_st(lds, RKEY_LDS + 8 * w, [kl, kh])
        gpu.barrier()
        if tid < 24:
            row = tid >> 2
            if ((mask >> row) & 1) == 1:
                kind = (tid & 2) == 0
                v = _lds_ld(lds, kind.select(RSUM_LDS, RKEY_LDS) + 8 * row, 2)
                half = ((tid & 1) == 0).select(Int32(v[0]), Int32(v[1]))
                bo.buffer_store(fx.Vector.from_elements([half, tag], Int32), rsrc(ws),
                                (RPART_OFF + 2048 * tid + 8 * s2) >> 2, cache_modifier=CM_DEV)  # fmt: skip


def _u64(lo, hi):
    return (Int64(hi) << 32) | Int64(fx.Uint32(lo))


def _u64_gt(a, b):
    return _k_gt(Int32(a >> 32), Int32(a), Int32(b >> 32), Int32(b))


@traced
def _fold_residual(tid, lds, a, tag, mask):
    """CTA 255, mask set: the 255 other CTAs' residual partials (+0xd600) folded into its own LDS row totals
    (u64 sum, u64 max key: order-free), then wave 0 pushes the rank's flagged rows to every rank's sym:sample
    + 0xc6000 and folds the ranks' -> per row (sum, key) at lanes j. Returns (sum lo, sum hi, key lo, key hi)
    of row = lane (wave 0)."""
    lane, w = tid % 64, tid // 64
    ws, tab = _w(a, 0x250), _w(a, 0x238)
    rk = _w(a, 0x240)
    rank, npes = Int32(rk), Int32(rk >> 32)
    par = Int32(_w(a, 0x248)) & 1
    c = tid & 255
    h = tid >> 8
    for q in range_constexpr(3):
        row = h + 2 * q
        sm, ky = Int64(0), Int64(0)
        if (row < 6) & (c != 255) & (((mask >> fx.min(row, Int32(5))) & 1) == 1):
            base = RPART_OFF + 8 * c + 2048 * (4 * row)
            v = [_ld(ws, (base + 2048 * k) >> 2, 2, CM_DEV) for k in range(4)]
            got = (
                (Int32(v[0][1]) == tag)
                & (Int32(v[1][1]) == tag)
                & (Int32(v[2][1]) == tag)
                & (Int32(v[3][1]) == tag)
            )
            while ~got:
                spin_pause()
                rocdl.s_sleep(1)
                v = [_ld(ws, (base + 2048 * k) >> 2, 2, CM_DEV) for k in range(4)]
                got = (
                    (Int32(v[0][1]) == tag)
                    & (Int32(v[1][1]) == tag)
                    & (Int32(v[2][1]) == tag)
                    & (Int32(v[3][1]) == tag)
                )
            sm = _u64(Int32(v[0][0]), Int32(v[1][0]))
            ky = _u64(Int32(v[2][0]), Int32(v[3][0]))
        for kk in (1, 2, 4, 8, 16, 32):
            sm = sm + _bperm64(sm, lane ^ kk)
            o = _bperm64(ky, lane ^ kk)
            ky = _u64_gt(o, ky).select(o, ky)
        if lane == 0:
            _lds_st(
                lds,
                RSCR_LDS + 16 * (4 * (2 * q + h) + (w & 3)),
                [Int32(sm), Int32(sm >> 32), Int32(ky), Int32(ky >> 32)],
            )
    gpu.barrier()
    if tid < 6:
        own_s = _lds_ld(lds, RSUM_LDS + 8 * tid, 2)
        own_k = _lds_ld(lds, RKEY_LDS + 8 * tid, 2)
        sm, ky = _u64(Int32(own_s[0]), Int32(own_s[1])), _u64(
            Int32(own_k[0]), Int32(own_k[1])
        )
        g = (tid & 1) + 2 * (tid >> 1)  # row tid = h + 2 q -> scratch group 2 q + h
        for k in range_constexpr(4):
            p4 = _lds_ld(lds, RSCR_LDS + 16 * (4 * g + k), 4)
            sm = sm + _u64(Int32(p4[0]), Int32(p4[1]))
            o = _u64(Int32(p4[2]), Int32(p4[3]))
            ky = _u64_gt(o, ky).select(o, ky)
        _lds_st(lds, RSUM_LDS + 8 * tid, [Int32(sm), Int32(sm >> 32)])
        _lds_st(lds, RKEY_LDS + 8 * tid, [Int32(ky), Int32(ky >> 32)])
    gpu.barrier()
    pv = [_ldm(tab, 64, 2 * p, npes > p, 2) for p in range(8)]
    peers = [
        rsrc((Int64(uniform(v[1])) << 32) | Int64(fx.Uint32(uniform(v[0])))) for v in pv
    ]
    ov = _ld(tab, rank * 2, 2)
    own = (Int64(uniform(ov[1])) << 32) | Int64(fx.Uint32(uniform(ov[0])))
    if (tid < 6) & (((mask >> fx.min(tid, Int32(5))) & 1) == 1):
        s2v = _lds_ld(lds, RSUM_LDS + 8 * tid, 2)
        k2v = _lds_ld(lds, RKEY_LDS + 8 * tid, 2)
        d1 = fx.Vector.from_elements([Int32(s2v[0]), tag, Int32(s2v[1]), tag], Int32)
        d2 = fx.Vector.from_elements([Int32(k2v[0]), tag, Int32(k2v[1]), tag], Int32)
        word = (RX2_OFF + ((par * 8 + rank) * 6 + tid) * 32) >> 2
        for pp in range_constexpr(8):
            if pp < npes:
                bo.buffer_store(d1, peers[pp], word, cache_modifier=CM_SYS)
                bo.buffer_store(d2, peers[pp], word + 4, cache_modifier=CM_SYS)
    r8, j8 = lane >> 3, lane & 7
    need = (
        (tid < 64)
        & (r8 < npes)
        & (j8 < 6)
        & (((mask >> fx.min(j8, Int32(5))) & 1) == 1)
    )
    word = (RX2_OFF + ((par * 8 + r8) * 6 + j8) * 32) >> 2
    e1 = _ldm(own, NB, word, need, 4, CM_SYS)
    e2 = _ldm(own, NB, word + 4, need, 4, CM_SYS)
    while need & (
        (Int32(e1[1]) != tag)
        | (Int32(e1[3]) != tag)
        | (Int32(e2[1]) != tag)
        | (Int32(e2[3]) != tag)
    ):
        spin_pause()
        e1 = _ldm(own, NB, word, need, 4, CM_SYS)
        e2 = _ldm(own, NB, word + 4, need, 4, CM_SYS)
    sm = need.select(_u64(Int32(e1[0]), Int32(e1[2])), Int64(0))
    ky = need.select(_u64(Int32(e2[0]), Int32(e2[2])), Int64(0))
    for kk in (8, 16, 32):
        sm = sm + _bperm64(sm, lane ^ kk)
        o = _bperm64(ky, lane ^ kk)
        ky = _u64_gt(o, ky).select(o, ky)
    return sm, ky


@traced
def _fold_out(tid, lds, a, tag, nlive, vp, tok, mask, accepted, rsum, rkey):
    """CTA 255 wave 0: the step's accept count n and token (residual rows: TileRT's ratio test on the ranks'
    residual masses), tok_out, the mtp_ctl write-back (dsp_pos, blk_tok / blk_pos, vtok / vpos, ar_acc /
    ar_num, ctl) and X row 0 = embed[token]."""
    lane = tid % 64
    w0 = tid < 64
    vtok = _w(a, 0x210)
    rk = _w(a, 0x240)
    npes = Int32(rk >> 32)
    dr = [_ld(vtok, 1 + j) for j in range(5)]
    n = nlive
    token = tok
    if vp[0] < 0:
        n = Int32(0)
    if mask != 0:
        # ratio test of row j + 1's residual mass against e_j = exp(acc_j)
        jn = fx.min(lane + 1, Int32(63))
        nsum = _bperm64(rsum, jn)
        mass = F32(0.0)
        if (lane + 1) < nlive:
            mf = (Int32(nsum >> 32).to(fx.Uint32).to(fx.Float64) * fx.Float64(4294967296.0)
                  + Int32(nsum).to(fx.Uint32).to(fx.Float64)) * fx.Float64(2.0 ** -60)  # fmt: skip
            # row j + 1's mass only when its mask bit j + 1 is set (TileRT: (mask >> lane) & 2)
            mass = (((mask >> lane) & 2) != 0).select(mf.to(F32), F32(0.0))
        j5 = fx.min(lane, Int32(4))
        ej = _exp_isa(_lds_ld(lds, 0x12694 + 4 * j5).bitcast(F32))
        ratio = ej
        if (lane + 1) < nlive:
            den = (F32(1.0) + mass) - ej
            ratio = (F32(0.0) < den).select(mass / den, F32(1.0))
        flag = (lane < nlive) & (_lds_ld(lds, 0x126AC + 4 * j5).bitcast(F32) <= ratio)
        cnt = Int32(0)
        for i in range_constexpr(
            6
        ):  # 64 - clz64(ballot(flag)): flags only on lanes < s33 <= 5
            fi = bperm(flag.select(Int32(1), Int32(0)).bitcast(F32), i).bitcast(Int32)
            cnt = (fi != 0).select(Int32(i + 1), cnt)
        n = (accepted < 0).select(cnt, accepted)
        rowkey = _bperm64(rkey, n & 63)
        klo = (nlive != accepted).select(Int32(rowkey), Int32(0))
        if (nlive == accepted) & (npes >= 1):
            kh, kl = Int32(0), Int32(0)
            for r in range_constexpr(8):
                if r < npes:
                    kv = _lds_ld(lds, RX_LDS + 136 * r + 88, 2)
                    g = _k_gt(Int32(kv[1]), Int32(kv[0]), kh, kl)
                    kh, kl = g.select(Int32(kv[1]), kh), g.select(Int32(kv[0]), kl)
            klo = kl
        token = ~klo
        if ((accepted | klo) > -1) & (n < nlive):
            token = _sel6([Int32(0)] + dr, n + 1)
    # tok_out: drafts below n, the token at n, -1 above
    outs = [
        (Int32(i) < n).select(
            dr[i] if i < 5 else Int32(-1), (Int32(i) == n).select(token, Int32(-1))
        )
        for i in range(6)
    ]
    tok_out, rec = _w(a, 0x258), _w(a, 0x260)
    if (tid == 0) & (tok_out != 0):
        bo.buffer_store(fx.Vector.from_elements(outs[:4], Int32), rsrc(tok_out), 0)
        bo.buffer_store(fx.Vector.from_elements(outs[4:], Int32), rsrc(tok_out), 4)
    if (tid == 0) & (rec != 0):
        bo.buffer_store(
            fx.Vector.from_elements([n, Int32(0), mask, Int32(0)], Int32), rsrc(rec), 0
        )
        bo.buffer_store(token, rsrc(rec), 4)
    ctl, avtok, avpos, dsp = _w(a, 0x2B8), _w(a, 0x2C0), _w(a, 0x2C8), _w(a, 0x2D0)
    rows, blk_tok, blk_pos = Int32(_w(a, 0x2D8)), _w(a, 0x2E0), _w(a, 0x2E8)
    nblk = Int32(_w(a, 0x2F0))
    ar_acc, ar_num, caps = _w(a, 0x2F8), _w(a, 0x300), _w(a, 0x308)
    pad = Int32(_w(a, 0x310))
    live0 = vp[0] >= 0
    tokx = token
    if ctl != 0:
        c01 = _ld(ctl, 0, 2)
        c35 = [_ld(ctl, 3 + k) for k in range(3)]
        c0, c1 = Int32(c01[0]), Int32(c01[1])
        old_vpos = (w0 & (lane < 6)).select(
            _ldm(avpos, NB, lane, w0 & (lane < 6), 1), Int32(-1)
        )
        gpu.barrier()
        if w0 & (lane < rows):
            ok = (lane < 6) & (lane <= n) & live0
            bo.buffer_store(ok.select(Int32(old_vpos), Int32(-1)), rsrc(dsp), lane)
        tokv = ((n < 6) & live0).select(token, c1)
        tokx = live0.select(token, tokv)
        base = live0.select(c0 + n + 1, c0)
        posv = lane + base + 1
        if w0 & (lane < nblk):
            bo.buffer_store((lane == 0).select(tokv, pad), rsrc(blk_tok), lane)
            bo.buffer_store(
                ((lane < 5) & (posv < c35[0])).select(posv, Int32(-1)),
                rsrc(blk_pos),
                lane,
            )
        if w0 & (lane < 6):
            bo.buffer_store((lane == 0).select(tokv, pad), rsrc(avtok), lane)
            bo.buffer_store((posv < c35[0]).select(posv, Int32(-1)), rsrc(avpos), lane)
        if tid == 0:
            if live0:
                cap_a, cap_n = Int32(caps), Int32(caps >> 32)
                cnt_a = _ld(ar_acc, 0)
                for i in range_constexpr(6):
                    if (cnt_a + 1 + i < cap_a) & (Int32(i) <= n):
                        bo.buffer_store(outs[i], rsrc(ar_acc), cnt_a + 1 + i)
                bo.buffer_store(fx.min(cnt_a + n + 1, cap_a - 1), rsrc(ar_acc), 0)
                cnt_n = _ld(ar_num, 0)
                if cnt_n + 1 < cap_n:
                    bo.buffer_store(n + 1, rsrc(ar_num), cnt_n + 1)
                bo.buffer_store(fx.min(cnt_n + 1, cap_n - 1), rsrc(ar_num), 0)
                bo.buffer_store(
                    fx.Vector.from_elements([c0 + n + 1, tokv], Int32), rsrc(ctl), 0
                )
                bo.buffer_store(c35[2] + n + 1, rsrc(ctl), 5)
            bo.buffer_store(n, rsrc(ctl), 2)
            bo.buffer_store(c35[1] + 1, rsrc(ctl), 4)
    # X row 0 = embed[token] (zeros past the vocabulary), wave 0, 10 x 16 B a lane
    if w0:
        tk = uniform(bperm(tokx.bitcast(F32), 0).bitcast(Int32))
        live = _u_lt(tk, Int32(VOCAB))
        src = _w(a, 0xE8 + 0xD8) + live.select(tk, Int32(0)).to(Int64) * HIDDEN_B
        dst = _w(a, 0xE8 + 0xE8)
        zero = fx.Vector.from_elements([Int32(0)] * 4, Int32)
        for k in range_constexpr(10):
            idx = lane + 64 * k
            v = live.select(
                fx.Vector(bo.buffer_load(rsrc(src), idx * 4, vec_width=4, dtype=T.i32)),
                zero,
            )
            for hc in range_constexpr(4):
                bo.buffer_store(v, rsrc(dst), (hc * HIDDEN_B) // 4 + idx * 4)


@traced
def _entry(tid, cta, lds, a):
    if cta < HEAD_CTAS:
        _head(tid, cta, lds, a)
        cnt = (1009 - cta) // 256 + 1
        tag, nlive, vp = _stats(tid, cta, lds, a, cnt)
        tok, accepted = Int32(-1), Int32(-1)
        if cta == HEAD_CTAS - 1:
            if vp[0] >= 0:
                _fold_gather(tid, lds, a, tag)
                gpu.barrier()
                _fold_publish(tid, lds, a, tag, nlive)
                tok, mask, accepted = _fold_accept(tid, lds, a, tag, nlive, vp)
            else:
                if tid == 0:
                    bo.buffer_store(fx.Vector.from_elements([Int32(0x10000), tag], Int32), rsrc(_w(a, 0x250)),
                                    MASK_OFF >> 2, cache_modifier=CM_DEV)  # fmt: skip
                    _lds_st(lds, MASK_LDS, [Int32(0)])
                gpu.barrier()
        else:
            _residual_rows(tid, cta, lds, a)
            gpu.barrier()
        _residual(tid, cta, lds, a, tag, vp, cnt)
        if cta == HEAD_CTAS - 1:
            mask = uniform(_lds_ld(lds, MASK_LDS))
            rsum, rkey = Int64(0), Int64(0)
            if mask != 0:
                rsum, rkey = _fold_residual(tid, lds, a, tag, mask)
            _fold_out(tid, lds, a, tag, nlive, vp, tok, mask, accepted, rsum, rkey)
    elif cta >= EMBED_CTA0:
        _embed_rows(tid, cta, a)
    elif (cta >= MP_CTA0) & (cta < MP_CTA0 + 40 * MP_SUBS):
        _main_proj(tid, cta, lds, a)
        if cta < MP_CTA0 + WKV_CTAS:
            _window_kv(tid, cta, lds, a)
    elif (cta >= NORM_CTA0) & (cta < NORM_CTA0 + 6):
        _main_norm(tid, cta, lds, a)
    elif (cta >= WIN_CTA0) & (cta < WIN_CTA0 + 18):
        _window_norm(tid, cta, lds, a)


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 99 kernarg words."""
    name = kernel_symbol("tk_head_seed_fold", rows=6, v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_BYTES // 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def head_seed_fold(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
        a56: Int64, a57: Int64, a58: Int64, a59: Int64, a60: Int64, a61: Int64, a62: Int64, a63: Int64,
        a64: Int64, a65: Int64, a66: Int64, a67: Int64, a68: Int64, a69: Int64, a70: Int64, a71: Int64,
        a72: Int64, a73: Int64, a74: Int64, a75: Int64, a76: Int64, a77: Int64, a78: Int64, a79: Int64,
        a80: Int64, a81: Int64, a82: Int64, a83: Int64, a84: Int64, a85: Int64, a86: Int64, a87: Int64,
        a88: Int64, a89: Int64, a90: Int64, a91: Int64, a92: Int64, a93: Int64, a94: Int64, a95: Int64,
        a96: Int64, a97: Int64, a98: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _entry(fx.thread_idx.x, fx.block_idx.x, lds,
              [a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18,
               a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37,
               a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56,
               a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75,
               a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a90, a91, a92, a93, a94,
               a95, a96, a97, a98])  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
        a56: Int64, a57: Int64, a58: Int64, a59: Int64, a60: Int64, a61: Int64, a62: Int64, a63: Int64,
        a64: Int64, a65: Int64, a66: Int64, a67: Int64, a68: Int64, a69: Int64, a70: Int64, a71: Int64,
        a72: Int64, a73: Int64, a74: Int64, a75: Int64, a76: Int64, a77: Int64, a78: Int64, a79: Int64,
        a80: Int64, a81: Int64, a82: Int64, a83: Int64, a84: Int64, a85: Int64, a86: Int64, a87: Int64,
        a88: Int64, a89: Int64, a90: Int64, a91: Int64, a92: Int64, a93: Int64, a94: Int64, a95: Int64,
        a96: Int64, a97: Int64, a98: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        head_seed_fold(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
            a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47,
            a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63,
            a64, a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79,
            a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a90, a91, a92, a93, a94, a95,
            a96, a97, a98,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
