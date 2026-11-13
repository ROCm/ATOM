# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The KV / index compressor (Attention._compress_batch) for the whole-layer kernel, layers 2/8/14 (ratio 2) and
20 (ratio 1).

    stage_cgemm (P1, beside kv / C / iq / iw; every CTA, waves 3..5 = the M tiles): wkv_gate [CW][5120] bf16 (CW = 1024 at
                ratio 2: values | scores, 512 at ratio 1) times the normed rows (axn) -> f32 partials CPART[slice][T][CW],
                CW / 16 N tiles x 256 / (CW / 16) K slices.
    stage_crow  (P2, after the next handoff): a wave a compress-plan row [ragged, batch, position, window_len]:
                ratio 2: the online softmax of its 2 sources (the previous position from the ring when j < window_len,
                else the input row) per channel, fp32; ratio 1: the bf16 value. RMSNorm (eps 1e-20, fp32),
                latent = bf16, RoPE of dims 448..511 at (pos // r) r (global table, fp32) -> bf16 main row;
                keys = bf16(latent . wk^T) [128], k_norm -> bf16, RoPE of 64..127 -> bf16 each, power-of-two FP8 (amax floor
                1e-10) -> the index plane's 8-row block.
    stage_cring (P2): a wave a write-plan row [ragged, batch, position, -]: the raw projections -> ring slot pos % 7:
                kv fp32, score fp32 + ape[pos % ratio]. Reads (stage_crow, j = 0: the PREVIOUS position's slot)
                never alias this step's writes (the 7-slot ring holds the 6 written positions apart).

Per-layer constants (LAYER_TAB): lflags bit 1 compressor, bit 2 ratio 1; cwkv cnw cape cwk cknw (weights), cplan wplan
(plan rows), ccap wcap (their capacities), cmain (the owner's main field, pool rows), crpp (rows a page at the ratio),
cprow (pool rows a page), cplane (the owner's index plane, bytes), ckv cscore (the owner's ring fields at slot 0, bytes),
cstate (state bytes a slot), btab bstride (block tables, int32, row stride), slots.
"""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels import mxfp4 as MX4
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import HID, KV8_ROWB, T, e8m0_of, expf, inv_pow2, m_tiles, rsq

F32, I32, I64, BF16 = fx.Float32, fx.Int32, fx.Int64, fx.BFloat16
LAT = D.KV_DIM                  # the compressed latent (head dim)
KEYS = 128                 # the index key
RING = 7                   # compress_ring_slots
CROW0 = 192    # first CTA of stage_crow up to 48 rows (a CTA a row; crow0)
CRING0 = 200   # first CTA of stage_cring


def i64(x):
    return fx.Int64(x)


def crow0(TR):
    """First CTA of stage_crow: a CTA a compress-plan row, CTAS - T_MAX past 48 rows (the plan has <= TR rows)."""
    return CROW0 if TR <= D.NARROW_T else D.CTAS - D.T_MAX


@ASTRewriter.transform
def stage_cgemm(c, r1):
    """r1: the layer is ratio 1 (runtime bool)."""
    t, b, w, l, MT = c["t"], c["b"], c["w"], c["l"], c["MT"]
    cw = r1.select(I32(LAT), I32(2 * LAT))
    ks_n = r1.select(I32(8), I32(4))           # K slices: 256 CTAs / (CW / 16) N tiles
    nt = b // ks_n
    ks = b % ks_n
    kper = HID // ks_n                          # 640 or 1280
    if (w >= 3) & (w < 6) & (w - 3 < MT):
        mis = m_tiles(MT, w - 3)
        k0 = ks * kper
        accs = [fx.Vector.filled(8, 0.0, F32) for _ in mis]
        for st in range(kper // 32):
            kk = k0 + st * 32 + (l // 16) * 16
            avs = [hx.gload(c["axn"] + i64((mi * 16 + l % 16) * HID + kk) * 2, BF16, 16) for mi in mis]
            bv = hx.gload(c["cwkv"] + i64((nt * 16 + l % 16) * HID + kk) * 2, BF16, 16)
            accs = [hx.wmma_bf16(avs[j], bv, accs[j]) for j in range(len(mis))]
        for j in range_constexpr(len(mis)):
            for i in range_constexpr(8):
                tok = mis[j] * 16 + (l // 16) * 8 + i
                c["stx"](c["cpart"] + i64((ks * T + tok) * (2 * LAT) + nt * 16 + l % 16) * 4, accs[j][i])


@ASTRewriter.transform
def _proj(c, r1, row, d0, col0):
    """The projection of input row `row`, columns col0 + d0 .. +15 (values: col0 0, scores: 512): the K slices' partials
    summed in slice order; ratio 1: rounded to bf16 (its GEMM's output dtype). A runtime slice loop (code size)."""
    ks_n = r1.select(I32(8), I32(4))
    v = [F32(0.0)] * 16
    # all 8 slices' loads issued together (a runtime loop waited one load latency a slice), summed in slice order
    ps = [hx.gload(c["cpart"] + i64((fx.min(I32(ks), ks_n - 1) * T + fx.max(row, I32(0))) * (2 * LAT) + col0 + d0) * 4, F32, 16)
          for ks in range(8)]
    for ks in range_constexpr(8):
        v = [(I32(ks) < ks_n).select(v[e] + ps[ks][e], v[e]) for e in range(16)]
    return [r1.select(v[e].to(BF16).to(F32), v[e]) for e in range(16)]


def _lane_sum(x):
    for o in range_constexpr(5):
        x = x + x.shuffle_xor(16 >> o, 32)
    return x


def _lane_max(x):
    for o in range_constexpr(5):
        x = fx.max(x, x.shuffle_xor(16 >> o, 32))
    return x


def _rope_pair(c, cpos, i):
    """cos / sin (f32 of the bf16 global table) of rotation pair i at position cpos."""
    return (hx.gload(c["rcos"] + (i64(cpos) * 32 + i64(i)) * 2, BF16).to(F32),
            hx.gload(c["rsin"] + (i64(cpos) * 32 + i64(i)) * 2, BF16).to(F32))


def _projn(c, r1, row, d0, col0, n):
    """_proj for n channels d0 .. d0 + n - 1 (all 8 slices' loads issued together, summed in slice order)."""
    ks_n = r1.select(I32(8), I32(4))
    ps = [hx.gload(c["cpart"] + i64((fx.min(I32(ks), ks_n - 1) * T + fx.max(row, I32(0))) * (2 * LAT) + col0 + d0) * 4, F32, n)
          for ks in range(8)]
    v = [F32(0.0)] * n
    for ks in range_constexpr(8):
        v = [(I32(ks) < ks_n).select(v[e] + ps[ks][e], v[e]) for e in range(n)]
    return [r1.select(v[e].to(BF16).to(F32), v[e]) for e in range(n)]


@ASTRewriter.transform
def stage_crow(c, r1):
    """A CTA a compress-plan row (CTAs CROW0 .. CROW0 + ccap - 1), its 8 waves splitting the 512 channels (2 a lane):
    each lane's loads are 1/8 of a one-wave-a-row layout (that one held a lane's 4 x 8 x 16 floats in flight, VGPR
    bound). RMSNorm: wave sums, then the 8 partials in wave order from LDS. Keys: the CTA's latent row (bf16, LDS) .
    wk^T as WMMA N tiles (wave w: keys 16 w ..), then wave 0 normalizes / quantizes the 128 keys."""
    kv8 = c.get("kv8", False)   # (a Python bool: read once, outside the traced branches)
    idx4 = c.get("idx4", False)   # (a Python bool: read once, outside the traced branches)
    t, b, w, l = c["t"], c["b"], c["w"], c["l"]
    # the row index per lane (opaque_v): with a wave-uniform index LLVM selects a scaled-SGPR-offset global load that
    # gfx1250 rejects ("scale_offset is not supported for this instruction")
    C0 = crow0(c["TR"])
    q = hx.opaque_v(b - C0)
    act = (b >= C0) & (b - C0 < c["ccap"])
    qq = fx.max(fx.min(q, c["ccap"] - 1), I32(0))
    rag = hx.gload(c["cplan"] + i64(qq * 4) * 4, I32)
    bat = hx.gload(c["cplan"] + i64(qq * 4 + 1) * 4, I32)
    pos = hx.gload(c["cplan"] + i64(qq * 4 + 2) * 4, I32)
    wl = hx.gload(c["cplan"] + i64(qq * 4 + 3) * 4, I32)
    live = act & (pos >= 0)
    ratio = r1.select(I32(1), I32(2))
    d0 = (w % 4) * 128 + l * 4   # waves 4..7 repeat waves 0..3's channels (only 0..3 store)
    lb = c["sm"]                     # bf16 [16][LAT]: row 0 the latent, rows 1..15 zero (the WMMA's M padding)
    kb = c["sm"] + 16 * LAT * 2      # f32 [KEYS]
    red = kb + KEYS * 4              # f32 [8] the waves' sums of squares
    if act:   # (CTA-uniform; a dead row (pos < 0) computes on clamped addresses, stores nothing)
        posc = fx.max(pos, I32(0))
        slot = hx.gload(c["slots"] + i64(fx.max(bat, I32(0))) * 4, I32)
        cpos = (posc // ratio) * ratio
        ci = posc // ratio
        phys = hx.gload(c["btab"] + (i64(fx.max(bat, I32(0))) * i64(c["bstride"]) + i64(ci // c["crpp"])) * 4, I32)
        nw = hx.gload(c["cnw"] + i64(d0) * 2, BF16, 4)
        ape0 = hx.gload(c["cape"] + i64(d0) * 4, F32, 4)
        ape1 = hx.gload(c["cape"] + i64(LAT + d0) * 4, F32, 4)
        rps = [_rope_pair(c, cpos, fx.max((d0 + e) // 2 - 224, I32(0))) for e in range(0, 4, 2)]
        cur_v = _projn(c, r1, rag, d0, 0, 4)
        pv = _projn(c, r1, rag - 1, d0, 0, 4)
        ps = _projn(c, r1, rag - 1, d0, LAT, 4)
        cs_ = _projn(c, r1, rag, d0, LAT, 4)
        s_prev = posc - 1
        from_ring = wl > 0
        rs = fx.max(s_prev, I32(0)) % RING
        rbase = i64(slot) * i64(c["cstate"]) + i64(rs * LAT + d0) * 4
        rkv = hx.gload(c["ckv"] + rbase, F32, 4)
        rsc = hx.gload(c["cscore"] + rbase, F32, 4)
        ok0 = s_prev >= 0
        comp = []
        for e in range_constexpr(4):
            kv0 = from_ring.select(rkv[e], pv[e])
            sc0 = from_ring.select(rsc[e], ps[e] + ape0[e])
            kv0 = ok0.select(kv0, F32(0.0))
            sc0 = ok0.select(sc0, F32(-1e30))
            sc1 = cs_[e] + ape1[e]
            m1 = fx.max(sc0, sc1)
            e0 = expf(sc0 - m1)
            e1 = expf(sc1 - m1)
            comp.append(r1.select(cur_v[e], (kv0 * e0 + cur_v[e] * e1) / (e0 + e1)))
        ws = _lane_sum(comp[0] * comp[0] + comp[1] * comp[1] + comp[2] * comp[2] + comp[3] * comp[3])
        if (l == 0) & (w < 4):
            hx.lstore(red + w * 4, ws)
        gpu.barrier()
        tot = hx.lload(red, F32)
        for ww in range_constexpr(1, 4):
            tot = tot + hx.lload(red + ww * 4, F32)
        rr = rsq(tot * F32(1.0 / LAT) + F32(D.RMS_EPS))
        nrm = [(comp[e] * rr) * nw[e].to(F32) for e in range(4)]
        isr = d0 >= LAT - 64
        rot = []
        for e in range_constexpr(0, 4, 2):
            cs, sn = rps[e // 2]
            rot.append(isr.select(nrm[e] * cs - nrm[e + 1] * sn, nrm[e]))
            rot.append(isr.select(nrm[e + 1] * cs + nrm[e] * sn, nrm[e + 1]))
        mrow = i64(phys) * i64(c["cprow"]) + i64(c["cmain"] + ci % c["crpp"])
        if fx.const_expr(kv8):
            # the FP8 pool's row (quantize_fp8 of the bf16 rotated row): a 32-dim group's 8 lanes share its code
            rb = [rot[e].to(BF16).to(F32) for e in range(4)]
            mam = fx.max(fx.max(abs(rb[0]), abs(rb[1])), fx.max(abs(rb[2]), abs(rb[3])))
            for o_ in range_constexpr(3):
                mam = fx.max(mam, mam.shuffle_xor(1 << o_, 32))
            mcode = e8m0_of(mam)
            minv = inv_pow2(mcode)
            mcl = [fx.min(fx.max(rb[e] * minv, F32(-448.0)), F32(448.0)) for e in range(4)]
            mword = hx.cvt_pk_fp8_w(hx.cvt_pk_fp8_w(0, mcl[0], mcl[1], False), mcl[2], mcl[3], True)
            if live & (w < 4):
                c["stx"](c["kv"] + mrow * KV8_ROWB + i64(d0), mword)
                if l % 8 == 0:
                    c["stx"](c["kv"] + mrow * KV8_ROWB + i64(LAT + d0 // 32), mcode.to(fx.Int8))
        else:
            if live & (w < 4):
                c["stx"](c["kv"] + (mrow * LAT + i64(d0)) * 2, fx.Vector.from_elements([rot[e].to(BF16) for e in range(4)], BF16).bitcast(fx.Int64)[0])
        if w < 4:
            hx.lstore(lb + d0 * 2, fx.Vector.from_elements([live.select(nrm[e], F32(0.0)).to(BF16) for e in range(4)], BF16))
        # M padding rows 1..15 of lb: zero (15 x 1024 B, 16 B a thread x 4)
        for it in range_constexpr(4):
            i = t + it * 256
            if i < 15 * LAT * 2 // 16:
                hx.lstore(lb + LAT * 2 + i * 16, fx.Vector.filled(4, 0, I32))
        gpu.barrier()
        wkr = c["cwk"] + i64((w * 16 + l % 16) * LAT) * 2
        bvs = [hx.gload(wkr + i64(ks * 32 + (l // 16) * 16) * 2, BF16, 16) for ks in range(LAT // 32)]
        acc = fx.Vector.filled(8, 0.0, F32)
        for ks in range_constexpr(LAT // 32):
            av = hx.lload(lb + ((l % 16) * LAT + ks * 32 + (l // 16) * 16) * 2, BF16, 16)
            acc = hx.wmma_bf16(av, bvs[ks], acc)
        if l < 16:   # lane l (< 16): row 0 (i = 0) of key 16 w + l
            hx.lstore(kb + (w * 16 + l) * 4, acc[0])
        gpu.barrier()
        if live & (w == 0):
            ka = hx.lload(kb + (l * 4) * 4, F32, 4)
            key = [ka[jj].to(BF16).to(F32) for jj in range(4)]
            sk = F32(0.0)
            for jj in range_constexpr(4):
                sk = sk + key[jj] * key[jj]
            rk = rsq(_lane_sum(sk) * F32(1.0 / KEYS) + F32(D.RMS_EPS))
            kw = hx.gload(c["cknw"] + i64(l * 4) * 2, BF16, 4)
            kn = [((key[jj] * rk) * kw[jj].to(F32)).to(BF16).to(F32) for jj in range(4)]
            ratio2 = r1.select(I32(1), I32(2))
            cpos2 = (posc // ratio2) * ratio2
            kr = []
            for jj in range_constexpr(0, 4, 2):
                j = l * 4 + jj
                ip = fx.max(j // 2 - 32, I32(0))
                cs2, sn2 = _rope_pair(c, cpos2, ip)
                isr2 = j >= KEYS - 64
                ev, od = kn[jj], kn[jj + 1]
                kr.append(isr2.select((ev * cs2 - od * sn2).to(BF16).to(F32), ev))
                kr.append(isr2.select((od * cs2 + ev * sn2).to(BF16).to(F32), od))
            row = posc // ratio2
            phys2 = hx.gload(c["btab"] + (i64(fx.max(bat, I32(0))) * i64(c["bstride"]) + i64(row // c["crpp"])) * 4, I32)
            flat = i64(phys2) * i64(c["crpp"]) + i64(row % c["crpp"])
            if fx.const_expr(idx4):
                # the FP4 plane (natural layout, index_write._write_index_fp4_kernel): 64 B a row, the even dim the low
                # nibble; one ceil E8M0 a 32-dim group (8 lanes) in the scale plane, 4 B a row
                am4 = fx.max(fx.max(abs(kr[0]), abs(kr[1])), fx.max(abs(kr[2]), abs(kr[3])))
                for o_ in range_constexpr(3):
                    am4 = fx.max(am4, am4.shuffle_xor(1 << o_, 32))
                code4, qs4 = MX4.ceil_scale(am4)
                c["stx"](c["cplane"] + flat * 64 + i64(l * 2), MX4.pack4(kr, qs4).to(fx.Int16))
                if l % 8 == 0:
                    c["stx"](c["iscl"] + flat * 4 + i64(l // 8), code4.to(fx.Int8))
            else:
                am = _lane_max(fx.max(fx.max(abs(kr[0]), abs(kr[1])), fx.max(abs(kr[2]), abs(kr[3]))))
                code = e8m0_of(am)
                inv = inv_pow2(code)
                cl = [fx.min(fx.max(kr[jj] * inv, F32(-448.0)), F32(448.0)) for jj in range(4)]
                word = hx.cvt_pk_fp8_w(hx.cvt_pk_fp8_w(0, cl[0], cl[1], False), cl[2], cl[3], True)
                tile = flat // 8
                tt = flat % 8
                blk = c["cplane"] + tile * 1056
                c["stx"](blk + i64((l // 4) * 128) + tt * 16 + i64((l % 4) * 4), word)
                if l == 0:
                    c["stx"](blk + 1024 + tt * 4, (code << 23).bitcast(F32))   # the fp32 scale 2^(code - 127)


@ASTRewriter.transform
def stage_cring(c, r1):
    t, b, w, l = c["t"], c["b"], c["w"], c["l"]
    q = (b - CRING0) * 8 + w
    act = (b >= CRING0) & (q < c["wcap"])
    qq = fx.max(fx.min(q, c["wcap"] - 1), I32(0))
    rag = hx.gload(c["wplan"] + i64(qq * 4) * 4, I32)
    bat = hx.gload(c["wplan"] + i64(qq * 4 + 1) * 4, I32)
    pos = hx.gload(c["wplan"] + i64(qq * 4 + 2) * 4, I32)
    if act & (pos >= 0):
        d0 = l * 16
        ratio = r1.select(I32(1), I32(2))
        slot = hx.gload(c["slots"] + i64(fx.max(bat, I32(0))) * 4, I32)
        v = _proj(c, r1, rag, d0, 0)
        s = _proj(c, r1, rag, d0, LAT)
        ape = hx.gload(c["cape"] + i64((pos % ratio) * LAT + d0) * 4, F32, 16)
        sc = [r1.select(v[e], s[e]) + ape[e] for e in range(16)]
        base = i64(slot) * i64(c["cstate"]) + i64((pos % RING) * LAT + d0) * 4
        for h in range_constexpr(4):
            c["stx"](c["ckv"] + base + h * 16, fx.Vector.from_elements(v[4 * h:4 * h + 4], F32))
            c["stx"](c["cscore"] + base + h * 16, fx.Vector.from_elements(sc[4 * h:4 * h + 4], F32))
