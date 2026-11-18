# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""Engram for the whole-layer kernel (layers 1 and 14). Semantics: atom/model_ops/engram/device/gate.py
(engram_post_wkv_reference).

Row k - 1 of the layer table carries lflags bit 7 ("the next layer has Engram") and that layer's Engram pointers: engw
[25600][6144] e4m3 (rows: 4 branch keys of 5120, then the shared value), engws [800][192] e8m0 (32 x 32 blocks), enggw
[4][5120] f32 (q_weight * k_weight), engemb [T][6144] bf16 (ATOM's staged rows, read in place).

    stage_engq   (layer k - 1, before A; every CTA): engemb -> MXFP8 eng8 [T][6144] / engs [T][192] (one 32-group a thread)
    stage_engkv  (layer k - 1, after A's handoff): kv = eng8 . engw^T, bf16 -> engkv [T][25600]; a wave task = (16-column
                 N tile, all MT M tiles)
The gate itself (R' += gate . value) is the attention front's (front._engram_gate)."""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from flydsl.expr import gpu, rocdl
from flydsl._mlir.dialects import llvm
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import CTAS, T, e8m0_of, inv_pow2, ld_op, q8words

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
EK = 6144            # engram embedding width
EG = EK // 32        # its 32-groups
HC, HD5 = D.HC, D.HID    # branches, hidden
EN = (HC + 1) * HD5  # 25600 kv columns
NT = EN // 16        # N tiles


def i64(x):
    return fx.Int64(x)


@ASTRewriter.transform
def stage_engq(c):
    """Group gi = b * 256 + t of T x EG: 32 bf16 -> amax -> e8m0 code -> 8 MXFP8 words (the kernel's q8words / e8m0_of,
    as every other activation quantization here). Rows >= TR: zeros (code of 0)."""
    t, b = c["t"], c["b"]
    gi = b * 256 + t
    if gi < T * EG:
        row = gi // EG
        g = gi % EG
        live = row < c["TR"]
        src = c["engemb"] + i64(fx.min(row, c["TR"] - 1) * EK + g * 32) * 2
        vs = [hx.gload(src + i64(16 * k), BF16, 8) for k in range(4)]
        vals = [live.select(vs[k][e].to(F32), F32(0.0)) for k in range(4) for e in range(8)]
        am = F32(0.0)
        for e in range_constexpr(32):
            am = fx.max(am, abs(vals[e]))
        code = e8m0_of(am)
        words = q8words(vals, inv_pow2(code))
        dst = c["eng8"] + i64(row * EK + g * 32)
        c["stx"](dst, fx.Vector.from_elements(words[:4], I32))
        c["stx"](dst + i64(16), fx.Vector.from_elements(words[4:], I32))
        c["stx"](c["engs"] + i64(row * EG + g), code.to(fx.Int8))


ED = 4                 # K128 steps of B operands in flight
ED_SH = 4              # (engtp) in flight on the sharded kv (a quarter of the tiles)
KCH = 3072             # K bytes of A staged in LDS at a time (2 chunks)
SPC = KCH // 128       # K128 steps a chunk
RS = KCH + 16          # padded LDS row of the A chunk (bank spread)
# (engtp) the kv sharded by TP rank: rank k computes the hidden columns [k * 1280, (k + 1) * 1280) of every branch
# (ETL = 80 N tiles a branch, ENT = 400 a rank: CTAs 0..49 x 8 waves) and pushes them to the 3 peers' receive regions as
# self-tagged LL words {rows 2p, 2p+1 | tag | rows 2p+2, 2p+3 | tag} (bf16, the same values it stores to its own engkv)
ETL = HD5 // 16 // 4
ENT = 5 * ETL
ETP_TG = 40960         # words a source rank in a receive region (attention.tp3's TP_TG)


def eng_eoff(TR):
    """Word offset of the Engram words in a source rank's slice of receive region par * 2 (past tp3's 256 x TPG words):
    ((TR + 7) // 8) row octets x ENT tiles x 16 cols x 2 words."""
    n = ((TR + 7) // 8) * ENT * 32
    eo = ETP_TG - n
    assert eo >= 256 * ((TR * 10 + 2) // 3), (TR, eo)
    return eo


def _pk2(a, b):
    return a.to(BF16).bitcast(fx.Int16).to(I32) & 0xFFFF | (b.to(BF16).bitcast(fx.Int16).to(I32) << 16)


@ASTRewriter.transform
def stage_engkv(c):
    """kv (see the module doc): a wave task = one 16-column N tile for all MT M tiles (1600 tasks = CTAs 0..199 x 8
    waves: every weight row read once). A (the MT*16 eng8 rows) is staged per CTA in LDS, KCH bytes at a time (async
    copies, issued after the chunk's first ED weight operands), so the A re-reads hit LDS, not L2. Runtime loops over
    the chunks and their K128 steps (code size: this sits in the layer loop). LDS [0, MT*16*RS + MT*16*KCH/32) is free
    here (between attention stages; the layer-resident residual streams sit at RKOFF)."""
    t, b, w, l, MT, sm = c["t"], c["b"], c["w"], c["l"], c["MT"], c["sm"]
    RA = MT * 16
    oa = sm
    osc = sm + RA * RS
    SH = "erecv" in c   # (engtp) this rank's ENT tiles, pushed to the peers
    # (engtp) a quarter of the tiles: 2x the K128 steps in flight a wave (the wave's serial load chain over its 98 KB
    # tile, not the bandwidth, set the stage's time); SPC = 24 is a multiple of both
    EDL = ED_SH if SH else ED
    if b * 8 < (ENT if SH else NT):
        q = b * 8 + w
        if fx.const_expr(SH):
            n0 = (q // ETL) * HD5 + (c["rank"] * ETL + q % ETL) * 16
        else:
            n0 = q * 16
        brow = c["engw"] + i64(n0 + l % 16) * EK
        bsc = c["engws"] + i64(((n0 + l % 16) // 32) * EG)

        def fb(k):
            return ld_op(brow, k * 128, l), hx.gload(bsc + i64(k * 4), I32, align=4)

        acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(MT)]
        for ch in range(EK // KCH):
            k0 = ch * SPC
            ring = [fb(k0 + i) for i in range(EDL)]
            if fx.const_expr("engemb" in c):
                # (engq-in-CTA) the chunk's A rows quantized here from the bf16 embeddings (stage_engq's arithmetic, one
                # 32-group a thread a round: rows >= TR zeros, code of 0), straight into LDS: no eng8 / engs round trip,
                # no stage_engq and no ATTN_A handoff before this stage
                for it in range((RA * (KCH // 32) + 255) // 256):   # (a runtime loop: code size / registers)
                    e = t + it * 256
                    if e < RA * (KCH // 32):
                        row = e // (KCH // 32)
                        g = e % (KCH // 32)
                        live = row < c["TR"]
                        src = c["engemb"] + i64(fx.min(row, c["TR"] - 1) * EK + ch * KCH + g * 32) * 2
                        vs = [hx.gload(src + i64(16 * k), BF16, 8) for k in range(4)]
                        vals = [live.select(vs[k][e_].to(F32), F32(0.0)) for k in range(4) for e_ in range(8)]
                        am = F32(0.0)
                        for e_ in range_constexpr(32):
                            am = fx.max(am, abs(vals[e_]))
                        code = e8m0_of(am)
                        words = q8words(vals, inv_pow2(code))
                        hx.lstore(oa + row * RS + g * 32, fx.Vector.from_elements(words[:4], I32))
                        hx.lstore(oa + row * RS + g * 32 + 16, fx.Vector.from_elements(words[4:], I32))
                        hx.lstore(osc + row * (KCH // 32) + g, code.to(fx.Int8))
            else:
                for it in range((RA * KCH // 16 + 255) // 256):   # A chunk: RA rows x KCH bytes, 16 B a copy
                    e = t + it * 256
                    if e < RA * KCH // 16:
                        row = e // (KCH // 16)
                        col = e % (KCH // 16)
                        hx.async_to_lds_b128(c["eng8"] + i64(row * EK + ch * KCH + col * 16), oa + row * RS + col * 16)
                for it in range_constexpr((RA * KCH // 32 // 16 + 255) // 256):   # its scale codes: RA rows x KCH / 32 bytes
                    e = t + it * 256
                    if e < RA * KCH // 32 // 16:
                        row = e // (KCH // 32 // 16)
                        col = e % (KCH // 32 // 16)
                        hx.async_to_lds_b128(c["engs"] + i64(row * EG + ch * (KCH // 32) + col * 16), osc + row * (KCH // 32) + col * 16)
            hx.wait_asynccnt0()
            gpu.barrier()
            for ks in range(0, SPC, EDL):
                for i in range_constexpr(EDL):
                    bv, bs = ring[i]
                    kk = ks + i
                    for mi in range_constexpr(MT):
                        ar = oa + (mi * 16 + l % 16) * RS + kk * 128 + (l // 16) * 16
                        parts = [hx.lload(ar + cc * 32, I32, 4) for cc in range(4)]
                        av = fx.Vector.from_elements([parts[cc][e_] for cc in range(4) for e_ in range(4)], I32)
                        asc = hx.lload(osc + (mi * 16 + l % 16) * (KCH // 32) + kk * 4, I32)
                        acc[mi] = hx.wmma_mx(av, asc, bv, bs, acc[mi])
                    ring[i] = fb(k0 + fx.min(ks + i + EDL, I32(SPC - 1)))   # (the chunk's last EDL refills: unused)
            gpu.barrier()   # every wave done with this chunk before the next overwrites it
        for mi in range_constexpr(MT):
            for i in range_constexpr(8):
                r = mi * 16 + (l // 16) * 8 + i
                c["stx"](c["engkv"] + i64(r * EN + n0 + l % 16) * 2, acc[mi][i].to(BF16))
            if fx.const_expr(SH):
                if mi * 16 + (l // 16) * 8 < c["TR"]:
                    for u in range_constexpr(2):
                        v = fx.Vector.from_elements([_pk2(acc[mi][4 * u], acc[mi][4 * u + 1]), c["etag"],
                                                     _pk2(acc[mi][4 * u + 2], acc[mi][4 * u + 3]), c["etag"]], I32)
                        slot = ((mi * 2 + l // 16) * ENT + q) * 32 + (l % 16) * 2 + u
                        for o in range_constexpr(4):
                            if o != c["rank"]:
                                hx.gstore(c["erecv"][o] + i64(c["rank"] * ETP_TG + slot) * 16, v, nt=True)


@ASTRewriter.transform
def eng_fetch(c, b, r, cc, kvv):
    """(engtp) The gate owner b's kv of (row r, hidden col cc), 5 branches: its own rank's shard from kvv (the local
    engkv loads), another rank's (owner b's 32 cols: rank b // 40's) from that rank's pushes to this rank's receive
    words (c: erecv, etag, rank), polled on their tags (bounded, the stop protocol as moe.recv_owner)."""
    ksrc = b // (160 // 4)
    rem = ksrc != c["rank"]
    q0 = (cc - ksrc * (HD5 // 4)) // 16
    r8 = r % 8
    eb = c["erecv"] + i64(ksrc * ETP_TG + ((r // 8) * ENT + q0) * 32 + (cc % 16) * 2 + r8 // 4) * 16
    gps = [eb + i64(h * ETL * 32 * 16) for h in range(5)]
    et0 = hx.clock()
    egot = [hx.gload(gp, I32, 4, nt=True) for gp in gps]
    ebad = I32(0)
    for h in range_constexpr(5):
        ebad = ebad | ((egot[h][1] != c["etag"]) | (egot[h][3] != c["etag"])).select(I32(1), I32(0))
    ebad = rem.select(ebad, I32(0))
    est = I32(0)
    while (ebad != 0) & (hx.clock() - et0 < fx.Int64(hx.POLL_TICKS)) & (est == fx.Int32(0)):
        hx.s_sleep(0)
        est = hx._stop_due(est, et0)
        egot = [hx.gload(gp, I32, 4, nt=True) for gp in gps]
        ebad = I32(0)
        for h in range_constexpr(5):
            ebad = ebad | ((egot[h][1] != c["etag"]) | (egot[h][3] != c["etag"])).select(I32(1), I32(0))
    hx._stop_mark((ebad != 0) & (est == fx.Int32(0)))
    hi = (r8 // 2) % 2 == 1
    odd = r % 2 == 1
    wvs = [hi.select(egot[h][2], egot[h][0]) for h in range(5)]
    return [rem.select(odd.select((wvs[h] & I32(-65536)).bitcast(F32), (wvs[h] << 16).bitcast(F32)), kvv[h]) for h in range(5)]


# ---- (engentry) the Engram kv of every Engram layer in the target entry, sharded by TP rank -----------------------------
ED_ENT = 8             # K128 steps of B operands in flight a wave (LDS ring)
RSF = EK + 16          # padded LDS row of the whole A (bank spread)
BRS = 144              # LDS row of a B stage (16 rows x 128 B, padded)
BST = 16 * BRS         # one B stage of a wave
NSLOT = 2              # Engram layers (rows before them: the layer table's rows with lflags ENG_ENTRY)


@ASTRewriter.transform
def stage_engkv_ent(c, e0, ncta, EDL, S):
    """The sharded kv (stage_engkv's SH tiles: rank k's 400 tiles a layer) of NSLOT Engram layers on the CTAs e0 ..
    e0 + ncta - 1 (the entry's CTAs without a front role): CTA-task j = (slot j // 50, tiles (j % 50) 8 + w), one a CTA.
    A (all of K) quantized in the CTA from the slot's bf16 embeddings (stage_engq's arithmetic) into LDS once; the B
    loads come by whole rows into the wave's LDS ring (4 async copies of 4 rows a K128 step: the coalesced ring of the
    entry kv, b64e8aa06), each lane reading its wmma operand back. Stores as stage_engkv: the slot's engkv (slot s at
    engkv + s T EN 2) and the self-tagged pushes to the peers (the slot's receive words). S: per-slot dicts (traced
    pointers / tags): engemb engw engws erecv etag (S(slot), slot a traced i32). The kv bits are those of stage_engkv."""
    t, b, w, l, MT, sm = c["t"], c["b"], c["w"], c["l"], c["MT"], c["sm"]
    TR = c["TR"]
    rank = c["rank"]
    RA = MT * 16
    oa = sm
    osc = sm + RA * RSF
    ob = osc + RA * EG + w * (EDL * BST)
    obw = ob + (l // 8) * BRS + (l % 8) * 16
    obr = ob + (l % 16) * BRS + (l // 16) * 16
    NK = EK // 128
    TPL = ENT // 8                      # CTA-tasks a slot
    bb = b - e0
    if (bb >= 0) & (bb < NSLOT * TPL) & (bb < ncta):
        sl = bb // TPL
        Sd = S(sl)   # (the slot's pointers / tag loaded for this CTA's slot only: fewer live scalars)
        engemb, engw, engws, etag, erecv = Sd["engemb"], Sd["engw"], Sd["engws"], Sd["etag"], Sd["erecv"]
        engkv = c["engkv"] + i64(sl * (T * EN * 2))
        q = (bb % TPL) * 8 + w
        n0 = (q // ETL) * HD5 + (rank * ETL + q % ETL) * 16
        bsrc = engw + i64(n0 + l // 8) * EK + i64((l % 8) * 16)
        bsc = engws + i64(((n0 + l % 16) // 32) * EG)

        def fbl(k, slot):
            for cc in range_constexpr(4):
                hx.async_to_lds_b128(bsrc + i64(cc * 4 * EK) + i64(k * 128), obw + slot * BST + cc * 4 * BRS)

        def fsc(k):
            return hx.gload(bsc + i64(k * 4), I32, align=4)

        for i in range_constexpr(EDL):   # the first EDL steps' B rows in flight while A is quantized
            fbl(i, i)
        sring = [fsc(i) for i in range(EDL)]
        for it in range((RA * EG + 255) // 256):   # A: RA rows x EG 32-groups, quantized here (rows >= TR: zeros, code 0)
            e = t + it * 256
            if e < RA * EG:
                row = e // EG
                g = e % EG
                live = row < TR
                src = engemb + i64(fx.min(row, TR - 1) * EK + g * 32) * 2
                vs = [hx.gload(src + i64(16 * k), BF16, 8) for k in range(4)]
                vals = [live.select(vs[k][e_].to(F32), F32(0.0)) for k in range(4) for e_ in range(8)]
                am = F32(0.0)
                for e_ in range_constexpr(32):
                    am = fx.max(am, abs(vals[e_]))
                code = e8m0_of(am)
                words = q8words(vals, inv_pow2(code))
                hx.lstore(oa + row * RSF + g * 32, fx.Vector.from_elements(words[:4], I32))
                hx.lstore(oa + row * RSF + g * 32 + 16, fx.Vector.from_elements(words[4:], I32))
                hx.lstore(osc + row * EG + g, code.to(fx.Int8))
        gpu.barrier()
        acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(MT)]
        for ks in range(0, NK, EDL):
            for i in range_constexpr(EDL):
                kk = ks + i
                rocdl.s_wait_asynccnt((EDL - 1) * 4)
                bp = [hx.lload(obr + i * BST + cc * 32, I32, 4) for cc in range(4)]
                bv = fx.Vector.from_elements([bp[cc][e_] for cc in range(4) for e_ in range(4)], I32)
                bs = sring[i]
                for mi in range_constexpr(MT):
                    ar = oa + (mi * 16 + l % 16) * RSF + kk * 128 + (l // 16) * 16
                    parts = [hx.lload(ar + cc * 32, I32, 4) for cc in range(4)]
                    av = fx.Vector.from_elements([parts[cc][e_] for cc in range(4) for e_ in range(4)], I32)
                    asc = hx.lload(osc + (mi * 16 + l % 16) * EG + kk * 4, I32)
                    acc[mi] = hx.wmma_mx(av, asc, bv, bs, acc[mi])
                llvm.inline_asm(None, [], "s_wait_dscnt 0x0", "", has_side_effects=True)
                kn = fx.min(kk + EDL, I32(NK - 1))
                fbl(kn, i)
                sring[i] = fsc(kn)
        hx.wait_asynccnt0()
        for mi in range_constexpr(MT):
            for i in range_constexpr(8):
                r = mi * 16 + (l // 16) * 8 + i
                c["stx"](engkv + i64(r * EN + n0 + l % 16) * 2, acc[mi][i].to(BF16))
            if mi * 16 + (l // 16) * 8 < TR:
                for u in range_constexpr(2):
                    v = fx.Vector.from_elements([_pk2(acc[mi][4 * u], acc[mi][4 * u + 1]), etag,
                                                 _pk2(acc[mi][4 * u + 2], acc[mi][4 * u + 3]), etag], I32)
                    slot = ((mi * 2 + l // 16) * ENT + q) * 32 + (l % 16) * 2 + u
                    for o in range_constexpr(4):
                        if o != rank:
                            hx.gstore(erecv[o] + i64(rank * ETP_TG + slot) * 16, v, nt=True)
