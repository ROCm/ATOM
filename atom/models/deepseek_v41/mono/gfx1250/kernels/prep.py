# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The step's preparation in one launch (``StepInputs._device_inputs``, captured in the width's preparation graph): what
the sampling parameters' host-to-device copy, ``step_meta.write_step_meta`` (no walk plans), ``step_plans.fill_step_plans``
and ``step_prep.prep_step`` wrote as four launches, the same values.

    CTA 0   the rows (one a thread): the window ring row (kvw, ctx_kvw) and key metadata (kmeta: first window position,
            window count, block-table offset, slot row base) of write_step_meta; prep_step's row copies (rpos, first
            positions, tids, dids, the draft geometry's slots, ibat, slots)
    CTA 1   the draft context's RoPE rows (ctx_cos / ctx_sin: table row = the row's position)
    CTA 2   the visible counts and compress / write plans of the two ratios (fill_step_plans: -1 first, then the rows in
            order by one thread)
    CTA 3   the sampling parameters (sparam) from the page-locked host rows (UVA)
    CTA 4   (eng) Engram's compressed ids (engram_compress: lookup[id], -1 on a dead row; the live leading rows)
    CTA 5   (eng) Engram's lookback snapshot and each verify prefix's cursor candidate (engram_snapshot: the
            compressed ids it reads recomputed from the ids, as CTA 4's: no ordering between the two)
    all     the zeroed counters (cnt, ss), mbest = INT64_MIN, the page tables (bcols columns of each request)

Integer widths (``w``: 4 or 8 bytes) are those of the tensors (``build``'s key)."""
from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx

I32, I64 = fx.Int32, fx.Int64
NCTA, THREADS = 16, 256
WORDS = 4   # step_meta.KMETA


def _ld(p, i, w):
    """Element i of an integer buffer of w-byte elements, as i64."""
    if w == 8:
        return hx.gload(p + I64(i) * 8, I64)
    return hx.gload(p + I64(i) * 4, I32).to(I64)


def _st(p, i, v, w):
    """Element i of an integer buffer of w-byte elements := v (truncated / extended as the buffer's)."""
    if w == 8:
        hx.gstore(p + I64(i) * 8, I64(v))
    else:
        hx.gstore(p + I64(i) * 4, I64(v).to(I32))


def build(widths, rows_per_request, bcols, ratios, extra_write, cap, eng=None, adv=False):
    """The launcher for the given integer widths (dict name -> 4 / 8) and the step's constants. eng: None, or (ngram,
    has a dead mask, stages cursor candidates): the Engram stage's compress and snapshot kernels
    (model_ops/engram/device/hashing.py: _compress_kernel, _engram_snapshot_kernel with COUNTED) in this launch.
    adv: the device round's advance (round/device_round.py ``_advance_kernel``) in this launch: row i of request q is at
    firstp[q] + 1 + anchor[q] - q x rows + i % rows (firstp: the previous step's first positions, the target's verify
    writes them; anchor: its verdict's row 2, still in vptr), in place of the host's positions (rewritten by CTA 0).
    Every launch also writes tcom (sample.COM_*): the step's first positions and, with a staged verify (ncom
    requests), the Engram commit's addresses, which the target's verify reads."""
    W = dict(widths)
    ROWS = rows_per_request
    RA, RB = ratios
    keyed = (tuple(sorted(W.items())), ROWS, bcols, RA, RB, extra_write, cap, eng, adv)
    ENG = eng is not None
    NGR, HAS_MASK, STAGE = eng if ENG else (1, False, False)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def body(pos: Int64, ids: Int64, bids: Int64, slots: Int64, cos: Int64, sin: Int64, bt: Int64, kvw: Int64,
             kmeta: Int64, vis1: Int64, vis2: Int64, cp1: Int64, wp1: Int64, cp2: Int64, wp2: Int64, spsrc: Int64,
             spdst: Int64, rpos: Int64, firstp: Int64, tids: Int64, dids: Int64, dgeom: Int64, ctxkvw: Int64,
             ctxcos: Int64, ctxsin: Int64, ibat: Int64, btab: Int64, slotsd: Int64, cnt: Int64, ss: Int64, mbest: Int64,
             n: fx.Int32, req: fx.Int32, nbt: fx.Int32, cc: fx.Int32, btstride: fx.Int32, slot_rows: fx.Int32,
             ring_slots: fx.Int32, wsize: fx.Int32, ncnt: fx.Int32, nss: fx.Int32, nmbest: fx.Int32, nsp: fx.Int32,
             lookup: Int64, eids: Int64, elive: Int64, edead: Int64, ecomp: Int64, ecu: Int64, ehist: Int64,
             ehidx: Int64, esnap: Int64, ecand: Int64, hstride: fx.Int32, nsnap: fx.Int32, cs0: fx.Int32, cs1: fx.Int32,
             tcom: Int64, ccur: Int64, ncom: fx.Int32, cw: fx.Int32, vptr: Int64):
        _ = keyed
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)

        def P(i):   # row i's position: the host's, or (adv) the device round's from the previous verdict
            if fx.const_expr(adv):
                q = I64(i) // I64(ROWS)
                return (_ld(firstp, q, W["firstp"]) + I64(1) + hx.gload(vptr + (I64(2 * 8) + q) * 4, I32).to(I64)
                        - q * I64(ROWS) + I64(i) % I64(ROWS))
            return _ld(pos, i, W["pos"])
        if b == 0:
            if t == 0:   # the launch's start stamp (sample.COM_T0)
                hx.gstore(tcom + I64(25 * 8), hx.clock())
            if t < n:   # write_step_meta (program 0) and prep_step's rows
                batch = _ld(bids, t, W["bids"])
                live = batch >= I64(0)
                brow = fx.max(batch, I64(0))
                slot = _ld(slots, brow, W["slots"])
                p = P(t)
                if fx.const_expr(adv):
                    _st(pos, t, p, W["pos"])
                rel = live.select(slot * I64(slot_rows) + p % I64(ring_slots), I64(-1))
                _st(kvw, t, rel, 4)
                _st(ctxkvw, t, rel, W["ctxkvw"])
                first = fx.max(p - I64(wsize) + I64(1), I64(0))
                _st(kmeta, t * WORDS, first, 4)
                _st(kmeta, t * WORDS + 1, live.select(p + I64(1) - first, I64(0)), 4)
                _st(kmeta, t * WORDS + 2, brow * I64(btstride), 4)
                _st(kmeta, t * WORDS + 3, slot * I64(slot_rows), 4)
                _st(rpos, t, p, W["rpos"])
                if t % ROWS == 0:   # (firstp itself: the target's verify, after every reader of the old one here)
                    hx.gstore(tcom + (I64(16) + I64(t // ROWS)) * 8, p)
                _st(tids, t, _ld(ids, t, W["ids"]), W["tids"])
                _st(ibat, t, batch, W["ibat"])
            if t < req * (ROWS - 1):
                _st(dids, t, _ld(ids, (t // (ROWS - 1)) * ROWS + 1 + t % (ROWS - 1), W["ids"]), W["dids"])
            if t < req:
                _st(dgeom, t + 8, _ld(slots, t, W["slots"]), 4)
            if t < nbt:
                _st(slotsd, t, _ld(slots, t, W["slots"]), W["slotsd"])
            if t < ncom:   # the commit's slots (sample.commit: ATOM's _commit_cursors' slots = history_index)
                hx.gstore(tcom + (I64(8) + I64(t)) * 8, _ld(ehidx, t, W["ehidx"]) if ENG else I64(0))
            if t == 0:   # sample.COM_*: N, CUR, CAND, CST, CS0, CS1, W, FIRSTP, ..., REQ
                hdr = [I64(ncom), ccur, ecand, I64(hstride), I64(cs0), I64(cs1), I64(cw), firstp]
                for z in range_constexpr(8):
                    hx.gstore(tcom + I64(z * 8), hdr[z])
                hx.gstore(tcom + I64(24 * 8), I64(req))
        if b == 1:   # the draft context's RoPE rows (f32 tables, 32 a row)
            for k0 in range_constexpr(0, 48 * 32, THREADS):
                k = t + k0
                if k < n * 32:
                    src = P(k // 32) * I64(32) + I64(k % 32)
                    hx.gstore(ctxcos + I64(k) * 4, hx.gload(cos + src * 4, fx.Float32))
                    hx.gstore(ctxsin + I64(k) * 4, hx.gload(sin + src * 4, fx.Float32))
        if b == 2:   # fill_step_plans: the visible counts, every plan row -1, then the live rows in order
            if t < n:
                p = P(t).to(I32)
                _st(vis1, t, (p + I32(1)) // I32(RA), W["vis"])
                _st(vis2, t, (p + I32(1)) // I32(RB), W["vis"])
            for f0 in range_constexpr(0, 4 * cap, THREADS):
                f = t + f0
                if f < 4 * cap:
                    pls = [cp1, wp1, cp2, wp2]
                    for z in range_constexpr(4):
                        hx.gstore(pls[z] + I64(f) * 4, I32(-1))
            hx.wait_storecnt0()
            gpu.barrier()
            if t == 0:
                plans = [(RA, cp1, wp1), (RB, cp2, wp2)]
                for z in range_constexpr(2):
                    R, cp, wp = plans[z]
                    ci = I32(0)
                    wi = I32(0)
                    for i_ in range(n):
                        i = I32(i_)
                        p = P(i).to(I32)
                        j = i % ROWS
                        bb = i // ROWS
                        end = p - j + ROWS
                        win = fx.max(I32(R - 1) - j, I32(0))
                        cm = ((p + I32(1)) % I32(R)) == I32(0)
                        wm = (p + I32(R + extra_write)) >= end
                        rec = [i, bb, p, win]
                        if cm & (ci < I32(cap)):
                            for u in range_constexpr(4):
                                hx.gstore(cp + I64(ci * 4 + u) * 4, rec[u])
                        if wm & (wi < I32(cap)):
                            for u in range_constexpr(4):
                                hx.gstore(wp + I64(wi * 4 + u) * 4, rec[u])
                        ci = ci + cm.select(I32(1), I32(0))
                        wi = wi + wm.select(I32(1), I32(0))
        if b == 3:   # the sampling parameters (f32), read from the page-locked host rows
            for k0 in range(0, nsp, THREADS):
                k = t + k0
                if k < nsp:
                    hx.gstore(spdst + I64(k) * 4, hx.gload(spsrc + I64(k) * 4, fx.Float32))
        if fx.const_expr(ENG):
            if b == 4:   # engram_compress (COUNTED): the live leading rows' compressed ids
                tokens = _ld(elive, 0, W["elive"])
                if I64(t) < tokens:
                    v = hx.gload(lookup + _ld(eids, t, W["eids"]) * 4, I32)
                    if fx.const_expr(HAS_MASK):
                        v = (_ld(edead, t, W["edead"]) != I64(0)).select(I32(-1), v)
                    hx.gstore(ecomp + I64(t) * 4, v)
            if b == 5:   # engram_snapshot (COUNTED): rows < nsnap; -2 past the live rows; a verify's cursor candidates
                tokens = _ld(elive, 0, W["elive"])
                for k0 in range(0, nsnap, THREADS):
                    tk = I64(t + k0)
                    if tk < I64(nsnap):
                        live = tk < tokens
                        batch = live.select(_ld(bids, live.select(tk, I64(0)), W["bids"]), I64(0))
                        local = tk - _ld(ecu, batch, W["ecu"])
                        row = _ld(ehidx, batch, W["ehidx"])
                        for shift in range_constexpr(NGR):
                            back = local - I64(shift)
                            fromc = live & (back >= I64(0))
                            x = fromc.select(tk - I64(shift), I64(0))
                            cv = hx.gload(lookup + _ld(eids, x, W["eids"]) * 4, I32).to(I64)
                            if fx.const_expr(HAS_MASK):
                                cv = (_ld(edead, x, W["edead"]) != I64(0)).select(I64(-1), cv)
                            fromh = live & (back < I64(0))
                            hv = _ld(ehist, fromh.select(row * I64(hstride) + I64(NGR - 1) + back, I64(0)), W["ehist"])
                            src = fromc.select(cv, fromh.select(hv, I64(-1)))
                            hx.gstore(esnap + (tk * I64(NGR) + I64(shift)) * 8, live.select(src, I64(-2)))
                            if fx.const_expr(STAGE and shift < NGR - 1):
                                if live:
                                    _st(ecand, batch * I64(cs0) + local * I64(cs1) + I64(NGR - 1 - shift), src, W["ecand"])
                        if fx.const_expr(STAGE):
                            if live:
                                position = P(tk - local)
                                _st(ecand, batch * I64(cs0) + local * I64(cs1), position + local + I64(1), W["ecand"])
        g = b * THREADS + t
        for s in range(0, ncnt, NCTA * THREADS):
            if s + g < ncnt:
                hx.gstore(cnt + I64(s + g) * 4, I32(0))
        for s in range(0, nss, NCTA * THREADS):
            if s + g < nss:
                hx.gstore(ss + I64(s + g) * 4, I32(0))
        for s in range(0, nmbest, NCTA * THREADS):
            if s + g < nmbest:
                hx.gstore(mbest + I64(s + g) * 8, I64(-9223372036854775808))
        for s in range(0, nbt * cc, NCTA * THREADS):
            e = s + g
            if e < nbt * cc:
                rr = e // cc
                cl = e % cc
                _st(btab, rr * bcols + cl, _ld(bt, I64(rr) * I64(btstride) + I64(cl), W["bt"]), W["btab"])

    @flyc.jit
    def launch(pos: Int64, ids: Int64, bids: Int64, slots: Int64, cos: Int64, sin: Int64, bt: Int64, kvw: Int64,
               kmeta: Int64, vis1: Int64, vis2: Int64, cp1: Int64, wp1: Int64, cp2: Int64, wp2: Int64, spsrc: Int64,
               spdst: Int64, rpos: Int64, firstp: Int64, tids: Int64, dids: Int64, dgeom: Int64, ctxkvw: Int64,
               ctxcos: Int64, ctxsin: Int64, ibat: Int64, btab: Int64, slotsd: Int64, cnt: Int64, ss: Int64, mbest: Int64,
               n: fx.Int32, req: fx.Int32, nbt: fx.Int32, cc: fx.Int32, btstride: fx.Int32, slot_rows: fx.Int32,
               ring_slots: fx.Int32, wsize: fx.Int32, ncnt: fx.Int32, nss: fx.Int32, nmbest: fx.Int32, nsp: fx.Int32,
               lookup: Int64, eids: Int64, elive: Int64, edead: Int64, ecomp: Int64, ecu: Int64, ehist: Int64,
               ehidx: Int64, esnap: Int64, ecand: Int64, hstride: fx.Int32, nsnap: fx.Int32, cs0: fx.Int32, cs1: fx.Int32,
               tcom: Int64, ccur: Int64, ncom: fx.Int32, cw: fx.Int32, vptr: Int64,
               stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(pos, ids, bids, slots, cos, sin, bt, kvw, kmeta, vis1, vis2, cp1, wp1, cp2, wp2, spsrc, spdst, rpos, firstp,
             tids, dids, dgeom, ctxkvw, ctxcos, ctxsin, ibat, btab, slotsd, cnt, ss, mbest, n, req, nbt, cc, btstride,
             slot_rows, ring_slots, wsize, ncnt, nss, nmbest, nsp, lookup, eids, elive, edead, ecomp, ecu, ehist,
             ehidx, esnap, ecand, hstride, nsnap, cs0, cs1, tcom, ccur, ncom, cw, vptr).launch(grid=(NCTA,), block=(THREADS,), smem=0,
                                                                        stream=stream)
    return launch
