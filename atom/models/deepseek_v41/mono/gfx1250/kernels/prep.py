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


def build(widths, rows_per_request, bcols, ratios, extra_write, cap):
    """The launcher for the given integer widths (dict name -> 4 / 8) and the step's constants."""
    W = dict(widths)
    ROWS = rows_per_request
    RA, RB = ratios
    keyed = (tuple(sorted(W.items())), ROWS, bcols, RA, RB, extra_write, cap)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def body(pos: Int64, ids: Int64, bids: Int64, slots: Int64, cos: Int64, sin: Int64, bt: Int64, kvw: Int64,
             kmeta: Int64, vis1: Int64, vis2: Int64, cp1: Int64, wp1: Int64, cp2: Int64, wp2: Int64, spsrc: Int64,
             spdst: Int64, rpos: Int64, firstp: Int64, tids: Int64, dids: Int64, dgeom: Int64, ctxkvw: Int64,
             ctxcos: Int64, ctxsin: Int64, ibat: Int64, btab: Int64, slotsd: Int64, cnt: Int64, ss: Int64, mbest: Int64,
             n: fx.Int32, req: fx.Int32, nbt: fx.Int32, cc: fx.Int32, btstride: fx.Int32, slot_rows: fx.Int32,
             ring_slots: fx.Int32, wsize: fx.Int32, ncnt: fx.Int32, nss: fx.Int32, nmbest: fx.Int32, nsp: fx.Int32):
        _ = keyed
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        if b == 0:
            if t < n:   # write_step_meta (program 0) and prep_step's rows
                batch = _ld(bids, t, W["bids"])
                live = batch >= I64(0)
                brow = fx.max(batch, I64(0))
                slot = _ld(slots, brow, W["slots"])
                p = _ld(pos, t, W["pos"])
                rel = live.select(slot * I64(slot_rows) + p % I64(ring_slots), I64(-1))
                _st(kvw, t, rel, 4)
                _st(ctxkvw, t, rel, W["ctxkvw"])
                first = fx.max(p - I64(wsize) + I64(1), I64(0))
                _st(kmeta, t * WORDS, first, 4)
                _st(kmeta, t * WORDS + 1, live.select(p + I64(1) - first, I64(0)), 4)
                _st(kmeta, t * WORDS + 2, brow * I64(btstride), 4)
                _st(kmeta, t * WORDS + 3, slot * I64(slot_rows), 4)
                _st(rpos, t, p, W["rpos"])
                if t % ROWS == 0:
                    _st(firstp, t // ROWS, p, W["firstp"])
                _st(tids, t, _ld(ids, t, W["ids"]), W["tids"])
                _st(ibat, t, batch, W["ibat"])
            if t < req * (ROWS - 1):
                _st(dids, t, _ld(ids, (t // (ROWS - 1)) * ROWS + 1 + t % (ROWS - 1), W["ids"]), W["dids"])
            if t < req:
                _st(dgeom, t + 8, _ld(slots, t, W["slots"]), 4)
            if t < nbt:
                _st(slotsd, t, _ld(slots, t, W["slots"]), W["slotsd"])
        if b == 1:   # the draft context's RoPE rows (f32 tables, 32 a row)
            for k0 in range_constexpr(0, 48 * 32, THREADS):
                k = t + k0
                if k < n * 32:
                    src = _ld(pos, k // 32, W["pos"]) * I64(32) + I64(k % 32)
                    hx.gstore(ctxcos + I64(k) * 4, hx.gload(cos + src * 4, fx.Float32))
                    hx.gstore(ctxsin + I64(k) * 4, hx.gload(sin + src * 4, fx.Float32))
        if b == 2:   # fill_step_plans: the visible counts, every plan row -1, then the live rows in order
            if t < n:
                p = _ld(pos, t, W["pos"]).to(I32)
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
                        p = _ld(pos, i, W["pos"]).to(I32)
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
               stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(pos, ids, bids, slots, cos, sin, bt, kvw, kmeta, vis1, vis2, cp1, wp1, cp2, wp2, spsrc, spdst, rpos, firstp,
             tids, dids, dgeom, ctxkvw, ctxcos, ctxsin, ibat, btab, slotsd, cnt, ss, mbest, n, req, nbt, cc, btstride,
             slot_rows, ring_slots, wsize, ncnt, nss, nmbest, nsp).launch(grid=(NCTA,), block=(THREADS,), smem=0,
                                                                        stream=stream)
    return launch
