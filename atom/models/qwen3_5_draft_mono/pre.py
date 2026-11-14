# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""Drafter pre-attention block, one launch, T <= 16 rows: embedding rows + both pre-fc norms -> fc -> input norm
(its rstd deferred through the q/k/v GEMM) -> q/k/v projection, on the caller's bf16 weights. The attention's own
kernels (q/k norm + rotary + cache write, the attention) follow. Runtime loops only; helpers from post.py."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
from atom.models.qwen3_5_draft_mono import device as hx
from atom.models.qwen3_5_draft_mono import post as Q
from atom.models.qwen3_5_draft_mono.post import F32, I32, BF16, i64, vec, bfr, rsq, burst, wmma_run, reduce8, rstd_reduce, st_x, RS, TMP, TMP2, PW, HID, EPS
from atom.models.qwen3_5_draft_mono.head8 import _ufl

NSTAMP = 16
NQ = 9216


def sum16(v):
    s = F32(v[0])
    for q in range(1, 16):
        s = s + F32(v[q])
    return s


@ASTRewriter.transform
def stage_fc(c, P):
    """b < 128: tile b of the fc (K 4096). The output is the layer's residual (bf16, plain) and, times (1 + w_in),
    the q/k/v stage's x; partial sums of squares for the deferred rstd."""
    b, w, l = c["b"], c["w"], c["l"]
    acc = fx.Vector.filled(8, 0.0, F32)
    if b < 128:
        burst(c, P["xfc"] + i64(w * 16384), 16, c["xbuf"], 0)
        hx.wait_asynccnt0()
        acc = wmma_run(c, 16, c["wbuf"], acc)
    burst(c, P["Wqkv"] + i64(b) * 65536 + i64(w * 8192), 8, c["wbuf"], 2)       # the next stage's first pass
    if b < 128:
        tot = reduce8(c, acc)
        if w == 0:
            tok = l % 16
            col = b * 16 + (l // 16) * 8
            g = hx.gload(P["gin"] + i64(col * 4), F32, 8)
            f = [bfr(tot[e]) for e in range(8)]
            hx.stx_dev(P["resid"] + i64((tok * HID + col) * 2), vec(f).to(BF16))
            st_x(P["xg"], b, c, [f[e] * F32(g[e]) for e in range(8)])
            ss = f[0] * f[0]
            for e in range_constexpr(1, 8):
                ss = ss + f[e] * f[e]
            hx.gstore(P["part"] + i64((tok * 256 + b * 2 + l // 16) * 4), ss, scope="agent")
        gpu.barrier()


@ASTRewriter.transform
def stage_qkv(c, P):
    """Tiles b, b + 256, b + 512 < 576 of [q|gate x 16 heads; k; v] (K 2048), weights double-buffered (two 8 KB halves).
    out rstd, bf16 -> qkv [tok][9216]."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    burst(c, P["xg"] + i64(w * 8192), 8, c["xbuf"], 0)
    rs = hx.lload(sm + RS + tok * 4, F32)
    for p in range(3):
        tile = b + p * 256
        if tile < NQ // 16:
            if tile + 256 < NQ // 16:
                burst(c, P["Wqkv"] + i64(tile + 256) * 65536 + i64(w * 8192), 8, c["wbuf"] + ((p + 1) % 2) * 8192, 2)
                rocdl.s_wait_asynccnt(16)
            else:
                hx.wait_asynccnt0()
            acc = wmma_run(c, 8, c["wbuf"] + (p % 2) * 8192, fx.Vector.filled(8, 0.0, F32))
            tot = reduce8(c, acc)
            if w == 0:
                hx.stx_dev(P["qkv"] + i64((tok * NQ + tile * 16 + (l // 16) * 8) * 2), vec([F32(tot[e]) * rs for e in range(8)]).to(BF16))
            gpu.barrier()


# ---------------------------------------------------------------------------------------------------------------------
# The part before the attention's own kernels, with ATOM's tensors: embedding + pre-fc norms -> fc -> input norm ->
# q/k/v projection. Launch arguments: ids [T] i64, hid [T][2048] bf16, nrows = T <= 16 (rows past T repeat the last).
# Outputs (buffers in the table): resid [16][2048] bf16 (the fc output = the layer's residual), qkv [16][9216] bf16 in
# the row order of the q/k/v weight the host bound (ATOM's: gate, q, k, v).
TABQ = "emb ge gh Wfc Wqkv gin xfc resid xg part qkv".split()


@ASTRewriter.transform
def stage_embed_dyn(c, P):
    b, t, sm = c["b"], c["t"], c["sm"]
    if b < 16:
        row = fx.min(b, P["nrows"] - 1)
        tid = _ufl(hx.gload(P["ids"] + i64(row * P["idw"]), I32))      # ids i32 or i64 (idw = 4 / 8 bytes): the low word
        e = hx.gload(P["emb"] + i64(tid) * (HID * 2) + i64(t * 16), BF16, 8).to(F32)
        h = hx.gload(P["hid"] + i64((row * HID + t * 8) * 2), BF16, 8).to(F32)
        we = hx.gload(P["ge"] + i64(t * 32), F32, 8)
        wh = hx.gload(P["gh"] + i64(t * 32), F32, 8)
        se = F32(e[0]) * F32(e[0])
        sh = F32(h[0]) * F32(h[0])
        for q in range_constexpr(1, 8):
            se = se + F32(e[q]) * F32(e[q])
            sh = sh + F32(h[q]) * F32(h[q])
        hx.lstore(sm + TMP + t * 4, se)
        hx.lstore(sm + TMP2 + t * 4, sh)
        gpu.barrier()
        tr = fx.min(t, I32(15))
        r1 = sum16(hx.lload(sm + TMP + tr * 64, F32, 16))
        r2 = sum16(hx.lload(sm + TMP2 + tr * 64, F32, 16))
        gpu.barrier()
        if t < 16:
            hx.lstore(sm + TMP + t * 4, r1)
            hx.lstore(sm + TMP2 + t * 4, r2)
        gpu.barrier()
        rse = rsq(sum16(hx.lload(sm + TMP, F32, 16)) * (1.0 / HID) + EPS)
        rsh = rsq(sum16(hx.lload(sm + TMP2, F32, 16)) * (1.0 / HID) + EPS)
        for part in range_constexpr(2):
            x, g, rs = (e, we, rse) if part == 0 else (h, wh, rsh)
            col = part * HID + t * 8
            hx.stx_dev(P["xfc"] + i64(col // 32) * 1024 + i64((((col % 32) // 16) * 16 + b) * 32 + (col % 16) * 2),
                       vec([F32(x[q]) * rs * F32(g[q]) for q in range(8)]).to(BF16))


def build_qkv():
    keyed = ("qkv", 2)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(tab: Int64, cnt: Int64, st: Int64, ids: Int64, hid: Int64, nrows: fx.Int32, idw: fx.Int32):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = t // 32
        c = dict(t=t, b=b, w=w, l=t % 32, sm=sm, wbuf=sm + w * PW, xbuf=sm + w * PW + 16384)
        P = {k: hx.sload64(tab + i64(8 * i)) for i, k in enumerate(TABQ)}
        P.update(ids=ids, hid=hid, nrows=nrows, idw=idw)

        def stamp(k):
            if t == 0:
                hx.gstore(st + i64(b) * (NSTAMP * 8) + i64(k * 8), hx.clock())

        stamp(0)
        if b < 128:
            burst(c, P["Wfc"] + i64(b) * 131072 + i64(w * 16384), 16, c["wbuf"], 2)
        stage_embed_dyn(c, P)
        hx.handoff(cnt, t, True)
        stamp(1)
        stage_fc(c, P)
        hx.handoff(cnt + i64(64), t, True)
        stamp(2)
        rstd_reduce(c, P["part"], 256)
        stage_qkv(c, P)
        hx.handoff(cnt + i64(128), t, True)
        stamp(3)
        _ = keyed

    @flyc.jit
    def launch(tab: Int64, cnt: Int64, st: Int64, ids: Int64, hid: Int64, nrows: fx.Int32, idw: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(tab, cnt, st, ids, hid, nrows, idw).launch(grid=(256,), block=(256,), smem=Q.LDS, cluster=(16, 1, 1), stream=stream)

    return launch
