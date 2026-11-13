# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The whole decode step as one persistent FlyDSL kernel (4-GPU TP, one build per rank and target width).

  target entry   the input ids' embedding -> block state (R = 4 copies, pre (1,0,0,0)) -> layer 0's attention front
  40 target layers  attention body (attention: A, kv, C, [indexer: iq / iw / compressor GEMM, iquant / compressor rows,
                 scorer + top-512 selection], D, E, F + wo_b cluster), TP1 all-reduce -> y1; then the two fronts with one
                 copy of the code (front.front_any: FFN front -> router ids -> routed + shared MoE (moe) -> combine
                 -> TP2 -> y2; attention front of the next layer); the draft context taps on the tap layers
  target tail    head -> 4-rank sampling -> verify (tail); the draft context KV (draft_ctx) inside the sampling
                 exchanges
  draft entry    the draft block's metadata from the verify, block state (anchor embedding / noise rows) -> draft stage
                 0's attention front
  3 draft layers the same layer code traced with the draft shape (TRD rows, 128-expert top-3 router, kmeta_d / kvw_d /
                 rpos_d metadata)
  draft tail     draft head -> Markov block drafts; the step epoch's end

Pointers: kernel args (attention), the device table `fp` (FPTRS, then ATAB, then the per-layer rows LKEYS) and the
step-tail table `tp`. Handoffs are fence-free with device-scope exchange stores. The receive buffers are double-buffered
by step parity (tp_epoch_begin / tp_epoch_end). Every handoff counter, the fixed LDS regions and the layer-table
rows are named in layout.py; the indexer's and the MoE's per-step arithmetic is in index_plan.py / moe_plan.py.
"""
from __future__ import annotations


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
import sys
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels import layout as LY
from atom.models.deepseek_v41.mono.gfx1250.kernels import front as FS
from atom.models.deepseek_v41.mono.gfx1250.kernels import moe as MS
from atom.models.deepseek_v41.mono.gfx1250.kernels import tail as ST
from atom.models.deepseek_v41.mono.gfx1250.kernels import attention as AS
from atom.models.deepseek_v41.mono.gfx1250.kernels import index_query as IS
from atom.models.deepseek_v41.mono.gfx1250.kernels import compressor as CS
from atom.models.deepseek_v41.mono.gfx1250.kernels import draft_ctx as CX
from atom.models.deepseek_v41.mono.gfx1250.kernels import index_score as SC
from atom.models.deepseek_v41.mono.gfx1250.kernels import topk as TK
from atom.models.deepseek_v41.mono.gfx1250.kernels import engram as ES
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import (CL, CTAS, KWOA, NWOA, QL, THREADS, handoff, i64, pf_rows, smem_layout, stage_A,
                         stage_C, stage_D, stage_E, stage_F_cluster, stage_kv, tp_allreduce)

DRAFT_NE, DRAFT_TOPK = D.DRAFT_EXPERTS, D.DRAFT_TOPK
DSPARK_W = D.DSPARK_W          # the DSpark block width: a build's draft requests = its draft rows // 5
TENTRY_ROW = LY.TARGET_ENTRY_ROW       # the layer-table row of the target entry's attention front (tables.StepTables: after 40 + 3 layers and the draft entry)

FRONT_LDS_OFF = LY.FRONT_LDS                # keep the TP payload ([TR][20] f32 at LDS 0) intact
FPTRS = ['R', 'y', 'pre', 'post', 'comb', 'fnh', 'fnl', 'hsc', 'hbs', 'nw', 'gate', 'gbias', 'Rout', 'xn', 'fx8', 'fxs', 'part', 'plog', 'cand', 'cands', 'npre', 'npost', 'ncomb', 'ids', 'wts', 'mout', 'mid8', 'midc', 'afnh', 'afnl', 'ahsc', 'ahbs', 'anw', 'aRout', 'axn', 'anpre', 'anpost', 'ancomb', 'tp2out', 'cyc', 'y1', 'y2', 'shplane', 'kmeta', 'knsel', 'ksel', 'kvw', 'rpos', 'iq', 'iwp', 'iq_out', 'iqs_out', 'iw_out', 'xh', 'cpart', 'ilvl', 'aux8', 'auxs', 'tpep', 'kmeta_d', 'kvw_d', 'rpos_d', 'dR', 'dy', 'dpre', 'dpost', 'dcomb', 'emb', 'noise', 'x8', 'xs', 'ctxtab', 'dgeom', 'tids', 'eng8', 'engs', 'engkv', 'engpart', 'mroute', 'mchist']   # the step's pointers, in a device table (kernel arg `fp`):
# as ~30 more kernel args they pushed SGPRs into spills and a 64-bit VGPR spill whose ~45 scratch reloads each waited,
# by the in-order loadcnt, for every outstanding load of the stage they sat in
ATAB = ("x8 xs wqkv wqkv_s qnw kvnw wqb wqb_s kv sel sink rcos rsin woa woa_s wob wob_s qkv_part qkv ss qr8 qrs kvn q "
        "apart aml o8 os_ y8 ys ypart out opart tpout recv0 recv1 recv2 recv3").split()   # attention pointers (after FPTRS in fp)
LKEYS = ("wqkv wqkv_s wqb wqb_s woa woa_s wob wob_s qnw kvnw sink fnh fnl hsc hbs nw gate gbias "
         "afnh afnl ahsc ahbs anw mw13 ms13 mw2 ms2 rcos rsin shplane ksel knsel kring lflags iqw iqws iww cwkv cnw cape cwk cknw cplan wplan ccap wcap cmain crpp cprow cplane ckv cscore cstate btab bstride slots ibat ibound iplane irpp ilog ilw isel irow imain ibmax icand engw engws enggw engemb").split()   # ksel/knsel: the layer's topk owner's selection; kring: ring start row   # rcos/rsin: this layer's RoPE row table (window or global, by the layer's ratio)   # a*: the attention front at the END of layer Ly = layer Ly+1's weights (host fills)
RECV_HALF = 16 * 40960 * 16 + (2 << 20)   # bytes a parity half of a receive buffer (TP regions + tail)
RKOFF = LY.RKEEP        # LDS: the residual streams kept between the fronts
LCB = LY.LAYER_BYTES             # (+ 128..255: the TP1 / TP2 column-group counters, gpub / gwait)  # bytes of handoff counters a layer (16: the indexer layers' extra phases use 4, 44.., 60)
IDX_LEVELS = 3        # selection-tree capacity (the index capacity buffers.ILW); actual passes depend on the current bound
MC_BYTES = (D.LAYERS + 4) * (4 * 8 * 256 * 4 + 8 * 8 * 4)   # mchist: [layers] ([4 rounds][8 rows][256] + [8 rows][8]) i32
IPF_KB = 32           # instruction prefetch (KB) before the fronts / the MoE
FPF_SPLIT = True      # (front prefetch layout; the step build does not prefetch the front weights)


def fptr(fp, name):
    """Pointer `name` from the table (wave-uniform: a scalar load, scalar cache)."""
    return hx.sload64(fp + i64(FPTRS.index(name) * 8))


@ASTRewriter.transform
def tp_epoch_begin(recv, fp, rank, zero=None):
    """(this step's receive bases, its epoch). Step e uses half e & 1 of every rank's receive buffer; this rank zeroes its
    own other half (the next step's) now: a peer writes there only in step e + 1, which it reaches after this rank's step-e
    pushes (issued after these stores drained at the first handoff), and every step-e word for this rank lands before this
    rank leaves step e (it polls them all)."""
    ep = hx.gload(fptr(fp, "tpep"), fx.Int32)
    half = i64(ep & 1) * RECV_HALF
    other = recv[rank] + i64((ep + 1) & 1) * RECV_HALF
    t = hx.opaque_v(fx.thread_idx.x)
    b = hx.opaque_s(fx.block_idx.x)
    zv = fx.Vector.from_elements([fx.Int32(0)] * 4, fx.Int32)
    n16 = RECV_HALF // 16
    if zero is None:
        zero = fx.Int32(1) != fx.Int32(0)
    if zero:
        for k in range_constexpr((n16 + CTAS * THREADS - 1) // (CTAS * THREADS)):   # (the last pass clamped: repeated zeros)
            hx.gstore(other + i64(fx.min(b * THREADS + t + k * (CTAS * THREADS), fx.Int32(n16 - 1))) * 16, zv, nt=True)
        hx.wait_storecnt0()
    return [r + half for r in recv], ep


@ASTRewriter.transform
def tp_epoch_end(fp, ep, fin=None):
    """The next step's epoch (CTA 0 thread 0, after the last handoff: every CTA read this step's at its start)."""
    t = hx.opaque_v(fx.thread_idx.x)
    b = hx.opaque_s(fx.block_idx.x)
    if fin is None:
        fin = fx.Int32(1) != fx.Int32(0)
    if (b == 0) & (t == 0) & fin:
        hx.gstore(fptr(fp, "tpep"), ep + 1)


class LazyC(dict):
    """Stage context whose pointer entries are emitted at each access: key -> thunk (a scalar table load, + offset)."""
    def __init__(self, thunks, **kw):
        super().__init__(**kw)
        self._th = dict(thunks)
        for k in thunks:
            dict.__setitem__(self, k, None)

    def __getitem__(self, k):
        if k in self._th:
            return self._th[k]()
        return dict.__getitem__(self, k)

    def get(self, k, d=None):
        return self[k] if k in self else d

    def with_(self, **kw):
        n = LazyC({k: v for k, v in self._th.items() if k not in kw},
                  **{k: dict.__getitem__(self, k) for k in self if k not in self._th and k not in kw})
        n.update(kw)
        return n


def cwith(c, **kw):
    return c.with_(**kw) if isinstance(c, LazyC) else dict(c, **kw)


def entry_front(fp, sm, b, row, ne, topk, last=False):
    """A segment entry's attention front (front CTAs): the block state (dR / dy / dpre / dpost / dcomb) through the
    attention front of layer-table row `row` (its a* weights) -> axn / x8 / xs / the gates; R kept in LDS (first)."""
    nof = fx.Int32(0) != fx.Int32(0)
    tq = hx.opaque_v(fx.thread_idx.x)
    cu = dict(t=tq, b=b, w=tq // 32, l=tq % 32, sm=sm + FRONT_LDS_OFF, rms_eps=D.RMS_EPS, hc_eps=1e-6, norm_eps=D.RMS_EPS,
              R=fptr(fp, "dR"), y=fptr(fp, "dy"), pre=fptr(fp, "dpre"), post=fptr(fp, "dpost"), comb=fptr(fp, "dcomb"),
              fnh=ltab(fp, row, "afnh")(), fnl=ltab(fp, row, "afnl")(), hsc=ltab(fp, row, "ahsc")(),
              hbs=ltab(fp, row, "ahbs")(), nw=ltab(fp, row, "anw")(), gate=ltab(fp, row, "gate")(),
              gbias=ltab(fp, row, "gbias")(), Rout=fptr(fp, "aRout"), xn=fptr(fp, "axn"), x8=fptr(fp, "x8"),
              xs=fptr(fp, "xs"), part=fptr(fp, "part"), plog=fptr(fp, "plog"), cand=fptr(fp, "cand"),
              cands=fptr(fp, "cands"), npre=fptr(fp, "anpre"), npost=fptr(fp, "anpost"), ncomb=fptr(fp, "ancomb"),
              ids=fptr(fp, "ids"), wts=fptr(fp, "wts"), isffn=nof, xs32=fx.Int32(1) != fx.Int32(0), merge=nof,
              rk=sm + RKOFF, first=fx.Int32(1) != fx.Int32(0), last=(fx.Int32(1) != fx.Int32(0)) if last else nof, hdev=True, ne=ne, topk=topk)
    FS.front_any(cu)


@ASTRewriter.transform
def gpub(base, b, t):
    """A TP CTA's y (device-scope stores) is out: count it in its column group b // 16 (16 CTAs a group)."""
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(base + i64(b // 16) * 4, 1)


@ASTRewriter.transform
def gpub3(base, b, t):
    """(fx-v3) A TP owner b < 160 (cols 32 b ..) is out: count it in its 320-col group b // 10 (10 owners a group)."""
    hx.wait_storecnt0()
    gpu.barrier()
    if (t == 0) & (b < 160):
        hx.atomic_add(base + i64(b // 10) * 4, 1)


@ASTRewriter.transform
def gwait(base, b, t, n=16):
    """A front CTA (g, p = b % 16): wait for the n TP CTAs of its column group (16; fx-v3 target: 10 owners)."""
    if t == 0:
        hx.wait_count(lambda: base + i64(b % 16) * 4, n, 7)
    gpu.barrier()


def _ctxtab(fp, ndr):   # (plain python) the draft context table when the step has draft layers
    return fptr(fp, "ctxtab") if ndr else None


def ltab(fp, Ly, name):
    """Thunk: per-layer pointer `name` of layer Ly (row Ly of the layer table)."""
    def f():
        return hx.sload64(fp + (i64(len(FPTRS) + len(ATAB)) + i64(Ly) * len(LKEYS) + i64(LKEYS.index(name))) * 8)
    return f


def lsel(fp, Ly, isf, n0, n1):
    """Per-layer pointer n0 (isf) or n1, one scalar load."""
    ix = isf.select(fx.Int32(LKEYS.index(n0)), fx.Int32(LKEYS.index(n1)))
    return hx.sload64(fp + (i64(len(FPTRS) + len(ATAB)) + i64(Ly) * len(LKEYS) + i64(ix)) * 8)


def atab(fp, name, off=None):
    """Thunk: pointer `name` of the attention table (after FPTRS in fp) [+ off]."""
    def f():
        v = hx.sload64(fp + i64((len(FPTRS) + ATAB.index(name)) * 8))
        return v if off is None else v + off
    return f


def iprefetch(kb):
    """Instruction prefetch of the next `kb` KB of code (s_prefetch_inst_pc_rel, 4 KB a request), issued before a
    handoff so the stage after it starts warm (the layer loop is larger than the instruction cache)."""
    for off in range(0, kb * 1024, 4096):
        llvm.inline_asm(None, [], f"s_prefetch_inst_pc_rel {off}, null, 31", "", has_side_effects=True)


CODE_END_OFF = 131024   # cyc scratch: the target / draft launches' end PCs (hx.code_end)


def build(TR, rank, NL=D.LAYERS, ndr=D.DRAFT_LAYERS, trd=None, part="all", sources="", moe_ids="ids", split=False):
    """The step kernel for TR target rows (TR // 6 requests) and trd draft rows (5 a request); NL target layers (NL < 40
    and ndr 0: a bring-up / shadow build of the first layers only). part: "all" (one launch), or the step as two launches
    on one stream: "target" (entry, target layers, head, sampling, verification) then "draft" (draft context, metadata,
    draft layers, draft head, Markov) -- the host reads the verdict after the first while the second runs.
    sources: the kernel sources' digest (the runner's atom.mono.plan.build_key.source_digest of this directory). Every
    parameter is in `keyed`, which the kernel and the launcher reference: all of them reach FlyDSL's JIT cache key (an edit to a helper
    the cache walk does not reach still rebuilds)."""
    assert part in ("all", "target", "draft")
    keyed = (("TR", TR), ("rank", rank), ("NL", NL), ("ndr", ndr), ("trd", trd), ("part", part), ("sources", sources), ("moe_ids", moe_ids), ("split", split))
    L = smem_layout(TR)
    TPG = TR * 10 // 3
    stx = hx.stx_dev
    SEGS = ([(0, NL, 1)] if part != "draft" else []) + ([(NL, NL + ndr, 4)] if ndr and part != "target" else [])
    NDR = ndr
    TRD = TR if trd is None else trd
    LD = smem_layout(TRD)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def body(recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64, cnt: Int64, scale: fx.Float32, fp: Int64,
             tp: Int64, lo: fx.Int32, hi: fx.Int32):
        recv = [recv0, recv1, recv2, recv3]
        recv, tep = tp_epoch_begin(recv, fp, rank, (lo == fx.Int32(0)) if split else None)
        for si in range_constexpr(len(SEGS)):   # layer segments, each its own loop and body (kind 1: target, 4: draft)
            seg_lo, seg_hi, seg_kind = SEGS[si]
            DRS = (seg_kind & 4) != 0
            FIRST_LY = -1
            if fx.const_expr(si == 0 and part != "draft"):
                if lo == 0:   # (split: only the launch that starts the step)
                    # the target entry: SinglePassHCState.from_embeddings (R = 4 copies of the token's embedding, pre = (1, 0, 0, 0),
                    # y = 0, post = 0, comb = I: stage_block_state with W = 1), then layer 0's attention front (its a*: layer-table
                    # row TENTRY_ROW): x8 / xs / the attention gates, R kept in LDS (layer 0's FFN front: first = 0)
                    t = hx.opaque_v(fx.thread_idx.x)
                    b = hx.opaque_s(fx.block_idx.x)
                    sm = hx.lds_base()
                    TCNT = cnt + i64((NL + NDR) * LCB)
                    mz = fptr(fp, "mchist")   # the multi-CTA selection's histograms / position counters: zeroed a step
                    for zz in range_constexpr((MC_BYTES // 16 + 65535) // 65536):
                        u = b * 256 + t + zz * 65536
                        if u < MC_BYTES // 16:
                            hx.gstore(mz + i64(u) * 16, fx.Vector.filled(4, 0, fx.Int32))
                    CX.stage_block_state(dict(t=t, b=b, aid=fptr(fp, "tids"), emb=fptr(fp, "emb"), noise=fptr(fp, "noise"),
                                              R=fptr(fp, "dR"), y=fptr(fp, "dy"), pre=fptr(fp, "dpre"), post=fptr(fp, "dpost"),
                                              comb=fptr(fp, "dcomb")), TR, 1)
                    handoff(TCNT + i64(LY.TARGET_STATE), t, False)
                    if b < FS.FRONT_CTAS:
                        entry_front(fp, sm, b, TENTRY_ROW, FS.NE, FS.TOPK, last=True)   # (v2) R -> aRout for layer 0's column owners
                    handoff(TCNT + i64(LY.TARGET_FRONT), t, False)
            TRx = TRD if DRS else TR
            Lx = LD if DRS else L
            DSUF = "_d" if DRS else ""
            RNE, RTOPK = (DRAFT_NE, DRAFT_TOPK) if DRS else (FS.NE, FS.TOPK)   # the segment's router shape
            if fx.const_expr(DRS):
                t = hx.opaque_v(fx.thread_idx.x)
                b = hx.opaque_s(fx.block_idx.x)
                sm = hx.lds_base()
                TCNT = cnt + i64((NL + NDR) * LCB)
                # the draft loop's code -> L2 (its first pass fetched it from HBM: draft layer 0 ran ~12 us slower)
                hx.code_prefetch(t, b, fptr(fp, "cyc") + i64(CODE_END_OFF + 8), 1024)
                if fx.const_expr(part == "all"):
                    ST.target(tp, t, b, sm, TCNT, recv, rank, head_d=4, ctx=fptr(fp, "ctxtab"), T=TR)
                # the draft context KV (main_proj .. ring) from the target loop's taps
                # (the draft context KV runs inside the target sampling's exchanges: ST.target(ctx=...))
                # the draft entry: the draft block's metadata from this step's verify, its block state, then draft stage 0's
                # attention front (layer-table row NL + NDR)
                if b == 0:
                    CX.draft_meta(dict(t=t, verdict=ST.tptr(tp, "verdict"), rpos=fptr(fp, "rpos"), dgeom=fptr(fp, "dgeom"),
                                       kmeta_d=fptr(fp, "kmeta_d"), kvw_d=fptr(fp, "kvw_d"), rpos_d=fptr(fp, "rpos_d")),
                                  TRD // DSPARK_W, DSPARK_W)
                CX.stage_block_state(dict(t=t, b=b, aid=ST.tptr(tp, "anc"), emb=fptr(fp, "emb"), noise=fptr(fp, "noise"),
                                          R=fptr(fp, "dR"), y=fptr(fp, "dy"), pre=fptr(fp, "dpre"), post=fptr(fp, "dpost"),
                                          comb=fptr(fp, "dcomb")), TRD // DSPARK_W, DSPARK_W)
                handoff(TCNT + i64(LY.DRAFT_STATE), t, False)
                if b < FS.FRONT_CTAS:
                    entry_front(fp, sm, b, NL + NDR, DRAFT_NE, DRAFT_TOPK, last=True)   # (R -> aRout for the owners)
                handoff(TCNT + i64(LY.DRAFT_FRONT), t, False)
            for Ly in range((lo if split and not DRS else seg_lo), (hi if split and not DRS else seg_hi)):
                # (fx-v3) every layer, target and draft: the contract-v3 body (layer_v3); the draft segment with its
                # shape (TRD rows, 128 experts top-3, the draft metadata) and its head input dh
                target_layer(dict(S=sys.modules[__name__], fp=fp, cnt=cnt, scale=scale, tep=tep, rank=rank,
                                     TR=TRD if DRS else TR, NL=NL, L0=seg_lo, LEND=seg_hi, DRS=DRS,
                                     moe_ids="ids" if DRS else moe_ids, dh=ST.tptr(tp, "dh") if DRS else None), Ly)
        if fx.const_expr(part == "target"):   # the step's first launch ends with the head, sampling and verification
            t = hx.opaque_v(fx.thread_idx.x)
            b = hx.opaque_s(fx.block_idx.x)
            sm = hx.lds_base()
            ST.target(tp, t, b, sm, cnt + i64((NL + NDR) * LCB), recv, rank, head_d=4,
                     ctx=_ctxtab(fp, NDR), T=TR)
        if fx.const_expr(NDR > 0 and part != "target"):
            t = hx.opaque_v(fx.thread_idx.x)
            b = hx.opaque_s(fx.block_idx.x)
            sm = hx.lds_base()
            ST.draft(tp, t, b, sm, cnt + i64((NL + NDR) * LCB), recv, rank, head_d=4)
        tp_epoch_end(fp, tep, (hi == fx.Int32(NL)) if split else None)
        hx.code_end(hx.opaque_v(fx.thread_idx.x), hx.opaque_s(fx.block_idx.x),
                    fptr(fp, "cyc") + i64(CODE_END_OFF + (8 if part == "draft" else 0)))
        _ = keyed

    @flyc.jit
    def launch(recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64, cnt: Int64, scale: fx.Float32, fp: Int64,
               tp: Int64, lo: fx.Int32, hi: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        """recv0..3: the 4 ranks' receive buffers; cnt: the step's zeroed handoff counters; scale: the
        attention softmax scale; fp: the pointer table (FPTRS, ATAB, layer rows); tp: the step tail's table."""
        _ = keyed
        body(recv0, recv1, recv2, recv3, cnt, scale, fp, tp, lo, hi).launch(
            grid=(CTAS,), block=(THREADS,), smem=max(L["LDS"], ST.HEAD_LDS if ndr else 0, FRONT_LDS_OFF + max(FS.FRONT_LDS, MS.MOE_LDS), RKOFF + FS.RK_BYTES), cluster=(CL, 1, 1), stream=stream)

    return launch, dict(L, TPG=TPG)


# ==============================================================================================================
# (merged from layer_v3.py)
# (fx-v3) The target layer body, contract v3 (design 6.3 / 6.7), one path, no switches.
# 
# The draft segment keeps the old body (step.py); every target layer (0 .. NL - 1) is this function. The special target
# layers (indexer, compressor, Engram, tap layers, the last layer) are runtime-gated sub-stages of the same chain.
# 
# Chain of one layer Ly (every data edge an LL tile flag unless named a handoff):
#   A      (newA: x from the TP2 owners' apflags) wqkv -> q latent / kv latent        [indexer / Engram-next: + ATTN_A]
#   kv, C  on A2 flags -> kv rows / q -> o-ready flags (kvflag, cflag)               [indexer / Engram-next: + KV_C]
#   (idx)  iq / iw / compressor / iquant / scorer / selection (handoffs: indexer layers only)
#   D      attention, on cflag / kvflag -> dflag;   D-idle CTAs: the attention mix partials + Sinkhorn of this layer
#   E4     wo_a on CTAs 192..255 (out tile, full K, head sum in the CTA), weights before the wait on dflag -> eflag
#   F3     wo_b (160 owners x 32 cols, full K), weights before the wait on eflag;  TP3 (TP1) -> y1
#   post   FFN mHC post in the 160 owners -> pflag (x1, R')       mix partials -> MIX_POST / L1 / Sinkhorn (idle CTAs)
#   router 128 CTAs (gate rows before the wait on pflag) -> LL scores;  shared UG (x8 self, on pflag)
#   top-k  every CTA (own copy, LDS);  MoE UG -> down (LL per expert) -> combine (DN_BLK) -> TP3 (TP2) -> y2
#   post   attention mHC post of layer Ly + 1 in the 160 owners -> apflag (next layer's A)   [last: xh]
#   end    LAYER_END handoff only after the last layer and the tap layers

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off


# plog regions (bytes from plog)
V3F0 = 917504            # the v3 LL flags: + site offset + (Ly % 2) * 4096
P2A, X1A, P3A = 524288, 786432, 851968   # attention side: mix partials, x1, level-1 sums
P2F, X1F, P3F = 65536, 327680, 458752    # FFN side
UGF = 40960              # MoE UG -> down flags
ENGP = 1081344           # Engram gate partials
DNF = 999424             # the MoE's LL flags (UG groups -> down, shared UG -> down): [parity] x 40 KB
# the v3 flag sites: (plog offset from V3F0, epoch32 site)
F_D, F_E, F_A2, F_C, F_KV, F_P, F_X, F_AP, F_W, F_ENG = ((0, 4), (8192, 5), (16384, 6), (24576, 7), (32768, 8),
                                                         (40960, 9), (49152, 10), (57344, 11), (65536, 12), (73728, 13))
# layer counters (bytes from the layer's counter base)
ATT_POST, AMIX_POST, AMIX_L1, ASINK = 440, 444, 448, 452



@ASTRewriter.transform
def target_layer(E, Ly):
    """One target layer. E: S (the step module: fptr ltab atab LazyC cwith iprefetch FRONT_LDS_OFF LCB RECV_HALF),
    fp cnt scale tep rank TR NL moe_ids."""
    S = E["S"]
    fp, cnt, tep, rank, TR, NL = E["fp"], E["cnt"], E["tep"], E["rank"], E["TR"], E["NL"]
    fptr, ltab, cwith = S.fptr, S.ltab, S.cwith
    moe_ids = E["moe_ids"]
    DRS = E.get("DRS", False)            # (the draft segment: TRD rows, 128 experts top-3, the draft metadata)
    RNE, RTOPK, DSUF = D.EXPERTS, FS.TOPK, ""
    if fx.const_expr(DRS):
        RNE, RTOPK, DSUF = D.DRAFT_EXPERTS, D.DRAFT_TOPK, "_d"
    L0, LEND = E.get("L0", 0), E.get("LEND", NL)   # the segment's layer range (target 0 .. NL, draft NL .. NL + NDR)
    FLO = S.FRONT_LDS_OFF
    IDXL = S.IDX_LEVELS
    t = hx.opaque_v(fx.thread_idx.x)
    b = hx.opaque_s(fx.block_idx.x)
    w = t // 32
    l = t % 32
    sm = hx.lds_base()
    lc = cnt + i64(Ly) * S.LCB
    par = Ly % 2
    Lp = fx.max(Ly - 1, fx.Int32(0))
    XHP = fptr(fp, "xh")           # the segment's head input (target: xh, draft: the draft head's dh)
    if fx.const_expr(DRS):
        XHP = E["dh"]
    newA = Ly >= L0 + 1            # a segment's first layer: x from its entry front
    lastL = Ly >= LEND - 1

    def plog(off):
        return fptr(fp, "plog") + i64(off)

    def flag(site, Lf):            # (pointer, tag) of a v3 flag site for layer Lf
        return plog(V3F0 + site[0]) + i64(Lf % 2) * 4096, hx.epoch32(tep, Lf, site[1])

    def lfl(bit):                  # lflags bit test (re-read at each use: not live across the layer)
        return (ltab(fp, Ly, "lflags")() & fx.Int64(bit)) != fx.Int64(0)

    dfl, dtg = flag(F_D, Ly)
    efl, etg = flag(F_E, Ly)
    a2fl, a2tg = flag(F_A2, Ly)
    cfl, ctg = flag(F_C, Ly)
    kvfl, kvtg = flag(F_KV, Ly)
    pfl, ptg = flag(F_P, Ly)
    apfl, aptg = flag(F_AP, Ly)
    apnfl, apntg = flag(F_AP, Ly + 1)
    wfl, wtg = flag(F_W, Ly)
    engfl, engtg = flag(F_ENG, Ly)
    fpf = dict(pflag=pfl, ptag=ptg)                     # FFN post (owners) -> x8_self / router / shared UG
    v3a = dict(a2flag=a2fl, a2tag=a2tg, cflag=cfl, ctag=ctg, kvflag=kvfl, kvtag=kvtg)
    v3f = dict(dflag=dfl, dtag=dtg, eflag=efl, etag=etg)

    # the attention context: pointers re-read from the tables at each use
    lz = {k: S.atab(fp, k) for k in ("tpout x8 xs kv sel qkv_part qkv qr8 qrs kvn q apart aml o8 y8 ys ypart out "
                                     "opart").split()}
    lz.update(os=S.atab(fp, "os_"), ss=S.atab(fp, "ss", i64(Ly) * (AS.SS_LAYER_F32 * 4)),
              recv=(lambda: [S.atab(fp, f"recv{o}")() + i64(tep & 1) * S.RECV_HALF for o in range(4)]))
    lz.update({k: ltab(fp, Ly, k) for k in "wqkv wqkv_s wqb wqb_s woa woa_s wob wob_s qnw kvnw sink rcos rsin".split()})
    Lx = smem_layout(TR)
    c = S.LazyC(lz, **dict(Lx, t=t, b=b, w=w, l=l, sm=sm, scale=E["scale"], stx=hx.stx_dev, rank=rank, xcpol=0))
    c["rpos"] = fptr(fp, "rpos" + DSUF)
    c._th.update(kmeta=lambda: fptr(fp, "kmeta" + DSUF), kvw=lambda: fptr(fp, "kvw" + DSUF),
                 ksel=ltab(fp, Ly, "ksel"), knsel=ltab(fp, Ly, "knsel"))
    for k_ in ("kmeta", "knsel", "ksel", "kvw"):
        dict.__setitem__(c, k_, None)
    c["kring_off"] = fx.Int32(ltab(fp, Ly, "kring")())
    RPC = Lx["RPC"]
    DCT = ((TR + RPC - 1) // RPC) * CL       # the D CTAs (row x key segment)

    if lfl(128):   # the row before an Engram layer: its embeddings -> MXFP8
        ES.stage_engq(cwith(c, engemb=ltab(fp, Ly, "engemb")(), eng8=fptr(fp, "eng8"), engs=fptr(fp, "engs")))

    # ---- A: x from the TP2 owners (apflag), wqkv -> latents (A2 flags)
    keepA = lfl(8 | 128)   # indexer / Engram-next layers: the KV_C handoff (iq / iw partials -> iquant: index-internal, contract 6.4)
    # (fx-v3) the attention input x8 made once: CTAs 224..255 publish it (global + 32 LL flags), A reads it from global
    pub = dict(pubflag=plog(DNF) + i64(par) * 40960 + i64(36864), pubtag=hx.epoch32(tep, Ly, 15))
    if newA:
        AS.stage_Apub(cwith(c, x1a=plog(X1A), part2=plog(P2A), anw=ltab(fp, Lp, "anw")(), apflag=apfl, aptag=aptg,
                            axn=fptr(fp, "axn"), norm_eps=D.RMS_EPS, **pub))
    stage_A(cwith(c, newA=newA, **pub, **v3a, apflag=apfl, aptag=aptg, attpost=lc + i64(ATT_POST), part2=plog(P2A),
                  x1a=plog(X1A), anw=ltab(fp, Lp, "anw")(), norm_eps=D.RMS_EPS, axn=fptr(fp, "axn")))
    pf_rows(c["wqb"] + i64(b) * (16 * QL), QL, 16, QL, t)
    pf_rows(c["wqb"] + i64(b + CTAS) * (16 * QL), QL, 16, QL, t)
    if lfl(128):   # (Engram-next layers only: the Engram kv reads every CTA's eng8; indexer layers wait on LL flags below)
        handoff(lc, t, True)   # LY.ATTN_A
    if lfl(128):   # the next layer's Engram kv
        ES.stage_engkv(cwith(c, eng8=fptr(fp, "eng8"), engs=fptr(fp, "engs"), engw=ltab(fp, Ly, "engw")(),
                             engws=ltab(fp, Ly, "engws")(), engkv=fptr(fp, "engkv")))

    # ---- kv, C (on A2 flags) -> kvflag / cflag
    stage_kv(cwith(c, **v3a))
    stage_C(cwith(c, **v3a))

    # ---- indexer layers: iq / iw / compressor GEMM, iquant / compressor rows, scorer + selection
    isidx = lfl(8)
    if isidx:
        # (contract 6.4) iq reads A2's q blocks + ss (the 40 q-block flags), iw / the compressor GEMM read axn (the 32
        # publishers' flags; layer 0: the entry front's, before the loop): LL waits, not the ATTN_A handoff
        ll_w = hx.opaque_v(fx.thread_idx.x) // 32
        ll_l = hx.opaque_v(fx.thread_idx.x) % 32
        hx.wait_flags(a2fl, AS.QL // 32, a2tg, ll_w, ll_l, 8)
        if newA:
            hx.wait_flags(pub["pubflag"], 32, pub["pubtag"], ll_w, ll_l, 9)
        gpu.barrier()
        ci = cwith(c, iqw=ltab(fp, Ly, "iqw")(), iqws=ltab(fp, Ly, "iqws")(), iww=ltab(fp, Ly, "iww")(),
                   iq=fptr(fp, "iq"), iwp=fptr(fp, "iwp"), axn=fptr(fp, "axn"))
        IS.stage_iq(ci)
        IS.stage_iw(ci)
        lfl1 = ltab(fp, Ly, "lflags")()
        hascomp = (lfl1 & fx.Int64(2)) != fx.Int64(0)
        r1 = (lfl1 & fx.Int64(4)) != fx.Int64(0)
        if hascomp:
            CS.stage_cgemm(cwith(c, cwkv=ltab(fp, Ly, "cwkv")(), axn=fptr(fp, "axn"), cpart=fptr(fp, "cpart")), r1)
    if keepA:
        handoff(lc + i64(LY.KV_C), t, True)
    if isidx:
        IS.stage_iquant(cwith(c, iq=fptr(fp, "iq"), iwp=fptr(fp, "iwp"), iq_out=fptr(fp, "iq_out"),
                              iqs_out=fptr(fp, "iqs_out"), iw_out=fptr(fp, "iw_out")))
        lfl2 = ltab(fp, Ly, "lflags")()
        if (lfl2 & fx.Int64(2)) != fx.Int64(0):
            r1b = (lfl2 & fx.Int64(4)) != fx.Int64(0)
            cc = cwith(c, cpart=fptr(fp, "cpart"), cnw=ltab(fp, Ly, "cnw")(), cape=ltab(fp, Ly, "cape")(),
                       cwk=ltab(fp, Ly, "cwk")(), cknw=ltab(fp, Ly, "cknw")(), cplan=ltab(fp, Ly, "cplan")(),
                       wplan=ltab(fp, Ly, "wplan")(), ccap=fx.Int32(ltab(fp, Ly, "ccap")()), wcap=fx.Int32(ltab(fp, Ly, "wcap")()),
                       cmain=fx.Int32(ltab(fp, Ly, "cmain")()), crpp=fx.Int32(ltab(fp, Ly, "crpp")()),
                       cprow=fx.Int32(ltab(fp, Ly, "cprow")()), cplane=ltab(fp, Ly, "cplane")(), ckv=ltab(fp, Ly, "ckv")(),
                       cscore=ltab(fp, Ly, "cscore")(), cstate=ltab(fp, Ly, "cstate")(), btab=ltab(fp, Ly, "btab")(),
                       bstride=fx.Int32(ltab(fp, Ly, "bstride")()), slots=ltab(fp, Ly, "slots")())
            CS.stage_crow(cc, r1b)
            CS.stage_cring(cc, r1b)
        if ltab(fp, Ly, "ibat")() == fx.Int64(0):   # (no scorer: the INDEX_QUANT handoff here; else inside the scorer)
            handoff(lc + i64(LY.INDEX_QUANT), t, True)
        if ltab(fp, Ly, "ibat")() != fx.Int64(0):
            cs_ = cwith(c, ibat=ltab(fp, Ly, "ibat")(), ibound=ltab(fp, Ly, "ibound")(), iplane=ltab(fp, Ly, "iplane")(),
                        irpp=fx.Int32(ltab(fp, Ly, "irpp")()), btab=ltab(fp, Ly, "btab")(), bstride=fx.Int32(ltab(fp, Ly, "bstride")()),
                        ilog=ltab(fp, Ly, "ilog")(), ilw=fx.Int32(ltab(fp, Ly, "ilw")()), isel=ltab(fp, Ly, "isel")(),
                        irow=ltab(fp, Ly, "irow")(), cprow=fx.Int32(ltab(fp, Ly, "cprow")()), imain=fx.Int32(ltab(fp, Ly, "imain")()),
                        iq_out=fptr(fp, "iq_out"), iw_out=fptr(fp, "iw_out"),
                        prewait=lambda: handoff(lc + i64(LY.INDEX_QUANT), hx.opaque_v(fx.thread_idx.x), True))
            lfl3 = ltab(fp, Ly, "lflags")()
            isrx = (lfl3 & fx.Int64(64)) != fx.Int64(0)
            cs_ = cwith(cs_, ibmax=ltab(fp, Ly, "ibmax")(), icand=ltab(fp, Ly, "icand")(), lvl=fptr(fp, "ilvl"),
                        iprod=(lfl3 & fx.Int64(32)) != fx.Int64(0), ts=None)
            if fx.const_expr(TR <= 6):   # (c1: one request) long FULL contexts score each key tile once for every row
                lg = hx.opaque_v(fx.thread_idx.x) % 32   # (short contexts keep the per-row tasks: more CTAs in parallel)
                bmx = TK.wave_max((lg < TR).select(hx.gload(cs_["ibound"] + i64(fx.min(lg, fx.Int32(TR - 1))) * 4, fx.Int32), fx.Int32(0)))
                if isrx | (bmx <= fx.Int32(SC.GROUPED_MIN_BOUND)):
                    SC.stage_score_any(cs_, isrx)
                else:
                    SC.stage_score_grouped(cs_)
            else:
                SC.stage_score_any(cs_, isrx)
            handoff(lc + i64(LY.SCORE), t, True)
            # (the selection tree's levels past the first only when a row has more than one level-0 list: every row's
            # input count n <= SEG -> level 0 already writes the final picks; its empty level-1 pass + handoff skipped)
            lnr = hx.opaque_v(fx.thread_idx.x) % 32
            bdr = hx.gload(cs_["ibound"] + i64(fx.min(lnr, fx.Int32(TR - 1))) * 4, fx.Int32)
            nrow = fx.max(bdr, fx.Int32(0))
            if isrx:
                nrow = TK.rx_context(cs_, fx.min(lnr, fx.Int32(TR - 1)), bdr)
            nmax = TK.wave_max((lnr < TR).select(nrow, fx.Int32(0)))
            sel_levels, blk_levels = TK.selection_levels(cs_, isrx)   # (2, or 3 past 8192 x 16 index rows)
            lsel = (nmax > fx.Int32(TK.SEG)).select(sel_levels, fx.Int32(1))
            npass = isrx.select(lsel, cs_["iprod"].select(sel_levels + blk_levels, lsel))
            if cs_["iprod"]:   # the candidate-block layer (its top-512 and top-2048 blocks): the selection tree
                for pi in range(npass):
                    pmode = isrx.select(fx.Int32(TK.MODE_RX), (pi < sel_levels).select(fx.Int32(TK.MODE_SEL), fx.Int32(TK.MODE_BLK)))
                    plev = (pi < sel_levels).select(pi, pi - sel_levels)
                    nlev = (pi < sel_levels).select(sel_levels, blk_levels)
                    TK.stage_select_any(cwith(cs_, ts=None), plev, nlev, pmode)
                    handoff(lc + i64(LY.SELECT0) + i64(pi) * 4, t, True)
            else:   # FULL / REINDEX: every row's top-512 by 8 CTAs a row (4 radix rounds, a group count each), one handoff
                mh = fptr(fp, "mchist")
                TK.select_mc(cwith(cs_, ts=None), mh + i64(Ly) * TK.MC_LAYER,
                             mh + i64(Ly) * TK.MC_LAYER + i64(TK.MC_HIST), isrx)
                handoff(lc + i64(LY.SELECT0), t, False)

    # ---- D (on cflag / kvflag) -> dflag;  D-idle CTAs: this layer's attention mix partials + Sinkhorn
    # (design 8 A/B, 11-12: TileRT's separate A2 merge CTAs +2.1 us; the merge stays in the hardware cluster)
    stage_D(cwith(c, **v3f, **v3a))
    MX0 = DCT
    NID = CTAS - MX0
    if newA & (b >= MX0):
        tqa = hx.opaque_v(fx.thread_idx.x)
        for kk in range_constexpr((FS.NOWN + NID - 1) // NID):
            ob = (b - MX0) + kk * NID
            if ob < FS.NOWN:
                FS.mix_owner_g(dict(t=tqa, b=ob, fnh=ltab(fp, Lp, "afnh")(), fnl=ltab(fp, Lp, "afnl")(), Rq=fptr(fp, "aRout"),
                                    part2=plog(P2A), cnt2=lc + i64(AMIX_POST)), TR)
        if (b >= FS.SK0) & (b < FS.SK0 + FS.NL1):
            FS.mix_l1(dict(t=tqa, part2=plog(P2A), part3=plog(P3A), cnt2=lc + i64(AMIX_POST), cnt3=lc + i64(AMIX_L1)), b - FS.SK0)
            if b == FS.SK0 + FS.NL1 - 1:
                FS.sinkhorn2(dict(t=tqa, w=tqa // 32, l=tqa % 32, sm=sm + FLO + 160000, part3=plog(P3A),
                                  hsc=ltab(fp, Lp, "ahsc")(), hbs=ltab(fp, Lp, "ahbs")(), npre=fptr(fp, "anpre"),
                                  npost=fptr(fp, "anpost"), ncomb=fptr(fp, "ancomb"), rms_eps=D.RMS_EPS, hc_eps=1e-6,
                                  cnt3=lc + i64(AMIX_L1), done=lc + i64(ASINK)))

    # ---- wo_a by head (dflag -> wflag), head fold (-> eflag), wo_b 160 x 32 (eflag), TP1 -> y1
    v3e = dict(wpart=c["ypart"], wflag=wfl, wtag=wtg, NDF=DCT)
    AS.stage_E4(cwith(c, **v3f, **v3e))   # (wo_a with the head sum in the CTA: no fold hop)
    AS.stage_F3(cwith(c, **v3f, NEF=64))
    AS.tp3(cwith(c, ytgt=fptr(fp, "y1")), par * 2, 2 * Ly + 1)
    hx.wait_storecnt0()
    gpu.barrier()

    # ---- FFN mHC post in the 160 owners (-> pflag); the router's gate rows issued first; mix partials
    tq6 = hx.opaque_v(fx.thread_idx.x)
    if b >= FS.RCTA0:   # (router_v2) the CTA's 3 experts' gate rows -> LDS while the owner post runs
        FS.router_gate_pre(dict(t=tq6, b=b, sm=sm + FLO, gate=ltab(fp, Ly, "gate")(), ne=RNE))
    if newA:   # pre / post / comb from this layer's attention-side Sinkhorn
        if tq6 == 0:
            hx.wait_count(lc + i64(ASINK), 1, 15)
        gpu.barrier()
    if b < FS.NOWN:
        FS.ffn_post_owner(dict(t=tq6, b=b, w=tq6 // 32, l=tq6 % 32, sm=sm + FLO, y=fptr(fp, "y1"),
                               R=fptr(fp, "aRout"), Rout=fptr(fp, "Rout"), x1g=plog(X1F),
                               pre=fptr(fp, "pre"), post=fptr(fp, "post"), comb=fptr(fp, "comb"),
                               fnh=ltab(fp, Ly, "fnh")(), fnl=ltab(fp, Ly, "fnl")(),
                               part2=plog(P2F), cnt=lc + i64(LY.FFN_POST), cnt2=lc + i64(LY.MIX_POST), **fpf), TR)
    if (b >= FS.CL) & (b < FS.RCTA0):   # (the router CTAs run their mix after the router)
        FS.ffn_mix_owner(dict(t=tq6, b=b, sm=sm + FLO, fnh=ltab(fp, Ly, "fnh")(), fnl=ltab(fp, Ly, "fnl")(),
                              part2=plog(P2F), cnt2=lc + i64(LY.MIX_POST)), TR)
    # (router_v2: 128 CTAs x 3 experts x every row, the gate read once; A/B 11-12 vs TileRT's per-row router: scores 47.5 vs 51.2 us)
    if b >= FS.RCTA0:
        FS.router_v2(dict(t=tq6, b=b, w=tq6 // 32, l=tq6 % 32, sm=sm + FLO, gate=ltab(fp, Ly, "gate")(), xn=fptr(fp, "xn"),
                          x8pub=lc + i64(LY.X8_PUB), x8n=FS.CL, x1g=plog(X1F), nw=ltab(fp, Ly, "nw")(),
                          postcnt=lc + i64(LY.FFN_POST), norm_eps=D.RMS_EPS, scores=fptr(fp, "plog"),
                          tag=hx.epoch32(tep, Ly, 1), gate_pre=True, ne=RNE, **fpf), TR)
        if b < FS.NOWN:   # the router CTAs' own mix partials (R'' kept in LDS past the router's)
            FS.ffn_mix_owner(dict(t=tq6, b=b, sm=sm + FLO, fnh=ltab(fp, Ly, "fnh")(), fnl=ltab(fp, Ly, "fnl")(),
                                  part2=plog(P2F), cnt2=lc + i64(LY.MIX_POST)), TR)
    S.iprefetch(28)
    if b < FS.CL:
        tq7 = hx.opaque_v(fx.thread_idx.x)
        FS.ffn_mix_owner(dict(t=tq7, b=b, sm=sm + FLO, fnh=ltab(fp, Ly, "fnh")(), fnl=ltab(fp, Ly, "fnl")(),
                              part2=plog(P2F), cnt2=lc + i64(LY.MIX_POST)), TR)

    # ---- (design 7) the FFN / MoE block per row (TileRT F0 / top-k wave 0 / H / I / G / G' / J)
    mfl = plog(DNF) + i64(par) * 40960
    cr = dict(t=t, b=b, w=w, l=l, sm=sm + FLO, rank=rank, recv=c["recv"], mflag=mfl, mtag=hx.epoch32(tep, Ly, 14),
              shflag0=6 * MS.TOPK * MS.NUG, ids_lds=moe_ids == "ids", ids=fptr(fp, moe_ids), ktop=RTOPK,
              w13=ltab(fp, Ly, "mw13")(), s13=ltab(fp, Ly, "ms13")(), w2=ltab(fp, Ly, "mw2")(), s2=ltab(fp, Ly, "ms2")(),
              shplane=ltab(fp, Ly, "shplane")(), mid8=fptr(fp, "mid8"), midc=fptr(fp, "midc"), nw=ltab(fp, Ly, "nw")(),
              x1g=plog(X1F), part2=plog(P2F), norm_eps=D.RMS_EPS,
              NP=TR * RTOPK, topk=RTOPK, wts=fptr(fp, "wts"), wts_lds=True, x8=fptr(fp, "fx8"), xs=fptr(fp, "fxs"),
              xflag=plog(V3F0 + F_X[0]) + i64(par) * 4096, xtag=hx.epoch32(tep, Ly, F_X[1]), **fpf)
    S.iprefetch(16)
    shc, rr_, ss_ = MS.rslot(b)
    if shc:   # the shared expert's UG (all rows), then (6 of them) the FFN mix sums + Sinkhorn
        FS.x8_self(dict(t=t, w=w, l=l, sm=sm + FLO, x1g=plog(X1F), part2=plog(P2F), nw=ltab(fp, Ly, "nw")(),
                        norm_eps=D.RMS_EPS, **fpf), TR)
        MS.shared_ug(cr, b - MS.SC0, TR)
        if (b >= MS.SC0 + 6) & (b < MS.SC0 + 6 + FS.NL1):
            FS.mix_l1(dict(t=t, part2=plog(P2F), part3=plog(P3F), cnt2=lc + i64(LY.MIX_POST), cnt3=lc + i64(LY.MIX_L1)),
                      b - (MS.SC0 + 6))
            if b == MS.SC0 + 6 + FS.NL1 - 1:
                FS.sinkhorn2(dict(t=t, w=w, l=l, sm=sm + FLO + 160000, part3=plog(P3F), hsc=ltab(fp, Ly, "hsc")(),
                                  hbs=ltab(fp, Ly, "hbs")(), npre=fptr(fp, "npre"), npost=fptr(fp, "npost"),
                                  ncomb=fptr(fp, "ncomb"), rms_eps=D.RMS_EPS, hc_eps=1e-6,
                                  cnt3=lc + i64(LY.MIX_L1), done=lc + i64(LY.SINK_DONE)))
    else:
        crr = dict(cr, row=fx.min(rr_, fx.Int32(TR - 1)))
        if rr_ < TR:
            MS.f0_row(crr, rr_)
            if ss_ == 0:   # the row's x8 to global for the union UG (one CTA a row)
                MS.x8_pub(crr, rr_)
            MS.shdn_prefetch(crr, ss_)
        # every row's top-k (wave r = row r) and the expert union (each (expert, group) of the UG once, its rows as N)
        FS.topk_v2(dict(t=t, b=b, w=w, l=l, sm=sm + FLO, scores=fptr(fp, "plog"), gbias=ltab(fp, Ly, "gbias")(),
                        tag=hx.epoch32(tep, Ly, 1), ne=RNE, ktop=RTOPK), TR, MS.O_PID, MS.O_WTS, moe_ids == "ids")
        gpu.barrier()
        MS.moe_union(crr)
        MS.ug_union(crr, rr_ * MS.NRC + ss_, TR)
        if rr_ < TR:
            MS.dn_row(crr, ss_)
            MS.push_row(crr, ss_, par * 2 + 1, 2 * Ly + 2)
    if b < FS.NOWN:   # (J) the owner's 32 cols of y2 from the 4 ranks' row partials
        MS.recv_owner(dict(t=t, b=b, rank=rank, recv=c["recv"], ytgt=fptr(fp, "y2")), par * 2 + 1, 2 * Ly + 2, TR)
    hx.wait_storecnt0()
    gpu.barrier()

    # ---- the attention mHC post of layer Ly + 1 in the 160 owners (-> apflag of Ly + 1); the last layer: xh
    if b < FS.NOWN:
        tq5 = hx.opaque_v(fx.thread_idx.x)
        if tq5 == 0:
            hx.wait_count(lc + i64(LY.SINK_DONE), 1, 15)
        gpu.barrier()
        FS.ffn_post_owner(dict(t=tq5, b=b, w=tq5 // 32, l=tq5 % 32, sm=sm + FLO, y=fptr(fp, "y2"),
                               R=fptr(fp, "Rout"), Rout=fptr(fp, "aRout"), x1g=plog(X1A),
                               pre=fptr(fp, "npre"), post=fptr(fp, "npost"), comb=fptr(fp, "ncomb"),
                               part2=plog(P2A), cnt=lc + i64(S.LCB + ATT_POST), pflag=apnfl, ptag=apntg,
                               engkv=fptr(fp, "engkv"), enggw=ltab(fp, Ly, "enggw")(), eng=lfl(128),
                               engp=plog(ENGP), engflag=engfl, engtag=engtg), TR)
        if lastL & (b < 16):
            FS.xh_out(dict(t=tq5, b=b, w=tq5 // 32, l=tq5 % 32, sm=sm + FLO, x1g=plog(X1A), part2=plog(P2A),
                           nw=ltab(fp, Ly, "anw")(), norm_eps=D.RMS_EPS, xh=XHP, pflag=apnfl, ptag=apntg), TR)
    # the next layer's stage-A weights (wqkv tile) and norm weight -> L2
    if (b < 224) & (Ly + 1 < LEND):
        pf_rows(ltab(fp, Ly + 1, "wqkv")() + i64((b // 2) * 16) * D.HID + i64((b % 2) * 2560), D.HID, 16, 2560, t)
    if b < D.HID * 2 // 128:
        pf_rows(ltab(fp, Ly, "anw")() + i64(b) * 128, 128, 1, 128, t)

    # ---- end: a device handoff only before the head and around the tap layers
    if lastL | lfl(16):
        handoff(lc + i64(LY.LAYER_END), t, True)
    if lfl(16):   # the draft context taps (before the next layer's post rewrites aRout)
        CX.stage_tap1(dict(t=hx.opaque_v(fx.thread_idx.x), b=b, aux8=fptr(fp, "aux8"), auxs=fptr(fp, "auxs")),
                      TR, Ly - (NL - 4), fptr(fp, "aRout"))
