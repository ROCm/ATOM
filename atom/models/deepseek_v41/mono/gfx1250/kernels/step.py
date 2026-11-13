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
FPTRS = ['R', 'y', 'pre', 'post', 'comb', 'fnh', 'fnl', 'hsc', 'hbs', 'nw', 'gate', 'gbias', 'Rout', 'xn', 'fx8', 'fxs', 'part', 'plog', 'cand', 'cands', 'npre', 'npost', 'ncomb', 'ids', 'wts', 'mout', 'mid8', 'midc', 'afnh', 'afnl', 'ahsc', 'ahbs', 'anw', 'aRout', 'axn', 'anpre', 'anpost', 'ancomb', 'tp2out', 'cyc', 'y1', 'y2', 'shplane', 'kmeta', 'knsel', 'ksel', 'kvw', 'rpos', 'iq', 'iwp', 'iq_out', 'iqs_out', 'iw_out', 'xh', 'cpart', 'ilvl', 'aux8', 'auxs', 'tpep', 'kmeta_d', 'kvw_d', 'rpos_d', 'dR', 'dy', 'dpre', 'dpost', 'dcomb', 'emb', 'noise', 'x8', 'xs', 'ctxtab', 'dgeom', 'tids', 'eng8', 'engs', 'engkv', 'engpart']   # the step's pointers, in a device table (kernel arg `fp`):
# as ~30 more kernel args they pushed SGPRs into spills and a 64-bit VGPR spill whose ~45 scratch reloads each waited,
# by the in-order loadcnt, for every outstanding load of the stage they sat in
ATAB = ("x8 xs wqkv wqkv_s qnw kvnw wqb wqb_s kv sel sink rcos rsin woa woa_s wob wob_s qkv_part qkv ss qr8 qrs kvn q "
        "apart aml o8 os_ y8 ys ypart out opart tpout recv0 recv1 recv2 recv3").split()   # attention pointers (after FPTRS in fp)
LKEYS = ("wqkv wqkv_s wqb wqb_s woa woa_s wob wob_s qnw kvnw sink fnh fnl hsc hbs nw gate gbias "
         "afnh afnl ahsc ahbs anw mw13 ms13 mw2 ms2 rcos rsin shplane ksel knsel kring lflags iqw iqws iww cwkv cnw cape cwk cknw cplan wplan ccap wcap cmain crpp cprow cplane ckv cscore cstate btab bstride slots ibat ibound iplane irpp ilog ilw isel irow imain ibmax icand engw engws enggw engemb").split()   # ksel/knsel: the layer's topk owner's selection; kring: ring start row   # rcos/rsin: this layer's RoPE row table (window or global, by the layer's ratio)   # a*: the attention front at the END of layer Ly = layer Ly+1's weights (host fills)
RECV_HALF = 16 * 40960 * 16 + (2 << 20)   # bytes a parity half of a receive buffer (TP regions + tail)
RKOFF = LY.RKEEP        # LDS: the residual streams kept between the fronts
LCB = LY.LAYER_BYTES             # (+ 128..255: the TP1 / TP2 column-group counters, gpub / gwait)  # bytes of handoff counters a layer (16: the indexer layers' extra phases use 4, 44.., 60)
IDX_LEVELS = 2        # selection-tree levels (1: bound <= 8192 index rows; 2: <= 8192 x 16 -- the 64K-token ILW)
IPF_KB = 32           # instruction prefetch (KB) before the fronts / the MoE
FPF_SPLIT = True      # (front prefetch layout; the step build does not prefetch the front weights)


def fptr(fp, name):
    """Pointer `name` from the table (wave-uniform: a scalar load, scalar cache)."""
    return hx.sload64(fp + i64(FPTRS.index(name) * 8))


def tp_epoch_begin(recv, fp, rank):
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
    for k in range_constexpr((n16 + CTAS * THREADS - 1) // (CTAS * THREADS)):   # (the last pass clamped: repeated zeros)
        hx.gstore(other + i64(fx.min(b * THREADS + t + k * (CTAS * THREADS), fx.Int32(n16 - 1))) * 16, zv, nt=True)
    hx.wait_storecnt0()
    return [r + half for r in recv], ep


@ASTRewriter.transform
def tp_epoch_end(fp, ep):
    """The next step's epoch (CTA 0 thread 0, after the last handoff: every CTA read this step's at its start)."""
    t = hx.opaque_v(fx.thread_idx.x)
    b = hx.opaque_s(fx.block_idx.x)
    if (b == 0) & (t == 0):
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


def entry_front(fp, sm, b, row, ne, topk):
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
              rk=sm + RKOFF, first=fx.Int32(1) != fx.Int32(0), last=nof, hdev=True, ne=ne, topk=topk)
    FS.front_any(cu, lambda k: None)


@ASTRewriter.transform
def gpub(base, b, t):
    """A TP CTA's y (device-scope stores) is out: count it in its column group b // 16 (16 CTAs a group)."""
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(base + i64(b // 16) * 4, 1)


@ASTRewriter.transform
def gwait(base, b, t):
    """A front CTA (g, p = b % 16): wait for the 16 TP CTAs of its column group."""
    if t == 0:
        hx.wait_count(lambda: base + i64(b % 16) * 4, 16)
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


def build(TR, rank, NL=D.LAYERS, ndr=D.DRAFT_LAYERS, trd=None, part="all", sources=""):
    """The step kernel for TR target rows (TR // 6 requests) and trd draft rows (5 a request); NL target layers (NL < 40
    and ndr 0: a bring-up / shadow build of the first layers only). part: "all" (one launch), or the step as two launches
    on one stream: "target" (entry, target layers, head, sampling, verification) then "draft" (draft context, metadata,
    draft layers, draft head, Markov) -- the host reads the verdict after the first while the second runs.
    sources: the kernel sources' digest (the runner's atom.mono.plan.build_key.source_digest of this directory). Every
    parameter is in `keyed`, which the kernel and the launcher reference: all of them reach FlyDSL's JIT cache key (an edit to a helper
    the cache walk does not reach still rebuilds)."""
    assert part in ("all", "target", "draft")
    keyed = (("TR", TR), ("rank", rank), ("NL", NL), ("ndr", ndr), ("trd", trd), ("part", part), ("sources", sources))
    L = smem_layout(TR)
    TPG = TR * 10 // 3
    stx = hx.stx_dev
    SEGS = ([(0, NL, 1)] if part != "draft" else []) + ([(NL, NL + ndr, 4)] if ndr and part != "target" else [])
    NDR = ndr
    TRD = TR if trd is None else trd
    LD = smem_layout(TRD)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def body(recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64, cnt: Int64, st: Int64, scale: fx.Float32, fp: Int64,
             tp: Int64):
        recv = [recv0, recv1, recv2, recv3]
        recv, tep = tp_epoch_begin(recv, fp, rank)
        for si in range_constexpr(len(SEGS)):   # layer segments, each its own loop and body (kind 1: target, 4: draft)
            seg_lo, seg_hi, seg_kind = SEGS[si]
            DRS = (seg_kind & 4) != 0
            FIRST_LY = -1
            if fx.const_expr(si == 0 and part != "draft"):
                # the target entry: SinglePassHCState.from_embeddings (R = 4 copies of the token's embedding, pre = (1, 0, 0, 0),
                # y = 0, post = 0, comb = I: stage_block_state with W = 1), then layer 0's attention front (its a*: layer-table
                # row TENTRY_ROW): x8 / xs / the attention gates, R kept in LDS (layer 0's FFN front: first = 0)
                t = hx.opaque_v(fx.thread_idx.x)
                b = hx.opaque_s(fx.block_idx.x)
                sm = hx.lds_base()
                TCNT = cnt + i64((NL + NDR) * LCB)
                CX.stage_block_state(dict(t=t, b=b, aid=fptr(fp, "tids"), emb=fptr(fp, "emb"), noise=fptr(fp, "noise"),
                                          R=fptr(fp, "dR"), y=fptr(fp, "dy"), pre=fptr(fp, "dpre"), post=fptr(fp, "dpost"),
                                          comb=fptr(fp, "dcomb")), TR, 1)
                handoff(TCNT + i64(LY.TARGET_STATE), t, False)
                if b < FS.FRONT_CTAS:
                    entry_front(fp, sm, b, TENTRY_ROW, FS.NE, FS.TOPK)
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
                if fx.const_expr(part == "all"):
                    if t == 0:
                        hx.gstore(st + i64(b) * (32 * 8) + i64(25 * 8), hx.clock())
                    ST.target(tp, t, b, sm, st, TCNT, recv, rank, head_d=4, ctx=fptr(fp, "ctxtab"), T=TR)
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
                    entry_front(fp, sm, b, NL + NDR, DRAFT_NE, DRAFT_TOPK)
                handoff(TCNT + i64(LY.DRAFT_FRONT), t, False)
            for Ly in range(seg_lo, seg_hi):
                kind = seg_kind
                t = hx.opaque_v(fx.thread_idx.x)
                b = hx.opaque_s(fx.block_idx.x)
                w = t // 32
                l = t % 32
                sm = hx.lds_base()
                lc = cnt + i64(Ly) * LCB
                # the attention pointers: re-read from the table at each use (scalar loads; as kernel args they pushed SGPRs
                # into spills), the per-layer ones from the layer row
                lz = {k: atab(fp, k) for k in ("tpout x8 xs kv sel qkv_part qkv qr8 qrs kvn q apart aml o8 y8 ys ypart out "
                                               "opart").split()}
                lz.update(os=atab(fp, "os_"), ss=atab(fp, "ss", i64(Ly) * (AS.SS_LAYER_F32 * 4)),
                          recv=(lambda: [atab(fp, f"recv{o}")() + i64(tep & 1) * RECV_HALF for o in range(4)]))
                lz.update({k: ltab(fp, Ly, k) for k in ("wqkv wqkv_s wqb wqb_s woa woa_s wob wob_s qnw kvnw sink rcos "
                                                        "rsin").split()})
                c = LazyC(lz, **dict(Lx, t=t, b=b, w=w, l=l, sm=sm, scale=scale, stx=stx, rank=rank, xcpol=0))
                c["rpos"] = fptr(fp, "rpos" + DSUF)   # each row's RoPE position (int64 [T]), the product's cache.rope_positions
                # the key geometry: step metadata, the layer's selection owner tables and ring start (a WINDOW layer: n_sel 0 table)
                c._th.update(kmeta=lambda: fptr(fp, "kmeta" + DSUF), knsel=lambda: fptr(fp, "knsel"), ksel=lambda: fptr(fp, "ksel"),
                             kvw=lambda: fptr(fp, "kvw" + DSUF))
                for k_ in ("kmeta", "knsel", "ksel", "kvw"):
                    dict.__setitem__(c, k_, None)
                c._th.update(ksel=ltab(fp, Ly, "ksel"), knsel=ltab(fp, Ly, "knsel"))
                c["kring_off"] = fx.Int32(ltab(fp, Ly, "kring")())
                stp = st + i64(b) * (32 * 8)

                def stamp(k):   # the layer-time stamps only: 0 (layer 0 start), 12 (layer 1 start), 18 (a layer's end)
                    if fx.const_expr(k not in (0, 12, 18)):
                        return
                    if t == 0:
                        hx.gstore(stp + i64(k * 8), hx.clock())

                if Ly == 0:
                    stamp(0)
                if Ly == 1:
                    stamp(12)
                if fx.const_expr((kind & 1) != 0):   # the row before an Engram layer (lflags bit 7): its embeddings -> MXFP8
                    if (ltab(fp, Ly, "lflags")() & fx.Int64(128)) != fx.Int64(0):
                        ES.stage_engq(cwith(c, engemb=ltab(fp, Ly, "engemb")(), eng8=fptr(fp, "eng8"), engs=fptr(fp, "engs")))
                stage_A(c)
                pf_rows(c["wqb"] + i64(b) * (16 * QL), QL, 16, QL, t)
                pf_rows(c["wqb"] + i64(b + CTAS) * (16 * QL), QL, 16, QL, t)
                handoff(lc, t, True)   # LY.ATTN_A (offset 0)
                if fx.const_expr((kind & 1) != 0):   # the next layer's Engram kv = emb . wkv^T (read by this layer's attention front)
                    if (ltab(fp, Ly, "lflags")() & fx.Int64(128)) != fx.Int64(0):
                        ES.stage_engkv(cwith(c, eng8=fptr(fp, "eng8"), engs=fptr(fp, "engs"), engw=ltab(fp, Ly, "engw")(),
                                             engws=ltab(fp, Ly, "engws")(), engkv=fptr(fp, "engkv")))
                stage_kv(c)
                stage_C(c)
                if fx.const_expr((kind & 1) != 0):   # an indexer layer: iq / iw beside kv / C (their inputs: A's q latent and ss, the front's axn)
                    isidx = (ltab(fp, Ly, "lflags")() & fx.Int64(8)) != fx.Int64(0)   # lflags: bit0 special body, 1 compressor, 2 ratio 1, 3 indexer, 4 ctx tap, 5 candidate producer, 6 REINDEX, 7 next layer has Engram
                    if isidx:
                        ci = cwith(c, iqw=ltab(fp, Ly, "iqw")(), iqws=ltab(fp, Ly, "iqws")(), iww=ltab(fp, Ly, "iww")(),
                                   iq=fptr(fp, "iq"), iwp=fptr(fp, "iwp"), axn=fptr(fp, "axn"))
                        IS.stage_iq(ci)
                        IS.stage_iw(ci)
                        lfl = ltab(fp, Ly, "lflags")()
                        hascomp = (lfl & fx.Int64(2)) != fx.Int64(0)
                        r1 = (lfl & fx.Int64(4)) != fx.Int64(0)
                        if hascomp:   # the compressor's wkv_gate GEMM (compressor)
                            CS.stage_cgemm(cwith(c, cwkv=ltab(fp, Ly, "cwkv")(), axn=fptr(fp, "axn"), cpart=fptr(fp, "cpart")), r1)
                handoff(lc + i64(LY.KV_C), t, True)
                if fx.const_expr((kind & 1) != 0):   # iquant, then its own handoff (the scorer reads iq_out / iw_out; lc + 4 is free)
                    if isidx:
                        IS.stage_iquant(cwith(c, iq=fptr(fp, "iq"), iwp=fptr(fp, "iwp"), iq_out=fptr(fp, "iq_out"),
                                              iqs_out=fptr(fp, "iqs_out"), iw_out=fptr(fp, "iw_out")))
                        lfl2 = ltab(fp, Ly, "lflags")()
                        if (lfl2 & fx.Int64(2)) != fx.Int64(0):   # the compressor's rows: main KV row + index row, and the ring
                            r1b = (lfl2 & fx.Int64(4)) != fx.Int64(0)
                            cc = cwith(c, cpart=fptr(fp, "cpart"), cnw=ltab(fp, Ly, "cnw")(), cape=ltab(fp, Ly, "cape")(),
                                       cwk=ltab(fp, Ly, "cwk")(), cknw=ltab(fp, Ly, "cknw")(), cplan=ltab(fp, Ly, "cplan")(),
                                       wplan=ltab(fp, Ly, "wplan")(), ccap=fx.Int32(ltab(fp, Ly, "ccap")()), wcap=fx.Int32(ltab(fp, Ly, "wcap")()), cmain=fx.Int32(ltab(fp, Ly, "cmain")()),
                                       crpp=fx.Int32(ltab(fp, Ly, "crpp")()), cprow=fx.Int32(ltab(fp, Ly, "cprow")()), cplane=ltab(fp, Ly, "cplane")(), ckv=ltab(fp, Ly, "ckv")(),
                                       cscore=ltab(fp, Ly, "cscore")(), cstate=ltab(fp, Ly, "cstate")(), btab=ltab(fp, Ly, "btab")(),
                                       bstride=fx.Int32(ltab(fp, Ly, "bstride")()), slots=ltab(fp, Ly, "slots")())
                            CS.stage_crow(cc, r1b)
                            CS.stage_cring(cc, r1b)
                        handoff(lc + i64(LY.INDEX_QUANT), t, True)
                        if ltab(fp, Ly, "ibat")() != fx.Int64(0):   # a layer whose table has no scorer inputs skips it
                            # FULL: the scores of every visible index row, then the top-512 -> isel / irow (the attention's ksel)
                            cs_ = cwith(c, ibat=ltab(fp, Ly, "ibat")(), ibound=ltab(fp, Ly, "ibound")(), iplane=ltab(fp, Ly, "iplane")(),
                                        irpp=fx.Int32(ltab(fp, Ly, "irpp")()), btab=ltab(fp, Ly, "btab")(), bstride=fx.Int32(ltab(fp, Ly, "bstride")()),
                                        ilog=ltab(fp, Ly, "ilog")(), ilw=fx.Int32(ltab(fp, Ly, "ilw")()), isel=ltab(fp, Ly, "isel")(),
                                        irow=ltab(fp, Ly, "irow")(), cprow=fx.Int32(ltab(fp, Ly, "cprow")()), imain=fx.Int32(ltab(fp, Ly, "imain")()),
                                        iq_out=fptr(fp, "iq_out"), iw_out=fptr(fp, "iw_out"))
                            lfl3 = ltab(fp, Ly, "lflags")()
                            isrx = (lfl3 & fx.Int64(64)) != fx.Int64(0)      # bit 6: REINDEX (layer 20's candidates)
                            cs_ = cwith(cs_, ibmax=ltab(fp, Ly, "ibmax")(), icand=ltab(fp, Ly, "icand")(), lvl=fptr(fp, "ilvl"),
                                        iprod=(lfl3 & fx.Int64(32)) != fx.Int64(0),   # bit 5: produces the candidate blocks (layer 20)
                                        ts=None)
                            # one instance of the scorer (FULL and REINDEX, isrx traced) and of the selection code for every pass (I-cache)
                            SC.stage_score_any(cs_, isrx)
                            handoff(lc + i64(LY.SCORE), t, True)
                            npass = isrx.select(fx.Int32(IDX_LEVELS), cs_["iprod"].select(fx.Int32(2 * IDX_LEVELS), fx.Int32(IDX_LEVELS)))
                            for pi in range(npass):   # sel or rx levels, then (layer 20) the blk levels
                                pmode = isrx.select(fx.Int32(TK.MODE_RX), (pi < IDX_LEVELS).select(fx.Int32(TK.MODE_SEL), fx.Int32(TK.MODE_BLK)))
                                plev = pi % IDX_LEVELS
                                TK.stage_select_any(cwith(cs_, ts=None), plev, IDX_LEVELS, pmode)
                                handoff(lc + i64(LY.SELECT0) + i64(pi) * 4, t, True)
                stage_D(c)
                pf_rows(c["woa"] + i64(b // 4) * (32 * KWOA) + i64((b % 4) * 1024), KWOA, 32, 1024, t)
                handoff(lc + i64(LY.ATTN_D), t, True)
                stage_E(c)
                pf_rows(c["wob"] + i64(b // 4) * (80 * NWOA) + i64((b % 4) * 512), NWOA, 80, 512, t)
                handoff(lc + i64(LY.ATTN_E), t, True)
                stage_F_cluster(c)
                gpu.barrier()
                par = Ly % 2
                tp_allreduce(cwith(c, ytgt=fptr(fp, 'y1')), par * 2, 2 * Ly + 1)
                gpub(lc + i64(LY.TP1_GROUPS), b, t)
                # the two fronts with one copy of the code: pass 0 = FFN front, MoE, TP2; pass 1 = attention front
                for hf in range(hx.opaque_s(fx.Int32(2))):
                    if hf == 0:
                        iprefetch(28)   # the front (the attention-front pass reuses it warm)
                    # a front CTA (g, p) reads y (TP1 / TP2's output) only in its cols 320p.. -- the 16 TP CTAs 16p..16p+15:
                    # it waits for their column-group count, not a device-wide handoff (the idle CTAs do not wait at all)
                    if b < FS.FRONT_CTAS:
                        gwait(lc + (hf == 0).select(i64(LY.TP1_GROUPS), i64(LY.TP2_GROUPS)), b, t)
                    if b < FS.FRONT_CTAS:
                        isf = hf == 0

                        def fsel(n0, n1):
                            ix = isf.select(fx.Int32(FPTRS.index(n0) * 8), fx.Int32(FPTRS.index(n1) * 8))
                            return hx.sload64(fp + i64(ix))
                        tq = hx.opaque_v(fx.thread_idx.x)
                        cu = dict(t=tq, b=b, w=tq // 32, l=tq % 32, sm=sm + FRONT_LDS_OFF, rms_eps=D.RMS_EPS, hc_eps=1e-6, norm_eps=D.RMS_EPS,
                                  R=fsel("R", "Rout"), y=fsel("y1", "y2"), pre=fsel("pre", "npre"), post=fsel("post", "npost"),
                                  comb=fsel("comb", "ncomb"), fnh=fsel("fnh", "afnh"), fnl=fsel("fnl", "afnl"),
                                  hsc=fsel("hsc", "ahsc"), hbs=fsel("hbs", "ahbs"), nw=fsel("nw", "anw"),
                                  gate=fptr(fp, "gate"), gbias=fptr(fp, "gbias"),
                                  # the last layer of each segment writes the head input: target -> xh, draft -> dh
                                  Rout=fsel("Rout", "aRout"), xn=((hf != 0) & (Ly == (NL + NDR - 1 if DRS else NL - 1))).select(ST.tptr(tp, "dh") if DRS else fptr(fp, "xh"), fsel("xn", "axn")), x8=isf.select(fptr(fp, "fx8"), c["x8"]),
                                  xs=isf.select(fptr(fp, "fxs"), c["xs"]), part=fptr(fp, "part"), plog=fptr(fp, "plog"),
                                  cand=fptr(fp, "cand"), cands=fptr(fp, "cands"), npre=fsel("npre", "anpre"),
                                  npost=fsel("npost", "anpost"), ncomb=fsel("ncomb", "ancomb"), ids=fptr(fp, "ids"),
                                  wts=fptr(fp, "wts"), isffn=isf, xs32=hf != 0, merge=isf, hdev=True, ne=RNE, topk=RTOPK)
                        cu.update(fnh=lsel(fp, Ly, isf, "fnh", "afnh"), fnl=lsel(fp, Ly, isf, "fnl", "afnl"),
                                  hsc=lsel(fp, Ly, isf, "hsc", "ahsc"), hbs=lsel(fp, Ly, isf, "hbs", "ahbs"),
                                  nw=lsel(fp, Ly, isf, "nw", "anw"), gate=ltab(fp, Ly, "gate")(), gbias=ltab(fp, Ly, "gbias")())
                        # the residual streams kept in LDS between the fronts; also to global (last) after the last layer and
                        # for the tap layers' inputs (the attention fronts of layers TAP0..TAP0+2)
                        tap0 = NL - 4
                        cu.update(rk=sm + RKOFF, first=isf & (Ly == FIRST_LY), last=(hf != 0) & ((Ly == NL - 1) | ((Ly >= tap0) & (Ly <= tap0 + 2))))
                        # the attention front before an Engram layer gates its value in (lflags bit 7; target body only)
                        if fx.const_expr((kind & 1) != 0):
                            cu.update(eng=(hf != 0) & ((ltab(fp, Ly, "lflags")() & fx.Int64(128)) != fx.Int64(0)),
                                      engkv=fptr(fp, "engkv"), enggw=ltab(fp, Ly, "enggw")(), engpart=fptr(fp, "engpart"), TR=TRx)
                        def x8pub(k):   # the FFN front's x8 / xs are out (front_C): count it (the early shared UG waits on it)
                            if fx.const_expr(k == 5 and MS.sh_early((TRx + MS.MAXTOK - 1) // MS.MAXTOK)):
                                if isf:
                                    hx.wait_storecnt0()
                                    gpu.barrier()
                                    if tq == 0:
                                        hx.atomic_add(lc + i64(LY.X8_PUB), 1)
                        FS.front_any(cu, x8pub)
                    if fx.const_expr(MS.sh_early((TRx + MS.MAXTOK - 1) // MS.MAXTOK)):
                        # the FFN front's idle CTAs: the shared expert's UG, K split over SHE_CTAS CTAs a unit, once every front
                        # CTA's x8 is out (the front leaves most CTAs idle)
                        if (hf == 0) & (b >= FS.FRONT_CTAS) & (b < FS.FRONT_CTAS + ((TRx + MS.MAXTOK - 1) // MS.MAXTOK) * MS.NUG * MS.SHE_CTAS):
                            tq3 = hx.opaque_v(fx.thread_idx.x)
                            if tq3 == 0:
                                hx.wait_count(lambda: lc + i64(LY.X8_PUB), FS.FRONT_CTAS)
                            gpu.barrier()
                            ce = dict(t=tq3, b=b, w=tq3 // 32, l=tq3 % 32, sm=sm + FRONT_LDS_OFF, NP=TRx * RTOPK, topk=RTOPK,
                                      x8=fptr(fp, "fx8"), xs=fptr(fp, "fxs"), shplane=ltab(fp, Ly, "shplane")(),
                                      mid8=fptr(fp, "mid8"), midc=fptr(fp, "midc"), out=fptr(fp, "mout"))
                            MS.ug_shared_early(ce, b - FS.FRONT_CTAS)
                    if hf == 0:
                        iprefetch(16)   # the routed MoE
                        handoff(lc + i64(LY.FRONT), t, True)
                        mfx8, mfxs, mwts, mout, mid8, midc = [fptr(fp, k) for k in ("fx8", "fxs", "wts", "mout", "mid8", "midc")]
                        tq2 = hx.opaque_v(fx.thread_idx.x)
                        cm = LazyC({'shplane': (lambda Ly=Ly: ltab(fp, Ly, 'shplane')())},   # (re-read at its uses: not live across the layer)
                                   t=tq2, b=b, w=tq2 // 32, l=tq2 % 32, sm=sm + FRONT_LDS_OFF, NP=TRx * RTOPK, topk=RTOPK, nexp_max=96,
                                   w13=ltab(fp, Ly, "mw13")(), s13=ltab(fp, Ly, "ms13")(), w2=ltab(fp, Ly, "mw2")(), s2=ltab(fp, Ly, "ms2")(), x8=mfx8, xs=mfxs, ids=fptr(fp, 'ids'), wts=mwts, out=mout,
                                   mid8=mid8, midc=midc, shcnt=lc + i64(LY.SHARED_DOWN))
                        MS.moe(cm, lc + i64(LY.MOE_UG))
                        gpu.barrier()
                        handoff(lc + i64(LY.MOE_DONE), t, True)
                        # TP2 payload = the MoE output rows, result -> y2 (the attention front's y)
                        MS.moe_combine(dict(t=hx.opaque_v(fx.thread_idx.x), b=b, out=fptr(fp, "mout"), topk=RTOPK), TRx, sm)
                        tp_allreduce(cwith(c, tpout=fptr(fp, "tp2out"), ytgt=fptr(fp, "y2")), par * 2 + 1, 2 * Ly + 2)
                        gpub(lc + i64(LY.TP2_GROUPS), b, t)
                handoff(lc + i64(LY.LAYER_END), t, True)
                if fx.const_expr((kind & 1) != 0):   # the target body: the draft context taps (read before the next front rewrites aRout)
                    if (ltab(fp, Ly, "lflags")() & fx.Int64(16)) != fx.Int64(0):
                        tap0 = NL - 4
                        CX.stage_tap1(dict(t=hx.opaque_v(fx.thread_idx.x), b=b, aux8=fptr(fp, "aux8"), auxs=fptr(fp, "auxs")),
                                      TRx, Ly - tap0, fptr(fp, "aRout"))
                stamp(18)
        if fx.const_expr(part == "target"):   # the step's first launch ends with the head, sampling and verification
            t = hx.opaque_v(fx.thread_idx.x)
            b = hx.opaque_s(fx.block_idx.x)
            sm = hx.lds_base()
            if t == 0:
                hx.gstore(st + i64(b) * (32 * 8) + i64(25 * 8), hx.clock())
            ST.target(tp, t, b, sm, st, cnt + i64((NL + NDR) * LCB), recv, rank, head_d=4,
                     ctx=_ctxtab(fp, NDR), T=TR)
        if fx.const_expr(NDR > 0 and part != "target"):
            t = hx.opaque_v(fx.thread_idx.x)
            b = hx.opaque_s(fx.block_idx.x)
            sm = hx.lds_base()
            if t == 0:
                hx.gstore(st + i64(b) * (32 * 8) + i64(26 * 8), hx.clock())
            ST.draft(tp, t, b, sm, st, cnt + i64((NL + NDR) * LCB), recv, rank, head_d=4)
        tp_epoch_end(fp, tep)
        _ = keyed

    @flyc.jit
    def launch(recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64, cnt: Int64, st: Int64, scale: fx.Float32, fp: Int64,
               tp: Int64, stream: fx.Stream = fx.Stream(None)):
        """recv0..3: the 4 ranks' receive buffers; cnt: the step's zeroed handoff counters; st: stamps; scale: the
        attention softmax scale; fp: the pointer table (FPTRS, ATAB, layer rows); tp: the step tail's table."""
        _ = keyed
        body(recv0, recv1, recv2, recv3, cnt, st, scale, fp, tp).launch(
            grid=(CTAS,), block=(THREADS,), smem=max(L["LDS"], ST.HEAD_LDS if ndr else 0, FRONT_LDS_OFF + max(FS.FRONT_LDS, MS.MOE_LDS), RKOFF + FS.RK_BYTES), cluster=(CL, 1, 1), stream=stream)

    return launch, dict(L, TPG=TPG)
