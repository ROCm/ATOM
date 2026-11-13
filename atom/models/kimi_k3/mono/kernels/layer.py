# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/layer.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""K2 ``layer_post``: one Kimi-K3 layer from the attention's core output (the
o_proj input) to the layer's MoE output, in one launch -- K2a's o_proj, its
all-reduce and the MLP AttnRes seam, then K2b's latent MoE (``moe``).

As ATOM's ``layer_post``, the two halves share the launch so that the MoE's
static weights stream while the AttnRes chain (partials -> reduce -> slice ->
gate -> mix) is still settling: the CTAs that will run the router, latent and
shared gate/up GEMVs issue their weights before they first wait.

    every CTA        o_proj (448 x 16 rows) -> partials to their columns'
                     owner rank (``OWN_COLS``): two tasks on CTAs 56 .. 247,
                     one on the rest (``o_slot``)
    CTAs 0 .. 55     K2a's reduce + AttnRes (slice, gate, mix -> x rows + XRDY)
                     of the first token group; then route (tokens bid, + 56 ..)
    CTAs 56 g ..     past 16 rows, token group g's AttnRes (g >= 1), after
                     their o_proj
    CTAs 56 .. 151   the shared gate/up (96 tasks, weights right after o_proj)
    CTAs 160 .. 243  router parts (28) and latent parts (56): weights first
    every CTA        ug (the first round placed, the rest from a queue); then
                     down (0 .. 223), or the shared SiTU and down (224 .. 255);
                     lnorm (56 .. 83), up (0 .. 55), reduce (200 .. 255)
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from atom.models.kimi_k3.mono.kernels.attention import back as k2a
from atom.models.kimi_k3.mono.kernels.attention.kda import (
    _SOURCES,
    COLS,
    HIDDEN,
    NA,
    TG,
    groups,
    slice_compute,
    slice_loads,
    stage_gate,
    stage_mix,
)
from atom.models.kimi_k3.mono.kernels.stages import moe as k2b
from atom.mono.device import gemv
from atom.mono.device.ops import kernel_symbol, load_ptr64, traced
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, preg, sreg
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import byte_layout, pair_layout
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.debug import region_ids

O0 = NA  # the CTAs past AttnRes's: o_proj's first round, then the early GEMVs
OTASKS = k2a.OTASKS  # 448
O_ROUNDS = (OTASKS + BLOCKS - 1) // BLOCKS  # 2: every CTA takes o_proj tasks


def o_slot(bid):
    """CTA bid's first o_proj task: CTAs O0 .. take 0 .., the AttnRes CTAs the
    last 56 slots, whose second task is past OTASKS (one task each: their
    reduce waits on the slowest partial)."""
    return (bid + BLOCKS - O0) % BLOCKS


E0 = 160  # router / latent CTAs (their weights in flight under their o_proj)
# timeline points: 0 start, 1 o_proj done / AR reduced, 2 early done / slice,
# 3 ug, 4 down/sdown, 5 lnorm, 6 up, 7 reduce, 8 gate, 9 mix (x published),
# 10 route, 11 route table, 12 latent quant, 13 route starts
TL_POINTS = 18  # + 14 route's logits polled, 15 lnorm's latent sums in, 16 down done,
# 17 the latent's first block polled

assert E0 + k2b.R_TASKS + k2b.L_TASKS <= BLOCKS and k2b.G_TASKS <= E0 - O0

REGIONS = ("part", "wgt", "msq", "xrdy", "ar", "dg") + k2b.REGIONS

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    # ``sym``: the field's short name in the kernel symbol (``symbol_params``)
    tokens: int = field(metadata={"sym": "s"})
    # the MLP AttnRes block sources
    nblocks: int = field(metadata={"sym": "nb"})
    eps: float = 1e-5  # mlp_res_norm
    out_eps: float = 1e-5  # post_attention_layernorm
    ln_eps: float = 1e-5  # the latent RMSNorm
    # the final reduce here (else the next layer's K1)
    reduce: bool = field(default=True, metadata={"sym": "r"})
    diag: bool = field(default=False, metadata={"sym": "g"})
    timeline: bool = field(default=False, metadata={"sym": "t"})


ABI = KernelAbi(
    (
        "core",
        "w_o",
        "prefix",
        "blocks",
        "blk_sm",
        "blk_sr",
        "ares_nw",
        "ares_qk",
        "in_nw",
        "w_gate",
        "bias",
        "w_ld",
        "w_sgu",
        "w_sd",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "ln_w",
        "w_up",
        "out",
        "scratch",
        "plain",
        "queue",
        "peers",
        "rank",
        "layer",
        # 1 on a block-write layer: the prefix restarts at the o_proj output (a
        # runtime argument, not a build: one build serves both kinds of layer)
        "reset",
        "diag",
        "tl",
    )
)


def peer_layout(s):
    """K2a's AR and DG regions, then K2b's: name -> (byte offset, bytes)."""
    pairs = {"ar": k2a.ar_pairs(s), "dg": k2a.dg_pairs(s), **k2b.peer_pairs(s)}
    return pair_layout(pairs.items())


def peer_bytes(s):
    lay = peer_layout(s)
    return max(o + n for o, n in lay.values())


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    a = k2a.scratch_layout(k2a.AttnPostBuild(tokens=s, nblocks=key.nblocks))
    end = max(o + n for o, n in a.values())
    b = k2b.scratch_layout(k2b.MoeBuild(tokens=s))
    lay = dict(a)
    for name, (o, n) in b.items():
        lay[name] = ((end + 15) // 16 * 16 + o, n)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def plain_layout(key: K2Build) -> dict:
    """K2b's plain words, then the bf16 x rows K2a's AttnRes writes for K2b
    (XROW): in the step's ``plain`` buffer (``byte_layout``)."""
    s = key.tokens
    lay = k2b.plain_layout(k2b.MoeBuild(tokens=s))
    end = max(o + n for o, n in lay.values())
    return {**lay, **byte_layout((("xrow", s * HIDDEN * 2),), start=end)}


def plain_bytes(key: K2Build) -> int:
    return max(o + n for o, n in plain_layout(key).values())


@traced
def stage_oproj(c, gi):
    """o_proj tasks gi, gi + 256 (< 448): every task's weights in flight at
    once (one memory latency, not two), then the MFMAs, a token group's core
    rows at a time. A CTA with one task loads its own slab again for the
    round it skips (cache hits), not task 447's on 64 CTAs at once."""
    # K2's bf16 GEMV weights (here, the MoE's early / shared down / up) a load
    # two slabs of 8 rows (``gemv.chunk_ops`` paired)
    ops = [
        k2a.o_loads(c, k2a.o_task_or(gi, gi + BLOCKS * j), paired=True)
        for j in range(O_ROUNDS)
    ]
    for g in range_constexpr(c["G"]):
        # ``stage_o`` ends at a barrier: the previous group's core rows are read
        cg = k2a.group_view(c, g)
        k2a.load_core(cg)
        for j in range_constexpr(O_ROUNDS):
            t = gi + BLOCKS * j
            if t < OTASKS:
                k2a.stage_o(cg, t, ops[j])


@traced
def stage_attnres(c, task, g, on, tls):
    """Task ``task`` (< NA) of token group g's AttnRes seam: the reduced o_proj
    -> slice -> gate (tasks 0 .. 15) -> mix -> x rows, XRDY. Group g runs on
    CTAs 56 g ..: the groups' hand-off chains side by side."""
    cg = k2a.group_view(c, g)
    enter_stage("k2.ar")
    # the slice's sources and weights in flight across the reduce's wait
    sld = slice_loads(cg, task)
    k2a.stage_reduce(cg, task)
    stamp(on, tls, c["tid"], 1)
    enter_stage("k2.slice")
    slice_compute(cg, task, sld)
    stamp(on, tls, c["tid"], 2)
    enter_stage("k2.gate")
    if task < cg["S"]:
        stage_gate(cg, task)
    stamp(on, tls, c["tid"], 8)
    enter_stage("k2.mix")
    stage_mix(cg, task)
    stamp(on, tls, c["tid"], 9)


@traced
def stage_attnres_far(c, bid, on, tls):
    """CTA bid's AttnRes task of the groups past the first (CTAs 56 .., after
    their o_proj, before their early GEMV: that waits for every group's x)."""
    for g in range_constexpr(1, c["G"]):
        if (bid >= NA * g) & (bid < NA * (g + 1)):
            stage_attnres(c, bid - NA * g, g, on, tls)


@traced
def stage_routes(c, bid):
    """CTA bid (< NA) after the AttnRes seam: route tokens bid, bid + 56, .."""
    s = c["S"]
    enter_stage("k2.route")
    for i in range_constexpr(-(-s // NA)):
        if bid + NA * i < s:
            k2b.stage_route(c, bid + NA * i)
    stamp(c["tl_on"], c["tls"], c["tid"], 10)


_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    # past one token group (``TG``) the per-token stages run a group at a
    # time, so their LDS holds one group; the latent ug multiplies is the
    # step's (``UgLds``). The width's real bound is LDS (64 rows fit).
    G, sg = groups(s), min(s, TG)
    # every group's XRDY flags polled at once (``early_compute``)
    assert s >= 1 and NA * G <= THREADS
    # the early GEMVs stage their K slice a slab at a time; ``xl`` holds the
    # widest slab any of the three kinds asks for (one slab, i.e. the whole
    # slice, at every row count up to 8)
    xcols = max(gemv.slab_width(k2b.early_shape(kind)[0], sg) for kind in (0, 1, 2))
    # the shared down's h rows of every group are in ``xl`` at once
    assert s * (k2b.SI + gemv.LDS_PAD) <= sg * (xcols + gemv.LDS_PAD)
    lay = scratch_layout(key)
    plain_lay = plain_layout(key)
    play = peer_layout(s)
    RouteLds = k2b.route_lds(s)

    @fx.struct
    class EarlyLds:
        xl: fx.Array[fx.Int32, sg * (xcols + gemv.LDS_PAD) // 2, 16]

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, sg * ns * COLS, 16]
        dl: fx.Array[fx.Float32, sg * COLS, 16]

    @fx.union
    class StageLds:
        early: EarlyLds
        attn: AttnLds
        ug: k2b.ug_lds(s)
        down: k2b.down_lds(s)
        rmask: k2b.mask_lds(s)
        rsel: k2b.RouteSelLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = kernel_symbol("k3_mono_k2", **symbol_params(key))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        plain: Int64,
        queue: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        reset: Int32,
        diag: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st = alloc.allocate(StageLds)
        el, al, ul, dl = st.early.peek(), st.attn.peek(), st.ug.peek(), st.down.peek()
        rsel = st.rsel.peek()
        # ``layer`` is this launch's slot of the step; the step's first kernel
        # finds every pair zeroed and peer-fenced
        if const_expr(key.diag):
            mb = Mailbox(layer, diag, region_ids(REGIONS))
        else:
            mb = Mailbox(layer)
        bases = peer_bases(peers, TP_)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in k2b.PEER_REGIONS + ("ar", "dg"):
            off = play[region][0]
            peer[region] = [preg(b, off, region) for b in bases]
            peer[region + "_own"] = preg(own, off, region)
        c = {
            "S": s,
            "G": G,
            "NS": ns,
            "NB": key.nblocks,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "delta": True,
            "reset": reset,
            "debug": False,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "ln_eps": key.ln_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "queue": queue,
            "red": lds.red.ptr,
            "peer": peer,
            "ar_peer": peer["ar"],
            "ar_own": peer["ar_own"],
            "dg_peer": peer["dg"],
            "dg_own": peer["dg_own"],
            "xl": el.xl.ptr,
            "rcand": rsel.cand.ptr,
            "rsig": rsel.sig.ptr,
            "vals": al.vals.ptr,
            "delta_lds": al.dl.ptr,
            "latq": ul.latq.ptr,
            "latsc": ul.latsc.ptr,
            "av": ul.av.ptr,
            "contrib": dl.contrib.ptr,
            "scan": rl.scan.ptr,
            "rt": k2b.route_table(rl, st.rmask.peek()),
            "xbuf": plain + fx.Int64(plain_lay["xrow"][0]),
            "xflags": NA * G,
            "tl_on": key.timeline,
            "tls": lds.tls.ptr,
            "inter": plain + fx.Int64(plain_lay["inter"][0]),
            "isc": plain + fx.Int64(plain_lay["isc"][0]),
            "ln": plain + fx.Int64(plain_lay["ln"][0]),
            "shp": plain + fx.Int64(plain_lay["shp"][0]),
            "args": {
                "core": core,
                "w_o": w_o,
                "prefix": prefix,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "x_dbg": 0,
                "w_gate": w_gate,
                "bias": bias,
                "w_ld": w_ld,
                "w_sgu": w_sgu,
                "w_sd": w_sd,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "ln_w": ln_w,
                "w_up": w_up,
                "out": out,
            },
        }
        for region in (
            "part",
            "wgt",
            "msq",
            "xrdy",
            "sgu",
            "h",
            "rpart",
            "lpart",
            "route",
            "ugf",
            "lmsq",
            "lnrdy",
            "shrdy",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        # ---- o_proj on every CTA (``o_slot``)
        eb = bid - E0
        lb = eb - k2b.R_TASKS
        is_r = (bid >= E0) & (eb < k2b.R_TASKS)
        is_l = (bid >= E0 + k2b.R_TASKS) & (lb < k2b.L_TASKS)
        # o_proj first (every AttnRes CTA waits on the slowest partial; loads
        # retire in order), the early weights after: they land before x does
        if bid >= O0:
            enter_stage("k2.o")
            stage_oproj(c, o_slot(bid))
            stamp(on, tls, tid, 1)
            stage_attnres_far(c, bid, on, tls)
        enter_stage("k2.early")
        if is_r:
            rops = k2b.early_loads(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS)
            k2b.early_compute(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS, rops)
        if is_l:
            lops = k2b.early_loads(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS)
            k2b.early_compute(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS, lops)
        if (bid >= O0) & ~is_r & ~is_l & (bid - O0 < k2b.G_TASKS):
            gi = bid - O0
            k2b.early_compute(c, 2, gi, 0, k2b.early_loads(c, 2, gi, 0))
        # ---- AttnRes: the reduced o_proj -> x; then route
        if bid < NA:
            enter_stage("k2.o")
            stage_oproj(c, o_slot(bid))
            stage_attnres(c, bid, 0, on, tls)
            stamp(on, tls, tid, 13)
            stage_routes(c, bid)
        if bid >= NA:
            stamp(on, tls, tid, 2)
        # ---- the routed experts
        # the route table first: its bitmaps share LDS with the latent
        # (``k2b.mask_lds``)
        enter_stage("k2.ug")
        nu = k2b.load_route(c)
        stamp(on, tls, tid, 11)
        k2b.load_latq(c)
        stamp(on, tls, tid, 12)
        k2b.run_ug(c, bid, nu)
        stamp(on, tls, tid, 3)
        # the shared SiTU after ug, on the shared down's CTAs: only the shared
        # down reads it (before ug it held the AttnRes CTAs' ug start back)
        enter_stage("k2.h")
        k2b.run_h(c, bid)
        enter_stage("k2.down")
        if bid < k2b.DOWN_TASKS:
            # the first w2 batch in flight before any ug flag is waited for
            dops = k2b.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
            k2b.stage_down(c, bid, nu, dops)
            stamp(on, tls, tid, 16)
        enter_stage("k2.sdown")
        k2b.run_sdown(c, bid, w_sd)
        stamp(on, tls, tid, 4)
        enter_stage("k2.lnorm")
        if (bid >= k2b.LN0) & (bid < k2b.LN0 + k2b.LN_TASKS):
            k2b.stage_lnorm(c, bid - k2b.LN0)
        # a wide step's lnorm tasks' second halves on CTAs 0 .., which wait
        # for LN anyway (up's)
        if const_expr(k2b.ln_halves(c["S"]) > 1):  # noqa: SIM102
            if bid < k2b.LN_TASKS:
                k2b.stage_lnorm(c, bid, 1)
        stamp(on, tls, tid, 5)
        enter_stage("k2.up")
        k2b.run_up(c, bid)
        stamp(on, tls, tid, 6)
        # compile-time gate outside, traced condition inside: `and` does not
        # combine a Python bool with a traced value
        enter_stage("k2.final")
        if const_expr(key.reduce):  # noqa: SIM102
            if (bid >= k2b.RED0) & (bid < k2b.RED0 + k2b.RED_TASKS):
                k2b.stage_reduce(c, bid - k2b.RED0)
        stamp(on, tls, tid, 7)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        plain: Int64,
        queue: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        reset: Int32,
        diag: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            w_o,
            prefix,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_gate,
            bias,
            w_ld,
            w_sgu,
            w_sd,
            w13,
            w13s,
            w2,
            w2s,
            ln_w,
            w_up,
            out,
            scratch,
            plain,
            queue,
            peers,
            rank,
            layer,
            reset,
            diag,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch


TP_ = k2b.TP
