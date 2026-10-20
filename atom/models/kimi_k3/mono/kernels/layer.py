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

    CTAs 0 .. 55     K2a's reduce + AttnRes (slice, gate, mix -> x rows + XRDY);
                     then route (0 .. S-1) and the shared SiTU (8 .. 8+S-1).
                     Those two CTA ranges overlap once S > 8 -- CTA 8 .. S-1
                     runs both, in that order -- which costs latency on 8 of
                     the 56 AttnRes CTAs and nothing else: every route task and
                     every SiTU task still has exactly one CTA.
    CTAs 56 .. 159   o_proj (448 x 16 rows) -> partials to every rank; then
                     the shared gate/up (96 tasks, weights right after o_proj)
    CTAs 160 .. 243  router parts (28) and latent parts (56): weights first
    every CTA        ug; down (0 .. 223) or the shared down (224 .. 255);
                     lnorm (56 .. 83), up (0 .. 55), reduce (200 .. 255)
"""

from dataclasses import dataclass

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
    stage_gate,
    stage_mix,
    stage_slice,
)
from atom.models.kimi_k3.mono.kernels.stages import gemv
from atom.models.kimi_k3.mono.kernels.stages import moe as k2b
from atom.mono.device.ops import load_ptr64, traced
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, preg, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import pair_layout
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.debug import region_ids

O0, O1 = NA, BLOCKS  # o_proj CTAs: every CTA past AttnRes's
O_N = O1 - O0  # 200
OTASKS = k2a.OTASKS  # 448
O_ROUNDS = (OTASKS + O_N - 1) // O_N  # 3
E0 = 160  # router / latent CTAs (their weights in flight under their o_proj)
H0 = 8  # the first shared-SiTU CTA (after its AttnRes work); see above
# timeline points: 0 start, 1 o_proj done / AR reduced, 2 early done / slice,
# 3 ug, 4 down/sdown, 5 lnorm, 6 up, 7 reduce, 8 gate, 9 mix (x published),
# 10 route, 11 route table, 12 latent quant, 13 route starts
TL_POINTS = 17  # + 14 (unused), 15 INTER loaded, 16 down done

assert E0 + k2b.R_TASKS + k2b.L_TASKS <= BLOCKS and k2b.G_TASKS <= E0 - O0

REGIONS = ("part", "wgt", "msq", "xrdy", "ar") + k2b.REGIONS

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int
    nblocks: int  # the MLP AttnRes block sources
    eps: float = 1e-5  # mlp_res_norm
    out_eps: float = 1e-5  # post_attention_layernorm
    ln_eps: float = 1e-5  # the latent RMSNorm
    reduce: bool = True  # the final reduce here (else the next layer's K1)
    reset: bool = False  # a block-write layer: the prefix restarts at the o_proj output
    diag: bool = False
    timeline: bool = False
    xp: int = 0  # timing experiments (7: route without its selection)


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
        "peers",
        "rank",
        "layer",
        "diag",
        "tl",
    )
)


def peer_layout(s):
    """K2a's AR region, then K2b's: name -> (byte offset, bytes)."""
    pairs = {"ar": k2a.ar_pairs(s), **k2b.peer_pairs(s)}
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
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


@traced
def stage_oproj(c, gi):
    """o_proj tasks gi, gi + 200, gi + 400 (< 448): every task's weights in
    flight at once (one memory latency, not three), then the MFMAs."""
    ops = [k2a.o_loads(c, fx.min(gi + O_N * j, OTASKS - 1)) for j in range(O_ROUNDS)]
    k2a.load_core(c)
    for j in range_constexpr(O_ROUNDS):
        t = gi + O_N * j
        if t < OTASKS:
            k2a.stage_o(c, t, ops[j])


_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    assert 1 <= s <= 16
    # the early GEMVs stage their K slice a slab at a time; ``xl`` holds the
    # widest slab any of the three kinds asks for (one slab, i.e. the whole
    # slice, at every row count up to 8)
    xcols = max(gemv.slab_width(k2b.early_shape(kind)[0], s) for kind in (0, 1, 2))
    lay = scratch_layout(key)
    play = peer_layout(s)

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * k2b.TOPK, 16]
        flag: fx.Array[fx.Int32, k2b.E, 16]
        expert: fx.Array[fx.Int32, k2b.max_u(s), 16]
        kof: fx.Array[fx.Int32, k2b.max_u(s) * k2b.kof_stride(s), 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class EarlyLds:
        xl: fx.Array[fx.Int32, s * (xcols + gemv.LDS_PAD) // 2, 16]

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]

    @fx.struct
    class UgLds:
        latq: fx.Array[fx.Int32, s * k2b.LAT // 8, 16]
        latsc: fx.Array[fx.Int32, s * k2b.LAT // 32, 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    @fx.struct
    class DownLds:
        interq: fx.Array[fx.Int32, s * k2b.TOPK * 48, 16]
        intsc: fx.Array[fx.Int32, s * k2b.TOPK * k2b.UG_GROUPS, 16]
        contrib: fx.Array[fx.Float32, s * k2b.TOPK * k2b.ROWS, 16]

    @fx.union
    class StageLds:
        early: EarlyLds
        attn: AttnLds
        ug: UgLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_k2_s{s}_nb{key.nblocks}_x{int(key.reset)}_r{int(key.reduce)}"
        f"_g{int(key.diag)}_t{int(key.timeline)}"
    )

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
        peers: Int64,
        rank: Int32,
        layer: Int32,
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
        # ``layer`` is this launch's slot of the step; the step's first kernel
        # finds every pair zeroed and peer-fenced
        if const_expr(key.diag):
            mb = Mailbox(layer, diag, region_ids(REGIONS))
        else:
            mb = Mailbox(layer)
        bases = peer_bases(peers, TP_)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("rlog", "lat", "latr", "final", "ar"):
            off = play[region][0]
            peer[region] = [preg(b, off, region) for b in bases]
            peer[region + "_own"] = preg(own, off, region)
        c = {
            "S": s,
            "NS": ns,
            "NB": key.nblocks,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "delta": True,
            "reset": key.reset,
            "debug": False,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "ln_eps": key.ln_eps,
            "xp": key.xp,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "peer": peer,
            "ar_peer": peer["ar"],
            "ar_own": peer["ar_own"],
            "xl": el.xl.ptr,
            "vals": al.vals.ptr,
            "delta_lds": al.dl.ptr,
            "latq": ul.latq.ptr,
            "latsc": ul.latsc.ptr,
            "av": ul.av.ptr,
            "interq": dl.interq.ptr,
            "intsc": dl.intsc.ptr,
            "contrib": dl.contrib.ptr,
            "scan": rl.scan.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
            },
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "xflags": NA,
            "tl_on": key.timeline,
            "tls": lds.tls.ptr,
            "ln": scratch + fx.Int64(lay["ln"][0]),
            "shp": scratch + fx.Int64(lay["shp"][0]),
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
            "inter",
            "isc",
            "lmsq",
            "lnrdy",
            "shrdy",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        # ---- o_proj on every CTA past AttnRes's
        eb = bid - E0
        lb = eb - k2b.R_TASKS
        is_r = (bid >= E0) & (eb < k2b.R_TASKS)
        is_l = (bid >= E0 + k2b.R_TASKS) & (lb < k2b.L_TASKS)
        # o_proj first (every AttnRes CTA waits on the slowest partial; loads
        # retire in order), the early weights after: they land before x does
        if is_r:
            enter_stage("k2.o")
            stage_oproj(c, bid - O0)
            stamp(on, tls, tid, 1)
            enter_stage("k2.early")
            rops = k2b.early_loads(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS)
            k2b.early_compute(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS, rops)
        if is_l:
            enter_stage("k2.o")
            stage_oproj(c, bid - O0)
            stamp(on, tls, tid, 1)
            enter_stage("k2.early")
            lops = k2b.early_loads(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS)
            k2b.early_compute(c, 1, lb // k2b.L_PARTS, lb % k2b.L_PARTS, lops)
        if (bid >= O0) & ~is_r & ~is_l:
            enter_stage("k2.o")
            stage_oproj(c, bid - O0)
            stamp(on, tls, tid, 1)
            gi = bid - O0
            if gi < k2b.G_TASKS:
                enter_stage("k2.early")
                k2b.early_compute(c, 2, gi, 0, k2b.early_loads(c, 2, gi, 0))
        # ---- AttnRes: the reduced o_proj -> x; then route / SiTU
        if bid < NA:
            enter_stage("k2.ar")
            k2a.stage_reduce(c, bid)
            stamp(on, tls, tid, 1)
            enter_stage("k2.slice")
            stage_slice(c, bid)
            stamp(on, tls, tid, 2)
            enter_stage("k2.gate")
            if bid < s:
                stage_gate(c, bid)
            stamp(on, tls, tid, 8)
            enter_stage("k2.mix")
            stage_mix(c, bid)
            stamp(on, tls, tid, 9)
            stamp(on, tls, tid, 13)
            enter_stage("k2.route")
            if bid < s:
                k2b.stage_route(c, bid)
            stamp(on, tls, tid, 10)
            enter_stage("k2.h")
            if (bid >= H0) & (bid < H0 + s):
                k2b.stage_h(c, bid - H0)
        if bid >= NA:
            stamp(on, tls, tid, 2)
        # ---- the routed experts
        # the latent (ready ~10 us before the route) quantized first
        enter_stage("k2.ug")
        k2b.load_latq(c)
        stamp(on, tls, tid, 12)
        nu = k2b.load_route(c)
        stamp(on, tls, tid, 11)
        total = nu * k2b.UG_GROUPS
        last = total - 1
        # reversed placement: the round's leftover tasks go to the high CTAs,
        # whose early work ends first (the AttnRes / route CTAs start ug last)
        task = BLOCKS - 1 - bid
        ops = k2b.ug_loads(c, fx.min(task, last))
        while task < total:
            nxt = task + BLOCKS
            ops_n = k2b.ug_loads(c, fx.min(nxt, last))
            k2b.stage_ug(c, task, ops)
            task = nxt
            ops = ops_n
        stamp(on, tls, tid, 3)
        enter_stage("k2.down")
        if bid < k2b.DOWN_TASKS:
            dops = k2b.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
            k2b.load_inter(c)
            stamp(on, tls, tid, 15)
            k2b.stage_down(c, bid, nu, dops)
            stamp(on, tls, tid, 16)
        enter_stage("k2.sdown")
        k2b.run_sdown_rest(c, bid, w_sd)
        k2b.run_sdown(c, bid, w_sd)
        stamp(on, tls, tid, 4)
        enter_stage("k2.lnorm")
        if (bid >= k2b.LN0) & (bid < k2b.LN0 + k2b.LN_TASKS):
            k2b.stage_lnorm(c, bid - k2b.LN0)
        stamp(on, tls, tid, 5)
        enter_stage("k2.up")
        if bid < k2b.UP_TASKS:
            k2b.stage_up(c, bid)
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
        peers: Int64,
        rank: Int32,
        layer: Int32,
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
            peers,
            rank,
            layer,
            diag,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch


TP_ = k2b.TP
