# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A run of consecutive ordinary layers in one launch (the run kernel): ``layer.py``'s kernel body
inside a runtime loop over the run (``nlay`` iterations, one copy of the code),
iteration i with layer i's arguments, read at use from the run's table
(``ltab``: ``RUN_ARGS`` int64 a layer in ``layer.ABI`` order; an Int32
argument the low word) by scalar buffer loads -- invariant, so LLVM schedules
and rematerializes them as kernel arguments. The launch-invariant tail
(scratch, numa_word, sym, peers, rank) is the kernel's own. A CTA barrier
hands the LDS union to the next iteration. What a launch boundary handed layer
i + 1 is handed over in the launch:

    PND    every iteration's reduce puts this rank's MoE output columns
           (``out``) as tagged bf16 pairs at its MoE tag too; the next
           iteration's seam A slice units poll their (row, column pair) of it
           just before the fold, its other loads (rows, gates, norm and mix
           weights) issued as soon as the CTA leaves the iteration before.
           The run's first iteration takes pend from memory (the entry
           prefetch's, from ``k1_pend``) by a select: no branch. One region
           for every iteration: iteration i + 1's reduce comes after all of
           its slices consumed iteration i's pairs
    res / gates   one uncached set for every hand-off inside the run
           (``runner.res_pair`` / ``gates_pair``); the run's first layer reads
           ``res_ffn`` / ``gates_ffn``, its last writes them (cached, for the
           next launch)

The peer bases and the slice's X8G base are scalar loads before the loop
(no vector round trip ahead of the first iteration's slice loads). The code is
warmed once at the launch's entry (``warm_iter``: at every iteration's top
too). Every other stage waits on its own hand-offs as in a launch (tags apart
by layer); K1's and K2b's regions are disjoint, so nothing an iteration writes
early is still read by the one before it.
"""

import contextlib
import os
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.meta import capture_user_location
from flydsl.expr.typing import T, Int32, Int64

from aiter.ops.flydsl.kernels import buffer_ops as bo
from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import compress as cp
from atom.models.deepseek_v41.mono.kernels import layer as kl
from atom.models.deepseek_v41.mono.kernels import moe as k2b
from atom.models.deepseek_v41.mono.kernels.attn_post import (
    attn_post_args,
    attn_post_context,
    attn_post_smem,
    run_attn_post,
    run_gates,
)
from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    attn_pre_args,
    attn_pre_context,
    attn_pre_smem,
    lds_union,
    run_attn_pre,
    shard_seam,
)
from atom.models.deepseek_v41.mono.kernels.debug import mailbox
from atom.models.deepseek_v41.mono.kernels.dims import HIDDEN, Dims
from atom.models.deepseek_v41.mono.kernels.layer import (
    CODE_WARM_SPAN,
    FLAGS,
    TL_POINTS,
    LayerBuild,
    compress_context,
    compress_stage,
    scratch_layouts,
    warm_layer_code,
    window_rows_tagged,
)
from atom.models.deepseek_v41.mono.kernels.layer_post import (
    gather_x,
    publish_x,
    publish_x_tok_par,
    seam_dot_norm,
)
from atom.models.deepseek_v41.mono.kernels.mailbox import epoch_of, fresh_s32, fresh_s64, sreg
from atom.models.deepseek_v41.mono.kernels.moe import (
    RED_COLS,
    moe_args,
    moe_context,
    moe_smem,
    run_moe,
)
from atom.models.deepseek_v41.mono.kernels.moe_shape import route_shape
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.ops import CM_DEV, bf_hi, bf_lo, fresh, kernel_symbol, launcher, rsrc, traced, uniform
from atom.mono.plan.execution import WAVES
from atom.mono.device.stamps import stamp_begin
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, first_task
from atom.mono.plan.layout import pair_layout
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)
ABI = KernelAbi(("ltab", "nlay", 'scratch', 'numa_word', 'sym', 'peers', 'rank', 'epoch'))
assert RED_COLS == k1.COLS
# a layer's table row: every ``layer.ABI`` argument, int64 each
RUN_ARGS = len(kl.ABI.names)
assert RUN_ARGS == 149
RUN_ARG_NAMES = kl.ABI.names
# the slot of the layer's MoE tag (the next iteration's PND tag)
# hand-off rows this launch rereads (res_pair / gates_pair, res_attn / gates_attn)
# loaded at device scope: a CU's resident line of an earlier iteration is never
# returned (default policy can return it stale); ATOM_MONO_HANDCM=0: the old loads
HAND_CM = CM_DEV if os.environ.get("ATOM_MONO_HANDCM", "1") == "1" else 0
K_WQKV = RUN_ARG_NAMES.index("k1_wqkv")
K_WQKV_S = RUN_ARG_NAMES.index("k1_wqkv_s")
# (RQW) the reduce CTAs' wave 7 reads the next layer's q-task wqkv_a tile into L2
# during the reduce; on by default, ATOM_MONO_RQW=0 builds without it
RQW_SITE = os.environ.get("ATOM_MONO_RQW", "1") == "1"
K_MOE = 146
assert RUN_ARG_NAMES[K_MOE] == "moe_layer"
# CDNA buffer resource flags (DATA_FORMAT 32, NUM_FORMAT float: raw loads ignore them)
_RSRC_FLAGS = (7 << 12) | (4 << 15)

# widths the run kernel is built for (the pair's): past 10 rows the layer body spills
RUN_MAX_TOKENS = 10


def run_mode() -> int:
    """``ATOM_MONO_RUN`` (0: no run kernel; 1, the default: the run kernel, its
    code warmed at the launch's entry only; 2: at every iteration's top too;
    3: as 1, PND polled before the slice's row loads (a
    diagnostic))."""
    return int(os.environ.get("ATOM_MONO_RUN", "1"))


@dataclass(frozen=True)
class RunBuild:
    layer: LayerBuild
    # the code warm at every iteration's top too (else the launch's entry only)
    warm_iter: bool
    # (diagnostic) PND polled before the slice's row loads
    late: bool = False


def runnable(key: LayerBuild) -> bool:
    """An ordinary layer build the run kernel takes: TP8, sharded, slice_x8,
    folding, no indexer / compressor / aux tap / embedding / token parallel /
    timeline (``layer_pair.pairable``'s)."""
    pre = key.pre
    return (
        pre.tp == 8 and pre.shard and pre.slice_x8 and pre.fold and not pre.feeds_index
        and not pre.aux and not pre.embed_in and not pre.tok_par and not pre.timeline
        and not key.compress_ratio and key.post.index_bound_max == 0
    )  # fmt: skip


def pnd_pairs(key: LayerBuild) -> int:
    return key.pre.tokens * (HIDDEN // key.pre.tp) // 2


def run_scratch_layouts(rkey: RunBuild):
    """``layer.scratch_layouts`` and PND past them (``layer_pair``'s place)."""
    pre, post = scratch_layouts(rkey.layer)
    end = max(o + n for layout in (pre, post) for o, n in layout.values())
    pre = dict(pre)
    pre.update(pair_layout([("pnd", pnd_pairs(rkey.layer))], start=end))
    return pre, post


def run_scratch_bytes(rkey: RunBuild) -> int:
    return max(o + n for layout in run_scratch_layouts(rkey) for o, n in layout.values())


def _rsrc_v4(addr, nbytes):
    """A raw buffer resource over ``nbytes`` at ``addr`` as the <4 x i32> the
    scalar buffer load takes."""
    r = bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)
    i128 = ir.IntegerType.get_signless(128)
    v4 = ir.VectorType.get([4], T.i32)
    return llvm.bitcast(v4, llvm.ptrtoint(i128, r))


def _s_load(rsrc, byte_off, words):
    """``words`` (1 or 2) dwords at ``byte_off`` of ``rsrc`` by
    ``s_buffer_load`` (readnone in IR; an invariant, dereferenceable load in
    MIR: sunk to its use and rematerialized as a kernel argument is)."""
    off = fx.Int32(byte_off).ir_value()
    pol = fx.Int32(0).ir_value()
    if words == 1:
        return fx.Int32(
            llvm.call_intrinsic(T.i32, "llvm.amdgcn.s.buffer.load.i32", [rsrc, off, pol], [], [])
        )
    v2 = ir.VectorType.get([2], T.i32)
    w = llvm.call_intrinsic(v2, "llvm.amdgcn.s.buffer.load.v2i32", [rsrc, off, pol], [], [])
    return fx.Int64(llvm.bitcast(T.i64, w))


class LayerTable:
    """Layer ``it``'s row of the run's argument table at ``ltab``."""

    def __init__(self, ltab, it):
        base = fx.Int64(ltab) + fx.Int64(it) * (8 * RUN_ARGS)
        self.r = _rsrc_v4(base, 8 * RUN_ARGS)

    def i64(self, k):
        return _s_load(self.r, 8 * k, 2)

    def i32(self, k):
        return _s_load(self.r, 8 * k, 1)


def _rqw_loads(tid, w, ws, task):
    """(RQW) The 16 rows of q task ``task`` (``_fp8_step_load``'s row group:
    16 x HIDDEN fp8, contiguous) a dword an L2 line, and its scale row
    (``(task // 2)``'s HIDDEN // 32 bytes) whole, every load in flight at once,
    kept by one sink: one wait, after the last issue."""
    lane = tid % 64
    r = rsrc(w + fx.Int64(task) * (16 * HIDDEN), 16 * HIDDEN)
    vals = [
        fx.Int32(bo.buffer_load(r, (lane + 64 * k) * 32, vec_width=1, dtype=T.i32))
        for k in range_constexpr(16 * HIDDEN // (64 * 128))
    ]
    rs = rsrc(ws + fx.Int64(task // 2) * (HIDDEN // 32), HIDDEN // 32)
    vals.append(fx.Int32(bo.buffer_load(rs, lane % (HIDDEN // 128), vec_width=1, dtype=T.i32)))
    llvm.inline_asm(
        None, [v.ir_value() for v in vals],
        "; rqw l2 " + " ".join(f"${i}" for i in range(len(vals))),
        ",".join(["v"] * len(vals)), has_side_effects=True,
    )  # fmt: skip


@traced
def _rqw_site(tid, on, row, task, scratch):
    """(RQW) Wave 7 of a CTA where ``on``: ``_rqw_loads`` of the weights of
    the table row at ``row`` (layer i+1's); the reduce's other waves skip it."""
    if on & (uniform(tid // 64) == WAVES - 1):
        nxt = LayerTable(row, 0)
        _rqw_loads(tid, nxt.i64(K_WQKV), nxt.i64(K_WQKV_S), task)


def pend_poll(mb, pnd, s, tp, tid, live, j):
    """A slice thread's two ``pend`` values from PND (slice task j of this
    rank's, the thread's (row, column pair) as ``slice_direct_loads_live``'s),
    polled where ``live`` (``layer_pair``'s)."""
    split = k1.slice_split(s)
    q = fx.min(tid, k1.slice_direct_threads(s) - 1)
    p = q // split
    t = p // (k1.COLS // 2)
    cc = p % (k1.COLS // 2)
    pair = t * (HIDDEN // tp // 2) + j * (k1.COLS // 2) + cc
    w = mb.poll_need([(pnd, pair, 1)], [live])[0][0]
    return [bf_lo(w), bf_hi(w)]


@traced
def _iter(C, ltab, it, tid, bid, lds, tls, pnd_off, shared, tail, nlay):
    key = C["key"]
    pre = C["pre"]
    post = C["post"]
    s = C["s"]
    timeline = C["timeline"]
    key_a = C["key_a"]
    key_b = C["key_b"]
    rs = C["rs"]
    layout_1 = C["layout_1"]
    layout = C["layout"]
    layout_a = C["layout_a"]
    layout_b = C["layout_b"]
    members = C["members"]
    comp = C["comp"]
    warm = C["warm"]
    warm_iter = C["warm_iter"]
    late = C["late"]
    scratch, numa_word, sym, peers, rank, ep = tail
    it = fx.Int32(it)
    tab = LayerTable(ltab, it)
    k1_res_in = tab.i64(0)
    k1_pend = tab.i64(1)
    k1_post_in = tab.i64(2)
    k1_comb_in = tab.i64(3)
    k1_pre_in = tab.i64(4)
    k1_hc_fn = tab.i64(5)
    k1_hc_scale = tab.i64(6)
    k1_hc_base = tab.i64(7)
    k1_attn_w = tab.i64(8)
    k1_wqkv = tab.i64(9)
    k1_wqkv_s = tab.i64(10)
    k1_qn_w = tab.i64(11)
    k1_kvn_w = tab.i64(12)
    k1_wqb = tab.i64(13)
    k1_wqb_s = tab.i64(14)
    k1_cos = tab.i64(15)
    k1_sin = tab.i64(16)
    k1_pos = tab.i64(17)
    k1_ring = tab.i64(18)
    k1_ring_scale = tab.i64(19)
    k1_ring_rows = tab.i64(20)
    k1_ring_off = tab.i32(21)
    k1_res_out = tab.i64(22)
    k1_post_out = tab.i64(23)
    k1_comb_out = tab.i64(24)
    k1_pre_out = tab.i64(25)
    k1_q_out = tab.i64(26)
    k1_normed = tab.i64(27)
    k1_qr_out = tab.i64(28)
    k1_qrs_out = tab.i64(29)
    k1_aux = tab.i64(30)
    k1_iq_w = tab.i64(31)
    k1_iq_ws = tab.i64(32)
    k1_iw_w = tab.i64(33)
    k1_iq_out = tab.i64(34)
    k1_iqs_out = tab.i64(35)
    k1_iw_out = tab.i64(36)
    iact = tab.i32(37)
    iq = tab.i64(38)
    iw = tab.i64(39)
    plane = tab.i64(40)
    iqs = tab.i64(41)
    pscale = tab.i64(42)
    iplan = tab.i64(43)
    itab = tab.i64(44)
    itab_stride = tab.i32(45)
    itab_len = tab.i32(46)
    ibatch = tab.i64(47)
    ishift = tab.i32(48)
    ibound = tab.i64(49)
    ilog = tab.i64(50)
    lstride = tab.i32(51)
    isel = tab.i64(52)
    icand = tab.i64(53)
    ilift = tab.i32(54)
    iprod = tab.i32(55)
    ibmax = tab.i64(56)
    bstride = tab.i32(57)
    icout = tab.i64(58)
    irow = tab.i64(59)
    ixsel = tab.i64(60)
    sbound = tab.i64(61)
    q = tab.i64(62)
    pool = tab.i64(63)
    pool_scale = tab.i64(64)
    window = tab.i64(65)
    window_scale = tab.i64(66)
    sel = tab.i64(67)
    topk = tab.i32(68)
    kmeta = tab.i64(69)
    table = tab.i64(70)
    rows_per_page = tab.i32(71)
    page_rows = tab.i32(72)
    main_off = tab.i32(73)
    ring_off = tab.i32(74)
    ring_slots = tab.i32(75)
    sink = tab.i64(76)
    qk_scale = tab.i32(77)
    pos = tab.i64(78)
    cos = tab.i64(79)
    sin = tab.i64(80)
    woa = tab.i64(81)
    woa_s = tab.i64(82)
    wob = tab.i64(83)
    wob_s = tab.i64(84)
    res_in = tab.i64(85)
    post_in = tab.i64(86)
    comb_in = tab.i64(87)
    pre_in = tab.i64(88)
    hc_fn = tab.i64(89)
    hc_scale = tab.i64(90)
    hc_base = tab.i64(91)
    ffn_w = tab.i64(92)
    res_out = tab.i64(93)
    post_out = tab.i64(94)
    comb_out = tab.i64(95)
    pre_out = tab.i64(96)
    normed = tab.i64(97)
    gate_w = tab.i64(98)
    bias = tab.i64(99)
    w13 = tab.i64(100)
    w13_s = tab.i64(101)
    w2 = tab.i64(102)
    w2_s = tab.i64(103)
    sgu = tab.i64(104)
    sgu_s = tab.i64(105)
    sw2 = tab.i64(106)
    sw2_s = tab.i64(107)
    out = tab.i64(108)
    queue_words = tab.i64(109)
    kvrow = tab.i64(110)
    c_w = tab.i64(111)
    c_parts = tab.i64(112)
    c_cplan = tab.i64(113)
    c_ncap = tab.i32(114)
    c_wplan = tab.i64(115)
    c_nwcap = tab.i32(116)
    c_slots = tab.i64(117)
    c_kv_state = tab.i64(118)
    c_score_state = tab.i64(119)
    c_state_stride = tab.i64(120)
    c_ring = tab.i32(121)
    c_ape = tab.i64(122)
    c_norm_w = tab.i64(123)
    c_cos = tab.i64(124)
    c_sin = tab.i64(125)
    c_pages = tab.i64(126)
    c_page_stride = tab.i64(127)
    c_page_scale = tab.i64(128)
    c_scale_stride = tab.i64(129)
    c_tables = tab.i64(130)
    c_table_stride = tab.i32(131)
    c_per_page = tab.i32(132)
    c_wk = tab.i64(133)
    c_k_norm_w = tab.i64(134)
    c_ivals = tab.i64(135)
    c_iscales = tab.i64(136)
    c_latent = tab.i64(137)
    c_keys = tab.i64(138)
    layer = tab.i32(144)
    post_layer = tab.i32(145)
    moe_layer = tab.i32(146)
    prev_moe = LayerTable(ltab, fx.max(it - 1, fx.Int32(0))).i32(K_MOE)
    if const_expr(warm_iter):
        # (run kernel, warm_iter) every iteration warms the code as a
        # launch does today; else only the launch's entry does
        warm_layer_code(warm, tid, bid)
    # the scratch regions' base a value of its own (``sync.fresh_s64``), not
    # a part of the merged kernarg load a compressing build reloads whole
    scratch = fresh_s64(scratch)
    # the kernarg tail's other late-read scalars too: no merged 8- / 4-dword
    # kernarg tuple held across the kernel
    sym, peers, numa_word = fresh_s64(sym), fresh_s64(peers), fresh_s64(numa_word)
    rank, layer = fresh_s32(rank), fresh_s32(layer)
    post_layer, moe_layer = fresh_s32(post_layer), fresh_s32(moe_layer)
    k1_args = lambda: attn_pre_args(  # noqa: E731
        k1_res_in, k1_pend, k1_post_in, k1_comb_in, k1_pre_in, k1_hc_fn,
        k1_hc_scale, k1_hc_base, k1_attn_w, k1_wqkv, k1_wqkv_s, k1_qn_w, k1_kvn_w,
        k1_wqb, k1_wqb_s, k1_cos, k1_sin, k1_pos, k1_ring, k1_ring_scale,
        k1_ring_rows, k1_ring_off, k1_res_out, k1_post_out, k1_comb_out, k1_pre_out, k1_q_out,
        k1_normed, k1_qr_out, k1_qrs_out, k1_aux, k1_iq_w, k1_iq_ws, k1_iw_w,
        k1_iq_out, k1_iqs_out, k1_iw_out, ibound,
    )  # fmt: skip

    # the seam A slice unit's first loads before any peer table's
    # (their readfirstlanes waited on them, the slice's loads after:
    # a cold round trip in series on every slice's path). At TP8 only (no
    # gain measured at TP4, whose kernel sits near the VGPR cap). Only where
    # it prefetches are K1's args made ahead of the tables: every other
    # build traces as before
    entry_pre, args_1 = None, None
    prefetch = pre.shard and pre.slice_x8 and pre.fold and pre.tp == 8
    assert prefetch, "the run kernel: TP8 sharded slice_x8 folding builds only"
    # (run) the reduce -> next iteration's seam A hand-off: PND, this rank's
    # MoE output columns as tagged bf16 pairs at the previous iteration's
    # MoE tag; the run's first iteration takes pend from memory (``k1_pend``,
    # the previous launch's ``ffn_out``), loaded by the entry prefetch in
    # every iteration and selected (branch-free: need = i > 0)
    pnd = sreg(scratch, pnd_off, "pnd")
    mb_p = mailbox(fresh_s32(prev_moe), scratch, pre.diag_off, layout_1, epoch=ep)
    later = it > 0
    got_late = None
    if const_expr(late):
        # (diagnostic, late) PND polled before any of the slice's loads:
        # its rows (res, gates: the previous iteration's seam F / gate
        # writes) read only past the previous iteration's reduce
        unit0 = first_task(bid, 0)
        own0 = k1.SLICES // pre.tp
        got_late = pend_poll(
            mb_p, pnd, s, pre.tp, tid, (unit0 < own0 * pre.tp) & later,
            fx.min(unit0, own0 * pre.tp - 1) % own0,
        )  # fmt: skip
    if const_expr(prefetch):
        args_1 = k1_args()
        if const_expr(HAND_CM):
            args_1["hand_cm"] = HAND_CM
        entry_pre = k1.slice_entry_prefetch(args_1, s, pre.tp, bid, rank)
    # (run) the peer bases and X8G base, loaded once before the loop
    bases, x8b = shared
    if const_expr(not prefetch):
        args_1 = k1_args()
    args_1["kvrow"] = kvrow
    args_a = attn_post_args(
        iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len,
        ibatch, ishift, ibound, ilog, lstride,
        isel, icand, ilift, iprod, ibmax, bstride, icout, irow, ixsel, sbound,
        q, pool, pool_scale, window, window_scale, sel, topk, kmeta, table,
        rows_per_page, page_rows, main_off, ring_off, ring_slots, sink, qk_scale,
        pos, cos, sin, woa, woa_s, wob, wob_s, res_in, post_in, comb_in, pre_in,
        hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out, pre_out,
        normed,
    )  # fmt: skip
    args_a["kvrow"] = kvrow
    if const_expr(HAND_CM):
        args_a["hand_cm"] = HAND_CM
    args_b = moe_args(
        normed, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out,
        queue_words,
    )  # fmt: skip
    args_1["numa_word"] = args_a["numa_word"] = args_b["numa_word"] = numa_word
    mb_1 = mailbox(layer, scratch, pre.diag_off, layout_1, epoch=ep)
    c1, co1 = attn_pre_context(
        pre, lds, mb_1, layout_1, scratch, args_1, rank, sym, peers, bases=bases
    )
    if const_expr(late):
        c1["pend_sel"] = lambda live, j, mem: [
            later.select(p, m) for p, m in zip(got_late, mem)
        ]
    else:
        c1["pend_sel"] = lambda live, j, mem: [
            later.select(p, m)
            for p, m in zip(pend_poll(mb_p, pnd, s, pre.tp, tid, live & later, j), mem)
        ]
    # (run) the slice's destination X8G base, loaded once before the loop
    c1["x8_dst_base"] = x8b
    # the own-rows view too (token parallel: a copy of ``c1``)
    c1["fused"] = co1["fused"] = True
    if const_expr(comp is not None):
        # the compressor reads K1's normed rows in this launch
        c1["normed_cm"] = co1["normed_cm"] = CM_DEV
    ca = attn_post_context(
        key_a, s, lds, mailbox(post_layer, scratch, post.diag_off, layout, epoch=ep),
        bases, layout_a, scratch, args_a, rank, sym,
    )  # fmt: skip
    ca["normed_cm"] = CM_DEV
    ca["fused"] = True
    # seam F sharded as seam A (``attn_pre_context`` from ``pre.shard``)
    if const_expr(pre.shard):
        shard_seam(ca, ca["peer"], bases, sym, pre.tp, "pmixgf")
    ca["k1_poll"] = mb_1.poll
    for flag in FLAGS:
        ca[flag] = c1[flag]
    kvt_rows = window_rows_tagged(key)
    c1["kvt_rows"] = co1["kvt_rows"] = ca["kvt_rows"] = kvt_rows
    if const_expr(kvt_rows):
        ca["kvt"], ca["k1_poll_need"] = c1["kvt"], mb_1.poll_need
    if const_expr(comp is not None):
        # the attention reads this launch's new entries from ``kvrow``
        # past their CRDY (``attention._selected_row`` / ``await_selection``)
        ca["cp_ratio"], ca["cp_ncap"], ca["crdy"] = comp.ratio, c_ncap, c1["crdy"]
    cb = moe_context(
        s, rs, lds["moe"], mailbox(moe_layer, scratch, post.diag_off, layout, epoch=ep),
        bases, layout_b, scratch, args_b, rank, sym, key_b.numa,
    )  # fmt: skip
    cb["x_cm"], cb["x_ready"] = CM_DEV, True
    # (run) every iteration's reduce puts its columns to PND too
    cb["pnd"] = pnd
    if const_expr(RQW_SITE):
        # (RQW) layer i+1's q-task wqkv_a tile into L2 from the
        # reduce CTAs' idle wave 7 (their q weights otherwise issue only past
        # the reduce)
        nxt_row = fx.Int64(ltab) + fx.Int64(fx.min(it + 1, nlay - 1)) * (8 * RUN_ARGS)
        rqw_task = first_task(bid, k1.GATE0)
        cb["reduce_side"] = lambda: _rqw_site(
            tid,
            (it + 1 < nlay) & (rqw_task < k1.Q_TASKS)
            & ((bid - rot + BLOCKS) % BLOCKS < k1.SLICES // pre.tp),
            nxt_row, rqw_task, scratch,
        )
    # the MoE output reduce-scattered too: the next seam A folds this
    # rank's columns only
    cb["shard_f"] = pre.shard
    assert key_b.after_seam != pre.tok_par
    if const_expr(pre.tok_par):
        # the other ranks' normed rows gathered for the MoE
        cb["tok_par"] = True
        cb["gather_x"] = lambda t: gather_x(cb, t)
        after_norm = lambda t: publish_x_tok_par(ca, cb, t)
    else:
        after_norm = lambda t: publish_x(ca, cb, t)
    if const_expr(rs.router_reads_seam):
        cb["dot_norm"] = seam_dot_norm(ca)
    stamp_begin(timeline, tls, tid, TL_POINTS)
    # seam A's gates after the attention, off the wo_a CTAs
    # (``seam_a_gates0``): only seam F reads them, and in K1 they would
    # hold up the wq_b tasks on their CTAs
    to_owner = k2a.wob_to_owner(ca)
    rot = k2a.seam_rot(s, ca["d"], to_owner, key_a.score0)
    gates_at = k2a.seam_a_gates0(s, ca["d"], to_owner, key_a.score0)
    cc = None
    if const_expr(comp is not None):
        cargs = {
            "w": c_w, "parts": c_parts, "cplan": c_cplan, "ncap": c_ncap,
            "wplan": c_wplan, "nwcap": c_nwcap, "slots": c_slots,
            "kv_state": c_kv_state, "score_state": c_score_state,
            "state_stride": c_state_stride, "ring": c_ring, "ape": c_ape,
            "norm_w": c_norm_w, "cos": c_cos, "sin": c_sin, "pages": c_pages,
            "page_stride": c_page_stride, "page_scale": c_page_scale,
            "scale_stride": c_scale_stride, "tables": c_tables,
            "table_stride": c_table_stride, "per_page": c_per_page, "wk": c_wk,
            "k_norm_w": c_k_norm_w, "ivals": c_ivals, "iscales": c_iscales,
            "latent": c_latent, "keys": c_keys,
        }  # fmt: skip
        cc = compress_context(
            comp, c1, cargs, lds["cp"].cpw.ptr, mb_1, k1_normed, kvrow, tid,
            bid, s, k1.APART_PROJ0 if key.q_apart else 0,
        )  # fmt: skip
    # the compressor: its projection right after K1's norm (its rows the
    # earliest, the pool's partials with them), its pool / key / ring
    # tasks after K1's stages; its LDS between barriers
    run_attn_pre(
        c1, co1, pre, tls, 0, lds["k1_iw"].nbl.ptr, gates=False,
        after_norm=None if comp is None else lambda: compress_stage(cp.project, cc),
        q_apart=key.q_apart, entry_pre=entry_pre,
    )  # fmt: skip
    if const_expr(comp is not None):
        compress_stage(cp.plan_rows, cc)
    # the LDS union is handed over by K2a's first barrier (``run_attn_post``)
    tl1 = len(k1.TL_STAGES)
    own_n = s // pre.tp if pre.tok_par else s
    run_attn_post(
        ca, key_a, bid, tls, tl1, after_norm, gates=False,
        before_woa=lambda: k1.run_seam_a_gates(c1, co1, bid, own_n, gates_at),
        after_index=(
            (lambda: k1.run_wqb_late(c1, pre, bid)) if pre.feeds_index else None
        ),
    )  # fmt: skip
    # K2b's placements turned with seam F's (``seam_rot``)
    mbid = (bid - rot + BLOCKS) % BLOCKS
    cb["bid"] = mbid
    # the FFN gates on the CTAs past the route's (K2b's placement, turned):
    # the ug tasks' first round reaches them only with many experts
    gates0 = (rs.route0 + s + rot) % BLOCKS
    run_moe(
        cb, key_b, mbid, tls, tl1 + len(k2a.TL_STAGES) - 1,
        before_ug=lambda: run_gates(ca, key_a, bid, gates0),
    )  # fmt: skip


# the run's loop: no LICM (a value hoisted out of it is live through the
# whole body -- the lane math LICM moved there took 256 VGPRs and scratch),
# no unrolling
_LOOP_ANNOTATION = "#llvm.loop_annotation<unroll = <disable = true>, licm = <disable = true>>"
# thread / CTA ids and the launch-invariant tail through a side-effecting move
# at each iteration's top: nothing derived from them leaves the loop
# (MachineLICM / codegen hoisting past the annotation: 256 VGPRs, scratch).
# Every helper reads the ids as ``fx.thread_idx.x`` / ``fx.block_idx.x``: those
# read the iteration's while its body is traced (``_iter_ids``)


def _volatile(v, wide):
    ty, cls, op = (T.i64, fx.Int64, "s_mov_b64") if wide else (T.i32, fx.Int32, "s_mov_b32")
    return cls(
        llvm.inline_asm(
            ty, [cls(v).ir_value()], f"{op} $0, $1 ; run iter", "=s,s",
            has_side_effects=True,
        )
    )  # fmt: skip


class _IdProxy:
    """``fx.thread_idx`` / ``fx.block_idx`` with the iteration's ``x``."""

    def __init__(self, orig, x):
        self._orig, self.x = orig, x

    def __getattr__(self, name):
        return getattr(self._orig, name)


@contextlib.contextmanager
def _iter_ids(tid, bid):
    saved = fx.thread_idx, fx.block_idx
    fx.thread_idx, fx.block_idx = _IdProxy(saved[0], tid), _IdProxy(saved[1], bid)
    try:
        yield
    finally:
        fx.thread_idx, fx.block_idx = saved


def _loop(C, ltab, nlay, tid, bid, lds, tls, pnd_off, shared, tail):
    """``for it in range(nlay)``: an ``scf.for`` built here (as flydsl's
    ``scf_for_dispatch`` builds one, no carried values) to carry the loop
    annotation, which rides ``scf.for`` -> ``cf`` -> ``llvm.br`` to
    ``!llvm.loop``."""
    loc = capture_user_location()
    zero, one = fx.Int32(0).ir_value(), fx.Int32(1).ir_value()
    for_op = scf.ForOp(zero, fx.Int32(nlay).ir_value(), one, loc=loc)
    for_op.attributes["loop_annotation"] = ir.Attribute.parse(_LOOP_ANNOTATION)
    with ir.InsertionPoint(for_op.body):
        it = fx.Int32(for_op.induction_variable)
        # masked back into their ranges: LLVM's known bits (a non-negative id
        # below THREADS / BLOCKS) keep the lane / task math unsigned shifts and
        # masks, as on the hardware ids
        assert THREADS & (THREADS - 1) == 0 and BLOCKS & (BLOCKS - 1) == 0
        t = fresh(tid) & (THREADS - 1)
        b = _volatile(bid, False) & (BLOCKS - 1)
        scratch, numa_word, sym, peers, rank, ep = tail
        tail_i = (
            _volatile(scratch, True), _volatile(numa_word, True), _volatile(sym, True),
            _volatile(peers, True), _volatile(rank, False), ep,
        )  # fmt: skip
        with _iter_ids(t, b):
            _iter(C, ltab, it, t, b, lds, tls, pnd_off, shared, tail_i, nlay)
        # the iteration's LDS union handed over to the next
        gpu.barrier()
        scf.YieldOp([], loc=capture_user_location())


def build_layer_run(rkey: RunBuild):
    key, warm_iter, late = rkey.layer, rkey.warm_iter, rkey.late
    assert runnable(key), rkey
    pre, post = key.pre, key.post
    s, timeline = pre.tokens, pre.timeline
    key_a, key_b = post.halves()
    rs = route_shape(key_b)
    layout_1, layout = run_scratch_layouts(rkey)
    layout_a = {n: layout[n] for n in k2a.scratch_layout(key_a)}
    layout_b = {n: layout[n] for n in (*k2b.scratch_layout(key_b), "xrdy")}
    keyed = key_tuple(rkey, SOURCES)
    members_1 = attn_pre_smem(pre)
    members_a = attn_post_smem(
        s, False, post.index_bound_max, Dims(post.tp), post.kv_fp8
    )
    members = dict(members_1)
    members.update(members_a)
    members["moe"] = moe_smem(s, rs, False)
    comp = None
    Parts = lds_union("LayerLds", members)
    warm = CODE_WARM_SPAN > 0
    pnd_off = layout_1["pnd"][0]
    own = k1.SLICES // pre.tp
    ranks = rs.dims.ranks
    C = dict(key=key, pre=pre, post=post, s=s, timeline=timeline, key_a=key_a, key_b=key_b, rs=rs,
             layout_1=layout_1, layout=layout, layout_a=layout_a, layout_b=layout_b,
             members=members, comp=comp, warm=warm, warm_iter=warm_iter, late=late)

    @fx.struct
    class Stamps:
        tls: fx.Array[fx.Int64, 1, 16]

    name = kernel_symbol(
        "v41_lrun", s=s, w=int(warm_iter), l=int(late), q=int(RQW_SITE), hc=int(HAND_CM != 0), kv8=int(post.kv_fp8), sh=int(pre.shard),
    )  # fmt: skip

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def run_kernel(
        ltab: Int64,
        nlay: Int32,
        scratch: Int64,
        numa_word: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        if const_expr(not warm_iter):
            # the code read into every XCD's L2 once, at the launch's entry
            warm_layer_code(warm, tid, bid)
        alloc = fx.SharedAllocator()
        tls = alloc.allocate(Stamps).peek().tls.ptr
        smem = alloc.allocate(Parts)
        lds = {n: getattr(smem, n).peek() for n in members}
        ltab = fresh_s64(ltab)
        # every rank's peer base and the slice's destination X8G base, once:
        # scalar loads (no vector round trip ahead of the first iteration's
        # slice loads; rematerializable as kernel arguments)
        ptab = _rsrc_v4(fresh_s64(peers), 8 * ranks)
        bases = [_s_load(ptab, 8 * p, 2) for p in range(ranks)]
        x8b = _s_load(ptab, (first_task(bid, 0) // own) * 8, 2)
        _loop(
            C, ltab, nlay, tid, bid, lds, tls, pnd_off, (bases, x8b),
            (scratch, numa_word, sym, peers, rank, epoch_of(epoch)),
        )  # fmt: skip
        _ = keyed

    @launcher(name)
    def launch(
        ltab: Int64,
        nlay: Int32,
        scratch: Int64,
        numa_word: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        run_kernel(
            ltab, nlay, scratch, numa_word, sym, peers, rank, epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(run_kernel, launch)
    return launch
