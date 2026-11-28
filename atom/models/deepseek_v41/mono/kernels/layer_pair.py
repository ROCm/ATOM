# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Two consecutive ordinary layers in one launch (the pair kernel): ``layer.py``'s
kernel body run twice on every CTA, body a with layer l's arguments, body b with
layer l + 1's (a CTA barrier hands the LDS union over). The body is layer.py's
(kept in step with it), with the hand-offs below. What the launch boundary
handed layer l + 1 is handed over in the launch:

    PND    body a's reduce puts this rank's MoE output columns (``out``) as
           tagged bf16 pairs at its MoE tag too; body b's seam A slice units
           poll their (row, column pair) of it in place of loading ``pend``.
           A reduce unit's columns are exactly a slice task's (RED_COLS == COLS)
    res / gates   body a's FFN seam writes the pair's own uncached residual and
           gates (``runner.res_pair`` / ``gates_pair``), body b's attention seam
           reads them on other XCDs; every single layer keeps the cached ones
    peers  body b reuses body a's peer bases and its slice's X8G base (no table
           load in body b: its slice path is kernarg -> entry prefetch -> PND)

``mode`` 1: body b polls PND first, then issues the slice's entry prefetch as a
launch does; 2 (the default): body b issues the entry prefetch as soon as its
CTA ends body a and polls PND just before the fold, overlapping the slice's
x-independent loads with layer l's tail. Body a warms its own code only
(``PAIR_WARM_A``; warming the whole pair kept the warming CTAs late into body
a), body b its own at its start. Every other stage of body b waits on its own
hand-offs as in a launch (tags apart by layer); K1's and K2b's regions are
disjoint, so nothing body b writes early is still read by body a.
"""

import os
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64

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
from atom.models.deepseek_v41.mono.kernels.code_warm import warm_code
from atom.models.deepseek_v41.mono.kernels.debug import mailbox
from atom.models.deepseek_v41.mono.kernels.dims import HIDDEN, Dims
from atom.models.deepseek_v41.mono.kernels.layer import (
    CODE_WARM_CTA0,
    CODE_WARM_SPAN,
    CODE_WARM_WAVE,
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
from atom.models.deepseek_v41.mono.kernels.mailbox import (
    epoch_of,
    fresh_s32,
    fresh_s64,
    sreg,
)
from atom.models.deepseek_v41.mono.kernels.moe import (
    RED_COLS,
    moe_args,
    moe_context,
    moe_smem,
    run_moe,
)
from atom.models.deepseek_v41.mono.kernels.moe_shape import route_shape
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    kernel_symbol,
    launcher,
    load_ptr64,
    traced,
    uniform,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp_begin
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, first_task
from atom.mono.plan.layout import pair_layout
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)
ABI = KernelAbi(tuple(f"a_{n}" for n in kl.ABI.names) + tuple(f"b_{n}" for n in kl.ABI.names))
assert RED_COLS == k1.COLS


# widths the pair is built for: past 10 rows its body pair spills (11 / 12: 20 B scratch)
PAIR_MAX_TOKENS = 10
# hand-off rows this launch rereads (res_pair / gates_pair, res_attn / gates_attn)
# loaded at device scope: a CU's resident line of an earlier iteration is never
# returned (default policy can return it stale); ATOM_MONO_HANDCM=0: the old loads
HAND_CM = CM_DEV if os.environ.get("ATOM_MONO_HANDCM", "1") == "1" else 0


def pair_mode() -> int:
    """``ATOM_MONO_PAIR`` (0, the default: no pairs -- the run kernel,
    ``layer_run``, takes every run of ordinary layers; 1: pairs, no prefetch;
    2: pairs, prefetch; pairs only where ``ATOM_MONO_RUN`` is 0)."""
    return int(os.environ.get("ATOM_MONO_PAIR", "0"))


@dataclass(frozen=True)
class PairBuild:
    layer: LayerBuild
    mode: int


def pairable(key: LayerBuild) -> bool:
    """An ordinary layer build the pair takes: TP8, sharded, slice_x8, folding,
    no indexer / compressor / aux tap / embedding / token parallel / timeline."""
    pre = key.pre
    return (
        pre.tp == 8 and pre.shard and pre.slice_x8 and pre.fold and not pre.feeds_index
        and not pre.aux and not pre.embed_in and not pre.tok_par and not pre.timeline
        and not key.compress_ratio and key.post.index_bound_max == 0
    )  # fmt: skip


def pnd_pairs(key: LayerBuild) -> int:
    return key.pre.tokens * (HIDDEN // key.pre.tp) // 2


def pair_scratch_layouts(pkey: PairBuild):
    """``layer.scratch_layouts`` and PND past them."""
    pre, post = scratch_layouts(pkey.layer)
    end = max(o + n for layout in (pre, post) for o, n in layout.values())
    pre = dict(pre)
    pre.update(pair_layout([("pnd", pnd_pairs(pkey.layer))], start=end))
    return pre, post


def pair_scratch_bytes(pkey: PairBuild) -> int:
    return max(o + n for layout in pair_scratch_layouts(pkey) for o, n in layout.values())


# body a's code warm span: a single ordinary layer's code (104..114 KB at s <= 10) and a margin
PAIR_WARM_A = 120 * 1024


@traced
def warm_pair_code(on, tid, bid, span):
    """``layer.warm_layer_code`` over ``span`` bytes from the pc."""
    if const_expr(on):  # noqa: SIM102
        if (bid >= CODE_WARM_CTA0) & (uniform(tid // 64) == CODE_WARM_WAVE):
            warm_code(tid % 64, span)


def pend_poll(mb, pnd, s, tp, tid, live, j):
    """A slice thread's two ``pend`` values from PND (slice task j of this
    rank's, the thread's (row, column pair) as ``slice_direct_loads_live``'s),
    polled where ``live``."""
    split = k1.slice_split(s)
    q = fx.min(tid, k1.slice_direct_threads(s) - 1)
    p = q // split
    t = p // (k1.COLS // 2)
    cc = p % (k1.COLS // 2)
    pair = t * (HIDDEN // tp // 2) + j * (k1.COLS // 2) + cc
    w = mb.poll_need([(pnd, pair, 1)], [live])[0][0]
    return [bf_lo(w), bf_hi(w)]


@traced
def _body(C, A, role, mode, tid, bid, lds, tls, pnd_off, pnd_tag, shared):
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
    k1_res_in = A["k1_res_in"]
    k1_pend = A["k1_pend"]
    k1_post_in = A["k1_post_in"]
    k1_comb_in = A["k1_comb_in"]
    k1_pre_in = A["k1_pre_in"]
    k1_hc_fn = A["k1_hc_fn"]
    k1_hc_scale = A["k1_hc_scale"]
    k1_hc_base = A["k1_hc_base"]
    k1_attn_w = A["k1_attn_w"]
    k1_wqkv = A["k1_wqkv"]
    k1_wqkv_s = A["k1_wqkv_s"]
    k1_qn_w = A["k1_qn_w"]
    k1_kvn_w = A["k1_kvn_w"]
    k1_wqb = A["k1_wqb"]
    k1_wqb_s = A["k1_wqb_s"]
    k1_cos = A["k1_cos"]
    k1_sin = A["k1_sin"]
    k1_pos = A["k1_pos"]
    k1_ring = A["k1_ring"]
    k1_ring_scale = A["k1_ring_scale"]
    k1_ring_rows = A["k1_ring_rows"]
    k1_ring_off = A["k1_ring_off"]
    k1_res_out = A["k1_res_out"]
    k1_post_out = A["k1_post_out"]
    k1_comb_out = A["k1_comb_out"]
    k1_pre_out = A["k1_pre_out"]
    k1_q_out = A["k1_q_out"]
    k1_normed = A["k1_normed"]
    k1_qr_out = A["k1_qr_out"]
    k1_qrs_out = A["k1_qrs_out"]
    k1_aux = A["k1_aux"]
    k1_iq_w = A["k1_iq_w"]
    k1_iq_ws = A["k1_iq_ws"]
    k1_iw_w = A["k1_iw_w"]
    k1_iq_out = A["k1_iq_out"]
    k1_iqs_out = A["k1_iqs_out"]
    k1_iw_out = A["k1_iw_out"]
    iact = A["iact"]
    iq = A["iq"]
    iw = A["iw"]
    plane = A["plane"]
    iqs = A["iqs"]
    pscale = A["pscale"]
    iplan = A["iplan"]
    itab = A["itab"]
    itab_stride = A["itab_stride"]
    itab_len = A["itab_len"]
    ibatch = A["ibatch"]
    ishift = A["ishift"]
    ibound = A["ibound"]
    ilog = A["ilog"]
    lstride = A["lstride"]
    isel = A["isel"]
    icand = A["icand"]
    ilift = A["ilift"]
    iprod = A["iprod"]
    ibmax = A["ibmax"]
    bstride = A["bstride"]
    icout = A["icout"]
    irow = A["irow"]
    ixsel = A["ixsel"]
    sbound = A["sbound"]
    q = A["q"]
    pool = A["pool"]
    pool_scale = A["pool_scale"]
    window = A["window"]
    window_scale = A["window_scale"]
    sel = A["sel"]
    topk = A["topk"]
    kmeta = A["kmeta"]
    table = A["table"]
    rows_per_page = A["rows_per_page"]
    page_rows = A["page_rows"]
    main_off = A["main_off"]
    ring_off = A["ring_off"]
    ring_slots = A["ring_slots"]
    sink = A["sink"]
    qk_scale = A["qk_scale"]
    pos = A["pos"]
    cos = A["cos"]
    sin = A["sin"]
    woa = A["woa"]
    woa_s = A["woa_s"]
    wob = A["wob"]
    wob_s = A["wob_s"]
    res_in = A["res_in"]
    post_in = A["post_in"]
    comb_in = A["comb_in"]
    pre_in = A["pre_in"]
    hc_fn = A["hc_fn"]
    hc_scale = A["hc_scale"]
    hc_base = A["hc_base"]
    ffn_w = A["ffn_w"]
    res_out = A["res_out"]
    post_out = A["post_out"]
    comb_out = A["comb_out"]
    pre_out = A["pre_out"]
    normed = A["normed"]
    gate_w = A["gate_w"]
    bias = A["bias"]
    w13 = A["w13"]
    w13_s = A["w13_s"]
    w2 = A["w2"]
    w2_s = A["w2_s"]
    sgu = A["sgu"]
    sgu_s = A["sgu_s"]
    sw2 = A["sw2"]
    sw2_s = A["sw2_s"]
    out = A["out"]
    queue_words = A["queue_words"]
    kvrow = A["kvrow"]
    c_w = A["c_w"]
    c_parts = A["c_parts"]
    c_cplan = A["c_cplan"]
    c_ncap = A["c_ncap"]
    c_wplan = A["c_wplan"]
    c_nwcap = A["c_nwcap"]
    c_slots = A["c_slots"]
    c_kv_state = A["c_kv_state"]
    c_score_state = A["c_score_state"]
    c_state_stride = A["c_state_stride"]
    c_ring = A["c_ring"]
    c_ape = A["c_ape"]
    c_norm_w = A["c_norm_w"]
    c_cos = A["c_cos"]
    c_sin = A["c_sin"]
    c_pages = A["c_pages"]
    c_page_stride = A["c_page_stride"]
    c_page_scale = A["c_page_scale"]
    c_scale_stride = A["c_scale_stride"]
    c_tables = A["c_tables"]
    c_table_stride = A["c_table_stride"]
    c_per_page = A["c_per_page"]
    c_wk = A["c_wk"]
    c_k_norm_w = A["c_k_norm_w"]
    c_ivals = A["c_ivals"]
    c_iscales = A["c_iscales"]
    c_latent = A["c_latent"]
    c_keys = A["c_keys"]
    scratch = A["scratch"]
    numa_word = A["numa_word"]
    sym = A["sym"]
    peers = A["peers"]
    rank = A["rank"]
    layer = A["layer"]
    post_layer = A["post_layer"]
    moe_layer = A["moe_layer"]
    tl = A["tl"]
    # the step's epoch: every hand-off of the body tagged with it (``kernels.mailbox``)
    ep = epoch_of(A["epoch"])
    if const_expr(role == 0):
        # body a warms its own code only (a single layer's span):
        # warming the whole pair's code would keep the warming CTAs 248..255
        # late into body a (their kv wqkv_a tasks -> KVT -> the merge); body b
        # warms its own at its start
        warm_pair_code(warm, tid, bid, PAIR_WARM_A)
    else:
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
    assert prefetch, "the pair: TP8 sharded slice_x8 folding builds only"
    # (pair) the reduce -> next seam A hand-off: PND, this rank's MoE output
    # columns as tagged bf16 pairs at body a's MoE tag
    pnd = sreg(scratch, pnd_off, "pnd")
    mb_p = mailbox(fresh_s32(pnd_tag), scratch, pre.diag_off, layout_1, epoch=ep)
    pend_src = None
    if const_expr(role == 1 and mode == 1):
        # body b, no prefetch: pend polled first, then the entry prefetch
        unit0 = first_task(bid, 0)
        own0 = k1.SLICES // pre.tp
        pend_src = pend_poll(
            mb_p, pnd, s, pre.tp, tid, unit0 < own0 * pre.tp,
            fx.min(unit0, own0 * pre.tp - 1) % own0,
        )  # fmt: skip
    if const_expr(role == 1 and mode == 2):
        # body b, prefetch: the slice's other loads first, pend polled late
        pend_src = "late"
    if const_expr(prefetch):
        args_1 = k1_args()
        if const_expr(HAND_CM and role == 1):
            # body b reads body a's res_pair / gates_pair
            args_1["hand_cm"] = HAND_CM
        entry_pre = k1.slice_entry_prefetch(
            args_1, s, pre.tp, bid, rank, pend=pend_src
        )
    if const_expr(role == 0):
        bases = peer_bases(peers, rs.dims.ranks)
        # the slice's destination X8G base with the peer tables
        x8b = load_ptr64(peers, first_task(bid, 0) // (k1.SLICES // pre.tp))
    else:
        # body b: body a's peer bases and X8G base, no table load
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
        # each body's seam F rereads res_attn / gates_attn
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
    if const_expr(role == 1 and mode == 2):
        c1["pend_late"] = lambda live, j: pend_poll(mb_p, pnd, s, pre.tp, tid, live, j)
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
    if const_expr(role == 0):
        # (pair) body a's reduce puts its columns to PND too
        cb["pnd"] = pnd
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
    return bases, x8b


def build_layer_pair(pkey: PairBuild):
    key, mode = pkey.layer, pkey.mode
    assert pairable(key) and mode in (1, 2), pkey
    pre, post = key.pre, key.post
    s, timeline = pre.tokens, pre.timeline
    key_a, key_b = post.halves()
    rs = route_shape(key_b)
    layout_1, layout = pair_scratch_layouts(pkey)
    layout_a = {n: layout[n] for n in k2a.scratch_layout(key_a)}
    layout_b = {n: layout[n] for n in (*k2b.scratch_layout(key_b), "xrdy")}
    keyed = key_tuple(pkey, SOURCES)
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
    C = dict(key=key, pre=pre, post=post, s=s, timeline=timeline, key_a=key_a, key_b=key_b, rs=rs,
             layout_1=layout_1, layout=layout, layout_a=layout_a, layout_b=layout_b,
             members=members, comp=comp, warm=warm)

    @fx.struct
    class Stamps:
        tls: fx.Array[fx.Int64, 1, 16]

    name = kernel_symbol(
        "v41_lpair", s=s, m=mode, hc=int(HAND_CM != 0), kv8=int(post.kv_fp8), sh=int(pre.shard),
    )  # fmt: skip

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def pair_kernel(
        a_k1_res_in: Int64,
        a_k1_pend: Int64,
        a_k1_post_in: Int64,
        a_k1_comb_in: Int64,
        a_k1_pre_in: Int64,
        a_k1_hc_fn: Int64,
        a_k1_hc_scale: Int64,
        a_k1_hc_base: Int64,
        a_k1_attn_w: Int64,
        a_k1_wqkv: Int64,
        a_k1_wqkv_s: Int64,
        a_k1_qn_w: Int64,
        a_k1_kvn_w: Int64,
        a_k1_wqb: Int64,
        a_k1_wqb_s: Int64,
        a_k1_cos: Int64,
        a_k1_sin: Int64,
        a_k1_pos: Int64,
        a_k1_ring: Int64,
        a_k1_ring_scale: Int64,
        a_k1_ring_rows: Int64,
        a_k1_ring_off: Int32,
        a_k1_res_out: Int64,
        a_k1_post_out: Int64,
        a_k1_comb_out: Int64,
        a_k1_pre_out: Int64,
        a_k1_q_out: Int64,
        a_k1_normed: Int64,
        a_k1_qr_out: Int64,
        a_k1_qrs_out: Int64,
        a_k1_aux: Int64,
        a_k1_iq_w: Int64,
        a_k1_iq_ws: Int64,
        a_k1_iw_w: Int64,
        a_k1_iq_out: Int64,
        a_k1_iqs_out: Int64,
        a_k1_iw_out: Int64,
        a_iact: Int32,
        a_iq: Int64,
        a_iw: Int64,
        a_plane: Int64,
        a_iqs: Int64,
        a_pscale: Int64,
        a_iplan: Int64,
        a_itab: Int64,
        a_itab_stride: Int32,
        a_itab_len: Int32,
        a_ibatch: Int64,
        a_ishift: Int32,
        a_ibound: Int64,
        a_ilog: Int64,
        a_lstride: Int32,
        a_isel: Int64,
        a_icand: Int64,
        a_ilift: Int32,
        a_iprod: Int32,
        a_ibmax: Int64,
        a_bstride: Int32,
        a_icout: Int64,
        a_irow: Int64,
        a_ixsel: Int64,
        a_sbound: Int64,
        a_q: Int64,
        a_pool: Int64,
        a_pool_scale: Int64,
        a_window: Int64,
        a_window_scale: Int64,
        a_sel: Int64,
        a_topk: Int32,
        a_kmeta: Int64,
        a_table: Int64,
        a_rows_per_page: Int32,
        a_page_rows: Int32,
        a_main_off: Int32,
        a_ring_off: Int32,
        a_ring_slots: Int32,
        a_sink: Int64,
        a_qk_scale: Int32,
        a_pos: Int64,
        a_cos: Int64,
        a_sin: Int64,
        a_woa: Int64,
        a_woa_s: Int64,
        a_wob: Int64,
        a_wob_s: Int64,
        a_res_in: Int64,
        a_post_in: Int64,
        a_comb_in: Int64,
        a_pre_in: Int64,
        a_hc_fn: Int64,
        a_hc_scale: Int64,
        a_hc_base: Int64,
        a_ffn_w: Int64,
        a_res_out: Int64,
        a_post_out: Int64,
        a_comb_out: Int64,
        a_pre_out: Int64,
        a_normed: Int64,
        a_gate_w: Int64,
        a_bias: Int64,
        a_w13: Int64,
        a_w13_s: Int64,
        a_w2: Int64,
        a_w2_s: Int64,
        a_sgu: Int64,
        a_sgu_s: Int64,
        a_sw2: Int64,
        a_sw2_s: Int64,
        a_out: Int64,
        a_queue_words: Int64,
        a_kvrow: Int64,
        a_c_w: Int64,
        a_c_parts: Int64,
        a_c_cplan: Int64,
        a_c_ncap: Int32,
        a_c_wplan: Int64,
        a_c_nwcap: Int32,
        a_c_slots: Int64,
        a_c_kv_state: Int64,
        a_c_score_state: Int64,
        a_c_state_stride: Int64,
        a_c_ring: Int32,
        a_c_ape: Int64,
        a_c_norm_w: Int64,
        a_c_cos: Int64,
        a_c_sin: Int64,
        a_c_pages: Int64,
        a_c_page_stride: Int64,
        a_c_page_scale: Int64,
        a_c_scale_stride: Int64,
        a_c_tables: Int64,
        a_c_table_stride: Int32,
        a_c_per_page: Int32,
        a_c_wk: Int64,
        a_c_k_norm_w: Int64,
        a_c_ivals: Int64,
        a_c_iscales: Int64,
        a_c_latent: Int64,
        a_c_keys: Int64,
        a_scratch: Int64,
        a_numa_word: Int64,
        a_sym: Int64,
        a_peers: Int64,
        a_rank: Int32,
        a_layer: Int32,
        a_post_layer: Int32,
        a_moe_layer: Int32,
        a_tl: Int64,
        a_epoch: Int64,
        b_k1_res_in: Int64,
        b_k1_pend: Int64,
        b_k1_post_in: Int64,
        b_k1_comb_in: Int64,
        b_k1_pre_in: Int64,
        b_k1_hc_fn: Int64,
        b_k1_hc_scale: Int64,
        b_k1_hc_base: Int64,
        b_k1_attn_w: Int64,
        b_k1_wqkv: Int64,
        b_k1_wqkv_s: Int64,
        b_k1_qn_w: Int64,
        b_k1_kvn_w: Int64,
        b_k1_wqb: Int64,
        b_k1_wqb_s: Int64,
        b_k1_cos: Int64,
        b_k1_sin: Int64,
        b_k1_pos: Int64,
        b_k1_ring: Int64,
        b_k1_ring_scale: Int64,
        b_k1_ring_rows: Int64,
        b_k1_ring_off: Int32,
        b_k1_res_out: Int64,
        b_k1_post_out: Int64,
        b_k1_comb_out: Int64,
        b_k1_pre_out: Int64,
        b_k1_q_out: Int64,
        b_k1_normed: Int64,
        b_k1_qr_out: Int64,
        b_k1_qrs_out: Int64,
        b_k1_aux: Int64,
        b_k1_iq_w: Int64,
        b_k1_iq_ws: Int64,
        b_k1_iw_w: Int64,
        b_k1_iq_out: Int64,
        b_k1_iqs_out: Int64,
        b_k1_iw_out: Int64,
        b_iact: Int32,
        b_iq: Int64,
        b_iw: Int64,
        b_plane: Int64,
        b_iqs: Int64,
        b_pscale: Int64,
        b_iplan: Int64,
        b_itab: Int64,
        b_itab_stride: Int32,
        b_itab_len: Int32,
        b_ibatch: Int64,
        b_ishift: Int32,
        b_ibound: Int64,
        b_ilog: Int64,
        b_lstride: Int32,
        b_isel: Int64,
        b_icand: Int64,
        b_ilift: Int32,
        b_iprod: Int32,
        b_ibmax: Int64,
        b_bstride: Int32,
        b_icout: Int64,
        b_irow: Int64,
        b_ixsel: Int64,
        b_sbound: Int64,
        b_q: Int64,
        b_pool: Int64,
        b_pool_scale: Int64,
        b_window: Int64,
        b_window_scale: Int64,
        b_sel: Int64,
        b_topk: Int32,
        b_kmeta: Int64,
        b_table: Int64,
        b_rows_per_page: Int32,
        b_page_rows: Int32,
        b_main_off: Int32,
        b_ring_off: Int32,
        b_ring_slots: Int32,
        b_sink: Int64,
        b_qk_scale: Int32,
        b_pos: Int64,
        b_cos: Int64,
        b_sin: Int64,
        b_woa: Int64,
        b_woa_s: Int64,
        b_wob: Int64,
        b_wob_s: Int64,
        b_res_in: Int64,
        b_post_in: Int64,
        b_comb_in: Int64,
        b_pre_in: Int64,
        b_hc_fn: Int64,
        b_hc_scale: Int64,
        b_hc_base: Int64,
        b_ffn_w: Int64,
        b_res_out: Int64,
        b_post_out: Int64,
        b_comb_out: Int64,
        b_pre_out: Int64,
        b_normed: Int64,
        b_gate_w: Int64,
        b_bias: Int64,
        b_w13: Int64,
        b_w13_s: Int64,
        b_w2: Int64,
        b_w2_s: Int64,
        b_sgu: Int64,
        b_sgu_s: Int64,
        b_sw2: Int64,
        b_sw2_s: Int64,
        b_out: Int64,
        b_queue_words: Int64,
        b_kvrow: Int64,
        b_c_w: Int64,
        b_c_parts: Int64,
        b_c_cplan: Int64,
        b_c_ncap: Int32,
        b_c_wplan: Int64,
        b_c_nwcap: Int32,
        b_c_slots: Int64,
        b_c_kv_state: Int64,
        b_c_score_state: Int64,
        b_c_state_stride: Int64,
        b_c_ring: Int32,
        b_c_ape: Int64,
        b_c_norm_w: Int64,
        b_c_cos: Int64,
        b_c_sin: Int64,
        b_c_pages: Int64,
        b_c_page_stride: Int64,
        b_c_page_scale: Int64,
        b_c_scale_stride: Int64,
        b_c_tables: Int64,
        b_c_table_stride: Int32,
        b_c_per_page: Int32,
        b_c_wk: Int64,
        b_c_k_norm_w: Int64,
        b_c_ivals: Int64,
        b_c_iscales: Int64,
        b_c_latent: Int64,
        b_c_keys: Int64,
        b_scratch: Int64,
        b_numa_word: Int64,
        b_sym: Int64,
        b_peers: Int64,
        b_rank: Int32,
        b_layer: Int32,
        b_post_layer: Int32,
        b_moe_layer: Int32,
        b_tl: Int64,
        b_epoch: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        tls = alloc.allocate(Stamps).peek().tls.ptr
        smem = alloc.allocate(Parts)
        lds = {n: getattr(smem, n).peek() for n in members}
        A = {"k1_res_in": a_k1_res_in, "k1_pend": a_k1_pend, "k1_post_in": a_k1_post_in, "k1_comb_in": a_k1_comb_in, "k1_pre_in": a_k1_pre_in, "k1_hc_fn": a_k1_hc_fn, "k1_hc_scale": a_k1_hc_scale, "k1_hc_base": a_k1_hc_base, "k1_attn_w": a_k1_attn_w, "k1_wqkv": a_k1_wqkv, "k1_wqkv_s": a_k1_wqkv_s, "k1_qn_w": a_k1_qn_w, "k1_kvn_w": a_k1_kvn_w, "k1_wqb": a_k1_wqb, "k1_wqb_s": a_k1_wqb_s, "k1_cos": a_k1_cos, "k1_sin": a_k1_sin, "k1_pos": a_k1_pos, "k1_ring": a_k1_ring, "k1_ring_scale": a_k1_ring_scale, "k1_ring_rows": a_k1_ring_rows, "k1_ring_off": a_k1_ring_off, "k1_res_out": a_k1_res_out, "k1_post_out": a_k1_post_out, "k1_comb_out": a_k1_comb_out, "k1_pre_out": a_k1_pre_out, "k1_q_out": a_k1_q_out, "k1_normed": a_k1_normed, "k1_qr_out": a_k1_qr_out, "k1_qrs_out": a_k1_qrs_out, "k1_aux": a_k1_aux, "k1_iq_w": a_k1_iq_w, "k1_iq_ws": a_k1_iq_ws, "k1_iw_w": a_k1_iw_w, "k1_iq_out": a_k1_iq_out, "k1_iqs_out": a_k1_iqs_out, "k1_iw_out": a_k1_iw_out, "iact": a_iact, "iq": a_iq, "iw": a_iw, "plane": a_plane, "iqs": a_iqs, "pscale": a_pscale, "iplan": a_iplan, "itab": a_itab, "itab_stride": a_itab_stride, "itab_len": a_itab_len, "ibatch": a_ibatch, "ishift": a_ishift, "ibound": a_ibound, "ilog": a_ilog, "lstride": a_lstride, "isel": a_isel, "icand": a_icand, "ilift": a_ilift, "iprod": a_iprod, "ibmax": a_ibmax, "bstride": a_bstride, "icout": a_icout, "irow": a_irow, "ixsel": a_ixsel, "sbound": a_sbound, "q": a_q, "pool": a_pool, "pool_scale": a_pool_scale, "window": a_window, "window_scale": a_window_scale, "sel": a_sel, "topk": a_topk, "kmeta": a_kmeta, "table": a_table, "rows_per_page": a_rows_per_page, "page_rows": a_page_rows, "main_off": a_main_off, "ring_off": a_ring_off, "ring_slots": a_ring_slots, "sink": a_sink, "qk_scale": a_qk_scale, "pos": a_pos, "cos": a_cos, "sin": a_sin, "woa": a_woa, "woa_s": a_woa_s, "wob": a_wob, "wob_s": a_wob_s, "res_in": a_res_in, "post_in": a_post_in, "comb_in": a_comb_in, "pre_in": a_pre_in, "hc_fn": a_hc_fn, "hc_scale": a_hc_scale, "hc_base": a_hc_base, "ffn_w": a_ffn_w, "res_out": a_res_out, "post_out": a_post_out, "comb_out": a_comb_out, "pre_out": a_pre_out, "normed": a_normed, "gate_w": a_gate_w, "bias": a_bias, "w13": a_w13, "w13_s": a_w13_s, "w2": a_w2, "w2_s": a_w2_s, "sgu": a_sgu, "sgu_s": a_sgu_s, "sw2": a_sw2, "sw2_s": a_sw2_s, "out": a_out, "queue_words": a_queue_words, "kvrow": a_kvrow, "c_w": a_c_w, "c_parts": a_c_parts, "c_cplan": a_c_cplan, "c_ncap": a_c_ncap, "c_wplan": a_c_wplan, "c_nwcap": a_c_nwcap, "c_slots": a_c_slots, "c_kv_state": a_c_kv_state, "c_score_state": a_c_score_state, "c_state_stride": a_c_state_stride, "c_ring": a_c_ring, "c_ape": a_c_ape, "c_norm_w": a_c_norm_w, "c_cos": a_c_cos, "c_sin": a_c_sin, "c_pages": a_c_pages, "c_page_stride": a_c_page_stride, "c_page_scale": a_c_page_scale, "c_scale_stride": a_c_scale_stride, "c_tables": a_c_tables, "c_table_stride": a_c_table_stride, "c_per_page": a_c_per_page, "c_wk": a_c_wk, "c_k_norm_w": a_c_k_norm_w, "c_ivals": a_c_ivals, "c_iscales": a_c_iscales, "c_latent": a_c_latent, "c_keys": a_c_keys, "scratch": a_scratch, "numa_word": a_numa_word, "sym": a_sym, "peers": a_peers, "rank": a_rank, "layer": a_layer, "post_layer": a_post_layer, "moe_layer": a_moe_layer, "tl": a_tl, "epoch": a_epoch}
        B = {"k1_res_in": b_k1_res_in, "k1_pend": b_k1_pend, "k1_post_in": b_k1_post_in, "k1_comb_in": b_k1_comb_in, "k1_pre_in": b_k1_pre_in, "k1_hc_fn": b_k1_hc_fn, "k1_hc_scale": b_k1_hc_scale, "k1_hc_base": b_k1_hc_base, "k1_attn_w": b_k1_attn_w, "k1_wqkv": b_k1_wqkv, "k1_wqkv_s": b_k1_wqkv_s, "k1_qn_w": b_k1_qn_w, "k1_kvn_w": b_k1_kvn_w, "k1_wqb": b_k1_wqb, "k1_wqb_s": b_k1_wqb_s, "k1_cos": b_k1_cos, "k1_sin": b_k1_sin, "k1_pos": b_k1_pos, "k1_ring": b_k1_ring, "k1_ring_scale": b_k1_ring_scale, "k1_ring_rows": b_k1_ring_rows, "k1_ring_off": b_k1_ring_off, "k1_res_out": b_k1_res_out, "k1_post_out": b_k1_post_out, "k1_comb_out": b_k1_comb_out, "k1_pre_out": b_k1_pre_out, "k1_q_out": b_k1_q_out, "k1_normed": b_k1_normed, "k1_qr_out": b_k1_qr_out, "k1_qrs_out": b_k1_qrs_out, "k1_aux": b_k1_aux, "k1_iq_w": b_k1_iq_w, "k1_iq_ws": b_k1_iq_ws, "k1_iw_w": b_k1_iw_w, "k1_iq_out": b_k1_iq_out, "k1_iqs_out": b_k1_iqs_out, "k1_iw_out": b_k1_iw_out, "iact": b_iact, "iq": b_iq, "iw": b_iw, "plane": b_plane, "iqs": b_iqs, "pscale": b_pscale, "iplan": b_iplan, "itab": b_itab, "itab_stride": b_itab_stride, "itab_len": b_itab_len, "ibatch": b_ibatch, "ishift": b_ishift, "ibound": b_ibound, "ilog": b_ilog, "lstride": b_lstride, "isel": b_isel, "icand": b_icand, "ilift": b_ilift, "iprod": b_iprod, "ibmax": b_ibmax, "bstride": b_bstride, "icout": b_icout, "irow": b_irow, "ixsel": b_ixsel, "sbound": b_sbound, "q": b_q, "pool": b_pool, "pool_scale": b_pool_scale, "window": b_window, "window_scale": b_window_scale, "sel": b_sel, "topk": b_topk, "kmeta": b_kmeta, "table": b_table, "rows_per_page": b_rows_per_page, "page_rows": b_page_rows, "main_off": b_main_off, "ring_off": b_ring_off, "ring_slots": b_ring_slots, "sink": b_sink, "qk_scale": b_qk_scale, "pos": b_pos, "cos": b_cos, "sin": b_sin, "woa": b_woa, "woa_s": b_woa_s, "wob": b_wob, "wob_s": b_wob_s, "res_in": b_res_in, "post_in": b_post_in, "comb_in": b_comb_in, "pre_in": b_pre_in, "hc_fn": b_hc_fn, "hc_scale": b_hc_scale, "hc_base": b_hc_base, "ffn_w": b_ffn_w, "res_out": b_res_out, "post_out": b_post_out, "comb_out": b_comb_out, "pre_out": b_pre_out, "normed": b_normed, "gate_w": b_gate_w, "bias": b_bias, "w13": b_w13, "w13_s": b_w13_s, "w2": b_w2, "w2_s": b_w2_s, "sgu": b_sgu, "sgu_s": b_sgu_s, "sw2": b_sw2, "sw2_s": b_sw2_s, "out": b_out, "queue_words": b_queue_words, "kvrow": b_kvrow, "c_w": b_c_w, "c_parts": b_c_parts, "c_cplan": b_c_cplan, "c_ncap": b_c_ncap, "c_wplan": b_c_wplan, "c_nwcap": b_c_nwcap, "c_slots": b_c_slots, "c_kv_state": b_c_kv_state, "c_score_state": b_c_score_state, "c_state_stride": b_c_state_stride, "c_ring": b_c_ring, "c_ape": b_c_ape, "c_norm_w": b_c_norm_w, "c_cos": b_c_cos, "c_sin": b_c_sin, "c_pages": b_c_pages, "c_page_stride": b_c_page_stride, "c_page_scale": b_c_page_scale, "c_scale_stride": b_c_scale_stride, "c_tables": b_c_tables, "c_table_stride": b_c_table_stride, "c_per_page": b_c_per_page, "c_wk": b_c_wk, "c_k_norm_w": b_c_k_norm_w, "c_ivals": b_c_ivals, "c_iscales": b_c_iscales, "c_latent": b_c_latent, "c_keys": b_c_keys, "scratch": b_scratch, "numa_word": b_numa_word, "sym": b_sym, "peers": b_peers, "rank": b_rank, "layer": b_layer, "post_layer": b_post_layer, "moe_layer": b_moe_layer, "tl": b_tl, "epoch": b_epoch}
        shared = _body(C, A, 0, mode, tid, bid, lds, tls, pnd_off, a_moe_layer, None)
        # body a's LDS union handed over to body b
        gpu.barrier()
        _body(C, B, 1, mode, tid, bid, lds, tls, pnd_off, a_moe_layer, shared)
        _ = keyed

    @launcher(name)
    def launch(
        a_k1_res_in: Int64,
        a_k1_pend: Int64,
        a_k1_post_in: Int64,
        a_k1_comb_in: Int64,
        a_k1_pre_in: Int64,
        a_k1_hc_fn: Int64,
        a_k1_hc_scale: Int64,
        a_k1_hc_base: Int64,
        a_k1_attn_w: Int64,
        a_k1_wqkv: Int64,
        a_k1_wqkv_s: Int64,
        a_k1_qn_w: Int64,
        a_k1_kvn_w: Int64,
        a_k1_wqb: Int64,
        a_k1_wqb_s: Int64,
        a_k1_cos: Int64,
        a_k1_sin: Int64,
        a_k1_pos: Int64,
        a_k1_ring: Int64,
        a_k1_ring_scale: Int64,
        a_k1_ring_rows: Int64,
        a_k1_ring_off: Int32,
        a_k1_res_out: Int64,
        a_k1_post_out: Int64,
        a_k1_comb_out: Int64,
        a_k1_pre_out: Int64,
        a_k1_q_out: Int64,
        a_k1_normed: Int64,
        a_k1_qr_out: Int64,
        a_k1_qrs_out: Int64,
        a_k1_aux: Int64,
        a_k1_iq_w: Int64,
        a_k1_iq_ws: Int64,
        a_k1_iw_w: Int64,
        a_k1_iq_out: Int64,
        a_k1_iqs_out: Int64,
        a_k1_iw_out: Int64,
        a_iact: Int32,
        a_iq: Int64,
        a_iw: Int64,
        a_plane: Int64,
        a_iqs: Int64,
        a_pscale: Int64,
        a_iplan: Int64,
        a_itab: Int64,
        a_itab_stride: Int32,
        a_itab_len: Int32,
        a_ibatch: Int64,
        a_ishift: Int32,
        a_ibound: Int64,
        a_ilog: Int64,
        a_lstride: Int32,
        a_isel: Int64,
        a_icand: Int64,
        a_ilift: Int32,
        a_iprod: Int32,
        a_ibmax: Int64,
        a_bstride: Int32,
        a_icout: Int64,
        a_irow: Int64,
        a_ixsel: Int64,
        a_sbound: Int64,
        a_q: Int64,
        a_pool: Int64,
        a_pool_scale: Int64,
        a_window: Int64,
        a_window_scale: Int64,
        a_sel: Int64,
        a_topk: Int32,
        a_kmeta: Int64,
        a_table: Int64,
        a_rows_per_page: Int32,
        a_page_rows: Int32,
        a_main_off: Int32,
        a_ring_off: Int32,
        a_ring_slots: Int32,
        a_sink: Int64,
        a_qk_scale: Int32,
        a_pos: Int64,
        a_cos: Int64,
        a_sin: Int64,
        a_woa: Int64,
        a_woa_s: Int64,
        a_wob: Int64,
        a_wob_s: Int64,
        a_res_in: Int64,
        a_post_in: Int64,
        a_comb_in: Int64,
        a_pre_in: Int64,
        a_hc_fn: Int64,
        a_hc_scale: Int64,
        a_hc_base: Int64,
        a_ffn_w: Int64,
        a_res_out: Int64,
        a_post_out: Int64,
        a_comb_out: Int64,
        a_pre_out: Int64,
        a_normed: Int64,
        a_gate_w: Int64,
        a_bias: Int64,
        a_w13: Int64,
        a_w13_s: Int64,
        a_w2: Int64,
        a_w2_s: Int64,
        a_sgu: Int64,
        a_sgu_s: Int64,
        a_sw2: Int64,
        a_sw2_s: Int64,
        a_out: Int64,
        a_queue_words: Int64,
        a_kvrow: Int64,
        a_c_w: Int64,
        a_c_parts: Int64,
        a_c_cplan: Int64,
        a_c_ncap: Int32,
        a_c_wplan: Int64,
        a_c_nwcap: Int32,
        a_c_slots: Int64,
        a_c_kv_state: Int64,
        a_c_score_state: Int64,
        a_c_state_stride: Int64,
        a_c_ring: Int32,
        a_c_ape: Int64,
        a_c_norm_w: Int64,
        a_c_cos: Int64,
        a_c_sin: Int64,
        a_c_pages: Int64,
        a_c_page_stride: Int64,
        a_c_page_scale: Int64,
        a_c_scale_stride: Int64,
        a_c_tables: Int64,
        a_c_table_stride: Int32,
        a_c_per_page: Int32,
        a_c_wk: Int64,
        a_c_k_norm_w: Int64,
        a_c_ivals: Int64,
        a_c_iscales: Int64,
        a_c_latent: Int64,
        a_c_keys: Int64,
        a_scratch: Int64,
        a_numa_word: Int64,
        a_sym: Int64,
        a_peers: Int64,
        a_rank: Int32,
        a_layer: Int32,
        a_post_layer: Int32,
        a_moe_layer: Int32,
        a_tl: Int64,
        a_epoch: Int64,
        b_k1_res_in: Int64,
        b_k1_pend: Int64,
        b_k1_post_in: Int64,
        b_k1_comb_in: Int64,
        b_k1_pre_in: Int64,
        b_k1_hc_fn: Int64,
        b_k1_hc_scale: Int64,
        b_k1_hc_base: Int64,
        b_k1_attn_w: Int64,
        b_k1_wqkv: Int64,
        b_k1_wqkv_s: Int64,
        b_k1_qn_w: Int64,
        b_k1_kvn_w: Int64,
        b_k1_wqb: Int64,
        b_k1_wqb_s: Int64,
        b_k1_cos: Int64,
        b_k1_sin: Int64,
        b_k1_pos: Int64,
        b_k1_ring: Int64,
        b_k1_ring_scale: Int64,
        b_k1_ring_rows: Int64,
        b_k1_ring_off: Int32,
        b_k1_res_out: Int64,
        b_k1_post_out: Int64,
        b_k1_comb_out: Int64,
        b_k1_pre_out: Int64,
        b_k1_q_out: Int64,
        b_k1_normed: Int64,
        b_k1_qr_out: Int64,
        b_k1_qrs_out: Int64,
        b_k1_aux: Int64,
        b_k1_iq_w: Int64,
        b_k1_iq_ws: Int64,
        b_k1_iw_w: Int64,
        b_k1_iq_out: Int64,
        b_k1_iqs_out: Int64,
        b_k1_iw_out: Int64,
        b_iact: Int32,
        b_iq: Int64,
        b_iw: Int64,
        b_plane: Int64,
        b_iqs: Int64,
        b_pscale: Int64,
        b_iplan: Int64,
        b_itab: Int64,
        b_itab_stride: Int32,
        b_itab_len: Int32,
        b_ibatch: Int64,
        b_ishift: Int32,
        b_ibound: Int64,
        b_ilog: Int64,
        b_lstride: Int32,
        b_isel: Int64,
        b_icand: Int64,
        b_ilift: Int32,
        b_iprod: Int32,
        b_ibmax: Int64,
        b_bstride: Int32,
        b_icout: Int64,
        b_irow: Int64,
        b_ixsel: Int64,
        b_sbound: Int64,
        b_q: Int64,
        b_pool: Int64,
        b_pool_scale: Int64,
        b_window: Int64,
        b_window_scale: Int64,
        b_sel: Int64,
        b_topk: Int32,
        b_kmeta: Int64,
        b_table: Int64,
        b_rows_per_page: Int32,
        b_page_rows: Int32,
        b_main_off: Int32,
        b_ring_off: Int32,
        b_ring_slots: Int32,
        b_sink: Int64,
        b_qk_scale: Int32,
        b_pos: Int64,
        b_cos: Int64,
        b_sin: Int64,
        b_woa: Int64,
        b_woa_s: Int64,
        b_wob: Int64,
        b_wob_s: Int64,
        b_res_in: Int64,
        b_post_in: Int64,
        b_comb_in: Int64,
        b_pre_in: Int64,
        b_hc_fn: Int64,
        b_hc_scale: Int64,
        b_hc_base: Int64,
        b_ffn_w: Int64,
        b_res_out: Int64,
        b_post_out: Int64,
        b_comb_out: Int64,
        b_pre_out: Int64,
        b_normed: Int64,
        b_gate_w: Int64,
        b_bias: Int64,
        b_w13: Int64,
        b_w13_s: Int64,
        b_w2: Int64,
        b_w2_s: Int64,
        b_sgu: Int64,
        b_sgu_s: Int64,
        b_sw2: Int64,
        b_sw2_s: Int64,
        b_out: Int64,
        b_queue_words: Int64,
        b_kvrow: Int64,
        b_c_w: Int64,
        b_c_parts: Int64,
        b_c_cplan: Int64,
        b_c_ncap: Int32,
        b_c_wplan: Int64,
        b_c_nwcap: Int32,
        b_c_slots: Int64,
        b_c_kv_state: Int64,
        b_c_score_state: Int64,
        b_c_state_stride: Int64,
        b_c_ring: Int32,
        b_c_ape: Int64,
        b_c_norm_w: Int64,
        b_c_cos: Int64,
        b_c_sin: Int64,
        b_c_pages: Int64,
        b_c_page_stride: Int64,
        b_c_page_scale: Int64,
        b_c_scale_stride: Int64,
        b_c_tables: Int64,
        b_c_table_stride: Int32,
        b_c_per_page: Int32,
        b_c_wk: Int64,
        b_c_k_norm_w: Int64,
        b_c_ivals: Int64,
        b_c_iscales: Int64,
        b_c_latent: Int64,
        b_c_keys: Int64,
        b_scratch: Int64,
        b_numa_word: Int64,
        b_sym: Int64,
        b_peers: Int64,
        b_rank: Int32,
        b_layer: Int32,
        b_post_layer: Int32,
        b_moe_layer: Int32,
        b_tl: Int64,
        b_epoch: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        pair_kernel(
            a_k1_res_in, a_k1_pend, a_k1_post_in, a_k1_comb_in, a_k1_pre_in, a_k1_hc_fn, a_k1_hc_scale, a_k1_hc_base, a_k1_attn_w, a_k1_wqkv, a_k1_wqkv_s, a_k1_qn_w, a_k1_kvn_w, a_k1_wqb, a_k1_wqb_s, a_k1_cos, a_k1_sin, a_k1_pos, a_k1_ring, a_k1_ring_scale, a_k1_ring_rows, a_k1_ring_off, a_k1_res_out, a_k1_post_out, a_k1_comb_out, a_k1_pre_out, a_k1_q_out, a_k1_normed, a_k1_qr_out, a_k1_qrs_out, a_k1_aux, a_k1_iq_w, a_k1_iq_ws, a_k1_iw_w, a_k1_iq_out, a_k1_iqs_out, a_k1_iw_out, a_iact, a_iq, a_iw, a_plane, a_iqs, a_pscale, a_iplan, a_itab, a_itab_stride, a_itab_len, a_ibatch, a_ishift, a_ibound, a_ilog, a_lstride, a_isel, a_icand, a_ilift, a_iprod, a_ibmax, a_bstride, a_icout, a_irow, a_ixsel, a_sbound, a_q, a_pool, a_pool_scale, a_window, a_window_scale, a_sel, a_topk, a_kmeta, a_table, a_rows_per_page, a_page_rows, a_main_off, a_ring_off, a_ring_slots, a_sink, a_qk_scale, a_pos, a_cos, a_sin, a_woa, a_woa_s, a_wob, a_wob_s, a_res_in, a_post_in, a_comb_in, a_pre_in, a_hc_fn, a_hc_scale, a_hc_base, a_ffn_w, a_res_out, a_post_out, a_comb_out, a_pre_out, a_normed, a_gate_w, a_bias, a_w13, a_w13_s, a_w2, a_w2_s, a_sgu, a_sgu_s, a_sw2, a_sw2_s, a_out, a_queue_words, a_kvrow, a_c_w, a_c_parts, a_c_cplan, a_c_ncap, a_c_wplan, a_c_nwcap, a_c_slots, a_c_kv_state, a_c_score_state, a_c_state_stride, a_c_ring, a_c_ape, a_c_norm_w, a_c_cos, a_c_sin, a_c_pages, a_c_page_stride, a_c_page_scale, a_c_scale_stride, a_c_tables, a_c_table_stride, a_c_per_page, a_c_wk, a_c_k_norm_w, a_c_ivals, a_c_iscales, a_c_latent, a_c_keys, a_scratch, a_numa_word, a_sym, a_peers, a_rank, a_layer, a_post_layer, a_moe_layer, a_tl, a_epoch, b_k1_res_in, b_k1_pend, b_k1_post_in, b_k1_comb_in, b_k1_pre_in, b_k1_hc_fn, b_k1_hc_scale, b_k1_hc_base, b_k1_attn_w, b_k1_wqkv, b_k1_wqkv_s, b_k1_qn_w, b_k1_kvn_w, b_k1_wqb, b_k1_wqb_s, b_k1_cos, b_k1_sin, b_k1_pos, b_k1_ring, b_k1_ring_scale, b_k1_ring_rows, b_k1_ring_off, b_k1_res_out, b_k1_post_out, b_k1_comb_out, b_k1_pre_out, b_k1_q_out, b_k1_normed, b_k1_qr_out, b_k1_qrs_out, b_k1_aux, b_k1_iq_w, b_k1_iq_ws, b_k1_iw_w, b_k1_iq_out, b_k1_iqs_out, b_k1_iw_out, b_iact, b_iq, b_iw, b_plane, b_iqs, b_pscale, b_iplan, b_itab, b_itab_stride, b_itab_len, b_ibatch, b_ishift, b_ibound, b_ilog, b_lstride, b_isel, b_icand, b_ilift, b_iprod, b_ibmax, b_bstride, b_icout, b_irow, b_ixsel, b_sbound, b_q, b_pool, b_pool_scale, b_window, b_window_scale, b_sel, b_topk, b_kmeta, b_table, b_rows_per_page, b_page_rows, b_main_off, b_ring_off, b_ring_slots, b_sink, b_qk_scale, b_pos, b_cos, b_sin, b_woa, b_woa_s, b_wob, b_wob_s, b_res_in, b_post_in, b_comb_in, b_pre_in, b_hc_fn, b_hc_scale, b_hc_base, b_ffn_w, b_res_out, b_post_out, b_comb_out, b_pre_out, b_normed, b_gate_w, b_bias, b_w13, b_w13_s, b_w2, b_w2_s, b_sgu, b_sgu_s, b_sw2, b_sw2_s, b_out, b_queue_words, b_kvrow, b_c_w, b_c_parts, b_c_cplan, b_c_ncap, b_c_wplan, b_c_nwcap, b_c_slots, b_c_kv_state, b_c_score_state, b_c_state_stride, b_c_ring, b_c_ape, b_c_norm_w, b_c_cos, b_c_sin, b_c_pages, b_c_page_stride, b_c_page_scale, b_c_scale_stride, b_c_tables, b_c_table_stride, b_c_per_page, b_c_wk, b_c_k_norm_w, b_c_ivals, b_c_iscales, b_c_latent, b_c_keys, b_scratch, b_numa_word, b_sym, b_peers, b_rank, b_layer, b_post_layer, b_moe_layer, b_tl, b_epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(pair_kernel, launch)
    return launch
