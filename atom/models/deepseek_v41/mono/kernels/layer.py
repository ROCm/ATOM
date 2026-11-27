# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The layer kernel: K1 (``attn_pre``) and K2 (``layer_post``) of one layer in
one launch -- the attention seam to the MoE's reduced output.

K2's stages follow K1's on every CTA, so the attention's loads and the output
projections' weights can go out while K1's stages still run elsewhere. What a
K1 launch handed K2 by its end is handed over by flag here, polled at K1's
tag, each written to memory no XCD caches (the scratch, and the runner's
``uncached`` hand-off buffers):

    SDONE  a slice task's R_new columns (``res_out`` -> K2a's slice)
    GDONE  a token's gates (``post_out`` / ``comb_out`` / ``pre_out`` -> K2a's slice)
    KRDY   a token's window row, also into ``kvrow`` laid out as the pool's
           planes (``dims.kvrow_bytes``: BF16 rows, or an FP8 pool's E4M3 rows
           and their scale rows): the attention reads this step's rows there,
           not from the pool, whose lines another XCD's L2 may hold from an
           earlier step
    QP     ``q`` itself as tagged pairs (wq_b -> the scores: one hop, no load after)
    KVT    in place of KRDY and ``kvrow`` where ``window_rows_tagged``: a token's
           window row as tagged pairs, a data word each, its group's E8M0 code
           in the tag's high half, polled by the scores and the merge
           themselves (no drain, no flag, no load after)
    IQDONE an index layer's (token, head) query, scale and head weight written
           (``index_query``'s quantization -> K2a's scorer: ``iq`` / ``iq_scale``
           / ``iw``, the runner's uncached buffers)

Each part keeps its mailbox tag; K2a's regions lie past K1's (the seam regions
lin / pmix / x8 shared as in a K2 launch: their K1 readers are done before K2a
writes them), the flags past both. The LDS is one union of the parts' members
(a barrier hands it over).

Every layer with nothing between its K1 and its K2 in the original path (no
compressor, no candidate table), its seam folding the owed post or taking a
settled residual (layer 0, an Engram layer): the consumer layers, and the
index-only layers (K1's index query, K2a's scorer). Token parallel (``tok_par``) as K1's and K2's: the seams on a
rank's own tokens, the other ranks' window rows into ``kvrow`` and their KRDY
from KG (``attn_pre.ring_row_from``), the normed rows gathered for the MoE
(``layer_post.gather_x``); the residual is then not column-sharded.
"""

from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import compress as cp
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
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
    lds_struct,
    lds_union,
    run_attn_pre,
    shard_seam,
)
from atom.models.deepseek_v41.mono.kernels.debug import mailbox
from atom.models.deepseek_v41.mono.kernels.dims import HIDDEN, KV_DIM, Dims
from atom.models.deepseek_v41.mono.kernels.index_query import IQ_GROUPS, iq_in_place
from atom.models.deepseek_v41.mono.kernels.layer_post import (
    LayerPostBuild,
    gather_x,
    publish_x,
    publish_x_tok_par,
    seam_dot_norm,
)
from atom.models.deepseek_v41.mono.kernels.moe import (
    moe_args,
    moe_context,
    moe_smem,
    run_moe,
)
from atom.models.deepseek_v41.mono.kernels.moe_shape import route_shape
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    kernel_symbol,
    launcher,
    rsrc,
    traced,
    uniform,
    warm_code,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp_begin, stamp_flush
from atom.mono.device.sync import fresh_s32, fresh_s64
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import pair_layout
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

# ``timeline`` stamps in the order the stages run: K1's, then K2's
TL_STAGES = k1.TL_STAGES + k2.TL_STAGES
TL_POINTS = 1 + len(TL_STAGES)
# K1's hand-over flags (module docstring), in its scratch past K2's regions
FLAGS = ("sdone", "gdone", "krdy", "qp", "iqdone")
# the window row as tagged pairs (KVT, past the flags): kv_norm puts a
# row's data words, its group's E8M0 code in each tag's high half
# (``Mailbox.put_coded``), and the score's idle waves and the merge poll the
# words themselves -- no ``kvrow`` copy, no publish drain, no KRDY. TP8, an FP8
# pool, no indexer / compressor / token parallel / DSpark block, at most
# KVT_MAX_TOKENS rows (the score's poll a lane: two quads a pair of rows); any
# other build keeps KRDY
KVT_MAX_TOKENS = 8


def window_rows_tagged(key) -> bool:
    """Whether ``key`` (a ``LayerBuild``) hands the window rows over by KVT."""
    pre, post = key.pre, key.post
    return (
        pre.tp == 8
        and pre.kv_fp8
        and not pre.feeds_index
        and not pre.tok_par
        and not key.compress_ratio
        and pre.tokens <= KVT_MAX_TOKENS
        and post.attn_sel
        and not post.block_rows
    )


# A layer's code meets its I-cache misses in an L2 a whole layer of weights has
# streamed through since its last launch: an index layer's (296 .. 308 KB at
# s6, once a step) and, as measured, an ordinary layer's (~100 KB, launched
# every layer) too. At entry one wave of each of the last 8 CTAs (one an XCD)
# reads it into its XCD's L2 (``warm_code``, from its pc up to CODE_WARM_SPAN
# bytes, clamped at the code's end) and waits for it there: those CTAs hold no
# slice, gate (block) or seam norm task, their first wait the normed rows'
# (wqkv_a, a compressor's projection). Every layer build; 0: off
CODE_WARM_SPAN = 320 * 1024
CODE_WARM_CTA0 = BLOCKS - 8
CODE_WARM_WAVE = WAVES - 1


@traced
def warm_layer_code(on, tid, bid):
    if const_expr(on):  # noqa: SIM102
        if (bid >= CODE_WARM_CTA0) & (uniform(tid // 64) == CODE_WARM_WAVE):
            warm_code(tid % 64, CODE_WARM_SPAN)


@dataclass(frozen=True)
class LayerBuild:
    pre: k1.AttnPreBuild
    post: LayerPostBuild
    # a compressing layer's compressor (``kernels.compress``) in the launch,
    # its ratio (0: none) and norms' eps -- scalars, for the JIT key; its new
    # main pool rows and index keys handed to K2a by CRDY
    compress_ratio: int = 0
    compress_eps: float = 0.0
    compress_k_eps: float = 0.0

    @property
    def q_apart(self) -> bool:
        """K1's q path on CTAs of its own, apart from the compressor's
        projection (``k1.APART_PROJ0``): a
        compressing layer past one token tile."""
        return bool(self.compress_ratio) and len(k1.token_tiles(self.pre.tokens)) > 1

    @property
    def compress(self) -> cp.CompressBuild | None:
        if not self.compress_ratio:
            return None
        pre = self.pre
        comp = cp.CompressBuild(
            rows=pre.tokens, ratio=self.compress_ratio, eps=self.compress_eps,
            k_eps=self.compress_k_eps, kv_fp8=pre.kv_fp8, timeline=pre.timeline,
        )  # fmt: skip
        if not self.q_apart:
            return comp
        # the projection on ``k1.APART_PROJ_CTAS`` CTAs whatever the ratio
        groups = comp.proj_ctas // cp.QUARTERS
        return replace(comp, quarters=k1.APART_PROJ_CTAS // groups)

    def __post_init__(self):
        pre, post = self.pre, self.post
        if self.compress_ratio:
            assert pre.feeds_index, "a compressing layer feeds its indexer"
            assert not pre.tok_par, "the compressor reads every row's normed"
        assert pre.tokens == post.tokens and pre.tp == post.tp
        assert pre.timeline == post.timeline and pre.diag_off == post.diag_off
        assert pre.feeds_index == (
            post.index_bound_max > 0
        ), "K1's index query for K2a's"
        assert pre.tok_par == post.tok_par and post.dp_ranks == 1
        assert pre.kv_fp8 == post.kv_fp8, "one pool's rows"
        assert pre.shard == post.shard, "both seams sharded or neither"


# K1's arguments (prefixed: K2 has its own residual, gates, tables) less its
# ``ibound`` (the layer's selection bounds: K2a's), K2's, the step rows'
# window rows, the shared tail with a tag a part
_K1 = tuple(
    f"k1_{n}" for n in k1.ABI.names[: k1.ABI.names.index("scratch")] if n != "ibound"
)
_K2 = k2.ABI.names[: k2.ABI.names.index("scratch")]
# the compressor's (prefixed; its input is K1's ``normed``, its scratch, tag
# and timeline the launch's): 0 where a build has none
COMPRESS_ARGS = tuple(
    f"c_{n}" for n in cp.ABI.names if n not in ("x", "scratch", "layer", "tl")
)
ABI = KernelAbi(
    (*_K1, *_K2, "kvrow", *COMPRESS_ARGS, "scratch", "numa_word", "sym", "peers",
     "rank", "layer", "post_layer", "moe_layer", "tl")
)  # fmt: skip


def scratch_layouts(key: LayerBuild) -> tuple[dict, dict]:
    """(K1's regions and the hand-over flags, K2's). K2's K2a seam regions lie
    at K1's offsets, its other regions past K1's; the flags past both. Two
    tables, not one: K1's ``x8`` / ``x8s`` and K2b's are different regions of
    one name."""
    s = key.pre.tokens
    post = k2.scratch_layout(key.post)
    pre = dict(k1.scratch_layout(s))
    end = max(o + n for o, n in (*pre.values(), *post.values()))
    heads = Dims(key.pre.tp).heads
    # IQDONE: a flag a (token, head), or a (token, head, group) where the
    # query is quantized in place (``iq_in_place``)
    iq_flags = IQ_GROUPS if iq_in_place(s, key.post.index_fp4) else 1
    sizes = (k1.SLICES, s, s, s * heads * k1.HEAD_DIM // 2, s * ip.HEADS * iq_flags)
    pairs = dict(zip(FLAGS, sizes, strict=True))
    if window_rows_tagged(key):
        # KVT: a step row's data words, a pair each
        pairs["kvt"] = s * (KV_DIM // 4)
    if key.compress is not None:
        # CRDY a step row (the one completing a compressed entry), then the
        # compressor's own hand-offs (``compress.scratch_layout``'s regions)
        pairs["crdy"] = s
        for name, n in cp.scratch_pairs(s):
            assert name not in pre and name not in post
            pairs[name] = n
    pre.update(pair_layout(pairs.items(), start=end))
    return pre, post


def compress_smem(comp: cp.CompressBuild):
    """The compressor's LDS, a member of the layer's union (its stages run
    between barriers after K1's)."""
    return lds_struct("CompressLds", {"cpw": fx.Array[fx.Int32, comp.lds_words, 16]})


def compress_context(comp, c1, cargs, lds, mb, normed, kvrow, tid, bid, s, proj0=0):
    """The compressor's context (``build_compress``'s) inside the layer kernel:
    its input K1's ``normed`` (awaited by the rows' x8f, read at device scope),
    its regions K1's layout's; its hand-off to K2a this step's new entries in
    ``kvrow`` past the window rows (``dims.kvrow_bytes``) and CRDY. Its tasks
    from CTA ``proj0`` (the projection's first; the plan rows' past it)."""
    lane, wave = tid % 64, tid // 64

    def x_wait():
        # every lane a row's flag (no traced branch in a closure)
        mb.poll([(c1["x8f"], fx.min(lane, fx.Int32(s - 1)), 1)])
        gpu.barrier()

    return {
        "key": comp, "tid": tid, "lane": lane, "wave": wave,
        "t": (bid + BLOCKS - proj0) % BLOCKS if proj0 else bid,
        "lds": lds, "ldsf": fx.recast_iter(fx.Float32, lds), "args": cargs,
        "r_w": rsrc(cargs["w"], comp.cols * HIDDEN * 2),
        "r_x": rsrc(normed, comp.rows * HIDDEN * 2),
        "r_parts": rsrc(cargs["parts"], cp.parts_bytes(comp.rows)),
        "mb": mb, "cpf": c1["cpf"], "css": c1["css"], "ckp": c1["ckp"],
        "crdy": c1["crdy"], "kvrow": kvrow,
        "x_wait": x_wait, "x_cm": CM_DEV, "stamp": lambda k: None,
    }  # fmt: skip


def compress_stage(stage, cc):
    """A compressor stage inside the layer kernel, between barriers (its LDS
    is a member of the layer's union)."""
    gpu.barrier()
    stage(cc)
    gpu.barrier()


def scratch_bytes(key: LayerBuild) -> int:
    return max(o + n for layout in scratch_layouts(key) for o, n in layout.values())


def build_layer(key: LayerBuild):
    pre, post = key.pre, key.post
    s, timeline = pre.tokens, pre.timeline
    key_a, key_b = post.halves()
    rs = route_shape(key_b)
    layout_1, layout = scratch_layouts(key)
    layout_a = {n: layout[n] for n in k2a.scratch_layout(key_a)}
    layout_b = {n: layout[n] for n in (*k2b.scratch_layout(key_b), "xrdy")}
    keyed = key_tuple(key, SOURCES)
    members_1 = attn_pre_smem(pre)
    members_a = attn_post_smem(
        s, False, post.index_bound_max, Dims(post.tp), post.kv_fp8
    )
    members = dict(members_1)
    members.update(members_a)
    members["moe"] = moe_smem(s, rs, False)
    comp = key.compress
    if comp is not None:
        members["cp"] = compress_smem(comp)
    Parts = lds_union("LayerLds", members)
    warm = CODE_WARM_SPAN > 0
    if warm:
        # the seam norm's CTAs end before WQKV0 or at NORM0 + its rows
        norm_rows = s // pre.tp if pre.tok_par else s
        assert max(k1.NORM0 + norm_rows, k1.WQKV0) <= CODE_WARM_CTA0

    @fx.struct
    class Stamps:
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]

    name = kernel_symbol(
        "v41_layer", s=s, a=int(pre.aux), tl=int(timeline),
        ix=int(post.index_bound_max > 0), i4=int(post.index_fp4),
        im=int(post.index_mfma), kv8=int(post.kv_fp8), tpar=int(pre.tok_par),
        sh=int(pre.shard), e=int(pre.embed_in), c=key.compress_ratio,
    )  # fmt: skip

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def layer_kernel(
        k1_res_in: Int64, k1_pend: Int64, k1_post_in: Int64, k1_comb_in: Int64,
        k1_pre_in: Int64, k1_hc_fn: Int64, k1_hc_scale: Int64, k1_hc_base: Int64,
        k1_attn_w: Int64, k1_wqkv: Int64, k1_wqkv_s: Int64, k1_qn_w: Int64,
        k1_kvn_w: Int64, k1_wqb: Int64, k1_wqb_s: Int64, k1_cos: Int64, k1_sin: Int64,
        k1_pos: Int64, k1_ring: Int64, k1_ring_scale: Int64, k1_ring_rows: Int64,
        k1_ring_off: Int32,
        k1_res_out: Int64, k1_post_out: Int64, k1_comb_out: Int64, k1_pre_out: Int64,
        k1_q_out: Int64, k1_normed: Int64, k1_qr_out: Int64, k1_qrs_out: Int64,
        k1_aux: Int64, k1_iq_w: Int64, k1_iq_ws: Int64, k1_iw_w: Int64,
        k1_iq_out: Int64, k1_iqs_out: Int64, k1_iw_out: Int64,
        iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64,
        pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64,
        lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32,
        ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, ixsel: Int64,
        sbound: Int64,
        q: Int64, pool: Int64, pool_scale: Int64, window: Int64, window_scale: Int64,
        sel: Int64, topk: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32,
        ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64,
        pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        normed: Int64,
        gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64,
        w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        queue_words: Int64, kvrow: Int64,
        c_w: Int64, c_parts: Int64, c_cplan: Int64, c_ncap: Int32, c_wplan: Int64,
        c_nwcap: Int32, c_slots: Int64, c_kv_state: Int64, c_score_state: Int64,
        c_state_stride: Int64, c_ring: Int32, c_ape: Int64, c_norm_w: Int64,
        c_cos: Int64, c_sin: Int64, c_pages: Int64, c_page_stride: Int64,
        c_page_scale: Int64, c_scale_stride: Int64, c_tables: Int64,
        c_table_stride: Int32, c_per_page: Int32, c_wk: Int64, c_k_norm_w: Int64,
        c_ivals: Int64, c_iscales: Int64, c_latent: Int64, c_keys: Int64,
        scratch: Int64, numa_word: Int64, sym: Int64, peers: Int64, rank: Int32,
        layer: Int32, post_layer: Int32, moe_layer: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        warm_layer_code(warm, tid, bid)
        # the scratch regions' base a value of its own (``sync.fresh_s64``), not
        # a part of the merged kernarg load a compressing build reloads whole
        scratch = fresh_s64(scratch)
        # the kernarg tail's other late-read scalars too: no merged 8- / 4-dword
        # kernarg tuple held across the kernel
        sym, peers, numa_word = fresh_s64(sym), fresh_s64(peers), fresh_s64(numa_word)
        rank, layer = fresh_s32(rank), fresh_s32(layer)
        post_layer, moe_layer = fresh_s32(post_layer), fresh_s32(moe_layer)
        alloc = fx.SharedAllocator()
        tls = alloc.allocate(Stamps).peek().tls.ptr
        smem = alloc.allocate(Parts)
        lds = {n: getattr(smem, n).peek() for n in members}
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
        if const_expr(prefetch):
            args_1 = k1_args()
            entry_pre = k1.slice_entry_prefetch(args_1, s, pre.tp, bid, rank)
        bases = peer_bases(peers, rs.dims.ranks)
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
        args_b = moe_args(
            normed, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out,
            queue_words,
        )  # fmt: skip
        args_1["numa_word"] = args_a["numa_word"] = args_b["numa_word"] = numa_word
        mb_1 = mailbox(layer, scratch, pre.diag_off, layout_1)
        c1, co1 = attn_pre_context(
            pre, lds, mb_1, layout_1, scratch, args_1, rank, sym, peers
        )
        # the own-rows view too (token parallel: a copy of ``c1``)
        c1["fused"] = co1["fused"] = True
        if const_expr(comp is not None):
            # the compressor reads K1's normed rows in this launch
            c1["normed_cm"] = co1["normed_cm"] = CM_DEV
        ca = attn_post_context(
            key_a, s, lds, mailbox(post_layer, scratch, post.diag_off, layout),
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
            s, rs, lds["moe"], mailbox(moe_layer, scratch, post.diag_off, layout),
            bases, layout_b, scratch, args_b, rank, sym, key_b.numa,
        )  # fmt: skip
        cb["x_cm"], cb["x_ready"] = CM_DEV, True
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
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)
        _ = keyed

    @launcher(name)
    def launch(
        k1_res_in: Int64, k1_pend: Int64, k1_post_in: Int64, k1_comb_in: Int64,
        k1_pre_in: Int64, k1_hc_fn: Int64, k1_hc_scale: Int64, k1_hc_base: Int64,
        k1_attn_w: Int64, k1_wqkv: Int64, k1_wqkv_s: Int64, k1_qn_w: Int64,
        k1_kvn_w: Int64, k1_wqb: Int64, k1_wqb_s: Int64, k1_cos: Int64, k1_sin: Int64,
        k1_pos: Int64, k1_ring: Int64, k1_ring_scale: Int64, k1_ring_rows: Int64,
        k1_ring_off: Int32,
        k1_res_out: Int64, k1_post_out: Int64, k1_comb_out: Int64, k1_pre_out: Int64,
        k1_q_out: Int64, k1_normed: Int64, k1_qr_out: Int64, k1_qrs_out: Int64,
        k1_aux: Int64, k1_iq_w: Int64, k1_iq_ws: Int64, k1_iw_w: Int64,
        k1_iq_out: Int64, k1_iqs_out: Int64, k1_iw_out: Int64,
        iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64,
        pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64,
        lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32,
        ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, ixsel: Int64,
        sbound: Int64,
        q: Int64, pool: Int64, pool_scale: Int64, window: Int64, window_scale: Int64,
        sel: Int64, topk: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32,
        ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64,
        pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        normed: Int64,
        gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64,
        w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        queue_words: Int64, kvrow: Int64,
        c_w: Int64, c_parts: Int64, c_cplan: Int64, c_ncap: Int32, c_wplan: Int64,
        c_nwcap: Int32, c_slots: Int64, c_kv_state: Int64, c_score_state: Int64,
        c_state_stride: Int64, c_ring: Int32, c_ape: Int64, c_norm_w: Int64,
        c_cos: Int64, c_sin: Int64, c_pages: Int64, c_page_stride: Int64,
        c_page_scale: Int64, c_scale_stride: Int64, c_tables: Int64,
        c_table_stride: Int32, c_per_page: Int32, c_wk: Int64, c_k_norm_w: Int64,
        c_ivals: Int64, c_iscales: Int64, c_latent: Int64, c_keys: Int64,
        scratch: Int64, numa_word: Int64, sym: Int64, peers: Int64, rank: Int32,
        layer: Int32, post_layer: Int32, moe_layer: Int32, tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        layer_kernel(
            k1_res_in, k1_pend, k1_post_in, k1_comb_in, k1_pre_in, k1_hc_fn,
            k1_hc_scale, k1_hc_base, k1_attn_w, k1_wqkv, k1_wqkv_s, k1_qn_w, k1_kvn_w,
            k1_wqb, k1_wqb_s, k1_cos, k1_sin, k1_pos, k1_ring, k1_ring_scale,
            k1_ring_rows, k1_ring_off, k1_res_out, k1_post_out, k1_comb_out, k1_pre_out, k1_q_out,
            k1_normed, k1_qr_out, k1_qrs_out, k1_aux, k1_iq_w, k1_iq_ws, k1_iw_w,
            k1_iq_out, k1_iqs_out, k1_iw_out,
            iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len,
            ibatch, ishift, ibound, ilog, lstride,
            isel, icand, ilift, iprod, ibmax, bstride, icout, irow, ixsel, sbound,
            q, pool, pool_scale, window, window_scale, sel, topk, kmeta, table,
            rows_per_page, page_rows, main_off, ring_off, ring_slots, sink, qk_scale,
            pos, cos, sin, woa, woa_s, wob, wob_s, res_in, post_in, comb_in, pre_in,
            hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out, pre_out,
            normed, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out, queue_words,
            kvrow,
            c_w, c_parts, c_cplan, c_ncap, c_wplan, c_nwcap, c_slots, c_kv_state,
            c_score_state, c_state_stride, c_ring, c_ape, c_norm_w, c_cos, c_sin,
            c_pages, c_page_stride, c_page_scale, c_scale_stride, c_tables,
            c_table_stride, c_per_page, c_wk, c_k_norm_w, c_ivals, c_iscales,
            c_latent, c_keys,
            scratch, numa_word, sym, peers, rank, layer, post_layer, moe_layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(layer_kernel, launch)
    return launch
