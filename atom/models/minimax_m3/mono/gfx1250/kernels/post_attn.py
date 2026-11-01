# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K4 ``mono_post`` (``kernels.post_attn`` on wave32 and WMMA).

The program -- stages, task placement, mailboxes, peer regions and the K4 ABI --
is the gfx950 kernel's; the stages are the gfx1250 modules under ``stages`` and
the LDS partial buffers are sized for 16 waves' WMMA accumulators.
"""

import os
from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    MAX_TOKENS,
    MOE_SLOTS,
    N_ROUTED,
    ONE_INDEX_HEAD,
    TOP_K,
    TOPK_BLOCKS,
    IndexHeads,
)
from atom.models.minimax_m3.mono.gfx1250.kernels.pre_attn import K1_ARGS_G1250, emit_k1
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.attention import (
    merge_defs,
    o_defs,
    split_defs,
)
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.moe_common import (
    DN_GW,
    moe_common_defs,
)
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.moe_narrow import (
    moe_narrow_defs,
)
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.moe_wide import moe_wide_defs
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.router import router_defs
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.select import select_defs
from atom.models.minimax_m3.mono.gfx1250.layout import (
    ACC_WORDS,
    MID_ROW,
    O_K,
    O_ROWS,
    PV_HALVES,
    SCRATCH,
    SPLIT_KEYS,
    UG_PER_SLOT,
    XN8_ROW,
    XSC_ROW,
    H,
    mailbox_regions,
    pool_words,
    stage_bases,
    sym_layout,
    ug_ctas_per_token,
    ug_tasks_of,
    wide_moe,
)
from atom.models.minimax_m3.mono.kernels.index_score import index_scale_log2e, step_rows
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    SCRATCH_HDONE as K1_SCRATCH_HDONE,
)
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    SCRATCH_RDONE as K1_SCRATCH_RDONE,
)
from atom.models.minimax_m3.mono.kernels.stages.ffn_reduce import ffn_reduce_defs
from atom.models.minimax_m3.mono.kernels.stages.shared import K1_STAMPS, TL_POINTS
from atom.models.minimax_m3.mono.sources import SOURCES
from atom.mono.device.gfx1250 import (
    BLOCKS,
    CM_DEV,
    THREADS,
    WAVE,
    WAVES,
    async_to_lds,
    butterfly,
    lds_addr,
    wait_async,
    xshfl,
)
from atom.mono.device.gfx1250 import block_max as block_max_of
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    kernel_symbol,
    launcher,
    ld_dev,
    load_ptr64,
    memrealtime,
    rsrc,
)
from atom.mono.device.sync import Mailbox, preg, shift, sreg
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.execution import first_task
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.debug import region_ids

__all__ = ["K4_ABI", "TL_POINTS", "TL_RECORD", "build_post_attn_kernel"]

_CURRENT_STREAM = fx.Stream(None)
TL_END = 16  # timeline point: the CTA's last stage done (free on gfx950's map)
# a CTA's timeline record: point k of wave w at k WAVES + w, each wave stamping
# its own (a tid == 0 stamp is a branch: a block split at every point, which
# moved the build's spills and schedule away from the real kernel's)
TL_RECORD = TL_POINTS * WAVES
assert TL_RECORD == THREADS
# gfx950 makes both all-reduces RS + AG from 12 tokens; on gfx1250 the one-hop
# push measured faster there (bs 3 x 4 verify rows: 9.29 -> 8.47 ms a step)
AG_RS_FROM_G1250 = MAX_TOKENS + 1


@dataclass(frozen=True)
class K4Build:
    npes: int = field(metadata={"sym": "tp"})
    tokens: int = field(metadata={"sym": "s"})
    init_blocks: int = field(metadata={"sym": "ib"})
    local_blocks: int = field(metadata={"sym": "lb"})
    index_heads: int = field(metadata={"sym": "ih"})
    index_own: int = field(metadata={"sym": "io"})
    index_topk: bool = field(metadata={"sym": "it"})
    x8_in: bool = field(metadata={"sym": "xi"})
    timeline: bool = field(metadata={"sym": "tl"})
    fuse_k1: bool
    sm_scale: float
    eps: float
    route_scale: float
    shared_weight: float
    swiglu_limit: float
    debug: bool
    skip: str  # debug: stages replaced by no-ops (ATOM_MONO_G1250_SKIP)
    tl_points: str  # debug: the timeline's points stamped (ATOM_MONO_TL_POINTS)
    ag_rs_from: int  # tokens from which the all-reduces are RS + AG


def build_post_attn_kernel(
    npes: int,
    sm_scale: float,
    eps: float,
    route_scale: float,
    shared_weight: float,
    swiglu_limit: float,
    init_blocks: int,
    local_blocks: int,
    tokens: int = 1,
    timeline: bool = False,
    fuse_k1: bool = False,
    heads: IndexHeads = ONE_INDEX_HEAD,
    debug: bool = False,
    index_topk: bool = True,
    x8_in: bool = False,
):
    """``kernels.post_attn.build_post_attn_kernel`` on gfx1250. ``x8_in``: K1 takes
    the MXFP8 rows the previous layer's FFN finish made (every sparse layer but
    the first); every layer's FFN finish makes them for the next."""
    assert not x8_in or fuse_k1
    assert 1 <= tokens <= MAX_TOKENS
    key = K4Build(
        npes=npes, tokens=tokens, init_blocks=init_blocks, local_blocks=local_blocks,
        index_heads=heads.count, index_own=heads.own, index_topk=index_topk,
        x8_in=x8_in, timeline=timeline, fuse_k1=fuse_k1, sm_scale=sm_scale, eps=eps, route_scale=route_scale,
        shared_weight=shared_weight, swiglu_limit=swiglu_limit, debug=debug,
        skip=os.environ.get("ATOM_MONO_G1250_SKIP", ""),
        tl_points=os.environ.get("ATOM_MONO_TL_POINTS", ""),
        ag_rs_from=int(os.environ.get("ATOM_MONO_G1250_AGRS_FROM", AG_RS_FROM_G1250)),
    )  # fmt: skip
    SKIP = set(key.skip.split(",")) - {""}
    # a stamp stalls its wave on the clock read: fewer points, a timeline nearer
    # the real kernel's (0 and TL_END always)
    TL_SET = {int(p) for p in key.tl_points.split(",") if p} | {0, TL_END}
    TL_ALL = not key.tl_points
    build_key = key_tuple(key, SOURCES)
    REGION_IDS = region_ids(
        d.name for d in mailbox_regions(tokens, heads.count, fuse_k1, index_topk)
    )
    W = npes
    G = BLOCKS
    BASE = stage_bases(tokens)
    SY = sym_layout(W)
    CM_K1 = 16 if fuse_k1 else 0  # SCOPE_DEV: K1's outputs are read device-coherent
    WIDE = wide_moe(tokens)
    POOL = pool_words(tokens)
    U_MAX = TOP_K * tokens + 1
    # narrow: the split's PV halves, then the MoE's up / down partials, one
    # accumulator set a token plus the shared expert's
    OPART = 1 if WIDE else max(PV_HALVES * O_K, (tokens + 1) * ACC_WORDS)
    # wide: the up / gate streams' partials, an accumulator set a round (block)
    UGR = (U_MAX * UG_PER_SLOT // 2 + BLOCKS - 1) // BLOCKS
    # and their LDS rings (a wave's chunks in flight), as deep as LDS allows
    UG_RING = 7 if tokens <= 12 else 6
    UG_SLOT_WORDS = (WAVE * 32 + WAVE * 4) // 4

    @fx.struct
    class Smem:
        x: fx.Array[fx.Int32, POOL, 16]
        red: fx.Array[fx.Float32, ACC_WORDS, 16]
        p8: fx.Array[fx.Int32, H * SPLIT_KEYS // 4, 16]
        q8: fx.Array[fx.Int32, O_K // 4, 16]
        opart: fx.Array[fx.Float32, OPART, 16]
        upart: fx.Array[fx.Float32, 2 * ACC_WORDS if WIDE else 1, 16]
        ugring: fx.Array[fx.Int32, WAVES * UG_RING * UG_SLOT_WORDS if WIDE else 1, 16]
        uexp: fx.Array[fx.Int32, U_MAX + 2, 16]
        umask: fx.Array[fx.Int32, U_MAX + 2 + WAVES, 16]
        urow: fx.Array[fx.Int32, (U_MAX + 1) * 16 if WIDE else 1, 16]
        uwt: fx.Array[fx.Float32, (U_MAX + 1) * 16 if WIDE else 1, 16]
        hst: fx.Array[fx.Float32, 3 * WAVES * H + WAVES, 16]
        misc: fx.Array[fx.Float32, O_ROWS * tokens, 16]
        # (wide: K1's codes live in the up / gate rings, not in use before the MoE)
        k1xsc: fx.Array[
            fx.Int32, tokens * (HIDDEN // 128) if x8_in and not WIDE else 1, 16
        ]
        k1rst: fx.Array[fx.Float32, 16, 16]
        route: fx.Array[fx.Int32, 8 * tokens, 16]
        rwt: fx.Array[fx.Float32, 8 * tokens, 16]
        blk: fx.Array[fx.Int32, 32, 16]
        tls: fx.Array[fx.Int64, TL_RECORD if timeline else 1, 16]
        qs: fx.Array[fx.Float32, HEAD_DIM // 2 * tokens if heads.count == 1 else 1, 16]

    kernel_name = kernel_symbol(
        "minimax_m3_mono_layer_g1250" if fuse_k1 else "minimax_m3_post_attn_g1250",
        **symbol_params(key),
    )
    ih_count, ih_own = heads.count, heads.own
    QS_CP = tokens * HIDDEN // 4
    assert not (fuse_k1 and heads.count > 1) or (
        QS_CP + HEAD_DIM // 2 * tokens * heads.count <= POOL
    )
    assert POOL >= tokens * HIDDEN // 4  # K1's x8 rows

    _bc = dict(locals())

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def post_attn_kernel(
        h_in: Int64,
        q: Int64,
        block_table: Int64,
        seq_lens: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        w_o: Int64,
        s_o: Int64,
        g_post: Int64,
        w_gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        h_mid: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        bt_width: Int32,
        q_len: Int32,
        tl: Int64,
        k1_args: Int64,
        positions: Int64,
        slot_mapping: Int64,
        res: Int64,
        batch_ids: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % WAVE
        wave = tid // WAVE
        g4 = lane // 16
        l16 = lane % 16
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.x.ptr
        red = lds.red.ptr
        p8 = lds.p8.ptr
        q8 = lds.q8.ptr
        fpool = fx.recast_iter(fx.Float32, xs)
        opart = fpool if WIDE else lds.opart.ptr
        upart = lds.upart.ptr
        ugring = lds.ugring.ptr
        uexp = lds.uexp.ptr
        umask = lds.umask.ptr
        urow = lds.urow.ptr
        uwt = lds.uwt.ptr
        hst = lds.hst.ptr
        misc = lds.misc.ptr
        route = lds.route.ptr
        rwt = lds.rwt.ptr
        keys = xs
        blk = lds.blk.ptr
        tls = lds.tls.ptr
        v4f = fx.Vector.make_type(4, fx.Float32)
        v2i = fx.Vector.make_type(2, fx.Int32)

        diag = scratch + fx.Int64(SCRATCH["diag"]) if debug else None
        mbox = Mailbox(layer, diag, REGION_IDS, gfx1250=True, steady=diag is not None)
        mb_put = mbox.put
        mb_put_bf = mbox.put_bf
        mb_put_words = mbox.put_words
        mb_poll = mbox.poll

        def mb(name):
            return scratch + fx.Int64(SCRATCH[name])

        def mr(name):
            return sreg(scratch, SCRATCH[name], name)

        def acquire():
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
            )

        def scores_copy(c, k=0):
            return shift(
                mr("scores"), (fx.Int64(c) * MAX_TOKENS + fx.Int64(k)) * (N_ROUTED * 16)
            )

        def route_key(v, e):
            b = v.bitcast(fx.Int32)
            b = (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)
            return (b & fx.Int32(-128)) | (127 - e)

        peer_dst = load_ptr64(peers, fx.min(wave, W - 1))
        AG_RS = tokens >= key.ag_rs_from

        def peer_addr(w):
            return load_ptr64(peers, w)

        def start(name):
            return first_task(bid, BASE[name])

        def stamp(k):
            """``timeline``: this wave passing point k (steady clock)."""
            if const_expr(timeline and (TL_ALL or k in TL_SET)):
                fx.ptr_store(memrealtime(True), tls + (k * WAVES + wave))

        def block_max(v):
            return block_max_of(v, lane, wave, red)

        def push_rows(region, row0, n_rows, only=None):
            """``post_attn.push_rows``, a pair a lane of wave w -> peer w."""
            half = n_rows // 2
            to = (wave < W) if only is None else (wave == only)
            for c in range_constexpr((half * tokens + WAVE - 1) // WAVE):
                i = lane + WAVE * c
                if to & (i < half * tokens):
                    mb_put_bf(
                        preg(peer_dst, SY[region], region),
                        (rank * MAX_TOKENS + i // half) * HIDDEN
                        + row0
                        + (i % half) * 2,
                        [fx.ptr_load(misc + i * 2), fx.ptr_load(misc + (i * 2 + 1))],
                    )

        if const_expr(timeline):
            fx.ptr_store(fx.Int64(0), tls + tid)
        stamp(0)

        if const_expr(fuse_k1):

            def k1_stamp(k):
                """K1's points 1-4 and 6 on K4's free stamps 27-31."""
                if const_expr(k in K1_STAMPS):
                    stamp(K1_STAMPS[k])

            (
                ar_in, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, index_cache,
                iq_out, scratch1, x8_prev, x8s_prev, ss_prev, h_next, x8_next,
                x8s_next, ss_next, g_next,
            ) = [load_ptr64(k1_args, i) for i in range(len(K1_ARGS_G1250))]  # fmt: skip
            qs_k1 = lds.qs.ptr
            if const_expr(ih_count > 1):
                qs_k1 = fpool + QS_CP
            emit_k1(
                tid, bid, lane, wave, xs, red, qs_k1, k1_stamp, layer,
                eps, index_scale_log2e(sm_scale), tokens, q_len, init_blocks,
                local_blocks, IndexHeads(ih_count, ih_own),
                (
                    ar_in, res, h_in, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin,
                    positions, slot_mapping, k_cache, v_cache, k_scale, v_scale,
                    index_cache, q, iq_out, block_table, seq_lens, mb("iscore"), scratch1,
                ),
                bt_width,
                True,
                diag,
                REGION_IDS,
                index_topk,
                x8_in=(
                    (
                        x8_prev,
                        x8s_prev,
                        ss_prev,
                        lds.ugring.ptr if WIDE else lds.k1xsc.ptr,
                        lds.k1rst.ptr,
                    )
                    if x8_in
                    else None
                ),
            )  # fmt: skip
            gpu.barrier()
            k1_mbox = Mailbox(
                layer, diag, REGION_IDS, gfx1250=True, steady=diag is not None
            )
            k1_poll = k1_mbox.poll
            hdone_mb = sreg(scratch1, K1_SCRATCH_HDONE, "k1.hdone")
            rdone_mb = sreg(scratch1, K1_SCRATCH_RDONE, "k1.rdone")
        else:
            k1_poll = hdone_mb = rdone_mb = None

        seq_lens_k, n_blks, long_rows = step_rows(
            seq_lens, tokens, q_len, IndexHeads(ih_count, ih_own)
        )
        scored = [nb > TOPK_BLOCKS for nb in n_blks]

        k4ctx = dict(_bc)
        k4ctx.update(locals())
        k4ctx.update(select_defs(k4ctx))
        stage_select_long = k4ctx["stage_select_long"]

        k4ctx.update(locals())
        k4ctx.update(split_defs(k4ctx))
        stage_split = k4ctx["stage_split"]

        if const_expr(index_topk):
            enter_stage("select_long")
            stage_select_long()
        enter_stage("split")
        if const_expr("split" not in SKIP):
            stage_split()
        stamp(1)

        k4ctx.update(locals())
        k4ctx.update(merge_defs(k4ctx))
        stage_merge = k4ctx["stage_merge"]
        enter_stage("merge")
        if const_expr("merge" not in SKIP):
            stage_merge()

        k4ctx.update(locals())
        k4ctx.update(o_defs(k4ctx))
        stage_o = k4ctx["stage_o"]
        enter_stage("o")
        if const_expr("o" not in SKIP):
            stage_o()
        stamp(4)

        XN8_WORDS = HIDDEN // 4
        XSC_WORDS = HIDDEN // 32
        XSC_LDS = tokens * XN8_ROW
        MID16_WORDS = MOE_SLOTS * INTER // 2
        MID8_WORDS = MOE_SLOTS * INTER // 4
        MID8_LDS = tokens * MID16_WORDS
        MIDSC_WORDS = MOE_SLOTS * INTER // 32
        MIDSC_LDS = MID8_LDS + tokens * MID8_WORDS

        k4ctx.update(locals())
        k4ctx.update(router_defs(k4ctx))
        stage_router = k4ctx["stage_router"]
        enter_stage("router")
        if const_expr("router" not in SKIP):
            stage_router()
        stamp(5)

        UG_KC = HIDDEN // 128
        UG_CPW = UG_KC // WAVES
        DN_KC = INTER // 128
        DN_UNITS = TOP_K * DN_KC
        DN_UPW = (DN_UNITS + DN_GW - 1) // DN_GW
        SH_UPW = (DN_KC + DN_GW - 1) // DN_GW
        ROUTED_TASKS = TOP_K * UG_PER_SLOT * tokens
        TPT = TOP_K * UG_PER_SLOT
        CPT = ug_ctas_per_token(tokens)
        UG_ITERS = max(ug_tasks_of(c, tokens) for c in range(G))

        mx8_scale = k4ctx["mx8_scale"]
        mx8_q = k4ctx["mx8_q"]

        def ffn_next_pre(t, row_tok, row_pair, row):
            """``ffn_next``'s inputs, out before the peers' wait: "a" polled (long
            landed), h_in (device scope) and g_next copied to the thread's red
            words (red is free in the FFN finish; a copy is not sunk to its use)."""
            got = mb_poll([(mr("a"), (row_tok * HIDDEN + row) // 2, 1)])
            slot = lds_addr(red) + tid * 8
            async_to_lds(
                fx.Int64(h_in) + fx.Int64((row_tok * HIDDEN + row) // 2 * 4),
                slot,
                b128=False,
                cm=CM_DEV,
            )
            async_to_lds(
                fx.Int64(g_next) + fx.Int64(row // 2 * 4), slot + 4, b128=False
            )
            return got[0][0]

        def ffn_next(t, row_tok, row_pair, row, f0, f1, a_word):
            """The next layer's K1 inputs of this layer's output rows row, row + 1
            of token row_tok (block t: the token's 16 threads): its residual acc =
            bf16(ffn) + h_mid (h_mid as the o reduce stores it, from "a" and h_in),
            acc (g_next + 1) as MXFP8, the block's sum of acc^2."""
            if const_expr(WIDE):
                wait_async(0)
                a0, a1 = bf2_f32(a_word)
                r0, r1 = bf2_f32(fx.ptr_load(red + tid * 2).bitcast(fx.Int32))
                gn0, gn1 = bf2_f32(fx.ptr_load(red + (tid * 2 + 1)).bitcast(fx.Int32))
            else:  # narrow: no live ranges across the peers' wait (VGPR-bound)
                got = mb_poll([(mr("a"), (row_tok * HIDDEN + row) // 2, 1)])
                a0, a1 = bf2_f32(got[0][0])
                r0, r1 = bf2_f32(fx.Int32(ld_dev(h_in, (row_tok * HIDDEN + row) // 2)))
                gn0, gn1 = bf2_f32(
                    fx.Int32(
                        bo.buffer_load(rsrc(g_next), row // 2, vec_width=1, dtype=T.i32)
                    )
                )
            h0 = bf16_round(f0) + bf16_round(a0 + r0)
            h1 = bf16_round(f1) + bf16_round(a1 + r1)
            bo.buffer_store(
                fx.Vector.from_elements([h0, h1], fx.Float32).to(fx.BFloat16),
                rsrc(h_next),
                row_tok * HIDDEN + row,
            )
            y0 = h0 * (gn0 + 1.0)
            y1 = h1 * (gn1 + 1.0)
            sc, inv = mx8_scale([y0, y1], (1, 2, 4, 8))
            lo = fx.Int32(
                rocdl.cvt_pk_fp8_f32(
                    T.i32, mx8_q(y0, inv), mx8_q(y1, inv), fx.Int32(0), False
                )
            )
            word = (lo & 0xFFFF) | (xshfl(lo, 1) << 16)
            ssq = butterfly(h0 * h0 + h1 * h1, (1, 2, 4, 8))
            if row_pair % 2 == 0:
                bo.buffer_store(word, rsrc(x8_next), (row_tok * HIDDEN + row) // 4)
            if row_pair == 0:
                bo.buffer_store(
                    fx.Int8(sc), rsrc(x8s_next), row_tok * (HIDDEN // 32) + t
                )
                bo.buffer_store(ssq, rsrc(ss_next), row_tok * (HIDDEN // 32) + t)

        if const_expr(not WIDE):
            ffn_next_pre = None
        k4ctx.update(locals())
        k4ctx.update(moe_common_defs(k4ctx))
        k4ctx.update(ffn_reduce_defs(k4ctx))

        W_UPP = tokens * (XN8_ROW + XSC_ROW)
        W_SIG = W_UPP + ACC_WORDS
        MID_ROWS = MOE_SLOTS * tokens
        MID_ROW_WORDS = INTER // 4
        MIDSC_ROW_WORDS = INTER // 32
        W_MIDSC = (MID_ROWS + 1) * MID_ROW
        UG_PAIRS = UG_PER_SLOT // 2
        UGW_ITERS = 2 * ((U_MAX * UG_PAIRS + G - 1) // G)
        assert DN_KC <= DN_GW
        ROUTE_ITERS = (tokens + WAVES - 1) // WAVES

        k4ctx.update(locals())
        k4ctx.update(moe_wide_defs(k4ctx))
        stage_moe_wide = k4ctx["stage_moe_wide"]

        k4ctx.update(locals())
        k4ctx.update(moe_narrow_defs(k4ctx))
        stage_moe = k4ctx["stage_moe"]

        enter_stage("ug")
        if const_expr("moe" not in SKIP):
            (stage_moe_wide if WIDE else stage_moe)()
        stamp(TL_END)
        if const_expr(timeline):
            gpu.barrier()
            fx.generic_store(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                    tl + fx.Int64((bid * TL_RECORD + tid) * 8),
                ),
                fx.ptr_load(tls + tid),
            )

    @launcher(kernel_name)
    def launch_post_attn(
        h_in: Int64,
        q: Int64,
        block_table: Int64,
        seq_lens: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        w_o: Int64,
        s_o: Int64,
        g_post: Int64,
        w_gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        h_mid: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        bt_width: Int32,
        q_len: Int32,
        tl: Int64,
        k1_args: Int64,
        positions: Int64,
        slot_mapping: Int64,
        res: Int64,
        batch_ids: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = build_key
        post_attn_kernel(
            h_in, q, block_table, seq_lens, k_cache, v_cache, k_scale, v_scale, w_o, s_o,
            g_post, w_gate, bias, w13, s13, w2, s2, h_mid, ar_out, scratch, sym, peers,
            rank, layer, bt_width, q_len, tl, k1_args, positions, slot_mapping, res,
            batch_ids,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)  # fmt: skip

    K4_ABI.check(post_attn_kernel, launch_post_attn)
    return launch_post_attn
