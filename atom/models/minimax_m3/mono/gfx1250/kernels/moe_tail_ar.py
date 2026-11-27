# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Routed MoE GEMM2 + its tail (gather-reduce, routed scale, shared add) + the TP4 one-shot all-reduce + residual
add + RMSNorm + MXFP8 in one launch (gfx1250, 8 decode rows).

The decode MoE of a TP4 grouped model (MiniMax-M3.1: N 6144, K 768, 128 experts, top-4, 8 rows) ends in two
launches: the grouped GEMM2 (``mxfp4_preshuffle_gfx1250_tdm``, bf16 rows per route) and the k5 one-shot all-reduce
with the deferred tail (``csrc/include/ar_k5_gfx1250.cuh``, ``k5_ar_rms_kernel<false, true, true>``: it gathers
the routes, pushes, sums, adds the residual, normalises and quantises). This launch does both with the same bits.

One launch, three roles, 256 threads a workgroup, no cluster and no in-launch counter:

* tile workgroup ``m * n_tiles + c`` (m-major, so the populated tiles lead the grid): the production GEMM2 tile
  body (``build_gemm_a8w4_tdm_task``, explicit tile, unchanged arithmetic) of m-tile ``m`` (= one expert: an
  expert has at most 8 rows) and column tile ``c``. Its (expert, valid rows) come from the route itself (top-k ids
  + ``topids_to_rows``, one load round) instead of the tile map's dependent lookups. Its bf16 rows go to the
  grouped output at device scope with no store drain behind them, 0xffff half-words rewritten to 0x7fff.
* the tile workgroups of m < 8 then gather token ``m``'s packs of column tile ``c`` (64 packs: 2 waves): each
  lane polls the four route rows' packs it needs (device scope, ``s_sleep 1``, count-bounded) until none holds
  the 0xffff sentinel -- the data is its own flag, k5's inbox rule applied to the grouped output, which this
  launch owns, arms once and re-arms after reading. Then k5's ``tail_math``: ``acc = acc + w[k] * (0 + g[k])`` in
  route order, bf16, ``* scale`` (rounded), ``+ addend`` (rounded); 0xffff -> 0x7fff; each 16-byte pack is stored,
  system scope, into slot ``rank`` of buffer ``round`` of the three peers' k5 inboxes, and, device scope, into
  this rank's own row scratch (the 8 rows behind the grouped output: armed and re-armed the same way).
* the last 8 workgroups: k5's receiver, a row a workgroup, a wave a 96-pack slab: re-arm the peers' slots of
  buffer ``round + 2`` (system scope, before the wait, as k5: the launch ends with no system-scope store
  outstanding), poll the three peers' slots and the own row scratch, the Gluon epilogue (rank-order fp32 sum,
  bf16, fp32 residual add, squares in order, lane and wave pair trees, ``v_rcp_f32`` / ``v_rsq_f32``, fp64 eps,
  ``(s * rstd) * (w + 1)``), k5's group-32 MXFP8 of the bf16 result (or bf16), counters advanced.

The only in-launch data edge is tile -> gather (data-as-flag, ~0.7 us on gfx1250 at any placement; a counter or a
store drain + barrier costs 1.0-1.3). The inbox, its counters and their peer mapping are k5's own: a launch
interleaves with k5 calls and with the o_proj launch on the same counters. No fused multiply-adds, IEEE fp32
denormals (FlyDSL's default). The one bit difference from production is a NaN's sign: a 0xffff half-word (a NaN)
in a GEMM2 row or in this rank's own gathered row is read here as 0x7fff.
"""

import contextlib

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from .. import moe_tail_ar_layout as LY
from aiter.ops.flydsl.kernels import mxfp4_preshuffle_gfx1250_tdm as _tdm
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

RANKS = LY.RANKS
HIDDEN = LY.HIDDEN
ROW_PACKS = LY.ROW_PACKS
RWAVES = LY.RWAVES
SLAB = LY.SLAB
BUFS = LY.BUFS
WAVE = LY.WAVE
SCOPE_DEV = 2 << 3  # gfx12 CPol SCOPE field: device (past the per-XCD L2)
SCOPE_SYS = 3 << 3  # system: what a peer's IPC-mapped memory needs
SENT_LO = 0x7FFF7FFF
SENT_HI = -0x7FFF8000  # 0x80008000 as int32
TLW = 8  # timeline words a workgroup
# shared down inside the launch (``sd``): 48 workgroups ahead of the tiles, wave w of workgroup d = output columns
# [128 d + 16 w, +16), production's ``batched_gemm_a8w8_mx32_..._sk1`` chain (6 scaled-fp8 16x16x128 steps from
# zero, bf16 out) on h (E4M3 [8, 768] + E8M0 [8, 24], written by an earlier launch)
SD_INTER = 768
SD_STEPS = SD_INTER // 128  # 6
SD_COLS = 128  # output columns a workgroup
NSD = HIDDEN // SD_COLS  # 48
SD_X = 0  # LDS: h B fragments [step 6][jj 4][kgrp 2][tok 8] x 16 B
SD_SC = SD_STEPS * 4 * 16 * 16  # 6144: h E8M0 words [tok][step]
(SDT_H, SDT_HS, SDT_DW, SDT_DS, SDT_SF) = range(5)  # the shared-down table (int64 words)


def _bits(w):
    """One bit (the half-word's top bit) per half-word of ``w`` that is 0xffff (k5's sentinel_mask)."""
    x = w ^ fx.Int32(-1)
    t = ((x & fx.Int32(SENT_LO)) + fx.Int32(SENT_LO)) | x
    return (t & fx.Int32(SENT_HI)) ^ fx.Int32(SENT_HI)


def _bf_lo(w):
    return (w << fx.Int32(16)).bitcast(fx.Float32)


def _bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def _f2bf2(lo, hi):
    """Two fp32 -> packed bf16 bits (round to nearest even, gfx1250's convert)."""
    v = Vec.from_elements([lo, hi], fx.Float32)
    return Vec(v.to(fx.BFloat16).bitcast(fx.Int32))[0]


def _rcp(v):
    # raw asm: k5 uses v_rcp_f32 itself (rcp_asm); the compiler may fold a typed 1/x to a correctly rounded one
    return fx.Float32(
        _llvm.InlineAsmOp(
            res=T.f32,
            operands_=[fx.Float32(v).ir_value()],
            asm_string="v_rcp_f32 $0, $1",
            constraints="=v,v",
            has_side_effects=True,
            is_align_stack=False,
        ).result
    )


def _rsq(v):
    return fx.Float32(_llvm.call_intrinsic(T.f32, "llvm.amdgcn.rsq.f32", [fx.Float32(v).ir_value()], [], []))


def _mem_clobber():
    """Keeps a poll loop's loads inside the loop (k5: asm volatile("" ::: "memory"))."""
    _llvm.InlineAsmOp(
        res=None, operands_=[], asm_string="", constraints="~{memory}", has_side_effects=True, is_align_stack=False
    )


def _now32():
    return fx.Int32(fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], [])) & fx.Int64(0x7FFFFFFF))


def _lds_barrier():
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _or32(v):
    """Wave-uniform OR of a per-lane i32."""
    for d_ in range(5):
        v = v | v.shuffle_xor(fx.Int32(1 << d_), fx.Int32(WAVE))
    return fx.Int32(rocdl.readfirstlane(T.i32, v))


def _max32(v):
    """Wave-uniform maximum of a per-lane i32."""
    for d_ in range(5):
        o_ = v.shuffle_xor(fx.Int32(1 << d_), fx.Int32(WAVE))
        v = (v < o_).select(o_, v)
    return fx.Int32(rocdl.readfirstlane(T.i32, v))


@contextlib.contextmanager
def flagged_c_stores(state=None):
    """While this launch's GEMM2 tile body is traced: its bf16 C stores (the LDS read-back + buffer stores of
    ``c_lds_store``, the only buffer stores of a stage-2 tile) go out at device scope with 0xffff half-words
    rewritten to 0x7fff, so the gather can poll them for the sentinel. The grouped GEMM builder has no option
    for the scope or the value of that store; ``state["c_stores"]`` counts the stores traced."""
    orig = _tdm.buffer_store

    def dev_store(data, rsrc, offset, mask=None, cache_modifier=0, **kw):
        if state is not None:
            state["c_stores"] = state.get("c_stores", 0) + 1
        v = Vec(data)
        words = [fx.Int32(v[c]) for c in range(4)]
        fixed = Vec.from_elements([w_ ^ _bits(w_) for w_ in words], fx.Int32)
        return orig(fixed.ir_value(), rsrc, offset, mask=mask, cache_modifier=SCOPE_DEV, **kw)

    _tdm.buffer_store = dev_store
    try:
        yield
    finally:
        _tdm.buffer_store = orig


def _mx_code(amax):
    """E8M0 code of a group whose |max| is ``amax`` (k5: floor 1e-10, ceil pow2 of amax / 448)."""
    a = fx.Float32(amax)
    a = (a < fx.Float32(1e-10)).select(fx.Float32(1e-10), a)
    scaled = a * fx.Int32(0x3B124925).bitcast(fx.Float32)  # fp32(1 / 448)
    bits = scaled.bitcast(fx.Int32)
    code = (bits >> fx.Int32(23)) + ((bits & fx.Int32(0x7FFFFF)) != fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
    e = fx.Int32(254) - code
    e = (e > fx.Int32(0)).select(e, fx.Int32(0))
    return code, (e << fx.Int32(23)).bitcast(fx.Float32)


def _fp8x4(vals, recip):
    """Four fp32 values times ``recip``, clamped to +-448, as packed E4M3 (v_cvt_pk_fp8_f32, nearest even)."""
    c = []
    for v in vals:
        y = v * recip
        y = (y < fx.Float32(-448.0)).select(fx.Float32(-448.0), y)
        y = (y > fx.Float32(448.0)).select(fx.Float32(448.0), y)
        c.append(y)
    pk = fx.rocdl.cvt_pk_fp8_f32(T.i32, c[0].ir_value(), c[1].ir_value(), fx.Int32(0).ir_value(), 0)
    pk = fx.rocdl.cvt_pk_fp8_f32(T.i32, c[2].ir_value(), c[3].ir_value(), pk, 1)
    return fx.Int32(pk)


def _fabs(v):
    return (fx.Float32(v).bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)).bitcast(fx.Float32)


def _fmax(a, b):
    return (a < b).select(b, a)


def _set_meta(meta_holder, meta):
    """Fill the GEMM2 body's per-tile (expert, valid rows). A plain function: a subscript store in the kernel
    body would read as state to the AST rewriter."""
    meta_holder[0] = meta


@flyc.jit
def launch_moe_tail_ar(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_m_tile_map: fx.Pointer,
    arg_ids: fx.Pointer,
    arg_rows: fx.Pointer,
    arg_gw: fx.Pointer,
    arg_add: fx.Pointer,
    arg_res: fx.Pointer,
    arg_nw: fx.Pointer,
    arg_out: fx.Pointer,
    arg_outs: fx.Pointer,
    arg_res_out: fx.Pointer,
    arg_ctr: fx.Pointer,
    arg_tl: fx.Pointer,
    i64_in0: fx.Int64,
    i64_in1: fx.Int64,
    i64_in2: fx.Int64,
    i64_in3: fx.Int64,
    i32_m: fx.Int32,
    i32_slot: fx.Int32,
    i32_rank: fx.Int32,
    i32_maxit: fx.Int32,
    f32_scale: fx.Float32,
    f64_eps: fx.Float64,
    stream: fx.Stream,
    K: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    a_is_fp4: Constexpr[int],
    n_experts: Constexpr[int],
    num_waves_per_tensor_tdm: Constexpr[int],
    tdm_b_th: Constexpr[int],
    lds_soa_load_interleave: Constexpr[int],
    rolled_drain: Constexpr[int],
    a_preshuffled: Constexpr[int],
    tokens: Constexpr[int],
    has_add: Constexpr[int],
    has_scale: Constexpr[int],
    max_rows: Constexpr[int],
    probe: Constexpr[int],
    fp8: Constexpr[int],
    tl: Constexpr[int],
    sd: Constexpr[int] = 0,
    arg_sdt: fx.Pointer = None,
):
    """``arg_c``: the grouped GEMM2 output bf16 [i32_m, 6144] followed by this rank's own gathered rows bf16
    [tokens, 6144], owned by this launch: every half-word 0xffff before the first launch (the launch re-arms
    what it reads); ``arg_a`` .. ``arg_m_tile_map``: GEMM2's operands
    as ``launch_gemm_a8w4_tdm`` takes them (the tile map is not read); ``arg_ids`` / ``arg_rows``: int32
    [tokens, 4] top-k expert ids and ``topids_to_rows``; ``arg_gw``: fp32 [tokens, 4] route weights; ``arg_add``:
    bf16 [tokens, 6144] addend (``has_add``), ``f32_scale`` the routed scale (``has_scale``); ``arg_res`` /
    ``arg_nw``: residual bf16 [tokens, 6144] and the Gemma norm weight; out: ``arg_out`` E4M3 [tokens, 6144] and
    ``arg_outs`` E8M0 [tokens, 192] (``fp8``; else ``arg_out`` bf16 [tokens, 6144], k5's bf16-out kernel),
    ``arg_res_out`` bf16 [tokens, 6144]. ``arg_ctr`` / ``i64_in*`` / ``i32_slot``: the k5 context (counters, the
    four inboxes, packs a sender a buffer = max_rows * 768).

    ``probe`` > 0: a receiver first polls one own-slot pack a lane with ``s_sleep probe`` (its rank's packs come
    first; no 12-load poll while GEMM2 streams). ``tl``: thread 0 of every workgroup stamps the 100 MHz steady
    counter into ``arg_tl`` (int32 [grid, 8]): tile workgroups entry / tile body left / rows seen / pushes issued /
    stores acked; receivers entry / poll start / own packs seen / all seen / outputs issued / stores acked, [6]
    wave 7's all seen, [7] after the epilogue's barrier."""
    if tokens != 8 or K % 128 or HIDDEN % tile_n or tile_n % (8 * WAVE) or tile_n // 8 > 8 * WAVE:
        raise ValueError(f"moe_tail_ar: 8 rows, tile_n a multiple of 256 dividing 6144 (got {tokens}, {tile_n})")
    holder = [None]
    meta_holder = [None]
    g2_task, block, g2_name, _, lay = _tdm.build_gemm_a8w4_tdm_task(
        K=K,
        tile_m=LY.TILE_M,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=1,
        n_warp=n_warp,
        out_is_f16=0,
        num_buffers=num_buffers,
        a_is_fp4=a_is_fp4,
        n_experts=n_experts,
        stage1_act=0,
        has_bias=0,
        stage1_quant_out=0,
        quant_wmma_rep=1,
        cluster_n=1,
        cluster_m=1,
        next_stage_prefetch=0,
        num_waves_per_tensor_tdm=num_waves_per_tensor_tdm,
        tdm_as_in_prologue=0,
        tdm_b_th=tdm_b_th,
        enable_ep_scatter=0,
        ep_arena_handle=0,
        ep_combine_input_offset=0,
        ep_slot_stride_bytes=0,
        ep_destination_stride=0,
        ep_world_size=0,
        ep_quant_bits=0,
        row_major_ascale=0,
        a_row_stride_bytes=0,
        a_scale_row_stride_bytes=0,
        lds_soa_load_interleave=lds_soa_load_interleave,
        persistent_workers=1,
        rolled_drain=rolled_drain,
        a_preshuffled=a_preshuffled,
        m_align=0,
        explicit_tile=1,
        lds_base=holder,
        tile_meta=meta_holder,
        # no store drain behind the tile: its fences and closing barrier order LDS and TDM only (the gather
        # polls the rows themselves; the LDS arena is not reused)
        raw_drain=1,
        # the role branch puts a branch right after a TDM issue (the gfx1250 hazard tdm_snop avoids)
        tdm_snop=1,
        # the C tile off input stage 0 and written by LDS read-back + buffer stores (the shared-fold
        # launcher's settings: a TDM store can read LDS bytes older than the workgroup's ds_stores)
        c_stage_apart=1,
        c_lds_store=1,
    )
    if block != RWAVES * WAVE:
        raise ValueError(f"moe_tail_ar: GEMM2 tile of {block} threads, need {RWAVES * WAVE} (n_warp 8)")
    arena_g2 = int(lay["arena"])
    RED_OFF = (arena_g2 + 15) // 16 * 16  # receiver: 8 fp32 wave partials
    arena = RED_OFF + RWAVES * 4
    NT = LY.n_tiles(tile_n)
    NTILE = LY.tile_wgs(tile_n)
    PPT = LY.tile_packs(tile_n)
    GWAVES = PPT // WAVE  # waves of a tile workgroup that gather
    ctr_recv = LY.ctr_recv(max_rows)
    ctr_err = LY.ctr_err(max_rows)
    kname = (
        f"moe_tail_ar_k5_gfx1250{'_tl' if tl else ''}_r{tokens}{'_add' if has_add else ''}"
        f"{'_sc' if has_scale else ''}{'' if fp8 else '_bf16'}_p{probe}{'_sd' if sd else ''}_" + g2_name
    )
    if sd and (tl or not has_add):
        raise ValueError("moe_tail_ar: the shared-down build has an addend and no timeline")
    NSD_ = NSD if sd else 0

    @flyc.kernel(name=kname, known_block_size=[block, 1, 1])
    def kernel_moe_tail_ar(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_m_tile_map: fx.Pointer,
        arg_ids: fx.Pointer,
        arg_rows: fx.Pointer,
        arg_gw: fx.Pointer,
        arg_add: fx.Pointer,
        arg_res: fx.Pointer,
        arg_nw: fx.Pointer,
        arg_out: fx.Pointer,
        arg_outs: fx.Pointer,
        arg_res_out: fx.Pointer,
        arg_ctr: fx.Pointer,
        arg_tl: fx.Pointer,
        i64_in0: fx.Int64,
        i64_in1: fx.Int64,
        i64_in2: fx.Int64,
        i64_in3: fx.Int64,
        i32_m: fx.Int32,
        i32_slot: fx.Int32,
        i32_rank: fx.Int32,
        i32_maxit: fx.Int32,
        f32_scale: fx.Float32,
        f64_eps: fx.Float64,
        arg_sdt: fx.Pointer,
    ):
        base_ptr = fx.SharedAllocator(static=True).allocate(arena)._ptr
        holder[0] = base_ptr
        tid = fx.Int32(fx.thread_idx.x)
        bid_raw = fx.Int32(fx.block_idx.x)
        bid = bid_raw - NSD_  # the original roles' index (the shared-down workgroups lead the grid)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        one = fx.Float32(1.0)
        i32_global = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        tlp = fx.recast_iter(i32_global, arg_tl)
        ctr = fx.recast_iter(i32_global, arg_ctr)
        ctr_rs = create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(arg_ctr)), num_records_bytes=LY.ctr_words(max_rows) * 4
        )

        def stamp(slot_, by=0):
            if const_expr(tl):
                if tid == by:
                    fx.ptr_store(_now32(), tlp + (bid * TLW + slot_))

        stamp(0)
        inbox_bytes = fx.Int32(BUFS * RANKS) * i32_slot * fx.Int32(16)
        ins = [i64_in0, i64_in1, i64_in2, i64_in3]
        sent = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)
        # this rank's own gathered rows: [tokens, 6144] bf16 right behind the grouped output
        xo_rs = create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(arg_c)) + fx.Int64(i32_m) * fx.Int64(HIDDEN * 2),
            num_records_bytes=tokens * HIDDEN * 2,
        )

        def peer_rs(i):
            """the k5 inbox of rank (rank + i) % 4"""
            base = fx.Int64(i64_in0)
            for d in range_constexpr(1, RANKS):
                base = (((i32_rank + i) & (RANKS - 1)) == d).select(fx.Int64(ins[d]), base)
            return create_buffer_resource_from_addr(base, num_records_bytes=inbox_bytes)

        sdt_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_sdt)), num_records_bytes=5 * 8)

        def sdt_word(i):
            w_ = Vec(buffer_load(sdt_rs, 2 * i, vec_width=2, dtype=T.i32))
            return (fx.Int64(fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(w_[1])))) << fx.Int64(32)) | fx.Int64(
                fx.Uint32(fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(w_[0])))))

        if const_expr(sd):
            sf_rs = create_buffer_resource_from_addr(sdt_word(SDT_SF), num_records_bytes=tokens * HIDDEN * 2)
        if bid_raw < NSD_:
            if const_expr(sd):
                # ---------------- shared down: output columns [128 d + 16 wave, +16) ----------------
                lane16 = lane % 16
                kgrp = lane // 16
                lds_idx = fx.Int64(fx.ptrtoint(base_ptr))
                lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
                lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)
                row = bid_raw * SD_COLS + wave * 16 + lane16
                dw_rs = create_buffer_resource_from_addr(sdt_word(SDT_DW), num_records_bytes=HIDDEN * SD_INTER)
                ds_rs = create_buffer_resource_from_addr(sdt_word(SDT_DS), num_records_bytes=HIDDEN * (SD_INTER // 32))
                ds_w = (row // 32) * (SD_INTER // 4) + row % 32
                dsc = [fx.Int32(buffer_load(ds_rs, ds_w + ks_ * 32, vec_width=1, dtype=T.i32, cache_modifier=1))
                       for ks_ in range_constexpr(SD_STEPS)]
                d_u = (row // 16) * SD_INTER + row % 16
                df = [
                    [Vec(buffer_load(dw_rs, (d_u + (ks_ * 8 + jj * 2 + kgrp) * 16) * 4, vec_width=4, dtype=T.i32,
                                     cache_modifier=1)) for jj in range_constexpr(4)]
                    for ks_ in range_constexpr(SD_STEPS)
                ]
                rocdl.sched_barrier(0)
                # h (plain: written by an earlier launch): 384 16-B chunks (token u // 48, K [16 (u % 48), +16)) over
                # the 256 threads; h scales: [tok][24] bytes = [tok][6] words, word i = tok * 6 + step
                h_rs = create_buffer_resource_from_addr(sdt_word(SDT_H), num_records_bytes=tokens * SD_INTER)
                hs_rs = create_buffer_resource_from_addr(sdt_word(SDT_HS), num_records_bytes=tokens * (SD_INTER // 32))
                for m_ in range_constexpr(2):
                    u = tid + 256 * m_
                    if u < tokens * (SD_INTER // 16):
                        hc = Vec(buffer_load(h_rs, u * 4, vec_width=4, dtype=T.i32))
                        tok = u // (SD_INTER // 16)
                        k0 = (u % (SD_INTER // 16)) * 16
                        lds_store_b128(lds_idx, SD_X + (((k0 // 128) * 4 + (k0 % 128) // 32) * 16 + ((k0 % 32) // 16) * 8
                                                         + tok) * 16, hc)
                if tid < tokens * SD_STEPS:
                    hw_ = fx.Int32(buffer_load(hs_rs, tid, vec_width=1, dtype=T.i32))
                    lds_store_b32(lds_idx, SD_SC + tid * 4, Vec.from_elements([hw_], fx.Int32))
                _lds_barrier()
                tok16 = lane16 % 8
                atom = fx.make_mma_atom(
                    fx.rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0, opsel_b=0))
                dacc = fx.make_rmem_tensor(8, fx.Float32)
                dacc.store(fx.full(8, 0.0, fx.Float32))

                def _cat4(v):
                    v01 = Vec(v[0]).shuffle(Vec(v[1]), list(range(8)))
                    v23 = Vec(v[2]).shuffle(Vec(v[3]), list(range(8)))
                    return v01.shuffle(v23, list(range(16)))

                def _rm(v):
                    t_ = fx.make_rmem_tensor(16, fx.Int32)
                    t_.store(v)
                    return t_

                for ks_ in range_constexpr(SD_STEPS):
                    xf = [lds_load_b128(lds_idx, SD_X + ((ks_ * 4 + jj) * 16 + kgrp * 8 + tok16) * 16)
                          for jj in range_constexpr(4)]
                    xs_ = fx.Int32(Vec(lds_load_b32(lds_idx, SD_SC + (tok16 * SD_STEPS + ks_) * 4))[0])
                    fx.gemm(atom, dacc, _rm(_cat4(df[ks_])), _rm(_cat4(xf)), dacc, scale_a=dsc[ks_], scale_b=xs_)
                # acc[i]: output column 8 kgrp + i of the wave's 16, token lane16 -> k5-sentinel words, device scope
                if lane16 < tokens:
                    av_ = Vec(dacc.load())
                    o_ = []
                    for c in range_constexpr(4):
                        w_ = _f2bf2(fx.Float32(av_[2 * c]), fx.Float32(av_[2 * c + 1]))
                        o_.append(w_ ^ _bits(w_))
                    buffer_store(Vec.from_elements(o_, fx.Int32).ir_value(), sf_rs,
                                 (lane16 * HIDDEN + bid_raw * SD_COLS + wave * 16 + kgrp * 8) * 2, offset_is_bytes=True,
                                 cache_modifier=SCOPE_DEV)
        elif bid < NTILE:
            # ---------------- GEMM2 tile (m-tile mt, column tile cblk), then token mt's gather ----------------
            mt = bid // NT
            cblk = bid % NT
            rows_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_rows)), num_records_bytes=tokens * 16)
            ids_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_ids)), num_records_bytes=tokens * 16)
            # the route, one (token, k) a lane: written by GEMM1's launch, visible at dispatch
            rows_v = fx.Int32(buffer_load(rows_rs, lane, vec_width=1, dtype=T.i32))
            ids_v = fx.Int32(buffer_load(ids_rs, lane, vec_width=1, dtype=T.i32))
            okv = (rows_v >= fx.Int32(0)) & (rows_v < i32_m)
            hq = okv & ((rows_v >> fx.Int32(4)) == mt)
            # the tile's expert (the same id on every route that lands in it) and valid rows
            expert = _or32(hq.select(ids_v, fx.Int32(0)))
            nrow = _max32(hq.select((rows_v & fx.Int32(15)) + 1, fx.Int32(0)))
            run = (nrow > 0) & (expert >= fx.Int32(0)) & (expert < n_experts)
            e_run = run.select(expert, fx.Int32(0))

            # the gather operands of token mt, loaded behind the tile
            gl = (mt < tokens) & (wave < GWAVES)
            tgs = gl.select(mt, fx.Int32(0))
            pc = gl.select(tid, fx.Int32(0))
            pcol = LY.pack_col(cblk, pc, tile_n)
            rk = [fx.Int32(rocdl.readlane(T.i32, rows_v, tgs * 4 + k)) for k in range(4)]
            gw_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_gw)), num_records_bytes=tokens * 16)
            gwv = Vec(buffer_load(gw_rs, tgs * 4, vec_width=4, dtype=T.i32))
            add_rs = create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(arg_add)), num_records_bytes=tokens * HIDDEN * 2
            )
            if const_expr(sd):
                add_rs = sf_rs  # the shared output comes from this launch's shared-down workgroups (polled below)
            av = Vec(buffer_load(add_rs, LY.pack_index(tgs, pcol) * 4, vec_width=4, dtype=T.i32))
            rnd = fx.Int32(buffer_load(ctr_rs, LY.slab(tgs, pcol), vec_width=1, dtype=T.i32))
            g_rs = []
            for k in range_constexpr(4):
                mapped = (rk[k] >= fx.Int32(0)) & (rk[k] < i32_m)
                row = mapped.select(rk[k], fx.Int32(0))
                # a row a descriptor (64-bit base, offsets inside the row); a route without a row reads zeros
                g_rs.append(
                    create_buffer_resource_from_addr(
                        fx.Int64(fx.ptrtoint(arg_c)) + fx.Int64(row) * fx.Int64(HIDDEN * 2),
                        num_records_bytes=mapped.select(fx.Int32(HIDDEN * 2), fx.Int32(0)),
                    )
                )

            if run:
                _set_meta(meta_holder, (e_run, nrow))
                g2_task(
                    arg_c,
                    arg_a,
                    arg_b,
                    arg_scale_a,
                    arg_scale_b,
                    arg_m_tile_map,
                    arg_m_tile_map,
                    arg_c,
                    arg_c,
                    i32_m,
                    fx.Int32(HIDDEN),
                    one,
                    one,
                    one,
                    mt,
                    cblk,
                )
            stamp(1)

            def g_load():
                _mem_clobber()
                return tuple(
                    Vec(buffer_load(g_rs[k], pcol * 4, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_DEV))
                    for k in range_constexpr(4)
                ) + ((Vec(buffer_load(add_rs, LY.pack_index(tgs, pcol) * 4, vec_width=4, dtype=T.i32,
                                      cache_modifier=SCOPE_DEV)),) if const_expr(sd) else ())

            def g_pending(vs):
                m_ = fx.Int32(0)
                for v in vs:
                    for c in range_constexpr(4):
                        m_ = m_ | _bits(fx.Int32(v[c]))
                return fx.Int32(rocdl.ballot(T.i32, m_ != 0)) != 0

            # the four route rows' packs: the data is its own flag (armed 0xffff, written once by their tiles)
            if const_expr(sd):
                (g0, g1, g2, g3, av) = g_load()
                gpend = gl & g_pending((g0, g1, g2, g3, av))
                git = fx.Int32(0)
                while gpend & (git < i32_maxit):
                    rocdl.s_sleep(1)
                    (g0, g1, g2, g3, av) = g_load()
                    gpend = g_pending((g0, g1, g2, g3, av))
                    git = git + 1
            else:
                (g0, g1, g2, g3) = g_load()
                gpend = gl & g_pending((g0, g1, g2, g3))
                git = fx.Int32(0)
                while gpend & (git < i32_maxit):
                    rocdl.s_sleep(1)
                    (g0, g1, g2, g3) = g_load()
                    gpend = g_pending((g0, g1, g2, g3))
                    git = git + 1
            if gl & gpend & (lane == 0):
                fx.atomic_add(ctr + ctr_err, fx.Int32(1), syncscope="agent")
            stamp(2)
            if gl:
                gv = (g0, g1, g2, g3)
                words = []
                for c in range_constexpr(4):
                    acc_lo = fx.Float32(0.0)
                    acc_hi = fx.Float32(0.0)
                    for k in range_constexpr(4):
                        wk = fx.Int32(gwv[k]).bitcast(fx.Float32)
                        gq = fx.Int32(gv[k][c])
                        acc_lo = acc_lo + wk * (_bf_lo(gq) + fx.Float32(0.0))
                        acc_hi = acc_hi + wk * (_bf_hi(gq) + fx.Float32(0.0))
                    o = _f2bf2(acc_lo, acc_hi)
                    if const_expr(has_add):
                        if const_expr(has_scale):
                            o = _f2bf2(_bf_lo(o) * f32_scale, _bf_hi(o) * f32_scale)
                        aq = fx.Int32(av[c])
                        o = _f2bf2(_bf_lo(o) + _bf_lo(aq), _bf_hi(o) + _bf_hi(aq))
                    words.append(o ^ _bits(o))
                data = Vec.from_elements(words, fx.Int32)
                pidx = LY.pack_index(tgs, pcol)
                off = LY.inbox_offset(rnd, i32_rank, i32_slot, pidx)
                for i in range_constexpr(1, RANKS):
                    buffer_store(data.ir_value(), peer_rs(i), off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)
                buffer_store(data.ir_value(), xo_rs, LY.xown_off(pidx), offset_is_bytes=True, cache_modifier=SCOPE_DEV)
                # consumed: the rows' packs armed again for the next launch
                for k in range_constexpr(4):
                    buffer_store(sent.ir_value(), g_rs[k], pcol * 16, offset_is_bytes=True, cache_modifier=SCOPE_DEV)
                if const_expr(sd):
                    buffer_store(sent.ir_value(), sf_rs, LY.pack_index(tgs, pcol) * 16, offset_is_bytes=True,
                                 cache_modifier=SCOPE_DEV)
            if const_expr(tl):
                stamp(3)
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
                stamp(4)
        else:
            # ---------------- k5's receiver: a row a workgroup, a wave a slab ----------------
            # (the grid has exactly `tokens` receivers: no dead workgroup, so no store sits under a branch --
            # a branch around a store costs an s_wait_xcnt at its join)
            w = wave
            rr_ = bid - NTILE
            g_ = rr_ * RWAVES + w
            srnd = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[g_])))
            rrnd = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[ctr_recv + g_])))
            p0 = rr_ * ROW_PACKS + w * SLAB + lane
            res_rs = create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(arg_res)), num_records_bytes=tokens * HIDDEN * 2
            )
            nw_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_nw)), num_records_bytes=HIDDEN * 2)
            out_rs = create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(arg_out)), num_records_bytes=tokens * HIDDEN * (1 if fp8 else 2)
            )
            os_rs = create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(arg_outs)), num_records_bytes=tokens * (HIDDEN // 32)
            )
            ro_rs = create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(arg_res_out)), num_records_bytes=tokens * HIDDEN * 2
            )
            my_base = fx.Int64(i64_in0)
            for d in range_constexpr(1, RANKS):
                my_base = (i32_rank == d).select(fx.Int64(ins[d]), my_base)
            my_rs = create_buffer_resource_from_addr(my_base, num_records_bytes=inbox_bytes)
            rs = []
            wt = []
            for j in range_constexpr(3):
                rs.append(Vec(buffer_load(res_rs, (p0 + 32 * j) * 4, vec_width=4, dtype=T.i32)))
                wt.append(Vec(buffer_load(nw_rs, (w * SLAB + 32 * j + lane) * 4, vec_width=4, dtype=T.i32)))

            def my_store(off_, data_):
                buffer_store(data_.ir_value(), my_rs, off_, offset_is_bytes=True, cache_modifier=SCOPE_SYS)

            def my_load(off_):
                # buffer_load takes a dword offset; not volatile: one wait a poll round, not one a load
                return Vec(buffer_load(my_rs, off_ // 4, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_SYS))

            # re-arm the peers' slots of buffer (round + 2) % 3, as k5 does
            rearm = (rrnd + 2) % BUFS
            for i in range_constexpr(1, RANKS):
                s_ = (i32_rank + i) & (RANKS - 1)
                for j in range_constexpr(3):
                    my_store(LY.inbox_offset(rearm, s_, i32_slot, p0 + 32 * j), sent)

            def offs(s_, j):
                return LY.inbox_offset(rrnd, s_, i32_slot, p0 + 32 * j)

            def own_load(j):
                return Vec(
                    buffer_load(xo_rs, LY.xown_off(p0 + 32 * j) // 4, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_DEV)
                )

            def load_all():
                # [0..2] this rank's own packs, [3 i + j] those of rank (rank + i) % 4
                _mem_clobber()
                return tuple(own_load(j) for j in range_constexpr(3)) + tuple(
                    my_load(offs((i32_rank + i) & (RANKS - 1), j))
                    for i in range_constexpr(1, RANKS)
                    for j in range_constexpr(3)
                )

            def pending(vs):
                m_ = fx.Int32(0)
                for v in vs:
                    for c in range_constexpr(4):
                        m_ = m_ | _bits(fx.Int32(v[c]))
                return fx.Int32(rocdl.ballot(T.i32, m_ != 0)) != 0

            stamp(1)
            if const_expr(probe):
                # this rank's own packs land first: one pack a lane until they are there
                _mem_clobber()
                pv = own_load(0)
                ppend = pending((pv,))
                pit = fx.Int32(0)
                while ppend & (pit < i32_maxit):
                    rocdl.s_sleep(probe)
                    _mem_clobber()
                    pv = own_load(0)
                    ppend = pending((pv,))
                    pit = pit + 1
            stamp(2)
            (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
            pend = pending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
            it = fx.Int32(0)
            while pend & (it < i32_maxit):
                rocdl.s_sleep(1)
                (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
                pend = pending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
                it = it + 1
            if pend & (lane == 0):
                fx.atomic_add(ctr + ctr_err, fx.Int32(1), syncscope="agent")
            stamp(3)
            stamp(6, by=(RWAVES - 1) * WAVE)
            rel = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
            # the packs are in registers: the own rows are armed again for the next launch (device scope) and the
            # slab's counters advance, behind the epilogue's arithmetic
            for j in range_constexpr(3):
                buffer_store(
                    sent.ir_value(), xo_rs, LY.xown_off(p0 + 32 * j), offset_is_bytes=True, cache_modifier=SCOPE_DEV
                )
            # rank order: the packs of absolute rank q are those at relative index (q - rank) % 4
            flat = []
            for q in range_constexpr(RANKS):
                ri = (fx.Int32(q) - i32_rank) & (RANKS - 1)
                for j in range_constexpr(3):
                    ws_ = []
                    for c in range_constexpr(4):
                        v_ = fx.Int32(rel[j][c])
                        for i in range_constexpr(1, RANKS):
                            v_ = (ri == i).select(fx.Int32(rel[3 * i + j][c]), v_)
                        ws_.append(v_)
                    flat.append(ws_)
            lane0 = lane == 0
            buffer_store(((srnd + 1) % BUFS).ir_value(), ctr_rs, g_ * 4, mask=lane0, offset_is_bytes=True)
            buffer_store(
                ((rrnd + 1) % BUFS).ir_value(), ctr_rs, (ctr_recv + g_) * 4, mask=lane0, offset_is_bytes=True
            )

            # k5's Gluon epilogue
            s = [[None] * 8 for _ in range(3)]
            for j in range_constexpr(3):
                for k in range_constexpr(8):
                    acc_ = None
                    for q in range_constexpr(RANKS):
                        wq = fx.Int32(flat[q * 3 + j][k // 2])
                        x_ = _bf_hi(wq) if k % 2 else _bf_lo(wq)
                        acc_ = x_ if acc_ is None else acc_ + x_
                    s[j][k] = acc_
            sq = None
            for j in range_constexpr(3):
                o = []
                for c in range_constexpr(4):
                    ab = _f2bf2(s[j][2 * c], s[j][2 * c + 1])
                    rw = fx.Int32(rs[j][c])
                    s[j][2 * c] = _bf_lo(ab) + _bf_lo(rw)
                    s[j][2 * c + 1] = _bf_hi(ab) + _bf_hi(rw)
                for k in range_constexpr(8):
                    e = s[j][k] * s[j][k]
                    sq = e if sq is None else sq + e
                for c in range_constexpr(4):
                    o.append(_f2bf2(s[j][2 * c], s[j][2 * c + 1]))
                buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), ro_rs, (p0 + 32 * j) * 16, offset_is_bytes=True)
            for d_ in range_constexpr(5):
                sq = sq + sq.shuffle_xor(fx.Int32(1 << d_), fx.Int32(WAVE))
            red = fx.recast_iter(
                fx.PointerType.get(elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4),
                base_ptr,
            ) + (RED_OFF // 4)
            if lane == 0:
                fx.ptr_store(sq, red + w)
            _lds_barrier()
            stamp(7)
            pr = [fx.Float32(red[i]) for i in range_constexpr(RWAVES)]
            total = ((pr[0] + pr[1]) + (pr[2] + pr[3])) + ((pr[4] + pr[5]) + (pr[6] + pr[7]))
            mean = total * _rcp(fx.Float32(float(HIDDEN)))
            rstd = _rsq(fx.Float32(fx.Float64(mean) + f64_eps))
            for j in range_constexpr(3):
                o = []
                for c in range_constexpr(4):
                    wv = fx.Int32(wt[j][c])
                    y0 = (s[j][2 * c] * rstd) * (_bf_lo(wv) + fx.Float32(1.0))
                    y1 = (s[j][2 * c + 1] * rstd) * (_bf_hi(wv) + fx.Float32(1.0))
                    o.append(_f2bf2(y0, y1))
                if const_expr(fp8):
                    # k5's group-32 MXFP8 of the bf16 norm output: a group is 4 lanes of a plane
                    yb = []
                    for c in range_constexpr(4):
                        yb.append(_bf_lo(o[c]))
                        yb.append(_bf_hi(o[c]))
                    amax = _fabs(yb[0])
                    for k in range_constexpr(1, 8):
                        amax = _fmax(amax, _fabs(yb[k]))
                    amax = _fmax(amax, amax.shuffle_xor(fx.Int32(1), fx.Int32(WAVE)))
                    amax = _fmax(amax, amax.shuffle_xor(fx.Int32(2), fx.Int32(WAVE)))
                    code, recip = _mx_code(amax)
                    q0 = _fp8x4(yb[0:4], recip)
                    q1 = _fp8x4(yb[4:8], recip)
                    buffer_store(
                        Vec.from_elements([q0, q1], fx.Int32).ir_value(),
                        out_rs,
                        LY.out_byte(rr_, w, lane, j),
                        offset_is_bytes=True,
                    )
                    buffer_store(
                        code.to(fx.Int8).ir_value(),
                        os_rs,
                        LY.scale_byte(rr_, w, lane, j),
                        mask=(lane % 4) == 0,
                        offset_is_bytes=True,
                    )
                else:
                    buffer_store(
                        Vec.from_elements(o, fx.Int32).ir_value(), out_rs, (p0 + 32 * j) * 16, offset_is_bytes=True
                    )
            if const_expr(tl):
                stamp(4)
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
                stamp(5)

    kernel_moe_tail_ar(
        arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_m_tile_map, arg_ids, arg_rows, arg_gw, arg_add,
        arg_res, arg_nw, arg_out, arg_outs, arg_res_out, arg_ctr, arg_tl,
        i64_in0, i64_in1, i64_in2, i64_in3, i32_m, i32_slot, i32_rank, i32_maxit, f32_scale, f64_eps,
        arg_sdt if arg_sdt is not None else arg_ctr,
    ).launch(grid=(LY.grid(tile_n, tokens) + NSD_, 1, 1), block=(block, 1, 1), stream=stream)  # fmt: skip


launch_moe_tail_ar.compile_hints["llvm_options"] = dict(_tdm.launch_gemm_a8w4_tdm.compile_hints["llvm_options"])
if _tdm.SUPPORT_EXPLICIT_VGPR_PARTITION:
    launch_moe_tail_ar.compile_hints["waves_per_eu"] = 1
    launch_moe_tail_ar.compile_hints["maxnreg"] = 1024
