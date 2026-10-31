# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K1 ``mono_pre`` (``kernels.pre_attn`` on wave32 and WMMA).

The tasks, their CTAs, the scratch layout and the numerical contracts are the
gfx950 kernel's. On 16 waves of 32:

- the norm: logical thread lt (8 elements of a 6144-wide row) is thread lt, and
  thread lt - 512 for lt >= 512; logical warp w of the reduction order is wave w
  (w < 16), else the second half of wave w - 16, so rstd reduces in the same order.
- the GEMV: a 64-k chunk is one 16x16x64 FP8 WMMA, A the preshuffled weight
  ((16, 16): lane l reads row l % 16 at k 64 kc + 32 c + 16 (l / 16), c = 0, 1:
  16 B of each of the chunk's two 512 B blocks), B the token's FP8 row.
- the head tasks: token k on wave k.
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    ONE_INDEX_HEAD,
    PAGE16,
    ROTARY_DIM,
    IndexHeads,
)
from atom.models.minimax_m3.mono.gfx1250.kernels.index_score import (
    emit_index_scores,
    q_dims,
)
from atom.models.minimax_m3.mono.kernels.common import (
    ar_pack_sumsq,
    per_token_fp8_scale,
)
from atom.models.minimax_m3.mono.kernels.index_score import (
    index_scale_log2e,
    step_rows,
)
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    DENSE_PAGE,
    ELEMS,
    HEAD_ROW_GROUPS,
    HT0,
    IK_TASK,
    IQ_TASK,
    K1_ABI,
    K1_ARGS,
    K_OFF,
    N_HEAD_TASKS,
    NT0,
    ONE_GEMV,
    OTHER_GEMV,
    PAGE_BYTES,
    ROWS_PER_TASK,
    SCRATCH_BYTES,
    SCRATCH_HDONE,
    SCRATCH_QKV,
    SCRATCH_RDONE,
    SCRATCH_X8,
    SCRATCH_X8S,
    TL_POINTS,
    V_OFF,
    XT0,
    bf16x8_f32,
    head_rope_inputs,
    ld_raw8,
)
from atom.models.minimax_m3.mono.sources import SOURCES
from atom.mono.device.gfx1250 import (
    BLOCKS,
    CM_DEV,
    CM_NT,
    THREADS,
    WAVE,
    WAVES,
    acc_row_sum,
    butterfly,
    publish,
    wave_max,
    wmma_fp8_k64,
)
from atom.mono.device.mx import FP8_MAX
from atom.mono.device.ops import (
    bf16_pair,
    bf16_round,
    div_rn,
    fp8_pack4,
    hw_rsq,
    kernel_symbol,
    lane_gather,
    launcher,
    rows_to_lds,
    rsrc,
    traced,
    uniform,
)
from atom.mono.device.sync import Addr, Mailbox, sreg
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.trace import Space, enter_stage

__all__ = ["K1_ABI", "K1_ARGS", "SCRATCH_BYTES", "TL_POINTS", "build_pre_attn_kernel", "emit_k1"]

K_CHUNKS = HIDDEN // 64  # 64-k chunks: one WMMA each
CHUNKS_PER_WAVE = K_CHUNKS // WAVES
LT_SPLIT = THREADS  # logical threads past it are held by threads lt - LT_SPLIT
N_LT = HIDDEN // 8
RED_FLOATS = WAVES * WAVE * 8  # every wave's 16x16 f32 accumulator
assert K_CHUNKS % WAVES == 0 and N_LT - LT_SPLIT <= THREADS // 2
assert MAX_TOKENS <= WAVES
_CURRENT_STREAM = fx.Stream(None)


@traced
def ar_block_sums(parts, lane, wave, red):
    """``common.ar_block_sums`` on wave32: per row (p0, p1), logical warp ``wave``
    holds p0 and logical warp 16 + ``wave`` p1 (waves < 8); a 16..1 butterfly per
    warp, then the 24 warp sums butterflied again. ``red`` needs 24 words a row."""
    bs = [
        (butterfly(p0, (16, 8, 4, 2, 1)), butterfly(p1, (16, 8, 4, 2, 1)))
        for p0, p1 in parts
    ]
    gpu.barrier()
    for k in range_constexpr(len(bs)):
        b0, b1 = bs[k]
        if lane == 0:
            fx.ptr_store(b0, red + (24 * k + wave))
        if (lane == 0) & (wave < 8):
            fx.ptr_store(b1, red + (24 * k + 16 + wave))
    gpu.barrier()
    tots = [
        butterfly(
            (lane < 24).select(fx.ptr_load(red + (24 * k + fx.min(lane, 23))), fx.Float32(0.0)),
            (16, 8, 4, 2, 1),
        )
        for k in range(len(parts))
    ]
    gpu.barrier()
    return tots


def head_norm_rope(e, lane, gw4, cs, sn, eps):
    """``kernels.pre_attn.head_norm_rope`` on a wave: lane l holds dims 4 l .. + 4."""
    d0 = lane * ELEMS
    half = ROTARY_DIM // 2
    first = d0 < half
    ss = fx.Float32(0.0)
    for j in range_constexpr(ELEMS):
        ss = ss + e[j] * e[j]
    ss = butterfly(ss, (16, 8, 4, 2, 1))
    rr = hw_rsq(ss / float(HEAD_DIM) + eps)
    n = [e[j] * rr * (gw4[j] + 1.0) for j in range(ELEMS)]
    in_rope = d0 < ROTARY_DIM
    partner_lane = first.select(lane + half // ELEMS, lane - half // ELEMS)
    out = []
    for j in range_constexpr(ELEMS):
        p = lane_gather(
            n[j].bitcast(fx.Int32), fx.min(fx.max(partner_lane, 0), WAVE - 1)
        ).bitcast(fx.Float32)
        ps = p * sn[j]
        rot = fmath.fma(n[j], cs[j], first.select(-ps, ps))
        out.append(in_rope.select(rot, n[j]))
    return out


@traced
def emit_pre_attn(
    tid, bid, lane, wave, x8, red, mb_put, mb_put_words, mb_poll, qkv_mb, x8_mb,
    x8s_mb, stamp, eps, tokens, heads, ptrs, signal=None, long_step=None,
    index_topk=True, dense=False, plain_norm=False,
):  # fmt: skip
    """``kernels.pre_attn.emit_pre_attn`` on gfx1250 (see the module)."""
    rows = heads.rows
    (
        ar,
        res,
        h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, positions,
        slot_mapping, k_cache, v_cache, k_scale, v_scale, index_cache, q_out, iq_out,
    ) = ptrs  # fmt: skip

    cm_out = CM_DEV if signal is not None else 0

    def publish_done(mb_addr, idx, who=None):
        if const_expr(signal is not None):
            publish(
                mb_put, mb_addr, idx, fx.Int32(1), (tid == 0) if who is None else who
            )

    has1 = tid < N_LT - LT_SPLIT
    lts = [tid, fx.min(tid + LT_SPLIT, N_LT - 1)]

    def load_norm_inputs(toks):
        raw_g = [ld_raw8(rsrc(g_in), lt) for lt in lts]
        raw_ar = [
            [
                (
                    ld_raw8(rsrc(ar), tk * (HIDDEN // 8) + lt),
                    ld_raw8(rsrc(res), tk * (HIDDEN // 8) + lt),
                )
                for lt in lts
            ]
            for tk in toks
        ]
        return raw_g, raw_ar

    def norm_quant(toks, raw_g, raw_ar, store_residual):
        gs = [bf16x8_f32(raw) for raw in raw_g]
        accs = [
            [
                [a + r for a, r in zip(bf16x8_f32(ra), bf16x8_f32(rr))]
                for ra, rr in raw_ar[j]
            ]
            for j in range(len(toks))
        ]
        for j in range_constexpr(len(toks)):
            if store_residual(j):
                r_h = rsrc(h_out)
                bo.buffer_store(
                    fx.Vector.from_elements(accs[j][0], fx.Float32).to(fx.BFloat16),
                    r_h,
                    toks[j] * HIDDEN + lts[0] * 8,
                    cache_modifier=cm_out,
                )
                if has1:
                    bo.buffer_store(
                        fx.Vector.from_elements(accs[j][1], fx.Float32).to(fx.BFloat16),
                        r_h,
                        toks[j] * HIDDEN + lts[1] * 8,
                        cache_modifier=cm_out,
                    )
        tots = ar_block_sums(
            [
                (
                    ar_pack_sumsq(accs[j][0]),
                    has1.select(ar_pack_sumsq(accs[j][1]), fx.Float32(0.0)),
                )
                for j in range(len(toks))
            ],
            lane,
            wave,
            red,
        )
        stamp(1)
        normed = []
        for j in range_constexpr(len(toks)):
            rstd = hw_rsq(tots[j] / float(HIDDEN) + eps)
            nj = [
                [accs[j][i][e] * rstd * (gs[i][e] + 1.0) for e in range(8)]
                for i in range(2)
            ]
            if const_expr(plain_norm):
                nj = [[bf16_round(x) for x in row] for row in nj]
            normed.append(nj)
        for j in range_constexpr(len(toks)):
            amax = fx.Float32(0.0)
            for e in range_constexpr(8):
                amax = fx.max(amax, fmath.absf(normed[j][0][e]))
                amax = fx.max(
                    amax, has1.select(fmath.absf(normed[j][1][e]), fx.Float32(0.0))
                )
            w_max = wave_max(amax)
            if lane == 0:
                fx.ptr_store(w_max, red + (j * WAVES + wave))
        gpu.barrier()
        x_scales = []
        for j in range_constexpr(len(toks)):
            amax = fx.ptr_load(red + j * WAVES)
            for w in range_constexpr(1, WAVES):
                amax = fx.max(amax, fx.ptr_load(red + (j * WAVES + w)))
            if const_expr(plain_norm):
                x_scale, x_inv = per_token_fp8_scale(amax)
            else:
                x_scale = (amax == 0.0).select(fx.Float32(1.0), amax / FP8_MAX)
                x_rcp = 1.0 / x_scale
            x_scales.append(x_scale)
            for i in range_constexpr(2):
                if const_expr(plain_norm):
                    q = [normed[j][i][e] * x_inv for e in range(8)]
                else:
                    q = [div_rn(normed[j][i][e], x_scale, x_rcp) for e in range(8)]
                w01 = [
                    fp8_pack4(q[0], q[1], q[2], q[3]),
                    fp8_pack4(q[4], q[5], q[6], q[7]),
                ]
                xo = x8 + (j * (HIDDEN // 4) + lts[i] * 2)
                if (i == 0) | has1:
                    fx.ptr_store(w01[0].bitcast(fx.Float32), xo)
                    fx.ptr_store(w01[1].bitcast(fx.Float32), xo + 1)
        gpu.barrier()
        stamp(2)
        return x_scales

    # ============================================ norm tasks (S > 2)
    enter_stage("k1.norm")
    if const_expr(tokens > 2):  # noqa: SIM102
        if (bid >= NT0) & (bid < NT0 + tokens):
            tk = bid - NT0
            raw_g, raw_ar = load_norm_inputs([tk])
            x_scale = norm_quant([tk], raw_g, raw_ar, lambda j: True)[0]
            if const_expr(signal is not None):
                publish_done(signal[1], tk)
            for i in range_constexpr(2):
                if (i == 0) | has1:
                    bo.buffer_store(
                        fx.Vector(
                            fx.ptr_load(
                                x8 + lts[i] * 2,
                                result_type=fx.Vector.make_type(2, fx.Int32),
                            )
                        ),
                        rsrc(x8_mb),
                        tk * (HIDDEN // 4) + lts[i] * 2,
                        cache_modifier=CM_DEV,
                    )
            publish(mb_put, x8s_mb, tk, x_scale, tid == 0)

    # ============================================ GEMV tasks (one per CTA)
    enter_stage("k1.gemv")
    own_q = (LOCAL_Q_HEADS + 2) * HEAD_ROW_GROUPS
    rg = (bid < own_q).select(
        bid,
        (bid < own_q + HEAD_ROW_GROUPS).select(
            heads.iq_off // ROWS_PER_TASK + bid - own_q,
            heads.ik_off // ROWS_PER_TASK + bid - own_q - HEAD_ROW_GROUPS,
        ),
    )
    gemv = bid < (ONE_GEMV if index_topk else own_q)
    if const_expr(heads.count > 1 and index_topk):
        xt = bid - XT0
        if (xt >= 0) & (xt < OTHER_GEMV):
            other = xt // HEAD_ROW_GROUPS
            head = other + (other >= heads.own).select(1, 0)
            rg = (LOCAL_Q_HEADS + 2 + head) * HEAD_ROW_GROUPS + xt % HEAD_ROW_GROUPS
            gemv = long_step()
    if gemv:
        if const_expr(tokens <= 2):
            raw_g, raw_ar = load_norm_inputs(list(range(tokens)))
        rocdl.sched_barrier(0)
        r_w = rsrc(w_qkv)
        # chunk kc's 512 B blocks 2 kc, 2 kc + 1 of row group rg: lane l's 16 B
        # of each, row l % 16, k = 64 kc + 32 c + 16 (l / 16) + [0, 16)
        wts = []
        for cc in range_constexpr(CHUNKS_PER_WAVE):
            kc = wave * CHUNKS_PER_WAVE + cc
            parts = []
            for c in range_constexpr(2):
                w4 = fx.Vector(
                    bo.buffer_load(
                        r_w,
                        ((rg * (HIDDEN // 32) + kc * 2 + c) * 512 + lane * 16) // 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
                parts += [w4[e] for e in range(4)]
            wts.append(fx.Vector.from_elements(parts, fx.Int32))
        ws = fx.Float32(
            bo.buffer_load(
                rsrc(s_qkv),
                rg * ROWS_PER_TASK + tid % ROWS_PER_TASK,
                vec_width=1,
                dtype=T.f32,
            )
        )
        if const_expr(tokens <= 2):
            x_scales = norm_quant(
                list(range(tokens)), raw_g, raw_ar, lambda j: bid == j
            )
            if const_expr(signal is not None):  # noqa: SIM102
                if bid < tokens:
                    publish_done(signal[1], bid)
        else:
            got = mb_poll([(x8s_mb, tk, 1) for tk in range(tokens)])
            x_scales = [got[tk][0].bitcast(fx.Float32) for tk in range(tokens)]
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
            )
            rows_to_lds(tid, x8_mb, tokens, HIDDEN // 4, x8, HIDDEN // 4)
            gpu.barrier()
            stamp(2)
        # B column l % 16 is token l % 16 (the last token past the step's)
        col_tok = fx.min(lane % 16, tokens - 1)
        c = fx.Vector.filled(8, 0.0, fx.Float32)
        for cc in range_constexpr(CHUNKS_PER_WAVE):
            kc = wave * CHUNKS_PER_WAVE + cc
            xb = []
            for h in range_constexpr(2):
                x4 = fx.Vector(
                    fx.ptr_load(
                        x8 + (col_tok * (HIDDEN // 4) + kc * 16 + h * 8 + (lane // 16) * 4),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                xb += [x4[e] for e in range(4)]
            c = wmma_fp8_k64(wts[cc], fx.Vector.from_elements(xb, fx.Int32), c)
        fx.ptr_store(c, red + (wave * WAVE + lane) * 8)
        gpu.barrier()
        if tid < ROWS_PER_TASK * tokens:
            r = tid % ROWS_PER_TASK
            tk = tid // ROWS_PER_TASK
            tot_r = acc_row_sum(red, r, tk)
            x_scale = x_scales[0]
            for k in range_constexpr(1, tokens):
                x_scale = (tk == k).select(x_scales[k], x_scale)
            mb_put(
                qkv_mb,
                tk * rows + rg * ROWS_PER_TASK + r,
                bf16_round(tot_r * x_scale * ws),
            )

        stamp(3)

    # ======================================= head tasks: norm / rope / cache
    enter_stage("k1.head")
    if (bid >= HT0) & (bid < HT0 + N_HEAD_TASKS):
        ht = bid - HT0
        tk = wave
        is_q = ht < LOCAL_Q_HEADS
        is_k = ht == LOCAL_Q_HEADS
        is_v = ht == LOCAL_Q_HEADS + 1
        is_iq = ht == LOCAL_Q_HEADS + 2
        base = tk * rows + is_q.select(
            ht * HEAD_DIM,
            is_k.select(
                K_OFF, is_v.select(V_OFF, is_iq.select(heads.iq_off, heads.ik_off))
            ),
        )
        live = (tk < tokens) if index_topk else (tk < tokens) & (ht < IQ_TASK)
        if live:
            d0 = lane * ELEMS
            slot = fx.Int32(
                bo.buffer_load(rsrc(slot_mapping), 2 * tk, vec_width=1, dtype=T.i32)
            )
            if const_expr(index_topk):
                gw = is_q.select(g_q, is_k.select(g_k, is_iq.select(g_iq, g_ik)))
            else:
                gw = is_q.select(g_q, g_k)
            gw4, cs, sn = head_rope_inputs(tk, gw, d0, cos_sin, positions)
            got = mb_poll([(qkv_mb, base + d0, 2), (qkv_mb, base + d0 + 2, 2)])
            stamp(4)
            e = [
                got[0][0].bitcast(fx.Float32),
                got[0][1].bitcast(fx.Float32),
                got[1][0].bitcast(fx.Float32),
                got[1][1].bitcast(fx.Float32),
            ]
            if is_v:
                am = fx.Float32(0.0)
                for j in range_constexpr(ELEMS):
                    am = fx.max(am, fmath.absf(e[j]))
                am = butterfly(am, (16, 8, 4, 2, 1), fx.max)
                vs = (am > 0.0).select(am / FP8_MAX, fx.Float32(1.0))
                if slot >= 0:
                    b = slot // PAGE16
                    t = slot % PAGE16
                    if lane == 0:
                        bo.buffer_store(
                            vs, rsrc(v_scale), b * PAGE16 + t, cache_modifier=cm_out
                        )
                    r_v = rsrc(v_cache)
                    for j in range_constexpr(ELEMS):
                        q8 = fp8_pack4(e[j] / vs, 0.0, 0.0, 0.0) & 0xFF
                        bo.buffer_store(
                            fx.Int8(q8),
                            r_v,
                            b * PAGE_BYTES + (d0 + j) * PAGE16 + t,
                            cache_modifier=cm_out,
                        )
            else:
                out = head_norm_rope(e, lane, gw4, cs, sn, eps)
                if is_q:
                    bo.buffer_store(
                        fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16),
                        rsrc(q_out),
                        (tk * LOCAL_Q_HEADS + ht) * HEAD_DIM + d0,
                        cache_modifier=cm_out,
                    )
                if is_iq:
                    bo.buffer_store(
                        fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16),
                        rsrc(iq_out),
                        tk * HEAD_DIM + d0,
                    )
                if is_k:
                    am = fx.Float32(0.0)
                    for j in range_constexpr(ELEMS):
                        am = fx.max(am, fmath.absf(out[j]))
                    am = butterfly(am, (16, 8, 4, 2, 1), fx.max)
                    ksc = (am > 0.0).select(am / FP8_MAX, fx.Float32(1.0))
                    if slot >= 0:
                        b = slot // PAGE16
                        t = slot % PAGE16
                        if lane == 0:
                            bo.buffer_store(
                                ksc,
                                rsrc(k_scale),
                                b * PAGE16 + t,
                                cache_modifier=cm_out,
                            )
                        if const_expr(dense):
                            qk = [out[j] / ksc for j in range(ELEMS)]
                            k_off = (
                                slot // DENSE_PAGE * (DENSE_PAGE * HEAD_DIM)
                                + (d0 // 16) * (DENSE_PAGE * 16)
                                + slot % DENSE_PAGE * 16
                                + d0 % 16
                            )
                        else:
                            qk = [bf16_round(out[j]) / ksc for j in range(ELEMS)]
                            k_off = (
                                b * PAGE_BYTES
                                + (d0 // 16) * (PAGE16 * 16)
                                + t * 16
                                + d0 % 16
                            )
                        bo.buffer_store(
                            fp8_pack4(qk[0], qk[1], qk[2], qk[3]),
                            rsrc(k_cache),
                            k_off // 4,
                            cache_modifier=cm_out,
                        )
                if (ht == IK_TASK) & (slot >= 0):
                    qi = [bf16_round(out[j]) for j in range(ELEMS)]
                    bo.buffer_store(
                        fp8_pack4(qi[0], qi[1], qi[2], qi[3]),
                        rsrc(index_cache),
                        (slot * HEAD_DIM + d0) // 4,
                        cache_modifier=cm_out,
                    )

        if const_expr(signal is not None):
            publish_done(signal[0], tid * N_HEAD_TASKS + ht, tid < tokens)


@traced
def emit_k1(
    tid, bid, lane, wave, x8, red, qs, stamp, layer, eps, index_scale, tokens,
    q_len, init_blocks, local_blocks, heads, ptrs, bt_width, signal=False, diag=None,
    region_ids=None, index_topk=True, dense=False, plain_norm=False,
):  # fmt: skip
    """``kernels.pre_attn.emit_k1`` on gfx1250. ``x8``: tokens * HIDDEN / 4 words
    of LDS, ``red``: RED_FLOATS floats, ``qs``: tokens * heads.count * HEAD_DIM / 2
    floats."""
    (
        ar, res, h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, positions,
        slot_mapping, k_cache, v_cache, k_scale, v_scale, index_cache, q_out, iq_out,
        block_table, seq_lens, iscore, scratch,
    ) = ptrs  # fmt: skip
    mbox = Mailbox(layer, diag, region_ids, gfx1250=True, steady=diag is not None)
    qkv_mb = sreg(scratch, SCRATCH_QKV, "k1.qkv")
    x8_mb = scratch + fx.Int64(SCRATCH_X8)
    x8s_mb = sreg(scratch, SCRATCH_X8S, "k1.x8s")
    hdone_mb = sreg(scratch, SCRATCH_HDONE, "k1.hdone")
    rdone_mb = sreg(scratch, SCRATCH_RDONE, "k1.rdone")
    mb_put = mbox.put
    mb_put_words = mbox.put_words
    mb_poll = mbox.poll

    def index_q_row(tk, head):
        """Token tk's index q of head ``head`` into qs (bf16 pairs), by this wave."""
        d0 = lane * ELEMS
        gw4, cs, sn = head_rope_inputs(tk, g_iq, d0, cos_sin, positions)
        base = tk * heads.rows + IndexHeads(heads.count, 0).iq_off + head * HEAD_DIM
        got = mb_poll([(qkv_mb, base + d0, 2), (qkv_mb, base + d0 + 2, 2)])
        e = [got[h][j].bitcast(fx.Float32) for h in range(2) for j in range(2)]
        out = head_norm_rope(e, lane, gw4, cs, sn, eps)
        row = qs + (tk * heads.count + head) * (HEAD_DIM // 2)
        fx.ptr_store(bf16_pair(out[0], out[1]), row + lane * 2)
        fx.ptr_store(bf16_pair(out[2], out[3]), row + (lane * 2 + 1))

    def make_index_q(first, last):
        for k0 in range_constexpr(0, tokens, WAVES):
            tk = wave + k0
            if (tk < tokens) & (tk >= first) & (tk <= last):
                index_q_row(tk, heads.own)
        gpu.barrier()

    def make_index_q_heads():
        for j0 in range_constexpr(0, tokens * heads.count, WAVES):
            j = wave + j0
            if j < tokens * heads.count:
                index_q_row(j // heads.count, j % heads.count)
        gpu.barrier()

    def q_frag_head(tk, c, head):
        """This lane's WMMA B operand of token tk's index q of head ``head``,
        k-chunk c: 16 bf16 at ``q_dims``."""
        row = (tk * heads.count + head) * (HEAD_DIM // 2)
        return fx.Vector(
            fx.ptr_load(
                qs + (row + q_dims(lane, c) // 2),
                result_type=fx.Vector.make_type(8, fx.Float32),
            )
        ).bitcast(fx.BFloat16)

    def q_frag(tk, c):
        return q_frag_head(tk, c, heads.own)

    def long_step():
        return step_rows(seq_lens, tokens, q_len, heads).any_long()

    emit_pre_attn(
        tid, bid, lane, wave, x8, red, mb_put, mb_put_words, mb_poll, qkv_mb, x8_mb,
        x8s_mb, stamp, eps, tokens, heads,
        (
            ar, res, h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, positions,
            slot_mapping, k_cache, v_cache, k_scale, v_scale, index_cache, q_out, iq_out,
        ),
        (hdone_mb, rdone_mb) if signal else None,
        long_step,
        index_topk,
        dense,
        plain_norm,
    )  # fmt: skip
    stamp(5)

    def wait_new_keys():
        mb_poll([(hdone_mb, fx.min(lane, tokens - 1) * N_HEAD_TASKS + IK_TASK, 1)])
        fx.memory_fence(
            syncscope=rocdl.SyncScope.Agent, ordering=fx.AtomicOrdering.Acquire
        )

    if const_expr(index_topk):
        score_base = ONE_GEMV + N_HEAD_TASKS
        enter_stage("k1.score")
        emit_index_scores(
            (bid + (BLOCKS - score_base)) % BLOCKS, lane, uniform(wave), red, mb_put,
            Addr("iscore", Space.SCRATCH, iscore),
            q_frag, tokens, q_len, init_blocks, local_blocks, index_scale, index_cache,
            block_table, seq_lens, bt_width, wait_new_keys if signal else None,
            make_index_q, heads, q_frag_head, make_index_q_heads,
        )  # fmt: skip
    stamp(6)


@dataclass(frozen=True)
class K1Build:
    tokens: int = field(metadata={"sym": "s"})
    init_blocks: int = field(metadata={"sym": "ib"})
    local_blocks: int = field(metadata={"sym": "lb"})
    index_heads: int = field(metadata={"sym": "ih"})
    index_own: int = field(metadata={"sym": "io"})
    dense: bool = field(metadata={"sym": "dn"})
    plain_norm: bool = field(metadata={"sym": "pn"})
    eps: float
    sm_scale: float


def build_pre_attn_kernel(
    eps: float,
    sm_scale: float,
    init_blocks: int,
    local_blocks: int,
    tokens: int = 1,
    timeline: bool = False,
    heads: IndexHeads = ONE_INDEX_HEAD,
    dense: bool = False,
    plain_norm: bool = False,
):
    """``kernels.pre_attn.build_pre_attn_kernel`` on gfx1250 (no timeline)."""
    assert 1 <= tokens <= MAX_TOKENS
    assert not timeline, "gfx1250 K1 has no timeline build"
    key = K1Build(
        tokens=tokens, init_blocks=init_blocks, local_blocks=local_blocks,
        index_heads=heads.count, index_own=heads.own, dense=dense,
        plain_norm=plain_norm, eps=eps, sm_scale=sm_scale,
    )  # fmt: skip
    build_key = key_tuple(key, SOURCES)
    index_scale = index_scale_log2e(sm_scale)

    @fx.struct
    class Smem:
        x8: fx.Array[fx.Int32, tokens * HIDDEN // 4, 16]
        red: fx.Array[fx.Float32, RED_FLOATS, 16]
        qs: fx.Array[fx.Float32, HEAD_DIM // 2 * tokens * heads.count, 16]

    kernel_name = kernel_symbol("minimax_m3_pre_attn_g1250", **symbol_params(key))
    ih_count, ih_own = heads.count, heads.own

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def pre_attn_kernel(
        ar: Int64,
        res: Int64,
        h_out: Int64,
        g_in: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        g_q: Int64,
        g_k: Int64,
        g_iq: Int64,
        g_ik: Int64,
        cos_sin: Int64,
        positions: Int64,
        slot_mapping: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        index_cache: Int64,
        q_out: Int64,
        iq_out: Int64,
        block_table: Int64,
        seq_lens: Int64,
        bt_width: Int32,
        q_len: Int32,
        iscore: Int64,
        scratch: Int64,
        layer: Int32,
        tl: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % WAVE
        wave = tid // WAVE
        lds = fx.SharedAllocator().allocate(Smem).peek()

        def stamp(k):
            pass

        emit_k1(
            tid, bid, lane, wave, lds.x8.ptr, lds.red.ptr, lds.qs.ptr, stamp, layer,
            eps, index_scale, tokens, q_len, init_blocks, local_blocks,
            IndexHeads(ih_count, ih_own),
            (
                ar, res, h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, positions,
                slot_mapping, k_cache, v_cache, k_scale, v_scale, index_cache, q_out,
                iq_out, block_table, seq_lens, iscore, scratch,
            ),
            bt_width,
            index_topk=not dense,
            dense=dense,
            plain_norm=plain_norm,
        )  # fmt: skip

    @launcher(kernel_name)
    def launch_pre_attn(
        ar: Int64,
        res: Int64,
        h_out: Int64,
        g_in: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        g_q: Int64,
        g_k: Int64,
        g_iq: Int64,
        g_ik: Int64,
        cos_sin: Int64,
        positions: Int64,
        slot_mapping: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        index_cache: Int64,
        q_out: Int64,
        iq_out: Int64,
        block_table: Int64,
        seq_lens: Int64,
        bt_width: Int32,
        q_len: Int32,
        iscore: Int64,
        scratch: Int64,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = build_key
        pre_attn_kernel(
            ar, res, h_out, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin,
            positions, slot_mapping, k_cache, v_cache, k_scale, v_scale, index_cache,
            q_out, iq_out, block_table, seq_lens, bt_width, q_len, iscore, scratch,
            layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    K1_ABI.check(pre_attn_kernel, launch_pre_attn)
    return launch_pre_attn
