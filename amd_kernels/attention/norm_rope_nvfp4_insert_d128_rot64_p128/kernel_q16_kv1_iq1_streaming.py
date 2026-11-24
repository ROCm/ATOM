# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``kernel.py``'s one-pair q16 kv1 iq1 rotary-64 bf16-q tile with nt q / index-q stores: the same bytes."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T, Vector as Vec
from .kernel import (
    D,
    BLK,
    NW,
    PRELOAD_HINTS,
    F448,
    F448N,
    _f32,
    _bf16x4,
    _pk_bf16,
    _norm_rope_out4,
    _maxnum,
    _minnum,
    _iq_fp8,
    _quant4,
    _v_row,
    _gptr,
    _k_scale_off,
)


def compile_streaming_q16_kv1_iq1():
    """Compile the fixed q16/kv1/iq1 bf16 one-pair tile with streaming Q stores."""
    nq, nkv, niq, rotary_dim, tp = 16, 1, 1, 64, 1
    NR = nq + nkv + 1 + niq  # slots that norm + rope
    S = NR + nkv * tp  # waves per 2 * tp tokens
    ROW = (nq + 2 * nkv + niq + 1) * D  # qkv row (elements)
    HALF = rotary_dim // 2
    SIDE = nkv * BLK * 72  # one side of a main page (bytes)
    TPW = 2 * tp  # tokens per wave
    Q_STORE_CM = 2  # nt: the TP4 QKV chain gains at 289-384 tokens (README, Performance)
    name = f"norm_rope_nvfp4_insert_d128_p128_q{nq}_kv{nkv}_iq{niq}_rot{rotary_dim}_pairs{tp}_qnt"

    # the first loads' arguments lead: the first 56 bytes arrive preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_qkv: fx.Int64,  # [n, ROW] bf16
        arg_pos: fx.Int64,  # positions [n] i64
        arg_slot: fx.Int64,  # main slots [n] i64, -1 skips
        arg_islot: fx.Int64,  # index slots [n] i64, -1 skips
        arg_cs: fx.Int64,  # cos_sin_cache [i32_cs_rows, rotary_dim] bf16
        arg_qw: fx.Int64,  # norm weights [128] bf16: q, k, index-q, index-k
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_q: fx.Int64,  # q_out [n, nq * 128] bf16
        arg_iq: fx.Int64,  # index_q_out rows of niq * 128 e4m3, row stride i32_iq_st
        arg_kv: fx.Int64,  # main pages [blocks, 2 * nkv, 128, 72] u8, stride i64_kv_st
        arg_idx: fx.Int64,  # index pages [blocks, 128, 72] u8, stride i64_idx_st
        arg_ks: fx.Int64,  # k / v global scales, fp32 scalars
        arg_vs: fx.Int64,
        i64_kv_st: fx.Int64,  # bytes
        i64_idx_st: fx.Int64,  # bytes
        i32_iq_st: fx.Int32,  # bytes
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wid = bx * NW + wave
        s = wid % S
        hl = lane % 32
        hi16 = (hl & 16) != 0
        dim0 = hl * 4
        toks = [(wid // S) * TPW + t * 2 + lane // 32 for t in range_constexpr(tp)]
        lives = [tk < i32_n for tk in toks]

        qkv = buffer_ops.create_buffer_resource_from_addr(
            arg_qkv, num_records_bytes=fx.Int64(i32_n) * (ROW * 2)
        )
        sl_r = buffer_ops.create_buffer_resource_from_addr(
            arg_slot, num_records_bytes=fx.Int64(i32_n) * 8
        )

        if s < NR:
            is_q = s < nq
            is_k = (s >= nq) & (s < nq + nkv)
            is_ik = s == nq + nkv
            is_kq = is_k | is_ik
            # the head's column (in heads) of the qkv row
            col = (s <= nq + nkv).select(
                is_ik.select(fx.Int32(nq + 2 * nkv + niq), s),
                s - (nq + nkv + 1) + (nq + 2 * nkv),
            )
            w_addr = is_q.select(
                arg_qw, is_k.select(arg_kw, is_ik.select(arg_ikw, arg_iqw))
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                w_addr, num_records_bytes=D * 2
            )
            pos_r = buffer_ops.create_buffer_resource_from_addr(
                arg_pos, num_records_bytes=fx.Int64(i32_n) * 8
            )
            isl_r = buffer_ops.create_buffer_resource_from_addr(
                arg_islot, num_records_bytes=fx.Int64(i32_n) * 8
            )
            cs_r = buffer_ops.create_buffer_resource_from_addr(
                arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (rotary_dim * 2)
            )
            xws, poss, slots = [], [], []
            for t in range_constexpr(tp):
                xws.append(
                    Vec(
                        buffer_ops.buffer_load(
                            qkv,
                            toks[t] * (ROW // 2) + col * (D // 2) + hl * 2,
                            vec_width=2,
                            dtype=fx.Int32,
                            mask=lives[t],
                        )
                    )
                )
                poss.append(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            pos_r,
                            toks[t] * 2,
                            vec_width=1,
                            dtype=fx.Int32,
                            mask=lives[t],
                        )
                    )
                )
                kslot = fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r,
                        toks[t] * 2,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lives[t] & is_k,
                    )
                )
                islot = fx.Int32(
                    buffer_ops.buffer_load(
                        isl_r,
                        toks[t] * 2,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lives[t] & is_ik,
                    )
                )
                slots.append(is_k.select(kslot, islot))
            w = _bf16x4(
                Vec(buffer_ops.buffer_load(wr, hl * 2, vec_width=2, dtype=fx.Int32))
            )
            first = dim0 < HALF
            in_rope = dim0 < rotary_dim
            ib = first.select(dim0, dim0 - HALF)
            css = []
            for t in range_constexpr(tp):
                cs_off = (poss[t] * rotary_dim + ib) // 2
                css.append(
                    (
                        _bf16x4(
                            Vec(
                                buffer_ops.buffer_load(
                                    cs_r,
                                    cs_off,
                                    vec_width=2,
                                    dtype=fx.Int32,
                                    mask=lives[t] & in_rope,
                                )
                            )
                        ),
                        _bf16x4(
                            Vec(
                                buffer_ops.buffer_load(
                                    cs_r,
                                    cs_off + HALF // 2,
                                    vec_width=2,
                                    dtype=fx.Int32,
                                    mask=lives[t] & in_rope,
                                )
                            )
                        ),
                    )
                )
            eps = i32_eps.bitcast(fx.Float32)
            qvs = []  # every pair first, so the scheduler can interleave them
            for t in range_constexpr(tp):
                out = _norm_rope_out4(
                    xws[t], css[t], w, eps, hi16, first, in_rope, HALF // 4
                )
                lo, hi = _pk_bf16(out[0], out[1]), _pk_bf16(out[2], out[3])
                qvs.append(Vec.from_elements([lo, hi], fx.Int32))
            if is_q:
                q_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_n) * (nq * D * 2)
                )
                for t in range_constexpr(tp):
                    buffer_ops.buffer_store(
                        qvs[t],
                        q_r,
                        toks[t] * (nq * D // 2) + s * (D // 2) + hl * 2,
                        mask=lives[t],
                        cache_modifier=Q_STORE_CM,
                    )
            if s > nq + nkv:
                iq_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_iq, num_records_bytes=fx.Int64(i32_n) * fx.Int64(i32_iq_st)
                )
                iq_col = (s - (nq + nkv + 1)) * D
                for t in range_constexpr(tp):
                    f = [
                        _minnum(_maxnum(x, _f32(F448N)), _f32(F448))
                        for x in _bf16x4(qvs[t])
                    ]
                    word = _iq_fp8(
                        f[2], f[3], _iq_fp8(f[0], f[1], fx.Int32(0), False), True
                    )
                    buffer_ops.buffer_store(
                        word,
                        iq_r,
                        toks[t] * i32_iq_st + iq_col + dim0,
                        mask=lives[t],
                        offset_is_bytes=True,
                        cache_modifier=Q_STORE_CM,
                    )
            if is_kq:
                gs = is_k.select(
                    fx.Float32(_gptr(arg_ks, T.f32, 4)[0]), fx.Float32(1.0)
                )
                inv_gs = fx.Float32(1.0) / gs
                quants = [
                    _quant4(_bf16x4(qvs[t]), gs, inv_gs)
                    for t in range_constexpr(tp)
                ]
                h = s - nq
                grp = hl // 4
                for t in range_constexpr(tp):
                    slot = slots[t]
                    tp_ = slot & (BLK - 1)
                    page = is_k.select(
                        arg_kv + fx.Int64(slot >> 7) * i64_kv_st,
                        arg_idx + fx.Int64(slot >> 7) * i64_idx_st,
                    )
                    d_off = (
                        is_k.select(h * (BLK * 64), fx.Int32(0)) + tp_ * 64 + hl * 2
                    )
                    s_off = is_k.select(
                        nkv * BLK * 64 + h * (BLK * 8) + _k_scale_off(tp_, grp),
                        BLK * 64 + tp_ * 8 + grp,
                    )
                    d_ptr = _gptr(page + fx.Int64(d_off), T.i16, 2)
                    s_ptr = _gptr(page + fx.Int64(s_off), T.i8, 1)
                    packed = quants[t][0].to(fx.Int16)
                    sf8 = quants[t][1].to(fx.Int8)
                    if lives[t] & (slot >= 0):
                        d_ptr[0] = packed
                        if hl % 4 == 0:
                            s_ptr[0] = sf8
        else:
            vh = (s - NR) // tp
            tok = (wid // S) * TPW + ((s - NR) % tp) * 2 + lane // 32
            live = tok < i32_n
            vcol = (nq + nkv + vh) * (D // 2)  # V head column (dwords)
            vs = fx.Float32(_gptr(arg_vs, T.f32, 4)[0])
            near_t = tok - 7 + hl
            has_near = (hl < 16) & (near_t >= 0) & (near_t < i32_n)
            near = has_near.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, near_t * 2, vec_width=1, dtype=fx.Int32, mask=has_near
                    )
                ),
                fx.Int32(-1),
            )
            slot = fx.Int32(
                buffer_ops.buffer_load(
                    sl_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live
                )
            )
            has_prev = live & (tok > 0)
            prev = has_prev.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, tok * 2 - 2, vec_width=1, dtype=fx.Int32, mask=has_prev
                    )
                ),
                fx.Int32(-1),
            )
            # the rows a run starting here can cover, before its slot is known
            rows = [
                Vec(
                    buffer_ops.buffer_load(
                        qkv,
                        (tok + r) * (ROW // 2) + vcol + hl * 2,
                        vec_width=2,
                        dtype=fx.Int32,
                        mask=tok + r < i32_n,
                    )
                )
                for r in range_constexpr(8)
            ]
            base = slot - slot % 8
            in_oct = (near >= base) & (near < base + 8)
            votes = fx.Int64(fx.rocdl.ballot(T.i64, in_oct))
            mine = (votes >> fx.Int64((lane // 32) * 32)).to(fx.Int32) & 0xFFFF
            n_run = fx.Int32(fx.math.ctpop(mine))
            run_first = (prev < 0) | ((prev >> 3) != (slot >> 3))
            tp_ = slot & (BLK - 1)
            v_side = arg_kv + fx.Int64(slot >> 7) * i64_kv_st + SIDE
            v_data = v_side + fx.Int64(vh * (BLK * 64) + (tp_ // 8) * (D * 4))
            v_sf = v_side + fx.Int64(nkv * BLK * 64 + vh * (BLK * 8) + tp_ - tp_ % 8)
            dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
            sp = _gptr(v_sf + fx.Int64((hl // 4) * BLK), T.i32, 4)
            go = live & (slot >= 0) & run_first
            inv_vs = fx.Float32(1.0) / vs
            # runs of <= 4 tokens (decode) take the short tile
            for run_rows, take in (
                (rows[:4], go & (n_run <= 4)),
                (rows, go & (n_run > 4)),
            ):
                if take:
                    old = [fx.Int32(dp[i]) for i in range_constexpr(4)]
                    old_sc = [fx.Int32(sp[i]) for i in range_constexpr(2)]
                    st = [fx.Int32(0)] * 9
                    for r in range_constexpr(len(run_rows)):
                        st = _v_row(st, run_rows[r], r, n_run, tp_ % 8, vs, inv_vs)
                    keep = ~st[4]
                    for i in range_constexpr(4):
                        dp[i] = st[i] | (old[i] & keep)
                    sc0 = st[5] | (old_sc[0] & ~st[7])
                    sc1 = st[6] | (old_sc[1] & ~st[8])
                    if hl % 4 == 0:
                        sp[0] = sc0
                        sp[1] = sc1

    @flyc.jit
    def launch_streaming_q16_kv1_iq1(
        arg_qkv: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_cs: fx.Int64,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_n: fx.Int32,
        i32_iq_st: fx.Int32,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_cs_rows: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        waves = (i32_n + (TPW - 1)) // TPW * S
        kernel(
            i32_n,
            i32_cs_rows,
            arg_qkv,
            arg_pos,
            arg_slot,
            arg_islot,
            arg_cs,
            arg_qw,
            arg_kw,
            arg_iqw,
            arg_ikw,
            arg_q,
            arg_iq,
            arg_kv,
            arg_idx,
            arg_ks,
            arg_vs,
            i64_kv_st,
            i64_idx_st,
            i32_iq_st,
            i32_eps,
        ).launch(
            grid=(fx.Int64((waves + (NW - 1)) // NW), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_streaming_q16_kv1_iq1.kernel_name = name
    launch_streaming_q16_kv1_iq1.compile_hints = PRELOAD_HINTS
    return launch_streaming_q16_kv1_iq1