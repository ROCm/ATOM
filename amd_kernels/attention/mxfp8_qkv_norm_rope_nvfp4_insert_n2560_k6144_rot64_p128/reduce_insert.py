# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MiniMax-M3.1 QKV split-K reduce + norm/RoPE + NVFP4 insert, fused (gfx950).

One launch in place of the MXFP8 GEMM's split-K reduce (``launch_splitk_reduce``)
AND the standalone ``norm_rope_nvfp4_insert`` (kernel.py): the QKV projection's fp32
partial slab(s) ``[SK, n, ROW]`` are summed in the reduce's order (0 + s0 + s1 +
...), rounded to bf16 exactly as the reduce's store (``.to(bf16)``), and fed to
the standalone kernel's norm / RoPE / NVFP4-insert arithmetic unchanged — bit for bit the
BASE chain ``mxfp8_gemm (bf16 out) -> norm_rope_nvfp4_insert`` for
SK == 1 (the GEMM's out_slab fp32 store) and ``... -> launch_splitk_reduce ->
insert`` for SK > 1. The bf16 QKV intermediate is never written.

Wave / lane structure is the standalone kernel's (kernel.py): a wave covers one
norm / RoPE slot (q / k / index-k / index-q head) of ``2 * tp`` tokens, 32 lanes
of 4 dims each per token. V has workgroups of its own before the norm workgroups,
8 tokens each: the rows of the 8 tokens and of the next 8 (halo) are summed,
rounded and quantized once into LDS, and each octet run's first token packs the
run's codes and scales from LDS with the run protocol's masks.
"""

from .packing import _v_words
from .quant import _quant4_table
from amd_kernels.warm_up import launcher

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import BFloat16, Float32, T
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir.dialects import llvm as _llvm

from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128.kernel_cooperative import (
    _lds_ptr,
    _lds_rmw,
    _lds_wait,
)

from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128.kernel import (
    BLK,
    F448,
    F448N,
    PRELOAD_HINTS,
    D,
    _bf16x4,
    _bits,
    _f32,
    _gptr,
    _iq_fp8,
    _k_scale_off,
    _maxnum,
    _minnum,
    _pk_bf16,
    _quant4,
    _e4m3_value,
    _rsq,
    _xor_lane,
    _xsum,
)

NW = 4  # slots (waves) per workgroup


@launcher
def compile_reduce_insert(tp: int, sk: int):
    """``tp``: token pairs per norm / RoPE wave (1 for decode; prefill keeps more
    loads in flight). ``sk``: the GEMM's split-K (1 = the out_slab fp32 store of
    a no-split row). V runs in workgroups of its own, 8 tokens each: every V wave sums and quantizes its token
    pair's rows and the pair 8 rows on (halo) once into LDS, and a run's first token packs the run from LDS
    (instead of each V lane loading and quantizing the 8 rows a run can cover, as the standalone insert does:
    -1.3..-1.8% on the chain at 65-256 rows, bitwise)."""
    nq, nkv, niq, rotary_dim = 16, 1, 1, 64
    assert tp in (1, 2) and sk in (2, 3)
    assert 1 <= sk <= 8
    NR = nq + nkv + 1 + niq  # slots that norm + rope
    ROW = (nq + 2 * nkv + niq + 1) * D  # qkv row (elements)
    HALF = rotary_dim // 2
    SIDE = nkv * BLK * 72  # one side of a main page (bytes)
    TPW = 2 * tp  # tokens per wave
    name = (
        f"mxfp8_qkv_norm_rope_nvfp4_insert_reduce_d128_p128_q{nq}_kv{nkv}_iq{niq}"
        f"_rot{rotary_dim}_pairs{tp}_s{sk}"
    )
    USE_TABLE = sk == 2
    name += "_vlds_vfirst_vinit_vtranspose"
    if USE_TABLE:
        name += "_vtable_paired_rope"
    # Whole four-head CTAs keep the sixteen q heads together and the three non-q heads together. The last
    # head slot is padding: every load/store that depends on its row uses ``s < NR`` in ``lives`` below.
    SN = 20  # norm / RoPE wave slots per 2 * tp tokens; V workgroups still come first
    name += "_heads20"
    if sk == 3:
        name += "_vhalo_cta"

    @fx.struct
    class Shared:
        vrow: fx.Array[fx.Int32, 16 * 32 * 2]  # a V workgroup's 8 rows + 8 halo rows: (codes, scale byte) per lane
        reciprocal: fx.Array[fx.Int32, 260 if USE_TABLE else 4, 16]
        vcnt: fx.Array[fx.Int32, 4]  # V waves whose rows are in vrow

    # the first loads' arguments lead: the first 56 bytes arrive preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_slab: fx.Int64,  # [sk, n, ROW] fp32 partial sums of the QKV GEMM
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
        # V workgroups first (their run chain is the longest), the norm / RoPE workgroups after them
        n_v_wg = (i32_n + 7) // 8
        wid = (fx.Int32(bx) - n_v_wg) * NW + wave
        s = wid % SN
        hl = lane % 32
        hi16 = (hl & 16) != 0
        dim0 = hl * 4
        toks = [(wid // SN) * TPW + t * 2 + lane // 32 for t in range_constexpr(tp)]
        lives = [(tk < i32_n) & (s < NR) for tk in toks]
        smem = fx.SharedAllocator().allocate(Shared).peek()

        slab = buffer_ops.create_buffer_resource_from_addr(
            arg_slab, num_records_bytes=fx.Int64(i32_n) * (sk * ROW * 4)
        )
        sl_r = buffer_ops.create_buffer_resource_from_addr(
            arg_slot, num_records_bytes=fx.Int64(i32_n) * 8
        )

        def load4(tok, col_elem, live):
            """This lane's 4 fp32 values of every split slab at (token, head-column
            element base)."""
            return [
                Vec(
                    buffer_ops.buffer_load(
                        slab,
                        (k * i32_n + tok) * ROW + col_elem + dim0,
                        vec_width=4,
                        dtype=Float32,
                        mask=live,
                    )
                )
                for k in range_constexpr(sk)
            ]

        def sum4(frags):
            """The split-K slabs summed in the reduce's order (0 + s0 + ... for
            sk > 1; bare for sk == 1, the no-split store), the reduce's RNE ->
            bf16 rounding, the insert's exact bf16 -> fp32 widening."""
            acc = Vec.filled(4, 0.0, Float32)
            for k in range_constexpr(sk):
                acc = frags[k] if sk == 1 else acc + frags[k]
            e = acc.to(BFloat16).to(Float32)
            return [e[i] for i in range_constexpr(4)]

        def row4(tok, col_elem, live):
            """This lane's 4 values at (token, head-column element base), fp32."""
            return sum4(load4(tok, col_elem, live))

        if fx.Int32(bx) >= n_v_wg:
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
            es, poss, slots = [], [], []
            for t in range_constexpr(tp):
                if const_expr(sk == 3):
                    es.append(row4(toks[t], col * D, lives[t]))
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
                slot_value = fx.Int32(-1)
                if const_expr(sk == 2):
                    # Most norm waves are q/index-q and need no slot loads or address arithmetic.
                    if is_kq:
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
                        slot_value = is_k.select(kslot, islot)
                else:
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
                    slot_value = is_k.select(kslot, islot)
                slots.append(slot_value)
            w = _bf16x4(
                Vec(buffer_ops.buffer_load(wr, hl * 2, vec_width=2, dtype=fx.Int32))
            )
            first = dim0 < HALF
            in_rope = dim0 < rotary_dim
            ib = first.select(dim0, dim0 - HALF)
            css = []
            for t in range_constexpr(tp):
                if const_expr(sk == 2):
                    # NeoX pairs (lanes h and h^8) share a cache row. Read cosine in one lane, sine in its
                    # partner, then exchange raw bf16 words; the arithmetic and its rounding remain unchanged.
                    raw = Vec(buffer_ops.buffer_load(
                        cs_r, (poss[t] * rotary_dim + dim0) // 2, vec_width=2,
                        dtype=fx.Int32, mask=lives[t] & in_rope,
                    ))
                    other = [_xor_lane(raw[i], HALF // 4, hi16) for i in range_constexpr(2)]
                    cosv = Vec.from_elements([first.select(raw[i], other[i]) for i in range_constexpr(2)], fx.Int32)
                    sinv = Vec.from_elements([first.select(other[i], raw[i]) for i in range_constexpr(2)], fx.Int32)
                    css.append((_bf16x4(cosv), _bf16x4(sinv)))
                else:
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
            if const_expr(sk == 2):
                # Issue position-dependent cache reads before the slab reads in this specialization.
                es = [row4(toks[t], col * D, lives[t]) for t in range_constexpr(tp)]
            eps = i32_eps.bitcast(fx.Float32)
            qvs = []  # every pair first, so the scheduler can interleave them
            for t in range_constexpr(tp):
                cos, sin = css[t]
                e = es[t]
                ss = e[0] * e[0] + e[1] * e[1]
                ss = ss + e[2] * e[2]
                ss = ss + e[3] * e[3]
                for m in (16, 8, 4, 2, 1):
                    ss = _xsum(ss, m)
                r = _rsq(fx.math.fma(ss, fx.Float32(0.0078125), eps))
                nrm = [
                    (e[i] * r) * (fx.Float32(1.0) + w[i]) for i in range_constexpr(4)
                ]
                out = []
                for i in range_constexpr(4):
                    pr = _xor_lane(_bits(nrm[i]), HALF // 4, hi16).bitcast(fx.Float32)
                    ps = pr * sin[i]
                    roped = fx.math.fma(nrm[i], cos[i], first.select(-ps, ps))
                    out.append(in_rope.select(roped, nrm[i]))
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
                    )
            if is_kq:
                gs = is_k.select(
                    fx.Float32(_gptr(arg_ks, T.f32, 4)[0]), fx.Float32(1.0)
                )
                inv_gs = fx.Float32(1.0) / gs
                quants = [
                    _quant4(_bf16x4(qvs[t]), gs, inv_gs) for t in range_constexpr(tp)
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
                    d_off = is_k.select(h * (BLK * 64), fx.Int32(0)) + tp_ * 64 + hl * 2
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
            # Only V workgroups exchange rows through LDS; norm workgroups need no CTA join.
            if const_expr(USE_TABLE):
                if tx == 0:
                    smem.vcnt.ptr[0] = fx.Int32(0)
            vbase = fx.Int32(bx) * 8
            tok = vbase + wave * 2 + lane // 32
            live = tok < i32_n
            vcol = (nq + nkv) * D  # the V head (TP4: one)
            vs = fx.Float32(_gptr(arg_vs, T.f32, 4)[0])
            inv_vs = fx.Float32(1.0) / vs
            fr_own = load4(tok, vcol, live)
            if const_expr(sk == 2):
                fr_halo = load4(tok + 8, vcol, tok + 8 < i32_n)
            near_t = tok - 7 + hl
            has_near = (hl < 16) & (near_t >= 0) & (near_t < i32_n)
            near = has_near.select(
                fx.Int32(buffer_ops.buffer_load(sl_r, near_t * 2, vec_width=1, dtype=fx.Int32, mask=has_near)),
                fx.Int32(-1),
            )
            if const_expr(sk == 3):
                # Every wave's neighbour vector includes the same two owner-boundary slots. Different octets
                # prove that no owned contiguous run reaches the halo. Keep owned slab prefetch unconditional.
                boundary_lane = fx.Int32(14) - wave * 2
                boundary0 = fx.Int32(fx.rocdl.readlane(T.i32, near, boundary_lane))
                boundary1 = fx.Int32(fx.rocdl.readlane(T.i32, near, boundary_lane + 1))
                use_halo = (boundary0 >= 0) & (boundary1 >= 0) & ((boundary0 >> 3) == (boundary1 >> 3))
                fr_halo = [Vec.filled(4, 0.0, Float32) for _ in range_constexpr(sk)]
                if use_halo:
                    fr_halo = load4(tok + 8, vcol, tok + 8 < i32_n)
            slot = fx.Int32(buffer_ops.buffer_load(sl_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live))
            has_prev = live & (tok > 0)
            prev = has_prev.select(
                fx.Int32(buffer_ops.buffer_load(sl_r, tok * 2 - 2, vec_width=1, dtype=fx.Int32, mask=has_prev)),
                fx.Int32(-1),
            )
            base = slot - slot % 8
            in_oct = (near >= base) & (near < base + 8)
            votes = fx.Int64(fx.rocdl.ballot(T.i64, in_oct))
            mine = (votes >> fx.Int64((lane // 32) * 32)).to(fx.Int32) & 0xFFFF
            n_run = fx.Int32(fx.math.ctpop(mine))
            run_first = (prev < 0) | ((prev >> 3) != (slot >> 3))
            go = live & (slot >= 0) & run_first
            # a lane that writes nothing reads page 0 (only V workgroups write these dwords)
            wslot = go.select(slot, fx.Int32(0))
            tp_ = wslot & (BLK - 1)
            v_side = arg_kv + fx.Int64(wslot >> 7) * i64_kv_st + SIDE
            v_data = v_side + fx.Int64((tp_ // 8) * (D * 4))
            v_sf = v_side + fx.Int64(nkv * BLK * 64 + tp_ - tp_ % 8)
            dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
            sp = _gptr(v_sf + fx.Int64((hl // 4) * BLK), T.i32, 4)
            vold = [fx.Int32(dp[i]) for i in range_constexpr(4)]
            vold_sc = [fx.Int32(sp[i]) for i in range_constexpr(2)]
            vrow = fx.Int32(fx.ptrtoint(smem.vrow.ptr))
            vcnt = fx.Int32(fx.ptrtoint(smem.vcnt.ptr))
            if const_expr(USE_TABLE):
                # Stage the exact scale reciprocals while slab and kept-page loads are in flight. This
                # V-only join also publishes the zeroed row counter; SK3 quantizes with ``_quant4`` instead.
                sv = _e4m3_value(tx)
                rec = (sv != 0.0).select(fx.Float32(1.0) / (vs * sv), fx.Float32(0.0))
                smem.reciprocal.ptr[tx] = rec.bitcast(fx.Int32)
                if tx == 0:
                    smem.reciprocal.ptr[256] = inv_vs.bitcast(fx.Int32)
                fx.gpu.barrier()
            for p in range_constexpr(2):
                tk = tok + p * 8
                fr = fr_own if p == 0 else fr_halo
                if const_expr(sk == 2):
                    need = True
                else:
                    need = (p == 0) or use_halo
                if need:
                    if const_expr(USE_TABLE):
                        nib_o, sf_o = _quant4_table(sum4(fr), smem.reciprocal.ptr)
                    else:
                        nib_o, sf_o = _quant4(sum4(fr), vs, inv_vs)
                    _llvm.StoreOp(fx.as_ir_value(Vec.from_elements([fx.Int32(nib_o), fx.Int32(sf_o)], fx.Int32)),
                                  _lds_ptr(vrow + ((tk - vbase) * 32 + hl) * 8), alignment=8)
            if const_expr(sk == 3):
                # All four waves of a V workgroup participate, so a CTA join publishes every owned/halo row.
                fx.gpu.barrier()
            else:
                if lane == 0:
                    _lds_rmw(vcnt, 1, _llvm.AtomicOrdering.release)
                _lds_wait(vcnt, NW)
            for run_rows, take in ((4, go & (n_run <= 4)), (8, go & (n_run > 4))):
                if take:
                    nibbles = []
                    scales = fx.Uint64(0)
                    for r in range_constexpr(run_rows):
                        got = Vec(_llvm.load(T.vec(2, T.i32), _lds_ptr(vrow + ((tok + r - vbase) * 32 + hl) * 8),
                                             alignment=8))
                        ok = (tok + r < i32_n) & (r < n_run)
                        nib = ok.select(got[0], fx.Int32(0))
                        sf = ok.select(got[1], fx.Int32(0))
                        nibbles.append(nib)
                        scales = scales | (sf.to(fx.Uint64) << (8 * r))
                    words = _v_words(nibbles)
                    p0 = tp_ % 8
                    mask = (fx.Uint32(0xFFFFFFFF) >> ((8 - n_run) * 4)) << (p0 * 4)
                    for i in range_constexpr(4):
                        dp[i] = (words[i] << (p0 * 4)) | (vold[i] & ~mask.to(fx.Int32))
                    smask = (fx.Uint64(0xFFFFFFFFFFFFFFFF) >> ((8 - n_run) * 8)) << (p0 * 8)
                    scales = scales << (p0 * 8)
                    sc0 = scales.to(fx.Int32) | (vold_sc[0] & ~smask.to(fx.Int32))
                    sc1 = (scales >> 32).to(fx.Int32) | (vold_sc[1] & ~(smask >> 32).to(fx.Int32))
                    if hl % 4 == 0:
                        sp[0] = sc0
                        sp[1] = sc1

    @flyc.jit
    def launch_qkv_reduce_insert(
        arg_slab: fx.Int64,
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
        waves = (i32_n + (TPW - 1)) // TPW * SN
        kernel(
            i32_n,
            i32_cs_rows,
            arg_slab,
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
            grid=(fx.Int64((waves + (NW - 1)) // NW + (i32_n + 7) // 8), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_qkv_reduce_insert.kernel_name = name
    launch_qkv_reduce_insert.compile_hints = PRELOAD_HINTS
    return launch_qkv_reduce_insert
