# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""QKV chain at 129-192 rows with the deferred GEMM's two-way K split and three K-waves per split.

Three matrix waves compute the same sequential K chains as the deferred GEMM, park their accumulators in LDS,
and reduce them in wave order. Four epilogue waves preload metadata concurrently; tail wave i takes its owner's
token pairs 2i and 2i + 1 (V: and halo rows 16 + 2i, 17 + 2i). Each (row block, head) has
four workgroups; after their sc1 slab stores, every workgroup waits for all four and owns 12 or 16 output rows.
V owners quantize 24 rows, including the halo needed by a run crossing an owner boundary. Norm and output masks
keep each owner's extra padded rows from writing. Positive producer increments sum to a weighted epoch of 16.
Owners wait for all 16, including every row block for V halos; nobody tops up, counts completions or resets. The host
assigns separate counters to each stream; each row block has its own, the V head one over all blocks.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import (
    PRELOAD_HINTS,
    W_CACHE_MOD,
    _addr,
    _lookahead,
    _mfma,
    _pack8,
    _pin_accumulators,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128.kernel import (
    BLK,
    F448,
    F448N,
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
    _rsq,
    _xor_lane,
    _xsum,
)
from .packing import _v_expand, _v_expanded_words
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import BFloat16, Float32, T
from flydsl.expr.typing import Vector as Vec

N, K = 2560, 6144
NQ, NKV, NIQ, ROT = 16, 1, 1, 64
RT, NI, KW, KS = 4, 4, 1, 2  # the deferred 1-16-row GEMM row
NB = N // (16 * NI)  # 80 column tiles
HEADS = N // D  # 20
TILES_PER_HEAD = D // (16 * NI)  # 4
WG_PER_HEAD = TILES_PER_HEAD * KS  # 12
SC1 = 0x10  # agent-scope (cross-XCD) stores / loads of the slabs
CNT_STRIDE = 32  # one counter per head, each on its own 128-byte line
EPOCH = 16  # what one call adds to each counter it uses (a power of 2 >= its producer count)
NR = NQ + NKV + 1 + NIQ  # norm / RoPE slots: 19
S = NR + NKV  # reduce-insert wave slots per token pair (one pair per wave): 20
ROW = N
HALF = ROT // 2
SIDE = NKV * BLK * 72


def _atomic_add(addr, v):
    """Returning scalar global RMW, called only by the epoch publisher/poller (one active lane).

    FlyDSL has no SMEM atomic wrapper. Native assembly fixes the SGPR instruction and return acknowledgement.
    The caller drains coherent slab stores before publication; the memory clobber prevents compiler reordering.
    """
    low = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = fx.Int32(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    inc = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))
    return fx.Int32(_llvm.inline_asm(T.i32, [fx.as_ir_value(inc), fx.as_ir_value(pointer)],
                     "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)", "=s,0,s,~{memory}",
                     has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT))


def _lds_ptr(addr):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(addr)))


def _lds_rmw(addr, v, order):
    """An LDS atomic add (workgroup scope) returning the old value."""
    return fx.Int32(_llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, _lds_ptr(addr), fx.Int32(v).ir_value(), order,
                                      syncscope="workgroup", alignment=4).result)


@flyc.jit
def _lds_wait(addr, target):
    """Spin until the LDS counter at ``addr`` reaches ``target`` (acquire: later LDS reads stay after it)."""
    f = _lds_rmw(addr, 0, _llvm.AtomicOrdering.acquire)
    while f < fx.Int32(target):
        f = _lds_rmw(addr, 0, _llvm.AtomicOrdering.acquire)


@launcher
def compile_qkv_k_waves(n_tokens: int):
    """The 48/64-row tiles preserve the deferred GEMM's K-wave sum and split order."""
    assert 129 <= n_tokens <= 192
    RT = 3 if n_tokens <= 144 else 4
    KW, NWG, PW, pairs = 3, 7, 2, 8
    OWNERS, ROWS_PER_OWNER = 4, RT * 4
    PRE_SCALES, PRE_WORDS = False, 10
    # Cold graph sweeps, including the ISA register gate: 188 / 184 VGPRs, no spills.
    law, laa, wcm = (1, 4, 0) if RT == 3 else (1, 2, 2)
    KSTEPS = K // 128
    SW = KSTEPS // (KS * KW)  # 16 K steps per workgroup
    SCB = K // 256
    LA_W, LA_A = _lookahead(SW, RT, NI, 1, 1)
    if law:
        LA_W, LA_A = law, laa
    NA = SW * RT
    name = f"mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128_ks2_kw{KW}_cooperative_tp{pairs}_rt{RT}_own{OWNERS}_wc{wcm}_sb_nw{NWG}_la{LA_W}x{LA_A}_vall_ep16_pairrun"

    name += "_vexpand_weighted"

    name += "_satomic"

    @fx.struct
    class Shared:
        ticket: fx.Array[fx.Int32, 4]
        gacc: fx.Array[fx.Int32, KW * RT * NI * 64 * 4, 16]
        pre: fx.Array[fx.Int32, 4 * PW * PRE_WORDS * 64]
        vrow: fx.Array[fx.Int32, 24 * 32 * 2]  # the V head's rows quantized: (codes, scale byte) per token and lane
        vcnt: fx.Array[fx.Int32, 4]  # V tail waves whose rows are in vrow

    @flyc.kernel(name=name, known_block_size=[64 * NWG, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_as: fx.Int64,
        arg_a: fx.Int64,
        arg_slab: fx.Int64,  # [3, m, 2560] fp32 partials (workspace)
        arg_cnt: fx.Int64,  # [20 * CNT_STRIDE] i32 head counters, a multiple of EPOCH between calls
        i32_cs_rows: fx.Int32,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_cs: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_iq_st: fx.Int32,
        i32_eps: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        mb = bx // (KS * NB)
        kz, nb = (bx % (KS * NB)) // NB, bx % NB
        tile_row = mb * (RT * 16)
        head = nb // TILES_PER_HEAD
        row_base = tile_row + (kz * TILES_PER_HEAD + nb % TILES_PER_HEAD - (WG_PER_HEAD - OWNERS)) * ROWS_PER_OWNER
        lane_off = q16 * 256 + l16 * 16
        k_base = (kz * KW + wave) * (SW * 2048)
        slab = buffer_ops.create_buffer_resource_from_addr(arg_slab, num_records_bytes=i32_m * (N * 4 * KS))

        def gemm():
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_a, num_records_bytes=((i32_m + RT*16-1) // (RT*16) * (RT*16)) * K)
            w_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K)
            as_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_as, num_records_bytes=a_scale_bytes)
            ws_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_ws, num_records_bytes=(N // 32) * SCB * 256)
            n16 = [nb * NI + ni for ni in range_constexpr(NI)]
            w_dw = [(n16[ni] * (16 * K) + k_base + lane_off) // 4 for ni in range_constexpr(NI)]
            a_dw = [((tile_row + mi * 16) * K + k_base + lane_off) // 4 for mi in range_constexpr(RT)]
            sc_lane = q16 * 16 + l16
            sc_k = (kz * KW + wave) * (SW // 2 * 64)
            ws_dw = ((n16[0] // 2) * SCB) * 64 + sc_k + sc_lane
            as_dw = (mb * RT // 2 * SCB * 64) + sc_k + sc_lane

            # A rows past m are the producer's padding: an output row reads only its own A row and rows past m are
            # never stored, so their loads are masked off (half of A's bytes at 8 rows)
            a_live = l16 < i32_m

            def load_pair(rsrc, dw, s, nt, mi=0):
                kw = dict(vec_width=4, dtype=fx.Int32)
                if const_expr(nt):
                    kw["cache_modifier"] = wcm
                else:
                    kw["mask"] = tile_row + l16 + mi * 16 < i32_m
                return _pack8(
                    buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048, **kw),
                    buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048 + 1024, **kw),
                )

            def load_scale(rsrc, dw, s):
                return fx.Int32(buffer_ops.buffer_load(rsrc, dw, vec_width=1, dtype=fx.Int32,
                                                       soffset_bytes=(s // 2) * 256))

            def load_w_step(s):
                bb = [load_pair(w_rsrc, w_dw[ni], s, True) for ni in range_constexpr(NI)]
                if const_expr(s % 2 == 0):
                    sw = [load_scale(ws_rsrc, ws_dw + cp * SCB * 64, s) for cp in range_constexpr(NI // 2)]
                    raw = [load_scale(as_rsrc, as_dw + rp * SCB * 64, s) for rp in range_constexpr((RT + 1) // 2)]
                    if const_expr(RT % 2):
                        odd = mb % 2 != 0
                        sa = []
                        for rp in range_constexpr((RT + 1) // 2):
                            nxt = raw[min(rp + 1, (RT + 1) // 2 - 1)]
                            fixed = ((raw[rp] >> 8) & 0x00FF00FF) | ((nxt & 0x00FF00FF) << 8)
                            sa.append(odd.select(fixed, raw[rp]))
                    else:
                        sa = raw
                    return bb, (sw, sa)
                return bb, None

            acc = [None] * (NI * RT)
            w_q = [None] * SW
            a_q = [None] * NA
            for s in range_constexpr(LA_W):
                w_q[s] = load_w_step(s)
            for t in range_constexpr(LA_A):
                a_q[t] = load_pair(a_rsrc, a_dw[t % RT], t // RT, False, t % RT)
            fx.rocdl.sched_barrier(0)
            sw = sa = None
            for s in range_constexpr(SW):
                if const_expr(s + LA_W < SW):
                    w_q[s + LA_W] = load_w_step(s + LA_W)
                bb, sc = w_q[s]
                if const_expr(s % 2 == 0):
                    sw, sa = sc
                for mi in range_constexpr(RT):
                    t = s * RT + mi
                    if const_expr(t + LA_A < NA):
                        p = t + LA_A
                        a_q[p] = load_pair(a_rsrc, a_dw[p % RT], p // RT, False, p % RT)
                    a_op = a_q[t]
                    a_q[t] = None
                    for ni in range_constexpr(NI):
                        ai = mi * NI + ni
                        acc[ai] = _mfma(bb[ni], a_op, acc[ai], sw[ni // 2], sa[mi // 2], ni % 2, mi % 2, s % 2)
                    fx.rocdl.sched_barrier(0)
                w_q[s] = None
            flat = _pin_accumulators(acc)
            for i in range_constexpr(RT * NI):
                _llvm.StoreOp(fx.as_ir_value(Vec(flat[i]).bitcast(fx.Int32)), _lds_ptr(
                    fx.Int32(fx.ptrtoint(smem.gacc.ptr)) + ((wave * RT * NI + i) * 64 + lane) * 16), alignment=16)


        def tail_pre(s, pair):
            """Every load of the reduce-insert wave (slot ``s``, one token pair) that does not read the slabs, issued
            while wave 0 runs the GEMM (the same addresses and masks as the reduce-insert's)."""
            L_i32_m = i32_m
            L_i32_cs_rows = i32_cs_rows
            L_arg_pos = arg_pos
            L_arg_slot = arg_slot
            L_arg_islot = arg_islot
            L_arg_cs = arg_cs
            L_arg_qw = arg_qw
            L_arg_kw = arg_kw
            L_arg_ks = arg_ks
            L_arg_vs = arg_vs
            L_arg_iqw = arg_iqw
            L_arg_ikw = arg_ikw
            L_lane = lane
            hl = L_lane % 32
            dim0 = hl * 4
            tok = pair * 2 + L_lane // 32
            live = tok < L_i32_m
            is_n = s < NR
            is_v = s >= NR
            is_q = s < NQ
            is_k = (s >= NQ) & (s < NQ + NKV)
            is_ik = s == NQ + NKV
            sl_r = buffer_ops.create_buffer_resource_from_addr(L_arg_slot, num_records_bytes=fx.Int64(L_i32_m) * 8)
            pos_r = buffer_ops.create_buffer_resource_from_addr(L_arg_pos, num_records_bytes=fx.Int64(L_i32_m) * 8)
            isl_r = buffer_ops.create_buffer_resource_from_addr(L_arg_islot, num_records_bytes=fx.Int64(L_i32_m) * 8)
            cs_r = buffer_ops.create_buffer_resource_from_addr(
                L_arg_cs, num_records_bytes=fx.Int64(L_i32_cs_rows) * (ROT * 2))
            w_addr = is_q.select(L_arg_qw, is_k.select(L_arg_kw, is_ik.select(L_arg_ikw, L_arg_iqw)))
            wr = buffer_ops.create_buffer_resource_from_addr(w_addr, num_records_bytes=D * 2)
            pos = fx.Int32(buffer_ops.buffer_load(pos_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live & is_n))
            mslot = fx.Int32(buffer_ops.buffer_load(sl_r, tok * 2, vec_width=1, dtype=fx.Int32,
                                                    mask=live & (is_k | is_v)))
            islot = fx.Int32(buffer_ops.buffer_load(isl_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live & is_ik))
            wv = Vec(buffer_ops.buffer_load(wr, hl * 2, vec_width=2, dtype=fx.Int32))
            first = dim0 < HALF
            in_rope = dim0 < ROT
            ib = first.select(dim0, dim0 - HALF)
            cs_off = (pos * ROT + ib) // 2
            cosv = Vec(buffer_ops.buffer_load(cs_r, cs_off, vec_width=2, dtype=fx.Int32, mask=live & in_rope & is_n))
            sinv = Vec(buffer_ops.buffer_load(cs_r, cs_off + HALF // 2, vec_width=2, dtype=fx.Int32,
                                              mask=live & in_rope & is_n))
            near_t = tok - 7 + hl
            has_near = is_v & (hl < 16) & (near_t >= 0) & (near_t < L_i32_m)
            near = has_near.select(
                fx.Int32(buffer_ops.buffer_load(sl_r, near_t * 2, vec_width=1, dtype=fx.Int32, mask=has_near)),
                fx.Int32(-1),
            )
            has_prev = is_v & live & (tok > 0)
            prev = has_prev.select(
                fx.Int32(buffer_ops.buffer_load(sl_r, tok * 2 - 2, vec_width=1, dtype=fx.Int32, mask=has_prev)),
                fx.Int32(-1),
            )
            if const_expr(PRE_SCALES):  # the global K / V scales, so the tail's quantizers do not wait for them
                ksb = fx.Float32(_gptr(L_arg_ks, T.f32, 4)[0]).bitcast(fx.Int32)
                vsb = fx.Float32(_gptr(L_arg_vs, T.f32, 4)[0]).bitcast(fx.Int32)
                return [mslot, islot, wv, cosv, sinv, near, prev, ksb, vsb]
            return [mslot, islot, wv, cosv, sinv, near, prev]

        def tail_post(s, pairs, pres):
            """The reduce-insert wave's slab reads and arithmetic (slot ``s``) for this wave's token pairs (loads of
            every pair first, then the arithmetic, then the stores), slabs at agent scope."""
            L_i32_m = i32_m
            L_i32_eps = i32_eps
            L_i32_iq_st = i32_iq_st
            L_i64_kv_st = i64_kv_st
            L_i64_idx_st = i64_idx_st
            L_arg_q = arg_q
            L_arg_iq = arg_iq
            L_arg_kv = arg_kv
            L_arg_idx = arg_idx
            L_arg_ks = arg_ks
            L_arg_vs = arg_vs
            L_slab = slab
            L_lane = lane
            L_row_base = row_base
            L_wave = wave
            L_vrow = fx.Int32(fx.ptrtoint(smem.vrow.ptr))
            L_vcnt = fx.Int32(fx.ptrtoint(smem.vcnt.ptr))
            hl = L_lane % 32
            hi16 = (hl & 16) != 0
            dim0 = hl * 4
            toks = [pair * 2 + L_lane // 32 for pair in pairs]
            lives = [(tk < L_i32_m) & (tk < L_row_base + ROWS_PER_OWNER) for tk in toks]
            eps = L_i32_eps.bitcast(fx.Float32)
            TP = len(pairs)

            def load4(tok, col_elem, live):
                return [
                    Vec(buffer_ops.buffer_load(L_slab, (k * L_i32_m + tok) * ROW + col_elem + dim0, vec_width=4,
                                               dtype=Float32, mask=live, cache_modifier=SC1))
                    for k in range_constexpr(KS)
                ]

            def sum4(frags):
                acc = Vec.filled(4, 0.0, Float32)
                for k in range_constexpr(KS):
                    acc = acc + frags[k]
                e = acc.to(BFloat16).to(Float32)
                return [e[i] for i in range_constexpr(4)]

            if s < NR:
                is_q = s < NQ
                is_k = (s >= NQ) & (s < NQ + NKV)
                is_ik = s == NQ + NKV
                is_kq = is_k | is_ik
                col = (s <= NQ + NKV).select(
                    is_ik.select(fx.Int32(NQ + 2 * NKV + NIQ), s),
                    s - (NQ + NKV + 1) + (NQ + 2 * NKV),
                )
                frags = [load4(toks[t], col * D, lives[t]) for t in range_constexpr(TP)]
                first = dim0 < HALF
                in_rope = dim0 < ROT
                qvs, slots = [], []
                for t in range_constexpr(TP):
                    mslot, islot, wv, cosv, sinv, near, prev = pres[t][:7]
                    e = sum4(frags[t])
                    slots.append(is_k.select(mslot, islot))
                    w = _bf16x4(wv)
                    cos, sin = _bf16x4(cosv), _bf16x4(sinv)
                    ss = e[0] * e[0] + e[1] * e[1]
                    ss = ss + e[2] * e[2]
                    ss = ss + e[3] * e[3]
                    for m in (16, 8, 4, 2, 1):
                        ss = _xsum(ss, m)
                    r = _rsq(fx.math.fma(ss, fx.Float32(0.0078125), eps))
                    nrm = [(e[i] * r) * (fx.Float32(1.0) + w[i]) for i in range_constexpr(4)]
                    out = []
                    for i in range_constexpr(4):
                        pr = _xor_lane(_bits(nrm[i]), HALF // 4, hi16).bitcast(fx.Float32)
                        ps = pr * sin[i]
                        roped = fx.math.fma(nrm[i], cos[i], first.select(-ps, ps))
                        out.append(in_rope.select(roped, nrm[i]))
                    lo, hi = _pk_bf16(out[0], out[1]), _pk_bf16(out[2], out[3])
                    qvs.append(Vec.from_elements([lo, hi], fx.Int32))
                if is_q:
                    q_r = buffer_ops.create_buffer_resource_from_addr(L_arg_q,
                                                                      num_records_bytes=fx.Int64(L_i32_m) * (NQ * D * 2))
                    for t in range_constexpr(TP):
                        buffer_ops.buffer_store(qvs[t], q_r, toks[t] * (NQ * D // 2) + s * (D // 2) + hl * 2,
                                                mask=lives[t])
                if s > NQ + NKV:
                    iq_r = buffer_ops.create_buffer_resource_from_addr(
                        L_arg_iq, num_records_bytes=fx.Int64(L_i32_m) * fx.Int64(L_i32_iq_st))
                    iq_col = (s - (NQ + NKV + 1)) * D
                    for t in range_constexpr(TP):
                        f = [_minnum(_maxnum(x, _f32(F448N)), _f32(F448)) for x in _bf16x4(qvs[t])]
                        word = _iq_fp8(f[2], f[3], _iq_fp8(f[0], f[1], fx.Int32(0), False), True)
                        buffer_ops.buffer_store(word, iq_r, toks[t] * L_i32_iq_st + iq_col + dim0, mask=lives[t],
                                                offset_is_bytes=True)
                if is_kq:
                    if const_expr(PRE_SCALES):
                        ks = pres[0][7].bitcast(fx.Float32)
                    else:
                        ks = fx.Float32(_gptr(L_arg_ks, T.f32, 4)[0])
                    gs = is_k.select(ks, fx.Float32(1.0))
                    inv_gs = fx.Float32(1.0) / gs
                    quants = [_quant4(_bf16x4(qvs[t]), gs, inv_gs) for t in range_constexpr(TP)]
                    h = s - NQ
                    grp = hl // 4
                    for t in range_constexpr(TP):
                        slot = slots[t]
                        tp_ = slot & (BLK - 1)
                        page = is_k.select(L_arg_kv + fx.Int64(slot >> 7) * L_i64_kv_st,
                                           L_arg_idx + fx.Int64(slot >> 7) * L_i64_idx_st)
                        d_off = is_k.select(h * (BLK * 64), fx.Int32(0)) + tp_ * 64 + hl * 2
                        s_off = is_k.select(NKV * BLK * 64 + h * (BLK * 8) + _k_scale_off(tp_, grp),
                                            BLK * 64 + tp_ * 8 + grp)
                        d_ptr = _gptr(page + fx.Int64(d_off), T.i16, 2)
                        s_ptr = _gptr(page + fx.Int64(s_off), T.i8, 1)
                        packed = quants[t][0].to(fx.Int16)
                        sf8 = quants[t][1].to(fx.Int8)
                        if lives[t] & (slot >= 0):
                            d_ptr[0] = packed
                            if hl % 4 == 0:
                                s_ptr[0] = sf8
            else:
                vh = 0
                vcol = (NQ + NKV + vh) * D
                if const_expr(PRE_SCALES):
                    vs = pres[0][8].bitcast(fx.Float32)
                else:
                    vs = fx.Float32(_gptr(L_arg_vs, T.f32, 4)[0])
                inv_vs = fx.Float32(1.0) / vs
                # each V wave sums, rounds and quantizes its own token pairs' rows (3 slab loads and one ``_quant4``
                # a lane and pair, as a q wave) into LDS; once all tail waves have, the first token of a run packs
                # the run's codes and scales from LDS (the reduce-insert's V wave loads the 8 rows a run can cover,
                # 24 slab vectors a lane, and quantizes them one after the other: the tail's long pole)
                # the halo pair: owner rows 16 + 2i (+1) of tail wave i, so V quantizes rows 0-23 of the owner
                v_toks = toks + [L_row_base + 16 + 2 * (L_wave - KW) + L_lane // 32]
                frags = [load4(tk, vcol, tk < L_i32_m) for tk in v_toks]
                # the runs (their slots were preloaded) and the octet dwords they keep, read before the wait so the
                # page round trip overlaps it; a lane that writes nothing reads page 0 (only this V tail writes these
                # dwords, so the values are the ones the reduce-insert reads right before its stores)
                runs = []
                for t in range_constexpr(TP):
                    mslot, islot, wv, cosv, sinv, near, prev = pres[t][:7]
                    slot = mslot
                    base = slot - slot % 8
                    in_oct = (near >= base) & (near < base + 8)
                    votes = fx.Int64(fx.rocdl.ballot(T.i64, in_oct))
                    mine = (votes >> fx.Int64((L_lane // 32) * 32)).to(fx.Int32) & 0xFFFF
                    n_run = fx.Int32(fx.math.ctpop(mine))
                    run_first = (prev < 0) | ((prev >> 3) != (slot >> 3))
                    go = lives[t] & (slot >= 0) & run_first
                    wslot = go.select(slot, fx.Int32(0))
                    tp_ = wslot & (BLK - 1)
                    v_side = L_arg_kv + fx.Int64(wslot >> 7) * L_i64_kv_st + SIDE
                    v_data = v_side + fx.Int64(vh * (BLK * 64) + (tp_ // 8) * (D * 4))
                    v_sf = v_side + fx.Int64(NKV * BLK * 64 + vh * (BLK * 8) + tp_ - tp_ % 8)
                    dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
                    sp = _gptr(v_sf + fx.Int64((hl // 4) * BLK), T.i32, 4)
                    old = [fx.Int32(dp[i]) for i in range_constexpr(4)] + [fx.Int32(sp[i]) for i in range_constexpr(2)]
                    runs.append((n_run, go, tp_, dp, sp, old))
                for t in range_constexpr(TP + 1):
                    nib_o, sf_o = _quant4(sum4(frags[t]), vs, inv_vs)
                    _llvm.StoreOp(fx.as_ir_value(Vec.from_elements([_v_expand(fx.Int32(nib_o)), fx.Int32(sf_o)], fx.Int32)),
                                  _lds_ptr(L_vrow + ((v_toks[t] - L_row_base) * 32 + hl) * 8), alignment=8)
                if L_lane == 0:
                    _lds_rmw(L_vcnt, TP + 1, _llvm.AtomicOrdering.release)
                _lds_wait(L_vcnt, 4 * (TP + 1))
                for t in range_constexpr(TP):
                    tok = toks[t]
                    n_run, go, tp_, dp, sp, oldv = runs[t]
                    for run_rows, take in ((4, go & (n_run <= 4)), (8, go & (n_run > 4))):
                        if take:
                            old = oldv[:4]
                            old_sc = oldv[4:6]
                            nibbles = []
                            scales = fx.Uint64(0)
                            for r in range_constexpr(run_rows):
                                got = Vec(_llvm.load(T.vec(2, T.i32),
                                                     _lds_ptr(L_vrow + ((tok + r - L_row_base) * 32 + hl) * 8), alignment=8))
                                ok = (tok + r < L_i32_m) & (r < n_run)
                                nib = ok.select(got[0], fx.Int32(0))
                                sf = ok.select(got[1], fx.Int32(0))
                                nibbles.append(nib)
                                scales = scales | (sf.to(fx.Uint64) << (8 * r))
                            words = _v_expanded_words(nibbles)
                            p0 = tp_ % 8
                            mask = (fx.Uint32(0xFFFFFFFF) >> ((8 - n_run) * 4)) << (p0 * 4)
                            for i in range_constexpr(4):
                                dp[i] = (words[i] << (p0 * 4)) | (old[i] & ~mask.to(fx.Int32))
                            smask = (fx.Uint64(0xFFFFFFFFFFFFFFFF) >> ((8 - n_run) * 8)) << (p0 * 8)
                            scales = scales << (p0 * 8)
                            sc0 = scales.to(fx.Int32) | (old_sc[0] & ~smask.to(fx.Int32))
                            sc1 = (scales >> 32).to(fx.Int32) | (old_sc[1] & ~(smask >> 32).to(fx.Int32))
                            if hl % 4 == 0:
                                sp[0] = sc0
                                sp[1] = sc1

        # head -> the reduce-insert's slot: q h, k 16, v 19, index-q 18, index-k 17
        slot_s = (head <= NQ).select(head, (head == NQ + 1).select(
            fx.Int32(NR), (head == NQ + 2).select(fx.Int32(NQ + NKV + 1), fx.Int32(NQ + NKV))))
        # A V owner's halo rows past its row block come from the next block's producers, so the V head has one
        # counter over all row blocks: every V owner waits for every V producer (a per-block counter lets an octet
        # run crossing a block boundary pack slab rows not yet stored: 271-417 page bytes wrong at 129-192 when the
        # next block's V workgroups start late)
        is_vh = head == NQ + 1
        nmb = (i32_m + RT * 16 - 1) // (RT * 16)
        cnt_addr = arg_cnt + fx.Int64(is_vh.select(fx.Int32(0), fx.Int32(mb)) * HEADS + head) * (CNT_STRIDE * 4)
        need = is_vh.select(nmb * WG_PER_HEAD, fx.Int32(WG_PER_HEAD))
        if tx == KW * 64:
            smem.vcnt.ptr[0] = fx.Int32(0)
        if wave < KW:
            gemm()
        else:
            # the tail's slab-independent loads while wave 0 runs the GEMM, parked in this wave's LDS slot
            for p in range_constexpr(PW):
                pre0 = tail_pre(slot_s, row_base // 2 + 2 * (wave - KW) + p)
                words0 = [pre0[0], pre0[1], *[Vec(pre0[k])[i] for k in (2, 3, 4) for i in range(2)], *pre0[5:]]
                for k in range_constexpr(PRE_WORDS):
                    smem.pre.ptr[(((wave - KW) * PW + p) * PRE_WORDS + k) * 64 + lane] = fx.Int32(words0[k])
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        if wave < KW:
            for i in range_constexpr(RT * NI):
                if wave == i % KW:
                    ps = [Vec(_llvm.load(T.vec(4, T.i32), _lds_ptr(
                        fx.Int32(fx.ptrtoint(smem.gacc.ptr)) + ((w * RT * NI + i) * 64 + lane) * 16),
                        alignment=16)).bitcast(fx.Float32) for w in range_constexpr(KW)]
                    v = ps[0]
                    for w in range_constexpr(1, KW):
                        v = v + ps[w]
                    row = tile_row + l16 + (i // NI) * 16
                    col = (nb * NI + i % NI) * 16 + q16 * 4
                    buffer_ops.buffer_store(v.bitcast(fx.Int32), slab, ((kz * i32_m + row) * N + col) * 4,
                                            offset_is_bytes=True, mask=row < i32_m, cache_modifier=SC1)
            fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        # Every non-V counter has four arrivals of +4. V spans three row blocks: the first block's
        # four arrivals add two each, the remaining eight add one. Both sums are EPOCH (16).
        # Owners wait for the full weighted epoch, including halo producers; arithmetic wraps modulo i32.
        if tx == 0:
            inc = is_vh.select((mb == 0).select(fx.Int32(2), fx.Int32(1)), fx.Int32(4))
            old0 = _atomic_add(cnt_addr, inc)
            ebase = old0 - (old0 & (EPOCH - 1))
            if (kz * TILES_PER_HEAD + nb % TILES_PER_HEAD >= WG_PER_HEAD - OWNERS):
                arrived = old0 + inc - ebase
                while arrived < EPOCH:
                    arrived = _atomic_add(cnt_addr, 0) - ebase
        fx.gpu.barrier()
        if (kz * TILES_PER_HEAD + nb % TILES_PER_HEAD >= WG_PER_HEAD - OWNERS) & (wave >= KW):
            pres1 = []
            for p in range_constexpr(PW):
                got = [fx.Int32(smem.pre.ptr[(((wave - KW) * PW + p) * PRE_WORDS + k) * 64 + lane])
                       for k in range_constexpr(PRE_WORDS)]
                pres1.append([got[0], got[1], Vec.from_elements(got[2:4], fx.Int32),
                              Vec.from_elements(got[4:6], fx.Int32), Vec.from_elements(got[6:8], fx.Int32), *got[8:]])
            # consecutive token pairs per tail wave (2i, 2i + 1): the first V-run tokens of two 8-token requests
            # (relative 0, 8) fall in different waves, which pack them at once (pairs i, i + 4 put both in tail wave
            # 0: M 129 / 144 / 176 / 192 -0.5 / -0.5 / -1.5 / -1.3%)
            tail_post(slot_s, [row_base // 2 + 2 * (wave - KW) + p for p in range(PW)], pres1)

    @flyc.jit
    def launch_qkv_k_waves_satomic(
        arg_a: fx.Int64,
        arg_w: fx.Int64,
        arg_as: fx.Int64,
        arg_ws: fx.Int64,
        arg_slab: fx.Int64,
        arg_cnt: fx.Int64,
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
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        i32_iq_st: fx.Int32,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_cs_rows: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, a_scale_bytes, arg_w, arg_ws, arg_as, arg_a, arg_slab, arg_cnt, i32_cs_rows, arg_pos, arg_slot,
            arg_islot, arg_cs, arg_qw, arg_kw, arg_iqw, arg_ikw, arg_q, arg_iq, arg_kv, arg_idx, arg_ks, arg_vs,
            i64_kv_st, i64_idx_st, i32_iq_st, i32_eps,
        ).launch(grid=(fx.Int64((i32_m + RT*16-1) // (RT*16) * KS * NB), 1, 1), block=(64 * NWG, 1, 1), stream=stream)

    launch_qkv_k_waves_satomic.kernel_name = name
    launch_qkv_k_waves_satomic.compile_hints = PRELOAD_HINTS
    return launch_qkv_k_waves_satomic
