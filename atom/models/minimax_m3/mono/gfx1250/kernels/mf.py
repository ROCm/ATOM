# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MoE front of a MiniMax-M3.1 decode layer as ONE launch (TP4, 8 rows, gfx1250), on the main stream:

    o_proj GEMV + k5 one-shot AR1 + residual + RMSNorm (Stage O, ``oproj_ar_gfx1250``, unchanged)
    -> the router (``moe_router_topk`` split-K 8: logits, biased sigmoid top-4, renorm, the MXFP4 token rows)
    || the whole shared expert (``quantize_fp8`` -> gate_up ``gemm_a8w8 mx32 sk4 fsk`` -> SwiGLU-OAI MXFP8
       -> down ``gemm_a8w8 mx32 sk1``),

with production's bits (each output element computed in production's order). The routed experts then run on
the routes and rows, and the shared output joins them in the gather: no side stream.

Workgroups (512 threads = 16 waves each, no clusters):

* ``0 .. 127``: o_proj GEMV tiles + the k5 push (Stage O).
* ``128 .. 131``: the k5 receivers, two rows each (Stage O). They also store x (the normed rows) into this
  launch's x buffer as k5-sentinel words (0xffff half-words -> 0x7fff), the in-launch data-as-flag.
* ``132 .. 139``: router CTA k = expert slice [16 k, +16). Wave s < 8 runs production's chain of K split s (24
  bf16 16x16x32 steps from zero, weights in VGPRs, issued at entry); the CTA sums the 8 splits in split order
  (LDS), rounds the logit to bf16 and publishes its slice; then it gates token row k from all 8 slices'
  logits (moe_router_topk's arithmetic) and writes row k's MXFP4 rows.
* ``140 .. 163``: gate_up CTA g = h columns [32 g, +32). Wave (s, t) = K split s of production's split-K 4 x
  tile t (8 gate rows + the same 8 up rows): a 12-step scaled-fp8 16x16x128 chain on x quantized to MXFP8
  in the CTA (quantize_fp8's bytes); split sum in split order (LDS), one bf16 rounding, SwiGLU-OAI as
  Triton's kernel, the group's E8M0 + E4M3 h, published as sentinel words.
* ``164 .. 187``: down CTA d = output columns [256 d, +256), wave w = 16 of them: a 6-step scaled-fp8
  16x16x128 chain on h from zero, bf16 out (the shared expert's output, read after the launch).

In-launch hand-offs carry their data: the consumers poll the words they read (k5's sentinel rule, so a word is
ready when no half-word is 0xffff; never a counter). Each hand-off buffer has 3 copies; launch E writes copy
E % 3 and re-arms copy (E + 1) % 3 (last written two launches ago, next written by the next launch). E is the
launch number: every role workgroup counts its own launches in a private word (they all run in every launch,
so they agree; no exit counter, no reset). Every wait is on a lower workgroup and bounded (timeouts counted).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.utility.mx_types import MxDtypeInt

from aiter.ops.flydsl.kernels import tdm_ops_gfx1250 as tdm_ops
from aiter.ops.flydsl.kernels.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops, pipeline_fence
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _MXFP8_COPY_AMAX_FLOOR,
    _ROUND_MODE,
    _cvt_scalef32_pk8_fp4_bf16,
    _cvt_scalef32_pk8_fp8_bf16,
)
from aiter.ops.flydsl.kernels.mxfp8_decode_gemm_gfx1250 import (
    A_KC,
    A_PAD,
    P_PAD,
    WAVE,
    WMMA_K,
    decode_gemm_config_error,
    decode_gemm_lds_bytes,
)
from aiter.ops.flydsl.kernels.oproj_ar_gfx1250 import (
    BUFS,
    _now32,
    RANKS,
    ROW_PACKS,
    RWAVES,
    SCOPE_SYS,
    SLAB,
    _bf_hi,
    _bf_lo,
    _bits,
    _f2bf2,
    _lds_barrier,
    _mem_clobber,
    _rcp,
    _rsq,
)
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from .rsh_tail import (
    FLT_MIN,
    LOG2E,
    RENORM_SUM_FLOOR,
    SCOPE_DEV,
    W_NT,
    _amax_bf16,
    _atomic_add_agent,
    _cat2,
    _cat4,
    _ld,
    _rmem,
    _rsrc,
    _uni,
    _word,
    sentinel_bits,
    x_flag_word,
)
from aiter.ops.flydsl.kernels.tensor_shim import AITER_FLYDSL_KERNARG_PRELOAD, AITER_FLYDSL_KERNARG_PRELOAD_COUNT, buf_copy_load, ptr_buf_tensor

HIDDEN = 6144
ROWS = 8
EXPERTS = 128
TOPK = 4
# router (moe_router_topk at 8 rows: split_k 8, bf16 WMMA 16x16x32)
R_SPLIT = 8
R_CHUNK = HIDDEN // R_SPLIT  # 768
R_STEPS = R_CHUNK // 32  # 24
R_XB = R_STEPS * 2 * 16 * 16  # 12288 B of x B fragments a split
# shared gate_up (gemm_a8w8_mx32_t16x32x512_..._sk4_fsk) and down (batched_gemm_a8w8_mx32_t16x64x128_..._sk1)
S_INTER = 768
S_SPLIT = 4
S_CHUNK = HIDDEN // S_SPLIT  # 1536
S_STEPS = S_CHUNK // 128  # 12
K_STEPS = HIDDEN // 128  # 48
GROUP = 32
N_GROUPS = S_INTER // GROUP  # 24
D_STEPS = S_INTER // 128  # 6
# roles
N_GEMV = 128
N_RECV = ROWS // 2
N_ROUTER = EXPERTS // 16  # 8
N_GU = N_GROUPS * S_SPLIT  # 96: (h group, K split) workgroups
N_DOWN = HIDDEN // 256  # 24
B_RECV = N_GEMV
B_ROUTER = B_RECV + N_RECV
B_GU = B_ROUTER + N_ROUTER
B_DOWN = B_GU + N_GU
GRID = B_DOWN + N_DOWN  # 260
# hand-off buffer copies (bytes a copy)
XBUF = ROWS * HIDDEN * 2  # x, bf16
LGBUF = ROWS * EXPERTS * 2  # router logits, bf16
HBUF = ROWS * S_INTER  # h, E4M3
HSBUF = ROWS * N_GROUPS * 4  # h scales, one E8M0 in the low byte of a dword (right after h in a copy)
HCOPY = HBUF + HSBUF  # 6912 B = 432 16-B chunks
NCOPY = 3
TLW = 8  # timeline stamps a workgroup
# LDS (bytes): router
LR_X = 0
LR_P = R_SPLIT * R_XB  # 98304: partials [split][tok][16] f32
LR_END = LR_P + R_SPLIT * ROWS * 16 * 4
# gate_up (one K split a workgroup)
LG_X = 0  # x MXFP8 B fragments of the split [step 12][jj 4][kgrp 2][tok 8] x 16 B
LG_SC = S_STEPS * 4 * 16 * 16  # 12288: x E8M0 words [tok][step]
LG_P = LG_SC + ROWS * S_STEPS * 4  # 12672: split 3's partials [tile][tok][16] f32
LG_END = LG_P + 4 * ROWS * 16 * 4
GPBUF = N_GROUPS * (S_SPLIT - 1) * 4 * ROWS * 16 * 8  # splits 0..2's partials, 64-bit (epoch << 32 | f32) words
RPBUF = R_SPLIT * ROWS * EXPERTS * 8  # rmode 1: the 64 router chains' partials [split][tok][expert], 64-bit words
# down
LD_X = 0  # h B fragments [step 6][jj 4][kgrp 2][tok 8] x 16 B
LD_SC = D_STEPS * 4 * 16 * 16  # 6144: h E8M0 words [tok][step]
LD_END = LD_SC + ROWS * D_STEPS * 4

# device table (int64 words), one a MoE layer (the scratch words are shared by every layer of a rank)
(T_RW, T_RB, T_NVR, T_GW, T_GS, T_DW, T_DS, T_IDS, T_WTS, T_XQ, T_XS, T_SH, T_XB, T_LG, T_HB, T_CNT,
 T_ERR, T_GP, T_RP) = range(19)
T_WORDS = 19
POLL_SLEEP = 1  # data-as-flag polls: each consumer polls only the words it reads (SYNC: s_sleep 1-4 equal)


@flyc.jit
def launch_mf(
    arg_a: fx.Pointer,
    arg_as: fx.Pointer,
    arg_w: fx.Pointer,
    arg_ws: fx.Pointer,
    arg_res: fx.Pointer,
    arg_nw: fx.Pointer,
    arg_out: fx.Pointer,
    arg_res_out: fx.Pointer,
    arg_ctr: fx.Pointer,
    arg_tab: fx.Pointer,
    i64_in0: fx.Int64,
    i64_in1: fx.Int64,
    i64_in2: fx.Int64,
    i64_in3: fx.Int64,
    i32_lda: fx.Int32,
    i32_slot: fx.Int32,
    i32_rank: fx.Int32,
    i32_maxit: fx.Int32,
    f64_eps: fx.Float64,
    stream: fx.Stream,
    K: Constexpr[int],
    alpha: Constexpr[float] = 1.702,
    beta: Constexpr[float] = 1.0,
    limit: Constexpr[float] = 7.0,
    need_renorm: Constexpr[int] = 1,
    roles: Constexpr[int] = 7,
    max_rows: Constexpr[int] = 256,
    tl: Constexpr[int] = 0,
    psleep: Constexpr[int] = POLL_SLEEP,
    rmode: Constexpr[int] = 0,
    shell: Constexpr[int] = 0,
    arg_tl: fx.Pointer = None,
):
    """Stage O's arguments (rows fixed at 8) + ``arg_tab``: the layer's table (``T_*`` int64 words).
    ``roles`` (debug): bit 0 router, bit 1 gate_up, bit 2 down (a role left out still counts its launches).
    ``tl`` (diagnostic): thread 0 of each workgroup stamps the 100 MHz steady counter into ``arg_tl`` (int32
    [GRID, TLW]): GEMV [entry, packs pushed]; receivers [entry, poll end, x stored]; router [entry, x ready, chains
    done, logits published, logits ready, end]; gate_up [entry, x ready, chains done, h stored]; down [entry,
    h ready, end]."""
    N = HIDDEN
    tile_m, nb, kw, wn, wdepth, wcpol, atdm = 32, 3, 16, 1, 1, 1, 1  # Stage O's (the decode GEMM's) config
    ks = 1
    err = decode_gemm_config_error(N, K, tile_m, nb, ks, kw, wn, wdepth, atdm)
    if err is not None:
        raise ValueError(f"mf: {err}")
    steps = K // WMMA_K
    steps_w = steps // kw
    cols = nb * 16
    ppr = cols // 8
    mbt = tile_m // 16
    nbw = nb // wn
    block = kw * wn * WAVE
    assert block == 512 and N // cols == N_GEMV
    wdepth = max(1, min(wdepth, steps_w))
    sa_opsel = mbt % 2 == 0
    sa_loads = mbt // 2 if sa_opsel else mbt
    KB4 = K // 128
    gemv_lds = decode_gemm_lds_bytes(tile_m, nb, kw, K, ks, atdm)
    PLANES = kw * tile_m * (nb * 16 * 4 + P_PAD)
    A_ROW = A_KC + A_PAD
    A_STAGE = tile_m * A_ROW
    P_ROW = cols * 4 + P_PAD
    P_PLANE = tile_m * P_ROW
    RED_OFF = (gemv_lds + 15) // 16 * 16
    lds_bytes = max(RED_OFF + 2 * RWAVES * 4, LR_END, LG_END, LD_END)
    ctr_recv = max_rows * RWAVES
    ctr_err = 2 * max_rows * RWAVES
    i32_m = ROWS
    kernel_name = f"mf_moe_front_gfx1250_k{K}{'' if roles == 7 else f'_r{roles}'}{'_tl' if tl else ''}{'' if psleep == POLL_SLEEP else f'_s{psleep}'}{'_rm1' if rmode else ''}{f'_sh{shell}' if shell else ''}"

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_mf(
        arg_a: fx.Pointer,
        arg_as: fx.Pointer,
        arg_w: fx.Pointer,
        arg_ws: fx.Pointer,
        arg_res: fx.Pointer,
        arg_nw: fx.Pointer,
        arg_out: fx.Pointer,
        arg_res_out: fx.Pointer,
        arg_ctr: fx.Pointer,
        arg_tab: fx.Pointer,
        i64_in0: fx.Int64,
        i64_in1: fx.Int64,
        i64_in2: fx.Int64,
        i64_in3: fx.Int64,
        i32_lda: fx.Int32,
        i32_slot: fx.Int32,
        i32_rank: fx.Int32,
        i32_maxit: fx.Int32,
        f64_eps: fx.Float64,
        arg_tl: fx.Pointer,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        arena = fx.SharedAllocator(static=False)
        arena.allocate(lds_bytes)
        base_ptr = arena.base_ptr
        lds_ptr = fx.recast_iter(fx.Uint8, base_ptr)
        lds_idx = fx.Int64(fx.ptrtoint(base_ptr))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)
        ctr = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4),
            arg_ctr,
        )
        tab = _rsrc(fx.ptrtoint(arg_tab))
        err_addr = _word(tab, T_ERR)
        inbox_bytes = fx.Int32(BUFS * RANKS) * i32_slot * fx.Int32(16)
        ins = [i64_in0, i64_in1, i64_in2, i64_in3]
        rsrc = [create_buffer_resource_from_addr(fx.Int64(p), num_records_bytes=inbox_bytes) for p in ins]

        def epoch():
            """This workgroup's launch number E (its private count + 1)."""
            cnt_rs = _rsrc(_word(tab, T_CNT), GRID * 4)
            return _uni(fx.Int32(buffer_load(cnt_rs, bid, vec_width=1, dtype=T.i32, cache_modifier=SCOPE_DEV))) + 1

        def count_out(e):
            if tid == 0:
                buffer_store(fx.Int32(e), _rsrc(_word(tab, T_CNT), GRID * 4), bid, cache_modifier=SCOPE_DEV)

        def copy_rs(tw, e, nbytes, ahead=0):
            """Resource of hand-off buffer copy (E + ahead) % 3."""
            idx = (e + ahead) % NCOPY
            return _rsrc(_word(tab, tw) + fx.Int64(idx) * fx.Int64(nbytes), nbytes)

        def timeout(pend):
            if pend & (lane == 0):
                _atomic_add_agent(err_addr, 1)

        def pending(chunks):
            m = fx.Int32(0)
            for v in chunks:
                for d in range(4):
                    m = m | sentinel_bits(fx.Int32(Vec(v)[d]))
            return fx.Int32(rocdl.ballot(T.i32, m != 0)) != 0

        ones = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)

        def stamp(slot, drain=False):
            if const_expr(tl):
                if const_expr(drain):
                    rocdl.s_wait_storecnt(0)
                    _lds_barrier()
                if tid == 0:
                    buffer_store(_now32(), _rsrc(fx.ptrtoint(arg_tl), GRID * TLW * 4), bid * TLW + slot)

        stamp(0)

        if bid < N_GEMV:
            # ---------------- the decode GEMM's tile + the k5 push (Stage O, unchanged) ----------------
            ntile = bid
            w_k = wave // wn
            w_n = wave % wn
            m0 = fx.Int32(0)
            ks0 = w_k * steps_w
            gA_base = fx.recast_iter(fx.Int8, arg_a)
            if wave < (K // A_KC):
                c_off = fx.Int64(wave * A_KC)
                desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=fx.Tensor(fx.make_view(fx.add_offset(gA_base, c_off),
                                                      fx.make_layout((tile_m, A_KC), (A_KC, 1)))),
                    lds_memref=fx.Tensor(fx.make_view(fx.add_offset(base_ptr, PLANES + wave * A_STAGE),
                                                      fx.make_layout((tile_m, A_KC), (A_ROW, 1)))),
                    global_offset=(0, 0),
                    tensor_shape=(tile_m, A_KC),
                    strides=(i32_lda, 1),
                    tile_shape=(tile_m, A_KC),
                    elem_bytes=1,
                    pad_interval=A_KC,
                    pad_amount=A_PAD,
                    num_warps=1,
                    workgroup_mask=0,
                    oob_outer_bound=i32_m,
                )
                tdm_ops.tensor_load_2d(desc)
            sa_buf = ptr_buf_tensor(arg_as, fx.Int32, num_records_bytes=i32_m * (K // 32))
            b_buf = ptr_buf_tensor(arg_w, fx.Int32, unit_elems=4)
            sb_buf = ptr_buf_tensor(arg_ws, fx.Int32)
            nblk0 = ntile * nb + w_n * nbw
            b_base = [(nblk0 + j) * K + lane for j in range_constexpr(nbw)]
            sb_col = [(nblk0 + j) * 16 + lane16 for j in range_constexpr(nbw)]
            sb_base = [(c // 32) * (K // 4) + c % 32 for c in sb_col]
            sa_row = [
                (m0 + (2 * p) * 16 + lane) if const_expr(sa_opsel) else (m0 + p * 16 + lane16)
                for p in range_constexpr(sa_loads)
            ]
            sa_base = [r * KB4 for r in sa_row]

            def _cat16(v):
                v01 = v[0].shuffle(v[1], list(range(8)))
                v23 = v[2].shuffle(v[3], list(range(8)))
                return v01.shuffle(v23, list(range(16)))

            def load_step(i):
                ksg = ks0 + i
                bfr = [
                    [
                        buf_copy_load(b_buf, b_base[j] + ksg * 128 + jj * 32, fx.Int32, 4, cache_modifier=wcpol)
                        for jj in range_constexpr(4)
                    ]
                    for j in range_constexpr(nbw)
                ]
                sb = [buf_copy_load(sb_buf, sb_base[j] + ksg * 32, fx.Int32) for j in range_constexpr(nbw)]
                sa = [buf_copy_load(sa_buf, sa_base[p] + ksg, fx.Int32) for p in range_constexpr(sa_loads)]
                return bfr, sb, sa

            def lds_a_frag(mi, i):
                s_ = w_k * steps_w + i
                base = PLANES + (s_ // 4) * A_STAGE + (mi * 16 + lane16) * A_ROW + (s_ % 4) * 128 + kgrp * 16
                return [Vec(lds_load_b128(lds_idx, base + 32 * jj)) for jj in range_constexpr(4)]

            atoms = [
                fx.make_mma_atom(
                    fx.rocdl.WMMAScale(
                        16, 16, WMMA_K, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32,
                        opsel_a=0, opsel_b=sel,
                    )
                )
                for sel in range_constexpr(2 if sa_opsel else 1)
            ]
            acc = [[fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(nbw)] for _ in range_constexpr(mbt)]
            for row in acc:
                for c in row:
                    c.store(fx.full(8, 0.0, fx.Float32))

            def _rm(v):
                t = fx.make_rmem_tensor(16, fx.Int32)
                t.store(v)
                return t

            def compute_step(st, i):
                bfr, sb, sa = st
                afr = [lds_a_frag(mi, i) for mi in range_constexpr(mbt)]
                bt = [_rm(_cat16(bfr[j])) for j in range_constexpr(nbw)]
                at = [_rm(_cat16(afr[mi])) for mi in range_constexpr(mbt)]
                for mi in range_constexpr(mbt):
                    for j in range_constexpr(nbw):
                        fx.gemm(
                            atoms[mi % 2 if sa_opsel else 0],
                            acc[mi][j],
                            bt[j],
                            at[mi],
                            acc[mi][j],
                            scale_a=sb[j],
                            scale_b=sa[mi // 2 if sa_opsel else mi],
                        )

            inflight = [load_step(i) for i in range_constexpr(wdepth)]
            rocdl.sched_barrier(0)
            pipeline_fence(outstanding=0, use_cluster=False)
            for i in range_constexpr(steps_w):
                st = inflight.pop(0)
                if const_expr(i + wdepth < steps_w):
                    inflight.append(load_step(i + wdepth))
                    rocdl.sched_barrier(0)
                compute_step(st, i)

            for mi in range_constexpr(mbt):
                for j in range_constexpr(nbw):
                    row = mi * 16 + lane16
                    col = (w_n * nbw + j) * 16 + kgrp * 8
                    v = Vec(acc[mi][j].load())
                    off = w_k * P_PLANE + row * P_ROW + col * 4
                    lds_store_b128(lds_idx, off, v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32))
                    lds_store_b128(lds_idx, off + 16, v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32))
            _lds_barrier()

            def plane_sum(r, c4):
                tot = None
                for k in range_constexpr(kw):
                    part = Vec(lds_load_b128(lds_idx, k * P_PLANE + r * P_ROW + c4 * 16)).bitcast(fx.Float32)
                    tot = part if tot is None else tot + part
                return tot

            p_units = tile_m * ppr
            for t in range_constexpr((p_units + block - 1) // block):
                u = tid + t * block
                r = u // ppr
                c8 = u % ppr
                ok = r < i32_m
                if const_expr(p_units % block):
                    ok = ok & (u < p_units)
                if ok:
                    lo4 = plane_sum(r, 2 * c8)
                    hi4 = plane_sum(r, 2 * c8 + 1)
                    v8 = lo4.shuffle(hi4, list(range(8)))
                    pk = Vec(v8.to(fx.BFloat16).bitcast(fx.Int32))
                    words = [fx.Int32(pk[c]) for c in range_constexpr(4)]
                    words = [w_ ^ _bits(w_) for w_ in words]
                    data = Vec.from_elements(words, fx.Int32)
                    pcol = ntile * ppr + c8
                    p = r * ROW_PACKS + pcol
                    g = r * RWAVES + pcol // SLAB
                    rnd = fx.Int32(ctr[g])
                    off = ((rnd * RANKS + i32_rank) * i32_slot + p) * 16
                    for d in range_constexpr(RANKS):
                        buffer_store(data.ir_value(), rsrc[d], off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)
            stamp(1)
            if const_expr(rmode == 1 and (roles & 1)):
                if bid < R_SPLIT * N_ROUTER:
                    # router chain (slice k, K split sp) on this GEMV workgroup after its push: wave 0 holds the 24.5 KB
                    # of weights (issued now, landing while the AR hops); waves 1-15 poll the split's x (12 KB) and
                    # stage it as B fragments; wave 0 runs production's 24-step chain and publishes its partials as
                    # 64-bit (epoch << 32 | f32) words for the 8 gate rows
                    e = epoch()
                    k = bid // R_SPLIT
                    sp = bid % R_SPLIT
                    w_rs = _rsrc(_word(tab, T_RW), (wave == 0).select(fx.Int32(EXPERTS * HIDDEN * 2), fx.Int32(0)))
                    a_dw = ((k * 16 + lane16) * HIDDEN + sp * R_CHUNK + kgrp * 8) // 2
                    wa = [[Vec(_ld(w_rs, a_dw + j * 16 + h * 8, 4, W_NT)) for h in range_constexpr(2)]
                          for j in range_constexpr(R_STEPS)]
                    rocdl.sched_barrier(0)
                    _lds_barrier()  # every wave is done with the GEMV's LDS planes
                    if wave >= 1:
                        x_rs = copy_rs(T_XB, e, XBUF)
                        t_ = tid - WAVE
                        ua = t_
                        ub = (t_ + 15 * WAVE < ROWS * (R_CHUNK // 8)).select(t_ + 15 * WAVE, t_)
                        da = ((ua // (R_CHUNK // 8)) * (HIDDEN // 8) + sp * (R_CHUNK // 8) + ua % (R_CHUNK // 8)) * 4
                        db = ((ub // (R_CHUNK // 8)) * (HIDDEN // 8) + sp * (R_CHUNK // 8) + ub % (R_CHUNK // 8)) * 4

                        def load_rx():
                            _mem_clobber()
                            return (Vec(_ld(x_rs, da, 4, SCOPE_DEV)), Vec(_ld(x_rs, db, 4, SCOPE_DEV)))

                        (xa, xb) = load_rx()
                        rpend = pending((xa, xb))
                        rit = fx.Int32(0)
                        while rpend & (rit < i32_maxit):
                            rocdl.s_sleep(psleep)
                            (xa, xb) = load_rx()
                            rpend = pending((xa, xb))
                            rit = rit + 1
                        timeout(rpend)
                        for uu, vv in ((ua, xa), (ub, xb)):
                            tok = uu // (R_CHUNK // 8)
                            cw = uu % (R_CHUNK // 8)
                            j = cw // 4
                            cc = cw % 4
                            lds_store_b128(lds_idx, ((j * 2 + cc // 2) * 16 + (cc % 2) * 8 + tok) * 16, vv)
                    _lds_barrier()
                    stamp(2)
                    if wave == 0:
                        tok16 = lane16 % 8

                        def x_frag(j):
                            return _cat2(
                                lds_load_b128(lds_idx, ((j * 2 + 0) * 16 + kgrp * 8 + tok16) * 16),
                                lds_load_b128(lds_idx, ((j * 2 + 1) * 16 + kgrp * 8 + tok16) * 16),
                            ).bitcast(fx.BFloat16)

                        wmma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
                        racc = fx.full(8, 0.0, fx.Float32)
                        PF = 2
                        xpend = [x_frag(j) for j in range_constexpr(PF)]
                        for j in range_constexpr(R_STEPS):
                            if const_expr(j + PF < R_STEPS):
                                xpend.append(x_frag(j + PF))
                            rocdl.sched_barrier(0)
                            ta = _rmem(racc, 8, fx.Float32)
                            fx.gemm(wmma, ta, _rmem(_cat2(wa[j][0], wa[j][1]).bitcast(fx.BFloat16), 16, fx.BFloat16),
                                    _rmem(xpend[j], 16, fx.BFloat16), ta)
                            racc = ta.load()
                            xpend[j] = None
                            rocdl.sched_barrier(0)
                        # acc[i]: expert 16 k + 8 kgrp + i of token lane16
                        if lane16 < ROWS:
                            av = Vec(racc)
                            rp_rs = _rsrc(_word(tab, T_RP), RPBUF)
                            for i in range_constexpr(8):
                                wd = Vec.from_elements([fx.Float32(av[i]).bitcast(fx.Int32), e], fx.Int32)
                                buffer_store(wd, rp_rs, ((lane16 * EXPERTS + k * 16 + kgrp * 8 + i) * R_SPLIT + sp) * 8,
                                             offset_is_bytes=True, cache_modifier=SCOPE_DEV)
                        stamp(3)
                    count_out(e)
        elif bid < B_ROUTER:
            # ---------------- k5's receiver (Stage O) + x as sentinel words ----------------
            e = epoch()
            x_rs = copy_rs(T_XB, e, XBUF)
            if const_expr(not (shell & 2)):
                # re-arm copy (E + 1) % 3 of this workgroup's two x rows (last written two launches ago)
                xr = copy_rs(T_XB, e, XBUF, 1)
                for m_ in range_constexpr(3):
                    u = tid + 512 * m_  # 16-B chunk of the 2 rows (1536 of them)
                    buffer_store(ones, xr, ((bid - N_GEMV) * 2 * HIDDEN + u * 8) * 2, offset_is_bytes=True,
                                 cache_modifier=SCOPE_DEV)
            half = wave // RWAVES
            w = wave % RWAVES
            r = (bid - N_GEMV) * 2 + half
            rr = r
            g = rr * RWAVES + w
            srnd = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[g])))
            rrnd = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[ctr_recv + g])))
            p0 = rr * ROW_PACKS + w * SLAB + lane
            res_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res)), num_records_bytes=i32_m * HIDDEN * 2)
            nw_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_nw)), num_records_bytes=HIDDEN * 2)
            out_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_out)), num_records_bytes=i32_m * HIDDEN * 2)
            ro_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res_out)), num_records_bytes=i32_m * HIDDEN * 2)
            my_base = fx.Int64(i64_in0)
            for d in range_constexpr(1, RANKS):
                my_base = (i32_rank == d).select(fx.Int64(ins[d]), my_base)
            my_rs = create_buffer_resource_from_addr(my_base, num_records_bytes=inbox_bytes)
            rs = []
            wt = []
            for j in range_constexpr(3):
                rs.append(Vec(buffer_load(res_rs, (p0 + 32 * j) * 4, vec_width=4, dtype=T.i32)))
                wt.append(Vec(buffer_load(nw_rs, (w * SLAB + 32 * j + lane) * 4, vec_width=4, dtype=T.i32)))
            sent = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)

            def my_store(off, data):
                buffer_store(data.ir_value(), my_rs, off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)

            def my_load(off):
                return Vec(buffer_load(my_rs, off // 4, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_SYS))

            rearm = (rrnd + 2) % BUFS
            for i in range_constexpr(1, RANKS):
                s_ = (i32_rank + i) & (RANKS - 1)
                for j in range_constexpr(3):
                    my_store(((rearm * RANKS + s_) * i32_slot + p0 + 32 * j) * 16, sent)

            def offs(s_, j):
                return ((rrnd * RANKS + s_) * i32_slot + p0 + 32 * j) * 16

            def load_all():
                _mem_clobber()
                return tuple(my_load(offs(s_, j)) for s_ in range_constexpr(RANKS) for j in range_constexpr(3))

            def kpending(vs):
                m = fx.Int32(0)
                for v in vs:
                    for c in range_constexpr(4):
                        m = m | _bits(fx.Int32(v[c]))
                return fx.Int32(rocdl.ballot(T.i32, m != 0)) != 0

            (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
            pend = kpending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
            it = fx.Int32(0)
            while pend & (it < i32_maxit):
                rocdl.s_sleep(1)
                (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
                pend = kpending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
                it = it + 1
            if pend & (lane == 0):
                fx.atomic_add(ctr + ctr_err, fx.Int32(1), syncscope="agent")
            stamp(1)
            flat = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
            ins_v = [[flat[s_ * 3 + j] for j in range(3)] for s_ in range(RANKS)]

            s = [[None] * 8 for _ in range(3)]
            for j in range_constexpr(3):
                for k in range_constexpr(8):
                    acc_ = None
                    for q in range_constexpr(RANKS):
                        wq = fx.Int32(ins_v[q][j][k // 2])
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
                    e_ = s[j][k] * s[j][k]
                    sq = e_ if sq is None else sq + e_
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
                fx.ptr_store(sq, red + wave)
            _lds_barrier()
            pr = [fx.Float32(red[half * RWAVES + i]) for i in range_constexpr(RWAVES)]
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
                buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), out_rs, (p0 + 32 * j) * 16, offset_is_bytes=True)
                # x for the in-launch consumers: the same words, k5's sentinel rule, device scope
                if const_expr(not (shell & 2)):
                    xo = Vec.from_elements([x_flag_word(o[c]) for c in range_constexpr(4)], fx.Int32)
                    buffer_store(xo.ir_value(), x_rs, (p0 + 32 * j) * 16, offset_is_bytes=True, cache_modifier=SCOPE_DEV)
            for j in range_constexpr(3):
                my_store(offs(i32_rank, j), sent)
            if lane == 0:
                fx.ptr_store((srnd + 1) % BUFS, ctr + g)
                fx.ptr_store((rrnd + 1) % BUFS, ctr + (ctr_recv + g))
            stamp(2, drain=True)
            count_out(e)
        elif bid < B_GU:
            # ---------------- router: slice k (experts [16 k, +16)), then gate row k ----------------
            k = bid - B_ROUTER
            e = epoch()
            live = wave < R_SPLIT
            sp = wave % R_SPLIT
            if const_expr(roles & 1):
                # the gate's inputs that do not depend on the logits, issued at entry (as production's router does):
                # the bias and the valid-row count (rows >= it get id -1, weight 0)
                bias_rs = _rsrc(_word(tab, T_RB))
                bias_v = [fx.Float32(buffer_load(bias_rs, lane + WAVE * i, vec_width=1, dtype=T.f32))
                          for i in range_constexpr(4)]
                nvr_addr = _word(tab, T_NVR)
                has_nvr = nvr_addr != fx.Int64(0)
                nvr = fx.Int32(buffer_load(_rsrc(has_nvr.select(nvr_addr, _word(tab, T_RB))), fx.Int32(0), vec_width=1,
                                           dtype=T.i32))
            if const_expr((roles & 1) and rmode == 1):
                # rmode 1: the chains ran on GEMV workgroups 0..63; this workgroup re-arms x row k of copy (E + 1) % 3,
                # writes row k's MXFP4 rows (waves 1-6) and gates row k from the 64 chains' partials (wave 0)
                if (wave >= 1) & (wave < 7):
                    b_ = tid - WAVE
                    x_rs = copy_rs(T_XB, e, XBUF)
                    xdw = (k * HIDDEN + b_ * 32) // 2

                    def load_q():
                        _mem_clobber()
                        return tuple(Vec(_ld(x_rs, xdw + cc * 4, 4, SCOPE_DEV)) for cc in range_constexpr(4))

                    (q0, q1, q2, q3) = load_q()
                    qpend = pending((q0, q1, q2, q3))
                    qit = fx.Int32(0)
                    while qpend & (qit < i32_maxit):
                        rocdl.s_sleep(psleep)
                        (q0, q1, q2, q3) = load_q()
                        qpend = pending((q0, q1, q2, q3))
                        qit = qit + 1
                    timeout(qpend)
                    ch = [q0, q1, q2, q3]
                    xq_rs = _rsrc(_word(tab, T_XQ), ROWS * (HIDDEN // 2))
                    xs_rs = _rsrc(_word(tab, T_XS), ROWS * (HIDDEN // 32))
                    bfs = [ch[cc].bitcast(fx.BFloat16) for cc in range_constexpr(4)]
                    amax = _amax_bf16(ch)
                    e8m0 = fx.Int32(emit_mx_e8m0_scale(amax, mode=_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1))
                    sc = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                    pq = [fx.Int32(_cvt_scalef32_pk8_fp4_bf16(bfs[cc], sc, i32_ty=T.i32)) for cc in range_constexpr(4)]
                    buffer_store(Vec.from_elements(pq, fx.Int32), xq_rs, k * (HIDDEN // 2) + b_ * 16, offset_is_bytes=True)
                    buffer_store(fx.Uint8(e8m0), xs_rs, k * (HIDDEN // 32) + b_, offset_is_bytes=True)
                stamp(1)
                if wave == 0:
                    rp_rs = _rsrc(_word(tab, T_RP), RPBUF)
                    # expert lane + 32 i of row k: its 8 splits' words are 64 contiguous bytes ([tok][expert][split]);
                    # one poll over all 16 16-byte chunks, then the split sum in split order and the bf16 logit

                    def load_p():
                        _mem_clobber()
                        return tuple(
                            Vec(_ld(rp_rs, ((k * EXPERTS + lane + WAVE * i) * R_SPLIT + 2 * q) * 2, 4, SCOPE_DEV))
                            for i in range_constexpr(4) for q in range_constexpr(4))

                    def p_pending(vs):
                        m = fx.Int32(0)
                        for v in vs:
                            m = m | (fx.Int32(Vec(v)[1]) - e) | (fx.Int32(Vec(v)[3]) - e)
                        return fx.Int32(rocdl.ballot(T.i32, m != 0)) != 0

                    (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15) = load_p()
                    ppend = p_pending((r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15))
                    pit = fx.Int32(0)
                    while ppend & (pit < i32_maxit):
                        rocdl.s_sleep(psleep)
                        (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15) = load_p()
                        ppend = p_pending((r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15))
                        pit = pit + 1
                    timeout(ppend)
                    rv = (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
                    logits = []
                    for i in range_constexpr(4):
                        tot = None
                        for s_ in range_constexpr(R_SPLIT):
                            pv = fx.Int32(Vec(rv[i * 4 + s_ // 2])[2 * (s_ % 2)]).bitcast(fx.Float32)
                            tot = pv if tot is None else tot + pv
                        logits.append(tot.to(fx.BFloat16).to(fx.Float32))
                    stamp(4)
                    out_w, out_id = _gate(logits, lane, bias_v, has_nvr, nvr, k, need_renorm)
                    if lane < TOPK:
                        buffer_store(out_w, _rsrc(_word(tab, T_WTS)), k * TOPK + lane)
                        buffer_store(out_id, _rsrc(_word(tab, T_IDS)), k * TOPK + lane)
            if const_expr((roles & 1) and rmode == 0):
                # weights first: wave s < 8 = split s of the slice's 16 rows (24 steps x 2 x 16 B, non-temporal); the
                # idle waves read through an empty resource (no traffic)
                w_rs = _rsrc(_word(tab, T_RW), live.select(fx.Int32(EXPERTS * HIDDEN * 2), fx.Int32(0)))
                a_dw = ((k * 16 + lane16) * HIDDEN + sp * R_CHUNK + kgrp * 8) // 2
                wa = [[Vec(_ld(w_rs, a_dw + j * 16 + h * 8, 4, W_NT)) for h in range_constexpr(2)]
                      for j in range_constexpr(R_STEPS)]
                rocdl.sched_barrier(0)
                # re-arm copy (E + 1) % 3 of x (row k) and of the logits (row k)
                if tid < EXPERTS * 2 // 16:
                    buffer_store(ones, copy_rs(T_LG, e, LGBUF, 1), (k * EXPERTS * 2) + tid * 16, offset_is_bytes=True,
                                 cache_modifier=SCOPE_DEV)
                # x: all 8 rows, 12 16-B chunks a thread (chunk u = tok * 768 + c8), polled until ready and staged
                # as the chains' B fragments; two passes of 6 (the weights hold 192 VGPRs meanwhile)
                x_rs = copy_rs(T_XB, e, XBUF)
                for ps in range_constexpr(2):

                    def load_x(ps=ps):
                        _mem_clobber()
                        return tuple(Vec(_ld(x_rs, (tid + 512 * (6 * ps + m_)) * 4, 4, SCOPE_DEV))
                                     for m_ in range_constexpr(6))

                    (c0, c1, c2, c3, c4, c5) = load_x()
                    pend = pending((c0, c1, c2, c3, c4, c5))
                    it = fx.Int32(0)
                    while pend & (it < i32_maxit):
                        rocdl.s_sleep(psleep)
                        (c0, c1, c2, c3, c4, c5) = load_x()
                        pend = pending((c0, c1, c2, c3, c4, c5))
                        it = it + 1
                    timeout(pend)
                    xc = (c0, c1, c2, c3, c4, c5)
                    for m_ in range_constexpr(6):
                        u = tid + 512 * (6 * ps + m_)
                        tok = u // (HIDDEN // 8)
                        c8_ = u % (HIDDEN // 8)
                        s_ = c8_ // (R_CHUNK // 8)
                        j = (c8_ % (R_CHUNK // 8)) // 4
                        cc = c8_ % 4
                        lds_store_b128(lds_idx, LR_X + s_ * R_XB + ((j * 2 + cc // 2) * 16 + (cc % 2) * 8 + tok) * 16,
                                       xc[m_])
                _lds_barrier()
                stamp(1)
                # the chains
                tok16 = lane16 % 8
                xb0 = LR_X + sp * R_XB

                def x_frag(j):
                    return _cat2(
                        lds_load_b128(lds_idx, xb0 + ((j * 2 + 0) * 16 + kgrp * 8 + tok16) * 16),
                        lds_load_b128(lds_idx, xb0 + ((j * 2 + 1) * 16 + kgrp * 8 + tok16) * 16),
                    ).bitcast(fx.BFloat16)

                wmma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
                racc = fx.full(8, 0.0, fx.Float32)
                PF = 2  # x fragments in flight (3 spills: 192 weight VGPRs a wave at the 256 cap)
                xpend = [x_frag(j) for j in range_constexpr(PF)]
                for j in range_constexpr(R_STEPS):
                    if const_expr(j + PF < R_STEPS):
                        xpend.append(x_frag(j + PF))
                    rocdl.sched_barrier(0)
                    ta = _rmem(racc, 8, fx.Float32)
                    fx.gemm(wmma, ta, _rmem(_cat2(wa[j][0], wa[j][1]).bitcast(fx.BFloat16), 16, fx.BFloat16),
                            _rmem(xpend[j], 16, fx.BFloat16), ta)
                    racc = ta.load()
                    xpend[j] = None
                    rocdl.sched_barrier(0)
                # partials [split][tok][16 experts] (acc[i]: expert 8 kgrp + i of token lane16)
                if live & (lane16 < ROWS):
                    av = Vec(racc)
                    for hh in range_constexpr(2):
                        part = Vec.from_elements([av[hh * 4 + i] for i in range_constexpr(4)], fx.Float32)
                        lds_store_b128(lds_idx, LR_P + ((sp * ROWS + lane16) * 16 + kgrp * 8 + hh * 4) * 4,
                                       part.bitcast(fx.Int32))
                _lds_barrier()
                stamp(2)
                # split sum in split order, bf16 logit; thread (tok, expert pair) -> its logits word (sentinel rule)
                if tid < ROWS * 8:
                    tok = tid // 8
                    ep = tid % 8
                    tl_ = [None, None]
                    for ee in range_constexpr(2):
                        tot = None
                        for s_ in range_constexpr(R_SPLIT):
                            v_ = Vec(lds_load_b32(lds_idx, LR_P + ((s_ * ROWS + tok) * 16 + 2 * ep + ee) * 4))
                            pv = fx.Int32(v_[0]).bitcast(fx.Float32)
                            tot = pv if tot is None else tot + pv
                        tl_[ee] = tot
                    wd = _f2bf2(tl_[0], tl_[1])
                    buffer_store(x_flag_word(wd), copy_rs(T_LG, e, LGBUF), (tok * EXPERTS + k * 16 + 2 * ep) * 2,
                                 offset_is_bytes=True, cache_modifier=SCOPE_DEV)
                stamp(3)
                # row k's MXFP4 rows (the router's quant_rows): MX block b of 32 values, thread b < 192
                if tid < HIDDEN // 32:
                    xq_rs = _rsrc(_word(tab, T_XQ), ROWS * (HIDDEN // 2))
                    xs_rs = _rsrc(_word(tab, T_XS), ROWS * (HIDDEN // 32))
                    ch = [Vec(_ld(x_rs, (k * HIDDEN + tid * 32) // 2 + cc * 4, 4, SCOPE_DEV)) for cc in range_constexpr(4)]
                    bfs = [ch[cc].bitcast(fx.BFloat16) for cc in range_constexpr(4)]
                    amax = _amax_bf16(ch)
                    e8m0 = fx.Int32(emit_mx_e8m0_scale(amax, mode=_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1))
                    sc = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                    pq = [fx.Int32(_cvt_scalef32_pk8_fp4_bf16(bfs[cc], sc, i32_ty=T.i32)) for cc in range_constexpr(4)]
                    buffer_store(Vec.from_elements(pq, fx.Int32), xq_rs, k * (HIDDEN // 2) + tid * 16, offset_is_bytes=True)
                    buffer_store(fx.Uint8(e8m0), xs_rs, k * (HIDDEN // 32) + tid, offset_is_bytes=True)
                # gate row k (wave 0): lane l holds experts l + 32 i, from the 8 slices' logits words
                if wave == 0:
                    lg_rs = copy_rs(T_LG, e, LGBUF)

                    def load_lg():
                        _mem_clobber()
                        return tuple(
                            fx.Int32(buffer_load(lg_rs, (k * EXPERTS + lane + WAVE * i) // 2, vec_width=1, dtype=T.i32,
                                                 cache_modifier=SCOPE_DEV))
                            for i in range_constexpr(4))

                    def lg_pending(ws_):
                        m = fx.Int32(0)
                        for w_ in ws_:
                            m = m | sentinel_bits(w_)
                        return fx.Int32(rocdl.ballot(T.i32, m != 0)) != 0

                    (g0, g1, g2, g3) = load_lg()
                    gpend = lg_pending((g0, g1, g2, g3))
                    git = fx.Int32(0)
                    while gpend & (git < i32_maxit):
                        rocdl.s_sleep(psleep)
                        (g0, g1, g2, g3) = load_lg()
                        gpend = lg_pending((g0, g1, g2, g3))
                        git = git + 1
                    timeout(gpend)
                    stamp(4)
                    odd = (lane % 2) == 1
                    logits = [odd.select(_bf_hi(w_), _bf_lo(w_)) for w_ in (g0, g1, g2, g3)]
                    out_w, out_id = _gate(logits, lane, bias_v, has_nvr, nvr, k, need_renorm)
                    if lane < TOPK:
                        buffer_store(out_w, _rsrc(_word(tab, T_WTS)), k * TOPK + lane)
                        buffer_store(out_id, _rsrc(_word(tab, T_IDS)), k * TOPK + lane)
            stamp(5, drain=True)
            count_out(e)
        elif bid < B_DOWN:
            # ---------------- shared gate_up, workgroup (h group g, K split s) ----------------
            # waves 0-3: tile t's 12-step chain of split s (weights in VGPRs, issued at entry); waves 4-15 poll and
            # quantize the split's x (no weight load in flight in a polling wave). Splits 0-2 publish their fp32
            # partials as 64-bit (epoch << 32 | value) words; split 3 sums the four in split order, then SwiGLU-OAI
            # and the group's MXFP8 h, published as sentinel words.
            u_ = bid - B_GU
            g = u_ // S_SPLIT
            s = u_ % S_SPLIT
            e = epoch()
            if const_expr(roles & 2):
                chain = wave < 4
                t = wave % 4
                row = (lane16 < 8).select(g * GROUP + t * 8 + lane16, S_INTER + g * GROUP + t * 8 + lane16 - 8)
                w_rs = _rsrc(_word(tab, T_GW), chain.select(fx.Int32(2 * S_INTER * HIDDEN), fx.Int32(0)))
                ws_rs = _rsrc(_word(tab, T_GS), chain.select(fx.Int32(2 * S_INTER * (HIDDEN // 32)), fx.Int32(0)))
                ws_w = (row // 32) * (HIDDEN // 4) + row % 32
                wsc = [fx.Int32(_ld(ws_rs, ws_w + (s * S_STEPS + ks_) * 32, 1, W_NT)) for ks_ in range_constexpr(S_STEPS)]
                w_u = (row // 16) * HIDDEN + row % 16
                wf = [
                    [Vec(_ld(w_rs, (w_u + ((s * S_STEPS + ks_) * 8 + jj * 2 + kgrp) * 16) * 4, 4, W_NT))
                     for jj in range_constexpr(4)]
                    for ks_ in range_constexpr(S_STEPS)
                ]
                rocdl.sched_barrier(0)
                # re-arm copy (E + 1) % 3 of this group's h bytes and h scales (the split-3 workgroup)
                if (s == S_SPLIT - 1) & (tid < ROWS * 2):
                    buffer_store(ones, copy_rs(T_HB, e, HCOPY, 1), (tid // 2) * S_INTER + g * GROUP + (tid % 2) * 16,
                                 offset_is_bytes=True, cache_modifier=SCOPE_DEV)
                if (s == S_SPLIT - 1) & (tid < ROWS):
                    buffer_store(fx.Int32(-1), copy_rs(T_HB, e, HCOPY, 1), HBUF + (tid * N_GROUPS + g) * 4,
                                 offset_is_bytes=True, cache_modifier=SCOPE_DEV)
                if wave >= 4:
                    # MX block b = tid - 128 (token b // 48, block b % 48 of the split): 4 chunks, polled, quantized
                    b_ = tid - 128
                    xtok = b_ // (S_CHUNK // 32)
                    xkb = b_ % (S_CHUNK // 32)
                    x_rs = copy_rs(T_XB, e, XBUF)
                    xdw = (xtok * HIDDEN + s * S_CHUNK + xkb * 32) // 2

                    def load_x():
                        _mem_clobber()
                        return tuple(Vec(_ld(x_rs, xdw + u * 4, 4, SCOPE_DEV)) for u in range_constexpr(4))

                    (c0, c1, c2, c3) = load_x()
                    pend = pending((c0, c1, c2, c3))
                    it = fx.Int32(0)
                    while pend & (it < i32_maxit):
                        rocdl.s_sleep(psleep)
                        (c0, c1, c2, c3) = load_x()
                        pend = pending((c0, c1, c2, c3))
                        it = it + 1
                    timeout(pend)
                    xch = [c0, c1, c2, c3]
                    bfs = [xch[cc].bitcast(fx.BFloat16) for cc in range_constexpr(4)]
                    xamax = _amax_bf16(xch)
                    xe8 = fx.Int32(
                        emit_mx_e8m0_scale(fx.max(xamax, fx.Float32(_MXFP8_COPY_AMAX_FLOOR)), mode=_ROUND_MODE,
                                           dtype=MxDtypeInt.FP8_E4M3))
                    xsc8 = (xe8 << fx.Int32(23)).bitcast(fx.Float32)
                    qv = [Vec(_cvt_scalef32_pk8_fp8_bf16(bfs[cc], xsc8, v2i32_ty=T.vec(2, T.i32))) for cc in range_constexpr(4)]
                    xks = xkb // 4
                    xq_ = xkb % 4
                    for kg in range_constexpr(2):
                        unit = Vec.from_elements([qv[2 * kg][0], qv[2 * kg][1], qv[2 * kg + 1][0], qv[2 * kg + 1][1]],
                                                 fx.Int32)
                        lds_store_b128(lds_idx, LG_X + ((xks * 4 + xq_) * 16 + kg * 8 + xtok) * 16, unit)
                    fx.ptr_store(fx.Uint8(xe8), fx.add_offset(lds_ptr, LG_SC + (xtok * S_STEPS + xks) * 4 + xq_))
                _lds_barrier()
                stamp(1)
                gp_rs = _rsrc(_word(tab, T_GP), GPBUF)
                if chain:
                    tok16 = lane16 % 8
                    atom = fx.make_mma_atom(
                        fx.rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0, opsel_b=0))

                    def gx_frag(ks_):
                        return [lds_load_b128(lds_idx, LG_X + ((ks_ * 4 + jj) * 16 + kgrp * 8 + tok16) * 16)
                                for jj in range_constexpr(4)]

                    def gx_scale(ks_):
                        return fx.Int32(Vec(lds_load_b32(lds_idx, LG_SC + (tok16 * S_STEPS + ks_) * 4))[0])

                    gacc = fx.make_rmem_tensor(8, fx.Float32)
                    gacc.store(fx.full(8, 0.0, fx.Float32))
                    PF = 1  # (2 spills at the 256-VGPR cap)
                    gpend_ = [(gx_frag(ks_), gx_scale(ks_)) for ks_ in range_constexpr(PF)]
                    for ks_ in range_constexpr(S_STEPS):
                        if const_expr(ks_ + PF < S_STEPS):
                            gpend_.append((gx_frag(ks_ + PF), gx_scale(ks_ + PF)))
                        rocdl.sched_barrier(0)
                        xf, xs_ = gpend_[ks_]
                        fx.gemm(atom, gacc, _rmem(_cat4(wf[ks_]), 16, fx.Int32), _rmem(_cat4(xf), 16, fx.Int32), gacc,
                                scale_a=wsc[ks_], scale_b=xs_)
                        gpend_[ks_] = None
                        rocdl.sched_barrier(0)
                    # acc[i]: tile row 8 kgrp + i (kgrp 0 gate, 1 up) of token lane16
                    av = Vec(gacc.load())
                    if lane16 < ROWS:
                        if s < S_SPLIT - 1:
                            pw = ((((g * (S_SPLIT - 1) + s) * 4 + t) * ROWS + lane16) * 16 + kgrp * 8) * 8
                            for i in range_constexpr(8):
                                wd = Vec.from_elements([fx.Float32(av[i]).bitcast(fx.Int32), e], fx.Int32)
                                buffer_store(wd, gp_rs, pw + i * 8, offset_is_bytes=True, cache_modifier=SCOPE_DEV)
                        else:
                            pb = LG_P + ((t * ROWS + lane16) * 16 + kgrp * 8) * 4
                            for hh in range_constexpr(2):
                                part = Vec.from_elements([av[hh * 4 + i] for i in range_constexpr(4)], fx.Float32)
                                lds_store_b128(lds_idx, pb + hh * 16, part.bitcast(fx.Int32))
                _lds_barrier()
                stamp(2)
                # split 3, waves 4-7: thread v = tid - 128 = token v // 16, tile (v % 16) // 4, h columns 8 tile + 2 q + {0, 1}
                if (s == S_SPLIT - 1) & (wave >= 4) & (wave < 8):
                    v_ = tid - 128
                    tok = v_ // 16
                    tile = (v_ % 16) // 4
                    q = v_ % 4

                    def load_p():
                        _mem_clobber()
                        return tuple(
                            Vec(_ld(gp_rs, (((((g * (S_SPLIT - 1) + sp_) * 4 + tile) * ROWS + tok) * 16 + 8 * gu + 2 * q)
                                            * 8) // 4, 4, SCOPE_DEV))
                            for sp_ in range_constexpr(S_SPLIT - 1) for gu in range_constexpr(2))

                    def p_pending(vs):
                        m = fx.Int32(0)
                        for v in vs:
                            m = m | (fx.Int32(Vec(v)[1]) - e) | (fx.Int32(Vec(v)[3]) - e)
                        return fx.Int32(rocdl.ballot(T.i32, m != 0)) != 0

                    (p0, p1, p2, p3, p4, p5) = load_p()
                    ppend = p_pending((p0, p1, p2, p3, p4, p5))
                    pit = fx.Int32(0)
                    while ppend & (pit < i32_maxit):
                        rocdl.s_sleep(psleep)
                        (p0, p1, p2, p3, p4, p5) = load_p()
                        ppend = p_pending((p0, p1, p2, p3, p4, p5))
                        pit = pit + 1
                    timeout(ppend)
                    pv = [[p0, p1], [p2, p3], [p4, p5]]  # [split][gate / up] -> (value, tag, value, tag)
                    gate = [None, None]
                    up = [None, None]
                    for sp_ in range_constexpr(S_SPLIT - 1):
                        for ee in range_constexpr(2):
                            gv_ = fx.Int32(Vec(pv[sp_][0])[2 * ee]).bitcast(fx.Float32)
                            uv_ = fx.Int32(Vec(pv[sp_][1])[2 * ee]).bitcast(fx.Float32)
                            gate[ee] = gv_ if sp_ == 0 else gate[ee] + gv_
                            up[ee] = uv_ if sp_ == 0 else up[ee] + uv_
                    base = LG_P + ((tile * ROWS + tok) * 16 + 2 * q) * 4
                    for ee in range_constexpr(2):
                        gv_ = fx.Int32(Vec(lds_load_b32(lds_idx, base + ee * 4))[0]).bitcast(fx.Float32)
                        uv_ = fx.Int32(Vec(lds_load_b32(lds_idx, base + 32 + ee * 4))[0]).bitcast(fx.Float32)
                        gate[ee] = gate[ee] + gv_
                        up[ee] = up[ee] + uv_
                    hb = []
                    for ee in range_constexpr(2):
                        gt = gate[ee].to(fx.BFloat16).to(fx.Float32)
                        ut = up[ee].to(fx.BFloat16).to(fx.Float32)
                        gt = fx.minnumf(gt, fx.Float32(limit))
                        ut = fx.minnumf(fx.maxnumf(ut, fx.Float32(-limit)), fx.Float32(limit))
                        a_ = fx.Float32(alpha) * gt
                        ex = fx.exp2((-a_) * fx.Float32(LOG2E))
                        sig = fx.Float32(1.0) / (fx.Float32(1.0) + ex)
                        hb.append(((gt * sig) * (ut + fx.Float32(beta))).to(fx.BFloat16))
                    hv = Vec.from_elements(hb, fx.BFloat16)
                    hf = hv.to(fx.Float32)
                    amax = fx.max(abs(fx.Float32(hf[0])), abs(fx.Float32(hf[1])))
                    for off in (1, 2, 4, 8):
                        amax = fx.max(amax, amax.shuffle_xor(fx.Int32(off), fx.Int32(WAVE)))
                    e8 = fx.Int32(emit_mx_e8m0_scale(fx.max(amax, fx.Float32(_MXFP8_COPY_AMAX_FLOOR)), mode=_ROUND_MODE,
                                                     dtype=MxDtypeInt.FP8_E4M3))
                    sc8 = (e8 << fx.Int32(23)).bitcast(fx.Float32)
                    hw = hv.bitcast(fx.Int32)
                    w0 = fx.Int32(Vec(hw)[0])
                    ws_ = [w0, w0.shuffle_xor(fx.Int32(1), fx.Int32(WAVE)), w0.shuffle_xor(fx.Int32(2), fx.Int32(WAVE)),
                           w0.shuffle_xor(fx.Int32(3), fx.Int32(WAVE))]
                    pairs = []
                    for want in range_constexpr(4):
                        pick = ws_[0]
                        for m_ in range_constexpr(1, 4):
                            pick = ((q ^ m_) == want).select(ws_[m_], pick)
                        pick = (q == want).select(ws_[0], pick)
                        pairs.append(pick)
                    v8 = Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16)
                    q8 = Vec(_cvt_scalef32_pk8_fp8_bf16(v8, sc8, v2i32_ty=T.vec(2, T.i32)))
                    if q == 0:
                        buffer_store(
                            Vec.from_elements([x_flag_word(fx.Int32(q8[0])), x_flag_word(fx.Int32(q8[1]))], fx.Int32),
                            copy_rs(T_HB, e, HCOPY), tok * S_INTER + g * GROUP + tile * 8, offset_is_bytes=True,
                            cache_modifier=SCOPE_DEV)
                    if (q == 0) & (tile == 0):
                        buffer_store(e8, copy_rs(T_HB, e, HCOPY), HBUF + (tok * N_GROUPS + g) * 4, offset_is_bytes=True,
                                     cache_modifier=SCOPE_DEV)
            stamp(3, drain=True)
            count_out(e)
        else:
            # ---------------- shared down: output columns [256 d + 16 wave, +16) ----------------
            d = bid - B_DOWN
            e = epoch()
            if const_expr(roles & 4):
                row = d * 256 + wave * 16 + lane16
                dw_rs = _rsrc(_word(tab, T_DW), HIDDEN * S_INTER)
                ds_rs = _rsrc(_word(tab, T_DS), HIDDEN * (S_INTER // 32))
                ds_w = (row // 32) * (S_INTER // 4) + row % 32
                dsc = [fx.Int32(_ld(ds_rs, ds_w + ks_ * 32, 1, W_NT)) for ks_ in range_constexpr(D_STEPS)]
                d_u = (row // 16) * S_INTER + row % 16
                df = [
                    [Vec(_ld(dw_rs, (d_u + (ks_ * 8 + jj * 2 + kgrp) * 16) * 4, 4, W_NT)) for jj in range_constexpr(4)]
                    for ks_ in range_constexpr(D_STEPS)
                ]
                rocdl.sched_barrier(0)
                # h then its scales, one copy = 432 16-B chunks: thread u < 432 polls chunk u (h: token u // 48,
                # K [16 (u % 48), +16); scales: dwords [4 (u - 384), +4) of [tok][24]); the rest poll chunk u - 432
                hc_rs = copy_rs(T_HB, e, HCOPY)
                n_h = HBUF // 16
                n_all = HCOPY // 16
                is_h = tid < n_h
                is_s = (tid >= n_h) & (tid < n_all)
                cu_ = (tid < n_all).select(tid, tid - n_all)

                def load_h():
                    _mem_clobber()
                    return (Vec(_ld(hc_rs, cu_ * 4, 4, SCOPE_DEV)),)

                (hc,) = load_h()
                pend = pending((hc,))
                it = fx.Int32(0)
                while pend & (it < i32_maxit):
                    rocdl.s_sleep(psleep)
                    (hc,) = load_h()
                    pend = pending((hc,))
                    it = it + 1
                timeout(pend)
                if is_h:
                    tok = tid // (S_INTER // 16)
                    k0 = (tid % (S_INTER // 16)) * 16
                    ks_ = k0 // 128
                    xq_ = (k0 % 128) // 32
                    kg = (k0 % 32) // 16
                    lds_store_b128(lds_idx, LD_X + ((ks_ * 4 + xq_) * 16 + kg * 8 + tok) * 16, hc)
                if is_s:
                    v = tid - n_h
                    for c in range_constexpr(4):
                        gi = v * 4 + c  # dword [tok][group] index
                        tok = gi // N_GROUPS
                        gg = gi % N_GROUPS
                        fx.ptr_store(fx.Uint8(fx.Int32(hc[c])),
                                     fx.add_offset(lds_ptr, LD_SC + (tok * D_STEPS + gg // 4) * 4 + gg % 4))
                _lds_barrier()
                stamp(1)
                tok16 = lane16 % 8
                atom = fx.make_mma_atom(
                    fx.rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0, opsel_b=0))
                dacc = fx.make_rmem_tensor(8, fx.Float32)
                dacc.store(fx.full(8, 0.0, fx.Float32))
                for ks_ in range_constexpr(D_STEPS):
                    xf = [lds_load_b128(lds_idx, LD_X + ((ks_ * 4 + jj) * 16 + kgrp * 8 + tok16) * 16)
                          for jj in range_constexpr(4)]
                    xs_ = fx.Int32(Vec(lds_load_b32(lds_idx, LD_SC + (tok16 * D_STEPS + ks_) * 4))[0])
                    fx.gemm(atom, dacc, _rmem(_cat4(df[ks_]), 16, fx.Int32), _rmem(_cat4(xf), 16, fx.Int32), dacc,
                            scale_a=dsc[ks_], scale_b=xs_)
                # acc[i]: output column 8 kgrp + i of the wave's 16, token lane16
                if lane16 < ROWS:
                    av = Vec(dacc.load())
                    o = [_f2bf2(fx.Float32(av[2 * c]), fx.Float32(av[2 * c + 1])) for c in range_constexpr(4)]
                    buffer_store(Vec.from_elements(o, fx.Int32), _rsrc(_word(tab, T_SH), ROWS * HIDDEN * 2),
                                 (lane16 * HIDDEN + d * 256 + wave * 16 + kgrp * 8) * 2, offset_is_bytes=True)
            stamp(2, drain=True)
            count_out(e)

    if arg_tl is None:
        arg_tl = arg_tab
    kernel_mf(
        arg_a, arg_as, arg_w, arg_ws, arg_res, arg_nw, arg_out, arg_res_out, arg_ctr, arg_tab,
        i64_in0, i64_in1, i64_in2, i64_in3, i32_lda, i32_slot, i32_rank, i32_maxit, f64_eps, arg_tl,
    ).launch(grid=(B_ROUTER if shell & 1 else GRID, 1, 1), block=(block, 1, 1), stream=stream)


def _gate(logits, lane, bias_v, has_nvr, nvr, row, need_renorm):
    """(weight, id) of top-k slot ``lane`` (lanes < 4) of token ``row`` from its 128 bf16 logits (lane l holds
    experts l + 32 i): moe_router_topk's arithmetic from the logit on (rsh_tail ``_gate_row``)."""
    neg_inf = float("-inf")
    ept = EXPERTS // WAVE
    vals, orig, idxs = [], [], []
    for i in range(ept):
        logit = logits[i]
        e2 = fx.exp2((-logit) * fx.Float32(LOG2E))
        score = fx.Float32(rocdl.rcp(T.f32, fx.Float32(1.0) + e2))
        score = (score < fx.Float32(FLT_MIN)).select(fx.Float32(0.0), score)
        is_nan = score != score  # noqa: PLR0124 -- traced value: the NaN test
        orig.append(is_nan.select(fx.Float32(0.0), score))
        v = score + bias_v[i]
        v = (v != v).select(fx.Float32(neg_inf), v)  # noqa: PLR0124
        vals.append(v)
        idxs.append(lane + WAVE * i)

    def cas(a, b):
        sw = vals[a] < vals[b]
        vals[a], vals[b] = sw.select(vals[b], vals[a]), sw.select(vals[a], vals[b])
        orig[a], orig[b] = sw.select(orig[b], orig[a]), sw.select(orig[a], orig[b])
        idxs[a], idxs[b] = sw.select(idxs[b], idxs[a]), sw.select(idxs[a], idxs[b])

    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
        cas(a, b)
    cursor = fx.Int32(0)
    wsum = fx.Float32(0.0)
    my_id = fx.Int32(0)
    my_w = fx.Float32(0.0)
    for kk in range(TOPK):
        head_v = fx.Float32(neg_inf)
        head_i = fx.Int32(0)
        head_o = fx.Float32(0.0)
        for i in range(ept - 1, -1, -1):
            at = cursor == i
            head_v = at.select(vals[i], head_v)
            head_i = at.select(idxs[i], head_i)
            head_o = at.select(orig[i], head_o)
        mx = head_v
        for off in (16, 8, 4, 2, 1):
            mx = fx.maxnumf(mx, mx.shuffle_xor(fx.Int32(off), fx.Int32(WAVE)))
        mask = fx.Int32(rocdl.ballot(T.i32, head_v == mx))
        win = (mask != 0).select(fx.Int32(fx.cttz(mask)), fx.Int32(0))
        win_id = fx.Int32(rocdl.readlane(T.i32, head_i, win))
        won = (cursor < ept) & (head_i == win_id)
        mine = won.select(head_o, fx.Float32(0.0))
        cursor = cursor + won.select(fx.Int32(1), fx.Int32(0))
        weight = fx.Int32(rocdl.readlane(T.i32, mine.bitcast(fx.Int32), win_id & (WAVE - 1))).bitcast(fx.Float32)
        is_k = lane == kk
        my_id = is_k.select(win_id, my_id)
        my_w = is_k.select(weight, my_w)
        if need_renorm:
            wsum = wsum + weight
    inv = (fx.Float32(1.0) / fx.maxnumf(wsum, fx.Float32(RENORM_SUM_FLOOR))) if need_renorm else fx.Float32(1.0)
    out_w = my_w * inv
    pad = has_nvr & (row >= nvr)
    return pad.select(fx.Float32(0.0), out_w), pad.select(fx.Int32(-1), my_id)


launch_mf.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}

__all__ = ["launch_mf", "GRID", "TLW", "T_WORDS", "XBUF", "LGBUF", "HBUF", "HSBUF", "HCOPY", "NCOPY", "GPBUF", "RPBUF"] + [
    n for n in dir() if n.startswith("T_")
]
