# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""DSpark draft layer after its attention core, as one launch (gfx1250, MiniMax-M3.1 TP4, decode rows <= 8):
o_proj -> all-reduce + residual + post-attention Gemma RMSNorm + MXFP8 -> gate_up -> SwiGLU-OAI + MXFP8 -> down ->
all-reduce + residual + the next Gemma RMSNorm (+ MXFP8). ``with_op=0`` is the MLP half alone (input: the
post-attention norm's MXFP8 output).

Production: quantize_fp8, o_proj, k5, gate_up, swiglu_oai_mxfp8, down, k5 (7 launches). Roles, by workgroup
(128 threads, clusters of 8), every wait on lower workgroups only (no residency assumption):

* OP (384, with_op): 16 o_proj columns; wave w K steps 4w..4w+3. Weights loaded at entry; the bf16 attention
  output is quantized in registers (fp8_groups' MXFP8, a group = two lanes), WMMA, wave-order plane sum, bf16,
  k5 packs pushed into every rank's inbox (round r).
* GU (384): gate_up unit u (32 h columns) x K quarter s; the weights are loaded at entry (with_op: they stream
  during o_proj and the first all-reduce). with_op: each cluster of 8 receives the first all-reduce for itself
  (CTA cr: row cr, k5's receiver + Gluon epilogue + MXFP8 into the cluster's region; cluster 0 also writes the
  residual and re-arms) and reads x back after the cluster barrier: no device-wide hand-off before gate_up;
  production's WMMA chain of the split;
  partials exchanged in the cluster; split-order sum, bf16, SwiGLU-OAI, bf16, group-32 MXFP8 -> h; GU_DONE.
* DN (384): 16 down columns, a wave 6 K steps; after GU_DONE, h, WMMA, plane sum, bf16, packs (round r + 1).
* RX2 (8): k5 receiver + Gluon epilogue (the next norm; k5 MXFP8 or bf16); with_op after GU_DONE, round r + 1.

Counters (``arg_ctl``, normal device memory, never reset): launch c waits for GU_DONE = (c + 1) * 384; the last workgroup out advances CALLS. Polls: one load, its wait, ``s_sleep 16``, bounded
(``i32_maxit``), timeouts counted in ``ctl[ERR]`` (receivers: k5's err word).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_load,
    ptr_buf_tensor,
)

from .dspark_draft_mlp_gfx1250 import (
    A_ROW,
    BLOCK,
    BUFS,
    CL,
    DN_PLANE,
    DN_PROW,
    DN_SW,
    HIDDEN,
    INTER,
    K1,
    K2,
    KS,
    LDS_BYTES,
    MAXR,
    N_DN,
    N_GU,
    N_RX,
    NU,
    RANKS,
    ROW_PACKS,
    RWAVES,
    SCOPE_DEV,
    SCOPE_SYS,
    SLAB,
    SPS,
    UP_NB,
    WAVE,
    WMMA_K,
    _bf_hi,
    _bf_lo,
    _bits,
    _cat16,
    _cluster_sync,
    _f2bf2,
    _fabs,
    _fmax,
    _fp8x4,
    _lds_barrier,
    _load_agent_i32,
    _mem_clobber,
    _mx_code,
    _now32,
    _rcp,
    _rmem,
    _rsq,
)

# weights stream non-temporally (TH NT, as production's decode GEMMs, wcpol 1): a layer's weights are read once a
# step and must not evict what the neighbouring kernels hit in L2 / MALL (checklist item 7)
WCPOL = 1
KO = 2048  # o_proj K (16 heads x 128 a rank)
OP_SW = KO // WMMA_K // 4  # 4 K steps an o_proj wave
N_OP = HIDDEN // 16  # 384
# ctl words (int32, normal device memory, a 128-byte line each, never reset)
GU_DONE, EXIT, CALLS, ERR, X_DONE = 0, 32, 64, 96, 128
CTL_WORDS = 256  # a build's block of counter words
CTL_BYTES = 2 * CTL_WORDS * 4
# scratch (bytes)
SCR_HS = 1024  # int32 [MAXR, NU] h scale codes
SCR_H = SCR_HS + MAXR * NU * 4  # fp8 [MAXR, INTER]
SCR_PART = 65536  # fp32 [NU, 4 waves, KS, 16 rows, 16 cols]
SCR_X = SCR_PART + NU * 4 * KS * 1024  # fp8 [N_RX, HIDDEN]: x (post-attention norm, MXFP8)
SCR_XS = SCR_X + N_RX * HIDDEN  # e8m0 [N_RX, HIDDEN / 32]
SCR_RES1 = SCR_XS + N_RX * HIDDEN // 32  # bf16 [N_RX, HIDDEN]: the residual after the first all-reduce
N_CLGU = N_GU // CL  # 48 gate_up clusters
XC_BYTES = N_RX * HIDDEN + N_RX * HIDDEN // 32  # a cluster's x (e4m3 [8, 6144]) + scales (e8m0 [8, 192])
SCR_XC = SCR_RES1 + N_RX * HIDDEN * 2
SCR_BYTES = SCR_XC + N_CLGU * XC_BYTES
MAXIT = 20000000


def layout(with_op):
    """Workgroup ranges: (OP0, RX1_0, GU0, DN0, RX2_0, GRID)."""
    if with_op:
        op0, rx1, gu0 = 0, N_OP, N_OP  # no first-receiver role: every gate_up cluster receives for itself
    else:
        op0 = rx1 = gu0 = 0
    dn0 = gu0 + N_GU
    rx2 = dn0 + N_DN
    return op0, rx1, gu0, dn0, rx2, rx2 + N_RX


@flyc.jit
def launch_draft_post(
    arg_a: fx.Pointer,
    arg_as: fx.Pointer,
    arg_wo: fx.Pointer,
    arg_wos: fx.Pointer,
    arg_w1: fx.Pointer,
    arg_w1s: fx.Pointer,
    arg_w2: fx.Pointer,
    arg_w2s: fx.Pointer,
    arg_res: fx.Pointer,
    arg_nw1: fx.Pointer,
    arg_nw: fx.Pointer,
    arg_out: fx.Pointer,
    arg_outs: fx.Pointer,
    arg_res_out: fx.Pointer,
    arg_ctr: fx.Pointer,
    arg_scr: fx.Pointer,
    arg_ctl: fx.Pointer,
    arg_tl: fx.Pointer,
    i64_in0: fx.Int64,
    i64_in1: fx.Int64,
    i64_in2: fx.Int64,
    i64_in3: fx.Int64,
    i32_m: fx.Int32,
    i32_lda: fx.Int32,
    i32_slot: fx.Int32,
    i32_rank: fx.Int32,
    i32_maxit: fx.Int32,
    f64_eps: fx.Float64,
    stream: fx.Stream,
    with_op: Constexpr[int] = 1,
    fp8_out: Constexpr[int] = 1,
    max_rows: Constexpr[int] = 256,
    tl: Constexpr[int] = 0,
    sw_alpha: Constexpr[float] = 1.702,
    sw_beta: Constexpr[float] = 1.0,
    sw_limit: Constexpr[float] = 7.0,
    gu_delay: Constexpr[int] = 0,
    poll_sleep: Constexpr[int] = 16,
):
    """with_op: ``arg_a`` bf16 [m, 2048] attention output (row stride ``i32_lda`` bytes), ``arg_wo``/``arg_wos``
    o_proj [6144, 2048], ``arg_res`` the layer's residual, ``arg_nw1`` the post-attention norm. Else ``arg_a`` e4m3
    [m, 6144] (row stride ``i32_lda`` bytes) + ``arg_as`` e8m0 [m, 192]: the post-attention norm's MXFP8 output,
    ``arg_res`` its residual. Both: gate_up ``arg_w1``/``arg_w1s`` [6144, 6144], down ``arg_w2``/``arg_w2s``
    [6144, 3072] (preshuffled e4m3 + shuffled e8m0); ``arg_nw`` the next norm; out ``arg_out`` e4m3 [m, 6144] +
    ``arg_outs`` e8m0 [m, 192] (fp8_out) or bf16, ``arg_res_out`` bf16 [m, 6144]. ``arg_ctr`` / ``i64_in*`` /
    ``i32_slot``: k5's counters, the four inboxes, max_rows * 768. ``arg_scr`` SCR_BYTES, ``arg_ctl`` CTL_BYTES
    of zeroed memory. ``tl``: thread 0 stamps the 100 MHz counter into ``arg_tl`` int32 [grid, 4]."""
    ctr_recv = max_rows * RWAVES
    ctr_err = 2 * max_rows * RWAVES
    alpha = float(sw_alpha)
    beta = float(sw_beta)
    limit = float(sw_limit)
    OP0, RX1_0, GU0, DN0, RX2_0, GRID = layout(with_op)
    kernel_name = (
        f"dspark_draft_{'post' if with_op else 'mlp'}_gfx1250{'_fp8' if fp8_out else '_bf16'}{'_tl' if tl else ''}"
        f"{f'_gd{gu_delay}' if gu_delay else ''}{f'_ps{poll_sleep}' if poll_sleep != 16 else ''}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK, 1, 1])
    def kernel_draft_post(
        arg_a: fx.Pointer,
        arg_as: fx.Pointer,
        arg_wo: fx.Pointer,
        arg_wos: fx.Pointer,
        arg_w1: fx.Pointer,
        arg_w1s: fx.Pointer,
        arg_w2: fx.Pointer,
        arg_w2s: fx.Pointer,
        arg_res: fx.Pointer,
        arg_nw1: fx.Pointer,
        arg_nw: fx.Pointer,
        arg_out: fx.Pointer,
        arg_outs: fx.Pointer,
        arg_res_out: fx.Pointer,
        arg_ctr: fx.Pointer,
        arg_scr: fx.Pointer,
        arg_ctl: fx.Pointer,
        arg_tl: fx.Pointer,
        i64_in0: fx.Int64,
        i64_in1: fx.Int64,
        i64_in2: fx.Int64,
        i64_in3: fx.Int64,
        i32_m: fx.Int32,
        i32_lda: fx.Int32,
        i32_slot: fx.Int32,
        i32_rank: fx.Int32,
        i32_maxit: fx.Int32,
        f64_eps: fx.Float64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        gptr = lambda p: fx.recast_iter(  # noqa: E731
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4), p
        )
        tlp = gptr(arg_tl)

        def stamp(slot):
            if const_expr(tl):
                if tid == 0:
                    fx.ptr_store(_now32(), tlp + (bid * 4 + slot))

        stamp(0)
        arena = fx.SharedAllocator(static=False)
        arena.allocate(LDS_BYTES)
        base_ptr = arena.base_ptr
        lds_idx = fx.Int64(fx.ptrtoint(base_ptr))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        lds_f32 = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4),
            base_ptr,
        )
        ctr = gptr(arg_ctr)
        # each build its own counter block: the epoch invariant EXIT = CALLS * GRID holds per grid size
        ctl = gptr(arg_ctl) + (0 if with_op else CTL_WORDS)
        scr_base = fx.Int64(fx.ptrtoint(arg_scr))
        inbox_bytes = fx.Int32(BUFS * RANKS) * i32_slot * fx.Int32(16)
        ins = [i64_in0, i64_in1, i64_in2, i64_in3]
        atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(16, 16, WMMA_K, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0, opsel_b=0)
        )

        def rsrc_at(off, nbytes):
            return create_buffer_resource_from_addr(scr_base + fx.Int64(off), num_records_bytes=nbytes)

        def wait_count(word, per_launch):
            """Thread 0 polls ctl[word] up to (CALLS + 1) * per_launch, then an LDS barrier."""
            if tid == 0:
                it = fx.Int32(0)
                target = (_load_agent_i32(ctl + CALLS) + fx.Int32(1)) * fx.Int32(per_launch)
                cur = _load_agent_i32(ctl + word)
                while ((cur - target) < fx.Int32(0)) & (it < i32_maxit):
                    rocdl.s_sleep(poll_sleep)
                    cur = _load_agent_i32(ctl + word)
                    it = it + 1
                if (cur - target) < fx.Int32(0):
                    fx.atomic_add(ctl + ERR, fx.Int32(1), syncscope="agent")
            _lds_barrier()

        def push_packs(n16, rnd_off):
            """Wave-order sum of the 4 LDS planes of a 16-column block, bf16, k5 packs to every inbox."""
            rsrc = [create_buffer_resource_from_addr(fx.Int64(p), num_records_bytes=inbox_bytes) for p in ins]
            r = tid // 2
            c8 = tid % 2
            if (tid < 32) & (r < i32_m):
                tot = None
                for k in range_constexpr(4):
                    lo4 = Vec(lds_load_b128(lds_idx, k * DN_PLANE + r * DN_PROW + c8 * 32)).bitcast(fx.Float32)
                    hi4 = Vec(lds_load_b128(lds_idx, k * DN_PLANE + r * DN_PROW + c8 * 32 + 16)).bitcast(fx.Float32)
                    pv = lo4.shuffle(hi4, list(range(8)))
                    tot = pv if tot is None else tot + pv
                pk = Vec(tot.to(fx.BFloat16).bitcast(fx.Int32))
                words = [fx.Int32(pk[c]) for c in range_constexpr(4)]
                words = [w_ ^ _bits(w_) for w_ in words]
                data = Vec.from_elements(words, fx.Int32)
                pcol = n16 * 2 + c8
                p = r * ROW_PACKS + pcol
                g = r * RWAVES + pcol // SLAB
                rnd = (_load_agent_i32(ctr + g) + rnd_off) % BUFS
                off = ((rnd * RANKS + i32_rank) * i32_slot + p) * 16
                for d in range_constexpr(RANKS):
                    buffer_store(data.ir_value(), rsrc[d], off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)

        def plane_store(acc):
            v = Vec(acc.load())
            off = wave * DN_PLANE + lane16 * DN_PROW + kgrp * 32
            lds_store_b128(lds_idx, off, v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32))
            lds_store_b128(lds_idx, off + 16, v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32))
            _lds_barrier()

        def receiver(r, mode):
            """k5's receiver + Gluon epilogue (rank-order sum, bf16, residual, Gemma RMSNorm, k5 MXFP8) of row r.

            mode "db" (with_op 0): one k5 call (round r of the counters): re-arm the peers' slots of buffer r + 2,
            poll, out / res_out, re-arm the own slot, counters + 1.
            mode "gu" (with_op, every gate_up cluster, CTA cr = row cr): the first all-reduce of round r as a
            private copy: x and its scales into the cluster's region (read back after the cluster barrier); cluster 0
            also stores the residual for the second receiver and re-arms the peers' slots of buffer r + 2. No own
            re-arm, no counter: the region's readers are many and the counters stay at r for the whole launch.
            mode "rx2" (with_op): after every gate_up workgroup (GU_DONE: all reads of buffer r done) the second
            all-reduce of round r + 1: re-arm all four slots of buffer r (= (r + 1) + 2), poll, out / res_out with
            the residual of the first, re-arm the own slot of buffer r + 1, counters + 2. The inbox and the counters
            end as after two k5 calls."""
            live = r < i32_m
            rr = live.select(r, fx.Int32(0))
            if const_expr(mode == "rx2"):
                wait_count(GU_DONE, N_GU)
            if const_expr(mode == "rx2"):
                res_rs = rsrc_at(SCR_RES1, N_RX * HIDDEN * 2)
                res_cm = SCOPE_DEV
            else:
                res_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res)), num_records_bytes=i32_m * HIDDEN * 2)
                res_cm = 0
            nw_rs = create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(arg_nw1 if mode == "gu" else arg_nw)), num_records_bytes=HIDDEN * 2
            )
            fp8 = mode == "gu" or fp8_out
            cl_base = (bid - GU0) // CL  # gu: this cluster
            if const_expr(mode == "gu"):
                out_rs = rsrc_at(SCR_XC, N_CLGU * XC_BYTES)
                out_off = cl_base * XC_BYTES
                os_rs = out_rs
                os_off = cl_base * XC_BYTES + N_RX * HIDDEN
                ro_rs = rsrc_at(SCR_RES1, N_RX * HIDDEN * 2)
                st_cm = 0
            else:
                out_rs = create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(arg_out)), num_records_bytes=i32_m * HIDDEN * (1 if fp8_out else 2))
                out_off = 0
                os_rs = create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(arg_outs)), num_records_bytes=i32_m * HIDDEN // 32)
                os_off = 0
                ro_rs = create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(arg_res_out)), num_records_bytes=i32_m * HIDDEN * 2)
                st_cm = 0
            store_res = (mode != "gu") | (cl_base == 0)
            res_store_cm = SCOPE_DEV if mode == "gu" else 0
            my_base = fx.Int64(i64_in0)
            for d in range_constexpr(1, RANKS):
                my_base = (i32_rank == d).select(fx.Int64(ins[d]), my_base)
            my_rs = create_buffer_resource_from_addr(my_base, num_records_bytes=inbox_bytes)
            sent = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)

            def my_store(off, data):
                buffer_store(data.ir_value(), my_rs, off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)

            def my_load(off):
                return Vec(buffer_load(my_rs, off // 4, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_SYS))

            slabs = [wave, wave + 4]
            g_ = [rr * RWAVES + sl for sl in slabs]
            srnd = [fx.Int32(rocdl.readfirstlane(T.i32, _load_agent_i32(ctr + g))) for g in g_]
            rrnd0 = [fx.Int32(rocdl.readfirstlane(T.i32, _load_agent_i32(ctr + (ctr_recv + g)))) for g in g_]
            rrnd = [(x_ + (1 if mode == "rx2" else 0)) % BUFS for x_ in rrnd0]
            p0 = [rr * ROW_PACKS + sl * SLAB + lane for sl in slabs]
            rs = [[Vec(buffer_load(res_rs, (p0[h] + 32 * j) * 4, vec_width=4, dtype=T.i32, cache_modifier=res_cm))
                   for j in range_constexpr(3)] for h in range_constexpr(2)]
            wt = [
                [Vec(buffer_load(nw_rs, (slabs[h] * SLAB + 32 * j + lane) * 4, vec_width=4, dtype=T.i32))
                 for j in range_constexpr(3)]
                for h in range_constexpr(2)
            ]
            # the start re-arm: the peers' slots of buffer (round + 2) (rx2 also the own slot: its readers are done)
            do_rearm = live if const_expr(mode != "gu") else (live & (cl_base == 0))
            if do_rearm:
                for h in range_constexpr(2):
                    rearm = (rrnd[h] + 2) % BUFS
                    for i in range_constexpr(0 if mode == "rx2" else 1, RANKS):
                        s_ = (i32_rank + i) & (RANKS - 1)
                        for j in range_constexpr(3):
                            my_store(((rearm * RANKS + s_) * i32_slot + p0[h] + 32 * j) * 16, sent)

            def pending(vs):
                m_ = fx.Int32(0)
                for v in vs:
                    for c in range_constexpr(4):
                        m_ = m_ | _bits(fx.Int32(v[c]))
                return fx.Int32(rocdl.ballot(T.i32, m_ != 0)) != 0

            if const_expr(mode != "gu"):
                stamp(1)
            s = []
            for h in range_constexpr(2):

                def offs(s_, j, h=h):
                    return ((rrnd[h] * RANKS + s_) * i32_slot + p0[h] + 32 * j) * 16

                def load_all(offs=offs):
                    _mem_clobber()
                    return tuple(my_load(offs(s_, j)) for s_ in range_constexpr(RANKS) for j in range_constexpr(3))

                (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
                pend = live & pending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
                it = fx.Int32(0)
                while pend & (it < i32_maxit):
                    rocdl.s_sleep(1)
                    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
                    pend = pending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
                    it = it + 1
                if live & pend & (lane == 0):
                    fx.atomic_add(ctr + ctr_err, fx.Int32(1), syncscope="agent")
                flat = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
                sh = [[None] * 8 for _ in range(3)]
                for j in range_constexpr(3):
                    for k in range_constexpr(8):
                        acc_ = None
                        for q in range_constexpr(RANKS):
                            wq = fx.Int32(flat[q * 3 + j][k // 2])
                            x_ = _bf_hi(wq) if k % 2 else _bf_lo(wq)
                            acc_ = x_ if acc_ is None else acc_ + x_
                        sh[j][k] = acc_
                s.append(sh)
            if const_expr(mode != "gu"):
                stamp(2)
            for h in range_constexpr(2):
                sq = None
                for j in range_constexpr(3):
                    o = []
                    for c in range_constexpr(4):
                        ab = _f2bf2(s[h][j][2 * c], s[h][j][2 * c + 1])
                        rw = fx.Int32(rs[h][j][c])
                        s[h][j][2 * c] = _bf_lo(ab) + _bf_lo(rw)
                        s[h][j][2 * c + 1] = _bf_hi(ab) + _bf_hi(rw)
                    for k in range_constexpr(8):
                        e = s[h][j][k] * s[h][j][k]
                        sq = e if sq is None else sq + e
                    for c in range_constexpr(4):
                        o.append(_f2bf2(s[h][j][2 * c], s[h][j][2 * c + 1]))
                    if live & store_res:
                        buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), ro_rs, (p0[h] + 32 * j) * 16,
                                     offset_is_bytes=True, cache_modifier=res_store_cm)
                for d_ in range_constexpr(5):
                    sq = sq + sq.shuffle_xor(fx.Int32(1 << d_), fx.Int32(WAVE))
                if lane == 0:
                    fx.ptr_store(sq, lds_f32 + slabs[h])
            _lds_barrier()
            pr = [fx.Float32(lds_f32[i]) for i in range_constexpr(RWAVES)]
            total = ((pr[0] + pr[1]) + (pr[2] + pr[3])) + ((pr[4] + pr[5]) + (pr[6] + pr[7]))
            mean = total * _rcp(fx.Float32(float(HIDDEN)))
            rstd = _rsq(fx.Float32(fx.Float64(mean) + f64_eps))
            for h in range_constexpr(2):
                for j in range_constexpr(3):
                    o = []
                    for c in range_constexpr(4):
                        wv = fx.Int32(wt[h][j][c])
                        y0 = (s[h][j][2 * c] * rstd) * (_bf_lo(wv) + fx.Float32(1.0))
                        y1 = (s[h][j][2 * c + 1] * rstd) * (_bf_hi(wv) + fx.Float32(1.0))
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
                        col0 = slabs[h] * 768 + 8 * lane + 256 * j
                        if live:
                            buffer_store(Vec.from_elements([q0, q1], fx.Int32).ir_value(), out_rs,
                                         out_off + rr * HIDDEN + col0, offset_is_bytes=True, cache_modifier=st_cm)
                            if lane % 4 == 0:
                                buffer_store(code.to(fx.Int8).ir_value(), os_rs, os_off + rr * (HIDDEN // 32) + col0 // 32,
                                             offset_is_bytes=True, cache_modifier=st_cm)
                    else:
                        if live:
                            buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), out_rs, (p0[h] + 32 * j) * 16,
                                         offset_is_bytes=True)
            if const_expr(mode != "gu"):
                if live:
                    ctr_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_ctr)),
                                                              num_records_bytes=(ctr_err + 1) * 4)
                    adv = 2 if mode == "rx2" else 1
                    for h in range_constexpr(2):
                        for j in range_constexpr(3):
                            my_store(((rrnd[h] * RANKS + i32_rank) * i32_slot + p0[h] + 32 * j) * 16, sent)
                        if lane == 0:
                            buffer_store(((srnd[h] + adv) % BUFS).ir_value(), ctr_rs, g_[h] * 4, offset_is_bytes=True,
                                         cache_modifier=SCOPE_DEV)
                            buffer_store(((rrnd0[h] + adv) % BUFS).ir_value(), ctr_rs, (ctr_recv + g_[h]) * 4,
                                         offset_is_bytes=True, cache_modifier=SCOPE_DEV)
            if const_expr(tl):
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
            if const_expr(mode != "gu"):
                stamp(3)

        if const_expr(with_op):
            in_op = bid < RX1_0
            in_rx1 = (bid >= RX1_0) & (bid < GU0)
        else:
            in_op = bid < 0
            in_rx1 = bid < 0
        if in_op:
            # ======================================================= o_proj, 16 columns, a wave 4 K steps
            if const_expr(with_op):
                n16 = bid - OP0
                b_buf = ptr_buf_tensor(arg_wo, fx.Int32, unit_elems=4)
                sb_buf = ptr_buf_tensor(arg_wos, fx.Int32)
                ks0 = wave * OP_SW
                bfr = [
                    [buf_copy_load(b_buf, n16 * KO + (ks0 + i) * 128 + jj * 32 + lane, fx.Int32, 4, cache_modifier=WCPOL)
                     for jj in range_constexpr(4)]
                    for i in range_constexpr(OP_SW)
                ]
                sb_col = n16 * 16 + lane16
                sb_base = (sb_col // 32) * (KO // 4) + sb_col % 32
                sb = [buf_copy_load(sb_buf, sb_base + (ks0 + i) * 32, fx.Int32, cache_modifier=WCPOL) for i in range_constexpr(OP_SW)]
                # the bf16 attention output: lane row lane16, K step i, chunk 2 jj + kgrp = 16 values (32 bytes)
                a_buf = ptr_buf_tensor(arg_a, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda)
                arow = lane16 * (i32_lda // 16)
                araw = [
                    [[buf_copy_load(a_buf, arow + (ks0 + i) * 16 + (2 * jj + kgrp) * 2 + hh, fx.Int32, 4)
                      for hh in range_constexpr(2)] for jj in range_constexpr(4)]
                    for i in range_constexpr(OP_SW)
                ]
                acc = fx.make_rmem_tensor(8, fx.Float32)
                acc.store(fx.full(8, 0.0, fx.Float32))
                for i in range_constexpr(OP_SW):
                    frag = []
                    sa = fx.Int32(0)
                    for jj in range_constexpr(4):
                        vals = []
                        for hh in range_constexpr(2):
                            for c in range_constexpr(4):
                                w_ = fx.Int32(araw[i][jj][hh][c])
                                vals.append(_bf_lo(w_))
                                vals.append(_bf_hi(w_))
                        amax = _fabs(vals[0])
                        for k in range_constexpr(1, 16):
                            amax = _fmax(amax, _fabs(vals[k]))
                        amax = _fmax(amax, amax.shuffle_xor(fx.Int32(16), fx.Int32(WAVE)))
                        code, recip = _mx_code(amax)
                        frag.append(Vec.from_elements([_fp8x4(vals[4 * q:4 * q + 4], recip) for q in range_constexpr(4)],
                                                      fx.Int32))
                        sa = sa | (code << fx.Int32(8 * jj))
                    fx.gemm(atom, acc, _rmem(_cat16(bfr[i])), _rmem(_cat16(frag)), acc, scale_a=sb[i], scale_b=sa)
                stamp(1)
                plane_store(acc)
                push_packs(n16, 0)
                if const_expr(tl):
                    rocdl.s_wait_storecnt(0)
                    _lds_barrier()
                stamp(2)
        elif bid < DN0:
            # ======================================================= gate_up, one n-block x one K quarter a wave
            gb = bid - GU0
            # with_op: the gate_up weights are needed only after the first all-reduce; issued gu_delay x s_sleep 16
            # (~0.5 us each) late, they do not compete with the o_proj weights on the way to the first hop
            for _ in range_constexpr(gu_delay if with_op else 0):
                rocdl.s_sleep(16)
            u = gb // KS
            s = gb % KS
            nblk = (wave < 2).select(u * 2 + wave, fx.Int32(UP_NB) + u * 2 + (wave - 2))
            ks0 = s * SPS
            b_buf = ptr_buf_tensor(arg_w1, fx.Int32, unit_elems=4)
            sb_buf = ptr_buf_tensor(arg_w1s, fx.Int32)
            b_base = nblk * K1 + lane  # 16-byte units: an n-block is K1 * 16 bytes
            sb_col = nblk * 16 + lane16
            sb_base = (sb_col // 32) * (K1 // 4) + sb_col % 32
            bfr = [
                [buf_copy_load(b_buf, b_base + (ks0 + i) * 128 + jj * 32, fx.Int32, 4, cache_modifier=WCPOL) for jj in range_constexpr(4)]
                for i in range_constexpr(SPS)
            ]
            sb = [buf_copy_load(sb_buf, sb_base + (ks0 + i) * 32, fx.Int32, cache_modifier=WCPOL) for i in range_constexpr(SPS)]
            QU = SPS * WMMA_K // 16  # 96 16-byte units of a row's quarter
            if const_expr(with_op):
                # the first all-reduce + post-attention norm, this cluster's own copy: CTA cr receives row cr
                cl = gb // CL
                receiver(gb % CL, "gu")
                rocdl.s_wait_storecnt(0)
                _cluster_sync()
                stamp(1)
                xbase = scr_base + fx.Int64(SCR_XC) + fx.Int64(cl) * fx.Int64(XC_BYTES)
                a_buf = ptr_buf_tensor(xbase, fx.Int32, unit_elems=4, num_records_bytes=i32_m * HIDDEN)
                a_base = 0
                lda16 = HIDDEN // 16
                sa_rs = create_buffer_resource_from_addr(xbase + fx.Int64(N_RX * HIDDEN),
                                                         num_records_bytes=i32_m * (K1 // 32))
                xcm = 0
            else:
                a_buf = ptr_buf_tensor(arg_a, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda)
                a_base = 0
                lda16 = i32_lda // 16
                sa_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_as)),
                                                         num_records_bytes=i32_m * (K1 // 32))
                xcm = 0
            xa = []
            for t in range_constexpr(MAXR * QU // BLOCK):
                q = tid + t * BLOCK
                r = q // QU
                c = q % QU
                row_ok = r < i32_m
                xa.append(buf_copy_load(a_buf, a_base + row_ok.select(r, fx.Int32(MAXR * 64)) * lda16 + s * QU + c,
                                        fx.Int32, 4, cache_modifier=xcm))
            sa = [fx.Int32(buffer_load(sa_rs, lane16 * (K1 // 128) + ks0 + i, vec_width=1, dtype=T.i32,
                                       cache_modifier=xcm)) for i in range_constexpr(SPS)]
            for t in range_constexpr(MAXR * QU // BLOCK):
                q = tid + t * BLOCK
                r = q // QU
                c = q % QU
                lds_store_b128(lds_idx, r * A_ROW + c * 16, xa[t])
            _lds_barrier()
            acc = fx.make_rmem_tensor(8, fx.Float32)
            acc.store(fx.full(8, 0.0, fx.Float32))
            for i in range_constexpr(SPS):
                base = lane16 * A_ROW + i * WMMA_K + kgrp * 16
                afr = [Vec(lds_load_b128(lds_idx, base + 32 * jj)) for jj in range_constexpr(4)]
                fx.gemm(atom, acc, _rmem(_cat16(bfr[i])), _rmem(_cat16(afr)), acc, scale_a=sb[i], scale_b=sa[i])
            stamp(2)
            part_rs = rsrc_at(SCR_PART, NU * 4 * KS * 1024)
            v = Vec(acc.load())
            poff = ((u * 4 + wave) * KS + s) * 1024 + lane16 * 64 + kgrp * 32
            buffer_store(v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32).ir_value(), part_rs, poff, offset_is_bytes=True)
            buffer_store(v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32).ir_value(), part_rs, poff + 16, offset_is_bytes=True)
            rocdl.s_wait_storecnt(0)
            _cluster_sync()
            # split s reduces rows s, s + 4, s + 8, s + 12: a wave one row, a lane one h column (gate and up)
            r = s + wave * 4
            nbg = lane // 16
            col = lane % 16

            def part_at(w_, sp):
                return ((u * 4 + w_) * KS + sp) * 256 + r * 16 + col

            gl = [fx.Float32(buffer_load(part_rs, part_at(nbg, sp), vec_width=1, dtype=T.f32)) for sp in range_constexpr(KS)]
            ul = [fx.Float32(buffer_load(part_rs, part_at(nbg + 2, sp), vec_width=1, dtype=T.f32))
                  for sp in range_constexpr(KS)]
            gt = ((gl[0] + gl[1]) + gl[2]) + gl[3]
            ut = ((ul[0] + ul[1]) + ul[2]) + ul[3]
            gu2 = _f2bf2(gt, ut)  # production's bf16 gate_up output
            g = _bf_lo(gu2)
            up = _bf_hi(gu2)
            g = (g > fx.Float32(limit)).select(fx.Float32(limit), g)
            up = (up < fx.Float32(-limit)).select(fx.Float32(-limit), up)
            up = (up > fx.Float32(limit)).select(fx.Float32(limit), up)
            sig = fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(fx.Float32(0.0) - fx.Float32(alpha) * g))
            act = g * sig * (up + fx.Float32(beta))
            a = _bf_lo(_f2bf2(act, act))
            amax = _fabs(a)
            for d_ in range_constexpr(5):
                amax = _fmax(amax, amax.shuffle_xor(fx.Int32(1 << d_), fx.Int32(WAVE)))
            code, recip = _mx_code(amax)
            a1 = a.shuffle_xor(fx.Int32(1), fx.Int32(WAVE))
            a2 = a.shuffle_xor(fx.Int32(2), fx.Int32(WAVE))
            a3 = a.shuffle_xor(fx.Int32(3), fx.Int32(WAVE))
            q4 = _fp8x4([a, a1, a2, a3], recip)
            h_rs = rsrc_at(SCR_H, MAXR * INTER)
            hs_rs = rsrc_at(SCR_HS, MAXR * NU * 4)
            if (r < i32_m) & (lane % 4 == 0):
                buffer_store(q4.ir_value(), h_rs, r * INTER + u * 32 + lane, offset_is_bytes=True, cache_modifier=SCOPE_DEV)
            if (r < i32_m) & (lane == 0):
                buffer_store(code.ir_value(), hs_rs, (r * NU + u) * 4, offset_is_bytes=True, cache_modifier=SCOPE_DEV)
            rocdl.s_wait_storecnt(0)
            _lds_barrier()
            if tid == 0:
                fx.atomic_add(ctl + GU_DONE, fx.Int32(1), syncscope="agent")
            stamp(3)
        elif bid < RX2_0:
            # ======================================================= down, 16 columns, a wave 6 K steps
            n16 = bid - DN0
            b_buf = ptr_buf_tensor(arg_w2, fx.Int32, unit_elems=4)
            sb_buf = ptr_buf_tensor(arg_w2s, fx.Int32)
            ks0 = wave * DN_SW
            bfr = [
                [buf_copy_load(b_buf, n16 * K2 + (ks0 + i) * 128 + jj * 32 + lane, fx.Int32, 4, cache_modifier=WCPOL) for jj in range_constexpr(4)]
                for i in range_constexpr(DN_SW)
            ]
            sb_col = n16 * 16 + lane16
            sb_base = (sb_col // 32) * (K2 // 4) + sb_col % 32
            sb = [buf_copy_load(sb_buf, sb_base + (ks0 + i) * 32, fx.Int32, cache_modifier=WCPOL) for i in range_constexpr(DN_SW)]
            wait_count(GU_DONE, N_GU)
            stamp(1)
            h_buf = ptr_buf_tensor(arg_scr, fx.Int32, unit_elems=4, num_records_bytes=SCR_H + i32_m * INTER)
            hs_rs = rsrc_at(SCR_HS, i32_m * NU * 4)
            hrow = (SCR_H + lane16 * INTER) // 16 + kgrp
            live_r = lane16 < i32_m
            afr = [
                [buf_copy_load(h_buf, hrow + (ks0 + i) * 8 + jj * 2, fx.Int32, 4, cache_modifier=SCOPE_DEV)
                 for jj in range_constexpr(4)]
                for i in range_constexpr(DN_SW)
            ]
            sav = [
                Vec(buffer_load(hs_rs, lane16 * NU + (ks0 + i) * 4, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_DEV))
                for i in range_constexpr(DN_SW)
            ]
            acc = fx.make_rmem_tensor(8, fx.Float32)
            acc.store(fx.full(8, 0.0, fx.Float32))
            for i in range_constexpr(DN_SW):
                sv = sav[i]
                sa = (fx.Int32(sv[0]) & fx.Int32(255)) | ((fx.Int32(sv[1]) & fx.Int32(255)) << fx.Int32(8)) | (
                    (fx.Int32(sv[2]) & fx.Int32(255)) << fx.Int32(16)) | (fx.Int32(sv[3]) << fx.Int32(24))
                sa = live_r.select(sa, fx.Int32(0))
                fx.gemm(atom, acc, _rmem(_cat16(bfr[i])), _rmem(_cat16(afr[i])), acc, scale_a=sb[i], scale_b=sa)
            plane_store(acc)
            push_packs(n16, 1 if with_op else 0)
            if const_expr(tl):
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
            stamp(2)
        else:
            receiver(bid - RX2_0, "rx2" if with_op else "db")

        # ------------------------------------------------------------------ exit: the last workgroup advances CALLS
        if tid == 0:
            calls = _load_agent_i32(ctl + CALLS)  # advanced only after every workgroup has counted
            old_e = fx.Int32(fx.atomic_add(ctl + EXIT, fx.Int32(1), syncscope="agent"))
            if old_e + fx.Int32(1) == (calls + fx.Int32(1)) * fx.Int32(GRID):
                fx.atomic_add(ctl + CALLS, fx.Int32(1), syncscope="agent")

    kernel_draft_post(
        arg_a, arg_as, arg_wo, arg_wos, arg_w1, arg_w1s, arg_w2, arg_w2s, arg_res, arg_nw1, arg_nw, arg_out, arg_outs,
        arg_res_out, arg_ctr, arg_scr, arg_ctl, arg_tl, i64_in0, i64_in1, i64_in2, i64_in3, i32_m, i32_lda, i32_slot,
        i32_rank, i32_maxit, f64_eps,
        value_attrs={"rocdl.cluster_dims": f"{CL},1,1"},
    ).launch(grid=(GRID, 1, 1), block=(BLOCK, 1, 1), stream=stream, cluster=(CL, 1, 1))


launch_draft_post.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
