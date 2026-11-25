# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSpark draft layer, MLP half, as one launch (gfx1250, MiniMax-M3.1 TP4, decode rows <= 16).

Production runs five launches: gate_up (MXFP8 GEMM, 4 K splits summed in a cluster), SwiGLU-OAI + MXFP8, down
(MXFP8 GEMM), then the k5 one-shot all-reduce + residual + Gemma RMSNorm (+ MXFP8 for the next layer's qkv).
This launch does all of it; its roles, by workgroup (128 threads, 4 waves):

* GU (384): gate_up. Unit u (96: h columns 32u..32u+31) x K quarter s (4). Wave w takes one 16-column n-block
  (gate 2u, 2u+1, up 2u, 2u+1) and the quarter's 12 K steps: all its weight loads are issued at entry, then one
  WMMA chain in K order (production's chain of a split). The FP32 partial goes to memory (device scope); the
  last of a unit's four workgroups to count sums the four in split order (production's split-K epilogue),
  rounds to bf16, runs SwiGLU-OAI (gate clamp, up clamp, bf16 store boundary) and the group-32 MXFP8 quant of
  the unit's 32 h columns, stores h (device scope), drains, and counts on ``ctl[GU_DONE]``.
* DN (384): down. 16 output columns, K 3072 = 4 waves x 6 steps; the weights are loaded at entry, so their
  stream overlaps gate_up. After all 96 units have counted, h (device-scope loads) goes through the WMMAs, the 4
  wave partials are summed in wave order, rounded to bf16, and pushed as k5 packs into every rank's inbox.
* RX (rows): k5's receiver and Gluon epilogue, one row a workgroup (a wave two slabs), with k5's MXFP8 output
  (``fp8_out``) or bf16 (the final norm).

The ctl words (normal device memory) are never reset (epoch-tagged, see CTL_BYTES). Polls are bounded
(``i32_maxit``), one load then its wait then ``s_sleep 16``, and count timeouts in ``ctl[ERR]`` (receiver
timeouts in k5's err word, as k5 does).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
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

WAVE = 32
BLOCK = 128
WMMA_K = 128
RANKS = 4
HIDDEN = 6144
INTER = 3072  # per rank
ROW_PACKS = HIDDEN // 8  # 768
RWAVES = 8  # k5: waves (slabs) of a row
SLAB = 96  # k5: packs a slab
BUFS = 3
SCOPE_SYS = 3 << 3  # gfx12 CPol SCOPE field
SCOPE_DEV = 2 << 3
SENT_LO = 0x7FFF7FFF
SENT_HI = -0x7FFF8000  # 0x80008000 as int32

K1 = HIDDEN  # gate_up K
K2 = INTER  # down K
KS = 4  # gate_up K splits (production's split-K 4)
SPS = K1 // WMMA_K // KS  # 12 K steps a split
NU = INTER // 32  # 96 units
N_GU = NU * KS  # 384
UP_NB = INTER // 16  # 192: first up n-block of the gate_up weight
N_DN = HIDDEN // 16  # 384
DN_SW = K2 // WMMA_K // 4  # 6 K steps a down wave
MAXR = 16
CL = 8  # cluster: two gate_up units' four K splits
N_RX = 8  # receiver workgroups (rows <= 8), one cluster
GRID = N_GU + N_DN + N_RX

# ctl words (int32, normal device memory, one 128-byte line each). Never reset: a launch adds N_GU to GU_DONE and
# GRID to EXIT, and its last workgroup advances CALLS (the launch count), so launch c waits for GU_DONE to
# reach (c + 1) * N_GU (int32 wrap-around differences).
GU_DONE, EXIT, CALLS, ERR = 0, 32, 64, 96
CTL_BYTES = 512
# scratch layout (bytes)
SCR_CTL = 0
SCR_HS = 1024  # int32 [MAXR, NU] h scale codes
SCR_H = SCR_HS + MAXR * NU * 4  # fp8 [MAXR, INTER]
SCR_PART = 65536  # fp32 [NU, 4 waves, KS, 16 rows, 16 cols]
SCR_BYTES = SCR_PART + NU * 4 * KS * 256 * 4

# LDS (bytes)
A_ROW = SPS * WMMA_K + 16  # 1552: an x row of a K quarter, padded
LDS_STAGE = MAXR * A_ROW  # bf16 gate/up [16 rows][64 cols]
STG_ROW = 64 * 2 + 16
LDS_OLD = LDS_STAGE + MAXR * STG_ROW
DN_PROW = 16 * 4 + 16
DN_PLANE = MAXR * DN_PROW
LDS_BYTES = max(LDS_OLD + 16, 4 * DN_PLANE, 64)


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


def _asm_f32(asm, v):
    # v_rcp_f32 / v_rsq_f32 by inline asm, as k5 does: the compiler may not keep the bare instruction otherwise
    return fx.Float32(
        _llvm.InlineAsmOp(
            res=T.f32,
            operands_=[fx.Float32(v).ir_value()],
            asm_string=asm,
            constraints="=v,v",
            has_side_effects=True,
            is_align_stack=False,
        ).result
    )


def _rcp(v):
    return _asm_f32("v_rcp_f32 $0, $1", v)


def _rsq(v):
    return fx.Float32(_llvm.call_intrinsic(T.f32, "llvm.amdgcn.rsq.f32", [fx.Float32(v).ir_value()], [], []))


def _mem_clobber():
    """Keeps the poll loop's loads inside the loop (k5: asm volatile("" ::: "memory"))."""
    _llvm.InlineAsmOp(
        res=None, operands_=[], asm_string="", constraints="~{memory}", has_side_effects=True, is_align_stack=False
    )


def _now32():
    return fx.Int32(fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], [])) & fx.Int64(0x7FFFFFFF))


def _lds_barrier():
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _cluster_sync():
    """Publish global stores to the cluster and observe the other CTAs' stores (op_merge's _cluster_sync)."""
    fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Release)
    cluster.cluster_barrier()
    fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Acquire)


def _load_agent_i32(ptr):
    return fx.Int32(
        fx.generic_load(
            ptr,
            volatile=True,
            memory_order=fx.AtomicOrdering.Monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        )
    )


def _cat16(v):
    v01 = v[0].shuffle(v[1], list(range(8)))
    v23 = v[2].shuffle(v[3], list(range(8)))
    return v01.shuffle(v23, list(range(16)))


def _rmem(v):
    t = fx.make_rmem_tensor(16, fx.Int32)
    t.store(v)
    return t


def _mx_code(amax):
    """E8M0 code of a group whose |max| is ``amax`` (fp8_groups / k5: floor 1e-10, ceil pow2 of amax / 448)."""
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


@flyc.jit
def launch_draft_mlp(
    arg_x: fx.Pointer,
    arg_xs: fx.Pointer,
    arg_w1: fx.Pointer,
    arg_w1s: fx.Pointer,
    arg_w2: fx.Pointer,
    arg_w2s: fx.Pointer,
    arg_res: fx.Pointer,
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
    fp8_out: Constexpr[int] = 1,
    max_rows: Constexpr[int] = 256,
    tl: Constexpr[int] = 0,
    sw_alpha: Constexpr[float] = 1.702,
    sw_beta: Constexpr[float] = 1.0,
    sw_limit: Constexpr[float] = 7.0,
    dn_delay: Constexpr[int] = 0,
):
    """``arg_x`` e4m3 [m, 6144] (row stride ``i32_lda``) + ``arg_xs`` e8m0 [m, 192] row-major: the post-attention
    norm's MXFP8 output. ``arg_w1``/``arg_w1s``: gate_up [6144 = gate | up, 6144], ``arg_w2``/``arg_w2s``: down
    [6144, 3072], both (16, 16)-preshuffled e4m3 + shuffle_mxfp8_bpreshuffle_scale e8m0. ``arg_res`` / ``arg_nw``:
    residual bf16 [m, 6144] and the next norm's weight (Gemma: the kernel adds 1). Out: ``arg_out`` e4m3 [m, 6144]
    + ``arg_outs`` e8m0 [m, 192] (fp8_out) or bf16 [m, 6144]; ``arg_res_out`` bf16 [m, 6144]. ``arg_ctr``: k5's
    counters (send [max_rows * 8], receive [max_rows * 8], err); ``i64_in*``: the four ranks' k5 inboxes; ``i32_slot``
    = max_rows * 768. ``arg_scr``: SCR_BYTES of scratch whose ctl words are zero before the first launch.
    ``tl``: thread 0 of every workgroup stamps the 100 MHz steady counter into ``arg_tl`` int32 [grid, 4]."""
    ctr_recv = max_rows * RWAVES
    ctr_err = 2 * max_rows * RWAVES
    alpha = float(sw_alpha)
    beta = float(sw_beta)
    limit = float(sw_limit)
    kernel_name = (f"dspark_draft_mlp_gfx1250{'_fp8' if fp8_out else '_bf16'}{'_tl' if tl else ''}"
                   f"{f'_d{dn_delay}' if dn_delay else ''}")

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK, 1, 1])
    def kernel_draft_mlp(
        arg_x: fx.Pointer,
        arg_xs: fx.Pointer,
        arg_w1: fx.Pointer,
        arg_w1s: fx.Pointer,
        arg_w2: fx.Pointer,
        arg_w2s: fx.Pointer,
        arg_res: fx.Pointer,
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
        tlp = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4), arg_tl
        )

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
        lds_i32 = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4), base_ptr
        )
        lds_f32 = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4),
            base_ptr,
        )
        ctr = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4), arg_ctr
        )
        scr_i32 = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4), arg_scr
        )
        ctl = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4), arg_ctl
        )
        scr_base = fx.Int64(fx.ptrtoint(arg_scr))
        inbox_bytes = fx.Int32(BUFS * RANKS) * i32_slot * fx.Int32(16)
        ins = [i64_in0, i64_in1, i64_in2, i64_in3]
        atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(16, 16, WMMA_K, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0, opsel_b=0)
        )

        if bid < N_GU:
            # ======================================================= gate_up, one n-block x one K quarter a wave
            u = bid // KS
            s = bid % KS
            nblk = (wave < 2).select(u * 2 + wave, fx.Int32(UP_NB) + u * 2 + (wave - 2))
            ks0 = s * SPS
            b_buf = ptr_buf_tensor(arg_w1, fx.Int32, unit_elems=4)
            sb_buf = ptr_buf_tensor(arg_w1s, fx.Int32)
            b_base = nblk * K1 + lane  # 16-byte units: an n-block is K1 * 16 bytes
            sb_col = nblk * 16 + lane16
            sb_base = (sb_col // 32) * (K1 // 4) + sb_col % 32
            # every weight load of the wave in flight at once
            bfr = [
                [buf_copy_load(b_buf, b_base + (ks0 + i) * 128 + jj * 32, fx.Int32, 4) for jj in range_constexpr(4)]
                for i in range_constexpr(SPS)
            ]
            sb = [buf_copy_load(sb_buf, sb_base + (ks0 + i) * 32, fx.Int32) for i in range_constexpr(SPS)]
            # x's K quarter -> LDS (rows >= m read as zero)
            a_buf = ptr_buf_tensor(arg_x, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda)
            sa_buf = ptr_buf_tensor(arg_xs, fx.Int32, num_records_bytes=i32_m * (K1 // 32))
            lda16 = i32_lda // 16
            QU = SPS * WMMA_K // 16  # 96 16-byte units of a row's quarter
            xa = []
            for t in range_constexpr(MAXR * QU // BLOCK):
                q = tid + t * BLOCK
                r = q // QU
                c = q % QU
                xa.append(buf_copy_load(a_buf, r * lda16 + s * QU + c, fx.Int32, 4))
            sa = [buf_copy_load(sa_buf, lane16 * (K1 // 128) + ks0 + i, fx.Int32) for i in range_constexpr(SPS)]
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
            stamp(1)
            # the partial: [unit][wave][split][row lane16][cols kgrp*8 ..], read back by the unit's cluster peers
            part_rs = create_buffer_resource_from_addr(scr_base + fx.Int64(SCR_PART), num_records_bytes=NU * 4 * KS * 1024)
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
            # SwiGLU-OAI (gate clamp, up clamp), its bf16 store boundary, then group-32 MXFP8 (a group = the wave)
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
            h_rs = create_buffer_resource_from_addr(scr_base + fx.Int64(SCR_H), num_records_bytes=MAXR * INTER)
            hs_rs = create_buffer_resource_from_addr(scr_base + fx.Int64(SCR_HS), num_records_bytes=MAXR * NU * 4)
            if (r < i32_m) & (lane % 4 == 0):
                buffer_store(q4.ir_value(), h_rs, r * INTER + u * 32 + lane, offset_is_bytes=True, cache_modifier=SCOPE_DEV)
            if (r < i32_m) & (lane == 0):
                buffer_store(code.ir_value(), hs_rs, (r * NU + u) * 4, offset_is_bytes=True, cache_modifier=SCOPE_DEV)
            rocdl.s_wait_storecnt(0)
            _lds_barrier()
            if tid == 0:
                fx.atomic_add(ctl + GU_DONE, fx.Int32(1), syncscope="agent")
            stamp(2)
        elif bid < N_GU + N_DN:
            # ======================================================= down, 16 columns, a wave 6 K steps
            n16 = bid - N_GU
            for _ in range_constexpr(dn_delay):
                rocdl.s_sleep(16)
            b_buf = ptr_buf_tensor(arg_w2, fx.Int32, unit_elems=4)
            sb_buf = ptr_buf_tensor(arg_w2s, fx.Int32)
            ks0 = wave * DN_SW
            bfr = [
                [buf_copy_load(b_buf, n16 * K2 + (ks0 + i) * 128 + jj * 32 + lane, fx.Int32, 4) for jj in range_constexpr(4)]
                for i in range_constexpr(DN_SW)
            ]
            sb_col = n16 * 16 + lane16
            sb_base = (sb_col // 32) * (K2 // 4) + sb_col % 32
            sb = [buf_copy_load(sb_buf, sb_base + (ks0 + i) * 32, fx.Int32) for i in range_constexpr(DN_SW)]
            # every gate_up unit has published h
            if tid == 0:
                it = fx.Int32(0)
                target = (_load_agent_i32(ctl + CALLS) + fx.Int32(1)) * fx.Int32(N_GU)
                cur = _load_agent_i32(ctl + GU_DONE)
                while ((cur - target) < fx.Int32(0)) & (it < i32_maxit):
                    rocdl.s_sleep(16)
                    cur = _load_agent_i32(ctl + GU_DONE)
                    it = it + 1
                if (cur - target) < fx.Int32(0):
                    fx.atomic_add(ctl + ERR, fx.Int32(1), syncscope="agent")
            _lds_barrier()
            stamp(1)
            h_buf = ptr_buf_tensor(arg_scr, fx.Int32, unit_elems=4, num_records_bytes=SCR_H + i32_m * INTER)
            hs_rs = create_buffer_resource_from_addr(scr_base + fx.Int64(SCR_HS), num_records_bytes=i32_m * NU * 4)
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
            v = Vec(acc.load())
            off = wave * DN_PLANE + lane16 * DN_PROW + kgrp * 32
            lds_store_b128(lds_idx, off, v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32))
            lds_store_b128(lds_idx, off + 16, v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32))
            _lds_barrier()
            # packs: thread -> row tid // 2, 8 columns (tid % 2); wave-order plane sum, bf16, sentinel rule
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
                rnd = fx.Int32(ctr[g])
                off = ((rnd * RANKS + i32_rank) * i32_slot + p) * 16
                for d in range_constexpr(RANKS):
                    buffer_store(data.ir_value(), rsrc[d], off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)
            if const_expr(tl):
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
            stamp(2)
        else:
            # ======================================================= k5 receiver, one row, a wave two slabs
            r = bid - (N_GU + N_DN)
            live = r < i32_m
            rr = live.select(r, fx.Int32(0))
            res_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res)), num_records_bytes=i32_m * HIDDEN * 2)
            nw_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_nw)), num_records_bytes=HIDDEN * 2)
            out_rs = create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(arg_out)), num_records_bytes=i32_m * HIDDEN * (1 if fp8_out else 2))
            os_p = fx.recast_iter(
                fx.PointerType.get(elem_ty=fx.Int8.ir_type, address_space=fx.AddressSpace.Global, alignment=1), arg_outs
            )
            ro_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res_out)), num_records_bytes=i32_m * HIDDEN * 2)
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
            srnd = [fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[g]))) for g in g_]
            rrnd = [fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[ctr_recv + g]))) for g in g_]
            p0 = [rr * ROW_PACKS + sl * SLAB + lane for sl in slabs]
            rs = [[Vec(buffer_load(res_rs, (p0[h] + 32 * j) * 4, vec_width=4, dtype=T.i32)) for j in range_constexpr(3)]
                  for h in range_constexpr(2)]
            wt = [
                [Vec(buffer_load(nw_rs, (slabs[h] * SLAB + 32 * j + lane) * 4, vec_width=4, dtype=T.i32))
                 for j in range_constexpr(3)]
                for h in range_constexpr(2)
            ]
            if live:
                for h in range_constexpr(2):
                    rearm = (rrnd[h] + 2) % BUFS
                    for i in range_constexpr(1, RANKS):
                        s_ = (i32_rank + i) & (RANKS - 1)
                        for j in range_constexpr(3):
                            my_store(((rearm * RANKS + s_) * i32_slot + p0[h] + 32 * j) * 16, sent)

            def pending(vs):
                m_ = fx.Int32(0)
                for v in vs:
                    for c in range_constexpr(4):
                        m_ = m_ | _bits(fx.Int32(v[c]))
                return fx.Int32(rocdl.ballot(T.i32, m_ != 0)) != 0

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
                    if live:
                        buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), ro_rs, (p0[h] + 32 * j) * 16,
                                     offset_is_bytes=True)
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
                    if const_expr(fp8_out):
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
                                         rr * HIDDEN + col0, offset_is_bytes=True)
                            if lane % 4 == 0:
                                fx.ptr_store(code.to(fx.Int8), os_p + (rr * (HIDDEN // 32) + col0 // 32))
                    else:
                        if live:
                            buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), out_rs, (p0[h] + 32 * j) * 16,
                                         offset_is_bytes=True)
            if live:
                for h in range_constexpr(2):
                    for j in range_constexpr(3):
                        my_store(((rrnd[h] * RANKS + i32_rank) * i32_slot + p0[h] + 32 * j) * 16, sent)
                    if lane == 0:
                        fx.ptr_store((srnd[h] + 1) % BUFS, ctr + g_[h])
                        fx.ptr_store((rrnd[h] + 1) % BUFS, ctr + (ctr_recv + g_[h]))
            if const_expr(tl):
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
            stamp(3)

        # ------------------------------------------------------------------ exit: the last workgroup resets
        if tid == 0:
            calls = _load_agent_i32(ctl + CALLS)  # advanced only after every workgroup has counted
            old_e = fx.Int32(fx.atomic_add(ctl + EXIT, fx.Int32(1), syncscope="agent"))
            if old_e + fx.Int32(1) == (calls + fx.Int32(1)) * fx.Int32(GRID):
                fx.atomic_add(ctl + CALLS, fx.Int32(1), syncscope="agent")

    kernel_draft_mlp(
        arg_x, arg_xs, arg_w1, arg_w1s, arg_w2, arg_w2s, arg_res, arg_nw, arg_out, arg_outs, arg_res_out, arg_ctr,
        arg_scr, arg_ctl, arg_tl, i64_in0, i64_in1, i64_in2, i64_in3, i32_m, i32_lda, i32_slot, i32_rank, i32_maxit, f64_eps,
        value_attrs={"rocdl.cluster_dims": f"{CL},1,1"},
    ).launch(grid=(GRID, 1, 1), block=(BLOCK, 1, 1), stream=stream, cluster=(CL, 1, 1))


launch_draft_mlp.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
