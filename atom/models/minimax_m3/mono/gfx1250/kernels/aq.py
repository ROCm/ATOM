# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AQ1: the MiniMax-M3.1 TP4 decode attention half's QKV chain and its CP index
score in ONE launch (gfx1250, one request x ``ql`` rows).

    QKV chain (``mxfp8_qkv_rope_nvfp4_gfx1250``, body unchanged)
      -> index q [ql, niq, 128] bf16           (in-launch: data-as-flag words)
      -> index-K NVFP4 rows of this step         (in-launch: one epoch word a row)
    CP score (ATOM ``indexer_cp_score_flydsl``, same instructions and order)
      -> fp32 scores [niq, ql, local blocks]    (read by the next launch: the exchange)

Workgroups (1-D grid, 12 waves each, hardware clusters of 16 along x):

* ``0 .. 16 * HEADS - 1``: the QKV chain; workgroup ``16 h + r`` is cluster rank r of
  head h (the production kernel's (rank, head) grid, row group 0). Its epilogue also
  publishes index-q rows as k5 sentinel words (0xffff -> 0x7fff) in copy E % 3 of the
  q hand-off buffer and re-arms copy (E + 1) % 3 (last written two launches ago), and
  the index-K head's rows set word ``row`` of the index-K flags to E after their
  NVFP4 stores and a device-scope release.
* ``16 * HEADS .. GRID - 1``: S score workgroups (S a multiple of 16; their clusters
  never use the cluster barrier). Workgroup s scores local blocks
  [s c, s c + c), c = ceil(live / S). Waves 0-3 run the scorer (keys stream into
  their LDS rings at entry, while the chain runs; blocks that hold this step's keys
  only after the index-K flags); waves 4-7 poll the index q (device-scope loads, the
  words they convert) and stage the B operands in their own LDS region; waves 8-11
  only pass the barriers.
* ``xchg`` (full AQ): then one workgroup an (index head h, query row r), last in the
  grid. Its 12 waves poll the scores of (h, r) (data-as-flag words, re-armed after
  reading), keep a bitonic top-16 of their 32-key tiles, and wave 0 merges the 12
  lists through LDS, pushes them to head h's owner (rank h, the CP mailbox), and on
  the owner receives the 4 shards' keys for row r and merges them into the indices,
  the sparse block table and ctx (ATOM ``_select_exchange_merge``'s bytes).

E is the launch number: every workgroup counts its own launches in a private word.
Every wait is bounded (``maxit`` polls); a timed-out wait counts in ``err`` and goes on.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.buffer_ops import buffer_load, buffer_store
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.mxfp8_qkv_rope_nvfp4_gfx1250 import (
    A_BATCH,
    CPOL_DEV,
    D,
    K_DIM,
    NKV,
    NQ,
    P_PAD,
    ROT,
    WAVE,
    WMMA_K,
    _bf16x4,
    _gptr,
    _lds_barrier,
    _nvfp4_lane,
    _rsq,
    _xor,
    chain_config_error,
    chain_heads,
    chain_lds_bytes,
)
from .rsh_tail import SCOPE_DEV, _rsrc, _uni, sentinel_bits
SCOPE_SYS = 3 << 3  # gfx12 CPol SCOPE field: system (peer memory over IPC, as k5 pushes)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

# ---- the scorer's constants (ATOM indexer_cp_score_flydsl) ----
SC_WAVES = 4
SC_THREADS = SC_WAVES * WAVE
PAGE = 128
MAX_BLOCKS = 64
SLOT_PAYLOAD = 32 * 64
SLOT = SLOT_PAYLOAD + 32 * 8
QOP_LANE = 64
E8M0_ONE = 0x7F7F7F7F7F7F7F7F
E8M0_LO = 0x7B7B7B7B7B7B7B7B
E4M3_MAX = 448.0
COPIES = 5
POLL_SLEEP = 1
TLW = 8  # timeline words a workgroup (tl builds): 0 entry, 1-7 role stamps (see the roles)


def _attr(text):
    return ir.Attribute.parse(text)


def _wmma(a, b, c, sa, sb, row_a, row_b):
    """D = A(e2m1, E4M3 scales row ``row_a``) x B(e4m3, E8M0 scales) + C (the scorer's)."""
    return fx.Vector(
        rocdl.wmma_scale16_f32_16x16x128_f8f6f4(
            fx.Vector.make_type(8, fx.Float32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(c),
            fx.as_ir_value(sa),
            fx.as_ir_value(sb),
            fmtA=_attr("#rocdl<matrix_format fp4_e2m1>"),
            fmtB=_attr("#rocdl<matrix_format fp8_e4m3>"),
            modC=_attr("#rocdl<wmma_c_modifier none>"),
            scaleAType=_attr(f"#rocdl<wmma_matrix_scale row{row_a}>"),
            fmtScaleA=_attr("#rocdl<wmma_matrix_scale_format e4m3>"),
            scaleBType=_attr(f"#rocdl<wmma_matrix_scale row{row_b}>"),
            fmtScaleB=_attr("#rocdl<wmma_matrix_scale_format e8>"),
            reuseA=False,
            reuseB=False,
        ).result
    )


def _to_fp8x8(v8):
    return fx.Vector(rocdl.cvt_scalef32_pk8_fp8_f32(T.i32x2, fx.as_ir_value(v8), fx.as_ir_value(fx.Float32(1.0))))


def _from_fp8x8(w2):
    return fx.Vector(
        rocdl.cvt_scale_pk8_f32_fp8(T.vec(8, T.f32), fx.as_ir_value(w2), fx.as_ir_value(fx.Int32(0x7F)), 0)
    )


def _swz(row, piece):
    return row * 64 + ((piece ^ ((row >> 2) & 3)) * 16)


def _lds_i32(base_u8, off, n):
    return fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, min(4 * n, 16)),
        fx.add_offset(base_u8, off),
    )


def getpc():
    """This instruction's PC (llvm.amdgcn.s.getpc)."""
    return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.getpc", [], [], []))


def prefetch_code_to(end_pc, max_req):
    """s_prefetch_inst over [this PC, end_pc) in 2 KiB requests, at most ``max_req``. ``end_pc`` must be a PC inside
    this kernel (recorded by an earlier launch with ``getpc``; 0 = none), so no request reaches past the code object:
    an unbounded pc-relative prefetch faulted on gfx1250 (SQC inst permission fault, 11-26). LLVM has no intrinsic
    for s_prefetch_inst."""
    pc0 = getpc()
    # a recorded PC past this one and within 1 MiB (a kernel's code is far smaller); else nothing is prefetched
    sane = (end_pc > pc0) & (end_pc - pc0 < fx.Int64(1 << 20))
    for k in range_constexpr(max_req):
        a = pc0 + fx.Int64(k * 2048)
        # a request that would pass end_pc re-fetches this PC's own line instead (branch-free, always mapped)
        a = (sane & (a + fx.Int64(2048) <= end_pc)).select(a, pc0)
        _llvm.InlineAsmOp(None, [a.ir_value()], "s_prefetch_inst $0, 0x0, null, 31", "s", has_side_effects=True)


def aq1_grid(niq: int, n_score: int) -> int:
    return 16 * chain_heads(niq) + n_score


def aq_xchg_wgs(niq: int, ql: int) -> int:
    """The exchange's workgroups: one an (index head, query row), padded to whole clusters of 16."""
    return (niq * ql + 15) // 16 * 16


def aq1_lds_bytes(cfg, ql: int, niq: int, ring: int, groups: int = 2) -> int:
    tile_m, ns, ks, kw = cfg[0], cfg[1], cfg[2], cfg[3]
    astage = cfg[7]
    nqt = (niq * ql + 15) // 16
    score = groups * SC_WAVES * ring * SLOT + MAX_BLOCKS * 32 * 4 + nqt * 2 * 32 * QOP_LANE
    return max(chain_lds_bytes(tile_m, ns, ks, kw, astage), score)


@flyc.jit
def launch_aq1(
    arg_a: fx.Pointer, arg_as: fx.Pointer, arg_w: fx.Pointer, arg_ws: fx.Pointer, arg_slab: fx.Pointer,
    arg_q: fx.Pointer, arg_iq: fx.Pointer, arg_qw: fx.Pointer, arg_kw: fx.Pointer, arg_iqw: fx.Pointer,
    arg_ikw: fx.Pointer, arg_cs: fx.Pointer, arg_pos: fx.Pointer, arg_slot: fx.Pointer, arg_islot: fx.Pointer,
    arg_kp: fx.Pointer, arg_vp: fx.Pointer, arg_ksc: fx.Pointer, arg_vsc: fx.Pointer, arg_ip: fx.Pointer,
    arg_isc: fx.Pointer,
    arg_bt: fx.Pointer, arg_len: fx.Pointer, arg_out: fx.Pointer,
    arg_aqq: fx.Pointer, arg_flag: fx.Pointer, arg_cnt: fx.Pointer, arg_err: fx.Pointer, arg_tl: fx.Pointer,
    arg_scb: fx.Pointer, arg_peers: fx.Pointer, arg_send: fx.Pointer, arg_recv: fx.Pointer, arg_tmo: fx.Pointer,
    arg_idx: fx.Pointer, arg_sbt: fx.Pointer, arg_sctx: fx.Pointer, arg_pc: fx.Pointer,
    i32_m: fx.Int32, i32_lda: fx.Int32, i32_mpad: fx.Int32, i32_cs_rows: fx.Int32,
    i64_kp_page: fx.Int64, i64_ks_page: fx.Int64, i64_ip_page: fx.Int64, i64_is_page: fx.Int64,
    i32_eps: fx.Int32,
    i32_bt_stride: fx.Int32, i32_local: fx.Int32, i32_global: fx.Int32, f32_scale: fx.Float32, i32_maxit: fx.Int32,
    stream: fx.Stream,
    tile_m: Constexpr[int], ns: Constexpr[int], ks: Constexpr[int], kw: Constexpr[int], wm: Constexpr[int],
    wn: Constexpr[int], depth: Constexpr[int], niq: Constexpr[int], astage: Constexpr[int], wcpol: Constexpr[int],
    ql: Constexpr[int], rank_cp: Constexpr[int], world: Constexpr[int], ring: Constexpr[int],
    n_score: Constexpr[int], tl: Constexpr[int] = 0, xchg: Constexpr[int] = 0, topk: Constexpr[int] = 16,
    init_blocks: Constexpr[int] = 0, local_keep: Constexpr[int] = 0, max_rows: Constexpr[int] = 64,
    local_max: Constexpr[int] = 1024, slot_keys: Constexpr[int] = 64, ipf_kib: Constexpr[int] = 0,
    owner_fast: Constexpr[int] = 1, merge_par: Constexpr[int] = 0, sc_groups: Constexpr[int] = 2,
    owner_v2: Constexpr[int] = 0, kchain: Constexpr[int] = 0,
):  # fmt: skip
    """``kchain`` 1: the QKV GEMM sums K as the tuned ``gemm_a8w8 ..._sk4_fsk`` does -- each K split one chain of
    WMMA steps from a zero accumulator (the split's K-waves hand the accumulator on through LDS, in K order), the
    splits summed in split order, one bf16 rounding -- instead of the K-waves' wave-order sum (0, the fused chain's
    order). For a layer whose QKV production runs that GEMM (the first decoder layer)."""
    cfg = (tile_m, ns, ks, kw, wm, wn, depth, astage, wcpol, 0)
    err = chain_config_error(tile_m, ns, ks, kw, wm, wn, depth, astage)
    if err is not None:
        raise ValueError(f"aq1: {err}")
    assert astage == 1 and tile_m == 16, "aq1: the 8-row decode config (one 16-row tile, LDS-staged activation)"
    K = K_DIM
    HEADS = chain_heads(niq)
    N = HEADS * D
    csize = ns * ks
    assert csize == 16 and n_score % 16 == 0 and n_score >= 16
    N_QKV = csize * HEADS
    # exchange workgroups: one an (index head, query row), padded to whole clusters of 16
    N_SEL = aq_xchg_wgs(niq, ql) if xchg else 0
    GRID = N_QKV + n_score + N_SEL
    assert not xchg or (topk == 16 and world * topk == 64 and niq == world and local_max % 32 == 0), "aq: the exchange"
    steps = K // WMMA_K // ks
    steps_w = steps // kw
    CPR = steps * WMMA_K // 16
    cols = D // ns
    nbt = cols // 16
    mbt = tile_m // 16
    mbw = mbt // wm
    nbw = nbt // wn
    waves = kw * wm * wn
    block = waves * WAVE
    assert block >= 3 * SC_THREADS, "aq1: the score role needs 12 waves a workgroup"
    # the exchange: wave w takes tiles w, w + 12, ... (up to 6) of 32 keys; wave 0 merges the 12 lists in pairs
    assert not xchg or (waves == 12 and local_max <= 6 * waves * 32), "aq: the exchange's waves / tiles"
    depth = max(1, min(depth, steps_w))
    sa_opsel = mbw % 2 == 0
    sa_loads = mbw // 2 if sa_opsel else mbw
    KB4 = K // 128
    per_thr = tile_m * CPR // block
    assert sc_groups in (2, 3), "aq: the score role's compute groups (2: waves 0-3, 8-11; 3: all 12 waves)"
    lds_bytes = aq1_lds_bytes(cfg, ql, niq, ring, sc_groups)
    # kchain: the accumulator hand-off between K-waves, past everything else (the activation slab is still read)
    RELAY = lds_bytes
    RELAY_PITCH = (D // ns) * 4
    if kchain:
        lds_bytes += tile_m * RELAY_PITCH
    # the grid is co-resident at 2 workgroups a CU (320 KB of LDS a CU): its workgroups wait on each other
    assert lds_bytes <= 160 * 1024, f"aq: {lds_bytes} B of LDS a workgroup (2 a CU need <= 160 KB)"
    # owner_v2: the request's block-table row in LDS past the exchange's lists ((waves + 6) x 16 keys x 8 B)
    PT_OFF = 4096
    assert not owner_v2 or (owner_fast and xchg and PT_OFF + local_max * world * 4 <= RELAY), "aq: owner_v2"
    P_ROW = cols * 4 + P_PAD
    P_PLANE = tile_m * P_ROW
    p_units = tile_m * cols // 4
    U_ROW = cols // 4
    rpr = (tile_m + csize - 1) // csize
    row_iters = (rpr + waves - 1) // waves
    IQ0 = NQ + 2 * NKV
    IK = IQ0 + niq
    cpol = CPOL_DEV
    # the scorer
    heads_i = niq
    rows_s = heads_i * ql
    nqt = (rows_s + 15) // 16
    assert 1 <= nqt <= 2 and ql <= 128 and 2 <= ring <= 8
    RING = ring
    RING_BYTES = sc_groups * SC_WAVES * RING * SLOT  # compute groups of 4 waves, a ring a wave
    PART = MAX_BLOCKS * 32
    Q_LDS = RING_BYTES + PART * 4  # B operands [p][lane] x 64 B, p = 2 qt + hl
    JH = 4 // (SC_THREADS // (32 * nqt))
    QCOPY = ql * niq * D * 2  # bytes of one index-q hand-off copy
    kernel_name = (
        f"{'aqx_qkv_score_xchg' if xchg else 'aq1_qkv_score'}_gfx1250_n{N}_t{tile_m}_ns{ns}_ks{ks}_w{kw}x{wm}x{wn}"
        f"_d{depth}_c{wcpol}"
        f"_h{heads_i}_q{ql}_r{rank_cp}w{world}_g{ring}_s{n_score}{'_tl' if tl else ''}"
        f"{f'_x{topk}i{init_blocks}l{local_keep}m{max_rows}' if xchg else ''}{f'_pf{ipf_kib}' if ipf_kib else ''}{'' if owner_fast else '_ow0'}"
        f"{'_mp' if merge_par else ''}{f'_sg{sc_groups}' if sc_groups != 2 else ''}{'_ov2' if owner_v2 else ''}{'_kc' if kchain else ''}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_aq1(
        arg_a: fx.Pointer, arg_as: fx.Pointer, arg_w: fx.Pointer, arg_ws: fx.Pointer, arg_slab: fx.Pointer,
        arg_q: fx.Pointer, arg_iq: fx.Pointer, arg_qw: fx.Pointer, arg_kw: fx.Pointer, arg_iqw: fx.Pointer,
        arg_ikw: fx.Pointer, arg_cs: fx.Pointer, arg_pos: fx.Pointer, arg_slot: fx.Pointer, arg_islot: fx.Pointer,
        arg_kp: fx.Pointer, arg_vp: fx.Pointer, arg_ksc: fx.Pointer, arg_vsc: fx.Pointer, arg_ip: fx.Pointer,
        arg_isc: fx.Pointer,
        arg_bt: fx.Pointer, arg_len: fx.Pointer, arg_out: fx.Pointer,
        arg_aqq: fx.Pointer, arg_flag: fx.Pointer, arg_cnt: fx.Pointer, arg_err: fx.Pointer, arg_tl: fx.Pointer,
        arg_scb: fx.Pointer, arg_peers: fx.Pointer, arg_send: fx.Pointer, arg_recv: fx.Pointer, arg_tmo: fx.Pointer,
        arg_idx: fx.Pointer, arg_sbt: fx.Pointer, arg_sctx: fx.Pointer, arg_pc: fx.Pointer,
        i32_m: fx.Int32, i32_lda: fx.Int32, i32_mpad: fx.Int32, i32_cs_rows: fx.Int32,
        i64_kp_page: fx.Int64, i64_ks_page: fx.Int64, i64_ip_page: fx.Int64, i64_is_page: fx.Int64,
        i32_eps: fx.Int32,
        i32_bt_stride: fx.Int32, i32_local: fx.Int32, i32_global: fx.Int32, f32_scale: fx.Float32,
        i32_maxit: fx.Int32,
    ):  # fmt: skip
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _bid_y, _bid_z = fx.block_idx
        bid = fx.Int32(bid_x)
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        cnt_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_cnt)), GRID * 4)
        err_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_err)), 16)
        flag_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_flag)), ql * 4)
        # E: this workgroup's launch number (its private count + 1)
        e_ = _uni(fx.Int32(buffer_load(cnt_rs, bid, vec_width=1, dtype=T.i32, cache_modifier=SCOPE_DEV))) + 1
        aqq_base = fx.Int64(fx.ptrtoint(arg_aqq))
        qcur_rs = _rsrc(aqq_base + fx.Int64((e_ % 3) * QCOPY), QCOPY)
        qnext_rs = _rsrc(aqq_base + fx.Int64(((e_ + 1) % 3) * QCOPY), QCOPY)

        scb_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_scb)), heads_i * ql * local_max * 4)
        arena = fx.SharedAllocator(static=False)
        arena.allocate(lds_bytes)
        lds_idx = fx.Int64(fx.ptrtoint(arena.base_ptr))

        def stamp(slot, who):
            # diagnostic timeline (tl): no stable API reads the steady counter
            if const_expr(tl):
                if tid == who:
                    now = fx.Int32(fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))
                                   & fx.Int64(0x7FFFFFFF))
                    buffer_store(now, _rsrc(fx.Int64(fx.ptrtoint(arg_tl)), GRID * 4 * TLW), bid * TLW + slot)

        stamp(0, 0)

        pc_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_pc)), 8)

        def err_pc_end():
            """This kernel's end PC recorded by an earlier launch (``arg_pc``, one buffer a compiled kernel)."""
            w2 = Vec(buffer_load(pc_rs, 0, vec_width=2, dtype=T.i32))
            return (fx.Int64(_uni(fx.Int32(w2[1]))) << fx.Int64(32)) | (fx.Int64(_uni(fx.Int32(w2[0])))
                                                                       & fx.Int64(0xFFFFFFFF))

        def timeout(pend):
            if pend & (lane == 0):
                fx.atomic_add(
                    fx.recast_iter(fx.Int32, fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                                                          fx.Int64(fx.ptrtoint(arg_err)))),
                    fx.Int32(1), syncscope="agent")

        if bid < N_QKV:
            # ======================= the QKV chain (production's body) =======================
            rank = bid % csize
            head = bid // csize
            mgrp = fx.Int32(0)
            npart = rank % ns
            kpart = rank // ns
            w_k = wave // (wm * wn)
            w_m = (wave // wn) % wm
            w_n = wave % wn
            lane16 = lane % 16
            kgrp = lane // 16
            m0 = mgrp * tile_m
            ks0 = kpart * steps

            lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

            r_base = rank * rpr
            e_rows = [r_base + wave + t * waves for t in range_constexpr(row_iters)]
            e_valid = [
                ((wave + t * waves) < rpr) & (r < tile_m) & (m0 + r < i32_m)
                for t, r in enumerate(e_rows)
            ]
            pos_buf = ptr_buf_tensor(arg_pos, fx.Int32, num_records_bytes=i32_m * 8)
            slot_buf = ptr_buf_tensor(arg_slot, fx.Int32, num_records_bytes=i32_m * 8)
            islot_buf = ptr_buf_tensor(arg_islot, fx.Int32, num_records_bytes=i32_m * 8)
            e_pos = [buf_copy_load(pos_buf, (m0 + r) * 2, fx.Int32) for r in e_rows]
            e_slot = [buf_copy_load(slot_buf, (m0 + r) * 2, fx.Int32) for r in e_rows]
            e_islot = [buf_copy_load(islot_buf, (m0 + r) * 2, fx.Int32) for r in e_rows]

            a_buf = ptr_buf_tensor(arg_a, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda)
            lda16 = i32_lda // 16
            sa_buf = ptr_buf_tensor(arg_as, fx.Int32, num_records_bytes=i32_m * (K // 32))
            b_buf = ptr_buf_tensor(arg_w, fx.Int32, unit_elems=4)
            sb_buf = ptr_buf_tensor(arg_ws, fx.Int32)

            nblk0 = head * 8 + npart * nbt + w_n * nbw
            b_base = [(nblk0 + j) * K + lane for j in range_constexpr(nbw)]
            sb_col = [(nblk0 + j) * 16 + lane16 for j in range_constexpr(nbw)]
            sb_base = [(c // 32) * (K // 4) + c % 32 for c in sb_col]
            mblk0 = w_m * mbw
            sa_row = [
                (m0 + (mblk0 + 2 * p) * 16 + lane)
                if const_expr(sa_opsel)
                else (m0 + (mblk0 + p) * 16 + lane16)
                for p in range_constexpr(sa_loads)
            ]
            sa_base = [r * KB4 for r in sa_row]
            st0 = w_k * steps_w

            def _cat16(v):
                v01 = v[0].shuffle(v[1], list(range(8)))
                v23 = v[2].shuffle(v[3], list(range(8)))
                return v01.shuffle(v23, list(range(16)))

            def load_b_step(i):
                ksg = ks0 + st0 + i
                bfr = [
                    [
                        buf_copy_load(b_buf, b_base[j] + ksg * 128 + jj * 32, fx.Int32, 4, cache_modifier=wcpol)
                        for jj in range_constexpr(4)
                    ]
                    for j in range_constexpr(nbw)
                ]
                sb = [buf_copy_load(sb_buf, sb_base[j] + ksg * 32, fx.Int32) for j in range_constexpr(nbw)]
                sa = [buf_copy_load(sa_buf, sa_base[p] + ksg, fx.Int32) for p in range_constexpr(sa_loads)]
                return [bfr, sb, sa, None]

            def a_load(u):
                return buf_copy_load(a_buf, (m0 + u // CPR) * lda16 + ks0 * 8 + u % CPR, fx.Int32, 4)

            def a_lds_off(u):
                r = u // CPR
                c = u % CPR
                return ((r // 16) * steps + c // 8) * 2048 + (c % 8) * 256 + ((r % 16) ^ (c % 16)) * 16

            frag_lane = [
                [(jj * 2 + kgrp) * 256 + (lane16 ^ (kgrp + (par * 8 + jj * 2))) * 16 for jj in range_constexpr(4)]
                for par in range_constexpr(2)
            ]

            def load_a_frag(mi, i):
                s = st0 + i
                base = ((mblk0 + mi) * steps + s) * 2048
                par = (w_k * steps_w + i) % 2 if const_expr(steps_w % 2) else i % 2
                if const_expr(steps_w % 2):
                    return [
                        Vec(lds_load_b128(lds_idx, base + (jj * 2 + kgrp) * 256
                                          + (lane16 ^ (kgrp + (par * 8 + jj * 2))) * 16))
                        for jj in range_constexpr(4)
                    ]
                return [Vec(lds_load_b128(lds_idx, base + frag_lane[par][jj])) for jj in range_constexpr(4)]

            atoms = [
                fx.make_mma_atom(
                    fx.rocdl.WMMAScale(16, 16, WMMA_K, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0,
                                       opsel_b=sel)
                )
                for sel in range_constexpr(2 if sa_opsel else 1)
            ]
            acc = [[fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(nbw)] for _ in range_constexpr(mbw)]
            for row_ in acc:
                for c in row_:
                    c.store(fx.full(8, 0.0, fx.Float32))

            def _rmem(v):
                t_ = fx.make_rmem_tensor(16, fx.Int32)
                t_.store(v)
                return t_

            def compute_step(st, afr):
                bfr, sb, sa, _ = st
                bt_ = [_rmem(_cat16(bfr[j])) for j in range_constexpr(nbw)]
                at_ = [_rmem(_cat16(afr[mi])) for mi in range_constexpr(mbw)]
                for mi in range_constexpr(mbw):
                    for j in range_constexpr(nbw):
                        fx.gemm(atoms[mi % 2 if sa_opsel else 0], acc[mi][j], bt_[j], at_[mi], acc[mi][j],
                                scale_a=sb[j], scale_b=sa[mi // 2 if sa_opsel else mi])

            batches = [
                [tid + (b0 + v) * block for v in range_constexpr(min(A_BATCH, per_thr - b0))]
                for b0 in range_constexpr(0, per_thr, A_BATCH)
            ]
            a_vals0 = [a_load(u) for u in batches[0]]
            rocdl.sched_barrier(0)
            inflight = [load_b_step(i) for i in range_constexpr(depth)]
            rocdl.sched_barrier(0)
            for u, val in zip(batches[0], a_vals0):
                lds_store_b128(lds_idx, a_lds_off(u), val)
            for units in batches[1:]:
                vals = [a_load(u) for u in units]
                for u, val in zip(units, vals):
                    lds_store_b128(lds_idx, a_lds_off(u), val)
            _lds_barrier()
            rocdl.sched_barrier(0)

            def run_steps(fl):
                for i in range_constexpr(steps_w):
                    st = fl.pop(0)
                    if const_expr(i + depth < steps_w):
                        fl.append(load_b_step(i + depth))
                        rocdl.sched_barrier(0)
                    afr = [load_a_frag(mi, i) for mi in range_constexpr(mbw)]
                    compute_step(st, afr)

            if const_expr(kchain):
                # one chain a K split: K-wave s continues the accumulator K-wave s - 1 left in LDS
                def relay_off(mi, j):
                    row = (mblk0 + mi) * 16 + lane16
                    col = (w_n * nbw + j) * 16 + kgrp * 8
                    return RELAY + row * RELAY_PITCH + col * 4

                for stage in range_constexpr(kw):
                    if w_k == stage:
                        if const_expr(stage > 0):
                            for mi in range_constexpr(mbw):
                                for j in range_constexpr(nbw):
                                    off = relay_off(mi, j)
                                    lo = Vec(lds_load_b128(lds_idx, off)).bitcast(fx.Float32)
                                    hi = Vec(lds_load_b128(lds_idx, off + 16)).bitcast(fx.Float32)
                                    acc[mi][j].store(lo.shuffle(hi, list(range(8))))
                        run_steps(list(inflight))
                        if const_expr(stage < kw - 1):
                            for mi in range_constexpr(mbw):
                                for j in range_constexpr(nbw):
                                    off = relay_off(mi, j)
                                    v = Vec(acc[mi][j].load())
                                    lds_store_b128(lds_idx, off, v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32))
                                    lds_store_b128(lds_idx, off + 16, v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32))
                    _lds_barrier()
            else:
                run_steps(inflight)
            stamp(1, 0)

            is_q = head < NQ
            is_k = head == NQ
            is_v = head == NQ + NKV
            is_iq = (head >= IQ0) & (head < IK)
            is_ik = head == IK
            hl = lane
            dim0 = hl * 4
            first = hl < (ROT // 8)
            in_rope = hl < (ROT // 4)
            ib = first.select(dim0, dim0 - ROT // 2)
            ib = in_rope.select(ib, fx.Int32(0))
            w_addr = is_q.select(
                fx.Int64(fx.ptrtoint(arg_qw)),
                is_k.select(
                    fx.Int64(fx.ptrtoint(arg_kw)),
                    is_ik.select(fx.Int64(fx.ptrtoint(arg_ikw)), fx.Int64(fx.ptrtoint(arg_iqw))),
                ),
            )
            nw_buf = ptr_buf_tensor(w_addr, fx.Int32, unit_elems=2, num_records_bytes=D * 2)
            nw = Vec(buf_copy_load(nw_buf, hl, fx.Int32, 2))
            slab_buf = ptr_buf_tensor(arg_slab, fx.Float32, unit_elems=4,
                                      num_records_bytes=fx.Int64(i32_mpad) * (ks * N * 4))
            _lds_barrier()
            for mi in range_constexpr(mbw):
                for j in range_constexpr(nbw):
                    row = (mblk0 + mi) * 16 + lane16
                    col = (w_n * nbw + j) * 16 + kgrp * 8
                    v = Vec(acc[mi][j].load())
                    off = w_k * P_PLANE + row * P_ROW + col * 4
                    lds_store_b128(lds_idx, off, v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32))
                    lds_store_b128(lds_idx, off + 16, v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32))
            _lds_barrier()
            for t in range_constexpr((p_units + block - 1) // block):
                u = tid + t * block
                r = u // U_ROW
                c4 = u % U_ROW
                tot = None
                for k in range_constexpr(kw):
                    if const_expr(kchain and k < kw - 1):
                        continue  # kchain: the last K-wave holds the split's whole chain
                    part = Vec(lds_load_b128(lds_idx, k * P_PLANE + r * P_ROW + c4 * 16)).bitcast(fx.Float32)
                    tot = part if tot is None else tot + part
                g = ((kpart * i32_mpad + m0 + r) * N + head * D + npart * cols) // 4 + c4
                if const_expr(p_units % block == 0):
                    buf_copy_store(slab_buf, g, tot, fx.Float32, 4, cache_modifier=cpol)
                else:
                    if u < p_units:
                        buf_copy_store(slab_buf, g, tot, fx.Float32, 4, cache_modifier=cpol)
            cs_buf = ptr_buf_tensor(arg_cs, fx.Int32, unit_elems=2, num_records_bytes=i32_cs_rows * (ROT * 2))
            e_cos = []
            e_sin = []
            for t in range_constexpr(row_iters):
                off = e_pos[t] * (ROT // 4) + ib // 4
                e_cos.append(Vec(buf_copy_load(cs_buf, off, fx.Int32, 2)))
                e_sin.append(Vec(buf_copy_load(cs_buf, off + ROT // 8, fx.Int32, 2)))

            rocdl.s_wait_storecnt(0)
            cluster.cluster_barrier()
            stamp(2, 0)

            eps = i32_eps.bitcast(fx.Float32)
            parts = []
            for t in range_constexpr(row_iters):
                row = m0 + e_rows[t]
                pk = []
                for k in range_constexpr(ks):
                    gi = ((k * i32_mpad + row) * N + head * D) // 4 + hl
                    pk.append(buf_copy_load(slab_buf, gi, fx.Float32, 4, cache_modifier=cpol))
                parts.append(pk)
            q_buf = ptr_buf_tensor(arg_q, fx.Int32, unit_elems=2, num_records_bytes=i32_m * (NQ * D * 2))
            iq_buf = ptr_buf_tensor(arg_iq, fx.Int32, unit_elems=2, num_records_bytes=i32_m * (niq * D * 2))
            w = _bf16x4(nw)
            kp_base = fx.Int64(fx.ptrtoint(arg_kp))
            vp_base = fx.Int64(fx.ptrtoint(arg_vp))
            ksc_base = fx.Int64(fx.ptrtoint(arg_ksc))
            vsc_base = fx.Int64(fx.ptrtoint(arg_vsc))
            ip_base = fx.Int64(fx.ptrtoint(arg_ip))
            isc_base = fx.Int64(fx.ptrtoint(arg_isc))
            for t in range_constexpr(row_iters):
                row = m0 + e_rows[t]
                valid = e_valid[t]
                acc4 = None
                for k in range_constexpr(ks):
                    pk = Vec(parts[t][k])
                    acc4 = pk if acc4 is None else acc4 + pk
                raw = acc4.to(fx.BFloat16)
                raw_w = raw.bitcast(fx.Int32)
                e = _bf16x4(raw_w)
                ss = e[0] * e[0]
                ss = fx.math.fma(e[1], e[1], ss)
                ss = fx.math.fma(e[2], e[2], ss)
                ss = fx.math.fma(e[3], e[3], ss)
                for msk in (16, 8, 4, 2, 1):
                    ss = ss + _xor(ss, msk)
                r = _rsq(fx.math.fma(ss, fx.Float32(0.0078125), eps))
                nrm = [(e[i] * r) * (fx.Float32(1.0) + w[i]) for i in range_constexpr(4)]
                cos = _bf16x4(e_cos[t])
                sin = _bf16x4(e_sin[t])
                y = []
                for i in range_constexpr(4):
                    pr = _xor(nrm[i], ROT // 8)
                    ps = pr * sin[i]
                    roped = fx.math.fma(nrm[i], cos[i], first.select(-ps, ps))
                    y.append(in_rope.select(roped, nrm[i]))
                yb = Vec.from_elements(y, fx.Float32).to(fx.BFloat16)
                yw = yb.bitcast(fx.Int32)
                if valid:
                    if is_q:
                        buf_copy_store(q_buf, (row * (NQ * D) + head * D) // 4 + hl, yw, fx.Int32, 2)
                    if is_iq:
                        buf_copy_store(iq_buf, (row * (niq * D) + (head - IQ0) * D) // 4 + hl, yw, fx.Int32, 2)
                        # AQ: the index q as k5 sentinel words in copy E % 3; copy (E + 1) % 3 re-armed
                        w0 = fx.Int32(Vec(yw)[0])
                        w1 = fx.Int32(Vec(yw)[1])
                        widx = (row * (niq * D) + (head - IQ0) * D) // 2 + hl * 2
                        buffer_store(Vec.from_elements([w0 ^ sentinel_bits(w0), w1 ^ sentinel_bits(w1)], fx.Int32),
                                     qcur_rs, widx, cache_modifier=SCOPE_DEV)
                        buffer_store(Vec.from_elements([fx.Int32(-1), fx.Int32(-1)], fx.Int32), qnext_rs, widx,
                                     cache_modifier=SCOPE_DEV)
                src = is_v.select(raw_w, yw)
                vq = _bf16x4(Vec(src))
                code, sb_ = _nvfp4_lane(vq, is_ik)
                slot = is_ik.select(e_islot[t], e_slot[t])
                page = fx.Int64(slot >> 7)
                pos = fx.Int64(slot & 127)
                kv = is_k | is_v
                pay = kv.select(
                    is_k.select(kp_base, vp_base) + page * i64_kp_page + pos * 64,
                    ip_base + page * i64_ip_page + pos * 64,
                )
                sca = kv.select(
                    is_k.select(ksc_base, vsc_base) + page * i64_ks_page + pos * 8,
                    isc_base + page * i64_is_page + pos * 8,
                )
                if valid & (slot >= 0) & (kv | is_ik):
                    fx.ptr_store(code.to(fx.Int16), _gptr(pay + fx.Int64(hl * 2), T.i16, 2))
                    if hl % 4 == 0:
                        fx.ptr_store(sb_.to(fx.Int8), _gptr(sca + fx.Int64(hl // 4), T.i8, 1))
                if valid & is_ik:
                    # AQ: this row's index-K is out: flag word ``row`` = E after a device-scope release
                    rocdl.s_wait_storecnt(0)
                    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope=fx.rocdl.SyncScope.Agent)
                    if hl == 0:
                        buffer_store(e_, flag_rs, row, cache_modifier=SCOPE_DEV)
        else:
            # ======================= the CP score (ATOM's scorer) =======================
            program = bid - N_QKV
            g = lane // 16
            u = lane % 16
            zero = fx.Int32(0)
            tokens = ql  # one request
            bt = fx.recast_iter(fx.Int32, arg_bt)
            last_col = i32_global - 1
            length = fx.Int32(fx.recast_iter(fx.Int32, arg_len)[0])
            request_blocks = (length + 127) // 128
            request_blocks = (request_blocks < i32_global).select(request_blocks, i32_global)
            live_end = (request_blocks - rank_cp + (world - 1)) // world
            live_end = (live_end < zero).select(zero, live_end)
            live_end = (live_end < i32_local).select(live_end, i32_local)
            chunk = (live_end + n_score - 1) // n_score
            start = program * chunk
            nblk = live_end - start
            nblk = (nblk < chunk).select(nblk, chunk)

            def local_of(j):
                return start + j

            min_cutoff = length - ql + 1
            first_masked = (((min_cutoff - PAGE) >> 7) + 1 - rank_cp + (world - 1)) // world
            mid = (first_masked < zero).select(zero, first_masked)
            # local blocks from `fresh` on hold keys this launch inserts (position length - ql on)
            first_new = (length - ql) >> 7
            fresh = (first_new - rank_cp + (world - 1)) // world
            fresh = (fresh < zero).select(zero, fresh)
            has_fresh = (start + nblk) > fresh

            def load_windows():
                ws = []
                for h in range_constexpr(MAX_BLOCKS // 32):
                    gblk = local_of(lane + 32 * h) * world + rank_cp
                    gblk = (gblk < last_col).select(gblk, last_col)
                    ws.append(fx.Int32(bt[gblk]))
                return ws

            if nblk > zero:
                windows = load_windows()
                ring_u8 = fx.recast_iter(fx.Uint8, arena.base_ptr)
                part = fx.recast_iter(fx.Float32, fx.add_offset(ring_u8, RING_BYTES))
                q_u8 = fx.add_offset(ring_u8, Q_LDS)

                def q_off(p, lane_, k):
                    return p * (32 * QOP_LANE) + lane_ * QOP_LANE + ((k ^ ((lane_ >> 2) & 3)) * 16)

                # compute group cg (waves 0-3: 0, waves 8-11: 1) scores blocks j = cg + 2 i; w4: the block quarter
                if const_expr(sc_groups == 3):
                    # waves 4-7 (the idx-q converters) are compute group 1 once the q is in LDS
                    cg = wave // SC_WAVES
                else:
                    cg = (wave >= 2 * SC_WAVES).select(fx.Int32(1), fx.Int32(0))
                w4 = wave % SC_WAVES
                rw = cg * SC_WAVES + w4
                n_g = (nblk - cg + (sc_groups - 1)) // sc_groups
                if const_expr(sc_groups == 3):
                    computes = wave >= zero
                else:
                    computes = (wave < SC_WAVES) | (wave >= 2 * SC_WAVES)
                ring_wave = lds_idx + fx.Int64(rw * (RING * SLOT))
                kd_base = fx.Int64(fx.ptrtoint(arg_ip))
                ks_base = fx.Int64(fx.ptrtoint(arg_isc))
                cp_src = [fx.Int64(w4 * 2048 + c * 512 + lane * 16) for c in range_constexpr(4)]
                cp_dst = [fx.Int64(_swz((c * 32 + lane) // 4, lane % 4)) for c in range_constexpr(4)]
                sc_src = fx.Int64(w4 * 256 + lane * 8)
                sc_dst = fx.Int64(SLOT_PAYLOAD + lane * 8)

                def page_of(j):
                    lo = fx.Int32(rocdl.readlane(T.i32, fx.as_ir_value(windows[0]), fx.as_ir_value(j % 32)))
                    hi = fx.Int32(rocdl.readlane(T.i32, fx.as_ir_value(windows[1]), fx.as_ir_value(j % 32)))
                    return (j < 32).select(lo, hi)

                def blk(i):
                    return cg + sc_groups * i

                def issue(i):
                    """Async copies of the group's i-th block (this wave's quarter) into its ring slot i % RING."""
                    page = fx.Int64(page_of(blk(i)))
                    slot = ring_wave + fx.Int64((i % RING) * SLOT)
                    pd = kd_base + page * i64_ip_page
                    for c in range_constexpr(4):
                        rocdl.global_load_async_to_lds_b128(
                            create_llvm_ptr(pd + cp_src[c], address_space=1),
                            create_llvm_ptr(slot + cp_dst[c], address_space=3),
                            0,
                        )
                    rocdl.global_load_async_to_lds_b64(
                        create_llvm_ptr(ks_base + page * i64_is_page + sc_src, address_space=1),
                        create_llvm_ptr(slot + sc_dst, address_space=3),
                        0,
                    )

                if computes:
                    # keys of blocks this launch does not write stream in now (they land while the chain runs)
                    for d in range_constexpr(RING - 1):
                        if (fx.Int32(d) < n_g) & (local_of(blk(fx.Int32(d))) < fresh):
                            issue(fx.Int32(d))
                if (wave >= SC_WAVES) & (wave < 2 * SC_WAVES):
                    # ---- the index q: poll the words this thread converts, then the B operands (hi / lo E4M3)
                    t_ = tid - SC_THREADS
                    t_lane = t_ % 32
                    t_qt = (t_ // 32) % nqt
                    t_jh = t_ // (32 * nqt)
                    tu, tg = t_lane % 16, t_lane // 16
                    tcol = tu + t_qt * 16
                    tlive = tcol < rows_s
                    tcol = tlive.select(tcol, zero)
                    # dword index of bf16 [row, head, 16 tg + 32 jq] in the [ql, niq, 128] copy
                    tbase = ((tcol // heads_i) * (niq * D) + (tcol % heads_i) * D + tg * 16) // 2

                    def q_load():
                        # flat: chunk jj, half hh at 2 jj + hh (16 bf16 = two 4-dword loads a chunk)
                        return tuple(
                            Vec(buffer_load(qcur_rs, tbase + (((t_jh * JH + jj) % 4) * 32) // 2 + 4 * hh,
                                            vec_width=4, dtype=T.i32, cache_modifier=SCOPE_DEV))
                            for jj in range_constexpr(JH)
                            for hh in range_constexpr(2)
                        )

                    def q_pending(qs):
                        m_ = fx.Int32(0)
                        for v_ in qs:
                            for c in range_constexpr(4):
                                m_ = m_ | sentinel_bits(fx.Int32(v_[c]))
                        m_ = tlive.select(m_, fx.Int32(0))
                        return fx.Int32(rocdl.ballot(T.i32, m_ != 0)) != 0

                    qs = q_load()
                    qpend = q_pending(qs)
                    qit = fx.Int32(0)
                    while qpend & (qit < i32_maxit):
                        rocdl.s_sleep(POLL_SLEEP)
                        qs = q_load()
                        qpend = q_pending(qs)
                        qit = qit + 1
                    timeout(qpend)
                    stamp(1, SC_THREADS)
                    for jj in range_constexpr(JH):
                        jq = t_jh * JH + jj
                        words = [fx.Int32(qs[2 * jj + hh][c]) for hh in range_constexpr(2) for c in range_constexpr(4)]
                        vals = Vec.from_elements(words, fx.Int32).bitcast(fx.BFloat16).to(fx.Float32)
                        hw, lw = [], []
                        for h in range_constexpr(2):
                            v8 = fx.Vector.from_elements(
                                [
                                    fx.minnumf(
                                        fx.maxnumf(tlive.select(fx.Float32(vals[h * 8 + e]), fx.Float32(0.0)),
                                                   fx.Float32(-E4M3_MAX)),
                                        fx.Float32(E4M3_MAX),
                                    )
                                    for e in range_constexpr(8)
                                ],
                                dtype=fx.Float32,
                            )
                            hi = _to_fp8x8(v8)
                            lo = _to_fp8x8((v8 - _from_fp8x8(hi)) * fx.Vector.filled(8, 16.0, fx.Float32))
                            hw += [hi[0], hi[1]]
                            lw += [lo[0], lo[1]]
                        for hl_, words_ in ((0, hw), (1, lw)):
                            fx.ptr_store(fx.Vector.from_elements(words_, dtype=fx.Int32),
                                         _lds_i32(q_u8, q_off(t_qt * 2 + hl_, t_lane, jq), 4))
                    # ---- this launch's index-K rows (only a workgroup holding their blocks waits)
                    if has_fresh & (wave == SC_WAVES):
                        fl = fx.Int32(buffer_load(flag_rs, (lane < ql).select(lane, fx.Int32(0)), vec_width=1,
                                                  dtype=T.i32, cache_modifier=SCOPE_DEV))
                        fpend = fx.Int32(rocdl.ballot(T.i32, (lane < ql) & (fl != e_))) != 0
                        fit = fx.Int32(0)
                        while fpend & (fit < i32_maxit):
                            rocdl.s_sleep(POLL_SLEEP)
                            fl = fx.Int32(buffer_load(flag_rs, (lane < ql).select(lane, fx.Int32(0)), vec_width=1,
                                                      dtype=T.i32, cache_modifier=SCOPE_DEV))
                            fpend = fx.Int32(rocdl.ballot(T.i32, (lane < ql) & (fl != e_))) != 0
                            fit = fit + 1
                        timeout(fpend)
                if tid < SC_THREADS:
                    nan = fx.Vector.from_elements([fx.Int32(0x7FC00000)], dtype=fx.Int32).bitcast(fx.Float32)[0]
                    for k in range_constexpr(PART // SC_THREADS):
                        fx.ptr_store(nan, part + (tid + k * SC_THREADS))
                _lds_barrier()
                if computes:
                    if has_fresh:
                        # the index-K rows were released at device scope; drop any stale line before reading
                        fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope=fx.rocdl.SyncScope.Agent)
                    q_hi, q_lo, cutoffs = [], [], []
                    for qt in range_constexpr(nqt):
                        col = u + qt * 16
                        col = (col < rows_s).select(col, zero)
                        cutoffs.append(length - ql + col // heads_i + 1)
                        ops = []
                        for hl_ in range_constexpr(2):
                            words = []
                            for k in range_constexpr(4):
                                v = fx.Vector(fx.ptr_load(_lds_i32(q_u8, q_off(qt * 2 + hl_, lane, k), 4),
                                                          result_type=fx.Vector.make_type(4, fx.Int32)))
                                words += [v[e] for e in range_constexpr(4)]
                            ops.append(fx.Vector.from_elements(words, dtype=fx.Int32))
                        q_hi.append(ops[0])
                        q_lo.append(ops[1])
                    # blocks holding this launch's keys stream in now, in block order after the early ones
                    for d in range_constexpr(RING - 1):
                        if (fx.Int32(d) < n_g) & (local_of(blk(fx.Int32(d))) >= fresh):
                            issue(fx.Int32(d))
                    lane_zero = fx.Int64((lane + length) >> 31)
                    s_hi = fx.Int64(E8M0_ONE) | lane_zero
                    s_lo = fx.Int64(E8M0_LO) | lane_zero
                    zero8 = fx.Vector.filled(8, 0.0, fx.Float32)
                    neg_inf = fx.Float32(float("-inf"))
                    a_off = [[_swz(tt * 16 + u, g + 2 * h) for h in range_constexpr(2)] for tt in range_constexpr(2)]

                    def partial_max(i, masked):
                        j = blk(i)
                        slot = rw * (RING * SLOT) + (i % RING) * SLOT
                        sa = fx.Vector(fx.ptr_load(_lds_i32(ring_u8, slot + SLOT_PAYLOAD + lane * 8, 2),
                                                   result_type=fx.Vector.make_type(2, fx.Int32))).bitcast(fx.Int64)[0]
                        key0 = (local_of(j) * world + rank_cp) * PAGE + w4 * 32
                        ms = [None] * nqt
                        for tt in range_constexpr(2):
                            lo4 = fx.Vector(fx.ptr_load(_lds_i32(ring_u8, slot + a_off[tt][0], 4),
                                                        result_type=fx.Vector.make_type(4, fx.Int32)))
                            hi4 = fx.Vector(fx.ptr_load(_lds_i32(ring_u8, slot + a_off[tt][1], 4),
                                                        result_type=fx.Vector.make_type(4, fx.Int32)))
                            a = fx.Vector.from_elements(
                                [lo4[e] for e in range_constexpr(4)] + [hi4[e] for e in range_constexpr(4)],
                                dtype=fx.Int32,
                            )
                            for qt in range_constexpr(nqt):
                                c = _wmma(a, q_hi[qt], zero8, sa, s_hi, tt, qt % 2)
                                c = _wmma(a, q_lo[qt], c, sa, s_lo, tt, qt % 2)
                                vals = [fx.Float32(c[v]) for v in range_constexpr(8)]
                                if const_expr(masked):
                                    pos0 = key0 + tt * 16 + g * 8
                                    vals = [(pos0 + v < cutoffs[qt]).select(vals[v], neg_inf)
                                            for v in range_constexpr(8)]
                                m = vals[0]
                                for v in range_constexpr(1, 8):
                                    m = fx.maxnumf(m, vals[v])
                                ms[qt] = m if tt == 0 else fx.maxnumf(ms[qt], m)
                        for qt in range_constexpr(nqt):
                            ms[qt] = fx.maxnumf(ms[qt], ms[qt].shuffle_xor(16, WAVE))
                        return ms

                    wr_row = u + g * 16
                    part_base = fx.Int64(fx.ptrtoint(part))

                    def put(ms, j):
                        val = ms[0]
                        if const_expr(nqt == 2):
                            val = (g == 0).select(ms[0], ms[1])
                        _llvm.atomicrmw(
                            _llvm.AtomicBinOp.fmax,
                            create_llvm_ptr(part_base + fx.Int64((j * 32 + wr_row) * 4), address_space=3),
                            fx.as_ir_value(val),
                            _llvm.AtomicOrdering.monotonic,
                            syncscope="workgroup",
                        )

                    for i in range(zero, n_g):
                        i = fx.Int32(i)
                        ia = i + (RING - 1)
                        if ia < n_g:
                            rocdl.s_wait_dscnt(0)
                            issue(ia)
                            rocdl.s_wait_asynccnt(COPIES * (RING - 1))
                        else:
                            rocdl.s_wait_asynccnt(0)
                        if local_of(blk(i)) < mid:
                            put(partial_max(i, False), blk(i))
                        else:
                            put(partial_max(i, True), blk(i))
                _lds_barrier()
                stamp(2, 0)
                out_ptr = fx.recast_iter(fx.Float32, arg_out)
                nb_ = (nblk > zero).select(nblk, fx.Int32(1))
                for it in range_constexpr((32 * MAX_BLOCKS + block - 1) // block):
                    idx = tid + it * block
                    jb = idx % nb_
                    row = idx // nb_
                    if (idx < rows_s * nblk) & (row < rows_s):
                        v = fx.Float32(fx.ptr_load(part + (jb * 32 + row))) * f32_scale
                        if const_expr(xchg):
                            # data-as-flag for the exchange workgroups (armed 0xffffffff, which no score is)
                            vb_ = v.bitcast(fx.Int32)
                            vb_ = (vb_ == fx.Int32(-1)).select(fx.Int32(0x7FC00001), vb_)
                            buffer_store(vb_, scb_rs, ((row % heads_i) * ql + row // heads_i) * local_max + local_of(jb),
                                         cache_modifier=SCOPE_DEV)
                        else:
                            dst = fx.add_offset(
                                out_ptr, ((row % heads_i) * tokens + row // heads_i) * i32_local + local_of(jb)
                            )
                            fx.ptr_store(v, dst)
        if const_expr(xchg):
            if bid >= N_QKV + n_score:
                # ======================= the CP candidate exchange (ATOM's _select_exchange_merge) ===========
                # workgroup hh * ql + r: this shard's top-16 of index head hh for query row r. Its waves split the
                # 32-key tiles (wave w: tiles w, w + waves, w + 2 waves), each keeps a bitonic top-16; wave 0 merges
                # the waves' lists through LDS and pushes the result to head hh's owner (rank hh). On the owner
                # (hh == rank) wave 0 then receives the 4 shards' keys for row r and merges them (bitonic too).
                # The packed (score, block) keys are totally ordered (invalid keys are all 0), so the top-16 and its
                # order do not depend on the reduction tree: production's bytes.
                sel = bid - N_QKV - n_score
                hh = sel // ql
                sel_r = sel % ql
                if sel < niq * ql:
                    if const_expr(ipf_kib):
                        # the role's straight-line code runs once a launch on this CU: pull it into the instruction
                        # cache while the scores are still being computed. Bounded by the kernel's own end PC, which
                        # an earlier launch recorded (0 before the first: no prefetch), so no request leaves the code.
                        if wave == 0:
                            prefetch_code_to(err_pc_end(), ipf_kib // 2)
                    zero_ = fx.Int32(0)
                    lane_lt = (fx.Int32(1) << lane) - 1  # lanes below this one

                    def pack(score, index, valid):
                        """_pack_score_key: (hi, lo) of (1 << 48) | (ordered << 16) | index, 0 where not valid."""
                        bits = score.bitcast(fx.Int32)
                        bits = (bits == fx.Int32(-0x80000000)).select(zero_, bits)
                        ordered = bits ^ (bits < zero_).select(fx.Int32(-1), fx.Int32(-0x80000000))
                        ordered = ((bits & fx.Int32(0x7FFFFFFF)) > fx.Int32(0x7F800000)).select(zero_, ordered)
                        hi = fx.Int32(0x10000) | ((ordered >> 16) & fx.Int32(0xFFFF))
                        lo = (ordered << 16) | index
                        return valid.select(hi, zero_), valid.select(lo, zero_)

                    def force(score, blockid, valid, local_start):
                        score = (valid & (blockid < init_blocks)).select(fx.Float32(1e30), score)
                        return (valid & (blockid >= local_start)).select(fx.Float32(1e29), score)

                    # ---- bitonic top-16 on (hi, lo) key pairs, one key a lane (hi < 2^17, so a signed 64-bit compare
                    # is the unsigned key order). Names end in _b: the kernel's AST rewrite sees nested functions' names.
                    lbit_b = {jb: (lane & fx.Int32(jb)) != zero_ for jb in (1, 2, 4, 8, 16)}
                    nbit_b = {jb: (lane & fx.Int32(jb)) == zero_ for jb in (1, 2, 4, 8, 16)}

                    def k64(kh_b, kl_b):
                        return (fx.Int64(kh_b) << fx.Int64(32)) | (fx.Int64(kl_b) & fx.Int64(0xFFFFFFFF))

                    def cx(key_b, jx_b, want_max):
                        """Compare-exchange with lane ^ jx_b: the larger key stays where want_max, else the smaller.
                        A key is (hi, lo, payload...): the payload words move with it."""
                        pk_b = [fx.Int32(gpu.shuffle_xor(e_b, jx_b, WAVE)) for e_b in key_b]
                        take_b = want_max == (k64(pk_b[0], pk_b[1]) > k64(key_b[0], key_b[1]))
                        return tuple(take_b.select(p_b, e_b) for p_b, e_b in zip(pk_b, key_b))

                    def kmax(ka_b, kb_b):
                        gt_b = k64(ka_b[0], ka_b[1]) > k64(kb_b[0], kb_b[1])
                        return tuple(gt_b.select(a_b, c_b) for a_b, c_b in zip(ka_b, kb_b))

                    def sort16(keys_b, dirs_b):
                        """Bitonic sequences in lanes 0-15 -> sorted (descending where dirs_b[i]); stage by stage."""
                        for jx_b in (8, 4, 2, 1):
                            keys_b = [cx(kq_b, jx_b, nbit_b[jx_b] if dq_b else lbit_b[jx_b]) for kq_b, dq_b in zip(keys_b, dirs_b)]
                        return keys_b

                    def top16(keys_b, dirs_b):
                        """The 16 largest of each 32-key tile, in lanes 0-15, sorted (descending where dirs_b[i]):
                        16-key halves sorted opposite ways (10 stages), the larger half (1), then sorted (4)."""
                        for kk_b, js_b in ((2, (1,)), (4, (2, 1)), (8, (4, 2, 1)), (16, (8, 4, 2, 1))):
                            for jx_b in js_b:
                                keys_b = [cx(tq_b, jx_b, lbit_b[jx_b] == lbit_b[kk_b]) for tq_b in keys_b]
                        keys_b = [cx(tq_b, 16, nbit_b[16]) for tq_b in keys_b]
                        return sort16(keys_b, dirs_b)

                    def merge16(ma_b, mb_b, desc_b):
                        """Top-16 of lists ma_b (descending) and mb_b (ascending), both in lanes 0-15."""
                        return sort16([kmax(ma_b, mb_b)], [desc_b])[0]

                    length_ = fx.Int32(fx.recast_iter(fx.Int32, arg_len)[0])
                    send_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_send)), world * max_rows * 4)
                    peers_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_peers)), world * 8)
                    recv_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_recv)), max_rows * 4)
                    if const_expr(owner_fast):
                        # read at the role's entry, while the scores are computed: only this workgroup advances its
                        # send counter (head hh, row r) and, on the owner, its receive counter (row r) in a launch,
                        # and the peer table is fixed; so the push and the poll need no dependent load later
                        call0 = _uni(fx.Int32(buffer_load(send_rs, hh * max_rows + sel_r, vec_width=1, dtype=T.i32,
                                                          cache_modifier=SCOPE_DEV)))
                        rcall0 = _uni(fx.Int32(buffer_load(recv_rs, sel_r, vec_width=1, dtype=T.i32,
                                                           cache_modifier=SCOPE_DEV)))
                        pw0 = Vec(buffer_load(peers_rs, 2 * hh, vec_width=2, dtype=T.i32))
                    token = sel_r
                    causal_blocks = (length_ - ql + token + 128) // 128
                    local_start = causal_blocks - local_keep
                    local_start = (local_start < zero_).select(zero_, local_start)
                    lim = (causal_blocks < i32_global).select(causal_blocks, i32_global)
                    live_row = (lim - rank_cp + (world - 1)) // world
                    live_row = (live_row < zero_).select(zero_, live_row)
                    live_row = (live_row < i32_local).select(live_row, i32_local)
                    # every slot the score workgroups wrote for this (head, row): this rank's live blocks
                    req_b = (length_ + 127) // 128
                    req_b = (req_b < i32_global).select(req_b, i32_global)
                    live_rank = (req_b - rank_cp + (world - 1)) // world
                    live_rank = (live_rank < zero_).select(zero_, live_rank)
                    live_rank = (live_rank < i32_local).select(live_rank, i32_local)
                    base_w = (hh * ql + sel_r) * local_max
                    ntiles = (live_rank + 31) // 32
                    # ---- the shard: wave w polls / re-arms / packs tiles w, w + W, ... (W = 12 waves), keeps
                    # their top-16 (descending, lanes 0-15) and leaves it in LDS as list w
                    lst_u8 = fx.recast_iter(fx.Uint8, arena.base_ptr)
                    if const_expr(owner_v2):
                        if hh == rank_cp:
                            # the request's block-table row (every page the owner can pick) into LDS while the scores
                            # are computed; read past the merge instead of a global load (the barrier after the
                            # shard lists orders these stores before wave 0's reads)
                            nbt_ = (length_ + 127) // 128
                            nbt_ = (nbt_ < i32_global).select(nbt_, i32_global)
                            nbt_ = (nbt_ < fx.Int32(local_max * world)).select(nbt_, fx.Int32(local_max * world))
                            bts_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_bt)), nbt_ * 4)
                            # one entry a load: a 16-B load that crosses the row's end reads zeros as a whole
                            for e1_ in range(tid, nbt_, fx.Int32(block)):
                                v1_ = fx.Int32(buffer_load(bts_rs, e1_, vec_width=1, dtype=T.i32))
                                fx.ptr_store(Vec.from_elements([v1_], fx.Int32), _lds_i32(lst_u8, PT_OFF + e1_ * 4, 1))

                    def shard_list(nslots):
                        offs = [(wave + waves * ts_b) * 32 + lane for ts_b in range_constexpr(nslots)]
                        inrs = [o < live_rank for o in offs]
                        wis = [inr.select(base_w + o, base_w) for inr, o in zip(inrs, offs)]

                        def sload():
                            return tuple(fx.Int32(buffer_load(scb_rs, wi, vec_width=1, dtype=T.i32,
                                                              cache_modifier=SCOPE_DEV)) for wi in wis)

                        def spend(vs_):
                            m_ = fx.Int32(0)
                            for inr, v_ in zip(inrs, vs_):
                                m_ = m_ | fx.Int32(rocdl.ballot(T.i32, inr & (fx.Int32(v_) == fx.Int32(-1))))
                            return m_ != zero_

                        vs = sload()
                        pend = spend(vs)
                        pit = zero_
                        while pend & (pit < i32_maxit):
                            rocdl.s_sleep(POLL_SLEEP)
                            vs = sload()
                            pend = spend(vs)
                            pit = pit + 1
                        timeout(pend)
                        stamp(4, 0)
                        tiles = []
                        for ts_b in range_constexpr(nslots):
                            if inrs[ts_b]:
                                # re-armed (one reader)
                                buffer_store(fx.Int32(-1), scb_rs, wis[ts_b], cache_modifier=SCOPE_DEV)
                            lvalid = offs[ts_b] < live_row
                            blockid = offs[ts_b] * world + rank_cp
                            valid = lvalid & (blockid < i32_global) & (blockid < causal_blocks)
                            score = fx.Float32(lvalid.select(fx.Float32(fx.Int32(vs[ts_b]).bitcast(fx.Float32)),
                                                             fx.Float32(-1e30)))
                            score = fx.Float32(force(score, blockid, valid, local_start))
                            tiles.append(pack(score, blockid + 1, valid))
                        if const_expr(nslots == 1):
                            rl_b = top16(tiles, [True])[0]
                        if const_expr(nslots == 3):
                            t0, t1, t2 = top16(tiles, [True, False, False])
                            rl_b = merge16(merge16(t0, t1, True), t2, True)
                        if const_expr(nslots == 6):
                            ts_ = top16(tiles, [True, False, True, False, True, False])
                            us_ = sort16([kmax(ts_[0], ts_[1]), kmax(ts_[2], ts_[3]), kmax(ts_[4], ts_[5])],
                                         [True, False, False])
                            rl_b = merge16(merge16(us_[0], us_[1], True), us_[2], True)
                        if lane < 16:
                            fx.ptr_store(Vec.from_elements([rl_b[1], rl_b[0]], fx.Int32),
                                         _lds_i32(lst_u8, (wave * 16 + lane) * 8, 2))

                    if ntiles > 3 * waves:
                        shard_list(6)
                    else:
                        if ntiles > waves:
                            shard_list(3)
                        else:
                            shard_list(1)
                    _lds_barrier()
                    l16 = lane & 15

                    def lget(lw_b, li_b):
                        v_ = Vec(fx.ptr_load(_lds_i32(lst_u8, (lw_b * 16 + li_b) * 8, 2),
                                             result_type=fx.Vector.make_type(2, fx.Int32)))
                        return fx.Int32(v_[1]), fx.Int32(v_[0])

                    if const_expr(merge_par):
                        stamp(5, 0)
                        # the first merge level on waves 0 .. npair-1 at once (wave p: lists 2p, 2p + 1 -> list
                        # waves + p, descending for even p, ascending for odd p), then wave 0 merges those as before.
                        # The top-16 under the keys' total order does not depend on the merge tree.
                        npair = (ntiles > 8).select(fx.Int32(6), fx.Int32(4))
                        if (ntiles > 4) & (wave < npair):
                            ap0_b = kmax(lget(2 * wave, l16), lget(2 * wave + 1, 15 - l16))
                            a_hi = ap0_b[0]
                            a_lo = ap0_b[1]
                            if (wave % 2) == zero_:
                                a_hi, a_lo = sort16([(a_hi, a_lo)], [True])[0]
                            else:
                                a_hi, a_lo = sort16([(a_hi, a_lo)], [False])[0]
                            if lane < 16:
                                fx.ptr_store(Vec.from_elements([a_lo, a_hi], fx.Int32),
                                             _lds_i32(lst_u8, ((waves + wave) * 16 + lane) * 8, 2))
                        _lds_barrier()
                    if wave == 0:
                        # ---- the 12 lists: pairs (list 2p descending, list 2p + 1 reversed) -> 6 -> 3 -> 2 -> 1
                        if const_expr(not merge_par):
                            stamp(5, 0)

                        def pairs(np_b, dirs_b):
                            if const_expr(merge_par and np_b > 2):
                                # the pairs' merges are done (lists waves + p, in dirs_b's order)
                                return [lget(waves + pp_b, l16) for pp_b in range_constexpr(np_b)]
                            return sort16([kmax(lget(2 * pp_b, l16), lget(2 * pp_b + 1, 15 - l16))
                                           for pp_b in range_constexpr(np_b)], dirs_b)

                        # as many levels as lists hold keys (lists past the tiles are all 0; wave-uniform branch)
                        t_hi = zero_
                        t_lo = zero_
                        if ntiles <= 2:
                            t_hi, t_lo = pairs(1, [True])[0]
                        else:
                            if ntiles <= 4:
                                t_hi, t_lo = merge16(*pairs(2, [True, False]), True)
                            else:
                                if ntiles <= 8:
                                    cs = pairs(4, [True, False, True, False])
                                    t_hi, t_lo = merge16(*sort16([kmax(cs[0], cs[1]), kmax(cs[2], cs[3])],
                                                                 [True, False]), True)
                                else:
                                    cs = pairs(6, [True, False, True, False, True, False])
                                    ds = sort16([kmax(cs[0], cs[1]), kmax(cs[2], cs[3]), kmax(cs[4], cs[5])],
                                                [True, False, False])
                                    t_hi, t_lo = merge16(merge16(ds[0], ds[1], True), ds[2], True)
                        stamp(1, 0)
                        # ---- push the shard's top-16 to head hh's owner (rank hh): tagged, bank call % 2
                        if const_expr(owner_fast):
                            call = call0
                        else:
                            call = _uni(fx.Int32(buffer_load(send_rs, hh * max_rows + sel_r, vec_width=1, dtype=T.i32,
                                                             cache_modifier=SCOPE_DEV)))
                        call = (call & fx.Int32((1 << 30) - 1)) + 1
                        if lane == 0:
                            buffer_store(call, send_rs, hh * max_rows + sel_r, cache_modifier=SCOPE_DEV)
                        tag = call % 32767 + 1
                        if const_expr(owner_fast):
                            pw = pw0
                        else:
                            pw = Vec(buffer_load(peers_rs, 2 * hh, vec_width=2, dtype=T.i32))
                        peer = ((fx.Int64(_uni(fx.Int32(pw[1]))) << fx.Int64(32))
                                | fx.Int64(fx.Uint32(_uni(fx.Int32(pw[0])))))
                        box_bytes = 2 * world * max_rows * slot_keys * 8
                        peer_rs = _rsrc(peer, box_bytes)
                        slot = ((call % 2) * world + rank_cp) * max_rows + sel_r
                        if lane < topk:
                            buffer_store(Vec.from_elements([t_lo, t_hi | (tag << 17)], fx.Int32), peer_rs,
                                         (slot * slot_keys + lane) * 2, cache_modifier=SCOPE_SYS)
                        stamp(6, 0)
                        if hh == rank_cp:
                            # ---- the owner: the 4 shards' keys for this row, then the merge (_merge_winners)
                            if const_expr(owner_fast):
                                rcall = rcall0
                            else:
                                rcall = _uni(fx.Int32(buffer_load(recv_rs, sel_r, vec_width=1, dtype=T.i32,
                                                                  cache_modifier=SCOPE_DEV)))
                            rcall = (rcall & fx.Int32((1 << 30) - 1)) + 1
                            rtag = rcall % 32767 + 1
                            bank = (rcall % 2) * world * max_rows
                            own_rs = peer_rs  # Peers[rank] is this rank's own mailbox
                            if const_expr(owner_v2):
                                # ---- owner_v2: shard s's 16 keys (sent sorted, descending) read reversed into lanes
                                # 0-15 of register s (ascending), merged into the running top-16 (descending) as soon
                                # as all 16 words carry this round's tag; the top-16 of a total order does not depend
                                # on the merge order, so the bytes are the all-at-once merge's
                                causal_len = length_ - ql + token + 1
                                valid_blocks = (causal_len + 127) // 128
                                m_start = valid_blocks - local_keep
                                m_start = (m_start < zero_).select(zero_, m_start)
                                lo16 = lane < 16
                                vsrc = [((bank + s_ * max_rows + sel_r) * slot_keys + (15 - lane)) * 2
                                        for s_ in range_constexpr(world)]

                                def vload():
                                    return tuple(Vec(buffer_load(own_rs, lo16.select(vsrc[s_], fx.Int32(0)), vec_width=2,
                                                                 dtype=T.i32, cache_modifier=SCOPE_SYS))
                                                 for s_ in range_constexpr(world))

                                def vin(wv_, s_):
                                    tgv_ = (fx.Int32(wv_[s_][1]) >> 17) & fx.Int32(0x7FFF)
                                    return fx.Int32(rocdl.ballot(T.i32, lo16 & (tgv_ == rtag))) == fx.Int32(0xFFFF)

                                def vkey(wv_, s_):
                                    kh_v = lo16.select(fx.Int32(wv_[s_][1]) & fx.Int32(0x1FFFF), zero_)
                                    kl_v = lo16.select(fx.Int32(wv_[s_][0]), zero_)
                                    blk_v = (kl_v & fx.Int32(0xFFFF)) - 1
                                    real_v = (kh_v != zero_) | (kl_v != zero_)
                                    sc_v = ((kh_v & fx.Int32(0xFFFF)) << 16) | ((kl_v >> 16) & fx.Int32(0xFFFF))
                                    bits_v = sc_v ^ (sc_v < zero_).select(fx.Int32(-0x80000000), fx.Int32(-1))
                                    val_v = force(bits_v.bitcast(fx.Float32), blk_v, real_v, m_start)
                                    return pack(val_v, blk_v + 1, real_v)

                                def vsort_asc(ka_v):
                                    # a full 16-key bitonic sort into ascending lanes 0-15 (an unsorted sender only)
                                    for kk_v, js_v in ((2, (1,)), (4, (2, 1)), (8, (4, 2, 1)), (16, (8, 4, 2, 1))):
                                        for jx_v in js_v:
                                            ka_v = cx(ka_v, jx_v, lbit_b[jx_v] != lbit_b[kk_v])
                                    return ka_v

                                def vmerge(rh_v, rl_v, key_v):
                                    # sorted ascending? (checked beside the merge; never false for AQ's / tl.topk's lists)
                                    ck_v = k64(key_v[0], key_v[1])
                                    nk_v = k64(fx.Int32(gpu.shuffle_down(key_v[0], 1, WAVE)),
                                               fx.Int32(gpu.shuffle_down(key_v[1], 1, WAVE)))
                                    badv_ = fx.Int32(rocdl.ballot(T.i32, (lane < 15) & (ck_v > nk_v)))
                                    mh_v, ml_v = sort16([kmax((rh_v, rl_v), key_v)], [True])[0]
                                    if badv_ != zero_:
                                        mh_v, ml_v = sort16([kmax((rh_v, rl_v), vsort_asc(key_v))], [True])[0]
                                    return mh_v, ml_v

                                run_h = zero_
                                run_l = zero_
                                got_ = zero_
                                wv = vload()
                                rit = zero_
                                while (got_ != fx.Int32((1 << world) - 1)) & (rit < i32_maxit):
                                    for s_ in range_constexpr(world):
                                        if (((got_ >> s_) & fx.Int32(1)) == zero_) & vin(wv, s_):
                                            run_h, run_l = vmerge(run_h, run_l, vkey(wv, s_))
                                            got_ = got_ | fx.Int32(1 << s_)
                                    if got_ != fx.Int32((1 << world) - 1):
                                        rocdl.s_sleep(POLL_SLEEP)
                                        wv = vload()
                                    rit = rit + 1
                                late = got_ != fx.Int32((1 << world) - 1)
                                if late & (lane == 0):
                                    fx.atomic_add(fx.recast_iter(fx.Int32, fx.inttoptr(
                                        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                                        fx.Int64(fx.ptrtoint(arg_tmo)))), fx.Int32(1), syncscope="agent")
                                if lane == 0:
                                    buffer_store(rcall, recv_rs, sel_r, cache_modifier=SCOPE_DEV)
                                stamp(2, 0)
                                m_hi = run_h
                                m_lo = run_l
                            else:
                                # register c: shard 2c's keys in lanes 0-15, shard 2c + 1's reversed in lanes 16-31
                                srcs = [((bank + ((lane + 32 * c) // topk) * max_rows + sel_r) * slot_keys
                                         + (lane < 16).select(lane, 31 - lane)) * 2 for c in range_constexpr(2)]
                                reals = [(lane + 32 * c) < world * topk for c in range_constexpr(2)]

                                def rload():
                                    return tuple(Vec(buffer_load(own_rs, reals[c].select(srcs[c], fx.Int32(0)),
                                                                 vec_width=2, dtype=T.i32, cache_modifier=SCOPE_SYS))
                                                 for c in range_constexpr(2))

                                def rlate(ws):
                                    m_ = fx.Int32(0)
                                    for c in range_constexpr(2):
                                        tg = (fx.Int32(ws[c][1]) >> 17) & fx.Int32(0x7FFF)
                                        m_ = m_ | fx.Int32(rocdl.ballot(T.i32, reals[c] & (tg != rtag)))
                                    return m_ != zero_

                                ws = rload()
                                late = rlate(ws)
                                rit = zero_
                                while late & (rit < i32_maxit):
                                    rocdl.s_sleep(POLL_SLEEP)
                                    ws = rload()
                                    late = rlate(ws)
                                    rit = rit + 1
                                if late & (lane == 0):
                                    fx.atomic_add(fx.recast_iter(fx.Int32, fx.inttoptr(
                                        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                                        fx.Int64(fx.ptrtoint(arg_tmo)))), fx.Int32(1), syncscope="agent")
                                if lane == 0:
                                    buffer_store(rcall, recv_rs, sel_r, cache_modifier=SCOPE_DEV)
                                stamp(2, 0)
                                # owner_fast: the poll's last read already holds every key (each 8-B word carries its own
                                # tag and is single-copy atomic), so no second system-scope read (production re-reads
                                # because its waves poll only one copy of each key). owner_fast 0: the re-read (A/B arm).
                                if const_expr(not owner_fast):
                                    ws = rload()
                                causal_len = length_ - ql + token + 1
                                btab = fx.recast_iter(fx.Int32, arg_bt)
                                valid_blocks = (causal_len + 127) // 128
                                m_start = valid_blocks - local_keep
                                m_start = (m_start < zero_).select(zero_, m_start)
                                cand = []
                                pgs = []
                                for c in range_constexpr(2):
                                    kh_ = reals[c].select(fx.Int32(ws[c][1]) & fx.Int32(0x1FFFF), zero_)
                                    kl_ = reals[c].select(fx.Int32(ws[c][0]), zero_)
                                    blk_ = (kl_ & fx.Int32(0xFFFF)) - 1
                                    real = (kh_ != zero_) | (kl_ != zero_)
                                    sc32 = ((kh_ & fx.Int32(0xFFFF)) << 16) | ((kl_ >> 16) & fx.Int32(0xFFFF))
                                    bits = sc32 ^ (sc32 < zero_).select(fx.Int32(-0x80000000), fx.Int32(-1))
                                    val = force(bits.bitcast(fx.Float32), blk_, real, m_start)
                                    # the candidate's page, read now (in flight during the merge) and carried with the key,
                                    # so the emit has no dependent table read after the merge
                                    if const_expr(owner_fast):
                                        # the candidate's page, read now (in flight during the merge); the merge carries
                                        # only the candidate's slot (lane + 32 c), and the winner's page is picked from
                                        # the slot afterwards, so the merge does not wait for the read
                                        pgs.append(real.select(fx.Int32(btab[real.select(blk_, zero_)]), zero_))
                                    cand.append((*pack(val, blk_ + 1, real), lane + fx.Int32(32 * c)))
                                # every sender's 16 keys come sorted (AQ's bitonic list, production's tl.topk) and the
                                # re-pinning is idempotent, so the registers are bitonic: lanes 0-15 non-increasing,
                                # 16-31 non-decreasing. Checked; if not, the full sort.
                                bad_b = zero_
                                for c in range_constexpr(2):
                                    ck_b = k64(cand[c][0], cand[c][1])
                                    nk_b = k64(fx.Int32(gpu.shuffle_down(cand[c][0], 1, WAVE)),
                                               fx.Int32(gpu.shuffle_down(cand[c][1], 1, WAVE)))
                                    ok_b = (((lane & 15) == 15) | ((lane < 16) & (ck_b >= nk_b))
                                            | ((lane >= 16) & (ck_b <= nk_b)))
                                    bad_b = bad_b | fx.Int32(rocdl.ballot(T.i32, ok_b == fx.Boolean(False)))
                                m_hi = zero_
                                m_lo = zero_
                                m_pg = zero_
                                if bad_b == zero_:
                                    m_hi, m_lo, m_pg = merge16(*sort16([cx(cand[0], 16, nbit_b[16]),
                                                                        cx(cand[1], 16, nbit_b[16])], [True, False]), True)
                                else:
                                    m_hi, m_lo, m_pg = merge16(*top16(cand, [True, False]), True)
                            stamp(7, 0)
                            # ---- indices + the NVFP4 page-128 table row (_emit_sparse_block_table_row, a page a block)
                            nsel = (valid_blocks < topk).select(valid_blocks, fx.Int32(topk))
                            tidx = (m_lo & fx.Int32(0xFFFF)) - 1
                            tidx = (lane < nsel).select(tidx, fx.Int32(-1))
                            idx_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_idx)), ql * topk * 4)
                            if lane < topk:
                                buffer_store(tidx, idx_rs, sel_r * topk + lane)
                            self_blk = (causal_len - 1) // 128
                            bvalid = (lane < topk) & (tidx >= zero_) & (tidx <= self_blk) & (causal_len > zero_)
                            is_tail = bvalid & (tidx == self_blk)
                            is_full = bvalid & (tidx < self_blk)
                            full_m = fx.Int32(rocdl.ballot(T.i32, is_full))
                            n_full = fx.Int32(fx.math.ctpop(full_m))
                            n_valid = fx.Int32(fx.math.ctpop(fx.Int32(rocdl.ballot(T.i32, bvalid))))
                            has_tail = fx.Int32(rocdl.ballot(T.i32, is_tail)) != zero_
                            earlier = fx.Int32(fx.math.ctpop(full_m & lane_lt))
                            bslot = is_full.select(earlier, n_full)
                            if const_expr(owner_v2):
                                pt_ = Vec(fx.ptr_load(_lds_i32(lst_u8, PT_OFF + bvalid.select(tidx, zero_) * 4, 1),
                                                      result_type=fx.Vector.make_type(1, fx.Int32)))
                                page = bvalid.select(fx.Int32(pt_[0]), zero_)
                            elif const_expr(owner_fast):
                                # btab[tidx]: the winner's candidate slot m_pg -> lane m_pg % 32 of register m_pg / 32
                                sl_ = m_pg & fx.Int32(31)
                                pg_w = (m_pg >= fx.Int32(32)).select(fx.Int32(gpu.shuffle_idx(pgs[1], sl_, WAVE)),
                                                                     fx.Int32(gpu.shuffle_idx(pgs[0], sl_, WAVE)))
                                page = bvalid.select(pg_w, zero_)
                            else:
                                page = bvalid.select(fx.Int32(btab[bvalid.select(tidx, zero_)]), zero_)
                            sbt_rs = _rsrc(fx.Int64(fx.ptrtoint(arg_sbt)), ql * topk * 4)
                            if bvalid:
                                buffer_store(page, sbt_rs, sel_r * topk + bslot)
                            if (lane < topk) & (lane >= n_valid):
                                buffer_store(zero_, sbt_rs, sel_r * topk + lane)
                            ctx = n_full * 128 + has_tail.select(causal_len - self_blk * 128, zero_)
                            nv128 = n_valid * 128
                            ctx = has_tail.select(ctx, (nv128 < causal_len).select(nv128, causal_len))
                            ctx = (ctx < zero_).select(zero_, ctx)
                            if lane == 0:
                                buffer_store(ctx, _rsrc(fx.Int64(fx.ptrtoint(arg_sctx)), ql * 4), sel_r)
        if tid == 0:
            buffer_store(e_, cnt_rs, bid, cache_modifier=SCOPE_DEV)
        if const_expr(ipf_kib):
            if (bid == 0) & (tid == 0):
                pc_ = getpc()
                buffer_store(Vec.from_elements([fx.Int32(pc_), fx.Int32(pc_ >> fx.Int64(32))], fx.Int32), pc_rs, 0)
        stamp(3, 0)

    kernel_aq1(
        arg_a, arg_as, arg_w, arg_ws, arg_slab, arg_q, arg_iq, arg_qw, arg_kw, arg_iqw, arg_ikw, arg_cs, arg_pos,
        arg_slot, arg_islot, arg_kp, arg_vp, arg_ksc, arg_vsc, arg_ip, arg_isc, arg_bt, arg_len, arg_out,
        arg_aqq, arg_flag, arg_cnt, arg_err, arg_tl,
        arg_scb, arg_peers, arg_send, arg_recv, arg_tmo, arg_idx, arg_sbt, arg_sctx, arg_pc,
        i32_m, i32_lda, i32_mpad, i32_cs_rows, i64_kp_page, i64_ks_page, i64_ip_page, i64_is_page, i32_eps,
        i32_bt_stride, i32_local, i32_global, f32_scale, i32_maxit,
        value_attrs={"rocdl.cluster_dims": f"{csize},1,1"},
    ).launch(
        grid=(GRID, 1, 1),
        block=(block, 1, 1),
        stream=stream,
        cluster=(csize, 1, 1),
    )  # fmt: skip


launch_aq1.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
