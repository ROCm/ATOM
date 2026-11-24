# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MoE GEMM1 for H 6144, I 768 (gfx950) on 128-row sort blocks: MXFP4 x MXFP4 gate / up, SwiGLU-OAI, MXFP4 requant.

``gemm1_prefill_h6144_i768`` (layouts there) with 128-row tiles: half its A tile and scale staging, and the next
tile's nine prologue groups go out under three epilogue blocks instead of seven. A second, 64-row body takes tiles
whose sorted id at row 64 is padding: no upper-half LDS reads, MFMAs, SwiGLU or quant stores (GEMM2 masks those
rows). Both bodies issue the same G2S loads (padding rows read out of range), so the vmcnt waits and the next tile's
prologue under the epilogue still hold; valid rows are bit for bit the full body's, and ``signal`` counts the same
ready units.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from .common.epilogue import _maxf_nn
from .common.loaders import (
    _N_WAVES,
    _SCALE_A_REGION,
    _SCALE_B_REGION,
    _SCALE_LDS_BYTES,
    _SCALE_SLOTS,
    G2SLoaderAsm,
    S2RLoader128B,
    ScaleGatherMoE,
    ScaleLoaderLDS,
    _as_f32,
    _Buf,
    _flat_frag,
    _g2s_thunks,
    _min,
    _permlane16_swap,
    _riffle,
    _s2r_thunks,
    _swizzled_col,
    _unflat_frag,
    wait_barrier,
)
from .a16w4_utils import READY_STRIDE, SC1, global_atomic_add_agent
from .common.tile_map import CLAIM_COUNTERS
from .i3072.gemm1_prefill_h6144_i3072_persistent import (
    N_XCDS,
    MfmaRowStationaryFp4,
    _buffer_atomic_add,
    _cu_count,
    _cvt_pk_fp4,
    _e8m0_pair,
    _swiglu_oai_x4,
)
from .mxfp8.gemm1_prefill_h6144_i3072 import (
    _G2SCM,
    _MFMA_RESULT_WAIT,
    BLOCK_K,
    _fmax3_abs,
)
from amd_kernels.warm_up import launcher


@launcher
def compile_moe_gemm1_mxfp4_h6144_i768(*, E: int, BLOCK_M: int = 128, signal: bool = False):
    """The persistent launcher: one workgroup per CU, ``grid_size`` = the tile map's size.

    Args:
        signal: build the body for ``gemm12_a4w4`` (``launch.tile``): write-through stores, and each tile counted in
            its m-block's ready counter at the next tile's tail wait (the last tile after the loop's final wait).
    """
    H, I = 6144, 768  # noqa: E741
    assert BLOCK_M == 128
    N_CTAS = _cu_count()
    assert N_CTAS % N_XCDS == 0, N_CTAS
    K = H
    BLOCK_K_BYTES = BLOCK_K
    BLOCK_N = 256  # 128 gate cols + the matching 128 up cols
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 16-row tiles per wave per LDS half
    N_TILES_B = LDS_BLOCK_N // 2 // 16  # = 4

    assert K % 256 == 0 and I % LDS_BLOCK_N == 0
    K_ITERS = (K // 2) // BLOCK_K  # fp4: 256 K per 128-B step
    UNROLL = 4 if (K_ITERS - 4) % 4 == 0 else 2
    assert K_ITERS >= 4 and (K_ITERS - 4) % UNROLL == 0, K_ITERS
    N_ACCUMS = N_TILES_A * N_TILES_B
    K_BYTES = K // 2
    N_BITS = (I // 128 - 1).bit_length()
    N0 = I // 16  # 16-column (gate + up) groups per expert

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES
    A_BUFS = 4 * a_lds_size
    LDS_TILES_BYTES = A_BUFS + 4 * b_lds_size

    A_WAVE_GROUPS = N_TILES_A // 2  # groups per wave per LDS half
    A_HALF_GROUPS = LDS_BLOCK_M // 32
    B_WAVE_GROUPS = 4  # consecutive groups per wave quarter
    B_HALF_GROUPS = 2  # quarter block b -> group G + b (via (b//2)*2 + b%2)
    SCALE_COLS_OUT = I // 32  # e8m0 per output row
    OUT_SC_BLOCKS_PER_ROW32 = SCALE_COLS_OUT // 8
    W_BYTES = (2 * I) * K_BYTES
    assert W_BYTES <= 0xFFFFFFFF, "buffer resources address 4 GB"

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, _SCALE_LDS_BYTES, 16]
        claim: fx.Array[fx.Int32, 4, 16]

    STORE_KW = {"cache_modifier": SC1} if signal else {}

    @flyc.jit
    def tile(
        _base_ptr,
        _scale_lds_ptr,
        claim_slot,
        bx,
        A,
        W13,
        OUT_Q,
        A_scale,
        W13_scale,
        OUT_scale,
        sorted_ids,
        sorted_expert_ids,
        n_tokens,
        num_m_blocks,
        a_scale_bytes,
        tile_map_t,
        grid_size,
        ready,
    ):
        """Workgroup ``bx``'s persistent tile loop on its LDS tiles (``_base_ptr``), scale slots (``_scale_lds_ptr``)
        and claim words (``claim_slot``); with ``signal``, ``ready`` holds ``gemm12_a4w4``'s m-block counters."""

        a_cur0 = _Buf(_base_ptr, 0 * a_lds_size)
        a_cur1 = _Buf(_base_ptr, 1 * a_lds_size)
        a_next0 = _Buf(_base_ptr, 2 * a_lds_size)
        a_next1 = _Buf(_base_ptr, 3 * a_lds_size)
        b_cur0 = _Buf(_base_ptr, A_BUFS + 0 * b_lds_size)
        b_cur1 = _Buf(_base_ptr, A_BUFS + 1 * b_lds_size)
        b_next0 = _Buf(_base_ptr, A_BUFS + 2 * b_lds_size)
        b_next1 = _Buf(_base_ptr, A_BUFS + 3 * b_lds_size)
        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        tid = fx.thread_idx.x
        lane_id = tid % 64
        wave_id = tid // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2

        ids_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BLOCK_M * 4)
        )
        eid_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4
        )
        # tile_map[grid_size] = valid entries; XCD x (workgroup w runs on XCD w % 8) claims its run [x * per_xcd,
        # min((x + 1) * per_xcd, valid)) through tile_map[grid_size - CLAIM_COUNTERS + x] (-1 from the tile map)
        tm_rsrc = buffer_ops.create_buffer_resource(
            tile_map_t, max_size=False, num_records_bytes=(grid_size + 1) * 4
        )
        xcd = bx % N_XCDS
        n_valid = fx.Int32(
            buffer_ops.buffer_load(
                tm_rsrc, grid_size, vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        per_xcd = (n_valid + fx.Int32(N_XCDS - 1)) // fx.Int32(N_XCDS)
        run_lo = xcd * per_xcd
        run_n = _min(per_xcd, n_valid - run_lo)
        # only thread 0 is in range: one claim per workgroup
        claim_off = (tid == 0).select(
            (grid_size - CLAIM_COUNTERS + xcd) * fx.Int32(4), fx.Int32(0x7FFFFFF0)
        )

        def _entry_of(c):
            """Tile-map entry of a claimed counter value c (uniform; run index c + 1), -1 past the run."""
            t = c + fx.Int32(1)
            ok = t < run_n
            e = fx.Int32(
                buffer_ops.buffer_load(
                    tm_rsrc,
                    ok.select(run_lo + t, fx.Int32(0)),
                    vec_width=1,
                    dtype=fx.Int32,
                    is_scalar=True,
                )
            )
            return ok.select(e, fx.Int32(-1))

        def _expert_of(entry):
            tile_i = (entry >= 0).select(entry >> N_BITS, fx.Int32(0))
            return fx.Int32(
                buffer_ops.buffer_load(
                    eid_rsrc, tile_i, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )

        def _a_offsets(entry):
            """Per-lane gathered A offsets of both LDS halves (sorted ids loads)."""
            m_base = (entry >= 0).select(entry >> N_BITS, fx.Int32(0)) * BLOCK_M
            offs = []
            for half in range_constexpr(2):
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    sid = fx.Int32(
                        buffer_ops.buffer_load(
                            ids_rsrc,
                            m_base + half * LDS_BLOCK_M + row,
                            vec_width=1,
                            dtype=fx.Int32,
                        )
                    )
                    # padded rows: tok == n_tokens -> OOB -> zeros
                    tok = sid & fx.Int32(0x00FFFFFF)
                    offs.append(tok * fx.Int32(K_BYTES) + _swizzled_col(row, col))
            return offs

        class _Tile:
            """Loaders and addresses of one tile (entry, its expert, its A offsets)."""

            def __init__(self, entry, expert, a_offs, live=True):
                """``live`` False: every load of the tile is out of range (no memory
                traffic; the loop exits before anything reads the LDS)."""
                self.entry = entry
                tile_i = entry >> N_BITS
                self.tile_j = entry & ((1 << N_BITS) - 1)
                self.m_base = tile_i * BLOCK_M
                weight_group0 = self.tile_j * fx.Int32(LDS_BLOCK_N // 16)
                group0 = expert * fx.Int32(N0) + weight_group0

                def _b_offsets(gu):
                    offs = []
                    for rnd in range_constexpr(N_TILES_B):
                        row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                        col = (lane_id // 8) * 16
                        # a16w4 order: gate block g at 2g, its up block at 2g + 1
                        blk = (weight_group0 + fx.Int32(row // 16)) * fx.Int32(
                            2
                        ) + fx.Int32(gu)
                        offs.append(
                            blk * fx.Int32(K_BYTES * 16)
                            + fx.Int32(
                                (row % 16) * 16
                                + (col // 64) * 1024
                                + ((col % 64) // 16) * 256
                                + (col % 16)
                            )
                        )
                    return offs

                self.scale_gather = ScaleGatherMoE(
                    A_scale,
                    W13_scale,
                    K,
                    lane_id,
                    wave_id,
                    _scale_lds_ptr,
                    a_scale_bytes * live,
                    E * 2 * I * (K // 32) * live,
                    A_WAVE_GROUPS,
                    A_HALF_GROUPS,
                    B_WAVE_GROUPS,
                    B_HALF_GROUPS,
                )
                self.scale_gather.set_wave_base(self.m_base, group0 * fx.Int32(32))
                a_rsrc = buffer_ops.create_buffer_resource(
                    A, max_size=False, num_records_bytes=n_tokens * K_BYTES * live
                )
                b_rsrc = buffer_ops.create_buffer_resource(
                    W13,
                    max_size=False,
                    num_records_bytes=fx.Int32(W_BYTES) * live,
                    base_byte_offset=fx.Int64(expert) * W_BYTES,
                )
                self.a0 = G2SLoaderAsm(a_rsrc, a_offs[:N_TILES_A], N_TILES_A, wave_id)
                self.a1 = G2SLoaderAsm(a_rsrc, a_offs[N_TILES_A:], N_TILES_A, wave_id)
                self.b0 = _G2SCM(b_rsrc, _b_offsets(0), N_TILES_B, wave_id, cm=" nt")
                self.b1 = _G2SCM(b_rsrc, _b_offsets(1), N_TILES_B, wave_id, cm=" nt")
                for ld in (self.a0, self.a1, self.b0, self.b1):
                    ld.set_wave_base(_base_ptr)

            def prologue_groups(self):
                """Scale blocks 0-2 and K steps 0, 1 into the cur / next buffers as groups of thunks, in the issue order
                the loop-top waits count (a loader's steps stay together: they chain m0)."""
                sg = self.scale_gather
                return [
                    [
                        lambda: sg.gather(0, _slot(0)),
                        lambda: sg.gather(1, _slot(1)),
                        lambda: sg.gather(2, _slot(2)),
                    ],
                    [lambda: self.a0.load(a_cur0, fx.Int32(0 * A_K_STEP))],
                    [lambda: self.b0.load(b_cur0, fx.Int32(0 * B_K_STEP))],
                    [lambda: self.b1.load(b_cur1, fx.Int32(0 * B_K_STEP))],
                    [lambda: self.a1.load(a_cur1, fx.Int32(0 * A_K_STEP))],
                    [lambda: self.a0.load(a_next0, fx.Int32(1 * A_K_STEP))],
                    [lambda: self.b0.load(b_next0, fx.Int32(1 * B_K_STEP))],
                    [lambda: self.b1.load(b_next1, fx.Int32(1 * B_K_STEP))],
                    [lambda: self.a1.load(a_next1, fx.Int32(1 * A_K_STEP))],
                ]

            def prologue(self):
                for grp in self.prologue_groups():
                    for th in grp:
                        th()

        A_K_STEP = BLOCK_K_BYTES
        B_K_STEP = 2 * 1024  # 128 B of K = two 1 KB blocks

        def _slot(g):
            return fx.Int32(g) % fx.Int32(_SCALE_SLOTS)

        _scale_base_ptr = _scale_lds_ptr

        def _publish(i, v):
            """LDS claim word i <- v (thread 0 hands a claimed index to the waves)."""
            claim_slot[i] = v

        out_rsrc = buffer_ops.create_buffer_resource(
            OUT_Q, max_size=False, num_records_bytes=num_m_blocks * (BLOCK_M * (I // 2))
        )
        osc_rsrc = buffer_ops.create_buffer_resource(
            OUT_scale,
            max_size=False,
            num_records_bytes=num_m_blocks * (BLOCK_M * SCALE_COLS_OUT),
        )
        g = lane_id // 16
        r16 = lane_id % 16

        # ---- first claim (exposed once per workgroup) ----
        first = _buffer_atomic_add(tm_rsrc, claim_off, 1)
        if tid == 0:
            _publish(0, first)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        entry = _entry_of(
            fx.Int32(
                _rocdl.readfirstlane(_T.i32, fx.as_ir_value(fx.Int32(claim_slot[0])))
            )
        )
        expert = _expert_of(entry)
        a_offs = _a_offsets(entry)
        if entry >= 0:
            _Tile(entry, expert, a_offs).prologue()

        # ``half`` is a Python constant: the runtime dispatch below emits two bodies with the prologue's G2S load counts
        def _run_tile(entry, expert, a_offs, prev_mb=None, half=False):
            """One tile (its prologue already issued); claims, prepares and issues the next tile's prologue and returns
            its (entry, expert, A offsets). The tile's objects live here: pre-loop objects whose methods the loop
            body called would become loop-carried state of the scf.while."""
            tile = _Tile(entry, expert, a_offs)
            a0_g2s, a1_g2s, b0_g2s, b1_g2s = tile.a0, tile.a1, tile.b0, tile.b1
            scale_gather = tile.scale_gather
            a_scale_ld = ScaleLoaderLDS(
                N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
            )
            b_scale_ld = ScaleLoaderLDS(
                N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
            )
            a_s2r = S2RLoader128B(wave_i, N_TILES_A)
            b_s2r = S2RLoader128B(wave_j, N_TILES_B)
            mfma = MfmaRowStationaryFp4(N_TILES_A, N_TILES_B)

            # gathers + a_cur0 landed: b0/b1/a1 + the 4 next batches may fly
            wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
            a0_frag = a_s2r.load(a_cur0)
            # b_cur0 AND b_cur1 landed: a1 + the 4 next batches may fly
            wait_barrier((3 * N_TILES_A) + (2 * N_TILES_B))
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            b1_frag = b_s2r.load(b_cur1, preshuffled=True)

            sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0))
            sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0))
            sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0 + sc0_sbC1)

            def _top_vmcnt(k2):
                return 2 * N_TILES_A + 2 * N_TILES_B + 1  # fp4: a gather every step

            def _seg2_vmcnt(k2):
                return 2 * N_TILES_A + N_TILES_B + 1

            _STEP0_VMCNT = 2 * N_TILES_A + 2 * N_TILES_B
            _STEP0_SEG2_VMCNT = 2 * N_TILES_A + N_TILES_B

            def _read_scale_thunks(kc_idx, holder):
                s = _slot(fx.Int32(kc_idx))

                def _r(dst, ld, half, _s=s):
                    holder[dst] = ld.read_half(_s, half)

                return [
                    lambda: _r(0, a_scale_ld, 0),
                    lambda: _r(1, a_scale_ld, 1),
                    lambda: _r(2, b_scale_ld, 0),
                    lambda: _r(3, b_scale_ld, 1),
                ]

            def _one_step(
                kc,
                k2,
                a0f,
                b0f,
                b1f_in,
                sc,
                accs,
                bufs,
                zero_acc=False,
                top_vmcnt=None,
                seg2_vmcnt=None,
            ):
                kc_i = fx.Int32(kc)
                top_vmcnt = _top_vmcnt(k2) if top_vmcnt is None else top_vmcnt
                seg2_vmcnt = _seg2_vmcnt(k2) if seg2_vmcnt is None else seg2_vmcnt
                ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                saR0, saR1, sbC = sc
                c00f, c01f, c10f, c11f = accs

                _a1 = [None] * N_TILES_A
                _a0n = [None] * N_TILES_A
                _b0n = [None] * N_TILES_B
                _b1n = [None] * N_TILES_B
                a_off = (kc_i + fx.Int32(2)) * fx.Int32(A_K_STEP)
                b_off = (kc_i + fx.Int32(2)) * fx.Int32(B_K_STEP)

                _scn = [None, None, None, None]  # type: list
                _rd_scn = _read_scale_thunks(kc_i + fx.Int32(1), _scn)
                _gk = _min(kc_i + fx.Int32(3), fx.Int32(K_ITERS - 1))
                _sc_gather = [lambda: scale_gather.gather(_gk, _slot(_gk))]

                wait_barrier(top_vmcnt)
                il = (
                    _riffle(
                        _g2s_thunks(a0_g2s, ac0, a_off, N_TILES_A),
                        ([] if half else _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False)),
                    )
                    + _rd_scn[:2]
                )
                c00f = mfma.call(
                    a0f, b0f, c00f, saR0, sbC, 0, k2, interleave=il, zero_acc=zero_acc
                )

                il = _riffle(_g2s_thunks(b0_g2s, bc0, b_off, N_TILES_B), _rd_scn[2:])
                # the second half's wait also under c01f's MFMAs (the barrier keeps its own): it overlaps them
                il += [lambda: fx.rocdl.s_waitcnt(vmcnt=seg2_vmcnt, lgkmcnt=0)]
                c01f = mfma.call(
                    a0f,
                    b1f_in,
                    c01f,
                    saR0,
                    sbC,
                    1,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                )
                a1f = _a1

                wait_barrier(seg2_vmcnt)
                il = (
                    _riffle(
                        _g2s_thunks(b1_g2s, bc1, b_off, N_TILES_B),
                        _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False),
                    )
                    + _sc_gather
                )
                if const_expr(not half):
                    c10f = mfma.call(
                        a1f, b0f, c10f, saR1, sbC, 0, k2, interleave=il, zero_acc=zero_acc
                    )
                else:
                    for th in il:
                        th()
                    c10f = c00f
                a0nf = _a0n

                il = _riffle(
                    _g2s_thunks(a1_g2s, ac1, a_off, N_TILES_A),
                    _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True)
                    + _s2r_thunks(b_s2r, bn1, _b1n, N_TILES_B, True),
                )
                # likewise the next step's top wait under c11f's MFMAs
                il += [lambda: fx.rocdl.s_waitcnt(vmcnt=_top_vmcnt(1 - k2), lgkmcnt=0)]
                if const_expr(not half):
                    c11f = mfma.call(
                        a1f,
                        b1f_in,
                        c11f,
                        saR1,
                        sbC,
                        1,
                        k2,
                        interleave=il,
                        zero_acc=zero_acc,
                    )
                else:
                    for th in il:
                        th()
                    c11f = c01f
                b0nf = _b0n
                b1nf = _b1n

                sc_next = (_scn[0], _scn[1], _scn[2] + _scn[3])
                new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
                return a0nf, b0nf, b1nf, sc_next, (c00f, c01f, c10f, c11f), new_bufs

            def _swap_bufs(bufs):
                ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                return (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

            n_a = 2 * N_TILES_A
            n_b = 2 * N_TILES_B
            n_ga = N_TILES_A // 2
            n_sc = 2 * n_ga + N_TILES_B
            _R = fx.as_ir_value

            def _flat_sc(sc):
                saR0, saR1, sbC = sc
                return (
                    [_R(v) for v in saR0] + [_R(v) for v in saR1] + [_R(v) for v in sbC]
                )

            def _unflat_sc(flat):
                o = 0
                saR0 = list(flat[o : o + n_ga])
                o += n_ga
                saR1 = list(flat[o : o + n_ga])
                o += n_ga
                sbC = list(flat[o : o + N_TILES_B])
                return (saR0, saR1, sbC)

            _accs0 = tuple([None] * N_ACCUMS for _ in range(4))
            a0f, b0f, b1f, sc, accs, _ = _one_step(
                0,
                0,
                a0_frag,
                b0_frag,
                b1_frag,
                sc0,
                _accs0,
                bufs0,
                zero_acc=True,
                top_vmcnt=_STEP0_VMCNT,
                seg2_vmcnt=_STEP0_SEG2_VMCNT,
            )
            a0f, b0f, b1f, sc, accs, _ = _one_step(
                1, 1, a0f, b0f, b1f, sc, accs, _swap_bufs(bufs0)
            )

            init_state = (
                _flat_frag(a0f)
                + _flat_frag(b0f)
                + _flat_frag(b1f)
                + _flat_sc(sc)
                + [_R(x) for x in accs[0]]
                + [_R(x) for x in accs[1]]
                + [_R(x) for x in accs[2]]
                + [_R(x) for x in accs[3]]
            )
            for kk, state in range(  # type: ignore[call-overload]
                2, K_ITERS - 2, UNROLL, init=init_state
            ):
                off = 0
                a0f = _unflat_frag(state[off : off + n_a], N_TILES_A)
                off += n_a
                b0f = _unflat_frag(state[off : off + n_b], N_TILES_B)
                off += n_b
                b1f = _unflat_frag(state[off : off + n_b], N_TILES_B)
                off += n_b
                sc = _unflat_sc(state[off : off + n_sc])
                off += n_sc
                c00f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c01f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c10f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c11f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                accs = (c00f, c01f, c10f, c11f)

                bufs = bufs0
                for u in range_constexpr(UNROLL):
                    a0f, b0f, b1f, sc, accs, bufs = _one_step(
                        kk + u, u % 2, a0f, b0f, b1f, sc, accs, bufs
                    )

                new_state = (
                    _flat_frag(a0f)
                    + _flat_frag(b0f)
                    + _flat_frag(b1f)
                    + _flat_sc(sc)
                    + [_R(x) for x in accs[0]]
                    + [_R(x) for x in accs[1]]
                    + [_R(x) for x in accs[2]]
                    + [_R(x) for x in accs[3]]
                )
                state = yield new_state

            off = 0
            a0_frag = _unflat_frag(state[off : off + n_a], N_TILES_A)
            off += n_a
            b0_frag = _unflat_frag(state[off : off + n_b], N_TILES_B)
            off += n_b
            b1_frag = _unflat_frag(state[off : off + n_b], N_TILES_B)
            off += n_b
            sc = _unflat_sc(state[off : off + n_sc])
            off += n_sc
            c00_frag = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c01_frag = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c10_frag = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c11_frag = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS

            # Tail step K_ITERS - 2 (no gather; reads the last scale block)
            saR0, saR1, sbC = sc
            _scn = [None, None, None, None]  # type: list
            _rd_scn = _read_scale_thunks(fx.Int32(K_ITERS - 1), _scn)
            _a1 = [None] * N_TILES_A
            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
            # claim the next tile now (thread 0; younger than every load the
            # remaining waits count, so they can only over-wait on it)
            nxt = _buffer_atomic_add(tm_rsrc, claim_off, 1)
            il = ([] if half else _s2r_thunks(a_s2r, a_cur1, _a1, N_TILES_A, False)) + _rd_scn
            c00_frag = mfma.call(
                a0_frag, b0_frag, c00_frag, saR0, sbC, 0, 0, interleave=il
            )
            a1_frag = _a1
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC, 1, 0)
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            _b1n = [None] * N_TILES_B
            wait_barrier(1 * N_TILES_A)
            il = (
                _s2r_thunks(a_s2r, a_next0, _a0n, N_TILES_A, False)
                + _s2r_thunks(b_s2r, b_next0, _b0n, N_TILES_B, True)
                + _s2r_thunks(b_s2r, b_next1, _b1n, N_TILES_B, True)
            )
            if const_expr(not half):
                c10_frag = mfma.call(
                    a1_frag, b0_frag, c10_frag, saR1, sbC, 0, 0, interleave=il
                )
            else:
                for th in il:
                    th()
                c10_frag = c00_frag
            if const_expr(not half):
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC, 1, 0)
            else:
                c11_frag = c01_frag
            a0_frag = _a0n
            b0_frag = _b0n
            b1_frag = _b1n

            # Tail step K_ITERS - 1 (its A half 1 sits in a_next1)
            _a1 = [None] * N_TILES_A
            wait_barrier(0)
            if tid == 0:
                _publish(1, nxt)
            if const_expr(signal):
                # every wave's loads and stores so far have completed, the previous
                # tile's stores included: count its block (4 waves) at no extra wait
                if tid == 0:
                    if prev_mb >= 0:
                        global_atomic_add_agent(
                            ready + fx.Int64(prev_mb * READY_STRIDE), 4
                        )
            saR0, saR1, sbC = (_scn[0], _scn[1], _scn[2] + _scn[3])
            il = ([] if half else _s2r_thunks(a_s2r, a_next1, _a1, N_TILES_A, False))
            c00_frag = mfma.call(
                a0_frag, b0_frag, c00_frag, saR0, sbC, 0, 1, interleave=il
            )
            a1_frag = _a1
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC, 1, 1)
            c00_frag = mfma.hold(c00_frag)
            if const_expr(not half):
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, saR1, sbC, 0, 1)
            else:
                c10_frag = c00_frag
            c01_frag = mfma.hold(c01_frag)
            if const_expr(not half):
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC, 1, 1)
            else:
                c11_frag = c01_frag
            c10_frag = mfma.hold(c10_frag)
            c11_frag = mfma.hold(c11_frag, wait=_MFMA_RESULT_WAIT)

            # every wave is done with the tile's LDS; the claim is visible
            wait_barrier(0)
            entry_n = _entry_of(
                fx.Int32(
                    _rocdl.readfirstlane(
                        _T.i32, fx.as_ir_value(fx.Int32(claim_slot[1]))
                    )
                )
            )
            expert_n = _expert_of(entry_n)
            a_offs_n = _a_offsets(entry_n)

            # ---- epilogue: swiglu-OAI + MXFP4 quant, sorted-row output ----
            col_base = tile.tile_j * LDS_BLOCK_N + wave_j * (N_TILES_B * 16)
            colgrp_base = col_base // 32

            def _epilogue(c_gate, c_up, base_row, inter=(), defer=False):
                """One quadrant pair (rows base_row + ti*16 + r16, the wave's 64 gate columns), row tiles in pairs
                (2 tp, 2 tp + 1) sharing the e8m0 exchange. ``inter[k]``: thunks run after the k-th (p, ti) block's
                SwiGLU. ``defer``: return the stores as thunks and fence each block (the inter thunks carry the next
                tile's prologue loads; unfenced, the scheduler bunches them and they stall at issue)."""
                stores = []
                k_it = 0
                for p in range_constexpr(N_TILES_B // 2):  # 32-col group p
                    colgrp = colgrp_base + p
                    sc_in_block = (colgrp % 4) * 64 + r16 * 4 + ((colgrp % 8) // 4) * 2
                    for tp in range_constexpr(N_TILES_A // 2):
                        hs = []
                        amaxs = []
                        for ti in (2 * tp, 2 * tp + 1):
                            gv = Vec(c_gate[mfma.idx(ti, 2 * p)])
                            gw = Vec(c_gate[mfma.idx(ti, 2 * p + 1)])
                            uv = Vec(c_up[mfma.idx(ti, 2 * p)])
                            uw = Vec(c_up[mfma.idx(ti, 2 * p + 1)])
                            h = []
                            for cg, cu in ((gv, uv), (gw, uw)):
                                h += _swiglu_oai_x4(
                                    [fx.Float32(cg[v]) for v in range_constexpr(4)],
                                    [fx.Float32(cu[v]) for v in range_constexpr(4)],
                                )
                            a01 = _fmax3_abs(h[0], h[1], h[2])
                            a34 = _fmax3_abs(h[3], h[4], h[5])
                            a67 = _fmax3_abs(h[6], h[7], a01)
                            amaxs.append(_maxf_nn(a67, a34))
                            hs.append(h)
                            if const_expr(k_it < len(inter)):
                                for th in inter[k_it]:
                                    th()
                            k_it += 1
                            if const_expr(defer):
                                fx.rocdl.sched_barrier(0)
                        e8m0s = _e8m0_pair(amaxs[0], amaxs[1])
                        for k in range_constexpr(2):
                            h = hs[k]
                            scale_f = _as_f32(e8m0s[k] << 23)
                            pa = _cvt_pk_fp4(fx.Int32(0), h[0], h[1], scale_f, 0)
                            pa = _cvt_pk_fp4(pa, h[2], h[3], scale_f, 1)
                            pb = _cvt_pk_fp4(fx.Int32(0), h[4], h[5], scale_f, 0)
                            pb = _cvt_pk_fp4(pb, h[6], h[7], scale_f, 1)
                            # after the swap, lane group g holds tile 2p + g%2, columns (g//2)*8 .. +8: one dword
                            pa, pb = _permlane16_swap(pa, pb)
                            row = base_row + (2 * tp + k) * 16 + r16
                            col = col_base + (2 * p + (g % 2)) * 16 + (g // 2) * 8
                            stores.append(
                                functools.partial(
                                    buffer_ops.buffer_store,
                                    pa | (pb << 16),
                                    out_rsrc,
                                    row * (I // 2) + col // 2,
                                    offset_is_bytes=True,
                                    **STORE_KW,
                                )
                            )
                            if const_expr(not defer):
                                stores.pop()()
                        row32 = (base_row // 32) + tp
                        blk = row32 * OUT_SC_BLOCKS_PER_ROW32 + colgrp // 8
                        pair = e8m0s[0] | (e8m0s[1] << 8)
                        pair16 = fx.Int32(pair).to(fx.Int16)
                        stores.append(
                            functools.partial(
                                buffer_ops.buffer_store,
                                pair16,
                                osc_rsrc,
                                blk * 256 + sc_in_block,
                                offset_is_bytes=True,
                                **STORE_KW,
                            )
                        )
                        if const_expr(not defer):
                            stores.pop()()
                return stores

            row_r0 = tile.m_base + wave_i * (N_TILES_A * 16)
            row_r1 = row_r0 + LDS_BLOCK_M
            # the next tile's prologue (past the run: tile 0 with every load out of range, drained before exit) goes out
            # under the first quadrant pair's blocks after block 0, which covers the sorted-ids latency; that pair's
            # stores wait until those loads are issued (the compiler's wait for the sorted ids would otherwise drain
            # them), then spread over the second pair's blocks
            live_n = entry_n >= 0
            nt = _Tile(
                live_n.select(entry_n, fx.Int32(0)),
                expert_n,
                a_offs_n,
                live=live_n.select(fx.Int32(1), fx.Int32(0)),
            )
            grp = nt.prologue_groups()
            inter0 = [
                [],
                grp[0] + grp[1] + grp[2],
                grp[3] + grp[4] + grp[5],
                grp[6] + grp[7] + grp[8],
            ]
            stores0 = _epilogue(c00_frag, c01_frag, row_r0, inter=inter0, defer=True)
            n_it = N_TILES_A * (N_TILES_B // 2)
            inter1 = [stores0[k::n_it] for k in range(n_it)]
            if const_expr(half):
                for store in stores0:
                    store()
            else:
                _epilogue(c10_frag, c11_frag, row_r1, inter=inter1)

            if const_expr(signal):
                return entry_n, expert_n, a_offs_n, entry >> N_BITS
            return entry_n, expert_n, a_offs_n

        if const_expr(signal):
            prev_mb = fx.Int32(-1)
            while entry >= 0:
                sid128 = fx.Int32(buffer_ops.buffer_load(
                    ids_rsrc, (entry >> N_BITS) * BLOCK_M + BLOCK_M // 2,
                    vec_width=1, dtype=fx.Int32, is_scalar=True))
                if (sid128 & 0x00FFFFFF) < n_tokens:
                    entry, expert, a_offs, prev_mb = _run_tile(entry, expert, a_offs, prev_mb, half=False)
                else:
                    entry, expert, a_offs, prev_mb = _run_tile(entry, expert, a_offs, prev_mb, half=True)
            # the last tile's stores (and sanitized prologue loads) have completed
            fx.rocdl.s_waitcnt(vmcnt=0)
            if lane_id == 0:
                if prev_mb >= 0:
                    global_atomic_add_agent(ready + fx.Int64(prev_mb * READY_STRIDE), 1)
        else:
            while entry >= 0:
                sid128 = fx.Int32(buffer_ops.buffer_load(
                    ids_rsrc, (entry >> N_BITS) * BLOCK_M + BLOCK_M // 2,
                    vec_width=1, dtype=fx.Int32, is_scalar=True))
                if (sid128 & 0x00FFFFFF) < n_tokens:
                    entry, expert, a_offs = _run_tile(entry, expert, a_offs, half=False)
                else:
                    entry, expert, a_offs = _run_tile(entry, expert, a_offs, half=True)

            # the last tile's (sanitized) prologue loads land before the LDS goes away
            fx.rocdl.s_waitcnt(vmcnt=0)

    @flyc.kernel(name="kernel_gemm1_mxfp4_h6144_i768_bm128_half_rows")
    def kernel_gemm1_mxfp4_h6144_i768_bm128_half_rows(
        A: fx.Tensor,
        W13: fx.Tensor,
        OUT_Q: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        tile_map_t: fx.Tensor,
        grid_size: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile(
            lds.all_lds.ptr,
            lds.scale_lds.ptr,
            lds.claim.ptr,
            fx.block_idx.x,
            A,
            W13,
            OUT_Q,
            A_scale,
            W13_scale,
            OUT_scale,
            sorted_ids,
            sorted_expert_ids,
            n_tokens,
            num_m_blocks,
            a_scale_bytes,
            tile_map_t,
            grid_size,
            fx.Int64(0),
        )

    @flyc.jit
    def launch_gemm1_mxfp4_h6144_i768_bm128_half_rows(
        A: fx.Tensor,
        W13: fx.Tensor,
        OUT_Q: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        tile_map_t: fx.Tensor,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm1_mxfp4_h6144_i768_bm128_half_rows(
            A,
            W13,
            OUT_Q,
            A_scale,
            W13_scale,
            OUT_scale,
            sorted_ids,
            sorted_expert_ids,
            n_tokens,
            num_m_blocks,
            a_scale_bytes,
            tile_map_t,
            grid_size,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(N_CTAS, 1, 1), block=(256, 1, 1), stream=stream)

    launch_gemm1_mxfp4_h6144_i768_bm128_half_rows.tile = tile
    launch_gemm1_mxfp4_h6144_i768_bm128_half_rows.n_ctas = N_CTAS
    launch_gemm1_mxfp4_h6144_i768_bm128_half_rows.n_tiles_n = I // 128
    launch_gemm1_mxfp4_h6144_i768_bm128_half_rows.lds_bytes = (LDS_TILES_BYTES, _SCALE_LDS_BYTES)
    return launch_gemm1_mxfp4_h6144_i768_bm128_half_rows
