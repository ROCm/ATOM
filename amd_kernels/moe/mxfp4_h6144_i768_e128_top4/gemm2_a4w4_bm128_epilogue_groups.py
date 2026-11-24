# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``gemm2_a4w4_bm128`` (128-row sort, at most 1536 tokens) in which a wave skips the epilogue of each of its 16-row
groups that is all padding; every MFMA, load and CTA barrier stays unconditional.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

from .common.epilogue import (
    _NUM_XCDS,
    _STORE_CPOL,
    _bf16x2,
    _lds_load_vec,
    _lds_store_vec,
    _v2i32,
)
from .a16w4_utils import READY_STRIDE, poll_ready, wait_ready_from
from .common.loaders import (
    _N_WAVES,
    _SCALE_A_REGION,
    _SCALE_B_REGION,
    _SCALE_SLOT_BYTES,
    G2SLoaderAsm,
    S2RLoader128B,
    ScaleGatherMoE,
    ScaleLoaderLDS,
    _asm_void,
    _Buf,
    _divmod_nonneg,
    _flat_frag,
    _g2s_thunks,
    _riffle,
    _s2r_thunks,
    _swizzled_col,
    _unflat_frag,
    wait_barrier,
)
from .i3072.gemm1_prefill_h6144_i3072_persistent import Mfma16x16x128Fp4
from .mxfp8.gemm1_prefill_h6144_i3072 import BLOCK_K
from amd_kernels.warm_up import launcher


_SCALE_SLOTS2 = 4


@launcher
def compile_moe_gemm2_mxfp4_h6144_i768(
    *,
    E: int,
    topk: int,
    n_tokens: int = 513,
    n_split: int = 6,
    sort_block_m: int = 128,
    wait_for: int = 0,
):
    """The launcher; ``launch.tile`` is the workgroup body of the one-launch GEMM1 + GEMM2 kernels.

    Args:
        n_tokens: picks the row-probe and padded-half epilogue forms; the launch takes the runtime M.
        wait_for: > 0: before loading the intermediate, the body waits until its m-block's ready count reaches it.
    """
    assert n_tokens <= 1536 and sort_block_m == 128
    # one wave ballots the eight row-group leaders from 1537 tokens; at <= 1536 the SMEM probes measured no slower
    wave_probes = n_tokens >= 1537
    group_skip = True
    # skip row half 1's epilogue when all its rows are padding: a gain at 1537-2048 tokens, noise at 3072 with GEMM1
    pad_skip = 1537 <= n_tokens <= 2048 and sort_block_m == 128
    probe_suffix = ("_wave_probes" if wave_probes else "") + ("_pad_skip" if pad_skip else "")
    H, I = 6144, 768  # noqa: E741
    BM = 128
    BN = 256
    K = I
    K_BYTES = K // 2  # A row bytes (packed e2m1)
    KW_BYTES = K // 2  # W2 row bytes
    BLOCK_K_BYTES = BLOCK_K
    EID_SHIFT = (sort_block_m // BM).bit_length() - 1  # 0 or 1
    assert sort_block_m in (BM, 2 * BM)
    N_TILES_ALL = H // BN
    assert H % BN == 0 and N_TILES_ALL % n_split == 0
    NT = N_TILES_ALL // n_split  # n-tiles per CTA
    K_ITERS = K_BYTES // BLOCK_K
    assert K_ITERS == 3 and K % 256 == 0
    assert NT % 2 == 0, "two n-tiles per mainloop iteration"
    FLAT = 2 * K_ITERS  # K-steps per iteration
    SCALE_BLOCKS = K_ITERS  # one 256-B scale block (8 K-groups = 256 K) per step
    LDS_BLOCK_M = BM // 2
    LDS_BLOCK_N = BN // 2
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 4
    N_TILES_B = LDS_BLOCK_N // 2 // 16  # 4
    N_ACCUMS = N_TILES_A * N_TILES_B
    NB = N_TILES_B
    NA = N_TILES_A
    OUT_ROW_BYTES = H * 2
    W2_BYTES = H * KW_BYTES
    assert W2_BYTES <= 0xFFFFFFFF
    B_TILE_BYTES = BN * KW_BYTES  # one n-tile of W2 (256 rows)

    B_STAGES = 2
    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES  # 16 KB
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES  # 16 KB
    A_BUFS = 4 * a_lds_size  # 64 KB: 2 stages x 2 halves
    LDS_TILES_BYTES = A_BUFS + 2 * B_STAGES * b_lds_size  # 128 KB: 64 A + 64 W2
    SCALE_LDS_BYTES2 = _SCALE_SLOTS2 * _SCALE_SLOT_BYTES  # 16 KB
    # epilogue staging, wave-private: 16 rows x (2 halves x 64 cols) bf16, 256 B rows
    STG_ROW = 2 * N_TILES_B * 16 * 2  # 256 B: the wave's two 128-B line segments
    STG_CH = STG_ROW // 16  # 16 chunks of 16 B per staged row
    STG_ROWS = 16  # one (row half, a-tile) at a time
    STG_WAVE = STG_ROWS * STG_ROW  # 4 KB
    STG_LDS_BYTES = _N_WAVES * STG_WAVE  # 16 KB
    N_FLUSH = (STG_ROWS * 2) // 8  # dwordx4 stores per round per lane: 32 lines, 8 each
    SID_LDS_BYTES = BM * 4  # the m-tile's sorted ids, read by every n-tile's epilogue
    assert LDS_TILES_BYTES + SCALE_LDS_BYTES2 + STG_LDS_BYTES + SID_LDS_BYTES <= 160 * 1024

    A_WAVE_GROUPS = N_TILES_A // 2  # 2
    A_HALF_GROUPS = LDS_BLOCK_M // 32  # 4
    B_WAVE_GROUPS = 2  # 32-row groups per wave (64 W2 rows)
    B_HALF_GROUPS = LDS_BLOCK_N // 32  # 4: half 1 = 128 rows further
    SC_COLS = K // 32
    W2_SC_BYTES = E * H * SC_COLS

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SCALE_LDS_BYTES2, 16]
        stage_lds: fx.Array[fx.Int8, STG_LDS_BYTES, 16]
        sids: fx.Array[fx.Int32, BM, 16]

    @flyc.jit
    def tile(
        _base_ptr,
        _scale_base_ptr,
        _stg_ptr,
        sid_slot,
        bx,
        A,
        W2,
        OUT,
        A_scale,
        W2_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        n_tokens,
        num_m_blocks,
        ready,
    ):
        """Workgroup ``bx``'s (m-tile, n-chunk) on the given LDS regions; ``ready``: the one-launch kernel's m-block
        counters (``wait_for`` > 0)."""

        a_cur0 = _Buf(_base_ptr, 0 * a_lds_size)
        a_cur1 = _Buf(_base_ptr, 1 * a_lds_size)
        a_next0 = _Buf(_base_ptr, 2 * a_lds_size)
        a_next1 = _Buf(_base_ptr, 3 * a_lds_size)
        # B: 2 stages x 2 halves
        b_st = [
            [_Buf(_base_ptr, A_BUFS + (2 * st + hf) * b_lds_size) for hf in range(2)]
            for st in range(B_STAGES)
        ]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        g4 = lane_id // 16
        r16 = lane_id % 16

        # ---- work item: (m-tile, n-chunk); consecutive m-tiles share an XCD ----
        nv_rsrc = buffer_ops.create_buffer_resource(
            num_valid_ids, max_size=False, num_records_bytes=4
        )
        num_valid = fx.Int32(
            buffer_ops.buffer_load(
                nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        n_work = (num_valid // fx.Int32(BM)) * fx.Int32(n_split)
        per_xcd = (n_work + fx.Int32(_NUM_XCDS - 1)) // fx.Int32(_NUM_XCDS)
        intra, xcd = _divmod_nonneg(bx, _NUM_XCDS)
        work = xcd * per_xcd + intra
        block_valid = (intra < per_xcd) & (work < n_work)
        work_safe = block_valid.select(work, fx.Int32(0))
        tile_i, chunk = _divmod_nonneg(work_safe, n_split)
        m_base = tile_i * BM
        eid_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4
        )
        expert = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                tile_i >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        ids_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
        )
        if const_expr(EID_SHIFT > 0):
            # a 2*BM-row sort pads each expert to 2*BM rows: a tile led by the sentinel (tok == n_tokens) is empty
            first_sid = fx.Int32(
                buffer_ops.buffer_load(
                    ids_rsrc, m_base, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            block_valid = block_valid & ((first_sid & fx.Int32(0x00FFFFFF)) < n_tokens)
        chunk_n0 = chunk * NT  # first n-tile (global index) of this CTA
        # no n-tile rotation: the physical tile is the logical one
        nt_rot = fx.Int32(0)

        def _pn(nt):
            """CTA-local logical n-tile (sweep order) -> physical n-tile of the chunk"""
            x = nt + nt_rot
            return (x >= fx.Int32(NT)).select(x - fx.Int32(NT), x)

        def _n_glob(nt):
            """Global n-tile of ``nt``, clamped: the redundant loads after the last tile re-fetch the last one."""
            ntc = (nt >= fx.Int32(NT)).select(fx.Int32(NT - 1), nt)
            return chunk_n0 + _pn(ntc)

        if block_valid:
            if const_expr(wait_for > 0):
                # thread 0's first poll overlaps the setup's loads
                ready_addr = ready + fx.Int64(tile_i * READY_STRIDE)
                got0 = poll_ready(ready_addr)
            if const_expr(wave_probes):
                # padding is a row suffix: lanes 0-7 probe the eight 16-row group leaders and the low 8 ballot bits
                # count the valid prefix in 16-row groups, in every wave the same count as the SMEM probes
                probe_sid = fx.Int32(buffer_ops.buffer_load(
                    ids_rsrc, m_base + (lane_id % 8) * 16,
                    vec_width=1, dtype=fx.Int32))
                probe_mask = fx.Int64(fx.rocdl.ballot(
                    T.i64, (probe_sid & 0x00FFFFFF) < n_tokens))
                m_active = fx.math.ctpop(probe_mask.to(fx.Int32) & 255).to(fx.Int32) * 16
            else:
                # the sort ends each expert's rows with the sentinel, so padding is a row suffix of the m-tile: one
                # scalar load (lgkmcnt, not vmcnt) per 16-row group leader counts the valid rows in 16-row groups
                m_active = fx.Int32(0)
                for g in range_constexpr(BM // 16):
                    sid_g = fx.Int32(
                        buffer_ops.buffer_load(
                            ids_rsrc,
                            m_base + g * 16,
                            vec_width=1,
                            dtype=fx.Int32,
                            is_scalar=True,
                        )
                    )
                    m_active = m_active + (
                        (sid_g & fx.Int32(0x00FFFFFF)) < n_tokens
                    ).select(fx.Int32(16), fx.Int32(0))

            sw_rsrc = buffer_ops.create_buffer_resource(
                sorted_weights,
                max_size=False,
                num_records_bytes=num_m_blocks * (BM * 4),
            )
            a_rsrc = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=num_m_blocks * (BM * K_BYTES)
            )
            b_rsrc = buffer_ops.create_buffer_resource(
                W2,
                max_size=False,
                num_records_bytes=W2_BYTES,
                base_byte_offset=fx.Int64(expert) * W2_BYTES,
            )
            # padded sorted rows (tok == n_tokens) fall outside the resource: dropped
            out_rows = n_tokens * fx.Int32(topk)
            out_rsrc = buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=out_rows * OUT_ROW_BYTES
            )

            # ---- A rows (sorted, contiguous): row * K_BYTES + swizzled col ----
            def _a_offsets(half):
                offs = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    grow = m_base + half * LDS_BLOCK_M + row
                    offs.append(grow * fx.Int32(K_BYTES) + _swizzled_col(row, col))
                return offs

            # padding rows (row_in_tile >= m_active) get an OOB offset: no memory access, zeros, still counted in vmcnt
            a_oob = num_m_blocks * fx.Int32(BM * K_BYTES)  # == a_rsrc records

            def _oob_wrap(offs, half):
                out = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    row_in_tile = fx.Int32(half * LDS_BLOCK_M) + row
                    out.append((row_in_tile >= m_active).select(a_oob, offs[rnd]))
                return out

            gl_off_a0 = _oob_wrap(_a_offsets(0), 0)
            gl_off_a1 = _oob_wrap(_a_offsets(1), 1)

            # ---- W2: expert row n*256 + hf*128 + r, 16-row blocks of KW_BYTES*16 B (a8 preshuffle inside);
            # the n-tile goes in soffset (B_TILE_BYTES per tile) ----
            e_row0 = expert * fx.Int32(H)

            def _b_offsets(hf):
                offs = []
                for rnd in range_constexpr(N_TILES_B):
                    row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id // 8) * 16
                    wrow = fx.Int32(hf * LDS_BLOCK_N + row)
                    off = (wrow // fx.Int32(16)) * fx.Int32(KW_BYTES * 16) + fx.Int32(
                        (row % 16) * 16
                        + (col // 64) * 1024
                        + ((col % 64) // 16) * 256
                        + (col % 16)
                    )
                    offs.append(off)
                return offs

            gl_off_b0 = _b_offsets(0)
            gl_off_b1 = _b_offsets(1)
            A_K_STEP = BLOCK_K_BYTES
            B_K_STEP = BLOCK_K_BYTES * 16

            mfma = Mfma16x16x128Fp4(N_TILES_A, N_TILES_B)
            sb_index = lambda j, hf: (j // 2, j % 2)  # noqa: E731  (32-row groups)

            scale_gather = ScaleGatherMoE(
                A_scale,
                W2_scale,
                K,
                lane_id,
                wave_id,
                _scale_base_ptr,
                num_m_blocks * (BM * SC_COLS),
                W2_SC_BYTES,
                A_WAVE_GROUPS,
                A_HALF_GROUPS,
                B_WAVE_GROUPS,
                B_HALF_GROUPS,
            )

            def _set_scale_tile(nt):
                scale_gather.set_wave_base(m_base, e_row0 + _n_glob(nt) * fx.Int32(BN))

            a_scale_ld = ScaleLoaderLDS(
                N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
            )
            b_scale_ld = ScaleLoaderLDS(
                N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
            )

            def _slot(nt, g):
                return (fx.Int32(nt) * fx.Int32(SCALE_BLOCKS) + fx.Int32(g)) % fx.Int32(
                    _SCALE_SLOTS2
                )

            a0_g2s = G2SLoaderAsm(a_rsrc, gl_off_a0, N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, gl_off_a1, N_TILES_A, wave_id)
            b0_g2s = G2SLoaderAsm(b_rsrc, gl_off_b0, N_TILES_B, wave_id)
            b1_g2s = G2SLoaderAsm(b_rsrc, gl_off_b1, N_TILES_B, wave_id)
            for ld in (a0_g2s, a1_g2s, b0_g2s, b1_g2s):
                ld.set_wave_base(_base_ptr)
            a_s2r = S2RLoader128B(wave_i, N_TILES_A)
            b_s2r = S2RLoader128B(wave_j, N_TILES_B)

            # ---- output rows / weights of the 4 rows this lane writes ----
            def _orow(sid):
                tok = sid & fx.Int32(0x00FFFFFF)  # padded: tok == n_tokens -> OOB
                slot = (sid >> 24) & fx.Int32(0xFF)
                return tok * fx.Int32(topk) + slot

            # routing weights of the lane's rows (row half h, tile ti, r16). The 32 output-row offsets are not held
            # across the main loop: they spilled, and each scratch reload before a store forced vmcnt(0) (stores count
            # in vmcnt).
            wave_row0 = m_base + wave_i * (N_TILES_A * 16)
            row_w = []
            for h in range_constexpr(2):
                for ti in range_constexpr(N_TILES_A):
                    row = wave_row0 + fx.Int32(h * LDS_BLOCK_M + ti * 16) + r16
                    row_w.append(
                        fx.Float32(
                            buffer_ops.buffer_load(
                                sw_rsrc, row, vec_width=1, dtype=fx.Float32
                            )
                        )
                    )
            # the m-tile's sorted ids go to LDS before the prologue: a global load in each epilogue would wait (in-order
            # vmcnt) behind every inline-asm A / W2 prefetch in flight. Two threads write each id (the same value); the
            # prologue's wait_barrier publishes them.
            sid_t = fx.thread_idx.x % BM
            sid_slot[sid_t] = fx.Int32(
                buffer_ops.buffer_load(
                    ids_rsrc, m_base + sid_t, vec_width=1, dtype=fx.Int32
                )
            )
            sid_lds = fx.Int32(fx.ptrtoint(sid_slot))
            stg_base = fx.Int32(fx.ptrtoint(_stg_ptr)) + wave_id * fx.Int32(STG_WAVE)

            def _stg_addr(row, chunk, half):
                """Staging address of (row 0..15, 16-B chunk, 8-B half); chunks are XOR-swizzled by the row so the
                tile-wise writes and the line-wise reads spread over the banks."""
                return (
                    stg_base
                    + row * fx.Int32(STG_ROW)
                    + ((chunk ^ (row % STG_CH)) * 16 + half * 8)
                )

            if const_expr(wait_for > 0):
                # until GEMM1 has stored all the m-tile's intermediate rows
                wait_ready_from(ready_addr, wait_for, n_split, got0)

            # ---- prologue: steps -2 and -1 of the steady-state order (B and A two steps ahead) ----
            # g0 B(0) a0(0) a1(0) g1 B(1) a0(1) g2 a1(1); scale block b is read at step b - 1
            _set_scale_tile(fx.Int32(0))
            scale_gather.gather(0, _slot(0, 0))
            b_soff0 = _n_glob(fx.Int32(0)) * fx.Int32(B_TILE_BYTES)
            b0_g2s.load(b_st[0][0], b_soff0 + fx.Int32(0 * B_K_STEP))
            b1_g2s.load(b_st[0][1], b_soff0 + fx.Int32(0 * B_K_STEP))
            a0_g2s.load(a_cur0, fx.Int32(0 * A_K_STEP))
            a1_g2s.load(a_cur1, fx.Int32(0 * A_K_STEP))
            scale_gather.gather(1, _slot(0, 1))
            b0_g2s.load(b_st[1][0], b_soff0 + fx.Int32(1 * B_K_STEP))
            b1_g2s.load(b_st[1][1], b_soff0 + fx.Int32(1 * B_K_STEP))
            a0_g2s.load(a_next0, fx.Int32(1 * A_K_STEP))
            scale_gather.gather(2, _slot(0, 2))
            a1_g2s.load(a_next1, fx.Int32(1 * A_K_STEP))

            # g0, B(0) and a0(0) landed: a1(0), g1, B(1), a0(1), g2, a1(1) may fly
            wait_barrier(NA + 1 + 2 * NB + NA + 1 + NA)
            a0_frag = a_s2r.load(a_cur0)
            b0_frag = b_s2r.load(b_st[0][0], preshuffled=True)
            b1_frag = b_s2r.load(b_st[0][1], preshuffled=True)
            sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0, 0))
            sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0, 0))
            sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)

            # Step kc issues B(kc+2) (b0, b1), then after the MID wait a0(kc+2), the scale gather and a1(kc+2), into
            # stage (step % 2). The top wait lands a1(kc), the MID wait a0(kc+1). The counts take loads only: stores
            # retire out of order with loads, so counting the epilogue's stores races.
            def _top_vmcnt(kc):
                return 2 * NB + 2 * NA + 1  # one gather per step

            def _mid_vmcnt(kc):
                # exact: step kc-1's gather and a1(kc+1), then B(kc+2), are the loads younger than a0(kc+1); a larger
                # count would release with a0(kc+1) in flight and c10f's s2r would read stale LDS
                return 2 * NB + NA + 1

            def _read_scale_thunks(nt, kc_next, holder):
                # the next step's scales: block kc_next of this tile or block 0 of the next (Python ints only: an ``if``
                # here would trace to an scf.if and lose the binding)
                s = _slot(nt + fx.Int32(kc_next // K_ITERS), kc_next % K_ITERS)

                def _r(dst, ld, half, _s=s):
                    holder[dst] = ld.read_half(_s, half)

                return [
                    lambda: _r(0, a_scale_ld, 0),
                    lambda: _r(1, a_scale_ld, 1),
                    lambda: _r(2, b_scale_ld, 0),
                    lambda: _r(3, b_scale_ld, 1),
                ]

            def _one_step(nt, kc, f, a0f, b0f, b1f_in, sc, accs, abufs):
                """One K-step: constexpr ``kc`` of n-tile ``nt``; the parity of ``f``, the iteration's constexpr flat
                step, picks the B stage."""
                k2 = f % 2
                ac0, ac1, an0, an1 = abufs
                bst = b_st[f % B_STAGES]  # consumed this step, refilled for step f + 2
                bnx = b_st[(f + 1) % B_STAGES]  # step f + 1's halves, read this step
                saR0, saR1, sbC0, sbC1 = sc
                c00f, c01f, c10f, c11f = accs
                zero_acc = kc == 0

                _a1 = [None] * NA
                _a0n = [None] * NA
                _b0n = [None] * NB
                _b1n = [None] * NB
                # A and B of step kc + 2 (B of tile nt + 1 from kc = K_ITERS - 2 on)
                ka = (kc + 2) % K_ITERS
                kb = (kc + B_STAGES) % K_ITERS
                nt_b = nt + fx.Int32((kc + B_STAGES) // K_ITERS)
                a_off = fx.Int32(ka * A_K_STEP)
                b_off = _n_glob(nt_b) * fx.Int32(B_TILE_BYTES) + fx.Int32(kb * B_K_STEP)

                _scn = [None, None, None, None]
                _rd_scn = _read_scale_thunks(nt, kc + 1, _scn)
                # Gather scales three steps ahead, crossing n-tiles as needed.
                future_block = kc + 3
                _g_blk = future_block % SCALE_BLOCKS
                _g_slot = _slot(nt + fx.Int32(future_block // SCALE_BLOCKS), _g_blk)
                _sc_gather = [lambda: scale_gather.gather(_g_blk, _g_slot)]

                wait_barrier(_top_vmcnt(kc))
                il = (
                    _riffle(
                        _g2s_thunks(b0_g2s, bst[0], b_off, NB),
                        _s2r_thunks(a_s2r, ac1, _a1, NA, False),
                    )
                    + _rd_scn[:2]
                )
                c00f = mfma.call(
                    a0f,
                    b0f,
                    c00f,
                    saR0,
                    sbC0,
                    0,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                # two G2S loaders are never riffled: each one's steps share m0 (s_mov, then s_add per step), so
                # interleaving them corrupts the LDS destinations. a0's loads go with c10f.
                il = _g2s_thunks(b1_g2s, bst[1], b_off, NB) + _rd_scn[2:]
                c01f = mfma.call(
                    a0f,
                    b1f_in,
                    c01f,
                    saR0,
                    sbC1,
                    1,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                a1f = _a1

                wait_barrier(_mid_vmcnt(kc))
                # a0(kc+2) may overwrite ac0 here: its last reads retired before the top barrier, and the s2r reads
                # below take an0. Concatenated, not riffled.
                il = (
                    _g2s_thunks(a0_g2s, ac0, a_off, NA)
                    + _s2r_thunks(a_s2r, an0, _a0n, NA, False)
                    + _sc_gather
                )
                c10f = mfma.call(
                    a1f,
                    b0f,
                    c10f,
                    saR1,
                    sbC0,
                    0,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                a0nf = _a0n
                il = _riffle(
                    _g2s_thunks(a1_g2s, ac1, a_off, NA),
                    _s2r_thunks(b_s2r, bnx[0], _b0n, NB, True)
                    + _s2r_thunks(b_s2r, bnx[1], _b1n, NB, True),
                )
                c11f = mfma.call(
                    a1f,
                    b1f_in,
                    c11f,
                    saR1,
                    sbC1,
                    1,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                sc_next = (_scn[0], _scn[1], _scn[2], _scn[3])
                return (
                    a0nf,
                    _b0n,
                    _b1n,
                    sc_next,
                    (c00f, c01f, c10f, c11f),
                    (an0, an1, ac0, ac1),
                )

            def _stage(accs, h, ti):
                """Routing-weighted BF16 of the 16 rows (half ``h``, a-tile ``ti``) x the wave's 128 columns into its
                staging buffer, 8 B per lane per tile; one wave's LDS ops run in order, so the writes follow the last
                flush's reads."""
                w = row_w[h * NA + ti]
                for hf in range_constexpr(2):
                    cq = accs[h * 2 + hf]
                    for j in range_constexpr(NB):
                        v = Vec(cq[mfma.idx(ti, j)])
                        d0 = _bf16x2(fx.Float32(v[0]) * w, fx.Float32(v[1]) * w)
                        d1 = _bf16x2(fx.Float32(v[2]) * w, fx.Float32(v[3]) * w)
                        # cols hf*128 + j*16 + 4*g4 of the wave's 128 -> chunk
                        chunk = fx.Int32(hf * 8 + j * 2) + g4 // 2
                        _lds_store_vec(_v2i32(d0, d1), _stg_addr(r16, chunk, g4 % 2), 2)

            def _flush(nt, h, ti, sids):
                """The staged 16 rows -> full 128-B output lines, 8 per store (non-temporal): a lane takes row
                4*(lane//16) + k, segment (lane//8)%2, 16 B at chunk lane%8. ``sids``: the group's 4 sorted ids."""
                col_wave = (_n_glob(nt) * fx.Int32(BN) + wave_j * (N_TILES_B * 16)) * 2
                hf_l = (lane_id // 8) % 2
                ch_l = lane_id % 8
                _asm_void([], "s_waitcnt lgkmcnt(0)", "")
                # the group's 4 reads, then its 4 stores: one LDS drain per group, not one lgkmcnt(0) per store
                datas = []
                for k in range_constexpr(N_FLUSH):
                    row = (lane_id // 16) * 4 + fx.Int32(k)
                    datas.append(_lds_load_vec(_stg_addr(row, hf_l * 8 + ch_l, 0), 4))
                for k in range_constexpr(N_FLUSH):
                    buffer_ops.buffer_store(
                        datas[k],
                        out_rsrc,
                        _orow(sids[k]) * fx.Int32(OUT_ROW_BYTES)
                        + col_wave
                        + hf_l * fx.Int32(LDS_BLOCK_N * 2)
                        + ch_l * fx.Int32(16),
                        offset_is_bytes=True,
                        cache_modifier=_STORE_CPOL,
                    )

            @flyc.jit
            def _epilogue(nt, accs, act):
                # Aligned 4-id LDS reads cover the lane's rows (lgkmcnt only).
                sid_row = wave_i * (N_TILES_A * 16) + (lane_id // 16) * 4
                sids_all = []
                for h in range_constexpr(act):
                    for ti in range_constexpr(NA):
                        sids_all.append(
                            Vec(
                                _lds_load_vec(
                                    sid_lds
                                    + (sid_row + fx.Int32(h * LDS_BLOCK_M + ti * 16))
                                    * 4,
                                    4,
                                )
                            )
                        )
                for h in range_constexpr(act):
                    for ti in range_constexpr(NA):
                        if const_expr(group_skip):
                            # wave-uniform branch: the staging is wave-private and the group holds no CTA barrier
                            if m_active > wave_i * (NA * 16) + h * LDS_BLOCK_M + ti * 16:
                                _stage(accs, h, ti)
                                _flush(
                                    nt,
                                    h,
                                    ti,
                                    sids_all[h * NA + ti],
                                )
                        else:
                            _stage(accs, h, ti)
                            _flush(
                                nt,
                                h,
                                ti,
                                sids_all[h * NA + ti],
                            )

            abufs0 = (a_cur0, a_cur1, a_next0, a_next1)
            n_a = 2 * NA
            n_b = 2 * NB
            n_sc = 2 * (NA // 2) + 2 * (NB // 2)
            _R = fx.as_ir_value

            def _flat_sc(sc):
                return [_R(v) for part in sc for v in part]

            def _unflat_sc(flat):
                sizes = (NA // 2, NA // 2, NB // 2, NB // 2)
                out, o = [], 0
                for n in sizes:
                    out.append(list(flat[o : o + n]))
                    o += n
                return tuple(out)

            init_state = (
                _flat_frag(a0_frag)
                + _flat_frag(b0_frag)
                + _flat_frag(b1_frag)
                + _flat_sc(sc0)
            )
            def _iteration(pair_idx, state, act):
                nt0 = fx.Int32(pair_idx) * fx.Int32(2)  # first n-tile of the iteration
                off = 0
                a0f = _unflat_frag(state[off : off + n_a], NA)
                off += n_a
                b0f = _unflat_frag(state[off : off + n_b], NB)
                off += n_b
                b1f = _unflat_frag(state[off : off + n_b], NB)
                off += n_b
                sc = _unflat_sc(state[off : off + n_sc])
                abufs = abufs0  # six rotations per iteration: back to the start
                for f in range_constexpr(FLAT):
                    nt = nt0 + fx.Int32(f // K_ITERS)
                    kc = f % K_ITERS
                    if const_expr(kc == 0):
                        accs = (
                            [None] * N_ACCUMS,
                            [None] * N_ACCUMS,
                            [None] * N_ACCUMS,
                            [None] * N_ACCUMS,
                        )
                        # K = 768: every gather of n-tile nt is for n-tile nt + 1
                        _set_scale_tile(nt + fx.Int32(1))
                    a0f, b0f, b1f, sc, accs, abufs = _one_step(
                        nt, kc, f, a0f, b0f, b1f, sc, accs, abufs
                    )
                    if const_expr(kc == K_ITERS - 1):
                        _epilogue(nt, accs, act)
                return _flat_frag(a0f) + _flat_frag(b0f) + _flat_frag(b1f) + _flat_sc(sc)

            # m_active is workgroup-uniform; both loops issue every load, MFMA and wait, the short one omits only row
            # half 1's staging and its out-of-bounds stores
            if const_expr(pad_skip):
                if m_active > fx.Int32(LDS_BLOCK_M):
                    for pair_idx, state in range(  # type: ignore[call-overload]
                        0, NT // 2, 1, init=init_state
                    ):
                        state = yield _iteration(pair_idx, state, 2)
                else:
                    for pair_idx, state in range(  # type: ignore[call-overload]
                        0, NT // 2, 1, init=init_state
                    ):
                        state = yield _iteration(pair_idx, state, 1)
            else:
                for pair_idx, state in range(  # type: ignore[call-overload]
                    0, NT // 2, 1, init=init_state
                ):
                    state = yield _iteration(pair_idx, state, 2)
            # the redundant loads of the tile after the last one must land before the LDS is handed on
            wait_barrier(0)

    @flyc.kernel(name="kernel_gemm2_mxfp4_h6144_i768_bm128_epilogue_groups" + probe_suffix)
    def kernel_gemm2_mxfp4_h6144_i768_bm128(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile(
            lds.all_lds.ptr,
            lds.scale_lds.ptr,
            lds.stage_lds.ptr,
            lds.sids.ptr,
            fx.block_idx.x,
            A,
            W2,
            OUT,
            A_scale,
            W2_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            n_tokens,
            num_m_blocks,
            fx.Int64(0),
        )

    @flyc.jit
    def launch_gemm2_mxfp4_h6144_i768_bm128_epilogue_groups(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm2_mxfp4_h6144_i768_bm128(
            A,
            W2,
            OUT,
            A_scale,
            W2_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            n_tokens,
            num_m_blocks,
            grid_size,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid_size, 1, 1), block=(256, 1, 1), stream=stream)

    launch_gemm2_mxfp4_h6144_i768_bm128_epilogue_groups.kernel_name = "kernel_gemm2_mxfp4_h6144_i768_bm128_epilogue_groups" + probe_suffix
    launch_gemm2_mxfp4_h6144_i768_bm128_epilogue_groups.tile = tile
    launch_gemm2_mxfp4_h6144_i768_bm128_epilogue_groups.lds_bytes = (
        LDS_TILES_BYTES, SCALE_LDS_BYTES2, STG_LDS_BYTES, BM
    )
    return launch_gemm2_mxfp4_h6144_i768_bm128_epilogue_groups
