# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""gfx950 MXFP8 GEMM with double-buffered LDS staging for both operands.

Adapted from AITER's eight-wave FP8 pipeline (PR #4714) and the vendored
MXFP8 scaled-MFMA implementation. Supports 2x2 and 2x4 wave layouts, per-32
E8M0 scales, BF16/FP16 output, and FP32 split-K partials.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Constexpr,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec

from .mxfp8_gemm_kernel import _scale_mma_atoms


def split_row_major_2d(index, minor_extent):
    """Split a linear row-major index into (major, minor) (aiter's helper)."""
    return index // minor_extent, index % minor_extent


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle_any(num_pid_m, num_pid_n, wgm):
    """Distribute a linear workgroup grid across XCDs."""
    NUM_XCDS = 8
    wgid = fx.Int32(fx.block_idx.x)
    num_wg = num_pid_m * num_pid_n
    xcd = wgid % NUM_XCDS
    intra_xcd = wgid // NUM_XCDS
    base = num_wg // NUM_XCDS
    extra = num_wg % NUM_XCDS
    wgid_remap = xcd * base + _min(xcd, extra) + intra_xcd
    num_in_group = wgm * num_pid_n
    group_id, intra_group = split_row_major_2d(wgid_remap, num_in_group)
    first_pid_m = group_id * wgm
    group_size_m = _min(num_pid_m - first_pid_m, wgm)
    pid_n, intra_m = split_row_major_2d(intra_group, group_size_m)
    return first_pid_m + intra_m, pid_n


@flyc.jit
def launch_gemm(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_work: fx.Pointer,
    arg_slab: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    i32_grid: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    out_dtype: Constexpr[str],
    waves_per_eu: Constexpr[int],
    xcd_swizzle: Constexpr[int],
    k_batch: Constexpr[int] = 1,
    waves_m: Constexpr[int] = 2,
    waves_n: Constexpr[int] = 4,
):
    """Launch the LDS-staged kernel using the common MXFP8 GEMM ABI. ``arg_work``,
    ``arg_slab`` and ``i32_grid`` stay in the kernel's arguments unread."""
    assert waves_m == 2 and waves_n in (2, 4)
    assert tile_m in ((64, 128) if waves_n == 2 else (128, 256))
    assert tile_n in ((64, 128) if waves_n == 2 else (128, 256))
    assert tile_k == 128 and k_batch >= 1 and K % (256 * k_batch) == 0
    assert N % tile_n == 0 and out_dtype in ("bf16", "fp16")
    BM, BN, BK = tile_m, tile_n, 128
    WAVES = waves_m * waves_n
    K_TILES = K // (BK * k_batch)
    SC_DW = (K // 256) * 64  # e8m0 scale stride per 32-row/col group, dwords
    N_TILES_A = BM // (32 * waves_m)  # 16-row A tiles per wave per quadrant
    N_TILES_B = BN // (32 * waves_n)  # 16-column tiles per wave per quadrant
    N_ACC = N_TILES_A * N_TILES_B
    AG = BM // 32  # A-scale groups per CTA
    NA = BM // (16 * WAVES)  # DMA instructions per A half
    BG = BN // 32
    NB = BN // (16 * WAVES)  # g2s LDS steps per B half
    A_LDS_B = (BM // 2) * BK  # per A half-buffer
    B_LDS_B = (BN // 2) * BK
    SA_B = AG * 256  # A-scale stage bytes
    SB_B = BG * 256  # B-scale stage bytes

    @fx.struct
    class SharedStorage:
        a_cur0: fx.Array[Int8, A_LDS_B, 16]
        a_cur1: fx.Array[Int8, A_LDS_B, 16]
        a_next0: fx.Array[Int8, A_LDS_B, 16]
        a_next1: fx.Array[Int8, A_LDS_B, 16]
        b_cur0: fx.Array[Int8, B_LDS_B, 16]
        b_cur1: fx.Array[Int8, B_LDS_B, 16]
        b_next0: fx.Array[Int8, B_LDS_B, 16]
        b_next1: fx.Array[Int8, B_LDS_B, 16]
        sa_s0: fx.Array[Int8, SA_B, 16]
        sb_s0: fx.Array[Int8, SB_B, 16]
        sa_s1: fx.Array[Int8, SA_B, 16]
        sb_s1: fx.Array[Int8, SB_B, 16]

    _kname = (
        f"flydsl_mxpsh_{waves_m}x{waves_n}w_{BM}x{BN}x{BK}_F8_F8_B16"
        f"_w{waves_per_eu}_x{xcd_swizzle}_sk{k_batch}"
    )

    @flyc.kernel(name=_kname, known_block_size=[64 * WAVES, 1, 1])
    def kernel_gemm(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        arg_work: fx.Int64,
        arg_slab: fx.Int64,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_grid: fx.Int32,
    ):
        scale_atoms = _scale_mma_atoms()
        if const_expr(k_batch > 1):
            split = fx.Int32(fx.block_idx.z)
            kt0 = split * K_TILES
        else:
            kt0 = fx.Int32(0)

        tid = fx.Int32(fx.thread_idx.x)
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16
        wave_m = wave // waves_n
        wave_n = wave % waves_n
        sc_lane = lane_div_16 * 16 + lane_mod_16

        n_blocks = i32_n // BN
        if const_expr(xcd_swizzle > 0):
            pid_m, pid_n = _xcd_swizzle_any(
                (i32_m + (BM - 1)) // BM, n_blocks, xcd_swizzle
            )
        else:
            pid_m, pid_n = split_row_major_2d(fx.Int32(fx.block_idx.x), n_blocks)
        bx_m = fx.Int32(pid_m) * BM
        by_n = fx.Int32(pid_n) * BN

        # A source view, bound to the last valid M row (ragged M OOB -> 0).
        _i8g = fx.PointerType.get(
            T.i8, address_space=fx.AddressSpace.Global, alignment=16
        )
        a_nrec = fx.Int64(i32_m) * fx.Int64(K)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(fx.inttoptr(_i8g, arg_a), fx.make_layout(65536 * K, 1))
            ),
            max_size=False,
            num_records_bytes=a_nrec,
        )
        a_div = fx.logical_divide(a_flat, fx.make_layout(1, 1))
        b_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(fx.inttoptr(_i8g, arg_b), fx.make_layout(65536 * K, 1))
            ),
            max_size=False,
            num_records_bytes=fx.Int64(N * K),
        )
        b_div = fx.logical_divide(b_flat, fx.make_layout(1, 1))

        # Scale DMA uses byte offsets and bounds, including padded A rows.
        _sc_layout = fx.make_layout(1 << 30, 1)
        a_sc_nrec = fx.Int64((i32_m + 31) // 32) * fx.Int64(SC_DW) * fx.Int64(4)
        b_sc_nrec = fx.Int64((N // 32) * SC_DW * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i8g, arg_scale_a), _sc_layout)),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i8g, arg_scale_b), _sc_layout)),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        # data halves indexed [tile_parity][row_half]; scale stages [chunk_parity]
        a_i8 = [
            [
                fx.recast_iter(_i8s, lds.a_cur0.ptr),
                fx.recast_iter(_i8s, lds.a_cur1.ptr),
            ],
            [
                fx.recast_iter(_i8s, lds.a_next0.ptr),
                fx.recast_iter(_i8s, lds.a_next1.ptr),
            ],
        ]
        b_i8 = [
            [
                fx.recast_iter(_i8s, lds.b_cur0.ptr),
                fx.recast_iter(_i8s, lds.b_cur1.ptr),
            ],
            [
                fx.recast_iter(_i8s, lds.b_next0.ptr),
                fx.recast_iter(_i8s, lds.b_next1.ptr),
            ],
        ]
        a_i32 = [
            [fx.recast_iter(Int32, p.ptr) for p in h]
            for h in ((lds.a_cur0, lds.a_cur1), (lds.a_next0, lds.a_next1))
        ]
        b_i32 = [
            [fx.recast_iter(Int32, p.ptr) for p in h]
            for h in ((lds.b_cur0, lds.b_cur1), (lds.b_next0, lds.b_next1))
        ]
        sc_i8 = [
            (fx.recast_iter(_i8s, lds.sa_s0.ptr), fx.recast_iter(_i8s, lds.sb_s0.ptr)),
            (fx.recast_iter(_i8s, lds.sa_s1.ptr), fx.recast_iter(_i8s, lds.sb_s1.ptr)),
        ]
        sc_i32 = [
            (
                fx.recast_iter(Int32, lds.sa_s0.ptr),
                fx.recast_iter(Int32, lds.sb_s0.ptr),
            ),
            (
                fx.recast_iter(Int32, lds.sa_s1.ptr),
                fx.recast_iter(Int32, lds.sb_s1.ptr),
            ),
        ]

        sa_stride = fx.Int32(fx.ptrtoint(lds.sa_s1.ptr)) - fx.Int32(
            fx.ptrtoint(lds.sa_s0.ptr)
        )
        sb_stride = fx.Int32(fx.ptrtoint(lds.sb_s1.ptr)) - fx.Int32(
            fx.ptrtoint(lds.sb_s0.ptr)
        )

        # NA/NB count A/B DMA instructions per wave. Wait only for loads
        # older than the following batches: prologue NA+NB, steady NA+2*NB,
        # penultimate tile NA. Scale loads precede those batches.
        # The inline-asm barrier must also drain LDS operations explicitly.
        def _wait_barrier(cnt):
            _llvm.inline_asm(
                None,
                [],
                f"s_waitcnt vmcnt({cnt}) lgkmcnt(0)\ns_barrier",
                "",
                has_side_effects=True,
            )

        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)

        def _g2s_a(dst_i8, half, kt):
            gl_base = (bx_m + half * (BM // 2)) * fx.Int32(K) + (kt0 + kt) * BK
            for st in range_constexpr(NA):
                row = lane // 8 + wave * 8 + st * (8 * WAVES)
                col = (lane % 8) * 16
                scol = col ^ (((row // 2) % 8) * 16)  # swizzle_128
                gmem = gl_base + row * fx.Int32(K) + scol
                dst = fx.make_view(
                    fx.add_offset(dst_i8, wave * 1024 + st * (1024 * WAVES)),
                    fx.make_layout(1, 1),
                )
                fx.copy(dma_atom, fx.slice(a_div, (None, gmem)), dst)

        def _g2s_b(dst_i8, half, kt):
            # aiter preshuffle (16,16): byte(n,k) = (n//16)*16K + (k//64)*1024
            # + ((k%64)//16)*256 + (n%16)*16 + k%16 (== base's _bq_view map).
            gl_base = (by_n + half * (BN // 2)) * fx.Int32(K) + (kt0 + kt) * 2048
            for st in range_constexpr(NB):
                row = lane % 8 + wave * 8 + st * (8 * WAVES)
                c16 = lane // 8
                gmem = (
                    gl_base
                    + (row // 16) * (K * 16)
                    + (row % 16) * 16
                    + (c16 // 4) * 1024
                    + (c16 % 4) * 256
                )
                dst = fx.make_view(
                    fx.add_offset(dst_i8, wave * 1024 + st * (1024 * WAVES)),
                    fx.make_layout(1, 1),
                )
                fx.copy(dma_atom, fx.slice(b_div, (None, gmem)), dst)

        def _g2s_sc(stage, chunk):
            # One 256-K scale chunk = AG+BG groups x 256B. Task id
            # t = wave*2 + (lane//16): t < AG -> A-scale group t; AG <= t <
            # AG+BG -> B-scale group t-AG. The copy atom adds lane*16 to the LDS
            # dst itself, so the dst base subtracts the in-wave lane offset.
            sa_i8 = fx.add_offset(sc_i8[0][0], stage * sa_stride)
            sb_i8 = fx.add_offset(sc_i8[0][1], stage * sb_stride)
            task = wave * 2 + lane // 16
            if lane // 16 < 2:
                if task < AG:
                    src = (
                        (bx_m // 32 + task) * fx.Int32(SC_DW)
                        + (kt0 // 2 + chunk) * 64
                        + (lane % 16) * 4
                    ) * 4
                    dst = fx.make_view(
                        fx.add_offset(sa_i8, task * 256 - lane * 16),
                        fx.make_layout(1, 1),
                    )
                    fx.copy(dma_atom, fx.slice(sa_flat, (None, src)), dst)
                if task >= AG:
                    if task < AG + BG:
                        src = (
                            (by_n // 32 + (task - AG)) * fx.Int32(SC_DW)
                            + (kt0 // 2 + chunk) * 64
                            + (lane % 16) * 4
                        ) * 4
                        dst = fx.make_view(
                            fx.add_offset(sb_i8, (task - AG) * 256 - lane * 16),
                            fx.make_layout(1, 1),
                        )
                        fx.copy(dma_atom, fx.slice(sb_flat, (None, src)), dst)

        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        lds_copy32 = fx.make_copy_atom(fx.UniversalCopy32b(), Int32)

        def _read16(base_i32, off_i32):
            t = fx.make_rmem_tensor(4, Int32)
            fx.copy(
                lds_copy,
                fx.make_view(fx.add_offset(base_i32, off_i32), fx.make_layout(4, 1)),
                t,
            )
            return Vec(fx.memref_load_vec(t))

        def _read4(base_i32, off_i32):
            t = fx.make_rmem_tensor(1, Int32)
            fx.copy(
                lds_copy32,
                fx.make_view(fx.add_offset(base_i32, off_i32), fx.make_layout(1, 1)),
                t,
            )
            return Vec(fx.memref_load_vec(t))[0]

        def _a_s2r(buf_i32):
            fr = []
            for i in range_constexpr(N_TILES_A):
                row = wave_m * (N_TILES_A * 16) + i * 16 + lane_mod_16
                h = []
                for st in range_constexpr(2):
                    col = lane_div_16 * 16 + st * 64
                    off = row * 128 + (col ^ (((row // 2) % 8) * 16))
                    h.append(_read16(buf_i32, off // 4))
                t = fx.make_rmem_tensor(8, Int32)
                t.store(h[0].shuffle(h[1], list(range(8))))
                fr.append(t)
            return fr

        def _b_s2r(buf_i32):
            # Read the preshuffled layout written by _g2s_b.
            fr = []
            for j in range_constexpr(N_TILES_B):
                row = wave_n * (N_TILES_B * 16) + j * 16 + lane_mod_16
                h = []
                for st in range_constexpr(2):
                    col = lane_div_16 * 16 + st * 64
                    off = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                    h.append(_read16(buf_i32, off // 4))
                t = fx.make_rmem_tensor(8, Int32)
                t.store(h[0].shuffle(h[1], list(range(8))))
                fr.append(t)
            return fr

        def _sc_s2r(stage, k2):
            _g4 = max(1, N_TILES_A // 2)  # A-scale groups per wave per half
            sa_i32 = fx.add_offset(sc_i32[0][0], stage * sa_stride // 4)
            sb_i32 = fx.add_offset(sc_i32[0][1], stage * sb_stride // 4)
            sa0 = [
                _read4(sa_i32, (wave_m * N_TILES_A // 2 + hh) * 64 + sc_lane)
                for hh in range_constexpr(_g4)
            ]
            sa1 = [
                _read4(sa_i32, (AG // 2 + wave_m * N_TILES_A // 2 + hh) * 64 + sc_lane)
                for hh in range_constexpr(_g4)
            ]
            if const_expr(N_TILES_A == 1):
                shift_a = (wave_m % 2) * 8
                sa0 = [v.shrui(shift_a) for v in sa0]
                sa1 = [v.shrui(shift_a) for v in sa1]
            sb_group = wave_n * N_TILES_B // 2
            sb0 = _read4(sb_i32, sb_group * 64 + sc_lane)
            sb1 = _read4(sb_i32, (BG // 2 + sb_group) * 64 + sc_lane)
            if const_expr(N_TILES_B == 1):
                # Adjacent waves select different 16-column halves of one scale group.
                shift = (wave_n % 2) * 8
                sb0 = sb0.shrui(shift)
                sb1 = sb1.shrui(shift)
            if const_expr(k2):
                # odd K-tile: the 256-K word's high 16b hold this tile's bytes.
                sa0 = [v.shrui(fx.Int32(16)) for v in sa0]
                sa1 = [v.shrui(fx.Int32(16)) for v in sa1]
                sb0 = sb0.shrui(fx.Int32(16))
                sb1 = sb1.shrui(fx.Int32(16))
            return sa0, sa1, sb0, sb1

        def _burst(accs_q, a_frag, b_frag, sa2, sb1, brackets=True):
            cfs = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(N_ACC)]
            for t in range_constexpr(N_ACC):
                cfs[t].store(Vec(accs_q[t]))
            if const_expr(brackets):
                rocdl.s_setprio(1)
            for i in range_constexpr(N_TILES_A):
                for j in range_constexpr(N_TILES_B):
                    cf = cfs[i * N_TILES_B + j]
                    fx.gemm(
                        scale_atoms[(i % 2, j % 2)],
                        cf,
                        a_frag[i],
                        b_frag[j],
                        cf,
                        scale_a=sa2[i // 2],
                        scale_b=sb1,
                    )
            if const_expr(brackets):
                rocdl.s_setprio(0)
                rocdl.s_barrier()
            return [cfs[t].load().ir_value() for t in range_constexpr(N_ACC)]

        zero = Vec.filled(4, 0.0, Float32).ir_value()
        accs = [[[zero] * N_ACC for _ in range(2)] for _ in range(2)]

        # ---- prologue: fill tile 0 (all halves + scale chunk 0), then tile 1
        _g2s_b(b_i8[0][0], 0, 0)
        _g2s_a(a_i8[0][0], 0, 0)
        _g2s_sc(0, 0)
        _g2s_b(b_i8[0][1], 1, 0)
        _g2s_a(a_i8[0][1], 1, 0)
        if wave_m == 1:
            rocdl.s_barrier()  # Stagger the two M wave groups.
        _wait_barrier(NB + NA)  # B0(0), A0(0), SC(0) landed
        _g2s_b(b_i8[1][0], 0, 1)
        _g2s_a(a_i8[1][0], 0, 1)
        _g2s_b(b_i8[1][1], 1, 1)
        if const_expr(K_TILES == 2):
            _wait_barrier(0)
        else:
            _wait_barrier(NB + NA + NB)  # B1(0), A1(0) landed

        # ---- steady state: tile k computes while k+1/k+2 stream in
        if const_expr(K_TILES > 2):
            init = [
                accs[i][j][t] for i in range(2) for j in range(2) for t in range(N_ACC)
            ]
            _k_end = fx.Index(K_TILES - 2)
            for iv, state in range(fx.Index(0), _k_end, fx.Index(2), init=init):
                accs = [
                    [
                        list(state[(i * 2 + j) * N_ACC : (i * 2 + j + 1) * N_ACC])
                        for j in range(2)
                    ]
                    for i in range(2)
                ]
                for parity in range_constexpr(2):
                    k = fx.Int32(iv) + fx.Int32(parity)
                    p = parity
                    sc_p = (k // 2) % 2
                    k2 = parity
                    # phase 1: read B0/A0/scales (tile k), issue A1(k+1) [+ SC(k+2)]
                    b0f = _b_s2r(b_i32[p][0])
                    a0f = _a_s2r(a_i32[p][0])
                    sa0, sa1, sb0, sb1 = _sc_s2r(sc_p, k2)
                    _g2s_a(a_i8[(parity + 1) % 2][1], 1, k + 1)
                    if const_expr(parity == 0):
                        _g2s_sc(((k // 2) + 1) % 2, (k + 2) // 2)
                    rocdl.s_barrier()
                    # phase 2
                    accs[0][0] = _burst(accs[0][0], a0f, b0f, sa0, sb0)
                    # phase 3: read B1(k), issue B0(k+2) into the just-freed buffer
                    b1f = _b_s2r(b_i32[p][1])
                    _g2s_b(b_i8[p][0], 0, k + 2)
                    rocdl.s_barrier()
                    # phase 4
                    accs[0][1] = _burst(accs[0][1], a0f, b1f, sa0, sb1)
                    # phase 5: read A1(k), issue A0(k+2)
                    a1f = _a_s2r(a_i32[p][1])
                    _g2s_a(a_i8[p][0], 0, k + 2)
                    rocdl.s_barrier()
                    # phase 6
                    accs[1][0] = _burst(accs[1][0], a1f, b0f, sa1, sb0)
                    # phase 7: issue B1(k+2); wait everything through phase 1 landed
                    _g2s_b(b_i8[p][1], 1, k + 2)
                    _wait_barrier(NB + NA + NB)
                    # phase 8
                    accs[1][1] = _burst(accs[1][1], a1f, b1f, sa1, sb1)
                results = yield [
                    accs[i][j][t]
                    for i in range(2)
                    for j in range(2)
                    for t in range(N_ACC)
                ]
            accs = [
                [
                    list(results[(i * 2 + j) * N_ACC : (i * 2 + j + 1) * N_ACC])
                    for j in range(2)
                ]
                for i in range(2)
            ]

        # ---- peel k = K_TILES-2 (even): no k+2 issues; final A1 tile here
        sc_p = ((K_TILES - 2) // 2) % 2
        b0f = _b_s2r(b_i32[0][0])
        a0f = _a_s2r(a_i32[0][0])
        sa0, sa1, sb0, sb1 = _sc_s2r(sc_p, 0)
        sa0s = [v.shrui(fx.Int32(16)) for v in sa0]  # tile K-1's halves (k2=1)
        sa1s = [v.shrui(fx.Int32(16)) for v in sa1]
        sb0s = sb0.shrui(fx.Int32(16))
        sb1s = sb1.shrui(fx.Int32(16))
        rocdl.s_barrier()
        accs[0][0] = _burst(accs[0][0], a0f, b0f, sa0, sb0)
        b1f = _b_s2r(b_i32[0][1])
        rocdl.s_barrier()
        accs[0][1] = _burst(accs[0][1], a0f, b1f, sa0, sb1)
        a1f = _a_s2r(a_i32[0][1])
        _g2s_a(a_i8[1][1], 1, K_TILES - 1)
        rocdl.s_barrier()
        accs[1][0] = _burst(accs[1][0], a1f, b0f, sa1, sb0)
        _wait_barrier(NA)  # B0(K-1) landed; only the A1(K-1) batch in flight
        b0f = _b_s2r(b_i32[1][0])  # tile K-1's B0 (replaces tile K-2's)
        rocdl.s_barrier()
        accs[1][1] = _burst(accs[1][1], a1f, b1f, sa1, sb1)

        # ---- peel k = K_TILES-1: drain, then compute
        a0f = _a_s2r(a_i32[1][0])
        _wait_barrier(0)
        accs[0][0] = _burst(accs[0][0], a0f, b0f, sa0s, sb0s)
        b1f = _b_s2r(b_i32[1][1])
        rocdl.s_barrier()
        accs[0][1] = _burst(accs[0][1], a0f, b1f, sa0s, sb1s)
        a1f = _a_s2r(a_i32[1][1])
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        accs[1][0] = _burst(accs[1][0], a1f, b0f, sa1s, sb0s, brackets=False)
        accs[1][1] = _burst(accs[1][1], a1f, b1f, sa1s, sb1s, brackets=False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        c_stride = N

        def _store_tile(store_elem, c_tile_addr, stride, rows_wg):
            """The wave's accumulators to ``c_tile_addr`` (this tile's first row),
            ``rows_wg`` rows of ``stride`` elements; lane (q, r) holds rows 4q .. 4q+3 of
            column r of each 16x16 tile."""
            store_bytes = 4 if store_elem == Float32 else 2
            c_nrec = fx.Int64(rows_wg) * fx.Int64(stride) * fx.Int64(store_bytes)
            c_ptr_ty = fx.PointerType.get(
                store_elem.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=store_bytes,
            )
            c_flat = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(
                        fx.make_view(
                            fx.inttoptr(c_ptr_ty, c_tile_addr), fx.make_layout(1 << 28, 1)
                        )
                    ),
                    max_size=False,
                    num_records_bytes=c_nrec,
                ),
                fx.make_layout(1, 1),
            )
            if const_expr(store_elem == Float32):
                c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), store_elem)
            else:
                c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), store_elem)
            c_rstride = fx.Int32(stride)
            for hm in range_constexpr(2):
                for i in range_constexpr(N_TILES_A):
                    row_local = (
                        hm * (BM // 2)
                        + wave_m * (BM // (2 * waves_m))
                        + i * 16
                        + lane_div_16 * 4
                    )
                    for hn in range_constexpr(2):
                        for j in range_constexpr(N_TILES_B):
                            col = (
                                hn * (BN // 2)
                                + wave_n * (BN // (2 * waves_n))
                                + j * 16
                                + lane_mod_16
                            )
                            acc = Vec(accs[hm][hn][i * N_TILES_B + j]).to(store_elem)
                            for ii in range_constexpr(4):
                                cf = fx.make_rmem_tensor(1, store_elem)
                                cf.store(Vec.from_elements([acc[ii]], store_elem))
                                fx.copy(
                                    c_copy,
                                    cf,
                                    c_flat[None, (row_local + ii) * c_rstride + col],
                                )

        out_elem = BFloat16 if const_expr(out_dtype == "bf16") else Float16
        _rows_rem = fx.Index(i32_m) - fx.Index(bx_m)
        _rows_wg = (_rows_rem < fx.Index(BM)).select(_rows_rem, fx.Index(BM))
        out_tile_addr = (
            arg_c + (fx.Int64(bx_m) * fx.Int64(c_stride) + fx.Int64(by_n)) * fx.Int64(2)
        )
        if const_expr(k_batch > 1):
            slab_addr = (
                arg_c
                + fx.Int64(split) * fx.Int64(i32_m) * fx.Int64(N) * 4
                + (fx.Int64(bx_m) * fx.Int64(N) + fx.Int64(by_n)) * fx.Int64(4)
            )
            _store_tile(Float32, slab_addr, N, _rows_wg)
        else:
            _store_tile(out_elem, out_tile_addr, c_stride, _rows_wg)

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    work_addr = fx.Int64(fx.ptrtoint(arg_work))
    slab_addr = fx.Int64(fx.ptrtoint(arg_slab))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = ((i32_m + (BM - 1)) // BM) * (i32_n // BN)
    kernel_gemm(
        c_addr,
        a_addr,
        b_addr,
        sa_addr,
        sb_addr,
        work_addr,
        slab_addr,
        i32_m,
        i32_n,
        i32_grid,
        value_attrs={"rocdl.waves_per_eu": wpe},
    ).launch(grid=(gx, 1, k_batch), block=(64 * WAVES, 1, 1), stream=stream)
