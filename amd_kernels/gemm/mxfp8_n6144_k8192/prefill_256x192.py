# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""256x192 MXFP8 GEMM, N6144 K8192, row-major codes and shuffled scales.

Four pipelined quadrant waves; each 48-column quadrant re-pairs its E8M0
words. BF16 output uses 64-column LDS blocks with the padding stores masked.
The 1281-2048 row window fits one CU round (192-256 workgroups).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T, Vector as Vec
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_4wave import (
    BLOCK_K,
    G2SLoaderAsm,
    MfmaRowStationary,
    PARENT_M,
    S2RLoader,
    _Buf,
    _LDS_BUF_BYTES,
    _N_WAVES,
    _PACK,
    _SCALE_A_REGION,
    _SCALE_B_REGION,
    _SCALE_SLOTS,
    _SCALE_SLOT_BYTES,
    _acc_read,
    _ceildiv,
    _cvt_pk_bf16,
    _divmod_nonneg,
    _g2s_thunks,
    _global_swizzle,
    _lds_ptr_t,
    _min,
    _riffle,
    _s2r_thunks,
    _uniform_i32,
    _xcd_swizzle,
    wait_barrier,
    ScaleGatherLDS as ScaleGatherBase,
)
PARENT_N = 192

class ScaleGatherLDS(ScaleGatherBase):
    def tile_group(self, a_base_tile, b_base_tile):
        """This wave's first 32-row group of a tile (wave-uniform)."""
        base_tile = fx.arith.select(
            self._is_a,
            a_base_tile + self._q * fx.Int32(128),
            b_base_tile + self._q * fx.Int32(128),
        )
        return _uniform_i32(base_tile // fx.Int32(32))

    def set_tile_voff(self, G):
        """This tile's per-lane gather offset with the K-block folded out (it rides
        the uniform soffset instead): computed once per tile, so the K loop carries
        no per-gather voffset math."""
        grp = fx.Int32(G) + self._blk
        self._voff_g = fx.as_ir_value(
            (grp * fx.Int32(self.row_i32) + self._in16 * fx.Int32(4)) * fx.Int32(4)
        )

class ScaleLoaderLDS:
    """The wave's scale dwords of a block: lane (klane, r) reads the dword of group
    (half, gi) at its (klane, mlane) position -- the layout the MFMA wants."""

    def __init__(self, n_tiles, lane_id, quarter, lds_base_ptr, region_off):
        self.n_groups = (n_tiles + 1) // _PACK
        self.n_tiles = n_tiles
        self.quarter = quarter
        self.lane_id = lane_id
        self._region_base = (
            fx.Int32(fx.ptrtoint(lds_base_ptr))
            + fx.Int32(region_off)
        )

    def read_half(self, slot, half):
        L = self.lane_id
        base = (
            self._region_base
            + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)
            + fx.Int32((L // 4) * 16 + (L % 4) * 4)
        )
        out = []
        for gi in range_constexpr(self.n_groups):
            row = self.quarter * (16 * self.n_tiles) + half * (32 * self.n_tiles) + gi * 32
            group = row // 32
            ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(base + group * 256))
            lo = fx.Int32(_llvm.LoadOp(fx.Int32.ir_type, ptr, alignment=4).result)
            if const_expr(self.n_tiles % 2 == 1):
                ptr2 = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(base + (group + 1) * 256))
                hi = fx.Int32(_llvm.LoadOp(fx.Int32.ir_type, ptr2, alignment=4).result)
                odd = (row % 32) != 0
                packed = (lo.shrui(fx.Int32(8)) & 0x00ff00ff) | ((hi & 0x00ff00ff) << 8)
                lo = odd.select(packed, lo)
            out.append(lo)

        return out

    def read(self, slot):
        return self.read_half(slot, 0), self.read_half(slot, 1)

class StoreCLds:
    """BF16 transpose for 48-column quadrants, padded to 64 columns in LDS.

    Each MFMA row writes three 16-column fragments into a 64-column block.
    A warp reads eight whole rows; the final two 8-column pieces are masked.
    Power-of-two LDS rows retain the original XOR bank-conflict swizzle.
    """

    def __init__(
        self,
        C,
        c_rows,
        c_cols,
        c_idx_fn,
        lds_base_ptr,
        wave_id,
        n_a,
        n_b,
        nt=False,
    ):
        self.c_cols = c_cols
        self.c_rows = c_rows
        self.c_idx_fn = c_idx_fn
        self.n_a = n_a
        self.n_b = n_b
        n_b = 4  # 48 useful columns, 64-column LDS rows
        self.pieces = 2 * n_b  # 16-byte pieces per block row (32 n_b bytes)
        self.rows_per_read = 64 // self.pieces  # rows one 64-lane read / store covers
        self.slabs = 16 // self.rows_per_read  # reads per block
        self.buf_bytes = 16 * 32 * n_b
        self.cpol = (
            2 if nt else 0
        )  # gfx950 buffer_store bits: 0x2 = nt (streaming, write-once output)
        self.rsrc = buffer_ops.create_buffer_resource(
            C, max_size=False, num_records_bytes=c_rows * c_cols * 2
        )
        lane = fx.thread_idx.x % 64
        g = lane // 16
        r = lane % 16
        row_b = 32 * n_b
        base = fx.Int32(fx.ptrtoint(lds_base_ptr)) + wave_id * fx.Int32(
            2 * self.buf_bytes
        )
        # write side: lane (g, r) holds columns 4 g .. 4 g + 3 of row r of each 16x16 tile
        self.w_off = [
            base
            + r * fx.Int32(row_b)
            + (((fx.Int32(2 * tj) + g // 2) ^ (r % self.pieces)) * fx.Int32(16))
            + (g % 2) * fx.Int32(8)
            for tj in range_constexpr(n_b)
        ]
        # read side: lane (r2 = lane // pieces, q = lane % pieces) takes piece q of row r2 of each slab
        q = lane % self.pieces
        r2 = lane // self.pieces
        self.r_off = [
            base
            + (r2 + h * self.rows_per_read) * fx.Int32(row_b)
            + ((q ^ (r2 % self.pieces)) * fx.Int32(16))
            for h in range_constexpr(self.slabs)
        ]
        self.st_off = (
            r2 * c_cols + q * 8
        ) * 2  # the per-lane part of the store byte offset

    @property
    def n_thunks(self):
        return self.n_a * (1 + self.slabs)

    def _write(self, c_frag, ti, tj, buf):
        v = c_frag[self.c_idx_fn(ti, tj)]
        rd = _acc_read
        d0 = _cvt_pk_bf16(rd(v, 0), rd(v, 1))
        d1 = _cvt_pk_bf16(rd(v, 2), rd(v, 3))
        vec = Vec.from_elements([fx.Int32(d0), fx.Int32(d1)], fx.Int32)
        ptr = _llvm.inttoptr(
            _lds_ptr_t(),
            fx.as_ir_value(self.w_off[tj] + fx.Int32(buf * self.buf_bytes)),
        )
        _llvm.StoreOp(fx.as_ir_value(vec), ptr, alignment=8)

    def _read_store(self, base_row, base_col, ti, h, buf):
        ptr = _llvm.inttoptr(
            _lds_ptr_t(), fx.as_ir_value(self.r_off[h] + fx.Int32(buf * self.buf_bytes))
        )
        vec4 = _ir.VectorType.get([4], fx.Int32.ir_type)
        data = Vec(_llvm.LoadOp(vec4, ptr, alignment=16).result)
        off = (
            (base_row + ti * 16 + h * self.rows_per_read) * self.c_cols + base_col
        ) * 2 + self.st_off
        buffer_ops.buffer_store(
            data, self.rsrc, off, offset_is_bytes=True, cache_modifier=self.cpol, mask=(fx.thread_idx.x % 8 < 6)
        )

    def thunks(self, c_frag, base_row, base_col):
        """n_a (1 + slabs) thunks per quadrant: the blocks through the two buffers -- W0 W1
        R0 W2 R1 ... (a wave's LDS ops run in order, so a buffer's re-write after its reads
        needs no wait); quadrants sharing the buffers must be concatenated, not
        interleaved."""
        order = [("w", 0)]
        for b in range(1, self.n_a):
            order += [("w", b), ("r", b - 1)]
        order.append(("r", self.n_a - 1))
        ts = []
        for kind, ti in order:
            if kind == "w":
                ts.append(
                    lambda ti=ti: [
                        self._write(c_frag, ti, tj, ti % 2) for tj in range(self.n_b)
                    ]
                )
            else:
                for h in range(self.slabs):
                    ts.append(
                        lambda ti=ti, h=h: self._read_store(
                            base_row, base_col, ti, h, ti % 2
                        )
                    )
        return ts

    def store(self, c_frag, base_row, base_col):
        for t in self.thunks(c_frag, base_row, base_col):
            t()

def compile_mxfp8_gemm_n6144_k8192_256x192(K: int = 8192, grid_max: int = 256):
    """Compile the row-major-code 256x192 tile at N 6144: the full o_proj (K 8192) or
    the TP4 o_proj (K 2048). ``grid_max`` 256 = persistent (one workgroup per CU); a
    larger value = one workgroup per tile (the 4-wave rows' plain grid inside DBO)."""
    MN, store_nt, tiles = None, True, (4, 3)
    use_xcd_remap = True
    N_TILES_A, N_TILES_B = tiles
    assert N_TILES_A == 4 and N_TILES_B == 3
    BLOCK_M = 64 * N_TILES_A
    BLOCK_N = 64 * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    SUB_M = PARENT_M // BLOCK_M  # sub-tiles per parent tile
    SUB_N = PARENT_N // BLOCK_N
    SUB_SPLIT = SUB_M * SUB_N
    assert K % 256 == 0
    K_ITERS = K // BLOCK_K  # even
    KB = K // 256  # scale blocks
    UNROLL = 4 if (K_ITERS - 4) % 4 == 0 else 2
    assert (K_ITERS - 4) % UNROLL == 0, K_ITERS
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert (
        LDS_BLOCK_M * BLOCK_K <= _LDS_BUF_BYTES
        and LDS_BLOCK_N * BLOCK_K <= _LDS_BUF_BYTES
    )
    # grid_max: workgroups of the launch (256 = one per CU, persistent); a larger value gives
    # every work item its own workgroup (a plain grid), which no co-resident kernel holds back
    GRID_MAX = grid_max
    name = f"flydsl_mxfp8_256x192_n6144_k{K}_nt" + ("" if grid_max == 256 else f"_g{grid_max}")
    SCALE_SLOTS = _SCALE_SLOTS
    OUT_LDS_BYTES = _N_WAVES * 2 * 16 * 32 * 4

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, 8 * _LDS_BUF_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SCALE_SLOTS * _SCALE_SLOT_BYTES, 16]
        out_lds: fx.Array[fx.Int8, OUT_LDS_BYTES, 16]

    @flyc.kernel(name=name)
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        tile_off: fx.Int32,
        n_work: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr

        def _buf(idx):
            return _Buf(_base_ptr, idx * _LDS_BUF_BYTES)

        a_cur0, a_cur1, a_next0, a_next1 = _buf(0), _buf(1), _buf(2), _buf(3)
        b_cur0, b_cur1, b_next0, b_next1 = _buf(4), _buf(5), _buf(6), _buf(7)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        if const_expr(MN is not None):
            c_m, c_n = const_expr(MN[0]), const_expr(MN[1])
        n_pid_m = _ceildiv(c_m, PARENT_M)  # the parent tiles' grid
        n_blocks = _ceildiv(c_n, PARENT_N)
        grid = _min(fx.Int32(n_work), fx.Int32(GRID_MAX))
        wg = fx.Int32(fx.block_idx.x)
        my_tiles = (fx.Int32(n_work) - wg + grid - fx.Int32(1)) // grid
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A_K_STEP = BLOCK_K
        B_K_STEP = 2 * 1024  # preshuffled (16, 16): a 16-row block spans 2 KB per 128 K

        def tile_coords(t):
            """work item -> (tile_i, tile_j) in units of this kernel's tile. The parent tiles
            are indexed in the XCD order of ``_xcd_swizzle``; a sub-tile launch keeps a
            workgroup on its XCD's parents: item ``t = 8 q + x`` is sub-tile ``q % SUB_SPLIT``
            of parent ``tile_off + 8 (q / SUB_SPLIT) + x`` (``tile_off`` a multiple of 8). A
            negative / out-of-range index lands beyond M so its stores are dropped by the
            buffer bounds."""
            if const_expr(use_xcd_remap and SUB_SPLIT > 1):
                q, x = _divmod_nonneg(t, 8)
                p_intra, sub = _divmod_nonneg(q, SUB_SPLIT)
                parent = fx.Int32(tile_off) + p_intra * 8 + x
            else:
                parent, sub = _divmod_nonneg(t, SUB_SPLIT)
                parent = fx.Int32(tile_off) + parent
            if const_expr(use_xcd_remap):
                pi, pj = _xcd_swizzle(n_pid_m, n_blocks, parent)
            else:
                pi, pj = parent // n_blocks, parent % n_blocks
            si, sj = _divmod_nonneg(sub, SUB_N)
            return pi * SUB_M + si, pj * SUB_N + sj

        def tile_bases(ti, tj):
            """the g2s bases of a tile (row-major A, preshuffled W) and its scale groups"""
            return (
                (ti * BLOCK_M) * K,
                (ti * BLOCK_M + LDS_BLOCK_M) * K,
                (tj * BLOCK_N) * K,
                (tj * BLOCK_N + LDS_BLOCK_N) * K,
            )

        mfma = MfmaRowStationary(N_TILES_A, N_TILES_B)
        _scale_base_ptr = lds.scale_lds.ptr
        scale_gather = ScaleGatherLDS(
            A_scale,
            B_scale,
            K,
            lane_id,
            wave_id,
            _scale_base_ptr,
            c_m,
            c_n,
            N_TILES_A,
            N_TILES_B,
        )
        scale_gather.set_wave_base()
        # the two scale-slot LDS addresses (slot 0 / 1) precomputed once; the
        # gather's K-block offset rides the uniform soffset inside the K loop
        _gm0_s = (
            scale_gather._wave_base_s,
            fx.as_ir_value(
                fx.Int32(scale_gather._wave_base_s) + fx.Int32(_SCALE_SLOT_BYTES)
            ),
        )
        a_scale_ld = ScaleLoaderLDS(
            N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
        )
        b_scale_ld = ScaleLoaderLDS(
            N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
        )

        def _slot(b):
            return fx.Int32(b) % fx.Int32(SCALE_SLOTS)

        gl_off_a = _global_swizzle(lane_id, wave_id, K, N_TILES_A, False)
        gl_off_b = _global_swizzle(lane_id, wave_id, K, N_TILES_B, True)
        a_rsrc = buffer_ops.create_buffer_resource(
            A, max_size=False, num_records_bytes=c_m * K
        )
        b_rsrc = buffer_ops.create_buffer_resource(
            B_T, max_size=False, num_records_bytes=c_n * K
        )
        a_g2s = G2SLoaderAsm(a_rsrc, gl_off_a, N_TILES_A, wave_id)
        b_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B, wave_id)
        a_g2s.set_wave_base(_base_ptr)
        b_g2s.set_wave_base(_base_ptr)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)
        store_c = StoreCLds(
            C,
            c_m,
            c_n,
            mfma.idx,
            lds.out_lds.ptr,
            wave_id,
            N_TILES_A,
            N_TILES_B,
            nt=store_nt,
        )

        # a step issues [ac0 (n_a) | bc0 (n_b) | bc1 (n_b) | ac1 (n_a)] loads per wave: at a
        # step's top the previous step's may all be in flight; mid-step the previous step's
        # ac1 and this step's first two batches
        _MAIN_VMCNT = 2 * N_TILES_A + 2 * N_TILES_B
        _SEG2_VMCNT = 2 * N_TILES_A + N_TILES_B

        def _read_scale_thunks(block, holder):
            # the caller passes the LDS slot itself (the block's K offset rides the
            # uniform soffset, so no block -> slot math is left)
            s = block

            def _r(dst, ld, half, _s=s):
                holder[dst] = ld.read_half(_s, half)

            return [
                lambda: _r(0, a_scale_ld, 0),
                lambda: _r(1, a_scale_ld, 1),
                lambda: _r(2, b_scale_ld, 0),
                lambda: _r(3, b_scale_ld, 1),
            ]

        def _g2s_bases(bases, step):
            a0, a1, b0, b1 = bases
            st = fx.Int32(step)
            return (
                fx.Int32(a0) + st * fx.Int32(A_K_STEP),
                fx.Int32(a1) + st * fx.Int32(A_K_STEP),
                fx.Int32(b0) + st * fx.Int32(B_K_STEP),
                fx.Int32(b1) + st * fx.Int32(B_K_STEP),
            )

        def _one_step(
            parity,
            a0f,
            b0f,
            b1f_in,
            sc,
            accs,
            bufs,
            load,
            gather,
            read_block,
            zero_acc=False,
            extra=None,
        ):
            """One K-step: this step's 64 MFMAs with, interleaved, the LDS reads for the next
            step, the global loads ``load`` of the step after (the resolved (a0, a1, b0,
            b1) soffsets), at even steps the gather ``gather`` of the next scale block (the
            precomputed (m0, soff)), at odd steps the read of scale block ``read_block``
            (the LDS slot itself) into the VGPR carry;
            ``extra(g, c00f, c01f)`` (called before group g = 0..3 with the accumulators
            finished so far, None before) returns more thunks for that group: the stores
            of this tile's c00 / c01 in the last step, of the previous tile's c10 / c11
            in the first."""
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            saR0, saR1, sbC0, sbC1 = sc
            c00f, c01f, c10f, c11f = accs
            _a1 = [None] * N_TILES_A
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            _b1n = [None] * N_TILES_B
            a0_off, a1_off, b0_off, b1_off = load
            _scn = [None, None, None, None]
            _rd_scn = _read_scale_thunks(read_block, _scn) if parity == 1 else []
            _sc_gather = (
                []
                if parity == 1
                else [lambda: scale_gather.gather_soff(gather[0], gather[1])]
            )
            ex = extra if extra is not None else (lambda g, c00, c01: ())

            wait_barrier(_MAIN_VMCNT)
            il = _riffle(
                _g2s_thunks(a_g2s, ac0, a0_off, N_TILES_A),
                _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False),
            ) + list(ex(0, None, None))
            c00f = mfma.call(
                a0f, b0f, c00f, saR0, sbC0, parity, interleave=il, zero_acc=zero_acc
            )
            il = _riffle(
                _g2s_thunks(b_g2s, bc0, b0_off, N_TILES_B), list(ex(1, c00f, None))
            )
            c01f = mfma.call(
                a0f, b1f_in, c01f, saR0, sbC1, parity, interleave=il, zero_acc=zero_acc
            )
            a1f = _a1

            wait_barrier(_SEG2_VMCNT)
            il = (_riffle(_g2s_thunks(b_g2s, bc1, b1_off, N_TILES_B), _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False))
                  + _sc_gather + _rd_scn[:2] + list(ex(2, c00f, c01f)))  # fmt: skip
            c10f = mfma.call(
                a1f, b0f, c10f, saR1, sbC0, parity, interleave=il, zero_acc=zero_acc
            )
            a0nf = _a0n
            il = (
                _riffle(
                    _g2s_thunks(a_g2s, ac1, a1_off, N_TILES_A),
                    _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True)
                    + _s2r_thunks(b_s2r, bn1, _b1n, N_TILES_B, True),
                )
                + _rd_scn[2:]
                + list(ex(3, c00f, c01f))
            )
            c11f = mfma.call(
                a1f, b1f_in, c11f, saR1, sbC1, parity, interleave=il, zero_acc=zero_acc
            )
            b0nf, b1nf = _b0n, _b1n

            sc_next = (_scn[0], _scn[1], _scn[2], _scn[3]) if parity == 1 else sc
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
            return a0nf, b0nf, b1nf, sc_next, (c00f, c01f, c10f, c11f), new_bufs

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        def _swap_bufs(bufs):
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            return (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

        n_a = N_TILES_A
        n_b = N_TILES_B
        n_ga = (N_TILES_A + 1) // _PACK
        n_gb = (N_TILES_B + 1) // _PACK
        n_sc = 2 * n_ga + 2 * n_gb
        _R = fx.as_ir_value

        def _flat_sc(sc):
            saR0, saR1, sbC0, sbC1 = sc
            return (
                [_R(v) for v in saR0]
                + [_R(v) for v in saR1]
                + [_R(v) for v in sbC0]
                + [_R(v) for v in sbC1]
            )

        def _unflat_sc(flat):
            o = 0
            saR0 = list(flat[o : o + n_ga])
            o += n_ga
            saR1 = list(flat[o : o + n_ga])
            o += n_ga
            sbC0 = list(flat[o : o + n_gb])
            o += n_gb
            sbC1 = list(flat[o : o + n_gb])
            return (saR0, saR1, sbC0, sbC1)

        # ---- prologue: the first tile's steps 0 / 1 and its scale blocks 0 / 1 ----
        t0 = wg
        ti0, tj0 = tile_coords(t0)
        bases0 = tile_bases(ti0, tj0)
        G0 = scale_gather.tile_group(ti0 * BLOCK_M, tj0 * BLOCK_N)
        scale_gather.set_tile_voff(G0)
        scale_gather.gather_soff(_gm0_s[0], 0)
        scale_gather.gather_soff(_gm0_s[1], 256)
        s0 = _g2s_bases(bases0, 0)
        s1 = _g2s_bases(bases0, 1)
        a_g2s.load(a_cur0, s0[0])
        b_g2s.load(b_cur0, s0[2])
        b_g2s.load(b_cur1, s0[3])
        a_g2s.load(a_cur1, s0[1])
        a_g2s.load(a_next0, s1[0])
        b_g2s.load(b_next0, s1[2])
        b_g2s.load(b_next1, s1[3])
        a_g2s.load(a_next1, s1[1])
        # 2 gathers + 32 loads in flight: a_cur0 landed at vmcnt(28), b_cur0 / b_cur1 at 20
        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier((3 * N_TILES_A) + (2 * N_TILES_B))
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0))
        sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0))
        sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        c10_prev = [zero4] * N_ACCUMS
        c11_prev = [zero4] * N_ACCUMS

        # ---- the tile loop: state = fragments of the tile's step 0, its scale block 0, the
        # previous tile's c10 / c11 (stored during this tile's first step), the tile index
        tile_state = (
            [_R(x) for x in a0_frag] + [_R(x) for x in b0_frag] + [_R(x) for x in b1_frag] + _flat_sc(sc0)
            + [_R(x) for x in c10_prev] + [_R(x) for x in c11_prev] + [_R(fx.Int32(t0)), _R(fx.Int32(-1))]
        )  # fmt: skip
        for jt, tstate in range(0, my_tiles, 1, init=tile_state):
            off = 0
            a0f = list(tstate[off : off + n_a])
            off += n_a
            b0f = list(tstate[off : off + n_b])
            off += n_b
            b1f = list(tstate[off : off + n_b])
            off += n_b
            sc = _unflat_sc(tstate[off : off + n_sc])
            off += n_sc
            c10p = list(tstate[off : off + N_ACCUMS])
            off += N_ACCUMS
            c11p = list(tstate[off : off + N_ACCUMS])
            off += N_ACCUMS
            t_cur = fx.Int32(tstate[off])
            t_prev = fx.Int32(tstate[off + 1])
            t_next = t_cur + grid
            ti, tj = tile_coords(t_cur)
            bases = tile_bases(ti, tj)
            G = scale_gather.tile_group(ti * BLOCK_M, tj * BLOCK_N)
            scale_gather.set_tile_voff(G)
            # the next tile (the last tile loads a tile beyond the grid: rows beyond M read
            # zero through the buffer bounds and nothing of it is stored)
            tin, tjn = tile_coords(t_next)
            bases_n = tile_bases(tin, tjn)
            Gn = scale_gather.tile_group(tin * BLOCK_M, tjn * BLOCK_N)
            # the previous tile's c10 / c11 store bases (row beyond M when there is none)
            tip, tjp = tile_coords(t_prev)
            has_prev = t_prev >= fx.Int32(0)
            prev_row1 = fx.arith.select(
                has_prev,
                tip * BLOCK_M + wave_i * (N_TILES_A * 16) + LDS_BLOCK_M,
                fx.Int32(c_m),
            )
            prev_col0 = fx.arith.select(
                has_prev, tjp * BLOCK_N + wave_j * (N_TILES_B * 16), fx.Int32(0)
            )
            prev_col1 = fx.arith.select(has_prev, prev_col0 + LDS_BLOCK_N, fx.Int32(0))
            base_row = ti * BLOCK_M + wave_i * (N_TILES_A * 16)
            base_col = tj * BLOCK_N + wave_j * (N_TILES_B * 16)

            accs = (
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
            )
            # step 0: zero accumulators, the previous tile's c10 / c11 stores spread over the
            # groups before the new c10 (born in group 2) / c11 (group 3), gather block 1
            prev_thunks = store_c.thunks(c10p, prev_row1, prev_col0) + store_c.thunks(
                c11p, prev_row1, prev_col1
            )
            n_prev3 = (
                len(prev_thunks) + 2
            ) // 3  # over the three groups before the new c11
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                0, a0f, b0f, b1f, sc, accs, bufs0, _g2s_bases(bases, 2), (_gm0_s[1], 256), None, zero_acc=True,
                extra=lambda g, c00, c01: prev_thunks[n_prev3 * g : n_prev3 * (g + 1)],
            )  # fmt: skip
            # step 1: read block 1
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                1, a0f, b0f, b1f, sc, accs, bufs, _g2s_bases(bases, 3), None, 1
            )
            # the four g2s streams advance by uniform strides and the gather
            # block by 256 B; carry them as uniform accumulators (incremented
            # per step) instead of recomputing base | (step << s) every K-step
            ka0 = fx.Int32(bases[0]) + fx.Int32(4 * A_K_STEP)
            ka1 = fx.Int32(bases[1]) + fx.Int32(4 * A_K_STEP)
            kb0 = fx.Int32(bases[2]) + fx.Int32(4 * B_K_STEP)
            kb1 = fx.Int32(bases[3]) + fx.Int32(4 * B_K_STEP)
            kg = fx.Int32(512)  # gather block soff at kk=2: block 2 << 8
            init_state = (
                [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                + [_R(x) for x in accs[0]] + [_R(x) for x in accs[1]] + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]]
                + [_R(ka0), _R(ka1), _R(kb0), _R(kb1), _R(kg)]
            )  # fmt: skip
            for kk, state in range(2, K_ITERS - 2, UNROLL, init=init_state):
                off = 0
                a0f = list(state[off : off + n_a])
                off += n_a
                b0f = list(state[off : off + n_b])
                off += n_b
                b1f = list(state[off : off + n_b])
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
                accs = (c00f, c01f, c10f, c11f)
                bufs = bufs0
                off += N_ACCUMS  # past c11f: the uniform address accumulators
                ka0 = fx.Int32(state[off])
                ka1 = fx.Int32(state[off + 1])
                kb0 = fx.Int32(state[off + 2])
                kb1 = fx.Int32(state[off + 3])
                kg = fx.Int32(state[off + 4])
                for u in range_constexpr(UNROLL):
                    if const_expr(UNROLL == 4):
                        # kk == 2 (mod 4): u=0 gathers an even block (slot 0), u=2 an
                        # odd one (slot 1) -- gather m0 slots folded; read slots stay
                        # runtime (see below).
                        lds_off = (ka0, ka1, kb0, kb1)
                        gath = None
                        if const_expr(u % 2 == 0):
                            gath = (
                                _gm0_s[u // 2],
                                kg if u == 0 else kg + fx.Int32(256),
                            )
                        rslot = None
                        if const_expr(u % 2 == 1):
                            # keep the read slot a runtime value: folding it to a
                            # constant made LLVM un-pair the scale ds_read2st64
                            # into 2 ds_read_b32 (+2 LDS/step at the ISA gate)
                            rslot = (
                                fx.Int32(kk) // fx.Int32(2) + fx.Int32(1 + u // 2)
                            ) % fx.Int32(2)
                    else:
                        kc = fx.Int32(kk) + fx.Int32(u)
                        block = kc // fx.Int32(2) + fx.Int32(1)
                        lds_off = _g2s_bases(bases, kc + fx.Int32(2))
                        gath = None
                        if const_expr(u % 2 == 0):
                            gath = (
                                fx.as_ir_value(
                                    fx.Int32(scale_gather._wave_base_s)
                                    + (block % fx.Int32(2))
                                    * fx.Int32(_SCALE_SLOT_BYTES)
                                ),
                                block * fx.Int32(256),
                            )
                        rslot = None
                        if const_expr(u % 2 == 1):
                            rslot = block % fx.Int32(2)
                    a0f, b0f, b1f, sc, accs, bufs = _one_step(
                        u % 2,
                        a0f,
                        b0f,
                        b1f,
                        sc,
                        accs,
                        bufs,
                        lds_off,
                        gath,
                        rslot,
                    )
                    ka0 = ka0 + fx.Int32(A_K_STEP)
                    ka1 = ka1 + fx.Int32(A_K_STEP)
                    kb0 = kb0 + fx.Int32(B_K_STEP)
                    kb1 = kb1 + fx.Int32(B_K_STEP)
                new_state = (
                    [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                    + [_R(x) for x in accs[0]] + [_R(x) for x in accs[1]] + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]]
                    + [_R(ka0), _R(ka1), _R(kb0), _R(kb1), _R(kg + fx.Int32(UNROLL * 128))]
                )  # fmt: skip
                state = yield new_state

            off = 0
            a0f = list(state[off : off + n_a])
            off += n_a
            b0f = list(state[off : off + n_b])
            off += n_b
            b1f = list(state[off : off + n_b])
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
            accs = (c00f, c01f, c10f, c11f)
            # step K_ITERS - 2 (even): loads of the next tile's step 0, gather of its block 0
            scale_gather.set_tile_voff(Gn)
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                0,
                a0f,
                b0f,
                b1f,
                sc,
                accs,
                bufs0,
                _g2s_bases(bases_n, 0),
                (_gm0_s[0], 0),
                None,
            )
            # step K_ITERS - 1 (odd): loads of the next tile's step 1, read of its block 0;
            # this tile's c00 / c01 stored over the groups after their last MFMAs (c00: 1..3,
            # c01: 2..3; the two quadrants' thunks stay concatenated, they share the LDS buffer)
            n_q3 = (
                2 * store_c.n_thunks + 2
            ) // 3  # thunks per group: two quadrants over 3 groups

            def _last_extra(g, c00, c01):
                # (compile-time python only: no mutable state -- the DSL rewriter turns
                # body-level literals into None -- and every `if` under const_expr)
                if const_expr(g == 0):
                    return ()
                t00 = store_c.thunks(c00, base_row, base_col)
                if const_expr(g == 1):
                    return t00[:n_q3]
                t01 = store_c.thunks(c01, base_row, base_col + LDS_BLOCK_N)
                if const_expr(g == 2):
                    return t00[n_q3:] + t01[: 2 * n_q3 - len(t00)]
                return t01[2 * n_q3 - len(t00) :]

            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                1,
                a0f,
                b0f,
                b1f,
                sc,
                accs,
                bufs,
                _g2s_bases(bases_n, 1),
                None,
                0,
                extra=_last_extra,
            )
            new_tile_state = (
                [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]] + [_R(t_next), _R(t_cur)]
            )  # fmt: skip
            tstate = yield new_tile_state

        # ---- epilogue: the last tile's c10 / c11 ----
        off = n_a + 2 * n_b + n_sc
        c10_last = list(tstate[off : off + N_ACCUMS])
        off += N_ACCUMS
        c11_last = list(tstate[off : off + N_ACCUMS])
        off += N_ACCUMS
        t_last = fx.Int32(tstate[off + 1])
        til, tjl = tile_coords(t_last)
        last_row1 = til * BLOCK_M + wave_i * (N_TILES_A * 16) + LDS_BLOCK_M
        last_col0 = tjl * BLOCK_N + wave_j * (N_TILES_B * 16)
        store_c.store(c10_last, last_row1, last_col0)
        store_c.store(c11_last, last_row1, last_col0 + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm_256x192(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        tile_off: fx.Int32,
        n_work: fx.Int32,
        stream: fx.Stream,
    ):
        """work items [tile_off, tile_off + n_work / SUB_SPLIT) of the parent grid
        ceil(M / 256) x ceil(N / 256); the whole GEMM is (0, parents * SUB_SPLIT)"""
        grid_x = _min(n_work, fx.Int32(GRID_MAX))
        kernel_gemm(
            A, B_T, C, A_scale, B_scale, c_m, c_n, tile_off, n_work,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)  # fmt: skip

    launch_gemm_256x192.kernel_name = name
    launch_gemm_256x192.sub_split = SUB_SPLIT
    return launch_gemm_256x192
