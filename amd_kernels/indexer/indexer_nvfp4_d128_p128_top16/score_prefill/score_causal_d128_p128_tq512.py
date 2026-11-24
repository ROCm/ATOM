# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One-head, single-request fp8 index scores on a compact causal task grid: each workgroup scores one four-block group
of a 512-row query tile with ``kernel.py``'s per-block math; a binary search maps its task to the tile and the group.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .kernel import (
    BLK, DQ_AT, GROUP, LDS_BYTES, MT, NI, NW, N_SLOTS, PAGE_BYTES, PAGE_CODES, PERM_EVEN,
    PERM_ODD, RAW_SLOTS, ROW_BYTES, ROWS_PER_WAVE, RS_T, SENTINEL_INIT, SENTINEL_LOCAL,
    SLOT_T, STAGE_W, TILE_Q, _asm_void, _dequant8, _global_i32_ptr, _maxf_nn,
    _mfma_fp8_16x16x128, _pack8, _perm, _xlane_max4_pair,
)


def compile_score_causal():
    """The one-head scorer on a single request's compact causal task grid (``kernel.py``'s keys, MFMA and sentinels)."""
    H = 1
    name = "nvfp4_fp8mfma_index_score_causal_d128_p128_tq512_h1_b4_tail"

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # K block slots (e4m3 rows)
        lut: fx.Array[fx.Int32, 512, 16]  # per scale byte: keys of magnitudes 0..3, 4..7
        raw: fx.Array[fx.Uint8, RAW_SLOTS * NW * STAGE_W, 16]  # DMA'd NVFP4 pages, per wave

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # idx_q [total_q, H, 128] e4m3, rows contiguous
        arg_kv: fx.Int64,  # NVFP4 index pages [num_blocks, 128, 72] u8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] i64
        arg_score: fx.Int64,  # score [H, total_q, S] fp32 (head stride i32_hs)
        arg_bt: fx.Int64,  # block table [batch, >= max_block] i32
        arg_cu: fx.Int64,  # cu_seqlens_q [batch + 1] i32
        arg_seq: fx.Int64,  # seq_lens [batch] i32
        i64_page_stride: fx.Int64,  # bytes between pages
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,  # query tiles per request
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i64_bt_bytes: fx.Int64,  # block table storage size (rows may be padded)
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
        i32_hs: fx.Int32,  # score head stride (elements)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        # one request: the causal trapezoid compacted into four-block tasks
        cu = _global_i32_ptr(arg_cu)
        b = fx.Int32(0)
        seq_start = fx.Int32(cu[0])
        q_len = fx.Int32(cu[1]) - seq_start
        seq_len = fx.Int32(_global_i32_ptr(arg_seq)[0])
        prefix = seq_len - q_len
        q_rows = q_len
        # padded queries can start before token zero: query tiles with an empty causal window are dropped first
        first_tile = (prefix < 0).select((-prefix) // TILE_Q, fx.Int32(0))
        qt = (q_len + TILE_Q - 1) // TILE_Q - first_tile
        a = (prefix + first_tile * TILE_Q + TILE_Q - 1) // TILE_Q
        prev = (qt > 0).select(qt - 1, fx.Int32(0))
        n_wg = prev * a + prev * (prev + 1) // 2 + (seq_len + TILE_Q - 1) // TILE_Q
        wid = pid
        lo = fx.Int32(0)
        hi_tile = qt
        for _ in range_constexpr(12):
            mid = (lo + hi_tile + 1) // 2
            before = mid * a + mid * (mid + 1) // 2
            take = before <= wid
            lo = take.select(mid, lo)
            hi_tile = take.select(hi_tile, mid - 1)
        q_tile = lo + first_tile
        seg = wid - lo * a - lo * (lo + 1) // 2
        row0 = q_tile * TILE_Q
        if (wid < n_wg) & (row0 < q_rows):
            # this task owns one four-block group of the tile's causal window
            hi_pos = prefix + (row0 + TILE_Q) // H
            hi = (hi_pos < seq_len).select(hi_pos, seq_len)
            nblk = (hi + (BLK - 1)) // BLK
            seg_len = GROUP
            blk0 = seg * seg_len
            blk1e = blk0 + seg_len
            blk1 = (blk1e < nblk).select(blk1e, nblk)
            if blk0 < blk1:
                qr = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_total_q) * (H * ROW_BYTES)
                )
                sr = buffer_ops.create_buffer_resource_from_addr(
                    arg_score,
                    num_records_bytes=(
                        fx.Int64(i32_hs) * (H - 1)
                        + fx.Int64(i32_total_q) * fx.Int64(i32_S)
                    )
                    * 4,
                )
                bt_row = b * i32_bt_stride
                last_blk = nblk - 1
                btr = buffer_ops.create_buffer_resource_from_addr(
                    arg_bt, num_records_bytes=i64_bt_bytes
                )

                def page_of(blk):
                    """Page id of block ``blk`` (past the end: the last block), a
                    scalar buffer load"""
                    bi = (blk < nblk).select(blk, last_blk)
                    return fx.Int32(
                        buffer_ops.buffer_load(
                            btr,
                            bt_row + bi,
                            vec_width=1,
                            dtype=fx.Int32,
                            is_scalar=True,
                        )
                    )

                # this lane's share of a page: token 32*wave + lane/2, dims
                # 64*(lane%2) .. +63 = code bytes [32*lane, +32) of the wave's 2 KB
                # and its 4 group scales. DMA (buffer_load ... lds, lane-linear LDS
                # writes): codes half h -> stage + h*1 KB + 16*lane, scales ->
                # stage + 2 KB + 4*lane, so each lane reads back its own bytes.
                raw_base_s = fx.Int32(
                    fx.rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(smem.raw.ptr)))
                )
                soff0 = fx.Int32(fx.rocdl.readfirstlane(T.i32, wave * 0))
                code_voff = [wave * 2048 + lane * 32 + h * 16 for h in range(2)]
                scale_voff = PAGE_CODES + wave * 256 + lane * 4

                def dma_raw(page, st):
                    kr = buffer_ops.create_buffer_resource_from_addr(
                        arg_kv + fx.Int64(page) * i64_page_stride,
                        num_records_bytes=PAGE_BYTES,
                    )
                    base = raw_base_s + (st * NW + wave) * STAGE_W
                    for h in range_constexpr(2):
                        _asm_void(
                            [base + h * 1024, code_voff[h], kr, soff0],
                            "s_mov_b32 m0, $0\n"
                            "buffer_load_dwordx4 $1, $2, $3 offen lds",
                            "s,v,s,s",
                        )
                    _asm_void(
                        [base + 2048, scale_voff, kr, soff0],
                        "s_mov_b32 m0, $0\nbuffer_load_dword $1, $2, $3 offen lds",
                        "s,v,s,s",
                    )

                raw_t = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.raw.ptr),
                        fx.make_layout(RAW_SLOTS * NW * STAGE_W // 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )
                raw_d = smem.raw.ptr

                def read_raw(st):
                    """This lane's 8 code dwords and its scale dword from stage st"""
                    b4 = (st * NW + wave) * (STAGE_W // 4)
                    out = []
                    for h in range_constexpr(2):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(
                            fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32),
                            fx.slice(raw_t, (None, b4 // 4 + h * 64 + lane)),
                            r,
                        )
                        v = Vec(r.load())
                        out += [fx.Int32(v[i]) for i in range_constexpr(4)]
                    sc = fx.Int32(
                        fx.make_view(
                            fx.recast_iter(fx.Int32, raw_d),
                            fx.make_layout(RAW_SLOTS * NW * STAGE_W // 4, 1),
                        )[b4 + 512 + lane]
                    )
                    return out + [sc]

                pg = [page_of(blk0 + i) for i in range_constexpr(2)]
                for i in range_constexpr(2):
                    dma_raw(pg[i], i)
                pg_next = page_of(blk0 + 2)

                # e4m3 key table -> LDS (2 dwords per thread)
                lut_g = _global_i32_ptr(arg_lut)
                lut_l = smem.lut.ptr
                for i in range_constexpr(2):
                    lut_l[tx * 2 + i] = fx.Int32(lut_g[tx * 2 + i])

                # this wave's query rows (request-local): row0 + wave*128 + ni*16 +
                # l16 = token * H + head; rows past q_len * H read the next request's
                # rows and are masked at store
                rows = [row0 + wave * ROWS_PER_WAVE + ni * 16 + l16 for ni in range(NI)]
                qpos = [prefix + r // H for r in rows]  # absolute query positions
                # Q^T B operands: lane (l16, q16) holds 16 B columns q16 and 4 + q16
                # of its row in the key order (even dims, odd dims per 8 dims)
                qf = []
                for ni in range_constexpr(NI):
                    qrow = (seq_start * H + rows[ni]) * ROW_BYTES
                    halves = []
                    for h in range_constexpr(2):
                        d = Vec(
                            buffer_ops.buffer_load(
                                qr,
                                (qrow + h * 64 + q16 * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        d = [fx.Int32(d[i]) for i in range_constexpr(4)]
                        halves.append(
                            Vec.from_elements(
                                [
                                    _perm(d[1], d[0], PERM_EVEN),
                                    _perm(d[1], d[0], PERM_ODD),
                                    _perm(d[3], d[2], PERM_EVEN),
                                    _perm(d[3], d[2], PERM_ODD),
                                ],
                                fx.Int32,
                            )
                        )
                    qf.append(_pack8(halves[0], halves[1]))

                # LDS as 16 B tiles
                lds16 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.a.ptr),
                        fx.make_layout(LDS_BYTES // 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )
                lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
                lut8 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.lut.ptr), fx.make_layout(512, 1)
                    ),
                    fx.make_layout(2, 1),
                )
                lut_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

                def lds_load16(tile):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(lds_atom, fx.slice(lds16, (None, tile)), r)
                    return r.load()

                def lds_store16(tile, vals):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(Vec.from_elements(vals, fx.Int32))
                    fx.copy(lds_atom, r, fx.slice(lds16, (None, tile)))

                def lut_load(sbyte):
                    r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                    fx.copy(lut_atom, fx.slice(lut8, (None, sbyte)), r)
                    v = Vec(r.load())
                    return fx.Int32(v[0]), fx.Int32(v[1])

                # dequant writer: token row t = 32*wave + lane/2, half hh; 16 B
                # column c of row t sits at c ^ ((t >> 1) & 7)
                t_row = wave * 32 + (lane >> 1)
                hh = lane & 1
                wr_base = t_row * RS_T
                wr_sw = (t_row >> 1) & 7

                def dq_tables(raw):
                    sc = raw[8]
                    return [
                        lut_load((sc >> (8 * g)) & 0xFF) for g in range_constexpr(4)
                    ]

                def dq_step(raw, tabs, dslot, k, pend):
                    """Code dword k of this lane's half -> 8 keys; every second one
                    stores 16 B (column 4*hh + k/2 of the token row)"""
                    lo, hi = _dequant8(raw[k], *tabs[k // 2])
                    if const_expr(k % 2 == 0):
                        return [lo, hi]
                    col = (hh * 4 + k // 2) ^ wr_sw
                    lds_store16(dslot * SLOT_T + wr_base + col, pend + [lo, hi])
                    return []

                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))
                # A operand of M-tile mt: row mt*16 + l16, 16 B columns q16 and 4 + q16,
                # swizzled by (row >> 1) & 7 = l16 >> 1
                rd_row = l16 * RS_T
                rd_col = [q16 ^ (l16 >> 1), (4 + q16) ^ (l16 >> 1)]

                def compute_block(slot, blk, masked, st, dslot, dequant_next):
                    """Block scores of this wave's 128 rows (8 values per lane,
                    identical across the 4 q16 lanes), optionally dequantizing the
                    next block's raw page from ``st`` into ``dslot``."""
                    if const_expr(masked):
                        vis = [
                            qpos[ni] - (blk * BLK + q16 * 4)
                            for ni in range_constexpr(NI)
                        ]
                    run = [neg_inf for _ in range(NI)]
                    if const_expr(dequant_next):
                        raw = read_raw(st)
                        tabs = dq_tables(raw)

                    def read_a(mt):
                        base = slot * SLOT_T + mt * 16 * RS_T + rd_row
                        return _pack8(
                            lds_load16(base + rd_col[0]), lds_load16(base + rd_col[1])
                        )

                    accs = [[None] * NI, [None] * NI]

                    def issue(mt, a):
                        acc = accs[mt % 2]
                        for ni in range_constexpr(NI):
                            acc[ni] = _mfma_fp8_16x16x128(a, qf[ni], zero4)

                    def reduce(mt):
                        acc = accs[mt % 2]
                        for ni in range_constexpr(NI):
                            v = acc[ni]
                            for i in range_constexpr(4):
                                x = v[i]
                                if const_expr(masked):
                                    x = (vis[ni] >= mt * 16 + i).select(x, neg_inf)
                                run[ni] = _maxf_nn(run[ni], x)

                    pend = []
                    a_next = read_a(0)
                    for mt in range_constexpr(MT):
                        a = a_next
                        if const_expr(mt + 1 < MT):
                            a_next = read_a(mt + 1)
                        issue(mt, a)
                        if const_expr(mt >= 1):
                            reduce(mt - 1)
                        ks = DQ_AT.get(mt, ()) if dequant_next else ()
                        for k in ks:
                            pend = dq_step(raw, tabs, dslot, k, pend)
                        n_valu = (16 if not masked else 48) + 10 * len(ks)
                        n_dsw = sum(k % 2 for k in ks)
                        for i in range_constexpr(NI):
                            fx.rocdl.sched_group_barrier(0x008, 1, 0)
                            fx.rocdl.sched_group_barrier(
                                0x002, (n_valu + NI - 1) // NI, 0
                            )
                            if const_expr(i < 2):
                                fx.rocdl.sched_group_barrier(0x100, 1, 0)
                            if const_expr(i == 4 and n_dsw):
                                fx.rocdl.sched_group_barrier(0x200, n_dsw, 0)
                        fx.rocdl.sched_barrier(0)
                    reduce(MT - 1)
                    out = []
                    for p in range_constexpr(NI // 2):
                        x, y = _xlane_max4_pair(run[2 * p], run[2 * p + 1])
                        out += [x, y]
                    return out

                sent_init = fx.Float32(SENTINEL_INIT)
                sent_local = fx.Float32(SENTINEL_LOCAL)

                def store_group(g0, sc):
                    """sc[j][ni]: scores of blocks g0..g0+3. Lane q16 = k stores rows
                    ni = 2k, 2k+1, 16 B per row (single dwords for a partial tail).
                    The row's init / local blocks get AITER's sentinels instead. No staging: one group per task."""
                    full = g0 + GROUP <= blk1
                    def store_now(vals, off, valid):
                        """16 B per row (single dwords for a partial tail)."""
                        if full:
                            buffer_ops.buffer_store(
                                fx.Vector.from_elements(vals, fx.Float32),
                                sr,
                                off,
                                mask=valid,
                            )
                        else:
                            for j in range_constexpr(GROUP):
                                buffer_ops.buffer_store(
                                    vals[j], sr, off + j, mask=valid & (g0 + j < blk1)
                                )

                    for h in range_constexpr(2):
                        row = rows[h]
                        vals = [sc[j][h] for j in range_constexpr(GROUP)]
                        for k in range_constexpr(1, NW):
                            is_k = q16 == k
                            row = is_k.select(rows[2 * k + h], row)
                            vals = [
                                is_k.select(sc[j][2 * k + h], vals[j])
                                for j in range_constexpr(GROUP)
                            ]
                        valid = row < q_rows
                        tok = row // H
                        nb_row = (
                            prefix + tok
                        ) // BLK + 1  # the row's causal block count
                        for j in range_constexpr(GROUP):
                            blk = g0 + j
                            is_local = (blk >= nb_row - i32_local) & (blk < nb_row)
                            is_init = (blk < i32_init) & (blk < nb_row)
                            vals[j] = is_local.select(
                                sent_local, is_init.select(sent_init, vals[j])
                            )
                        off = (row % H) * i32_hs + (seq_start + tok) * i32_S + g0
                        store_now(vals, off, valid)

                # an unmasked group must precede the tile's first query row
                v = prefix + row0 - (BLK - 1)
                last_full = (v < 0).select(fx.Int32(-1), v // BLK)

                # prologue: table and the first raw pages in LDS, block blk0
                # dequantized into slot 0
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()
                raw0 = read_raw(0)
                tabs0 = dq_tables(raw0)
                pend = []
                for k in range_constexpr(8):
                    pend = dq_step(raw0, tabs0, 0, k, pend)
                fx.gpu.barrier()

                def group(g0, pg_next, masked):
                    """Four blocks with two raw-page slots and alternating key slots, fetching only pages a later
                    block consumes; the last block needs neither a next-block dequant nor another LDS handoff."""
                    sc = []
                    for j in range_constexpr(GROUP):
                        blk = g0 + j
                        if const_expr(j + 2 < GROUP):
                            dma_raw(pg_next, (j + 2) % RAW_SLOTS)
                        if const_expr(j + 3 < GROUP):
                            pg_next = page_of(blk + 3)
                        sc.append(
                            compute_block(
                                j % N_SLOTS,
                                blk,
                                masked,
                                (j + 1) % RAW_SLOTS,
                                (j + 1) % N_SLOTS,
                                j + 1 < GROUP,
                            )
                        )
                        if const_expr(j + 1 < GROUP):
                            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                            fx.gpu.barrier()
                    store_group(g0, sc)
                    return pg_next

                if last_full >= blk0 + GROUP - 1:
                    group(blk0, pg_next, False)
                else:
                    group(blk0, pg_next, True)

    @flyc.jit
    def launch_index_score_causal_d128_p128_tq512(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_cu: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_hs: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_lut,
            arg_score,
            arg_bt,
            arg_cu,
            arg_seq,
            i64_page_stride,
            i32_bt_stride,
            i32_S,
            i32_nseg,
            i32_qt,
            i32_total_q,
            i32_nb,
            i64_bt_bytes,
            i32_init,
            i32_local,
            i32_hs,
            # two waves per SIMD, as in kernel.py's staged scorer
            value_attrs={"rocdl.waves_per_eu": 2},
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch_index_score_causal_d128_p128_tq512.kernel_name = name
    return launch_index_score_causal_d128_p128_tq512
