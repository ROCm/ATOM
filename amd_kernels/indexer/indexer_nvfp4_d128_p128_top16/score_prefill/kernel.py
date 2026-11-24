# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Prefill index scorer over the NVFP4 index pages, fp8 keys (e4m3_rn_satfinite(e2m1 x scale)), FlyDSL (gfx950): per
(row, block) the max over the row's causal tokens of q . k with the init (1e30) / local (1e29) sentinels; bit for bit
the scores of ``kernel_shared.py``'s fp8 route.

A workgroup is 4 waves x 128 query rows (a 512-row tile) x a segment of the request's context. A query row is a (token,
head) pair, row = token * H + head (H = 1, 2 or 4 heads share one index-K head), so every dequantized key block serves
all 512 rows. Q^T stays in VGPRs; one ``v_mfma_f32_16x16x128_f8f6f4`` per (16-token M-tile, 16-row N-tile) covers the
whole head. Two 16 KB LDS slots hold e4m3 key rows (16 B column c of row r at c ^ ((r >> 1) & 7): an A fragment's 16
rows hit 16 bank groups). Two accumulator sets: the previous tile's max runs under this tile's MFMAs.

Lane l of wave w owns token 32w + l/2, dims 64(l%2) .. +63 of a page. Its 32 code bytes and 4 scale bytes reach a
per-wave LDS staging area by ``buffer_load ... lds`` DMA one block ahead; block b+1 is dequantized into the other key
slot under block b's MFMAs. A 256-entry LDS table holds per scale byte the e4m3 keys of the 8 e2m1 magnitudes
(``host.nvfp4_e4m3_lut``), so 8 codes cost two ``v_perm_b32`` plus the sign bits. Code dword k of a half gives the even
dims (8k + 0, 2, 4, 6), then the odd ones; Q is loaded in the same order.

The DMA is the intrinsic (``_dma``), page ids a constant-space ``s_load`` (``page_lookup``): inline-asm DMAs spilled
more and ran 3-8% slower.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _mlir_ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

NW = 4  # waves per workgroup
ROWS_PER_WAVE = 128  # 8 MFMA N-tiles of query rows
TILE_Q = NW * ROWS_PER_WAVE  # 512 query rows per workgroup
BLK = 128  # tokens per sparse block (one page)
D = 128  # index head dim
ROW_BYTES = D  # 128 B per e4m3 key row in LDS
BLOCK_BYTES = BLK * ROW_BYTES  # 16 KB
RS_T = ROW_BYTES // 16  # 16 B tiles per LDS row
SLOT_T = BLK * RS_T  # tiles per block slot
N_SLOTS = 2
LDS_BYTES = N_SLOTS * BLOCK_BYTES  # 32 KB
PAGE_CODES = BLK * D // 2  # NVFP4 page: [128 x 64 B] e2m1 codes, then [128 x 8 B] e4m3 scales
PAGE_BYTES = PAGE_CODES + BLK * 8
GROUP = 4  # blocks buffered per score store (16 B per row)
# Store groups written per row together (``staged``): the first SPAN - 1 wait in LDS, so memory gets whole 64 B
# pieces; L2 evicts half-written 128 B score lines, and 16 B groups reached memory as 32 B writes.
SPAN = 4
STAGE_T = (SPAN - 1) * NW * 2 * 64  # 16 B LDS tiles: (group, wave, row half, lane)
NI = ROWS_PER_WAVE // 16  # N-tiles (query rows) per wave
MT = BLK // 16  # M-tiles (tokens) per block
SENTINEL_INIT = 1e30  # AITER's forced-selection scores (its top-k orders by them)
SENTINEL_LOCAL = 1e29
PERM_EVEN, PERM_ODD = 0x06040200, 0x07050301
STAGE_W = 2048 + 256  # raw bytes per wave and block: 32 tokens x 64 B codes + their scales
RAW_SLOTS = 2  # raw pages DMA'd one block ahead (two blocks ahead measured slower)
# dequant schedule: M-tile -> code dwords dequantized under its MFMAs (raw reads and table lookups under M-tile 0)
DQ_AT = {1: (0,), 2: (1,), 3: (2,), 4: (3,), 5: (4,), 6: (5,), 7: (6, 7)}


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _asm_void(operands, asm, constraints):
    """Side-effecting void inline asm (LLVM sees no memory op and inserts no waitcnt: the DMA wait is explicit)."""
    llvm.inline_asm(
        None, [_ir(o) for o in operands], asm, constraints, has_side_effects=True
    )


def _dma(rsrc, lds_addr, voff, size):
    """``buffer_load ... lds``: ``size`` bytes per lane into LDS at ``lds_addr`` (the intrinsic: LLVM schedules it as
    a load and counts it in vmcnt)."""
    lds = llvm.IntToPtrOp(_mlir_ir.Type.parse("!llvm.ptr<3>"), _ir(fx.Int32(lds_addr)))
    fx.rocdl.raw_ptr_buffer_load_lds(
        rsrc,
        lds.result,
        fx.Int32(size),
        fx.Int32(voff),
        fx.Int32(0),
        fx.Int32(0),
        aux=0,
    )


def page_lookup(arg_bt, bt_row, end):
    """``page_of(blk)``: the page of block ``blk`` of block-table row ``bt_row`` (from ``end``: block ``end - 1``), a
    constant-space ``s_load`` (before an ``s_buffer_load`` LLVM waits for the in-flight LDS DMAs)."""
    bt = llvm.IntToPtrOp(_mlir_ir.Type.parse("!llvm.ptr<4>"), _ir(fx.Int64(arg_bt))).result

    def page_of(blk):
        bi = (blk < end).select(blk, end - 1)
        p = buffer_ops.get_element_ptr(bt, byte_offset=_ir((bt_row + bi) * 4), elem_type=T.i8)
        return fx.Int32(llvm.LoadOp(T.i32, p, alignment=4, invariant=True).result)

    return page_of


def _global_i32_ptr(addr_i64):
    """Typed global i32 pointer at a raw device address (``p[i]`` loads)."""
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _pack8(h0, h1):
    """Two 4 x i32 halves -> the 8 x i32 operand of the fp8 16x16x128 MFMA: VGPRs 0-3 = K bytes [16*klane, +16),
    VGPRs 4-7 = [64 + 16*klane, +16) of the lane's row (klane = lane // 16)."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements(
        [a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32
    )


def _mfma_fp8_16x16x128(a8, b8, acc4):
    """``v_mfma_f32_16x16x128_f8f6f4`` (e4m3 x e4m3, f32 accumulate) through the ROCDL scaled op with zero block
    scales (an intrinsic, so LLVM sees the MFMA -> VALU hazard). Lane L gets D[4 * (L // 16) + i][L % 16]."""
    zero = fx.as_ir_value(fx.Int32(0))
    e4m3 = _mlir_ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>")
    opsel0 = _mlir_ir.IntegerAttr.get(_mlir_ir.IntegerType.get_signless(32), 0)
    res = _rocdl_ops.mfma_scale_f32_16x16x128_f8f6f4(
        Vec.make_type(4, fx.Float32),
        fx.as_ir_value(a8),
        fx.as_ir_value(b8),
        fx.as_ir_value(acc4),
        e4m3,
        e4m3,
        opsel0,
        zero,
        opsel0,
        zero,
    ).result
    return Vec(res)


def _permlane_swap(op, d_a, d_b):
    """v_permlane{16,32}_swap (raw rocdl op; fx has no wrapper)."""
    pair_ty = _mlir_ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
    return (
        fx.Int32(llvm.extractvalue(T.i32, res, [0])),
        fx.Int32(llvm.extractvalue(T.i32, res, [1])),
    )


def _maxf_nn(a, b):
    """v_max_f32 with nnan (without it LLVM canonicalizes both inputs first)."""
    fm = _mlir_ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def _xlane_max4_pair(x, y):
    """Max over lanes L, L^16, L^32, L^48 of two values at once, both broadcast to every lane: swap32(x, y) + max ->
    [X2 | Y2]; swap16 + max -> [X4 X4 | Y4 Y4]; swap32 -> [X4 | X4], [Y4 | Y4]."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(y))
    m = _bits(_maxf_nn(_as_f32(a), _as_f32(b)))
    a, b = _permlane_swap(_rocdl.permlane16_swap, m, m)
    big = _bits(_maxf_nn(_as_f32(a), _as_f32(b)))
    a, b = _permlane_swap(_rocdl.permlane32_swap, big, big)
    return _as_f32(a), _as_f32(b)


def _perm(hi, lo, sel):
    return fx.Int32(_rocdl.perm_b32(fx.Int32(hi), fx.Int32(lo), fx.Int32(sel)))


def _dequant8(c, t_lo, t_hi):
    """8 e2m1 codes (one dword: dims 0..7, low nibble the even dim) with the group's table -> the e4m3 keys of the
    even dims (0, 2, 4, 6) and of the odd dims (1, 3, 5, 7)."""
    ml = _perm(t_hi, t_lo, c & fx.Int32(0x07070707))
    mh = _perm(t_hi, t_lo, (c >> 4) & fx.Int32(0x07070707))
    return ml ^ ((c << 4) & fx.Int32(0x80808080 - (1 << 32))), mh ^ (
        c & fx.Int32(0x80808080 - (1 << 32))
    )


def compile_score_prefill(heads: int, staged: bool):
    """One kernel per index-head count (1, 2 or 4) and store mode (``staged``: SPAN groups per row write; else each
    group as it completes); all sizes are runtime arguments."""
    assert heads in (1, 2, 4), heads
    H = heads
    name = f"nvfp4_fp8mfma_index_score_prefill_d128_p128_tq{TILE_Q}_h{H}" + ("" if staged else "_g4")

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # K block slots (e4m3 rows)
        lut: fx.Array[fx.Int32, 512, 16]  # per scale byte: keys of magnitudes 0..3, 4..7
        raw: fx.Array[fx.Uint8, RAW_SLOTS * NW * STAGE_W, 16]  # DMA'd NVFP4 pages, per wave
        stage: fx.Array[fx.Int32, STAGE_T * 4 if staged else 4, 16]  # staged score groups, lane-private

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
        i64_bt_bytes: fx.Int64,  # block table storage size (unused: the launch signature of every prefill scorer)
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
        i32_hs: fx.Int32,  # score head stride (elements)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        n_wg = i32_qt * i32_nseg * i32_nb
        wid = pid
        q_tile = wid % i32_qt
        seg = (wid // i32_qt) % i32_nseg
        b = wid // (i32_qt * i32_nseg)

        bc = (wid < n_wg).select(b, fx.Int32(0))  # padded workgroups read request 0
        cu = _global_i32_ptr(arg_cu)
        seq_start = fx.Int32(cu[bc])
        q_len = fx.Int32(cu[bc + 1]) - seq_start
        seq_len = fx.Int32(_global_i32_ptr(arg_seq)[bc])
        prefix = seq_len - q_len  # rows are aligned to the end of the request
        q_rows = q_len * H  # request-local query rows: token * H + head
        row0 = q_tile * TILE_Q
        if (wid < n_wg) & (row0 < q_rows):
            # causal window of this query tile, split into segments of 4k blocks
            hi_pos = prefix + (row0 + TILE_Q) // H
            hi = (hi_pos < seq_len).select(hi_pos, seq_len)
            nblk = (hi + (BLK - 1)) // BLK
            seg_len = ((nblk + i32_nseg - 1) // i32_nseg + (GROUP - 1)) & ~(GROUP - 1)
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
                page_of = page_lookup(arg_bt, b * i32_bt_stride, nblk)

                # this lane's share of a page: token 32*wave + lane/2, dims
                # 64*(lane%2) .. +63 = code bytes [32*lane, +32) of the wave's 2 KB
                # and its 4 group scales. DMA (buffer_load ... lds, lane-linear LDS
                # writes): codes half h -> stage + h*1 KB + 16*lane, scales ->
                # stage + 2 KB + 4*lane, so each lane reads back its own bytes.
                raw_base_s = fx.Int32(
                    fx.rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(smem.raw.ptr)))
                )
                code_voff = [wave * 2048 + lane * 32 + h * 16 for h in range(2)]
                scale_voff = PAGE_CODES + wave * 256 + lane * 4

                def dma_raw(page, st):
                    kr = buffer_ops.create_buffer_resource_from_addr(
                        arg_kv + fx.Int64(page) * i64_page_stride,
                        num_records_bytes=PAGE_BYTES,
                    )
                    base = raw_base_s + (st * NW + wave) * STAGE_W
                    for h in range_constexpr(2):
                        _dma(kr, base + h * 1024, code_voff[h], 16)
                    _dma(kr, base + 2048, scale_voff, 4)

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

                def compute_block(slot, blk, masked, st, dslot):
                    """Block scores of this wave's 128 rows (8 values per lane,
                    identical across the 4 q16 lanes), and the dequant of the next
                    block's raw page (staging buffer ``st``) into ``dslot``."""
                    if const_expr(masked):
                        vis = [
                            qpos[ni] - (blk * BLK + q16 * 4)
                            for ni in range_constexpr(NI)
                        ]
                    run = [neg_inf for _ in range(NI)]
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
                        ks = DQ_AT.get(mt, ())
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

                stage16 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.stage.ptr),
                        fx.make_layout((STAGE_T if staged else 1) * 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )

                def stage_tile(k, h):
                    return ((k * NW + wave) * 2 + h) * 64 + lane

                def store_group(g0, sc):
                    """sc[j][ni]: scores of blocks g0..g0+3. Lane q16 = k stores rows
                    ni = 2k, 2k+1, 16 B per row (single dwords for a partial tail).
                    The row's init / local blocks get AITER's sentinels instead.
                    ``staged``: a group waits in LDS unless it ends its row's SPAN-group
                    piece or the segment; that one stores the piece's staged groups
                    (from the segment's first) and itself back to back."""
                    full = g0 + GROUP <= blk1
                    if const_expr(staged):
                        pos = (g0 >> 2) & (SPAN - 1)  # group within the 16-block piece
                        piece0 = g0 & ~(SPAN * GROUP - 1)
                        first = (piece0 < blk0).select(
                            (blk0 >> 2) & (SPAN - 1), fx.Int32(0)
                        )
                        flush = (pos == SPAN - 1) | (g0 + GROUP >= blk1)

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

                    def flush_piece(h, vals, off, valid):
                        """The piece's staged groups, then this one, back to back."""
                        for k in range_constexpr(SPAN - 1):
                            t = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            fx.copy(
                                lds_atom, fx.slice(stage16, (None, stage_tile(k, h))), t
                            )
                            buffer_ops.buffer_store(
                                Vec(t.load()),
                                sr,
                                off - (pos - k) * GROUP,
                                mask=valid & (first <= k) & (pos > k),
                            )
                        store_now(vals, off, valid)

                    def stage_group(h, vals):
                        t = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        t.store(Vec.from_elements([_bits(x) for x in vals], fx.Int32))
                        fx.copy(
                            lds_atom, t, fx.slice(stage16, (None, stage_tile(pos, h)))
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
                        if const_expr(staged):
                            if flush:
                                flush_piece(h, vals, off, valid)
                            else:
                                stage_group(h, vals)
                        else:
                            store_now(vals, off, valid)

                n_groups = (blk1 - blk0 + (GROUP - 1)) // GROUP
                # groups whose 4 blocks every row of the tile sees whole (block
                # blk*128+127 <= prefix + row0 / H) need no causal select
                v = prefix + row0 // H - (BLK - 1)
                last_full = (v < 0).select(fx.Int32(-1), v // BLK)
                d = last_full - (GROUP - 1) - blk0
                nf = (d < 0).select(fx.Int32(0), d // GROUP + 1)
                n_full = (nf < n_groups).select(nf, n_groups)

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
                    """Blocks g0 .. g0+3: per block, DMA the raw page of block b+2
                    (its page id fetched one block earlier), compute block b while
                    dequantizing block b+1 into the other slot, wait for the raw page
                    of block b+2, barrier. Block blk0 + n's raw page sits in staging
                    buffer n % 2 (g0 - blk0 is a multiple of 4)."""
                    sc = []
                    for j in range_constexpr(GROUP):
                        blk = g0 + j
                        dma_raw(pg_next, (j + 2) % RAW_SLOTS)
                        pg_next = page_of(blk + 3)
                        sc.append(
                            compute_block(
                                j % N_SLOTS,
                                blk,
                                masked,
                                (j + 1) % RAW_SLOTS,
                                (j + 1) % N_SLOTS,
                            )
                        )
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        fx.gpu.barrier()
                    store_group(g0, sc)
                    return pg_next

                def one(st):
                    return fx.Int32(st[0] if isinstance(st, (list, tuple)) else st)

                for g, st in range(
                    fx.Index(0), fx.Index(n_full), fx.Index(1), init=[_ir(pg_next)]
                ):
                    pgn = group(blk0 + fx.Int32(g) * GROUP, one(st), False)
                    res = yield [_ir(pgn)]
                for g, st in range(
                    fx.Index(n_full),
                    fx.Index(n_groups),
                    fx.Index(1),
                    init=[_ir(one(res))],
                ):
                    pgn = group(blk0 + fx.Int32(g) * GROUP, one(st), True)
                    res = yield [_ir(pgn)]

    @flyc.jit
    def launch(
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
            # staged: LLVM picks 264 VGPRs (1 wave / SIMD) unless held to 2 waves
            **({"value_attrs": {"rocdl.waves_per_eu": 2}} if staged else {}),
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
