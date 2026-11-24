# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Decode index scorer over the NVFP4 index pages: per (head, row, block) the max over the row's visible tokens of
q . k, with AITER's init (1e30) / local (1e29) sentinels. Keys are e4m3_rn_satfinite(e2m1 x scale) on the fp8 MFMA.

A workgroup takes a chunk of up to 64 requests (lane r of every wave holds request r's state and looks up its page)
and the block positions b = split, split + splits, ...; its 4 waves share each position. The distinct pages at b (a
prefix-shared page is one page for the whole chunk) go round robin over the waves. A wave dequantizes its page into
the 8 A fragments of the 16x16x128 fp8 MFMA (tile t: lane (col, kl) holds token 16 t + col, dims 32 kl .. +31, even
dims then odd dims per 8, through the 256-entry e4m3 table in LDS), issues the next page's loads tile by tile into the
freed registers, and scores the page against every 16-column group holding it (16 / R requests of R = H * query_len
rows; their Q rows sit in LDS in the key's dim order, XOR-swizzled). Pages load with ``nt``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..score_prefill.kernel import (
    PERM_EVEN,
    PERM_ODD,
    _as_f32,
    _bits,
    _dequant8,
    _global_i32_ptr,
    _maxf_nn,
    _mfma_fp8_16x16x128,
    _pack8,
    _perm,
    _permlane_swap,
)

NW = 4  # waves per workgroup
MAX_REQS = 64  # requests per chunk (one per lane)
BLK = 128  # tokens per block (one page)
D = 128  # index head dim
PAGE_CODES = BLK * D // 2  # [128 x 64 B] e2m1 codes, then [128 x 8 B] e4m3 group scales
PAGE_BYTES = PAGE_CODES + BLK * 8
MT = BLK // 16  # token tiles per page
SENTINEL_INIT = 1e30  # AITER's forced-selection scores (its top-k orders by them)
SENTINEL_LOCAL = 1e29
NT = 2  # buffer cache policy of the page loads
# Blocks per LDS-buffered score unit: QROWS x 36 B of LDS keeps 2-4 workgroups per CU; flushing 32 B of a row measured
# the same as whole 128 B lines.
UMAX = 8


def _xlane_max4(x):
    """Max over lanes L, L^16, L^32, L^48, broadcast to every lane."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    m = _bits(_maxf_nn(_as_f32(a), _as_f32(b)))
    a, b = _permlane_swap(_rocdl.permlane16_swap, m, m)
    return _maxf_nn(_as_f32(a), _as_f32(b))


MAX_ROWS = 32  # H * query_len


def padded_rows(rows: int) -> int:
    """A request's columns in the chunked kernel: rows rounded up to a power of two (padding columns are not stored)."""
    return 1 << (rows - 1).bit_length()


def chunk_max(rows: int) -> int:
    """Requests per chunk: one per lane, the chunk's 256 e4m3 Q rows in LDS (32 KB)."""
    return min(MAX_REQS, 256 // rows)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def compile_score_decode(
    rows: int,
    heads: int,
    buffered: bool = False,
    device_positions: bool = False,
):
    """The chunked scorer for ``rows`` = H * query_len (1..32); all sizes are runtime arguments.

    Args:
        buffered: a unit's scores gather in LDS and leave as whole rows, else they store directly.
        device_positions: (fp8, direct stores, per-wave chunks of 2-3 requests) shared positions instead when the
            device-side lengths give every block position its own workgroup.
    """
    assert 1 <= rows <= MAX_ROWS and heads in (1, 2, 4) and rows % heads == 0
    assert not device_positions or not buffered
    H = heads
    QL = rows // H
    R = padded_rows(rows)  # a request's columns; column j = query row j // H
    RPC = max(1, 16 // R)  # requests per 16-column group
    CPR = max(1, R // 16)  # 16-column groups per request
    GMASK = (1 << RPC) - 1
    QROWS = chunk_max(R) * R
    QROW_BYTES = D
    name = (
        f"nvfp4_fp8mfma_index_score_decode_d128_p128_r{rows}h{H}"
        + ("_buf" if buffered else "")
        + ("_devpos" if device_positions else "")
    )

    # buffered: a unit's scores leave LDS as whole rows of 4-block pieces instead of scattered dwords
    SB_STRIDE = UMAX + 1  # padded: 16 lanes store a row each without bank conflicts

    @fx.struct
    class Shared:
        q: fx.Array[fx.Uint8, QROWS * QROW_BYTES, 16]  # the chunk's Q rows
        lut: fx.Array[
            fx.Int32, 512, 16
        ]  # per scale byte: keys of magnitudes 0..3, 4..7
        seq: fx.Array[fx.Int32, MAX_REQS]
        # the unit's scores as f32 bits (a Float32 array's pointer is None inside
        # nested functions) + 64 dummy slots; unused without ``buffered``
        sbuf: fx.Array[fx.Int32, (QROWS * SB_STRIDE + 64) if buffered else 16, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [num_reqs * QL, H, 128] e4m3
        arg_kv: fx.Int64,  # NVFP4 index pages [num_blocks, 128, 72] u8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] i64
        arg_score: fx.Int64,  # [H, num_reqs * QL, S] fp32
        arg_bt: fx.Int64,  # block table [num_reqs, >= max_blocks] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs] i32
        i64_page_stride: fx.Int64,  # bytes between pages
        i32_q_st: fx.Int32,  # q stride(0), stride(1) (bytes)
        i32_q_sh: fx.Int32,
        i64_q_bytes: fx.Int64,
        i32_s_sh: fx.Int32,  # score stride(0), stride(1) (elements)
        i32_s_st: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_chunk: fx.Int32,  # requests per chunk (<= 64)
        i32_nch: fx.Int32,
        i32_splits: fx.Int32,  # workgroups per chunk
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
        i32_per_wave: fx.Int32,  # 1: each wave its own positions (chunks < NW)
        i32_unit: fx.Int32,  # consecutive positions per unit (<= UMAX; 0: none)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        col, kl = lane % 16, lane // 16
        ch = pid % i32_nch
        split = pid // i32_nch
        req0 = ch * i32_chunk
        left = i32_num_reqs - req0
        nr = (left < i32_chunk).select(left, i32_chunk)

        # per-request state, lane r = request req0 + r
        r_ok = lane < nr
        seq_r = fx.Int32(
            _global_i32_ptr(arg_seq)[r_ok.select(req0 + lane, fx.Int32(0))]
        )
        seq_r = r_ok.select(seq_r, fx.Int32(0))
        nblk_r = (seq_r + (BLK - 1)) // BLK
        first_row = seq_r - (QL - 1)  # position of the request's first query row
        # blocks before rbf_r are seen whole by every row: init / local ones there
        # only need their sentinel
        rbf_r = ((first_row < 0).select(fx.Int32(0), first_row) + (BLK - 1)) // BLK
        ll = nblk_r - i32_local
        loc_last_r = (ll < 0).select(fx.Int32(0), ll)
        fm_r = seq_r - QL  # tokens past this position are masked for some row

        btr = buffer_ops.create_buffer_resource_from_addr(
            arg_bt, num_records_bytes=i64_bt_bytes
        )
        sr = buffer_ops.create_buffer_resource_from_addr(
            arg_score, num_records_bytes=i64_s_bytes
        )
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=i64_q_bytes
        )

        q16 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.q.ptr),
                fx.make_layout(QROWS * QROW_BYTES // 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        seq_l = smem.seq.ptr
        scb = smem.sbuf.ptr
        per_wave = i32_per_wave != fx.Int32(0)
        # i32_unit 0: scores go straight to memory (small footprint); else the
        # waves' shared units are buffered in LDS (per-wave mode stores directly)
        if const_expr(buffered):
            direct = per_wave | (i32_unit == fx.Int32(0))
            to_lds = (i32_per_wave == fx.Int32(0)) & (i32_unit != fx.Int32(0))
            usz = (i32_unit == fx.Int32(0)).select(fx.Int32(1), i32_unit)
        else:
            direct = i32_unit >= fx.Int32(0)  # always
            to_lds = i32_unit < fx.Int32(0)  # never
            usz = fx.Int32(1)

        def prologue():
            """The chunk's Q rows, the e4m3 table and the lengths -> LDS: row i = request i // R, column j = i % R
            (query row j // H, head j % H); 16 B column c8 at c8 ^ (i & 7), dwords permuted to the key order (even
            dims, odd dims per 8)."""
            for k in range_constexpr(QROWS * 8 // (64 * NW)):
                u = tx + k * 64 * NW
                i = u // 8
                c8 = u % 8
                rr = i // R
                j = i % R
                # rows of missing requests stay unwritten: only their own MFMA
                # columns read them, and those scores are never stored
                if (rr < nr) & (j // H < QL):  # padding columns stay unwritten
                    tq = (req0 + rr) * QL + j // H
                    off = (tq * i32_q_st + (j % H) * i32_q_sh + c8 * 16) // 4
                    d = Vec(
                        buffer_ops.buffer_load(
                            qr, off, vec_width=4, dtype=fx.Int32, mask=rr < nr
                        )
                    )
                    d = [fx.Int32(d[e]) for e in range_constexpr(4)]
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(
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
                    fx.copy(atom16, r, fx.slice(q16, (None, i * 8 + (c8 ^ (i & 7)))))
            lut_g = _global_i32_ptr(arg_lut)
            lut_l = smem.lut.ptr
            for e in range_constexpr(2):
                lut_l[tx * 2 + e] = fx.Int32(lut_g[tx * 2 + e])
            seq_l[lane] = seq_r  # every wave writes the same 64 values

        prologue()
        fx.gpu.barrier()

        lut8 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.lut.ptr), fx.make_layout(512, 1)
            ),
            fx.make_layout(2, 1),
        )
        lut_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

        def lut_load(sbyte):
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            fx.copy(lut_atom, fx.slice(lut8, (None, sbyte)), r)
            v = Vec(r.load())
            return fx.Int32(v[0]), fx.Int32(v[1])

        def q_frag(cg):
            """B operand of column group cg: Q row cg * 16 + col, 16 B columns 2 kl and 2 kl + 1 (dims 32 kl .. +31)."""
            i = cg * 16 + col
            halves = []
            for h in range_constexpr(2):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(
                    atom16, fx.slice(q16, (None, i * 8 + ((2 * kl + h) ^ (i & 7)))), r
                )
                halves.append(r.load())
            return _pack8(halves[0], halves[1])

        code_off = (col * 64 + kl * 16) // 4  # dwords
        scale_off = col * 4 + kl  # u16 units: token col's groups 2 kl, 2 kl + 1

        def page_rsrc(page):
            return buffer_ops.create_buffer_resource_from_addr(
                arg_kv + fx.Int64(page) * i64_page_stride, num_records_bytes=PAGE_BYTES
            )

        def load_tile(kr, ok, t):
            """Tile t of a page: this lane's 16 B of codes (token 16 t + col, dims 32 kl ..) and its two group scales;
            ``ok`` false: no memory access."""
            c = Vec(
                buffer_ops.buffer_load(
                    kr,
                    code_off,
                    vec_width=4,
                    dtype=fx.Int32,
                    mask=ok,
                    soffset_bytes=t * 1024,
                    cache_modifier=NT,
                )
            )
            sc = fx.Int32(
                fx.Int16(
                    buffer_ops.buffer_load(
                        kr,
                        scale_off,
                        vec_width=1,
                        dtype=fx.Int16,
                        mask=ok,
                        soffset_bytes=PAGE_CODES + t * 128,
                        cache_modifier=NT,
                    )
                ).to(fx.Int32)
            )
            return c, sc

        def dequant_tile(c, sc):
            """The A fragment of a tile (e4m3 keys, even dims then odd dims per 8)."""
            t0 = lut_load(sc & 0xFF)
            t1 = lut_load((sc >> 8) & 0xFF)
            f = []
            for e in range_constexpr(4):
                lo, hi = _dequant8(fx.Int32(Vec(c)[e]), *(t0 if e < 2 else t1))
                f += [lo, hi]
            return Vec.from_elements(f, fx.Int32)

        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(float("-inf"))

        def frag_max(frags, qfs, masked, lims):
            """Per column group (B operands qfs[g]): the max over this lane's 32 scores (4 tokens x 8 tiles) of its
            column, then over the column's 4 lane groups; tile t's results are reduced under tile t + 1's MFMA."""
            ng = len(qfs)
            runs = [neg_inf] * ng
            accs = [[None] * ng for _ in range(MT)]

            def reduce(t, runs):
                out = []
                for g in range_constexpr(ng):
                    run = runs[g]
                    v = accs[t][g]
                    for e in range_constexpr(4):
                        x = v[e]
                        if const_expr(masked):
                            x = (lims[g] >= t * 16 + e).select(x, neg_inf)
                        run = _maxf_nn(run, x)
                    out.append(run)
                return out

            for t in range_constexpr(MT):
                for g in range_constexpr(ng):
                    accs[t][g] = _mfma_fp8_16x16x128(frags[t], qfs[g], zero4)
                if const_expr(t >= 1):
                    runs = reduce(t - 1, runs)
            runs = reduce(MT - 1, runs)
            return [_xlane_max4(r) for r in runs]

        sent_init = fx.Float32(SENTINEL_INIT)
        sent_local = fx.Float32(SENTINEL_LOCAL)

        def row_value(mx, seq, q_off, b):
            """AITER's sentinels on block b of the row at q_off (a block both init and local is local)."""
            v1 = seq - QL + q_off + 1  # the row's visible tokens
            rb = ((v1 < 0).select(fx.Int32(0), v1) + (BLK - 1)) // BLK
            l0 = rb - i32_local
            local0 = (l0 < 0).select(fx.Int32(0), l0)
            vis = b < rb
            v = (vis & (b < i32_init)).select(sent_init, mx)
            return (vis & (b >= local0)).select(sent_local, v)

        def put(i_row, jpos, v, cond, addr):
            """Score v of Q row i_row at the unit's position jpos: into the LDS buffer, or straight to memory
            (unbuffered or per-wave mode). Branch-free: other lanes write a dummy slot."""
            if const_expr(not buffered):
                buffer_ops.buffer_store(v, sr, addr, mask=cond)
            else:
                buffer_ops.buffer_store(v, sr, addr, mask=cond & direct)
                idx = (cond & to_lds).select(
                    i_row * SB_STRIDE + jpos, QROWS * SB_STRIDE + lane
                )
                (scb + idx).store(_bits(v))  # a call: `scb[idx] = v` reads as rebinding

        def flush(ub):
            """The unit at block ub: every present request's rows, 4 consecutive blocks per thread and pass (scalar
            stores: a vector-store path took 161 instead of 118 VGPRs)."""
            NCH = UMAX // 4
            k = fx.Int32(0)
            while k < fx.Int32(QROWS * NCH // (64 * NW)):
                item = tx + k * (64 * NW)
                i = item // NCH
                c = item % NCH
                rr = i // R
                jc = i % R
                qoff = jc // H
                nb_c = (fx.Int32(seq_l[rr]) + (BLK - 1)) // BLK
                b0c = ub + c * 4
                ok = (rr < nr) & (qoff < QL) & (c * 4 < i32_unit)
                addr = (jc % H) * i32_s_sh + ((req0 + rr) * QL + qoff) * i32_s_st + b0c
                for e in range_constexpr(4):
                    v = _as_f32(fx.Int32(scb[i * SB_STRIDE + c * 4 + e]))
                    buffer_ops.buffer_store(
                        v,
                        sr,
                        addr + e,
                        mask=ok & (b0c + e < nb_c) & (c * 4 + e < i32_unit),
                    )
                k = k + 1

        def score_cgs(frags, hm, masked_any, cgs, b, jpos):
            """Column groups cgs (Q row i = cg * 16 + col: request i // R, column j = i % R = query row j // H, head
            j % H): scores, the causal mask where a hit request is at its last blocks, sentinels."""
            info = []
            for cg in cgs:
                i = cg * 16 + col
                crr = i // R
                st_j = i % R
                info.append((cg, crr, st_j // H, st_j % H))
            qfs = [q_frag(cg) for cg, _, _, _ in info]
            seqs = [fx.Int32(seq_l[crr]) for _, crr, _, _ in info]
            mxs = [neg_inf for _ in info]
            if masked_any:
                lims = [
                    seq_c - QL + qoff - b * BLK - kl * 4
                    for seq_c, (_, _, qoff, _) in zip(seqs, info)
                ]
                mxs = frag_max(frags, qfs, True, lims)
            else:
                mxs = frag_max(frags, qfs, False, None)
            for mx, seq_c, (cg, crr, qoff, head) in zip(mxs, seqs, info):
                hit = ((hm >> fx.Int64(crr)) & fx.Int64(1)) != fx.Int64(0)
                ov = row_value(mx, seq_c, qoff, b)
                otq = (req0 + crr) * QL + qoff
                put(
                    cg * 16 + col,
                    jpos,
                    ov,
                    hit & (kl == 0) & (qoff < QL),
                    head * i32_s_sh + otq * i32_s_st + b,
                )

        def score_groups(frags, hm, mmask, b, jpos):
            """Every column group holding the page (requests hm)."""
            left_m = hm
            while left_m != fx.Int64(0):
                lead_req = fx.Int32(fx.math.cttz(left_m).to(fx.Int32))
                if const_expr(CPR == 1):
                    cg = lead_req // RPC
                    gbits = fx.Int64(GMASK) << fx.Int64(cg * RPC)
                    left_m = left_m & ~gbits
                    score_cgs(frags, hm, (mmask & gbits) != fx.Int64(0), [cg], b, jpos)
                else:
                    gbits = fx.Int64(1) << fx.Int64(lead_req)
                    left_m = left_m & ~gbits
                    masked_any = (mmask & gbits) != fx.Int64(0)
                    score_cgs(
                        frags,
                        hm,
                        masked_any,
                        [lead_req * CPR + g for g in range(CPR)],
                        b,
                        jpos,
                    )

        # The workgroup's waves take the same positions, in units of i32_unit consecutive blocks: unit split, + splits,
        # ...; page k of the workgroup's position `it` (lane order of first use) goes to wave (k + it) % NW. With fewer
        # requests than waves per chunk (per_wave), wave w takes its own units split * NW + w, + splits * NW: walking
        # the shared positions cost a dependent table load per position.
        if const_expr(device_positions):
            # The host plans from max_seq_len (graphs: max_model_len). With every block position on its own workgroup
            # (device lengths), shared positions put one page per wave where per-wave positions walk all 2-3 requests'
            # pages; from two positions per workgroup per-wave positions win. Per-wave chunks hold < NW requests: lanes
            # 0..NW-2 (0 past nr).
            nb_max = fx.Int32(fx.rocdl.readlane(T.i32, nblk_r, fx.Int32(0)))
            for r in range_constexpr(1, NW - 1):
                nb_r = fx.Int32(fx.rocdl.readlane(T.i32, nblk_r, fx.Int32(r)))
                nb_max = (nb_r > nb_max).select(nb_r, nb_max)
            shared_pos = (
                (i32_unit == fx.Int32(0)) & (nb_max <= i32_splits) & (nr >= 2)
            )
            per_wave = per_wave & ~shared_pos
        u_step = per_wave.select(i32_splits * NW, i32_splits)
        ub = per_wave.select(split * NW + wave, split) * usz
        it = fx.Int32(0)
        while fx.Int64(fx.rocdl.ballot(T.i64, ub < nblk_r)) != fx.Int64(0):
            b = ub
            jpos = fx.Int32(0)
            while (jpos < usz) & (
                fx.Int64(fx.rocdl.ballot(T.i64, b < nblk_r)) != fx.Int64(0)
            ):
                in_r = b < nblk_r
                skip = (b < rbf_r) & ((b < i32_init) | (b >= loc_last_r))
                need_r = in_r & ~skip
                bid = fx.Int32(
                    buffer_ops.buffer_load(
                        btr,
                        need_r.select((req0 + lane) * i32_bt_stride + b, fx.Int32(0)),
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                )
                bid = need_r.select(bid, fx.Int32(-1))
                # sentinel-only blocks (wave 0 stores)
                if (fx.Int64(fx.rocdl.ballot(T.i64, in_r & skip)) != fx.Int64(0)) & (
                    (wave == 0) | per_wave
                ):
                    # a dynamic loop: unrolled, its QL * H stores' addresses are
                    # all computed up front (fp8 chunked kernel 154 -> 187 VGPRs)
                    jr = fx.Int32(0)
                    while jr < fx.Int32(QL * H):
                        sv = row_value(neg_inf, seq_r, jr // H, b)
                        stq = (req0 + lane) * QL + jr // H
                        put(
                            lane * R + jr,
                            jpos,
                            sv,
                            in_r & skip,
                            (jr % H) * i32_s_sh + stq * i32_s_st + b,
                        )
                        jr = jr + 1
                pend = fx.Int64(fx.rocdl.ballot(T.i64, need_r))
                mmask = fx.Int64(
                    fx.rocdl.ballot(T.i64, need_r & (b * BLK + (BLK - 1) > fm_r))
                )
                # this wave's pages, as the mask of their first lanes
                mine = fx.Int64(0)
                k = it
                while pend != fx.Int64(0):
                    lead = fx.Int32(fx.math.cttz(pend).to(fx.Int32))
                    page = fx.Int32(fx.rocdl.readlane(T.i32, bid, lead))
                    pend = pend & ~fx.Int64(fx.rocdl.ballot(T.i64, bid == page))
                    mine = mine | (((k % NW) == wave) | per_wave).select(
                        fx.Int64(1) << fx.Int64(lead), fx.Int64(0)
                    )
                    k = k + 1
                ok0 = mine != fx.Int64(0)
                l0 = fx.Int32(fx.math.cttz(mine).to(fx.Int32))
                kr0 = page_rsrc(
                    fx.Int32(fx.rocdl.readlane(T.i32, bid, ok0.select(l0, fx.Int32(0))))
                )
                c0, s0 = load_tile(kr0, ok0, 0)
                c1, s1 = load_tile(kr0, ok0, 1)
                c2, s2 = load_tile(kr0, ok0, 2)
                c3, s3 = load_tile(kr0, ok0, 3)
                c4, s4 = load_tile(kr0, ok0, 4)
                c5, s5 = load_tile(kr0, ok0, 5)
                c6, s6 = load_tile(kr0, ok0, 6)
                c7, s7 = load_tile(kr0, ok0, 7)
                while mine != fx.Int64(0):
                    lead = fx.Int32(fx.math.cttz(mine).to(fx.Int32))
                    page = fx.Int32(fx.rocdl.readlane(T.i32, bid, lead))
                    hm = fx.Int64(fx.rocdl.ballot(T.i64, bid == page))
                    mine = mine & ~(fx.Int64(1) << fx.Int64(lead))
                    okn = mine != fx.Int64(0)
                    ln = fx.Int32(fx.math.cttz(mine).to(fx.Int32))
                    krn = page_rsrc(
                        fx.Int32(
                            fx.rocdl.readlane(T.i32, bid, okn.select(ln, fx.Int32(0)))
                        )
                    )
                    # tile t of this page -> keys, then tile t of the next page
                    # loads into the freed registers (the barriers keep LLVM from
                    # hoisting the loads, which copies the carried registers and
                    # waits at the loop latch)
                    cs = [c0, c1, c2, c3, c4, c5, c6, c7]
                    ss = [s0, s1, s2, s3, s4, s5, s6, s7]
                    fs = []
                    for t in range_constexpr(MT):
                        fs.append(dequant_tile(cs[t], ss[t]))
                        cs[t], ss[t] = load_tile(krn, okn, t)
                        fx.rocdl.sched_barrier(0)
                    score_groups(fs, hm, mmask, b, jpos)
                    c0, c1, c2, c3, c4, c5, c6, c7 = cs
                    s0, s1, s2, s3, s4, s5, s6, s7 = ss
                b = b + 1
                jpos = jpos + 1
                it = it + 1
            if const_expr(buffered):  # noqa: SIM102 (the outer if is static)
                if to_lds:
                    fx.gpu.barrier()  # every wave's scores of the unit are in LDS
                    flush(ub)
                    fx.gpu.barrier()  # before the next unit's scores overwrite them
            ub = ub + u_step * usz

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_q_st: fx.Int32,
        i32_q_sh: fx.Int32,
        i64_q_bytes: fx.Int64,
        i32_s_sh: fx.Int32,
        i32_s_st: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_chunk: fx.Int32,
        i32_nch: fx.Int32,
        i32_splits: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_per_wave: fx.Int32,
        i32_unit: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_lut,
            arg_score,
            arg_bt,
            arg_seq,
            i64_page_stride,
            i32_q_st,
            i32_q_sh,
            i64_q_bytes,
            i32_s_sh,
            i32_s_st,
            i64_s_bytes,
            i32_bt_stride,
            i64_bt_bytes,
            i32_num_reqs,
            i32_chunk,
            i32_nch,
            i32_splits,
            i32_init,
            i32_local,
            i32_per_wave,
            i32_unit,
        ).launch(
            grid=(fx.Int64(i32_nch * i32_splits), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
