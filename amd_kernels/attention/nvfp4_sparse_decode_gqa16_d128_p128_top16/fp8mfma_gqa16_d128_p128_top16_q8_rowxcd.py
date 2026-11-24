# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``fp8/kernel.py`` at 8 rows per request, 16 query heads, 1 KV head, 4 splits, rows spread over the 8 XCDs.

What ``bf16mfma_gqa16_d128_p128_top16_q8_rowxcd.py`` is to kernel.py: row bx runs on XCD bx % 8 and a row's splits share
it; the first block's loads issue before the table barrier, and the V table words are consumed together. The Q / K / V /
P roundings and every row's sum order are fp8/kernel.py's.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .fp8.kernel import P_SCALE, REF_SPAN, _k_fp8x8, _mfma128, _mfma_fp8, _pack_fp8x8, _q_fp8x8
from .kernel import (
    AGENT_SCOPE, BLK, D, DATA_B, HG, KPW, NDT, NKS, NW, PART_F, PRELOAD_HINTS, RAW, SCALE_B, _as_f32, _bits, _exp2,
    _f16x2_e4m3, _global_atomic_add, _global_i32_ptr, _maxf_nn, _mfma, _mxfp8_group, _pk_bf16, _xlane4, dequant_f16,
)


def compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_q8_rowxcd(
    splits: int = 1,
    topk: int = 16,
    xcds: int = 0,
    quant: bool = False,
    tail: bool = False,
):
    """The launcher; the host's ``fp8/kernel.py`` key at splits 4, topk 16, xcds 8, no tail."""
    assert splits == 4 and topk == 16 and xcds == 8 and not tail
    assert 1 <= topk <= 64 and splits >= 1
    assert not tail or xcds > 0, "the tail grid places a pair's splits by XCD"
    SPLITS, TOPK, XCDS, TAIL = splits, topk, xcds, tail
    PART_SCOPE = 0 if XCDS else AGENT_SCOPE
    exact_fp16 = False  # (the bf16 kernel's fp16 branches stay dead here)
    mfma = "fp8"  # the MFMA operand dtype
    name = (
        f"nvfp4_{mfma}mfma_sparse_gqa16_d128_p128_top{TOPK}_q8_rowxcd_s{SPLITS}"
        + (f"_xcd{XCDS}" if XCDS else "")
        + ("_mxfp8" if quant else "")
        + ("_tail" if tail else "")
    )

    name += "_vlut"

    @fx.struct
    class Shared:
        # partial O per (wave, d-tile, lane) as 16 B tiles, then (max, sum) per (wave, lane)
        o: fx.Array[fx.Int32, NW * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * 2 * 64]
        # the e4m3 magnitude table (lo, hi word per scale byte); its 8 B alignment folds the LDS offset into ds_read_b64
        lut: fx.Array[fx.Int32, 512, 8]
        ticket: fx.Array[fx.Int32, 4]

    def store_out(arg_out, i32_rows, i32_H, row, kvh, lane, acc, inv):
        """O of ``row`` (``acc`` x ``inv``) in bf16."""
        n, g = lane % 16, lane // 16
        # lane (n, g) holds dims 8 (4 g + e) + dt of head n: per e, 8 consecutive dims
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        o_row = (row * i32_H + kvh * HG + n) * (D // 2) + g * 16
        for e in range_constexpr(4):
            val = Vec.from_elements(
                [
                    _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                    for j in range_constexpr(4)
                ],
                fx.Int32,
            )
            buffer_ops.buffer_store(val, outr, o_row + e * 4, mask=row < i32_rows)

    def store_q(arg_qc, arg_qs, i32_rows, i32_H, i32_tile, row, kvh, lane, acc, inv):
        """The o_proj input of ``row``: MXFP8 codes of O (row-major, or the (16, 16) tiles when ``i32_tile`` is 16)
        and e8m0 scales in AITER's layout."""
        n, g = lane % 16, lane // 16
        words = [
            _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
            for e in range_constexpr(4)
            for j in range_constexpr(4)
        ]
        dws, sb = _mxfp8_group(words)
        k_row = i32_H * D  # row width (elements)
        head = kvh * HG + n
        k0 = head * D + g * 32  # the lane's first dim
        rows16 = (i32_rows + 15) // 16 * 16
        qcr = buffer_ops.create_buffer_resource_from_addr(
            arg_qc, num_records_bytes=fx.Int64(rows16) * fx.Int64(k_row)
        )
        shuffled = i32_tile > 0
        # (16, 16) tiles: byte (m, k) at (m//16)*16K + (k//64)*1024 + ((k%64)//16)*256 + (m%16)*16 + k%16; the lane's
        # 32 bytes are two 16 B runs 256 B apart
        off_t = (
            (row // 16) * (16 * k_row)
            + (k0 // 64) * 1024
            + ((k0 % 64) // 16) * 256
            + (row % 16) * 16
        ) // 4
        off_l = (row * k_row + k0) // 4
        off = shuffled.select(off_t, off_l)
        step = shuffled.select(fx.Int32(64), fx.Int32(4))
        for h in range_constexpr(2):
            val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
            buffer_ops.buffer_store(val, qcr, off + h * step, mask=row < i32_rows)
        # scale (m, g): [M//32, K//256, klane 4, mlane 16, kpack 2, mpack 2]
        gi = head * 4 + g
        s_off = (
            ((((row // 32) * (k_row // 256) + gi // 8) * 4 + gi % 4) * 16 + row % 16)
            * 2
            + (gi % 8) // 4
        ) * 2 + (row % 32) // 16
        rows32 = (i32_rows + 31) // 32 * 32
        qsr = buffer_ops.create_buffer_resource_from_addr(
            arg_qs, num_records_bytes=fx.Int64(rows32) * fx.Int64(k_row // 32)
        )
        buffer_ops.buffer_store(
            sb.to(fx.Int32).to(fx.Int8), qsr, s_off, mask=row < i32_rows
        )

    # the arguments of the first (dependent) loads lead: the first 56 bytes arrive preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,  # rows per request
        i32_topk_hs: fx.Int32,  # topk stride over KV heads (elements)
        i32_bt_s: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        arg_seq: fx.Int64,  # [reqs] int32
        arg_topk: fx.Int64,  # [Hkv, >= rows, TOPK] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_lut: fx.Int64,  # [256] int64 K magnitude tables
        arg_q: fx.Int64,  # [rows, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 * Hkv, 128, 72] uint8
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_out: fx.Int64,  # [rows, H, 128] bf16
        arg_part: fx.Int64,  # splits: [grid rows * Hkv, S, 64, PART_F] fp32
        arg_cnt: fx.Int64,  # splits: [grid rows * Hkv] int32 tickets, zero
        arg_qc: fx.Int64,  # quant: [ceil(rows / 16) * 16, H * 128] e4m3 codes
        arg_qs: fx.Int64,  # quant: [ceil(rows / 32) * 32, H * 4] e8m0 scales
        i64_part_bytes: fx.Int64,
        i32_page_byte: fx.Int32,  # cache page stride (bytes)
        i32_tile: fx.Int32,  # quant: 0 row-major codes, 16 the tile layout
        f32_c: fx.Float32,  # sm_scale * log2(e)
        i32_heavy: fx.Int32,  # tail: whole-round pairs (a multiple of XCDS)
        i32_tpx: fx.Int32,  # tail: tail pairs per XCD
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        i32_H = fx.Int32(16)
        i32_hkv = fx.Int32(1)
        tx = fx.thread_idx.x
        bx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = _global_i32_ptr(arg_topk)
        bt = _global_i32_ptr(arg_bt)
        seq_p = _global_i32_ptr(arg_seq)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        if const_expr(not exact_fp16):
            lut_g = _global_i32_ptr(arg_lut)
            smem.lut.ptr[tx] = fx.Int32(lut_g[tx])
            smem.lut.ptr[tx + 256] = fx.Int32(lut_g[tx + 256])
        lut64 = fx.recast_iter(fx.Int64, smem.lut.ptr)

        def tab(sc):
            """(lo, hi) words of scale byte ``sc``'s table: one ds_read_b64."""
            pair = Vec.from_elements([fx.Int64(lut64[sc])], fx.Int64).bitcast(fx.Int32)
            return [pair[0], pair[1]]

        split = fx.block_idx.z
        if const_expr(TAIL):
            # 1-D grid. Workgroups [0, heavy) are the pairs p = row * Hkv + kvh in [0, heavy), one split each, XCD x
            # taking the contiguous pairs [x heavy / XCDS, (x + 1) heavy / XCDS) (a request's pairs stay together).
            # Then the tail pairs heavy + x tpx + j at SPLITS splits each, all splits of a pair on XCD x (their
            # partials meet in its L2). Pairs past rows x Hkv are padding.
            xcd = bx % XCDS
            heavy = bx < i32_heavy
            t_slot = (bx - i32_heavy) // XCDS
            tid_t = xcd * i32_tpx + t_slot // SPLITS  # the tail pair (scratch row)
            pair = heavy.select(
                xcd * (i32_heavy // XCDS) + bx // XCDS, i32_heavy + tid_t
            )
            split = heavy.select(fx.Int32(0), t_slot % SPLITS)
            split = fx.Int32(fx.rocdl.readfirstlane(T.i32, split))
            tid_t = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid_t))
            heavy_i = fx.Int32(
                fx.rocdl.readfirstlane(T.i32, heavy.select(fx.Int32(1), fx.Int32(0)))
            )
            row = fx.Int32(fx.rocdl.readfirstlane(T.i32, pair // i32_hkv))
            kvh = fx.Int32(fx.rocdl.readfirstlane(T.i32, pair % i32_hkv))
        else:
            if const_expr(XCDS > 0):
                # grid.x is a multiple of XCDS: row bx and all its z splits run on XCD bx % 8
                row = bx
            else:
                row = bx
        row_ok = row < i32_rows
        rowc = row_ok.select(row, fx.Int32(0))
        req = rowc // 8
        pos = fx.Int32(seq_p[req]) - 8 + rowc % 8
        # a graph's padding rows (seq_len 0) and the grid's padding see nothing
        pos = (row_ok & (pos >= 0)).select(pos, fx.Int32(-1))
        pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pos))
        req = fx.Int32(fx.rocdl.readfirstlane(T.i32, req))
        self_blk = (pos >= 0).select(pos // BLK, fx.Int32(-1))

        # lane = top-k slot: its logical block and page (-1: none)
        slot_ok = lane < TOPK
        blk = fx.Int32(
            topk[kvh * i32_topk_hs + rowc * TOPK + slot_ok.select(lane, fx.Int32(0))]
        )
        ok = slot_ok & (blk >= 0) & (blk <= self_blk)
        page = fx.Int32(bt[req * i32_bt_s + ok.select(blk, fx.Int32(0))])
        page = ok.select(page, fx.Int32(-1))
        blk = ok.select(blk, fx.Int32(-1))
        vmask = fx.Int64(fx.rocdl.ballot(T.i64, ok))
        n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))
        if const_expr(SPLITS > 1):
            # this split's blocks: ranks [lo, lo + chunk) in slot order
            below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
            rank = fx.Int32(fx.math.ctpop(vmask & below).to(fx.Int32))
            chunk = (n_v + (SPLITS - 1)) // SPLITS
            if const_expr(TAIL):
                chunk = (heavy_i == fx.Int32(1)).select(n_v, chunk)  # whole row
            lo = split * chunk
            vmask = fx.Int64(
                fx.rocdl.ballot(T.i64, ok & (rank >= lo) & (rank < lo + chunk))
            )
            n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))

        # causal limit of this wave's key offsets in the own block; Q^T
        lim = pos % BLK - w * KPW
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        q_row = (rowc * i32_H + kvh * HG + n) * (D // 2) + g * 16
        qf = [
            fx.Vector(
                buffer_ops.buffer_load(qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32)
            )
            for ks in range_constexpr(NKS)
        ]

        q8 = []
        for ks in range_constexpr(NKS):
            q8 += _q_fp8x8([qf[ks][i] for i in range_constexpr(4)], True)
        q8 = Vec.from_elements(q8, fx.Int32)  # B of the 16x16x128 scores MFMA

        # block offsets (bytes) of this KV head's K data / scales, V data / scales
        side_b = i32_hkv * (DATA_B + SCALE_B)
        k_data = kvh * DATA_B
        k_scl = i32_hkv * DATA_B + kvh * SCALE_B
        v_data = side_b + kvh * DATA_B
        v_scl = side_b + i32_hkv * DATA_B + kvh * SCALE_B
        # lane map: K rows 32 w + 8 (n // 4) + 4 st + n % 4, bytes [16 g, 16 g + 16) (dims 32 g..), their scales
        # (groups 2 g, 2 g + 1) one dword; V keys 32 w + 8 g + j of dims 8 n.. one 32-byte tile, their scales (group
        # n // 2) 8 contiguous bytes
        k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
        k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
        k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
        v_load = (v_data + ((w * KPW + 8 * g) // 8 * 16 + n) * 32) // 4
        v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4

        def load_block(pg):
            """This lane's NVFP4 dwords of page ``pg`` (RAW): K rows st = 0, 1 (4 + 4), their scales (1), V dims
            8 n + dt (8), the scales of their keys (2)."""
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_cache + fx.Int64(pg) * fx.Int64(i32_page_byte),
                num_records_bytes=i32_page_byte,
            )
            raw = []
            for st in range_constexpr(2):
                kd = fx.Vector(
                    buffer_ops.buffer_load(pr, k_load[st], vec_width=4, dtype=fx.Int32)
                )
                raw += [kd[i] for i in range_constexpr(4)]
            raw.append(
                fx.Int32(
                    buffer_ops.buffer_load(
                        pr, k_scale_load, vec_width=1, dtype=fx.Int32
                    )
                )
            )
            for h in range_constexpr(2):
                vd = fx.Vector(
                    buffer_ops.buffer_load(
                        pr, v_load + 4 * h, vec_width=4, dtype=fx.Int32
                    )
                )
                raw += [vd[i] for i in range_constexpr(4)]
            vs = fx.Vector(
                buffer_ops.buffer_load(pr, v_scale_load, vec_width=2, dtype=fx.Int32)
            )
            raw += [vs[i] for i in range_constexpr(2)]
            return raw

        def dequant_k(raw):
            """K frags kf[st][ks] of a block (4 dwords of bf16 pairs each)."""
            if const_expr(exact_fp16):
                kf = []
                for st in range_constexpr(2):
                    ks = _f16x2_e4m3(raw[8], st == 1)
                    kf.append(
                        [
                            Vec.from_elements(
                                dequant_f16(
                                    raw[st * 4 + j],
                                    [
                                        Vec.from_elements(
                                            [ks[j // 2], ks[j // 2]], fx.Float16
                                        )
                                        for _ in range_constexpr(4)
                                    ],
                                ),
                                fx.Int32,
                            )
                            for j in range_constexpr(NKS)
                        ]
                    )
                return kf
            kf = []
            for st in range_constexpr(2):
                tabs = []
                for grp in range_constexpr(2):
                    sb = raw[8].shrui(fx.Int32(8 * (2 * st + grp))) & fx.Int32(0xFF)
                    tabs.append(tuple(tab(sb)))
                words = []
                for ks in range_constexpr(NKS):
                    words += _k_fp8x8(raw[st * 4 + ks], *tabs[ks // 2], True)
                kf.append(Vec.from_elements(words, fx.Int32))  # A of the scores MFMA
            return kf

        def dequant_v(raw, cur_blk):
            """V^T operands of a block: the eight e4m3 key bytes of each output dim (one i64), through the table."""
            if const_expr(exact_fp16):
                sp = []
                for j in range_constexpr(4):
                    pair = _f16x2_e4m3(raw[17 + j // 2], j % 2 == 1)
                    # unwritten tail scale bytes can be NaN, and P = 0 cannot mask a NaN V
                    sp.append(
                        Vec.from_elements(
                            [
                                (
                                    cur_blk * BLK + w * KPW + 8 * g + 2 * j + k <= pos
                                ).select(pair[k], fx.Float16(0.0))
                                for k in range_constexpr(2)
                            ],
                            fx.Float16,
                        )
                    )
                return [
                    Vec.from_elements(dequant_f16(raw[9 + dt], sp), fx.Int32)
                    for dt in range_constexpr(NDT)
                ]

            def perm(hi, lo, sel):
                return fx.Int32(fx.rocdl.perm_b32(hi, lo, sel))

            x = [raw[9 + dt] for dt in range_constexpr(8)]
            for dt in range_constexpr(4):
                a, b = x[dt], x[dt + 4]
                x[dt] = perm(b, a, fx.Int32(0x05040100))
                x[dt + 4] = perm(b, a, fx.Int32(0x07060302))
            for r in (0, 1, 4, 5):
                a, b = x[r], x[r + 2]
                x[r] = perm(b, a, fx.Int32(0x06020400))
                x[r + 2] = perm(b, a, fx.Int32(0x07030501))
            vt = [
                tab((raw[17 + k // 4] >> (8 * (k % 4))) & 255)
                for k in range_constexpr(8)
            ]
            # a tied vector identity makes every table word ready before its first use (no serial LDS waits)
            all_tabs = Vec.from_elements(
                [vt[k][j] for k in range_constexpr(8) for j in range_constexpr(2)],
                fx.Int32,
            )
            all_tabs = Vec(_llvm.inline_asm(
                Vec.make_type(16, fx.Int32), [all_tabs.ir_value()], "",
                "=v,0,~{memory}", has_side_effects=True,
            ))
            vt = [[all_tabs[2 * k + j] for j in range_constexpr(2)] for k in range_constexpr(8)]
            z = []
            for kp in range_constexpr(4):
                even, odd = [], []
                for par in range_constexpr(2):
                    c = x[2 * kp + par]
                    te, to = vt[2 * kp], vt[2 * kp + 1]
                    even.append(
                        perm(te[1], te[0], c & 0x07070707) ^ ((c << 4) & -2139062144)
                    )
                    odd.append(
                        perm(to[1], to[0], (c >> 4) & 0x07070707) ^ (c & -2139062144)
                    )
                z.extend([even, odd])
            words = [[fx.Int32(0), fx.Int32(0)] for dt in range_constexpr(8)]
            for par in range_constexpr(2):
                for half in range_constexpr(2):
                    a, b, c, d = [z[4 * half + k][par] for k in range_constexpr(4)]
                    u0 = perm(b, a, fx.Int32(0x05010400))
                    u1 = perm(b, a, fx.Int32(0x07030602))
                    v0 = perm(d, c, fx.Int32(0x05010400))
                    v1 = perm(d, c, fx.Int32(0x07030602))
                    vals = [
                        perm(v0, u0, fx.Int32(0x05040100)),
                        perm(v0, u0, fx.Int32(0x07060302)),
                        perm(v1, u1, fx.Int32(0x05040100)),
                        perm(v1, u1, fx.Int32(0x07060302)),
                    ]
                    for i in range_constexpr(4):
                        words[par + 2 * i][half] = vals[i]
            vf = [Vec.from_elements(w, fx.Int32).bitcast(fx.Int64)[0] for w in words]
            return vf


        zero4 = Vec.filled(4, 0.0, fx.Float32)
        # offset (within this wave's 32 keys) of S^T tile st, row 4 g + e
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        first = (n_v > 0).select(
            fx.Int32(fx.math.cttz(vmask).to(fx.Int32)), fx.Int32(0)
        )

        def slot_at(mask):
            """The lowest slot left in ``mask`` (the first one again when empty)."""
            return (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )

        def page_of(slot):
            pg = fx.Int32(fx.rocdl.readlane(T.i32, page, slot))
            return (pg >= 0).select(pg, fx.Int32(0))

        def attend(kf, vf, cur_blk, live, m_t, r_t, l_t, o_t):
            """One dequantized block into the running (reference, max, sum, O): S^T, the own block's causal mask,
            this wave's 32 keys as one P chunk (P against the chunk's own max), O^T += V^T P^T. A block that is not
            ``live`` (or a chunk with no visible key) is masked whole: nothing moves, P is 0."""
            a = []
            for st in range_constexpr(2):
                acc = _mfma128(kf[st], q8, zero4)
                a += [acc[e] for e in range_constexpr(4)]
            diag = cur_blk == self_blk
            dead = live == fx.Int32(0)
            a = [
                (dead | (diag & (key_off[i] > lim))).select(neg_inf, a[i])
                for i in range_constexpr(8)
            ]
            mx = a[0]
            for i in range_constexpr(1, 8):
                mx = _maxf_nn(mx, a[i])
            # maxima in scaled units: max(a) * c == max(a * c) (c > 0)
            mc = _xlane4(mx, _maxf_nn) * c  # the chunk's max: P's reference
            r_new = _maxf_nn(r_t, mc)  # the running max
            m_new = (mc == neg_inf).select(
                m_t, _maxf_nn(mc, r_new - fx.Float32(REF_SPAN))
            )
            m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
            alpha = _exp2(m_t - m_use)
            p = [_exp2(fx.math.fma(x, c, -m_use)) for x in a]
            ps = p[0]
            for i in range_constexpr(1, 8):
                ps = ps + p[i]
            pf = _pack_fp8x8([x * fx.Float32(P_SCALE) for x in p])
            # alpha == 1 leaves O's bits unchanged: the fp16 path skips that rescale
            av = Vec.filled(4, alpha, fx.Float32)
            if const_expr(exact_fp16):
                if alpha != fx.Float32(1.0):
                    o_t = [o_t[dt] * av for dt in range_constexpr(NDT)]
                o_t = [_mfma(vf[dt], pf, o_t[dt], True) for dt in range_constexpr(NDT)]
            else:
                o_t = [_mfma_fp8(vf[dt], pf, o_t[dt] * av) for dt in range_constexpr(NDT)]
            return m_new, r_new, l_t * alpha + ps, o_t

        # two blocks (A, B) per iteration, the next A's loads in flight; an odd last B is masked. fp16 dequantizes V
        # after the next load, bf16 before it.
        init = [vmask, neg_inf, fx.Float32(0.0)]
        init += [zero4 for _ in range_constexpr(NDT)]
        init += load_block(page_of(slot_at(vmask)))
        if const_expr(not exact_fp16):
            fx.gpu.barrier()  # the table is in LDS; the first block's loads are in flight

        init.append(neg_inf)  # the running max (bounds the reference)
        n_st = 3 + NDT
        results = init
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index((n_v + 1) // 2),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            m_t = fx.Float32(st_[1])
            l_t = fx.Float32(st_[2])
            o_t = [Vec(st_[3 + dt]) for dt in range_constexpr(NDT)]
            raw_a = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            r_t = fx.Float32(st_[n_st + RAW])
            blk_a = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem)))
            rem_b = rem & (rem - fx.Int64(1))
            raw_b = load_block(page_of(slot_at(rem_b)))
            fx.rocdl.sched_barrier(0)
            kf_a = dequant_k(raw_a)
            if const_expr(not exact_fp16):
                vf_a = dequant_v(raw_a, blk_a)
            rem_n = rem_b & (rem_b - fx.Int64(1))
            fx.rocdl.sched_barrier(0)
            raw_n = load_block(page_of(slot_at(rem_n)))
            fx.rocdl.sched_barrier(0)
            if const_expr(exact_fp16):
                vf_a = dequant_v(raw_a, blk_a)
            m_t, r_t, l_t, o_t = attend(kf_a, vf_a, blk_a, fx.Int32(1), m_t, r_t, l_t, o_t)
            blk_b = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem_b)))
            live_b = (rem_b != fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))
            kf_b = dequant_k(raw_b)
            vf_b = dequant_v(raw_b, blk_b)
            m_t, r_t, l_t, o_t = attend(kf_b, vf_b, blk_b, live_b, m_t, r_t, l_t, o_t)
            state = [rem_n, m_t, l_t] + o_t + raw_n + [r_t]
            results = yield [fx.as_ir_value(x) for x in state]

        # combine the waves' partials (wave 0)
        o_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        ml = smem.ml.ptr
        ml[(w * 2) * 64 + lane] = fx.Float32(results[1])
        ml[(w * 2 + 1) * 64 + lane] = fx.Float32(results[2])
        for dt in range_constexpr(NDT):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(Vec(results[3 + dt]).bitcast(fx.Int32))
            fx.copy(lds_atom, r, fx.slice(o_lds, (None, (w * NDT + dt) * 64 + lane)))
        fx.gpu.barrier()
        if w == 0:
            ms = [fx.Float32(ml[(ww * 2) * 64 + lane]) for ww in range_constexpr(NW)]
            big = ms[0]
            for ww in range_constexpr(1, NW):
                big = _maxf_nn(big, ms[ww])
            big_raw = big
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_tot = l_tot + fx.Float32(ml[(ww * 2 + 1) * 64 + lane]) * wt[ww]
            l_tot = _xlane4(l_tot, lambda x, y: x + y)
            acc = [zero4 for _ in range_constexpr(NDT)]
            for ww in range_constexpr(NW):
                wv = Vec.filled(4, wt[ww], fx.Float32)
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        lds_atom,
                        fx.slice(o_lds, (None, (ww * NDT + dt) * 64 + lane)),
                        r,
                    )
                    acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
            if const_expr(SPLITS > 1):
                # this split's partial; the row's last split combines. In the tail grid the whole-round workgroups
                # (heavy_i) write directly.
                if const_expr(TAIL):
                    tid = tid_t
                    zsp = split
                    split_on = heavy_i == fx.Int32(0)
                else:
                    tid = row * i32_hkv + kvh
                    zsp = fx.block_idx.z
                    split_on = True
                if split_on:
                    pr = buffer_ops.create_buffer_resource_from_addr(
                        arg_part, num_records_bytes=i64_part_bytes
                    )
                    row_f = (tid * SPLITS + zsp) * 64 + lane
                    for dt in range_constexpr(NDT):
                        buffer_ops.buffer_store(
                            acc[dt].bitcast(fx.Int32),
                            pr,
                            row_f * PART_F + dt * 4,
                            cache_modifier=PART_SCOPE,
                        )
                    buffer_ops.buffer_store(
                        Vec.from_elements([_bits(big_raw), _bits(l_tot)], fx.Int32),
                        pr,
                        row_f * PART_F + NDT * 4,
                        cache_modifier=PART_SCOPE,
                    )
                    # the partial stores are done before the ticket is taken
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    if lane == 0:
                        smem.ticket.ptr[0] = _global_atomic_add(
                            arg_cnt + fx.Int64(tid) * 4, 1
                        )
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    if (
                        fx.Int32(
                            fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0]))
                        )
                        == SPLITS - 1
                    ):
                        rows_f = [
                            (tid * SPLITS + sp) * 64 + lane
                            for sp in range_constexpr(SPLITS)
                        ]
                        mls = [
                            Vec(
                                buffer_ops.buffer_load(
                                    pr,
                                    rf * PART_F + NDT * 4,
                                    vec_width=2,
                                    dtype=fx.Int32,
                                    cache_modifier=PART_SCOPE,
                                )
                            )
                            for rf in rows_f
                        ]
                        bigs = [_as_f32(x[0]) for x in mls]
                        top = bigs[0]
                        for sp in range_constexpr(1, SPLITS):
                            top = _maxf_nn(top, bigs[sp])
                        top = (top == neg_inf).select(fx.Float32(0.0), top)
                        ws = [fx.Float32(fx.math.exp2(b - top)) for b in bigs]
                        l_all = fx.Float32(0.0)
                        for sp in range_constexpr(SPLITS):
                            l_all = l_all + _as_f32(mls[sp][1]) * ws[sp]
                        inv_all = (l_all > fx.Float32(0.0)).select(
                            v_scale / (l_all * fx.Float32(P_SCALE)), fx.Float32(0.0)
                        )
                        acc_all = [zero4 for _ in range_constexpr(NDT)]
                        for sp in range_constexpr(SPLITS):
                            wv = Vec.filled(4, ws[sp], fx.Float32)
                            for dt in range_constexpr(NDT):
                                part = Vec(
                                    buffer_ops.buffer_load(
                                        pr,
                                        rows_f[sp] * PART_F + dt * 4,
                                        vec_width=4,
                                        dtype=fx.Int32,
                                        cache_modifier=PART_SCOPE,
                                    )
                                ).bitcast(fx.Float32)
                                acc_all[dt] = acc_all[dt] + part * wv
                        if const_expr(quant):
                            store_q(
                                arg_qc,
                                arg_qs,
                                i32_rows,
                                i32_H,
                                i32_tile,
                                row,
                                kvh,
                                lane,
                                acc_all,
                                inv_all,
                            )
                        else:
                            store_out(
                                arg_out,
                                i32_rows,
                                i32_H,
                                row,
                                kvh,
                                lane,
                                acc_all,
                                inv_all,
                            )
                        if lane == 0:
                            cnt_p = _global_i32_ptr(arg_cnt + fx.Int64(tid) * 4)
                            cnt_p[0] = fx.Int32(0)
                if const_expr(TAIL):  # noqa: SIM102 (compile-time, then runtime)
                    if heavy_i == fx.Int32(1):
                        inv = (l_tot > fx.Float32(0.0)).select(
                            v_scale / (l_tot * fx.Float32(P_SCALE)), fx.Float32(0.0)
                        )
                        if const_expr(quant):
                            store_q(
                                arg_qc,
                                arg_qs,
                                i32_rows,
                                i32_H,
                                i32_tile,
                                row,
                                kvh,
                                lane,
                                acc,
                                inv,
                            )
                        else:
                            store_out(
                                arg_out, i32_rows, i32_H, row, kvh, lane, acc, inv
                            )
            else:
                inv = (l_tot > fx.Float32(0.0)).select(
                    v_scale / (l_tot * fx.Float32(P_SCALE)), fx.Float32(0.0)
                )
                if const_expr(quant):
                    store_q(
                        arg_qc,
                        arg_qs,
                        i32_rows,
                        i32_H,
                        i32_tile,
                        row,
                        kvh,
                        lane,
                        acc,
                        inv,
                    )
                else:
                    store_out(arg_out, i32_rows, i32_H, row, kvh, lane, acc, inv)

    @flyc.jit
    def launch_fp8_q8_rowxcd(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_lut: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        arg_qc: fx.Int64,
        arg_qs: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_rows: fx.Int32,
        i32_tile: fx.Int32,
        i32_ql: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        i32_grid_x: fx.Int32,
        i32_heavy: fx.Int32,
        i32_tpx: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows,
            i32_ql,
            i32_topk_hs,
            i32_bt_s,
            i32_H,
            i32_hkv,
            arg_seq,
            arg_topk,
            arg_bt,
            arg_lut,
            arg_q,
            arg_cache,
            arg_ks,
            arg_vs,
            arg_out,
            arg_part,
            arg_cnt,
            arg_qc,
            arg_qs,
            i64_part_bytes,
            i32_page_b,
            i32_tile,
            f32_c,
            i32_heavy,
            i32_tpx,
        ).launch(
            grid=(i32_grid_x, 1, 1) if TAIL else (i32_grid_x, i32_hkv, SPLITS),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_fp8_q8_rowxcd.kernel_name = name
    launch_fp8_q8_rowxcd.compile_hints = PRELOAD_HINTS
    return launch_fp8_q8_rowxcd
