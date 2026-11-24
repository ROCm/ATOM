# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``router_mxfp4_quant_line_layout.py`` plus the shared expert's MXFP8 quant of the same bf16 fragments (one amax per
32 values), its scales in the dense GEMM's 32-row / 256-K tiled layout (gfx950)."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common.epilogue import (
    _lds_load_vec,
    _lds_store_vec,
)
from .common.launch import _run_compiled
from .router_mxfp4_quant import (
    _amax_bf16,
    _bf16_to_f32,
    _cvt_pk_fp4_bf16,
    _e8m0_e2m1_even_bf16,
    _i32_max,
    _swap16,
    _swap32,
    router_k_split,
)
from .shared_quant import _e8m0_fp8, _fp8
from .common.loaders import _as_f32
from amd_kernels.warm_up import launcher


BLOCK_M = 128  # rows per CTA (8 MFMA row tiles)
N_WAVES = 8  # one 16-expert n-tile per wave: E = 128
KT_PER_CHUNK = 2  # k-tiles (32 K each) per LDS stage / quant chunk / load batch: 153 vs 184 us with 4 at 16384 rows
LDS_ROW_B = KT_PER_CHUNK * 64 + 16  # padded A row in LDS (conflict-free reads)
LDS_SLOTS = 2  # A stages: write c + 1 while reading c
_LD_NT = 2  # non-temporal loads of y (read once per K half)
_ST_NT = 0x2  # non-temporal stores for the write-once logits
# A chunks in flight in registers: 2 is -0.5..-2% vs 3 (4: +1%) and keeps every loop-carried value in place
A_AHEAD = 2


@functools.cache
@launcher
def compile_router_mxfp4_mxfp8_quant_line_layout(K_SPLIT: int):
    """Per k-tile the hi, mid and lo terms' MFMAs, K split ``K_SPLIT`` ways."""
    H, E = 6144, 128
    KC = KT_PER_CHUNK
    assert E == 16 * N_WAVES and H % (K_SPLIT * 32 * KC) == 0
    KH = H // K_SPLIT  # K per CTA
    NCHUNK = KH // (32 * KC)
    ROW_DW = H // 2  # bf16 row in dwords
    SROW_B = H // 32  # scale row in bytes
    N_RT = BLOCK_M // 16
    N_THREADS = 64 * N_WAVES
    PIECES = BLOCK_M * KC * 4  # 16 B pieces of an A stage (row x 64 B per k-tile)
    PIECES_PER_LANE = PIECES // N_THREADS
    SEGS = KC * 4  # 16 B segments per row per stage
    SLOT_B = BLOCK_M * LDS_ROW_B
    assert PIECES % N_THREADS == 0 and LDS_SLOTS * SLOT_B <= 160 * 1024
    name = f"router_bf16x3_mxfp4_even_mxfp8_quant_h6144_e128_ks{K_SPLIT}"
    assert NCHUNK % 2 == 0  # the 2-chunk loop body
    # chunks per code / scale store, 4 (a row's 128 B of codes = one line) or 2: -6% at 32768 rows vs per-k-tile stores
    QC = 4 if NCHUNK % 4 == 0 else 2
    assert NCHUNK % QC == 0
    name += f"_store{QC * KC * 16}"
    # the next A load after the compute, except in the short split-16 body (same chunks and queue)
    A_AFTER_COMPUTE = K_SPLIT != 16
    name += "_weight_lines" + ("_a_after_compute" if A_AFTER_COMPUTE else "_a_ahead")
    QB = QC * KC  # scale bytes / 16 B code pieces per row and group
    Q4ST_ROW_B = QB * 16 + 16
    Q4ST_WAVE_B = 16 * Q4ST_ROW_B

    @fx.struct
    class Shared:
        a_lds: fx.Array[fx.Int8, LDS_SLOTS * SLOT_B, 16]
        q4_lds: fx.Array[fx.Int8, N_WAVES * Q4ST_WAVE_B, 16]

    @flyc.kernel(name=name, known_block_size=[N_THREADS, 1, 1])
    def kernel_router_quant(
        Y: fx.Tensor,
        W: fx.Tensor,
        Q4: fx.Tensor,
        S4: fx.Tensor,
        Q8: fx.Tensor,
        S8: fx.Tensor,
        LOGITS: fx.Tensor,
        n_rows: fx.Int32,
    ):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = tx // 64
        g = lane // 16  # K group of the A / B fragment, row group of C
        r16 = lane % 16
        row0 = (fx.block_idx.x // K_SPLIT) * BLOCK_M
        khalf = fx.block_idx.x % K_SPLIT
        k0_dw = khalf * (KH // 2)  # this CTA's K start in dwords
        lds = fx.SharedAllocator().allocate(Shared).peek()
        lds_base = fx.Int32(fx.ptrtoint(lds.a_lds.ptr))
        q4st_base = fx.Int32(fx.ptrtoint(lds.q4_lds.ptr)) + wave * Q4ST_WAVE_B
        y_rsrc = buffer_ops.create_buffer_resource(
            Y, max_size=False, num_records_bytes=n_rows * (H * 2)
        )
        w_rsrc = buffer_ops.create_buffer_resource(W, max_size=True)
        q4_rsrc = buffer_ops.create_buffer_resource(
            Q4, max_size=False, num_records_bytes=n_rows * (H // 2)
        )
        s4_rsrc = buffer_ops.create_buffer_resource(
            S4, max_size=False, num_records_bytes=n_rows * SROW_B
        )
        q8_rsrc = buffer_ops.create_buffer_resource(
            Q8, max_size=False, num_records_bytes=n_rows * H
        )
        s8_rsrc = buffer_ops.create_buffer_resource(
            S8, max_size=False, num_records_bytes=((n_rows + 31) // 32 * 32) * SROW_B
        )
        l_rsrc = buffer_ops.create_buffer_resource(
            LOGITS, max_size=False, num_records_bytes=n_rows * (K_SPLIT * E * 4)
        )
        # the A pieces this lane stages: piece p = tx + j * N_THREADS -> row p // SEGS, segment p % SEGS
        piece_rows = [(tx + j * N_THREADS) // SEGS for j in range(PIECES_PER_LANE)]
        piece_segs = [(tx + j * N_THREADS) % SEGS for j in range(PIECES_PER_LANE)]
        piece_gdw = [
            (row0 + piece_rows[j]) * ROW_DW + k0_dw + piece_segs[j] * 4
            for j in range(PIECES_PER_LANE)
        ]
        piece_lds = [
            lds_base + piece_rows[j] * LDS_ROW_B + piece_segs[j] * 16
            for j in range(PIECES_PER_LANE)
        ]
        # this lane's A fragment addresses in a slot (row rt * 16 + r16, K group g)
        frag_lds = [
            [
                lds_base + (rt * 16 + r16) * LDS_ROW_B + kt * 64 + g * 16
                for kt in range(KC)
            ]
            for rt in range(N_RT)
        ]
        # fp32 B [K/32, wave, half, g, r16, 4]: a wave load reads 64 adjacent 16-byte pieces, whole 128-byte lines
        b_row_dw = k0_dw * 2 * E + wave * (2 * 64 * 4) + lane * 4
        # the quant row this wave owns: row tile ``wave`` (row wave * 16 + r16)
        qrow = row0 + wave * 16 + r16
        qfrag_lds = [
            lds_base + (wave * 16 + r16) * LDS_ROW_B + kt * 64 + g * 16
            for kt in range(KC)
        ]

        def load_a(c):
            """This lane's staged pieces of chunk ``c`` (global -> registers); none past the CTA's last chunk."""
            k_dw = c * (KC * 16)
            return [
                buffer_ops.buffer_load(
                    y_rsrc,
                    piece_gdw[j] + k_dw,
                    vec_width=4,
                    dtype=fx.Int32,
                    mask=(c < NCHUNK).ir_value(),
                    cache_modifier=_LD_NT,
                )
                for j in range(PIECES_PER_LANE)
            ]

        def store_a(pieces, slot):
            for j in range_constexpr(PIECES_PER_LANE):
                _lds_store_vec(pieces[j], piece_lds[j] + slot * SLOT_B, 4)

        def load_b(c):
            """fp32 B fragments [half][kt] of chunk ``c``, this wave's n-tile (two 4 x f32 halves of the lane's 8 K)."""
            k_dw = c * (KC * 16) * 2 * E
            return [
                [
                    buffer_ops.buffer_load(
                        w_rsrc,
                        b_row_dw + k_dw + kt * 32 * E + hf * 256,
                        vec_width=4,
                        dtype=fx.Float32,
                    )
                    for kt in range(KC)
                ]
                for hf in range(2)
            ]

        def split_b(b, kt):
            """The k-tile's fp32 B as bf16 hi = RN(w), mid = RN(w - hi), lo = RN(w - hi - mid): the router's split."""
            w = Vec.from_elements(
                [fx.Float32(Vec(b[0][kt])[i]) for i in range(4)]
                + [fx.Float32(Vec(b[1][kt])[i]) for i in range(4)],
                fx.Float32,
            )
            hi = w.to(fx.BFloat16)
            r1 = w - _bf16_to_f32(hi)
            mid = r1.to(fx.BFloat16)
            lo = (r1 - _bf16_to_f32(mid)).to(fx.BFloat16)
            return hi, mid, lo

        def mfma(a4, b8, acc):
            return Vec(
                rocdl.mfma_f32_16x16x32_bf16(
                    T.vec(4, T.f32),
                    [
                        Vec(a4).bitcast(fx.BFloat16).ir_value(),
                        b8.ir_value(),
                        acc.ir_value(),
                        0,
                        0,
                        0,
                    ],
                )
            )

        sb_acc = []

        def quant_tile(c, hq, kt, a4):
            """Quantizes this wave's row tile at k-tile ``kt`` of chunk ``c`` (``a4``: this lane's A fragment of it)."""
            v = Vec(a4)  # 4 x i32 = 8 bf16 of this lane's K
            dw = [fx.Int32(v[d]) for d in range(4)]
            am = _amax_bf16(dw)
            lo32, hi32 = _swap32(am)
            am = _i32_max(lo32, hi32)
            ev16, od16 = _swap16(am)
            am = _i32_max(ev16, od16)
            grp = khalf * (KH // 32) + c * KC + kt
            sb4 = _e8m0_e2m1_even_bf16(am)
            scale4 = _as_f32(sb4 << 23)
            code4 = fx.Int32(0)
            for j in range_constexpr(4):
                code4 = _cvt_pk_fp4_bf16(code4, dw[j], scale4, j)
            _lds_store_vec(
                Vec.from_elements([code4], fx.Int32).ir_value(),
                q4st_base + r16 * Q4ST_ROW_B + (hq * KC + kt) * 16 + g * 4,
                1,
            )
            sb_acc.append(sb4)
            # The amax is already common to the four K lanes of this 32-value group.
            e8 = _e8m0_fp8(_as_f32(am << 16))
            scale8 = _as_f32(e8 << 23)
            pk8 = []
            for j in range_constexpr(2):
                a, b = dw[2*j], dw[2*j+1]
                z = _fp8(fx.Int32(0), _as_f32(a << 16), _as_f32(a & fx.Int32(-65536)), scale8, False)
                z = _fp8(z, _as_f32(b << 16), _as_f32(b & fx.Int32(-65536)), scale8, True)
                pk8.append(z)
            buffer_ops.buffer_store(Vec.from_elements(pk8, fx.Int32), q8_rsrc,
                                    (qrow * H + grp * 32 + g * 8) // 4)
            r32, c8 = qrow % 32, grp % 8
            scale_off = (
                ((qrow // 32) * (H // 256) + grp // 8) * 256
                + (c8 % 4) * 64 + (r32 % 16) * 4 + (c8 // 4) * 2 + r32 // 16
            )
            # The separate scale shuffle reads zero outside the valid input rows.
            e8 = (qrow < n_rows).select(e8, fx.Int32(0))
            buffer_ops.buffer_store(e8.to(fx.Int8), s8_rsrc, scale_off,
                                    mask=(g == 0), offset_is_bytes=True)

        def step(c, slot, hq, b, acc_hi, acc_lo):
            """One chunk: A fragments from LDS slot ``slot``, MFMAs; each wave also quantizes its row tile (16 B more
            LDS reads per k-tile)."""
            for kt in range_constexpr(KC):
                a_kt = [
                    _lds_load_vec(frag_lds[rt][kt] + slot * SLOT_B, 4)
                    for rt in range(N_RT)
                ]
                aq = _lds_load_vec(qfrag_lds[kt] + slot * SLOT_B, 4)
                # the k-tile's 9 LDS reads ahead of its MFMAs: LLVM sinks each to its MFMAs behind an lgkmcnt(0) (12
                # full waits per chunk); pinned 5 waits, 194 VGPRs, -7..-12% at 2048-32768 rows
                fx.rocdl.sched_barrier(0)
                b_hi, b_mid, b_lo = split_b(b, kt)
                for rt in range_constexpr(N_RT):
                    acc_hi[rt] = mfma(a_kt[rt], b_hi, acc_hi[rt])
                    acc_lo[rt] = mfma(a_kt[rt], b_mid, acc_lo[rt])
                    acc_lo[rt] = mfma(a_kt[rt], b_lo, acc_lo[rt])
                quant_tile(c, hq, kt, aq)
            return acc_hi, acc_lo

        def store_q(c):
            """After chunk ``c``, the last of a group of QC: the wave's 16 rows x QB * 16 B of MXFP4 codes (QB lanes per
            row) and QB scale bytes per row in dwords (lanes g == 0)."""
            assert len(sb_acc) == QB
            grp0 = khalf * (KH // 32) + (c - (QC - 1)) * KC
            words = []
            for w4 in range_constexpr(QB // 4):
                v = sb_acc[4 * w4]
                for j in range_constexpr(1, 4):
                    v = v | (sb_acc[4 * w4 + j] << (8 * j))
                words.append(v)
            sb_acc.clear()
            buffer_ops.buffer_store(
                Vec.from_elements(words, fx.Int32).ir_value() if len(words) > 1 else words[0],
                s4_rsrc,
                qrow * SROW_B + grp0,
                mask=(g == fx.Int32(0)).ir_value(),
                offset_is_bytes=True,
            )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            k4_b = grp0 * 16
            RPP = 64 // QB  # rows per pass
            rq = lane // QB
            for ps in range_constexpr(16 // RPP):
                v4 = _lds_load_vec(q4st_base + (ps * RPP + rq) * Q4ST_ROW_B + (lane % QB) * 16, 4)
                buffer_ops.buffer_store(
                    v4,
                    q4_rsrc,
                    (row0 + wave * 16 + ps * RPP + rq) * (H // 2) + k4_b + (lane % QB) * 16,
                    offset_is_bytes=True,
                )

        def flat_b(b):
            return [b[hf][kt] for hf in range(2) for kt in range(KC)]

        def unflat_b(vals):
            return [[vals[hf * KC + kt] for kt in range(KC)] for hf in range(2)]

        # prologue: stage chunk 0; A(1 .. DA) and B(0) in registers
        DA = A_AHEAD
        store_a(load_a(fx.Int32(0)), 0)
        a_q = [load_a(fx.Int32(1 + j)) for j in range_constexpr(DA)]
        b0 = load_b(fx.Int32(0))
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        n_b = 2 * KC
        n_a = PIECES_PER_LANE
        init_state = (
            [zero4.ir_value() for _ in range(2 * N_RT)]
            + flat_b(b0)
            + [v for aa in a_q for v in aa]
        )
        def chunk(c, h, hq, acc_hi, acc_lo, b, a_q):
            """Chunk c (parity h): write A(c + 1) to slot 1 - h (last read in chunk c - 1, behind the barrier), load
            A(c + 1 + DA) (after the compute with ``A_AFTER_COMPUTE``) and B(c + 1), compute chunk c, store a finished
            group's codes and scales, barrier."""
            store_a(a_q[0], 1 - h)
            a_new = a_q[0] if A_AFTER_COMPUTE else load_a(c + 1 + DA)
            b_next = load_b(c + 1)
            acc_hi, acc_lo = step(c, h, hq, b, acc_hi, acc_lo)
            if const_expr(hq == QC - 1):
                store_q(c)
            a_new = load_a(c + 1 + DA) if A_AFTER_COMPUTE else a_new
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            return acc_hi, acc_lo, b_next, a_q[1:] + [a_new]

        def unpack(state):
            acc_hi = [Vec(state[rt]) for rt in range(N_RT)]
            acc_lo = [Vec(state[N_RT + rt]) for rt in range(N_RT)]
            b = unflat_b(state[2 * N_RT : 2 * N_RT + n_b])
            o = 2 * N_RT + n_b
            a_q = [state[o + j * n_a : o + (j + 1) * n_a] for j in range(DA)]
            return acc_hi, acc_lo, b, a_q

        def pack(acc_hi, acc_lo, b, a_q):
            return (
                [v.ir_value() for v in acc_hi + acc_lo]
                + flat_b(b)
                + [v for aa in a_q for v in aa]
            )

        # loop-carried: the accumulators, this chunk's B and the next DA chunks' A. QC (even) chunks per iteration bring
        # the A queue and B back to their own registers (one chunk per iteration: 20 v_mov_b64 per chunk) and keep the
        # LDS slots and the group's store static; the last QC chunks follow the loop.
        for pi, state in range(  # type: ignore[call-overload]
            0, (NCHUNK - QC) // QC, 1, init=init_state
        ):
            acc_hi, acc_lo, b, a_q = unpack(state)
            for h in range_constexpr(QC):
                acc_hi, acc_lo, b, a_q = chunk(
                    fx.Int32(pi) * QC + h, h % 2, h, acc_hi, acc_lo, b, a_q
                )
            state = yield pack(acc_hi, acc_lo, b, a_q)
        acc_hi, acc_lo, b, a_q = unpack(state)
        for j in range_constexpr(QC - 1):
            acc_hi, acc_lo, b, a_q = chunk(
                fx.Int32(NCHUNK - QC + j), j % 2, j, acc_hi, acc_lo, b, a_q
            )
        state = pack(acc_hi, acc_lo, b, a_q)
        acc_hi = [Vec(state[rt]) for rt in range(N_RT)]
        acc_lo = [Vec(state[N_RT + rt]) for rt in range(N_RT)]
        b = unflat_b(state[2 * N_RT : 2 * N_RT + n_b])
        acc_hi, acc_lo = step(
            fx.Int32(NCHUNK - 1), (NCHUNK - 1) % LDS_SLOTS, QC - 1, b, acc_hi, acc_lo
        )
        store_q(fx.Int32(NCHUNK - 1))
        # logits of this K slice: the lane holds rows g * 4 + i of each row tile, expert wave * 16 + r16
        for rt in range_constexpr(N_RT):
            out = acc_hi[rt] + acc_lo[rt]
            for i in range_constexpr(4):
                row = row0 + rt * 16 + g * 4 + i
                # Mask excess rows: buffer bounds alone include the next K slice.
                buffer_ops.buffer_store(
                    fx.Float32(out[i]),
                    l_rsrc,
                    (khalf * n_rows + row) * E + wave * 16 + r16,
                    mask=(row < n_rows).ir_value(),
                    cache_modifier=_ST_NT,
                )

    @flyc.jit
    def launch_router_mxfp4_mxfp8_quant_line_layout_i768(
        Y: fx.Tensor,
        W: fx.Tensor,
        Q4: fx.Tensor,
        S4: fx.Tensor,
        Q8: fx.Tensor,
        S8: fx.Tensor,
        LOGITS: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream,
    ):
        grid = ((n_rows + BLOCK_M - 1) // BLOCK_M) * K_SPLIT
        kernel_router_quant(
            Y,
            W,
            Q4,
            S4,
            Q8,
            S8,
            LOGITS,
            n_rows,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": f"{N_THREADS},{N_THREADS}",
            },
        ).launch(grid=(grid, 1, 1), block=(N_THREADS, 1, 1), stream=stream)

    launch_router_mxfp4_mxfp8_quant_line_layout_i768.kernel_name = name
    return launch_router_mxfp4_mxfp8_quant_line_layout_i768


def router_mxfp4_mxfp8_quant_line_layout(x: torch.Tensor, gate_weight: torch.Tensor):
    """BF16 [M, 6144] -> (MXFP4 codes, scales), (FP8 codes, scales), fp32 [M, S, 128] router partials.

    ``gate_weight``: ``layout.router_line_layout``'s copy. The FP8 scales are 32-row / 256-K tiled, padding rows zero
    (as the dense scale shuffle).
    """
    m, h = x.shape
    assert h == 6144 and x.dtype == torch.bfloat16 and x.is_contiguous()
    ks = router_k_split(m)
    q4 = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    s4 = torch.empty((m, h // 32), dtype=torch.uint8, device=x.device)
    q8 = torch.empty((m, h), dtype=torch.uint8, device=x.device)
    s8 = torch.empty(((m + 31) // 32 * 32, h // 32), dtype=torch.uint8, device=x.device)
    parts = torch.empty((ks, m, 128), dtype=torch.float32, device=x.device)
    _run_compiled(
        compile_router_mxfp4_mxfp8_quant_line_layout(ks),
        x.view(torch.int32).view(-1), gate_weight.view(-1),
        q4.view(-1), s4.view(-1), q8.view(-1), s8.view(-1), parts.view(-1), m,
        torch.cuda.current_stream(),
    )
    return (q4, s4), (q8, s8), parts.permute(1, 0, 2)
