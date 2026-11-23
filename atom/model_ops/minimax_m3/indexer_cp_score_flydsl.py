"""MiniMax-M3 indexer context-parallel FP4 block scorer for gfx1250 (FlyDSL).

Drop-in for ``indexer_context_parallel._context_score`` on the FP4 index cache
with ``fill_dead_slots=False``: the same [heads, tokens, local] FP32 scores, bit
for bit (slots past a request's live blocks are left unwritten, as there).

Math -- the Triton scorer's ``nvfp4_scaled_dot`` + ``nvfp4_block_max``, same
instructions, operand roles and accumulation order: per 16-key tile one
``v_wmma_scale16_f32_16x16x128_f8f6f4`` with A = the e2m1 key rows and their
E4M3 per-16 scales (keys on M), B = the query's E4M3 hi term (E8M0 scale 1),
then the same instruction with B = the lo term (E8M0 2^-4) accumulating onto
it; the causal mask (-inf); the max over the block's keys (``v_max*_num_f32``
and ``ds_max_num_f32``: maximumNumber, order-free); one multiply by
sm_scale * log2(e). The query split uses Triton's conversions
(``v_cvt_scalef32_pk8_fp8_f32`` / ``v_cvt_scale_pk8_f32_fp8``).

Work. Grid (requests, P) sized by the batch (not by max_model_len); 4 waves a
workgroup. Workgroup p scores the request's local blocks [p c, p c + c),
c = ceil(live / P) <= 64 (balanced to a block). Wave w owns key tiles 2w, 2w+1
(32 keys) of every block: it streams its quarter of a block (2 KiB of e2m1
rows, 256 B of scales) global -> LDS with ``global_load_async_to_lds`` into a
private ring of RING slots -- the data in flight costs no registers -- and
reads the WMMA operands back from LDS (16 B pieces XOR-swizzled: conflict
free). The query is converted once a workgroup and staged through LDS. 8
WMMAs a block cover all 32 query rows; the four waves' partial maxima meet by
LDS atomic max; after the loop one barrier, and each store instruction writes
one row's consecutive blocks (coalesced: scattered 4-byte score stores were
the single largest cost of the earlier layouts).
"""

import functools

import numpy as np
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

WAVE = 32
WAVES = 4  # a workgroup; wave w owns key tiles 2w, 2w+1 of every block
THREADS = WAVE * WAVES
PAGE = 128
MAX_BLOCKS = 64  # blocks a workgroup owns at most (two block-table VGPRs)
SLOT_PAYLOAD = 32 * 64  # a wave's quarter block: 32 key rows (16 B pieces swizzled)
SLOT = SLOT_PAYLOAD + 32 * 8  # + their 32 scale rows
QOP_LANE = 64  # LDS bytes of one lane's 16-dword B operand (16 B pieces swizzled)
LOG2E = 1.4426950409
E8M0_ONE = 0x7F7F7F7F7F7F7F7F  # q_hi's scale bytes (2^0)
E8M0_LO = 0x7B7B7B7B7B7B7B7B  # q_lo's scale bytes (2^-4)
E4M3_MAX = 448.0
COPIES = 5  # async copies of a quarter block: 4 x 512 B payload + 256 B scales


def _attr(text):
    return ir.Attribute.parse(text)


def _wmma(a, b, c, sa, sb, row_a, row_b):
    """D = A(e2m1, E4M3 scales row ``row_a``) x B(e4m3, E8M0 scales) + C."""
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
    """8 fp32 -> 8 e4m3 in two dwords (v_cvt_scalef32_pk8_fp8_f32, scale 1)."""
    return fx.Vector(
        rocdl.cvt_scalef32_pk8_fp8_f32(
            T.i32x2, fx.as_ir_value(v8), fx.as_ir_value(fx.Float32(1.0))
        )
    )


def _from_fp8x8(w2):
    """8 e4m3 (two dwords) -> 8 fp32 (v_cvt_scale_pk8_f32_fp8, E8M0 1.0)."""
    return fx.Vector(
        rocdl.cvt_scale_pk8_f32_fp8(
            T.vec(8, T.f32), fx.as_ir_value(w2), fx.as_ir_value(fx.Int32(0x7F)), 0
        )
    )


def _swz(row, piece):
    """LDS byte offset of 16 B piece ``piece`` (0..3) of 64 B row ``row``:
    pieces XOR-swizzled by (row / 4) % 4, so 16 B reads of rows u = 0..15 at
    one piece hit distinct banks."""
    return row * 64 + ((piece ^ ((row >> 2) & 3)) * 16)


def _lds_i32(base_u8, off, n):
    p = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, min(4 * n, 16)),
        fx.add_offset(base_u8, off),
    )
    return p


@functools.lru_cache(maxsize=None)
def build(heads: int, ql: int, rank: int, world: int, ring: int):
    """Launcher for (index heads, query rows a request, CP rank / world);
    ``ring``: LDS slots (blocks in flight) a wave."""
    rows = heads * ql
    nqt = (rows + 15) // 16  # 16-row query tiles
    assert 1 <= nqt <= 2, rows
    assert ql <= 128  # at most two blocks of a request hold masked keys
    assert 2 <= ring <= 8
    RING = ring
    name = f"m3_cp_score_fp4_gfx1250_h{heads}_q{ql}_r{rank}w{world}_g{ring}"
    RING_BYTES = WAVES * RING * SLOT
    PART = MAX_BLOCKS * 32  # maxima [block][row], combined by LDS atomic max
    assert nqt * 2 * 32 * QOP_LANE <= WAVES * SLOT  # B operands staged in the last slots
    JH = 4 // (THREADS // (32 * nqt))  # K chunks a converting thread takes

    @fx.struct
    class SharedStorage:
        ring_buf: fx.Array[fx.Uint8, RING_BYTES, 16]
        part: fx.Array[fx.Float32, PART, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def cp_score_kernel(
        arg_q: fx.Pointer,
        arg_kd: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_len: fx.Pointer,
        arg_out: fx.Pointer,
        i32_q_tok: fx.Int32,
        i32_q_head: fx.Int32,
        i32_kd_page: fx.Int32,
        i32_ks_page: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_local: fx.Int32,
        i32_global: fx.Int32,
        i32_batch: fx.Int32,
        i32_programs: fx.Int32,
        f32_scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % WAVE
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tid // WAVE)))
        g = lane // 16
        u = lane % 16
        b = fx.Int32(gpu.block_id("x"))
        program = fx.Int32(gpu.block_id("y"))
        tokens = i32_batch * ql
        zero = fx.Int32(0)

        # ---- the query rows this thread converts (thread t: B lane t % 32 of
        # tile (t // 32) % nqt, K chunks [JH h, JH h + JH), h = t // 32 nqt)
        q_ptr = fx.recast_iter(fx.BFloat16, arg_q)
        t_lane = tid % 32
        t_qt = (tid // 32) % nqt
        t_jh = tid // (32 * nqt)
        t_on = t_jh * JH < 4
        tu, tg = t_lane % 16, t_lane // 16
        tcol = tu + t_qt * 16
        tlive = (tcol < rows) & t_on
        tcol = tlive.select(tcol, zero)
        tbase = fx.Int64(b * ql + tcol // heads) * fx.Int64(i32_q_tok) + fx.Int64((tcol % heads) * i32_q_head)
        bt = fx.recast_iter(fx.Int32, arg_bt)
        last_col = i32_global - 1

        def load_windows(local_fn):
            ws = []
            for h in range_constexpr(MAX_BLOCKS // 32):
                gblk = local_fn(lane + 32 * h) * world + rank
                gblk = (gblk < last_col).select(gblk, last_col)
                ws.append(fx.Int32(bt[b * i32_bt_stride + gblk]))
            return ws

        length = fx.Int32(fx.recast_iter(fx.Int32, arg_len)[b])
        request_blocks = (length + 127) // 128
        request_blocks = (request_blocks < i32_global).select(request_blocks, i32_global)
        live_end = (request_blocks - rank + (world - 1)) // world
        live_end = (live_end < zero).select(zero, live_end)
        live_end = (live_end < i32_local).select(live_end, i32_local)
        # workgroup p: local blocks [start, start + nblk), c = ceil(live / P)
        chunk = (live_end + i32_programs - 1) // i32_programs
        start = program * chunk
        nblk = live_end - start
        nblk = (nblk < chunk).select(nblk, chunk)

        def local_of(j):
            return start + j

        # Local blocks from `mid` on may hold keys past a row's cutoff.
        min_cutoff = length - ql + 1
        first_masked = (((min_cutoff - PAGE) >> 7) + 1 - rank + (world - 1)) // world
        mid = (first_masked < zero).select(zero, first_masked)
        if nblk > zero:
            windows = load_windows(local_of)
            # query rows (issued after the block table: loads return in order)
            q_raw = []
            for jj in range_constexpr(JH):
                jq = (t_jh * JH + jj) % 4
                q_raw.append(
                    fx.Vector(
                        fx.ptr_load(
                            fx.add_offset(q_ptr, tbase + fx.Int64(tg * 16 + jq * 32)),
                            result_type=fx.Vector.make_type(16, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                )

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            part = lds.part.ptr
            ring_u8 = fx.recast_iter(fx.Uint8, lds.ring_buf.ptr)
            # the B operands stage in each wave's last ring slot (free until
            # the main loop's first copy): operand p = 2 qt + hl in wave p's
            q_u8 = ring_u8

            def q_off(p, lane_, k):
                return (p * RING + (RING - 1)) * SLOT + lane_ * QOP_LANE + ((k ^ ((lane_ >> 2) & 3)) * 16)

            lds_base = fx.Int64(fx.ptrtoint(lds.ring_buf.ptr))
            ring_wave = lds_base + fx.Int64(wave * (RING * SLOT))
            kd_base = fx.Int64(fx.ptrtoint(arg_kd))
            ks_base = fx.Int64(fx.ptrtoint(arg_ks))
            # copy c of a quarter: lane's 16 B chunk 32 c + lane of the 2 KiB
            # (row (32 c + lane) // 4, 16 B piece lane % 4), pieces swizzled.
            cp_src = [fx.Int64(wave * 2048 + c * 512 + lane * 16) for c in range_constexpr(4)]
            cp_dst = [fx.Int64(_swz((c * 32 + lane) // 4, lane % 4)) for c in range_constexpr(4)]
            sc_src = fx.Int64(wave * 256 + lane * 8)
            sc_dst = fx.Int64(SLOT_PAYLOAD + lane * 8)

            def page_of(j):
                if const_expr(MAX_BLOCKS == 32):
                    return fx.Int32(rocdl.readlane(T.i32, fx.as_ir_value(windows[0]), fx.as_ir_value(j)))
                lo = fx.Int32(rocdl.readlane(T.i32, fx.as_ir_value(windows[0]), fx.as_ir_value(j % 32)))
                hi = fx.Int32(rocdl.readlane(T.i32, fx.as_ir_value(windows[1]), fx.as_ir_value(j % 32)))
                return (j < 32).select(lo, hi)

            def issue(j):
                """Async copies of block j's quarter into ring slot j % RING."""
                page = fx.Int64(page_of(j))
                slot = ring_wave + fx.Int64((j % RING) * SLOT)
                pd = kd_base + page * fx.Int64(i32_kd_page)
                for c in range_constexpr(4):
                    rocdl.global_load_async_to_lds_b128(
                        create_llvm_ptr(pd + cp_src[c], address_space=1),
                        create_llvm_ptr(slot + cp_dst[c], address_space=3),
                        0,
                    )
                rocdl.global_load_async_to_lds_b64(
                    create_llvm_ptr(ks_base + page * fx.Int64(i32_ks_page) + sc_src, address_space=1),
                    create_llvm_ptr(slot + sc_dst, address_space=3),
                    0,
                )

            for d in range_constexpr(RING - 1):
                if fx.Int32(d) < nblk:
                    issue(fx.Int32(d))

            # ---- query -> B operands (hi / lo E4M3), once a workgroup ------
            if t_on:
                for jj in range_constexpr(JH):
                    jq = t_jh * JH + jj
                    vals = q_raw[jj]
                    hw, lw = [], []
                    for h in range_constexpr(2):
                        v8 = fx.Vector.from_elements(
                            [
                                fx.minnumf(
                                    fx.maxnumf(
                                        tlive.select(fx.Float32(vals[h * 8 + e]), fx.Float32(0.0)),
                                        fx.Float32(-E4M3_MAX),
                                    ),
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
                    for hl, words in ((0, hw), (1, lw)):
                        fx.ptr_store(fx.Vector.from_elements(words, dtype=fx.Int32),
                                     _lds_i32(q_u8, q_off(t_qt * 2 + hl, t_lane, jq), 4))
            # partial maxima start as NaN (the identity of maximumNumber)
            nan = fx.Vector.from_elements([fx.Int32(0x7FC00000)], dtype=fx.Int32).bitcast(fx.Float32)[0]
            for k in range_constexpr(PART // THREADS):
                fx.ptr_store(nan, part + (tid + k * THREADS))
            gpu.barrier()
            q_hi, q_lo, cutoffs = [], [], []
            for qt in range_constexpr(nqt):
                col = u + qt * 16
                col = (col < rows).select(col, zero)
                cutoffs.append(length - ql + col // heads + 1)
                ops = []
                for hl in range_constexpr(2):
                    words = []
                    for k in range_constexpr(4):
                        v = fx.Vector(fx.ptr_load(
                            _lds_i32(q_u8, q_off(qt * 2 + hl, lane, k), 4),
                            result_type=fx.Vector.make_type(4, fx.Int32)))
                        words += [v[e] for e in range_constexpr(4)]
                    ops.append(fx.Vector.from_elements(words, dtype=fx.Int32))
                q_hi.append(ops[0])
                q_lo.append(ops[1])
            rocdl.s_wait_dscnt(0)
            gpu.barrier()  # every wave holds its operands: the last slots are free
            # B scales must sit in VGPRs (an inline / SGPR i64 scale reads
            # wrong): OR in a lane-varying zero the compiler cannot fold.
            lane_zero = fx.Int64((lane + length) >> 31)
            s_hi = fx.Int64(E8M0_ONE) | lane_zero
            s_lo = fx.Int64(E8M0_LO) | lane_zero

            zero8 = fx.Vector.filled(8, 0.0, fx.Float32)
            neg_inf = fx.Float32(float("-inf"))
            a_off = [[_swz(tt * 16 + u, g + 2 * h) for h in range_constexpr(2)] for tt in range_constexpr(2)]

            def partial_max(j, masked):
                """[nqt] per-lane max over this wave's 32 keys of block j (both
                lane halves hold every row of their tile after the cross-half
                max)."""
                slot = wave * (RING * SLOT) + (j % RING) * SLOT
                sa = fx.Vector(fx.ptr_load(_lds_i32(ring_u8, slot + SLOT_PAYLOAD + lane * 8, 2),
                                           result_type=fx.Vector.make_type(2, fx.Int32))).bitcast(fx.Int64)[0]
                key0 = (local_of(j) * world + rank) * PAGE + wave * 32
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
                            vals = [
                                (pos0 + v < cutoffs[qt]).select(vals[v], neg_inf)
                                for v in range_constexpr(8)
                            ]
                        m = vals[0]
                        for v in range_constexpr(1, 8):
                            m = fx.maxnumf(m, vals[v])
                        ms[qt] = m if tt == 0 else fx.maxnumf(ms[qt], m)
                for qt in range_constexpr(nqt):
                    ms[qt] = fx.maxnumf(ms[qt], ms[qt].shuffle_xor(16, WAVE))
                return ms

            wr_row = u + g * 16  # lane (u, g) writes row 16 g + u (tile g)

            part_base = fx.Int64(fx.ptrtoint(part))

            def put(ms, j):
                val = ms[0]
                if const_expr(nqt == 2):
                    val = (g == 0).select(ms[0], ms[1])
                from flydsl._mlir.dialects import llvm as _llvm
                _llvm.atomicrmw(
                    _llvm.AtomicBinOp.fmax,
                    create_llvm_ptr(part_base + fx.Int64((j * 32 + wr_row) * 4), address_space=3),
                    fx.as_ir_value(val),
                    _llvm.AtomicOrdering.monotonic,
                    syncscope="workgroup",
                )

            # ---- main loop: block j + RING - 1 streams in while j is scored
            for j in range(zero, nblk):
                j = fx.Int32(j)
                ja = j + (RING - 1)
                if ja < nblk:
                    rocdl.s_wait_dscnt(0)  # slot ja % RING's last readers are done
                    issue(ja)
                    rocdl.s_wait_asynccnt(COPIES * (RING - 1))
                else:
                    rocdl.s_wait_asynccnt(0)
                if local_of(j) < mid:
                    put(partial_max(j, False), j)
                else:
                    put(partial_max(j, True), j)
            gpu.barrier()
            # ---- combine the four waves' maxima, scale, store: one row's
            # consecutive blocks per instruction (lane = block) --------------
            out_ptr = fx.recast_iter(fx.Float32, arg_out)
            for it in range_constexpr(32 * MAX_BLOCKS // THREADS):
                idx = tid + it * THREADS
                jb = idx % MAX_BLOCKS
                row = idx // MAX_BLOCKS
                if (jb < nblk) & (row < rows):
                    v = fx.Float32(fx.ptr_load(part + (jb * 32 + row))) * f32_scale
                    dst = fx.add_offset(out_ptr, ((row % heads) * tokens + b * ql + row // heads) * i32_local + local_of(jb))
                    fx.ptr_store(v, dst)

    @flyc.jit
    def launch(
        arg_q: fx.Pointer,
        arg_kd: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_len: fx.Pointer,
        arg_out: fx.Pointer,
        i32_q_tok: fx.Int32,
        i32_q_head: fx.Int32,
        i32_kd_page: fx.Int32,
        i32_ks_page: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_local: fx.Int32,
        i32_global: fx.Int32,
        i32_batch: fx.Int32,
        i32_programs: fx.Int32,
        f32_scale: fx.Float32,
        stream: fx.Stream,
    ):
        cp_score_kernel(
            arg_q,
            arg_kd,
            arg_ks,
            arg_bt,
            arg_len,
            arg_out,
            i32_q_tok,
            i32_q_head,
            i32_kd_page,
            i32_ks_page,
            i32_bt_stride,
            i32_local,
            i32_global,
            i32_batch,
            i32_programs,
            f32_scale,
        ).launch(
            grid=(fx.Int64(i32_batch), fx.Int64(i32_programs), 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch


# Workgroups at once: MI455X has 256 CUs and the kernel fits 6 (VGPRs) of
# them a CU; 4 a CU left the most bandwidth in sweeps (bs 16-32, 64K-256K).
CP_FLYDSL_WORKGROUPS = 1024
# LDS ring slots a wave (blocks in flight: RING - 1). 4 x 4 waves x 2.25 KiB +
# 8 KiB of maxima = 45 KiB a workgroup; 3 and 5 were slower.
CP_FLYDSL_RING = 4


def programs_per_request(batch: int, local: int, workgroups: int = CP_FLYDSL_WORKGROUPS) -> int:
    """Workgroups a request: the batch's share of `workgroups`, and enough
    that a workgroup's chunk (ceil(local / P)) fits its 64-block window."""
    return max(1, workgroups // max(1, batch), -(-local // MAX_BLOCKS))


def supported(idx_q, index_cache, index_scale, max_query_len) -> bool:
    """Whether `cp_scores_fp4` takes this call (the caller falls back to the
    Triton scorer otherwise)."""
    heads = idx_q.shape[1]
    return (
        1 <= heads * max_query_len <= 32
        and max_query_len <= 128
        and idx_q.dtype == torch.bfloat16
        and idx_q.stride(2) == 1
        and index_cache.dtype == torch.uint8
        and index_scale.dtype == torch.uint8
        and index_cache.stride()[1:] == (64, 1)
        and index_scale.stride()[1:] == (8, 1)
        and index_cache.stride(0) % 16 == 0
        and index_scale.stride(0) % 8 == 0
        and index_cache.data_ptr() % 16 == 0
        and index_scale.data_ptr() % 8 == 0
        and max(index_cache.stride(0), index_scale.stride(0), idx_q.stride(0)) < 2**31
    )


def cp_scores_fp4(
    idx_q,
    index_cache,
    index_scale,
    block_table,
    seq_lens,
    max_seq_len,
    rank,
    world,
    max_query_len,
    sm_scale,
    scores,
    workgroups=CP_FLYDSL_WORKGROUPS,
    ring=CP_FLYDSL_RING,
):
    """Fill ``scores`` [heads, tokens, ceil(blocks / world)] FP32 like
    ``_context_score`` with ``fill_dead_slots=False`` (slots past a request's
    live blocks are left unwritten). Inputs as `indexer_context_scores`
    validates them; ``supported`` must hold."""
    tokens, heads, _ = idx_q.shape
    if tokens == 0:
        return scores
    batch = seq_lens.numel()
    blocks = -(-max_seq_len // PAGE)
    local = -(-blocks // world)
    launch = build(heads, max_query_len, rank, world, ring)
    with torch.cuda.device(idx_q.device):
        _run_compiled(
            launch,
            ptr_arg(idx_q, fx.BFloat16),
            ptr_arg(index_cache, fx.Int32),
            ptr_arg(index_scale, fx.Int32),
            ptr_arg(block_table, fx.Int32),
            ptr_arg(seq_lens, fx.Int32),
            ptr_arg(scores, fx.Float32),
            idx_q.stride(0),
            idx_q.stride(1),
            index_cache.stride(0),
            index_scale.stride(0),
            block_table.stride(0),
            local,
            blocks,
            batch,
            programs_per_request(batch, local, workgroups),
            # Triton rounds the constexpr sm_scale * log2(e) to FP32 the same way
            float(np.float32(sm_scale * LOG2E)),
            torch.cuda.current_stream(idx_q.device).cuda_stream,
        )
    return scores
