# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Independent fp8-MFMA page scans at two requests.

Read the first page-table entry as a scalar. Prefetch later entries after LUT
staging, bounded by device lengths. This avoids unused table reads at short
contexts while keeping the later pages in registers for long contexts.
The LUT, MFMA and fixed maximum order match the general one-head scorer.
``xcd``: the XCD-local unit-to-block order below (same bits, full score lines
per XCD); the host uses it at 8 rows.
"""
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, const_expr
from amd_kernels.warm_up import launcher
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from amd_kernels.common import buffer_ops
from .kernel import (_xlane_max4, _maxf_nn, _dequant8, _perm, PERM_EVEN,
                     PERM_ODD, _mfma_fp8_16x16x128, PAGE_BYTES, PAGE_CODES, NT)


@functools.cache
@launcher
def compile_score_first_page(ql: int, xcd: bool = False):
    assert ql in (4, 8)
    NW = 1
    QL = ql
    name = f"nvfp4_fp8mfma_index_score_d128_p128_r{QL}h1_w{NW}_firstpage" + ("_xcd" if xcd else "")

    @fx.struct
    class Shared:
        lut: fx.Array[fx.Int32, 512, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(q: fx.Int64, kv: fx.Int64, lut: fx.Int64, score: fx.Int64,
               bt: fx.Int64, seqs: fx.Int64, page_stride: fx.Int64,
               q_stride: fx.Int32, score_stride: fx.Int32, score_bytes: fx.Int64,
               table_bytes: fx.Int64, max_blocks: fx.Int32,
               splits: fx.Int32, reqs: fx.Int32, bt_stride: fx.Int32, init: fx.Int32, local: fx.Int32):
        sm = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        col, kl = lane % 16, lane // 16
        req = fx.block_idx.y
        b0 = (fx.block_idx.x) * NW + wave
        if const_expr(xcd):
            # NW == 1. Units below splits rounded down to 256 score block
            # 256 (y // 32) + 32 (x % 8) + y % 32 (y = x // 8): a bijection
            # there, so the 32 blocks of one 128-B score line come from
            # workgroups of one XCD (equal x % 8 share one); the rest keep x.
            yg = b0 >> 3
            remap = ((yg >> 5) << 8) + ((b0 & 7) << 5) + (yg & 31)
            b0 = (b0 < ((splits >> 8) << 8)).select(remap, b0)
        step = splits * NW
        def resource(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)
        # The first scalar table entry and Q do not depend on the length.
        tab = fx.Int32(buffer_ops.buffer_load(resource(bt, table_bytes), req * bt_stride + b0,
                         vec_width=1, dtype=fx.Int32, is_scalar=True))
        seq = fx.Int32(buffer_ops.buffer_load(resource(seqs + req.to(fx.Int64) * 4, 4), fx.Int32(0),
                         vec_width=1, dtype=fx.Int32, is_scalar=True))
        seq = (seq < 0).select(fx.Int32(0), seq)
        qr = resource(q + req.to(fx.Int64) * QL * q_stride, QL * q_stride)
        qs = []
        for h in range_constexpr(2):
            d = Vec(buffer_ops.buffer_load(qr, ((col * q_stride + kl * 32) >> 2) + h * 4,
                    vec_width=4, dtype=fx.Int32, mask=col < QL))
            qs += [_perm(fx.Int32(d[1]), fx.Int32(d[0]), PERM_EVEN),
                   _perm(fx.Int32(d[1]), fx.Int32(d[0]), PERM_ODD),
                   _perm(fx.Int32(d[3]), fx.Int32(d[2]), PERM_EVEN),
                   _perm(fx.Int32(d[3]), fx.Int32(d[2]), PERM_ODD)]
        qf = Vec.from_elements(qs, fx.Int32)
        lr = resource(lut, 2048)
        lut_quads = fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32, sm.lut.ptr),
                fx.make_layout(512, 1)), fx.make_layout(4, 1))
        atom128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        for k in range_constexpr(512 // (64 * NW * 4)):
            lv = Vec(buffer_ops.buffer_load(lr, tx * 4 + k * 64 * NW * 4, vec_width=4, dtype=fx.Int32))
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(lv)
            fx.copy(atom128, r, fx.slice(lut_quads, (None, tx + k * 64 * NW)))
        fx.gpu.barrier()
        lut_view = fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32, sm.lut.ptr),
                       fx.make_layout(512, 1)), fx.make_layout(2, 1))
        atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
        def lut_load(s):
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            fx.copy(atom, fx.slice(lut_view, (None, s)), r)
            v = Vec(r.load())
            return fx.Int32(v[0]), fx.Int32(v[1])
        def tile(kr, t):
            c = Vec(buffer_ops.buffer_load(kr, col * 16 + kl * 4,
                vec_width=4, dtype=fx.Int32, soffset_bytes=t * 1024, cache_modifier=NT))
            s = fx.Int32(fx.Int16(buffer_ops.buffer_load(kr, col * 4 + kl,
                vec_width=1, dtype=fx.Int16, soffset_bytes=PAGE_CODES + t * 128,
                cache_modifier=NT)).to(fx.Int32))
            return c, s
        sr = resource(score, score_bytes)
        nblk = ((seq + 127) >> 7)
        # Later entries overlap the first page; one lane holds each entry.
        future = fx.Int32(buffer_ops.buffer_load(resource(bt, table_bytes), req * bt_stride + b0 + (lane + 1) * step,
                     vec_width=1, dtype=fx.Int32, mask=b0 + (lane + 1) * step < ((seq + 127) >> 7)))
        neg_inf = fx.Float32(float('-inf'))
        # Peeling removes the first/future selector from the page loop.
        @flyc.jit
        def page_value(b: fx.Int32, page: fx.Int32):
            kr = resource(kv + page.to(fx.Int64) * page_stride, PAGE_BYTES)
            cs, ss = [], []
            for t in range_constexpr(8):
                c, s = tile(kr, t)
                cs.append(c)
                ss.append(s)
            fx.rocdl.sched_barrier(0)
            frags = []
            for t in range_constexpr(8):
                t0, t1 = lut_load(ss[t] & 255), lut_load((ss[t] >> 8) & 255)
                fs = []
                for e in range_constexpr(4):
                    lo, hi = _dequant8(fx.Int32(cs[t][e]), *(t0 if e < 2 else t1))
                    fs += [lo, hi]
                frags.append(Vec.from_elements(fs, fx.Int32))
            fx.rocdl.sched_barrier(0)
            # The dots do not depend on the causal mask. Issue them once,
            # then keep the original token order in either maximum reduction.
            accs = [_mfma_fp8_16x16x128(frags[t], qf, Vec.filled(4, 0.0, fx.Float32))
                    for t in range_constexpr(8)]
            def reduce_page(masked):
                run = neg_inf
                lim = seq - QL + col - b * 128 - kl * 4
                for t in range_constexpr(8):
                    for e in range_constexpr(4):
                        x = accs[t][e]
                        if const_expr(masked):
                            x = (lim >= t * 16 + e).select(x, neg_inf)
                        run = _maxf_nn(run, x)
                return _xlane_max4(run)
            mx = neg_inf
            if b * 128 + 127 > seq - QL:
                mx = reduce_page(True)
            else:
                mx = reduce_page(False)
            row_len = seq - QL + col + 1
            rb = (((row_len < 0).select(fx.Int32(0), row_len) + 127) >> 7)
            l0 = rb - local
            l0 = (l0 < 0).select(fx.Int32(0), l0)
            vis = b < rb
            v = (vis & (b < init)).select(fx.Float32(1e30), mx)
            v = (vis & (b >= l0)).select(fx.Float32(1e29), v)
            return v
        b = b0
        if b < nblk:
            v = page_value(b, tab)
            # Complete the future table prefetch before the first score store.
            # The later loop then reads ready lanes without waiting for that store.
            fx.rocdl.s_waitcnt(vmcnt=0)
            buffer_ops.buffer_store(v, sr, (req * QL + col) * score_stride + b,
                                    mask=(col < QL) & (kl == 0))
            b = b + step
            j = fx.Int32(0)
            while b < nblk:
                page = fx.Int32(fx.rocdl.readlane(T.i32, future, j))
                v = page_value(b, page)
                buffer_ops.buffer_store(v, sr, (req * QL + col) * score_stride + b,
                                        mask=(col < QL) & (kl == 0))
                b = b + step
                j = j + 1

    @flyc.jit
    def launch_first_page(q: fx.Int64, kv: fx.Int64, lut: fx.Int64, score: fx.Int64,
               bt: fx.Int64, seqs: fx.Int64, page_stride: fx.Int64,
               q_stride: fx.Int32, score_stride: fx.Int32, score_bytes: fx.Int64,
               table_bytes: fx.Int64, max_blocks: fx.Int32,
               splits: fx.Int32, reqs: fx.Int32, bt_stride: fx.Int32, init: fx.Int32, local: fx.Int32, stream: fx.Stream):
        kernel(q, kv, lut, score, bt, seqs, page_stride, q_stride, score_stride,
               score_bytes, table_bytes, max_blocks, splits, reqs, bt_stride, init, local).launch(
               grid=(fx.Int64(splits), fx.Int64(reqs), 1), block=(64 * NW, 1, 1), stream=stream)
    launch_first_page.kernel_name = name
    return launch_first_page
