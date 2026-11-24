# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two waves per page, fp8 MFMA, register Q and scalar page-table loads.

Each wave scores half the page. LDS combines their maxima in a fixed order;
only wave zero writes the score. The host uses this at one or three requests.
Units past the request's length leave right after the length load.
The same LUT and one fp8 MFMA per token dot preserve the ordinary scorer bits."""
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
def compile_score_page(ql: int):
    assert ql in (4, 8)
    NW = 2
    QL = ql
    TILES = 8 // NW
    name = f"nvfp4_fp8mfma_index_score_page_d128_p128_r{QL}h1_w{NW}_scale_first"

    @fx.struct
    class Shared:
        lut: fx.Array[fx.Int32, 512, 16]
        partial: fx.Array[fx.Int32, NW * 16, 16]

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
        b0 = fx.block_idx.x
        step = splits
        def resource(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)
        # The first scalar table load is independent of the length. Later
        # pages reload their scalar entry: no unused strided prefetch or readlane.
        tab = fx.Int32(buffer_ops.buffer_load(resource(bt, table_bytes), req * bt_stride + b0,
                         vec_width=1, dtype=fx.Int32, is_scalar=True))
        seq = fx.Int32(buffer_ops.buffer_load(resource(seqs + req.to(fx.Int64) * 4, 4), fx.Int32(0),
                         vec_width=1, dtype=fx.Int32, is_scalar=True))
        seq = (seq < 0).select(fx.Int32(0), seq)
        nblk = ((seq + 127) >> 7)
        # A unit past the request's blocks (most of the 1M plan's 1536 at short
        # contexts) stops before the Q and LUT loads: M 8 / 32K -8%.
        if b0 < nblk:
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
            # Stage the unmodified LUT with 16-byte loads and LDS writes.
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
            def codes(kr,t):
                return Vec(buffer_ops.buffer_load(kr, col * 16 + kl * 4,
                    vec_width=4, dtype=fx.Int32, soffset_bytes=t * 1024, cache_modifier=NT))
            def scale(kr,t):
                return fx.Int32(fx.Int16(buffer_ops.buffer_load(kr, col * 4 + kl,
                    vec_width=1, dtype=fx.Int16, soffset_bytes=PAGE_CODES + t * 128,
                    cache_modifier=NT)).to(fx.Int32))

            sr = resource(score, score_bytes)
            j = fx.Int32(0)
            b = b0
            neg_inf = fx.Float32(float('-inf'))
            # The LUT staging has already drained the prologue VMEM. Keep that
            # fact explicit so the loop does not wait for prior score stores.
            fx.rocdl.s_waitcnt(vmcnt=0)
            while b < nblk:
                page = tab
                if j != 0:
                    page = fx.Int32(buffer_ops.buffer_load(resource(bt, table_bytes), req * bt_stride + b,
                                  vec_width=1, dtype=fx.Int32, is_scalar=True))
                kr = resource(kv + page.to(fx.Int64) * page_stride, PAGE_BYTES)
                # Issue scales first: their LUT reads can overlap the later
                # code loads. The dequantization and maximum order are unchanged.
                ss = [scale(kr,t + wave * TILES) for t in range_constexpr(TILES)]
                cs = [codes(kr,t + wave * TILES) for t in range_constexpr(TILES)]
                fx.rocdl.sched_barrier(0)
                frags = []
                for t in range_constexpr(TILES):
                    t0, t1 = lut_load(ss[t] & 255), lut_load((ss[t] >> 8) & 255)
                    fs = []
                    for e in range_constexpr(4):
                        lo, hi = _dequant8(fx.Int32(cs[t][e]), *(t0 if e < 2 else t1))
                        fs += [lo, hi]
                    frags.append(Vec.from_elements(fs, fx.Int32))
                fx.rocdl.sched_barrier(0)
                def reduce_page(masked):
                    run = neg_inf
                    lim = seq - QL + col - b * 128 - kl * 4 - wave * TILES * 16
                    accs = []
                    for t in range_constexpr(TILES):
                        af = frags[t]
                        acc = _mfma_fp8_16x16x128(af, qf, Vec.filled(4, 0.0, fx.Float32))
                        accs.append(acc)
                        if const_expr(t >= 1):
                            for e in range_constexpr(4):
                                x = accs[t - 1][e]
                                if const_expr(masked):
                                    x = (lim >= (t - 1) * 16 + e).select(x, neg_inf)
                                run = _maxf_nn(run, x)
                    for e in range_constexpr(4):
                        x = accs[TILES - 1][e]
                        if const_expr(masked):
                            x = (lim >= (TILES - 1) * 16 + e).select(x, neg_inf)
                        run = _maxf_nn(run, x)
                    return _xlane_max4(run)
                mx = neg_inf
                if b * 128 + 127 > seq - QL:
                    mx = reduce_page(True)
                else:
                    mx = reduce_page(False)
                if kl == 0:
                    sm.partial.ptr[wave * 16 + col] = mx.bitcast(fx.Int32)
                fx.gpu.barrier()
                mx = fx.Int32(sm.partial.ptr[col]).bitcast(fx.Float32)
                for w in range_constexpr(1, NW):
                    mx = _maxf_nn(mx, fx.Int32(sm.partial.ptr[w * 16 + col]).bitcast(fx.Float32))
                fx.gpu.barrier()
                row_len = seq - QL + col + 1
                rb = (((row_len < 0).select(fx.Int32(0), row_len) + 127) >> 7)
                l0 = rb - local
                l0 = (l0 < 0).select(fx.Int32(0), l0)
                vis = b < rb
                v = (vis & (b < init)).select(fx.Float32(1e30), mx)
                v = (vis & (b >= l0)).select(fx.Float32(1e29), v)
                buffer_ops.buffer_store(v, sr, (req * QL + col) * score_stride + b,
                                        mask=(col < QL) & (kl == 0) & (wave == 0))
                b = b + step
                j = j + 1

    @flyc.jit
    def launch_score_page_scale_first(q: fx.Int64, kv: fx.Int64, lut: fx.Int64, score: fx.Int64,
               bt: fx.Int64, seqs: fx.Int64, page_stride: fx.Int64,
               q_stride: fx.Int32, score_stride: fx.Int32, score_bytes: fx.Int64,
               table_bytes: fx.Int64, max_blocks: fx.Int32,
               splits: fx.Int32, reqs: fx.Int32, bt_stride: fx.Int32, init: fx.Int32, local: fx.Int32, stream: fx.Stream):
        kernel(q, kv, lut, score, bt, seqs, page_stride, q_stride, score_stride,
               score_bytes, table_bytes, max_blocks, splits, reqs, bt_stride, init, local).launch(
               grid=(fx.Int64(splits), fx.Int64(reqs), 1), block=(64 * NW, 1, 1), stream=stream)
    launch_score_page_scale_first.kernel_name = name
    # The 16-word preload covers the pointers and page/Q/score strides (chain at 1 / 3 requests: -1-2%).
    launch_score_page_scale_first.compile_hints = {"llvm_options": {
        "amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}
    return launch_score_page_scale_first
