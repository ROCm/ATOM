"""C1: ``out_proj`` over the gated DeltaNet output + the attention HC write.

Task ``t`` (160) owns hidden columns ``16 t .. 16 t + 15``: one n-block of the
MXFP4 weight over K 6144 (48 k_iters): wave ``w < 4`` chains k_iters
``w, w + 4, ...``, then ``((w0 + w1) + w2) + w3``; waves 4-7 load out of range
and skip the GEMM. The epilogue is the fused HC write,
``h_out[b] = bf16(res_b + bf16(bf16(proj) * inj_b))``, for the 4 streams.

The residual is the attention HC's input streams: the layer input, prefetched
from ``h_in`` before any poll, or in the PLE layer P2's ``h'`` (``ph``).

The written streams are the MLP HC's input: C1 publishes them as pairs
(``w``, one per bf16 word) and, per stream, the square sum of its 16 values
(``whs``, the half-chunk sums) for M2's norm. ``h_out`` is only written by the attention-half kernel.
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels import p2
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.device_ops import xred
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    HC_WORDS,
    a_operand,
    add,
    bf16r,
    mb_ix,
    mfma_operands,
    row_ix,
    task_index,
)
from atom.models.qwen4_exp.mono.kernels.stages.hc import OOB

C1_KI = p2.k_iters(C.delta_value)  # 48
C1_CHAIN = C1_KI // 4  # 12 k_iters per wave
ROW_WORDS = C.hidden // 2


def c1(c, p, store_out=False):
    tid, lane, wave = c.tid, c.lane, c.wave
    t = task_index(c, "c1")
    x8 = c.lds_i + c.L.lds["x8"]
    xe = c.lds_i + c.L.lds["xe"]
    red = c.lds_f + c.L.lds["red"]
    rows = c.rows
    rs = c.L.stride["red"]

    if t < c.L.tasks["c1"]:
        # k_iter ``w + 4 j`` is in scale dword ``j``, byte ``w``.
        part = wave % 4
        off = (wave < 4).select(fx.Int32(0), fx.Int32(OOB))
        ws = [
            fx.Vector(
                bo.buffer_load(
                    p.w_out,
                    ((t * C1_KI + part + 4 * j) * 64 + lane) * 4 + off,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for j in range(C1_CHAIN)
        ]
        ss = [
            fx.Int32(
                bo.buffer_load(
                    p.s_out,
                    (t * C1_CHAIN + j) * 64 + lane + off,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            for j in range(C1_CHAIN)
        ]
        # Thread ``tid < 32``: stream ``tid // 8``, bf16 word ``tid % 8`` of the tile.
        # Batched builds: thread ``32 r + i`` is thread i of request row r.
        if const_expr(rows == 1):
            b, rt = fx.min(tid, 31) // 8, 0
        else:
            tt = fx.min(tid, 32 * rows - 1)
            b, rt = tt % 32 // 8, tt // 32
        word = b * ROW_WORDS + t * 8 + tid % 8
        if const_expr(c.kind == "delta_ple"):
            # The PLE layer's residual is P2's h' (published before A1).
            res = mbx.poll([(c.mb_ph, mb_ix(word, rt), 1)], c.tag)[0][0]
        else:
            res = fx.Int32(
                bo.buffer_load(
                    p.h_in, row_ix(word, rt, HC_WORDS), vec_width=1, dtype=T.i32
                )
            )
        # A1 published the injection weights long before: take them now, not
        # one more round trip after the GEMM.
        inj = mbx.poll([(c.mb_inj, mb_ix(b, rt), 1)], c.tag)[0][0].bitcast(fx.Float32)

        # The exponents are each B3 task's last store: wait on them first.
        # Batched builds: wave r waits on request row r's.
        rw = 0 if rows == 1 else fx.min(wave, rows - 1)
        if (wave == 0) if rows == 1 else (wave < rows):
            e = mbx.poll(
                [(c.mb_de, mb_ix(lane + 64 * m, rw), 1) for m in range(3)], c.tag
            )
            for m in range_constexpr(3):
                fx.ptr_store(e[m][0], xe + row_ix(lane + 64 * m, rw, c.L.stride["xe"]))
        gpu.barrier()
        hi = fx.min(tid, 255) + 512
        sx = c.L.stride["x8"]
        specs = []
        for r in range_constexpr(rows):
            specs += [(c.mb_d8, mb_ix(tid * 2, r), 2), (c.mb_d8, mb_ix(hi * 2, r), 2)]
        v = mbx.poll(specs, c.tag)
        for r in range_constexpr(rows):
            fx.ptr_store(v[2 * r][0], x8 + row_ix(tid * 2, r, sx))
            fx.ptr_store(v[2 * r][1], x8 + row_ix(tid * 2 + 1, r, sx))
        if tid < 256:
            for r in range_constexpr(rows):
                fx.ptr_store(v[2 * r + 1][0], x8 + row_ix(hi * 2, r, sx))
                fx.ptr_store(v[2 * r + 1][1], x8 + row_ix(hi * 2 + 1, r, sx))
        gpu.barrier()

        if wave < 4:
            xr, er = mfma_operands(c, x8, xe)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for j in range_constexpr(C1_CHAIN):
                kit = part + 4 * j
                ea = fx.ptr_load(er + (kit * 4 + lane // 16))
                sw = ss[j] >> (part * 8)
                acc = p2.mfma_p2(acc, a_operand(xr, kit, lane), ws[j], ea, sw, 0, 0)
            if lane < 16:
                for r in range_constexpr(rows):
                    fx.ptr_store(acc[r], red + row_ix(wave * 16 + lane, r, rs))
        gpu.barrier()

        if tid < 32 * rows:
            j = (tid % 8) * 2
            p0 = fx.ptr_load(red + row_ix(j, rt, rs))
            p1 = fx.ptr_load(red + row_ix(j + 1, rt, rs))
            for w in range_constexpr(1, 4):
                p0 = p0 + fx.ptr_load(red + row_ix(w * 16 + j, rt, rs))
                p1 = p1 + fx.ptr_load(red + row_ix(w * 16 + j + 1, rt, rs))
            r0, r1 = mbx.bf2_f32(res)
            o0 = r0 + bf16r(bf16r(p0) * inj)
            o1 = r1 + bf16r(bf16r(p1) * inj)
            out = mbx.bf16_pair(o0, o1)
            if const_expr(store_out):
                bo.buffer_store(out, p.h_out, row_ix(word, rt, HC_WORDS))
            mbx.put(c.mb_w, mb_ix(word, rt), out, c.tag)
            # The tile's 16 values of stream b sit on 8 adjacent lanes.
            v0, v1 = mbx.bf2_f32(out)
            sq = v0 * v0 + v1 * v1
            for off in (1, 2, 4):
                sq = xred(sq, off, add)
            if tid % 8 == 0:
                mbx.put(c.mb_whs, mb_ix(b * c.L.tasks["c1"] + t, rt), sq, c.tag)


c1 = ASTRewriter.transform(c1)
