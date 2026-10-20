"""Wide builds' A2/M2 (``layout_wide.for_wide``: more than ``MAX_ROWS``
rows): hc.py's ``a2`` with request row ``r``'s up result taken from its lanes
of the MFMA tile (``acc[r % 4]``, ``common_wide``) and the pre-polls batched.
Past ``WAVES`` rows its epilogue runs in passes of ``WAVES`` request rows
(:func:`_a2_out`).

Derived from hc.py; hc.py stays the source.
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels import p2
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.device_ops import xred
from atom.models.qwen4_exp.mono.kernels.layout import WAVES
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    HC_WORDS,
    bf16r,
    mb_ix,
    mfma_operands,
    row_ix,
    task_index,
)
from atom.models.qwen4_exp.mono.kernels.stages.common_wide import (
    pre_poll_wide,
    row_lanes,
)
from atom.models.qwen4_exp.mono.kernels.stages.hc import (
    DOWN_GROUPS,
    UP_KI,
    UP_NB,
    halves_sums,
    lane_gather,
    norm_inv,
    norm_load,
    norm_sums,
    poll_inv,
    sigmoid_acc,
)


def _a2_out(c, hc, prod, sp, t, ek):
    """Pass ``ek`` of A2's epilogue past ``WAVES`` rows: wave ``w`` mixes
    request row ``w + 8 ek``, as ``a2``'s wave ``r`` does row ``r``."""
    lane = c.lane
    q = c.wave + WAVES * ek
    rq = fx.min(q, c.rows - 1)
    if q < c.rows:
        col = lane % 32
        s = fx.Float32(0.0)
        for b in range_constexpr(C.hc_count):
            s = s + fx.ptr_load(prod + row_ix(b * 32 + col, rq, sp))
        mixed = bf16r(s * 0.25)
        amax = fmath.absf(mixed)
        for off in (1, 2, 4, 8, 16):
            amax = xred(amax, off, fx.max)
        ex, scale = p2.e8m0_fp8(amax)
        v8 = lane_gather(mixed, (lane % 4) * 8, 8)
        lo = p2.fp8_word(v8[0], v8[1], v8[2], v8[3], scale)
        hi = p2.fp8_word(v8[4], v8[5], v8[6], v8[7], scale)
        v4 = lane_gather(mixed, (lane % 8) * 4, 4)
        if lane < 4:
            mbx.put2(hc.x8, mb_ix(t * 8 + lane * 2, rq), lo, hi, c.tag)
        if lane < 8:
            mbx.put2(
                hc.xbf,
                mb_ix(t * 16 + lane * 2, rq),
                mbx.bf16_pair(v4[0], v4[1]),
                mbx.bf16_pair(v4[2], v4[3]),
                c.tag,
            )
        if lane == 0:
            mbx.put(hc.xe, mb_ix(t, rq), ex, c.tag)


_a2_out = ASTRewriter.transform(_a2_out)


def a2(c, hc):
    tid, lane, wave = c.tid, c.lane, c.wave
    rows = c.rows
    t = task_index(c, hc.up)
    l8 = c.lds_i + c.L.lds["l8"]
    le = c.lds_i + c.L.lds["le"]
    prod = c.lds_f + c.L.lds["prod"]
    sl8, sle, sp = c.L.stride["l8"], c.L.stride["le"], c.L.stride["prod"]
    v4i = fx.Vector.make_type(4, fx.Int32)
    stream = wave // 2
    atom = wave % 2
    nb = t * 2 + stream * UP_NB + atom
    j = lane % 16
    idx = stream * C.hidden + t * 32 + atom * 16 + j

    # Batched builds: hint ``m`` is producer ``m % n`` of request row ``m // n``.
    def hint(m):
        k, r = (m, 0) if rows == 1 else (m % DOWN_GROUPS, m // DOWN_GROUPS)
        return hc.lre, mb_ix(k, r)

    def c1_hint(m):
        k, r = (m, 0) if rows == 1 else (m % hc.producers, m // hc.producers)
        return hc.hs, mb_ix(k * hc.hint_stride, r)

    if t < c.L.tasks[hc.up]:
        wu = [
            fx.Vector(
                bo.buffer_load(
                    hc.w_up, ((nb * 4 + kit) * 64 + lane) * 4, vec_width=4, dtype=T.i32
                )
            )
            for kit in range(UP_KI)
        ]
        su = fx.Int32(bo.buffer_load(hc.s_up, nb * 64 + lane, vec_width=1, dtype=T.i32))
        ww = fx.Int32(bo.buffer_load(hc.norm_w, idx // 2, vec_width=1, dtype=T.i32))
        if const_expr(hc.polled):
            if const_expr(not hc.wide_inv):
                pre_poll_wide(c1_hint, hc.producers * rows, c.tag)
                hws = [
                    v[0]
                    for v in mbx.poll(
                        [(hc.h, mb_ix(idx // 2, r), 1) for r in range(rows)], c.tag
                    )
                ]
                halves_sums(c, hc)
        else:
            hws = [
                fx.Int32(
                    bo.buffer_load(
                        hc.h, row_ix(idx // 2, r, HC_WORDS), vec_width=1, dtype=T.i32
                    )
                )
                for r in range(rows)
            ]
            if const_expr(not hc.wide_inv):
                for r in range_constexpr(rows):
                    norm_sums(c, hc, norm_load(hc.h, tid, r), r)
        if const_expr(not hc.wide_inv):
            norm_inv(c, hc)

        # The low-rank pack: 80 fp8 words then zeros to 96, 10 exponents then
        # 127; from INV_ROWS rows the streams' inverse RMS (A1's task 0 of each row).
        pre_poll_wide(hint, DOWN_GROUPS * rows, c.tag)
        if const_expr(hc.polled and hc.wide_inv):
            # A1's tasks of a row polled all its streams before publishing.
            hws = [
                v[0]
                for v in mbx.poll(
                    [(hc.h, mb_ix(idx // 2, r), 1) for r in range(rows)], c.tag
                )
            ]
        if tid < C.hc_lowrank // 8:
            vs = mbx.poll([(hc.lr8, mb_ix(tid * 2, r), 2) for r in range(rows)], c.tag)
            for r in range_constexpr(rows):
                fx.ptr_store(
                    fx.Vector.from_elements(vs[r], fx.Int32),
                    l8 + row_ix(tid * 2, r, sl8),
                )
        if (tid >= 64) & (tid < 64 + DOWN_GROUPS):
            es = mbx.poll([(hc.lre, mb_ix(tid - 64, r), 1) for r in range(rows)], c.tag)
            for r in range_constexpr(rows):
                fx.ptr_store(es[r][0], le + row_ix(tid - 64, r, sle))
        if (tid >= 128) & (tid < 128 + 16):
            for r in range_constexpr(rows):
                fx.ptr_store(
                    fx.Int32(0), l8 + row_ix(tid - 128 + C.hc_lowrank // 4, r, sl8)
                )
        if (tid >= 192) & (tid < 192 + 2):
            for r in range_constexpr(rows):
                fx.ptr_store(
                    fx.Int32(p2.SCALE_ONE), le + row_ix(tid - 192 + DOWN_GROUPS, r, sle)
                )
        if const_expr(hc.wide_inv):
            poll_inv(c, hc)
        gpu.barrier()

        xl8, xle = mfma_operands(c, l8, le, "l8", "le")
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for kit in range_constexpr(UP_KI):
            off = kit * 32 + (lane // 16) * 4
            a0 = fx.Vector(fx.ptr_load(xl8 + off, result_type=v4i))
            a1_ = fx.Vector(fx.ptr_load(xl8 + (off + 16), result_type=v4i))
            a = fx.Vector.from_elements(
                [a0[e] for e in range(4)] + [a1_[e] for e in range(4)], fx.Int32
            )
            ea = fx.ptr_load(xle + (kit * 4 + lane // 16))
            acc = p2.mfma_p2(acc, a, wu[kit], ea, su, 0, kit)
        for r in range_constexpr(rows):
            hw = hws[r]
            rw = bf16r(sigmoid_acc(bf16r(acc[r % 4])))
            x = (j % 2 == 0).select(hw << 16, hw & fx.Int32(-65536)).bitcast(fx.Float32)
            w = (j % 2 == 0).select(ww << 16, ww & fx.Int32(-65536)).bitcast(fx.Float32)
            inv = fx.ptr_load(
                c.lds_f + row_ix(c.L.lds["inv"] + stream, r, c.L.stride["inv"])
            )
            y = bf16r((x * inv) * w)
            if row_lanes(lane, r):
                fx.ptr_store(
                    bf16r(rw * y), prod + row_ix(stream * 32 + atom * 16 + j, r, sp)
                )
        gpu.barrier()

        # Lane L (and L + 32): mixed[32 t + L]. Batched builds: wave r, of
        # request row r.
        rq = 0 if rows == 1 else fx.min(wave, rows - 1)
        if (wave == 0) if rows == 1 else (wave < rows):
            col = lane % 32
            s = fx.Float32(0.0)
            for b in range_constexpr(C.hc_count):
                s = s + fx.ptr_load(prod + row_ix(b * 32 + col, rq, sp))
            mixed = bf16r(s * 0.25)
            amax = fmath.absf(mixed)
            for off in (1, 2, 4, 8, 16):
                amax = xred(amax, off, fx.max)
            ex, scale = p2.e8m0_fp8(amax)
            v8 = lane_gather(mixed, (lane % 4) * 8, 8)
            lo = p2.fp8_word(v8[0], v8[1], v8[2], v8[3], scale)
            hi = p2.fp8_word(v8[4], v8[5], v8[6], v8[7], scale)
            v4 = lane_gather(mixed, (lane % 8) * 4, 4)
            if lane < 4:
                mbx.put2(hc.x8, mb_ix(t * 8 + lane * 2, rq), lo, hi, c.tag)
            if lane < 8:
                mbx.put2(
                    hc.xbf,
                    mb_ix(t * 16 + lane * 2, rq),
                    mbx.bf16_pair(v4[0], v4[1]),
                    mbx.bf16_pair(v4[2], v4[3]),
                    c.tag,
                )
            if lane == 0:
                mbx.put(hc.xe, mb_ix(t, rq), ex, c.tag)
        for ek in range_constexpr(1, -(-rows // WAVES)):
            _a2_out(c, hc, prod, sp, t, ek)


a2 = ASTRewriter.transform(a2)
