"""Grouped verify PLE mix (``build_layer(..., group=S)``): ple.py's P2 for a
verify step of ``rows / S`` requests of ``S`` rows each.

Row ``r`` is row ``j = r % S`` of its group. Its conv state is its group's
window (its table entry's ``conv_in``, ``group_meta``) then the ``nrm`` of
the group's earlier rows, the seq build's chain within the group.

Derived from ple.py; ple.py stays the one-request source.
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import GRID
from atom.models.qwen4_exp.mono.kernels.device_ops import xred
from atom.models.qwen4_exp.mono.kernels.layout import TASKS
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    HC_WORDS,
    bf16r,
    bf16x8,
    mb_ix,
    pick_row,
    row_ix,
    task_index,
    task_rounds,
)
from atom.models.qwen4_exp.mono.kernels.stages.delta import _theirs
from atom.models.qwen4_exp.mono.kernels.stages.hc import OOB, fma, sigmoid_acc
from atom.models.qwen4_exp.mono.kernels.stages.ple import (
    PV,
    PV_FROM_PK,
    PV_FROM_PN,
    ROW,
    TAPS,
    _bf,
    _gate,
    _half,
    _half_sum,
    _mix_specs,
    _next_state,
    _normalize,
    _odd_half,
    _res,
    _vecs,
    ple_row,
    ple_state,
    stream_inv,
)
from atom.models.qwen4_exp.mono.kernels.stages.qsa import _request


def _group_taps(c, rq, ch, col, odd, taps):
    """Row ``rq``'s 9 state elements (bf16 bits): the last 9 of its group's
    window (``taps``) then the ``nrm`` of the group's rows before it. Polls
    only those rows (the others' specs repeat this row's own ``pv`` word,
    already published): a later row waits on this one."""
    j = rq % c.group
    gb = rq - j
    specs = [
        (
            c.mb_isc,
            (j > q).select(mb_ix(ch // 2, gb + q), mb_ix(PV_FROM_PN + col // 2, rq)),
            1,
        )
        for q in range(c.group - 1)
    ]
    got = mbx.poll(specs, c.tag)
    hist = taps + [odd.select(_half(w[0], 1), _half(w[0], 0)) for w in got]
    return [pick_row(j, [hist[r + k] for r in range(c.group)]) for k in range(TAPS)]


def ple_mix(c, p):
    t = task_index(c, "p2")
    for k in range_constexpr(task_rounds(c, "p2")):
        if k > 0:
            gpu.barrier()  # after the last round's LDS reads
        _ple_mix(c, p, t if k == 0 else t + k * GRID)


def _ple_mix(c, p, t):
    tid, lane, wave = c.tid, c.lane, c.wave
    rq, tq = _request(c, t, TASKS["p2"])
    s = ple_state(p)
    sig_l = c.lds_f + c.L.lds["p_sig"]
    inv_l = c.lds_f + c.L.lds["p_inv"]
    j = wave % 4
    key = wave < 4
    r_nq = _res(s.nq, "nq")
    r_nk = _res(s.nk, "nk")
    r_nc = _res(s.nc, "nc")
    r_cw = _res(s.conv_w, "conv_w")
    # Every row's table entry holds its group's window and its own snapshot.
    cin, cout = ple_row(p, rq, "conv_in"), ple_row(p, rq, "conv_out")
    r_cin = _res(cin, "conv_in")
    r_cout = _res(cout, "conv_out")
    # Stream words of this lane's vectors; waves 4-7 read the key side's
    # operands out of range.
    vw = [key.select(j * ROW + 4 * (lane + 64 * m), OOB) for m in range(PV)]
    # The pairs polled: the key's stream j (waves 0-3) or the value (4-7).
    pbase = key.select(j * ROW, PV_FROM_PK)
    # Conv channel of thread tid < 128 (the others read out of range).
    conv = tid < 128
    cj = (tid // 32) % 4
    col = tq * 32 + tid % 32
    ch = cj * C.hidden + col
    odd = (tid % 2) == 1  # parity of ch and of its first state element
    sw = ch * TAPS // 2

    def hint(m):
        return c.mb_pv, mb_ix(m * 8 + 7, rq)

    if t < c.L.tasks["p2"]:
        hv = [
            fx.Vector(
                bo.buffer_load(
                    p.h_in, row_ix(w, rq, HC_WORDS), vec_width=4, dtype=T.i32
                )
            )
            for w in vw
        ]
        qw = [fx.Vector(bo.buffer_load(r_nq, w, vec_width=4, dtype=T.i32)) for w in vw]
        kw = [fx.Vector(bo.buffer_load(r_nk, w, vec_width=4, dtype=T.i32)) for w in vw]
        cw = fx.Vector(
            bo.buffer_load(r_cw, conv.select(ch * 2, OOB), vec_width=2, dtype=T.i32)
        )
        st = fx.Vector(
            bo.buffer_load(r_cin, conv.select(sw, OOB), vec_width=4, dtype=T.i32)
        )
        st4 = fx.Int32(
            bo.buffer_load(r_cin, conv.select(sw + 4, OOB), vec_width=1, dtype=T.i32)
        )
        hw = fx.Int32(
            bo.buffer_load(
                p.h_in,
                conv.select(row_ix(ch // 2, rq, HC_WORDS), OOB),
                vec_width=1,
                dtype=T.i32,
            )
        )
        ncw = fx.Int32(
            bo.buffer_load(r_nc, conv.select(ch // 2, OOB), vec_width=1, dtype=T.i32)
        )

        # The query norm does not wait on P1 (waves 4-7 normalize zeros).
        hs = _vecs(hv)
        qn = _normalize(hs, stream_inv(hs, lane), _vecs(qw))

        mbx.pre_poll(hint, c.L.tasks["p1"], c.tag)
        got = mbx.poll(_mix_specs(c, pbase, lane, col, rq), c.tag)
        xs = [bf16x8(got[2 * m] + got[2 * m + 1]) for m in range(PV)]

        if wave < 4:
            kn = _normalize(xs, stream_inv(xs, lane), _vecs(kw))
            sigma = _gate(kn, qn)
            if lane == 0:
                fx.ptr_store(sigma, sig_l + j)
        gpu.barrier()
        if wave >= 4:
            gated = [[bf16r(fx.ptr_load(sig_l + j) * x) for x in xv] for xv in xs]
            ginv = stream_inv(gated, lane)
            if lane == 0:
                fx.ptr_store(ginv, inv_l + j)
        gpu.barrier()

        if wave < 2:
            gv = bf16r(fx.ptr_load(sig_l + cj) * _odd_half(odd, got[2 * PV][0]))
            nrm = bf16r((gv * fx.ptr_load(inv_l + cj)) * _odd_half(odd, ncw))
            # State element k (bf16 bits): half ``odd + k`` of the 5 words.
            sv = [st[0], st[1], st[2], st[3], st4]
            taps = [
                odd.select(
                    _half(sv[(k + 1) // 2], (k + 1) % 2), _half(sv[k // 2], k % 2)
                )
                for k in range(TAPS)
            ]
            nrm_other = xred(nrm, 1, _theirs)
            if lane % 2 == 0:
                mbx.put(
                    c.mb_isc, mb_ix(ch // 2, rq), mbx.bf16_pair(nrm, nrm_other), c.tag
                )
            taps = _group_taps(c, rq, ch, col, odd, taps)
            w0, w1 = mbx.bf2_f32(cw[0])
            w2, w3 = mbx.bf2_f32(cw[1])
            acc = fma(_bf(taps[0]), w0, fx.Float32(0.0))
            acc = fma(_bf(taps[3]), w1, acc)
            acc = fma(_bf(taps[6]), w2, acc)
            acc = fma(nrm, w3, acc)
            rc = bf16r(acc)
            act = bf16r(rc * sigmoid_acc(rc))
            out = bf16r(_odd_half(odd, hw) + bf16r(gv + act))

            nxt, last = _next_state(odd, taps, nrm)
            bo.buffer_store(nxt, r_cout, sw + tid % 2)
            bo.buffer_store(last, r_cout, sw + 4)

            other = xred(out, 1, _theirs)
            sq = _half_sum(out * out)
            if lane % 2 == 0:
                mbx.put(c.mb_ph, mb_ix(ch // 2, rq), mbx.bf16_pair(out, other), c.tag)
            if lane % 16 == 0:
                mbx.put(
                    c.mb_phs,
                    mb_ix(cj * c.L.tasks["c1"] + tq * 2 + (tid % 32) // 16, rq),
                    sq,
                    c.tag,
                )


_ple_mix = ASTRewriter.transform(_ple_mix)
