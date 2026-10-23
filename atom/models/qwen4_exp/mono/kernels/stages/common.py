"""Device helpers shared by the stages: math variants, reductions, MXFP4 GEMV units.

A stage is a module-level function ``stage(c, p)`` called from the kernel body.
``c`` is the kernel context (thread ids, tag, LDS views, mailbox addresses) and
``p`` the buffer resources of the kernel arguments, both made once at kernel
entry. Stage functions with runtime control flow go through
``ASTRewriter.transform``; inside a runtime region they only read attributes
of ``c``/``p`` and call module-level functions, so neither becomes carried
state.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels import p2
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import GRID
from atom.models.qwen4_exp.mono.kernels.device_ops import exp, rcp, rsrc, xred
from atom.models.qwen4_exp.mono.kernels.layout import MAX_ROWS, ROW_PAIRS

LN2 = 0.6931471805599453


def task_index(c, name):
    """This CTA's task of stage ``name`` (``>= c.L.tasks[name]`` when it has none)."""
    return (c.bid + (GRID - c.L.base[name])) & (GRID - 1)


def task_rounds(c, name):
    """Rounds of stage ``name``'s tasks: a batched build's stage may have more
    tasks than CTAs, and round ``k`` runs task ``task_index + k GRID``."""
    return -(-c.L.tasks[name] // GRID)


# ------------------------------------------------------------- request rows
# A batched build (``c.rows`` > 1) decodes one token of each of ``rows``
# requests. Request row ``r`` of a mailbox region sits ``ROW_PAIRS`` pairs on,
# of a per-row LDS entry ``c.L.stride[name]`` words on, of ``h_in`` / ``h_out``
# ``HC_WORDS`` words on. Row indices are Python ints or runtime values
# (wave-uniform wherever they pick a buffer resource); row 0 as a Python ``0``
# adds nothing, so one-request builds stay as they were.

HC_WORDS = C.hc_width // 2


def row_ix(i, r, stride):
    """Index ``i`` in request row ``r`` of rows ``stride`` apart."""
    if isinstance(r, int) and r == 0:
        return i
    return i + r * stride


def pick_row(r, values):
    """``values[r]`` for the wave-uniform row ``r`` (a select chain)."""
    v = values[0]
    for q in range(1, len(values)):
        v = (r == q).select(values[q], v)
    return v


def mb_ix(i, r):
    """Pair ``i`` of request row ``r`` of a mailbox region."""
    return row_ix(i, r, ROW_PAIRS)


def lds_row(c, view, name, r=0):
    """``view`` (``c.lds_i`` or ``c.lds_f``) at row ``r`` of LDS entry ``name``."""
    return view + row_ix(c.L.lds[name], r, c.L.stride[name])


def table_entry(r, i):
    """Entry ``i`` (wave-uniform) of the int64 table behind buffer resource
    ``r``. Scalar loads: a vector load would wait for every vector load and
    store the wave issued before it (vmcnt is in order)."""
    words = [
        fx.Int32(bo.buffer_load(r, 2 * i + k, vec_width=1, is_scalar=True))
        for k in (0, 1)
    ]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Int64)[0]


DELTA_STATE = ("conv_in", "conv_out", "rec_in", "rec_out")


def delta_state(c, p, r):
    """Buffer resources of request row ``r``'s DeltaNet state, in
    :data:`DELTA_STATE` order. The ``conv_in`` argument points at an int64
    table ``[rows][DELTA_STATE]`` of their addresses, in every build: ATOM's
    state slots change per request, and a captured graph keeps the table's
    address while the step rewrites its entries."""
    return tuple(
        rsrc(table_entry(p.conv_in, r * 4 + k)) for k in range(len(DELTA_STATE))
    )


def _hw(name, value):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(value).ir_value()], [], [])
    )


def log2_hw(x):
    return _hw("llvm.amdgcn.log.f32", x)


def exp2_hw(x):
    """``v_exp_f32``: the hardware 2^x, no range reduction."""
    return _hw("llvm.amdgcn.exp2.f32", x)


def bf16r(x):
    """Round an f32 to bf16 (RNE) and widen it back."""
    return fx.Float32(fx.Float32(x).to(fx.BFloat16))


def bf16_bits(x):
    """bf16 bits of ``x`` in the low half of an int32."""
    return (
        fx.Vector.from_elements([x, x], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
        & 0xFFFF
    )


def sigmoid_fast(x):
    """``v_rcp(1 + v_exp(-x * log2 e))``: the native DeltaNet sigmoid."""
    return rcp(fx.Float32(1.0) + exp(fx.Float32(0.0) - x))


def softplus_fast(x):
    """Native DeltaNet softplus: hardware exp2/log2, identity above 20."""
    sp = log2_hw(fx.Float32(1.0) + exp(x)) * LN2
    return (x > 20.0).select(x, sp)


def add(a, b):
    return a + b


def wave_sum(v):
    for off in (32, 16, 8, 4, 2, 1):
        v = xred(v, off, add)
    return v


def wave_max(v):
    for off in (32, 16, 8, 4, 2, 1):
        v = xred(v, off, fx.max)
    return v


def sum16(v):
    """Sum over each 16-lane row; every lane of the row gets the same bits."""
    for off in (8, 4, 2, 1):
        v = xred(v, off, add)
    return v


# ---------------------------------------------------------- MXFP4 GEMV
# A stored MXFP4 weight is n-blocks of 16 rows; n-block ``nb`` holds ``ki``
# k_iters of 1 KiB, lane ``l`` owning 16 bytes of each. ``scales_k4`` puts the
# scales of k_iters ``4 g .. 4 g + 3`` of lane ``l`` in one dword, picked by
# the MFMA's opsel. The MXFP8 activation sits in LDS: fp8 words at ``x8``,
# one exponent word per 32 values at ``xe``.


def p2_issue(r_w, r_s, nb, ki, g, lane, off=0):
    """Issue the weight and scale loads of k_iter group ``g`` of n-block ``nb``;
    ``off`` = ``OOB`` loads zeros without touching memory."""
    ws = [
        fx.Vector(
            bo.buffer_load(
                r_w,
                ((nb * ki + g * 4 + q) * 64 + lane) * 4 + off,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for q in range_constexpr(4)
    ]
    sw = fx.Int32(
        bo.buffer_load(
            r_s, (nb * (ki // 4) + g) * 64 + lane + off, vec_width=1, dtype=T.i32
        )
    )
    return ws, sw


def a_operand(x8, kit, lane):
    """Lane ``lane``'s fp8 A operand of k_iter ``kit``: two 16-byte halves 64
    bytes apart (see :func:`p2.mfma_p2`)."""
    v4i = fx.Vector.make_type(4, fx.Int32)
    off = kit * 32 + (lane // 16) * 4
    a0 = fx.Vector(fx.ptr_load(x8 + off, result_type=v4i))
    a1 = fx.Vector(fx.ptr_load(x8 + (off + 16), result_type=v4i))
    return fx.Vector.from_elements(
        [a0[e] for e in range(4)] + [a1[e] for e in range(4)], fx.Int32
    )


def p2_mma(acc, unit, g, x8, xe, lane):
    """``acc += W[group g] @ x`` for a unit from :func:`p2_issue`.

    Every lane holds the same activation, so each 16-row tile of the result
    holds 16 copies of the GEMV row.
    """
    ws, sw = unit
    for q in range_constexpr(4):
        kit = g * 4 + q
        ea = fx.ptr_load(xe + (kit * 4 + lane // 16))
        acc = p2.mfma_p2(acc, a_operand(x8, kit, lane), ws[q], ea, sw, 0, q)
    return acc


def mfma_row(c):
    """Request row of this lane's MFMA A row in a batched build. Lane ``l``
    carries A row ``l % 16`` (:func:`p2.mfma_p2`), filled from request row
    ``min(l % 16, rows - 1)``: request ``r``'s result is ``acc[r]`` of lanes
    0-15, bit-identical to a one-request build's ``acc[0]`` (every element
    depends only on its own A row). Point the A operand and its exponents at
    the row: ``p2_mma(acc, unit, g, x8 + r * X8_ROW, xe + r * XE_ROW, lane)``."""
    return fx.min(c.lane % 16, c.rows - 1)


def mfma_operands(c, x8, xe, x8_name="x8", xe_name="xe"):
    """``(x8, xe)`` of this lane's :func:`mfma_row` (per-row LDS entries
    ``x8_name`` / ``xe_name``); unchanged in a one-request build."""
    if c.rows == 1:
        return x8, xe
    r = mfma_row(c)
    return x8 + r * c.L.stride[x8_name], xe + r * c.L.stride[xe_name]


def payload_wait(c, hint, n):
    """The wait ahead of a payload poll whose producers end with the hint
    pairs ``hint`` (``n`` of them, as :func:`mbx.pre_poll`). Builds of up to
    ``MAX_ROWS`` rows only barrier: the payload poll spins itself, one round
    trip after the producers' stores instead of the hint's two. Wider builds
    keep the pre-poll's throttle (their payload polls hold more pairs per
    thread, on more CTAs). Every thread must call this."""
    if c.L.rows <= MAX_ROWS:
        gpu.barrier()
    else:
        mbx.pre_poll(hint, n, c.tag)


def poll_mixed(c, x8, xe):
    """Load A2's MXFP8 mixed row into LDS (fp8 words at ``x8``, exponents at
    ``xe``). Wave 0 waits on the 80 exponent pairs (each A2 task's last
    store), then every thread loads its fp8 words. Ends with a CTA barrier."""
    if c.rows > 1:
        _poll_mixed_rows(c, x8, xe)
    else:
        _poll_mixed_one(c, x8, xe)


def _poll_mixed_one(c, x8, xe):
    tid, lane = c.tid, c.lane
    if c.wave == 0:
        e = mbx.poll(
            [(c.mb_xe, lane, 1), (c.mb_xe, fx.min(lane + 64, C.hidden // 32 - 1), 1)],
            c.tag,
        )
        fx.ptr_store(e[0][0], xe + lane)
        if lane < 16:
            fx.ptr_store(e[1][0], xe + (lane + 64))
    gpu.barrier()
    if tid < C.hidden // 8:
        v = mbx.poll([(c.mb_x8, tid * 2, 2)], c.tag)[0]
        fx.ptr_store(v[0], x8 + tid * 2)
        fx.ptr_store(v[1], x8 + (tid * 2 + 1))
    gpu.barrier()


def _poll_mixed_rows(c, x8, xe):
    """:func:`poll_mixed` of every request row into the rows of LDS ``x8`` /
    ``xe``. Only wave 0 spins, on row 0's exponents as in the one-row build
    (each A2 task publishes all rows together); then thread ``i < 320`` loads
    its fp8 words of every row and exponent ``i % 80`` of row ``1 + i // 80``,
    normally on the first try."""
    tid, lane, wave = c.tid, c.lane, c.wave
    sx, se = c.L.stride["x8"], c.L.stride["xe"]
    n = C.hidden // 32
    if wave == 0:
        e = mbx.poll(
            [
                (c.mb_xe, mb_ix(lane, 0), 1),
                (c.mb_xe, mb_ix(fx.min(lane + 64, n - 1), 0), 1),
            ],
            c.tag,
        )
        fx.ptr_store(e[0][0], xe + lane)
        if lane < 16:
            fx.ptr_store(e[1][0], xe + (lane + 64))
    gpu.barrier()
    if tid < C.hidden // 8:
        k = fx.min(tid, (c.rows - 1) * n - 1)
        er, ej = 1 + k // n, k % n
        specs = [(c.mb_x8, mb_ix(tid * 2, r), 2) for r in range(c.rows)]
        v = mbx.poll(specs + [(c.mb_xe, mb_ix(ej, er), 1)], c.tag)
        for r in range_constexpr(c.rows):
            fx.ptr_store(v[r][0], x8 + (r * sx + tid * 2))
            fx.ptr_store(v[r][1], x8 + (r * sx + tid * 2 + 1))
        if tid < (c.rows - 1) * n:
            fx.ptr_store(v[c.rows][0], xe + (er * se + ej))
    gpu.barrier()


_poll_mixed_one = ASTRewriter.transform(_poll_mixed_one)
_poll_mixed_rows = ASTRewriter.transform(_poll_mixed_rows)


def lds_f4(ptr, off):
    """Four f32 from a 16-byte aligned LDS offset, as a Python list."""
    v4f = fx.Vector.make_type(4, fx.Float32)
    v = fx.Vector(fx.ptr_load(ptr + off, result_type=v4f))
    return [v[e] for e in range(4)]


def bf16x8(words):
    """Four packed bf16 words -> eight f32, low half first."""
    out = []
    for e in range(4):
        w = words[e]
        out += [
            (w << 16).bitcast(fx.Float32),
            (w & fx.Int32(-65536)).bitcast(fx.Float32),
        ]
    return out
