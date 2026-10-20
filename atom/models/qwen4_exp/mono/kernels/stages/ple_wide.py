"""Wide builds' P1 (``layout_wide.for_wide``): ple.py's ``ple_proj`` with
every request row's key and value results taken from its lanes of the MFMA
tile (``acc[r % 4]``, ``common_wide``).

Derived from ple.py; ple.py stays the source.
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels import p2
from atom.models.qwen4_exp.mono.kernels.device_ops import rsrc, xred
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    a_operand,
    bf16r,
    mb_ix,
    mfma_operands,
    p2_issue,
    p2_mma,
    row_ix,
    task_index,
)
from atom.models.qwen4_exp.mono.kernels.stages.common_wide import row_lanes
from atom.models.qwen4_exp.mono.kernels.stages.delta import _theirs
from atom.models.qwen4_exp.mono.kernels.stages.ple import (
    KEY_NB,
    P_GROUPS,
    P_KI,
    STREAM_VECS,
    _pack_vec,
    _res,
    ple_row,
    ple_state,
)


def _put_block_wide(c, region, nb, v, r):
    """``ple._put_block`` of request row ``r`` of an MFMA result tile: its 16
    bf16 outputs ``v`` on lanes ``16 (r // 4) ..`` (``common_wide``)."""
    lane = c.lane
    out = bf16r(v)
    other = xred(out, 1, _theirs)
    if row_lanes(lane, r) & (lane % 2 == 0):
        mbx.put(
            region,
            mb_ix(nb * 8 + (lane % 16) // 2, r),
            mbx.bf16_pair(out, other),
            c.tag,
        )


_put_block_wide = ASTRewriter.transform(_put_block_wide)


def _value_chains_wide(units, x8, xe, lane, rows):
    """``ple._value_chains`` of a wide build: request row ``r``'s value is
    ``acc[r % 4]`` of its lanes (``common_wide``)."""
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
    for g in range_constexpr(P_GROUPS):
        ws, sw = units[g]
        for q in range_constexpr(4):
            kit = g * 4 + q
            ea = fx.ptr_load(xe + (kit * 4 + lane // 16))
            accs[q] = p2.mfma_p2(accs[q], a_operand(x8, kit, lane), ws[q], ea, sw, 0, q)
    return [
        ((accs[0][r % 4] + accs[1][r % 4]) + accs[2][r % 4]) + accs[3][r % 4]
        for r in range(rows)
    ]


def ple_proj(c, p):
    tid, lane, wave = c.tid, c.lane, c.wave
    rows = c.rows
    t = task_index(c, "p1")
    s = ple_state(p)
    x8 = c.lds_i + c.L.lds["p_x8"]
    xe = c.lds_i + c.L.lds["p_xe"]
    sx, se = c.L.stride["p_x8"], c.L.stride["p_xe"]
    xr, er = mfma_operands(c, x8, xe, "p_x8", "p_xe")
    key = wave < 4
    nb = key.select(t * 4 + wave, (wave == 4).select(t, KEY_NB))
    r_w = _res(key.select(s.key_w, s.val_w), "key_w")
    r_s = _res(key.select(s.key_s, s.val_s), "key_s")
    r_emb = [rsrc(s.emb)] + [rsrc(ple_row(p, r, "emb")) for r in range(1, rows)]

    if t < c.L.tasks["p1"]:
        # The embedding rows first: the barrier keeps every wave's row loads
        # ahead of the weight streams in the CU's load pipe.
        vec = fx.min(tid, STREAM_VECS - 1) * 4
        xs = [
            fx.Vector(bo.buffer_load(r_emb[r], vec, vec_width=4, dtype=T.i32))
            for r in range(rows)
        ]
        rocdl.sched_barrier(0)
        gpu.barrier()
        # Waves 5-7 read out of range (n-block 640): zeros, no memory traffic.
        units = [p2_issue(r_w, r_s, nb, P_KI, g, lane) for g in range(P_GROUPS)]

        # The pack: thread ``tid < 320`` (waves 0-4) owns vector ``tid`` of
        # every row.
        for rp in range_constexpr(rows):
            words, ex = _pack_vec(xs[rp])
            if wave < 5:
                fx.ptr_store(words, x8 + row_ix(tid * 2, rp, sx))
                if lane % 4 == 0:
                    fx.ptr_store(ex, xe + row_ix(tid // 4, rp, se))
        gpu.barrier()

        if wave < 4:
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for g in range_constexpr(P_GROUPS):
                acc = p2_mma(acc, units[g], g, xr, er, lane)
            for rk in range_constexpr(rows):
                _put_block_wide(c, c.mb_pk, nb, acc[rk % 4], rk)
        if wave == 4:
            vs = _value_chains_wide(units, xr, er, lane, rows)
            for rv in range_constexpr(rows):
                _put_block_wide(c, c.mb_pv, nb, vs[rv], rv)


ple_proj = ASTRewriter.transform(ple_proj)
