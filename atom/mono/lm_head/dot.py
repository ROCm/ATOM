# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A vocab head's dot ``bf16(x . W[v])`` over this rank's vocab rows, a task
a CTA: the part every head launch shares (the LM head, ``lm_head.kernel``;
DeepSeek-V4.1's draft head, ``deepseek_v41/mono/kernels/draft_head``).

``W`` is the LM head shard as loaded (row-major [vocab, K] bf16). A task's
``blocks`` blocks of ``VT`` rows, a wave 16 rows (the MFMA's A), x's rows
(every row of the step, in LDS: whole or in ``kparts`` K slices landed in
turn) the B columns, 16 a chunk. The weights stream in units of 128 K
elements a wave: each load reads 4 rows x 256 contiguous bytes (whole cache
lines), the unit goes through the wave's own LDS tile into the MFMA's layout
(same K order at any slicing), one bf16 MFMA a 32-element step and chunk.
Each logit, rounded to bf16, goes to the caller's ``put``.

A build key using these provides: ``rows``, ``vocab``, ``hidden``,
``prefetch`` (K steps in flight a wave), ``blocks``, ``chunks``, ``steps``,
``kparts`` / ``psteps`` / ``pitch`` (x's K slicing in LDS), ``a_stage`` (the
waves' staging tiles, ``STAGE_WORDS``) and ``x_in_launch`` (x written by this
launch: read at device scope after every row's flag in the context's ``hxr``).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.mono.device.gemv import rows_loads, rows_store
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    butterfly,
    mfma_bf16,
    rms_rstd,
    traced,
)
from atom.mono.plan.execution import WAVES

RG = 16  # vocab rows a row group (the MFMA's M) and rows a chunk of x (its N)
KSTEP = 32  # K an MFMA step (the MFMA's K)
# K steps a weight unit: a wave's 16 rows x 128 elements, read 4 rows x 256
# contiguous bytes an instruction, through LDS into the MFMA's layout
UNIT = 4
STAGE_PITCH = UNIT * KSTEP // 2 + 4  # words a staged row (+4: bank spread)
STAGE_WORDS = WAVES * RG * STAGE_PITCH  # every wave's staging tile
VT = WAVES * RG  # vocab rows a block
LDS_BYTES = 160 * 1024


def lds_quad(c, at):
    """LDS words ``at`` .. + 3 (``at`` a multiple of 4) in one 16-byte read."""
    return fx.Vector(
        fx.ptr_load(c["lds"] + at, result_type=fx.Vector.make_type(4, fx.Int32))
    )


@traced
def x_ready(c):
    """``x_in_launch``: every row's flag (``hxr``), a lane a row."""
    p = c["key"]
    if const_expr(p.x_in_launch):
        lane = c["lane"]
        c["mb"].poll(
            [
                (c["hxr"], fx.min(lane + 64 * k, fx.Int32(p.rows - 1)), 1)
                for k in range(-(-p.rows // 64))
            ]
        )


def issue(c, blk, u):
    """Weight unit ``u`` of the wave's 16 W rows of block ``blk`` (K steps
    UNIT u ..): load s reads rows 4 s .. + 3, 256 contiguous bytes each (lane
    ``lane``: row 4 s + lane // 16, its 16 B at (lane % 16) 16), whole cache
    lines an instruction; a row group past ``vocab`` reads 0 (past the
    descriptor)."""
    p, lane = c["key"], c["lane"]
    rg = (c["t"] * p.blocks + blk) * WAVES + c["wave"]
    live = rg < p.vocab // RG
    loads = []
    for s in range_constexpr(RG // 4):
        row = rg * RG + 4 * s + lane // 16
        off = row * (p.hidden // 2) + u * (UNIT * KSTEP // 2) + (lane % 16) * 4
        off = live.select(off, fx.Int32(p.vocab * p.hidden // 2))
        load = bo.buffer_load(
            c["r_w"], off, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
        )
        loads.append(fx.Vector(load))
    return loads


def stage_unit(c, loads):
    """A unit's loads (``issue``) through the wave's LDS tile -> its UNIT K
    steps' MFMA A operands, lane ``lane`` row lane % 16, K 8 (lane // 16) ..
    + 7 of each step: the wave's own tile, in program order, no barrier."""
    p, lane = c["key"], c["lane"]
    tile = p.a_stage + c["wave"] * (RG * STAGE_PITCH)
    for s in range_constexpr(RG // 4):
        at = tile + (4 * s + lane // 16) * STAGE_PITCH + (lane % 16) * 4
        fx.ptr_store(loads[s], c["lds"] + at)
    row = tile + (lane % 16) * STAGE_PITCH + (lane // 16) * 4
    return [lds_quad(c, row + t * (KSTEP // 2)) for t in range(UNIT)]


@traced
def land_x(c, part):
    """K slice ``part`` of x [rows, K] into LDS (row r at word r pitch): 16 B
    a load, every load issued before the stores; device scope where this
    launch wrote x (``x_in_launch``)."""
    p, k = c["key"], c["key"].psteps * KSTEP
    cm = CM_DEV if p.x_in_launch else 0
    loads = rows_loads(
        c["tid"], c["args"]["x"], p.rows, k, p.hidden, k0=part * k, cm=cm
    )
    rows_store(c["tid"], loads, p.rows, k, c["lds"], 2 * p.pitch)


@traced
def dot_block(c, blk, put):
    """Block ``blk``'s logits: per weight unit the load ``prefetch`` K steps
    ahead, the unit through the wave's LDS tile (``stage_unit``), then one
    MFMA a K step and 16-row chunk of x (x's next K slice landing once every
    wave is past the last); ``put(c, blk, ch, vals)`` each chunk's 4 logits
    a lane (x row lane % 16 + 16 ch, vocab rows 4 (lane // 16) .. + 3 of the
    wave's group), rounded to bf16."""
    p, lane = c["key"], c["lane"]
    xrows = [
        c["ldsf"] + fx.min(lane % 16 + RG * ch, fx.Int32(p.rows - 1)) * p.pitch
        + (lane // 16) * 4
        for ch in range(p.chunks)
    ]  # fmt: skip
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.chunks)]
    units = p.steps // UNIT
    q = [issue(c, blk, u) for u in range(min(p.prefetch // UNIT, units))]
    n_pre = len(q)
    for u in range_constexpr(units):
        i0 = u * UNIT
        # x's slices land past the block's first weight loads: they stream
        # while x is awaited (``x_ready``) and loaded
        if const_expr(i0 % p.psteps == 0 and (p.kparts > 1 or blk == 0)):
            gpu.barrier()  # every wave past the last slice (or the last block)
            if const_expr(blk == 0 and u == 0):
                x_ready(c)
            land_x(c, i0 // p.psteps)
            gpu.barrier()
        if const_expr(u + n_pre < units):
            q.append(issue(c, blk, u + n_pre))
        a = stage_unit(c, q[u])
        for t in range_constexpr(UNIT):
            for ch in range_constexpr(p.chunks):
                b = fx.ptr_load(
                    xrows[ch] + (i0 + t) % p.psteps * (KSTEP // 2),
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
                accs[ch] = mfma_bf16(
                    a[t].bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), accs[ch]
                )
    for ch in range_constexpr(p.chunks):
        put(c, blk, ch, [bf16_round(accs[ch][e]) for e in range(4)])


def dots(c, put):
    """Every block's logits (``dot_block``), a block at a time."""
    p = c["key"]
    # unrolled: a runtime loop holds the next block's addresses live beside
    # the prefetch and spills
    for blk in range_constexpr(p.blocks):
        if const_expr(blk > 0):
            rocdl.sched_barrier(0)
        dot_block(c, blk, put)


@traced
def row_rstd(c, groups, lives):
    """The final RMSNorm's rstd of a row a head norms (``kernel.norm_row``,
    V4.1's ``draft_head.tail_row``): ``groups`` this thread's 8-column f32
    groups of it, live where ``lives``, summed over the CTA through LDS
    (``a_red``), rounded as aiter's norm (``rms_rstd``)."""
    p, lane = c["key"], c["lane"]
    sq = fx.Float32(0.0)
    for k in range_constexpr(len(groups)):
        for e in range_constexpr(8):
            sq = sq + lives[k].select(groups[k][e] * groups[k][e], fx.Float32(0.0))
    sq = butterfly(sq, (1, 2, 4, 8, 16, 32))
    if lane == 0:
        fx.ptr_store(sq, c["ldsf"] + (p.a_red + c["wave"]))
    gpu.barrier()
    tot = fx.ptr_load(c["ldsf"] + p.a_red)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(c["ldsf"] + (p.a_red + w))
    return rms_rstd(tot, p.hidden, p.eps)
