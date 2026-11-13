# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Program -> one resident ``@flyc.kernel`` and its ``@flyc.jit`` launcher.

Kernel ABI (the same for every program):

    mono(step: Int64, layer: Int32, table: Int64, scratch: Int64, tl: Int64)

- ``step``: device address of an int32 step counter; the launch tag is ``step * LAYER_SLOTS + layer + 1``, so
  bumping the counter once per decode step (or giving each launch its own ``layer``) makes every launch's tag new
  and scratch never needs clearing;
- ``table``: device address of an int64 pointer table, ``program.args`` order;
- ``scratch``: the regions' buffer (``program.scratch_bytes``);
- ``tl``: unused (0); kept in the ABI and in the launch-context stash.

The body walks the stages in order. Each stage gets a fresh context: the launch context re-read from its LDS
stash, the stage's pointer-table entries and region addresses recomputed behind opaque moves, so LLVM cannot
hold one stage's values in registers across later stages (the spills / scratch that would break residency).
Every region is addressed through scratch's buffer resource (one resource for the whole mailbox); dual-copy
regions read the copy of the CTA's I/O die (``device.mailbox``).
"""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen38.mono.kernels.device import mailbox, ops
from atom.models.qwen38.mono.kernels.program import LAYER_SLOTS, Program, plan
from atom.models.qwen38.mono.sources import SOURCES
from atom.mono.plan.trace import enter_stage

_CURRENT_STREAM = fx.Stream(None)


def load_table(table, names) -> dict:
    """The pointer table's entries (``names``, in order) as uniform Int64s, scalar loads at kernel start (constant
    address space, invariant)."""
    out = {}
    for i, name in enumerate(names):
        addr = fx.Int64(table) + fx.Int64(8 * i)
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<4>"), addr.ir_value()).result
        out[name] = fx.Int64(
            llvm.LoadOp(T.i64, ptr, alignment=8, invariant=True).result
        )
    return out


def first_task(bid, base, grid):
    """CTA ``bid``'s first task of a stage placed from CTA ``base``."""
    return (bid - base % grid + grid) % grid


traced = ASTRewriter.transform  # (what ``ops.traced`` is)


@traced
def _hold(first, tasks, units):
    """A CTA with a task of the stage sleeps ``units`` x 64 clocks (every wave)."""
    if first < tasks:
        for _ in range_constexpr(units // 127):
            llvm.InlineAsmOp(None, [], "s_sleep 127", "", has_side_effects=True)
        if const_expr(units % 127):
            llvm.InlineAsmOp(
                None, [], f"s_sleep {units % 127}", "", has_side_effects=True
            )


@traced
def _task_loop(k, emit, first, tasks, grid):
    for t in range(first, tasks, grid):
        emit(k, fx.Int32(t))


def _ctx_stash(p, tag, bid, scratch, table, tl):
    """Every lane stores the launch context into the LDS words at ``p`` (one value per word, every wave the same: a
    wave later reads back its own stores, no barrier needed): tag, CTA id, scratch, table and ``tl`` pointers.
    """
    vals = [tag, bid, scratch, scratch >> 32, table, table >> 32, tl, tl >> 32]
    for i, v in enumerate(vals):
        fx.ptr_store(fx.Int32(v), p + i)


def _ctx_load(p):
    """(tag, bid, scratch, table) from the stash: uniform, through an opaque address (one load per stage: LLVM can
    neither CSE them across stages nor hoist them to the kernel top)."""
    q = p + ops.fresh(fx.Int32(0))
    w = [ops.uniform(fx.ptr_load(q + i)) for i in range(6)]
    return w[0], w[1], _i64(w[2], w[3]), _i64(w[4], w[5])


def _i64(lo, hi):
    return (fx.Int64(hi) << 32) | fx.Int64(fx.Uint32(lo))


def _entry(table, i):
    """Pointer-table entry ``i`` (an invariant scalar load)."""
    addr = fx.Int64(table) + fx.Int64(8 * i)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<4>"), addr.ir_value()).result
    return fx.Int64(llvm.LoadOp(T.i64, ptr, alignment=8, invariant=True).result)


def _stage_ctx(program: Program, base: SimpleNamespace, stage) -> SimpleNamespace:
    """The context a stage's tasks see (attribute reads only)."""
    t = program.target
    tag, bid, scratch, table = _ctx_load(base.ctx)
    args = {name: _entry(table, program.args.index(name)) for name in stage.args}
    wave = ops.uniform(ops.fresh(base.tid) // t.wave)
    # dual regions through the shared resource: ``Addr.dual`` (the header word; ``mailbox`` picks the CTA's copy at
    # each load, stores both)
    routed = set(program.dual_regions)
    mbs = {
        r.name: mailbox.Addr(
            r.name,
            r.space,
            scratch + fx.Int64(program.scratch[r.name]),
            base=scratch,
            off=program.scratch[r.name],
            dual=program.dual_bytes if r.name in routed else None,
            spread=r.name in routed and r.name in program.dual_spread,
        )
        for r in program.regions
    }
    lane = base.tid % t.wave
    return SimpleNamespace(
        program=program,
        scenario=program.scenario,
        rows=program.scenario.rows,
        grid=t.grid,
        threads=t.threads,
        tid=base.tid,
        lane=lane,
        wave=wave,
        bid=bid,
        tag=tag,
        base=stage.base
        % t.grid,  # the stage's first CTA (Python int): task t runs on (base + t) % grid
        target=t,
        lds=base.lds,
        ldsf=base.ldsf,
        args=args,
        mb=mbs,
    )


def _emit_stage(program: Program, base: SimpleNamespace, i: int, stage) -> None:
    enter_stage(stage.name)
    k = _stage_ctx(program, base, stage)
    if i > 0 and stage.lds_words and program.stages[i - 1].lds_words:
        # Stages overlay one LDS arena and a CTA's waves leave a stage at different times: without this a fast
        # wave's next-stage LDS writes clobber what a slow sibling still reads.
        gpu.barrier()
    first = first_task(k.bid, stage.base, program.target.grid)
    if stage.hold:
        _hold(first, stage.tasks, stage.hold)
    _task_loop(k, stage.emit, first, stage.tasks, k.grid)


def _emit_stages(program: Program, base: SimpleNamespace) -> None:
    """Plain Python (not rewritten): the stage loop unrolls at trace time."""
    for i, stage in enumerate(program.stages):
        _emit_stage(program, base, i, stage)


def build(program: Program):
    """The ``@flyc.jit`` launcher of ``program`` (planned here if not yet): ``launch(step, layer, table, scratch, tl,
    stream=...)``."""
    if program.lds_layout is None:
        plan(program)
    t = program.target
    kernel_symbol, ld_i32, uniform = ops.kernel_symbol, ops.ld_i32, ops.uniform
    keyed = program.key() + (("sources", SOURCES),)
    # the arena, then the launch context
    ctx_off, lds_words = (
        program.lds_layout.ctx,
        program.lds_layout.words,
    )

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, lds_words, 16]

    @flyc.kernel(
        name=kernel_symbol(program.name, s=program.scenario.name, h=program.digest()),
        known_block_size=[t.threads, 1, 1],
    )
    def mono(step: Int64, layer: Int32, table: Int64, scratch: Int64, tl: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        # (unused values, kept in their place: the IR order LLVM's folding of the stages depends on)
        _ = fx.recast_iter(fx.Int64, lds + ctx_off)
        tag = uniform(ld_i32(step, 0)) * LAYER_SLOTS + layer + 1
        base = SimpleNamespace(
            tid=tid,
            bid=bid,
            wave=uniform(tid // t.wave),
            tag=tag,
            lds=lds,
            ldsf=fx.recast_iter(fx.Float32, lds),
            args=load_table(table, program.args),
            ctx=lds + ctx_off,
        )
        _ctx_stash(lds + ctx_off, tag, bid, scratch, table, tl)
        _emit_stages(program, base)

    @flyc.jit
    def launch(
        step: Int64,
        layer: Int32,
        table: Int64,
        scratch: Int64,
        tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = keyed
        mono(step, layer, table, scratch, tl).launch(
            grid=(t.grid,), block=(t.threads,), stream=stream
        )

    return launch
