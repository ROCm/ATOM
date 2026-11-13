# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Program -> one resident ``@flyc.kernel`` and its ``@flyc.jit`` launcher.

Kernel ABI (the same for every program):

    mono(step: Int64, layer: Int32, table: Int64, scratch: Int64, priv: Int64)

- ``step``: device address of an int32 step counter; the launch tag is
  ``step * LAYER_SLOTS + layer + 1``, so bumping the counter once per decode
  step (or giving each launch its own ``layer``) makes every launch's tag new
  and scratch never needs clearing;
- ``table``: device address of an int64 pointer table, ``program.args`` order;
- ``scratch``: the regions' buffer (``program.scratch_bytes``): every launch's tags are new, so programs that never
  run at once (a kind's programs of other widths) may share one;
- ``priv``: the program's own buffer (``program.private_bytes``): word 0 its dual-copy word, then the regions
  updated with atomics (``Region.single``), which count from a multiple of a launch's adds and so cannot be shared.

A TP program with peer regions also reads ``PEERS_ARG`` (every rank's peer buffer) and ``RANK_ARG``
(this rank, as a value) from the table; its stages see ``k.tp``, ``k.rank``, ``k.peers`` and their
peer regions' ``k.mb`` entries in this rank's peer buffer (``device.gfx950.peer``); a shared program
(``Program.peer_shared``) also ``PEER_TOP_ARG``, its group's peer buffer bytes (``k.peer_top``), and its group's
step phase (``k.peer_phase``: the peer table's entry ``tp``, ``PeerGroup.end_step``).

The body walks the stages in order. Each stage gets a fresh context (``_stage_ctx``): wave
index, pointer-table loads and region addresses recomputed behind opaque moves,
so LLVM cannot hold one stage's values in registers across later stages (the
spills / scratch that would break residency).

Target: the program's (``Program.target``). The device helpers the emitter calls (opaque moves,
the step load, table loads, the mailbox's ``Addr`` and debug hooks, the kernel symbol)
come from the device package (``device.gfx950``: ``ops``, ``mailbox``), the launch grid and CTA
count from ``target.grid``.
"""

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from ..device.gfx950 import mailbox
from ..device.gfx950.ops import (
    fresh_s,
    fresh_s64,
    kernel_symbol,
    ld_i32,
    load_ptr64,
    uniform,
)
from ..plan import trace
from ..plan.program import (
    DUAL_PAGE,
    LAYER_SLOTS,
    OWN_ARG,
    PEER_TOP_ARG,
    PEERS_ARG,
    RANK_ARG,
    PlanError,
    Program,
)
from ..plan.trace import Space, enter_stage

_PKG = Path(__file__).resolve().parents[1]


_SOURCE_DIGEST: dict = (
    {}
)  # the source files' (path, mtime, size) -> their digest (the last one)


def _source_files() -> list[tuple[tuple[str, ...], str]]:
    """Every ``.py`` under the package: (its relative path's parts, its path), in ``sorted(Path.rglob)``'s order."""
    out = []
    for root, _, names in os.walk(_PKG):
        rel = os.path.relpath(root, _PKG)
        parts = () if rel == os.curdir else tuple(rel.split(os.sep))
        out += [
            (parts + (n,), os.path.join(root, n)) for n in names if n.endswith(".py")
        ]
    return sorted(out)


def source_digest() -> str:
    """Hash of every hi source file: part of each build key, because FlyDSL
    does not key helper sources reached only through dynamic control flow. Hashed again only when a
    file's mtime or size moved (a walk and a stat a file: ~2 ms, where the pathlib walk and the reads took
    ~32 ms a call, 14 of a snapshot's 17 s of composing; the same digest).
    """
    files = _source_files()
    sig = tuple(
        (p, st.st_mtime_ns, st.st_size) for p, f in files for st in (os.stat(f),)
    )
    digest = _SOURCE_DIGEST.get(sig)
    if digest is None:
        h = hashlib.sha256()
        for parts, f in files:
            h.update("/".join(parts).encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
        _SOURCE_DIGEST.clear()
        digest = _SOURCE_DIGEST[sig] = h.hexdigest()[:16]
    return digest


def _table_entry(table, i):
    """Entry ``i`` of the pointer table as a scalar load (constant address space, invariant)."""
    ptr = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<4>"), (fx.Int64(table) + fx.Int64(8 * i)).ir_value()
    ).result
    return fx.Int64(llvm.LoadOp(T.i64, ptr, alignment=8, invariant=True).result)


def load_table(table, names) -> dict:
    """The pointer table's entries (``names``, in order) as uniform Int64s, read
    once at kernel start with scalar loads (constant address space, invariant):
    reading it at every stage entry put a serial vector round trip (~0.75 µs)
    before each stage's first task."""
    out = {}
    for i, name in enumerate(names):
        addr = fx.Int64(table) + fx.Int64(8 * i)
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<4>"), addr.ir_value()).result
        out[name] = fx.Int64(
            llvm.LoadOp(T.i64, ptr, alignment=8, invariant=True).result
        )
    return out


def _load_step(step):
    """The step counter as a scalar load (constant address space: the scalar cache starts every launch cold, and the
    counter only changes between launches)."""
    ptr = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<4>"), fx.Int64(step).ir_value()
    ).result
    return fx.Int32(llvm.LoadOp(T.i32, ptr, alignment=4, invariant=True).result)


def _load_live(step, word: int = 1):
    """``Program.dyn_rows``: the launch's live rows (``word`` 1, the int32 after the step counter), spec (2) or batch
    (3): a scalar load, as the step's (it changes only between launches). Re-read at each stage entry rather than held
    across the kernel (a stage that uses none of them loads none: the loads are dead code then).
    """
    ptr = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<4>"), (fx.Int64(step) + fx.Int64(4 * word)).ir_value()
    ).result
    return fx.Int32(llvm.LoadOp(T.i32, ptr, alignment=4, invariant=True).result)


def first_task(bid, base, grid):
    """CTA ``bid``'s first task of a stage placed from CTA ``base``."""
    return (bid - base % grid + grid) % grid


traced = ASTRewriter.transform  # (what every device package's ``traced`` is)


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


@traced
def _fill_dead(k, put_fill, first, tasks: int, per: int, unit: int, rows: int, regs):
    """``Program.dyn_rows``: every task of a ``row_tasks`` / ``req_tasks`` stage (``per`` tasks a unit of ``unit``
    rows: a row, or a request's ``spec`` rows at the build's stride) puts part ``t % per`` of its unit's dead rows
    (``k.n`` ..) of the row-major regions ``regs`` [(name, pairs a row)] as (0, tag) pairs: a dead task fills its own
    rows, a live one the dead tail of its unit. The readers then poll every row with no clamp (each clamp holds an
    index register: the 16-row linears spilled with them)."""
    for t in range(first, tasks, k.grid):
        u, part = t // per, t % per
        for name, rp in regs:
            a = k.mb[name]
            share = -(-rp // per)
            for r in range_constexpr(unit):
                row = u * unit + r
                for j in range_constexpr(-(-share // k.threads)):
                    q = part * share + j * k.threads + k.tid
                    if (
                        (row >= k.n)
                        & (row < rows)
                        & (q < rp)
                        & (q < (part + 1) * share)
                    ):
                        put_fill(a, row * rp + q, k.tag)


@traced
def _task_once(k, emit, first, tasks):
    """A stage of at most one task a CTA without the loop: no preheader for LLVM to hoist the task's whole address
    set-up into ahead of its first instructions (an entry-issued stage's, ``Stage.pre``).
    """
    if first < tasks:
        emit(k, fx.Int32(first))


def _stage_ctx(program: Program, base: SimpleNamespace, stage) -> SimpleNamespace:
    """The context a stage's tasks see (attribute reads only)."""
    t = program.target
    tag, bid, scratch = base.tag, base.bid, fresh_s64(base.scratch)
    wave = args = (
        None  # (the opaque copies in their order of before, after the mailboxes: same code)
    )
    rows_of = _region_rows(program) if program.dyn_rows else {}
    pairs_of = {r.name: r.pairs for r in program.regions} if rows_of else {}
    row_kw = lambda name: (
        {"row_pairs": pairs_of[name] // rows_of[name], "nbytes": pairs_of[name] * 8}
        if name in rows_of
        else {}
    )
    mbs = {
        r.name: mailbox.Addr(
            r.name,
            r.space,
            scratch + fx.Int64(program.scratch[r.name]),
            **row_kw(r.name),
        )
        for r in program.regions
        if r.space is not Space.PEER and r.name not in program.private
    }
    if program.private:  # the atomics' regions: the launch's own private buffer
        priv = fresh_s64(base.priv)
        mbs.update({name: mailbox.Addr(name, Space.SCRATCH, priv + fx.Int64(off), **row_kw(name))
                    for name, off in program.private.items()})  # fmt: skip
    if (
        program.dual_on
    ):  # by the regions' resources: loads read this CTA's copy (``base.dual``)
        from ..device.gfx950 import dual

        bits = fresh_s(base.dual)
        if program.dual_pages == 1:
            dual.local(bits * dual.HALF)
        else:  # a page's copy bit at bit 8 x page
            for pg in range(program.dual_pages):
                dual.local(((bits >> (8 * pg)) & 1) * dual.HALF, pg)
        for name in program.dual_regions:
            dual.root(
                mbs[name].value,
                name in program.dual_spread,
                program.scratch[name] // DUAL_PAGE,
            )
    peers = rank = None
    if program.peer:
        peers, rank = fresh_s64(base.peers), fresh_s(base.rank)
        # fast entry: this rank's peer buffer from the table, a scalar load a stage (an opaque table pointer: not one
        # SGPR pair held across the kernel, as the vector load of ``peers[rank]`` a stage was not)
        own = (_table_entry(fresh_s64(base.table), program.args.index(OWN_ARG)) if program.fast_entry
               else load_ptr64(peers, rank))  # fmt: skip
        mbs.update(
            {
                name: mailbox.Addr(name, Space.PEER, own + fx.Int64(off))
                for name, off in program.peer.items()
            }
        )
    if (
        wave is None
    ):  # our fork's order: lane, then the wave's and the args' opaque copies
        lane = base.tid % t.wave
        wave = fresh_s(base.wave)
        args = {name: fresh_s64(base.args[name]) for name in stage.args}
    tid = base.tid
    k = SimpleNamespace(
        program=program,
        scenario=program.scenario,
        rows=program.scenario.rows,
        # the live rows (``Program.dyn_rows``): a uniform Int32 read at the stage's entry, else the Python int ``rows``
        n=(
            _load_live(fresh_s64(base.step))
            if program.dyn_rows
            else program.scenario.rows
        ),
        # ... and the live spec and batch (``n = b x s``, rows packed at stride ``s``); Python ints without dyn_rows
        s=(
            _load_live(fresh_s64(base.step), 2)
            if program.dyn_rows
            else program.scenario.spec
        ),
        b=(
            _load_live(fresh_s64(base.step), 3)
            if program.dyn_rows
            else program.scenario.batch
        ),
        grid=t.grid,
        threads=t.threads,
        tid=tid,
        lane=lane,
        wave=wave,
        bid=bid,
        tag=tag,
        base=stage.base
        % t.grid,  # the stage's first CTA (Python int): task t runs on (base + t) % grid
        target=t,  # the program's target (Python): a stage plans with it, not module constants
        idle=stage.idle,
        # the kernel-start pointer loads themselves (no per-stage copy): for hooks that read
        # an arg the stage does not own (``Stage.idle``) without a stage-long SGPR pair
        kargs=base.args,
        lds=base.lds,
        ldsf=base.ldsf,
        # the table was read once at kernel start; opaque copies keep LLVM from
        # merging one stage's address math with another's
        args=args,
        mb=mbs,
        # carried LDS (``LdsCarry``) this stage writes or reads: name -> its LDS words (Int32 ptr)
        carry={
            name: base.lds + off for name, off in carried_by(program, stage).items()
        },
        tp=program.scenario.tp,
        rank=rank,
        peers=peers,
        own=(
            own if peers is not None else None
        ),  # this rank's peer buffer (the entry marks' dependency)
        # ``Program.peer_shared``: the group's peer buffer bytes (its parity-1 slots mirrored from there)
        peer_top=(
            fresh_s64(base.args[PEER_TOP_ARG])
            if peers is not None and program.peer_shared
            else None
        ),
        # ... and its step phase (``PeerGroup.end_step``: the peer table's entry ``tp``), added to the tag's parity
        peer_phase=(
            fresh_s(base.peer_phase)
            if peers is not None and program.peer_shared
            else None
        ),
        peer_pairs={r.name: r.pairs for r in program.regions if r.space is Space.PEER},
    )
    for (
        name,
        v,
    ) in (
        k.args.items()
    ):  # the memory-model record (no IR): accesses through these are the arg's
        trace.root(v, trace.ARG, name)
    trace.root(scratch, trace.INTERNAL, "scratch")
    if peers is not None:
        trace.root(peers, trace.INTERNAL, PEERS_ARG)
    return k


def _row_major(program: Program) -> dict[str, int]:
    """``Program.dyn_rows``: the row-major scratch regions and their rows: a channel's (``Region.rows``), and a region a
    ``Stage.row_tasks`` stage writes one pair a task of (a FLAG channel's flags: a dead row's task puts none).
    """
    out = {r.name: r.rows for r in program.regions if r.rows}
    for s in program.stages:
        if s.row_tasks:
            for r in s.writes:
                if r.name not in out and r.pairs == s.tasks:
                    out[r.name] = program.scenario.rows
    space = {r.name: r.space for r in program.regions}
    return {n: rows for n, rows in out.items() if space[n] is not Space.PEER}


def _bounded(stage) -> bool:
    """A dyn_rows stage that runs only its live rows' tasks (``row_tasks`` / ``req_tasks`` / ``live_shape``; not a
    ``live_tasks`` bound read from the launch's context, whose tasks serve every row).
    """
    return bool(stage.row_tasks or stage.req_tasks or stage.live_shape)


def _filled(program: Program) -> dict[str, int]:
    """The row-major regions a dyn_rows build makes whole by filling their dead rows (``_fill_dead``): no writer skips
    dead rows inside (``Stage.partial_rows``), and one at least runs only its live tasks (its dead tasks fill instead).
    """
    rm = _row_major(program)
    by = {s.name: s for s in program.stages}
    out = {}
    for r in program.regions:
        ws = [by[w] for w in r.writers if w in by]
        if (
            r.name in rm
            and ws
            and not any(s.partial_rows for s in ws)
            and any(_bounded(s) for s in ws)
        ):
            out[r.name] = rm[r.name]
    return out


def _region_rows(program: Program) -> dict[str, int]:
    """``Program.dyn_rows``: the row-major regions whose readers clamp and writers mask dead rows: those a
    ``Stage.partial_rows`` stage writes. Every other region is whole: its writers put every row (an unbounded stage computes its dead rows from its
    inputs, a bounded one fills them), so its readers poll every row unchanged."""
    filled = _filled(program)
    by = {s.name: s for s in program.stages}
    out = {}
    for r in program.regions:
        ws = [by[w] for w in r.writers if w in by]
        if any(s.partial_rows for s in ws) or (
            r.name not in filled and any(_bounded(s) for s in ws)
        ):
            out[r.name] = r
    rm = _row_major(program)
    return {n: rows for n, rows in rm.items() if n in out}


def _roots(base) -> None:
    """The memory-model record's roots (``plan.trace``): kernel-wide addresses and the launch tag."""
    for name, v in base.args.items():
        trace.root(v, trace.ARG, name)
    if getattr(base, "peers", None) is not None:
        trace.root(base.peers, trace.INTERNAL, PEERS_ARG)
    trace.root(base.tag, trace.TAG, "tag")


def carried_by(program: Program, stage) -> dict:
    """Word offsets of the carried LDS regions ``stage`` declares or reads (``LdsCarry``)."""
    offs = {name: off for name, off, _ in program.lds_layout.carried}
    out = {c.name: offs[c.name] for c in stage.carry}
    for s in program.stages:
        out.update({c.name: offs[c.name] for c in s.carry if stage.name in c.readers})
    return out


def _emit_stages(program: Program, base: SimpleNamespace) -> None:
    """Plain Python (not rewritten): the stage loop unrolls at trace time."""
    if program.dual_on:  # (gfx950's dual-copy mailboxes)
        from ..device.gfx950 import dual

        ctx = dual.emitting(True)
    else:
        ctx = nullcontext()
    with ctx:
        for i, stage in enumerate(program.stages):
            _emit_stage(program, base, i, stage)


def _prev_lds(program: Program, i: int) -> bool:
    """Whether the stage right before stage ``i`` uses LDS."""
    return i > 0 and bool(program.stages[i - 1].lds_words)


def _emit_stage(program: Program, base: SimpleNamespace, i: int, stage) -> None:
    _emit_stage_entry(program, base, i, stage)


def _emit_stage_entry(program: Program, base: SimpleNamespace, i: int, stage) -> None:
    enter_stage(stage.name)
    k = _stage_ctx(program, base, stage)
    if i > 0 and stage.lds_words and _prev_lds(program, i):
        # Stages overlay one LDS arena and a CTA's waves leave a stage at different
        # times: without this a fast wave's next-stage LDS writes clobber what a slow
        # sibling still reads.
        gpu.barrier()
    first = first_task(k.bid, stage.base, program.target.grid)
    dyn = program.dyn_rows
    with mailbox.live_rows(k.n) if dyn else nullcontext():
        _emit_stage_body(program, base, i, stage, k, first, dyn)


def _emit_stage_body(
    program: Program, base: SimpleNamespace, i: int, stage, k, first, dyn
) -> None:
    if (
        stage.pre is not None
    ):  # the stage's entry issue (``Stage.pre``), before the task loop's set-up
        k.pre = stage.pre(k, first)
    if stage.hold:
        _hold(first, stage.tasks, stage.hold)
    shape = dyn  # the live shape's task bounds
    live = stage.live_tasks is not None and (shape or not stage.live_shape)
    tasks = (
        stage.live_tasks(k) if live else stage.tasks
    )  # ``Stage.live_tasks``: a prefix
    if shape and stage.row_tasks:  # the live rows' tasks
        if stage.tasks != program.scenario.rows * stage.row_tasks:
            raise PlanError(
                f"{stage.name}: {stage.tasks} tasks, not rows x row_tasks {stage.row_tasks}"
            )
        live_rows = k.n * stage.row_tasks
        tasks = live_rows if not live else fx.min(tasks, live_rows)
    k.live = tasks  # (``emit`` maps a task from the live count without recomputing it)
    if stage.pre is not None and stage.tasks <= k.grid:
        _task_once(k, stage.emit, first, tasks)
    else:
        _task_loop(k, stage.emit, first, tasks, k.grid)
    if shape and (stage.row_tasks or stage.req_tasks):
        fill = _filled(program)  # the stage's tasks fill their units' dead rows
        regs = [
            (r.name, r.pairs // fill[r.name]) for r in stage.writes if r.name in fill
        ]
        if regs:
            per, unit = (
                (stage.row_tasks, 1)
                if stage.row_tasks
                else (stage.req_tasks, program.scenario.spec)
            )
            _fill_dead(
                k,
                mailbox.put_fill,
                first,
                stage.tasks,
                per,
                unit,
                program.scenario.rows,
                regs,
            )


def build(program: Program):
    """The ``@flyc.jit`` launcher of ``program`` (planned here if not yet):
    ``launch(step, layer, table, scratch, priv, stream=...)``."""
    t = program.target
    keyed = program.key() + (("sources", source_digest()),)
    lds_words = (
        program.lds_layout.words
    )  # the planner's: the arena, then the carried regions

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, lds_words, 16]

    @flyc.kernel(
        name=kernel_symbol(program.name, s=program.scenario.name, h=program.digest()),
        known_block_size=[t.threads, 1, 1],
    )
    def mono(step: Int64, layer: Int32, table: Int64, scratch: Int64, priv: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        for name, v in (
            ("step", step),
            ("table", table),
            ("scratch", scratch),
            ("priv", priv),
        ):
            trace.root(v, trace.INTERNAL, name)  # the memory-model record (no IR)
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        if const_expr(
            program.peer and program.fast_entry
        ):  # a scalar load with the table's (no vector round trip)
            tag = _load_step(step) * LAYER_SLOTS + layer + 1
        else:
            tag = uniform(ld_i32(step, 0)) * LAYER_SLOTS + layer + 1
        base = SimpleNamespace(
            tid=tid,
            bid=bid,
            wave=uniform(tid // t.wave),
            tag=tag,
            step=step,
            lds=lds,
            ldsf=fx.recast_iter(fx.Float32, lds),
            args=load_table(table, program.args),
            table=table,
            scratch=scratch,
            priv=priv,
        )
        if const_expr(
            program.dual_on
        ):  # the copy bits of this CTA's XCD, bit 8 p for page p (word 0 of priv)
            hdr = llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<4>"), fx.Int64(priv).ir_value()
            )
            w = fx.Int32(
                llvm.LoadOp(T.i32, hdr.result, alignment=4, invariant=True).result
            )
            # the hardware's XCD: CTA id mod 8 is one only up to the launch's first XCD
            xcc = llvm.InlineAsmOp(
                T.i32,
                [],
                "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)",
                "=s",
                has_side_effects=False,
            )
            base.dual = (w >> (fx.Int32(xcc.result) & 7)) & sum(
                1 << (8 * pg) for pg in range(program.dual_pages)
            )
        if const_expr(
            program.peer and program.fast_entry
        ):  # scalar loads with the other entries (one round trip)
            base.peers = base.args[PEERS_ARG]
            base.rank = fx.Int32(
                base.args[RANK_ARG]
            )  # a value, not an address: its low word
        elif const_expr(program.peer):
            # the rank is a value in the table, not an address: its low word
            base.peers = load_ptr64(table, program.args.index(PEERS_ARG))
            base.rank = uniform(ld_i32(table, 2 * program.args.index(RANK_ARG)))
        _roots(base)
        base.peer_phase = None
        if const_expr(
            program.peer and program.peer_shared
        ):  # the group's step phase (peer table entry tp, low word)
            base.peer_phase = uniform(ld_i32(base.peers, 2 * program.scenario.tp))
        _emit_stages(program, base)

    @flyc.jit
    def launch(
        step: Int64,
        layer: Int32,
        table: Int64,
        scratch: Int64,
        priv: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL's launch default
    ):
        _ = keyed
        mono(step, layer, table, scratch, priv).launch(
            grid=(t.grid,), block=(t.threads,), stream=stream
        )

    return launch
