# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A layer program's plan: plain Python, no FlyDSL.

A ``Program`` is an ordered list of ``Stage``s. Stage ``s`` runs ``s.tasks`` tasks; task ``t`` runs on CTA
``(s.base + t) % G`` (``G``: the program's ``target.grid``, the resident CTAs of a launch) and every CTA walks the
stages in program order. Every hand-off between stages goes through a ``Region`` of tagged (value, tag) pairs in
scratch; a region is declared by the stage(s) that put it and may only be polled by later stages (or its own stage
if it is an exchange). With all CTAs co-resident that is deadlock-free without any grid barrier.

``plan()`` resolves the layout: region byte offsets in scratch (256 B aligned; dual-copy regions first, in runs at
the start of a 2 MiB page), the LDS (the largest stage's arena, then the launch context's stash), default
placements, and the pointer-table ABI.

**Dual-copy regions** (``Program.dual``): a copy of every tagged scratch region of at most ``dual_max`` bytes on
each I/O die, laid out by MI355X's home rule (an address's home die flips every 8 KiB inside a 2 MiB page) in the
first page of scratch: producers store both copies, a CTA polls the copy of its die
(``device.mailbox``). Atomically updated regions (``Region.single``) keep one copy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from atom.models.qwen38.mono.kernels.target import MI355X, Target
from atom.mono.plan.trace import Space

PAIR_BYTES = 8  # a (value, tag) pair
REGION_ALIGN = 256
LAYER_SLOTS = 128  # launch tag = step * LAYER_SLOTS + layer + 1: layers 0..127 a step
# dual-copy regions: the home rule's page (the run area lies in one page)
DUAL_PAGE = 2 << 20
DUAL_RUN = 8192  # dual-copy regions: a run of one home (copy 0 in the even runs of the area, copy 1 in the odd)
DUAL_HEADER = 256  # dual-copy regions: bytes after the run area; word 0 = the copy of each XCD (runtime.dual)
CTX_WORDS = 8  # the launch context's LDS stash (``emit.kernel``: tag, CTA id, scratch, table, tl)


@dataclass(frozen=True)
class LdsLayout:
    """A program's LDS, in int32 words: the stage arena (stages overlay it) from word 0, then the launch context's
    stash at ``ctx`` (16 B aligned); ``words`` in all."""

    arena: int
    ctx: int
    words: int


@dataclass(frozen=True)
class Scenario:
    """What a launch serves: ``batch`` requests x ``spec`` rows each (1 = decode, >1 = MTP verify). Row ``r`` is
    request ``r // spec`` at offset ``r % spec``."""

    batch: int = 1
    spec: int = 1

    @property
    def rows(self) -> int:
        return self.batch * self.spec

    @property
    def name(self) -> str:
        return f"B{self.batch}S{self.spec}"


@dataclass(frozen=True)
class Region:
    """A hand-off buffer: ``pairs`` (value, tag) pairs, put by ``writers``."""

    name: str
    pairs: int
    writers: tuple[str, ...]
    space: Space = Space.SCRATCH
    exchange: bool = False  # polled by its own writer stage too (an all-reduce)
    # updated with atomics: one copy in a ``Program.dual`` build (never a dual region)
    single: bool = False


@dataclass
class Stage:
    """One stage: ``tasks`` tasks, each emitted by ``emit(k, t)`` (a traced free function or a closure over Python
    constants; ``k``: the stage's kernel context, ``t``: the traced task index)."""

    name: str
    tasks: int
    emit: Callable
    lds_words: int = 0  # int32 words of LDS one task needs
    writes: tuple[Region, ...] = ()  # the regions this stage puts
    args: tuple[str, ...] = ()  # pointer-table entries the stage reads
    base: int | None = None  # first CTA; None: placed after the previous stage
    key: tuple = ()  # every Python constant ``emit`` closes over (build key)
    # Entry hold: a CTA with a task of this stage sleeps ``hold`` s_sleep units (64 clocks each) before its first
    # task, so an early starter stops taking bandwidth from the critical path. 0: none.
    hold: int = 0


@dataclass
class Program:
    name: str
    scenario: Scenario
    stages: list[Stage]
    target: Target = MI355X
    # SGPR spills into VGPR lanes pass the resource gate (a spill to memory still shows as scratch and fails)
    sgpr_spill_ok: bool = False
    dual: bool = False  # dual-copy regions (module docstring)
    # the largest dual region in bytes (0: any that fits the run area)
    dual_max: int = 0
    # resolved by plan()
    scratch: dict[str, int] = field(default_factory=dict)  # region -> byte offset
    scratch_bytes: int = 0
    lds_words: int = 0
    args: tuple[str, ...] = ()  # pointer-table ABI, in order
    lds_layout: LdsLayout | None = None
    # dual regions: the run area at the start of scratch (both copies), then DUAL_HEADER bytes
    dual_bytes: int = 0
    dual_spread: tuple[
        str, ...
    ] = ()  # the dual regions larger than a run (their accesses are placed)
    dual_regions: tuple[str, ...] = ()  # the regions laid out with two copies

    @property
    def dual_on(self) -> bool:
        """The program has dual-copy regions (``plan``): the scratch header word, the copy choice, the hooks."""
        return self.dual_bytes > 0

    def is_dual(self, r: Region) -> bool:
        """Region ``r`` is selected for a copy per I/O die (one that does not fit keeps one copy)."""
        if r.single or not self.dual:
            return False
        return not self.dual_max or r.pairs * PAIR_BYTES <= self.dual_max

    @property
    def regions(self) -> list[Region]:
        return [r for s in self.stages for r in s.writes]

    def key(self) -> tuple:
        """Every build-relevant value, as scalars: the JIT cache key and the kernel symbol's digest (a closure's
        Python objects are not keyed by FlyDSL). Stage keys go through ``buildkey.stable``.
        """
        from atom.models.qwen38.mono.kernels.buildkey import stable

        out = [self.name, self.scenario.name, self.target.arch, self.sgpr_spill_ok]
        if self.dual_on:
            out.append(f"dual@{self.dual_bytes}:" + ",".join(self.dual_spread))
        if self.target.grid != self.target.cus:
            out.append(f"grid:{self.target.grid}")
        out += [self.scratch_bytes, self.lds_words]
        out += list(self.args)
        for s in self.stages:
            out += [
                s.name,
                s.tasks,
                s.base,
                s.lds_words,
                repr(stable(s.key)),
                getattr(s.emit, "__qualname__", ""),
            ]
            if s.hold:
                out.append(f"hold:{s.hold}")
            out += [
                f"{r.name}:{r.pairs}:{r.space.value}:{r.exchange}:{','.join(r.writers)}"
                for r in s.writes
            ]
        out += [f"{k}@{v}" for k, v in sorted(self.scratch.items())]
        return tuple(out)

    def digest(self) -> str:
        return hashlib.sha256(repr(self.key()).encode()).hexdigest()[:12]


class PlanError(ValueError):
    pass


def _dual_size(r: Region) -> int:
    return -(-r.pairs * PAIR_BYTES // REGION_ALIGN) * REGION_ALIGN


def _dual_pack(
    regions: list[Region],
) -> tuple[dict[str, int], list[Region], tuple, int]:
    """The run area's layout of ``regions`` (in this order of preference) -> (copy 0 offsets, the regions that do not
    fit, the spread ones, the area's bytes).

    The area is a sequence of 16 KiB slots [copy 0 run | copy 1 run] in one DUAL_PAGE: a region up to a run long
    sits in one run (first fit, 256 B aligned), a longer one takes whole consecutive slots (logical byte o at
    2 (o & ~8191) + o % 8192, ``mailbox.spread_byte``). Copy 1 is DUAL_RUN bytes after copy 0.
    """
    slot, fill = 2 * DUAL_RUN, []  # fill[i]: bytes used of slot i's run
    per = DUAL_PAGE // slot  # slots a page
    offs, out, spread = {}, [], []
    for r in regions:
        size = _dual_size(r)
        if size > DUAL_RUN:
            n = -(-size // DUAL_RUN)
            at = len(fill)
            if n > per or at + n > per:
                out.append(r)
                continue
            offs[r.name] = at * slot
            fill += [DUAL_RUN] * (n - 1) + [size - (n - 1) * DUAL_RUN]
            spread.append(r.name)
            continue
        i = next((i for i, f in enumerate(fill) if f + size <= DUAL_RUN), None)
        if i is None:
            if len(fill) + 1 > per:
                out.append(r)
                continue
            fill.append(0)
            i = len(fill) - 1
        offs[r.name] = i * slot + fill[i]
        fill[i] += size
    return offs, out, tuple(sorted(spread)), len(fill) * slot


def _dual_layout(program: Program, regions: list[Region]) -> list[Region]:
    """Lay the dual regions out in runs at the start of scratch (``_dual_pack``, the largest first) -> the regions
    left for the single-copy layout. ``scratch[name]`` is copy 0's start."""
    dual = [r for r in regions if program.is_dual(r)]
    offs, _, program.dual_spread, program.dual_bytes = _dual_pack(
        sorted(dual, key=lambda r: -r.pairs)
    )
    program.scratch.update(offs)
    program.dual_regions = tuple(r.name for r in dual if r.name in offs)
    return [r for r in regions if r.name not in offs]


def plan(program: Program) -> Program:
    """Resolve layout, placement and ABI in place; return ``program``."""
    t = program.target
    names = [s.name for s in program.stages]
    if len(set(names)) != len(names):
        raise PlanError(f"duplicate stage names {names}")
    off = 0
    regions = program.regions
    if len({r.name for r in regions}) != len(regions):
        raise PlanError(
            "a region is declared by two stages; declare it once with several writers"
        )
    program.dual_bytes, program.dual_spread, program.dual_regions = 0, (), ()
    if any(program.is_dual(r) for r in regions):
        regions = _dual_layout(program, regions)
        off = program.dual_bytes + DUAL_HEADER
    for r in regions:
        off = -(-off // REGION_ALIGN) * REGION_ALIGN
        program.scratch[r.name] = off
        off += r.pairs * PAIR_BYTES
    program.scratch_bytes = -(-off // REGION_ALIGN) * REGION_ALIGN
    program.lds_words = max([s.lds_words for s in program.stages] + [0])
    ctx = -(-max(program.lds_words, 4) // 4) * 4  # 16 B aligned
    program.lds_layout = LdsLayout(program.lds_words, ctx, ctx + CTX_WORDS)
    if program.lds_layout.words * 4 > t.lds_bytes:
        raise PlanError(
            f"LDS {program.lds_layout.words * 4} B (stages {program.lds_words * 4} B) > {t.lds_bytes} B"
        )
    nxt = 0
    for s in program.stages:
        if s.tasks < 1:
            raise PlanError(f"stage {s.name} has {s.tasks} tasks")
        if s.base is None:
            s.base = nxt % t.grid
        nxt = s.base + s.tasks
    args = []
    for s in program.stages:
        args += [a for a in s.args if a not in args]
    program.args = tuple(args)
    return program
