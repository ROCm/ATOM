# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A monokernel's plan: plain Python, no FlyDSL.

A ``Program`` is an ordered list of ``Stage``s. Stage ``s`` runs ``s.tasks``
tasks; task ``t`` runs on CTA ``(s.base + t) % G`` (``G``: the program's
``target.grid``, the resident CTAs of a launch) and every CTA walks the
stages in program order. Every hand-off between stages goes through a
``Region`` of tagged (value, tag) pairs in scratch; a region is declared by the
stage(s) that put it and may only be polled by later stages (or its own stage
if it is an exchange). With all CTAs co-resident that is deadlock-free without
any grid barrier. One exception, declared: a stage's ``prev_reads`` regions are
the *previous launch's*, written by the program's last stage (a cross-launch
hand-off: that launch has completed every stage, so they are all there).

``plan()`` resolves the layout: region byte offsets in scratch (256 B aligned),
the LDS arena (the largest stage's need: stages run one after another on a CTA,
so their LDS overlays), default placements, and the pointer-table ABI.

**Tensor parallelism** (``Scenario.tp`` > 1): every rank launches the same program on its own GPU.
Regions in ``Space.PEER`` live in a symmetric per-rank peer buffer (``runtime.peer``), not in
scratch: each has ``PEER_SLOTS`` x ``tp`` copies of ``pairs`` pairs, [launch-tag parity][source rank],
so rank ``r`` puts into slot ``r`` of every peer's buffer and polls the ``tp`` slots of its own
(``device.gfx950.peer``). Two parity slots suffice when every launch on a peer group
exchanges with every rank (a rank then cannot get a launch ahead of a peer still polling): the
runtime checks that consecutive launches on one peer group alternate tag parity. Programs with peer
regions get two more ABI entries, ``PEERS_ARG`` (the int64 table of every rank's peer buffer) and
``RANK_ARG`` (this rank, as a value).

**Programs of different peer layouts on one group** (``Program.peer_shared``): the parity argument above
holds for one program's regions only. A rank that has finished launch N may already make launch N + 1's
early puts (a stage that waits on nothing of its own launch, e.g. a chained seam summing the previous
launch's partials) while a slower peer still polls launch N's slots; with another program's layout those
puts can land on exactly those slots (the next launch's tag over this launch's: the poll never completes).
A shared program lays each region's parity-0 copy out from the buffer's bottom as usual (``tp`` slots)
and its parity-1 copy mirrored from the group's top (``PEER_TOP_ARG``, the group's bytes, bound at table
time): every program's parity-0 copies lie in ``[0, S)`` and its parity-1 copies in ``[top - S, top)``
(``S``: its ``peer_bytes`` / 2), disjoint for any two programs whose ``peer_bytes`` the group holds; a
region at the same offset and size in two programs (``peer_first``: the cross-launch regions) keeps one
address in both.

Across steps: a step of an odd number of launches on a group ends on the parity it began with, and the next step's
first launch (often one that reads nothing of its launch: a step's first layer) would share the previous launch's
copy. A shared program adds its group's step phase to its tag's parity (the peer table's entry ``tp``, low word),
which ``PeerGroup.end_step`` after a step's last launch flips when the step's launch count was odd: consecutive
launches alternate across steps of any widths.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from ..target import MI355X, Target
from .trace import Space

PAIR_BYTES = 8  # a (value, tag) pair
HOST = (
    "host"  # writer name of regions the host publishes before a launch (tests, shells)
)
REGION_ALIGN = 256
LAYER_SLOTS = 128  # launch tag = step * LAYER_SLOTS + layer + 1: layers 0..127 a step
PEER_SLOTS = 2  # copies of a peer region by launch-tag parity (module docstring)
PEERS_ARG, RANK_ARG = (
    "_peers",
    "_rank",
)  # the ABI entries of a program with peer regions
OWN_ARG = "_own"  # ... and with ``Program.fast_entry``: this rank's peer buffer
PEER_TOP_ARG = "_peer_top"  # ... and with ``Program.peer_shared``: the group's peer buffer bytes, as a value
_SHARE_PEER_GROUPS = False  # ``share_peer_groups``: ``Program.peer_shared``'s default for programs planned from now
DUAL_PAGE = (
    2 << 20
)  # dual-copy regions: the home rule's page (a region's runs lie in one page)
DUAL_PAGES = 4  # dual-copy regions: pages of run area at most (the scratch header word: 8 bits, one an XCD, a page)
DUAL_RUN = 8192  # dual-copy regions: a run of one home (copy 0 in the even runs of the area, copy 1 in the odd)
DUAL_HEADER = 256  # dual-copy regions: bytes after the run area (kept free)
# a program's private buffer (the launch's ``priv`` argument, its own, never shared with another program's scratch):
# word 0 the dual-copy word (the copy of each XCD, ``runtime.dual``), the regions updated with atomics
# (``Region.single``: their counts must start from a multiple of the adds a launch makes) from PRIVATE_HEADER on
PRIVATE_HEADER = 256


def share_peer_groups(on: bool = True) -> None:
    """Programs planned from now on that leave ``Program.peer_shared`` unset share their peer groups with programs of
    other peer layouts (module docstring): what an engine hook calls once, before it builds its programs, when it
    launches several programs on one ``PeerGroup`` (a chain of layer kinds, a head, a drafter's stages).
    """
    global _SHARE_PEER_GROUPS
    _SHARE_PEER_GROUPS = bool(on)


@dataclass(frozen=True)
class LdsLayout:
    """A program's LDS, in int32 words (``plan``): the stage arena (stages overlay it) from word 0, then the carried
    regions; ``words`` in all."""

    arena: int
    words: int
    # carried regions (``LdsCarry``): ((name, word offset, words), ...); each lies past the arena of
    # every stage its interval spans and past the regions whose intervals meet it (``plan``)
    carried: tuple = ()


@dataclass(frozen=True)
class LdsCarry:
    """LDS one stage hands to later stages of the same program on the same CTA (
    e.g. weights staged during an earlier stage). Declared by its writer
    stage (``Stage.carry``): ``words`` int32 words, read by the stages named in ``readers`` (later
    in program order).

    ``plan`` places it where no stage of its interval (writer or entry .. last reader) has its
    arena, so stages between keep their own LDS, and stages outside the interval may still overlay
    it. Contract (``plan``): every CTA that runs a reader task runs a writer task (the content is
    per CTA). The emitter joins (a CTA barrier) before each reader stage; the stage context gives writers and readers ``k.carry[name]``. Unified
    programs: ``carry_spans`` (its rules and the spans per variant)."""

    name: str
    words: int
    readers: tuple[str, ...]


@dataclass(frozen=True)
class LdsKeep:
    """LDS words ``lo .. hi`` a stage leaves in place, at an offset of its block's own layout, for later stages of the
    same block on the same CTA (``readers``): e.g. MoEExperts' rt slot table, read by its ug / down tasks. Unlike an
    ``LdsCarry`` the planner does not place it, so it constrains nothing in the build; ``plan`` only refuses a stage
    between the writer and the last reader, on a CTA they share, whose arena reaches ``lo`` (it would overwrite the
    words: an inserted or re-placed stage). ``aware``: stages that keep
    the layout (the block's own; their ``lds_words`` may cover the range). Plan-time only: not part of the build key.
    """

    name: str
    lo: int
    hi: int
    readers: tuple[str, ...]
    aware: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """What a launch serves: ``batch`` requests x ``spec`` rows each (1 = decode,
    >1 = MTP verify). Row ``r`` is request ``r // spec`` at offset ``r % spec``."""

    batch: int = 1
    spec: int = 1
    tp: int = 1  # tensor-parallel ranks running the program together (module docstring)

    @property
    def rows(self) -> int:
        return self.batch * self.spec

    @property
    def name(self) -> str:
        return f"B{self.batch}S{self.spec}" + (f"T{self.tp}" if self.tp > 1 else "")


@dataclass(frozen=True)
class Region:
    """A hand-off buffer: ``pairs`` (value, tag) pairs, put by ``writers``."""

    name: str
    pairs: int
    writers: tuple[str, ...]
    space: Space = Space.SCRATCH
    exchange: bool = False  # polled by its own writer stage too (an all-reduce)
    # "": one copy unless the program's ``dual`` selection takes it. "numa2": a dual-copy region in any program, one
    # copy homed on each I/O die (``Target`` NUMA notes; the layout, stores and loads of ``Program.dual``'s regions:
    # ``device.gfx950.dual``); the bytes of each copy are the single copy's (same formats, same bits).
    # Scratch only, not host regions; tagged regions and flags (a bulk + flag payload also works, but the producer's
    # drain then waits for the far copy). It must fit the run area (``PlanError``). ``numa_stages`` selects regions
    # this way.
    home: str = ""
    single: bool = (
        False  # updated with atomics: one copy in a ``Program.dual`` build (never a dual region)
    )
    # row-major over the scenario's rows (``Channel.regions``): pair p is row p // (pairs // rows). Read only by a
    # ``Program.dyn_rows`` build (its dead rows' polls / puts are masked); 0: not row-major. Not part of the key.
    rows: int = 0


@dataclass
class Stage:
    """One stage: ``tasks`` tasks, each emitted by ``emit(k, t)``.

    ``emit`` is a traced free function (or a closure over Python constants):
    ``k`` is the stage's kernel context (``emit.kernel._stage_ctx``), ``t``
    the traced task index. It must not call methods on, or assign attributes of,
    local objects inside dynamic control flow.
    """

    name: str
    tasks: int
    emit: Callable
    lds_words: int = 0  # int32 words of LDS one task needs
    writes: tuple[Region, ...] = ()  # the regions this stage puts
    args: tuple[str, ...] = ()  # pointer-table entries the stage reads
    base: int | None = None  # first CTA; None: placed after the previous stage
    key: tuple = ()  # every Python constant ``emit`` closes over (build key)
    # LDS this stage hands to later stages on the same CTA (``LdsCarry``); none: as before
    carry: tuple = ()
    # LDS this stage leaves in its block's own layout for later stages (``LdsKeep``: a plan-time check, not keyed)
    lds_keep: tuple = ()
    # Work a task does while it waits (``blocks.gfx950.warm.Idle``: e.g. warming a later
    # stage's weights). A block that supports it calls ``idle_issue`` where its own loads
    # are in flight and it is about to wait, and ``idle_sink`` at the task's end; its repr
    # is part of the build key. None: nothing.
    idle: object = None
    # Regions this stage polls as the *previous launch* wrote them (its tag and parity: e.g. a layer's first seam
    # summing the previous layer's TP partials, ``peer.sum_partials(tag=)``). Each must be written by the program's
    # last stage only (``plan.check``), and the previous launch must be this program's or one sharing its peer
    # layout. A step's first layer has no such producer: it is a program variant without the read (plain input).
    prev_reads: tuple[str, ...] = ()
    # Args this stage stores only for later launches (a cache's new entries): no stage of this launch uses the
    # elements it stores (they reach this launch's readers through a channel; a reader may load the stale bytes and
    # replace them). ``plan.check.check_memory`` takes it at its word for these args' same-launch pairs (the
    # many-launch check is what tests the claim); not part of the build.
    later_args: tuple[str, ...] = ()
    # What the stage moves and computes, for its speed-of-light floor (a cost, or a callable of the
    # scenario returning one). Measurement metadata only: not part of the build key, never read by the emitter.
    cost: object = None
    # Entry hold: a CTA with a task of this stage sleeps ``hold`` s_sleep units (64 clocks each) before its first
    # task, so an early starter stops taking bandwidth from the critical path (a stage hold). 0: none
    # (the build is unchanged). A causal probe sets it one stage at a time.
    hold: int = 0
    # With ``hold``, which CTAs with a task hold: "" every one; "first" those with no task of an earlier stage (of every
    # variant this stage runs in: the stage is their first work); "after" those with one
    hold_where: str = ""
    # Entry issue: ``pre(k, first)`` runs before the stage's task loop (and before the loop-invariant set-up LLVM hoists
    # there), on every CTA (``first``: the CTA's first task, >= ``tasks`` on a CTA without one: clamp what it
    # addresses); its value is ``k.pre`` in ``emit``. For a stage whose first task's loads should be in flight from
    # the stage's first instructions (a layer's first poll). None: nothing (the build is unchanged).
    pre: Callable | None = None
    # Live tasks: ``live_tasks(k)`` -> the tasks this launch runs (a traced Int32, at most ``tasks``), from what the
    # launch reads (e.g. a context length from ``pos``): the stage's task loop stops there, so tasks past it cost
    # nothing; ``emit`` sees the count as ``k.live``. The block orders its tasks so the live ones are a prefix and
    # keys the choice in ``key``. Or (``live_shape``) a count from a dyn_rows build's live shape. None: every task (the
    # build is unchanged).
    live_tasks: Callable | None = None
    # ``Program.dyn_rows``: the stage's tasks are ``row_tasks`` a row, row-major (tasks == rows x row_tasks: task t
    # serves row t // row_tasks), so a dyn_rows build runs the first n x row_tasks only. 0: every task (the stage
    # handles dead rows itself, or has none). Not part of the key (no code unless dyn_rows).
    row_tasks: int = 0
    # ... or ``req_tasks`` a request, request-major (tasks == batch x req_tasks): the first b x req_tasks run.
    req_tasks: int = 0
    # ``Program.dyn_rows``: ``live_tasks`` is a count from the live shape (``k.n``, ``k.s``, ``k.b``: e.g. the row
    # chunks holding a live row, the live requests' data-dependent splits), so the stage runs its live rows' tasks
    # only, as a ``row_tasks`` / ``req_tasks`` one does; it is called in dyn_rows builds only (a static build runs every
    # task). Not part of the key.
    live_shape: bool = False
    # ``Program.dyn_rows``: the stage computes and puts every row, dead ones included (from its clamped inputs, e.g. a
    # Linear's row tiles): a region all of whose writers do so is whole, so its puts are not masked and its polls not
    # clamped (each clamp held its own index register: a linear's polls spilled). Not part of the key.
    all_rows: bool = False
    # ``Program.dyn_rows``: the stage skips dead rows' work inside (``k.n`` in its own code, e.g. an MoE route counting
    # the live picks only), so the row-major regions it writes may hold stale dead rows: their readers clamp dead rows
    # onto the last live pair and its dead rows' puts are dropped. Any other stage puts every row (an unbounded stage
    # computes them from its inputs, a bounded one fills them). Not part of the key.
    partial_rows: bool = False


@dataclass
class Program:
    name: str
    scenario: Scenario
    stages: list[Stage]
    target: Target = MI355X
    # SGPR spills into VGPR lanes pass the resource gate (a composition over the SGPR budget: lane moves, no memory);
    # a spill to memory still shows as scratch and fails
    sgpr_spill_ok: bool = False
    host_regions: tuple[Region, ...] = ()  # regions the host publishes (writer HOST)
    # peer regions laid out first, in this order (the rest after them in declaration order): a chain of programs pins
    # the regions one launch writes and the next reads (``Stage.prev_reads``) at the same offset in every program, and
    # every other region lands after them, so no program's own regions overlap them
    peer_first: tuple[str, ...] = ()
    # the program shares its peer group with programs of other peer layouts (module docstring): parity-1 copies
    # mirrored from the group's top (``PEER_TOP_ARG``). None (resolved by plan()): ``share_peer_groups``' setting,
    # off by default; off: the two copies side by side in each region, the layout, key and code of before
    peer_shared: bool | None = None
    # plain targets (args / dense regions) whose same-launch hand-off problems ``plan.check.check_memory`` reports
    # (``MonoKernel.record.waived``) instead of raising: a documented finding awaiting its fix, never a default
    plain_waive: tuple[str, ...] = ()
    # A short kernel entry for TP programs: this rank's peer buffer is a pointer-table entry (``OWN_ARG``) and the
    # peer table / rank / own entries are scalar loads with the others, instead of vector loads of the peer table and
    # then of its own entry (two dependent round trips before a stage can address a peer region). Off: unchanged.
    # None (resolved by plan()): on when a stage asks for an entry issue (``Stage.pre``: its first loads are only as
    # early as the peer base they address)
    fast_entry: bool | None = None
    # Symbolic shapes (the dyn measurements): the program is planned for its scenario (the maximum: R
    # rows, B x S) and every launch serves the live rows ``n``, spec ``s`` and batch ``b`` read from the words after
    # the step counter (``Buffers.set_live``): stages see them as ``k.n``, ``k.s``, ``k.b`` (uniform Int32s; without
    # dyn_rows the scenario's Python ints, so code written against them traces exactly as before). Rows >= n are dead:
    # bounded stages run the live tasks only and fill the dead rows' slots. Off: unchanged (key, code).
    dyn_rows: bool = False
    # resolved by plan()
    scratch: dict[str, int] = field(default_factory=dict)  # region -> byte offset
    scratch_bytes: int = 0
    lds_words: int = 0
    args: tuple[str, ...] = ()  # pointer-table ABI, in order
    peer: dict[str, int] = field(
        default_factory=dict
    )  # peer region -> byte offset in the peer buffer
    peer_bytes: int = 0  # the peer buffer a rank needs (0: no peer regions)
    lds_layout: LdsLayout | None = None  # the arena and the emitter's reserved words
    dual_bytes: int = (
        0  # dual regions: the run area at the start of scratch (both copies), then DUAL_HEADER bytes
    )
    dual_spread: tuple[
        str, ...
    ] = ()  # the dual regions larger than a run (their accesses are placed)
    dual_regions: tuple[
        str, ...
    ] = ()  # the regions laid out with two copies (a selected one may not fit)
    private: dict[str, int] = field(
        default_factory=dict
    )  # Region.single scratch regions -> byte offset in priv
    private_bytes: int = (
        0  # the private buffer a launch needs (PRIVATE_HEADER at least)
    )

    @property
    def dual_on(self) -> bool:
        """The program has dual-copy regions (``plan``): the scratch header word, the copy choice, the hooks."""
        return self.dual_bytes > 0

    @property
    def dual_pages(self) -> int:
        """Pages of the run area (each with its own copy bits in the header word)."""
        return -(-self.dual_bytes // DUAL_PAGE)

    def is_dual(self, r: Region) -> bool:
        """Region ``r`` is laid out with a copy per I/O die: ``Region.home`` "numa2" (its layout: ``plan``)."""
        if r.single or r.space is not Space.SCRATCH or r in self.host_regions:
            return False
        return bool(r.home)

    @property
    def regions(self) -> list[Region]:
        return list(self.host_regions) + [r for s in self.stages for r in s.writes]

    def key(self) -> tuple:
        """Every build-relevant value, as scalars: the JIT cache key (a closure's
        Python objects are not keyed by FlyDSL, see atom.mono.plan.build_key). Stage keys go
        through ``buildkey.stable``: a parameter dataclass keyed by its fields off their defaults,
        so a new defaulted field leaves existing programs' keys alone."""
        from .buildkey import KEY_VERSION, stable

        # (False, False: the timeline and lds_guard debug builds' flags of before, kept so the keys stay)
        out = [self.name, self.scenario.name, self.target.arch, False, False]
        if self.fast_entry and self.peer:  # keyed only when it changes the build
            out.append("fast_entry")
        if self.dyn_rows:  # keyed only when on
            out.append("dyn_rows")
        if (
            self.dual_on
        ):  # keyed only when on (the layout itself is in the offsets below)
            out.append(f"dual@{self.dual_bytes}:" + ",".join(self.dual_spread))
        if self.target.grid != self.target.cus:  # keyed only when not one CTA per CU
            out.append(f"grid:{self.target.grid}")
        out += [self.scratch_bytes, self.lds_words]
        out += list(self.args)
        out += [f"host:{r.name}:{r.pairs}" for r in self.host_regions]
        for s in self.stages:
            out += [
                s.name,
                s.tasks,
                s.base,
                s.lds_words,
                repr(stable(s.key)),
                getattr(s.emit, "__qualname__", ""),
            ]
            if s.idle is not None:
                out.append(f"idle:{stable(s.idle)!r}")
            if s.hold:  # keyed only when used: other programs keep their key
                out.append(
                    f"hold:{s.hold}" + (f":{s.hold_where}" if s.hold_where else "")
                )
            if s.pre is not None:  # keyed only when used
                out.append(
                    f"pre:{getattr(getattr(s.pre, 'func', s.pre), '__qualname__', '')}"
                )
            if s.prev_reads:  # keyed only when used: other programs keep their key
                out.append(f"prev:{','.join(s.prev_reads)}")
            if (
                s.carry
            ):  # keyed only when used (":early:None": the text of the keys before, kept)
                out.append(f"carry:{stable(s.carry)!r}:early:None")
            out += [
                f"{r.name}:{r.pairs}:{r.space.value}:{r.exchange}:{','.join(r.writers)}"
                + (f":{r.home}" if r.home else "")
                for r in s.writes
            ]
        out += [f"{k}@{v}" for k, v in sorted(self.scratch.items())]
        if self.peer:  # keyed only with peer regions: TP1 programs keep their key
            out += [f"peer:{k}@{v}" for k, v in sorted(self.peer.items())] + [
                f"peer_bytes:{self.peer_bytes}"
            ]
            if self.peer_shared:  # keyed only when on
                out.append("peer_shared")
        if self.lds_layout is not None and self.lds_layout.carried:
            out.append(f"carried:{stable(self.lds_layout.carried)!r}")
        out.append(f"key:v{KEY_VERSION}")
        return tuple(out)

    def digest(self) -> str:
        return hashlib.sha256(repr(self.key()).encode()).hexdigest()[:12]

    def describe(self) -> str:
        """Human-readable plan, for diffs and agents."""
        lines = [
            f"program {self.name} {self.scenario.name}: scratch {self.scratch_bytes} B, LDS {self.lds_words * 4} B"
        ]
        for s in self.stages:
            writes = ", ".join(f"{r.name}[{r.pairs}]" for r in s.writes)
            lines.append(
                f"  {s.name:<16} tasks {s.tasks:>4} base {s.base:>3} lds {s.lds_words * 4:>6} B  -> {writes}"
            )
        if self.dual_on:
            lines.append(f"  dual: {len(self.dual_regions)} regions, run area {self.dual_bytes} B, spread "
                         f"{', '.join(self.dual_spread) or '-'}")  # fmt: skip
        lines.append(f"  args: {', '.join(self.args)}")
        return "\n".join(lines)


class PlanError(ValueError):
    pass


def _numa_auto(r: Region, regions: set[str], max_bytes: int) -> bool:
    stem = r.name.rsplit(".", 1)[0] if r.name.endswith((".q", ".s")) else r.name
    dense = f"{stem}.f" in regions and not r.name.endswith(
        ".f"
    )  # a FLAG channel's payload (untagged)
    return r.pairs * PAIR_BYTES <= max_bytes and not dense and "ctr" not in r.name


def _dual_size(r: Region) -> int:
    return -(-r.pairs * PAIR_BYTES // REGION_ALIGN) * REGION_ALIGN


def _dual_pack(
    regions: list[Region], pages: int = DUAL_PAGES
) -> tuple[dict[str, int], list[Region], tuple, int]:
    """The run area's layout of ``regions`` (in this order of preference) -> (copy 0 offsets, the regions that do not
    fit, the spread ones, the area's bytes).

    The area is a sequence of 16 KiB slots [copy 0 run | copy 1 run]: a region up to a run long sits in one run
    (first fit, 256 B aligned), a longer one takes whole consecutive slots (logical byte o at 2 (o & ~8191) + o %
    8192, ``device.gfx950.dual``). Copy 1 is DUAL_RUN bytes after copy 0. A region's slots lie in one
    DUAL_PAGE (the page whose runs alternate homes); the area takes up to ``pages`` pages.
    """
    slot, fill = 2 * DUAL_RUN, []  # fill[i]: bytes used of slot i's run
    per = DUAL_PAGE // slot  # slots a page
    offs, out, spread = {}, [], []
    for r in regions:
        size = _dual_size(r)
        if size > DUAL_RUN:
            n = -(-size // DUAL_RUN)
            at = len(fill)
            if (
                at % per + n > per
            ):  # not across a page boundary: the next page (the skipped slots stay free)
                at = -(-at // per) * per
            if n > per or at + n > pages * per:
                out.append(r)
                continue
            offs[r.name] = at * slot
            fill += (
                [0] * (at - len(fill))
                + [DUAL_RUN] * (n - 1)
                + [size - (n - 1) * DUAL_RUN]
            )
            spread.append(r.name)
            continue
        i = next((i for i, f in enumerate(fill) if f + size <= DUAL_RUN), None)
        if i is None:
            if len(fill) + 1 > pages * per:
                out.append(r)
                continue
            fill.append(0)
            i = len(fill) - 1
        offs[r.name] = i * slot + fill[i]
        fill[i] += size
    return offs, out, tuple(sorted(spread)), len(fill) * slot


def _dual_order(regions: list[Region]) -> list[Region]:
    """The packing order: regions that ask for copies (``Region.home``) first, then the largest first (the spread
    regions take whole slots)."""
    return sorted(regions, key=lambda r: (not r.home, -r.pairs))


def numa_stages(
    stages: list[Stage], names: tuple[str, ...], extra: tuple[Region, ...] = ()
) -> list[Stage]:
    """``stages`` (in place) with the named regions, and with ``auto[:KiB]`` every small tagged scratch region
    put as "numa2" (dual copies, ``Region.home``). For a composition that turns dual copies on for its
    programs; ``extra``: other regions of the program (host regions), to tell a FLAG payload by its sibling flag
    region. The automatic candidates are taken smallest first while their bytes fit one 2 MiB page (one copy each)
    and every chosen region still fits the run area (named ones always; ``plan`` raises if they do not fit).
    """
    regions = {r.name for r in extra} | {r.name for s in stages for r in s.writes}
    auto = next((n for n in names if n.split(":")[0] == "auto"), None)
    max_bytes = (int(auto.split(":")[1]) if auto and ":" in auto else 64) << 10

    def named(r):
        return any(r.name == n or r.name.startswith(n + ".") for n in names)

    def pick(r):
        return (
            r.space is Space.SCRATCH
            and not r.home
            and (named(r) or (auto and _numa_auto(r, regions, max_bytes)))
        )

    cands = sorted(
        {r.name: r for s in stages for r in s.writes if pick(r)}.values(),
        key=lambda r: r.pairs,
    )
    old = [
        r for s in stages for r in s.writes if r.home
    ]  # already dual: they keep their place in the area
    chosen = [r for r in cands if named(r)]
    budget = DUAL_PAGE - DUAL_HEADER - sum(_dual_size(r) for r in chosen)
    for r in cands:
        if named(r) or _dual_size(r) > budget:
            continue
        trial = [replace(x, home="numa2") for x in old + chosen + [r]]
        if not _dual_pack(_dual_order(trial))[1]:
            chosen.append(r)
            budget -= _dual_size(r)
    keep = {r.name for r in chosen}
    for s in stages:
        s.writes = tuple(
            replace(r, home="numa2") if r.name in keep else r for r in s.writes
        )
    return stages


def _dual_layout(program: Program, regions: list[Region]) -> list[Region]:
    """Lay the dual regions out in runs at the start of scratch (``_dual_pack``) -> the regions left for the
    single-copy layout. ``scratch[name]`` is copy 0's start. A ``dual`` selection that no longer fits keeps one copy;
    a region that asks for copies (``Region.home``) must fit."""
    dual = [r for r in regions if program.is_dual(r)]
    pages = DUAL_PAGES
    offs, out, program.dual_spread, program.dual_bytes = _dual_pack(
        _dual_order(dual), pages
    )
    bad = [r.name for r in out if r.home]
    if bad:
        raise PlanError(
            f"{program.name}: dual-copy regions {bad} do not fit the run area ({DUAL_PAGES} pages)"
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
    tp = program.scenario.tp
    peer_off = 0
    if program.peer_shared is None:
        program.peer_shared = _SHARE_PEER_GROUPS
    copies = (
        1 if program.peer_shared else PEER_SLOTS
    )  # (shared: the parity-1 copies mirrored from the top)
    first = {n: i for i, n in enumerate(program.peer_first)}
    regions = sorted(
        regions,
        key=lambda r: (
            (0, first[r.name]) if r.space is Space.PEER and r.name in first else (1, 0)
        ),
    )
    for r in regions:
        if r.home and (
            r.home != "numa2"
            or r.space is not Space.SCRATCH
            or r in program.host_regions
        ):
            raise PlanError(
                f"region {r.name}: home {r.home!r} (only 'numa2', scratch regions a stage writes)"
            )
    program.dual_bytes, program.dual_spread, program.dual_regions = 0, (), ()
    # the atomics' regions in the private buffer (the scratch may be shared with other programs: module docstring)
    priv_off, program.private = PRIVATE_HEADER, {}
    for r in regions:
        if r.single and r.space is Space.SCRATCH and r not in program.host_regions:
            priv_off = -(-priv_off // REGION_ALIGN) * REGION_ALIGN
            program.private[r.name] = priv_off
            priv_off += r.pairs * PAIR_BYTES
    program.private_bytes = -(-priv_off // REGION_ALIGN) * REGION_ALIGN
    regions = [r for r in regions if r.name not in program.private]
    if any(program.is_dual(r) for r in regions):  # dual copies
        regions = _dual_layout(program, regions)
        off = program.dual_bytes + DUAL_HEADER
    for r in regions:
        if r.space is Space.PEER:
            if tp < 2:
                raise PlanError(f"peer region {r.name} in a TP1 program")
            peer_off = -(-peer_off // REGION_ALIGN) * REGION_ALIGN
            program.peer[r.name] = peer_off
            peer_off += copies * tp * r.pairs * PAIR_BYTES
            continue
        off = -(-off // REGION_ALIGN) * REGION_ALIGN
        program.scratch[r.name] = off
        off += r.pairs * PAIR_BYTES
    program.scratch_bytes = -(-off // REGION_ALIGN) * REGION_ALIGN
    program.peer_bytes = (
        -(-peer_off // REGION_ALIGN)
        * REGION_ALIGN
        * (PEER_SLOTS if program.peer_shared else 1)
    )
    if not program.peer:  # (no peer regions: nothing to share)
        program.peer_shared = False
    program.lds_words = max([s.lds_words for s in program.stages] + [0])
    carried, top = _place_carried(program)
    words = -(-max(program.lds_words, top, 4) // 4) * 4  # 16 B aligned
    program.lds_layout = LdsLayout(program.lds_words, words, carried)
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
    _check_carried(program)
    _check_kept(program)
    if program.fast_entry is None:
        program.fast_entry = any(s.pre is not None for s in program.stages)
    args = []
    for s in program.stages:
        args += [a for a in s.args if a not in args]
    if program.peer:
        args += [PEERS_ARG, RANK_ARG] + ([OWN_ARG] if program.fast_entry else [])
        args += [PEER_TOP_ARG] if program.peer_shared else []
    program.args = tuple(args)
    return program


def carry_interval(stages: list[Stage], i: int, c: LdsCarry) -> tuple[int, int]:
    """(first, last) stage index of carried region ``c`` of stage ``i``: from its writer to its last reader."""
    index = {s.name: j for j, s in enumerate(stages)}
    missing = [r for r in c.readers if r not in index]
    if missing:
        raise PlanError(
            f"carried LDS {c.name} of {stages[i].name}: no reader stage {missing}"
        )
    if not c.readers:
        raise PlanError(f"carried LDS {c.name} of {stages[i].name} has no readers")
    last = max(index[r] for r in c.readers)
    for r in c.readers:
        if index[r] <= i:
            raise PlanError(
                f"carried LDS {c.name}: reader {r} does not run after its writer {stages[i].name}"
            )
    return i, last


def carry_spans(program: Program, i: int, c: LdsCarry) -> dict[int, set[int]]:
    """The stages (indices into ``program.stages``) during which carried region ``c`` of stage ``i`` must stay intact:
    {0: its ``carry_interval``}."""
    lo, hi = carry_interval(program.stages, i, c)
    return {0: set(range(lo, hi + 1))}


def _place_carried(program: Program) -> tuple[tuple, int]:
    """Word offsets of every carried region: past the arena of each stage of its spans
    (``carry_spans``) and past the placed regions whose spans meet it in a variant they both live in
    (16 B aligned). Returns ((name, offset, words), ...) and the first word past them all (0: none).
    """
    stages = program.stages
    placed, top, names = [], 0, set()
    for i, s in enumerate(stages):
        for c in s.carry:
            if c.name in names:
                raise PlanError(f"carried LDS {c.name} declared twice")
            names.add(c.name)
            spans = carry_spans(program, i, c)
            off = max(stages[j].lds_words for span in spans.values() for j in span)
            for _, pspans, poff, pwords in sorted(placed, key=lambda p: p[2]):
                meet = any(span & pspans[v] for v, span in spans.items() if v in pspans)
                if meet and off < poff + pwords and poff < off + c.words:
                    off = poff + pwords
            off = -(-off // 4) * 4
            placed.append((c.name, spans, off, c.words))
            top = max(top, off + c.words)
    return tuple((n, off, w) for n, _, off, w in placed), top


def _check_kept(program: Program) -> None:
    """``LdsKeep``: no stage between a keeping stage and its last reader, on a CTA the writer and a reader share,
    has an arena reaching the kept words (unless it is ``aware`` of them)."""
    grid = program.target.grid
    stages = program.stages
    index = {s.name: j for j, s in enumerate(stages)}

    def ctas(stage):
        return {(stage.base + t) % grid for t in range(stage.tasks)}

    for i, s in enumerate(stages):
        for keep in s.lds_keep:
            readers = [index[r] for r in keep.readers if r in index]
            if not readers:
                continue
            shared = ctas(s) & set().union(*(ctas(stages[j]) for j in readers))
            for j in range(i + 1, max(readers)):
                o = stages[j]
                if (
                    o.name in keep.aware
                    or o.lds_words <= keep.lo
                    or not (ctas(o) & shared)
                ):
                    continue
                raise PlanError(
                    f"stage {o.name} ({o.lds_words} LDS words) runs between {s.name} and its reader(s) "
                    f"{list(keep.readers)} on CTAs {sorted(ctas(o) & shared)[:8]}: it would overwrite LDS words "
                    f"{keep.lo} .. {keep.hi} ({keep.name}) the readers expect intact; place it elsewhere or after them"
                )


def _check_carried(program: Program) -> None:
    """Every CTA that runs a reader task of a carried region runs a task of its writer."""
    grid = program.target.grid

    def ctas(stage):
        return {(stage.base + t) % grid for t in range(stage.tasks)}

    by_name = {s.name: s for s in program.stages}
    for s in program.stages:
        for c in s.carry:
            mine = ctas(s)
            for r in c.readers:
                other = ctas(by_name[r]) - mine
                if other:
                    raise PlanError(
                        f"carried LDS {c.name}: reader {r} runs on CTAs {sorted(other)[:8]} where its writer "
                        f"{s.name} does not (carried LDS stays on its CTA)"
                    )
