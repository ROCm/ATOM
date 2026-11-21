# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether a compiled gfx12 kernel's line polls re-read memory.

A poll of epoch-tagged lines is written ``load; while stale: pause; load``.
Its loads are ordinary (non-volatile) buffer loads where a kernel issues a
round's loads together, and LLVM is free to hoist, merge or sink such a load
when nothing in the loop may write memory: the loop then spins on a value it
read once. Only the compiled code says whether it did, build by build, so
this module reads the code object's disassembly:

* the control-flow graph of the kernel (``s_branch`` / ``s_cbranch_*``
  targets from the instruction words) and its natural loops;
* a *poll loop*: the smallest loop around a spin pause (``atom.mono.polling``'s
  instruction, the one ``atom.mono.device.ops.spin_pause`` emits);
* per poll loop, walked from its pause back to its pause: the system-scope
  loads (the lines), the ``s_wait_loadcnt`` between and after them.

``hazards``: what makes a build unsafe to bind -- a poll loop that loads
nothing or one of whose loads some way round it skips, loads with no wait
behind them on some way back to the pause, system-scope loads outside the
poll loops without their copy inside (a load hoisted, merged or sunk),
a wait on anything but the loads between a load and its ``s_wait_loadcnt
0`` (a message's return, a store, a barrier: the stale polls of the sampling
kernel's first design waited for the realtime clock's message with a load
out), control flow this module does not follow. ``notes``: what is safe but not as built for (a round's loads waited
one by one in a build whose loads go out together).

Nothing here imports FlyDSL or torch: ``analyze`` takes disassembly text.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"
# a build's name says its polls' loads go out together
TOGETHER_NAME = re.compile(r"_pt(?=_|$)")
_COMMENT = re.compile(r"//\s*([0-9A-Fa-f]{6,16}):\s*((?:[0-9A-Fa-f]{8}\s*)+)")
_LABEL = re.compile(r"^([0-9A-Fa-f]{8,16})\s+<([^>]+)>:\s*$")
_INDIRECT = ("s_setpc", "s_swappc", "s_call", "s_rfe")


@dataclass(frozen=True)
class Insn:
    addr: int
    size: int
    text: str
    word: int

    @property
    def mnemonic(self) -> str:
        return self.text.split(None, 1)[0] if self.text else ""

    @property
    def branch(self) -> bool:
        return self.mnemonic == "s_branch" or self.mnemonic.startswith("s_cbranch_")

    @property
    def target(self) -> int:
        """A branch's target: the word's low 16 bits, signed, in words past
        the next instruction."""
        simm = self.word & 0xFFFF
        if simm >= 0x8000:
            simm -= 0x10000
        return self.addr + 4 + 4 * simm

    @property
    def sys_load(self) -> bool:
        return "SCOPE_SYS" in self.text and "_load_" in self.mnemonic

    @property
    def load(self) -> bool:
        m = self.mnemonic
        return m.startswith(("buffer_load_", "global_load_")) and "_lds_" not in m

    @property
    def load_wait(self) -> bool:
        return self.mnemonic == "s_wait_loadcnt"

    @property
    def waits_else(self) -> bool:
        """Something a wave waits on other than its loads: a message's
        return (the realtime clock), stores, LDS, a barrier, a sleep."""
        m = self.mnemonic
        return (
            (m.startswith("s_wait_") and m != "s_wait_loadcnt")
            or m.startswith(("s_sendmsg", "s_barrier"))
            or m == "s_sleep"
        )

    @property
    def message(self) -> bool:
        return self.mnemonic.startswith("s_sendmsg") or self.mnemonic == "s_wait_kmcnt"

    @property
    def operand(self) -> int | None:
        parts = self.text.split()
        try:
            return int(parts[1], 0) if len(parts) > 1 else None
        except ValueError:
            return None


@dataclass
class PollLoop:
    """One poll loop: ``loads`` system-scope loads on every way round from
    its pause (of ``loop_loads`` in it), ``waits_between`` load waits between
    the first and the last of them, ``drained`` an ``s_wait_loadcnt 0``
    behind every load on every way back to the pause, ``simple`` one way
    round (else ``lines`` are the loop's instructions in address order)."""

    header: int
    pause: int
    loads: int
    loop_loads: int
    waits_between: int
    drained: bool
    simple: bool
    lines: list[str] = field(default_factory=list)
    # what else the wave waits on between a load and its s_wait_loadcnt 0
    # (mnemonics), and whether the loop asks for a message (the realtime
    # clock) at all
    waits_else: tuple[str, ...] = ()
    messages: bool = False
    pause_text: str = ""


@dataclass
class PollIsa:
    kernel: str
    instructions: int
    sys_loads: int
    inside: int
    outside: int
    loops: list[PollLoop]
    hazards: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def together(self) -> bool:
        """Every poll's loads back to back, one drain a round."""
        return bool(self.loops) and all(
            lp.waits_between == 0 and lp.drained for lp in self.loops
        )

    def summary(self) -> dict:
        return {
            "kernel": self.kernel,
            "instructions": self.instructions,
            "sys_loads": self.sys_loads,
            "in_poll_loops": self.inside,
            "outside": self.outside,
            "poll_loops": [
                {
                    "at": hex(lp.pause),
                    "loads": lp.loads,
                    "waits_between": lp.waits_between,
                    "drained": lp.drained,
                    "waits_else": list(lp.waits_else),
                    "messages": lp.messages,
                    "pause": lp.pause_text,
                }
                for lp in self.loops
            ],
            "together": self.together,
            "hazards": list(self.hazards),
            "notes": list(self.notes),
        }


def spin_pause_asm() -> str:
    from atom.mono.polling import SPIN_PAUSE_ASM

    return SPIN_PAUSE_ASM


def disassemble(elf: bytes) -> str:
    """``llvm-objdump -d`` of a code object (raises where there is none)."""
    tool = OBJDUMP if shutil.which(OBJDUMP) else shutil.which("llvm-objdump")
    if tool is None:
        raise RuntimeError("no llvm-objdump to read the code object with")
    with tempfile.NamedTemporaryFile(suffix=".elf") as f:
        f.write(elf)
        f.flush()
        return subprocess.run(
            [tool, "-d", "--mcpu=gfx1250", f.name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout


def parse(text: str) -> dict[str, list[Insn]]:
    """Each function's instructions. Every instruction line carries the
    printer's ``// address: words`` comment; one without is an error (its
    address and size are what the control flow is read from)."""
    functions: dict[str, list[Insn]] = {}
    current = None
    for line in text.splitlines():
        label = _LABEL.match(line.strip())
        if label:
            current = functions.setdefault(label.group(2), [])
            continue
        if current is None or not line.strip() or not line[:1].isspace():
            continue
        m = _COMMENT.search(line)
        body = line.split("//")[0].strip()
        if m is None:
            if body:
                raise ValueError(f"no address on instruction line: {line.strip()!r}")
            continue
        # objdump may print the raw bytes ahead of the instruction
        body = re.sub(r"^[0-9a-f]+:\s+(?:[0-9a-f]{2}\s)+\s*", "", body)
        words = m.group(2).split()
        current.append(
            Insn(int(m.group(1), 16), 4 * len(words), body, int(words[0], 16))
        )
    return {name: insns for name, insns in functions.items() if insns}


def _is_pause(insn: Insn, pause: str) -> bool:
    want = pause.split()
    if insn.mnemonic != want[0]:
        return False
    return len(want) == 1 or insn.operand == int(want[1], 0)


def _blocks(insns: list[Insn]):
    """Basic blocks as (first, last) instruction indices, each block's
    successors, and whether the control flow is all direct branches."""
    index = {insn.addr: i for i, insn in enumerate(insns)}
    leaders = {0}
    direct = True
    for i, insn in enumerate(insns):
        if insn.mnemonic.startswith(_INDIRECT):
            direct = False
        if insn.branch:
            if insn.target in index:
                leaders.add(index[insn.target])
            else:
                direct = False
            if i + 1 < len(insns):
                leaders.add(i + 1)
        elif insn.mnemonic == "s_endpgm" and i + 1 < len(insns):
            leaders.add(i + 1)
    starts = sorted(leaders)
    spans = [
        (s, (starts[k + 1] if k + 1 < len(starts) else len(insns)) - 1)
        for k, s in enumerate(starts)
    ]
    block_of = {}
    for b, (first, last) in enumerate(spans):
        for i in range(first, last + 1):
            block_of[i] = b
    succ = []
    for first, last in spans:
        insn = insns[last]
        out = []
        if insn.branch and insn.target in index:
            out.append(block_of[index[insn.target]])
        fall = insn.mnemonic not in ("s_branch", "s_endpgm")
        if fall and last + 1 < len(insns):
            out.append(block_of[last + 1])
        succ.append(sorted(set(out)))
    return spans, succ, block_of, index, direct


def _dominators(succ):
    count = len(succ)
    pred = [[] for _ in range(count)]
    for b, outs in enumerate(succ):
        for s in outs:
            pred[s].append(b)
    reach, stack = {0}, [0]
    while stack:
        for s in succ[stack.pop()]:
            if s not in reach:
                reach.add(s)
                stack.append(s)
    everything = set(reach)
    dom = {b: set(everything) for b in reach}
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for b in sorted(reach):
            if b == 0:
                continue
            preds = [dom[p] for p in pred[b] if p in reach]
            new = set.intersection(*preds) | {b} if preds else {b}
            if new != dom[b]:
                dom[b], changed = new, True
    return dom, pred, reach


def _loops(succ):
    """Natural loops, one a header: header -> its blocks."""
    dom, pred, reach = _dominators(succ)
    loops: dict[int, set[int]] = {}
    for b in sorted(reach):
        for header in succ[b]:
            if header in dom[b]:
                # the header's own predecessors are not walked: the loop
                # ends at its header
                body, stack = {header, b}, ([] if b == header else [b])
                while stack:
                    for p in pred[stack.pop()]:
                        if p in reach and p not in body:
                            body.add(p)
                            stack.append(p)
                loops.setdefault(header, set()).update(body)
    return loops


def _walk(insns, spans, succ, block_of, index, loop, start):
    """A round of the loop as it runs, from ``start`` (a pause): its block
    to the end, then each block's one successor inside the loop not yet
    passed, until none is left (the head of the pause's own block last).
    None where two such successors are open: more than one way on."""
    home = block_of[start]
    seen = list(range(start, spans[home][1] + 1))
    passed, block = {home}, home
    for _ in range(len(spans) + 1):
        inside = [s for s in succ[block] if s in loop]
        fresh = [s for s in inside if s not in passed]
        if len(fresh) > 1:
            return None
        if not fresh:
            if home in inside:
                seen.extend(range(spans[home][0], start))
            return seen
        block = fresh[0]
        passed.add(block)
        seen.extend(range(spans[block][0], spans[block][1] + 1))
    return None


def _skippable(succ, loop, home, without):
    """Whether block ``home`` (the pause's) is reached again from itself
    inside ``loop`` with block ``without`` left out: a way round the loop
    that does not pass ``without``."""
    seen, stack = set(), [s for s in succ[home] if s in loop and s != without]
    while stack:
        block = stack.pop()
        if block == home:
            return True
        if block in seen:
            continue
        seen.add(block)
        stack.extend(s for s in succ[block] if s in loop and s != without)
    return False


def _to_drain(insns, spans, succ, block_of, loop, load, pause):
    """From ``load`` on, inside the loop, up to the first ``s_wait_loadcnt
    0`` on every way: (whether some way reaches the pause again without one,
    the instructions passed before it)."""
    seen, stack, undrained = set(), [load + 1], False
    while stack:
        i = stack.pop()
        while True:
            if i in seen:
                break
            if i == pause:
                undrained = True
                break
            insn = insns[i]
            if insn.load_wait and insn.operand == 0:
                break
            seen.add(i)
            block = block_of[i]
            if i == spans[block][1]:
                stack.extend(spans[s][0] for s in succ[block] if s in loop)
                break
            i += 1
    return undrained, seen


def analyze_function(name: str, insns: list[Insn], pause: str) -> PollIsa:
    spans, succ, block_of, index, direct = _blocks(insns)
    hazards, notes, found = [], [], []
    if not direct:
        hazards.append("control flow not followed (an indirect or outside branch)")
    loops = _loops(succ)
    in_poll_loop: set[int] = set()
    done = set()
    for p, insn in enumerate(insns):
        if not _is_pause(insn, pause):
            continue
        around = [(len(body), h) for h, body in loops.items() if block_of[p] in body]
        if not around:
            # a pause on a straight path polls nothing
            continue
        _, header = min(around)
        if header in done:
            continue
        done.add(header)
        loop = loops[header]
        members = [i for b in loop for i in range(spans[b][0], spans[b][1] + 1)]
        loop_loads = [i for i in members if insns[i].sys_load]
        any_load = any(insns[i].load for i in members)
        at = insn.addr
        if not any_load:
            hazards.append(
                f"the poll loop at {at:#x} loads nothing: it re-reads no line"
            )
            found.append(
                PollLoop(insns[spans[header][0]].addr, at, 0, 0, 0, False, True)
            )
            continue
        if not loop_loads:
            # a wait on words of another scope (a flag): not a line poll
            continue
        in_poll_loop.update(loop_loads)
        home = block_of[p]
        # a load off some way round the loop is not re-read every round
        skipped = [
            i
            for i in loop_loads
            if block_of[i] != home and _skippable(succ, loop, home, block_of[i])
        ]
        undrained, before_drain = [], set()
        for i in loop_loads:
            open_way, passed = _to_drain(insns, spans, succ, block_of, loop, i, p)
            before_drain |= passed
            if open_way:
                undrained.append(i)
        # with a load out, the wave must wait for nothing but its loads: a
        # wait for a message's return there (the realtime clock) is what the
        # stale polls of the sampling kernel's first design had
        waits_else = tuple(
            sorted({insns[i].mnemonic for i in before_drain if insns[i].waits_else})
        )
        messages = any(insns[i].message for i in members)
        way = _walk(insns, spans, succ, block_of, index, loop, p)
        if way is not None and not set(loop_loads) <= set(way):
            way = None
        simple = way is not None
        if way is None:
            # more than one way round: the loop's instructions in address
            # order (every load is still on every way, or it is a hazard)
            way = sorted(members)
        loads = [k for k, i in enumerate(way) if insns[i].sys_load]
        waits = [k for k, i in enumerate(way) if insns[i].load_wait]
        between = sum(1 for k in waits if loads[0] < k < loads[-1])
        found.append(
            PollLoop(
                insns[spans[header][0]].addr,
                at,
                len(loop_loads) - len(skipped),
                len(loop_loads),
                between,
                not undrained,
                simple,
                [f"{insns[i].addr:#07x} {insns[i].text}" for i in way],
                waits_else,
                messages,
                insn.text,
            )
        )
        if waits_else:
            hazards.append(
                f"the poll loop at {at:#x}: the wave waits on {', '.join(waits_else)}"
                " between a load and its s_wait_loadcnt 0"
            )
        if messages:
            notes.append(f"the poll loop at {at:#x} asks for a message every round")
        if skipped:
            hazards.append(
                f"the poll loop at {at:#x}: {len(skipped)} of its {len(loop_loads)}"
                " system-scope loads can be skipped on a way round it"
            )
        if undrained:
            hazards.append(
                f"the poll loop at {at:#x}: no s_wait_loadcnt 0 behind its loads"
            )
        if between:
            notes.append(
                f"the poll loop at {at:#x}: {between} load waits between its"
                f" {len(loads)} loads"
            )
    sys_loads = [i for i, insn in enumerate(insns) if insn.sys_load]
    inside = len(in_poll_loop)
    outside = len(sys_loads) - inside
    # a poll's first loads ahead of its loop are copies of the loop's (the
    # compiler may keep one copy of a load for both, entered past the
    # pause): more loads outside than inside is one without its copy inside
    if outside > inside:
        hazards.append(
            f"{outside} system-scope loads outside the poll loops, {inside} inside:"
            " a load was hoisted, merged or sunk"
        )
    return PollIsa(
        name, len(insns), len(sys_loads), inside, outside, found, tuple(hazards),
        tuple(notes),
    )  # fmt: skip


def analyze(text: str, pause: str | None = None) -> list[PollIsa]:
    """``analyze_function`` of each kernel in a disassembly."""
    pause = spin_pause_asm() if pause is None else pause
    return [analyze_function(n, insns, pause) for n, insns in parse(text).items()]


def together_hazards(text: str, pause: str | None = None) -> tuple[str, ...]:
    """Why a build whose polls issue their loads together must not be bound
    (empty: it re-reads every line every round)."""
    out = []
    reports = analyze(text, pause)
    if not reports:
        return ("no kernel in the disassembly",)
    for report in reports:
        out.extend(report.hazards)
        if not any(lp.loads for lp in report.loops):
            out.append("no poll loop re-reading system-scope lines")
    return tuple(out)


def code_object_hazards(name: str, elf: bytes) -> tuple[str, ...]:
    """``together_hazards`` of a code object whose kernel ``name`` says its
    polls go out together; none for any other kernel. A code object that
    cannot be read is a hazard: an unread poll must not pass for a safe one."""
    if not TOGETHER_NAME.search(name or ""):
        return ()
    try:
        found = together_hazards(disassemble(elf))
    except Exception as why:  # noqa: BLE001 -- unread is unsafe, whatever the cause
        return (f"poll loops unread ({type(why).__name__}: {why})",)
    return tuple(f"polls together: {h}" for h in found)


def launcher_hazards(launcher) -> tuple[str, ...]:
    """``code_object_hazards`` of each binary a compiled launcher holds."""
    from atom.mono.runtime.compile import _code_object, _symbol

    out = []
    for artifact in launcher._mem_cache.values():
        elf = _code_object(artifact._ir_text)
        out.extend(code_object_hazards(_symbol(elf), elf))
    return tuple(out)


# gfx12 CPol of a poll's load: SCOPE_SYS, with or without the volatile bit
_SCOPE_SYS, _VOLATILE = 3 << 3, 1 << 31


def _compiled_hazards(kernel) -> tuple[str, ...]:
    """``launcher_hazards`` of ``kernel``, compiled here, and one more: the
    system-scope loads inside its poll loops are half the loads its build
    traced (a poll is written ``load; while stale: pause; load``). The
    disassembly alone cannot tell a hoisted load from one the compiler kept
    a single copy of."""
    from aiter.ops.flydsl.kernels import buffer_ops as bo

    from atom.mono.runtime.compile import _code_object, compile_kernels

    traced = [0]
    plain = bo.buffer_load

    def counted(*args, **kwargs):
        modifier = kwargs.get("cache_modifier", 0)
        if isinstance(modifier, int) and modifier & ~_VOLATILE == _SCOPE_SYS:
            traced[0] += 1
        return plain(*args, **kwargs)

    bo.buffer_load = counted
    try:
        (launcher,) = compile_kernels([(kernel.launcher, kernel.abi)])
    finally:
        bo.buffer_load = plain
    found = list(launcher_hazards(launcher))
    try:
        inside = sum(
            report.inside
            for artifact in launcher._mem_cache.values()
            for report in analyze(disassemble(_code_object(artifact._ir_text)))
        )
    except Exception as why:  # noqa: BLE001 -- unread is unsafe, whatever the cause
        return (*found, f"poll loops unread ({type(why).__name__}: {why})")
    if 2 * inside != traced[0]:
        found.append(
            f"polls together: {inside} system-scope loads in the poll loops of"
            f" {traced[0]} traced (two a line)"
        )
    return tuple(found)


def bound_build(build, what: str, log=None, hazards_of=None):
    """``build(None)``, a kernel as its switch has it -- unless its polls
    issue their loads together (its name says so) and its compiled poll loops
    do not re-read every line every round (``launcher_hazards``): then
    ``build(False)``, the volatile loads, with a warning. A build that does
    not re-read is never handed on. Every rank compiles the same source with
    the same compiler, so the ranks of a collective launch choose alike."""
    kernel = build(None)
    if not TOGETHER_NAME.search(kernel.name):
        return kernel
    hazards = (hazards_of or _compiled_hazards)(kernel)
    if not hazards:
        return kernel
    if log is None:
        import logging

        log = logging.getLogger("atom")
    log.warning(
        "%s: %s keeps volatile polls (%s)", what, kernel.name, ", ".join(hazards)
    )
    kernel = build(False)
    if TOGETHER_NAME.search(kernel.name):
        raise RuntimeError(f"{what}: {kernel.name} still polls together")
    return kernel
