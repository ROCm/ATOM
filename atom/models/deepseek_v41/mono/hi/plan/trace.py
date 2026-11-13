# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ATOM atom/mono/plan/trace.py (MIT, Copyright (C) 2026 Advanced
# Micro Devices, Inc.).
"""A kernel's mailbox accesses, recorded while FlyDSL traces it.

Tracing runs the kernel body's Python once, so every put / poll executes once as
Python: noting it costs the device nothing. ``enter_stage`` is called where the
program starts each stage, so the record is ``(stage, region, put | poll)`` in
program order. Plain functions on a module-level recorder: a traced body must
not call methods on local objects.

**Memory operations** (``Record.mem``, for the memory-model rules of ``plan.check``): every buffer / global
load and store the body traces (``buffer_ops``, ``ops.global_load`` / ``global_store``) and every ``s_waitcnt``,
in trace order, with what the address is derived from. The emitter registers the values addresses start from
(``root``): each stage's pointer-table entries (an *arg*: plain device memory), every region's ``Addr`` (scratch or
peer) and the launch tag. An access's target is the nearest root its address value reaches through the IR's
operands (its def chain, through ``scf`` region arguments and results); a store also records, for each 8 B half of
its data (or its one word), whether a word of it is derived from the tag. Nothing here emits IR: the kernels are
the same with or without a recorder.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs
    PEER = "peer"  # peer-symmetric memory: system-scope hand-off between GPUs


@dataclass(frozen=True)
class Access:
    stage: str
    region: str
    space: Space
    kind: str  # "put" | "poll"
    seq: int = (
        -1
    )  # position in the record's trace order (``Mem.seq``'s sequence); -1: not ordered (hand-made)


# root kinds: where an address comes from
ARG, REGION, PEER, INTERNAL, TAG = "arg", "region", "peer", "internal", "tag"


@dataclass(frozen=True)
class Mem:
    """One traced memory operation (or wait) of a stage, in trace order."""

    stage: str
    seq: int
    op: str  # "load" | "store" | "wait"
    kind: str = (
        ""  # the target's root kind: ARG / REGION / PEER / INTERNAL, "unknown" (no root reached), "" (a wait)
    )
    target: str = ""  # arg or region name ("" for a wait / unknown)
    cm: int = (
        0  # cache-policy bits (``ops.CM_*``); a wait: its vmcnt (-1: not a vmcnt wait)
    )
    nbytes: int = 0
    # a store: per 8 B half of the data (a 4 B store: its word), a word derived from the tag; None: words not seen
    tagged: tuple | None = None
    # the root kind of each name in ``target`` (in its order), for a target with several candidates (an address select
    # over buffers: ``kind`` "mixed" when they differ); () for one candidate (its kind is ``kind``)
    kinds: tuple = ()


@dataclass
class Record:
    stages: list[str] = field(default_factory=list)
    accesses: list[Access] = field(default_factory=list)
    mem: list[Mem] = field(default_factory=list)
    n: int = 0  # the next sequence number
    roots: dict = field(default_factory=dict)  # ir.Value -> (kind, name)
    memo: dict = field(default_factory=dict)  # ir.Value -> frozenset of roots reached
    quiet: int = 0  # > 0: memory operations not recorded (debug-build bookkeeping)
    waived: list = field(
        default_factory=list
    )  # the memory-model problems the program waived (``check_memory``)

    def next_seq(self) -> int:
        self.n += 1
        return self.n - 1


_record: Record | None = None


@contextmanager
def recording():
    """Record the accesses of kernels traced inside this block."""
    global _record
    outer, _record = _record, Record()
    from flydsl.expr import rocdl

    waitcnt = rocdl.s_waitcnt

    def s_waitcnt(*args, **kwargs):  # records the wait, then the real one
        wait(kwargs.get("vmcnt", -1) if not args else -1)
        return waitcnt(*args, **kwargs)

    rocdl.s_waitcnt = s_waitcnt
    try:
        yield _record
    finally:
        rocdl.s_waitcnt = waitcnt
        _record = outer


def _stage() -> str:
    return _record.stages[-1] if _record.stages else "<before any stage>"


def enter_stage(name: str) -> None:
    if _record is not None:
        _record.stages.append(name)


def note(region: str, space: Space, kind: str) -> None:
    if _record is not None:
        _record.accesses.append(
            Access(_stage(), region, space, kind, _record.next_seq())
        )


@contextmanager
def quiet():
    """Memory operations traced inside are not recorded (a debug build's own records)."""
    if _record is None:
        yield
        return
    _record.quiet += 1
    try:
        yield
    finally:
        _record.quiet -= 1


# ---- memory operations


def _ir(v):
    """The ``ir.Value`` behind a DSL value, or None. Only instance attributes are read: a DSL property may build IR
    (a lazily materialized value), and the record must not change the kernel."""
    from flydsl._mlir import ir

    for _ in range(10):
        if isinstance(v, ir.Value):
            # a plain Value: DSL subclasses overload == (it would build an arith.cmpi in a dict lookup)
            return v if type(v) is ir.Value else ir.Value(v)
        d = getattr(v, "__dict__", None) or {}
        nxt = d.get("_value", d.get("value"))
        if nxt is None or nxt is v:
            break
        v = nxt
    return v if isinstance(v, ir.Value) else None


def root(value, kind: str, name: str) -> None:
    """Accesses through addresses derived from ``value`` target (``kind``, ``name``)."""
    if _record is not None:
        v = _ir(value)
        if v is not None:
            _record.roots[v] = (kind, name)


def _op(owner):
    return getattr(owner, "operation", owner)


def _plain(v):
    from flydsl._mlir import ir

    return v if type(v) is ir.Value else ir.Value(v)


def _preds(v):
    """The values ``v`` is computed from: an op result's operands (for an op with regions also its terminators'
    operands at the result's position); a region argument's init operand (``scf.for`` / ``scf.while``).
    """
    from flydsl._mlir import ir

    owner = v.owner
    if isinstance(owner, ir.Block):  # a region argument
        parent = owner.owner
        if parent is None:
            return []
        op = _op(parent)
        i = ir.BlockArgument(v).arg_number
        if op.name == "scf.for":
            return (
                [_plain(op.operands[3 + i - 1])]
                if i >= 1 and 3 + i - 1 < len(op.operands)
                else []
            )
        if op.name == "scf.while":
            return [_plain(op.operands[i])] if i < len(op.operands) else []
        return []
    op = _op(owner)
    out = [_plain(x) for x in op.operands]
    if len(op.regions):
        i = ir.OpResult(v).result_number
        for region in op.regions:
            for block in region.blocks:
                ops = list(block.operations)
                if not ops:
                    continue
                term = _op(ops[-1])
                vals = list(term.operands)
                if term.name == "scf.condition":
                    vals = vals[1:]
                if i < len(vals):
                    out.append(_plain(vals[i]))
    return out


def roots_of(value) -> frozenset:
    """Every root ``value`` is derived from (memoized; stops at roots)."""
    v0 = _ir(value)
    if v0 is None or _record is None:
        return frozenset()
    memo, roots = _record.memo, _record.roots
    stack, busy = [(v0, False)], set()
    while stack:
        v, done = stack.pop()
        if v in memo:
            continue
        if v in roots:
            memo[v] = frozenset([roots[v]])
            continue
        preds = _preds(v)
        if not done:
            busy.add(v)
            stack.append((v, True))
            stack.extend((p, False) for p in preds if p not in memo and p not in busy)
            continue
        busy.discard(v)
        acc = set()
        for p in preds:
            acc |= memo.get(p, frozenset())
        memo[v] = frozenset(acc)
    return memo.get(v0, frozenset())


def _target(addr):
    """-> (kind, target, kinds): the root kind ("mixed" when the candidates differ), the names ``|``-joined, and
    each name's kind when there are several."""
    rs = [r for r in roots_of(addr) if r[0] != TAG]
    if not rs:
        return "unknown", "", ()
    if len(rs) > 1:  # e.g. a select between two buffers: name them all
        rs = sorted(rs, key=lambda r: (r[1], r[0]))
        kinds = {k for k, _ in rs}
        return (
            (kinds.pop() if len(kinds) == 1 else "mixed"),
            "|".join(n for _, n in rs),
            tuple(k for k, _ in rs),
        )
    return rs[0][0], rs[0][1], ()


def _bits(t):
    from flydsl._mlir import ir

    if isinstance(t, ir.IntegerType):
        return t.width
    if isinstance(t, ir.F32Type):
        return 32
    if isinstance(t, (ir.BF16Type, ir.F16Type)):
        return 16
    return None


def _words(data):
    """The 32-bit words of a store's data (its ``vector.from_elements`` operands, or the value itself), or None."""
    from flydsl._mlir import ir

    v = _ir(data)
    if v is None:
        return None
    if not isinstance(v.type, ir.VectorType):
        return [v] if _bits(v.type) == 32 else None
    owner = v.owner
    if isinstance(owner, ir.Block):
        return None
    op = _op(owner)
    if op.name == "vector.from_elements":
        el = [_plain(x) for x in op.operands]
        if el and all(_bits(x.type) == 32 for x in el):
            return el
    return None


def _tagged(data, nbytes):
    words = _words(data)
    if words is None:
        return None
    tagged = [any(r[0] == TAG for r in roots_of(w)) for w in words]
    if nbytes <= 4:
        return (tagged[0],) if tagged else None
    return tuple(any(tagged[h : h + 2]) for h in range(0, len(tagged), 2))


def mem(op: str, addr, cm: int = 0, nbytes: int = 0, data=None) -> None:
    """A load / store of ``nbytes`` through ``addr`` (a buffer resource or a global address), cache bits ``cm``."""
    if _record is None or _record.quiet:
        return
    kind, target, kinds = _target(addr)
    tagged = _tagged(data, nbytes) if op == "store" and kind in (REGION, PEER) else None
    _record.mem.append(
        Mem(
            _stage(),
            _record.next_seq(),
            op,
            kind,
            target,
            int(cm),
            nbytes,
            tagged,
            kinds,
        )
    )


def wait(vmcnt: int) -> None:
    """An ``s_waitcnt`` (``vmcnt`` -1: not a vector-memory wait)."""
    if _record is None or _record.quiet:
        return
    _record.mem.append(
        Mem(
            _stage(),
            _record.next_seq(),
            "wait",
            cm=vmcnt if isinstance(vmcnt, int) else -1,
        )
    )
