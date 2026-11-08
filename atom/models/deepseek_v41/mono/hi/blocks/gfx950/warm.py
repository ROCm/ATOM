# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cache warming: touch a later ``Linear``'s weight bytes early, so that
its own (nontemporal) weight loads hit L2 / the 256 MB Infinity Cache instead of HBM.

**Unused**: no served schedule sets an ``Idle`` hook (every placement tried lost). ``Linear`` calls
``idle_issue`` / ``idle_sink`` at its wait points, no-ops without a hook.

A touch is one dword load per ``line`` bytes. It reads nothing from the mailbox and writes nothing, so it
changes no result, only timing. The **``Idle`` hook** (``Stage.idle``): a host stage's task
issues the touches where its own loads are in flight and it is about to wait
(blocks call ``idle_issue`` there and ``idle_sink`` at the task's end; a block
without the calls ignores the hook). At most ``cap`` touches per thread (one VGPR
each, held until the sink).

Which bytes (``mode``):

- ``"own"``: those of the target task(s) this CTA will run (the target stage's
  placement is read from the program at emit time), K steps ``skip`` (default: past
  the target's register prefetch, which the task issues itself first) to the end of
  its K slice, K step by K step (the order the task streams them);
- ``"spread"``: fraction ``[lo, hi)`` of the target's whole weight, split evenly
  over the 256 CTAs.

Cost: loads complete in issue order per wave, so every wait of the same wave after
the touches waits for them too.
"""

from dataclasses import dataclass

import flydsl.expr as fx

from ...device.gfx950.ops import fresh, traced
from ...plan.program import Scenario
from ...target import MI355X

THREADS = MI355X.threads
GRID = MI355X.cus
OOB = 0x3FFFFFFF  # a word offset past every buffer: the load returns 0 without touching memory


@dataclass(frozen=True)
class _WPlan:
    name: str
    target: str  # the target Linear's stage name (and weight arg prefix)
    mode: str
    tasks_t: int  # target tasks
    blocks: int
    g: int
    ks: int
    steps: int
    n_rg: int
    skip: int  # K steps of a task's slice not touched (the prefetched ones)
    lo: float
    hi: float
    w_bytes: int
    line: int
    cm: int
    tasks: int  # GRID
    cap: int  # touches per thread (an Idle hook)

    @property
    def lpu(self) -> int:  # lines per 1 KiB unit
        return 1024 // self.line

    @property
    def own_lines(self) -> int:
        return self.g * (self.ks - self.skip) * self.lpu

    @property
    def spread_lines(self) -> int:  # lines per task
        lo, hi = (
            int(self.lo * self.w_bytes) // self.line,
            int(self.hi * self.w_bytes) // self.line,
        )
        return -(-(hi - lo) // self.tasks)


def _plan(name, target, sc, mode, tasks, skip, lo, hi, line, cm, cap) -> _WPlan:
    if mode not in ("own", "spread"):
        raise ValueError(f"warm mode {mode}")
    p = target.plan(sc)
    if skip is None:  # the K steps every wave's register prefetch covers
        skip = min(p.ks, (p.prefetch // p.gw) * p.wk)
    return _WPlan(
        name=name,
        target=p.name,
        mode=mode,
        tasks_t=p.tasks,
        blocks=p.blocks,
        g=p.g,
        ks=p.ks,
        steps=p.steps,
        n_rg=p.n_rg,
        skip=min(skip, p.ks),
        lo=lo,
        hi=hi,
        w_bytes=p.w_bytes,
        line=line,
        cm=cm,
        tasks=tasks,
        cap=cap,
    )


@dataclass(frozen=True)
class Idle:
    """A ``Stage.idle`` hook: touches of ``w``'s bytes (module docstring)."""

    w: _WPlan

    @classmethod
    def of(
        cls,
        target,
        sc: Scenario,
        mode="own",
        skip=None,
        lo=0.0,
        hi=1.0,
        line=128,
        cm=0,
        cap=4,
    ) -> "Idle":
        name = f"idle.{target.name}"
        return cls(_plan(name, target, sc, mode, GRID, skip, lo, hi, line, cm, cap))

    @property
    def args(self) -> tuple[str, ...]:
        return ()  # read through the kernel's own pointer loads (``k.kargs``), no stage copy

    @property
    def kargs(self) -> tuple[str, ...]:
        return (f"{self.w.target}.w",)


def idle_issue(k):
    """A block's wait point: issue the stage's idle touches (None if it has none)."""
    hook = getattr(k, "idle", None)
    if hook is None:
        return None
    w = hook.w
    if w.mode == "own":
        tt = (
            fresh(k.bid) - _target_base(k, w) % GRID + GRID
        ) % GRID  # VGPR math: SGPRs are scarce in hosts
        n = min(w.cap, -(-w.own_lines // THREADS))
        offs = [
            _own_off(k, w, tt, fx.Int32(j * THREADS) + k.tid, tt < w.tasks_t)
            for j in range(n)
        ]
    else:
        n = min(w.cap, -(-w.spread_lines // THREADS))
        offs = [_spread_off(w, k.bid, fx.Int32(j * THREADS) + k.tid) for j in range(n)]
    # global loads off the kernel-start pointer: no buffer descriptor, no stage-long SGPR copy
    ptr = k.kargs[f"{w.target}.w"]
    return [_touch_global(ptr, o) for o in offs]


# ------------------------------------------------------------------ device code


def idle_sink(k, h):
    """The task's end: keep the idle touches alive (their values feed a store that never runs)."""
    if h:
        _sink(k, k.idle.w, _fold(h))


def _target_base(k, w):
    for s in k.program.stages:
        if s.name == w.target:
            return s.base
    raise ValueError(f"warm {w.name}: no stage {w.target} in the program")


def _touch_global(ptr, word):
    """One dword at word ``word`` of ``ptr`` (out-of-range words: word 0, a harmless repeat)."""
    word = (word == fx.Int32(OOB)).select(fx.Int32(0), word)
    addr = ptr + (fx.Int64(word) << 2)
    return fx.Int32(
        fx.ptr_load(
            fx.inttoptr(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4), addr
            )
        )
    )


def _fold(vals):
    acc = vals[0]
    for v in vals[1:]:
        acc = acc ^ v
    return acc


@traced
def _sink(k, w, acc):
    if (acc == fx.Int32(0x5EED)) & (k.tag == 0):  # tags are >= 1
        ptr = getattr(k, "kargs", k.args)[f"{w.target}.w"]
        fx.generic_store(
            fx.inttoptr(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4), ptr
            ),
            acc,
        )


def _own_off(k, w, tt, li, valid):
    """Word offset of line ``li`` of target task ``tt``'s touched range (K step major:
    K step ``skip + li // (g * lpu)``, then row group, then line in the unit)."""
    s, b = tt // w.blocks, tt % w.blocks
    per_st = w.g * w.lpu
    st = w.skip + li // per_st
    rem = li % per_st
    rg = fx.min(b * w.g + rem // w.lpu, fx.Int32(w.n_rg - 1))
    byte = (rg * w.steps + s * w.ks + st) * 1024 + (rem % w.lpu) * w.line
    return (valid & (li < w.own_lines)).select(byte // 4, fx.Int32(OOB))


def _spread_off(w, t, li):
    lo = int(w.lo * w.w_bytes) // w.line
    hi = int(w.hi * w.w_bytes) // w.line
    line = fx.Int32(lo) + t * w.spread_lines + li
    ok = (li < w.spread_lines) & (line < hi)
    return ok.select(line * (w.line // 4), fx.Int32(OOB))
