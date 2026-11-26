# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dual-copy scratch on MI355X's home rule (``Program.dual``).

The tagged-pair calibration retries use the same mailbox backoff as execution.
SPIN_MAX bounds retry count, not elapsed time; changing backoff changes this
calibration's timing samples as well as production polling.

The rule (on two MI355X boxes, agreement 1.0 on every page): within a 2 MiB page, memory alternates its
home I/O die every 8 KiB (home = address bit 13 XOR a bit of the page). A dual program lays both copies of its dual
regions out arithmetically by it (``plan.program._dual_layout``: copy 0 in the area's even runs, copy 1 in the odd
ones, the area at the start of a 2 MiB-aligned scratch), so copy 0 is homed on one IOD and copy 1 on the other. What
is not arithmetic is which: the page bit (the physical address) and which XCDs sit on which IOD. Both come from a
timing probe, which also checks the rule:

- ``pp_runs``: a ping-pong (16 B tagged puts, one CTA each way, ``ITERS`` round trips) in each listed run, between
  the CTA among 0-7 on XCD ``xp`` and the CTA among 8-15 on XCD ``xc`` (the hardware's XCC id: the dealing of CTAs to
  XCDs may move). A round trip is fast when the run is homed on both CTAs' IOD.
- ``calibrate`` / ``page_parity`` (``atom.mono.runtime.numa``'s, the shared mono path's, on this probe): every
  XCD pair on one even and one odd run: pairs on one IOD show the parity gap, and its sign is
  the run parity local to them -> each XCD's copy (``word``: bit x = the copy XCD x reads). Then a sweep of every run
  of the area with one pair of each IOD: every run must take the home the rule gives it (agreement 1.0). A run whose
  home disagrees is timed again (``RETIME`` repetitions, up to ``RETIME_ROUNDS`` rounds) before it counts: one noisy
  round trip in a sweep of hundreds of runs gave agreement 0.9961 and word 0 on a scratch whose rule held.

``scratch_word`` (each dual scratch): the copy word of that page, its runs checked, up to ``ATTEMPTS`` calibrations; a
failed check (also logged) gives word 0 (every CTA reads copy 0: correct, the stores still go to both, but up to a few
% slower a layer). Every dual scratch logs its word (ATOM's log, ``dsv41hi.runtime.dual: <program>: word ...``): a served
log shows each program's.

The same scratch serves ``Region.home`` "numa2" regions (V4.1's ``numa_stages``): one mechanism, two selections.
"""

from __future__ import annotations

import logging
from functools import partial

import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu
from flydsl.expr.typing import T

from atom.mono.runtime.numa import calibrate, page_parity

from ..device.gfx950 import buffer_ops as bo
from ..device.gfx950.ops import CM_DEV, memrealtime, rsrc, spin_pause, traced
from ..plan.program import DUAL_PAGE, DUAL_RUN, Program, Scenario, Stage

logger = logging.getLogger("atom")

ITERS = 24  # round trips a run
SPIN_MAX = 1 << 18  # a poll gives up (a role no CTA took: the check then fails)
ATTEMPTS = 3  # calibrations of a scratch's page 0 before it settles on copy 0 (``scratch_word``)
REPORT: dict = (
    {}
)  # (address, runs) -> the last calibration's facts (word, gaps, agreement)


def _xcc():
    v = llvm.InlineAsmOp(
        T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s", has_side_effects=True
    )
    return fx.Int32(v.result) & 7


@traced
def _pp(k, t):
    """cfg: [xp, xc, n]; ``runs[j]``: a run's word offset in ``buf``; ``out[j]``: ticks of ITERS round trips there
    (a 16 B tagged put at the run's word 0, the answer at word 512)."""
    cr = rsrc(k.args["cfg"])
    xp = fx.Int32(bo.buffer_load(cr, 0, vec_width=1, dtype=T.i32))
    xc = fx.Int32(bo.buffer_load(cr, 1, vec_width=1, dtype=T.i32))
    n = fx.Int32(bo.buffer_load(cr, 2, vec_width=1, dtype=T.i32))
    buf, out, runs = rsrc(k.args["buf"]), rsrc(k.args["out"]), rsrc(k.args["runs"])
    x = _xcc()
    isp = (k.bid < 8) & (x == xp)
    isc = (k.bid >= 8) & (k.bid < 16) & (x == xc)
    salt = (k.tag & 0x7FF) << 20
    if (isp | isc) & (k.tid == 0):
        me = isp.select(fx.Int32(0), fx.Int32(512))
        yo = isp.select(fx.Int32(512), fx.Int32(0))
        j = fx.Int32(0)
        while j < n:
            w = fx.Int32(bo.buffer_load(runs, j, vec_width=1, dtype=T.i32))
            t0 = memrealtime()
            i = fx.Int32(0)
            while i < ITERS:
                tg = salt + j * 32 + i + 1
                if isp:
                    bo.buffer_store(fx.Vector.from_elements([tg, tg, tg, tg], fx.Int32), buf, w + me,
                                    cache_modifier=CM_DEV)  # fmt: skip
                v = fx.Vector(
                    bo.buffer_load(
                        buf, w + yo, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
                    )
                )
                s = fx.Int32(0)
                while ((v[1] != tg) | (v[3] != tg)) & (s < SPIN_MAX):
                    spin_pause()
                    v = fx.Vector(
                        bo.buffer_load(
                            buf, w + yo, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
                        )
                    )
                    s = s + 1
                if isc:
                    bo.buffer_store(fx.Vector.from_elements([tg, tg, tg, tg], fx.Int32), buf, w + me,
                                    cache_modifier=CM_DEV)  # fmt: skip
                i = i + 1
            if isp:
                bo.buffer_store(fx.Int32(memrealtime() - t0), out, j)
            j = j + 1
    gpu.barrier()


def _program() -> Program:
    st = Stage(
        "pp",
        256,
        partial(_pp),
        args=("cfg", "buf", "out", "runs"),
        base=0,
        key=("dual_pp", ITERS),
    )
    return Program("dual_pp", Scenario(1, 1), [st], plain_waive=("buf", "out", "runs"))


_K: list = []


def pp_runs(buf: torch.Tensor, runs: list[int], xp: int, xc: int) -> torch.Tensor:
    """µs per round trip of the XCD ``xp`` -> ``xc`` pair in each run (byte offsets into ``buf``, zeroed int32)."""
    from .kernel import MonoKernel

    if not _K:
        k = MonoKernel(_program())
        _K.append((k, k.buffers()))
    kern, bufs = _K[0]
    dev = buf.device
    cfg = torch.tensor([xp, xc, len(runs), 0], dtype=torch.int32, device=dev)
    rw = torch.tensor([r // 4 for r in runs], dtype=torch.int32, device=dev)
    out = torch.zeros(len(runs), dtype=torch.int32, device=dev)
    bufs.bump()
    kern.launch(
        kern.table({"cfg": cfg, "buf": buf, "out": out, "runs": rw}, device=dev),
        bufs,
        layer=0,
    )
    torch.cuda.synchronize(dev)
    return out.float().cpu() * 0.01 / ITERS


def _calibrated(page: torch.Tensor, nruns: int) -> tuple[int, float, dict]:
    """``calibrate`` (``atom.mono.runtime.numa``'s, on this module's probe) up to ``ATTEMPTS`` times (the page zeroed
    after each, more repetitions each time) until the rule holds; the last attempt's result otherwise
    (facts["attempts"])."""
    for attempt in range(1, ATTEMPTS + 1):
        word, agree, facts = calibrate(page, nruns, reps=3 * attempt, probe=pp_runs)
        page.zero_()
        if agree == 1.0:
            break
    facts["attempts"] = attempt
    return word, agree, facts


def scratch_word(scratch: torch.Tensor, area: int, name: str = "") -> int:
    """The copy word of a dual scratch (zeroed, 2 MiB aligned; its first ``area`` bytes the run area, one or more
    pages): bits 8 p .. 8 p + 7 the copy of each XCD in page p. Page 0 is calibrated (``_calibrated``: up to
    ``ATTEMPTS``), every further page's runs checked with page 0's pairs (``page_parity``). A failed check warns and
    gives that page 0 (every CTA reads its copy 0); a failed page 0 gives 0. The word is logged with
    ``name`` (the program's)."""
    if scratch.data_ptr() % DUAL_PAGE:
        raise ValueError(
            f"dual scratch 0x{scratch.data_ptr():x} is not {DUAL_PAGE} B aligned"
        )
    pages = -(-area // DUAL_PAGE)
    first = min(max(area, 2 * DUAL_RUN), DUAL_PAGE)
    page = scratch[:first].view(torch.int32)
    word, agree, facts = _calibrated(page, first // DUAL_RUN)
    REPORT[(scratch.data_ptr(), area)] = facts
    if agree < 1.0:
        logger.warning("dsv41hi.runtime.dual: scratch 0x%x (%s): home rule agreement %.4f after %d calibrations (%s): "
                       "every CTA polls copy 0", scratch.data_ptr(), name, agree, facts["attempts"],
                       facts.get("error", ""))  # fmt: skip
        _log(name, 0, facts)
        return 0
    out = word
    for p in range(1, pages):
        n = max(min(area - p * DUAL_PAGE, DUAL_PAGE), 2 * DUAL_RUN)
        pg = scratch[p * DUAL_PAGE : p * DUAL_PAGE + n].view(torch.int32)
        flip, f = page_parity(pg, n // DUAL_RUN, facts["pairs"], probe=pp_runs)
        pg.zero_()
        facts[f"page{p}"] = f
        if flip is None:
            logger.warning("dsv41hi.runtime.dual: scratch 0x%x (%s) page %d: the home rule does not hold (%s): every "
                           "CTA polls that page's copy 0", scratch.data_ptr(), name, p, f)  # fmt: skip
            continue
        out |= (word ^ (0xFF if flip else 0)) << (8 * p)
    _log(name, out, facts)
    return out


def _log(name: str, word: int, facts: dict) -> None:
    """One log line a dual scratch: its program, word, agreement, calibrations and the runs timed again."""
    retimed = sum(
        len(f.get("retimed", ()))
        for f in [facts, *(v for k, v in facts.items() if k.startswith("page"))]
    )
    logger.info("dsv41hi.runtime.dual: %s: word 0x%x, agreement %.4f, %d calibration(s), %d run(s) timed again",
                name or "scratch", word, facts.get("agree", 0.0), facts.get("attempts", 1), retimed)  # fmt: skip
