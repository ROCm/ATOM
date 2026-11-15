# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dual-copy scratch on MI355X's home rule (``Program.dual``).

The rule: within a 2 MiB page, memory alternates its home I/O die every 8 KiB (home = address bit 13 XOR a bit of
the page). A dual program lays both copies of its dual regions out arithmetically by it
(``program._dual_layout``: copy 0 in the area's even runs, copy 1 in the odd ones, the area at the start of a
2 MiB-aligned scratch), so copy 0 is homed on one IOD and copy 1 on the other. What
is not arithmetic is which: the page bit (the physical address) and which XCDs sit on which IOD. Both come from a
timing probe, which also checks the rule:

- ``pp_runs``: a ping-pong (16 B tagged puts, one CTA each way, ``ITERS`` round trips) in each listed run, between
  the CTA among 0-7 on XCD ``xp`` and the CTA among 8-15 on XCD ``xc`` (the hardware's XCC id: the dealing of CTAs to
  XCDs may move). A round trip is fast when the run is homed on both CTAs' IOD.
- ``calibrate``: every XCD pair on one even and one odd run: pairs on one IOD show the parity gap, and its sign is
  the run parity local to them -> each XCD's copy (``word``: bit x = the copy XCD x reads). Then a sweep of every run
  of the area with one pair of each IOD: every run must take the home the rule gives it (agreement 1.0).

``available()`` (once per process and device, on a fresh page of its own): the full-page check before building
dual programs; False (and a warning) -> single-copy builds. ``scratch_word`` (each dual scratch): the copy word of
that page, its runs checked; a failed check (also logged) gives word 0 (every CTA reads copy 0: correct, the stores
still go to both).
"""

from __future__ import annotations

import warnings
from functools import partial

import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device.ops import (
    CM_DEV,
    memrealtime,
    rsrc,
    spin_pause,
    traced,
)
from atom.models.qwen38.mono.kernels.program import (
    DUAL_PAGE,
    DUAL_RUN,
    Program,
    Scenario,
    Stage,
)

ITERS = 24  # round trips a run
SPIN_MAX = 1 << 18  # a poll gives up (a role no CTA took: the check then fails)
# a pair's parity gap (fraction of its round trip) that counts as "on one IOD"
GAP = 0.08


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
    return Program("dual_pp", Scenario(1, 1), [st])


_K: list = []


def pp_runs(buf: torch.Tensor, runs: list[int], xp: int, xc: int) -> torch.Tensor:
    """µs per round trip of the XCD ``xp`` -> ``xc`` pair in each run (byte offsets into ``buf``, zeroed int32)."""
    from atom.models.qwen38.mono.kernels.kernel import MonoKernel

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


def calibrate(page: torch.Tensor, nruns: int, reps: int = 3) -> tuple[int, float, dict]:
    """``page``: a zeroed int32 tensor at a 2 MiB boundary, at least ``nruns`` runs. -> (word, agreement, facts):
    bit x of ``word`` = the copy (run parity) local to XCD x; agreement = the share of the runs whose measured home
    is the rule's (1.0 when the rule holds)."""
    even, odd = 0, DUAL_RUN
    gap = torch.zeros(8, 8)
    for xp in range(8):
        for xc in range(8):
            t = (
                torch.stack([pp_runs(page, [even, odd], xp, xc) for _ in range(reps)])
                .min(0)
                .values
            )
            # > 0: the even run (copy 0) is local to the pair
            gap[xp, xc] = float((t[1] - t[0]) / t.min())
    strong = gap.abs() > GAP
    # each XCD's pairs on its own IOD
    vote = (gap * strong).sum(1) + (gap * strong).sum(0)
    # bit x: copy 1 (odd runs) is XCD x's
    word = sum(int(vote[x] < 0) << x for x in range(8))
    facts = {"gap": [[round(float(v), 3) for v in r] for r in gap], "word": word}
    groups = [[x for x in range(8) if (word >> x) & 1 == c] for c in (0, 1)]
    if any(len(g) != 4 for g in groups):
        facts["error"] = f"XCD groups {groups} (expected 4 + 4)"
        return word, 0.0, facts
    # one pair of each IOD (its strongest), every run of the area: the measured home must be the rule's
    pairs = [
        max(
            ((a, b) for a in g for b in g if a != b), key=lambda ab: abs(float(gap[ab]))
        )
        for g in groups
    ]
    runs = [i * DUAL_RUN for i in range(nruns)]
    t = [
        torch.stack([pp_runs(page, runs, *ab) for _ in range(2)]).min(0).values
        for ab in pairs
    ]
    near0 = (
        t[0] < t[1]
    ).int()  # the run is local to group 0 (copy 0's XCDs) rather than group 1
    want = 1 - torch.arange(nruns) % 2  # group 0 reads copy 0 = the even runs
    agree = float((near0 == want).float().mean())
    facts.update(
        pairs=pairs, agree=agree, margin=round(float((t[0] - t[1]).abs().median()), 3)
    )
    return word, agree, facts


_AVAILABLE: dict = {}


def available(device=None) -> bool:
    """The home rule holds on this device (a full-page check, once per process and device); else warn: the
    recipe builds single copies."""
    if not torch.cuda.is_available():
        return False
    dev = torch.device(
        "cuda", torch.cuda.current_device() if device is None else device
    )
    if dev not in _AVAILABLE:
        raw = torch.zeros(2 * DUAL_PAGE, dtype=torch.uint8, device=dev)
        page = raw[-raw.data_ptr() % DUAL_PAGE :][:DUAL_PAGE].view(torch.int32)
        try:
            _, agree, facts = calibrate(page, DUAL_PAGE // DUAL_RUN)
        except Exception as e:  # noqa: BLE001 - a probe that cannot run: no dual builds
            agree, facts = 0.0, {"error": f"{type(e).__name__}: {e}"}
        ok = agree == 1.0
        if not ok:
            why = f"agreement {agree:.4f} {facts.get('error', '')}".strip()
            warnings.warn(f"atom.models.qwen38.mono.kernels.dual: the 8 KiB home rule does not hold on {dev} ({why}): "
                          "dual-copy programs are built with one copy", stacklevel=2)  # fmt: skip
        _AVAILABLE[dev] = ok
    return _AVAILABLE[dev]


def scratch_word(scratch: torch.Tensor, area: int) -> int:
    """The copy word of a dual scratch (zeroed, 2 MiB aligned; its first ``area`` bytes the run area, one page): bit
    x the copy XCD ``x`` polls (``calibrate``). A failed calibration warns and gives 0 (every CTA polls copy 0).
    """
    if scratch.data_ptr() % DUAL_PAGE:
        raise ValueError(
            f"dual scratch 0x{scratch.data_ptr():x} is not {DUAL_PAGE} B aligned"
        )
    first = min(max(area, 2 * DUAL_RUN), DUAL_PAGE)
    page = scratch[:first].view(torch.int32)
    word, agree, facts = calibrate(page, first // DUAL_RUN)
    page.zero_()
    if agree < 1.0:
        warnings.warn(f"atom.models.qwen38.mono.kernels.dual: scratch 0x{scratch.data_ptr():x}: home rule agreement "
                      f"{agree:.4f} ({facts.get('error', '')}): every CTA polls copy 0", stacklevel=2)  # fmt: skip
        return 0
    return word
