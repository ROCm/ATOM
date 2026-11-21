# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Which copy of a dual area (``atom.mono.plan.layout.dual_layout``) each XCD
reads, page by page: the word ``device.numa.local_copy`` takes.

MI355X homes each 8 KiB run of a 2 MiB page on one of its two I/O dies,
alternating (the home is address bit 13 XOR a bit of the page). A dual area
keeps copy 0 in its even runs and copy 1 in its odd ones, so each copy is
homed on one die; which one, and which XCDs sit on which die, are not
arithmetic: a ping-pong probe times them (``calibrate``; ``probe=`` another
round-trip timer):

- ``pp_runs``: one CTA on XCD ``xp`` and one on XCD ``xc`` trade 16-byte
  tagged puts ``ITERS`` times in each listed run; a round trip is fast when
  the run is homed on both CTAs' die.
- ``calibrate``: every XCD pair on one even and one odd run (a pair on one
  die shows the parity gap, its sign the run parity local to it) -> each
  XCD's copy; then every run of the page timed with one pair of each die,
  each run's measured home checked against the rule (agreement 1.0).
- ``scratch_word``: page 0 calibrated, every further page's parity checked
  with page 0's pairs. A failed check gives that page copy 0 for every XCD:
  still correct (every put writes both copies), only slower.
"""

import functools
import logging
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.mono.device.numa import xcc_id
from atom.mono.device.ops import CM_DEV, kernel_symbol, launcher, memrealtime, rsrc
from atom.mono.plan.build_key import key_tuple, source_digest
from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.layout import NUMA_PAGE, NUMA_RUN

logger = logging.getLogger("atom")

ITERS = 24  # round trips a run
SPIN_MAX = 1 << 18  # a poll gives up (no CTA of the asked XCD: the check fails)
GAP = 0.08  # a pair's parity gap (of its round trip) that counts as "on one die"
RETIME = 8  # repetitions of a run whose measured home disagrees with the rule
RETIME_ROUNDS = 2  # rounds of that before the run counts against the rule
ATTEMPTS = 3  # calibrations of page 0 before it settles on copy 0
THREADS = 64
_CURRENT_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class PingPongBuild:
    iters: int


@functools.cache
def _build(key: PingPongBuild):
    iters = key.iters
    keyed = key_tuple(key, source_digest("mono"))
    name = kernel_symbol("mono_numa_pp", it=iters)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def ping_pong(
        buf: Int64, runs: Int64, out: Int64, xp: Int32, xc: Int32, n: Int32,
        salt: Int32,
    ):  # fmt: skip
        """CTA (< 8, on XCD xp) puts a tag at word 0 of each run and waits for
        its echo at word 512, the one (8 .. 15, on XCD xc) echoes it: ``out``
        [j] the ticks of ``iters`` round trips in run ``runs`` [j] (words)."""
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        rb, rr, ro = rsrc(buf), rsrc(runs), rsrc(out)
        x = xcc_id()
        isp = (bid < 8) & (x == xp)
        isc = (bid >= 8) & (bid < 16) & (x == xc)
        if (isp | isc) & (tid == 0):
            me = isp.select(fx.Int32(0), fx.Int32(512))
            yo = isp.select(fx.Int32(512), fx.Int32(0))
            j = fx.Int32(0)
            while j < n:
                w = fx.Int32(bo.buffer_load(rr, j, vec_width=1, dtype=T.i32))
                t0 = memrealtime()
                i = fx.Int32(0)
                while i < iters:
                    tg = salt + j * 32 + i + 1
                    put = fx.Vector.from_elements([tg, tg, tg, tg], fx.Int32)
                    if isp:
                        bo.buffer_store(put, rb, w + me, cache_modifier=CM_DEV)
                    v = fx.Vector(
                        bo.buffer_load(
                            rb, w + yo, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
                        )
                    )
                    s = fx.Int32(0)
                    while ((v[1] != tg) | (v[3] != tg)) & (s < SPIN_MAX):
                        rocdl.s_sleep(1)
                        v = fx.Vector(
                            bo.buffer_load(
                                rb, w + yo, vec_width=4, dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )  # fmt: skip
                        s = s + 1
                    if isc:
                        bo.buffer_store(put, rb, w + me, cache_modifier=CM_DEV)
                    i = i + 1
                if isp:
                    bo.buffer_store(fx.Int32(memrealtime() - t0), ro, j)
                j = j + 1
        gpu.barrier()

    @launcher(name)
    def launch(
        buf: Int64, runs: Int64, out: Int64, xp: Int32, xc: Int32, n: Int32,
        salt: Int32, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        ping_pong(buf, runs, out, xp, xc, n, salt).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch


_SALT = [0]


def pp_runs(buf: torch.Tensor, runs: list[int], xp: int, xc: int) -> torch.Tensor:
    """µs a round trip of the XCD ``xp`` -> ``xc`` pair in each run (byte
    offsets into ``buf``, zeroed int32), each call's tags its own."""
    dev = buf.device
    rw = torch.tensor([r // 4 for r in runs], dtype=torch.int32, device=dev)
    out = torch.zeros(len(runs), dtype=torch.int32, device=dev)
    _SALT[0] = (_SALT[0] + 1) & 0x7FF
    _run_compiled(
        _build(PingPongBuild(ITERS)), buf.data_ptr(), rw.data_ptr(), out.data_ptr(),
        xp, xc, len(runs), _SALT[0] << 20, torch.cuda.current_stream(dev),
    )  # fmt: skip
    torch.cuda.synchronize(dev)
    return out.float().cpu() * 0.01 / ITERS


def _homes(page, nruns, pairs, want, reps=2, either=False, probe=pp_runs):
    """Each of the first ``nruns`` runs of ``page``: 1 when it is local to
    ``pairs[0]``'s die rather than ``pairs[1]``'s. A run disagreeing with
    ``want`` (``either``: or with ``1 - want`` when most runs do) is timed
    again (``RETIME`` repetitions, ``RETIME_ROUNDS`` rounds), its last
    measurement counting. ``probe``: the round trips' timer (``pp_runs``'s
    signature). -> (the homes, facts)."""

    def time(idx, n):
        runs = [i * NUMA_RUN for i in idx]
        t = [
            torch.stack([probe(page, runs, *ab) for _ in range(n)]).min(0).values
            for ab in pairs
        ]
        return (t[0] < t[1]).int(), (t[0] - t[1]).abs()

    near0, diff = time(range(nruns), reps)
    if either and float((near0 == want).float().mean()) < 0.5:
        want = 1 - want
    retimed = []
    for _ in range(RETIME_ROUNDS):
        bad = [i for i in range(nruns) if int(near0[i]) != int(want[i])]
        if not bad:
            break
        retimed += bad
        again, _ = time(bad, RETIME)
        near0[bad] = again
    return near0, {
        "margin": round(float(diff.median()), 3),
        "retimed": sorted(set(retimed)),
    }


def calibrate(
    page: torch.Tensor, nruns: int, reps: int = 3, probe=pp_runs
) -> tuple[int, float, dict]:
    """``page``: zeroed int32 at a page boundary, ``nruns`` runs at least. ->
    (word, agreement, facts): bit x of ``word`` the copy (run parity) XCD x
    reads; agreement the share of runs whose measured home is the rule's.
    ``probe``: the round trips' timer (``pp_runs``'s signature)."""
    # host tensors, beside the probe's timings whatever the default device
    gap = torch.zeros(8, 8, device="cpu")
    for xp in range(8):
        for xc in range(8):
            t = (
                torch.stack([probe(page, [0, NUMA_RUN], xp, xc) for _ in range(reps)])
                .min(0)
                .values
            )
            # > 0: the even run (copy 0) is local to the pair
            gap[xp, xc] = float((t[1] - t[0]) / t.min())
    strong = gap.abs() > GAP
    vote = (gap * strong).sum(1) + (gap * strong).sum(0)
    word = sum(int(vote[x] < 0) << x for x in range(8))
    facts = {"gap": [[round(float(v), 3) for v in r] for r in gap], "word": word}
    groups = [[x for x in range(8) if (word >> x) & 1 == c] for c in (0, 1)]
    if any(len(g) != 4 for g in groups):
        facts["error"] = f"XCD groups {groups} (expected 4 + 4)"
        return word, 0.0, facts
    # one pair of each die (its strongest), every run: the rule's home each
    pairs = [
        max(
            ((a, b) for a in g for b in g if a != b), key=lambda ab: abs(float(gap[ab]))
        )
        for g in groups
    ]
    # group 0 reads copy 0, the even runs
    want = 1 - torch.arange(nruns, device="cpu") % 2
    near0, f = _homes(page, nruns, pairs, want, probe=probe)
    agree = float((near0 == want).float().mean())
    facts.update(pairs=pairs, agree=agree, margin=f["margin"], retimed=f["retimed"])
    return word, agree, facts


def page_parity(
    page: torch.Tensor, nruns: int, pairs, probe=pp_runs
) -> tuple[int | None, dict]:
    """A further page, with page 0's pairs: 0 when its even runs are local to
    the XCDs page 0's are, 1 to the others, None when the rule does not hold."""
    even = 1 - torch.arange(nruns, device="cpu") % 2
    near0, facts = _homes(page, nruns, pairs, even, either=True, probe=probe)
    same = float((near0 == even).float().mean())
    facts["same"] = same
    return (0 if same == 1.0 else 1 if same == 0.0 else None), facts


def scratch_word(scratch: torch.Tensor, area: int, name: str = "") -> int:
    """The copy word of a dual area: ``scratch`` zeroed and page aligned, its
    first ``area`` bytes the area; bits 8 p .. 8 p + 7 page p's copy of each
    XCD. Left zeroed."""
    if scratch.data_ptr() % NUMA_PAGE:
        raise ValueError(f"dual area 0x{scratch.data_ptr():x} is not page aligned")
    first = min(max(area, 2 * NUMA_RUN), NUMA_PAGE)
    page = scratch[:first].view(torch.int32)
    for attempt in range(1, ATTEMPTS + 1):
        word, agree, facts = calibrate(page, first // NUMA_RUN, reps=3 * attempt)
        page.zero_()
        if agree == 1.0:
            break
    if agree < 1.0:
        logger.warning(
            "mono numa: %s: home rule agreement %.4f after %d calibrations (%s): "
            "every CTA reads copy 0", name, agree, attempt, facts.get("error", ""),
        )  # fmt: skip
        return 0
    out = word
    for p in range(1, -(-area // NUMA_PAGE)):
        n = max(min(area - p * NUMA_PAGE, NUMA_PAGE), 2 * NUMA_RUN)
        pg = scratch[p * NUMA_PAGE : p * NUMA_PAGE + n].view(torch.int32)
        flip, f = page_parity(pg, n // NUMA_RUN, facts["pairs"])
        pg.zero_()
        if flip is None:
            logger.warning(
                "mono numa: %s page %d: the home rule does not hold (%s): every "
                "CTA reads its copy 0", name, p, f,
            )  # fmt: skip
            continue
        out |= (word ^ (0xFF if flip else 0)) << (8 * p)
    logger.info("mono numa: %s: word 0x%x, %d calibration(s)", name, out, attempt)
    return out
