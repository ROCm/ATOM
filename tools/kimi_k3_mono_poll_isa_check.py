# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every serving width's poll-together builds as compiled: do they re-read?

Compiles (no launch) each kernel whose polls can issue their loads together
-- the fused K2 tail, the o_proj all-reduce (one-shot and two-hop), the
gathering all-reduce (seven- and six-element lines), the draft projection
all-reduce and the MLA back half -- built for TP4 and TP8 at every serving
width with ``poll_together=True``, and reads its code object
(``atom.utils.poll_isa``):

* every poll loop (the smallest loop round a spin pause) holds its
  system-scope loads on every way round from the pause, back to back, with
  an ``s_wait_loadcnt 0`` behind them and none between them;
* between a load and that wait the wave waits on nothing else -- no message
  (the realtime clock), store, LDS or barrier wait, no sleep: the stale polls
  of the sampling kernel's first design waited for the clock's message with
  a load out;
* the loads inside the poll loops are half the loads the build traced (a
  poll is written ``load; while stale: pause; load``): none hoisted, merged
  or sunk -- counted at trace time, apart from the disassembly;
* as many again sit ahead of the loops (the first loads), or none;
* the builder's own refusal (``poll_isa.launcher_hazards``, what
  ``poll_isa.bound_build`` binds by) finds nothing;
* no scratch, no VGPR or SGPR spill.

A build failing any of these is printed with its poll loops' disassembly
and the run exits 1. REF=1 also compiles the default build of each (its
loads VOLATILE): the same number of loads in its poll loops, each waited
out before the next.

    python tools/kimi_k3_mono_poll_isa_check.py
    env: KINDS (tail,tail_gr,tail_rs,oproj,oproj2,gather,draft_o,draft_down,
    mla), TPS (4,8), ROWS (8..64 step 8, 5..60 step 5, 42, 43), SHARD (i/n:
    every n-th build from the i-th), REF (0), SHOW (0: every build's loops),
    TOGETHER (1; 0 reads the default builds alone), TOOL_BOUND (unset; N: the
    bounded tool builds, whose polls give up past N)
"""

import json
import os
import sys
import time

from aiter.ops.flydsl.kernels import buffer_ops as bo

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as dlay
from atom.models.kimi_k3.mono.kernels.stages import draft_proj_ar_gfx1250 as dp
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tk
from atom.models.kimi_k3.mono.kernels.stages import mla_back_gfx1250 as mb
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.compile import _code_object, compile_kernels, kernel_resources
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import poll_isa

WIDTHS = sorted({*range(8, 65, 8), *range(5, 61, 5), 42, 43})
ROWS = (
    [int(r) for r in os.environ["ROWS"].split(",")]
    if os.environ.get("ROWS")
    else WIDTHS
)
TPS = [int(t) for t in os.environ.get("TPS", "4,8").split(",")]
KINDS = os.environ.get(
    "KINDS", "tail,tail_gr,tail_rs,oproj,oproj2,gather,draft_o,draft_down,mla"
).split(",")
REF = os.environ.get("REF", "0") == "1"
SHOW = os.environ.get("SHOW", "0") == "1"
TOGETHER = os.environ.get("TOGETHER", "1") == "1"
# the bounded tool builds in place of the serving ones (polls give up past it)
TOOL_BOUND = int(os.environ["TOOL_BOUND"]) if os.environ.get("TOOL_BOUND") else None
BOUNDED = {} if TOOL_BOUND is None else {"poll_bound": TOOL_BOUND}
SHARD = [int(v) for v in os.environ.get("SHARD", "0/1").split("/")]
EPS = 1e-5
OPROJ_PROJ = 12288
DRAFT_TOTAL = {"draft_o": 8192, "draft_down": 14336}
# the MLA back half's groups: (TP, DCP, heads a rank)
MLA_SHAPES = {8: (8, 8, 12), 4: (4, 4, 24)}

# the system-scope loads a build traces: every poll emits its loads twice
_buffer_load = bo.buffer_load
TRACED = {"loads": 0}


def _counted_load(*args, **kwargs):
    modifier = kwargs.get("cache_modifier", 0)
    if isinstance(modifier, int) and modifier & ~ar.VOLATILE == ar.SCOPE_SYS:
        TRACED["loads"] += 1
    return _buffer_load(*args, **kwargs)


bo.buffer_load = _counted_load


def plan(rows, nblocks):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=GFX1250.max_workgroup_lds_bytes,
        tokens=rows,
        nblocks=nblocks,
    )


def builder(kind, tp, rows):
    """``build(together)`` of a kind at a width, or why there is none."""
    if kind in ("tail", "tail_gr", "tail_rs"):
        if rows > tk.tail_line_rows():
            return None, "past the tail's widths"
        if kind == "tail_rs" and rows > tk.tail_resident_rows():
            return None, "past the resident tail's widths"

        def build(together):
            os.environ["ATOM_KIMI_MONO_TAIL_RESIDENT"] = (
                "1" if kind == "tail_rs" else "0"
            )
            try:
                return tk.build_k2_tail(
                    tp, rows, EPS, identity_routes=kind != "tail_gr",
                    poll_together=together, **BOUNDED,
                )  # fmt: skip
            finally:
                del os.environ["ATOM_KIMI_MONO_TAIL_RESIDENT"]

        return build, None
    if kind in ("oproj", "oproj2"):
        return (
            lambda together: oar.build_oproj_allreduce(
                plan(rows, 4),
                tp,
                OPROJ_PROJ // tp,
                oneshot=kind == "oproj",
                poll_together=together,
                **BOUNDED,
            ),
            None,
        )
    if kind == "gather":
        return (
            lambda together: ar.build_gather_allreduce(
                tp, rows, poll_together=together, **BOUNDED
            ),
            None,
        )
    if kind in DRAFT_TOTAL:
        k = DRAFT_TOTAL[kind] // tp
        why = dlay.refusal(rows, k, tp)
        if why is not None:
            return None, why
        return (
            lambda together: dp.build_draft_proj_ar(
                rows, tp, k, poll_together=together, **BOUNDED
            ),
            None,
        )
    if kind == "mla":
        group, dcp, heads = MLA_SHAPES[tp]
        return (
            lambda together: mb.build_mla_back(
                plan(rows, 1),
                group,
                dcp,
                heads,
                poll_together=together,
                bound=TOOL_BOUND,
            ),
            None,
        )
    raise ValueError(kind)


def read(build, together):
    """A build compiled and read: (entry, reports)."""
    started = time.time()
    kernel = build(together)
    TRACED["loads"] = 0
    (launcher,) = compile_kernels([(kernel.launcher, kernel.abi)])
    traced = TRACED["loads"]
    entries = []
    for artifact in launcher._mem_cache.values():
        elf = _code_object(artifact._ir_text)
        resources = kernel_resources(elf)
        (report,) = poll_isa.analyze(poll_isa.disassemble(elf))
        entries.append((resources, report))
    ((resources, report),) = entries
    spills = {
        key: resources[key]
        for key in ("scratch", "vgpr_spill", "sgpr_spill")
        if resources[key]
    }
    entry = {
        "kernel": resources["name"],
        "instructions": report.instructions,
        "traced_loads": traced,
        "sys_loads": report.sys_loads,
        "inside": report.inside,
        "outside": report.outside,
        "loops": [[lp.loads, lp.waits_between, lp.drained] for lp in report.loops],
        # per poll loop: its pause, what else the wave waits on between a
        # load and its s_wait_loadcnt 0, whether it asks for a message
        "pauses": sorted({lp.pause_text for lp in report.loops}),
        "waits_else": sorted({m for lp in report.loops for m in lp.waits_else}),
        "messages": any(lp.messages for lp in report.loops),
        "together": report.together,
        "hazards": list(report.hazards),
        "notes": list(report.notes),
        "spills": spills,
        "vgpr": resources["vgpr"],
        "sgpr": resources["sgpr"],
        "s": round(time.time() - started, 1),
    }
    return entry, report, launcher


def show(report):
    for lp in report.loops:
        print(f"--- {report.kernel}: poll loop at {lp.pause:#x}, {lp.loads} loads")
        for line in lp.lines:
            print("    " + line)
    print(flush=True)


def main():
    builds = [(kind, tp, rows) for kind in KINDS for tp in TPS for rows in ROWS]
    builds = builds[SHARD[0] :: SHARD[1]]
    print(
        json.dumps(
            {
                "run": {
                    "builds": len(builds),
                    "kinds": KINDS,
                    "tps": TPS,
                    "rows": ROWS,
                    "pause": poll_isa.spin_pause_asm(),
                    "together": TOGETHER,
                }
            }
        ),
        flush=True,
    )
    failed, refused, passed = [], 0, 0
    for kind, tp, rows in builds:
        head = {"kind": kind, "tp": tp, "rows": rows}
        build, why = builder(kind, tp, rows)
        if build is None:
            refused += 1
            print(json.dumps({**head, "no_build": why}), flush=True)
            continue
        try:
            entry, report, launcher = read(build, TOGETHER)
        except (ValueError, MonoUnsupported) as refusal:
            refused += 1
            print(json.dumps({**head, "no_build": str(refusal)}), flush=True)
            continue
        problems = list(entry["hazards"])
        named = bool(poll_isa.TOGETHER_NAME.search(entry["kernel"]))
        if named != TOGETHER:
            problems.append(f"named together: {named}")
        if not report.inside:
            problems.append("no poll loop with system-scope loads")
        if 2 * report.inside != entry["traced_loads"]:
            problems.append(
                f"{report.inside} loads in the poll loops of {entry['traced_loads']}"
                " traced (two a line)"
            )
        if TOGETHER:
            if not report.together:
                problems.append("a round's loads are not back to back")
            entry["builder_refuses"] = list(poll_isa.launcher_hazards(launcher))
            if entry["builder_refuses"]:
                problems.append("the builder refuses it")
        if entry["spills"]:
            problems.append(f"resources {entry['spills']}")
        if REF and TOGETHER:
            ref, ref_report, _ = read(build, False)
            entry["default"] = {
                "kernel": ref["kernel"],
                "inside": ref["inside"],
                "loops": ref["loops"],
                "hazards": ref["hazards"],
                "pauses": ref["pauses"],
                "waits_else": ref["waits_else"],
                "messages": ref["messages"],
            }
            if ref["inside"] != entry["inside"] or ref["hazards"]:
                problems.append("the default build's poll loops differ")
            if any(
                lp.waits_between != lp.loads - 1 or not lp.drained
                for lp in ref_report.loops
            ):
                problems.append("a default build's load is not waited out by itself")
        entry["ok"] = not problems
        entry["problems"] = problems
        print(json.dumps({**head, **entry}), flush=True)
        if problems or SHOW:
            show(report)
        if problems:
            failed.append(entry["kernel"])
        else:
            passed += 1
    print(
        json.dumps(
            {
                "summary": {
                    "builds": len(builds),
                    "ok": passed,
                    "no_build": refused,
                    "failed": failed,
                }
            }
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
