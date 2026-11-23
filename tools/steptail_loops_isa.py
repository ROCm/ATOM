# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every bounded wait of the two served step-tail kernels as compiled: each
loop that pauses (``s_sleep``), from its back-branch's target through the
back-branch. Checked a loop: it reads memory inside, every read is waited for
(``s_wait_loadcnt``) before the clock's message, its return or any other wait,
no read is out at the back-branch, and the innermost loop reads no clock.

    python tools/steptail_loops_isa.py
    env: HEADS (64x8,8x8: rows x qlen), BLOCKS (9x4,16x4,8x7: requests x
    drafts), SHOW (1: print each loop's reads, waits and clock reads)
"""

import os
import shutil
import subprocess

from atom.models.kimi_k3.step_tail import head_dispatch
from atom.models.kimi_k3.step_tail import kernels as K
from atom.models.kimi_k3.step_tail import markov_kernels as MK
from atom.models.kimi_k3.step_tail.sources import MARKOV_SOURCES, SOURCES
from atom.mono.runtime.compile import _code_object, compile_only


def pairs(name, default):
    return [
        tuple(int(v) for v in s.split("x"))
        for s in os.environ.get(name, default).split(",")
        if s
    ]


HEADS = pairs("HEADS", "64x8,8x8")
BLOCKS = pairs("BLOCKS", "9x4,16x4,8x7")
SHOW = os.environ.get("SHOW", "1") == "1"
KINDS = (
    (("buffer_load", "global_load"), "read"),
    (("s_wait_loadcnt",), "wait-read"),
    (("s_sendmsg",), "clock"),
    (("s_wait_kmcnt",), "wait-clock"),
    (("s_sleep",), "pause"),
    (("s_wait_", "s_barrier"), "wait-other"),
)


def code(launch):
    """The build's instructions: (address or None, text)."""
    with compile_only():
        launch(0, 0, 0, 0)
    objdump = shutil.which("llvm-objdump") or "/opt/rocm/llvm/bin/llvm-objdump"
    out = []
    for artifact in launch._mem_cache.values():
        path = f"/tmp/steptail_loops_isa_{os.getpid()}.o"
        with open(path, "wb") as f:
            f.write(_code_object(artifact._ir_text))
        text = subprocess.run(
            [objdump, "-d", "--no-show-raw-insn", path],
            capture_output=True,
            text=True,
            check=False,
        )
        os.remove(path)
        for ln in text.stdout.splitlines():
            if not ln[:1].isspace() or not ln.strip():
                continue
            body, _, comment = ln.partition("//")
            try:
                address = int(comment.strip().split(":")[0].split(" ")[0], 16)
            except ValueError:
                address = None
            out.append((address, body.strip()))
    return out


def targets(lines):
    """A branch's destination instruction (its operand: words from the next
    instruction), by instruction index."""
    at = {a: i for i, (a, _) in enumerate(lines) if a is not None}
    out = {}
    for i, (address, text) in enumerate(lines):
        if address is None or not text.startswith(("s_branch", "s_cbranch")):
            continue
        try:
            words = int(text.split()[-1], 0)
        except ValueError:
            continue
        words = words - 65536 if words >= 32768 else words
        to = at.get(address + 4 + 4 * words)
        if to is not None:
            out[i] = to
    return out


def kind(text):
    for prefixes, name in KINDS:
        if text.startswith(prefixes):
            return name
    return ""


def audit(what, lines):
    """A bounded wait is a pausing loop with no loop inside (its rounds) and
    the loop right around it (which reads the clock once a chunk of rounds).
    Larger pausing loops are the launch's own rounds around its waits."""
    to = targets(lines)
    kinds = [kind(text) for _, text in lines]
    loops = sorted(
        (
            (start, end)
            for end, start in to.items()
            if start <= end and "pause" in kinds[start : end + 1]
        ),
        key=lambda r: r[1] - r[0],
    )
    # the whole launch in address order: the clock's message or its wait while a global read is still out
    out, exposed, clocks = False, [], 0
    for i, k in enumerate(kinds):
        if k == "read":
            out = True
        elif k == "wait-read":
            out = False
        elif k in ("clock", "wait-clock"):
            clocks += k == "clock"
            if out:
                exposed.append(i)
    good, waits, rounds = bool(loops) and not exposed, 0, 0
    for start, end in loops:
        inner = [
            r for r in loops if r != (start, end) and start <= r[0] and r[1] <= end
        ]
        if len(inner) > 1:
            rounds += 1
            continue
        waits += 1
        events = [(i, kinds[i]) for i in range(start, end + 1) if kinds[i]]
        reads = [i for i, k in events if k == "read"]
        out, early = False, []
        for i, k in events:
            if k == "read":
                out = True
            elif k == "wait-read":
                out = False
            elif out:
                early.append((i, k))
        own_clock = [
            i
            for i, k in events
            if k == "clock" and not any(a <= i <= b for a, b in inner)
        ]
        ok = bool(reads) and not early and not out and (bool(inner) or not own_clock)
        good &= ok
        print(
            f"  loop {start}..{end} ({end - start + 1} instructions, "
            f"{'the rounds' if not inner else 'around the rounds'}): reads inside {len(reads)}, "
            f"anything waited for with a read out {early or 'never'}, a read out at the back-branch {out}, "
            f"clock reads of its own {len(own_clock)}: {'ok' if ok else 'NOT THE FORM'}",
            flush=True,
        )
        if SHOW and not inner:
            for i, k in events:
                print(f"       {i:5d}  {k:10s} {lines[i][1][:84]}")
            print(f"       {end:5d}  back       {lines[end][1][:84]} -> {start}")
    print(
        f"{what}: {len(lines)} instructions; {waits} loops of bounded waits (rounds, and the loop around them), "
        f"every one in the form: {good}; {clocks} clock messages in the launch, one sent or waited for with a "
        f"global read out: {exposed or 'never'}; {rounds} larger pausing loops (the launch's rounds) not judged",
        flush=True,
    )
    return good


def main():
    good = True
    for rows, qlen in HEADS:
        launch = K.build_head_verify(
            rows, qlen, head_dispatch.head_splits(rows), sources=SOURCES
        )
        good &= audit(
            K.head_verify_name(rows, qlen, head_dispatch.head_splits(rows)),
            code(launch),
        )
    for requests, drafts in BLOCKS:
        splits = head_dispatch.head_splits(requests * drafts)
        launch = MK.build_draft_markov(
            requests, drafts, 256, splits, sources=MARKOV_SOURCES
        )
        good &= audit(MK.draft_markov_name(requests, drafts, 256, splits), code(launch))
    print("PASSED 0" if good else "FAILED 1", flush=True)


if __name__ == "__main__":
    main()
