# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft launch's cross-rank poll as compiled, a shape after another:
every system-scope load with what surrounds it (is the loop's reload after the
pause and before the back-branch?), then how two shapes' loops differ.

    python tools/steptail_markov_isa.py
    env: SHAPES (8x4,9x4: requests x drafts), DESIGN ("": the served design;
    v1: the first design), VOLATILE (0; v1: the poll's loads volatile),
    v1's other poll knobs SLEEP, POLLROUNDS, SETTLE, CLOCKFIRST (0),
    BEFORE (12), AFTER (30: instructions to show around a load)
"""

import difflib
import os
import shutil
import subprocess

from atom.models.kimi_k3.step_tail import head_dispatch
from atom.mono.runtime.compile import _code_object, compile_only

SHAPES = [
    tuple(int(v) for v in s.split("x"))
    for s in os.environ.get("SHAPES", "8x4,9x4").split(",")
]
DESIGN = os.environ.get("DESIGN", "")
VOLATILE = os.environ.get("VOLATILE", "0") == "1"
KNOBS = {
    "volatile": VOLATILE,
    "sleep": int(os.environ.get("SLEEP", "0")),
    "rounds": int(os.environ.get("POLLROUNDS", "0")),
    "settle": os.environ.get("SETTLE", "0") == "1",
    "clockfirst": os.environ.get("CLOCKFIRST", "0") == "1",
}
BEFORE = int(os.environ.get("BEFORE", "12"))
AFTER = int(os.environ.get("AFTER", "30"))
BRANCHES = ("s_branch", "s_cbranch")


def disassemble(requests, drafts):
    """The build's instructions: (address or None, text)."""
    if DESIGN == "v1":
        import steptail_markov_v1 as MK  # beside this tool

        extra = dict(KNOBS)
    else:
        from atom.models.kimi_k3.step_tail import markov_kernels as MK

        extra = {}
    launch = MK.build_draft_markov(
        requests,
        drafts,
        256,
        head_dispatch.head_splits(requests * drafts),
        sources="isa",
        **extra,
    )
    with compile_only():
        launch(0, 0, 0, 0)
    objdump = shutil.which("llvm-objdump") or "/opt/rocm/llvm/bin/llvm-objdump"
    out = []
    for artifact in launch._mem_cache.values():
        path = f"/tmp/steptail_markov_isa_{os.getpid()}.o"
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
            code, _, comment = ln.partition("//")
            address = None
            token = comment.strip().split(":")[0].split(" ")[0]
            try:
                address = int(token, 16)
            except ValueError:
                pass
            out.append((address, code.strip()))
    return out


def target(lines, i):
    """The instruction a branch at ``i`` goes to (its operand: words from the
    next instruction), or None."""
    address, code = lines[i]
    if address is None or not code.startswith(BRANCHES):
        return None
    try:
        words = int(code.split()[-1], 0)
    except ValueError:
        return None
    words = words - 65536 if words >= 32768 else words
    to = address + 4 + 4 * words
    return next((j for j, (a, _) in enumerate(lines) if a == to), None)


def show(lines, lo, hi, mark):
    for j in range(max(0, lo), min(len(lines), hi)):
        to = target(lines, j)
        note = "" if to is None else f"   -> {to} ({'back' if to <= j else 'forward'})"
        print(f"   {'>' if j == mark else ' '} {j:5d}  {lines[j][1][:96]}{note}")


def main():
    loops = {}
    for requests, drafts in SHAPES:
        lines = disassemble(requests, drafts)
        code = [c for _, c in lines]
        loads = [i for i, c in enumerate(code) if "SCOPE_SYS" in c and "_load" in c]
        stores = [i for i, c in enumerate(code) if "SCOPE_SYS" in c and "_store" in c]
        pauses = [i for i, c in enumerate(code) if c.startswith("s_sleep")]
        print(
            f"b{requests} t{drafts} design {DESIGN or 'served'} knobs {KNOBS if DESIGN else ''}: {len(lines)} instructions, "
            f"system-scope loads at {loads}, system-scope stores {len(stores)} ({stores[:1]}..{stores[-1:]}), "
            f"pauses at {pauses}, s_wait_loadcnt {sum(c.startswith('s_wait_loadcnt') for c in code)}",
            flush=True,
        )
        for i in loads:
            paused = [j for j in pauses if i - 6 <= j < i]
            # the first branch after the load that goes back to or before the pause
            back = next(
                (
                    j
                    for j in range(i, min(i + 80, len(lines)))
                    if target(lines, j) is not None and target(lines, j) <= i
                ),
                None,
            )
            wait = next(
                (
                    j - i
                    for j in range(i, min(i + 8, len(lines)))
                    if code[j].startswith("s_wait_loadcnt")
                ),
                None,
            )
            print(
                f"--- load at {i}: a pause in the 6 instructions before it: {bool(paused)}; s_wait_loadcnt at "
                f"+{wait}; the first branch back past it: "
                f"{None if back is None else f'{back} -> {target(lines, back)}'}",
                flush=True,
            )
            show(lines, i - BEFORE, i + AFTER + 1, i)
            if paused and back is not None:
                loops[(requests, drafts)] = code[target(lines, back) : back + 1]
    if len(loops) == 2:
        (a, ta), (b, tb) = loops.items()
        diff = [
            ln
            for ln in difflib.unified_diff(ta, tb, lineterm="", n=0)
            if not ln.startswith(("---", "+++", "@@"))
        ]
        print(
            f"the poll loop (back-branch target through the back-branch): b{a[0]} t{a[1]} {len(ta)} instructions, "
            f"b{b[0]} t{b[1]} {len(tb)}; differing lines {len(diff)}:"
        )
        for ln in diff:
            print("    ", ln[:110])


if __name__ == "__main__":
    main()
