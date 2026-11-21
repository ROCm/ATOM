# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The head + verify kernel's cross-rank poll as compiled: the system-scope
loads and stores, and what surrounds each load (the waits and the pause of a
poll round), from the serving build's code object.

    python tools/steptail_isa.py
    env: ROWS (8), QLEN (8), CONTEXT (10), LOOP (0: lines before the first WMMA to show)
"""

import os
import shutil
import subprocess

from atom.models.kimi_k3.step_tail import head_dispatch
from atom.models.kimi_k3.step_tail import kernels as K
from atom.models.kimi_k3.step_tail.sources import SOURCES
from atom.mono.runtime.compile import _code_object, compile_only

ROWS = int(os.environ.get("ROWS", "8"))
QLEN = int(os.environ.get("QLEN", "8"))
CONTEXT = int(os.environ.get("CONTEXT", "10"))
LOOP = int(os.environ.get("LOOP", "0"))


def main():
    launch = K.build_head_verify(
        ROWS, QLEN, head_dispatch.head_splits(ROWS), sources=SOURCES
    )
    with compile_only():
        launch(0, 0, 0, 0)
    objdump = shutil.which("llvm-objdump") or "/opt/rocm/llvm/bin/llvm-objdump"
    for artifact in launch._mem_cache.values():
        path = "/tmp/steptail_isa.o"
        with open(path, "wb") as f:
            f.write(_code_object(artifact._ir_text))
        text = subprocess.run(
            [objdump, "-d", "--no-show-raw-insn", path],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [ln.rstrip() for ln in text.stdout.splitlines() if ln.strip()]
        print("instructions", len(lines), text.stderr[:200])
        sys_at = [i for i, ln in enumerate(lines) if "SCOPE_SYS" in ln]
        loads = [i for i in sys_at if "_load" in lines[i]]
        stores = [i for i in sys_at if "_store" in lines[i]]
        print(f"system-scope loads {len(loads)}, stores {len(stores)}")
        for name in ("s_wait_loadcnt", "s_wait_storecnt", "s_sleep", "s_barrier"):
            print(name, sum(name in ln for ln in lines))
        for i in loads:
            print(f"--- load at {i}")
            for ln in lines[max(0, i - CONTEXT) : i + CONTEXT + 1]:
                print("   ", ln[:110])
        if stores:
            print(f"--- stores at {stores[0]}..{stores[-1]}")
            for ln in lines[stores[0] - 3 : stores[-1] + 3]:
                print("   ", ln[:110])
        wmma = [i for i, ln in enumerate(lines) if "v_wmma" in ln]
        print(f"--- {len(wmma)} WMMA; the K step around the first ones")
        if LOOP and wmma:
            for ln in lines[max(0, wmma[0] - LOOP) : wmma[min(len(wmma) - 1, 3)] + 6]:
                print("   ", ln.split("//")[0].rstrip()[:96])
        os.remove(path)


if __name__ == "__main__":
    main()
