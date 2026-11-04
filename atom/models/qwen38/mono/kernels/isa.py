# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The final ISA of a built kernel: disassembly, its normalized text, a summary.

Used to check a change on the assembly (did the intended instruction change happen, and did nothing else move:
waits, cache bits, barriers, spills):

    from atom.models.qwen38.mono.kernels.isa import disassemble, normalize, summary
    text = disassemble(kern)          # kern: MonoKernel
    print(summary(text))
"""

import os
import re
import subprocess
import tempfile
from collections import Counter

from atom.mono.runtime.compile import _code_object


def _objdump() -> str:
    """ROCm's llvm-objdump: under ``ROCM_PATH`` (or /opt/rocm when unset: serving environments often don't set it),
    else the one on ``PATH``."""
    import shutil

    for root in dict.fromkeys(
        r for r in (os.environ.get("ROCM_PATH"), "/opt/rocm") if r
    ):
        for sub in (("lib", "llvm", "bin"), ("llvm", "bin")):
            path = os.path.join(root, *sub, "llvm-objdump")
            if os.path.exists(path):
                return path
    found = shutil.which("llvm-objdump")
    if found is None:
        raise FileNotFoundError(
            "llvm-objdump: not under $ROCM_PATH or /opt/rocm ({lib/,}llvm/bin) nor on PATH"
        )
    return found


def objdump(elf: bytes, mcpu: str, *flags: str) -> str:
    """``llvm-objdump -d --mcpu=<mcpu> <flags>`` of a code object."""
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(elf)
        f.flush()
        cmd = [_objdump(), "-d", f"--mcpu={mcpu}", *flags, f.name]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def code_object(kern) -> bytes:
    """The code object of ``kern`` (a ``MonoKernel``)."""
    return _code_object(next(iter(kern.launcher._mem_cache.values()))._ir_text)


def disassemble(kern, mcpu: str | None = None) -> str:
    """Disassembly of ``kern``'s (a ``MonoKernel``) code object for its program's target
    (``mcpu``: override)."""
    return objdump(
        code_object(kern), mcpu or kern.program.target.mcpu, "--no-show-raw-insn"
    )


_ADDR = re.compile(r"^\s*[0-9a-f]+:\s*", re.MULTILINE)
_LABEL = re.compile(r"<[^>]+>")
_COMMENT = re.compile(r"\s*//.*$")


def normalize(text: str) -> str:
    """Instruction text without addresses or label names: stable across relinks."""
    lines = [
        _COMMENT.sub("", _LABEL.sub("<L>", _ADDR.sub("", ln))).strip()
        for ln in text.splitlines()
    ]
    return "\n".join(
        ln
        for ln in lines
        if ln and not ln.startswith(("Disassembly", "/")) and not ln.endswith(":")
    )


def summary(text: str) -> dict:
    """Instruction mix: totals and the counts that matter for a resident kernel (the keys from
    ``wmma`` on count other-architecture instructions: zero on gfx950)."""
    ops = Counter(ln.split()[0] for ln in normalize(text).splitlines() if ln.split())
    waits = {
        k: v for k, v in ops.items() if k.startswith("s_wait_") and k != "s_waitcnt"
    }
    return {
        "instructions": sum(ops.values()),
        "mfma": sum(v for k, v in ops.items() if "mfma" in k),
        "global_or_buffer_loads": sum(
            v for k, v in ops.items() if k.startswith(("buffer_load", "global_load"))
        ),
        "stores": sum(
            v for k, v in ops.items() if k.startswith(("buffer_store", "global_store"))
        ),
        "lds": sum(v for k, v in ops.items() if k.startswith("ds_")),
        "s_waitcnt": ops.get("s_waitcnt", 0),
        "s_barrier": ops.get("s_barrier", 0),
        "scratch": sum(v for k, v in ops.items() if k.startswith("scratch_")),
        "wmma": sum(v for k, v in ops.items() if "wmma" in k),
        "tdm": sum(v for k, v in ops.items() if k.startswith("tensor_")),
        "async_lds": sum(
            v for k, v in ops.items() if "async" in k and not k.startswith("s_wait")
        ),
        "s_wait": sum(waits.values()),
        "s_wait_kinds": dict(sorted(waits.items())),
        "s_barrier_split": ops.get("s_barrier_wait", 0),
    }
