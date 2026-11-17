# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The final ISA of a built kernel (a ``MonoKernel``): its code object, disassembly and a normalized hash
(``isa_hash``: the instructions without addresses or label names), and the ``.text`` range.
"""

import hashlib
import os
import re
import subprocess
import tempfile

from .compile import _code_object, code_bytes  # noqa: F401 - code_bytes: re-exported


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


def disassemble(kern, mcpu: str | None = None) -> str:
    """Disassembly of ``kern``'s (a ``MonoKernel``) code object for its program's target
    (``mcpu``: override)."""
    artifact = next(iter(kern.launcher._mem_cache.values()))
    elf = _code_object(artifact._ir_text)
    mcpu = mcpu or kern.program.target.mcpu
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(elf)
        f.flush()
        cmd = [_objdump(), "-d", f"--mcpu={mcpu}", "--no-show-raw-insn", f.name]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return out


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


def isa_hash(kern) -> str:
    return hashlib.sha256(normalize(disassemble(kern)).encode()).hexdigest()[:16]


def code_object(kern) -> bytes:
    """``kern``'s (a ``MonoKernel``) code object (the ELF the runtime loads)."""
    return _code_object(next(iter(kern.launcher._mem_cache.values()))._ir_text)


def text_range(elf: bytes) -> tuple[int, int]:
    """(address, bytes) of a code object's ``.text`` (the kernel's code, its tail padding included)."""
    import struct

    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    names = struct.unpack_from("<Q", elf, shoff + shstrndx * shentsize + 0x18)[0]
    for i in range(shnum):
        h = shoff + i * shentsize
        name = struct.unpack_from("<I", elf, h)[0]
        if elf[names + name : names + name + 6] == b".text\0":
            addr, _, size = struct.unpack_from("<QQQ", elf, h + 0x10)
            return addr, size
    raise RuntimeError("no .text in the code object")
