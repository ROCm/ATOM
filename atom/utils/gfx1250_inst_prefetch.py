# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Instruction prefetch prologue for Triton / Gluon kernels on gfx1250.

A kernel that a CU runs once per launch over KB of straight-line code stalls on every new 128-byte
instruction line when it starts with a cold instruction cache: ~0.4-0.5 us per KB, whether the code
comes from L2 or HBM. A CU's instruction cache holds < 64 KB, so in a layer of several kernels every
kernel starts cold (the GDN prefill layer: fused KKT 6 us warm -> 20.5 us cold, chunk_o 4.2 -> 10.3).
The kernel descriptor's launch prefetch (INST_PREF_SIZE) does not help on this part.

`install()` adds a Triton compiler stage hook that edits the AMDGCN assembly of the selected kernels
(name matched by a regular expression): right after the entry it enables scalar prefetch (MODE bit 24)
and issues `s_prefetch_inst_pc_rel` requests for the code that follows, 4 KB per instruction. The
prefetched range starts after these instructions and ends at least 16 bytes before the end of the
kernel's code as compiled without them (`codeLenInByte`; inserting code at the entry only moves later
code to higher addresses, alignment included), so it never reaches past the kernel: prefetching past
a code object faults the queue. No arithmetic changes: outputs are bitwise the same.

The hook changes Triton's cache key of every kernel compiled after it is installed (the selection is
part of the key), so the first run recompiles them once.
"""

import re

_LINE = 128  # instruction cache line
_PER_INSTR = 32  # an s_prefetch_inst covers at most 32 lines
_HEAD = 128  # bytes left unprefetched: covers the required prologue before the block (<= 96 bytes)
_VERSION = 2  # part of the Triton cache key: bump when the inserted code changes
_installed: dict = {}


def prefetch_block(code_bytes: int) -> list:
    """Assembly lines that prefetch the code after them, for a kernel of `code_bytes` (compiled without
    them): setreg (8 bytes), then the prefetches (8 each). Placed at most 96 bytes after the entry, the
    range [block + 16, block + 16 + lines * 128) ends inside the kernel."""
    lines = (code_bytes - _HEAD) // _LINE
    if lines <= 0:
        return []
    out = ["\ts_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 24, 1), 1 ; scalar prefetch on (inst prefetch)"]
    i = 0
    while lines > 0:
        n = min(_PER_INSTR, lines)
        # koffset counts from the next instruction: prefetch i starts at block + 16 + 4 KB * i
        out.append(f"\ts_prefetch_inst_pc_rel {i * _PER_INSTR * _LINE - 8 * i:#x}, null, {n - 1}")
        lines -= n
        i += 1
    return out


def add_prefetch(asm: str, name: str) -> str:
    """The AMDGCN assembly `asm` of kernel `name` with the prefetch block at its entry."""
    m = re.search(r"codeLenInByte = (\d+)", asm)
    if not m or "s_prefetch_inst_pc_rel" in asm:  # (once: a second block would move the first)
        return asm
    block = prefetch_block(int(m.group(1)))
    if not block:
        return asm
    lines = asm.split("\n")
    try:
        at = next(i for i, ln in enumerate(lines) if ln.startswith(name + ":"))
    except StopIteration:
        return asm
    # gfx1250 kernels must begin with the compiler's prologue (s_mov_b64, v_nop, global_prefetch_b8,
    # after an optional MODE setreg): the block goes right after it, at most 8 instructions in.
    insts = 0
    for j in range(at + 1, min(at + 60, len(lines))):
        t = lines[j].strip()
        if not t or t.startswith((";", ".", "//")) or t.endswith(":"):
            continue
        insts += 1
        if t.startswith("global_prefetch_b8"):
            return "\n".join(lines[: j + 1] + block + lines[j + 1:])
        if insts >= 8:
            break
    return asm  # no recognized prologue: leave the kernel alone


def install(pattern: str) -> None:
    """Prefetch the code of the gfx1250 Triton / Gluon kernels whose name matches `pattern` (re.search)
    in every kernel compiled from now on. Idempotent per pattern; "" does nothing."""
    if not pattern or pattern in _installed:
        return
    from triton import knobs

    sel = re.compile(pattern)
    prev = knobs.runtime.add_stages_inspection_hook
    tag = f"-instpf{_VERSION}[{pattern}]"

    def hook(*args):
        if not args:  # the cache-key query
            key, digest = prev() if prev is not None else ("", "")
            return key + tag, digest + tag
        backend, stages, options, language, capability = args
        if prev is not None:
            prev(*args)
        if getattr(options, "arch", "") != "gfx1250" or "amdgcn" not in stages:
            return
        make_amdgcn = stages["amdgcn"]

        def amdgcn(src, metadata):
            asm = make_amdgcn(src, metadata)
            name = metadata.get("name", "")
            return add_prefetch(asm, name) if sel.search(name) else asm

        stages["amdgcn"] = amdgcn

    knobs.runtime.add_stages_inspection_hook = hook
    _installed[pattern] = hook


def install_from_env() -> None:
    """install(ATOM_INST_PREFETCH_KERNELS)."""
    from atom.utils import envs

    install(envs.ATOM_INST_PREFETCH_KERNELS)
