# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MFMA srcC hazards the compiler leaves in gfx950 code, checked on the built ISA (``MonoKernel``: ``ResourceError``).

An MFMA whose srcC is exactly the vDst of an earlier MFMA gets the result forwarded, and LLVM inserts no wait states
for an exact overlap. Forwarding needs equal pass counts: ``v_mfma_f32_16x16x16_bf16`` (4 passes) reading the vDst of
``v_mfma_f32_16x16x32_bf16`` (8 passes) as srcC read stale elements (the first two of a lane's four) with up to 4 wait
states between, never with 5. The writer may be in flight for the other wave of the SIMD too, so it shows only in
some builds.

``srcc_hazards(text)``: every MFMA of straight-line code whose srcC exactly overlaps the vDst of the latest MFMA of a
different opcode writing it, with fewer wait states between than ``GAPS`` (or ``UNKNOWN_GAP`` for a pair not in it).
Each instruction counts one wait state, ``s_nop k`` k + 1 (a lower bound). Partial overlaps are LLVM's
(GCNHazardRecognizer handles them); a writer in another basic block is not seen.
"""

import re

# (writer, reader) -> wait states the reader needs after the writer (measured: 16x16x32 bf16 -> 16x16x16 bf16, the
# writer's passes - the reader's + 1)
GAPS = {("v_mfma_f32_16x16x32_bf16", "v_mfma_f32_16x16x16_bf16"): 5}
UNKNOWN_GAP = 18  # a pair not measured: the longest XDL op's passes + 2

_INS = re.compile(r"^\s+([a-z_][a-z0-9_]*)\s*(.*?)\s*(?://\s*([0-9A-Fa-f]+):.*)?$")
_RANGE = re.compile(r"^[va]\[(\d+):(\d+)\]$")


def _regs(op: str) -> tuple:
    m = _RANGE.match(op.strip())
    return (op.strip()[0], int(m.group(1)), int(m.group(2))) if m else ()


def srcc_hazards(text: str) -> list[str]:
    """The hazards of a disassembly (``llvm-objdump -d``): one line each (address, writer, reader, wait states)."""
    ins = []
    for ln in text.splitlines():
        m = _INS.match(ln)
        if m:
            ins.append((m.group(3) or "?", m.group(1), m.group(2)))
    out = []
    for i, (addr, mn, args) in enumerate(ins):
        if not mn.startswith("v_mfma"):
            continue
        ops = re.split(r",\s*", args)
        c = _regs(ops[3]) if len(ops) > 3 else ()
        if not c:
            continue
        ws, j = 0, i - 1
        while j >= 0 and ws < UNKNOWN_GAP:
            pm, pa = ins[j][1], ins[j][2]
            if pm.startswith("s_nop"):
                ws += int(pa.split()[0], 0) + 1 if pa else 1
            elif pm.startswith(("s_cbranch", "s_branch", "s_setpc", "s_endpgm")):
                break
            else:
                if pm.startswith("v_mfma"):
                    d = _regs(re.split(r",\s*", pa)[0])
                    if (
                        d and d[0] == c[0] and d[1] <= c[2] and c[1] <= d[2]
                    ):  # the latest MFMA writing any of it
                        need = GAPS.get((pm, mn), UNKNOWN_GAP)
                        if d == c and pm != mn and ws < need:
                            out.append(
                                f"{addr}: {mn} srcC {ws} wait states after {pm} (needs {need})"
                            )
                        break
                ws += 1
            j -= 1
    return out
