# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A V4.1 layer kernel's code read into its XCD's L2 at kernel entry
(``layer.warm_layer_code``): gfx9 fetches no instruction ahead, so a cold
stage's I-cache misses would go to HBM."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.ops import i64_words, rsrc

# an L2 line: ``warm_code`` reads a dword of each
L2_LINE = 128
# the pc and lld's ``_etext`` (the end of the code object's last read-only
# segment: its .text) in one asm: ``s_getpc_b64`` gives the address of the next
# instruction, the ``s_mov``, whose literal sits 4 bytes past it, so the
# pc-relative literal + 4 is ``_etext - pc``. Hidden, the reference binds at
# the link (no dynamic relocation; a default-visibility one is refused)
_CODE_TAIL_ASM = ".hidden _etext\ns_getpc_b64 $0\ns_mov_b32 $1, _etext@rel32@lo+4"


def code_tail():
    """(pc, the bytes from the pc to the end of the code object's text)."""
    pair = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64, T.i32]), [], _CODE_TAIL_ASM, "=s,=s",
        has_side_effects=True,
    )  # fmt: skip
    return (
        fx.Int64(llvm.extractvalue(T.i64, pair, [0])),
        fx.Int32(llvm.extractvalue(T.i32, pair, [1])),
    )


def warm_code(lane, span):
    """The kernel's code from here on, ``span`` bytes at most, read into this
    XCD's L2 (gfx9 fetches no instruction ahead: a cold stage's I-cache misses
    go to HBM): a dword of each L2 line, a wave's 64 lanes 8 KB a load. A
    buffer whose records end at the code object's text (``code_tail``): the
    lanes past it load nothing, so no span reads outside the code object. The
    loads are kept by one use, here (LLVM drops an unused load; a volatile one
    would take the system scope's bypass and a wait each): the wave waits for
    them all, so it runs on a CTA with nothing to do for that long."""
    pc, tail = code_tail()
    base = pc & fx.Int64(-L2_LINE)
    r = rsrc(base, fx.Int64(tail + (i64_words(pc)[0] & (L2_LINE - 1))))
    words = L2_LINE // 4
    acc = None
    for k in range_constexpr(-(-span // (64 * L2_LINE))):
        v = fx.Int32(
            bo.buffer_load(r, lane * words + k * 64 * words, vec_width=1, dtype=T.i32)
        )
        acc = v if acc is None else acc | v
    llvm.inline_asm(
        None, [acc.ir_value()], "; code warm $0", "v", has_side_effects=True
    )
