# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::lib::ds_v4_1_flash::(anonymous)::admit_kernel(AdmitArgs)``.
"""A request's admission on gfx950 as a FlyDSL port of TileRT's ``admit_kernel``: kernarg ABI unchanged (``AdmitArgs``,
176 B, read by offset as 22 i64 words; ``tk.host.admit_args`` builds it), grid 64 x 256, launched outside the step
graphs (``TkRank.admit``).

``AdmitArgs`` (byte offsets): ``plain_acc`` 0x00 / its int32 count 0x08, ``ar_acc`` 0x10 / 0x18, ``ar_num`` 0x20 /
0x28, ``ctl`` 0x30, ``blk_tok`` 0x38, ``blk_pos`` 0x40, rows 0x48, ``vtok`` 0x50, ``vpos`` 0x58, verify rows 0x60,
``embed`` 0x68, ``X`` 0x70, pos 0x78, first 0x7c, cap 0x80, mtp 0x84, ``samp_params`` 0x88 (null: untouched), its 4
int64 values 0x90.

Thread g (block b, lane t: g = 256 b + t) of G = 64 x 256, every store an int32 unless said:

- ``plain_acc[i] = 0`` for i = g, g + G, ... below its count;
- ``ar_acc[i] = mtp ? (i == 0 ? 1 : i == 1 ? first : 0) : 0``, ``ar_num[i] = mtp && i < 2`` the same way;
- g < 16: ``ctl[g]`` = pos - 1 (g 0), first (g 1), 1 (g 5) when mtp; cap (g 3); else 0 (all 16 written);
- ``samp_params`` given, g < 4: ``samp_params[g]`` = the g-th int64 of the args;
- then only when mtp: block 0, t < rows: ``blk_tok[t] = t ? 128799 : first``, ``blk_pos[t] = t < 5 && pos + t < cap ?
  pos + t : -1``; t < verify rows: ``vtok[t]`` the same, ``vpos[t] = pos + t < cap ? pos + t : -1``; block b < rows:
  row b of ``X`` (4 mHC copies, 0x2800 B apart, 0xa000 B a row) = the bf16 embedding of b ? 128799 : first (zeros
  for a token >= 129280), 16 B a lane at t, t + 256, t + 512 < 640.

The same on the CPU: ``tk.host.admit_reference`` (the GPU check's model).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.ops import kernel_symbol, launcher, rsrc, traced

Int32, Int64 = fx.Int32, fx.Int64
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 22  # AdmitArgs, 176 B
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
GRID, THREADS = 64, 256
TOTAL = GRID * THREADS
PAD_TOKEN = 128799  # 0x1f71f
VOCAB = 129280
ROW_BYTES = 0x2800  # one embedding row (5120 bf16)
X_ROW = 0xA000  # a row of X: 4 copies
VERSION = "v1"


def _lo(w):
    return Int32(w)


def _hi(w):
    return Int32(w >> 32)


def _st(addr, word, v):
    bo.buffer_store(Int32(v), rsrc(addr), word)


@traced
def _fill(addr, n, g, value_of):
    """``addr[i] = value_of(i)`` for i = g, g + TOTAL, ... < n."""
    i = g
    while i < n:
        _st(addr, i, value_of(i))
        i = i + TOTAL


@traced
def _entry(t, b, a):
    g = b * THREADS + t
    pos, first = _lo(a[15]), _hi(a[15])
    cap, mtp = _lo(a[16]), _hi(a[16]) != 0
    zero, one = Int32(0), Int32(1)
    _fill(a[0], _lo(a[1]), g, lambda i: zero)
    _fill(
        a[2],
        _lo(a[3]),
        g,
        lambda i: mtp.select((i == 0).select(one, (i == 1).select(first, zero)), zero),
    )
    _fill(a[4], _lo(a[5]), g, lambda i: (mtp & (i < 2)).select(one, zero))
    if g < 16:
        v = (g == 3).select(cap, zero)
        v = (mtp & (g == 0)).select(pos - 1, v)
        v = (mtp & (g == 1)).select(first, v)
        v = (mtp & (g == 5)).select(one, v)
        _st(a[6], g, v)
    if (a[17] != 0) & (g < 4):
        p = (g == 0).select(
            a[18], (g == 1).select(a[19], (g == 2).select(a[20], a[21]))
        )
        bo.buffer_store(
            fx.Vector.from_elements([_lo(p), _hi(p)], Int32), rsrc(a[17]), g * 2
        )
    if mtp:
        rows, vrows = _lo(a[9]), _lo(a[12])
        if b == 0:
            tok = (t == 0).select(first, Int32(PAD_TOKEN))
            p = pos + t
            if t < rows:
                _st(a[7], t, tok)
                _st(a[8], t, ((t < 5) & (p < cap)).select(p, Int32(-1)))
            if t < vrows:
                _st(a[10], t, tok)
                _st(a[11], t, (p < cap).select(p, Int32(-1)))
        if b < rows:
            tok = (b == 0).select(first, Int32(PAD_TOKEN))
            ok = fx.Uint32(tok) < fx.Uint32(VOCAB)
            src = a[13] + ok.select(tok, zero).to(Int64) * ROW_BYTES
            dst = a[14] + b.to(Int64) * X_ROW
            for k in range_constexpr(3):
                c = t + 256 * k
                if c < 640:
                    v = fx.Vector(
                        bo.buffer_load(rsrc(src), c * 4, vec_width=4, dtype=T.i32)
                    )
                    v = fx.Vector.from_elements(
                        [ok.select(v[e], zero) for e in range(4)], Int32
                    )
                    for h in range_constexpr(4):
                        bo.buffer_store(v, rsrc(dst), c * 4 + h * (ROW_BYTES // 4))


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 22 kernarg words."""
    name = kernel_symbol("tk_admit", v=VERSION)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def admit(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64,
    ):  # fmt: skip
        _entry(fx.thread_idx.x, fx.block_idx.x, [a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11,
                                                 a12, a13, a14, a15, a16, a17, a18, a19, a20, a21])  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        admit(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
              a20, a21).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
