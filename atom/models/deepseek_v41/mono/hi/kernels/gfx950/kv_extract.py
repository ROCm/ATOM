# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::toktab::kv_extract_kernel(u8 const*, float const*, u16*)``.
"""The DSpark token tables' kv on gfx950 as a FlyDSL port of TileRT's ``toktab::kv_extract_kernel``: kernarg ABI
unchanged (``ws``, ``kv_norm``, ``out``: 3 pointers, 24 B), one CTA of 256 threads a row (grid = rows), run after the
QKV kernel (``q_kv_xf_kg``) of the token table build (``tk.toktab_device``).

Row r: the window kv GEMM's 512 bf16 outputs from the QKV workspace's LL packs ({bf16x2, epoch}, 8 B, at
``ws + 0x2a400 + r * 0x1c00``: lane t takes features t and t + 256), RMS-normed as the QKV kernel's kv-norm does it
(``qkv.qkv_kvnorm``: a + b^2 fused, xor 32..2 butterflies, the last one on lane 0, waves added in order from 0,
1 / sqrt(fma(sum, 1/512, 1e-20)) IEEE), ``kv_norm * (x * rstd)`` to bf16 (RNE) at ``out + r * 1024`` B.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.hi.kernels.gfx950.common import bf_rne_hi, f32c
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.qkv import (
    KV_LL,
    LL_QA,
    _lds_ld,
    _lds_st,
    _shfl_xor,
)
from atom.mono.device.ops import kernel_symbol, launcher, rsrc, traced

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 3
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
THREADS = 256
ROW_PACKS = 0x1C00  # a token row's LL packs (1280 q_a + 512 kv columns, 8 B a pair)
VERSION = "v1"


def _ld(addr, word):
    return Int32(bo.buffer_load(rsrc(addr), word, vec_width=1, dtype=T.i32))


@traced
def _entry(tid, row, ws, gamma, out, lds):
    lane, w = tid % 64, tid // 64
    word = (LL_QA + KV_LL + row * ROW_PACKS + (tid // 2) * 8) // 4
    pa = _ld(ws, word)  # pack t / 2: features t & ~1, t | 1
    pb = _ld(ws, word + 256)  # pack 128 + t / 2: feature 256 + t
    sh = (tid & 1) * 16
    a = ((pa >> sh) << 16).bitcast(F32)
    b = ((pb >> sh) << 16).bitcast(F32)
    v = fmath.fma(a, a, b * b)
    for k in range_constexpr(5):
        v = v + _shfl_xor(v, lane, 32 >> k)
    last = _shfl_xor(v, lane, 1)
    if lane == 0:
        _lds_st(lds, w * 4, [(v + last).bitcast(Int32)])
    gpu.barrier()
    tot = F32(0.0)
    for q in range_constexpr(4):
        tot = tot + _lds_ld(lds, q * 4, 1).bitcast(F32)
    rstd = F32(1.0) / fmath.sqrt(fmath.fma(tot, f32c(0x3B000000), f32c(0x1E3CE508)))
    g1 = _ld(gamma, tid).bitcast(F32)
    g2 = _ld(gamma, tid + 256).bitcast(F32)
    ya = (bf_rne_hi(g1 * (a * rstd)) >> 16).to(fx.Int16)
    yb = (bf_rne_hi(g2 * (b * rstd)) >> 16).to(fx.Int16)
    bo.buffer_store(ya, rsrc(out), row * 1024 + tid * 2, offset_is_bytes=True)
    bo.buffer_store(yb, rsrc(out), row * 1024 + 512 + tid * 2, offset_is_bytes=True)


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 3 kernarg words (grid = rows)."""
    name = kernel_symbol("tk_toktab_kv_extract", v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kv_extract(a00: Int64, a01: Int64, a02: Int64):
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _entry(fx.thread_idx.x, fx.block_idx.x, a00, a01, a02, lds)

    @launcher(name)
    def launch(a00: Int64, a01: Int64, a02: Int64, stream: fx.Stream = _CURRENT_STREAM):
        kv_extract(a00, a01, a02).launch(grid=(6,), block=(THREADS,), stream=stream)

    return name, launch
