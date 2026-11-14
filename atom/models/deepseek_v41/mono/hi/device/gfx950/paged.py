# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's paged V4.1 storage, read and written in place by a program (the engine's KV pool and FP4 index plane).

A KV owner (a compressing layer) keeps its compressed entries in ATOM's pool, a page of ``PAGE`` positions holding
``rpp = PAGE / ratio`` rows of the owner, and its FP4 index keys in a plane of its own, dense in 8-key units:

- entry ``e`` of the request with batch id ``bid``: page ``ptab[bid * stride + e // rpp]``, main row ``page *
  page_rows + e % rpp`` (bf16 [512], 1 KiB; in ATOM's fp8 pool MXFP8, ``KV8_ROW_BYTES``) from the owner's base
  (the pool + its main offset rows);
- its index key: unit ``u = (page * rpp + e % rpp) // 8``, lane ``t`` the remainder; the 16 B of 32-element group
  ``c`` at ``u * 512 + (t % 4) * 128 + c * 32 + (t // 4) * 16`` of the values plane, its E8M0 byte at ``u * 32 + c *
  8 + t`` of the scales plane (ATOM's FP4 ``preshuffle`` layout). In ATOM's ``mfma`` layout (gfx950's OPUS scorer,
  ``ATOM_V4_UNIFIED_MQA=opus``) unit ``u`` is rows ``r = (u % 8) * 8 + t`` of the 64-row group ``G = u // 8``: the
  16 B at ``G * 4096 + c * 1024 + r * 16``, the E8M0 byte at ``G * 256 + (c % 2) * 128 + (r % 32) * 4 + (c // 2) * 2
  + r // 32`` (``mfma``: the functions below).

Two pointer-table args (``REQ``): the step's block table (``ptab``, ATOM's) and the request words (``req``, int32
``REQ_WORDS``): the page geometry at ``GEO`` (table stride, page rows, the compress ring's slot stride, 0), then from
``BID`` / ``SLOT`` / ``LIVE`` a request's batch id, its ATOM slot (0 when dead), whether it is a real one (``MAX_REQ``
each), from ``ROW_LIVE`` a row's (``MAX_ROWS``): the geometry bound by the hook, the rest written by the begin-step
launch. Addresses are 64-bit (the pool is far past 4 GiB). Every store into ATOM's storage is masked by the row's liveness: a padded request or a band's dead
one must not write a real request's pages; every read clamps its batch id and its page into range.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, pow2, rcp_pow2
from atom.mono.device.ops import mxfp8_to_bf16x8

from ...plan import trace
from . import dual
from .ops import ld_i32, uniform

PAGE = 128  # ATOM's V4.1 page in positions (``hi_gate.BLOCK_SIZE``)
ROW_BYTES = 1024  # a main row: bf16 [512]
# a main row of ATOM's fp8 pool (``--kv_cache_dtype fp8``): MXFP8, 512 E4M3 values then their 16 E8M0 group-32 scales
KV8_ROW_BYTES = 528
KV8_GROUP = 32  # the values an E8M0 code of such a row scales
# ATOM's FP8 activations' amax floor (``blockscale_kernels.quantization.AMAX_FLOOR``): its writers' code of an all-zero
# (or tinier) group is the floor's
AMAX_FLOOR = 1e-10
REQ = ("req", "ptab")  # the args every in-place block takes
MAX_REQ, MAX_ROWS = (
    8,
    64,
)  # the request words' requests and rows (a step's, a band's dead ones included)
GEO, BID, SLOT, LIVE, ROW_LIVE = (
    0,
    4,
    4 + MAX_REQ,
    4 + 2 * MAX_REQ,
    4 + 3 * MAX_REQ,
)  # word offsets in ``req``
REQ_WORDS = ROW_LIVE + MAX_ROWS


def rpp(ratio: int) -> int:
    """An owner's rows a page at compress ``ratio``."""
    return PAGE // ratio


def page_of(k, b, e, ratio: int, uni: bool = True):
    """The page of entry ``e`` (a compressed index, >= 0) of request ``b``: ``ptab[bid * stride + e // rpp]``
    (``uni``: a wave-uniform scalar; else a lane's own), the column and the page clamped into range (a dead request's
    or a padded entry's read lands on some valid row; its result is not used)."""
    bid = fx.max(uniform(ld_i32(k.args["req"], BID + b)), fx.Int32(0))
    stride = uniform(ld_i32(k.args["req"], GEO))
    col = fx.min(e // rpp(ratio), stride - 1)
    page = fx.max(ld_i32(k.args["ptab"], bid * stride + col), fx.Int32(0))
    return uniform(page) if uni else page


def main_row(k, b, e, ratio: int, uni: bool = True):
    """Entry ``e``'s main row (Int64) from the owner's base."""
    page_rows = uniform(ld_i32(k.args["req"], GEO + 1))
    page = page_of(k, b, e, ratio, uni)
    return fx.Int64(page) * fx.Int64(page_rows) + fx.Int64(e % rpp(ratio))


def row_addr(k, base_arg: str, b, e, ratio: int, uni: bool = True, row_bytes: int = ROW_BYTES):  # fmt: skip
    """Entry ``e``'s main row address (64-bit) in the owner's rows at ``k.args[base_arg]`` (``row_bytes``: the
    pool's row, ``ROW_BYTES`` or ``KV8_ROW_BYTES``)."""
    return fx.Int64(k.args[base_arg]) + main_row(k, b, e, ratio, uni) * row_bytes


def key_unit(k, b, e, ratio: int, uni: bool = True):
    """Entry ``e``'s FP4 index key: (unit (Int64), lane t). (rpp is a multiple of 8: ``t`` is ``e % 8``.)"""
    r = rpp(ratio)
    unit = fx.Int64(page_of(k, b, e, ratio, uni)) * (r // 8) + fx.Int64((e % r) // 8)
    return unit, e % 8


def key_value_addr(k, vals_arg: str, unit, t, group, mfma: bool = False):
    """The 16 B of 32-element group ``group`` of the key at (``unit``, ``t``) in the values plane (``mfma``: the
    ``mfma`` layout's)."""
    if mfma:
        row = (unit % 8) * 8 + fx.Int64(t)
        off = (unit // 8) * 4096 + fx.Int64(group * 1024) + row * 16
        return fx.Int64(k.args[vals_arg]) + off
    off = unit * 512 + fx.Int64((t % 4) * 128 + group * 32 + (t // 4) * 16)
    return fx.Int64(k.args[vals_arg]) + off


def key_scale_addr(k, scales_arg: str, unit, t, group, mfma: bool = False):
    """The E8M0 byte of group ``group`` of the key at (``unit``, ``t``) in the scales plane (``mfma``: the
    ``mfma`` layout's)."""
    if mfma:
        word, byte = mfma_scale_word(k, scales_arg, unit, t, group)
        return word + fx.Int64(byte)
    return fx.Int64(k.args[scales_arg]) + unit * 32 + fx.Int64(group * 8 + t)


def mfma_scale_word(k, scales_arg: str, unit, t, group):
    """In the ``mfma`` layout: (the 4-byte aligned address of the word holding the E8M0 byte of group ``group`` of
    the key at (``unit``, ``t``), that byte's index in it)."""
    row = (unit % 8) * 8 + fx.Int64(t)
    word = (unit // 8) * 256 + fx.Int64((group % 2) * 128) + (row % 32) * 4
    byte = fx.Int32((group // 2) * 2) + fx.Int32(row // 32)
    return fx.Int64(k.args[scales_arg]) + word, byte


def row_live(k, row):
    """Whether step row ``row`` is a real one (a live request's row: its stores reach ATOM's storage)."""
    return ld_i32(k.args["req"], ROW_LIVE + row) != 0


def req_slot(k, b):
    """Request ``b``'s ATOM slot (clamped to 0 for a dead one)."""
    return fx.max(uniform(ld_i32(k.args["req"], SLOT + b)), fx.Int32(0))


def ring_stride(k):
    """The compress ring's request-slot stride (f32 words)."""
    return uniform(ld_i32(k.args["req"], GEO + 2))


def global_store(addr, value, nbytes: int = 4):
    """``value`` (an Int32, its low byte with ``nbytes`` 1; a vector of 4 Int32 with 16) at a 64-bit global
    address."""
    trace.mem("store", addr, 0, nbytes, value)  # the memory-model record (no IR)
    dual.global_access(addr)
    ptr = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))
    ).result
    v = as_ir_value(value)
    if nbytes == 1:
        v = arith.TruncIOp(ir.IntegerType.get_signless(8), v).result
    llvm.StoreOp(v, ptr, alignment=nbytes)


# ------------------------------------------------------------------ ATOM's fp8 pool rows (MXFP8)
#
# A row: 512 E4M3 values (byte d: dim d), then the 16 E8M0 codes of its 32-dim groups (byte 512 + g). ATOM's writers
# (``fp8_groups``: the compressor's ``fused_compress``, the window's ``rope_window``, ``quantize_fp8``) quantize a bf16
# row group by group: the code of the smallest power of two >= max(amax, AMAX_FLOOR) x fp32(1 / 448), the values times
# its exact reciprocal, clamped to +-448, E4M3 rounded to nearest even (the hardware conversion). Its readers take
# value x 2^(code - 127): exact in bf16 (a 4-bit significand by a power of two).


def kv8_code(amax):
    """ATOM's writer's E8M0 code of a group whose largest |value| is ``amax`` (f32, >= 0)."""
    return code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX))


def kv8_scaled(v, code):
    """``v`` (f32) of a group of code ``code`` divided by its scale (exactly) and clamped: what the E4M3 byte rounds."""
    return clamp_fp8(v * rcp_pow2(code))


def kv8_value(scaled, code):
    """The value the row holds for ``kv8_scaled``'s ``scaled``: E4M3 (RNE) times the scale, f32 (a bf16 exactly)."""
    w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, scaled, scaled, fx.Int32(0), False))
    return fx.Float32(rocdl.cvt_f32_fp8(T.f32, w.ir_value(), 0)) * pow2(code)


def kv8_code_of(word, group):
    """Group ``group``'s code from the code word holding it (the row's code bytes 4 (group // 4) .. + 4), Int32."""
    return (fx.Int32(word) >> ((group % 4) * 8)) & 0xFF


def kv8_words(w0, w1, code):
    """Eight E4M3 values (two words, dims in byte order) of a group of code ``code`` as four bf16-pair words: the
    words a bf16 row holds there (exact)."""
    v = mxfp8_to_bf16x8(w0, w1, pow2(code)).bitcast(fx.Int32)
    return [v[e] for e in range(4)]
