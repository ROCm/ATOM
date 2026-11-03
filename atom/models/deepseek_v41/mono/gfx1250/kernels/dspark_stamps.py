# SPDX-License-Identifier: MIT
"""Optional gfx1250 local stage records, with no cross-launch clock subtraction.

Only thread zero stamps its stage arrival into LDS. Global records are flushed
once, after the kernel's mathematical work; no extra CTA barrier is introduced.
Stage arrival is not a claim that every producer has finished. Release builds
trace all calls to nothing and preserve their original symbols and LDS layout.
"""
import os

import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels import buffer_ops as bo
from atom.mono.device.ops import rsrc, traced
from atom.models.deepseek_v41.mono.gfx1250.kernels.reference_sync import memrealtime
from atom.mono.device.ops import CM_DEV

ENABLED = os.environ.get("ATOM_DSV41_GFX1250_STAMPS", "0") == "1"
MAX_ROWS = 48
SQ_WORDS = MAX_ROWS * 160 * 2
LAYERS = 64
BLOCKS = 256
SLOTS = 16
WORDS = LAYERS * BLOCKS * SLOTS * 2
SIGNATURE = 0x57300000
TICK_US = 0.01


def k1_offset_words():
    # Import after the K1 module has defined its private gemv loader. An eager
    # P1A import cycles through raw_handoff -> WO -> partially imported K1,
    # binding WO to the wildcard-imported gfx950 loader instead.
    from atom.models.deepseek_v41.mono.gfx1250.kernels.p1a_attention import P1A_STATE_WORDS
    return SQ_WORDS + P1A_STATE_WORDS


def k2_offset_words():
    return k1_offset_words() + WORDS


@traced
def begin(tls, tid):
    if const_expr(ENABLED):
        if tid == 0:
            for slot in range(SLOTS):
                fx.ptr_store(fx.Int64(0), tls + slot)


@traced
def stamp(tls, tid, slot):
    if const_expr(ENABLED):
        if tid == 0:
            fx.ptr_store(memrealtime(True), tls + slot)


@traced
def flush(tls, base, tid, bid, layer, rows):
    if const_expr(ENABLED):
        if tid == 0:
            at = ((layer % LAYERS) * BLOCKS + bid) * SLOTS
            old = fx.Vector(bo.buffer_load(rsrc(base), (at + 14) * 2,
                                          vec_width=2, dtype=T.i32, cache_modifier=CM_DEV))
            generation = (fx.Int64(fx.Uint32(old[0])) | (fx.Int64(old[1]) << 32)) + fx.Int64(1)
            for slot in range(14):
                value = fx.ptr_load(tls + slot)
                bo.buffer_store(fx.Vector.from_elements([fx.Int32(value), fx.Int32(value >> 32)], fx.Int32), rsrc(base), (at + slot) * 2, cache_modifier=CM_DEV)
            # Exporters must observe the original stream's completed event.
            for slot, value in ((14, generation), (15, fx.Int64(SIGNATURE | rows))):
                bo.buffer_store(fx.Vector.from_elements([fx.Int32(value), fx.Int32(value >> 32)], fx.Int32), rsrc(base), (at + slot) * 2, cache_modifier=CM_DEV)
