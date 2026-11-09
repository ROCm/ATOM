# SPDX-License-Identifier: MIT
"""Per-CTA stage stamps of one gfx1250 K2 layer (debug only).

``ATOM_GFX1250_K2_STAMP_LAYER=<layer>`` builds the K2s with ``llvm.readsteadycounter``
(100 MHz; gfx1250 has no s_memrealtime) stamps at their stage ends, kept in LDS and flushed by the chosen
layer only into int64 [BLOCKS][POINTS] past the runner's mhc_sq/state
workspace. ``tl`` carries that workspace on gfx1250, so the base and the layer
tag go through late_meta words, not live kernel arguments.
"""
import os

import flydsl.expr as fx
from flydsl.expr import const_expr

from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

from atom.mono.device.ops import traced
from atom.mono.plan.execution import BLOCKS

STAMP_LAYER = int(os.environ.get("ATOM_GFX1250_K2_STAMP_LAYER", "-1"))
ON = STAMP_LAYER >= 0
POINTS = 25
STAMP_WORDS = BLOCKS * POINTS * 2
BASE_META = 264
LAYER_META = 266
# point k: the end of STAGES[k - 1]; point 0 the kernel start
STAGES = ("setup", "indexer", "score", "pv", "wo", "seam+front", "gates",
          "ug", "down", "candidates")
# 48-row seam sub-points (dspark_mhtjstest_split), past the stages
# (half h's reduce: 11 + 2 h, its tile: 12 + 2 h)
SEAM_POINTS = {"reduce0": 11, "tile0": 12, "reduce1": 13, "tile1": 14, "rstd": 15, "seam": 16}
# 48-row down: its owner columns (W2 + pushes to the peers) done
DOWN_OWNED = 17
# 48-row down phases: shared GEMV, routed expert walk, then the combine
DOWN_SHARED, DOWN_ROUTED, DOWN_FINISH = 22, 23, 24
# wo_a done (its pushes), wo_b's input rows polled
WOA_DONE, WOB_INPUT = 18, 19
# a wo_a task's rows in LDS, its matrices done
WOA_ROWS_IN, WOA_MMA = 20, 21


def offset_bytes(max_rows):
    """The stamp region's byte offset in the runner's mhc_sq/state workspace."""
    from atom.models.deepseek_v41.mono.gfx1250.kernels.p1a_attention import P1A_STATE_WORDS
    return (max_rows * 160 * 2 + P1A_STATE_WORDS) * 4


@traced
def stage_meta(late_meta, tl, offset_bytes, layer):
    """Thread 0: the stamp base (``tl`` + ``offset_bytes``) and the layer tag."""
    if const_expr(ON):
        base = tl + fx.Int64(offset_bytes)
        fx.ptr_store(fx.Int32(base), late_meta + BASE_META)
        fx.ptr_store(fx.Int32(base >> 32), late_meta + BASE_META + 1)
        fx.ptr_store(layer, late_meta + LAYER_META)


def _clock():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))


@traced
def begin(tls, tid):
    if const_expr(ON):  # noqa: SIM102
        if tid < POINTS:
            fx.ptr_store(fx.Int64(0), tls + tid)
    mark(tls, tid, 0)


def mark_any(tls, k):
    """Point k, every thread (no branch: usable inside another traced ``if``)."""
    if const_expr(ON and tls is not None):
        fx.ptr_store(_clock(), tls + k)


@traced
def mark(tls, tid, k):
    """Point k by thread 0; a no-op in a kernel without stamps (``tls`` None)."""
    if const_expr(ON and tls is not None):  # noqa: SIM102
        if tid == 0:
            fx.ptr_store(_clock(), tls + k)


@traced
def flush(tls, c, tid, bid, tag):
    """The chosen layer's CTA record -> the stamp region."""
    if const_expr(ON):
        from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_ct2_index_attention import (
            late_pointer, scalar_word)
        layer = scalar_word(fx.ptr_load(c["late_meta"] + LAYER_META))
        if layer == fx.Int32(tag):
            if tid < POINTS:
                base = late_pointer(c, BASE_META)
                fx.generic_store(
                    fx.inttoptr(
                        fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                        base + fx.Int64((bid * POINTS + tid) * 8),
                    ),
                    fx.ptr_load(tls + tid),
                )
