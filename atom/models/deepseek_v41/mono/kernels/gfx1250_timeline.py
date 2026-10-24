# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Optional gfx1250 K2 phase intervals, buffered in private CTA LDS.

Each phase records first entry, last exit and accumulated elapsed ticks.
Repeated routed-down tasks therefore separate their compute and wait time.
Only the final flush writes global memory, as in gfx950's phase stamps.
All clocks use the platform steady 100 MHz counter, not shader cycles.
"""

import os

import flydsl.expr as fx
from flydsl.expr import const_expr

from atom.models.deepseek_v41.mono.kernels.gfx1250_k1 import _clock
from atom.mono.device.ops import traced

ENABLED = os.environ.get("ATOM_DSV41_GFX1250_K2_TIMELINE", "0") == "1"
STAGES = (
    "index", "score_ready", "score", "pv_wait", "pv", "woa_wait", "woa",
    "wob_wait", "wob", "wob_publish", "attn_tp_put", "reduce_local_wait",
    "attn_tp_sum", "slice", "norm_wait", "norm", "router_wait", "router",
    "route_wait", "route", "union_wait", "union", "shared_up_wait", "shared_up",
    "gates_wait", "gates", "ug_wait", "ug", "shared_down_wait", "shared_down",
    "routed_down_wait", "routed_down", "combine", "ffn_publish", "ffn_tp_sum",
)
# epoch, CTA entry, CTA exit, expert union count; then 3 words per phase.
HEADER, POINTS = 4, 4 + 3 * len(STAGES)
LDS_BYTES = (POINTS * 8 + 15) // 16 * 16


@traced
def begin(tls, tid, epoch):
    if const_expr(tls is not None):
        if tid < POINTS:
            fx.ptr_store(fx.Int64(0), tls + tid)
        fx.gpu.barrier()
        if tid == 0:
            fx.ptr_store(fx.Int64(epoch), tls)
            fx.ptr_store(_clock(), tls + 1)


@traced
def mark(tls, tid, stage, ending=False):
    if const_expr(tls is not None):
        if tid == 0:
            at = tls + HEADER + 3 * STAGES.index(stage)
            now = _clock()
            if const_expr(ending):
                before, total = fx.ptr_load(at + 1), fx.ptr_load(at + 2)
                fx.ptr_store(total + now - before, at + 2)
                fx.ptr_store(now, at + 1)
            else:
                if fx.ptr_load(at) == 0:
                    fx.ptr_store(now, at)
                fx.ptr_store(now, at + 1)


@traced
def union_count(tls, tid, count):
    if const_expr(tls is not None):
        if tid == 0:
            fx.ptr_store(fx.Int64(count), tls + 3)


@traced
def flush(tls, output, tid, bid):
    if const_expr(tls is not None):
        if tid == 0:
            fx.ptr_store(_clock(), tls + 2)
        fx.gpu.barrier()
        if tid < POINTS:
            fx.ptr_store(fx.ptr_load(tls + tid), output + bid * POINTS + tid)
