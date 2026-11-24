# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A top-p row's draw in the LM head launch (``kernel``): the row's
``helpers`` CTAs (CTA t: helper t // rows of row t % rows, helper 0 the
row's lead), each over its slice of the row's logits (``top_p_logits``),
first by the row's largest logits (``top_p_candidates``), and where its
top-p set reaches past them, every rank alike, by the two-level scan
(``top_p_scan``). The lead puts the row's id into ``ids``; the pick leaves
the row alone."""

import flydsl.expr as fx
from flydsl.expr import gpu

from atom.mono.device.ops import LOG2E, hw_rcp, traced, uniform_f32
from atom.mono.lm_head import top_p_candidates, top_p_scan
from atom.mono.lm_head import top_p_logits as tl


@traced
def draw_slot(c):
    """CTA t < rows x helpers: its part of row t % rows's draw, where the row
    draws within its top-p (``top_p_logits.draws_top_p``)."""
    p = c["key"]
    r, j = c["t"] % p.rows, c["t"] // p.rows
    # the CTA's branch: one value for every thread
    t, top_p = (uniform_f32(v) for v in tl.row_sampling(c, r))
    if tl.draws_top_p(t, top_p, p.greedy_t):
        gpu.barrier()  # the rank reduction's LDS readers past it
        cr = fx.Float32(LOG2E) * hw_rcp(t)
        span = tl.slice_span(p, j)
        decided = top_p_candidates.draw(c, r, j, cr, top_p, span)
        if ~decided:
            top_p_scan.scan_draw(c, r, j, cr, top_p, span)
