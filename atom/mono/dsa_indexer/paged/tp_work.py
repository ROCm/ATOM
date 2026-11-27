# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Request metadata and rank-local contiguous windows of the actual layer."""

import flydsl.expr as fx

from atom.mono.device.ops import (
    ld_i32,
    traced,
    uniform,
    wave_incl_scan,
    wave_max,
    wave_sum,
)
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged import tp_plan as tp
from atom.mono.dsa_indexer.selection_plan import SHORT_VERIFY_CTX


@traced
def row_work(c, row):
    request = ld_i32(c["args"]["ix_meta"], row * 4)
    last = ld_i32(c["args"]["ix_meta"], row * 4 + 2)
    raw = ld_i32(c["args"]["ibound"], row)
    request_bound = ld_i32(c["args"]["ibound"], fx.max(last, 0))
    mode = tp.work_kind(
        request_bound, c["ix_total"], c["ix_requests"], c["world"], c["ix_local"]
    )
    mode = (request >= 0).select(mode, fx.Int32(tp.LOCAL))
    first, count = tp.key_window(
        raw, request_bound, fx.max(request, 0), c["world"], c["rank"], mode
    )
    count = (request >= 0).select(count, fx.Int32(0))
    return mode, request % c["world"], first, count


@traced
def initialize(c):
    lane = c["lane"]
    row = fx.min(lane, c["S"] - 1)
    meta = c["args"]["ix_meta"]
    request = ld_i32(meta, row * 4)
    first = ld_i32(meta, row * 4 + 1)
    last = ld_i32(meta, row * 4 + 2)
    bound = c["args"]["ibound"]
    live = (lane < c["S"]) & (request >= 0)
    start = live & (lane == first)
    request_bound = ld_i32(bound, fx.max(last, 0))
    first_bound = ld_i32(bound, fx.max(first, 0))
    c["ix_total"] = uniform(wave_sum(start.select(request_bound, fx.Int32(0))))
    c["ix_requests"] = uniform(ld_i32(meta, 3)) & 0xFFFF
    c["ix_local"] = (
        uniform(wave_max((start & (first_bound > SHORT_VERIFY_CTX)).select(1, 0))) == 0
    )


@traced
def initialize_scores(c):
    initialize(c)
    lane = c["lane"]
    row = fx.min(lane, c["S"] - 1)
    live = (lane < c["S"]) & (ld_i32(c["args"]["ix_meta"], row * 4) >= 0)
    bound = c["args"]["ibound"]
    mode, owner, _offset, count = row_work(c, row)
    active = (
        live
        & ((mode != tp.OWNED) | (owner == c["rank"]))
        & (count > 0)
        & (ld_i32(bound, row) > ip.TOPK)
    )
    c["ix_active_rows"] = uniform(wave_sum(active.select(1, 0)))
    c["ix_active_prefix"] = wave_incl_scan(active.select(1, 0), lane) - 1
    c["ix_active"] = active
