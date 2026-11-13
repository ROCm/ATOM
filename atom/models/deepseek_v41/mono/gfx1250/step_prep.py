# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A served step's per-row preparation in one launch (``StepInputs._device_inputs``): the copies of the step's
positions / ids / batch ids / slots / window rows into the step tables, the draft context's RoPE rows (gathered by
position), the page tables (a fixed ``BCOLS`` columns a request) and the scratch the launch counts or maxes into
(zeroed / -inf). One Triton launch instead of ~20 small copies and fills (their GPU time and the gaps between them
were ~40 us of the step-to-step gap)."""

import triton
import triton.language as tl


@triton.jit
def _st(ptr, offs, v, mask):
    tl.store(ptr + offs, v.to(ptr.dtype.element_ty), mask=mask)


@triton.jit
def _prep_kernel(
    pos,
    ids,
    batch_ids,
    slots_src,
    kvw,
    cos_tab,
    sin_tab,
    bt,
    rpos,
    first_pos,
    tids,
    dids,
    dgeom_slots,
    ctx_kvw,
    ctx_cos,
    ctx_sin,
    ibat,
    btab,
    slots_dst,
    cnt,
    ss,
    mbest,
    n,
    req,
    nbt,
    cc,
    bt_stride,
    n_cnt,
    n_ss,
    n_mbest,
    ROWS: tl.constexpr,
    BCOLS: tl.constexpr,
    IDX: tl.constexpr,
    COMP: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    if pid == 0:  # the rows
        i = tl.arange(0, BLOCK)
        live = i < n
        p = tl.load(pos + i, mask=live, other=0)
        _st(rpos, i, p, live)
        _st(first_pos, i // ROWS, p, live & (i % ROWS == 0))
        _st(tids, i, tl.load(ids + i, mask=live, other=0), live)
        nd = req * (ROWS - 1)
        dm = i < nd
        _st(
            dids,
            i,
            tl.load(
                ids + (i // (ROWS - 1)) * ROWS + 1 + i % (ROWS - 1), mask=dm, other=0
            ),
            dm,
        )
        rm = i < req
        _st(dgeom_slots, i, tl.load(slots_src + i, mask=rm, other=0), rm)
        _st(ctx_kvw, i, tl.load(kvw + i, mask=live, other=0), live)
        if IDX:
            _st(ibat, i, tl.load(batch_ids + i, mask=live, other=0), live)
            if COMP:
                bm = i < nbt
                _st(slots_dst, i, tl.load(slots_src + i, mask=bm, other=0), bm)
    if pid == 1:  # the draft context's RoPE rows: table row = the row's position
        k = tl.arange(0, BLOCK * 32)
        r = k // 32
        live = r < n
        p = tl.load(pos + r, mask=live, other=0)
        src = p.to(tl.int64) * 32 + k % 32
        _st(ctx_cos, k, tl.load(cos_tab + src, mask=live, other=0.0), live)
        _st(ctx_sin, k, tl.load(sin_tab + src, mask=live, other=0.0), live)
    c = tl.arange(0, CHUNK)
    for s in range(pid * CHUNK, n_cnt, npid * CHUNK):
        _st(cnt, s + c, tl.zeros([CHUNK], tl.int32), s + c < n_cnt)
    for s in range(pid * CHUNK, n_ss, npid * CHUNK):
        _st(ss, s + c, tl.zeros([CHUNK], tl.int32), s + c < n_ss)
    for s in range(pid * CHUNK, n_mbest, npid * CHUNK):
        _st(
            mbest,
            s + c,
            tl.full([CHUNK], -9223372036854775808, tl.int64),
            s + c < n_mbest,
        )
    if IDX:  # the page tables, cc columns of each of the nbt requests
        tot = nbt * cc
        for s in range(pid * CHUNK, tot, npid * CHUNK):
            e = s + c
            m = e < tot
            rr = e // cc
            cl = e % cc
            _st(
                btab,
                rr * BCOLS + cl,
                tl.load(bt + rr.to(tl.int64) * bt_stride + cl, mask=m, other=0),
                m,
            )


def prep_step(
    *,
    positions,
    input_ids,
    batch_ids,
    slots,
    kvw,
    rope,
    block_tables,
    out,
    rows,
    req,
    rows_per_request,
    bcols,
    idx,
    comp,
    cnt,
    ss,
    mbest,
    programs=16,
):
    """``out``: the step tables' destinations (``rpos first_pos tids dids dgeom_slots ctx_kvw ctx_cos ctx_sin ibat
    btab slots``); ``cnt`` / ``ss``: int32 views zeroed; ``mbest``: an int64 view set to INT64_MIN. On the current
    stream."""
    bt = block_tables
    nbt, cc = (bt.shape[0], min(bt.shape[1], bcols)) if idx else (0, 0)
    _prep_kernel[(programs,)](
        positions,
        input_ids,
        batch_ids,
        slots,
        kvw,
        rope[0],
        rope[1],
        bt,
        out["rpos"],
        out["first_pos"],
        out["tids"],
        out["dids"],
        out["dgeom_slots"],
        out["ctx_kvw"],
        out["ctx_cos"],
        out["ctx_sin"],
        out["ibat"],
        out["btab"],
        out["slots"],
        cnt,
        ss,
        mbest,
        rows,
        req,
        nbt,
        cc,
        bt.stride(0) if idx else 0,
        cnt.numel(),
        ss.numel(),
        mbest.numel(),
        ROWS=rows_per_request,
        BCOLS=bcols,
        IDX=idx,
        COMP=comp,
        BLOCK=64,
        CHUNK=1024,
    )
