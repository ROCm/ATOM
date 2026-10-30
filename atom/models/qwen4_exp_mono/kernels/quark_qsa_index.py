# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA index query/raw publication and compressed BF16 cache preparation.

q1 forwards its own raw row. q4 first publishes ALL raw/position rows, then
compresses on the same stream. Logical positions, request tables and closing
compressed slots retain the Native causal and cache ownership contract.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(x, w, cos, sin, pos0, pos1, pos2, D: tl.constexpr):
    d = tl.arange(0, D)
    ss = tl.sum(x * x, 1) / D
    xn = (x * tl.rsqrt(ss + 1.0e-6)[:, None] * w).to(tl.bfloat16)
    f = d % 32
    axis = tl.where((f % 3 == 1) & (f < 33), 1, 0)
    axis = tl.where((f % 3 == 2) & (f < 30), 2, axis)
    pos = tl.where(axis == 0, pos0, tl.where(axis == 1, pos1, pos2))
    c = tl.load(cos + pos * 32 + f).to(tl.bfloat16).to(tl.float32)
    s = tl.load(sin + pos * 32 + f).to(tl.bfloat16).to(tl.float32)
    pair = tl.where(d < 32, d + 32, d - 32)
    p = tl.gather(xn, tl.broadcast_to(pair[None, :], xn.shape), 1).to(tl.float32)
    p = tl.where(d[None, :] < 32, -p, p)
    xc = (xn.to(tl.float32) * c[None, :]).to(tl.bfloat16).to(tl.float32)
    ys = (p * s[None, :]).to(tl.bfloat16).to(tl.float32)
    return tl.where(d[None, :] < 64, (xc + ys).to(tl.bfloat16), xn)


@triton.jit
def _compress(
    KW,
    COS,
    SIN,
    SLOT,
    CSLOT,
    TABLE,
    REQ,
    LOGICAL,
    RAW,
    PC,
    CC,
    row,
    raw,
    BS: tl.constexpr,
    TBS: tl.constexpr,
    PAGES: tl.constexpr,
    REQUESTS: tl.constexpr,
    FORWARD_CURRENT: tl.constexpr,
):
    d = tl.arange(0, 128)
    slot = tl.load(SLOT + row).to(tl.int64)
    valid = (slot >= 0) & (slot < PAGES * BS)
    cs = tl.load(CSLOT + row).to(tl.int64)
    req = tl.load(REQ + row)
    end = tl.load(LOGICAL + row)
    if (
        valid
        & (cs >= 0)
        & (cs < PAGES * (BS // 4))
        & (end >= 3)
        & (req >= 0)
        & (req < REQUESTS)
    ):
        accum = tl.full((128,), 0, tl.float32)
        for i in tl.static_range(3):
            lp = end - 3 + i
            page = tl.load(TABLE + req * TBS + lp // BS, lp // BS < TBS, -1)
            val = tl.load(
                RAW + (tl.maximum(page, 0).to(tl.int64) * BS + lp % BS) * 128 + d,
                (page >= 0) & (page < PAGES),
                0,
            ).to(tl.float32)
            accum = accum + val
        if FORWARD_CURRENT:
            last = raw.to(tl.float32)
        else:
            last = tl.load(RAW + slot * 128 + d).to(tl.float32)
        accum = accum + last
        pooled = (accum / 4).to(tl.bfloat16).to(tl.float32)
        first = end - 3
        page = tl.load(TABLE + req * TBS + first // BS, first // BS < TBS, -1).to(
            tl.int64
        )
        fp = (tl.maximum(page, 0) * BS + first % BS) * 3
        pv = (page >= 0) & (page < PAGES)
        f0 = tl.load(PC + fp, pv, 0)
        f1 = tl.load(PC + fp + 1, pv, 0)
        f2 = tl.load(PC + fp + 2, pv, 0)
        kw = tl.load(KW + d).to(tl.float32) + 1
        ck = _norm_rope(pooled[None, :], kw[None, :], COS, SIN, f0, f1, f2, 128)
        tl.store(CC + cs * 128 + d, tl.reshape(ck, (128,)))


@triton.jit
def _index_front(
    QK,
    QW,
    KW,
    COS,
    SIN,
    POS,
    SLOT,
    CSLOT,
    TABLE,
    REQ,
    LOGICAL,
    RAW,
    PC,
    CC,
    OQ,
    STRIDE: tl.constexpr,
    PS: tl.constexpr,
    PA: tl.constexpr,
    BS: tl.constexpr,
    TBS: tl.constexpr,
    PAGES: tl.constexpr,
    REQUESTS: tl.constexpr,
    VERIFY: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 128)
    h = tl.arange(0, 4)
    slot = tl.load(SLOT + row).to(tl.int64)
    p0 = tl.load(POS + row * PS)
    p1 = tl.load(POS + row * PS + PA)
    p2 = tl.load(POS + row * PS + 2 * PA)
    q = tl.load(QK + row * STRIDE + h[:, None] * 128 + d[None, :]).to(tl.float32)
    qw = tl.load(QW + d).to(tl.float32) + 1
    oq = _norm_rope(q, qw[None, :], COS, SIN, p0, p1, p2, 128)
    tl.store(OQ + row * 512 + h[:, None] * 128 + d[None, :], oq)
    raw = tl.load(QK + row * STRIDE + 512 + d)
    valid = (slot >= 0) & (slot < PAGES * BS)
    tl.store(RAW + slot * 128 + d, raw, valid)
    tl.store(PC + slot * 3, p0, valid)
    tl.store(PC + slot * 3 + 1, p1, valid)
    tl.store(PC + slot * 3 + 2, p2, valid)
    if not VERIFY:
        _compress(
            KW,
            COS,
            SIN,
            SLOT,
            CSLOT,
            TABLE,
            REQ,
            LOGICAL,
            RAW,
            PC,
            CC,
            row,
            raw,
            BS,
            TBS,
            PAGES,
            REQUESTS,
            True,
        )


@triton.jit
def _index_compress(
    KW,
    COS,
    SIN,
    SLOT,
    CSLOT,
    TABLE,
    REQ,
    LOGICAL,
    RAW,
    PC,
    CC,
    BS: tl.constexpr,
    TBS: tl.constexpr,
    PAGES: tl.constexpr,
    REQUESTS: tl.constexpr,
):
    _compress(
        KW,
        COS,
        SIN,
        SLOT,
        CSLOT,
        TABLE,
        REQ,
        LOGICAL,
        RAW,
        PC,
        CC,
        tl.program_id(0),
        tl.full((128,), 0, tl.float32),
        BS,
        TBS,
        PAGES,
        REQUESTS,
        False,
    )


def prepare(module, hidden, positions, meta, *, projected=None, query_length=1):
    assert query_length in (1, 4)
    rows = hidden.shape[0]
    qk = (
        module.indexer.index_qk_proj(hidden, otype=hidden.dtype)
        if projected is None
        else projected
    )
    iq = torch.empty(
        (hidden.shape[0], 4, 128), dtype=hidden.dtype, device=hidden.device
    )
    rope = module.indexer.rotary_emb
    _index_front[(hidden.shape[0],)](
        qk,
        module.indexer.q_layernorm.weight,
        module.indexer.k_layernorm.weight,
        rope.cos_cache,
        rope.sin_cache,
        positions,
        meta.slot_mapping,
        meta.compressed_slot_mapping,
        meta.block_tables,
        meta.token_to_req,
        meta.logical_positions,
        module.raw_key_cache,
        module.rope_position_cache,
        module.compressed_key_cache,
        iq,
        qk.stride(0),
        positions.stride(-1),
        positions.stride(0) if positions.ndim == 2 and positions.shape[0] == 3 else 0,
        module.raw_key_cache.shape[1],
        meta.block_tables.stride(0),
        module.raw_key_cache.shape[0],
        meta.block_tables.shape[0],
        query_length == 4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if query_length == 4:
        # Stream ordering publishes every q4 raw/position row before any
        # closing group reads another query's data. CSLOT owns the only write.
        _index_compress[(rows,)](
            module.indexer.k_layernorm.weight,
            rope.cos_cache,
            rope.sin_cache,
            meta.slot_mapping,
            meta.compressed_slot_mapping,
            meta.block_tables,
            meta.token_to_req,
            meta.logical_positions,
            module.raw_key_cache,
            module.rope_position_cache,
            module.compressed_key_cache,
            module.raw_key_cache.shape[1],
            meta.block_tables.stride(0),
            module.raw_key_cache.shape[0],
            meta.block_tables.shape[0],
            num_warps=4,
            enable_fp_fusion=False,
        )
    return iq
