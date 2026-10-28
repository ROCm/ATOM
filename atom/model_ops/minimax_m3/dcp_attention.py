# SPDX-License-Identifier: MIT
"""M3 block-interleaved DCP: owned KV, global index selection and LSE merge.

The initial compute layout is S=B=128 with DCP=TP and all KV heads per rank.
The same kernels cover ragged cached prefill, verify rows, and draft windows.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_value(payload, scales, offset, scale_offset, dim, valid):
    byte = tl.load(payload + offset, valid, other=0).to(tl.int32)
    code = (byte >> ((dim % 2) * 4)) & 15
    mag = code & 7
    value = tl.where(mag < 4, mag * 0.5, tl.where(mag < 6, mag - 2.0, mag * 2.0 - 8.0))
    value = tl.where(code >= 8, -value, value)
    scale = (
        tl.load(scales + scale_offset, valid, other=0)
        .to(tl.uint8)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    return value * scale


@triton.jit
def _index_scores(
    Q,
    K,
    KS,
    BT,
    Req,
    Length,
    Out,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    BTS: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCKS: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    FP4: tl.constexpr,
    SCALE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_TILE: tl.constexpr = 1,
):
    row, first = tl.program_id(0), tl.program_id(1) * BLOCK_TILE
    req = tl.load(Req + row)
    length = tl.load(Length + row)
    heads, d, t = tl.arange(0, N), tl.arange(0, 128), tl.arange(0, 128)
    q = tl.load(
        Q + row * QS0 + heads[None, :] * QS1 + d[:, None],
        heads[None, :] < HEADS,
        other=0,
    )
    cols = first + tl.arange(0, BLOCK_TILE)
    tl.store(
        Out + (heads[:, None] * ROWS + row) * BLOCKS + cols[None, :],
        -float("inf"),
        (heads[:, None] < HEADS) & (cols[None, :] < BLOCKS),
    )
    live = tl.cdiv(tl.cdiv(tl.maximum(length, 0), 128), WORLD)
    for block in range(first, tl.minimum(first + BLOCK_TILE, tl.minimum(live, BLOCKS))):
        score = tl.full((N,), -float("inf"), tl.float32)
        global_block = block * WORLD + RANK
        if global_block * 128 < length:
            page = tl.load(BT + req * BTS + block).to(tl.int64)
            if FP4:
                # Match the native FP4 scorer's even/odd FP16 dot products.
                e = tl.arange(0, 64) * 2
                base = page * 128 * 64 + t[:, None] * 64 + e[None, :] // 2
                scale = page * 128 * 8 + t[:, None] * 8 + e[None, :] // 16
                ke = _fp4_value(K, KS, base, scale, e[None, :], t[:, None] >= 0).to(
                    tl.float16
                )
                ko = _fp4_value(K, KS, base, scale, e[None, :] + 1, t[:, None] >= 0).to(
                    tl.float16
                )
                qe = tl.load(
                    Q + row * QS0 + heads[None, :] * QS1 + e[:, None],
                    heads[None, :] < HEADS,
                    other=0,
                ).to(tl.float16)
                qo = tl.load(
                    Q + row * QS0 + heads[None, :] * QS1 + e[:, None] + 1,
                    heads[None, :] < HEADS,
                    other=0,
                ).to(tl.float16)
                dots = tl.dot(ke, qe) + tl.dot(ko, qo)
            else:
                k = tl.load(K + page * 128 * 128 + t[:, None] * 128 + d[None, :]).to(
                    q.dtype
                )
                dots = tl.dot(k, q)
            dots = tl.where(
                global_block * 128 + t[:, None] < length, dots * SCALE, -float("inf")
            )
            score = tl.max(dots, 0)
        tl.store(Out + (heads * ROWS + row) * BLOCKS + block, score, heads < HEADS)


def index_topk(
    q,
    k,
    ks,
    table,
    req,
    lengths,
    group,
    rank,
    world,
    topk,
    init,
    local,
    scale,
    max_seq_len=None,
):
    """Score owned blocks, restore global order, use the native packed selector."""
    from atom.model_ops.dcp_ops import dcp_all_gather
    from atom.model_ops.minimax_m3.index_topk import _launch_select, _require_packable

    heads, rows, blocks = q.shape[1], q.shape[0], table.shape[1]
    if max_seq_len is not None:
        blocks = min(blocks, max(1, triton.cdiv(max_seq_len, 128 * world)))
    _require_packable(blocks * world)
    scores = torch.empty((heads, rows, blocks), device=q.device, dtype=torch.float32)
    _index_scores[(rows, triton.cdiv(blocks, 256))](
        q,
        k,
        ks if ks is not None else k,
        table,
        req,
        lengths,
        scores,
        q.stride(0),
        q.stride(1),
        table.stride(0),
        heads,
        rows,
        blocks,
        world,
        rank,
        ks is not None,
        scale * 1.4426950409,
        max(16, triton.next_power_of_2(heads)),
        BLOCK_TILE=256,
    )
    scores = (
        dcp_all_gather(group, scores, -1)
        .view(heads, rows, world, blocks)
        .transpose(2, 3)
        .reshape(heads, rows, blocks * world)
    )
    indices = torch.empty((heads, rows, topk), device=q.device, dtype=torch.int32)
    # Each query is a one-row request with its own global causal length.
    _launch_select(
        scores,
        indices,
        lengths,
        lengths,
        rows,
        topk,
        init,
        local,
        table,
        (indices, indices, 0, 0),
        heads,
        1,
        1,
        max(1024, triton.next_power_of_2(topk)),
        4,
        False,
        8,
    )
    return indices


@triton.jit
def _attention(
    Q,
    K,
    V,
    KS,
    VS,
    BT,
    Req,
    Length,
    Start,
    Topk,
    Out,
    Lse,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    BTS: tl.constexpr,
    ROWS: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    PAGE: tl.constexpr,
    PACK: tl.constexpr,
    FP4: tl.constexpr,
    SCALES: tl.constexpr,
    SPARSE: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    G: tl.constexpr,
):
    row, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req, length, start = tl.load(Req + row), tl.load(Length + row), tl.load(Start + row)
    g, d, t = tl.arange(0, G), tl.arange(0, 128), tl.arange(0, 128)
    qh = head * (HQ // HK) + g
    q = tl.load(
        Q + row * QS0 + qh[:, None] * QS1 + d[None, :], g[:, None] < HQ // HK, other=0
    )
    m = tl.full((G,), -float("inf"), tl.float32)
    z = tl.full((G,), 0.0, tl.float32)
    acc = tl.full((G, 128), 0.0, tl.float32)
    if SPARSE:
        count = TOPK
    else:
        count = tl.cdiv(tl.cdiv(tl.maximum(length, 0), 128), WORLD)
    for i in range(split, count, SPLITS):
        if SPARSE:
            block = tl.load(Topk + (head * ROWS + row) * TOPK + i)
        else:
            block = i * WORLD + RANK
        if (
            (block >= 0)
            & (block % WORLD == RANK)
            & (block * 128 < length)
            & ((block + 1) * 128 > start)
        ):
            page = tl.load(BT + req * BTS + block // WORLD).to(tl.int64)
            valid = (block * 128 + t < length) & (block * 128 + t >= start)
            if FP4:
                off = ((page * HK + head) * 128 + t[:, None]) * 64 + d[None, :] // 2
                sc = ((page * HK + head) * 128 + t[:, None]) * 8 + d[None, :] // 16
                k = _fp4_value(K, KS, off, sc, d[None, :], valid[:, None]).to(q.dtype)
                v = _fp4_value(V, VS, off, sc, d[None, :], valid[:, None]).to(q.dtype)
            else:
                phys = page * (128 // PAGE) + t // PAGE
                tok = t % PAGE
                base = (phys[:, None] * HK + head) * PAGE * 128
                ko = (
                    base
                    + ((d[None, :] // PACK) * PAGE + tok[:, None]) * PACK
                    + d[None, :] % PACK
                )
                vo = (
                    base
                    + ((tok[:, None] // PACK) * 128 + d[None, :]) * PACK
                    + tok[:, None] % PACK
                )
                k = tl.load(K + ko, valid[:, None], other=0.0).to(tl.float32)
                v = tl.load(V + vo, valid[:, None], other=0.0).to(tl.float32)
                if SCALES:
                    so = (phys * HK + head) * PAGE + tok
                    k *= tl.load(KS + so, valid, other=0)[:, None]
                    v *= tl.load(VS + so, valid, other=0)[:, None]
                k, v = k.to(q.dtype), v.to(q.dtype)
            scores = tl.dot(q, tl.trans(k)) * SCALE
            scores = tl.where(valid[None, :], scores, -float("inf"))
            new_m = tl.maximum(m, tl.max(scores, 1))
            safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            p = tl.exp(scores - safe_m[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
            z = z * alpha + tl.sum(p, 1)
            m = new_m
    result = acc / tl.where(z > 0, z, 1.0)[:, None]
    tl.store(
        Out + ((split * ROWS + row) * HQ + qh[:, None]) * 128 + d[None, :],
        result,
        g[:, None] < HQ // HK,
    )
    tl.store(
        Lse + (split * ROWS + row) * HQ + qh,
        tl.where(z > 0, m + tl.log(z), -float("inf")),
        g < HQ // HK,
    )


def merge_partials(outputs, lses):
    """Stable softmax merge, including entirely empty shards (0, -inf)."""
    maximum = lses.amax(0)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = (lses - maximum).exp()
    denominator = weights.sum(0)
    out = (outputs.float() * weights.unsqueeze(-1)).sum(0) / denominator.clamp_min(
        1e-30
    ).unsqueeze(-1)
    lse = torch.where(
        denominator > 0, maximum + denominator.clamp_min(1e-30).log(), -torch.inf
    )
    return out, lse


def local_attention(
    q, cache, table, req, lengths, starts, *, world, rank, scale, topk=None, nvfp4=False
):
    """Return one rank's output and natural-log LSE, without collectives."""
    k, v, ks, vs = cache
    rows, hq, _ = q.shape
    hk = k.shape[1]
    parts = 4
    output = torch.empty((parts, rows, hq, 128), dtype=torch.float32, device=q.device)
    lse = torch.empty((parts, rows, hq), dtype=torch.float32, device=q.device)
    _attention[(rows, hk, parts)](
        q,
        k,
        v,
        ks if ks is not None else k,
        vs if vs is not None else v,
        table,
        req,
        lengths,
        starts,
        topk if topk is not None else table,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        table.stride(0),
        rows,
        hq,
        hk,
        world,
        rank,
        128 if nvfp4 else k.shape[3],
        1 if nvfp4 else k.shape[4],
        nvfp4,
        ks is not None,
        topk is not None,
        topk.shape[-1] if topk is not None else 0,
        parts,
        scale,
        max(16, triton.next_power_of_2(hq // hk)),
    )
    return merge_partials(output, lse)


def attention(
    q,
    cache,
    table,
    req,
    lengths,
    starts,
    *,
    world,
    rank,
    group,
    scale,
    topk=None,
    nvfp4=False,
):
    """Local attention followed by an FP32 distributed softmax reduction."""
    from atom.model_ops.dcp_ops import dcp_all_gather

    out, local_lse = local_attention(
        q,
        cache,
        table,
        req,
        lengths,
        starts,
        world=world,
        rank=rank,
        scale=scale,
        topk=topk,
        nvfp4=nvfp4,
    )
    rows, hq = q.shape[:2]
    if world == 1:
        return out.to(q.dtype)
    all_lse = dcp_all_gather(group, local_lse, 0).view(world, rows, hq)
    maximum = all_lse.amax(0)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    denominator = (all_lse - maximum).exp().sum(0).clamp_min(1e-30)
    weighted = out * ((local_lse - maximum).exp() / denominator).unsqueeze(-1)
    return group.all_reduce(weighted).to(q.dtype)
