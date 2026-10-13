# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next MoE for a few decode rows over FP8 blockscale experts.

At 12-48 rows each rank sees only a handful of local experts. The stock path
(two sort kernels, an activation quantization, the 1-stage asm MoE) gives each
expert a few workgroups that walk the whole hidden dimension, so a single
local expert already costs ~25 us. Here the rows of an expert are cut into
blocks of 16 and every (block, N tile) is a work item:

* `plan`: one program per local expert lists its (token, slot) pairs and
  blocks; pairs are packed by expert, so no sort is needed;
* `up`: per (block, N tile), the gate and up dots with the hidden dimension
  split across the warps of the program, summed in split order, then
  `silu(gate) * up` in BF16;
* `down`: per (block, N tile), the down projection into an FP32 row per pair;
* `combine`: one program per (token, N tile) sums its pairs with the routing
  weights in slot order.

Every sum has a fixed order, so the result is deterministic. Activations stay
BF16 (AITER quantizes them to FP8 per 1x128 group), which is closer to the
FP32 reference than AITER; the FP8 weights are widened in registers. Weights
stay in AITER's `shuffle_weight((16, 16))` layout and `[E, N // 128, K // 128]`
block scales.
"""

import functools

import torch
import triton
import triton.language as tl

# The side of a scale block and of the shuffled layout's row groups times 8;
# N tiles and K steps stay inside one scale block.
_SCALE_BLOCK = 128
MAX_TOKENS = 64
_BUCKET_CAP = tl.constexpr(MAX_TOKENS)
BM = 16
# Slots per token the router may file: 10 routed, shared, sentinel.
_MAX_SLOTS = 16
_UP_BN = 16
_UP_BK = 64
_UP_SPLIT = 4
_DOWN_BN = 32
_DOWN_BK = 32
_DOWN_SPLIT = 4
_PROGRAMS = 1024


@triton.jit
def _plan_kernel(
    ids_ptr,
    expert_map_ptr,
    pairs_ptr,
    blk_expert_ptr,
    blk_start_ptr,
    blk_count_ptr,
    n_blocks_ptr,
    NUM_PAIRS,
    P: tl.constexpr,
    BINS: tl.constexpr,
    BM: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    """Program `e` lists the pairs routed to local expert `e` and its blocks.

    Pairs are packed by expert in id order and each expert's rows are cut
    into blocks of BM; an expert's pairs start after all pairs of lower
    experts and its blocks after their blocks. No sort.
    """
    e = tl.program_id(0)
    p = tl.arange(0, P)
    valid = p < NUM_PAIRS
    gid = tl.load(ids_ptr + p, mask=valid, other=0)
    loc = tl.load(expert_map_ptr + gid, mask=valid, other=-1)
    mine = (loc == e).to(tl.int32)
    count = tl.sum(mine, 0)
    # Comparisons on a histogram result miscompile (Triton 3.7, gfx950), so
    # its bins are only combined arithmetically or masked by bin index.
    if e == 0:
        hist = tl.histogram(tl.where(loc >= 0, loc, BINS - 1), BINS)
        hist = tl.where(tl.arange(0, BINS) < BINS - 1, hist, 0)
        tl.store(n_blocks_ptr, tl.sum((hist + BM - 1) // BM, 0))
    if count > 0:
        hist = tl.histogram(tl.where(loc >= 0, loc, BINS - 1), BINS)
        below = tl.where(tl.arange(0, BINS) < e, hist, 0)
        start = tl.sum(below, 0)
        first = tl.sum((below + BM - 1) // BM, 0)
        tl.store(pairs_ptr + start + tl.cumsum(mine, 0) - 1, p, mask=mine == 1)
        b = tl.arange(0, MAX_BLOCKS_PER_EXPERT)
        bmask = b * BM < count
        tl.store(blk_expert_ptr + first + b, e, mask=bmask)
        tl.store(blk_start_ptr + first + b, start + b * BM, mask=bmask)
        tl.store(blk_count_ptr + first + b, tl.minimum(count - b * BM, BM), mask=bmask)


@triton.jit
def _block_rows(
    pairs_ptr, blk_expert_ptr, blk_start_ptr, blk_count_ptr, blk, BM: tl.constexpr
):
    e = tl.load(blk_expert_ptr + blk).to(tl.int64)
    start = tl.load(blk_start_ptr + blk)
    rows = tl.arange(0, BM)
    rmask = rows < tl.load(blk_count_ptr + blk)
    pair = tl.load(pairs_ptr + start + rows, mask=rmask, other=0)
    return e, pair, rmask


@triton.jit
def _shuffled_split_tile(w_ptr, rows, k0, K: tl.constexpr, BK: tl.constexpr):
    """[S, rows, BK] FP8 tiles of a shuffled weight at K offsets `k0 [S]`,
    widened to BF16."""
    kc = k0[:, None] // 16 + tl.arange(0, BK // 16)[None, :]
    col = (kc // 2) * 512 + (kc % 2) * 256
    row = (rows // 16) * (16 * K) + (rows % 16) * 16
    off = (
        row[None, :, None, None]
        + col[:, None, :, None]
        + tl.arange(0, 16)[None, None, None, :]
    )
    w = tl.load(w_ptr + off)
    return tl.reshape(w, (k0.shape[0], rows.shape[0], BK)).to(tl.bfloat16)


@triton.jit
def _split_dot(w_ptr, s_ptr, a, n, k0, K: tl.constexpr, BK: tl.constexpr):
    """`[S, n, BM]` partial products of weight rows `n` with `a [S, BM, BK]`
    at K offsets `k0 [S]`, block scales `s_ptr[k0 // 128]` applied."""
    w = _shuffled_split_tile(w_ptr, n, k0, K, BK)
    return (
        tl.dot(w, tl.permute(a, (0, 2, 1))) * tl.load(s_ptr + k0 // 128)[:, None, None]
    )


@triton.jit
def _up_kernel(
    x_ptr,
    pairs_ptr,
    blk_expert_ptr,
    blk_start_ptr,
    blk_count_ptr,
    n_blocks_ptr,
    w13_ptr,
    s13_ptr,
    h_ptr,
    stride_x,
    SLOTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """`h = silu(x @ gate.T) * (x @ up.T)` per (block, N tile) work item.

    One tile walking all of the hidden dimension is latency bound, so the
    hidden dimension is split SPLIT ways across the warps of a program (a
    batched dot) and the partials are summed in split order.
    """
    TILES: tl.constexpr = INTER // BN
    SPAN: tl.constexpr = HIDDEN // SPLIT
    sp = tl.arange(0, SPLIT)
    items = tl.load(n_blocks_ptr) * TILES
    for w in range(tl.program_id(0), items, tl.num_programs(0)):
        e, pair, rmask = _block_rows(
            pairs_ptr, blk_expert_ptr, blk_start_ptr, blk_count_ptr, w // TILES, BM
        )
        tok = (pair // SLOTS).to(tl.int64)
        j0 = w % TILES * BN
        j = j0 + tl.arange(0, BN)
        w_gate = w13_ptr + e * (2 * INTER * HIDDEN)
        w_up = w_gate + INTER * HIDDEN
        s_gate = (
            s13_ptr
            + e * (2 * INTER // 128) * (HIDDEN // 128)
            + (j0 // 128) * (HIDDEN // 128)
        )
        s_up = s_gate + (INTER // 128) * (HIDDEN // 128)
        acc_g = tl.zeros((SPLIT, BN, BM), dtype=tl.float32)
        acc_u = tl.zeros((SPLIT, BN, BM), dtype=tl.float32)
        for kk in tl.static_range(0, SPAN, BK):
            k0 = sp * SPAN + kk
            x = tl.load(
                x_ptr
                + tok[None, :, None] * stride_x
                + k0[:, None, None]
                + tl.arange(0, BK)[None, None, :],
                mask=rmask[None, :, None],
                other=0.0,
            )
            acc_g += _split_dot(w_gate, s_gate, x, j, k0, HIDDEN, BK)
            acc_u += _split_dot(w_up, s_up, x, j, k0, HIDDEN, BK)
        g = tl.trans(tl.sum(acc_g, 0))
        u = tl.trans(tl.sum(acc_u, 0))
        tl.store(
            h_ptr + pair[:, None].to(tl.int64) * INTER + j[None, :],
            (g * tl.sigmoid(g) * u).to(h_ptr.dtype.element_ty),
            mask=rmask[:, None],
        )


@triton.jit
def _down_kernel(
    h_ptr,
    pairs_ptr,
    blk_expert_ptr,
    blk_start_ptr,
    blk_count_ptr,
    n_blocks_ptr,
    w2_ptr,
    s2_ptr,
    y_ptr,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Down projection per pair into `y [pairs, H]`, FP32, by pair id."""
    TILES: tl.constexpr = HIDDEN // BN
    SPAN: tl.constexpr = INTER // SPLIT
    sp = tl.arange(0, SPLIT)
    items = tl.load(n_blocks_ptr) * TILES
    for w in range(tl.program_id(0), items, tl.num_programs(0)):
        e, pair, rmask = _block_rows(
            pairs_ptr, blk_expert_ptr, blk_start_ptr, blk_count_ptr, w // TILES, BM
        )
        n0 = w % TILES * BN
        n = n0 + tl.arange(0, BN)
        w_base = w2_ptr + e * (HIDDEN * INTER)
        s_base = (
            s2_ptr + e * (HIDDEN // 128) * (INTER // 128) + (n0 // 128) * (INTER // 128)
        )
        acc = tl.zeros((SPLIT, BN, BM), dtype=tl.float32)
        for kk in tl.static_range(0, SPAN, BK):
            k0 = sp * SPAN + kk
            h = tl.load(
                h_ptr
                + pair[None, :, None].to(tl.int64) * INTER
                + k0[:, None, None]
                + tl.arange(0, BK)[None, None, :],
                mask=rmask[None, :, None],
                other=0.0,
            )
            acc += _split_dot(w_base, s_base, h, n, k0, INTER, BK)
        tl.store(
            y_ptr + pair[:, None].to(tl.int64) * HIDDEN + n[None, :],
            tl.trans(tl.sum(acc, 0)),
            mask=rmask[:, None],
        )


@triton.jit
def _combine_kernel(
    y_ptr,
    ids_ptr,
    weights_ptr,
    expert_map_ptr,
    out_ptr,
    SLOTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BN: tl.constexpr,
):
    t = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for s in tl.static_range(SLOTS):
        pair = t * SLOTS + s
        local = tl.load(expert_map_ptr + tl.load(ids_ptr + pair)) >= 0
        if local:
            w = tl.load(weights_ptr + pair)
            acc += w * tl.load(y_ptr + pair.to(tl.int64) * HIDDEN + n)
    tl.store(out_ptr + t * HIDDEN + n, acc.to(out_ptr.dtype.element_ty))


def _split_fits(k: int, split: int, bk: int) -> bool:
    """Each split's K span is whole BK chunks and no chunk straddles a scale
    block (or a 16-wide shuffle group)."""
    return (
        k % split == 0
        and (k // split) % bk == 0
        and _SCALE_BLOCK % bk == 0
        and bk % 16 == 0
    )


def supports(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, s13, s2) -> bool:
    """Shapes and formats these kernels are written for."""
    if s13 is None or s2 is None:
        return False
    experts, two_inter, hidden = w13.shape
    inter = two_inter // 2
    return (
        x.dtype == torch.bfloat16
        and w13.dtype == w2.dtype
        and w13.element_size() == 1
        and w2.shape == (experts, hidden, inter)
        and _split_fits(hidden, _UP_SPLIT, _UP_BK)
        and _split_fits(inter, _DOWN_SPLIT, _DOWN_BK)
        and s13.numel()
        == experts * (two_inter // _SCALE_BLOCK) * (hidden // _SCALE_BLOCK)
        and s2.numel() == experts * (hidden // _SCALE_BLOCK) * (inter // _SCALE_BLOCK)
    )


def moe_small_batch(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    buckets: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """`sum_k w_k * expert_k(x)` over this rank's experts, one row per token.

    `topk_ids [T, S]` hold global expert ids; `expert_map` maps them to local
    experts, -1 for experts owned elsewhere. `w13 [E, 2I, H]` (gate rows then
    up rows) and `w2 [E, H, I]` are FP8 in the (16, 16) shuffled layout with
    `[E, 2I / 128, H / 128]` and `[E, H / 128, I / 128]` block scales.

    `buckets` from `route_buckets`, filled by the router, replace the plan
    launch.
    """
    tokens, slots = topk_ids.shape
    experts, two_inter, hidden = w13.shape
    inter = two_inter // 2
    num_pairs = tokens * slots
    dev = x.device
    # An expert takes each token at most once.
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
    if buckets is None:
        max_blocks = min(experts, num_pairs) + num_pairs // BM
        meta = torch.empty(
            num_pairs + 3 * max_blocks + 1, dtype=torch.int32, device=dev
        )
        pairs = meta[:num_pairs]
        blk_expert, blk_start, blk_count = meta[num_pairs:-1].view(3, max_blocks)
        n_blocks = meta[-1:]
        _plan_kernel[(experts,)](
            topk_ids,
            expert_map,
            pairs,
            blk_expert,
            blk_start,
            blk_count,
            n_blocks,
            num_pairs,
            P=triton.next_power_of_2(num_pairs),
            BINS=triton.next_power_of_2(experts + 1),
            BM=BM,
            MAX_BLOCKS_PER_EXPERT=triton.next_power_of_2(triton.cdiv(tokens, BM)),
            num_warps=4,
        )
        blocks = (pairs, blk_expert, blk_start, blk_count, n_blocks)
    else:
        assert tokens <= MAX_TOKENS
        blocks = buckets[1:]
    h = torch.empty((num_pairs, inter), dtype=x.dtype, device=dev)
    _up_kernel[(_PROGRAMS,)](
        x,
        *blocks,
        w13,
        s13,
        h,
        x.stride(0),
        SLOTS=slots,
        HIDDEN=hidden,
        INTER=inter,
        BM=BM,
        BN=_UP_BN,
        BK=_UP_BK,
        SPLIT=_UP_SPLIT,
        num_warps=4,
        num_stages=1,
    )
    y = torch.empty((num_pairs, hidden), dtype=torch.float32, device=dev)
    _down_kernel[(_PROGRAMS,)](
        h,
        *blocks,
        w2,
        s2,
        y,
        HIDDEN=hidden,
        INTER=inter,
        BM=BM,
        BN=_DOWN_BN,
        BK=_DOWN_BK,
        SPLIT=_DOWN_SPLIT,
        num_warps=4,
        num_stages=1,
    )
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=dev)
    bn3 = 512
    _combine_kernel[(tokens, hidden // bn3)](
        y,
        topk_ids,
        topk_weights,
        expert_map,
        out,
        SLOTS=slots,
        HIDDEN=hidden,
        BN=bn3,
        num_warps=4,
    )
    return out


@functools.cache
def route_buckets(device: torch.device, experts: int) -> tuple[torch.Tensor, ...]:
    """The router's per-expert pair buckets and block list, shared by every
    layer: `(route, pairs, blk_expert, blk_start, blk_count, n_blocks)`.

    `route` starts with the zeroed per-expert counts and the ticket, which the
    router's last program leaves zero again; `pairs` is `[experts,
    MAX_TOKENS]`, expert `e`'s pairs from `e * MAX_TOKENS`.
    """
    bins = triton.next_power_of_2(experts)
    max_blocks = experts + MAX_TOKENS * _MAX_SLOTS // BM
    route = torch.zeros(
        bins + 2 + 3 * max_blocks + experts * MAX_TOKENS,
        dtype=torch.int32,
        device=device,
    )
    n_blocks = route[bins + 1 : bins + 2]
    blk_expert, blk_start, blk_count = route[bins + 2 : bins + 2 + 3 * max_blocks].view(
        3, max_blocks
    )
    pairs = route[bins + 2 + 3 * max_blocks :]
    return route, pairs, blk_expert, blk_start, blk_count, n_blocks


@triton.jit
def file_pairs(
    route_ptr,
    expert_map_ptr,
    gid,
    pair,
    valid,
    BINS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BM: tl.constexpr,
):
    """Router side of `route_buckets`: one program per token files its local
    `pair`s (global expert `gid`) under their experts; the last program to
    finish lists the blocks and zeroes the counts and ticket.

    The order within a bucket varies between runs, but each pair's expert
    output does not depend on its position, so results stay deterministic.
    """
    local = tl.load(expert_map_ptr + gid, mask=valid, other=-1)
    mine = local >= 0
    index = tl.atomic_add(route_ptr + local, 1, mask=mine, sem="relaxed")
    pairs_ptr = route_ptr + BINS + 2 + 3 * MAX_BLOCKS
    tl.store(pairs_ptr + local * _BUCKET_CAP + index, pair, mask=mine)
    tl.debug_barrier()
    if tl.atomic_add(route_ptr + BINS, 1) == tl.num_programs(0) - 1:
        bins = tl.arange(0, BINS)
        # Every other program is done; read the counts at L2, then clear.
        count = tl.atomic_add(route_ptr + bins, 0, sem="relaxed")
        tl.store(route_ptr + bins, tl.zeros((BINS,), tl.int32))
        blocks = (count + BM - 1) // BM
        first = tl.cumsum(blocks, 0) - blocks
        b = tl.arange(0, _BUCKET_CAP // BM)[None, :]
        bmask = b * BM < count[:, None]
        blk = route_ptr + BINS + 2 + first[:, None] + b
        tl.store(blk, bins[:, None] + 0 * b, mask=bmask)
        tl.store(blk + MAX_BLOCKS, bins[:, None] * _BUCKET_CAP + b * BM, mask=bmask)
        tl.store(
            blk + 2 * MAX_BLOCKS,
            tl.minimum(count[:, None] - b * BM, BM),
            mask=bmask,
        )
        tl.store(route_ptr + BINS + 1, tl.sum(blocks, 0))
        tl.store(route_ptr + BINS, 0)
