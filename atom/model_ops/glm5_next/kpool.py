# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""kpool: pooled indexer-key compression for GLM-5.3-Flash.

The sparse indexer does not score every token. Each group of ``index_kpool``
consecutive tokens is compressed into ONE cached entry, top-k runs at that pool
granularity (``index_topk // index_kpool`` pools), and each selected pool then
expands back to the ``index_kpool`` token positions it covers. The trailing
incomplete pool ("the tail") is always selected, so the newest tokens are never
dropped.

The compression is, per pool ``p`` and per dimension ``d``:

    w[slot, d] = softmax over slot of (gate[p, slot, d] + ape[slot, d])
    pooled[p, d] = sum_slot w[slot, d] * k[p, slot, d]

Note the softmax runs **over the pool's slots, independently per dimension** --
it is not a scalar per-slot gate. Getting that wrong produces plausible-looking
values and a quiet accuracy loss, which is why `pool_compress_ref` below exists
as the oracle the Triton kernel is tested against.

``pooled`` is then Hadamard-128 rotated and quantized to FP8 with a ue8m0
(power-of-two) scale, matching the basis the cached keys are scored in.

Ported from vLLM PR #53906 (`vllm/models/glm5next/nvidia/ops/kpool_compress.py`).
"""

from __future__ import annotations

import math

import torch
from aiter import dtypes

# The GLM-5.3-Flash indexer head dimension is fixed at 128 and the FP8 quant
# block spans the whole head, so both are compile-time constants here.
INDEX_HEAD_DIM = 128
_INV_SQRT_128 = 1.0 / math.sqrt(128.0)
FP8_DTYPE = dtypes.fp8
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)

# --------------------------------------------------------------------------
# Reference implementations (the correctness oracle; not the fast path)
# --------------------------------------------------------------------------


def pool_compress_ref(
    k: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Softmax-pool whole pools of indexer keys.

    Args:
        k:    ``[num_pools, pool, head_dim]`` layer-normed indexer keys.
        gate: ``[num_pools, pool, head_dim]`` per-token gate scores.
        ape:  ``[pool, head_dim]`` learned per-slot bias.

    Returns:
        ``[num_pools, head_dim]`` pooled keys, in fp32.
    """
    scores = gate.float() + ape.float().unsqueeze(0)
    # dim=1 is the slot axis: one softmax per (pool, dim) over the pool's slots.
    weights = scores.softmax(dim=1)
    return (weights * k.float()).sum(dim=1)


def hadamard128_ref(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over a 128-wide last dim.

    The ``1/sqrt(128)`` normalization is NOT cosmetic. ``H`` is applied to the
    pooled keys AND to the indexer query (`fwht128_quant_fp8`), so with the
    normalization ``<Hq, Hk> == <q, k>`` exactly and the rotation is a pure
    quantization-conditioning change. Dropping it would scale every logit by
    128 -- harmless for top-k on its own -- but the FP8 scale is ue8m0, i.e.
    rounded to a POWER OF TWO, and ``1/sqrt(128) == 2**-3.5`` is not one. The
    two conventions therefore quantize to genuinely different bytes, so this
    must match the reference (vLLM PR #53906 `_hadamard128`) exactly.
    """
    assert x.shape[-1] == 128, f"expected head_dim 128, got {x.shape[-1]}"
    out = x.float().clone()
    step = 1
    while step < 128:
        view = out.view(*out.shape[:-1], 128 // (2 * step), 2, step)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        step *= 2
    return out * _INV_SQRT_128


def quant_fp8_ue8m0_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vector absmax native-FP8 quant with a power-of-two scale."""
    fp8_max = FP8_MAX
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / fp8_max)))
    q = (x / scale).clamp(-fp8_max, fp8_max)
    return q.to(FP8_DTYPE), scale.squeeze(-1)


def compress_pools_ref(
    k: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full compression: softmax-pool -> Hadamard-128 -> FP8/ue8m0."""
    pooled = pool_compress_ref(k, gate, ape)
    # bf16 round-trips mirror the fused kernel's intermediate precision.
    pooled = pooled.to(torch.bfloat16).float()
    rotated = hadamard128_ref(pooled).to(torch.bfloat16).float()
    return quant_fp8_ue8m0_ref(rotated)


# --------------------------------------------------------------------------
# Pool -> token expansion
# --------------------------------------------------------------------------


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    """How many pools to select so expanding them yields ``topk`` tokens."""
    assert topk % pool_size == 0, (topk, pool_size)
    return topk // pool_size


def expand_pools_to_tokens(
    pool_ids: torch.Tensor,
    pool_valid: torch.Tensor,
    topk: int,
    pool_size: int,
) -> torch.Tensor:
    """Expand selected pool ids to token ids.

    Args:
        pool_ids:   ``[rows, topk // pool_size]`` selected pool indices.
        pool_valid: same shape, False where the slot is padding.

    Returns:
        ``[rows, topk]`` token indices, ``-1`` where invalid.
    """
    assert pool_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    offsets = torch.arange(pool_size, device=pool_ids.device, dtype=torch.int64)
    token_ids = pool_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(pool_ids.shape[0], topk)
    valid = (
        pool_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(pool_ids.shape[0], topk)
    )
    return torch.where(
        valid,
        token_ids.to(torch.int32),
        torch.full_like(token_ids, -1, dtype=torch.int32),
    )


def expand_and_append_tail_ref(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Torch oracle for `expand_pools_and_append_tail` (compact layout).

    History for the row's VALID pools, then the tail immediately after, then
    -1. The compaction is what makes the ragged `sparse_kv_indptr` count line
    up with the entries actually written.
    """
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    device = pool_ids.device
    out = torch.full((rows, out_cols), -1, dtype=torch.int32, device=device)
    seq = seq_lens.to(torch.int64)
    pool_len = seq // pool_size
    tail_count = seq - pool_len * pool_size
    n_valid = torch.minimum(pool_len, torch.full_like(pool_len, n_groups))
    for r in range(rows):
        cur = []
        for g in range(int(n_valid[r])):
            pid = int(pool_ids[r, g])
            if pid < 0:
                cur.extend([-1] * pool_size)
            else:
                cur.extend(pid * pool_size + o for o in range(pool_size))
        start = int(pool_len[r]) * pool_size
        cur.extend(start + t for t in range(int(tail_count[r])))
        out[r, : len(cur)] = torch.tensor(cur, dtype=torch.int32, device=device)
    return out


def append_tail_to_topk(
    topk_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Append the trailing incomplete pool's tokens.

    ``index_kpool_always_select_tail``: the in-progress pool is never compressed
    yet, so its raw tokens are appended after the expanded history rather than
    being scored.
    """
    tail = pool_size - 1
    if tail == 0:
        return topk_tokens
    rows = topk_tokens.shape[0]
    device = topk_tokens.device
    pooled_end = (seq_lens // pool_size) * pool_size  # first tail token
    offs = torch.arange(tail, device=device, dtype=torch.int64)
    tail_ids = pooled_end.to(torch.int64).unsqueeze(1) + offs.unsqueeze(0)
    tail_valid = tail_ids < seq_lens.to(torch.int64).unsqueeze(1)
    tail_ids = torch.where(
        tail_valid,
        tail_ids.to(torch.int32),
        torch.full_like(tail_ids, -1, dtype=torch.int32),
    )
    return torch.cat([topk_tokens, tail_ids.view(rows, tail)], dim=1)


# --------------------------------------------------------------------------
# Triton kernels (the fast path; every one is checked against a *_ref above)
# --------------------------------------------------------------------------

import triton
import triton.language as tl


@triton.jit
def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    """One Walsh-Hadamard butterfly stage over a flat ``N = GROUPS*2*STRIDE``.

    Kept flat rather than 2D so ``BLOCK_R`` rows transform in one pass: every
    stage's ``(GROUPS, 2, STRIDE)`` tiling has ``2*STRIDE`` dividing 128, so a
    butterfly pair never straddles a row boundary.
    """
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@triton.jit
def _fwht128_rows(x, N: tl.constexpr, ROWS: tl.constexpr):
    """Orthonormal Hadamard-128 on ``ROWS`` contiguous 128-wide rows."""
    x = _fwht_stage(x, N, ROWS * 64, 1)
    x = _fwht_stage(x, N, ROWS * 32, 2)
    x = _fwht_stage(x, N, ROWS * 16, 4)
    x = _fwht_stage(x, N, ROWS * 8, 8)
    x = _fwht_stage(x, N, ROWS * 4, 16)
    x = _fwht_stage(x, N, ROWS * 2, 32)
    x = _fwht_stage(x, N, ROWS, 64)
    # 1/sqrt(128) is exact in fp32; see hadamard128_ref for why it matters.
    return x * 0.08838834764831845


@triton.jit
def _kpool_pool_rotate_kernel(
    k_ptr,
    gate_ptr,
    ape_ptr,
    out_ptr,
    n_pools,
    HEAD_DIM: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """softmax(gate + ape) over a pool's slots, per dimension -> Hadamard-128.

    The softmax runs over the SLOT axis independently for each of the HEAD_DIM
    dimensions. A scalar-per-slot gate is the plausible wrong reading and
    produces finite, plausible-looking keys; `pool_compress_ref` is the oracle
    that separates them.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_pools
    offs = tl.arange(0, HEAD_DIM)
    base = rows[:, None] * (POOL * HEAD_DIM) + offs[None, :]

    # Two passes over the (compile-time) POOL slots: a max for numerical
    # stability, then the weighted sum. POOL is 4, so this fully unrolls.
    m = tl.full((BLOCK_R, HEAD_DIM), float("-inf"), tl.float32)
    for s in tl.static_range(POOL):
        g = tl.load(gate_ptr + base + s * HEAD_DIM, mask=rmask[:, None], other=0.0).to(
            tl.float32
        )
        a = tl.load(ape_ptr + s * HEAD_DIM + offs).to(tl.float32)
        m = tl.maximum(m, g + a[None, :])

    acc = tl.zeros((BLOCK_R, HEAD_DIM), tl.float32)
    den = tl.zeros((BLOCK_R, HEAD_DIM), tl.float32)
    for s in tl.static_range(POOL):
        g = tl.load(gate_ptr + base + s * HEAD_DIM, mask=rmask[:, None], other=0.0).to(
            tl.float32
        )
        a = tl.load(ape_ptr + s * HEAD_DIM + offs).to(tl.float32)
        w = tl.exp(g + a[None, :] - m)
        kv = tl.load(k_ptr + base + s * HEAD_DIM, mask=rmask[:, None], other=0.0).to(
            tl.float32
        )
        acc += w * kv
        den += w
    pooled = acc / den
    # Match the reference's bf16 materialization between pool and rotate.
    pooled = pooled.to(tl.bfloat16).to(tl.float32)

    N: tl.constexpr = BLOCK_R * HEAD_DIM
    rot = _fwht128_rows(tl.reshape(pooled, (N,)), N, BLOCK_R)
    rot = tl.reshape(rot, (BLOCK_R, HEAD_DIM))
    tl.store(
        out_ptr + rows[:, None] * HEAD_DIM + offs[None, :],
        rot.to(tl.bfloat16),
        mask=rmask[:, None],
    )


def pool_and_rotate(
    k: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Softmax-pool whole pools and rotate into the cached-key basis.

    Args:
        k:    ``[num_pools, pool, head_dim]`` bf16 layer-normed indexer keys.
        gate: ``[num_pools, pool, head_dim]`` bf16 per-token gate scores.
        ape:  ``[pool, head_dim]`` per-slot bias.

    Returns:
        ``[num_pools, head_dim]`` bf16, ready for
        ``indexer_k_quant_and_cache`` -- which applies exactly the per-vector
        absmax ue8m0 FP8 quantization `quant_fp8_ue8m0_ref` describes, in the
        byte layout `cp_gather_indexer_k_quant_cache` reads back. Doing the
        quant+scatter with aiter's own op rather than a hand-written one keeps
        this file out of the cache's byte layout entirely.
    """
    num_pools, pool, head_dim = k.shape
    assert head_dim == INDEX_HEAD_DIM, f"kpool assumes head_dim 128, got {head_dim}"
    assert gate.shape == k.shape, (k.shape, gate.shape)
    assert ape.shape == (pool, head_dim), ape.shape
    out = torch.empty((num_pools, head_dim), dtype=torch.bfloat16, device=k.device)
    if num_pools == 0:
        return out
    block_r = 8
    _kpool_pool_rotate_kernel[(triton.cdiv(num_pools, block_r),)](
        k.contiguous(),
        gate.contiguous(),
        ape.contiguous().to(torch.float32),
        out,
        num_pools,
        HEAD_DIM=head_dim,
        POOL=pool,
        BLOCK_R=block_r,
        num_warps=4,
    )
    return out


@triton.jit
def _fwht_quant_kernel(
    q_ptr,
    qout_ptr,
    sout_ptr,
    n_rows,
    HEAD_DIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    FP8_AMAX: tl.constexpr,
):
    """Fused Hadamard-128 + per-row absmax FP8 (ue8m0) quant of the query."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    offs = tl.arange(0, HEAD_DIM)
    x = tl.load(
        q_ptr + rows[:, None] * HEAD_DIM + offs[None, :],
        mask=rmask[:, None],
        other=0.0,
    ).to(tl.float32)

    N: tl.constexpr = BLOCK_R * HEAD_DIM
    x = _fwht128_rows(tl.reshape(x, (N,)), N, BLOCK_R)
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (BLOCK_R, HEAD_DIM))

    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    # Keep the division spelling used by the reference and aiter's key
    # quantizer. Multiplying by a rounded reciprocal can cross the ceil(log2)
    # step and choose a scale one binade away at boundary values.
    scale = tl.exp2(tl.ceil(tl.log2(absmax / FP8_AMAX)))
    y = tl.minimum(tl.maximum(x / scale[:, None], -FP8_AMAX), FP8_AMAX)

    tl.store(
        qout_ptr + rows[:, None] * HEAD_DIM + offs[None, :], y, mask=rmask[:, None]
    )
    tl.store(sout_ptr + rows, scale, mask=rmask)


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the indexer query into the cached-key basis, then FP8-quant it.

    The pooled keys are stored Hadamard-rotated, so the query MUST be rotated
    by the same orthonormal transform or the dot products are meaningless. This
    replaces the plain `quant_func` call the token-granular path uses.

    Returns ``(q_fp8 [rows, 128], scale [rows, 1] fp32)`` -- the shapes
    ``Indexer.forward_impl`` already folds into ``weights``.
    """
    assert q.ndim == 2 and q.shape[1] == INDEX_HEAD_DIM, q.shape
    n_rows = q.shape[0]
    q_fp8 = torch.empty((n_rows, INDEX_HEAD_DIM), dtype=FP8_DTYPE, device=q.device)
    q_scale = torch.empty((n_rows, 1), dtype=torch.float32, device=q.device)
    if n_rows == 0:
        return q_fp8, q_scale
    block_r = 32
    _fwht_quant_kernel[(triton.cdiv(n_rows, block_r),)](
        q.contiguous(),
        q_fp8,
        q_scale,
        n_rows,
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_R=block_r,
        FP8_AMAX=FP8_MAX,
        num_warps=2,
    )
    return q_fp8, q_scale


# --------------------------------------------------------------------------
# Pool <-> physical cache addressing
# --------------------------------------------------------------------------


def pool_slot_mapping(
    pool_block_table: torch.Tensor,
    pool_ids: torch.Tensor,
    req_idx: torch.Tensor,
    pool_rows: int,
) -> torch.Tensor:
    """Physical cache slots for a batch's pools. ``-1`` passes through.

    Args:
        pool_block_table: the request's own ``[bs, n_blocks]`` block table.
                          One index block per KV block, so no remapping.
        pool_ids:         ``[n]`` per-entry pool id, ``-1`` where there is no
                          pool to write.
        req_idx:          ``[n]`` which request each entry belongs to. A
                          prefill batch interleaves requests, so a single
                          block-table row is not enough -- using one would
                          scatter every request but the first into another's
                          blocks.

    ``indexer_k_quant_and_cache`` skips slots < 0, which is how an incomplete
    pool is expressed without branching on the write.
    """
    valid = pool_ids >= 0
    ids = pool_ids.to(torch.int64).clamp_min(0)
    rows = req_idx.to(torch.int64)
    blocks = pool_block_table.to(torch.int64)[rows, ids // pool_rows]
    slots = blocks * pool_rows + (ids % pool_rows)
    return torch.where(valid, slots, torch.full_like(slots, -1))


# --------------------------------------------------------------------------
# Pool -> token expansion (fused)
# --------------------------------------------------------------------------


@triton.jit
def _expand_pools_and_append_tail_kernel(
    pool_ids_ptr,
    seq_lens_ptr,
    pool_base_ptr,
    tok_base_ptr,
    out_ptr,
    topk,
    out_cols,
    pid_s0,
    out_s0,
    POOL_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    HAS_POOL_BASE: tl.constexpr,
    HAS_TOK_BASE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols < out_cols

    seq_len = tl.load(seq_lens_ptr + row)
    pool_len = seq_len // POOL_SIZE
    tail_start = pool_len * POOL_SIZE
    tail_count = seq_len - tail_start  # in [0, POOL_SIZE)

    # Prefill scores one buffer holding EVERY request's pools, so the selected
    # ids are batch-global and the token ids the converter wants are
    # request-local plus that request's key offset. Both rebases are folded in
    # here rather than costing a separate pass over a [rows, 2176] int32 array.
    pool_base = tl.load(pool_base_ptr + row) if HAS_POOL_BASE else 0
    tok_base = tl.load(tok_base_ptr + row) if HAS_TOK_BASE else 0

    # The row is consumed RAGGEDLY: `sparse_kv_indptr` gives each row exactly
    # `min(pool_len, n_groups) * POOL_SIZE + tail_count` entries and nothing
    # past that is ever read. So the tail has to sit immediately after the last
    # VALID history entry, not at a fixed column `topk`. Anchoring it at `topk`
    # silently drops the newest 1..POOL_SIZE-1 tokens on every row whose
    # sequence is not pool-aligned -- three steps out of four -- and feeds the
    # attention `-1` entries in their place.
    n_valid = tl.minimum(pool_len, topk // POOL_SIZE)
    hist_end = n_valid * POOL_SIZE

    is_history = cols < hist_end
    g = cols // POOL_SIZE
    o = cols % POOL_SIZE
    pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)
    local_pid = pid - pool_base
    hist_out = tl.where(
        (pid >= 0) & (local_pid >= 0),
        (local_pid * POOL_SIZE + o + tok_base).to(tl.int32),
        -1,
    )

    # Tail: the request's trailing incomplete pool, never scored
    # (index_kpool_always_select_tail).
    tail_off = cols - hist_end
    is_tail = (tail_off >= 0) & (tail_off < tail_count)
    tail_out = tl.where(is_tail, (tail_start + tail_off + tok_base).to(tl.int32), -1)

    tl.store(
        out_ptr + row * out_s0 + cols,
        tl.where(is_history, hist_out, tail_out),
        mask=mask,
    )


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    out: torch.Tensor | None = None,
    pool_base: torch.Tensor | None = None,
    tok_base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused ``expand_pools_to_tokens`` + ``append_tail_to_topk``.

    Args:
        pool_ids:  ``[rows, topk // pool_size]`` selected pool indices, -1 padded.
        seq_lens:  ``[rows]`` int32 TOKEN-granular sequence length (pos + 1).
        out:       optional preallocated ``[rows, topk + pool_size - 1]`` int32.
        pool_base: optional ``[rows]`` int32 subtracted from each selected pool
                   id to make it request-local (prefill scores one batch-wide
                   pool buffer).
        tok_base:  optional ``[rows]`` int32 added to every emitted token id
                   (the request's offset in the converter's key space).

    Returns ``[rows, topk + pool_size - 1]`` int32 token indices, -1 where
    invalid. One launch instead of the ~25 elementwise kernels the torch
    spelling costs.
    """
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    if out is None:
        out = torch.empty((rows, out_cols), dtype=torch.int32, device=pool_ids.device)
    assert out.shape[-1] >= out_cols, (out.shape, out_cols)
    if rows == 0:
        return out
    block_cols = 128
    _expand_pools_and_append_tail_kernel[(rows, triton.cdiv(out_cols, block_cols))](
        pool_ids,
        seq_lens,
        pool_base if pool_base is not None else pool_ids,
        tok_base if tok_base is not None else pool_ids,
        out,
        topk,
        out_cols,
        pool_ids.stride(0),
        out.stride(0),
        POOL_SIZE=pool_size,
        BLOCK_COLS=block_cols,
        HAS_POOL_BASE=pool_base is not None,
        HAS_TOK_BASE=tok_base is not None,
    )
    return out


# --------------------------------------------------------------------------
# The tail buffer: the in-progress pool's raw K and gate, per request
# --------------------------------------------------------------------------
#
# Layout, per indexer layer: [num_slots, 2, POOL, HEAD_DIM] bf16, where index 0
# on the second axis is K and 1 is the gate score. A token at absolute position
# `p` occupies row `p % POOL`. It rides the same per-request state slots KDA's
# recurrent state uses, so it inherits their lifetime, forking and relocation.


@triton.jit
def _kpool_seed_tail_kernel(
    k_ptr,
    gate_ptr,
    positions_ptr,
    cu_seqlens_q_ptr,
    slot_idx_in_ptr,
    slot_idx_out_ptr,
    tail_ptr,
    HEAD_DIM: tl.constexpr,
    POOL: tl.constexpr,
):
    """Persist each prefill request's trailing incomplete pool.

    Chunk boundaries are pool-aligned, so every non-final chunk ends with
    ``seq_len % POOL == 0`` and seeds nothing; the final chunk always contains
    the whole tail. The ``i < q_start`` guard makes a violation of that
    alignment drop the write rather than read another request's tokens.
    """
    r = tl.program_id(0)
    j = tl.program_id(1)
    q_start = tl.load(cu_seqlens_q_ptr + r)
    q_end = tl.load(cu_seqlens_q_ptr + r + 1)
    if q_end <= q_start:
        return
    seq_len = tl.load(positions_ptr + q_end - 1).to(tl.int32) + 1
    tail_count = seq_len % POOL
    if j >= tail_count:
        return
    # Absolute position of tail token j, mapped back to its row in the batch.
    i = q_end - 1 - (tail_count - 1 - j)
    src_slot = tl.load(slot_idx_in_ptr + r).to(tl.int64)
    dst_slot = tl.load(slot_idx_out_ptr + r).to(tl.int64)
    if src_slot < 0 or dst_slot < 0:
        return
    offs = tl.arange(0, HEAD_DIM)
    src_base = src_slot * (2 * POOL * HEAD_DIM) + j * HEAD_DIM
    dst_base = dst_slot * (2 * POOL * HEAD_DIM) + j * HEAD_DIM
    from_chunk = i >= q_start
    kval = tl.load(k_ptr + i * HEAD_DIM + offs, mask=from_chunk, other=0.0)
    gval = tl.load(gate_ptr + i * HEAD_DIM + offs, mask=from_chunk, other=0.0)
    kval = tl.where(from_chunk, kval, tl.load(tail_ptr + src_base + offs))
    gval = tl.where(
        from_chunk,
        gval,
        tl.load(tail_ptr + src_base + POOL * HEAD_DIM + offs),
    )
    tl.store(tail_ptr + dst_base + offs, kval)
    tl.store(
        tail_ptr + dst_base + POOL * HEAD_DIM + offs,
        gval,
    )


def kpool_seed_tail(
    tail: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    slot_idx: torch.Tensor,
    pool_size: int,
    slot_idx_in: torch.Tensor | None = None,
) -> None:
    """Seed the per-request tail buffer at the end of prefill."""
    num_requests = cu_seqlens_q.shape[0] - 1
    if num_requests <= 0:
        return
    assert tail.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    _kpool_seed_tail_kernel[(num_requests, pool_size)](
        k.contiguous(),
        gate.contiguous(),
        positions,
        cu_seqlens_q,
        slot_idx if slot_idx_in is None else slot_idx_in,
        slot_idx,
        tail,
        HEAD_DIM=k.shape[-1],
        POOL=pool_size,
    )


@triton.jit
def _kpool_decode_stash_and_pool_kernel(
    k_ptr,
    gate_ptr,
    positions_ptr,
    slot_idx_in_ptr,
    slot_idx_out_ptr,
    tail_ptr,
    ape_ptr,
    out_ptr,
    HEAD_DIM: tl.constexpr,
    POOL: tl.constexpr,
):
    """Stash one decode token into the tail, then pool the (possibly complete)
    pool.

    The pooled vector is computed for EVERY request, complete or not; the
    caller marks incomplete pools with slot ``-1`` so the cache write skips
    them. That keeps this kernel branch-free on the write and its shapes fixed,
    which is what CUDAGraph capture requires -- a device-value-dependent
    early-out here would read as "the indexer stopped writing" on a healthy
    model.
    """
    r = tl.program_id(0)
    src_slot = tl.load(slot_idx_in_ptr + r).to(tl.int64)
    dst_slot = tl.load(slot_idx_out_ptr + r).to(tl.int64)
    if src_slot < 0 or dst_slot < 0:
        return
    pos = tl.load(positions_ptr + r).to(tl.int32)
    phase = pos % POOL
    offs = tl.arange(0, HEAD_DIM)
    src_base = src_slot * (2 * POOL * HEAD_DIM)
    dst_base = dst_slot * (2 * POOL * HEAD_DIM)
    cur_k = tl.load(k_ptr + r * HEAD_DIM + offs)
    cur_g = tl.load(gate_ptr + r * HEAD_DIM + offs)

    # Read the partial pool from the input slot and materialize its updated
    # form in the output slot. They differ when a prefix-cache checkpoint forks
    # into a newly allocated slot.
    m = tl.full((HEAD_DIM,), float("-inf"), tl.float32)
    for s in tl.static_range(POOL):
        g = tl.where(
            s == phase,
            cur_g,
            tl.load(tail_ptr + src_base + POOL * HEAD_DIM + s * HEAD_DIM + offs),
        )
        tl.store(tail_ptr + dst_base + POOL * HEAD_DIM + s * HEAD_DIM + offs, g)
        g = g.to(tl.float32)
        a = tl.load(ape_ptr + s * HEAD_DIM + offs).to(tl.float32)
        m = tl.maximum(m, g + a)
    acc = tl.zeros((HEAD_DIM,), tl.float32)
    den = tl.zeros((HEAD_DIM,), tl.float32)
    for s in tl.static_range(POOL):
        g = tl.where(
            s == phase,
            cur_g,
            tl.load(tail_ptr + src_base + POOL * HEAD_DIM + s * HEAD_DIM + offs),
        ).to(tl.float32)
        kv = tl.where(
            s == phase,
            cur_k,
            tl.load(tail_ptr + src_base + s * HEAD_DIM + offs),
        )
        tl.store(tail_ptr + dst_base + s * HEAD_DIM + offs, kv)
        a = tl.load(ape_ptr + s * HEAD_DIM + offs).to(tl.float32)
        w = tl.exp(g + a - m)
        acc += w * kv.to(tl.float32)
        den += w
    pooled = (acc / den).to(tl.bfloat16).to(tl.float32)

    N: tl.constexpr = HEAD_DIM
    rot = _fwht128_rows(tl.reshape(pooled, (N,)), N, 1)
    tl.store(out_ptr + r * HEAD_DIM + offs, rot.to(tl.bfloat16))


def kpool_decode_stash_and_pool(
    tail: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    positions: torch.Tensor,
    slot_idx: torch.Tensor,
    ape: torch.Tensor,
    pool_size: int,
    out: torch.Tensor | None = None,
    slot_idx_in: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stash the decode step's token and return each request's pooled vector.

    The result is only *meaningful* for requests whose pool completed this step
    (``positions % pool_size == pool_size - 1``); the caller gates the cache
    write on that with a ``-1`` slot.
    """
    num_requests = k.shape[0]
    head_dim = k.shape[-1]
    if out is None:
        out = torch.zeros(
            (num_requests, head_dim), dtype=torch.bfloat16, device=k.device
        )
    if num_requests == 0:
        return out
    _kpool_decode_stash_and_pool_kernel[(num_requests,)](
        k.contiguous(),
        gate.contiguous(),
        positions,
        slot_idx if slot_idx_in is None else slot_idx_in,
        slot_idx,
        tail,
        ape.contiguous().to(torch.float32),
        out,
        HEAD_DIM=head_dim,
        POOL=pool_size,
    )
    return out
