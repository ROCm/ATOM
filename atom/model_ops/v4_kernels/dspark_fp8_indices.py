# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Index construction for the DSpark FP8 block attention path.

The fp8 draft path feeds `aiter.mla.mla_decode_fwd_v4_nm` (through
`sparse_attn_v4_paged_decode`) instead of the bf16 Triton `sparse_attn`. That
kernel addresses KV as one pool of rows plus a CSR index list per query row, so
the `[window ++ draft-block]` KV DSpark attends to has to be expressed as pool
row ids rather than a materialised `[B, W+T, 512]` tensor.

`dspark_build_indices` emits all three in one launch:

- `kv_indices` / `kv_indptr` — the ragged CSR, one query row per draft position,
  `N = B*T` rows.
- `draft_rows` — the ring rows the draft block's own KV is scattered into, fused
  into the `qk_norm_rope_maybe_quant` launch that produces it.

`qo_indptr` comes from `dspark_qo_indptr`. Shapes are statically known: no
`.item()`, no data-dependent allocation, so a captured CUDA graph replays it.
The buffers are shape-keyed and persistent, and the capacity is the worst case
rather than a tight fit, so trailing slack is simply never read.

The pool row formula is not restated here -- the kernel calls
`pool_index.window_row`, which is what that module exists for.

The `[B,T,W+T]` gather indices the bf16 path uses (`_dspark_block_topk_idxs`)
are a broadcast along T — every draft position attends to the identical set — so
one KV list per request is sufficient here, which is exactly what CSR expresses.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.v4_pool_geometry import WindowParams
from atom.model_ops.v4_kernels.pool_index import window_constexprs, window_row


def dspark_kv_index_capacity(batch: int, draft: int, window: int) -> int:
    """Elements to reserve for `kv_indices`, plus the trailing dump slot."""
    return batch * draft * (window + draft) + 1


def dspark_qo_indptr(batch: int, draft: int, device) -> torch.Tensor:
    """`qo_indptr` for `N = B*T` single-token queries: `arange(N+1)`.

    The asm wrapper runs `max_seqlen_q = 1`, i.e. one "sequence" per query row —
    the same convention the V4 target's decode metadata uses
    (`deepseek_v4_attn.py:3727`).

    Cached (`_SCRATCH`, defined below) rather than rebuilt: the value depends
    only on the shape, so a fresh `arange` per call would be an allocation and a
    launch per stage per step for a constant. A persistent buffer is also what
    graph capture wants — a fresh one each forward replays against a dead
    address.
    """
    key = ("qo", batch, draft, str(device))
    hit = _SCRATCH.get(key)
    if hit is None:
        hit = torch.arange(batch * draft + 1, dtype=torch.int32, device=device)
        _SCRATCH[key] = hit
    return hit


# ---------------------------------------------------------------------------
# Fused build.
#
# Spelled as torch ops this is ~50 elementwise launches (~286us at B=128 T=4
# W=128 on MI355X) over a few thousand int32 -- launch-bound, in front of an
# attention kernel that reads only W+T rows. `write_v4_paged_decode_indices` is
# the target's answer to the same problem, and this is that for the draft's
# `[window ++ draft-block]` list.
#
# ONE launch. The fill needs the CSR offsets before it can address its slice,
# but the summand is only `min(anchor+1, W) + T`, so a program rebuilds the
# whole prefix from the B anchors itself instead of waiting on a second kernel.
# B is the CG-padded batch, so that redundant BLOCK_B reduction is far cheaper
# than the extra launch and the serialization it forces.
# ---------------------------------------------------------------------------


@triton.jit
def _dspark_index_kernel(
    anchors_ptr,  # [B] int64
    slots_ptr,  # [B] int64
    kv_indptr_ptr,  # [B*T+1] int32 out
    draft_rows_ptr,  # [B*T] int32 out
    out_ptr,  # [capacity] int32 out
    B,
    ring_start,
    T: tl.constexpr,
    W: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per query row: its CSR offset, its draft ring row, and its
    whole `[valid window ++ draft block]` list.

    Per-request row length is `min(anchor+1, W) + T` and every draft position of
    a request shares it, so the CSR offset of row `b*T+t` is
    `T * exclusive_prefix(len)[b] + t * len[b]`. The prefix is rebuilt here from
    the anchors instead of read from an earlier kernel -- that is what collapses
    the build to a single launch (see the note above).

    Window slot `j` is absolute position `anchor-n+1+j` for `n = min(anchor+1,W)`
    valid slots -- the `p >= 0` suffix `_build_block_plan` marks valid -- and the
    draft half continues at `anchor+1`. Both are non-negative, which matters:
    Triton's remainder follows C, so a negative position would not wrap.
    """
    i = tl.program_id(0)
    b = i // T
    t = i % T

    # Exclusive prefix over requests, recomputed per program.
    bb = tl.arange(0, BLOCK_B)
    live = bb < B
    all_anchors = tl.load(anchors_ptr + bb, mask=live, other=0)
    per_req = tl.where(live, (tl.minimum(all_anchors + 1, W) + T) * T, 0)
    base = tl.sum(tl.where(bb == b, tl.cumsum(per_req, axis=0) - per_req, 0))

    anchor = tl.load(anchors_ptr + b)
    slot = tl.load(slots_ptr + b)
    n_valid = tl.minimum(anchor + 1, W)
    length = n_valid + T
    start = base + t * length

    tl.store(kv_indptr_ptr + i, start.to(tl.int32))
    if i == 0:
        tl.store(kv_indptr_ptr + B * T, tl.sum(per_req).to(tl.int32))

    # This row's own draft KV lands at absolute position anchor+1+t.
    tl.store(
        draft_rows_ptr + i,
        window_row(
            slot,
            anchor + 1 + t,
            ring_start,
            RING_SLOTS,
            SLOT_ROWS,
            RING_STRIDE,
            RUN_ROWS,
        ).to(tl.int32),
    )

    j = tl.arange(0, BLOCK_K)
    in_window = j < n_valid
    pos = tl.where(in_window, anchor - n_valid + 1 + j, anchor + 1 + (j - n_valid))
    row = window_row(
        slot, pos, ring_start, RING_SLOTS, SLOT_ROWS, RING_STRIDE, RUN_ROWS
    )
    tl.store(out_ptr + start + j, row.to(tl.int32), mask=j < length)


_SCRATCH: dict = {}
# Shapes `dspark_build_indices` has actually filled, so a stage that reads the
# buffers back cannot be handed an uninitialised `torch.empty`.
_BUILT: set = set()


def _scratch(batch: int, draft: int, device):
    """Persistent `(kv_indptr, draft_rows)` for a shape.

    `write_v4_paged_decode_indices` states the principle for the target's
    equivalent: "All inputs are persistent forward_vars buffers -- no allocator
    churn." This path runs once per DSpark stage per step, so a fresh pair of
    allocations each call is exactly the churn that warns against. Keyed by
    shape, which is fixed for the process (`T = min(mtp_k, window)`, and B is
    the CUDA-graph-padded batch).
    """
    key = (batch, draft, str(device))
    hit = _SCRATCH.get(key)
    if hit is None:
        n = batch * draft
        hit = (
            torch.empty(n + 1, dtype=torch.int32, device=device),
            torch.empty(n, dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = hit
    return hit


def dspark_kv_index_scratch(
    batch: int, draft: int, window: int, device
) -> torch.Tensor:
    """Persistent `kv_indices` buffer for a shape — see :func:`_scratch`."""
    key = ("idx", batch, draft, window, str(device))
    hit = _SCRATCH.get(key)
    if hit is None:
        hit = torch.empty(
            dspark_kv_index_capacity(batch, draft, window),
            dtype=torch.int32,
            device=device,
        )
        _SCRATCH[key] = hit
    return hit


def dspark_indices_view(
    batch: int, draft: int, window: int, device, out_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read back what :func:`dspark_build_indices` last wrote for this shape.

    DSpark runs one block through every stage and these indices are
    stage-invariant (all DSpark layers share compress ratio 0, hence one
    `WindowParams`, and each layer's plane view is base-row-relative), so stage
    0 builds and the rest read. The buffers are the shape-keyed persistent ones,
    so this is pure lookup — no launch, no allocation.

    Freshness comes from call order, not from here: `_DSparkInner.forward` walks
    the stages in order and stage 0 refills the buffers at the top of every
    block. What this DOES catch is a stage reading a shape that was never built
    at all, which would feed the asm kernel a `torch.empty`.
    """
    key = (batch, draft, str(device))
    if key not in _BUILT:
        raise RuntimeError(
            f"DSpark kv indices for B={batch} T={draft} on {device} were never "
            "built; stage 0 must run before any stage reads them back."
        )
    kv_indptr, draft_rows = _scratch(batch, draft, device)
    capacity = dspark_kv_index_capacity(batch, draft, window)
    return out_indices[:capacity], kv_indptr, draft_rows


def dspark_build_indices(
    window: WindowParams,
    slots: torch.Tensor,  # [B] per-request ring slot
    anchors: torch.Tensor,  # [B] per-request anchor position
    draft_width: int,  # T
    draft_window: int,  # W
    out_indices: torch.Tensor,  # [>= capacity] int32, preallocated
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Everything the asm path needs, in one launch.

    Returns ``(kv_indices, kv_indptr, draft_rows)``; build ``qo_indptr`` with
    :func:`dspark_qo_indptr`.
    """
    B = anchors.shape[0]
    T, W = draft_width, draft_window
    capacity = dspark_kv_index_capacity(B, T, W)
    if out_indices.numel() < capacity:
        raise ValueError(
            f"DSpark kv_indices buffer holds {out_indices.numel()} < {capacity} "
            f"needed for B={B} T={T} W={W}."
        )

    dev = anchors.device
    kv_indptr, draft_rows = _scratch(B, T, dev)
    anchors_i64 = anchors.to(torch.int64)
    slots_i64 = slots.to(torch.int64)

    _dspark_index_kernel[(B * T,)](
        anchors_i64,
        slots_i64,
        kv_indptr,
        draft_rows,
        out_indices,
        B,
        window.ring_start,
        T=T,
        W=W,
        **window_constexprs(window),
        BLOCK_B=triton.next_power_of_2(B),
        BLOCK_K=triton.next_power_of_2(W + T),
    )
    _BUILT.add((B, T, str(dev)))
    return out_indices[:capacity], kv_indptr, draft_rows
