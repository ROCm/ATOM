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

`qo_indptr` and the scatter's `batch_ids` are constants that ride in the same
`DSparkIndexBuffers` bundle. Everything is allocated once at `max_num_seqs` and
only ever sliced, and shapes are statically known -- no `.item()`, no
data-dependent allocation -- so a captured CUDA graph replays it.

The pool row formula is not restated here -- the kernel calls
`pool_index.window_row`, which is what that module exists for.

The `[B,T,W+T]` gather indices the bf16 path uses (`_dspark_block_topk_idxs`)
are a broadcast along T — every draft position attends to the identical set — so
one KV list per request is sufficient here, which is exactly what CSR expresses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.v4_pool_geometry import WindowParams
from atom.model_ops.v4_kernels.pool_index import window_constexprs, window_row


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


@dataclass
class DSparkIndexBuffers:
    """The fp8 path's index buffers, allocated once at `max_num_seqs`.

    Sized at the maximum batch and only ever sliced -- the idiom
    `write_v4_paged_decode_indices` states as "All inputs are persistent
    forward_vars buffers, no allocator churn", and that the drafter's own
    `_init_draft_block_buffers` already follows. Nothing here is keyed by
    shape, because every one of these is prefix-stable in the batch: the first
    `B*T` entries of the max-batch buffer ARE the batch-`B` answer.

    `qo_indptr` and `batch_ids` are constants, filled at allocation and never
    written again. The other three are refilled by `dspark_build_indices` at
    the top of each block; `built_for` records the batch they hold, so a stage
    reading them back cannot be handed a stale or uninitialised slice.
    """

    kv_indices: torch.Tensor  # [max_b*T*(W+T)] int32
    kv_indptr: torch.Tensor  # [max_b*T+1] int32
    draft_rows: torch.Tensor  # [max_b*T] int32
    qo_indptr: torch.Tensor  # [max_b*T+1] int32, constant ramp
    batch_ids: torch.Tensor  # [max_b*T] int32, constant [0]*T ++ [1]*T ++ ...
    built_for: int = -1


def dspark_index_buffers(
    max_batch: int, draft: int, window: int, device
) -> DSparkIndexBuffers:
    """Allocate :class:`DSparkIndexBuffers` for a process's largest batch.

    `qo_indptr` is `arange(N+1)`: the asm wrapper runs `max_seqlen_q = 1`, one
    "sequence" per query row, the same convention the V4 target's decode
    metadata uses (`deepseek_v4_attn.py:3727`).

    `batch_ids` is the token -> request map the fused SWA scatter gates on
    (`bid >= 0`). It carries no CG-pad sentinels, and does not need any: the
    draft runs at `context.batch_size`, which is `scheduled_bs`, the REAL decode
    batch (`model_runner.py:2609`) -- not the padded `effective_bs` the target's
    metadata is built at. The target can alias `cu_seqlens_q[:bs]` for its own
    (`deepseek_v4_attn.py:2348`) because at one token per sequence that slice is
    already `arange(bs)`; DSpark runs T tokens per request and needs each id
    repeated T times, so there is no existing buffer to slice.
    """
    n = max_batch * draft
    i32 = {"dtype": torch.int32, "device": device}
    return DSparkIndexBuffers(
        kv_indices=torch.empty(n * (window + draft), **i32),
        kv_indptr=torch.empty(n + 1, **i32),
        draft_rows=torch.empty(n, **i32),
        qo_indptr=torch.arange(n + 1, **i32),
        batch_ids=torch.arange(max_batch, **i32)
        .view(max_batch, 1)
        .expand(max_batch, draft)
        .reshape(-1)
        .contiguous(),
    )


def dspark_index_views(
    bufs: DSparkIndexBuffers, batch: int, draft: int, window: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The `(kv_indices, kv_indptr, draft_rows)` slices for a batch.

    DSpark runs one block through every stage and the values are
    stage-invariant (all DSpark layers share compress ratio 0, hence one
    `WindowParams`, and each layer's plane view is base-row-relative), so stage
    0 builds and the rest read this back -- no launch, no allocation.
    """
    if bufs.built_for != batch:
        raise RuntimeError(
            f"DSpark kv indices hold batch {bufs.built_for}, not {batch}; "
            "stage 0 must build them before any stage reads them back."
        )
    n = batch * draft
    return (
        bufs.kv_indices[: n * (window + draft)],
        bufs.kv_indptr[: n + 1],
        bufs.draft_rows[:n],
    )


def dspark_build_indices(
    window: WindowParams,
    slots: torch.Tensor,  # [B] per-request ring slot
    anchors: torch.Tensor,  # [B] per-request anchor position
    draft_width: int,  # T
    draft_window: int,  # W
    bufs: DSparkIndexBuffers,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Everything the asm path needs, in one launch.

    Fills ``bufs`` in place and returns its ``(kv_indices, kv_indptr,
    draft_rows)`` slices; ``qo_indptr`` and ``batch_ids`` are constants already
    sitting in the same bundle.
    """
    B = anchors.shape[0]
    T, W = draft_width, draft_window
    if bufs.batch_ids.numel() < B * T:
        raise ValueError(
            f"DSpark index buffers hold {bufs.batch_ids.numel() // T} requests "
            f"< B={B}; they are sized at max_num_seqs."
        )
    bufs.built_for = B
    kv_indices, kv_indptr, draft_rows = dspark_index_views(bufs, B, T, W)
    anchors_i64 = anchors.to(torch.int64)
    slots_i64 = slots.to(torch.int64)

    _dspark_index_kernel[(B * T,)](
        anchors_i64,
        slots_i64,
        kv_indptr,
        draft_rows,
        kv_indices,
        B,
        window.ring_start,
        T=T,
        W=W,
        **window_constexprs(window),
        BLOCK_B=triton.next_power_of_2(B),
        BLOCK_K=triton.next_power_of_2(W + T),
    )
    return kv_indices, kv_indptr, draft_rows
