# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Index construction for the DSpark FP8 block attention path.

The fp8 draft path feeds `aiter.mla.mla_decode_fwd_v4_nm` (through
`sparse_attn_v4_paged_decode`) instead of the bf16 Triton `sparse_attn`. That
kernel addresses KV as one pool of rows plus a CSR index list per query row, so
the `[window ++ draft-block]` KV DSpark attends to has to be expressed as pool
row ids rather than a materialised `[B, W+T, 512]` tensor.

`DSparkIndexBuffers.build` fills, in one launch:

- `kv_indices` / `kv_indptr` — the ragged CSR, one query row per draft position,
  `N = B*T` rows.
- `draft_rows` — the ring rows the draft block's own KV is scattered into, fused
  into the `qk_norm_rope_maybe_quant` launch that produces it.

and `DSparkIndexBuffers.views` reads them back.

`qo_indptr` is a constant riding in the same `DSparkIndexBuffers` bundle; the
scatter's `batch_ids` is filled there too but follows how much of the batch is
real, via `mask_pad_tail`. Everything is allocated once at `max_num_seqs` and
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
    `_init_draft_block_buffers` already follows. Nothing is keyed by shape,
    because every one of these is prefix-stable in the batch: the first `B*T`
    entries of the max-batch buffer ARE the batch-`B` answer.

    `draft_width` / `draft_window` are the shape the buffers were cut for, kept
    here so no caller can hand a slice a different one; `qo_indptr` is a
    constant filled at allocation and never written again. `built_for` is the
    batch `kv_indices` / `kv_indptr` / `draft_rows` currently hold; `batch_ids`
    additionally follows how much of that batch is real, via `mask_pad_tail`.
    """

    kv_indices: torch.Tensor  # [max_b*T*(W+T)] int32
    kv_indptr: torch.Tensor  # [max_b*T+1] int32
    draft_rows: torch.Tensor  # [max_b*T] int32
    qo_indptr: torch.Tensor  # [max_b*T+1] int32, constant ramp
    batch_ids: torch.Tensor  # [max_b*T] int32, [0]*T ++ [1]*T ++ ..., -1 on pad
    max_batch: int
    draft_width: int  # T
    draft_window: int  # W
    built_for: int = -1

    @classmethod
    def allocate(
        cls, max_batch: int, draft: int, window: int, device
    ) -> DSparkIndexBuffers:
        """Allocate the bundle for the largest batch (`draft` = T, `window` = W).

        `qo_indptr` is `arange(N+1)`: the asm wrapper runs `max_seqlen_q = 1`, one
        "sequence" per query row -- the same per-token convention the V4 target uses
        for its own decode AND verify forwards (`deepseek_v4_attn.py:3727`).

        `batch_ids` is the token -> request map the fused SWA scatter gates on
        (`bid >= 0`); `mask_pad_tail` rewrites it when a batch is padded. Filled
        here rather than left ``empty`` because the startup sweep runs the block
        first, and a buffer whose contract gives `-1` a meaning should not start
        out undefined. The target can alias `cu_seqlens_q[:bs]` for its own
        (`deepseek_v4_attn.py:2348`) because at one token per sequence that slice
        is already `arange(bs)`; DSpark runs T tokens per request and needs each
        id repeated T times, so there is nothing to alias.
        """
        n = max_batch * draft
        i32 = {"dtype": torch.int32, "device": device}
        return cls(
            kv_indices=torch.empty(n * (window + draft), **i32),
            kv_indptr=torch.empty(n + 1, **i32),
            draft_rows=torch.empty(n, **i32),
            qo_indptr=torch.arange(n + 1, **i32),
            batch_ids=torch.arange(n, **i32) // draft,
            max_batch=max_batch,
            draft_width=draft,
            draft_window=window,
        )

    def mask_pad_tail(self, row_ids: torch.Tensor, real_batch: int, batch: int) -> None:
        """Sentinel the rows past `real_batch`, so the fused SWA write skips them.

        Restores the prefix as well as marking the tail: a sentinel left on a row
        that has since become real would drop that request's draft KV silently.
        `row_ids` is the builder's `arange`, broadcast one id across the row's T
        tokens -- one strided copy, no repeated map to keep anywhere.
        """
        t = self.draft_width
        self.batch_ids[: real_batch * t].view(real_batch, t).copy_(
            row_ids[:real_batch].unsqueeze(1)
        )
        self.batch_ids[real_batch * t : batch * t].fill_(-1)

    def build(
        self,
        window: WindowParams,
        slots: torch.Tensor,  # [B] per-request ring slot
        anchors: torch.Tensor,  # [B] per-request anchor position
    ) -> None:
        """Fill this bundle for one block, in one launch. Read it with :meth:`views`.

        `T` and `W` come from the bundle, so the slices can never be cut to a shape
        the buffers were not allocated for.
        """
        B = anchors.shape[0]
        T, W = self.draft_width, self.draft_window
        if B > self.max_batch:
            raise ValueError(
                f"DSpark index buffers hold {self.max_batch} requests < B={B}; "
                "they are sized at max_num_seqs."
            )
        kv_indices, kv_indptr, draft_rows = self._ragged(B)

        _dspark_index_kernel[(B * T,)](
            anchors.to(torch.int64),
            slots.to(torch.int64),
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
        # Last: a launch that raised must not leave the bundle claiming a batch.
        self.built_for = B

    def views(self, batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The `(kv_indices, kv_indptr, draft_rows)` this built bundle holds.

        The one accessor: :meth:`build` only fills, and every stage -- including the
        one that filled -- reads through here. DSpark runs one block through every
        stage and the values are stage-invariant (all DSpark layers share compress
        ratio 0, hence one `WindowParams`, and each layer's plane view is
        base-row-relative), so stage 0 builds and the rest just slice.
        """
        if self.built_for != batch:
            raise RuntimeError(
                f"DSpark kv indices hold batch {self.built_for}, not {batch}; "
                "stage 0 must build them before any stage reads them back."
            )
        return self._ragged(batch)

    def _ragged(self, batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three built buffers sliced to `batch`, unchecked (see :meth:`views`)."""
        n = batch * self.draft_width
        return (
            self.kv_indices[: n * (self.draft_window + self.draft_width)],
            self.kv_indptr[: n + 1],
            self.draft_rows[:n],
        )
