# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused CSA topk translate + packed-write kernel.

Replaces the chain (per CSA layer, per fwd):

    block_idx = topk_local // csa_block_capacity
    slot      = topk_local %  csa_block_capacity
    safe_bid  = batch_id.clamp(0).long()
    safe_blk  = block_idx.long().clamp(0, mnbps-1)
    phys      = block_tables[safe_bid_expanded, safe_blk]      # fancy index
    paged     = swa_pages + phys * csa_block_capacity + slot
    packed_write(paged, kv_indices_csa, kv_indptr_csa, ...)     # triton

with a single triton kernel that does the indexer-topk → paged offset
translation + bounded packed write entirely in registers — no per-layer
[T, K] intermediates, no fancy index, no separate launch.

CG benefits (V4-Pro: 31 CSA layers per fwd):
  - 0 transient [T, K] tensor allocs (was 5+/layer × 31 → 155+/fwd)
  - 1 captured graph node per layer instead of 7-8

Correctness equivalence with the prior path:
  - paged_decode reads `kv_indices_csa[indptr[t] : indptr[t+1]]` whose length
    is exactly `window_size + n_committed_csa[bid]`. The fused kernel writes
    only `[0, valid_k)` (mask `(k < valid_k) & (k < index_topk)`); the tail
    `[valid_k, index_topk)` is never read downstream, so no `-1` fill is
    needed. CG-padded slots (batch_id=-1) bail in the kernel preamble.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _csa_translate_pack_kernel(
    topk_local_ptr,  # [T, index_topk] int32 — indexer raw output
    block_tables_ptr,  # [bs, mnbps] int32 — page table
    n_committed_csa_per_seq_ptr,  # [bs] int32 — per-seq valid count
    kv_indptr_csa_ptr,  # [T+1] int32 — packed cumsum (per-token)
    batch_id_per_token_ptr,  # [T] int32 — token → seq, sentinel -1
    kv_indices_csa_ptr,  # [total_indices] int32 — destination
    swa_pages,  # i32 — SWA region size, runtime int
    mnbps,  # i32 — max blocks per seq, runtime int
    window_size: tl.constexpr,
    index_topk: tl.constexpr,
    csa_block_capacity: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_kb = tl.program_id(1)

    # CG-padded slot sentinel: builder fills [actual_T:padded_T] with -1
    # so the captured kernel grid (= padded_T) bails on padded entries.
    bid = tl.load(batch_id_per_token_ptr + pid_t)
    if bid < 0:
        return

    valid_k = tl.load(n_committed_csa_per_seq_ptr + bid)

    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    in_range = (k_offs < valid_k) & (k_offs < index_topk)

    topk = tl.load(
        topk_local_ptr + pid_t * index_topk + k_offs,
        mask=in_range,
        other=0,
    )

    # Translate seq-local row → physical paged offset.
    blk_idx = topk // csa_block_capacity
    slot = topk % csa_block_capacity
    # Defensive clamp: indexer may emit garbage in CG-padded tail cols
    # (within the per-token row, beyond `valid_k`). `in_range` already
    # masks those, but clamp keeps the gather pointer in-bounds even
    # when triton speculatively computes addresses for masked-off lanes.
    blk_safe = tl.minimum(tl.maximum(blk_idx, 0), mnbps - 1)
    phys = tl.load(
        block_tables_ptr + bid * mnbps + blk_safe,
        mask=in_range,
        other=0,
    )
    paged = swa_pages + phys * csa_block_capacity + slot

    write_base = tl.load(kv_indptr_csa_ptr + pid_t) + window_size
    tl.store(
        kv_indices_csa_ptr + write_base + k_offs,
        paged,
        mask=in_range,
    )


def csa_translate_pack(
    topk_local: torch.Tensor,
    block_tables: torch.Tensor,
    n_committed_csa_per_seq: torch.Tensor,
    kv_indptr_csa: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    kv_indices_csa: torch.Tensor,
    *,
    swa_pages: int,
    window_size: int,
    csa_block_capacity: int,
) -> None:
    """Fused topk translate + packed write into `kv_indices_csa` (in-place).

    Args:
      topk_local:               [T, index_topk] int32 — indexer's seq-local
                                row indices, may contain garbage beyond
                                `valid_k` and -1 in tails.
      block_tables:             [bs, mnbps] int32 — logical block → physical.
      n_committed_csa_per_seq:  [bs] int32 — RAW per-seq committed count
                                (`ctx_len // 4`). The kernel clamps internally
                                via `(k < valid_k) & (k < index_topk)` mask;
                                callers MUST pass the raw value (NOT
                                pre-clamped to index_topk), since the indexer
                                also reads this same buffer and needs the raw
                                count. Indexed by `batch_id_per_token[t]`.
      kv_indptr_csa:            [T+1] int32 — per-token packed cumsum
                                (CG-padded: tail repeats last value → kv_len=0).
      batch_id_per_token:       [T] int32 — token → seq, sentinel -1 for
                                CG-padded slots.
      kv_indices_csa:           [total_indices] int32 — destination buffer;
                                this kernel writes the CSA section
                                `[indptr[t]+window_size, indptr[t]+window_size+min(valid_k, index_topk))`.
      swa_pages:                SWA region size — `num_slots * window_size`,
                                fixed at CG capture time. Keyword-only.
      window_size:              SWA window slots reserved at the head of each
                                per-token region (constexpr). Keyword-only.
      csa_block_capacity:       `block_size // ratio = 128 // 4 = 32` (constexpr;
                                triton can strength-reduce // and %). Keyword-only.
    """
    T, index_topk = topk_local.shape
    if T == 0:
        return

    if kv_indptr_csa.numel() < T + 1:
        raise ValueError(f"kv_indptr_csa.numel()={kv_indptr_csa.numel()} < T+1={T + 1}")
    if batch_id_per_token.numel() < T:
        raise ValueError(
            f"batch_id_per_token.numel()={batch_id_per_token.numel()} < T={T}"
        )
    mnbps = block_tables.size(1)

    BLOCK_K = min(64, triton.next_power_of_2(index_topk))
    grid = (T, triton.cdiv(index_topk, BLOCK_K))
    _csa_translate_pack_kernel[grid](
        topk_local,
        block_tables,
        n_committed_csa_per_seq,
        kv_indptr_csa,
        batch_id_per_token,
        kv_indices_csa,
        swa_pages,
        mnbps,
        window_size=window_size,
        index_topk=index_topk,
        csa_block_capacity=csa_block_capacity,
        BLOCK_K=BLOCK_K,
    )


def csa_translate_pack_reference(
    topk_local: torch.Tensor,
    block_tables: torch.Tensor,
    n_committed_csa_per_seq: torch.Tensor,
    kv_indptr_csa: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    kv_indices_csa: torch.Tensor,
    *,
    swa_pages: int,
    window_size: int,
    csa_block_capacity: int,
) -> None:
    """Pure-torch reference. Mirrors the original PyTorch chain + packed write.

    Same contract as `csa_translate_pack`: `n_committed_csa_per_seq` is the
    raw per-seq count; this reference also clamps via `min(n, index_topk)`
    to match the kernel's mask behavior.
    """
    T, index_topk = topk_local.shape
    indptr = kv_indptr_csa.to(torch.int64)
    counts = n_committed_csa_per_seq.to(torch.int64)
    bids = batch_id_per_token.to(torch.int64)
    mnbps = block_tables.size(1)
    for t in range(T):
        bid = int(bids[t].item())
        if bid < 0:
            continue
        # Mirror kernel's `(k < valid_k) & (k < index_topk)` clamp.
        n = min(int(counts[bid].item()), index_topk)
        if n == 0:
            continue
        topk = topk_local[t, :n].to(torch.int64)
        blk_idx = (topk // csa_block_capacity).clamp(0, mnbps - 1)
        slot = topk % csa_block_capacity
        phys = block_tables[bid, blk_idx].to(torch.int64)
        paged = swa_pages + phys * csa_block_capacity + slot
        base = int(indptr[t].item()) + window_size
        kv_indices_csa[base : base + n] = paged.to(kv_indices_csa.dtype)
