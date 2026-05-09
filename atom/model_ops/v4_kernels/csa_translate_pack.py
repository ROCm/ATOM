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

Per-token write offset (`skip_prefix_len_per_token[t]`) accommodates the
two-source paged_prefill layout:
  - decode:           skip = window_size       (full SWA prefix)
  - pure prefill:     skip = 0                 (no SWA history in `unified_kv`)
  - chunked prefill:  skip = prior_swa_count   (variable per-token)

Correctness equivalence with the prior path:
  - paged_decode reads `kv_indices_csa[indptr[t] : indptr[t+1]]` whose length
    is exactly `skip_prefix_len_per_token[t] + n_committed_csa[bid]`. The
    fused kernel writes only `[0, valid_k)` (mask
    `(k < valid_k) & (k < index_topk)`); the tail `[valid_k, index_topk)` is
    never read downstream, so no `-1` fill is needed. CG-padded slots
    (batch_id=-1) bail in the kernel preamble. Indexer raw output may
    contain -1 in tail cols (kernel-native sentinel) — those cells are
    masked off via `(topk >= 0)` so we never write a garbage paged offset.
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
    skip_prefix_len_per_token_ptr,  # [T] int32 — per-token write offset
    kv_indices_csa_ptr,  # [total_indices] int32 — destination
    swa_pages,  # i32 — SWA region size, runtime int
    mnbps,  # i32 — max blocks per seq, runtime int
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
    # Indexer's raw seq-local output uses -1 as the kernel-native sentinel
    # for tail cols where no candidate was selected (per-token visibility
    # cap < n_committed_csa[seq]). Skip writes for negative entries so we
    # don't translate a garbage row into a paged offset.
    valid_topk = in_range & (topk >= 0)

    # Translate seq-local row → physical paged offset.
    blk_idx = topk // csa_block_capacity
    slot = topk % csa_block_capacity
    # Defensive clamp: keep block_tables gather in-bounds even on masked-off
    # lanes (triton speculatively computes addresses).
    blk_safe = tl.minimum(tl.maximum(blk_idx, 0), mnbps - 1)
    phys = tl.load(
        block_tables_ptr + bid * mnbps + blk_safe,
        mask=valid_topk,
        other=0,
    )
    paged = swa_pages + phys * csa_block_capacity + slot

    skip = tl.load(skip_prefix_len_per_token_ptr + pid_t)
    write_base = tl.load(kv_indptr_csa_ptr + pid_t) + skip
    tl.store(
        kv_indices_csa_ptr + write_base + k_offs,
        paged,
        mask=valid_topk,
    )


def csa_translate_pack(
    topk_local: torch.Tensor,
    block_tables: torch.Tensor,
    n_committed_csa_per_seq: torch.Tensor,
    kv_indptr_csa: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    skip_prefix_len_per_token: torch.Tensor,
    kv_indices_csa: torch.Tensor,
    *,
    swa_pages: int,
    csa_block_capacity: int,
) -> None:
    """Fused topk translate + packed write into `kv_indices_csa` (in-place).

    Args:
      topk_local:                  [T, index_topk] int32 — indexer's seq-local
                                   row indices; tails may carry -1 sentinels
                                   (kernel-native; we skip writes on those).
      block_tables:                [bs, mnbps] int32 — logical block → physical.
      n_committed_csa_per_seq:     [bs] int32 — RAW per-seq committed count
                                   (`ctx_len // 4`). The kernel clamps
                                   internally via
                                   `(k < valid_k) & (k < index_topk)` mask;
                                   callers MUST pass the raw value (NOT
                                   pre-clamped to index_topk), since the
                                   indexer also reads this same buffer and
                                   needs the raw count. Indexed by
                                   `batch_id_per_token[t]`.
      kv_indptr_csa:               [T+1] int32 — per-token packed cumsum
                                   (CG-padded: tail repeats last value
                                   → kv_len=0).
      batch_id_per_token:          [T] int32 — token → seq, sentinel -1 for
                                   CG-padded slots.
      skip_prefix_len_per_token:   [T] int32 — per-token write offset within
                                   `kv_indices_csa[indptr[t] : indptr[t+1]]`
                                   the CSA section starts at. Decode passes
                                   `[window_size, ...]` (full SWA prefix);
                                   pure prefill passes 0; chunked prefill
                                   passes `prior_swa_count_per_token`.
      kv_indices_csa:              [total_indices] int32 — destination buffer;
                                   this kernel writes the CSA section
                                   `[indptr[t]+skip[t],
                                     indptr[t]+skip[t]+min(valid_k, index_topk))`.
      swa_pages:                   SWA region size — `num_slots * window_size`,
                                   fixed at CG capture time. Keyword-only.
      csa_block_capacity:          `block_size // ratio = 128 // 4 = 32`
                                   (constexpr; triton can strength-reduce
                                   // and %). Keyword-only.
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
    if skip_prefix_len_per_token.numel() < T:
        raise ValueError(
            "skip_prefix_len_per_token.numel()="
            f"{skip_prefix_len_per_token.numel()} < T={T}"
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
        skip_prefix_len_per_token,
        kv_indices_csa,
        swa_pages,
        mnbps,
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
    skip_prefix_len_per_token: torch.Tensor,
    kv_indices_csa: torch.Tensor,
    *,
    swa_pages: int,
    csa_block_capacity: int,
) -> None:
    """Pure-torch reference. Mirrors the original PyTorch chain + packed write.

    Same contract as `csa_translate_pack`: `n_committed_csa_per_seq` is the
    raw per-seq count; this reference also clamps via `min(n, index_topk)`
    to match the kernel's mask behavior. Negative `topk_local` entries
    (kernel-native -1 sentinels) are skipped to mirror the kernel's
    `topk >= 0` write mask.
    """
    T, index_topk = topk_local.shape
    indptr = kv_indptr_csa.to(torch.int64)
    counts = n_committed_csa_per_seq.to(torch.int64)
    bids = batch_id_per_token.to(torch.int64)
    skips = skip_prefix_len_per_token.to(torch.int64)
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
        valid = topk >= 0  # mirror kernel's per-cell `topk >= 0` write mask
        blk_idx = (topk // csa_block_capacity).clamp(0, mnbps - 1)
        slot = topk % csa_block_capacity
        phys = block_tables[bid, blk_idx].to(torch.int64)
        paged = swa_pages + phys * csa_block_capacity + slot
        base = int(indptr[t].item()) + int(skips[t].item())
        # Scatter only at cells where valid; cells with topk<0 keep their
        # prior value (matches the kernel's masked tl.store).
        for k in range(n):
            if bool(valid[k].item()):
                kv_indices_csa[base + k] = int(paged[k].item())
