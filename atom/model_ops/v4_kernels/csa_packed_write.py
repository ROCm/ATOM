# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CSA packed-indices write kernel.

V4 CSA's per-token KV section is packed-cumsum (variable per-token kv_len),
so the indexer's per-layer `topk_local` cannot be written via a simple slice.
Slice writes would overflow into the next token's region whenever
`valid_count[t] < index_topk`. This kernel does the per-token bounded write
via a fixed grid `(T, ceil(index_topk / BLOCK_K))` so CUDAGraph sees a
shape-stable launch.

Caller pre-translates `topk_local` → physical paged offsets and clamps
`valid_count_per_seq = min(n_committed_csa, index_topk)` (per-seq, NOT
per-token: ctx_len is sequence-level, so MTP base/draft tokens of one req
share the same valid_count). The kernel reads
`valid_count_per_seq[batch_id_per_token[t]]` to keep the only per-token
indirection on the single shared `batch_id_per_token` mapping. See
`atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_DECODE_DESIGN.en.md` §6.4 (or
`.zh.md` for the original Chinese).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _csa_packed_write_kernel(
    paged_compress_ptr,  # [T, index_topk] int32 — translated paged offsets
    kv_indices_ptr,  # [total_indices_csa] int32 — destination buffer
    kv_indptr_ptr,  # [T+1] int32 — packed cumsum
    valid_count_per_seq_ptr,  # [bs] int32 — per-seq valid topk count
    batch_id_per_token_ptr,  # [T] int32 — token → seq mapping
    window_size: tl.constexpr,
    index_topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_kb = tl.program_id(1)

    bid = tl.load(batch_id_per_token_ptr + pid_t)
    # CG-padded slot sentinel: builder fills [actual_T:padded_T] with -1 so
    # the captured kernel grid (= padded_T) bails on padded entries instead
    # of dereferencing valid_count_per_seq with a negative index.
    if bid < 0:
        return
    valid_k = tl.load(valid_count_per_seq_ptr + bid)

    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    in_range = (k_offs < valid_k) & (k_offs < index_topk)

    src = tl.load(
        paged_compress_ptr + pid_t * index_topk + k_offs,
        mask=in_range,
        other=0,
    )

    write_base = tl.load(kv_indptr_ptr + pid_t) + window_size
    tl.store(
        kv_indices_ptr + write_base + k_offs,
        src,
        mask=in_range,
    )


def csa_packed_write(
    paged_compress: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr_csa: torch.Tensor,
    valid_count_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    window_size: int,
    index_topk: int,
) -> None:
    """In-place packed write of per-token CSA topk into the CSA kv_indices buffer.

    Args:
      paged_compress:       [T, index_topk] int32 — already-translated physical
                            paged offsets (caller adds
                            `swa_pages + block_id * cap + slot`). Entries past
                            `valid_count_per_seq[batch_id_per_token[t]]` are
                            ignored.
      kv_indices:           [total_indices_csa] int32 — destination buffer;
                            written in-place at slots
                            `[indptr[t]+window_size, indptr[t]+window_size+vk)`
                            where `vk = valid_count_per_seq[batch_id[t]]`.
      kv_indptr_csa:        [T+1] int32 — packed cumsum of per-token CSA kv_len.
      valid_count_per_seq:  [bs] int32 — `min(n_committed_csa, index_topk)`,
                            looked up by `batch_id_per_token[t]`.
      batch_id_per_token:   [T] int32 — the single per-token mapping shared
                            across Phase B/C/E.
      window_size:          SWA window slots reserved at the head of each
                            per-token CSA section.
      index_topk:           indexer's topk width (constexpr — must be the same
                            for all CG-captured shapes).
    """
    T = paged_compress.size(0)
    if T == 0:
        return

    if paged_compress.size(1) != index_topk:
        raise ValueError(
            f"paged_compress.size(1)={paged_compress.size(1)} != index_topk={index_topk}"
        )
    # kv_indptr / batch_id_per_token may be sized to a CG-padded T (>= T).
    # Kernel grid uses paged_compress.size(0) as authoritative; tail entries
    # in those buffers are sentinel-skipped (batch_id=-1) inside the kernel.
    if kv_indptr_csa.numel() < T + 1:
        raise ValueError(f"kv_indptr_csa.numel()={kv_indptr_csa.numel()} < T+1={T + 1}")
    if batch_id_per_token.numel() < T:
        raise ValueError(
            f"batch_id_per_token.numel()={batch_id_per_token.numel()} < T={T}"
        )

    BLOCK_K = min(64, triton.next_power_of_2(index_topk))
    grid = (T, triton.cdiv(index_topk, BLOCK_K))
    _csa_packed_write_kernel[grid](
        paged_compress,
        kv_indices,
        kv_indptr_csa,
        valid_count_per_seq,
        batch_id_per_token,
        window_size=window_size,
        index_topk=index_topk,
        BLOCK_K=BLOCK_K,
    )


def csa_packed_write_reference(
    paged_compress: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr_csa: torch.Tensor,
    valid_count_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    window_size: int,
) -> None:
    """Pure-torch reference. In-place write into `kv_indices`."""
    T = paged_compress.size(0)
    indptr = kv_indptr_csa.to(torch.int64)
    counts_per_seq = valid_count_per_seq.to(torch.int64)
    bids = batch_id_per_token.to(torch.int64)
    for t in range(T):
        bid = int(bids[t].item())
        if bid < 0:  # CG-padded sentinel
            continue
        n = int(counts_per_seq[bid].item())
        if n == 0:
            continue
        base = int(indptr[t].item()) + window_size
        kv_indices[base : base + n] = paged_compress[t, :n].to(kv_indices.dtype)
