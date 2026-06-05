# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plugin-local Triton ops for the vLLM DeepSeek V4 bridge.

Kernels here are specific to the bridge's proxy KV-cache layout (the unified
``swa_pages + block_id`` compress region) rather than to native ATOM, so they
live alongside the bridge instead of in ``atom.model_ops.v4_kernels``.

``write_v4_decode_hca_compress_tail`` is the decode-time companion to
``atom.model_ops.v4_kernels.write_v4_paged_decode_indices``: that core kernel
fills the SWA window prefix shared by the SWA / CSA / HCA index buffers; this
one appends the HCA compress tail straight from the GPU block table, so the
whole decode HCA index segment is built on-GPU with no per-step host round trip.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_decode_hca_compress_tail_kernel(
    batch_id_per_token_ptr,  # [>=T] int — sentinel -1 in CG pad tail
    positions_ptr,  # [>=T] int — global token position
    hca_indptr_ptr,  # [>=T+1] int32 — ragged (SWA prefix + HCA committed)
    n_committed_hca_per_seq_ptr,  # [num_reqs] int32 — per-seq HCA entry count
    block_tables_ptr,  # [num_reqs, MAX_BLOCKS] int — per-seq paged block ids
    bt_stride_bs,  # block_tables row stride (elements)
    hca_indices_ptr,  # [>=hca_indptr[T]] int32 OUT — HCA compress tail written
    swa_pages,  # num_slots * cs — boundary into the compress region
    win: tl.constexpr,  # SWA window — per-token prefix length cap
    BLOCK_J: tl.constexpr,  # next_pow2(win) — HCA loop chunk size
):
    """One program per token. Writes the HCA *compress-tail* segment
    ``[hca_indptr[t] + n_swa, +n_hca)`` where ``n_swa = min(pos+1, win)`` is the
    SWA window-prefix length (filled separately by
    ``write_v4_paged_decode_indices``) and the j-th committed HCA entry maps to
    physical page ``swa_pages + block_tables[bid, j]``.

    The window-prefix region ``[hca_indptr[t], +n_swa)`` is left untouched here
    (the sibling kernel writes it); together they cover the full per-token HCA
    segment ``[hca_indptr[t], hca_indptr[t+1])``, so no ``-1`` pre-fill is
    needed. CG-padded tail tokens (batch_id == -1) carry a zero-length segment
    and are skipped.

    Decode analogue of the HCA compress section in
    ``_v4_paged_prefill_indices_kernel`` (``prefix_swa_count`` there is the
    decode ``n_swa`` here).
    """
    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return  # CG-padded sentinel — leave outputs untouched
    pos = tl.load(positions_ptr + t)
    n_swa = tl.minimum(pos + 1, win)
    n_hca = tl.load(n_committed_hca_per_seq_ptr + bid)
    base = tl.load(hca_indptr_ptr + t) + n_swa
    bt_row_base = bid * bt_stride_bs
    i = tl.arange(0, BLOCK_J)
    for j in tl.range(0, n_hca, BLOCK_J):
        k = j + i
        mask = k < n_hca
        bt = tl.load(block_tables_ptr + bt_row_base + k, mask=mask, other=0)
        tl.store(hca_indices_ptr + base + k, swa_pages + bt, mask=mask)


def write_v4_decode_hca_compress_tail(
    *,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    hca_indptr: torch.Tensor,
    n_committed_hca_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    hca_indices: torch.Tensor,
    T: int,
    win: int,
    swa_pages: int,
) -> None:
    """In-place GPU fill of the decode HCA compress-tail paged offsets.

    Companion to ``write_v4_paged_decode_indices``: that kernel fills the SWA
    window-prefix of the SWA / CSA / HCA index buffers; this one appends the HCA
    compress tail (``swa_pages + block_tables[bid, j]``) after each token's SWA
    prefix in ``hca_indices``. Together they write the full per-token HCA
    segment, so the caller need not pre-fill ``-1``.

    Replaces the prior CPU scatter in the bridge's decode metadata build
    (block-table D2H + numpy ``repeat``/``cumsum``/fancy-index + H2D), so the
    whole HCA index segment is produced on-GPU with no per-step host round trip
    — matching the existing on-GPU prefill build.

    All tensors are GPU tensors. Per-seq inputs are indexed by
    ``batch_id_per_token`` inline (no caller pre-gather).

    Args:
      batch_id_per_token:      ``[>=T]``   int — token→seq map; -1 skipped.
      positions:               ``[>=T]``   int — global token positions; the
                                           SWA prefix length is ``min(pos+1, win)``.
      hca_indptr:              ``[>=T+1]`` int32 — ragged HCA indptr (same one
                                           passed to ``write_v4_paged_decode_indices``).
      n_committed_hca_per_seq: ``[num_reqs]`` int32 — per-seq HCA entry count.
      block_tables:            ``[num_reqs, mnbs]`` int — per-seq paged blocks.
      hca_indices:             ``[>=hca_indptr[T]]`` int32 OUT — compress tail
                                           written; SWA prefix left to sibling.
      T:                       int — real token count (grid size).
      win:                     int — SWA window size.
      swa_pages:               int — ``num_slots * cs`` boundary in unified_kv.
    """
    if T == 0:
        return
    assert batch_id_per_token.dim() == 1 and batch_id_per_token.shape[0] >= T
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert hca_indptr.dim() == 1 and hca_indptr.shape[0] >= T + 1
    assert n_committed_hca_per_seq.dim() == 1
    assert block_tables.dim() == 2
    assert hca_indices.dim() == 1

    BLOCK_J = triton.next_power_of_2(win)
    _v4_decode_hca_compress_tail_kernel[(T,)](
        batch_id_per_token,
        positions,
        hca_indptr,
        n_committed_hca_per_seq,
        block_tables,
        block_tables.stride(0),
        hca_indices,
        swa_pages,
        win=win,
        BLOCK_J=BLOCK_J,
    )


def write_v4_decode_hca_compress_tail_reference(
    *,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    hca_indptr: torch.Tensor,
    n_committed_hca_per_seq: torch.Tensor,
    block_tables: torch.Tensor,
    hca_indices: torch.Tensor,
    T: int,
    win: int,
    swa_pages: int,
) -> None:
    """Pure-PyTorch equivalent of ``write_v4_decode_hca_compress_tail``.
    For unit-test bit-exact verification and dump-bisect debugging. Mirrors the
    kernel: per-token compress tail written at ``[hca_indptr[t] + n_swa, +n_hca)``;
    the SWA prefix region is left untouched; -1 batch_id tokens skipped.
    """
    if T == 0:
        return
    bid_cpu = batch_id_per_token[:T].cpu().tolist()
    pos_cpu = positions[:T].cpu().tolist()
    hca_indptr_cpu = hca_indptr.cpu().tolist()
    n_hca_cpu = n_committed_hca_per_seq.cpu().tolist()
    bt_cpu = block_tables.cpu()
    device = hca_indices.device
    for t in range(T):
        bid = int(bid_cpu[t])
        if bid < 0:
            continue
        n_swa = min(int(pos_cpu[t]) + 1, win)
        n_hca = int(n_hca_cpu[bid])
        if n_hca <= 0:
            continue
        base = int(hca_indptr_cpu[t]) + n_swa
        bt = bt_cpu[bid, :n_hca].to(device).to(hca_indices.dtype)
        hca_indices[base : base + n_hca] = swa_pages + bt
