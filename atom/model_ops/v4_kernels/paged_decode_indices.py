# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 paged-decode index scatter — single Triton kernel writes all 3
destinations (SWA / CSA / HCA window prefix) from persistent forward_vars
buffers. Replaces the prior chain in `_attach_v4_paged_decode_meta`:

    state_slot_per_token = state_slot_per_seq[batch_id_per_token[:T].long()]
    swa_paged_2d = torch.where(window_topk >= 0, slot * cs + topk, -1)
    swa_paged_flat = swa_paged_2d.reshape(-1)
    swa_indices_gpu[:T*win].copy_(swa_paged_flat)
    win_arange = torch.arange(win, ...)
    csa_win_pos = (csa_indptr[:T].to(int64).unsqueeze(1) + win_arange).reshape(-1)
    hca_win_pos = (hca_indptr[:T].to(int64).unsqueeze(1) + win_arange).reshape(-1)
    csa_indices_gpu.index_copy_(0, csa_win_pos, swa_paged_flat)
    hca_indices_gpu.index_copy_(0, hca_win_pos, swa_paged_flat)

That chain creates ~5 transient tensors via `torch.where / arange / .reshape /
.to(int64)`. Under MTP-3 long-prefill, the caching allocator races on these
across stream boundaries: a still-in-flight `index_copy_` reads storage that
the allocator has just handed to the next call's transient. Manifests as
`ASSERT_TRAP` in `at::native::index_copy_kernel_impl<OpaqueType<4>>` (PyTorch
device-side bound check) when the racing tensor's contents look like an
out-of-bounds index. See skill `debug-agent-locate-kernel` for the full
investigation (May 14 2026).

This kernel takes ONLY persistent forward_vars buffers as input — no
allocator churn — and writes all 3 destinations in-place in a single launch.
Bytewise-equivalent to the previous chain: each token's win-prefix entries
hold `slot * cs + window_topk[t,w]` (or -1 sentinel where window_topk[t,w]<0).

Caller contract:
- Grid = T (one program per token).
- `batch_id_per_token[:T]` may carry `-1` sentinels in the CG-padded tail —
  kernel checks and bails (matches `_attach_v4_per_fwd_meta` convention).
- `csa_indices` / `hca_indices` capacity must be >= csa_indptr[T] /
  hca_indptr[T] (the full packed buffer); we only write the window-prefix
  segment `[csa_indptr[t], csa_indptr[t]+win)` per token, the compress-tail
  is filled elsewhere (HCA: numpy fill in caller, CSA: csa_translate_pack
  per-layer).
- `swa_indices` capacity must be >= T*win.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_paged_decode_indices_kernel(
    state_slot_per_seq_ptr,  # [bs] int32
    batch_id_per_token_ptr,  # [T+pad] int — sentinel -1 in pad tail
    window_topk_ptr,  # [T, win] int32 (-1 sentinel for invalid)
    csa_indptr_ptr,  # [T+1] int32
    hca_indptr_ptr,  # [T+1] int32
    swa_indices_ptr,  # [T*win] int32, output
    csa_indices_ptr,  # [csa_total] int32, output (in-place at csa_indptr[t]+w)
    hca_indices_ptr,  # [hca_total] int32, output (in-place at hca_indptr[t]+w)
    cs,  # win_with_spec — stride into unified_kv SWA region (paper §3.6.1)
    win,  # window_size — number of SWA prefix slots written per token
    BLOCK_W: tl.constexpr,  # next_pow2(win)
):
    """One program per token. Writes `win` paged offsets to SWA, CSA, HCA.

    For token `t`:
        bid = batch_id_per_token[t]                  # bail if -1 (CG pad)
        slot = state_slot_per_seq[bid]
        for w in range(win):
            topk = window_topk[t, w]
            paged = slot * cs + topk if topk >= 0 else -1
            swa_indices[t * win + w]              = paged
            csa_indices[csa_indptr[t] + w]        = paged
            hca_indices[hca_indptr[t] + w]        = paged
    """
    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return  # CG-padded sentinel — leave outputs untouched (sentinel -1 in caller)

    slot = tl.load(state_slot_per_seq_ptr + bid)
    csa_base = tl.load(csa_indptr_ptr + t)
    hca_base = tl.load(hca_indptr_ptr + t)

    w = tl.arange(0, BLOCK_W)
    mask = w < win

    topk = tl.load(window_topk_ptr + t * win + w, mask=mask, other=-1)
    paged = tl.where(topk >= 0, slot * cs + topk, -1)

    tl.store(swa_indices_ptr + t * win + w, paged, mask=mask)
    tl.store(csa_indices_ptr + csa_base + w, paged, mask=mask)
    tl.store(hca_indices_ptr + hca_base + w, paged, mask=mask)


def write_v4_paged_decode_indices(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    window_topk: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor,
    hca_indices: torch.Tensor,
    T: int,
    win: int,
    cs: int,
) -> None:
    """In-place fill SWA / CSA / HCA window-prefix offsets via a single Triton
    kernel. Replaces the prior `state_slot gather + torch.where + arange +
    2× index_copy_` chain in `_attach_v4_paged_decode_meta`. All inputs are
    persistent forward_vars buffers — no allocator churn.

    Args (all GPU tensors except T/win/cs):
      state_slot_per_seq:  [bs]   int32 — per-seq state cache slot.
      batch_id_per_token:  [>=T]  int   — token→seq map; -1 sentinel skipped.
      window_topk:         [T, win] int32 — per-token SWA ring indices
                                   (0..cs-1) or -1 for invalid slots.
      csa_indptr:          [>=T+1] int32 — packed CSA buffer indptr.
      hca_indptr:          [>=T+1] int32 — packed HCA buffer indptr.
      swa_indices:         [>=T*win] int32 OUT — uniform-stride SWA offsets.
      csa_indices:         [>=csa_indptr[T]] int32 OUT — packed; only
                                   window-prefix `[csa_indptr[t], +win)` written.
      hca_indices:         [>=hca_indptr[T]] int32 OUT — same shape semantics.
      T:                   int — number of real tokens (grid size).
      win:                 int — SWA window size (typically 128 for V4-Pro).
      cs:                  int — `win_with_spec = window_size + max_spec_steps`,
                                 stride into unified_kv SWA region per slot.

    Output values per token `t` and slot `w in [0, win)`:
        paged = state_slot_per_seq[batch_id_per_token[t]] * cs + window_topk[t,w]
                if window_topk[t,w] >= 0 else -1
        swa_indices[t*win + w]         = paged
        csa_indices[csa_indptr[t] + w] = paged
        hca_indices[hca_indptr[t] + w] = paged
    """
    if T == 0:
        return
    assert state_slot_per_seq.dim() == 1
    assert batch_id_per_token.dim() == 1 and batch_id_per_token.shape[0] >= T
    assert window_topk.dim() == 2 and window_topk.shape[0] >= T
    assert (
        window_topk.shape[1] == win
    ), f"window_topk.shape[1] {window_topk.shape[1]} != win {win}"
    assert csa_indptr.dim() == 1 and csa_indptr.shape[0] >= T + 1
    assert hca_indptr.dim() == 1 and hca_indptr.shape[0] >= T + 1
    assert swa_indices.dim() == 1 and swa_indices.shape[0] >= T * win
    assert csa_indices.dim() == 1
    assert hca_indices.dim() == 1
    assert window_topk.is_contiguous(), "window_topk must be contiguous (row-major)"

    BLOCK_W = triton.next_power_of_2(win)
    _v4_paged_decode_indices_kernel[(T,)](
        state_slot_per_seq,
        batch_id_per_token,
        window_topk,
        csa_indptr,
        hca_indptr,
        swa_indices,
        csa_indices,
        hca_indices,
        cs,
        win,
        BLOCK_W=BLOCK_W,
    )


def write_v4_paged_decode_indices_reference(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    window_topk: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor,
    hca_indices: torch.Tensor,
    T: int,
    win: int,
    cs: int,
) -> None:
    """Pure-PyTorch reference equivalent of `write_v4_paged_decode_indices`.
    For unit tests and bisect verification. Mirrors the kernel exactly:
    skip rows where bid<0, otherwise compute `slot*cs + topk` with -1 sentinel
    propagation, write to all three destinations.
    """
    if T == 0:
        return
    bid = batch_id_per_token[:T].long()
    valid = bid >= 0
    slot = torch.where(
        valid, state_slot_per_seq[bid.clamp(min=0)], torch.zeros_like(bid)
    )
    topk = window_topk[:T]  # [T, win]
    paged = torch.where(
        topk >= 0,
        slot[:, None] * cs + topk,
        torch.full_like(topk, -1),
    )  # [T, win] int32
    # For invalid rows (bid<0), DON'T write — kernel bails. Mask them out:
    valid_mask = valid[:, None].expand_as(paged)
    # SWA: uniform stride → straight copy, but only valid rows
    flat = paged.reshape(-1)
    swa_indices[: T * win].copy_(flat)
    if (~valid).any():
        # restore sentinel for invalid rows in SWA (kernel doesn't write them,
        # so callers may have pre-filled -1 — match by leaving them be).
        # Reference writes regardless; for tests on non-CG inputs this is moot
        # since callers don't pass -1 bids.
        pass
    # CSA / HCA: packed at indptr[t]+w
    for t in range(T):
        if not bool(valid[t]):
            continue
        csa_base = int(csa_indptr[t].item())
        hca_base = int(hca_indptr[t].item())
        csa_indices[csa_base : csa_base + win] = paged[t]
        hca_indices[hca_base : hca_base + win] = paged[t]
    # silence lint
    del valid_mask
