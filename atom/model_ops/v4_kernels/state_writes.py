# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State-write Triton kernels for V4 attention backend.

Replaces the per-seq Python state writes in `deepseek_v4.py` (PR-A Phase 1).
Inputs are flat batched tensors; per-token slot/position lookups happen
inside the kernel — no `.item()` syncs.

Currently implemented:
- `swa_write`: writes `swa_kv[slot_per_token[t], position[t] % win, :] = kv[t, :]`
  for each token in the flat batch. Equivalent semantics to the legacy
  prefill (lines 1429-1437) and decode (line 1446) writes.
- `update_compressor_states`: unified in-place update of Compressor's
  per-request `kv_state` + `score_state` ring buffers, covering both prefill
  (B-side overlap context + tail) and decode (every token at `pos % STATE_SIZE`
  in a single ring). Layout follows paper §3.6.1 (per-request fixed-size state
  cache) but indexes the buffer as ONE ring of size `STATE_SIZE = 2*ratio`
  (CSA overlap) or `ratio` (HCA). Token at absolute `pos` always lands at
  `kv_state[slot, pos % STATE_SIZE]` — no segment switching, no roll. The
  Compressor's softmax-pool consumer reads two halves whose A-side / B-side
  identity alternates by block-id parity; see `Compressor.forward` for that
  consumer-side logic.

Caller contract (`swa_write`):
- `kv`            [num_tokens, head_dim] flat — same layout as the model's
                  per-seq slice `seq_kv.squeeze(0)` concatenated across seqs.
- `positions`    [num_tokens] int — absolute token positions, model already has these.
- `slot_per_token` [num_tokens] int — built once per step in the metadata builder
                  by repeating `slot_per_seq` along `cu_seqlens_q` deltas.
- `swa_kv`       [num_slots, win, head_dim] in-place buffer.
- `win`          int sliding-window size (e.g. 128).

For long-prefill (`seqlen > win`), the caller must pre-filter `kv` and the
position/slot tensors so that only the last `win` tokens per seq are passed
in (otherwise multiple tokens in the same seq map to the same `pos % win`
and the write race is non-deterministic). The metadata builder produces a
`swa_keep_mask` for this purpose. Decode steps (seqlen=1) and short prefill
(seqlen<=win) bypass the mask.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swa_write_kernel(
    kv_ptr,  # [num_tokens, head_dim]
    positions_ptr,  # [num_tokens] int (sentinel = -1 for inactive slots)
    slot_ptr,  # [num_tokens] int
    swa_kv_ptr,  # [num_slots, win, head_dim]
    swa_kv_slot_stride,  # = win * head_dim
    swa_kv_pos_stride,  # = head_dim
    head_dim,
    win,
    BLOCK_D: tl.constexpr,
):
    """Fixed grid + sentinel mask (CUDAGraph-safe).

    The grid size equals the buffer capacity (`positions.shape[0]`) regardless
    of how many tokens this fwd actually writes. Inactive grid programs are
    indicated by `positions[tok_id] < 0` and bail before any load/store.
    """
    tok_id = tl.program_id(0)
    pos = tl.load(positions_ptr + tok_id)
    if pos < 0:
        return
    slot = tl.load(slot_ptr + tok_id)
    ring_idx = pos % win

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    src = tl.load(
        kv_ptr + tok_id * head_dim + d_offsets,
        mask=d_mask,
    )
    dst = (
        swa_kv_ptr
        + slot * swa_kv_slot_stride
        + ring_idx * swa_kv_pos_stride
        + d_offsets
    )
    tl.store(dst, src, mask=d_mask)


def swa_write(
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_per_token: torch.Tensor,
    swa_kv: torch.Tensor,
    win: int,
) -> None:
    """In-place write `swa_kv[slot_per_token[t], pos[t] % win, :] = kv[t, :]`.

    Args:
        kv: [num_tokens, head_dim] flat batched KV (BF16).
        positions: [num_tokens] int absolute positions. Slots with `position
            < 0` are skipped — caller may pass a fixed-capacity buffer with
            sentinel-filled tail to keep the grid size constant for CUDAGraph.
        slot_per_token: [num_tokens] int per-request slot ids.
        swa_kv: [num_slots, win, head_dim] in-place ring buffer.
        win: sliding-window size.
    """
    assert kv.dim() == 2, f"kv must be [N, D], got {kv.shape}"
    assert positions.dim() == 1
    assert slot_per_token.dim() == 1
    assert swa_kv.dim() == 3, f"swa_kv must be [S, W, D], got {swa_kv.shape}"
    num_tokens, head_dim = kv.shape
    assert positions.shape[0] == num_tokens
    assert slot_per_token.shape[0] == num_tokens
    assert swa_kv.shape[1] == win
    assert swa_kv.shape[2] == head_dim
    assert kv.is_contiguous() and swa_kv.is_contiguous()

    if num_tokens == 0:
        return

    # head_dim is small (e.g. 64-128 for V4 SWA layer), so a single Triton
    # block per token covers it. Round up to the next power of two for tl.
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (num_tokens,)
    _swa_write_kernel[grid](
        kv,
        positions,
        slot_per_token,
        swa_kv,
        swa_kv.stride(0),
        swa_kv.stride(1),
        head_dim,
        win,
        BLOCK_D=BLOCK_D,
    )


def swa_write_reference(
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_per_token: torch.Tensor,
    swa_kv: torch.Tensor,
    win: int,
) -> None:
    """Pure-PyTorch reference equivalent of `swa_write`. For tests / dump-bisect."""
    ring_idx = positions % win
    swa_kv[slot_per_token, ring_idx] = kv


# === Unified Compressor state save (plan path) ==========================
# Paper §3.6.1: per-request fixed-size state cache for "uncompressed tail
# tokens + previous block as overlap context (B-side, eq 11)". ATOM keeps
# this as a single ring of size `STATE_SIZE = 2*ratio` (CSA overlap) or
# `ratio` (HCA). Each token at absolute `pos` writes to slot
# `pos % STATE_SIZE`; the consumer (`fused_compress.*` kernel) reads its K
# source rows per-source-position, dispatching INPUT vs state cache by the
# `k_static >= window_len` plan field (where `window_len` is the count of
# leading K-loop iterations that go to state cache, encoded per-boundary in
# `compress_plan`).
#
# Write window selection (HOST side, in compress_plan.make_compress_plans):
#   write_plan rows = tokens whose absolute `pos >= max(0, seq_len - STATE_SIZE)`.
#   This preserves the last STATE_SIZE absolute positions of this forward
#   regardless of how it was scheduled (fresh prefill, chunked prefill,
#   single decode, MTP-N). The kernel below writes those rows
#   unconditionally — no in-kernel mask.


@triton.jit
def _update_compressor_states_kernel(
    kv_ptr,  # [N, dim] (strided allowed)
    kv_row_stride,
    score_ptr,  # [N, dim] (strided allowed)
    score_row_stride,
    ape_ptr,  # [RATIO, dim]
    write_plan_ptr,  # [num_write, 4] int32 (ragged_id, batch_id, position, _)
    state_slot_mapping_ptr,  # [bs] int32 — per-seq state cache slot
    kv_state_ptr,
    kv_state_slot_stride,
    kv_state_pos_stride,
    score_state_ptr,
    score_state_slot_stride,
    score_state_pos_stride,
    dim,
    STATE_SIZE: tl.constexpr,  # = 2*RATIO if OVERLAP else RATIO
    OVERLAP: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """SGLang plan-style write: one program per row in `write_plan_ptr`.

    Each plan row = (ragged_id, batch_id, position, _). The plan was
    pre-filtered on the host to include only tokens whose `position` falls in
    the per-seq "last STATE_SIZE absolute positions" window — so the kernel
    writes unconditionally (no in-kernel mask), keeping it minimal.

    Destination (uniform):
      dst = position % STATE_SIZE
      slot = state_slot_mapping[batch_id]

    Score write fuses ape lookup: `score + ape[position % RATIO]`.
    """
    pid = tl.program_id(0)
    plan_base = write_plan_ptr + pid * 4
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)
    position = tl.load(plan_base + 2)

    # Fixed-grid + sentinel for CUDAGraph compat: caller may pass a buffer
    # padded to max capacity; rows beyond `num_write` carry position = -1
    # and are skipped here.
    if position < 0:
        return

    slot = tl.load(state_slot_mapping_ptr + batch_id)
    dst = position % STATE_SIZE
    ring_idx_ape = position % RATIO

    d = tl.arange(0, BLOCK_D)
    m = d < dim

    kv_v = tl.load(kv_ptr + ragged_id * kv_row_stride + d, mask=m).to(tl.float32)
    sc_v = tl.load(score_ptr + ragged_id * score_row_stride + d, mask=m).to(tl.float32)
    ape_v = tl.load(ape_ptr + ring_idx_ape * dim + d, mask=m).to(tl.float32)

    tl.store(
        kv_state_ptr + slot * kv_state_slot_stride + dst * kv_state_pos_stride + d,
        kv_v,
        mask=m,
    )
    tl.store(
        score_state_ptr
        + slot * score_state_slot_stride
        + dst * score_state_pos_stride
        + d,
        sc_v + ape_v,
        mask=m,
    )


def update_compressor_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    write_plan: torch.Tensor,  # [num_write, 4] int32
    num_write: int,
    state_slot_mapping: torch.Tensor,  # [bs] int32 — per-seq state slot
    ratio: int,
    overlap: bool,
) -> None:
    """In-place update of Compressor's per-request `kv_state`/`score_state`
    ring buffer (size `2*ratio` for overlap CSA, `ratio` for HCA), driven by
    a SGLang-style packed `write_plan`.

    The plan is pre-filtered on the host to include only tokens whose
    `position` falls in the per-seq "last STATE_SIZE absolute positions"
    window — the kernel writes unconditionally, no in-kernel mask.

    Args:
      kv:           [N, dim] flat batched KV (typically fp32 or bf16, cast inside).
      score:        [N, dim] flat batched score (NOT pre-added with ape;
                    kernel fuses ape addition).
      ape:          [ratio, dim] absolute position embedding.
      kv_state:     [num_slots, S, dim] in-place ring buffer.
                    S = 2*ratio if overlap else ratio.
      score_state:  same shape as kv_state.
      write_plan:   [num_write, 4] int32 — packed (ragged_id, batch_id,
                    position, _); each row = one token to write.
      num_write:    grid size (CPU scalar, == write_plan.shape[0] but kept
                    explicit to avoid GPU sync).
      state_slot_mapping: [bs] int32 — per-seq state cache slot.
      ratio, overlap: compress geometry.
    """
    assert kv.dim() == 2 and score.dim() == 2
    assert kv.shape == score.shape, f"{kv.shape} vs {score.shape}"
    assert ape.dim() == 2 and ape.shape[0] == ratio
    state_size = (2 if overlap else 1) * ratio
    assert (
        kv_state.shape[1] == state_size
    ), f"kv_state.shape[1]={kv_state.shape[1]}, expected {state_size}"
    dim = kv.shape[1]
    assert write_plan.dim() == 2 and write_plan.shape[1] == 4
    assert write_plan.dtype == torch.int32
    assert state_slot_mapping.dim() == 1 and state_slot_mapping.dtype == torch.int32
    # Grid = plan buffer capacity (fixed at builder __init__ time), NOT the
    # per-fwd `num_write`. Inactive rows past `num_write` carry sentinel
    # `position=-1` (filled host-side in `make_compress_plans`); the kernel
    # bails on those, so this is functionally identical to the variable-grid
    # version while keeping the launch CUDAGraph-capturable.
    grid_size = write_plan.shape[0]
    if grid_size == 0:
        return

    # Strided kv / score allowed (zero-copy split halves of fused upstream
    # GEMM); inner column stride must be 1 (kernel uses `+ d`).
    assert kv.stride(-1) == 1 and score.stride(-1) == 1
    BLOCK_D = triton.next_power_of_2(dim)
    _update_compressor_states_kernel[(grid_size,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        write_plan,
        state_slot_mapping,
        kv_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state,
        score_state.stride(0),
        score_state.stride(1),
        dim,
        STATE_SIZE=state_size,
        OVERLAP=int(overlap),
        RATIO=ratio,
        BLOCK_D=BLOCK_D,
    )


def update_compressor_states_reference(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    write_plan: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    ratio: int,
    overlap: bool,
) -> None:
    """Pure-PyTorch reference equivalent of `update_compressor_states` (plan path).

    `write_plan[i] = (ragged_id, batch_id, position, _)` — each row is one
    token to write.  No mask (host filtered).
    """
    state_size = (2 if overlap else 1) * ratio
    plan_cpu = write_plan.detach().cpu()
    slot_map_cpu = state_slot_mapping.detach().cpu()
    for i in range(plan_cpu.shape[0]):
        ragged_id, batch_id, position, _ = plan_cpu[i].tolist()
        slot = int(slot_map_cpu[batch_id].item())
        dst = position % state_size
        kv_state[slot, dst] = kv[ragged_id]
        score_state[slot, dst] = score[ragged_id] + ape[position % ratio]
