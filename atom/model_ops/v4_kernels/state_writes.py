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
    positions_ptr,  # [num_tokens] int
    slot_ptr,  # [num_tokens] int
    swa_kv_ptr,  # [num_slots, win, head_dim]
    swa_kv_slot_stride,  # = win * head_dim
    swa_kv_pos_stride,  # = head_dim
    head_dim,
    win,
    BLOCK_D: tl.constexpr,
):
    tok_id = tl.program_id(0)
    pos = tl.load(positions_ptr + tok_id)
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
        positions: [num_tokens] int absolute positions.
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


# === Unified Compressor state save (all phases) =========================
# Paper §3.6.1: per-request fixed-size state cache for "uncompressed tail
# tokens + previous block as overlap context (B-side, eq 11)". ATOM keeps
# this as a single ring of size `STATE_SIZE = 2*ratio` (CSA overlap) or
# `ratio` (HCA). Each token at absolute `pos` writes to slot
# `pos % STATE_SIZE`; the consumer (`fused_compress.*` kernel) reads its K
# source rows per-source-position, dispatching INPUT vs state cache by the
# `s >= start_pos` test.
#
# Write mask (uniform — no prefill/decode discrimination):
#   Always preserve the last STATE_SIZE absolute positions of this forward:
#     write_start_pos = max(0, context_n - STATE_SIZE)
#     do_write        = pos >= write_start_pos
#   This is the smallest invariant that keeps the next forward's first
#   compress-boundary read fully populated regardless of how this forward
#   was scheduled (fresh prefill, chunked prefill, single decode, MTP-N).


@triton.jit
def _update_compressor_states_kernel(
    kv_ptr,  # [N, dim]
    score_ptr,  # [N, dim]
    ape_ptr,  # [RATIO, dim]
    positions_ptr,  # [N]            absolute token positions
    seq_idx_per_token_ptr,  # [N]            which seq each token belongs to
    cu_seqlens_q_ptr,  # [num_seqs+1]   per-seq token offsets
    context_lens_ptr,  # [num_seqs]     per-seq total context length
    state_slot_mapping_ptr,  # [num_seqs]     per-seq state slot
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
    """One program per token. Uniform `dst = pos % STATE_SIZE`; per-seq mask
    derived from `cu_seqlens_q` + `context_lens`.

    Per-seq derivations:
      seqlen_q   = cu_seqlens_q[seq+1] - cu_seqlens_q[seq]
      context_n  = context_lens[seq]                   (incl. tokens in this fwd)
      start_pos  = context_n - seqlen_q

    Per-token write mask (unified across fresh prefill / chunked prefill /
    single-token decode / MTP-N):
      Always preserve the last STATE_SIZE positions of THIS forward in state
      cache (so the next forward has a full B-side overlap window for its
      first compress-boundary read):

        write_start_pos = max(0, context_n - STATE_SIZE)
        do_write        = pos >= write_start_pos

      For fresh prefill with seqlen_q >= STATE_SIZE this writes positions
      [seqlen-STATE_SIZE, seqlen). For shorter forwards it writes everything.
      No is_prefill / is_fresh_prefill discrimination — pure function of
      context length.

    Per-token destination:
      dst = pos % STATE_SIZE                           (uniform; consumer
                                                       resolves A/B by parity
                                                       or per-source position)
    """
    tok = tl.program_id(0)
    pos = tl.load(positions_ptr + tok)
    seq_idx = tl.load(seq_idx_per_token_ptr + tok)
    ring_idx_ape = pos % RATIO  # ape lookup is always per-block-position

    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seqlen_q = seq_end - seq_start
    context_n = tl.load(context_lens_ptr + seq_idx)

    # Unified write window: last STATE_SIZE absolute positions of this fwd.
    write_start_pos = tl.maximum(context_n - STATE_SIZE, 0)
    do_write = pos >= write_start_pos

    dst = pos % STATE_SIZE
    slot = tl.load(state_slot_mapping_ptr + seq_idx)

    d = tl.arange(0, BLOCK_D)
    m = (d < dim) & do_write

    kv_v = tl.load(kv_ptr + tok * dim + d, mask=m).to(tl.float32)
    sc_v = tl.load(score_ptr + tok * dim + d, mask=m).to(tl.float32)
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
    positions: torch.Tensor,
    context_lens: torch.Tensor,
    ratio: int,
    overlap: bool,
    # Batched-mode inputs (caller has full attn_metadata):
    cu_seqlens_q: torch.Tensor | None = None,
    state_slot_mapping: torch.Tensor | None = None,
    seq_idx_per_token: torch.Tensor | None = None,
    # Single-seq fallback (constructs trivial 1-seq metadata):
    slot: int | None = None,
) -> None:
    """In-place update of Compressor's per-request `kv_state`/`score_state`
    ring buffer (size `2*ratio` for overlap CSA, `ratio` for HCA). Each
    token at absolute `pos` writes to slot `pos % STATE_SIZE` — uniform
    across prefill (only tail + B-side overlap window are written) and
    decode (every token in this fwd writes).

    `context_lens` is required in both call modes — the kernel uses it to
    discriminate fresh prefill (`context_n == seqlen_q`) vs decode /
    prefix-cache (`context_n > seqlen_q`). Caller pulls it from
    `attn_metadata.context_lens` (e.g. `var["context_lens"].gpu[:bs]` for
    batched, or a per-seq slice `[seq_idx:seq_idx+1]` for single-seq).

    Two call modes:

    1. **Batched** (preferred — caller has full attn_metadata):
        `cu_seqlens_q`, `state_slot_mapping`, `seq_idx_per_token` from
        `forward_context.attn_metadata`. One launch handles all seqs.

    2. **Single-seq fallback**: pass `slot=` + `context_lens=` (1-element
        tensor). Wrapper builds trivial 1-seq cu_seqlens_q /
        state_slot_mapping / seq_idx_per_token.

    All per-token write decisions (mask, dst) happen inside the kernel.

    Args (common):
      kv:          [N, dim]                — flat batched KV.
      score:       [N, dim]                — flat batched score (NOT
                                             pre-added with ape; kernel adds it).
      ape:         [ratio, dim]            — absolute position embedding.
      positions:   [N]                     — absolute token positions.
      context_lens: [num_seqs] int         — per-seq end-of-context
                                             absolute position. Required.
      kv_state:    [num_slots, S, dim] in-place ring buffer.
                                           S = 2*ratio if overlap else ratio.
      score_state: same shape as kv_state.
    """
    assert kv.dim() == 2 and score.dim() == 2
    assert kv.shape == score.shape, f"{kv.shape} vs {score.shape}"
    assert ape.dim() == 2 and ape.shape[0] == ratio
    assert (
        context_lens is not None
    ), "context_lens is required (no positions-derived fallback)"
    state_size = (2 if overlap else 1) * ratio
    assert (
        kv_state.shape[1] == state_size
    ), f"kv_state.shape[1]={kv_state.shape[1]}, expected {state_size}"
    n = kv.shape[0]
    if n == 0:
        return
    dim = kv.shape[1]
    device = kv.device

    # Single-seq fallback: build trivial 1-seq cu_seqlens_q /
    # state_slot_mapping / seq_idx_per_token. context_lens MUST be
    # supplied by caller (no positions-derived shortcut).
    if cu_seqlens_q is None:
        assert slot is not None, "single-seq mode requires slot="
        cu_seqlens_q = torch.tensor([0, n], device=device, dtype=torch.int32)
        state_slot_mapping = torch.tensor([slot], device=device, dtype=torch.int32)
        seq_idx_per_token = torch.zeros(n, device=device, dtype=torch.int32)
    else:
        assert (
            state_slot_mapping is not None and seq_idx_per_token is not None
        ), "batched mode requires state_slot_mapping, seq_idx_per_token"

    BLOCK_D = triton.next_power_of_2(dim)
    _update_compressor_states_kernel[(n,)](
        kv if kv.is_contiguous() else kv.contiguous(),
        score if score.is_contiguous() else score.contiguous(),
        ape,
        positions,
        seq_idx_per_token,
        cu_seqlens_q,
        context_lens,
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
    positions: torch.Tensor,
    context_n: int,
    ratio: int,
    overlap: bool,
    slot: int,
) -> None:
    """Pure-PyTorch reference equivalent of `update_compressor_states`.

    Unified write mask (no prefill/decode discrimination): preserve the last
    STATE_SIZE positions of THIS forward, where STATE_SIZE = 2*ratio (overlap)
    or ratio (HCA). `context_n` is the absolute end-of-context position +1
    (i.e., `start_pos + seqlen` for this fwd).
    """
    state_size = (2 if overlap else 1) * ratio
    write_start = max(0, context_n - state_size)
    for i in range(kv.size(0)):
        p = int(positions[i].item())
        if p < write_start:
            continue
        dst = p % state_size
        kv_state[slot, dst] = kv[i]
        score_state[slot, dst] = score[i] + ape[p % ratio]
