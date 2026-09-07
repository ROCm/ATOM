# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State-write Triton kernels for V4 attention backend.

Replaces the per-seq Python state writes in `deepseek_v4.py` (PR-A Phase 1).
Inputs are flat batched tensors; per-token slot/position lookups happen
inside the kernel — no `.item()` syncs.

Currently implemented:
- `swa_write`: writes the LAST `min(tok_n_b, write_per_batch)` tokens of
  every seq `b ∈ [0, bs)` into `pool[window.index(slot, positions[src])] =
  kv[src, :]`, where `window` is the layer's class's `WindowParams` and the
  row formula lives in `v4_pool_geometry`. `src_id` is derived inside the
  kernel from `cu_seqlens_q + row_in_batch` — no shared per-token
  `write_indices` GPU buffer (which had a DMA-tear race when the next fwd's
  CPU rewrite landed mid-H2D). The ring holds `window_size + max_spec_steps`
  positions — for non-MTP that reduces to `window_size`; for MTP-k draft
  tokens get their own ring slots separate from the verified token's slot.
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
- `kv`                  [T, head_dim] flat — full per-fwd KV (forward_vars).
- `positions`           [T] int — full positions buffer (forward_vars).
- `cu_seqlens_q`        [bs+1] int — per-fwd cumulative seqlens (so
                        seq `i` covers token rows `[cu_seqlens_q[i], cu_seqlens_q[i+1])`
                        in `kv` / `positions`). Per-seq token count is
                        derived inside the kernel as `cu_seqlens_q[i+1] -
                        cu_seqlens_q[i]`.
- `state_slot_per_seq`  [bs] int — `state_slot_mapping_gpu_i32`.
- `pool`                [rows, head_dim] this layer's whole plane view,
                        written in place. A row outside it is dropped, not
                        wrapped — see the bound in `_swa_write_kernel`.
- `window`              `WindowParams` for this layer's compress class; carries
                        the ring size `window_size + max_spec_steps` (e.g.
                        128 + 0 = 128 non-MTP; 128 + 1 = 129 MTP-1).
- `write_per_batch`     int — max tokens to write per seq this fwd
                        (= `min(max_q_len, window.ring_slots)`). Used as Triton
                        `constexpr` for grid sizing.

Grid = `(bs, write_per_batch)`; each program writes one (seq, row-in-seq)
token. Per-seq actual count is `min(token_num_per_seq[bs], write_per_batch)`;
threads whose `row_in_batch >= actual_count` bail. The kernel derives
`src_id = cu_seqlens_q[i+1] - actual_count + row_in_batch` — selects the
LAST `actual_count` tokens of seq `i` in `kv` / `positions`, no shared
GPU index buffer needed (no DMA race window).
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import WindowParams
from atom.model_ops.v4_kernels.pool_index import (
    row_offset,
    window_constexprs,
    window_row,
)
from atom.utils.decorators import mark_trace


@triton.jit
def _swa_write_kernel(
    kv_ptr,  # [T, head_dim]
    positions_ptr,  # [T] int — full positions
    cu_seqlens_q_ptr,  # [bs+1] int — per-seq cumulative seqlens
    state_slot_per_seq_ptr,  # [bs] int — state_slot_mapping_gpu_i32
    pool_ptr,  # this layer's whole unified-pool view, [rows, head_dim]
    pool_row_stride,  # = head_dim
    pool_rows,  # rows in that view; nothing may be written past it
    head_dim,
    ring_start,
    WRITE_PER_BATCH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
):
    """SWA ring write. 2D grid `(bs, WRITE_PER_BATCH)`. Program `(b, r)`
    writes the `r`-th of the last-N tokens of seq `b`, where
    `N = min(tok_n_b, WRITE_PER_BATCH)` and
    `tok_n_b = cu_seqlens_q[b+1] - cu_seqlens_q[b]`. Threads with `r >= N` bail.

    `src_id = cu_seqlens_q[b+1] - N + r` — selects directly from `kv` /
    `positions` with NO shared GPU index buffer (no DMA race window).

    The destination is this request's own ring slot, `window_row(slot, pos)`
    — an index into this layer's view, which spans the whole pool.

    A private ring is what #1417 replaced with `block_tables` addressing,
    because a request resuming someone else's cached prefix had never written
    that prefix into its own ring and read stale rows. The ring is back because
    a checkpoint now carries the ring: it lives in the same per-request entry
    the compressor state does, so resuming copies the window in. Reverting the
    addressing WITHOUT that copy reintroduces #1417 exactly.
    """
    batch_idx = tl.program_id(0)
    row_in_batch = tl.program_id(1)

    cu_start = tl.load(cu_seqlens_q_ptr + batch_idx)
    cu_end = tl.load(cu_seqlens_q_ptr + batch_idx + 1)
    tok_n = cu_end - cu_start
    if tok_n <= 0:
        return
    write_n = tl.minimum(tok_n, WRITE_PER_BATCH)
    if row_in_batch >= write_n:
        return

    src_id = cu_end - write_n + row_in_batch

    pos = tl.load(positions_ptr + src_id)
    slot = tl.load(state_slot_per_seq_ptr + batch_idx)
    dst_row = window_row(
        slot, pos, ring_start, RING_SLOTS, SLOT_ROWS, RING_STRIDE, RUN_ROWS
    )

    # `cu_seqlens_q` and `write_n` bound the reads; nothing bounded the write.
    # `slot` and `pos` are the row's only inputs and both come from per-forward
    # staging, so a stale one could address anywhere in a 150 GB pool and only
    # surface as a fault in an unrelated kernel a thousand dispatches later.
    # The fused twin in aiter took this same guard when the ring came back.
    if dst_row < 0 or dst_row >= pool_rows:
        return

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    src = tl.load(
        kv_ptr + src_id * head_dim + d_offsets,
        mask=d_mask,
    )
    dst = pool_ptr + row_offset(dst_row, pool_row_stride) + d_offsets
    tl.store(dst, src, mask=d_mask)


@mark_trace
def swa_write(
    kv: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    pool: torch.Tensor,
    window: WindowParams,
    write_per_batch: int,
    *,
    k_packed: torch.Tensor | None = None,
    k_rope: torch.Tensor | None = None,
    pool_rope: torch.Tensor | None = None,
    prefix: str = "",
) -> None:
    """SWA ring in-place write, dispatching on the kv-cache layout.

    Native 2buff fp8 (``pool_rope`` provided): the op-quantized extend K comes
    in as ``k_packed`` (fp8 NoPE) + ``k_rope`` (bf16 RoPE tail), in the
    ``[T, *]`` or ``[T, 1, *]`` layout produced by the quant kernel; delegates to
    :func:`swa_write_2buff_prepacked`, which scatters both into their planes
    (``pool`` = NoPE, ``pool_rope`` = RoPE) — a pure dtype-agnostic copy, no
    requant. The bf16 ``kv`` arg is unused on this path (the caller may pass
    ``None``).

    Otherwise (bf16): for the last `min(tok_n_b, write_per_batch)` tokens of
    every seq `b ∈ [0, bs)` this fwd
    (`tok_n_b = cu_seqlens_q[b+1] - cu_seqlens_q[b]`,
    `bs = state_slot_per_seq.shape[0]`), write `kv[r]` into that request's
    window at `window.index(slot, pos)`.

    The ring holds `window.ring_slots = window_size + max_spec_steps` positions,
    not `window_size`: a spec round writes the verified token plus
    `max_spec_steps` drafts at consecutive positions, and a ring sized
    `window_size` would alias the drafts onto `[p_0-window_size+1 .. p_0]` — the
    verified query at `p_0` would then read future tokens.

    Args:
        kv: [T, head_dim] per-fwd KV (BF16). bf16 path only; `T = cu_seqlens_q[bs]`.
            May be ``None`` on the fp8 2buff path (``k_packed`` is used instead).
        positions: [T'] int — full forward_vars["positions"] (`T' >= T`). Under
            PCP these must be the all-gathered full-sequence positions: the ring
            index is absolute-position modulo, so a 1/W shard would alias.
        cu_seqlens_q: [bs+1] int — exact size (`bs == state_slot_per_seq.shape[0]`).
        state_slot_per_seq: [bs] int32 — per-request state slot
            (`state_slot_mapping_gpu_i32`). Its `shape[0]` is the grid X dim and
            source-of-truth for `bs`.
        pool: [rows, head_dim] — this layer's whole view of the NoPE plane, not
            just its window part. Rows come from `window.index`, which addresses
            the plane relative to this view's base.
        window: this layer's compress class's `WindowParams`
            (`UnifiedPoolGeometry.window_params`).
        write_per_batch: `min(max_q_len, window.ring_slots)` — max tokens written
            per seq this fwd (grid y dim, kernel `constexpr`).
        k_packed: [T, 512] or [T, 1, 512] fp8 NoPE extend K — fp8 2buff path only.
        k_rope: [T, rope_head_dim] or [T, 1, rope_head_dim] bf16 RoPE tail — fp8
            2buff path only.
        pool_rope: [rows, rope_head_dim] bf16 RoPE plane view — presence selects
            the fp8 2buff path.
    """
    if pool_rope is not None:
        # fp8 2buff: scatter the op-quantized extend K (k_packed/k_rope) into both
        # planes. Flatten the [T, 1, *] quant-kernel views to [T, *]; the bf16
        # `kv` source is unused here.
        swa_write_2buff_prepacked(
            k_packed.view(k_packed.shape[0], -1),
            k_rope.view(k_rope.shape[0], -1),
            positions,
            cu_seqlens_q,
            state_slot_per_seq,
            pool,
            pool_rope,
            window,
            write_per_batch,
        )
        return
    assert kv.dim() == 2, f"kv must be [T, D], got {kv.shape}"
    assert positions.dim() == 1
    assert (
        state_slot_per_seq.dim() == 1
    ), f"state_slot_per_seq must be [bs], got {state_slot_per_seq.shape}"
    bs = state_slot_per_seq.shape[0]
    assert cu_seqlens_q.dim() == 1 and cu_seqlens_q.shape[0] >= bs + 1
    assert pool.dim() == 2, f"pool must be [rows, D], got {pool.shape}"
    T, head_dim = kv.shape
    assert positions.shape[0] >= T, f"positions {positions.shape[0]} < kv T={T}"
    assert pool.shape[1] == head_dim
    assert kv.is_contiguous() and pool.is_contiguous()
    assert (
        bs > 0 and write_per_batch > 0
    ), f"bs={bs}, write_per_batch={write_per_batch} must be positive"
    assert write_per_batch <= window.ring_slots, (
        f"write_per_batch={write_per_batch} exceeds the ring "
        f"({window.ring_slots}): two tokens of the SAME seq would map to one row "
        "and race. Paged addressing was injective on position and needed no cap; "
        "a ring is not."
    )

    # head_dim is small (e.g. 64-128 for V4 SWA layer), so a single Triton
    # block per token covers it. Round up to the next power of two for tl.
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (bs, write_per_batch)

    _swa_write_kernel[grid](
        kv,
        positions,
        cu_seqlens_q,
        state_slot_per_seq,
        pool,
        pool.stride(0),
        pool.shape[0],
        head_dim,
        window.ring_start,
        WRITE_PER_BATCH=write_per_batch,
        BLOCK_D=BLOCK_D,
        **window_constexprs(window),
    )


def swa_write_reference(
    kv: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    pool: torch.Tensor,
    window: WindowParams,
    write_per_batch: int,
) -> None:
    """Pure-PyTorch reference equivalent of `swa_write` (ring). For tests.

    Mirrors the kernel: for each seq `b ∈ [0, bs)`
    (`bs = state_slot_per_seq.shape[0]`), take the last
    `min(cu_seqlens_q[b+1] - cu_seqlens_q[b], write_per_batch)` rows of `kv`
    for that seq and write them at `window.index(slot, pos)`. The row comes from
    the geometry rather than from a second copy of the formula, so this checks
    the kernel's transcription and not its author's memory.
    """
    bs = state_slot_per_seq.shape[0]
    cu_cpu = cu_seqlens_q[: bs + 1].tolist()
    for b in range(bs):
        cu_start = int(cu_cpu[b])
        cu_end = int(cu_cpu[b + 1])
        tok_n = cu_end - cu_start
        write_n = min(tok_n, write_per_batch)
        if write_n <= 0:
            continue
        src_ids = torch.arange(
            cu_end - write_n, cu_end, dtype=torch.long, device=kv.device
        )
        src_kv = kv[src_ids]
        src_pos = positions[src_ids].tolist()
        slot = int(state_slot_per_seq[b])
        dst_row = torch.tensor(
            [window.index(slot, int(p)) for p in src_pos],
            dtype=torch.long,
            device=kv.device,
        )
        pool[dst_row] = src_kv


@triton.jit
def _swa_scatter_rows_kernel(
    kv_ptr,  # [T, head_dim]
    dest_row_ptr,  # [T] int32 — plane row for this token
    batch_id_per_token_ptr,  # [T] int — -1 on CG-pad tokens
    pool_ptr,  # [rows, head_dim] this layer's plane view
    pool_row_stride,
    pool_rows,  # rows in that view; nothing may be written past it
    head_dim,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    dst_row = tl.load(dest_row_ptr + t)
    bid = tl.load(batch_id_per_token_ptr + t)
    # Upper bound as well as lower: `dest_rows` comes from another kernel fed
    # by the same staging, so it is no more trustworthy than the row
    # `swa_write` derives inline.
    if dst_row < 0 or dst_row >= pool_rows or bid < 0:
        return
    d = tl.arange(0, BLOCK_D)
    m = d < head_dim
    tl.store(
        pool_ptr + row_offset(dst_row, pool_row_stride) + d,
        tl.load(kv_ptr + t * head_dim + d, mask=m),
        mask=m,
    )


@mark_trace
def swa_scatter_rows(
    kv: torch.Tensor,
    dest_rows: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    pool: torch.Tensor,
    *,
    k_packed: torch.Tensor | None = None,
    k_rope: torch.Tensor | None = None,
    pool_rope: torch.Tensor | None = None,
    prefix: str = "",
) -> None:
    """Write each token's KV to a row the caller already chose.

    The decode counterpart of :func:`swa_write`. Decode knows every token's
    destination before the layers run — one row per compress class, built once
    by `write_v4_paged_decode_indices` — so the row need not be re-derived 41
    times, and the fused write in the norm/RoPE kernel can take the same array
    instead of carrying the window's geometry into another repo.

    Args:
        kv:        [T, head_dim] — bf16 path.
        dest_rows: [>=T] int32 — plane row per token.
        batch_id_per_token: [>=T] int — `-1` on CG-pad tokens, which are
                   skipped. The same gate the fused writes apply, so a padded
                   replay writes nothing whichever backend ran.
        pool:      [rows, head_dim] this layer's NoPE plane view.
        k_packed / k_rope / pool_rope: the fp8 2buff triple, as in `swa_write`.
    """
    if pool_rope is not None:
        assert k_packed is not None and k_rope is not None
        flat = (t.reshape(t.shape[0], -1) for t in (k_packed, k_rope))
        nope, rope = flat
        swa_scatter_rows(nope, dest_rows, batch_id_per_token, pool)
        swa_scatter_rows(rope, dest_rows, batch_id_per_token, pool_rope)
        return
    assert kv.dim() == 2 and pool.dim() == 2
    assert kv.shape[1] == pool.shape[1]
    T = kv.shape[0]
    if T == 0:
        return
    assert dest_rows.shape[0] >= T and batch_id_per_token.shape[0] >= T
    head_dim = kv.shape[1]
    _swa_scatter_rows_kernel[(T,)](
        kv.contiguous(),
        dest_rows,
        batch_id_per_token,
        pool,
        pool.stride(0),
        pool.shape[0],
        head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


def swa_scatter_rows_reference(
    kv: torch.Tensor,
    dest_rows: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    pool: torch.Tensor,
) -> None:
    """Pure-torch equivalent of :func:`swa_scatter_rows` (bf16 path)."""
    T = kv.shape[0]
    live = (dest_rows[:T] >= 0) & (batch_id_per_token[:T] >= 0)
    pool[dest_rows[:T][live].long()] = kv[live]


def swa_write_2buff_prepacked(
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    pool_nope: torch.Tensor,
    pool_rope: torch.Tensor,
    window: WindowParams,
    write_per_batch: int,
) -> None:
    """Native 2buff fp8 SWA ring write: scatter of the LAST
    ``min(tok_n_b, write_per_batch)`` tokens of every seq into the two SWA ring
    pools (fp8 NoPE + bf16 RoPE). The K is ALREADY in the 2buff layout
    (nope-fp8 ``[T,512]`` + rope-bf16 ``[T,64]``), produced upstream by the
    compute-only 2buff quant (:func:`qk_norm_rope_maybe_quant_fp8_2buff`). This
    is a pure dtype-agnostic scatter (reuses :func:`swa_write` once per pool);
    NO torch quantization happens here.

    Both planes are this layer's views of ``unified_kv`` / ``unified_kv_rope``
    and take the SAME row index, ``window.index(slot, pos)`` — that is what a
    shared `kv_indices` buffer means, and why the two planes materialize one row
    space rather than sharing an envelope.

    Args:
        k_packed:        [T, 512] fp8 — quantized K nope + inline e8m0 scale + pad.
        k_rope:          [T, 64]  bf16 — rotated K-PE (not quantized).
        state_slot_per_seq: [bs] int32 — per-request state slot.
        pool_nope:       [rows, 512] fp8 NoPE plane view for this layer.
        pool_rope:       [rows, 64]  bf16 RoPE plane view for this layer.
        window:          this layer's class's ``WindowParams``.
        (other args as :func:`swa_write`.)
    """
    from atom.model_ops.v4_kernels.v4_quant import V4_DIM_QK_PACKED, V4_DIM_ROPE

    assert (
        k_packed.dim() == 2 and k_packed.shape[1] == V4_DIM_QK_PACKED
    ), f"k_packed must be [T,{V4_DIM_QK_PACKED}] fp8, got {tuple(k_packed.shape)}"
    assert (
        k_rope.dim() == 2 and k_rope.shape[1] == V4_DIM_ROPE
    ), f"k_rope must be [T,{V4_DIM_ROPE}] bf16, got {tuple(k_rope.shape)}"
    assert pool_nope.dim() == 2 and pool_nope.shape[1] == V4_DIM_QK_PACKED
    assert pool_rope.dim() == 2 and pool_rope.shape[1] == V4_DIM_ROPE

    swa_write(
        k_packed.contiguous(),
        positions,
        cu_seqlens_q,
        state_slot_per_seq,
        pool_nope,
        window,
        write_per_batch,
    )
    swa_write(
        k_rope.contiguous(),
        positions,
        cu_seqlens_q,
        state_slot_per_seq,
        pool_rope,
        window,
        write_per_batch,
    )


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
    kv_row_stride: tl.constexpr,
    score_ptr,  # [N, dim] (strided allowed)
    score_row_stride: tl.constexpr,
    ape_ptr,  # [RATIO, dim]
    write_plan_ptr,  # [num_write, 4] int32 (ragged_id, batch_id, position, _)
    state_slot_mapping_ptr,  # [bs] int32 — per-seq state cache slot
    kv_state_ptr,
    kv_state_slot_stride: tl.constexpr,
    kv_state_pos_stride: tl.constexpr,
    score_state_ptr,
    score_state_slot_stride: tl.constexpr,
    score_state_pos_stride: tl.constexpr,
    dim: tl.constexpr,
    STATE_SIZE: tl.constexpr,  # ring buffer modulo = kv_state.shape[1] (≥ K_pool;
    #   V4-Pro spec decode: K_pool + max_spec_steps to keep R's rejected writes
    #   out of R+1's read window; non-spec or pre-spec models: exactly K_pool)
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

    # 64 bits on the slot term, which is not optional: the compressor state
    # lives at the front of a slot in the shared plane, so consecutive slots
    # are a whole slot apart and the product runs the length of the pool. At
    # 152 GB that is 17x past what a 32-bit multiply holds, and it wraps
    # silently. `dst` and `d` stay inside one entry and need no widening.
    kv_slot_base = row_offset(slot, kv_state_slot_stride)
    score_slot_base = row_offset(slot, score_state_slot_stride)
    tl.store(
        kv_state_ptr + kv_slot_base + dst * kv_state_pos_stride + d,
        kv_v,
        mask=m,
    )
    tl.store(
        score_state_ptr + score_slot_base + dst * score_state_pos_stride + d,
        sc_v + ape_v,
        mask=m,
    )


@mark_trace
def update_compressor_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    write_plan: torch.Tensor,  # [num_write, 4] int32
    state_slot_mapping: torch.Tensor,  # [bs] int32 — per-seq state slot
    ratio: int,
    overlap: bool,
    prefix: str = "",
) -> None:
    """In-place update of Compressor's per-request `kv_state`/`score_state`
    ring buffer (size ≥ `K_pool = (1+overlap)*ratio`; V4-Pro widens to
    `K_pool + max_spec_steps` for spec decode, keeps `K_pool` for non-spec),
    driven by a SGLang-style packed `write_plan`.

    The plan is pre-filtered on the host to include only tokens whose
    `position` falls in the per-seq "last K_pool absolute positions" window
    (`write_starts = max(0, context_lens - K_pool)` in `make_compress_plans`)
    — the kernel writes unconditionally, no in-kernel mask. Note that the
    write window is K_pool, NOT STATE_SIZE; the extra STATE_SIZE - K_pool
    slots exist purely as aliasing slack for spec rollback (see
    `csa_main_state_shape` comment in `deepseek_v4_attn.py`).

    Args:
      kv:           [N, dim] flat batched KV (typically fp32 or bf16, cast inside).
      score:        [N, dim] flat batched score (NOT pre-added with ape;
                    kernel fuses ape addition).
      ape:          [ratio, dim] absolute position embedding.
      kv_state:     [num_slots, S, dim] in-place ring buffer. S ≥ K_pool;
                    V4-Pro: S = K_pool + max_spec_steps.
      score_state:  same shape as kv_state.
      write_plan:   [grid, 4] int32 — packed (ragged_id, batch_id, position, _);
                    each active row = one token to write. `grid` (== shape[0])
                    is the caller-supplied slice length: the decode-tight
                    `running_bs * min(qlen, K_pool)` on the CUDAGraph path, tight
                    `num_write` on the eager path, or the full buffer capacity
                    for the extend-shaped verify path. Inactive tail rows carry
                    sentinel `position=-1` and are skipped.
      state_slot_mapping: [bs] int32 — per-seq state cache slot.
      ratio, overlap: compress geometry.
    """
    assert kv.dim() == 2 and score.dim() == 2
    assert kv.shape == score.shape, f"{kv.shape} vs {score.shape}"
    assert ape.dim() == 2 and ape.shape[0] == ratio
    K_pool = (2 if overlap else 1) * ratio  # pool window (lower bound)
    state_size = kv_state.shape[1]  # ring buffer modulo (≥ K_pool)
    assert (
        state_size >= K_pool
    ), f"kv_state.shape[1]={state_size}, must be ≥ K_pool={K_pool}"
    dim = kv.shape[1]
    assert write_plan.dim() == 2 and write_plan.shape[1] == 4
    assert write_plan.dtype == torch.int32
    assert state_slot_mapping.dim() == 1 and state_slot_mapping.dtype == torch.int32
    # Grid = the write-plan slice length (fixed at capture on the CUDAGraph
    # path). Inactive tail rows carry sentinel `position=-1` (filled host-side
    # in `make_compress_plans`); the kernel bails on those, so a padded slice is
    # functionally identical to a tight one while staying CUDAGraph-capturable.
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
    state_size = kv_state.shape[1]  # ring buffer modulo (≥ (1+overlap)*ratio)
    plan_cpu = write_plan.detach().cpu()
    slot_map_cpu = state_slot_mapping.detach().cpu()
    for i in range(plan_cpu.shape[0]):
        ragged_id, batch_id, position, _ = plan_cpu[i].tolist()
        # Skip sentinel rows (position = -1) exactly like the kernel. Without
        # this, Python's negative modulo (`-1 % state_size == state_size-1`)
        # would silently write a garbage row into the ring.
        if position < 0:
            continue
        slot = int(slot_map_cpu[batch_id].item())
        dst = position % state_size
        kv_state[slot, dst] = kv[ragged_id]
        score_state[slot, dst] = score[ragged_id] + ape[position % ratio]


# === DSpark rolling window gather (read side of the SWA ring) =============
# DSpark's block drafter attends `[rolling target window ++ draft block]`. The
# window KV lives in the shared pool (`unified_kv`, draft layer slice bound as
# `attn.swa_plane`), addressed by the request's state slot exactly like the V4
# target SWA. The block-sparse attention still wants a DENSE `[B, W, D]`
# window tensor (it concatenates the in-forward draft KV and runs `sparse_attn`),
# so this kernel materialises that window from the pool.
#
# Window slot `s ∈ [0, W)` for seq `b` holds the target token at absolute
# position `p = anchor_pos[b] - (W - 1) + s`. Slots with `p < 0` are unfilled
# (the caller's `valid_target` mask drops them; we zero them here so a stray read
# is harmless). Filled slots map to the pool via the same addressing as the
# write:
#     src_row = window.index(state_slot_per_seq[b], p)
#
# `draft_window <= window.ring_slots` is a hard precondition, asserted below:
# the draft's `window_size` and the target's `win_with_spec` come from different
# configs, and a wider draft window would silently read rows the ring already
# recycled.


@triton.jit
def _dspark_paged_window_gather_kernel(
    pool_ptr,  # this layer's whole unified-pool view, [rows, head_dim]
    pool_row_stride,  # = head_dim
    state_slot_per_seq_ptr,  # [B] int32 per-request ring slot
    anchor_pos_ptr,  # [B] int — per-seq anchor absolute position
    out_ptr,  # [B, W, head_dim] dense window output
    out_seq_stride,  # = W * head_dim
    out_slot_stride,  # = head_dim
    head_dim,
    ring_start,
    W: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
):
    """2D grid `(B, W)`. Program `(b, s)` gathers window slot `s` of seq `b`
    from the paged pool into `out[b, s, :]`. Unfilled slots (`p < 0`) write 0."""
    b = tl.program_id(0)
    s = tl.program_id(1)

    anchor = tl.load(anchor_pos_ptr + b)
    p = anchor - (W - 1) + s

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    out_base = out_ptr + b * out_seq_stride + s * out_slot_stride

    if p < 0:
        # Unfilled slot: zero it (valid_target masks it out in attention anyway).
        tl.store(
            out_base + d_offsets,
            tl.zeros([BLOCK_D], dtype=out_ptr.dtype.element_ty),
            mask=d_mask,
        )
        return

    slot = tl.load(state_slot_per_seq_ptr + b)
    src_row = window_row(
        slot, p, ring_start, RING_SLOTS, SLOT_ROWS, RING_STRIDE, RUN_ROWS
    )
    src = tl.load(
        pool_ptr + row_offset(src_row, pool_row_stride) + d_offsets,
        mask=d_mask,
    )
    tl.store(out_base + d_offsets, src, mask=d_mask)


def dspark_paged_window_gather(
    pool: torch.Tensor,  # [rows, D] this layer's plane view (attn.unified_kv)
    state_slot_per_seq: torch.Tensor,  # [B] int32 per-request ring slot
    anchor_pos: torch.Tensor,  # [B] int — anchor absolute position per seq
    draft_window: int,
    window: WindowParams,
) -> torch.Tensor:  # [B, draft_window, head_dim]
    """Materialise the dense `[B, W, head_dim]` rolling window from the pool,
    addressed by the state slot (mirrors `swa_write`). Slot `s` holds absolute
    position `anchor_pos[b] - (W-1) + s`; `p < 0` slots are zeroed.
    """
    assert pool.dim() == 2, f"pool must be [rows, D], got {pool.shape}"
    assert (
        state_slot_per_seq.dim() == 1
    ), f"state_slot_per_seq must be [B], got {state_slot_per_seq.shape}"
    assert draft_window <= window.ring_slots, (
        f"rolling window {draft_window} exceeds the SWA ring "
        f"({window.ring_slots} slots): the oldest slot the window needs has "
        "already been overwritten. The draft's `window_size` and the target's "
        "`win_with_spec` are separate configs."
    )
    B = state_slot_per_seq.shape[0]
    head_dim = pool.shape[1]
    assert anchor_pos.shape[0] >= B
    assert pool.is_contiguous()

    out = torch.zeros(B, draft_window, head_dim, device=pool.device, dtype=pool.dtype)
    if B == 0 or draft_window == 0:
        return out
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (B, draft_window)
    _dspark_paged_window_gather_kernel[grid](
        pool,
        pool.stride(0),
        state_slot_per_seq,
        anchor_pos.to(torch.int32),
        out,
        out.stride(0),
        out.stride(1),
        head_dim,
        window.ring_start,
        W=draft_window,
        BLOCK_D=BLOCK_D,
        **window_constexprs(window),
    )
    return out


def dspark_paged_window_gather_reference(
    pool: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    anchor_pos: torch.Tensor,
    draft_window: int,
    window: WindowParams,
) -> torch.Tensor:
    """Pure-torch reference for `dspark_paged_window_gather` (unit tests)."""
    B = state_slot_per_seq.shape[0]
    _, head_dim = pool.shape
    out = torch.zeros(B, draft_window, head_dim, device=pool.device, dtype=pool.dtype)
    for b in range(B):
        anchor = int(anchor_pos[b].item())
        for s in range(draft_window):
            p = anchor - (draft_window - 1) + s
            if p < 0:
                continue
            slot = int(state_slot_per_seq[b])
            out[b, s] = pool[window.index(slot, p)]
    return out


# === DSpark paged window gather — native 2buff fp8 variant =================
# fp8 KV cache stores the rolling target window in the SAME 2buff layout as the
# V4 target: NoPE lanes fp8-quantized (per-64-elt e8m0 tile scale, inline in the
# 512B `pool_nope` row) + RoPE lanes bf16 in a parallel `pool_rope` plane, both
# ring-addressed by the request's state slot. DSpark's block
# attention wants a DENSE bf16 `[B, W, head_dim]` window, so this kernel gathers
# BOTH pools and dequantizes the NoPE half on the fly (fp8_val * 2^(B-127)),
# concatenating the bf16 RoPE tail — a fused analog of
# `dspark_paged_window_gather` + `dequantize_v4_2buff_to_bf16`.


@triton.jit
def _dspark_paged_window_gather_2buff_kernel(
    nope_fp8_ptr,  # [rows, 512] fp8 plane view (NoPE 448 | dup-e8m0-scale 14 | pad 50)
    nope_u8_ptr,  # same buffer, uint8 view — reads the e8m0 scale bytes
    rope_ptr,  # [rows, ROPE] bf16 plane view
    nope_row_stride,  # = 512
    rope_row_stride,  # = ROPE
    state_slot_per_seq_ptr,  # [B] int32 per-request ring slot
    anchor_pos_ptr,  # [B] int
    out_ptr,  # [B, W, NOPE+ROPE] bf16 dense window
    out_seq_stride,  # = W * (NOPE+ROPE)
    out_slot_stride,  # = NOPE+ROPE
    ring_start,
    W: tl.constexpr,
    NOPE: tl.constexpr,  # 448
    ROPE: tl.constexpr,  # 64
    TILE: tl.constexpr,  # 64
    NUM_TILES: tl.constexpr,  # 7
    PACK_OFF_SCALE: tl.constexpr,  # 448
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
):
    """2D grid `(B, W)`. Program `(b, s)` gathers + dequantizes window slot `s`
    of seq `b` into `out[b, s, :]` (bf16). Unfilled slots (`p < 0`) write 0."""
    b = tl.program_id(0)
    s = tl.program_id(1)

    anchor = tl.load(anchor_pos_ptr + b)
    p = anchor - (W - 1) + s
    out_base = out_ptr + b * out_seq_stride + s * out_slot_stride

    d_tile = tl.arange(0, TILE)
    r_cols = tl.arange(0, ROPE)

    if p < 0:
        # Unfilled slot: zero it (valid_target masks it out in attention anyway).
        zero_t = tl.zeros([TILE], dtype=out_ptr.dtype.element_ty)
        for t in tl.static_range(NUM_TILES):
            tl.store(out_base + t * TILE + d_tile, zero_t)
        tl.store(
            out_base + NOPE + r_cols,
            tl.zeros([ROPE], dtype=out_ptr.dtype.element_ty),
        )
        return

    slot = tl.load(state_slot_per_seq_ptr + b)
    src_row = window_row(
        slot, p, ring_start, RING_SLOTS, SLOT_ROWS, RING_STRIDE, RUN_ROWS
    )

    # NoPE: per-64-elt tile fp8 dequant. e8m0 byte B decodes to 2^(B-127); B==0
    # is the all-zero-tile sentinel -> scale 0.0 (mirrors _e8m0_to_fp32_pow2).
    for t in tl.static_range(NUM_TILES):
        cols = t * TILE + d_tile
        nope_base = row_offset(src_row, nope_row_stride)
        x = tl.load(nope_fp8_ptr + nope_base + cols).to(tl.float32)
        byte = tl.load(nope_u8_ptr + nope_base + PACK_OFF_SCALE + 2 * t).to(tl.int32)
        scale = tl.where(byte > 0, tl.exp2((byte - 127).to(tl.float32)), 0.0)
        tl.store(out_base + cols, (x * scale).to(out_ptr.dtype.element_ty))

    # RoPE tail: bf16 passthrough.
    r = tl.load(rope_ptr + row_offset(src_row, rope_row_stride) + r_cols)
    tl.store(out_base + NOPE + r_cols, r.to(out_ptr.dtype.element_ty))


def dspark_paged_window_gather_2buff(
    pool_nope: torch.Tensor,  # [rows, 512] fp8 plane view
    pool_rope: torch.Tensor,  # [rows, rope_dim] bf16 plane view
    state_slot_per_seq: torch.Tensor,  # [B] int32 per-request ring slot
    anchor_pos: torch.Tensor,  # [B] int
    draft_window: int,
    window: WindowParams,
) -> torch.Tensor:  # [B, draft_window, V4_DIM_QK] bf16
    """Materialise + dequantize the dense bf16 `[B, W, 512]` rolling window from
    the two 2buff planes (NoPE fp8 + RoPE bf16), addressed by the state slot
    (mirrors `swa_write_2buff_prepacked`). Slot `s` holds absolute position
    `anchor_pos[b] - (W-1) + s`; `p < 0` slots are zeroed.
    """
    from atom.model_ops.v4_kernels.v4_quant import (
        V4_DIM_NOPE,
        V4_DIM_QK,
        V4_DIM_QK_PACKED,
        V4_DIM_ROPE,
        V4_NUM_TILES,
        V4_PACK_OFF_SCALE,
        V4_TILE,
    )

    assert pool_nope.dim() == 2 and pool_nope.shape[1] == V4_DIM_QK_PACKED, (
        f"pool_nope must be [rows,{V4_DIM_QK_PACKED}] fp8, "
        f"got {tuple(pool_nope.shape)}"
    )
    assert pool_rope.dim() == 2 and pool_rope.shape[1] == V4_DIM_ROPE, (
        f"pool_rope must be [rows,{V4_DIM_ROPE}] bf16, " f"got {tuple(pool_rope.shape)}"
    )
    assert (
        state_slot_per_seq.dim() == 1
    ), f"state_slot_per_seq must be [B], got {state_slot_per_seq.shape}"
    assert draft_window <= window.ring_slots, (
        f"rolling window {draft_window} exceeds the SWA ring "
        f"({window.ring_slots} slots): the oldest slot the window needs has "
        "already been overwritten. The draft's `window_size` and the target's "
        "`win_with_spec` are separate configs."
    )
    assert pool_nope.is_contiguous() and pool_rope.is_contiguous()
    B = state_slot_per_seq.shape[0]
    out = torch.empty(
        B, draft_window, V4_DIM_QK, device=pool_nope.device, dtype=torch.bfloat16
    )
    if B == 0 or draft_window == 0:
        return out
    grid = (B, draft_window)
    _dspark_paged_window_gather_2buff_kernel[grid](
        pool_nope,
        pool_nope.view(torch.uint8),
        pool_rope,
        pool_nope.stride(0),
        pool_rope.stride(0),
        state_slot_per_seq,
        anchor_pos.to(torch.int32),
        out,
        out.stride(0),
        out.stride(1),
        window.ring_start,
        W=draft_window,
        NOPE=V4_DIM_NOPE,
        ROPE=V4_DIM_ROPE,
        TILE=V4_TILE,
        NUM_TILES=V4_NUM_TILES,
        PACK_OFF_SCALE=V4_PACK_OFF_SCALE,
        **window_constexprs(window),
    )
    return out


def dspark_paged_window_gather_2buff_reference(
    pool_nope: torch.Tensor,
    pool_rope: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    anchor_pos: torch.Tensor,
    draft_window: int,
    window: WindowParams,
) -> torch.Tensor:
    """Pure-torch reference for `dspark_paged_window_gather_2buff` (unit tests):
    gather each 2buff pool with the single-pool reference, then dequantize.
    """
    from atom.model_ops.v4_kernels.v4_quant import dequantize_v4_2buff_to_bf16

    nope = dspark_paged_window_gather_reference(
        pool_nope, state_slot_per_seq, anchor_pos, draft_window, window
    )  # [B, W, 512] fp8
    rope = dspark_paged_window_gather_reference(
        pool_rope, state_slot_per_seq, anchor_pos, draft_window, window
    )  # [B, W, rope] bf16
    return dequantize_v4_2buff_to_bf16(nope, rope)
