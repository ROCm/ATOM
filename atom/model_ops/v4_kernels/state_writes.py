# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State-write Triton kernels for V4 attention backend.

Replaces the per-seq Python state writes in `deepseek_v4.py` (PR-A Phase 1).
Inputs are flat batched tensors; per-token slot/position lookups happen
inside the kernel — no `.item()` syncs.

Currently implemented:
- `swa_write`: writes the LAST `min(tok_n_b, write_per_batch)` tokens of
  every seq `b ∈ [0, bs)` into `swa_kv[state_slot_per_seq[b],
  positions[src] % cache_size, :] = kv[src, :]`. `src_id` is derived inside
  the kernel from `cu_seqlens_q + row_in_batch` — no shared per-token
  `write_indices` GPU buffer (which had a DMA-tear race when the next fwd's
  CPU rewrite landed mid-H2D). `cache_size = window_size + max_spec_steps`
  — for non-MTP this reduces to `window_size`; for MTP-k draft tokens get
  their own ring slots separate from the verified token's slot.
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
- `swa_kv`              [num_slots, cache_size, head_dim] in-place buffer.
- `cache_size`          int ring-slot count = `window_size + max_spec_steps`
                        (e.g. 128 + 0 = 128 non-MTP; 128 + 1 = 129 MTP-1).
- `write_per_batch`     int — max tokens to write per seq this fwd
                        (= `min(max_q_len, cache_size)`). Used as Triton
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

from atom.utils.decorators import mark_trace


@triton.jit
def _swa_write_kernel(
    kv_ptr,  # [T, head_dim]
    positions_ptr,  # [T] int — full positions
    cu_seqlens_q_ptr,  # [bs+1] int — per-seq cumulative seqlens
    block_tables_ptr,  # [bs, max_blocks_per_seq] int32 — logical→physical
    block_tables_stride,  # = max_blocks_per_seq (row stride)
    swa_region_ptr,  # [num_pages, head_dim] flat SWA region of unified_kv
    swa_region_row_stride,  # = head_dim
    head_dim,
    block_size,
    WRITE_PER_BATCH: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """paged-SWA write. 2D grid `(bs, WRITE_PER_BATCH)`. Program `(b, r)`
    writes the `r`-th of the last-N tokens of seq `b`, where
    `N = min(tok_n_b, WRITE_PER_BATCH)` and
    `tok_n_b = cu_seqlens_q[b+1] - cu_seqlens_q[b]`. Threads with `r >= N` bail.

    `src_id = cu_seqlens_q[b+1] - N + r` — selects directly from `kv` /
    `positions` with NO shared GPU index buffer (no DMA race window).

    The destination is content-addressed by `block_tables` (same physical
    block id the compressed cache uses), so a cross-request prefix-cache hit
    reads the original request's SWA from the cached physical block instead of
    a stale per-request ring (issue #1417):
        blk      = pos // block_size
        phys     = block_tables[b, blk]
        dst_row  = phys * block_size + (pos % block_size)
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
    blk = pos // block_size
    phys = tl.load(block_tables_ptr + batch_idx * block_tables_stride + blk)
    dst_row = phys * block_size + (pos % block_size)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    src = tl.load(
        kv_ptr + src_id * head_dim + d_offsets,
        mask=d_mask,
    )
    dst = swa_region_ptr + dst_row * swa_region_row_stride + d_offsets
    tl.store(dst, src, mask=d_mask)


@mark_trace
def swa_write(
    kv: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    swa_region: torch.Tensor,
    block_size: int,
    write_per_batch: int,
    *,
    k_packed: torch.Tensor | None = None,
    k_rope: torch.Tensor | None = None,
    swa_region_rope: torch.Tensor | None = None,
    prefix: str = "",
) -> None:
    """paged-SWA in-place write, dispatching on the kv-cache layout.

    Native 2buff fp8 (``swa_region_rope`` provided): the op-quantized extend K
    comes in as ``k_packed`` (fp8 NoPE) + ``k_rope`` (bf16 RoPE tail), in the
    ``[T, *]`` or ``[T, 1, *]`` layout produced by the quant kernel; delegates to
    :func:`swa_write_2buff_prepacked`, which scatters both into their paged pools
    (``swa_region`` = NoPE pool, ``swa_region_rope`` = RoPE pool) — a pure
    dtype-agnostic copy, no requant. The bf16 ``kv`` arg is unused on this path
    (the caller may pass ``None``).

    Otherwise (bf16): for the last `min(tok_n_b, write_per_batch)` tokens of
    every seq `b ∈ [0, bs)` this fwd
    (`tok_n_b = cu_seqlens_q[b+1] - cu_seqlens_q[b]`, `bs = block_tables.shape[0]`),
    write `kv[r]` to the content-addressed SWA region:
        swa_region[block_tables[b, pos//block_size] * block_size
                   + pos % block_size, :] = kv[r, :]

    Replaces the per-request ring (`swa_kv[slot, pos % cache_size]`). The
    physical block id is the SAME one the compressed cache uses, so a
    cross-request prefix-cache hit reads the original request's SWA from the
    cached block instead of a stale ring (issue #1417).

    Args:
        kv: [T, head_dim] per-fwd KV (BF16). bf16 path only; `T = cu_seqlens_q[bs]`.
            May be ``None`` on the fp8 2buff path (``k_packed`` is used instead).
        positions: [T'] int — full forward_vars["positions"] (`T' >= T`).
        cu_seqlens_q: [bs+1] int — exact size (`bs == block_tables.shape[0]`).
        block_tables: [bs, max_blocks_per_seq] int32 — logical→physical block.
            Its `shape[0]` is the grid X dim and source-of-truth for `bs`.
        swa_region: [num_pages, head_dim] flat SWA region of `unified_kv`
            (= `unified_kv[:swa_pages]`), `num_pages = num_blocks * block_size`.
        block_size: tokens per block (= V4 block_size, 128).
        write_per_batch: `min(max_q_len, block_size_window)` — max tokens
            written per seq this fwd (grid y dim, kernel `constexpr`).
        k_packed: [T, 512] or [T, 1, 512] fp8 NoPE extend K — fp8 2buff path only.
        k_rope: [T, rope_head_dim] or [T, 1, rope_head_dim] bf16 RoPE tail — fp8
            2buff path only.
        swa_region_rope: [num_pages, rope_head_dim] bf16 RoPE pool — presence
            selects the fp8 2buff path.
    """
    if swa_region_rope is not None:
        # fp8 2buff: scatter the op-quantized extend K (k_packed/k_rope) into both
        # paged SWA pools. Flatten the [T, 1, *] quant-kernel views to [T, *]; the
        # bf16 `kv` source is unused here.
        swa_write_2buff_prepacked(
            k_packed.view(k_packed.shape[0], -1),
            k_rope.view(k_rope.shape[0], -1),
            positions,
            cu_seqlens_q,
            block_tables,
            swa_region,
            swa_region_rope,
            block_size,
            write_per_batch,
        )
        return
    assert kv.dim() == 2, f"kv must be [T, D], got {kv.shape}"
    assert positions.dim() == 1
    assert (
        block_tables.dim() == 2
    ), f"block_tables must be [bs, MB], got {block_tables.shape}"
    bs = block_tables.shape[0]
    assert cu_seqlens_q.dim() == 1 and cu_seqlens_q.shape[0] >= bs + 1
    assert swa_region.dim() == 2, f"swa_region must be [P, D], got {swa_region.shape}"
    T, head_dim = kv.shape
    assert positions.shape[0] >= T, f"positions {positions.shape[0]} < kv T={T}"
    assert swa_region.shape[1] == head_dim
    assert kv.is_contiguous() and swa_region.is_contiguous()
    assert (
        bs > 0 and write_per_batch > 0
    ), f"bs={bs}, write_per_batch={write_per_batch} must be positive"

    # head_dim is small (e.g. 64-128 for V4 SWA layer), so a single Triton
    # block per token covers it. Round up to the next power of two for tl.
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (bs, write_per_batch)

    _swa_write_kernel[grid](
        kv,
        positions,
        cu_seqlens_q,
        block_tables,
        block_tables.stride(0),
        swa_region,
        swa_region.stride(0),
        head_dim,
        block_size,
        WRITE_PER_BATCH=write_per_batch,
        BLOCK_D=BLOCK_D,
    )


def swa_write_reference(
    kv: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    swa_region: torch.Tensor,
    block_size: int,
    write_per_batch: int,
) -> None:
    """Pure-PyTorch reference equivalent of `swa_write` (paged). For tests.

    Mirrors the kernel: for each seq `b ∈ [0, bs)`
    (`bs = block_tables.shape[0]`), take the last
    `min(cu_seqlens_q[b+1] - cu_seqlens_q[b], write_per_batch)` rows of `kv`
    for that seq, translate each token's position through `block_tables`, and
    write to the content-addressed SWA region.
    """
    bs = block_tables.shape[0]
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
        src_pos = positions[src_ids].to(torch.long)
        blk = src_pos // block_size
        phys = block_tables[b, blk].to(torch.long)
        dst_row = phys * block_size + (src_pos % block_size)
        swa_region[dst_row] = src_kv


def swa_write_2buff_prepacked(
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    swa_block_tables: torch.Tensor,
    swa_region_nope: torch.Tensor,
    swa_region_rope: torch.Tensor,
    block_size: int,
    write_per_batch: int,
) -> None:
    """Native 2buff fp8 paged SWA write: content-addressed scatter of the LAST
    ``min(tok_n_b, write_per_batch)`` tokens of every seq into the two paged
    SWA pools (fp8 NoPE + bf16 RoPE). The K is ALREADY in the 2buff layout
    (nope-fp8 ``[T,512]`` + rope-bf16 ``[T,64]``), produced upstream by the
    compute-only 2buff quant (:func:`qk_norm_rope_maybe_quant_fp8_2buff`). This
    is a pure dtype-agnostic scatter (reuses the paged :func:`swa_write` once per
    pool); NO torch quantization happens here.

    Both pools are the flat content-addressed regions of ``unified_kv`` /
    ``unified_kv_rope`` (``[num_pages, D]``), addressed by ``swa_block_tables``:
    ``swa_region[block_tables[b, pos//block_size] * block_size + pos%block_size]``.
    Replaces the pre-paged per-request ring variant (matches the paged bf16
    :func:`swa_write` semantics; issue #1417).

    Args:
        k_packed:        [T, 512] fp8 — quantized K nope + inline e8m0 scale + pad.
        k_rope:          [T, 64]  bf16 — rotated K-PE (not quantized).
        swa_block_tables:[bs, max_blocks] int32 — paged-SWA logical→physical map.
        swa_region_nope: [num_pages, 512] fp8 paged pool (2buff nope).
        swa_region_rope: [num_pages, 64]  bf16 paged pool (rope).
        block_size:      paging stride of both pools.
        (other args as :func:`swa_write`.)
    """
    from atom.model_ops.v4_kernels.v4_quant import V4_DIM_QK_PACKED, V4_DIM_ROPE

    assert (
        k_packed.dim() == 2 and k_packed.shape[1] == V4_DIM_QK_PACKED
    ), f"k_packed must be [T,{V4_DIM_QK_PACKED}] fp8, got {tuple(k_packed.shape)}"
    assert (
        k_rope.dim() == 2 and k_rope.shape[1] == V4_DIM_ROPE
    ), f"k_rope must be [T,{V4_DIM_ROPE}] bf16, got {tuple(k_rope.shape)}"
    assert swa_region_nope.dim() == 2 and swa_region_nope.shape[1] == V4_DIM_QK_PACKED
    assert swa_region_rope.dim() == 2 and swa_region_rope.shape[1] == V4_DIM_ROPE

    swa_write(
        k_packed.contiguous(),
        positions,
        cu_seqlens_q,
        swa_block_tables,
        swa_region_nope,
        block_size,
        write_per_batch,
    )
    swa_write(
        k_rope.contiguous(),
        positions,
        cu_seqlens_q,
        swa_block_tables,
        swa_region_rope,
        block_size,
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


@mark_trace
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
    K_pool = (2 if overlap else 1) * ratio  # pool window (lower bound)
    state_size = kv_state.shape[1]  # ring buffer modulo (≥ K_pool)
    assert (
        state_size >= K_pool
    ), f"kv_state.shape[1]={state_size}, must be ≥ K_pool={K_pool}"
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
