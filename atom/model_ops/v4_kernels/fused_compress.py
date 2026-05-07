# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Compressor boundary kernel for V4 attention (CSA / sparse_attn path).

Replaces the Python pool → RMSNorm → RoPE → (quant) → kv_cache scatter
chain in `Compressor.forward` (see `atom/models/deepseek_v4.py`).

SGLang plan-style batched dispatch (vs. the earlier per-seq launcher):
  Each compression boundary across the entire fwd is one row in
  `compress_plan_gpu` — a packed `[num_compress, 4] int32` tensor where each
  row is `[ragged_id, batch_id, position, window_len]`. The kernel grid is
  `num_compress` (zero waste, no early-exit), and each program does ONE 4×i32
  load to get all the metadata it needs:

    ragged_id  → row index in the ragged kv_in / score_in stream
    batch_id   → seq index → state_slot_mapping[batch_id], block_table[batch_id]
    position   → absolute token position (drives RoPE + paged scatter)
    window_len → number of leading K-loop iterations that read state cache
                 (instead of the ragged input). K = STATE_SIZE.

State cache vs. input vs. padding dispatch (replaces the old `s >= start_pos`
test):

    s = position - K + 1 + k_static
    is_padding = s < 0
    is_input   = k_static >= window_len
    is_state   = (~is_input) & (~is_padding)

Correctness invariant (caller-side):
  This kernel reads state cache as-of-the-end-of-the-PREVIOUS-fwd. Therefore
  the caller MUST invoke this kernel BEFORE `update_compressor_states` runs
  (which would overwrite the historic positions this kernel needs).

Output:
  Returns `[num_compress, head_dim]` BF16 tensor. Rows are in plan order
  (= ragged_id ascending = per-seq grouped). Caller slices per seq via
  `plan.cu_compress_cpu[i]:plan.cu_compress_cpu[i+1]`.

  Also writes valid rows into the paged kv_cache at compressed index
  `position // RATIO` (when `block_table` is provided).

TODO: FP8/FP4 quant fusion. Currently stores raw BF16 (no act_quant). The
`scale_fmt="ue8m0"` round-to-even f32→e8m0 path requires porting aiter's
`f32_to_e8m0` bit manipulation into Triton (~20 lines per quant block).
Skipping for this PR; follow-up PR will add it.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from atom.model_ops.v4_kernels.compress_plan import CompressPlan


@triton.jit
def _fused_compress_attn_kernel(
    # ── source: INPUT (this fwd's projection) ───────────────────────────
    kv_in_ptr,  # [num_q_tokens, dim_full] fp32 (strided allowed)
    kv_in_row_stride,  # row stride; ≥ dim_full when fused upstream split
    score_in_ptr,  # [num_q_tokens, dim_full] fp32 (raw, no ape; strided allowed)
    score_in_row_stride,
    dim_full,  # = 2*head_dim if OVERLAP else head_dim
    # ── plan: per-boundary packed metadata ──────────────────────────────
    plan_ptr,  # [num_compress, 4] int32 (ragged_id, batch_id, position, window_len)
    # ── source: state cache (previous fwd's writes; score has ape) ──────
    kv_state_ptr,
    kv_state_slot_stride,
    kv_state_pos_stride,
    score_state_ptr,
    score_state_slot_stride,
    score_state_pos_stride,
    state_slot_mapping_ptr,  # [bs] int32 — per-seq state cache slot
    # ── ape (for INPUT-source rows only) ────────────────────────────────
    ape_ptr,  # [RATIO, dim_full] fp32
    # ── RMSNorm ─────────────────────────────────────────────────────────
    rms_weight_ptr,  # [head_dim] fp32
    rms_eps,
    # ── RoPE (separate cos / sin caches) ────────────────────────────────
    cos_cache_ptr,  # [max_seq, rope_head_dim/2] bf16 (after .squeeze)
    sin_cache_ptr,
    cos_sin_pos_stride,  # = rope_head_dim // 2
    # ── KV cache scatter (paged) ────────────────────────────────────────
    kv_cache_ptr,  # [num_blocks, k_per_block, head_dim] bf16
    kv_cache_block_stride,
    kv_cache_token_stride,
    block_table_ptr,  # [bs, max_blocks_per_seq] int32
    block_table_seq_stride,  # row stride
    k_per_block,
    # ── output to caller (post norm+rope BF16 for sparse_attn input) ────
    out_ptr,  # [num_compress, head_dim] bf16
    out_token_stride,
    head_dim,
    rope_head_dim,
    # ── constexpr ───────────────────────────────────────────────────────
    BLOCK_D: tl.constexpr,  # = next_pow2(head_dim)
    HALF_ROPE: tl.constexpr,  # = rope_head_dim // 2
    OVERLAP: tl.constexpr,
    RATIO: tl.constexpr,
    STATE_SIZE: tl.constexpr,  # = 2*RATIO if OVERLAP else RATIO
    K: tl.constexpr,  # = STATE_SIZE (softmax-pool reduce dim)
    HAS_BLOCK_TABLE: tl.constexpr,
):
    """One program per boundary in the plan. Grid = plan capacity (CUDAGraph-
    safe fixed grid); inactive rows are sentinel-marked (position == -1) and
    bail before any load/store/scatter."""
    pid = tl.program_id(0)
    plan_base = plan_ptr + pid * 4
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)
    position = tl.load(plan_base + 2)
    window_len = tl.load(plan_base + 3)

    if position < 0:
        return

    slot = tl.load(state_slot_mapping_ptr + batch_id)

    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim

    # ── 1. Per-source-position load + online softmax-pool ──────────────
    NEG_INF: tl.constexpr = float("-inf")
    m_acc = tl.full([BLOCK_D], NEG_INF, tl.float32)
    kv_acc = tl.zeros([BLOCK_D], tl.float32)
    w_acc = tl.zeros([BLOCK_D], tl.float32)

    # Dynamic loop (NOT unrolled) — K=128 (HCA) would otherwise blow up hsaco.
    for k_static in tl.range(K):
        s = position - K + 1 + k_static
        is_padding = s < 0
        is_input = k_static >= window_len
        # is_state = (~is_input) & (~is_padding)

        # B-side (k < RATIO): cols [:head_dim]   (= col_off=0)
        # A-side (k >= RATIO): cols [head_dim:]  (= col_off=head_dim)
        # HCA (no overlap, K=RATIO): col_off=0 always (k_static < RATIO).
        col_off = (k_static >= RATIO) * head_dim if OVERLAP else 0
        ape_row = k_static % RATIO  # [0, RATIO) — same for B/A sides

        # Input source: row index in ragged stream.
        # k_static = K-1 corresponds to s = position (the boundary token itself,
        # = ragged_id row). Earlier k_static values map to earlier ragged rows.
        in_row = ragged_id - (K - 1 - k_static)
        kv_a = tl.load(
            kv_in_ptr + in_row * kv_in_row_stride + col_off + d,
            mask=is_input & d_mask,
            other=0.0,
        )

        # State cache source: ring slot indexed by absolute s.
        s_safe = tl.maximum(s, 0)
        ring = s_safe % STATE_SIZE
        kv_b = tl.load(
            kv_state_ptr
            + slot * kv_state_slot_stride
            + ring * kv_state_pos_stride
            + col_off
            + d,
            mask=(~is_input) & (~is_padding) & d_mask,
            other=0.0,
        )
        kv_k = kv_a + kv_b  # exactly one path active per source pos

        # ── score_k load ──
        score_a = tl.load(
            score_in_ptr + in_row * score_in_row_stride + col_off + d,
            mask=is_input & d_mask,
            other=NEG_INF,
        )
        ape_v = tl.load(
            ape_ptr + ape_row * dim_full + col_off + d,
            mask=is_input & d_mask,
            other=0.0,
        )
        score_b = tl.load(
            score_state_ptr
            + slot * score_state_slot_stride
            + ring * score_state_pos_stride
            + col_off
            + d,
            mask=(~is_input) & (~is_padding) & d_mask,
            other=NEG_INF,
        )
        score_k = tl.where(is_input, score_a + ape_v, score_b)

        # ── Online softmax-pool accumulate ──
        m_new = tl.maximum(m_acc, score_k)
        scale = tl.where(m_acc == NEG_INF, 0.0, tl.exp(m_acc - m_new))
        w_k = tl.where(score_k == NEG_INF, 0.0, tl.exp(score_k - m_new))
        kv_acc = kv_acc * scale + w_k * kv_k
        w_acc = w_acc * scale + w_k
        m_acc = m_new

    compressed = kv_acc / w_acc  # [BLOCK_D] fp32

    # ── 2. RMSNorm (fp32) ──────────────────────────────────────────────
    rms_w = tl.load(rms_weight_ptr + d, mask=d_mask, other=0.0)
    compressed_masked = tl.where(d_mask, compressed, 0.0)
    var = tl.sum(compressed_masked * compressed_masked, axis=0) / head_dim
    rrms = tl.rsqrt(var + rms_eps)
    normed = compressed_masked * rrms * rms_w  # [BLOCK_D] fp32

    # ── 3. RoPE on rope_head_dim segment (GPT-J interleaved, fp32) ────
    comp_pos = (position // RATIO) * RATIO
    NUM_PAIRS: tl.constexpr = BLOCK_D // 2
    NOPE_PAIRS = (head_dim - rope_head_dim) // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even_v, odd_v = tl.split(pair_2d)  # each [NUM_PAIRS]

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    cos_per_pair = tl.load(
        cos_cache_ptr + comp_pos * cos_sin_pos_stride + cs_idx,
        mask=is_rope_pair,
        other=1.0,
    ).to(tl.float32)
    sin_per_pair = tl.load(
        sin_cache_ptr + comp_pos * cos_sin_pos_stride + cs_idx,
        mask=is_rope_pair,
        other=0.0,
    ).to(tl.float32)

    new_even = even_v * cos_per_pair - odd_v * sin_per_pair
    new_odd = odd_v * cos_per_pair + even_v * sin_per_pair
    rotated = tl.interleave(new_even, new_odd)  # [BLOCK_D] fp32

    # ── 4. Cast to BF16 + store ────────────────────────────────────────
    rotated_bf16 = rotated.to(tl.bfloat16)

    # Output: row pid (plan order = per-seq grouped, caller uses cu_compress).
    tl.store(out_ptr + pid * out_token_stride + d, rotated_bf16, mask=d_mask)

    # KV cache scatter (paged): block_table[batch_id, ci // k_per_block][ci % k_per_block]
    if HAS_BLOCK_TABLE:
        ci = position // RATIO
        block_in_seq = ci // k_per_block
        slot_in_block = ci % k_per_block
        physical_block = tl.load(
            block_table_ptr + batch_id * block_table_seq_stride + block_in_seq
        ).to(tl.int64)
        cache_addr = (
            physical_block * kv_cache_block_stride
            + slot_in_block * kv_cache_token_stride
            + d
        )
        tl.store(kv_cache_ptr + cache_addr, rotated_bf16, mask=d_mask)


def fused_compress_attn(
    *,
    # Source tensors (ragged across all seqs in batch)
    kv_in: torch.Tensor,  # [num_q_tokens, dim_full] fp32
    score_in: torch.Tensor,  # [num_q_tokens, dim_full] fp32 (raw, no ape)
    kv_state: torch.Tensor,  # [num_slots, STATE_SIZE, dim_full] fp32
    score_state: torch.Tensor,  # same shape, score has ape pre-added
    # Plan + per-seq metadata
    plan: CompressPlan,
    state_slot_mapping: torch.Tensor,  # [bs] int32 — per-seq state cache slot
    # Compressor params
    ape: torch.Tensor,  # [ratio, dim_full] fp32
    rms_weight: torch.Tensor,  # [head_dim] fp32
    rms_eps: float,
    cos_cache: torch.Tensor,  # [max_seq, ..., rope_head_dim/2] bf16/fp16
    sin_cache: torch.Tensor,  # same shape
    # KV cache scatter
    kv_cache: Optional[torch.Tensor],  # [num_blocks, k_per_block, head_dim] bf16
    block_tables: Optional[torch.Tensor],  # [bs, max_blocks_per_seq] int32
    k_per_block: int,
    # Geometry
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Batched fused per-source-position pool + RMSNorm + RoPE + bf16 kv_cache
    scatter, dispatched via SGLang-style packed plan.

    Returns `[num_compress, head_dim]` BF16 tensor in plan order (= ragged_id
    ascending = per-seq grouped). Caller uses `plan.cu_compress_cpu[i:i+2]` to
    slice per-seq chunks.

    Returns None if `plan.num_compress == 0` AND no `out` buffer is provided.
    When `out` is supplied (CUDAGraph path), kernel ALWAYS launches at full
    plan capacity — inactive rows are sentinel-skipped inside the kernel —
    and `out` is returned unchanged when `num_compress == 0`.

    Caller MUST invoke BEFORE `update_compressor_states` (state cache reads
    must see previous-fwd data).
    """
    device = kv_in.device
    num_compress = plan.num_compress
    plan_capacity = plan.compress_plan_gpu.shape[0]
    if plan_capacity == 0:
        return out  # nothing to do; out (or None) returned as-is.
    if num_compress == 0 and out is None:
        # Eager path: nothing to do, no caller buffer to fill.
        return None

    # Validate shapes
    dim_full = (2 if overlap else 1) * head_dim
    state_size = (2 if overlap else 1) * ratio
    assert (
        kv_in.dim() == 2 and kv_in.shape[1] == dim_full
    ), f"kv_in {kv_in.shape}, expected [*, {dim_full}]"
    assert score_in.shape == kv_in.shape
    assert kv_state.shape[1] == state_size and kv_state.shape[2] == dim_full
    assert score_state.shape == kv_state.shape
    assert ape.shape == (ratio, dim_full)
    assert rms_weight.shape == (head_dim,)
    assert plan.compress_plan_gpu.shape == (
        plan_capacity,
        4,
    ), f"plan {plan.compress_plan_gpu.shape}, expected ({plan_capacity}, 4)"
    assert plan.compress_plan_gpu.dtype == torch.int32
    assert num_compress <= plan_capacity, (
        f"plan.num_compress ({num_compress}) > capacity ({plan_capacity}); "
        f"caller must size the plan buffer to the worst-case num_compress."
    )
    assert state_slot_mapping.dim() == 1 and state_slot_mapping.dtype == torch.int32
    assert cos_cache.shape[-1] == rope_head_dim // 2
    assert sin_cache.shape[-1] == rope_head_dim // 2
    assert (
        cos_cache.stride(0) == rope_head_dim // 2
    ), f"cos_cache outer stride {cos_cache.stride(0)} != rope_head_dim/2"
    # kv_in / score_in row-strided allowed (e.g. zero-copy split halves of the
    # fused wkv_gate output). Inner column stride must be 1 — kernel uses
    # `+ d` for the BLOCK_D offset.
    assert kv_in.stride(-1) == 1 and score_in.stride(-1) == 1
    assert kv_state.is_contiguous() and score_state.is_contiguous()
    assert ape.is_contiguous() and rms_weight.is_contiguous()
    has_bt = block_tables is not None
    if has_bt:
        assert kv_cache is not None and kv_cache.dim() == 3
        assert block_tables.dim() == 2 and block_tables.is_contiguous()
        bt_seq_stride = block_tables.stride(0)
    else:
        bt_seq_stride = 0

    if out is None:
        # Eager path: allocate output sized to the actual num_compress so the
        # returned tensor has the legacy [num_compress, head_dim] shape.
        out = torch.empty(num_compress, head_dim, dtype=out_dtype, device=device)
    else:
        # CUDAGraph path: caller-provided buffer of capacity ≥ plan_capacity.
        # Validate shape; kernel will write into rows [0:num_compress) and
        # leave [num_compress:plan_capacity) untouched (those plan rows are
        # sentinel-skipped). Inactive output rows therefore carry stale data
        # but no consumer reads them (caller slices via cu_compress_cpu).
        assert (
            out.shape[0] >= plan_capacity and out.shape[1] == head_dim
        ), f"out {tuple(out.shape)}, expected ≥ ({plan_capacity}, {head_dim})"
        assert out.dtype == out_dtype
        assert out.is_contiguous() or out.stride(1) == 1

    BLOCK_D = triton.next_power_of_2(head_dim)
    HALF_ROPE = rope_head_dim // 2
    K = state_size

    # Fixed grid for CUDAGraph compat: launch one program per plan row;
    # sentinel rows (position=-1) skipped inside the kernel. When out is
    # eager-allocated above, grid still shrinks to num_compress (no pad).
    grid = (plan_capacity if out.shape[0] >= plan_capacity else num_compress,)
    _fused_compress_attn_kernel[grid](
        kv_in,
        kv_in.stride(0),
        score_in,
        score_in.stride(0),
        dim_full,
        plan.compress_plan_gpu,
        kv_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state,
        score_state.stride(0),
        score_state.stride(1),
        state_slot_mapping,
        ape,
        rms_weight,
        rms_eps,
        cos_cache,
        sin_cache,
        cos_cache.stride(0),
        kv_cache if has_bt else cos_cache,  # placeholder when no scatter
        kv_cache.stride(0) if has_bt else 0,
        kv_cache.stride(1) if has_bt else 0,
        block_tables if has_bt else state_slot_mapping,  # placeholder
        bt_seq_stride,
        k_per_block,
        out,
        out.stride(0),
        head_dim,
        rope_head_dim,
        BLOCK_D=BLOCK_D,
        HALF_ROPE=HALF_ROPE,
        OVERLAP=int(overlap),
        RATIO=ratio,
        STATE_SIZE=state_size,
        K=K,
        HAS_BLOCK_TABLE=int(has_bt),
    )

    return out  # [num_compress, head_dim]


def fused_compress_attn_reference(
    *,
    kv_in: torch.Tensor,
    score_in: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    plan: CompressPlan,
    state_slot_mapping: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_tables: Optional[torch.Tensor],
    k_per_block: int,
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Optional[torch.Tensor]:
    """Pure-PyTorch reference equivalent of `fused_compress_attn` (plan path).

    Returns `[num_compress, head_dim]` BF16 in plan order. None if num_compress=0.
    """
    if plan.num_compress == 0:
        return None
    device = kv_in.device
    K = (2 if overlap else 1) * ratio
    state_size = K
    plan_cpu = plan.compress_plan_gpu.detach().cpu()
    slot_map_cpu = state_slot_mapping.detach().cpu()
    if block_tables is not None:
        bt_cpu = block_tables.detach().cpu()
    else:
        bt_cpu = None

    out = torch.empty(plan.num_compress, head_dim, dtype=out_dtype, device=device)

    for pid in range(plan.num_compress):
        ragged_id, batch_id, position, window_len = plan_cpu[pid].tolist()
        slot = int(slot_map_cpu[batch_id].item())

        kv_rows = []
        score_rows = []
        for k in range(K):
            s = position - K + 1 + k
            if overlap:
                col_off = head_dim if k >= ratio else 0
            else:
                col_off = 0
            ape_row = k % ratio
            d_slice = slice(col_off, col_off + head_dim)
            is_padding = s < 0
            is_input = k >= window_len

            if is_padding:
                kv_rows.append(
                    torch.zeros(head_dim, dtype=torch.float32, device=device)
                )
                score_rows.append(
                    torch.full(
                        (head_dim,), float("-inf"), dtype=torch.float32, device=device
                    )
                )
            elif is_input:
                in_row = ragged_id - (K - 1 - k)
                kv_rows.append(kv_in[in_row, d_slice].float())
                score_rows.append(
                    score_in[in_row, d_slice].float() + ape[ape_row, d_slice].float()
                )
            else:
                ring = s % state_size
                kv_rows.append(kv_state[slot, ring, d_slice].float())
                score_rows.append(score_state[slot, ring, d_slice].float())

        kv_stack = torch.stack(kv_rows, dim=0)  # [K, head_dim]
        sc_stack = torch.stack(score_rows, dim=0)  # [K, head_dim]
        weights = torch.softmax(sc_stack, dim=0)
        compressed = (weights * kv_stack).sum(dim=0)  # [head_dim] fp32

        var = (compressed * compressed).mean()
        normed = compressed * torch.rsqrt(var + rms_eps) * rms_weight.float()

        comp_pos = (position // ratio) * ratio
        rope_seg = normed[-rope_head_dim:].clone()
        cos_v = cos_cache[comp_pos].view(-1).float()
        sin_v = sin_cache[comp_pos].view(-1).float()
        even = rope_seg[0::2]
        odd = rope_seg[1::2]
        new_even = even * cos_v - odd * sin_v
        new_odd = odd * cos_v + even * sin_v
        rotated_seg = torch.stack([new_even, new_odd], dim=-1).flatten()
        normed[-rope_head_dim:] = rotated_seg

        out_bf16 = normed.to(out_dtype)
        out[pid] = out_bf16

        if bt_cpu is not None and kv_cache is not None:
            ci = position // ratio
            block_in_seq = ci // k_per_block
            slot_in_block = ci % k_per_block
            physical = int(bt_cpu[batch_id, block_in_seq].item())
            kv_cache[physical, slot_in_block] = out_bf16

    return out
