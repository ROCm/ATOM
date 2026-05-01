# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Compressor boundary kernel for V4 attention (CSA / sparse_attn path).

Replaces the Python pool → RMSNorm → RoPE → (quant) → kv_cache scatter
chain in `Compressor.forward` (see `atom/models/deepseek_v4.py`).

Generality (per design plan §"Core Insight"):
  Each source position `s` of a boundary token at absolute position `p` is
  resolved independently:
    s < 0                      → -inf padding   (block-0 B-side)
    s >= start_pos             → INPUT          (this fwd's projection at row s-start_pos)
    0 <= s < start_pos         → state cache    (slot, ring=`s % STATE_SIZE`)
  This handles fresh prefill, chunked prefill (start_pos > 0), single-token
  decode, and MTP-N decode uniformly — no `is_prefill` flag, no caller-supplied
  start_pos: kernel loads `start_pos = positions[0]` and
  `end_pos = context_lens[0]` itself.

Correctness invariant (caller-side):
  This kernel reads state cache as-of-the-end-of-the-PREVIOUS-fwd. Therefore
  the caller MUST invoke this kernel BEFORE `update_compressor_states` runs
  (which would overwrite the historic positions this kernel needs).

Grid:
  (n_max,) where n_max = (token_num + ratio - 1) // ratio. This is the
  start_pos-free upper bound on boundary count. Programs whose derived
  `pos >= end_pos` early-exit (≤ ratio-1 wasted programs per launch). The
  output buffer rows for early-exited programs are left UNINITIALIZED.

Output:
  Returns `[1, n_max, head_dim]` BF16 padded tensor — rows
  `[n_actual, n_max)` are uninitialized garbage. Downstream `sparse_attn`
  is gather-based (`atom/model_ops/sparse_attn_v4.py:74-79`) and only reads
  rows referenced by `topk_idxs`, which lies in `[0, n_actual)`. Padded rows
  are never accessed; their content does not affect numerics.

  Also writes valid rows into the paged kv_cache at compressed index
  `pos // RATIO` (when `block_table` is provided). Early-exited programs do
  not scatter.

TODO: FP8/FP4 quant fusion. Currently stores raw BF16 (no act_quant). The
`scale_fmt="ue8m0"` round-to-even f32→e8m0 path requires porting aiter's
`f32_to_e8m0` bit manipulation into Triton (~20 lines per quant block).
Skipping for this PR; follow-up PR will add it.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_compress_attn_kernel(
    # ── source: INPUT (this fwd's projection) ───────────────────────────
    kv_in_ptr,  # [token_num, dim_full] fp32
    score_in_ptr,  # [token_num, dim_full] fp32 (raw, no ape)
    dim_full,  # = 2*head_dim if OVERLAP else head_dim
    # ── per-seq metadata (loaded inside kernel) ─────────────────────────
    positions_ptr,  # [token_num] int — start_pos = positions[0]
    context_lens_ptr,  # [1] int — end_pos = context_lens[0]
    # ── source: state cache (previous fwd's writes; score has ape) ──────
    kv_state_ptr,
    kv_state_slot_stride,
    kv_state_pos_stride,
    score_state_ptr,
    score_state_slot_stride,
    score_state_pos_stride,
    slot,
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
    block_table_ptr,  # [max_blocks_per_seq] int32
    k_per_block,
    # ── output to caller (post norm+rope BF16 for sparse_attn input) ────
    out_ptr,  # [n_max, head_dim] bf16 (rows past n_actual are GARBAGE)
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
    """One program per slot in the n_max-sized grid. Programs whose derived
    `pos >= end_pos` early-exit (≤ RATIO-1 per launch)."""
    prog_id = tl.program_id(0)
    start_pos = tl.load(positions_ptr)
    end_pos = tl.load(context_lens_ptr)

    # k_min = start_pos // RATIO is the smallest global boundary index whose
    # absolute position (k+1)*RATIO - 1 is >= start_pos (correct for both
    # boundary-aligned and non-aligned start_pos; see plan).
    k_global = start_pos // RATIO + prog_id
    pos = (k_global + 1) * RATIO - 1
    if pos >= end_pos:
        return

    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim

    # ── 1. Per-source-position load + online softmax-pool ──────────────
    # Online softmax: keep running (max, weighted_kv_acc, weight_acc) per d-lane.
    NEG_INF: tl.constexpr = float("-inf")
    m_acc = tl.full([BLOCK_D], NEG_INF, tl.float32)
    kv_acc = tl.zeros([BLOCK_D], tl.float32)
    w_acc = tl.zeros([BLOCK_D], tl.float32)

    # Dynamic loop (NOT unrolled) — K=128 (HCA) would otherwise produce a
    # ~148KB hsaco vs ~16KB for K=8 (CSA), and short-prefill HCA cases
    # (no boundary in window) launch one early-exit-only program per layer
    # whose per-launch overhead scales with hsaco size.
    for k_static in tl.range(K):
        s = pos - K + 1 + k_static
        is_padding = s < 0
        is_input = (s >= start_pos) & (~is_padding)
        # is_state = (~is_input) & (~is_padding)

        # B-side (k < RATIO): cols [:head_dim]   (= col_off=0)
        # A-side (k >= RATIO): cols [head_dim:]  (= col_off=head_dim)
        # HCA (no overlap, K=RATIO): col_off=0 always (k_static < RATIO).
        col_off = (k_static >= RATIO) * head_dim if OVERLAP else 0
        ape_row = k_static % RATIO  # [0, RATIO) — same for B/A sides

        # ── kv_k load (mutually-exclusive masks; loads sum) ──
        s_off_in_input = s - start_pos  # only valid when is_input
        kv_a = tl.load(
            kv_in_ptr + s_off_in_input * dim_full + col_off + d,
            mask=is_input & d_mask,
            other=0.0,
        )
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
        kv_k = (
            kv_a + kv_b
        )  # exactly one path active per source pos (or both 0 for padding)

        # ── score_k load ──
        score_a = tl.load(
            score_in_ptr + s_off_in_input * dim_full + col_off + d,
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
        # Padding rows: score=-inf (other=NEG_INF on both branches → tl.where picks NEG_INF).
        score_k = tl.where(is_input, score_a + ape_v, score_b)

        # ── Online softmax-pool accumulate ──
        # Guard against -inf - -inf = NaN by masking the exp() inputs:
        #   if m_acc == -inf (no valid row yet)   → scale = 0
        #   if score_k == -inf (padding row)      → w_k = 0
        m_new = tl.maximum(m_acc, score_k)
        scale = tl.where(m_acc == NEG_INF, 0.0, tl.exp(m_acc - m_new))
        w_k = tl.where(score_k == NEG_INF, 0.0, tl.exp(score_k - m_new))
        kv_acc = kv_acc * scale + w_k * kv_k
        w_acc = w_acc * scale + w_k
        m_acc = m_new

    compressed = (
        kv_acc / w_acc
    )  # [BLOCK_D] fp32; padding lanes get 0/0=nan but masked off below

    # ── 2. RMSNorm (fp32) ──────────────────────────────────────────────
    rms_w = tl.load(rms_weight_ptr + d, mask=d_mask, other=0.0)
    # Mask padded lanes to 0 before squaring so var is over real elements only.
    compressed_masked = tl.where(d_mask, compressed, 0.0)
    var = tl.sum(compressed_masked * compressed_masked, axis=0) / head_dim
    rrms = tl.rsqrt(var + rms_eps)
    normed = compressed_masked * rrms * rms_w  # [BLOCK_D] fp32

    # ── 3. RoPE on rope_head_dim segment (GPT-J interleaved, fp32) ────
    # vLLM's reshape/split/interleave pattern (compress_quant_cache.py).
    # Reshape `normed` [BLOCK_D] → [NUM_PAIRS, 2], split into even/odd halves,
    # rotate rope pairs only (nope pairs gated to identity via masked cos/sin),
    # interleave back to [BLOCK_D].
    comp_pos = (pos // RATIO) * RATIO
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

    # 4a. Output buffer (caller wraps as [1, n_max, head_dim] for sparse_attn).
    # Write at row prog_id (NOT k_global) so caller-side topk indices
    # [0, n_actual) map directly to output rows.
    tl.store(out_ptr + prog_id * out_token_stride + d, rotated_bf16, mask=d_mask)

    # 4b. KV cache scatter (paged): block_table[ci // k_per_block][ci % k_per_block]
    if HAS_BLOCK_TABLE:
        ci = pos // RATIO
        block_in_seq = ci // k_per_block
        slot_in_block = ci % k_per_block
        physical_block = tl.load(block_table_ptr + block_in_seq).to(tl.int64)
        cache_addr = (
            physical_block * kv_cache_block_stride
            + slot_in_block * kv_cache_token_stride
            + d
        )
        tl.store(kv_cache_ptr + cache_addr, rotated_bf16, mask=d_mask)


def fused_compress_attn(
    *,
    # Source tensors
    kv_in: torch.Tensor,  # [token_num, dim_full] fp32
    score_in: torch.Tensor,  # [token_num, dim_full] fp32 (raw, no ape)
    kv_state: torch.Tensor,  # [num_slots, STATE_SIZE, dim_full] fp32
    score_state: torch.Tensor,  # same shape, score has ape pre-added
    slot: int,
    # Per-token absolute positions (kernel loads positions[0] for start_pos)
    positions: torch.Tensor,
    # Per-seq end-of-context (kernel loads context_lens[0] for end_pos)
    context_lens: torch.Tensor,
    # Compressor params
    ape: torch.Tensor,  # [ratio, dim_full] fp32
    rms_weight: torch.Tensor,  # [head_dim] fp32
    rms_eps: float,
    cos_cache: torch.Tensor,  # [max_seq, 1, 1, rope_head_dim/2] bf16/fp16
    sin_cache: torch.Tensor,  # same shape
    # KV cache scatter
    kv_cache: Optional[torch.Tensor],  # [num_blocks, k_per_block, head_dim] bf16
    block_table: Optional[torch.Tensor],
    k_per_block: int,
    # Geometry
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Optional[torch.Tensor]:
    """Fused per-source-position pool + RMSNorm + RoPE + bf16 kv_cache scatter.

    Returns `[1, n_max, head_dim]` BF16 PADDED tensor where
    `n_max = (token_num + ratio - 1) // ratio` is the start_pos-free boundary
    upper bound. Rows `[n_actual, n_max)` are UNINITIALIZED — downstream
    `sparse_attn` is gather-based and only reads rows referenced by topk_idxs
    (range `[0, n_actual)`), so padded rows are never accessed and their
    content does not affect numerics.

    Returns None if `token_num == 0`.

    Caller MUST invoke BEFORE `update_compressor_states` (so state cache still
    holds previous-fwd data). See module docstring for the full invariant.

    `start_pos` and `end_pos` are derived inside the kernel from
    `tl.load(positions_ptr)` and `tl.load(context_lens_ptr)` respectively —
    no caller-supplied start_pos.

    TODO: FP8/FP4 quant fusion. For now stores raw BF16; downstream values
    lack the QAT round-trip from `act_quant_inplace`. This is acceptable for
    initial integration; follow-up PR will add ue8m0-format FP8 quant.
    """
    token_num = positions.shape[0]
    if token_num == 0:
        return None

    n_max = (token_num + ratio - 1) // ratio

    device = positions.device

    # Validate shapes
    dim_full = (2 if overlap else 1) * head_dim
    state_size = (2 if overlap else 1) * ratio
    assert kv_in.shape == (
        token_num,
        dim_full,
    ), f"kv_in {kv_in.shape} != ({token_num}, {dim_full})"
    assert score_in.shape == kv_in.shape
    assert kv_state.shape[1] == state_size and kv_state.shape[2] == dim_full
    assert score_state.shape == kv_state.shape
    assert ape.shape == (ratio, dim_full)
    assert rms_weight.shape == (head_dim,)
    assert context_lens.numel() >= 1, "context_lens must be a tensor with >=1 element"
    # cos/sin cache: [max_seq, ..., rope_head_dim/2] — last dim is per-pair freq.
    # _V4RoPE stores 4D [max_seq, 1, 1, rope_head_dim/2]; tests pass 2D.
    # We use stride(0) to address regardless of intermediate dims.
    assert cos_cache.shape[-1] == rope_head_dim // 2
    assert sin_cache.shape[-1] == rope_head_dim // 2
    assert cos_cache.stride(0) == rope_head_dim // 2, (
        f"cos_cache outer stride {cos_cache.stride(0)} != rope_head_dim/2 "
        f"{rope_head_dim // 2}; non-contiguous?"
    )
    assert kv_in.is_contiguous() and score_in.is_contiguous()
    assert kv_state.is_contiguous() and score_state.is_contiguous()
    assert ape.is_contiguous() and rms_weight.is_contiguous()
    if block_table is not None:
        assert kv_cache is not None and kv_cache.dim() == 3
        # kv_cache may be a view of a per-req cache pool (PR3-pre2c-B); strides
        # are passed to the kernel explicitly so non-contiguous-but-strided is OK.
        assert block_table.is_contiguous()

    # `torch.empty` (NOT zeros) — padded rows [n_actual, n_max) are intentionally
    # uninitialized. Downstream sparse_attn (gather-based via kv[batch_idx, safe_idxs])
    # only reads rows referenced by topk_idxs ∈ [0, n_actual), so padded rows
    # are never accessed; their content does not affect numerics.
    out = torch.empty(n_max, head_dim, dtype=out_dtype, device=device)

    BLOCK_D = triton.next_power_of_2(head_dim)
    HALF_ROPE = rope_head_dim // 2
    K = state_size
    has_bt = block_table is not None

    grid = (n_max,)
    _fused_compress_attn_kernel[grid](
        kv_in,
        score_in,
        dim_full,
        positions,
        context_lens,
        kv_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state,
        score_state.stride(0),
        score_state.stride(1),
        slot,
        ape,
        rms_weight,
        rms_eps,
        cos_cache,
        sin_cache,
        cos_cache.stride(0),  # cos_sin_pos_stride (= rope_head_dim//2 for contig)
        kv_cache if has_bt else cos_cache,  # placeholder if no scatter
        kv_cache.stride(0) if has_bt else 0,
        kv_cache.stride(1) if has_bt else 0,
        block_table if has_bt else positions,  # placeholder
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

    return out.unsqueeze(0)  # [1, n_max, head_dim] (padded)


def fused_compress_attn_reference(
    *,
    kv_in: torch.Tensor,
    score_in: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    slot: int,
    positions: torch.Tensor,
    context_lens: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    k_per_block: int,
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Optional[torch.Tensor]:
    """Pure-PyTorch reference equivalent of `fused_compress_attn`.

    Replicates the per-source-position scheme + RMSNorm + GPT-J RoPE + bf16
    scatter, matching the kernel byte-for-byte (modulo Triton/PyTorch fp
    reduction order — practically `allclose(atol=1e-5)`).

    Returns `[1, n_max, head_dim]` BF16 PADDED tensor (same convention as the
    Triton kernel). Padded rows `[n_actual, n_max)` are uninitialized and must
    not be compared against the kernel output.
    """
    token_num = positions.shape[0]
    if token_num == 0:
        return None

    n_max = (token_num + ratio - 1) // ratio
    start_pos = int(positions[0].item())
    end_pos = int(context_lens[0].item())
    device = kv_in.device
    K = (2 if overlap else 1) * ratio
    state_size = K
    k_min = start_pos // ratio

    out = torch.empty(n_max, head_dim, dtype=out_dtype, device=device)

    for prog_id in range(n_max):
        k_global = k_min + prog_id
        p = (k_global + 1) * ratio - 1
        if p >= end_pos:
            continue  # padded row — leave out[prog_id] uninitialized

        # Gather K source rows
        kv_rows = []
        score_rows = []
        for k in range(K):
            s = p - K + 1 + k
            if overlap:
                col_off = head_dim if k >= ratio else 0
            else:
                col_off = 0
            ape_row = k % ratio
            d_slice = slice(col_off, col_off + head_dim)

            if s < 0:
                kv_rows.append(
                    torch.zeros(head_dim, dtype=torch.float32, device=device)
                )
                score_rows.append(
                    torch.full(
                        (head_dim,), float("-inf"), dtype=torch.float32, device=device
                    )
                )
            elif s >= start_pos:
                row = s - start_pos
                kv_rows.append(kv_in[row, d_slice].float())
                score_rows.append(
                    score_in[row, d_slice].float() + ape[ape_row, d_slice].float()
                )
            else:
                ring = s % state_size
                kv_rows.append(kv_state[slot, ring, d_slice].float())
                score_rows.append(score_state[slot, ring, d_slice].float())

        kv_stack = torch.stack(kv_rows, dim=0)  # [K, head_dim]
        sc_stack = torch.stack(score_rows, dim=0)  # [K, head_dim]
        weights = torch.softmax(sc_stack, dim=0)
        compressed = (weights * kv_stack).sum(dim=0)  # [head_dim] fp32

        # RMSNorm
        var = (compressed * compressed).mean()
        normed = compressed * torch.rsqrt(var + rms_eps) * rms_weight.float()

        # RoPE on rope_head_dim segment, GPT-J interleaved.
        comp_pos = (p // ratio) * ratio
        rope_seg = normed[-rope_head_dim:].clone()
        cos_v = cos_cache[comp_pos].view(-1).float()  # [rope_head_dim/2]
        sin_v = sin_cache[comp_pos].view(-1).float()
        even = rope_seg[0::2]
        odd = rope_seg[1::2]
        new_even = even * cos_v - odd * sin_v
        new_odd = odd * cos_v + even * sin_v
        rotated_seg = torch.stack([new_even, new_odd], dim=-1).flatten()
        normed[-rope_head_dim:] = rotated_seg

        out_bf16 = normed.to(out_dtype)
        out[prog_id] = out_bf16

        # KV cache scatter
        if block_table is not None and kv_cache is not None:
            ci = p // ratio
            block_in_seq = ci // k_per_block
            slot_in_block = ci % k_per_block
            physical = int(block_table[block_in_seq].item())
            kv_cache[physical, slot_in_block] = out_bf16

    return out.unsqueeze(0)  # [1, n_max, head_dim] (padded)
