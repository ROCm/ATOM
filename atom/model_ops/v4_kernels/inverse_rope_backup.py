# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused inverse RoPE Triton kernel for V4 attention output.

Replaces the two-step `freqs_for_positions` + `_apply_rotary_emb(inverse=True)`
with a single kernel that indexes the cos/sin cache by positions and does the
inverse rotation in-place. GPT-J (interleaved) style only.

Two modes:
  - quant_out=False (default): pure inverse RoPE, in-place BF16 write to `x`.
    Used by `_V4RoPE.inverse`.
  - quant_out=True: fused inverse RoPE + per-D128 amax + UE8M0-rounded scale +
    FP8 e4m3 cast + packed-i32 scale write. Used by `DeepseekV4Attention.
    forward_impl` when feeding the wo_a grouped LoRA via the flydsl
    fp8_einsum kernel. The quant axis groups heads into `n_groups` chunks of
    `n_heads_per_group * head_dim = d_per_group` columns each.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Mode A: pure inverse RoPE (rope slice only). Unchanged from PR1.
# ---------------------------------------------------------------------------
@triton.jit
def _inverse_rope_gptj_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    stride_x_s,
    stride_x_h,
    stride_x_d,
    stride_cos_s,
    stride_cos_d,
    S,
    BLOCK_S: tl.constexpr,
    BLOCK_RD: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_RD)
    s_mask = s_offs < S

    pos = tl.load(pos_ptr + s_offs, mask=s_mask)

    # GPT-J with reuse_freqs_front_part: cos/sin have rd//2 entries,
    # each used for a pair of adjacent elements → index = d_offs // 2
    d_cos_offs = d_offs // 2
    cos_offs = pos[:, None] * stride_cos_s + d_cos_offs[None, :] * stride_cos_d
    cos_mask = s_mask[:, None] & (d_cos_offs < BLOCK_RD_HALF)[None, :]
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    x_offs = (
        s_offs[:, None] * stride_x_s + pid_h * stride_x_h + d_offs[None, :] * stride_x_d
    )
    x_mask = s_mask[:, None] & (d_offs < BLOCK_RD)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # GPT-J inverse: swap pairs and negate evens of (x * sin), then add x * cos.
    # Forward: out[2i]   =  x[2i]*cos - x[2i+1]*sin
    #          out[2i+1] =  x[2i]*sin + x[2i+1]*cos
    # Inverse: out[2i]   =  x[2i]*cos + x[2i+1]*sin
    #          out[2i+1] = -x[2i]*sin + x[2i+1]*cos
    x_sin = x * sin
    even_mask = (d_offs % 2 == 0)[None, :]
    x_negated = tl.where(even_mask, -x_sin, x_sin)
    x_negated = tl.reshape(x_negated, (BLOCK_S, BLOCK_RD_HALF, 2))
    x_negated = tl.flip(x_negated, 2)
    x_rotated = tl.reshape(x_negated, (BLOCK_S, BLOCK_RD))

    out = x * cos + x_rotated
    out = out.to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + x_offs, out, mask=x_mask)


# ---------------------------------------------------------------------------
# Mode B: fused inverse RoPE + per-D128 UE8M0 quant epilogue.
#
# Grid:   (n_groups_G, cdiv(S, BLOCK_S))
# Per program:
#   1. Loops over the n_heads_per_group heads of this group
#   2. For each head: load nope + rope slices, apply inverse RoPE on rope slice
#   3. Compute amax across (nope|rope) within each 128-D-col group of d_per_group
#   4. Round amax/448 up to next power of 2 (UE8M0 byte)
#   5. q = x / s_pow2 → fp8 e4m3
#   6. Store fp8 row to y_ptr (B, G, d_per_group)
#   7. Pack 4 adjacent UE8M0 bytes per row into one i32; store to s_ptr
#      (B, G, d_per_group // 512)
#
# Tiling chosen to keep register footprint bounded:
#   - One D128 group at a time (128 BF16 cols per row).
#   - n_heads_per_group heads (typically 8 at V4-Pro tp=1) traversed in a
#     compile-time loop so head-relative rope masking stays static.
# ---------------------------------------------------------------------------
@triton.jit
def _inverse_rope_quant_kernel(
    x_ptr,  # bf16 [S, n_heads, head_dim]
    y_ptr,  # fp8  [S, G, d_per_group]
    s_ptr,  # i32  [S, G, d_per_group // 512]
    cos_ptr,  # bf16 [max_pos, 1, 1, rd//2]
    sin_ptr,
    pos_ptr,  # i32/i64 [S]
    stride_x_s,
    stride_x_h,
    stride_x_d,
    stride_y_s,
    stride_y_g,
    stride_y_d,
    stride_s_s,
    stride_s_g,
    stride_s_d,
    stride_cos_s,
    stride_cos_d,
    S,
    BLOCK_S: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    N_HEADS_PER_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # 128
    PACK4: tl.constexpr,  # 4 (UE8M0 bytes per i32)
    FP8_MAX: tl.constexpr,  # 448.0
):
    pid_g = tl.program_id(0)  # which group of heads (G dim)
    pid_s = tl.program_id(1)  # which BLOCK_S row tile

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    pos = tl.load(pos_ptr + s_offs, mask=s_mask, other=0)

    # rope cos/sin slice for this fwd's positions, shared across all heads.
    rope_d_offs = tl.arange(0, ROPE_DIM)
    d_cos_offs = rope_d_offs // 2  # reuse_freqs_front_part
    cos_offs = pos[:, None] * stride_cos_s + d_cos_offs[None, :] * stride_cos_d
    cos_mask = s_mask[:, None]
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask, other=0.0)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask, other=0.0)
    even_mask = (rope_d_offs % 2 == 0)[None, :]

    # Walk each head inside this group: load → (rope inverse on tail) → emit
    # quantized chunks. d_per_group = N_HEADS_PER_GROUP * HEAD_DIM. We process
    # GROUP_SIZE=128 D-cols at a time so amax + UE8M0 + fp8 fits in registers.
    # NOPE_DIM and ROPE_DIM are both multiples of GROUP_SIZE in V4 (448+64=512,
    # 128 | 512), so each head spans HEAD_DIM // GROUP_SIZE block-rows.
    NUM_D128_PER_HEAD: tl.constexpr = HEAD_DIM // GROUP_SIZE  # 4 at V4-Pro
    NUM_D128_PER_GROUP: tl.constexpr = N_HEADS_PER_GROUP * NUM_D128_PER_HEAD  # 32

    # Base pointer of (this group, this row tile) in y / s (1D, per row).
    y_row_base = s_offs * stride_y_s + pid_g * stride_y_g  # [BLOCK_S]
    s_row_base = s_offs * stride_s_s + pid_g * stride_s_g  # [BLOCK_S]

    # Buffer the per-D128 UE8M0 bytes for this row tile so we can pack 4 → i32
    # at the end. Stored as int32 (one byte per slot), shape [BLOCK_S, NUM_D128_PER_GROUP].
    # tl arrays must be constexpr-sized → NUM_D128_PER_GROUP is constexpr.

    # We can't easily index-assign a tl tensor by a runtime index. Instead, write
    # each D128 group's scale byte to s_ptr-as-int8 first into a *temporary* i32
    # scratch path: do it per-pack4 using a simple structured loop that emits
    # one i32 every 4 D128 groups.
    #
    # Strategy: pack4_group_idx in [0, NUM_D128_PER_GROUP // PACK4); within
    # each pack4_group, process 4 consecutive D128 groups, accumulate their
    # UE8M0 bytes into 4 scalars (per row), pack into one i32, store.

    NUM_PACKS: tl.constexpr = NUM_D128_PER_GROUP // PACK4

    for pack_idx in tl.static_range(NUM_PACKS):
        # Accumulator for the 4 UE8M0 bytes in this pack (one per row).
        b0 = tl.zeros([BLOCK_S], dtype=tl.int32)
        b1 = tl.zeros([BLOCK_S], dtype=tl.int32)
        b2 = tl.zeros([BLOCK_S], dtype=tl.int32)
        b3 = tl.zeros([BLOCK_S], dtype=tl.int32)

        for sub_idx in tl.static_range(PACK4):
            d128_idx = pack_idx * PACK4 + sub_idx
            # Which head + which D128-within-head does this map to?
            head_in_group = d128_idx // NUM_D128_PER_HEAD
            d128_in_head = d128_idx % NUM_D128_PER_HEAD
            head_global = pid_g * N_HEADS_PER_GROUP + head_in_group
            d_in_head_start = d128_in_head * GROUP_SIZE  # 0, 128, 256, 384

            # Load the 128-col chunk for this (row tile, head, D128) — bf16.
            # NOPE region spans d_in_head in [0, NOPE_DIM); ROPE region spans
            # d_in_head in [NOPE_DIM, HEAD_DIM). V4 has NOPE_DIM=448, ROPE_DIM=64,
            # so D128 groups 0, 1, 2 are pure nope (one of which straddles the
            # rope boundary at d=384..511 if NOPE_DIM=448 < 512).
            #
            # General handling: this D128 chunk has some prefix in nope (len
            # = max(0, min(GROUP_SIZE, NOPE_DIM - d_in_head_start))) and the
            # remainder in rope. We handle that via per-element column index
            # and per-element nope/rope dispatch (kept simple via two masked
            # loads + a tl.where).
            chunk_d = d_in_head_start + tl.arange(0, GROUP_SIZE)  # [GROUP_SIZE]
            in_nope_mask = chunk_d < NOPE_DIM  # [GROUP_SIZE]

            # Single load over the entire 128-col chunk (head_dim contiguous).
            x_offs = (
                s_offs[:, None] * stride_x_s
                + head_global * stride_x_h
                + chunk_d[None, :] * stride_x_d
            )
            x = tl.load(x_ptr + x_offs, mask=s_mask[:, None], other=0.0)

            # Apply inverse RoPE on the rope sub-slice in place (within registers).
            # rope-relative col index = chunk_d - NOPE_DIM, valid only where !in_nope.
            rope_rel = chunk_d - NOPE_DIM  # [GROUP_SIZE]
            # Build per-chunk cos/sin gather indexed by rope_rel (clamped to a
            # safe in-range value when in nope; we'll mask the result anyway).
            rope_rel_safe = tl.where(in_nope_mask, 0, rope_rel)
            # cos has shape [BLOCK_S, ROPE_DIM]; gather along its D-axis.
            chunk_cos = tl.gather(
                cos, rope_rel_safe[None, :].broadcast_to((BLOCK_S, GROUP_SIZE)), axis=1
            )
            chunk_sin = tl.gather(
                sin, rope_rel_safe[None, :].broadcast_to((BLOCK_S, GROUP_SIZE)), axis=1
            )

            # Even/odd within the rope pair (per ELEMENT, but rope_rel may be
            # outside ROPE_DIM for nope cols — even_mask on rope_rel still
            # works because we mask the rotated contribution to 0 in nope).
            even_chunk_mask = (rope_rel % 2 == 0)[None, :]
            x_sin = x * chunk_sin
            x_neg = tl.where(even_chunk_mask, -x_sin, x_sin)
            # Pair-swap: reshape last dim into (GROUP_SIZE/2, 2), flip, reshape back.
            x_neg_2 = tl.reshape(x_neg, (BLOCK_S, GROUP_SIZE // 2, 2))
            x_neg_2 = tl.flip(x_neg_2, 2)
            x_rot = tl.reshape(x_neg_2, (BLOCK_S, GROUP_SIZE))
            x_after_rope = x * chunk_cos + x_rot

            # Where in_nope, the rope rotation must contribute zero; just keep x.
            x_post = tl.where(in_nope_mask[None, :], x, x_after_rope)

            # Per-row amax across the 128 cols of this D128 chunk.
            x_f32 = x_post.to(tl.float32)
            amax = tl.max(tl.abs(x_f32), axis=1)  # [BLOCK_S] fp32
            # Scale = amax / FP8_MAX, rounded UP to next pow2 via UE8M0 byte trick:
            #   byte = ((bits + 0x007FFFFF) & 0xFF800000) >> 23
            # Then s_pow2 = 2^(byte - 127). Clamp byte to [0, 254].
            #
            # Note: the `0x007FFFFF` add (full mantissa-mask) is what gives us
            # round-UP-to-next-pow2 semantics, matching host
            # `fp32_to_ue8m0_byte` (frexp-based) and C++ `dsv4_rotate_quant.cu`
            # (`if (u32 & 0x7FFFFF) exponent += 1;`). Using `0x400000` instead
            # yields round-to-NEAREST-pow2, which loses an exponent whenever
            # the mantissa is in the lower half of its binade — producing
            # FP8 outputs that are 2× too large for ~half the blocks.
            scale_f32 = amax * (1.0 / FP8_MAX)
            # Replace zero / sub-normal with the smallest pow2 we encode (byte=0).
            scale_f32 = tl.maximum(scale_f32, 1.1754944e-38)  # 2^-126
            # Round amax/448 UP to next pow2 via the UE8M0 trick. Work in
            # uint32 to keep the 0xFF800000 mask in-range; cast back for shift.
            bits_u = scale_f32.to(tl.uint32, bitcast=True)
            byte_u = (
                (bits_u + tl.full([], 0x007FFFFF, tl.uint32))
                & tl.full([], 0xFF800000, tl.uint32)
            ) >> 23
            byte = byte_u.to(tl.int32)
            byte = tl.minimum(tl.maximum(byte, 0), 254)
            # s_pow2 = 2^(byte - 127), reconstruct from the UE8M0 byte so the
            # consumer's dequant matches bit-for-bit.
            pow2_bits = byte.to(tl.uint32) << 23
            s_pow2 = pow2_bits.to(tl.float32, bitcast=True)

            # Quantize → fp8 e4m3 (manual: divide, clamp, cast).
            q = x_f32 / s_pow2[:, None]
            q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
            q_fp8 = q.to(tl.float8e4nv)

            # Store the fp8 chunk into y at (s_offs, pid_g, d128_idx * GROUP_SIZE + i)
            y_d_offs = d128_idx * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
            y_offs = y_row_base[:, None] + y_d_offs[None, :] * stride_y_d
            tl.store(y_ptr + y_offs, q_fp8, mask=s_mask[:, None])

            # Stash the UE8M0 byte into the right slot of (b0, b1, b2, b3).
            if sub_idx == 0:
                b0 = byte
            elif sub_idx == 1:
                b1 = byte
            elif sub_idx == 2:
                b2 = byte
            else:
                b3 = byte

        # Pack and store the i32 scale word for this pack.
        packed = (
            (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24)
        )
        # Column `pack_idx` within (s_offs, pid_g).
        s_off_col = s_row_base + pack_idx * stride_s_d
        tl.store(s_ptr + s_off_col, packed, mask=s_mask)


def inverse_rope_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    quant_out: bool = False,
    n_groups: int | None = None,
    nope_dim: int | None = None,
):
    """In-place inverse RoPE (GPT-J style) on the rope slice of attention output.

    Args:
        x: ``[num_tokens, n_heads, head_dim]`` bf16 — typically ``o`` (full
           head_dim including nope+rope) when ``quant_out=True``, or just the
           rope slice ``o[..., -rd:]`` when ``quant_out=False``.
        cos: ``[max_seq_len, 1, 1, rotary_dim // 2]`` bf16 cache.
        sin: ``[max_seq_len, 1, 1, rotary_dim // 2]`` bf16 cache.
        positions: ``[num_tokens]`` int — per-token position ids.
        quant_out: when True, returns ``(y_fp8, sx_i32)`` quantized output
                   suitable as direct input to ``fp8_einsum``. ``x`` is
                   read-only in this mode.
        n_groups: when ``quant_out=True``, number of head-groups to fold the
                  ``n_heads`` axis into. Must divide ``n_heads`` evenly.
                  ``d_per_group = (n_heads // n_groups) * head_dim``.
        nope_dim: when ``quant_out=True``, size of the leading nope slice in
                  each head. Defaults to ``head_dim - cos.shape[-1] * 2``
                  (rope-slice = full rotary_dim, nope = remainder).

    Returns:
        ``None`` when ``quant_out=False``.
        ``(y_fp8, sx_i32)`` when ``quant_out=True``:
          - ``y_fp8``: ``[num_tokens, n_groups, d_per_group]`` float8_e4m3fn
          - ``sx_i32``: ``[num_tokens, n_groups, d_per_group // 512]`` int32
                       (4 UE8M0 bytes packed little-endian per i32).
    """
    if not quant_out:
        S, H, rd = x.shape
        BLOCK_RD = triton.next_power_of_2(rd)
        BLOCK_S = 32
        grid = (H, triton.cdiv(S, BLOCK_S))
        _inverse_rope_gptj_kernel[grid](
            x,
            cos,
            sin,
            positions,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            cos.stride(0),
            cos.stride(3),
            S,
            BLOCK_S=BLOCK_S,
            BLOCK_RD=BLOCK_RD,
            BLOCK_RD_HALF=BLOCK_RD // 2,
            num_warps=4,
        )
        return None

    # quant_out=True path: full head_dim input, fused per-D128 UE8M0 + fp8 cast.
    assert n_groups is not None, "n_groups required when quant_out=True"
    S, H, head_dim = x.shape
    assert H % n_groups == 0, f"n_heads={H} must be divisible by n_groups={n_groups}"
    n_heads_per_group = H // n_groups
    d_per_group = n_heads_per_group * head_dim
    GROUP_SIZE = 128
    PACK4 = 4
    assert (
        d_per_group % (GROUP_SIZE * PACK4) == 0
    ), f"d_per_group={d_per_group} must be divisible by {GROUP_SIZE * PACK4}"
    # cos.shape[-1] = rotary_dim // 2 (reuse_freqs_front_part layout)
    rope_dim = cos.shape[-1] * 2
    if nope_dim is None:
        nope_dim = head_dim - rope_dim
    assert (
        nope_dim + rope_dim == head_dim
    ), f"nope_dim={nope_dim} + rope_dim={rope_dim} != head_dim={head_dim}"
    assert (
        head_dim % GROUP_SIZE == 0
    ), f"head_dim={head_dim} must be multiple of {GROUP_SIZE}"
    assert (
        rope_dim <= GROUP_SIZE
    ), f"rope_dim={rope_dim} must be <= {GROUP_SIZE} (one rope D128 chunk)"

    device = x.device
    y = torch.empty(S, n_groups, d_per_group, dtype=torch.float8_e4m3fn, device=device)
    sx = torch.empty(
        S,
        n_groups,
        d_per_group // (GROUP_SIZE * PACK4),
        dtype=torch.int32,
        device=device,
    )

    BLOCK_S = 32
    grid = (n_groups, triton.cdiv(S, BLOCK_S))
    _inverse_rope_quant_kernel[grid](
        x,
        y,
        sx,
        cos,
        sin,
        positions,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        sx.stride(0),
        sx.stride(1),
        sx.stride(2),
        cos.stride(0),
        cos.stride(3),
        S,
        BLOCK_S=BLOCK_S,
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        N_HEADS_PER_GROUP=n_heads_per_group,
        GROUP_SIZE=GROUP_SIZE,
        PACK4=PACK4,
        FP8_MAX=448.0,
        num_warps=4,
    )
    return y, sx
