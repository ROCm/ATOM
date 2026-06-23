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
# Ported from vLLM's `_fused_inv_rope_fp8_quant_per_head`
# (vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py).
#
# Key structural wins adopted from vLLM:
#   - Grid = (n_heads, cdiv(S, BLOCK_S)) → one program per (head, token-tile),
#     each handling ONE head's CHUNKS_PER_HEAD D128 blocks. Massively more
#     workgroups than the old (n_groups, cdiv(S,32)) grid (which serialized
#     all 32 D128 chunks of a group in one WG → ~8 WGs at small B).
#   - No tl.gather / tl.flip: inverse RoPE via the `offsets ^ 1` partner-load
#     trick + even/odd tl.where. Much cheaper than the gather+flip path.
#
# Kept from OUR kernel (so output is BIT-IDENTICAL → true drop-in):
#   - Separate cos/sin caches `[max_pos, 1, 1, rope_dim//2]` with the
#     reuse_freqs_front_part `// 2` indexing (NOT vLLM's fused cos||sin cache).
#   - UE8M0 byte = round-UP via `(bits + 0x007FFFFF) & 0xFF800000) >> 23`,
#     clamp [0, 254], scale floor 2^-126 — matches host `fp32_to_ue8m0_byte`
#     and the C++ reference (vLLM's `exp2(ceil(log2))` differs on ~64% of
#     inputs, so we do NOT use it).
#   - Output layout: y_fp8 [S, n_groups, d_per_group] row-major; sx int32
#     [S, n_groups, d_per_group//512] with 4 UE8M0 bytes packed LE per i32.
# ---------------------------------------------------------------------------
# Grid = (num_tokens, n_groups * heads_per_group): ONE program per
# (token, head), each handling that head's CHUNKS_PER_HEAD D128 blocks. This
# vLLM-derived layout (vllm fused_inv_rope_fp8_quant) beats the prior
# (n_heads, cdiv(S, BLOCK_S)) token-tiled grid by ~1.6-1.9× at prefill on
# gfx950 (MI355X): the flat per-head HEAD_DIM load is fully coalesced and
# carries far less register/ALU pressure than the old 2D [BLOCK_S, HEAD_DIM]
# tile, so the kernel stops being VALU-issue-bound (measured VALUBusy 79% →
# memory-streaming) at large S. num_warps=1 (one wavefront issues the whole
# head row; extra warps only add scheduling overhead). No @triton.autotune —
# the grid is a pure function of (tokens, heads), so it is CUDA-graph-capture
# safe and spike-free with no per-batch search.
#
# RoPE math runs in FP32 (matches vLLM). This is NOT byte-identical to the
# older BF16-rope kernel — ~0.8% of fp8 bytes differ by ≤1 UE8M0 ULP — but
# stays well within fp8 e4m3 tolerance (≈6% per-element ULP); the downstream
# fp8_einsum consumer is unaffected (see op_tests/test_inverse_rope_quant.py
# and test_v4_fp8_wo_a.py). FP32 rope removes the BF16 pack/unpack overhead
# that was the remaining gap to vLLM's kernel.


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
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    N_HEADS_PER_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # 128
    PACK4: tl.constexpr,  # 4 (UE8M0 bytes per i32)
    FP8_MAX: tl.constexpr,  # 448.0
):
    # One program per (token, head). Grid = (num_tokens, n_heads).
    # int64 ids: stride math overflows int32 past S=32768 (IMA).
    pid_s = tl.program_id(0).to(tl.int64)  # token
    pid_h = tl.program_id(1).to(tl.int64)  # global head in [0, n_heads)

    head_in_group = pid_h % N_HEADS_PER_GROUP
    pid_g = pid_h // N_HEADS_PER_GROUP

    CHUNKS_PER_HEAD: tl.constexpr = HEAD_DIM // GROUP_SIZE  # 4 at V4-Pro
    # Where this head's D128 chunks land within the group's d_per_group axis.
    qb_start = head_in_group * CHUNKS_PER_HEAD  # first D128 idx of this head

    pos = tl.load(pos_ptr + pid_s)

    # Whole-head column offsets [HEAD_DIM]; rope lives in the tail.
    d_offs = tl.arange(0, HEAD_DIM)
    rope_abs_start = NOPE_DIM  # rope occupies [NOPE_DIM, HEAD_DIM)
    is_rope = d_offs >= rope_abs_start  # [HEAD_DIM]
    rope_local = d_offs - rope_abs_start  # valid where is_rope

    # Flat per-(token, head) load of the whole head row: [HEAD_DIM]. Fully
    # coalesced (contiguous along d). RoPE math in FP32 (vLLM-style) — load is
    # bf16, immediately upcast. Not byte-identical to the old BF16-rope kernel
    # (~0.8% of fp8 bytes differ by ≤1 ULP), but within fp8 tolerance.
    base = x_ptr + pid_s * stride_x_s + pid_h * stride_x_h
    x = tl.load(base + d_offs * stride_x_d).to(tl.float32)  # [HEAD_DIM] fp32

    # Partner element for the interleaved (GPT-J) rope pair: index ^ 1.
    x_partner = tl.load(base + (d_offs ^ 1) * stride_x_d, mask=is_rope, other=0.0).to(
        tl.float32
    )

    # cos/sin gathered by the rope-pair index (reuse_freqs_front_part: //2).
    # cos cache is [max_pos, 1, 1, rd//2]; index = rope_local // 2.
    cs_idx = tl.maximum(rope_local >> 1, 0)  # [HEAD_DIM]
    cos_addr = pos * stride_cos_s + cs_idx * stride_cos_d
    cos_v = tl.load(cos_ptr + cos_addr, mask=is_rope, other=1.0)
    sin_v = tl.load(sin_ptr + cos_addr, mask=is_rope, other=0.0)

    # Inverse (transpose) GPT-J rotation in FP32: even lanes x*cos + partner*sin,
    # odd lanes x*cos - partner*sin. Applied only on rope cols.
    is_even = (rope_local & 1) == 0  # [HEAD_DIM]
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)  # [HEAD_DIM] fp32

    # Per-D128-block amax over the head's CHUNKS_PER_HEAD chunks.
    x_blk = tl.reshape(x, (CHUNKS_PER_HEAD, GROUP_SIZE))
    amax = tl.max(tl.abs(x_blk), axis=1)  # [CHUNKS_PER_HEAD]

    # UE8M0 byte via the bit-trick round-up: ((bits + 0x7FFFFF) & 0xFF800000)
    # >> 23, clamp [0,254]. Matches the DeepGEMM host reference. The pow2 scale
    # is reconstructed from the byte so the consumer's dequant is bit-exact.
    scale_f32 = amax * (1.0 / FP8_MAX)
    scale_f32 = tl.maximum(scale_f32, 1.1754944e-38)  # 2^-126 floor
    bits_u = scale_f32.to(tl.uint32, bitcast=True)
    byte_u = (
        (bits_u + tl.full([], 0x007FFFFF, tl.uint32))
        & tl.full([], 0xFF800000, tl.uint32)
    ) >> 23
    byte = tl.minimum(tl.maximum(byte_u.to(tl.int32), 0), 254)  # [CHUNKS_PER_HEAD]
    s_pow2 = (byte.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    # Quantize: divide each col by its block's pow2 scale, clamp, fp8 cast.
    s_pow2_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(s_pow2, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    q = x / s_pow2_exp
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
    q_fp8 = q.to(tl.float8e4nv)

    # Store fp8 head into y at (s, g, qb_start*GROUP_SIZE + d_offs).
    y_d0 = qb_start * GROUP_SIZE
    y_offs = pid_s * stride_y_s + pid_g * stride_y_g + (y_d0 + d_offs) * stride_y_d
    tl.store(y_ptr + y_offs, q_fp8)

    # Pack scale bytes into one i32 word. Each i32 covers PACK4 consecutive D128
    # blocks (= 512 cols). When CHUNKS_PER_HEAD == PACK4 (V4-Pro: 4 == 4) each
    # head's blocks map to EXACTLY ONE word (qb_start is a multiple of PACK4,
    # so word = qb_start // PACK4 = head_in_group and lanes are 0..PACK4-1).
    # No cross-head word sharing → a single packed store per head, no atomics.
    tl.static_assert(
        CHUNKS_PER_HEAD == PACK4,
        "packed-scale store assumes CHUNKS_PER_HEAD == PACK4 (one word/head)",
    )
    lane_shift = tl.arange(0, CHUNKS_PER_HEAD) * 8  # [CHUNKS]
    packed = tl.sum((byte & 0xFF) << lane_shift)  # scalar i32
    word = qb_start // PACK4  # == head_in_group
    s_addr = pid_s * stride_s_s + pid_g * stride_s_g + word * stride_s_d
    tl.store(s_ptr + s_addr, packed)


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

    # Grid = (S, n_heads): ONE program per (token, head). Each program loads
    # that head's flat HEAD_DIM row (fully coalesced) and writes its
    # CHUNKS_PER_HEAD D128 blocks + one packed scale word. Pure function of
    # (tokens, heads) → CUDA-graph-capture safe, no @triton.autotune, no
    # per-batch search spike. num_warps=1: a single wavefront issues the head
    # row; extra warps only add scheduling overhead.
    n_heads = n_groups * n_heads_per_group
    grid = (S, n_heads)
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
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        N_HEADS_PER_GROUP=n_heads_per_group,
        GROUP_SIZE=GROUP_SIZE,
        PACK4=PACK4,
        FP8_MAX=448.0,
        num_warps=1,
    )
    return y, sx
