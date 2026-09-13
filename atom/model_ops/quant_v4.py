# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DeepSeek-V4 activation quantization helpers (PR1 torch fallbacks).

Three ops the V4 reference inference uses for Quantization-Aware Training (QAT)
simulation on activations:

    act_quant_inplace        — block-wise FP8 e4m3 round-trip (BF16 -> FP8 -> BF16)
    fp4_act_quant_inplace    — block-wise FP4 e2m1 round-trip (BF16 -> FP4 -> BF16)
    rotate_activation        — Walsh-Hadamard transform with 1/sqrt(N) scaling

The reference TileLang kernels live in /data/DeepSeek-V4-Pro/inference/kernel.py.
PR1 ships pure-torch fallbacks for numerical correctness; production-perf paths
land alongside the AITER `sparse_attn` kernel in PR4.

These ops do NOT change tensor shape or dtype — they round-trip the values
in-place to simulate the precision loss of low-bit storage.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

# FP4 e2m1 representable magnitudes (positive half). Symmetric around 0.
# Reference: /data/DeepSeek-V4-Pro/inference/convert.py:11-14
_FP4_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# FP4 e2m1 full lookup table (16 entries: low nibble 0..7 = positive,
# low nibble 8..15 = negative). Matches convert.py:11-14 exactly.
_FP4_LOOKUP = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def dequant_fp4_e2m1(
    packed: torch.Tensor,
    scale: torch.Tensor,
    fp4_block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed FP4 e2m1 weights with per-block ue8m0 scale.

    On-disk format used by DeepSeek-V4-Pro expert weights:
      - `packed`: int8 [..., out, in/2]. Each byte holds 2 FP4 values:
            byte = (high_nibble << 4) | low_nibble
            position 2*j   ← FP4_LOOKUP[low_nibble]
            position 2*j+1 ← FP4_LOOKUP[high_nibble]
      - `scale`: float8_e8m0fnu [..., out, in/fp4_block_size]. Power-of-2
            scaling factor for each contiguous block of `fp4_block_size`
            values along the input dim.

    Reference: convert.py:cast_e2m1fn_to_e4m3fn (lines 17-52). The first
    half of that function does the unpack; we then apply the per-block
    scale directly to BF16 instead of repacking into FP8 e4m3.

    Args:
        packed: int8 tensor with shape [..., out, in/2]
        scale: float8_e8m0fnu (or any float) tensor with shape [..., out, in/fp4_block_size]
        fp4_block_size: scaling block size along input dim (default 32)
        out_dtype: dtype to return (default bfloat16)

    Returns:
        Dequantized tensor with shape [..., out, in], dtype=out_dtype.
    """
    assert packed.dtype == torch.int8, f"packed must be int8, got {packed.dtype}"
    assert packed.dim() >= 2, f"packed must be ≥2D, got shape {packed.shape}"

    *prefix, out_dim, in_packed = packed.shape
    in_dim = in_packed * 2
    assert (
        in_dim % fp4_block_size == 0
    ), f"unpacked in_dim {in_dim} not divisible by fp4_block_size {fp4_block_size}"
    expected_scale = (*prefix, out_dim, in_dim // fp4_block_size)
    assert (
        tuple(scale.shape) == expected_scale
    ), f"scale shape {tuple(scale.shape)} != expected {expected_scale}"

    # Unpack: each byte → 2 FP4 values via lookup table.
    table = _FP4_LOOKUP.to(packed.device)  # [16] FP32
    u = packed.view(torch.uint8)
    low = (u & 0x0F).long()  # [..., out, in/2]
    high = ((u >> 4) & 0x0F).long()
    # Stack so adjacent positions are (low, high), then flatten the trailing pair.
    unpacked = torch.stack([table[low], table[high]], dim=-1)  # [..., out, in/2, 2]
    unpacked = unpacked.reshape(*prefix, out_dim, in_dim)  # [..., out, in]

    # Apply per-block scale. ue8m0 cast to float gives the linear scale value
    # (since float8_e8m0fnu represents pure powers of 2).
    s = scale.float()  # [..., out, in/block]
    s_expanded = s.repeat_interleave(fp4_block_size, dim=-1)  # [..., out, in]
    dequant = unpacked * s_expanded

    return dequant.to(out_dtype)


def act_quant_inplace(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> None:
    """In-place BF16 -> FP8 e4m3 -> BF16 round-trip, blocked along the last dim.

    Reference: inference/kernel.py:act_quant with `inplace=True`.

    Args:
        x:          tensor to quantize in-place; last dim must be divisible by block_size
        block_size: number of elements per scaling block (typical: 64 or 128)
        scale_fmt:  None         -> FP32 scale (no special rounding)
                    "ue8m0"      -> round scale UP to nearest power of 2 (MXFP-style)
    """
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    *prefix, n = x.shape
    assert n % block_size == 0, f"last dim {n} not divisible by block_size {block_size}"

    blocks = x.reshape(*prefix, n // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax * fp8_max_inv
    if scale_fmt == "ue8m0":
        # Match reference (ref_full_generate / aiter): round-to-even via
        # f32_to_e8m0 + e8m0_to_f32. Earlier `ceil(log2(scale))` matched the
        # TileLang reference but differed from the aiter-backed ref_full_generate
        # path by up to 1 binade — see notes/17_root_cause_input_quant.md.
        from aiter.utility import fp4_utils as _fp4u

        e8m0 = _fp4u.f32_to_e8m0(scale.contiguous())
        scale = _fp4u.e8m0_to_f32(e8m0)

    # Quantize -> FP8 -> dequantize
    quant_fp8 = (
        (blocks / scale).clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
    )
    dequant = quant_fp8.float() * scale

    x.copy_(dequant.reshape(*prefix, n).to(x.dtype))


# ---------------------------------------------------------------------------
# FP4 activation quantization — Triton kernel
# ---------------------------------------------------------------------------

# FP4 e2m1 magnitudes + midpoint thresholds.
# Magnitudes: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# Midpoints:  0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0

# Reverse lookup: magnitude → nibble index (0-7)
# {0.0→0, 0.5→1, 1.0→2, 1.5→3, 2.0→4, 3.0→5, 4.0→6, 6.0→7}
# Sign bit (bit 3) is set for negative values → nibble += 8


def _encode_fp4_to_nibble(
    abs_val: torch.Tensor, sign_mask: torch.Tensor
) -> torch.Tensor:
    """Torch reference: map snapped |v| to 4-bit nibble (0-15).
    
    Nibble 0-7 positive, 8-15 negative (sign bit = 8).
    Matches ``_FP4_LOOKUP`` indexing: LOOKUP[nibble] gives the fp32 value.
    """
    nibble = torch.zeros_like(abs_val, dtype=torch.int32)
    nibble = torch.where((abs_val > 0.25) & (abs_val <= 0.75), 1, nibble)
    nibble = torch.where((abs_val > 0.75) & (abs_val <= 1.25), 2, nibble)
    nibble = torch.where((abs_val > 1.25) & (abs_val <= 1.75), 3, nibble)
    nibble = torch.where((abs_val > 1.75) & (abs_val <= 2.5), 4, nibble)
    nibble = torch.where((abs_val > 2.5) & (abs_val <= 3.5), 5, nibble)
    nibble = torch.where((abs_val > 3.5) & (abs_val <= 5.0), 6, nibble)
    nibble = torch.where(abs_val > 5.0, 7, nibble)
    # sign bit
    nibble = torch.where(sign_mask, nibble + 8, nibble)
    return nibble


@triton.jit
def _fp4_act_quant_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    N: int,
    block_size: tl.constexpr,
    num_rows: int,
    fp4_max: tl.constexpr,
    fp4_max_inv: tl.constexpr,
    eps_amax_min: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """FP4 e2m1 activation quantization kernel: BF16 → packed uint8 + E8M0 scale.

    Grid: ``(ceil(num_rows / BLOCK_M), N // block_size)``.

    Loads inputs as individual 1D column tensors (Triton doesn't support
    column-indexing 2D register tensors), then encodes and packs pairs
    inside a ``tl.static_range`` loop.

    Pack format: byte = (nibble_high << 4) | nibble_low.
    Nibble 0-7 = positive magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    Nibble 8-15 = same magnitudes negative.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < num_rows

    # --- Step 1: per-row amax (single pass over columns) ---
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for c in tl.static_range(block_size):
        col_ptr = x_ptr + rows * N + (pid_b * block_size + c)
        col = tl.load(col_ptr, mask=row_mask, other=0.0)
        amax = tl.maximum(amax, tl.abs(col))
    amax = tl.where(amax < eps_amax_min, eps_amax_min, amax)  # [BLOCK_M]

    # --- Step 2: ue8m0 scale = 2^ceil(log2(amax / 6)) ---
    log2_val = tl.math.log2(amax * fp4_max_inv)
    log2_floor = tl.math.floor(log2_val)
    log2_ceil = tl.where(log2_val == log2_floor, log2_val, log2_floor + 1.0)
    scale = tl.math.exp2(log2_ceil)  # [BLOCK_M]

    # --- Step 3-4-5: per-pair reload, snap, encode, and pack ---
    N_PAIRS: tl.constexpr = block_size // 2
    for p in tl.static_range(N_PAIRS):
        i0 = 2 * p
        i1 = 2 * p + 1

        # Reload this pair of columns from global memory.
        v0 = tl.load(
            x_ptr + rows * N + (pid_b * block_size + i0),
            mask=row_mask, other=0.0,
        )
        v1 = tl.load(
            x_ptr + rows * N + (pid_b * block_size + i1),
            mask=row_mask, other=0.0,
        )

        # Normalize + clamp column i0
        n0 = v0 / scale
        n0 = tl.clamp(n0, -fp4_max, fp4_max)

        # Snap |n0| to nearest FP4 magnitude + encode to nibble
        a0 = tl.abs(n0)
        nib0 = tl.zeros((BLOCK_M,), dtype=tl.int32)
        nib0 = tl.where((a0 > 0.25) & (a0 <= 0.75), 1, nib0)
        nib0 = tl.where((a0 > 0.75) & (a0 <= 1.25), 2, nib0)
        nib0 = tl.where((a0 > 1.25) & (a0 <= 1.75), 3, nib0)
        nib0 = tl.where((a0 > 1.75) & (a0 <= 2.5), 4, nib0)
        nib0 = tl.where((a0 > 2.5) & (a0 <= 3.5), 5, nib0)
        nib0 = tl.where((a0 > 3.5) & (a0 <= 5.0), 6, nib0)
        nib0 = tl.where(a0 > 5.0, 7, nib0)
        nib0 = nib0 + tl.where(n0 < 0.0, 8, 0)

        # Normalize + clamp column i1
        n1 = v1 / scale
        n1 = tl.clamp(n1, -fp4_max, fp4_max)

        # Snap + encode column i1
        a1 = tl.abs(n1)
        nib1 = tl.zeros((BLOCK_M,), dtype=tl.int32)
        nib1 = tl.where((a1 > 0.25) & (a1 <= 0.75), 1, nib1)
        nib1 = tl.where((a1 > 0.75) & (a1 <= 1.25), 2, nib1)
        nib1 = tl.where((a1 > 1.25) & (a1 <= 1.75), 3, nib1)
        nib1 = tl.where((a1 > 1.75) & (a1 <= 2.5), 4, nib1)
        nib1 = tl.where((a1 > 2.5) & (a1 <= 3.5), 5, nib1)
        nib1 = tl.where((a1 > 3.5) & (a1 <= 5.0), 6, nib1)
        nib1 = tl.where(a1 > 5.0, 7, nib1)
        nib1 = nib1 + tl.where(n1 < 0.0, 8, 0)

        # Pack: low nibble = nib0, high nibble = nib1
        byte_val = nib0 | (nib1 << 4)

        # Store packed byte
        out_col = pid_b * N_PAIRS + p
        tl.store(
            packed_ptr + rows * (N // 2) + out_col,
            byte_val.to(tl.uint8),
            mask=row_mask,
        )

    # --- Step 6: write scale (E8M0 per row per block) ---
    tl.store(
        scale_ptr + rows * (N // block_size) + pid_b,
        scale,
        mask=row_mask,
    )


def _fp4_act_quant_triton(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 32,
) -> None:
    """Dispatch FP4 quantize+pack to Triton kernel."""
    *prefix, n = x.shape
    num_rows = 1
    for d in prefix:
        num_rows *= d

    if not x.is_cuda:
        x = x.cuda()
        packed = packed.cuda()
        scale = scale.cuda()

    # Tune BLOCK_M to row count for occupancy.
    if num_rows >= 512:
        BLOCK_M = 128
    elif num_rows >= 128:
        BLOCK_M = 64
    elif num_rows >= 32:
        BLOCK_M = 32
    else:
        BLOCK_M = max(1, num_rows)

    num_blocks = n // block_size
    grid = (triton.cdiv(num_rows, BLOCK_M), num_blocks)

    _fp4_act_quant_kernel[grid](
        x,
        packed,
        scale,
        n,
        block_size,
        num_rows,
        fp4_max=6.0,
        fp4_max_inv=1.0 / 6.0,
        eps_amax_min=6.0 * (2.0**-126),
        BLOCK_M=BLOCK_M,
    )


def _fp4_act_quant_torch(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 32,
) -> None:
    """Pure-torch FP4 quantize+pack reference implementation (writes float32 scale)."""
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    eps_amax = 6.0 * (2.0**-126)

    *prefix, n = x.shape

    # Reshape to blocks: [..., n_blocks, block_size]
    blocks = x.reshape(*prefix, n // block_size, block_size).float()

    # Per-block amax + ue8m0 scale
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=eps_amax)
    block_scale = torch.pow(2.0, torch.ceil(torch.log2(amax * fp4_max_inv)))

    # Normalize + clamp
    normalized = (blocks / block_scale).clamp(min=-fp4_max, max=fp4_max)

    # Snap to nearest FP4 magnitude
    fp4_vals = _FP4_MAGNITUDES.to(normalized.device)
    diff = (normalized.abs().unsqueeze(-1) - fp4_vals).abs()
    snapped_mag = fp4_vals[diff.argmin(dim=-1)]
    snapped = torch.where(normalized < 0, -snapped_mag, snapped_mag)

    # Encode to nibbles: map snapped |v| to 0-7, set sign bit
    nibbles = _encode_fp4_to_nibble(
        snapped.abs(), (normalized < 0)
    )  # [..., n_blocks, block_size]

    # Pack pairs: byte = (nibble_high << 4) | nibble_low
    nibbles_even = nibbles[..., ::2]
    nibbles_odd = nibbles[..., 1::2]
    packed_vals = (nibbles_even | (nibbles_odd << 4)).to(torch.uint8)
    packed_out = packed_vals.reshape(*prefix, n // 2)

    # scale: [..., n_blocks] (float32), remove keepdim
    scale_out = block_scale.squeeze(-1)

    packed.copy_(packed_out)
    scale.copy_(scale_out)


def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 → packed FP4 e2m1 uint8 + E8M0 scale, per-1×block_size blocks.

    This is the production quantize-and-pack kernel required by the
    DeepSeek-V4 Indexer and Compressor paths (issue #807).

    Uses a Triton kernel on AMD ROCm for performance; falls back to a
    pure-torch implementation when ``ATOM_FP4_TORCH_FALLBACK=1`` is set
    or CUDA is unavailable.

    Pack format (matching ``dequant_fp4_e2m1``):
      - ``packed``: uint8 [..., N//2], 2 FP4 values per byte.
        Byte = (nibble_high << 4) | nibble_low, where nibble ∈ [0, 15].
        Nibbles 0-7 map to positive magnitudes, 8-15 to negatives.
      - ``scale``: float8_e8m0fnu [..., N//block_size], per-block ue8m0 scale.

    Args:
        x:          BF16 tensor [..., N]; N must be even and divisible by
                    ``block_size``.
        block_size: elements per scaling block (default 32 for FP4 e2m1).

    Returns:
        (packed_uint8, scale_float8_e8m0fnu) tuple.
    """
    *prefix, n = x.shape
    assert n % 2 == 0, f"last dim {n} must be even for nibble packing"
    assert n % block_size == 0, f"last dim {n} not divisible by block_size {block_size}"

    num_rows = 1
    for d in prefix:
        num_rows *= d

    # Allocate output tensors.
    # Triton kernel writes float32 scale (float8_e8m0fnu not a supported
    # pointer type in Triton); we convert after kernel execution.
    packed_out = torch.empty(*prefix, n // 2, dtype=torch.uint8, device=x.device)
    scale_f32 = torch.empty(*prefix, n // block_size, dtype=torch.float32, device=x.device)

    _use_triton = (
        os.environ.get("ATOM_FP4_TORCH_FALLBACK", "0") != "1"
        and x.is_cuda
        and num_rows >= 1
    )

    if _use_triton:
        try:
            _fp4_act_quant_triton(x, packed_out, scale_f32, block_size)
            return packed_out, scale_f32.to(torch.float8_e8m0fnu)
        except Exception:
            pass

    _fp4_act_quant_torch(x, packed_out, scale_f32, block_size)
    return packed_out, scale_f32.to(torch.float8_e8m0fnu)


# ---------------------------------------------------------------------------
# FP4 QAT round-trip simulation (kept for backward compatibility)
# ---------------------------------------------------------------------------


def fp4_act_quant_inplace(x: torch.Tensor, block_size: int = 32) -> None:
    """In-place BF16 → FP4 e2m1 → BF16 round-trip (QAT simulation).

    DO NOT use this for production inference — it's a precision-loss
    simulator for quantisation-aware training.  Use ``fp4_act_quant``
    for actual quantize-and-pack.

    Args:
        x:          tensor to quantize in-place; last dim must be divisible
                    by ``block_size``.
        block_size: number of elements per scaling block (default 32).
    """
    # Use the quantize-then-dequantize approach via fp4_act_quant
    packed, scale = fp4_act_quant(x, block_size)
    dequant = dequant_fp4_e2m1(
        packed.long(), scale, fp4_block_size=block_size, out_dtype=x.dtype
    )
    x.copy_(dequant)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Walsh-Hadamard transform along last dim with 1/sqrt(N) scaling.

    Reference: inference/model.py:rotate_activation, which delegates to the
    `fast_hadamard_transform` package. We provide a pure-torch fallback since
    that package fails to build on AMD ROCm.

    Iterative radix-2 butterfly (FFT-style): O(N log N) ops, log2(N) passes.
    For each pass `h` = 1, 2, 4, ..., N/2: pair (x[k+j], x[k+j+h]) becomes
    (a+b, a-b). After all passes, multiply by 1/sqrt(N) for normalization.

    Args:
        x: tensor whose last dim is a power of 2 (typically 128 or 512)
    Returns:
        Hadamard-transformed tensor, same shape and dtype as x
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"last dim {n} must be a power of 2"

    orig_dtype = x.dtype
    *prefix, _ = x.shape
    flat = x.reshape(-1, n).float().contiguous()

    h = 1
    while h < n:
        # Group consecutive 2h-element segments; pair element j with element j+h.
        view = flat.view(-1, n // (2 * h), 2, h)
        a = view[..., 0, :]
        b = view[..., 1, :]
        flat = torch.stack([a + b, a - b], dim=-2).reshape(-1, n)
        h *= 2

    flat = flat * (n**-0.5)
    return flat.reshape(*prefix, n).to(orig_dtype)


# ---------------------------------------------------------------------------
# Self-test (run as `python -m atom.model_ops.quant_v4`)
# ---------------------------------------------------------------------------


def _selftest():
    torch.manual_seed(0)

    # ---- FP8 round-trip: error bounded by ~1/448 per block ----
    x = torch.randn(2, 16, 256, dtype=torch.bfloat16) * 3.0
    x_orig = x.clone()
    act_quant_inplace(x, block_size=128, scale_fmt=None)
    rel_err = (
        ((x.float() - x_orig.float()).abs() / x_orig.float().abs().clamp(min=1e-3))
        .mean()
        .item()
    )
    # FP8 e4m3 has ~3-bit mantissa => ~6% per-element ULP, ~2-3% mean rel error.
    assert rel_err < 0.05, f"FP8 round-trip relative error too large: {rel_err}"
    print(f"[act_quant_inplace fp32-scale]  OK  mean_rel_err={rel_err:.2e}")

    x = x_orig.clone()
    act_quant_inplace(x, block_size=128, scale_fmt="ue8m0")
    rel_err_ue = (
        ((x.float() - x_orig.float()).abs() / x_orig.float().abs().clamp(min=1e-3))
        .mean()
        .item()
    )
    assert (
        rel_err_ue < 0.05
    ), f"FP8 ue8m0 round-trip relative error too large: {rel_err_ue}"
    print(f"[act_quant_inplace ue8m0-scale] OK  mean_rel_err={rel_err_ue:.2e}")

    # ---- FP4 quantize+pack round-trip validation ----
    gpu_available = os.environ.get("ATOM_FP4_TORCH_FALLBACK", "0") != "1" and torch.cuda.is_available()
    route = "triton" if gpu_available else "torch"

    x = torch.randn(2, 16, 64, dtype=torch.bfloat16) * 2.0
    if gpu_available:
        x = x.cuda()
    x_orig = x.clone()

    # Quantize + dequantize round-trip
    fp4_act_quant_inplace(x, block_size=32)
    rel_err = (
        ((x.float() - x_orig.float()).abs() / x_orig.float().abs().clamp(min=1e-3))
        .mean()
        .item()
    )
    assert rel_err < 0.30, f"FP4 round-trip relative error too large: {rel_err}"

    # Check all values land on valid FP4 grid
    blocks = x.reshape(2, 16, 2, 32).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2**-126)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    normalized = (blocks / scale).abs()
    valid_grid = _FP4_MAGNITUDES.to(normalized.device)
    on_grid = (
        (normalized.unsqueeze(-1) - valid_grid).abs().min(dim=-1).values.max().item()
    )
    assert on_grid < 1e-4, f"FP4 values off grid by {on_grid}"
    print(
        f"[fp4_act_quant_inplace] {route}  OK  "
        f"mean_rel_err={rel_err:.2e}  off_grid={on_grid:.2e}"
    )

    # ---- FP4 quantize+pack: verify packed outputs ----
    if gpu_available:
        print("\n--- FP4 quantize+pack self-tests ---")

    test_configs = [
        ((2, 16, 64), 32, "small-2D"),
        ((4, 32, 128), 32, "medium-2D"),
        ((1, 128, 256), 32, "dsv4-indexer-kv"),
        ((1, 32), 32, "single-block"),
    ]

    for shape, block_size, label in test_configs:
        x_in = torch.randn(*shape, dtype=torch.bfloat16) * 2.0
        *prefix, n = shape

        if gpu_available:
            x_in = x_in.cuda()
            packed_t, scale_t = torch.empty(*prefix, n // 2, dtype=torch.uint8, device="cuda"), \
                               torch.empty(*prefix, n // block_size, dtype=torch.float8_e8m0fnu, device="cuda")
            _fp4_act_quant_triton(x_in, packed_t, scale_t, block_size)
        else:
            packed_t, scale_t = torch.empty(*prefix, n // 2, dtype=torch.uint8), \
                               torch.empty(*prefix, n // block_size, dtype=torch.float8_e8m0fnu)
            _fp4_act_quant_torch(x_in, packed_t, scale_t, block_size)

        # Verify packed output can be dequantized correctly
        packed_long = packed_t.int().reshape(*prefix, n // block_size, block_size // 2)
        scale_f = scale_t.float().reshape(*prefix, n // block_size)
        dequant = dequant_fp4_e2m1(packed_long, scale_f, fp4_block_size=block_size)
        blocks_v = dequant.float().reshape(*prefix, n // block_size, block_size)
        amax_v = blocks_v.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2**-126)
        scale_v = torch.pow(2.0, torch.ceil(torch.log2(amax_v / 6.0)))
        norm_v = (blocks_v / scale_v).abs()
        off_grid = (
            (norm_v.unsqueeze(-1) - _FP4_MAGNITUDES.to(norm_v.device)).abs()
            .min(dim=-1).values.max().item()
        )
        status = "OK" if off_grid < 1e-4 else "FAIL"
        print(f"  [{label}] {status}  off_grid={off_grid:.2e}")
        assert off_grid < 1e-4, f"FP4 packed values off grid for {label}: {off_grid}"

    # Edge cases
    x_zero = torch.zeros(4, 64, dtype=torch.bfloat16)
    fp4_act_quant_inplace(x_zero, block_size=32)
    assert (x_zero == 0).all(), "All-zero round-trip should stay all-zero"
    print("[fp4_act_quant_inplace] all-zero  OK")

    x_big = torch.full((1, 32), 100.0, dtype=torch.bfloat16)
    fp4_act_quant_inplace(x_big, block_size=32)
    expected = 96.0
    actual = x_big.float().item()
    assert abs(actual - expected) < 0.5, f"Extreme value: expected {expected}, got {actual}"
    print(f"[fp4_act_quant_inplace] extreme-value  OK  {expected} ~ {actual:.1f}")

    # ---- Hadamard transform: orthogonality H @ H^T = I ----
    n = 128
    eye = torch.eye(n, dtype=torch.float32)
    h = rotate_activation(eye)
    # H @ H^T should be identity for an orthogonal transform
    hht = h @ h.T
    err = (hht - torch.eye(n)).abs().max().item()
    assert err < 1e-5, f"Hadamard not orthogonal: max abs err = {err}"
    print(f"[rotate_activation orthogonality] OK  max_abs_err={err:.2e}")

    # Hadamard inverse: applying twice = identity (Hadamard is involutive after normalization)
    x = torch.randn(2, 4, 64, dtype=torch.float32)
    twice = rotate_activation(rotate_activation(x))
    err = (twice - x).abs().max().item()
    assert err < 1e-5, f"Hadamard not involutive: max abs err = {err}"
    print(f"[rotate_activation involution]   OK  max_abs_err={err:.2e}")

    print("ALL OK")


if __name__ == "__main__":
    _selftest()
