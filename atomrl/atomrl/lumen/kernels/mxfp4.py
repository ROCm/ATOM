# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen-owned MXFP4 Triton kernels.

Contains only kernels that AITER does not provide:
  - ASM-accelerated MXFP4 quantization (RTN + SR) via gfx950 VOP3 instructions
  - Software fallback MXFP4 quantization (RTN + SR)
  - Packed FP4 transpose
  - Blockwise Hadamard transform (RHT)

For kernels that AITER provides (RTN quant via dynamic_mxfp4_quant, GEMM,
dequant), see ``lumen/ops/quantize/ops.py`` which wraps AITER directly.

gfx950 ISA instructions used:
  - ``v_cvt_scalef32_pk_fp4_f32``    : 2×FP32 → packed FP4 byte (RTN)
  - ``v_cvt_scalef32_sr_pk_fp4_f32`` : 2×FP32 → packed FP4 byte (SR)
  - ``v_cvt_scalef32_pk_fp4_bf16``   : 2×BF16 → packed FP4 byte (RTN)
  - ``v_cvt_scalef32_sr_pk_fp4_bf16``: 2×BF16 → packed FP4 byte (SR)

ASM SR is unbiased without pre-scaling — no correction factors needed.

Fused Hadamard + Quant kernel:
  - ``_fused_hadamard_quant_mxfp4_kernel``: BF16 → Hadamard rotate (in-register) → FP4 quantize → write
    Eliminates one global memory roundtrip vs separate hadamard + quant kernels.
"""

import math
import os

import triton
import triton.language as tl

FP4_E2M1_MAX = 6.0
_E2M1_EMAX = 2  # largest normal biased exponent for E2M1

# Philox rounds behind stochastic rounding. Triton defaults to 10; Philox4x32-7
# already passes BigCrush, and rounding noise asks less of a generator than a
# simulation does. The rounds are a third of the kernel's instructions and it is
# VALU-bound, so dropping three takes 8-12% off the dual-layout quantizer.
#
# Read from the environment because it must be a compile-time constant: Triton
# rejects a global that changed after tracing, so a sweep cannot patch it in
# place. Lowering it is a numerics change.
SR_PHILOX_ROUNDS_DEFAULT = 7
SR_PHILOX_ROUNDS = int(
    os.environ.get("LUMEN_SR_PHILOX_ROUNDS", SR_PHILOX_ROUNDS_DEFAULT)
)
SR_PHILOX_ROUNDS_C = tl.constexpr(SR_PHILOX_ROUNDS)

# Not a Megatron argument, so it is absent from the argument dump every run logs;
# an overridden run would otherwise be indistinguishable from a default one.
if SR_PHILOX_ROUNDS != SR_PHILOX_ROUNDS_DEFAULT and os.environ.get("RANK", "0") == "0":
    print(
        f"[lumen] LUMEN_SR_PHILOX_ROUNDS={SR_PHILOX_ROUNDS} "
        f"(default {SR_PHILOX_ROUNDS_DEFAULT}): stochastic-rounding dither uses "
        f"fewer Philox rounds; this changes gradient numerics.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Scale calculation (shared by RTN and SR paths)
# ---------------------------------------------------------------------------


@triton.jit
def _calculate_fp4_scales(
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
):
    """Compute E8M0 block scales for MXFP4 quantization.

    Follows the same pattern as AITER's ``_calculate_scales`` for MXFP8 but
    with ``target_max_pow2 = 2`` (FP4 E2M1 emax).
    """
    E8M0_EXPONENT_BIAS: tl.constexpr = 127

    tl.static_assert(BLOCK_N % QUANT_BLOCK_SIZE == 0)
    if IS_2D_BLOCK:
        tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)

    if x.type.element_ty == tl.float32:
        hp_int_dtype = tl.int32
        hp_mbits: tl.constexpr = 23
        hp_ebits: tl.constexpr = 8
        hp_exp_bias: tl.constexpr = 127
    else:
        hp_int_dtype = tl.int16
        hp_mbits: tl.constexpr = 7
        hp_ebits: tl.constexpr = 8
        hp_exp_bias: tl.constexpr = 127

    sbits: tl.constexpr = 1
    # FP4 E2M1: 1 mantissa bit, max exponent pow2 = 2
    mbits: tl.constexpr = 1
    target_max_pow2: tl.constexpr = 2

    NEW_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    if IS_2D_BLOCK:
        NEW_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        x_r = x.reshape(NEW_BLOCK_M, QUANT_BLOCK_SIZE, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        x_r = tl.permute(x_r, (0, 2, 1, 3))
        max_abs = tl.max(tl.abs(x_r), axis=-1)
        max_abs = tl.max(max_abs, axis=-1)
    else:
        x_r = x.reshape(BLOCK_M, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x_r), axis=-1)
    max_abs = max_abs.to(x.type.element_ty)

    # Round-even on the scale (adaptive rounding from pytorch/ao)
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + sbits)) - 1) << hp_mbits
    max_abs = (max_abs + val_to_add) & mask

    # NOTE: the masked exponent field is unsigned; cast to signed int32 before
    # subtracting the biases, otherwise ``extracted_pow2 - target_max_pow2`` can
    # underflow to a huge positive value and clamp to the max E8M0 scale (2**128).
    extracted_pow2 = ((max_abs >> hp_mbits) & 0b11111111).to(tl.int32) - hp_exp_bias
    scale_e8m0_unbiased = extracted_pow2 - target_max_pow2

    scale_e8m0_unbiased = tl.minimum(
        tl.maximum(scale_e8m0_unbiased, -1 * E8M0_EXPONENT_BIAS), E8M0_EXPONENT_BIAS + 1
    )
    scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS

    return scale_e8m0_biased.to(tl.uint8)


# ---------------------------------------------------------------------------
# FP4 packing: ASM (gfx950) + software fallback
# ---------------------------------------------------------------------------


@triton.jit
def _generate_randval(m: tl.constexpr, n: tl.constexpr, philox_seed, philox_offset):
    """An (m, n) tile of random 32-bit words for stochastic rounding.

    Philox emits four words per round whether or not the caller reads them, so
    taking one word per element pays four times the rounds it needs. Spreading
    each round's four words across four adjacent columns covers the same tile in
    a quarter of the rounds, which took SR from 44% of kernel time to 12% on the
    shape it was tuned on (0.146 ms -> 0.059 ms against 0.057 ms RTN).

    That 12% is not what production pays. At the tiles and shapes a Qwen3-8B step
    actually issues, SR is 22-26% of the dual-layout quantizer on every gradient
    shape (report §5.19, ``bench_dual_layout_tiles.py --axis features``), and is
    the largest single feature in it. ``SR_PHILOX_ROUNDS`` is the knob left: it is
    already 7 against Philox's standard 10, and SR dither has no cryptographic
    requirement, so it can go lower -- but that is a numerics change and belongs
    in the precision harness, not here.
    """
    # A launch-level offset alone repeats the same Philox counters in every
    # program. Give each tile a disjoint counter range so reduction axes see
    # independent rounding errors instead of one repeated dither pattern.
    tile_id = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    if n % 4 == 0:
        QN: tl.constexpr = n // 4
        ms = tl.arange(0, m)
        ns = tl.arange(0, QN)
        tile_offset = philox_offset + tile_id * m * QN
        rng_offsets = tile_offset + ms[:, None] * QN + ns[None, :]
        r0, r1, r2, r3 = tl.randint4x(philox_seed, rng_offsets, SR_PHILOX_ROUNDS_C)
        return tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(m, n)
    # Narrow tiles can't be quartered; fall back to one round per element.
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    tile_offset = philox_offset + tile_id * m * n
    rng_offsets = tile_offset + ms[:, None] * n + ns[None, :]
    r1, _, _, _ = tl.randint4x(philox_seed, rng_offsets, SR_PHILOX_ROUNDS_C)
    return r1


@triton.jit
def _pack_fp4(
    x,
    scales,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_SR: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
):
    """Pack BF16/FP32 values into MXFP4 (2 FP4 nibbles per byte).

    ASM path uses gfx950 ``v_cvt_scalef32_[sr_]pk_fp4_f32`` instructions.
    These instructions take the E8M0 scale as a uint8 and produce a packed
    byte with bits[3:0]=fp4(src_lo) and bits[7:4]=fp4(src_hi).

    For SR, the hardware uses the seed from Src1 for the first FP4 lane,
    then internally derives a new random value via v_prng_b32 for the second
    lane. No software pre-scaling is needed — the result is unbiased.
    """
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    # Split into pairs for the pk (pack-2) instruction
    x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))

    # E8M0 scale → FP32 power-of-2 for the conversion instruction
    scales = tl.where(scales < 1, 1, scales)
    scales_fp32 = (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    F32_MIN_NORMAL: tl.constexpr = 2**-126
    min_frag = F32_MIN_NORMAL
    scales_fp32 = tl.where(scales_fp32 < min_frag, min_frag, scales_fp32)

    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        scales_fp32 = (
            scales_fp32.expand_dims(axis=(1, 3))
            .broadcast_to(
                SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE
            )
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )
    else:
        scales_fp32 = (
            scales_fp32.expand_dims(axis=2)
            .broadcast_to(BLOCK_M, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE)
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )

    if USE_SR:
        if USE_ASM:
            # ASM pk instruction processes 2 values per call; one seed per pair
            randval0 = _generate_randval(
                BLOCK_M, HALF_BLOCK_N, philox_seed, philox_offset
            )
        else:
            # Software path applies per-element noise
            randval0 = _generate_randval(BLOCK_M, BLOCK_N, philox_seed, philox_offset)
    else:
        randval0 = 0

    if USE_ASM:
        if x0.type.element_ty == tl.float32:
            if not USE_SR:
                # RTN: v_cvt_scalef32_pk_fp4_f32  dst, src0(f32_lo), src1(f32_hi), src2(scale)
                y = tl.inline_asm_elementwise(
                    asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];",
                    constraints="=&v,v,v,v",
                    args=[x0, x1, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
            else:
                # SR: v_cvt_scalef32_sr_pk_fp4_f32  dst, src0(u64=hi|lo), src1(seed), src2(scale)
                x_packed = (x1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | x0.to(
                    tl.uint32, bitcast=True
                )
                y = tl.inline_asm_elementwise(
                    asm="v_cvt_scalef32_sr_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];",
                    constraints="=&v,v,v,v",
                    args=[x_packed, randval0, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
        else:
            # BF16 input
            if not USE_SR:
                x_packed_bf16 = (
                    x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
                ) | x0.to(tl.uint16, bitcast=True)
                y = tl.inline_asm_elementwise(
                    asm="v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2 op_sel:[0,0,0,0];",
                    constraints="=&v,v,v",
                    args=[x_packed_bf16, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
            else:
                x_packed_bf16 = (
                    x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
                ) | x0.to(tl.uint16, bitcast=True)
                y = tl.inline_asm_elementwise(
                    asm="v_cvt_scalef32_sr_pk_fp4_bf16 $0, $1, $2, $3 op_sel:[0,0,0,0];",
                    constraints="=&v,v,v,v",
                    args=[x_packed_bf16, randval0, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )

        # Output is already packed: bits[3:0]=fp4(x0), bits[7:4]=fp4(x1).
        # The pk instruction writes the packed byte into the low 8 bits of a
        # 32-bit VGPR (op_sel selects byte 0); a uint8 asm output constraint is
        # not allocatable to a `v` register, so we declare uint32 and mask.
        y = (y & 0xFF).to(tl.uint8)
        y = y.reshape(BLOCK_M, HALF_BLOCK_N)
    else:
        # Software fallback: manual FP4 E2M1 conversion.
        #
        # NOT a validated path, and not equivalent to the ASM one above:
        # ``quantize.assert_mxfp4_arch_supported`` refuses any architecture that
        # would land here, because its rounding is wrong in four ways that no
        # test covers (see that function for the list, and note the scale
        # clamped at the top of this kernel is not the byte the caller stores,
        # so a subnormal-amax block is off by 2x). Correcting them is a numerics
        # change that needs the precision harness behind it. Until then this
        # exists to be read and developed against, not trained on.
        x_scaled = x / scales_fp32.expand_dims(axis=2).broadcast_to(
            BLOCK_M, HALF_BLOCK_N, 2
        ).reshape(BLOCK_M, BLOCK_N)

        if USE_SR:
            noise = randval0.to(tl.float32) * (1.0 / 4294967296.0)
            x_scaled = x_scaled + (noise - 0.5) * 0.01

        # Clamp and round to nearest FP4 E2M1 representable value
        abs_val = tl.abs(x_scaled)
        sign = tl.where(x_scaled < 0.0, 1.0, 0.0)

        # FP4 E2M1 levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        # Boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
        code = tl.zeros_like(abs_val).to(tl.uint8)
        code = tl.where(abs_val >= 0.25, tl.full(code.shape, 1, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 0.75, tl.full(code.shape, 2, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 1.25, tl.full(code.shape, 3, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 1.75, tl.full(code.shape, 4, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 2.50, tl.full(code.shape, 5, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 3.50, tl.full(code.shape, 6, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 5.00, tl.full(code.shape, 7, dtype=tl.uint8), code)

        sign_u8 = sign.to(tl.uint8)
        fp4_code = (sign_u8 << 3) | code

        # Pack two FP4 codes into one uint8 (AITER convention: even=low nibble)
        codes_reshaped = fp4_code.reshape(BLOCK_M, HALF_BLOCK_N, 2)
        even, odd = tl.split(codes_reshaped)
        y = even | (odd << 4)

    return y


# ---------------------------------------------------------------------------
# Main conversion kernel: BF16/FP32 → packed MXFP4 + E8M0 scales
# ---------------------------------------------------------------------------


@triton.jit
def _convert_to_mxfp4_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_SR: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """BF16/FP32 → packed MXFP4 with E8M0 block scales.

    Computes E8M0 shared exponent scales, then quantizes each element to
    FP4 E2M1 using either ASM instructions (gfx950) or software fallback.
    Two FP4 values are packed per output byte.

    When ``USE_ASM=True`` and ``USE_SR=True``, the hardware SR is unbiased
    without any pre-scaling — no correction factors (4/3 or 16/9) are needed
    on the GEMM output.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    x = tl.load(x_ptr + offs_x)

    scales = _calculate_fp4_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
    )

    y = _pack_fp4(
        x,
        scales,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
    )

    # Store packed FP4 output (HALF_BLOCK_N bytes per row)
    offs_yn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    tl.store(y_ptr + offs_y, y)
    tl.store(s_ptr + offs_s, scales)


# ---------------------------------------------------------------------------
# Packed FP4 transpose kernel
# ---------------------------------------------------------------------------


@triton.jit
def _transpose_packed_fp4_kernel(
    in_ptr,
    out_ptr,
    M,
    N_packed: tl.constexpr,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N_PACKED: tl.constexpr,
):
    """Transpose packed FP4: (M, N//2) -> (N, M//2).

    Unpacks nibbles, transposes, repacks.

    Input and output stay row-major; AITER owns GEMM layout conversion.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    N = N_packed * 2
    BLOCK_N: tl.constexpr = BLOCK_N_PACKED * 2
    BLOCK_M_HALF: tl.constexpr = BLOCK_M // 2

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_packed = pid_n * BLOCK_N_PACKED + tl.arange(0, BLOCK_N_PACKED)

    mask = (rm[:, None] < M) & (rn_packed[None, :] < N_packed)
    in_offs = rm[:, None] * stride_im + rn_packed[None, :] * stride_in
    packed = tl.load(in_ptr + in_offs, mask=mask, other=0).to(tl.uint8)

    # AITER convention: even in low nibble, odd in high nibble
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    unpacked = tl.reshape(tl.join(even, odd), (BLOCK_M, BLOCK_N))

    transposed = tl.trans(unpacked)

    reshaped = tl.reshape(transposed, (BLOCK_N, BLOCK_M_HALF, 2))
    t_even, t_odd = tl.split(reshaped)
    repacked = t_even | (t_odd << 4)

    rn_full = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_packed = pid_m * BLOCK_M_HALF + tl.arange(0, BLOCK_M_HALF)
    out_mask = (rn_full[:, None] < N) & (rm_packed[None, :] < (M // 2))
    tl.store(
        out_ptr + rn_full[:, None] * stride_om + rm_packed[None, :] * stride_on,
        repacked,
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Blockwise Hadamard Transform kernel
# ---------------------------------------------------------------------------


@triton.jit
def _hadamard_transform_kernel(
    x_ptr,
    out_ptr,
    sign_ptr,
    M,
    N: tl.constexpr,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    G: tl.constexpr,
):
    """Blockwise Random Hadamard Transform via butterfly algorithm O(G log G).

    Applies H_g @ diag(S) to blocks of G elements along the last dimension.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m + tl.arange(0, 1)
    rn = pid_n * G + tl.arange(0, G)

    mask = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(
        x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn, mask=mask, other=0.0
    ).to(tl.float32)

    signs = tl.load(sign_ptr + tl.arange(0, G)).to(tl.float32)
    x = x * signs[None, :]

    # Unroll at compile time: derive h from the static induction index and keep
    # h/groups as constexpr ints (a loop-carried h becomes a traced tensor and
    # breaks the reshape shapes).
    log2_g: tl.constexpr = int(math.log2(G))
    for i in tl.static_range(log2_g):
        h = 1 << i
        groups = G // (2 * h)
        # (1, groups, 2, h): split each 2h-group into two h-blocks (top/bot).
        # Triton 3.7 has no slice indexing, so permute the pair axis last and
        # use tl.split.
        x_reshaped = tl.reshape(x, (1, groups, 2, h))
        x_perm = tl.permute(x_reshaped, (0, 1, 3, 2))  # (1, groups, h, 2)
        top, bot = tl.split(x_perm)  # each (1, groups, h)
        new_top = top + bot
        new_bot = top - bot
        # Reassemble as (1, groups, 2, h) -> (1, G).
        stacked = tl.join(new_top, new_bot)  # (1, groups, h, 2)
        stacked = tl.permute(stacked, (0, 1, 3, 2))  # (1, groups, 2, h)
        x = tl.reshape(stacked, (1, G))

    x = x * (1.0 / tl.sqrt(float(G)))

    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on, x, mask=mask)


# ---------------------------------------------------------------------------
# Hadamard butterfly subroutine (in-register, no memory traffic)
# ---------------------------------------------------------------------------


@triton.jit
def _hadamard16_butterfly(x, ROWS: tl.constexpr):
    """In-register Hadamard-16 butterfly for x of shape (ROWS, 16).

    Hardcoded for G=16 (log2(16)=4 stages). Avoids the constexpr reshape
    issues of a generic loop by unrolling all 4 stages explicitly.
    Returns (ROWS, 16) normalized.
    """
    # Stage 0: h=1, groups=8 -> (ROWS, 8, 2, 1)
    x_r = tl.reshape(x, (ROWS, 8, 2, 1))
    x_p = tl.permute(x_r, (0, 1, 3, 2))
    top, bot = tl.split(x_p)
    x = tl.reshape(tl.permute(tl.join(top + bot, top - bot), (0, 1, 3, 2)), (ROWS, 16))

    # Stage 1: h=2, groups=4 -> (ROWS, 4, 2, 2)
    x_r = tl.reshape(x, (ROWS, 4, 2, 2))
    x_p = tl.permute(x_r, (0, 1, 3, 2))
    top, bot = tl.split(x_p)
    x = tl.reshape(tl.permute(tl.join(top + bot, top - bot), (0, 1, 3, 2)), (ROWS, 16))

    # Stage 2: h=4, groups=2 -> (ROWS, 2, 2, 4)
    x_r = tl.reshape(x, (ROWS, 2, 2, 4))
    x_p = tl.permute(x_r, (0, 1, 3, 2))
    top, bot = tl.split(x_p)
    x = tl.reshape(tl.permute(tl.join(top + bot, top - bot), (0, 1, 3, 2)), (ROWS, 16))

    # Stage 3: h=8, groups=1 -> (ROWS, 1, 2, 8)
    x_r = tl.reshape(x, (ROWS, 1, 2, 8))
    x_p = tl.permute(x_r, (0, 1, 3, 2))
    top, bot = tl.split(x_p)
    x = tl.reshape(tl.permute(tl.join(top + bot, top - bot), (0, 1, 3, 2)), (ROWS, 16))

    return x * 0.25  # 1/sqrt(16) = 0.25


@triton.jit
def _hadamard16_mfma(x, hmat_ptr, ROWS: tl.constexpr):
    """Hadamard-16 of ``x`` (any shape whose last axis groups into 16) via MFMA.

    ``hmat_ptr`` holds diag(sign) @ H16 / sqrt(16), which is the same linear map
    the butterfly applies. Its entries are ±1/4 and the caller's values are
    BF16-exact, so routing it through the matrix unit costs nothing in accuracy
    and replaces four stages of cross-lane reshuffling with one instruction —
    measured 33% off the dual-layout quantizer, 1.49 → 2.21 TB/s.

    Caller must pass BF16 ``x``; an FP32 operand would be truncated here.
    """
    G: tl.constexpr = 16
    hmat = tl.load(hmat_ptr + tl.arange(0, G)[:, None] * G + tl.arange(0, G)[None, :])
    return tl.dot(x.reshape(ROWS, G), hmat, out_dtype=tl.float32)


# ---------------------------------------------------------------------------
# Fused Hadamard + MXFP4 Quantization kernel
# ---------------------------------------------------------------------------


@triton.jit
def _fused_hadamard_quant_mxfp4_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    sign_ptr,
    hmat_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_SR: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """BF16 → Hadamard-16 rotate (in-register) → packed MXFP4 + E8M0 scales.

    Fuses hadamard_transform + convert_to_mxfp4 into a single kernel,
    eliminating one global memory roundtrip. Hardcoded for G=16.

    A BF16 input takes the matrix-unit rotation, matching what the dual-layout
    and WGrad-activation quantizers do so the three agree bit for bit; an FP32
    input keeps the butterfly, which does not have to narrow the operand.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    G: tl.constexpr = 16
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    NUM_GROUPS: tl.constexpr = BLOCK_N // G
    ROWS: tl.constexpr = BLOCK_M * NUM_GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    x_in = tl.load(x_ptr + offs_x)

    # --- Hadamard-16 in registers (zero memory traffic) ---
    if x_in.type.element_ty == tl.bfloat16:
        x = _hadamard16_mfma(x_in, hmat_ptr, ROWS=ROWS).reshape(BLOCK_M, BLOCK_N)
    else:
        sign = tl.load(sign_ptr + tl.arange(0, G)).to(tl.float32)
        x = x_in.to(tl.float32).reshape(ROWS, G) * sign[None, :]
        x = _hadamard16_butterfly(x, ROWS=ROWS).reshape(BLOCK_M, BLOCK_N)

    # --- FP4 quantization in registers ---
    scales = _calculate_fp4_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
    )

    y = _pack_fp4(
        x,
        scales,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
    )

    # --- Write packed FP4 + scales (single write, no intermediate BF16) ---
    offs_yn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    tl.store(y_ptr + offs_y, y)

    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    offs_s = offs_m[:, None] * stride_sm + offs_sn[None, :] * stride_sn
    tl.store(s_ptr + offs_s, scales)


# ---------------------------------------------------------------------------
# Dual-layout quantization: one read of x, both MXFP4 layouts backward needs
# ---------------------------------------------------------------------------


@triton.jit
def _dual_layout_quant_mxfp4_kernel(
    x_ptr,
    a_ptr,
    as_ptr,
    b_ptr,
    bs_ptr,
    sign_ptr,
    hmat_ptr,
    stride_xm,
    stride_xn,
    stride_am,
    stride_an,
    stride_asm,
    stride_asn,
    stride_bm,
    stride_bn,
    stride_bsm,
    stride_bsn,
    philox_seed,
    philox_offset_a,
    philox_offset_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_SR_A: tl.constexpr,
    USE_SR_B: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """One tile read of x → MXFP4 along n *and* Hadamard-rotated MXFP4 along m.

    Backward quantizes the gradient twice: row-major for DGrad, then rotated and
    transposed for WGrad. Run separately the second pass has to read x^T as a
    view, which measures 1.70x slower than the same shape dense because of the
    register layout the strided load forces (report §5.10). Here the tile is read
    once, densely, and transposed in registers.

    Both outputs' scale blocks live entirely inside the tile, so BLOCK_M and
    BLOCK_N each have to be a whole number of quant blocks and BLOCK_M a whole
    number of Hadamard groups. No cross-tile reduction is needed.
    """
    G: tl.constexpr = 16
    tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_N % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_M % G == 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_in = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    x = x_in.to(tl.float32)

    # --- A: row-major, quant blocks along n ---
    a_scales = _calculate_fp4_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
    )
    a = _pack_fp4(
        x,
        a_scales,
        philox_seed,
        philox_offset_a,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_SR=USE_SR_A,
        USE_ASM=USE_ASM,
    )

    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_an = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    tl.store(a_ptr + offs_m[:, None] * stride_am + offs_an[None, :] * stride_an, a)
    offs_asn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    tl.store(
        as_ptr + offs_m[:, None] * stride_asm + offs_asn[None, :] * stride_asn,
        a_scales,
    )

    # --- B: transposed, Hadamard-16 along m, quant blocks along m ---
    ROWS_B: tl.constexpr = BLOCK_N * (BLOCK_M // G)
    if x_in.type.element_ty == tl.bfloat16:
        xt = _hadamard16_mfma(tl.trans(x_in), hmat_ptr, ROWS=ROWS_B).reshape(
            BLOCK_N, BLOCK_M
        )
    else:
        sign = tl.load(sign_ptr + tl.arange(0, G)).to(tl.float32)
        xt = tl.trans(x).reshape(ROWS_B, G) * sign[None, :]
        xt = _hadamard16_butterfly(xt, ROWS=ROWS_B).reshape(BLOCK_N, BLOCK_M)

    b_scales = _calculate_fp4_scales(
        xt,
        BLOCK_M=BLOCK_N,
        BLOCK_N=BLOCK_M,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
    )
    b = _pack_fp4(
        xt,
        b_scales,
        philox_seed,
        philox_offset_b,
        BLOCK_M=BLOCK_N,
        BLOCK_N=BLOCK_M,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_SR=USE_SR_B,
        USE_ASM=USE_ASM,
    )

    HALF_BLOCK_M: tl.constexpr = BLOCK_M // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    offs_bn = pid_m * HALF_BLOCK_M + tl.arange(0, HALF_BLOCK_M)
    tl.store(b_ptr + offs_n[:, None] * stride_bm + offs_bn[None, :] * stride_bn, b)
    offs_bsn = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    tl.store(
        bs_ptr + offs_n[:, None] * stride_bsm + offs_bsn[None, :] * stride_bsn,
        b_scales,
    )


# ---------------------------------------------------------------------------
# Fused dequant + transpose: packed FP4 (M, K/2) → BF16 (K, M)
# ---------------------------------------------------------------------------


@triton.jit
def _fp4_e2m1_decode(code):
    """Decode a 4-bit FP4 E2M1 code to float32. bits[3]=sign, bits[2:0]=magnitude."""
    magnitude = code & 0x07
    sign = (code >> 3).to(tl.float32)
    val = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(
                            magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where(sign > 0.5, -val, val)


@triton.jit
def _dequant_hadamard_quant_mxfp4_kernel(
    fp4_ptr,
    in_scale_ptr,
    out_ptr,
    out_scale_ptr,
    hmat_ptr,
    stride_fm,
    stride_fk,
    stride_ism,
    stride_isk,
    stride_om,
    stride_on,
    stride_osm,
    stride_osn,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_SR: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """Packed FP4 (M, K/2) → Hadamard-rotated, transposed packed FP4 (K, M/2).

    WGrad needs the activation rotated and transposed, but forward only stored it
    row-major in FP4. Done in two passes that costs a full BF16 (K, M) buffer
    written and read back — 4x the bytes of either FP4 end. Here the tile is
    decoded, transposed and rotated in registers, so the BF16 form never reaches
    memory and the kernel moves half a byte per element on each side.

    The output's quant blocks run along M, so BLOCK_M has to be a whole number of
    both quant blocks and Hadamard groups; BLOCK_K likewise for the input scales.
    """
    G: tl.constexpr = 16
    tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_M % G == 0)
    tl.static_assert(BLOCK_K % QUANT_BLOCK_SIZE == 0)

    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    HALF_BLOCK_K: tl.constexpr = BLOCK_K // 2
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // QUANT_BLOCK_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_fk = pid_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)

    packed = tl.load(
        fp4_ptr + offs_m[:, None] * stride_fm + offs_fk[None, :] * stride_fk
    ).to(tl.uint8)
    vals = tl.reshape(
        tl.join(
            _fp4_e2m1_decode(packed & 0x0F), _fp4_e2m1_decode((packed >> 4) & 0x0F)
        ),
        (BLOCK_M, BLOCK_K),
    )

    offs_sk = pid_k * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)
    in_scale_offs = offs_m[:, None] * stride_ism + offs_sk[None, :] * stride_isk
    scale_raw = tl.load(in_scale_ptr + in_scale_offs).to(tl.int32)
    # E8M0 stores the exponent biased by 127, i.e. the float is 2^(raw-127),
    # which is the raw byte shifted into an fp32's exponent field.
    scale_f32 = (scale_raw.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    x = vals * (
        scale_f32.reshape(BLOCK_M, SCALE_BLOCK_K, 1)
        .broadcast_to(BLOCK_M, SCALE_BLOCK_K, QUANT_BLOCK_SIZE)
        .reshape(BLOCK_M, BLOCK_K)
    )

    # An E2M1 magnitude needs two mantissa bits and the E8M0 scale is a power of
    # two, so every decoded value is exact in BF16 and the matrix-unit rotation
    # below is lossless.
    ROWS: tl.constexpr = BLOCK_K * (BLOCK_M // G)
    xt = _hadamard16_mfma(
        tl.trans(x.to(tl.bfloat16)),
        hmat_ptr,
        ROWS=ROWS,
    ).reshape(BLOCK_K, BLOCK_M)

    out_scales = _calculate_fp4_scales(
        xt,
        BLOCK_M=BLOCK_K,
        BLOCK_N=BLOCK_M,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
    )
    y = _pack_fp4(
        xt,
        out_scales,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_K,
        BLOCK_N=BLOCK_M,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
    )

    HALF_BLOCK_M: tl.constexpr = BLOCK_M // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_on = pid_m * HALF_BLOCK_M + tl.arange(0, HALF_BLOCK_M)
    tl.store(out_ptr + offs_k[:, None] * stride_om + offs_on[None, :] * stride_on, y)
    offs_osn = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    tl.store(
        out_scale_ptr + offs_k[:, None] * stride_osm + offs_osn[None, :] * stride_osn,
        out_scales,
    )


@triton.jit
def _dequant_transpose_mxfp4_kernel(
    fp4_ptr,
    scale_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    stride_fm,
    stride_fk,
    stride_sm,
    stride_sk,
    stride_ok,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Fused dequant + transpose: read packed FP4 (M, K/2) + 1D scales → write BF16 (K, M).

    Combines convert_from_mxfp4 and .t().contiguous() into a single kernel,
    eliminating one full BF16 (M, K) intermediate write.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    HALF_BLOCK_K: tl.constexpr = BLOCK_K // 2
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // QUANT_BLOCK_SIZE

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_packed = pid_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)

    mask_fp4 = (rm[:, None] < M) & (rk_packed[None, :] < (K // 2))
    packed = tl.load(
        fp4_ptr + rm[:, None] * stride_fm + rk_packed[None, :] * stride_fk,
        mask=mask_fp4,
        other=0,
    ).to(tl.uint8)

    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F

    vals_even = _fp4_e2m1_decode(even)
    vals_odd = _fp4_e2m1_decode(odd)

    # Interleave: (BLOCK_M, BLOCK_K)
    vals = tl.reshape(tl.join(vals_even, vals_odd), (BLOCK_M, BLOCK_K))

    # Load and expand 1D scales: (M, K/block_size)
    rk_scale = pid_k * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)
    mask_scale = (rm[:, None] < M) & (rk_scale[None, :] < (K // QUANT_BLOCK_SIZE))
    scale_raw = tl.load(
        scale_ptr + rm[:, None] * stride_sm + rk_scale[None, :] * stride_sk,
        mask=mask_scale,
        other=127,
    ).to(tl.int32)
    # E8M0 → float: 2^(stored_exp - 127)
    scale_f32 = ((scale_raw.to(tl.uint32)) << 23).to(tl.float32, bitcast=True)

    # Expand scales: (BLOCK_M, SCALE_BLOCK_K) → (BLOCK_M, BLOCK_K)
    scale_expanded = (
        scale_f32.reshape(BLOCK_M, SCALE_BLOCK_K, 1)
        .broadcast_to(BLOCK_M, SCALE_BLOCK_K, QUANT_BLOCK_SIZE)
        .reshape(BLOCK_M, BLOCK_K)
    )

    result = (vals * scale_expanded).to(tl.bfloat16)

    # Write in transposed layout: out[k, m] = result[m, k]
    rk_full = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_out = (rk_full[:, None] < K) & (rm[None, :] < M)
    result_t = tl.trans(result)  # (BLOCK_K, BLOCK_M)
    tl.store(
        out_ptr + rk_full[:, None] * stride_ok + rm[None, :] * stride_om,
        result_t,
        mask=mask_out,
    )
