# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Low-level quantization/dequantization ops wrapping Triton and C++ kernels.

These are pure stateless functions — no autograd, no scaling history.  For
autograd-aware quantized linear, see :mod:`~.linear`.  For the nn.Module
wrapper, see :mod:`atomrl.lumen.modules.quantize`.
"""

import functools
import logging
import random

import torch
import triton
from aiter.ops.quant import per_tensor_quant_triton
from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
    quant_fp8_blockwise_kernel,
    quant_fp8_blockwise_segment_m_kernel,
    requant_fp8_row_to_col_kernel,
)
from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
    _convert_from_mxfp8_kernel,
    _convert_to_mxfp8_kernel,
)
from torch.library import triton_op, wrap_triton

logger = logging.getLogger(__name__)
_HAVE_REQUANT_ROW_TO_COL = True


@functools.cache
def _triton_target(device: int):
    """Triton's target for one device, asked once.

    ``get_current_target()`` queries the HIP runtime for the device properties
    on every call. The quantize path asks per launch, and a training step makes
    hundreds of those, which is dead CPU time in front of kernels the GPU is
    already waiting for. A device's architecture cannot change under a live
    process, so the answer is cacheable; keying on the device keeps it right for
    a process that switches between unlike GPUs.
    """
    return triton.runtime.driver.active.get_current_target()


def is_cdna4():
    target = _triton_target(torch.cuda.current_device())
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


def triton_arch() -> str:
    """Architecture name of the current device, e.g. ``gfx950``.

    Same answer as AITER's ``get_arch()``, without its per-call device query.
    """
    target = _triton_target(torch.cuda.current_device())
    return "" if target is None else target.arch


# ---------------------------------------------------------------------------
# Tensorwise Quantization
# ---------------------------------------------------------------------------


def quant_fp8_tensorwise_impl(x, scale, dtype):
    # Triton static quant. The HIP ``static_per_tensor_quant`` kernel aborts
    # the process on this AITER build, so it is not on the call path.
    if dtype not in (torch.float8_e5m2, torch.float8_e5m2fnuz):
        try:
            y, _ = per_tensor_quant_triton(x, scale=scale, quant_dtype=dtype)
            return y.view(x.shape)
        except (ValueError, RuntimeError):
            pass
    qmax = torch.finfo(dtype).max
    return (x.float() / scale).clamp(-qmax, qmax).to(dtype)


def dequant_fp8_tensorwise_impl(x, scale_inv, dtype):
    return x.to(dtype) * scale_inv


# ---------------------------------------------------------------------------
# Blockwise Quantization
# ---------------------------------------------------------------------------


@triton_op("lumen::quant_fp8_blockwise_impl", mutates_args=())
def quant_fp8_blockwise_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor using blockwise FP8 scaling along *axis*.

    Returns ``(x_fp8, x_scales)`` where ``x_scales`` is per-block in float32.
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2

    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (
        (triton.cdiv(M, block_size), N)
        if axis == 0
        else (M, triton.cdiv(N, block_size))
    )
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    wrap_triton(quant_fp8_blockwise_kernel)[grid](
        x,
        x_fp8,
        x_scales,
        # The column outputs are only written when DUAL, which is a constexpr:
        # Triton drops those stores, so reusing the primary buffers is free.
        x_fp8,
        x_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
        axis,
        False,  # DUAL
    )
    return x_fp8, x_scales


@quant_fp8_blockwise_impl.register_fake
def quant_fp8_blockwise_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (
        (triton.cdiv(M, block_size), N)
        if axis == 0
        else (M, triton.cdiv(N, block_size))
    )
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    return x_fp8, x_scales


@triton_op("lumen::quant_fp8_blockwise_dual_axis_impl", mutates_args=())
def quant_fp8_blockwise_dual_axis_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dual-axis blockwise FP8 quant: emit row-wise (1×B) AND col-wise (B×1) in one pass.

    Loads each BLOCK×BLOCK tile of BF16 ``x`` once and writes both layouts —
    saves one BF16 read + amax pass vs calling ``quant_fp8_blockwise_impl``
    twice.  Used in the blockwise2d (Jet-RL §4.2) backward where the same
    grad tensor feeds both DGrad (1×128 along axis=1) and WGrad (128×1 along
    axis=0).

    Returns ``(x_fp8_row, x_scales_row, x_fp8_col, x_scales_col)`` with
    shapes ``(M, N)``, ``(M, N/B)``, ``(M, N)``, ``(M/B, N)``.  Scales are
    stored as dequant multipliers (``amax / FP8_MAX``), matching the
    convention of ``quant_fp8_blockwise_impl``.
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape

    x_fp8_row = torch.empty((M, N), dtype=dtype, device=x.device)
    x_fp8_col = torch.empty((M, N), dtype=dtype, device=x.device)
    x_scales_row = torch.empty(
        (M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device
    )
    x_scales_col = torch.empty(
        (triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device
    )

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    # One kernel serves both: AXIS=1 is the row-wise (1xB) copy, DUAL adds the
    # col-wise (Bx1) copy from the same loaded tile.
    wrap_triton(quant_fp8_blockwise_kernel)[grid](
        x,
        x_fp8_row,
        x_scales_row,
        x_fp8_col,
        x_scales_col,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
        1,  # AXIS
        True,  # DUAL
    )
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


@quant_fp8_blockwise_dual_axis_impl.register_fake
def quant_fp8_blockwise_dual_axis_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    M, N = x.shape
    return (
        torch.empty((M, N), dtype=dtype, device=x.device),
        torch.empty(
            (M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device
        ),
        torch.empty((M, N), dtype=dtype, device=x.device),
        torch.empty(
            (triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device
        ),
    )


if _HAVE_REQUANT_ROW_TO_COL:

    @triton_op("lumen::requant_fp8_row_to_col", mutates_args=())
    def requant_fp8_row_to_col(
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        dtype: torch.dtype,
        block_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-quantize FP8 (row-wise 1×block) → FP8 (col-wise block×1) without BF16 roundtrip.

        ``x_fp8`` is (M, K) FP8 with row-wise 1×block_size scales ``x_scales`` (M, K//block_size).
        Returns ``(y_fp8, y_scales)`` where ``y_fp8`` is (M, K) FP8 and ``y_scales`` is
        (M//block_size, K) float32 col-wise dequant scales.  Used in the blockwise/blockwise2d
        WGrad backward to avoid a BF16 intermediate activation copy.
        """
        assert (
            x_fp8.is_contiguous() and x_fp8.dim() == 2
        ), "x_fp8 must be 2D and contiguous"
        M, K = x_fp8.shape
        y_fp8 = torch.empty((M, K), dtype=dtype, device=x_fp8.device)
        y_scales = torch.empty(
            (triton.cdiv(M, block_size), K), dtype=torch.float32, device=x_fp8.device
        )
        grid = (triton.cdiv(M, block_size), triton.cdiv(K, block_size))
        wrap_triton(requant_fp8_row_to_col_kernel)[grid](
            x_fp8,
            x_scales,
            y_fp8,
            y_scales,
            M,
            K,
            block_size,
            torch.finfo(dtype).max,
        )
        return y_fp8, y_scales

else:

    def requant_fp8_row_to_col(x_fp8, x_scales, dtype, block_size=128):  # type: ignore[misc]
        raise RuntimeError(
            "requant_fp8_row_to_col_kernel not available in this aiter build; "
            "install the pinned official ROCm/aiter checkout from /opt/aiter"
        )


if _HAVE_REQUANT_ROW_TO_COL:

    @requant_fp8_row_to_col.register_fake
    def requant_fp8_row_to_col_meta(
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        dtype: torch.dtype,
        block_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x_fp8.dim() == 2, "x_fp8 must be 2D"
        M, K = x_fp8.shape
        return (
            torch.empty((M, K), dtype=dtype, device=x_fp8.device),
            torch.empty(
                (triton.cdiv(M, block_size), K),
                dtype=torch.float32,
                device=x_fp8.device,
            ),
        )


def quant_fp8_blockwise_segment_m_impl(
    x: torch.Tensor,
    batch_size: int,
    seg_lens: torch.Tensor,
    seg_indptr: torch.Tensor,
    scales_seg_indptr: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
):
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)

    scales_shape = (triton.cdiv(M, block_size) + batch_size, N)
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_size) + seg_lens.shape[0], triton.cdiv(N, block_size))
    quant_fp8_blockwise_segment_m_kernel[grid](
        x,
        x_fp8,
        x_scales,
        N,
        batch_size,
        seg_indptr,
        scales_seg_indptr,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8, x_scales


# ---------------------------------------------------------------------------
# MXFP8 Conversion
# ---------------------------------------------------------------------------


@triton_op("lumen::convert_to_mxfp8", mutates_args={})
def convert_to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: bool | None = None,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
    float8_dtype_pt: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(
        data_hp.shape[axis] % block_size == 0,
        f"tensor shape ({data_hp.shape}) at axis={axis} is not divisible by {block_size}",
    )
    assert not is_2d_block or data_hp.size(-2) % block_size == 0
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4() and float8_dtype_pt == torch.float8_e4m3fn
    elif use_asm and float8_dtype_pt == torch.float8_e5m2:
        use_asm = False
        logger.warning(
            f"ASM mode doesn't support {float8_dtype_pt}, falling back to non-ASM implementation"
        )

    data_hp = data_hp.transpose(axis, -1)
    data_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, data_shape[-1])
    data_lp = torch.empty(
        data_shape, dtype=float8_dtype_pt, device=data_hp.device
    ).reshape(-1, data_shape[-1])

    if is_2d_block:
        scales_shape = (
            *data_shape[:-2],
            data_shape[-2] // block_size,
            data_shape[-1] // block_size,
        )
    else:
        scales_shape = (*data_shape[:-1], data_shape[-1] // block_size)
    scales = torch.ones(scales_shape, dtype=torch.uint8, device=data_hp.device).reshape(
        -1, scales_shape[-1]
    )
    stride_xm, stride_xn = data_hp.stride()
    stride_ym, stride_yn = data_lp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    assert M % block_size == 0, "tensor M shape must align to block size"
    assert N % block_size == 0, "tensor N shape must align to block size"

    BLOCK_M = min(64, M)
    BLOCK_N = min(64, N)
    BLOCK_M = max(BLOCK_M, block_size)
    BLOCK_N = max(BLOCK_N, block_size)

    if philox_seed is None:
        philox_seed = random.randint(0, 2**31 - 2)
    if philox_offset is None:
        philox_offset = random.randint(0, 2**31 - 2)
    wrap_triton(_convert_to_mxfp8_kernel)[grid](
        data_hp,
        data_lp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )

    return data_lp.reshape(data_shape).transpose(axis, -1), scales.reshape(
        scales_shape
    ).transpose(axis, -1)


@triton_op("lumen::convert_from_mxfp8", mutates_args={})
def convert_from_mxfp8(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: bool | None = None,
) -> torch.Tensor:
    assert output_dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4() and data_lp.dtype == torch.float8_e4m3fn
    elif use_asm and data_lp.dtype == torch.float8_e5m2:
        use_asm = False
        logger.warning(
            f"ASM mode doesn't support {data_lp.dtype}, falling back to non-ASM implementation"
        )

    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)
    orig_shape = data_lp.shape
    data_lp = data_lp.reshape(-1, orig_shape[-1])

    scales = scales.reshape(-1, orig_shape[-1] // block_size)
    data_hp = data_lp.new_empty(orig_shape, dtype=output_dtype).reshape(
        -1, orig_shape[-1]
    )

    stride_xm, stride_xn = data_lp.stride()
    stride_ym, stride_yn = data_hp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    BLOCK_M = min(64, M)
    BLOCK_N = min(64, N)
    wrap_triton(_convert_from_mxfp8_kernel)[grid](
        data_lp,
        data_hp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_ASM=use_asm,
    )
    return data_hp.reshape(orig_shape).transpose(axis, -1)


@convert_to_mxfp8.register_fake
def _fake_convert_to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: bool | None = None,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
    float8_dtype_pt: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape

    data_lp = data_hp.new_empty(data_hp.shape, dtype=float8_dtype_pt)
    if is_2d_block:
        scales_shape = (
            *orig_shape[:-2],
            orig_shape[-2] // block_size,
            orig_shape[-1] // block_size,
        )
    else:
        scales_shape = (*orig_shape[:-1], orig_shape[-1] // block_size)

    scales = data_hp.new_empty(scales_shape, dtype=torch.uint8)
    return data_lp, scales.transpose(axis, -1)


@convert_from_mxfp8.register_fake
def _fake_convert_from_mxfp8(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: bool | None = None,
) -> torch.Tensor:
    data_hp = data_lp.new_empty(data_lp.shape, dtype=output_dtype)
    return data_hp


# ---------------------------------------------------------------------------
# MXFP4 Conversion
# ---------------------------------------------------------------------------


def _dividing_block(dim: int, cap: int, floor: int = 1) -> int:
    """Largest power-of-two block ``<= cap`` that divides *dim*.

    The MXFP4 quantize kernels address their tiles without bounds masks, so a
    grid built with ``cdiv`` over a block that does not divide the dimension
    leaves the last program reading past the input and writing past the scale
    tensor. Shrinking the block keeps every program wholly in range. Training
    shapes are multiples of the cap and are unaffected.
    """
    block = 1 << (min(cap, dim).bit_length() - 1)
    while block > floor and dim % block:
        block >>= 1
    return max(block, floor)


def convert_to_mxfp4(
    data_hp: torch.Tensor,
    block_size: int = 32,
    axis: int = -1,
    use_sr: bool = False,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
    swizzle_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert BF16/FP32 -> packed MXFP4 (uint8) + E8M0 scales (uint8).

    Uses AITER ``dynamic_mxfp4_quant`` for round-to-nearest (RTN) by default.
    Uses Lumen SR kernel when ``use_sr=True`` (for gradient quantization only —
    SR on forward tensors is detrimental per NVFP4 paper §4.4).

    Returns:
        (data_fp4, scales) — packed uint8 + uint8 E8M0 scales.
    """
    assert data_hp.dtype in (torch.float32, torch.bfloat16)

    if axis == 0 or axis == -2:
        data_hp = data_hp.transpose(-2, -1).contiguous()

    orig_shape = data_hp.shape
    data_2d = data_hp.reshape(-1, orig_shape[-1]).contiguous()
    M, N = data_2d.shape

    assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"
    assert N % 2 == 0, f"N={N} must be even for packing"

    use_asm = is_cdna4()

    if swizzle_scale:
        raise ValueError("MXFP4 layout conversion is owned by AITER")

    if not swizzle_scale and not use_sr and not use_asm:
        # AITER RTN path (non-ASM fallback)
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        data_bf16 = (
            data_2d.to(torch.bfloat16) if data_2d.dtype != torch.bfloat16 else data_2d
        )
        fp4_packed, scales_e8m0 = dynamic_mxfp4_quant(data_bf16)
        fp4_packed = fp4_packed.view(torch.uint8)
        scales_e8m0 = scales_e8m0.view(torch.uint8)
        expected_scale_cols = N // block_size
        scales_e8m0 = scales_e8m0[:M, :expected_scale_cols].contiguous()
    else:
        # Lumen unified kernel: ASM (gfx950) or software fallback
        from atomrl.lumen.kernels.mxfp4 import _convert_to_mxfp4_kernel

        # Only the SR path reads the counter, and drawing two Python randoms per
        # launch is measurable on a step that issues hundreds of them.
        if use_sr:
            if philox_seed is None:
                philox_seed = random.randint(0, 2**31 - 2)
            if philox_offset is None:
                philox_offset = random.randint(0, 2**31 - 2)
        else:
            philox_seed = philox_seed or 0
            philox_offset = philox_offset or 0

        n_scale_cols = N // block_size
        fp4_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=data_2d.device)
        scales_e8m0 = torch.empty(
            (M, n_scale_cols),
            dtype=torch.uint8,
            device=data_2d.device,
        )

        BLOCK_M = _dividing_block(M, 64)
        BLOCK_N = _dividing_block(N, 64, floor=block_size)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _convert_to_mxfp4_kernel[grid](
            data_2d,
            fp4_packed,
            scales_e8m0,
            data_2d.stride(0),
            data_2d.stride(1),
            fp4_packed.stride(0),
            fp4_packed.stride(1),
            scales_e8m0.stride(0),
            scales_e8m0.stride(1),
            philox_seed,
            philox_offset,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            QUANT_BLOCK_SIZE=block_size,
            IS_2D_BLOCK=False,
            USE_SR=use_sr,
            USE_ASM=use_asm,
        )

    out_shape = (*orig_shape[:-1], N // 2)
    scale_shape = (*orig_shape[:-1], N // block_size)

    if axis == 0 or axis == -2:
        return (
            fp4_packed.reshape(out_shape).transpose(-2, -1).contiguous(),
            scales_e8m0.reshape(scale_shape).transpose(-2, -1).contiguous(),
        )

    return fp4_packed.reshape(out_shape), scales_e8m0.reshape(scale_shape)


def convert_to_mxfp4_2d(
    data_hp: torch.Tensor,
    block_size: int = 32,
    use_sr: bool = False,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
    shuffle_data: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert BF16/FP32 -> packed MXFP4 with 2D (block×block) tile scaling.

    Each block×block tile shares a single E8M0 scale. The quantized
    representation is transpose-invariant: transposing the packed data
    and the 2D scale grid produces the same quantized values, preserving
    the chain rule across forward and backward passes (NVFP4 paper §4.3).

    Uses the same ``_convert_to_mxfp4_kernel`` as 1D but with
    ``IS_2D_BLOCK=True``, which makes ``_calculate_fp4_scales`` compute
    per-tile (block×block) amax instead of per-row-block (1×block) amax.

    Returns:
        (data_fp4_packed, scales_2d) where scales_2d has shape
        ``(M//block, N//block)`` in uint8 E8M0 format.
    """
    assert data_hp.dtype in (torch.float32, torch.bfloat16)
    orig_shape = data_hp.shape
    data_2d = data_hp.reshape(-1, orig_shape[-1]).contiguous()
    M, N = data_2d.shape

    assert M % block_size == 0, f"M={M} not divisible by block_size={block_size}"
    assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"

    sm, sn = M // block_size, N // block_size
    use_asm = is_cdna4()

    if shuffle_data:
        raise ValueError("MXFP4 layout conversion is owned by AITER")

    from atomrl.lumen.kernels.mxfp4 import _convert_to_mxfp4_kernel

    # Weights take this path once per optimizer step with RTN, where the counter
    # is unused; only draw randoms when the kernel will actually read them.
    if use_sr:
        if philox_seed is None:
            philox_seed = random.randint(0, 2**31 - 2)
        if philox_offset is None:
            philox_offset = random.randint(0, 2**31 - 2)
    else:
        philox_seed = philox_seed or 0
        philox_offset = philox_offset or 0

    fp4_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=data_2d.device)
    scales_2d = torch.empty((sm, sn), dtype=torch.uint8, device=data_2d.device)

    BLOCK_M = _dividing_block(M, 64, floor=block_size)
    BLOCK_N = _dividing_block(N, 64, floor=block_size)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _convert_to_mxfp4_kernel[grid](
        data_2d,
        fp4_packed,
        scales_2d,
        data_2d.stride(0),
        data_2d.stride(1),
        fp4_packed.stride(0),
        fp4_packed.stride(1),
        scales_2d.stride(0),
        scales_2d.stride(1),
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=True,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )

    out_shape = (*orig_shape[:-1], N // 2)
    return fp4_packed.reshape(out_shape), scales_2d


def convert_from_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
    axis: int = -1,
) -> torch.Tensor:
    """Convert packed MXFP4 + E8M0 scales -> BF16/FP32.

    Uses AITER ``mxfp4_to_f32`` + ``e8m0_to_f32`` when available.
    Falls back to a pure-Python dequant path otherwise.
    """
    assert output_dtype in (torch.float32, torch.bfloat16)

    if axis == 0 or axis == -2:
        data_fp4 = data_fp4.transpose(-2, -1).contiguous()
        scales = scales.transpose(-2, -1).contiguous()

    orig_packed_shape = data_fp4.shape
    data_flat = data_fp4.reshape(-1, orig_packed_shape[-1])
    scales_flat = scales.reshape(-1, scales.shape[-1])
    M, N_packed = data_flat.shape
    N = N_packed * 2

    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    values = mxfp4_to_f32(data_flat.view(torch.uint8))
    scale_f32 = e8m0_to_f32(scales_flat.view(torch.uint8))
    scale_expanded = (
        scale_f32.unsqueeze(-1).expand(M, N // block_size, block_size).reshape(M, N)
    )
    result = (values * scale_expanded).to(output_dtype)

    out_shape = (*orig_packed_shape[:-1], N)
    if axis == 0 or axis == -2:
        return result.reshape(out_shape).transpose(-2, -1).contiguous()
    return result.reshape(out_shape)


def convert_from_mxfp4_2d(
    data_fp4: torch.Tensor,
    scales_2d: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    """Dequantize packed MXFP4 with 2D (block×block) E8M0 scales -> BF16/FP32.

    ``scales_2d`` has shape ``(M//block, N//block)`` — one E8M0 scale per tile.
    """
    assert output_dtype in (torch.float32, torch.bfloat16)

    orig_packed_shape = data_fp4.shape
    data_flat = data_fp4.reshape(-1, orig_packed_shape[-1])
    M, N_packed = data_flat.shape
    N = N_packed * 2

    sm, sn = scales_2d.shape[-2], scales_2d.shape[-1]
    assert sm == M // block_size and sn == N // block_size, (
        f"scales_2d shape {scales_2d.shape} doesn't match data shape ({M}, {N}) "
        f"with block_size={block_size}"
    )

    # Unpack nibbles
    _mxfp4_lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=data_flat.device,
    )
    unpacked = data_flat.view(torch.uint8).repeat_interleave(2, dim=-1)
    unpacked[..., ::2] = unpacked[..., ::2] & 0xF
    unpacked[..., 1::2] = unpacked[..., 1::2] >> 4
    values = _mxfp4_lut[unpacked.long()]  # (M, N)

    # E8M0 scales → float: 2^(stored_exp - 127)
    scale_f32 = torch.pow(
        2.0, scales_2d.view(torch.uint8).to(torch.float32) - 127.0
    )  # (sm, sn)
    # Expand 2D scales to per-element: (sm, 1, sn, 1) → (sm, block, sn, block) → (M, N)
    scale_expanded = (
        scale_f32.view(sm, 1, sn, 1)
        .expand(sm, block_size, sn, block_size)
        .reshape(M, N)
    )
    result = (values * scale_expanded).to(output_dtype)

    out_shape = (*orig_packed_shape[:-1], N)
    return result.reshape(out_shape)


def convert_to_mxfp4_dual_axis(
    data_hp: torch.Tensor,
    block_size: int = 32,
    use_sr: bool = True,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emit both axis=1 (1x32) and axis=0 (32x1) MXFP4 quantizations.

    Uses different PRNG offsets for the two axes to avoid correlation.

    Returns:
        (row_fp4, row_scales, col_fp4, col_scales)
    """
    if philox_seed is None:
        philox_seed = random.randint(0, 2**31 - 2)
    if philox_offset is None:
        philox_offset = random.randint(0, 2**31 - 2)

    row_fp4, row_scales = convert_to_mxfp4(
        data_hp,
        block_size=block_size,
        axis=-1,
        use_sr=use_sr,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
    )
    col_fp4, col_scales = convert_to_mxfp4(
        data_hp,
        block_size=block_size,
        axis=0,
        use_sr=use_sr,
        philox_seed=philox_seed + 1,
        philox_offset=philox_offset,
    )
    return row_fp4, row_scales, col_fp4, col_scales


def transpose_packed_fp4(
    data_fp4: torch.Tensor,
    shuffle_data: bool = False,
    in_shuffled: bool = False,
) -> torch.Tensor:
    """Transpose a packed MXFP4 matrix: (M, N//2) -> (N, M//2).

    Uses Lumen Triton kernel (no AITER equivalent).

    Layout conversion belongs to AITER; this helper only transposes row-major
    packed data.
    """
    from atomrl.lumen.kernels.mxfp4 import _transpose_packed_fp4_kernel

    M, N_packed = data_fp4.shape
    N = N_packed * 2
    assert M % 2 == 0, f"M={M} must be even for packed transpose"
    if shuffle_data or in_shuffled:
        raise ValueError("MXFP4 layout conversion is owned by AITER")

    output = torch.empty((N, M // 2), dtype=torch.uint8, device=data_fp4.device)

    BLOCK_M = min(32, M)
    BLOCK_N_PACKED = min(16, N_packed)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_packed, BLOCK_N_PACKED))
    _transpose_packed_fp4_kernel[grid](
        data_fp4,
        output,
        M,
        N_packed,
        data_fp4.stride(0),
        data_fp4.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N_PACKED=BLOCK_N_PACKED,
    )
    return output


_hadamard_cache: dict[tuple[int, torch.device], torch.Tensor] = {}


def _get_hadamard_matrix(g: int, device: torch.device) -> torch.Tensor:
    """Return the normalized g×g Hadamard matrix, cached per (g, device)."""
    key = (g, device)
    if key not in _hadamard_cache:
        H = torch.tensor([[1.0]], device=device)
        while H.shape[0] < g:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        _hadamard_cache[key] = H * (1.0 / (g**0.5))
    return _hadamard_cache[key]


def hadamard_transform(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    g: int = 64,
) -> torch.Tensor:
    """Apply blockwise Random Hadamard Transform.

    Computes (x * diag(S)) @ H_g for blocks of g elements along the last
    dimension, where H_g is the normalized Hadamard matrix.
    """
    assert x.shape[-1] % g == 0, f"N={x.shape[-1]} not divisible by g={g}"
    assert (g & (g - 1)) == 0, f"g={g} must be a power of 2"

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape

    H = _get_hadamard_matrix(g, x.device)
    x_blocked = x_2d.float().reshape(M, N // g, g)
    x_blocked = x_blocked * sign_vector.float()
    out = (x_blocked @ H).to(x.dtype).reshape(orig_shape)
    return out


_HADAMARD_CACHE: dict = {}


def _get_hadamard_matrix_normalized(g: int, device: torch.device) -> torch.Tensor:
    """Return a (g, g) normalized Hadamard matrix, cached per (g, device)."""
    key = (g, device)
    if key not in _HADAMARD_CACHE:
        H = _get_hadamard_matrix(g, device)
        _HADAMARD_CACHE[key] = H
    return _HADAMARD_CACHE[key]


_RHT_MATRIX_ATTR = "_lumen_rht_matrix_bf16"


def _rht_matrix_bf16(sign_vector: torch.Tensor, g: int) -> torch.Tensor:
    """``diag(sign) @ H_g / sqrt(g)`` as BF16, cached on the sign vector.

    The quantizers apply the rotation on the matrix unit, which wants the whole
    map as one operand rather than a sign multiply followed by a butterfly. For
    the g=16 the kernels are built around every entry is ±1/4, so BF16 holds the
    matrix exactly. Caching it on the sign vector ties its lifetime to the
    signs it was built from.
    """
    cached = getattr(sign_vector, _RHT_MATRIX_ATTR, None)
    if (
        cached is not None
        and cached.shape[0] == g
        and cached.device == sign_vector.device
    ):
        return cached
    mat = (
        torch.diag(sign_vector.float()) @ _get_hadamard_matrix(g, sign_vector.device)
    ).to(torch.bfloat16)
    setattr(sign_vector, _RHT_MATRIX_ATTR, mat)
    return mat


def hadamard_quant_mxfp4(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    block_size: int = 32,
    g: int = 16,
    use_sr: bool = True,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Hadamard rotation + MXFP4 quantization in a single kernel launch.

    Equivalent to ``convert_to_mxfp4(hadamard_transform(x, sign, g), ...)``,
    but eliminates one global memory roundtrip (no intermediate BF16 write).

    The Hadamard is applied via matmul with a precomputed (g, g) matrix.
    The sign_vector is baked into the Hadamard matrix (diag(sign) @ H).

    Returns:
        (fp4_packed, scales_e8m0) — same format as ``convert_to_mxfp4``.
    """
    assert x.dtype in (torch.float32, torch.bfloat16)
    assert x.shape[-1] % g == 0, f"N={x.shape[-1]} not divisible by g={g}"
    assert x.shape[-1] % block_size == 0

    orig_shape = x.shape
    # Deliberately not .contiguous(): the kernel addresses x through both
    # strides, so a transposed view works as-is. Callers wanting x^T would
    # otherwise have to materialise it, which on Qwen3-8B's wgrad path cost
    # more GPU time than the wgrad GEMM itself.
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape

    use_asm = is_cdna4()

    # Only the SR path reads the counter (see convert_to_mxfp4).
    if use_sr:
        if philox_seed is None:
            philox_seed = random.randint(0, 2**31 - 2)
        if philox_offset is None:
            philox_offset = random.randint(0, 2**31 - 2)
    else:
        philox_seed = philox_seed or 0
        philox_offset = philox_offset or 0

    fp4_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    scales_e8m0 = torch.empty((M, N // block_size), dtype=torch.uint8, device=x.device)

    BLOCK_M = _dividing_block(M, 64)
    BLOCK_N = _dividing_block(N, 64, floor=max(block_size, g))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    from atomrl.lumen.kernels.mxfp4 import _fused_hadamard_quant_mxfp4_kernel

    _fused_hadamard_quant_mxfp4_kernel[grid](
        x_2d,
        fp4_packed,
        scales_e8m0,
        sign_vector,
        _rht_matrix_bf16(sign_vector, g),
        x_2d.stride(0),
        x_2d.stride(1),
        fp4_packed.stride(0),
        fp4_packed.stride(1),
        scales_e8m0.stride(0),
        scales_e8m0.stride(1),
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )

    out_shape = (*orig_shape[:-1], N // 2)
    scale_shape = (*orig_shape[:-1], N // block_size)
    return fp4_packed.view(out_shape), scales_e8m0.view(scale_shape)


def dual_layout_quant_mxfp4(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    block_size: int = 32,
    g: int = 16,
    use_sr_row: bool = True,
    use_sr_transposed: bool = True,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
    swizzle_scale: bool = False,
    shuffle_col: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both MXFP4 layouts the backward pass needs, from one read of *x*.

    Equivalent to::

        row, row_s = convert_to_mxfp4(x, block_size, axis=-1, use_sr=use_sr_row)
        col, col_s = hadamard_quant_mxfp4(x.t(), sign_vector, block_size, g,
                                          use_sr=use_sr_transposed)

    but reads *x* once, densely. The two-call form has to take x^T as a view,
    whose strided load forces a register layout that costs 1.70x on the same
    shape (report §5.10).

    *x* must be 2D and contiguous, with both dimensions a whole number of quant
    blocks and the row count a whole number of Hadamard groups.

    Returns:
        ``(row_fp4, row_scales, col_fp4, col_scales)`` — ``row_*`` matching
        ``convert_to_mxfp4``, ``col_*`` matching ``hadamard_quant_mxfp4(x.t())``.
    """
    assert x.dim() == 2, f"expected 2D, got {tuple(x.shape)}"
    assert (
        x.is_contiguous()
    ), "x must be contiguous; the point is to avoid a strided read"
    assert x.dtype in (torch.float32, torch.bfloat16)
    M, N = x.shape
    assert (
        M % block_size == 0 and N % block_size == 0
    ), f"({M}, {N}) not a whole number of {block_size}-blocks"
    assert M % g == 0, f"M={M} not divisible by g={g}"
    if swizzle_scale or shuffle_col:
        raise ValueError("MXFP4 layout conversion is owned by AITER")

    # The counter only reaches the kernel through the SR path; an all-RTN call
    # that drew from Python's RNG anyway would also shift the stream every SR
    # caller after it reads.
    if use_sr_row or use_sr_transposed:
        if philox_seed is None:
            philox_seed = random.randint(0, 2**31 - 2)
        if philox_offset is None:
            philox_offset = random.randint(0, 2**31 - 2)
    else:
        philox_seed = philox_seed or 0
        philox_offset = philox_offset or 0

    n_scale_a, n_scale_b = N // block_size, M // block_size
    scale_a_shape = (M, n_scale_a)
    scale_b_shape = (N, n_scale_b)

    row_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    row_scales = torch.empty(scale_a_shape, dtype=torch.uint8, device=x.device)
    col_fp4 = torch.empty((N, M // 2), dtype=torch.uint8, device=x.device)
    col_scales = torch.empty(scale_b_shape, dtype=torch.uint8, device=x.device)

    # BLOCK_M is the transposed output's contiguous run (BLOCK_M/2 bytes), so it
    # wants to be the larger of the two. (256, 32) measured fastest across
    # Qwen3-8B's wgrad shapes.
    BLOCK_M = _dividing_block(M, 256, floor=max(block_size, g))
    BLOCK_N = _dividing_block(N, 32, floor=block_size)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    from atomrl.lumen.kernels.mxfp4 import _dual_layout_quant_mxfp4_kernel

    _dual_layout_quant_mxfp4_kernel[grid](
        x,
        row_fp4,
        row_scales,
        col_fp4,
        col_scales,
        sign_vector,
        _rht_matrix_bf16(sign_vector, g),
        x.stride(0),
        x.stride(1),
        row_fp4.stride(0),
        row_fp4.stride(1),
        row_scales.stride(0),
        row_scales.stride(1),
        col_fp4.stride(0),
        col_fp4.stride(1),
        col_scales.stride(0),
        col_scales.stride(1),
        # Separate streams: the two outputs feed different GEMMs and correlating
        # their rounding noise would defeat the point of using SR on both.
        philox_seed,
        philox_offset,
        philox_offset + 0x9E3779B9,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        USE_SR_A=use_sr_row,
        USE_SR_B=use_sr_transposed,
        USE_ASM=is_cdna4(),
    )

    return row_fp4, row_scales, col_fp4, col_scales


def dequant_hadamard_quant_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    sign_vector: torch.Tensor,
    block_size: int = 32,
    g: int = 16,
    use_sr: bool = False,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
    swizzle_scale: bool = False,
    shuffle_data: bool = False,
    in_scale_swizzled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed MXFP4 (M, K/2) → Hadamard-rotated, transposed MXFP4 (K, M/2).

    Equivalent to::

        hadamard_quant_mxfp4(dequant_transpose_mxfp4(data, scales), sign_vector)

    but never writes the BF16 (K, M) intermediate, which is four times the bytes
    of either FP4 end and has to be read straight back.

    Both dimensions must be a whole number of quant blocks and *M* a whole number
    of Hadamard groups, so that every output quant block lies inside one tile.

    Layout conversion belongs to AITER; inputs and outputs remain row-major.
    """
    assert data_fp4.dim() == 2, f"expected 2D, got {tuple(data_fp4.shape)}"
    M, K_packed = data_fp4.shape
    K = K_packed * 2
    assert (
        M % block_size == 0 and K % block_size == 0
    ), f"({M}, {K}) not a whole number of {block_size}-blocks"
    assert M % g == 0, f"M={M} not divisible by g={g}"
    assert g == 16, f"kernel is hardcoded for g=16, got {g}"
    if swizzle_scale or shuffle_data or in_scale_swizzled:
        raise ValueError("MXFP4 layout conversion is owned by AITER")

    # Activations take this path with RTN, where the kernel never reads the
    # counter; drawing from Python's RNG then would only shift the stream that
    # the gradient's SR reads next.
    if use_sr:
        if philox_seed is None:
            philox_seed = random.randint(0, 2**31 - 2)
        if philox_offset is None:
            philox_offset = random.randint(0, 2**31 - 2)
    else:
        philox_seed = philox_seed or 0
        philox_offset = philox_offset or 0

    from atomrl.lumen.kernels.mxfp4 import _dequant_hadamard_quant_mxfp4_kernel

    n_scale_cols = M // block_size
    out = torch.empty((K, M // 2), dtype=torch.uint8, device=data_fp4.device)
    out_scales = torch.empty(
        (K, n_scale_cols),
        dtype=torch.uint8,
        device=data_fp4.device,
    )

    # BLOCK_M is the output's contiguous run (BLOCK_M/2 bytes) and carries the
    # quant blocks, so it wants to be the larger of the two. (128, 64) measured
    # 5-9% ahead of (128, 32) across Qwen3-8B's activation shapes: FP4 on both
    # sides leaves few bytes to hide latency behind, so the wider tile wins.
    BLOCK_M = _dividing_block(M, 128, floor=max(block_size, g))
    BLOCK_K = _dividing_block(K, 64, floor=block_size)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _dequant_hadamard_quant_mxfp4_kernel[grid](
        data_fp4,
        scales,
        out,
        out_scales,
        _rht_matrix_bf16(sign_vector, g),
        data_fp4.stride(0),
        data_fp4.stride(1),
        scales.stride(0),
        scales.stride(1),
        out.stride(0),
        out.stride(1),
        out_scales.stride(0),
        out_scales.stride(1),
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        QUANT_BLOCK_SIZE=block_size,
        USE_SR=use_sr,
        USE_ASM=is_cdna4(),
    )
    return out, out_scales


def dequant_transpose_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Fused dequant + transpose: packed FP4 (M, K/2) + 1D scales → BF16 (K, M).

    Equivalent to ``convert_from_mxfp4(data, scales).t().contiguous()`` but
    eliminates one full BF16 (M, K) intermediate write.
    """
    from atomrl.lumen.kernels.mxfp4 import _dequant_transpose_mxfp4_kernel

    orig_packed_shape = data_fp4.shape
    data_flat = data_fp4.reshape(-1, orig_packed_shape[-1])
    scales_flat = scales.reshape(-1, scales.shape[-1])
    M, K_packed = data_flat.shape
    K = K_packed * 2

    output = torch.empty((K, M), dtype=torch.bfloat16, device=data_fp4.device)

    BLOCK_M = min(32, M)
    BLOCK_K = min(64, K)
    BLOCK_K = max(BLOCK_K, block_size)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _dequant_transpose_mxfp4_kernel[grid](
        data_flat,
        scales_flat,
        output,
        M,
        K,
        data_flat.stride(0),
        data_flat.stride(1),
        scales_flat.stride(0),
        scales_flat.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        QUANT_BLOCK_SIZE=block_size,
    )
    return output
