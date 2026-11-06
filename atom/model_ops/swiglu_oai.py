# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""SwiGLU-OAI activation used by MiniMax-M3.

MiniMax-M3 stores dense and expert gate/up activations in split layout:
``[gate | up]``.  The activation is:

    gate * sigmoid(alpha * gate) * (up + beta)

with optional clamping.  ATOM only supports this path on AMD GPU, so the
implementation is Triton-only.

``swiglu_oai_split_reducing`` takes the output of a gate/up GEMM that left its
split-K reduction to its consumer (aiter.gemm_a8w8_bpreshuffle_splitk_deferred):
the kernel sums the fp32 partials as aiter's split-K reduce does, writes the
reduced gate/up back and activates it, all in one launch.

``swiglu_oai_split_quant`` / ``swiglu_oai_split_reducing_quant`` also quantize
the activation per token to FP8 for a per-token FP8 down projection: the same
kernel with one program per row, bitwise the activation followed by aiter's
``dynamic_per_token_scaled_quant``.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard
from triton.language.extra.hip import libdevice

# Widest per-rank intermediate size the quantizing kernel keeps in one program
# (the row's amax needs the whole row).
SWIGLU_OAI_QUANT_MAX_INTER = 8192


@triton.jit
def _splitk_reduced(ptr, split_stride, offs, mask, NUM_SPLITS: tl.constexpr):
    # aiter's gemm_a8w8_splitk_reduce, element for element: an fp32 sum in split
    # order, rounded once to bf16 (round to nearest even).
    acc = tl.load(ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, NUM_SPLITS):
        acc += tl.load(ptr + s * split_stride + offs, mask=mask, other=0.0)
    return acc.to(tl.bfloat16)


@triton.jit
def _per_token_fp8(x, mask, FP8_MAX: tl.constexpr):
    """aiter's dynamic_per_token_scaled_quant (csrc/kernels/quant_kernels.cu) of
    one row, op for op, on the row rounded to bf16 as that kernel reads it: amax
    starts at 0 and skips NaNs (fmaxf), scale = amax * float(1 / FP8_MAX), and
    each element is x * rcp(scale), the hardware reciprocal (v_rcp_f32, not a
    divide), clamped to +-FP8_MAX and rounded to nearest even. aiter builds it
    with -fgpu-flush-denormals-to-zero, so a subnormal scale is 0. A zero row
    thus has scale 0 and quantizes 0 * inf, as there."""
    x = x.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(x), 0.0), axis=0), 0.0)
    scale = amax * (1.0 / FP8_MAX)
    scale = tl.where(scale < 1.1754943508222875e-38, 0.0, scale)  # FLT_MIN
    q = libdevice.fast_dividef(x, scale)  # x * rcp(scale)
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
    return q, scale


@triton.jit
def _swiglu_oai_kernel(
    gate_up_ptr,
    out_ptr,
    n_inter: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    alpha,
    beta,
    limit,
    has_limit: tl.constexpr,
    block_i: tl.constexpr,
    # Split-K partials of a contiguous gate_up (NUM_SPLITS > 0 only): fp32
    # [NUM_SPLITS, >= rows, 2 * n_inter].
    part_ptr,
    part_stride_split,
    NUM_SPLITS: tl.constexpr,
    # Per-token FP8 output (QUANT only): out is FP8, its row scales fp32 [rows],
    # and one program covers a row (block_i >= n_inter).
    scale_ptr,
    QUANT: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    # Inductor may pass Python float arguments as fp64, unlike eager
    # Triton. Keep the activation in fp32 in both execution modes.
    alpha = tl.cast(alpha, tl.float32)
    beta = tl.cast(beta, tl.float32)
    limit = tl.cast(limit, tl.float32)

    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * block_i + tl.arange(0, block_i)
    mask = cols < n_inter

    if NUM_SPLITS > 0:
        part = part_ptr + row * (2 * n_inter)
        gate_bf = _splitk_reduced(part, part_stride_split, cols, mask, NUM_SPLITS)
        up_bf = _splitk_reduced(
            part, part_stride_split, n_inter + cols, mask, NUM_SPLITS
        )
        tl.store(gate_up_ptr + row * stride_gm + cols * stride_gn, gate_bf, mask=mask)
        tl.store(
            gate_up_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
            up_bf,
            mask=mask,
        )
        gate = gate_bf.to(tl.float32)
        up = up_bf.to(tl.float32)
    else:
        gate = tl.load(
            gate_up_ptr + row * stride_gm + cols * stride_gn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gate_up_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    if has_limit:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)

    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    if QUANT:
        out, scale = _per_token_fp8(out, mask, FP8_MAX)
        tl.store(scale_ptr + row, scale)
    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def swiglu_oai_quant_supported(n_inter: int, quant_dtype: torch.dtype) -> bool:
    """Whether ``swiglu_oai_split_quant`` takes a per-rank intermediate size of
    ``n_inter`` to ``quant_dtype``: OCP FP8 E4M3, the row in one program."""
    return quant_dtype == torch.float8_e4m3fn and n_inter <= SWIGLU_OAI_QUANT_MAX_INTER


def _swiglu_oai(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    out_dtype: torch.dtype | None,
    gate_up_partials: torch.Tensor | None,
    quant_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"SwiGLU-OAI expects an even last dimension, got {gate_up.shape[-1]}."
        )
    if not gate_up.is_cuda:
        raise RuntimeError("SwiGLU-OAI is only supported on AMD GPU tensors.")

    orig_shape = gate_up.shape
    two_i = orig_shape[-1]
    n_inter = two_i // 2
    x2 = gate_up.reshape(-1, two_i)
    out = torch.empty(
        (x2.shape[0], n_inter),
        dtype=quant_dtype or out_dtype or gate_up.dtype,
        device=gate_up.device,
    )

    # Unused unless NUM_SPLITS > 0; any tensor stands in for the pointer.
    partial_args = {"part_ptr": x2, "part_stride_split": 0, "NUM_SPLITS": 0}
    if gate_up_partials is not None:
        if not (
            gate_up.dim() == 2
            and gate_up.dtype == torch.bfloat16
            and gate_up.stride() == (two_i, 1)
            and gate_up_partials.dtype == torch.float32
            and gate_up_partials.dim() == 3
            and gate_up_partials.shape[1] >= gate_up.shape[0]
            and gate_up_partials.stride()[1:] == (two_i, 1)
        ):
            raise ValueError(
                "gate_up_partials must be the fp32 split-K partials of the "
                "contiguous bf16 [rows, 2I] gate_up"
            )
        partial_args = {
            "part_ptr": gate_up_partials,
            "part_stride_split": gate_up_partials.stride(0),
            "NUM_SPLITS": gate_up_partials.shape[0],
        }

    if quant_dtype is None:
        scale = None
        # Unused unless QUANT; any tensor stands in for the pointer.
        quant_args = {"scale_ptr": out, "QUANT": False, "FP8_MAX": 0.0}
        block_i = 512 if n_inter >= 2048 else 256
        num_warps = 4
    else:
        if not swiglu_oai_quant_supported(n_inter, quant_dtype):
            raise ValueError(
                f"SwiGLU-OAI per-token quant supports float8_e4m3fn up to "
                f"{SWIGLU_OAI_QUANT_MAX_INTER} wide, got {quant_dtype} x {n_inter}"
            )
        scale = torch.empty((x2.shape[0], 1), dtype=torch.float32, device=out.device)
        quant_args = {
            "scale_ptr": scale,
            "QUANT": True,
            "FP8_MAX": torch.finfo(quant_dtype).max,
        }
        # One program per row: the scale needs the row's amax. Reading 4 to 8
        # split-K partials per element, a row needs more waves in flight.
        block_i = triton.next_power_of_2(n_inter)
        num_warps = 16 if gate_up_partials is None else 32
    grid = (x2.shape[0], triton.cdiv(n_inter, block_i))
    _swiglu_oai_kernel[grid](
        x2,
        out,
        n_inter,
        x2.stride(0),
        x2.stride(1),
        out.stride(0),
        out.stride(1),
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        has_limit=limit is not None,
        block_i=block_i,
        num_warps=num_warps,
        **partial_args,
        **quant_args,
    )
    out = out.reshape(*orig_shape[:-1], n_inter)
    if scale is not None:
        scale = scale.reshape(*orig_shape[:-1], 1)
    return out, scale


def swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    out_dtype: torch.dtype | None = None,
    gate_up_partials: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply MiniMax-M3 SwiGLU-OAI to a split-layout ``[..., 2I]`` tensor.

    ``gate_up_partials``: the claimed split-K partials (``aiter.splitk_partials``)
    of a contiguous bf16 2-D ``gate_up``, reduced here and written back into it."""
    return _swiglu_oai(
        gate_up, alpha, beta, limit, out_dtype, gate_up_partials, quant_dtype=None
    )[0]


def swiglu_oai_split_quant(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    gate_up_partials: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``swiglu_oai_split`` of a bf16 ``gate_up`` quantized per token, in one
    kernel: ``(x, x_scale)`` bitwise as aiter's ``get_hip_quant(QuantType.
    per_Token)`` (dynamic_per_token_scaled_quant) gives them for the bf16
    activation, x_scale fp32 ``[..., 1]``. ``gate_up_partials`` as there."""
    if gate_up.dtype != torch.bfloat16:
        raise ValueError(f"SwiGLU-OAI quant expects bf16 gate_up, got {gate_up.dtype}")
    return _swiglu_oai(
        gate_up, alpha, beta, limit, None, gate_up_partials, quant_dtype=quant_dtype
    )


def _claim_splitk_partials(gate_up: torch.Tensor) -> torch.Tensor | None:
    try:
        from aiter import splitk_partials
    except ImportError:  # aiter without consumer-side split-K reduction
        return None
    return splitk_partials(gate_up)


def _swiglu_oai_split_reducing_fake(
    gate_up: torch.Tensor, alpha: float, beta: float, limit: float
) -> torch.Tensor:
    return gate_up.new_empty((*gate_up.shape[:-1], gate_up.shape[-1] // 2))


@torch_compile_guard(gen_fake=_swiglu_oai_split_reducing_fake, mutates_args=[])
def swiglu_oai_split_reducing(
    gate_up: torch.Tensor, alpha: float, beta: float, limit: float
) -> torch.Tensor:
    """``swiglu_oai_split`` of a gate/up GEMM output that may still carry its
    split-K partials (`LinearBase.splitk_consumer_max_m`). An opaque op: the
    partials are claimed at run time by the output's address, which a traced
    graph cannot see."""
    return swiglu_oai_split(
        gate_up, alpha, beta, limit, gate_up_partials=_claim_splitk_partials(gate_up)
    )


def _swiglu_oai_split_reducing_quant_fake(
    gate_up: torch.Tensor, alpha: float, beta: float, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = gate_up.shape[:-1]
    return (
        gate_up.new_empty((*rows, gate_up.shape[-1] // 2), dtype=torch.float8_e4m3fn),
        gate_up.new_empty((*rows, 1), dtype=torch.float32),
    )


@torch_compile_guard(gen_fake=_swiglu_oai_split_reducing_quant_fake, mutates_args=[])
def swiglu_oai_split_reducing_quant(
    gate_up: torch.Tensor, alpha: float, beta: float, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``swiglu_oai_split_reducing`` quantized per token to FP8 E4M3, as
    ``swiglu_oai_split_quant``."""
    return swiglu_oai_split_quant(
        gate_up, alpha, beta, limit, gate_up_partials=_claim_splitk_partials(gate_up)
    )
