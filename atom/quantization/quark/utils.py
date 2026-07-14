# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any
import regex as re
import triton
import triton.language as tl
import torch
from aiter import QuantType


def deep_compare(dict1: Any, dict2: Any) -> bool:
    if type(dict1) is not type(dict2):
        return False
    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)
    elif isinstance(dict1, list):
        return set(dict1) == set(dict2)
    else:
        return dict1 == dict2


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    """
    for target in targets:
        if _is_equal_or_regex_match(layer_name, target):
            return True
    return False


def _is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


@triton.jit
def _weight_dequant_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):  # type: ignore[no-untyped-def]
    """
    Triton kernel for dequantizing FP8 weights using scaling factors.

    This kernel is provided by deepseek-ai for efficient FP8 weight dequantization.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant_fp8(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantize FP8 weight tensor using inverse scale with Triton kernel.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    _weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# Optional E8M0 dtype: only available on newer torch builds.
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)


def _mx_block_scale_dtype():
    """The block-scale dtype mandated by the MX (microscaling) format: E8M0.

    Every MX scheme (``QuantType.per_1x32``) stores a shared power-of-two block
    scale in E8M0, regardless of whether the elements are FP4 or FP8 — this is
    fixed by the MX spec, not a per-call choice. Resolving it here gives both the
    MXFP4 and MXFP8 online-quant paths a single source of truth, so callers pass
    a value derived from the format rather than a repeated literal.
    """
    from aiter import dtypes

    return dtypes.fp8_e8m0


def weight_dequant_mxfp8(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """Dequantize an MXFP8 weight to the default float dtype."""
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, K = x.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    n_blocks = K // block_size
    assert s.shape == (M, n_blocks), f"scale shape {tuple(s.shape)} != {(M, n_blocks)}"

    if _E8M0_DTYPE is not None and s.dtype == _E8M0_DTYPE:
        # E8M0 dtype decodes straight to the 2**(e-127) multiplier.
        scale = s.to(torch.float32)
    else:
        # Raw E8M0 integer codes stored as uint8 / float.
        scale = torch.exp2(s.to(torch.float32) - 127.0)

    out_dtype = torch.get_default_dtype()
    y = x.to(torch.float32).reshape(M, n_blocks, block_size)
    y = y * scale.unsqueeze(-1)
    return y.reshape(M, K).to(out_dtype)


def quant_mxfp4_online_even(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online MXFP4 weight quant via the aiter HIP kernel with ``Even`` round mode.

    Round-half-to-even on the FP4/E2M1 grid + an E8M0 block scale (note: on
    gfx942 ``Even`` falls back to round-half-away in software). Returns the
    packed weight viewed as ``dtypes.fp4x2`` and the block scale as
    ``dtypes.fp8_e8m0``.

    Shared by the Linear and MoE online-quant paths so both stay in sync.
    ``quant_mxfp4_hip`` requires a 2D contiguous fp16/bf16 input, so we
    normalise the input accordingly before calling it.
    """
    from aiter import dtypes
    from aiter.ops.quant import quant_mxfp4_hip
    from aiter.utility.mx_types import MxScaleRoundModeInt

    orig_shape = weight.shape
    q_in = weight.contiguous()
    if q_in.dtype not in (torch.float16, torch.bfloat16):
        q_in = q_in.to(torch.bfloat16)
    # quant_mxfp4_hip requires 2D input; reshape higher-rank tensors
    # (e.g. fused MoE expert weights [num_experts, N, K]) before the call
    # and restore the leading dimensions on the outputs.
    if q_in.dim() > 2:
        q_in = q_in.reshape(-1, q_in.shape[-1])
    q_weight, weight_scale = quant_mxfp4_hip(q_in, round_mode=MxScaleRoundModeInt.Even)
    if len(orig_shape) > 2:
        leading = orig_shape[:-1]
        q_weight = q_weight.reshape(*leading, q_weight.shape[-1])
        weight_scale = weight_scale.reshape(*leading, weight_scale.shape[-1])
    return q_weight.view(dtypes.fp4x2), weight_scale.view(_mx_block_scale_dtype())


def quantize_weight_to_fp8_128x128_blockscale(weight, quant_dtype):
    """Quantize a 2D weight to FP8 with 128x128 block scales.

    Returns:
        q_weight: quantized weight with the same shape as input ``weight``.
        scale: per-block scale with shape ``(ceil(N/128), ceil(K/128))``.
    """
    assert weight.dim() == 2, f"expected 2D weight, got shape={tuple(weight.shape)}"

    w = weight.to(torch.float32).contiguous()
    n, k = w.shape
    n_blocks = (n + 127) // 128
    k_blocks = (k + 127) // 128
    n_padded = n_blocks * 128
    k_padded = k_blocks * 128

    if n_padded != n or k_padded != k:
        w = torch.nn.functional.pad(w, (0, k_padded - k, 0, n_padded - n))

    w_blocks = w.view(n_blocks, 128, k_blocks, 128).permute(0, 2, 1, 3).contiguous()

    finfo = torch.finfo(quant_dtype)
    block_amax = w_blocks.abs().amax(dim=(2, 3))
    scale = (block_amax / finfo.max).clamp_min(torch.finfo(torch.float32).tiny)

    q_blocks = torch.clamp(
        w_blocks / scale.unsqueeze(-1).unsqueeze(-1), min=finfo.min, max=finfo.max
    ).to(quant_dtype)

    q_weight = (
        q_blocks.permute(0, 2, 1, 3)
        .contiguous()
        .view(n_padded, k_padded)[:n, :k]
        .contiguous()
    )
    return q_weight, scale.contiguous()


def quant_weight_online(
    weight: torch.Tensor,
    online_quant_type: QuantType,
    online_quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch online weight quantization by target dtype / scheme.

    Single entry point shared by the Linear and MoE online-quant paths so both
    stay in sync:

    - MXFP4 (``dtypes.fp4x2``): use the aiter HIP kernel with ``Even`` round
      mode (:func:`quant_mxfp4_online_even`), matching the offline Quark kernel.
    - per_1x128 FP8: use :func:`quantize_weight_to_fp8_128x128_blockscale` to
      produce a true 128x128 block scale of shape ``(N//128, K//128)``. This is
      what the blockscale GEMM consumes; ``get_hip_quant(per_1x128)`` would
      instead produce a 1x128-along-K scale ``(N, K//128)`` that is inconsistent
      with the GEMM and collapses generation.
    - MXFP8 (``per_1x32`` + ``dtypes.fp8``): fp8 weights with a per-32 block
      scale. See the e8m0 note below for why ``scale_type`` must be forced.
    - other FP8 (incl. ptpc_fp8 per-token / per-channel): use the aiter quant
      function resolved from ``get_hip_quant(online_quant_type)``.

    :param weight: The (already dequantized) weight tensor to quantize.
    :param online_quant_type: Online quantization scheme, used to resolve the
        FP8 quant function via ``get_hip_quant``.
    :param online_quant_dtype: Target online quantization dtype.
    :return: ``(q_weight, weight_scale)``.
    """
    from aiter import dtypes, get_hip_quant

    if online_quant_dtype == dtypes.fp4x2:
        return quant_mxfp4_online_even(weight)
    if online_quant_type == QuantType.per_1x128:
        return quantize_weight_to_fp8_128x128_blockscale(weight, online_quant_dtype)
    quant_func = get_hip_quant(online_quant_type)
    # A per_1x32 scheme *is* MX (microscaling): its block scale is E8M0 by
    # definition of the format, independent of the element dtype. So the scale
    # dtype is derived from the scheme, not chosen per case — the MXFP4 branch
    # above already relies on this (aiter forces e8m0 for fp4x2), and here the
    # MXFP8 (fp8) branch needs the same E8M0 scale. We only have to pass it
    # explicitly because aiter's per_1x32 fp8 quantizer keeps scale_type=fp32 as
    # a backward-compat default, whereas the whole consuming side is E8M0:
    # Fp8MoEMethod.create_weights allocates an e8m0 (uint8) scale buffer and the
    # MXFP8 GEMM / flydsl MoE kernels only read that byte scale. Without it the
    # scale silently reverts to fp32 and weight loading / inference break.
    if online_quant_type == QuantType.per_1x32:
        mx_scale_type = _mx_block_scale_dtype()
        return quant_func(
            weight,
            quant_dtype=online_quant_dtype,
            scale_type=mx_scale_type,
        )
    return quant_func(weight, quant_dtype=online_quant_dtype)
