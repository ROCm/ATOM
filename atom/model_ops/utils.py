# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Tuple, Optional, List, Union, Any
import torch
from aiter import per_tensor_quant, dtypes, QuantType
from aiter.ops.shuffle import shuffle_weight
from torch import nn
from collections.abc import Iterable, Mapping
from types import MappingProxyType
import regex as re
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale


def requantize_with_max_scale(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_widths: List[int],
    normalize_e4m3fn_to_e4m3fnuz=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    if normalize_e4m3fn_to_e4m3fnuz:
        quant_dtype = torch.float8_e4m3fnuz
        weight = weight.view(torch.float8_e4m3fn)
        max_w_scale = weight_scale.max() * 2.0
    else:
        quant_dtype = weight.dtype
        max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (
        weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint or normalize_e4m3fn_to_e4m3fnuz:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight.view(quant_dtype)[start:end, :], _ = per_tensor_quant(
                weight_dq, max_w_scale, quant_dtype=quant_dtype
            )
            start = end

    return max_w_scale, weight.view(quant_dtype)


def shuffle_weights(
    *tensors: torch.Tensor, layout: tuple[int, int] = (16, 16)
) -> tuple[torch.Tensor, ...]:
    """
    Applies shuffle_weight function from AITER to each
    input tensor and returns them.

    Rearranges (shuffles) the input tensor/s
    into a specified block layout for optimized computation.

    Args:
        *tensors: Variable number of torch.Tensor objects.
        layout: A pair of integers specifying the
        block sizes used to divide the tensors during shuffling.
        Default is (16, 16).

    Returns:
    A Tuple of shuffled tensors.
    """
    return tuple(shuffle_weight(tensor, layout=layout) for tensor in tensors)


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def per_tensor_dequantize(
        tensor: torch.Tensor, inv_scale: Union[float,
                                               torch.Tensor]) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def get_and_maybe_dequant_weights(layer: nn.Module) -> torch.Tensor:
    if layer.quant_type != QuantType.No:
        # NOTE: This should only be used offline, since it's O(N^3)
        eye = torch.eye(
            layer.input_size,
            dtype=torch.bfloat16,
            device=layer.weight.device,
        )
        dequant_weights = layer(eye)
        del eye
        # standardize to (output, input)
        return dequant_weights.T
    return layer.weight


# utility for tensor dims > 2 cases
def b_dynamic_mxfp4_quant(x):
    h, b, d = x.shape
    x, x_scales = dynamic_mxfp4_quant(x.reshape(-1, d))
    return x.view(h, b, d // 2), x_scales.view(h, b, d // 32)
    #return x.view(h, b, d // 2), x_scales.view(h, b, d // 32)


def mxfp4_to_f32(x, is_threed):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=-1)
    if is_threed:
        x[..., ::2] = x[..., ::2] & 0xF
        x[..., 1::2] = x[..., 1::2] >> 4
    else:
        x[:, ::2] = x[:, ::2] & 0xF
        x[:, 1::2] = x[:, 1::2] >> 4

    mxfp4_list = [
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
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    # Convert the input tensor `x` (assumed to be in e8m0 format) to float32.
    # e8m0 is a custom 8-bit floating point format with 8 bits for exponent, 0 for mantissa.
    # This means the value is essentially 2^(exponent - 127), similar to how IEEE-754 stores floats.

    # Convert x to float32 for computation, and compute the power of 2 by subtracting the bias (127).
    x_f32 = 2 ** ((x.to(torch.float32)) - 127)

    # If the exponent value was 255 (i.e., 2^(128)), this is a special case usually used to represent NaN or Inf.
    # Since this custom format has no mantissa, treat 2^128 as NaN.
    x_f32[x_f32 == 128] = float("nan")
    return x_f32

def quark_post_load_weights(self_attn: nn.Module, w: torch.Tensor, quant_format: str):
    if "mxfp4" in quant_format:
     
        # when dtype is bf16, the processing flow is to dynamic quantize bf16 tensor to uint8 tensor
        # do w_kc (bf16) first to get the w_kc(uint8) w_s_kc(uint8)
        # and w_vc repeating the same procedure of w_kc to get  w_vc(uint8) w_s_vc(uint8)
        if w.dtype == torch.bfloat16:
            # w_kc, w_vc = w.split(
            # [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            w_kc, w_s_kc = b_dynamic_mxfp4_quant(w_kc.transpose(-2, -1))
            w_kc = w_kc.transpose(-2, -1)
            w_s_kc = w_s_kc.transpose(-2, -1)
            w_vc, w_s_vc = b_dynamic_mxfp4_quant(w_vc)
            w_s_kc = w_s_kc.transpose(1, 2).contiguous().transpose(1, 2)
            w_s_vc = w_s_vc.contiguous().transpose(1, 2)
        elif w.dtype == torch.uint8:  # static quant for mxfp4
            # when dtype is uint8, it means the w has been quantized to mxfp4 format
            # but we must separate it to w_kc and w_vc.
            # The quantized tensor size is only half of original tensor size
            # and the scaling factor is 1/32, the transpose behavior will be not correct
            # need to upcast it to fp32 to separate w to w_kc and w_vc
            # to ensure the following transpose behavior is correct
            # and then do mxfp4 quant again
            w = mxfp4_to_f32(w, True).to(torch.bfloat16)
            w_scales = self_attn.kv_b_proj.weight_scale.repeat_interleave(32, dim=-1)
            w_scales = e8m0_to_f32(w_scales).to(torch.bfloat16)
            w = w * w_scales
            w_kc, w_vc = w.unflatten(
                0, (-1, (self_attn.qk_nope_head_dim + self_attn.v_head_dim))
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            w_kc, w_s_kc = b_dynamic_mxfp4_quant(w_kc.transpose(-2, -1))
            w_kc = w_kc.transpose(-2, -1)
            w_s_kc = w_s_kc.transpose(-2, -1)
            w_vc, w_s_vc = b_dynamic_mxfp4_quant(w_vc)
            w_s_kc = w_s_kc.transpose(1, 2).contiguous().transpose(1, 2)
            w_s_vc = w_s_vc.contiguous().transpose(1, 2)

        return w_kc, w_s_kc, w_vc, w_s_vc

# # utility for tensor dims > 2 cases
# def b_dynamic_mxfp4_quant(x):
#     h, b, d = x.shape
#     x, x_scales = dynamic_mxfp4_quant(x.reshape(-1, d))
#     return x.view(h, b, d // 2), x_scales.view(h, b, d // 32)

# def quark_post_load_weights(self_attn: nn.Module, w: torch.Tensor, quant_format: str):
#     if "mxfp4" in quant_format:
#         # when dtype is bf16, the processing flow is to dynamic quantize bf16 tensor to uint8 tensor
#         # do w_kc (bf16) first to get the w_kc(uint8) w_s_kc(uint8)
#         # and w_vc repeating the same procedure of w_kc to get  w_vc(uint8) w_s_vc(uint8)
#         if w.dtype == torch.bfloat16:
#             # w_kc, w_vc = w.split(
#             # [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
#             w_kc, w_vc = w.unflatten(
#                 0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
#             ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
#             w_kc, w_s_kc = b_dynamic_mxfp4_quant(w_kc.transpose(-2, -1))
#             w_kc = w_kc.transpose(-2, -1)
#             w_s_kc = w_s_kc.transpose(-2, -1)
#             w_vc, w_s_vc = b_dynamic_mxfp4_quant(w_vc)
#             w_s_kc = w_s_kc.transpose(1, 2).contiguous().transpose(1, 2)
#             w_s_vc = w_s_vc.contiguous().transpose(1, 2)
#         elif w.dtype == torch.uint8:  # static quant for mxfp4
#             # when dtype is uint8, it means the w has been quantized to mxfp4 format
#             # but we must separate it to w_kc and w_vc.
#             # The quantized tensor size is only half of original tensor size
#             # and the scaling factor is 1/32, the transpose behavior will be not correct
#             # need to upcast it to fp32 to separate w to w_kc and w_vc
#             # to ensure the following transpose behavior is correct
#             # and then do mxfp4 quant again
#             w = mxfp4_to_f32(w).to(torch.bfloat16)
#             w_scales = self_attn.kv_b_proj.weight_scale.repeat_interleave(32, dim=-1)
#             w_scales = e8m0_to_f32(w_scales).to(torch.bfloat16)
#             w = w * w_scales
#             w_kc, w_vc = w.unflatten(
#                 0, (-1, (self_attn.qk_nope_head_dim + self_attn.v_head_dim))
#             ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
#             w_kc, w_s_kc = b_dynamic_mxfp4_quant(w_kc.transpose(-2, -1))
#             w_kc = w_kc.transpose(-2, -1)
#             w_s_kc = w_s_kc.transpose(-2, -1)
#             w_vc, w_s_vc = b_dynamic_mxfp4_quant(w_vc)
#             w_s_kc = w_s_kc.transpose(1, 2).contiguous().transpose(1, 2)
#             w_s_vc = w_s_vc.contiguous().transpose(1, 2)

#         return w_kc, w_s_kc, w_vc, w_s_vc