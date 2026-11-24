# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Paired gate/up MXFP8 GEMM + SwiGLU-OAI, bf16 out, at 1-8 rows, for the dense MLP N 6144 (gate | up of I 3072)
x K 6144 (TP4): the one-wave fp32 form and the two-wave form (``gateup_swiglu_bf16_onewave``,
``gateup_swiglu_bf16``). Callers: the dense decoder block's MLP and the DSpark draft's MLP."""

import functools

import torch


@functools.cache
def _gateup_swiglu_bf16_launcher(form):
    if form == "fp32":
        from .gateup_swiglu_bf16_onewave import compile_mxfp8_gateup_swiglu_bf16_onewave
        return compile_mxfp8_gateup_swiglu_bf16_onewave()
    from .gateup_swiglu_bf16 import compile_mxfp8_gateup_swiglu_bf16
    return compile_mxfp8_gateup_swiglu_bf16(form)


def mxfp8_gateup_swiglu_bf16(activation,activation_scale,weight,weight_scale,m,
                            alpha=1.702,beta=1.0,limit=7.0,form='fp32'):
    """Paired gate/up GEMM and model SwiGLU, BF16 [m,3072], for 1-8 rows.

    Weight is the existing packed [6144,6144] gate|up layout. Activation codes
    use the 16x16 layout, padded to 16 rows; scales use AITER's layout.
    Each dot product rounds to BF16 before activation. ``form`` is the caller's
    numerical contract (fp32 or bf16_chain), independent of the row count.
    Target fp32 uses one K wave; draft bf16_chain retains two K waves.
    The FP32 dot summation order differs between these kernel configurations.
    The BF16 output feeds the existing down linear's quantizer without changing it.
    """
    import struct
    from amd_kernels.common.tensor_shim import _run_compiled
    assert 1<=m<=8 and form in ('fp32','bf16_chain')
    assert activation.shape==(16,6144) and activation.element_size()==1 and activation.is_contiguous()
    assert weight.shape==(6144,6144) and weight.element_size()==1 and weight.is_contiguous()
    assert activation_scale.shape[0]>=32 and activation_scale.shape[1]==192
    out=torch.empty((m,3072),device=activation.device,dtype=torch.bfloat16)
    u8=lambda t:t.view(torch.uint8).view(-1)
    bits=lambda v:struct.unpack('<i',struct.pack('<f',v))[0]
    _run_compiled(_gateup_swiglu_bf16_launcher(form),u8(activation),u8(weight),u8(out),u8(activation_scale),
                  u8(weight_scale),m,activation_scale.numel(),1,192,bits(alpha),bits(beta),bits(limit),torch.cuda.current_stream())
    return out


def warmup_mxfp8_gateup_swiglu_bf16(activation,activation_scale,weight,weight_scale,forms=('fp32','bf16_chain')):
    """Compile the reached activation forms before graphs or serving."""
    for form in forms:
        mxfp8_gateup_swiglu_bf16(activation,activation_scale,weight,weight_scale,8,form=form)
