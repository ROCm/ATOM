# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_M3_ATTN_QOUT: o_proj's MXFP8 input straight from the NVFP4 sparse decode attention.

On pure-decode steps of 8..128 rows the attention (aiter ``sparse_pa_nvfp4_decode(quant_out=...)``) writes the
e4m3 codes and E8M0 scales of o_proj's input and parks them on its impl; ``oproj_input`` hands them to the
o_proj. Everything else (prefill, pad rows, dummy runs, an aiter without ``quant_out``) quantizes the bf16
attention output exactly as ``native_quant_linear`` would (``quantize_fp8``): the codes are byte-identical.
The choice is made inside the custom op at run time, so the compiled graph has one shape for every row count.
"""

import functools

import torch
from aiter import dtypes
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.config import get_current_atom_config
from atom.model_ops.blockscale import quantize_fp8

FP8_DTYPE = dtypes.fp8


@functools.cache
def aiter_supports() -> bool:
    try:
        import inspect

        from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode import (
            sparse_pa_nvfp4_decode,
        )
    except ImportError:
        return False
    return "quant_out" in inspect.signature(sparse_pa_nvfp4_decode).parameters


def _fake(attn_output: torch.Tensor, layer_name: str):
    m = attn_output.shape[0]
    k = attn_output.numel() // m
    return (
        torch.empty((m, k), dtype=FP8_DTYPE, device=attn_output.device),
        torch.empty(
            (m, k // 32), dtype=torch.float8_e8m0fnu, device=attn_output.device
        ),
    )


@torch_compile_guard(gen_fake=_fake, mutates_args=[])
def oproj_input(
    attn_output: torch.Tensor, layer_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    impl = (
        get_current_atom_config()
        .compilation_config.static_forward_context[layer_name]
        .impl
    )
    res = getattr(impl, "_qout_result", None)
    impl._qout_result = None
    m = attn_output.shape[0]
    if res is not None and res[2] == m:
        return res[0], res[1]
    return quantize_fp8(attn_output.reshape(m, -1))
