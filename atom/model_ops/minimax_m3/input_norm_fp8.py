# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M3 Gemma norm and per-token FP8 using the AITER fused kernel."""

import torch
from aiter.utility import dtypes

try:
    import aiter.ops.fused_qk_rmsnorm_group_quant as _fused_norm_ops
except ModuleNotFoundError as exc:
    if exc.name != "aiter.ops.fused_qk_rmsnorm_group_quant":
        raise
    _fused_norm_ops = None

_fused_per_token_quant = getattr(
    _fused_norm_ops, "fused_qk_rmsnorm_per_token_quant", None
)


def fused_m3_gemma_norm_fp8(x, weight, epsilon, residual=None):
    assert _fused_per_token_quant is not None
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.ndim == 2 and x.shape[1] == 6144 and x.stride(1) == 1
    assert dtypes.fp8 == torch.float8_e4m3fn
    out = torch.empty(x.shape, dtype=dtypes.fp8, device=x.device)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    res_out = torch.empty_like(x) if residual is not None else None
    # AITER quantizes the FP32 norm result directly, without rounding to BF16.
    if x.shape[0] > 0:
        _fused_per_token_quant(
            out,
            scale,
            x,
            weight,
            epsilon,
            q_res_out=res_out,
            q_residual=residual,
            gemma_norm=True,
        )
    # First decoder layer keeps the original residual stream by reference.
    return out, scale, x if residual is None else res_out


def supports_m3_fused_gemma_fp8(
    hidden_width: int, *, tp_replicated_o_proj: bool = False
) -> bool:
    """Select the validated SP4 layout on gfx950 before graph tracing."""
    from atom.distributed.ulysses_sp import get_sp_world_size

    if (
        not torch.cuda.is_available()
        or _fused_per_token_quant is None
        or get_sp_world_size() != (1 if tp_replicated_o_proj else 4)
        or hidden_width != 6144
        or dtypes.fp8 != torch.float8_e4m3fn
    ):
        return False
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.config import get_current_atom_config
    from atom.plugin.prepare import is_plugin_mode

    return (
        not is_plugin_mode()
        and get_tensor_model_parallel_world_size() == (4 if tp_replicated_o_proj else 1)
        and get_current_atom_config().torch_dtype == torch.bfloat16
        and get_gfx_runtime() == "gfx950"
    )
