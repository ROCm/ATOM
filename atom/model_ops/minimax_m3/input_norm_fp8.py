"""M3 Gemma norm and per-token FP8 using the existing AITER fused kernel."""

import torch
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_per_token_quant
from aiter.utility import dtypes


def fused_m3_gemma_norm_fp8(x, weight, epsilon, residual=None):
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.ndim == 2 and x.shape[1] == 6144 and x.stride(1) == 1
    assert dtypes.fp8 == torch.float8_e4m3fn
    out = torch.empty(x.shape, dtype=dtypes.fp8, device=x.device)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    res_out = torch.empty_like(x) if residual is not None else None
    # AITER's per-token entry quantizes the FP32 norm result directly. Its
    # numerical contract is validated by model accuracy, not BF16 byte parity.
    if x.shape[0] > 0:
        fused_qk_rmsnorm_per_token_quant(
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


def supports_m3_fused_gemma_fp8(hidden_width: int) -> bool:
    """Startup-only support gate; caller caches this before graph tracing."""
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size

    from atom.config import get_current_atom_config
    from atom.distributed.ulysses_sp import get_sp_world_size
    from atom.utils import envs

    if (
        not getattr(envs, "ATOM_SP_FUSED_GEMMA_FP8", False)
        or get_sp_world_size() != 4
        or get_tensor_model_parallel_world_size() != 1
        or hidden_width != 6144
        or dtypes.fp8 != torch.float8_e4m3fn
        or get_current_atom_config().torch_dtype != torch.bfloat16
    ):
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.plugin.prepare import is_plugin_mode

    return not is_plugin_mode() and get_gfx_runtime() == "gfx950"
