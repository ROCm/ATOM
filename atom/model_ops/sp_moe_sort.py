# SPDX-License-Identifier: MIT
"""Startup eligibility for M3 SP4 tiled MoE sorting."""

from functools import lru_cache
from inspect import Parameter, signature


@lru_cache(maxsize=1)
def _aiter_supports_tiled_sort(fused_moe) -> bool:
    try:
        option = signature(fused_moe).parameters.get("use_tiled_sort")
    except (TypeError, ValueError):
        return False
    return option is not None and option.kind in (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    )


def supports_m3_sp_tiled_sort(layer, *, tp_replicated_o_proj: bool = False) -> bool:
    """Cache on the M3 expert layer; AITER also checks the runtime sort contract."""
    from aiter import QuantType
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.distributed.ulysses_sp import get_sp_world_size
    from atom.model_ops.moe import Mxfp4MoEMethod, fused_moe
    from atom.plugin.prepare import is_plugin_mode

    return not (
        get_sp_world_size() != (1 if tp_replicated_o_proj else 4)
        or (tp_replicated_o_proj and get_tensor_model_parallel_world_size() != 4)
        or is_plugin_mode()
        or type(layer.quant_method) is not Mxfp4MoEMethod
        or layer.quant_method.quant_type != QuantType.per_1x32
        or layer.use_ep
        or layer.custom_routing_function is not None
        or layer.expert_layout.uses_dispatch_remap
        or layer.hidden_size != 6144
        or layer.intermediate_size_per_partition != 768
        or layer.global_num_experts != 128
        or layer.local_num_experts != 129
        or layer.top_k != 4
        or layer.num_fused_shared_experts != 1
        or not _aiter_supports_tiled_sort(fused_moe)
        or get_gfx_runtime() != "gfx950"
    )
