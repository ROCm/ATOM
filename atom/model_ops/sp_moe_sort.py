# SPDX-License-Identifier: MIT
"""Startup eligibility for the optional M3 SP4 tiled sorting path."""


def supports_m3_sp_tiled_sort(layer) -> bool:
    """Cache on the M3 expert layer; AITER also checks the runtime sort contract."""
    from aiter import QuantType
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.distributed.ulysses_sp import get_sp_world_size
    from atom.model_ops.moe import Mxfp4MoEMethod
    from atom.plugin.prepare import is_plugin_mode
    from atom.utils import envs

    return not (
        not envs.ATOM_SP_MOE_TILED_SORT
        or get_sp_world_size() != 4
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
        or get_gfx_runtime() != "gfx950"
    )
