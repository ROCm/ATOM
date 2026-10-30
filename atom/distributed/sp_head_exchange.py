# SPDX-License-Identifier: MIT
"""M3 SP4 head exchange directly into the o-projection's final row layout."""

import torch

from atom.utils import envs

try:
    import aiter.ops.sp_head_exchange as _head_exchange_ops
except ModuleNotFoundError as exc:
    if exc.name != "aiter.ops.sp_head_exchange":
        raise
    _head_exchange_ops = None

# Resolve optional AITER support before dispatch; execution failures propagate.
_exchange = getattr(_head_exchange_ops, "sp_head_exchange", None)


def head_exchange_communicator(x, *, group=None):
    """Select M3's prefill exchange; decode uses the regular head gather."""
    if (
        not x.is_cuda
        or _exchange is None
        or not envs.ATOM_USE_CUSTOM_ALL_GATHER
        or x.ndim != 2
        or x.shape[0] < 8192
        or x.shape[0] % 4
        or x.shape[1] not in (1024, 2048)
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        return None
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size

    from atom.config import get_current_atom_config
    from atom.distributed.ulysses_sp import get_sp_group, get_sp_world_size
    from atom.plugin.prepare import is_plugin_mode

    if is_plugin_mode():
        return None
    if group is None:
        if get_sp_world_size() != 4 or get_tensor_model_parallel_world_size() != 1:
            return None
        group = get_sp_group()
    elif (
        group.world_size != 4
        or get_sp_world_size() != 1
        or get_tensor_model_parallel_world_size() != 4
        or not getattr(get_current_atom_config(), "m3_tp_replicated_o_proj", False)
    ):
        return None
    architectures = (
        getattr(get_current_atom_config().hf_config, "architectures", None) or ()
    )
    if not any(
        arch
        in (
            "MiniMaxM3SparseForCausalLM",
            "MiniMaxM3SparseForConditionalGeneration",
        )
        for arch in architectures
    ):
        return None
    ca = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    if ca is None or ca.disabled or getattr(ca, "_pool", None) is None:
        return None
    if not ca.should_custom_ag(x):
        return None
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return ca if get_gfx_runtime() == "gfx950" else None


def exchange_heads(x, ca, *, registered=False):
    """Read just this rank's token slice from each peer and concatenate heads.

    All operations use the communicator's current stream. Registered scratch
    must be produced immediately before this call; end_sync protects its reuse.
    """
    out = torch.empty((x.shape[0] // 4, x.shape[1] * 4), device=x.device, dtype=x.dtype)
    pool = ca._pool["input"]
    graph_registered = (
        ca._IS_CAPTURING
        and torch.cuda.is_current_stream_capturing()
        and ca.enable_register_for_capturing
    )
    _exchange(
        ca._ptr,
        x,
        out,
        pool.data_ptr,
        pool.max_size,
        not (registered or graph_registered),
        80,
    )
    return out
