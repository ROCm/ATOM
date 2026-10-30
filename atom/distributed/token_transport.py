# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Token and head transport over one process group.

Shared by Ulysses SP, the TP replicated o_proj and the MoE token gathers.
Every choice below depends only on the payload shape and the group, which are
the same on every rank, so all ranks pick the same collective.
"""

from functools import cache

import torch

from atom.utils import envs

# Small payloads can benefit from the registered-buffer all-gather. Large
# prefills use all-to-all to avoid receiving unused heads.
_ALLGATHER_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
# Output rows per rank from which an attention output takes the prefill head
# exchange (FP8 with a gathered amax, the four-rank exchange kernel); decode
# stays below it on the all-gather.
HEAD_EXCHANGE_MIN_LOCAL_ROWS = 2048
# AITER custom_all_reduce kMaxBlocks.
_CA_MAX_BLOCKS = 80


def _ca_comm(group):
    ca = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    return None if ca is None or ca.disabled else ca


def _pynccl(group):
    comm = getattr(getattr(group, "device_communicator", None), "pynccl_comm", None)
    return None if comm is None or comm.disabled else comm


def custom_gather_ok(payload: torch.Tensor, group) -> bool:
    """Whether AITER's registered-buffer all-gather supports this payload."""
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return False
    ca = _ca_comm(group)
    return ca is not None and bool(ca.should_custom_ag(payload))


def custom_reduce_scatter_ok(payload: torch.Tensor, group) -> bool:
    ca = _ca_comm(group)
    return ca is not None and bool(ca.should_custom_rs(payload, 0))


def prefer_custom_gather(payload: torch.Tensor, group) -> bool:
    """Use registered-buffer all-gather only for small supported payloads."""
    if payload.numel() * payload.element_size() > _ALLGATHER_MAX_PAYLOAD_BYTES:
        return False
    return custom_gather_ok(payload, group)


def gather_tokens(x: torch.Tensor, group) -> torch.Tensor:
    """Concatenate every rank's rows, rank-major."""
    x = x.contiguous()
    if custom_gather_ok(x, group):
        return group.all_gather(x, use_custom=True, dim=0)
    pynccl = _pynccl(group)
    if pynccl is not None:
        out = x.new_empty((x.shape[0] * group.world_size, *x.shape[1:]))
        pynccl.all_gather(out, x)
        return out
    return group.all_gather(x, dim=0)


def reduce_scatter_tokens(x: torch.Tensor, group) -> torch.Tensor:
    """Sum over ranks and keep this rank's rank-major row block."""
    x = x.contiguous()
    # The communicator's PyNccl `reduce_scatter` reduces into a temporary and
    # copies it out; for rows, `reduce_scatterv` writes the shard directly.
    if not custom_reduce_scatter_ok(x, group) and _pynccl(group) is not None:
        return group.device_communicator.reduce_scatterv(x, dim=0)
    return group.reduce_scatter(x, dim=0)


def gather_heads(x: torch.Tensor, group) -> torch.Tensor:
    """``[S, H/W*D] -> [S/W, H*D]``: give up the sequence, regain all heads."""
    w = group.world_size
    if w <= 1:
        return x
    x = x.reshape(x.shape[0], -1)
    s_local = x.shape[0] // w
    # AITER's last-dimension gather requires 16-byte alignment.
    if x.shape[1] * x.element_size() % 16 == 0 and prefer_custom_gather(x, group):
        start = group.rank_in_group * s_local
        return group.all_gather(x, use_custom=True, dim=1)[start : start + s_local]
    ca = head_exchange_communicator(x, group)
    if ca is not None:
        return exchange_heads(x, ca)
    from atom.distributed.sp_kernels import all_to_all_into

    send = x.reshape(w, s_local, x.shape[1]).contiguous()
    recv = torch.empty_like(send)
    all_to_all_into(recv, send, group)
    # recv[i] is head-group i for our token chunk -> concatenate on heads.
    return recv.permute(1, 0, 2).reshape(s_local, -1)


@cache
def _head_exchange_kernel():
    """AITER's four-rank head exchange, or None if this AITER lacks it."""
    try:
        import aiter.ops.sp_head_exchange as ops
    except ModuleNotFoundError as exc:
        if exc.name != "aiter.ops.sp_head_exchange":
            raise
        return None
    return getattr(ops, "sp_head_exchange", None)


def head_exchange_communicator(x, group):
    """Select the four-rank prefill exchange; decode uses the regular gather."""
    if (
        not x.is_cuda
        or group.world_size != 4
        or not envs.ATOM_USE_CUSTOM_ALL_GATHER
        or x.ndim != 2
        or x.shape[0] < 4 * HEAD_EXCHANGE_MIN_LOCAL_ROWS
        or x.shape[0] % 4
        or x.shape[1] * x.element_size() % 16
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or _head_exchange_kernel() is None
    ):
        return None
    from atom.plugin.prepare import is_plugin_mode

    if is_plugin_mode():
        return None
    ca = _ca_comm(group)
    if ca is None or getattr(ca, "_pool", None) is None:
        return None
    if not ca.should_custom_ag(x):
        return None
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return ca if get_gfx_runtime() == "gfx950" else None


def exchange_heads(x, ca, *, registered=False):
    """Read just this rank's token slice from each peer and concatenate heads.

    The output is already in the o-projection's row layout. All operations use
    the communicator's current stream. Registered scratch must be produced
    immediately before this call; end_sync protects its reuse.
    """
    out = torch.empty((x.shape[0] // 4, x.shape[1] * 4), device=x.device, dtype=x.dtype)
    pool = ca._pool["input"]
    graph_registered = (
        ca._IS_CAPTURING
        and torch.cuda.is_current_stream_capturing()
        and ca.enable_register_for_capturing
    )
    _head_exchange_kernel()(
        ca._ptr,
        x,
        out,
        pool.data_ptr,
        pool.max_size,
        not (registered or graph_registered),
        _CA_MAX_BLOCKS,
    )
    return out
