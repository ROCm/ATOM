# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Token and head transport over one process group.

Shared by Ulysses SP and the MoE token gathers.
Every choice below depends only on the payload shape and the group, which are
the same on every rank, so all ranks pick the same collective.
"""

import torch

from atom.utils import envs

# Below this, sending W times the bytes through the registered-buffer
# all-gather still beats RCCL's all-to-all; above it, the all-to-all's smaller
# payload wins. Measured on SP4 hidden rows: the two cross at ~3.7 MB per rank,
# and the gather only falls further behind past it. `indexer_cp`'s 512 KiB is
# for its key gather, a different payload and a different pair of kernels.
ALLGATHER_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
# Output rows per rank from which an attention output takes the prefill head
# exchange (FP8 with a gathered amax, the four-rank exchange kernel); decode
# stays below it on the all-gather.
HEAD_EXCHANGE_MIN_LOCAL_ROWS = 2048


def _ca_comm(group):
    ca = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    return None if ca is None or ca.disabled else ca


def _pynccl(group):
    comm = getattr(getattr(group, "device_communicator", None), "pynccl_comm", None)
    return None if comm is None or comm.disabled else comm


def custom_gather_ok(
    payload: torch.Tensor, group, max_bytes: int | None = None
) -> bool:
    """Whether AITER's registered-buffer all-gather takes ``payload``, and, with
    ``max_bytes``, whether the payload is small enough to prefer it. Reads only
    the env, the group and the payload's shape, so every rank agrees."""
    if max_bytes is not None and payload.numel() * payload.element_size() > max_bytes:
        return False
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return False
    ca = _ca_comm(group)
    return ca is not None and bool(ca.should_custom_ag(payload))


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


def gather_heads(x: torch.Tensor, group) -> torch.Tensor:
    """``[S, H/W*D] -> [S/W, H*D]``: give up the sequence, regain all heads."""
    w = group.world_size
    if w <= 1:
        return x
    x = x.reshape(x.shape[0], -1)
    s_local = x.shape[0] // w
    # AITER's last-dimension gather requires 16-byte alignment.
    if x.shape[1] * x.element_size() % 16 == 0 and custom_gather_ok(
        x, group, ALLGATHER_MAX_PAYLOAD_BYTES
    ):
        start = group.rank_in_group * s_local
        return group.all_gather(x, use_custom=True, dim=1)[start : start + s_local]
    ca = head_exchange_communicator(x, group)
    if ca is not None:
        return ca.head_exchange(x)
    from atom.distributed.sp_kernels import all_to_all_into

    send = x.reshape(w, s_local, x.shape[1]).contiguous()
    recv = torch.empty_like(send)
    all_to_all_into(recv, send, group)
    # recv[i] is head-group i for our token chunk -> concatenate on heads.
    return recv.permute(1, 0, 2).reshape(s_local, -1)


def head_exchange_communicator(x, group):
    """Select the four-rank prefill exchange; decode uses the regular gather."""
    if (
        not x.is_cuda
        or x.shape[0] < 4 * HEAD_EXCHANGE_MIN_LOCAL_ROWS
        or x.dtype != torch.bfloat16
    ):
        return None
    from atom.plugin.prepare import is_plugin_mode

    if is_plugin_mode() or not custom_gather_ok(x, group):
        return None
    ca = _ca_comm(group)
    if not ca.should_head_exchange(x):
        return None
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return ca if get_gfx_runtime() == "gfx950" else None
