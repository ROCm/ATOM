# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Head transport for Ulysses SP, and the custom all-gather admission it
shares with other row gathers.

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


def _pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """``x`` followed by zero rows up to ``rows``."""
    if x.shape[0] == rows:
        return x
    out = x.new_empty((rows, *x.shape[1:]))
    out[: x.shape[0]].copy_(x)
    out[x.shape[0] :].zero_()
    return out


def gather_heads(
    x: torch.Tensor, group, rows: int | None = None, ca=None
) -> torch.Tensor:
    """``[S, H/W*D] -> [S/W, H*D]``: give up the sequence, regain all heads.

    ``x`` may hold only the leading rows of S = ``rows``; the rest are sent as
    zeros, padded inside the send buffer. ``ca`` is the layer's
    `head_exchange_comm`.
    """
    w = group.world_size
    if w <= 1:
        return x
    x = x.reshape(x.shape[0], -1)
    rows = x.shape[0] if rows is None else rows
    s_local = rows // w
    row_bytes = x.shape[1] * x.element_size()
    # AITER's last-dimension gather requires 16-byte alignment.
    small = row_bytes % 16 == 0 and rows * row_bytes <= ALLGATHER_MAX_PAYLOAD_BYTES
    view = None
    if not small and takes_head_exchange(rows, w):
        view = head_exchange_view(ca, x, rows)
    if view is x:
        return ca.head_exchange(x)
    if view is not None:
        # Pad in the registered pool: this copy replaces the exchange's own.
        view[: x.shape[0]].copy_(x)
        view[x.shape[0] :].zero_()
        return ca.head_exchange(view, registered=True)
    x = _pad_rows(x, rows)
    if small and custom_gather_ok(x, group):
        start = group.rank_in_group * s_local
        return group.all_gather(x, use_custom=True, dim=1)[start : start + s_local]
    send = x.reshape(w, s_local, x.shape[1]).contiguous()
    recv = torch.empty_like(send)
    group.device_communicator.all_to_all(recv, send)
    # recv[i] is head-group i for our token chunk -> concatenate on heads.
    return recv.permute(1, 0, 2).reshape(s_local, -1)


def takes_head_exchange(rows: int, world: int) -> bool:
    """Whether an attention output of ``rows`` rows is long enough for the
    four-rank head exchange."""
    return rows >= world * HEAD_EXCHANGE_MIN_LOCAL_ROWS


def head_exchange_comm(group):
    """The communicator of the four-rank head exchange, if the env and GPU
    allow it. Static: a layer decides it once, when built."""
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return None
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return _ca_comm(group) if get_gfx_runtime() == "gfx950" else None


def head_exchange_view(ca, x: torch.Tensor, rows: int, dtype=None):
    """The ``[rows, x.shape[1]]`` buffer of ``dtype`` that ``ca`` exchanges
    from: ``x`` itself when it fits as is, else a registered staging view.
    None when the exchange does not take it."""
    dtype = x.dtype if dtype is None else dtype
    if ca is None or x.dtype != torch.bfloat16:
        return None
    if (rows, dtype) == (x.shape[0], x.dtype):
        view = x
    else:
        view = ca.registered_input_view((rows, x.shape[1]), dtype)
        if view is None:
            return None
    wire = view.view(torch.bfloat16)
    if ca.should_custom_ag(wire) and ca.should_head_exchange(wire):
        return view
    return None
