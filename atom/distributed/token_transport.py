# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Head-exchange limits and buffers for Ulysses SP.

Every choice below depends only on the payload shape and the group, which are
the same on every rank, so all ranks pick the same collective.
"""

import torch

from atom.utils import envs

# Per-rank payloads up to this are all-gathered rather than sent all-to-all.
# Measured on SP4 the two cross at ~3.7 MB; 2 MiB keeps a margin.
ALLGATHER_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
# Output rows per rank from which an attention output takes the prefill head
# exchange (FP8 with a gathered amax, the four-rank exchange kernel); decode
# stays below it on the all-gather.
HEAD_EXCHANGE_MIN_LOCAL_ROWS = 2048


def head_exchange_comm(group):
    """The communicator of the four-rank head exchange, unless the env opts
    out. Static: a layer decides it once, when built."""
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return None
    return group.device_communicator.ca_comm


def head_exchange_view(ca, x: torch.Tensor, rows: int, dtype=None):
    """The ``[rows, x.shape[1]]`` buffer of ``dtype`` that ``ca`` exchanges
    from: ``x`` itself when it fits as is, else a registered staging view.
    None when the exchange does not take it."""
    dtype = x.dtype if dtype is None else dtype
    if ca is None:
        return None
    if (rows, dtype) == (x.shape[0], x.dtype):
        view = x
    else:
        view = ca.registered_input_view((rows, x.shape[1]), dtype)
    if view is not None and ca.should_head_exchange(view):
        return view
    return None
