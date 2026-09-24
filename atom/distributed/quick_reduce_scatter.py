# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental INT4 reduce-scatter, disabled until model-level validation.

The kernel uses QuickReduce's existing IPC allocation and device flag colors.
Calls using one communicator must remain serialized on a single stream, just
as for the existing QuickAllReduce implementation. There is only one payload
communication phase and one quantization; no full all-reduce is performed.
"""

import os

import torch


def try_quick_reduce_scatter(x: torch.Tensor, group) -> torch.Tensor | None:
    """Return this rank's sum shard, or None for the normal RS fallback.

    Selection depends only on identical rank-invariant configuration and
    padded tensor metadata. This experimental path needs the matching AITER
    ``module_quick_reduce_scatter`` extension; an explicitly enabled missing
    extension raises instead of silently reporting a benchmark of fallback.
    """
    if os.environ.get("ATOM_SP_QUICK_REDUCE_SCATTER", "0") != "1":
        return None
    comm = getattr(getattr(group, "device_communicator", None), "qr_comm", None)
    if comm is None or comm.disabled or comm.world_size != 4:
        return None
    if getattr(comm.qr_quant_level, "name", None) != "INT4":
        return None
    if x.ndim == 0 or x.shape[0] % comm.world_size or not x.is_contiguous():
        return None
    if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda:
        return None
    size = x.numel() * x.element_size()
    # M3/gfx950: 24 MiB beat custom BF16 RS by 19%; at 6 MiB the 2%
    # difference was too small to justify enabling another lossy operation.
    # Decode continues to use its normal RS.
    minimum = int(os.environ.get("ATOM_SP_QUICK_RS_MIN_BYTES", str(24 * 1024**2)))
    if size < minimum or size > comm.qr_max_size or size % (16 * comm.world_size):
        return None

    from aiter.ops.quick_reduce_scatter import qr_reduce_scatter

    out = x.new_empty((x.shape[0] // comm.world_size, *x.shape[1:]))
    qr_reduce_scatter(comm._ptr, x, out, bool(comm.use_fp16_kernels))
    return out
