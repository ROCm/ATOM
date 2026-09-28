# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""INT4 reduce-scatter for large SP4 tensors on gfx950.

The kernel uses QuickReduce's existing IPC allocation and device flag colors.
Calls using one communicator must remain serialized on a single stream, just
as for the existing QuickAllReduce implementation. There is only one payload
communication phase and one quantization; no full all-reduce is performed.
"""

from importlib import import_module

import torch


def try_quick_reduce_scatter(x: torch.Tensor, group) -> torch.Tensor | None:
    """Return this rank's sum shard, or None for the normal RS fallback.

    Use the existing INT4 QuickReduce communicator for supported prefill
    tensors. Missing companion AITER support retains the normal collective;
    failures while building or executing an available kernel propagate.
    """
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
    # Below 24 MiB, keep the normal reduce-scatter used by decode.
    if size < 24 * 1024**2 or size > comm.qr_max_size or size % (16 * comm.world_size):
        return None
    arch = getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "")
    arch = arch.split(":", 1)[0]
    if arch != "gfx950":
        return None

    try:
        module = import_module("aiter.ops.quick_reduce_scatter")
    except ModuleNotFoundError as exc:
        if exc.name not in {"aiter", "aiter.ops", "aiter.ops.quick_reduce_scatter"}:
            raise
        return None
    qr_reduce_scatter = getattr(module, "qr_reduce_scatter", None)
    if qr_reduce_scatter is None:
        return None

    out = x.new_empty((x.shape[0] // comm.world_size, *x.shape[1:]))
    qr_reduce_scatter(comm._ptr, x, out, bool(comm.use_fp16_kernels))
    return out
