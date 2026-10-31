# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""INT4 reduce-scatter for large SP4 tensors on gfx950.

The kernel uses QuickReduce's existing IPC allocation and device flag colors.
Calls using one communicator must remain serialized on a single stream, just
as for the existing QuickAllReduce implementation. There is only one payload
communication phase and one quantization; no full all-reduce is performed.
"""

import logging

import torch

logger = logging.getLogger("atom")

try:
    import aiter.ops.quick_reduce_scatter as _qrs_ops
except ModuleNotFoundError as exc:
    if exc.name not in {"aiter", "aiter.ops", "aiter.ops.quick_reduce_scatter"}:
        raise
    _qrs_ops = None

# Resolve optional AITER support before dispatch; execution failures propagate.
_qr_reduce_scatter = getattr(_qrs_ops, "qr_reduce_scatter", None)

# MiniMax-M3 SP4 on gfx950: INT4 beat the BF16 custom reduce-scatter by 19% at
# 24 MiB but only 2% at 6 MiB, too little for another lossy step. Decode stays
# below it on the exact reduce-scatter.
QUICK_RS_MIN_BYTES = 24 << 20
_announced = False


def try_quick_reduce_scatter(x: torch.Tensor, group) -> torch.Tensor | None:
    """Return this rank's sum shard, or None for the normal RS fallback.

    Use the existing INT4 QuickReduce communicator for supported prefill
    tensors. Missing companion AITER support retains the normal collective;
    failures while building or executing an available kernel propagate.
    """
    comm = getattr(getattr(group, "device_communicator", None), "qr_comm", None)
    if _qr_reduce_scatter is None or comm is None or comm.disabled:
        return None
    if comm.world_size != 4:
        return None
    if getattr(comm.qr_quant_level, "name", None) != "INT4":
        return None
    if x.ndim == 0 or x.shape[0] % comm.world_size or not x.is_contiguous():
        return None
    if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda:
        return None
    size = x.numel() * x.element_size()
    if (
        size < QUICK_RS_MIN_BYTES
        or size > comm.qr_max_size
        or size % (16 * comm.world_size)
    ):
        return None
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        return None

    global _announced
    if not _announced:
        _announced = True
        logger.info(
            "AITER_QUICK_REDUCE_QUANTIZATION=INT4 also quantizes the SP MoE "
            "reduce-scatter of tensors from %d MiB",
            QUICK_RS_MIN_BYTES >> 20,
        )
    out = x.new_empty((x.shape[0] // comm.world_size, *x.shape[1:]))
    _qr_reduce_scatter(comm._ptr, x, out, bool(comm.use_fp16_kernels))
    return out
