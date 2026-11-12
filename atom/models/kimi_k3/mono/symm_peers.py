# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A gfx1250 mono kernel's receive banks on every TP peer, over torch.symm_mem
with the MORI backend: the transport aiter's custom all-reduce maps across
trays (HIP IPC reaches only the local tray)."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class SymmPeers:
    """``buffer``: this rank's banks. ``peers``: every rank's base address as
    this rank maps it, an int64 per rank. ``epochs``: the kernel's launch
    counters (``k2_allreduce_gfx1250``)."""

    buffer: torch.Tensor
    peers: torch.Tensor
    epochs: torch.Tensor


def map_symm_peers(
    nbytes: int, counters: int, group, tp: int, device: torch.device
) -> SymmPeers | None:
    """``nbytes`` of banks on every rank of ``group``, or None on every rank
    unless all of them map every peer. Only where aiter's custom all-reduce
    already took the MORI symm_mem transport: it chose, and voted on, the
    process-global backend this allocation then uses."""
    if os.getenv("AITER_CUSTOM_AR_USE_SYMM_MEM", "0") not in ("1", "true", "True"):
        return None
    import torch.distributed._symmetric_memory as symm_mem

    if symm_mem.get_backend(device) != "MORI":
        return None
    buffer = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
    buffer.zero_()
    torch.cuda.synchronize(device)
    # a host collective: at bind time, never inside a capture
    ptrs = list(symm_mem.rendezvous(buffer, group).buffer_ptrs)
    mapped = torch.tensor([int(len(ptrs) == tp and all(ptrs))])
    dist.all_reduce(mapped, op=dist.ReduceOp.MIN, group=group)
    if not mapped.item():
        return None
    return SymmPeers(
        buffer,
        torch.tensor(ptrs, dtype=torch.int64, device=device),
        torch.zeros(counters, dtype=torch.int32, device=device),
    )
