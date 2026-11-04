# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Uncached device memory: the peer buffers' and a step's own scratch.

A launch's hand-offs (tagged pairs, flags, plain rows behind them) are read
by CTAs on other XCDs. In cached memory the consumer's device-scope load goes
to the producer XCD's L2 for the line; uncached, it is served at the line's
home memory.

Uncached memory is never freed while the process runs: nothing here calls
``hipFree``, so a width rolled back, a refused build, a runner dropped on a
model reload all keep theirs. On MI355X / ROCm 7.2 a freed and reused
uncached page, once a kernel had polled it, gave later steps rank-disagreeing
reads (sgl-amd-mono #76). Its users' ``close`` only closes peer mappings.
"""

from __future__ import annotations

import torch

from atom.mono.runtime.hip_ipc import allocate_uncached


def ipc_heap():
    """aiter's IPC heap (``PeerBuffer``'s handle exchange), imported on use:
    importing it at module scope would make every user's CPU test require an
    AITER build."""

    from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

    return UncachedIpcHeap


class DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address, for a
    non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def alloc_uncached(nbytes: int) -> int:
    """``nbytes`` of zeroed uncached device memory, never freed (module doc):
    the raw address."""
    return allocate_uncached(nbytes)


def uncached_bytes(nbytes: int, device: torch.device) -> torch.Tensor:
    """A uint8 view of ``nbytes`` of zeroed uncached memory (``alloc_uncached``)."""
    return torch.as_tensor(DeviceBytes(alloc_uncached(nbytes), nbytes), device=device)
