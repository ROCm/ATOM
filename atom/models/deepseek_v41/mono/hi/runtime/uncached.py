# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Uncached device memory for in-launch scratch (``MonoKernel.buffers(uncached=True)``): retired and reused, never
freed.

A monokernel's mailboxes are written by one CTA and polled by others, often on another XCD. In ordinary (cached)
device memory, a consumer's read can go through the producer XCD's L2 before the line reaches memory. Uncached memory
(``hipExtMallocWithFlags`` 0x3, as ``PeerGroup``'s buffers) keeps every access at the line's home. Another
monokernel's largest single win was allocating its in-launch scratch this way (-2.0..-2.8 µs a launch).

Freeing uncached memory corrupts later allocations: silent wrong results and faults in later kernels (seen with
``PeerGroup.close``, and by others; neither barriers nor unmap-then-free help). So a block is never freed. When the
last tensor viewing it is gone it returns to this module's pool, and the next request of at most its size takes it
again, zeroed. ``zeros(nbytes)`` gives a uint8 tensor of ``nbytes`` (a view of a block of at least that size).
"""

from __future__ import annotations

import torch

from atom.mono.runtime import hip_ipc

_FREE: list[tuple[int, int, int]] = []  # (nbytes, ptr, device) blocks no tensor views
_STATS = {"blocks": 0, "bytes": 0, "reused": 0}


class _Block:
    """One uncached allocation, as ``__cuda_array_interface__`` for ``torch.as_tensor``; the tensor keeps it alive,
    and it goes back to the pool when the tensor is gone."""

    def __init__(self, ptr: int, nbytes: int, device: int):
        self.ptr, self.nbytes, self.device = ptr, nbytes, device
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
        }

    def __del__(self):
        _FREE.append((self.nbytes, self.ptr, self.device))


def _take(nbytes: int, device: int) -> tuple[int, int] | None:
    """The smallest pooled block on ``device`` of at least ``nbytes`` (removed from the pool), or None."""
    fits = [(n, i) for i, (n, _, d) in enumerate(_FREE) if d == device and n >= nbytes]
    if not fits:
        return None
    n, p, _ = _FREE.pop(min(fits)[1])
    return n, p


def zeros(nbytes: int, device="cuda") -> torch.Tensor:
    """``nbytes`` of zeroed uncached device memory as a uint8 tensor (module docstring)."""
    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        got = _take(nbytes, index)
        if got is None:  # (a new block: zeroed by the allocation, whole 4 KiB pages)
            size = -(-max(nbytes, 1) // 4096) * 4096
            got = (size, hip_ipc.allocate_uncached(size))
            torch.cuda.synchronize()  # (the allocation's zeroing done before any stream uses it)
            _STATS["blocks"] += 1
            _STATS["bytes"] += size
        else:
            _STATS["reused"] += 1
            torch.cuda.synchronize()  # (a reused block's last kernels are done before it is zeroed)
            hip_ipc.check_hip_error(
                hip_ipc.load_runtime().hipMemset(got[1], 0, got[0]),
                operation="hipMemset",
            )
            torch.cuda.synchronize()
        size, ptr = got
        t = torch.as_tensor(_Block(ptr, size, index), device=f"cuda:{index}")
    return t[:nbytes]
