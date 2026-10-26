# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric uncached peer memory for in-kernel TP exchanges (gfx1250), on aiter's
UncachedIpcHeap: every rank allocates ``nbytes`` (hipDeviceMallocUncached), the IPC
handles are exchanged over a CPU (gloo) group, and every rank maps every peer's.

Like atom.mono.runtime.peer_memory.PeerBuffer, without its gfx950 step-fence import."""

import torch


def _heap():
    from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

    return UncachedIpcHeap


class PeerBuffer:
    """``local``: this rank's base; ``addresses``: int64 device tensor [npes] of every
    rank's base (its own at ``rank``)."""

    def __init__(self, nbytes: int, group, rank: int, npes: int, device):
        heap = _heap()
        self.local = heap.alloc_uncached(nbytes)
        self.rank = rank
        self._opened = []
        handles = heap.gather_object_list_via_broadcast(
            group, heap.get_mem_handle_bytes(self.local)
        )
        addresses = []
        for peer, handle in enumerate(handles):
            if peer == rank:
                addresses.append(self.local)
            else:
                base = heap.open_mem_handle(handle)
                self._opened.append(base)
                addresses.append(base)
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self):
        heap = _heap()
        for base in self._opened:
            heap.close_mem_handle(base)
        heap.free_device_mem(self.local)
        self._opened = []
