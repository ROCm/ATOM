# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric peer memory for in-kernel TP reductions."""

import torch
from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

from atom.mono.runtime.consensus import bind_agreed
from atom.mono.runtime.step_begin import FENCE_BYTES


class PeerBuffer:
    """One uncached buffer per rank, every rank holding every peer's address
    (``addresses``, int64, indexed by TP rank). What lives where, and why no rank
    overwrites a peer's unread data, is the model's layout's to say.

    ``bytes`` is the data region, the part zeroed between steps; the step
    fence's slots sit past it (``step_begin``).

    The constructor is collective over ``group`` (the handle exchange, then an
    agreement that every rank mapped every peer): every rank must reach it, so a
    runner reaches it only after ``tp_agree``. A failed mapping on any rank raises
    ``MonoUnsupported`` on all of them.
    """

    def __init__(
        self,
        nbytes: int,
        group,
        rank: int,
        npes: int,
        device: torch.device,
    ):
        self.local = 0
        self.rank = rank
        self._opened: list[int] = []
        self.bytes = self.addresses = None
        handle = None

        def allocate():
            self.local = UncachedIpcHeap.alloc_uncached(nbytes + FENCE_BYTES)
            # Non-owning view: HIP memory is released only by close().
            storage = torch._C._construct_storage_from_data_pointer(
                self.local, device, nbytes
            )
            self.bytes = torch.empty(0, dtype=torch.uint8, device=device).set_(
                storage, 0, (nbytes,), (1,)
            )

        def export():
            nonlocal handle
            handle = UncachedIpcHeap.get_mem_handle_bytes(self.local)

        def map_peers():
            addresses = []
            for peer, peer_handle in enumerate(handles):
                base = self.local
                if peer != rank:
                    base = UncachedIpcHeap.open_mem_handle(peer_handle)
                    self._opened.append(base)
                addresses.append(base)
            self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

        try:
            bind_agreed(allocate, group)
            bind_agreed(export, group)
            handles = (
                UncachedIpcHeap.gather_object_list_via_broadcast(group, handle)
                if npes > 1
                else [handle]
            )
            bind_agreed(map_peers, group)
        except Exception:
            self.close()
            raise

    def kernel_args(self) -> dict:
        """A kernel's view of the group, by ABI name: this rank's buffer
        (``sym``), the table of every rank's (``peers``) and the TP ``rank``."""
        return {
            "sym": self.local,
            "peers": self.addresses.data_ptr(),
            "rank": self.rank,
        }

    def close(self) -> None:
        """Close the peer mappings and free the local buffer; idempotent. Only once no
        kernel of any rank can still touch it (the runner is being dropped)."""
        self.addresses = None
        opened, self._opened = self._opened, []
        for base in opened:
            UncachedIpcHeap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            UncachedIpcHeap.free_device_mem(self.local)
            self.local = 0
