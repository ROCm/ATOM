# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric peer memory for in-kernel TP reductions."""

import torch
from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

from atom.mono.runtime.consensus import MonoUnsupported, tp_agree
from atom.mono.runtime.step_begin import FENCE_BYTES


class _DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address, for a
    non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class SymmetricPeerBuffer:
    """Dedicated IPC storage for an epoch-tagged complete-layer kernel.

    The HIP calls reuse AITER's IPC helpers; there is no second HIP runtime.
    A caching-allocator tensor can live inside a much larger segment, and HIP
    exports that complete segment rather than the small tensor view. Multiple
    exact widths would then map many GiB of unrelated storage on every peer.
    Use one page-rounded uncached allocation so the exported handle covers only
    this arena. Unlike PeerBuffer this storage is not a StepMailboxes arena.
    Its owner supplies its epoch/scrub protocol. All ranks close in the same
    order.
    """

    def __init__(self, nbytes: int, rank: int = 0, npes: int = 1, group=None):
        import torch.distributed as dist

        if nbytes <= 0 or not 0 <= rank < npes:
            raise ValueError("invalid symmetric peer allocation")
        if npes > 1:
            if not dist.is_initialized():
                raise MonoUnsupported("epoch peers require an initialized CPU TP group")
            group = dist.group.WORLD if group is None else group
            if (
                dist.get_backend(group) != "gloo"
                or dist.get_world_size(group) != npes
                or dist.get_rank(group) != rank
            ):
                raise MonoUnsupported("epoch peers require the matching gloo TP group")
        self.rank, self.npes, self.group = rank, npes, group
        self._opened = []
        self._fenced = False
        self.local_address = 0
        self.storage = None
        failure = None
        try:
            self.local_address = UncachedIpcHeap.alloc_uncached(nbytes)
            device = torch.device("cuda", torch.cuda.current_device())
            self.storage = torch.as_tensor(
                _DeviceBytes(self.local_address, nbytes), device=device
            )
            mine = None
            if npes > 1:
                mine = (
                    UncachedIpcHeap.get_mem_handle_bytes(self.local_address),
                    0,
                )
        # HIP allocation/export failures use backend-specific exception types;
        # capture all of them so every TP rank reaches the agreement below.
        except Exception as error:  # noqa: BLE001
            failure = error
        if not tp_agree(failure is None, group):
            if self.local_address:
                self.storage = None
                UncachedIpcHeap.free_device_mem(self.local_address)
                self.local_address = 0
            raise MonoUnsupported(
                "epoch peer allocation/export failed on a TP rank"
            ) from failure
        addresses = [self.local_address]
        if npes > 1:
            peers = [None] * npes
            dist.all_gather_object(peers, mine, group=group)
            addresses, failure = [], None
            try:
                for peer, (handle, offset) in enumerate(peers):
                    if peer == rank:
                        addresses.append(self.local_address)
                    else:
                        mapped = UncachedIpcHeap.open_mem_handle(handle)
                        self._opened.append(mapped)
                        addresses.append(mapped + offset)
            # Mapping can fail through HIP, torch, or the IPC extension; keep
            # peers in the same handshake and report one agreed refusal.
            except Exception as error:  # noqa: BLE001
                failure = error
            if not tp_agree(failure is None, group):
                for mapped in self._opened:
                    UncachedIpcHeap.close_mem_handle(mapped)
                self._opened.clear()
                self.storage = None
                UncachedIpcHeap.free_device_mem(self.local_address)
                self.local_address = 0
                raise MonoUnsupported("epoch peer memory handshake failed") from failure
            dist.barrier(group=group)
        failure = None
        try:
            self.addresses = torch.tensor(
                addresses, dtype=torch.int64, device=self.storage.device
            )
        # Device table construction can raise from either torch or HIP.
        except Exception as error:  # noqa: BLE001
            failure = error
        if not tp_agree(failure is None, group):
            self.close()
            raise MonoUnsupported(
                "epoch peer address table allocation failed"
            ) from failure

    def close(self):
        import torch.distributed as dist

        if not self._fenced:
            if self.storage is not None:
                torch.cuda.synchronize(self.storage.device)
            if self.npes > 1:
                if not dist.is_initialized():
                    raise RuntimeError(
                        "close epoch peers before destroying the TP process group"
                    )
                dist.barrier(group=self.group)
            self._fenced = True
        pending, first = [], None
        for mapped in self._opened:
            try:
                UncachedIpcHeap.close_mem_handle(mapped)
            # Close all handles even when one backend-specific close fails.
            except Exception as error:  # noqa: BLE001
                pending.append(mapped)
                first = first or error
        self._opened = pending
        if first is not None:
            raise first
        if self.local_address:
            self.storage = None
            UncachedIpcHeap.free_device_mem(self.local_address)
            self.local_address = 0
        self.addresses = None


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
        self.local = UncachedIpcHeap.alloc_uncached(nbytes + FENCE_BYTES)
        # a view, not an owner: close() frees the memory under it
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self.rank = rank
        self._opened: list[int] = []
        addresses = [self.local]
        if npes > 1:
            handles = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, UncachedIpcHeap.get_mem_handle_bytes(self.local)
            )
            addresses, failure = [], None
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        base = UncachedIpcHeap.open_mem_handle(handle)
                        self._opened.append(base)
                        addresses.append(base)
            except RuntimeError as err:  # every rank must still reach the agreement
                failure = err
            if not tp_agree(failure is None, group):
                self.close()
                raise MonoUnsupported(
                    f"peer memory handshake failed on {'this' if failure else 'another'}"
                    " TP rank"
                ) from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

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
        opened, self._opened = self._opened, []
        for base in opened:
            UncachedIpcHeap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            UncachedIpcHeap.free_device_mem(self.local)
            self.local = 0
