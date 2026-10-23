# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4 persistent-grid admission and owned epoch IPC arena."""

import torch

from atom.mono.runtime.compile import _SYMBOL_KEY, _code_object
from atom.mono.runtime.consensus import MonoUnsupported, tp_agree
from atom.mono.runtime.peer_memory import _DeviceBytes, _ipc_heap


def _entry_symbol(elf: bytes) -> str:
    """The raw HIP entry name from its ``.symbol`` (a msgpack str, ``<name>.kd``)."""
    at = elf.find(_SYMBOL_KEY)
    if at < 0:
        raise RuntimeError("no .symbol in the code object")
    v = elf[at + len(_SYMBOL_KEY) :]
    if 0xA0 <= v[0] <= 0xBF:  # fixstr
        n, v = v[0] & 0x1F, v[1:]
    elif v[0] == 0xD9:  # str8
        n, v = v[1], v[2:]
    elif v[0] == 0xDA:  # str16
        n, v = int.from_bytes(v[1:3], "big"), v[3:]
    else:
        raise RuntimeError(f"unexpected msgpack tag {v[0]:#x} for .symbol")
    name = v[:n].decode().removesuffix(".kd")
    return name


def require_resident_grid(launcher, grid: int, threads: int) -> None:
    """Check HIP occupancy for the actual binaries before a persistent launch.

    Scratch admission is a separate check. A successful occupancy query alone
    does not prove mailbox/protocol safety or admit another device topology.
    """
    import ctypes

    import torch

    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    hip.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    hip.hipModuleUnload.argtypes = [ctypes.c_void_p]
    for name in (
        "hipModuleLoadData",
        "hipModuleGetFunction",
        "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
        "hipModuleUnload",
    ):
        getattr(hip, name).restype = ctypes.c_int

    def checked(status):
        if status:
            raise RuntimeError(f"HIP persistent residency query failed: {status}")

    artifacts = list(launcher._mem_cache.values())
    if not artifacts:
        raise RuntimeError("compile before checking persistent residency")
    cus = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    for artifact in artifacts:
        elf = _code_object(artifact._ir_text)
        # The metadata's descriptor is '<kernel>.kd'; HIP takes the entry name.
        symbol = _entry_symbol(elf)
        module, function, active = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_int()
        checked(
            hip.hipModuleLoadData(
                ctypes.byref(module), ctypes.create_string_buffer(elf)
            )
        )
        try:
            checked(
                hip.hipModuleGetFunction(
                    ctypes.byref(function), module, symbol.encode()
                )
            )
            checked(
                hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                    ctypes.byref(active), function, threads, 0
                )
            )
            if grid > cus * active.value:
                raise RuntimeError(
                    f"persistent grid {grid} cannot fit {cus} CUs x {active.value} CTAs/CU"
                )
        finally:
            checked(hip.hipModuleUnload(module))


class EpochPeerBuffer:
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
            self.local_address = _ipc_heap().alloc_uncached(nbytes)
            device = torch.device("cuda", torch.cuda.current_device())
            self.storage = torch.as_tensor(
                _DeviceBytes(self.local_address, nbytes), device=device
            )
            mine = None
            if npes > 1:
                mine = (
                    _ipc_heap().get_mem_handle_bytes(self.local_address),
                    0,
                )
        # HIP allocation/export failures use backend-specific exception types;
        # capture all of them so every TP rank reaches the agreement below.
        except Exception as error:  # noqa: BLE001
            failure = error
        if not tp_agree(failure is None, group):
            if self.local_address:
                self.storage = None
                _ipc_heap().free_device_mem(self.local_address)
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
                        mapped = _ipc_heap().open_mem_handle(handle)
                        self._opened.append(mapped)
                        addresses.append(mapped + offset)
            # Mapping can fail through HIP, torch, or the IPC extension; keep
            # peers in the same handshake and report one agreed refusal.
            except Exception as error:  # noqa: BLE001
                failure = error
            if not tp_agree(failure is None, group):
                for mapped in self._opened:
                    _ipc_heap().close_mem_handle(mapped)
                self._opened.clear()
                self.storage = None
                _ipc_heap().free_device_mem(self.local_address)
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
                _ipc_heap().close_mem_handle(mapped)
            # Close all handles even when one backend-specific close fails.
            except Exception as error:  # noqa: BLE001
                pending.append(mapped)
                first = first or error
        self._opened = pending
        if first is not None:
            raise first
        if self.local_address:
            self.storage = None
            _ipc_heap().free_device_mem(self.local_address)
            self.local_address = 0
        self.addresses = None
