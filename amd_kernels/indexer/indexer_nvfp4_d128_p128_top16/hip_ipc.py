# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Uncached device buffers that the other GPUs map (HIP IPC), for ``split4``: ``hipExtMallocWithFlags(...,
hipDeviceMallocUncached)``, a zero fill and ``hipIpcGetMemHandle``, then ``hipIpcOpenMemHandle(...,
hipIpcMemLazyEnablePeerAccess)`` on the peers, through ctypes bound to torch's own HIP runtime (an image can carry
several ``libamdhip64.so``). Buffers are never freed.
"""

import ctypes
import functools
import os

import torch

HIP_DEVICE_MALLOC_UNCACHED = 0x3  # hipDeviceMallocUncached
HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1  # hipIpcMemLazyEnablePeerAccess
HIP_IPC_HANDLE_SIZE = 64


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * HIP_IPC_HANDLE_SIZE)]


@functools.cache
def _hip() -> ctypes.CDLL:
    path = os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so")
    lib = ctypes.CDLL(path if os.path.exists(path) else "libamdhip64.so")
    lib.hipExtMallocWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return lib


def _check(rc: int, what: str) -> None:
    assert rc == 0, f"{what} failed: hipError {rc}"


def allocate_shared_buffer_and_handle(nbytes: int) -> tuple[int, torch.Tensor]:
    """(device pointer, IPC handle as a CPU uint8 tensor) of ``nbytes`` of zeroed uncached memory on the current
    device."""
    torch.cuda.init()
    hip = _hip()
    ptr = ctypes.c_void_p()
    _check(
        hip.hipExtMallocWithFlags(ctypes.byref(ptr), nbytes, HIP_DEVICE_MALLOC_UNCACHED),
        "hipExtMallocWithFlags",
    )
    _check(hip.hipMemset(ptr, 0, nbytes), "hipMemset")
    _check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
    handle = _IpcHandle()
    _check(hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr), "hipIpcGetMemHandle")
    raw = bytes(bytearray(handle.reserved))
    return ptr.value, torch.frombuffer(bytearray(raw), dtype=torch.uint8)


def open_mem_handle(handle: torch.Tensor) -> int:
    """A peer's buffer (``allocate_shared_buffer_and_handle``'s handle) mapped on the current device."""
    h = _IpcHandle()
    ctypes.memmove(h.reserved, bytes(handle.tolist()), HIP_IPC_HANDLE_SIZE)
    ptr = ctypes.c_void_p()
    _check(
        _hip().hipIpcOpenMemHandle(
            ctypes.byref(ptr), h, HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
        ),
        "hipIpcOpenMemHandle",
    )
    return ptr.value
