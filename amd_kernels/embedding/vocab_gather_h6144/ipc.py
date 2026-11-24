# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-input registration, adapted from the existing norm family."""
import ctypes
import functools

import torch
import torch.distributed as dist

from amd_kernels.common.hip_runtime import load_hip_runtime

WORLD = 4


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = load_hip_runtime()
    lib.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return lib


def _alloc_handle(ptr: int) -> tuple[int, bytes]:
    """(base of the allocation holding ptr, its IPC handle's 64 raw bytes)."""
    hip = _hip()
    base, size = ctypes.c_void_p(), ctypes.c_size_t()
    rc = hip.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ptr)
    assert rc == 0, f"hipMemGetAddressRange failed: {rc}"
    h = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(h), base)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    # the raw bytes: a c_char field reads back only up to its first NUL
    return base.value, ctypes.string_at(ctypes.addressof(h), ctypes.sizeof(h))


def _open_handle(raw: bytes) -> int:
    hip = _hip()
    h = _IpcHandle()
    ctypes.memmove(ctypes.addressof(h), raw, len(raw))
    ptr = ctypes.c_void_p()
    rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)  # lazy peer access
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def exchange(obj, rank: int, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails
    on gloo under inference mode)."""
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]
