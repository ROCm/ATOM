# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""ctypes bindings for the HIP virtual memory management (VMM) API.

Only what a shareable KV pool needs: create physical memory that can be
exported, export / import its handle, and map it into this process's address
space. The structs mirror
``hip_runtime_api.h`` (identical layout in ROCm 7.2 and 7.14).

Handle types (``hipMemAllocationHandleType``):

* ``POSIX_FD`` -- a dma-buf file descriptor; another process on the same host
  takes a duplicate of it (``pidfd_getfd``). Every ROCm with VMM.
* ``FABRIC`` -- a 128-bit globally unique handle another host in the same
  scale-up domain (Helios UALoE) imports. ROCm >= 7.14 (TheRock).
"""

from __future__ import annotations

import ctypes
import functools
import os

HIP_SUCCESS = 0
HIP_IPC_HANDLE_SIZE = 64

HANDLE_TYPE_POSIX_FD = 0x1
HANDLE_TYPE_FABRIC = 0x8

_ALLOCATION_TYPE_PINNED = 0x1
_LOCATION_TYPE_DEVICE = 0x1
_ACCESS_PROT_READ_WRITE = 0x3
_GRANULARITY_RECOMMENDED = 0x1


class HipError(RuntimeError):
    """A HIP VMM call returned an error code."""

    def __init__(self, call: str, code: int) -> None:
        self.call = call
        self.code = code
        super().__init__(f"{call} failed: {error_string(code)} ({code})")


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class _MemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", _MemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_int)]


@functools.lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    """libamdhip64, the same instance torch loaded (RTLD_GLOBAL keeps one)."""
    names = ["libamdhip64.so"]
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    names.append(os.path.join(rocm, "lib", "libamdhip64.so"))
    last: OSError | None = None
    for name in names:
        try:
            lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError as exc:
            last = exc
    else:
        raise OSError(f"cannot load libamdhip64: {last}")

    vp, sz, i, u64 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_ulonglong
    signatures = {
        "hipGetErrorString": ([i], ctypes.c_char_p),
        "hipGetLastError": ([], i),
        "hipGetDevice": ([ctypes.POINTER(i)], i),
        "hipMemGetAllocationGranularity": (
            [ctypes.POINTER(sz), ctypes.POINTER(_MemAllocationProp), i],
            i,
        ),
        "hipMemCreate": (
            [ctypes.POINTER(vp), sz, ctypes.POINTER(_MemAllocationProp), u64],
            i,
        ),
        "hipMemRelease": ([vp], i),
        "hipMemExportToShareableHandle": ([vp, vp, i, u64], i),
        "hipMemImportFromShareableHandle": ([ctypes.POINTER(vp), vp, i], i),
        "hipMemAddressReserve": ([ctypes.POINTER(vp), sz, sz, vp, u64], i),
        "hipMemAddressFree": ([vp, sz], i),
        "hipMemMap": ([vp, sz, sz, vp, u64], i),
        "hipMemUnmap": ([vp, sz], i),
        "hipMemSetAccess": ([vp, sz, ctypes.POINTER(_MemAccessDesc), sz], i),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(lib, name, None)
        if fn is None:
            continue  # probed by callers that can do without it
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


def error_string(code: int) -> str:
    try:
        msg = _lib().hipGetErrorString(code)
    except OSError:
        return "unknown"
    return msg.decode() if msg else "unknown"


def _fail(call: str, code: int) -> HipError:
    """The error for a failed call, with HIP's per-thread last error cleared:
    left set, the next torch launch on this thread reports it as its own."""
    _lib().hipGetLastError()
    return HipError(call, code)


def _check(call: str, code: int) -> None:
    if code != HIP_SUCCESS:
        raise _fail(call, code)


def current_device() -> int:
    dev = ctypes.c_int()
    _check("hipGetDevice", _lib().hipGetDevice(ctypes.byref(dev)))
    return dev.value


def _device_prop(device: int, handle_type: int) -> _MemAllocationProp:
    prop = _MemAllocationProp()
    prop.type = _ALLOCATION_TYPE_PINNED
    prop.requestedHandleType = handle_type
    prop.location.type = _LOCATION_TYPE_DEVICE
    prop.location.id = device
    return prop


def granularity(device: int, handle_type: int) -> int:
    prop = _device_prop(device, handle_type)
    out = ctypes.c_size_t()
    _check(
        "hipMemGetAllocationGranularity",
        _lib().hipMemGetAllocationGranularity(
            ctypes.byref(out), ctypes.byref(prop), _GRANULARITY_RECOMMENDED
        ),
    )
    return out.value


def mem_create(nbytes: int, device: int, handle_type: int) -> int:
    prop = _device_prop(device, handle_type)
    handle = ctypes.c_void_p()
    _check(
        "hipMemCreate",
        _lib().hipMemCreate(ctypes.byref(handle), nbytes, ctypes.byref(prop), 0),
    )
    return handle.value


def mem_release(handle: int) -> None:
    _check("hipMemRelease", _lib().hipMemRelease(handle))


def export_posix_fd(handle: int) -> int:
    fd = ctypes.c_int(-1)
    _check(
        "hipMemExportToShareableHandle",
        _lib().hipMemExportToShareableHandle(
            ctypes.byref(fd), handle, HANDLE_TYPE_POSIX_FD, 0
        ),
    )
    return fd.value


def export_fabric(handle: int) -> bytes:
    buf = (ctypes.c_ubyte * HIP_IPC_HANDLE_SIZE)()
    _check(
        "hipMemExportToShareableHandle",
        _lib().hipMemExportToShareableHandle(
            ctypes.byref(buf), handle, HANDLE_TYPE_FABRIC, 0
        ),
    )
    return bytes(buf)


def import_posix_fd(fd: int) -> int:
    handle = ctypes.c_void_p()
    _check(
        "hipMemImportFromShareableHandle",
        _lib().hipMemImportFromShareableHandle(
            ctypes.byref(handle), ctypes.c_void_p(fd), HANDLE_TYPE_POSIX_FD
        ),
    )
    return handle.value


def import_fabric(blob: bytes) -> int:
    if len(blob) != HIP_IPC_HANDLE_SIZE:
        raise ValueError(
            f"a fabric handle is {HIP_IPC_HANDLE_SIZE} bytes, got {len(blob)}"
        )
    buf = (ctypes.c_ubyte * HIP_IPC_HANDLE_SIZE).from_buffer_copy(blob)
    handle = ctypes.c_void_p()
    _check(
        "hipMemImportFromShareableHandle",
        _lib().hipMemImportFromShareableHandle(
            ctypes.byref(handle), ctypes.byref(buf), HANDLE_TYPE_FABRIC
        ),
    )
    return handle.value


def map_chunks(chunks: list[tuple[int, int]], align: int, device: int) -> int:
    """Map ``(handle, nbytes)`` chunks back to back into one reserved range and
    grant ``device`` read-write access to all of it. Returns the base address;
    undone by :func:`unmap_chunks` with the same sizes."""
    total = sum(n for _, n in chunks)
    ptr = ctypes.c_void_p()
    _check(
        "hipMemAddressReserve",
        _lib().hipMemAddressReserve(ctypes.byref(ptr), total, align, None, 0),
    )
    base = ptr.value
    mapped: list[tuple[int, int]] = []
    try:
        offset = 0
        for handle, nbytes in chunks:
            _check("hipMemMap", _lib().hipMemMap(base + offset, nbytes, 0, handle, 0))
            mapped.append((base + offset, nbytes))
            offset += nbytes
        desc = _MemAccessDesc()
        desc.location.type = _LOCATION_TYPE_DEVICE
        desc.location.id = device
        desc.flags = _ACCESS_PROT_READ_WRITE
        _check(
            "hipMemSetAccess",
            _lib().hipMemSetAccess(base, total, ctypes.byref(desc), 1),
        )
    except BaseException:
        for addr, nbytes in mapped:
            _lib().hipMemUnmap(addr, nbytes)
        _lib().hipMemAddressFree(base, total)
        raise
    return base


def unmap_chunks(base: int, sizes: list[int]) -> None:
    offset = 0
    for nbytes in sizes:
        _check("hipMemUnmap", _lib().hipMemUnmap(base + offset, nbytes))
        offset += nbytes
    _check("hipMemAddressFree", _lib().hipMemAddressFree(base, offset))


# --- dma-buf fd duplication across processes on one host ---------------------

_SYS_PIDFD_OPEN = 434
_SYS_PIDFD_GETFD = 438


@functools.lru_cache(maxsize=1)
def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def dup_fd_from(pid: int, fd: int) -> int:
    """A duplicate, in this process, of ``fd`` open in process ``pid``.

    ``pidfd_getfd`` (Linux >= 5.6) needs ptrace access to ``pid``: the same
    user, or ``CAP_SYS_PTRACE``.
    """
    libc = _libc()
    pidfd = libc.syscall(_SYS_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))
    if pidfd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"pidfd_open({pid}): {os.strerror(err)}")
    try:
        local = libc.syscall(
            _SYS_PIDFD_GETFD, ctypes.c_int(pidfd), ctypes.c_int(fd), ctypes.c_uint(0)
        )
        if local < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"pidfd_getfd({pid}, {fd}): {os.strerror(err)}")
        return local
    finally:
        os.close(pidfd)
